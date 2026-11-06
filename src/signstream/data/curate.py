"""Dataset curation: deduplication, quality filtering, and label auditing."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from signstream.config import DataConfig
from signstream.data.corpus import GLOSSES
from signstream.data.keypoints import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    normalize_sequence,
    resample_sequence,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DedupeReport:
    """Result of duplicate detection within each gloss."""

    n_input: int
    n_dropped: int
    dropped_ids: list[str] = field(default_factory=list)
    duplicate_of: dict[str, str] = field(default_factory=dict)
    groups: list[list[str]] = field(default_factory=list)
    kept_ids: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Result of frame-count, missing-data, and motion checks."""

    n_input: int
    n_dropped: int
    dropped_ids: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    kept_ids: list[str] = field(default_factory=list)

    @property
    def counts_by_reason(self) -> dict[str, int]:
        return {reason: len(ids) for reason, ids in self.reasons.items()}


@dataclass
class LabelAuditReport:
    """Model-free label audit with held-out agreement statistics."""

    n_input: int
    n_flagged: int
    top_fraction: float
    flagged_ids: list[str] = field(default_factory=list)
    suggested_labels: dict[str, str] = field(default_factory=dict)
    disagreement: dict[str, float] = field(default_factory=dict)
    injected_total: int = 0
    true_positives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    flagged_injected_ids: list[str] = field(default_factory=list)
    missed_injected_ids: list[str] = field(default_factory=list)


@dataclass
class CurationResult:
    """Curated records plus the complete audit trail."""

    kept_indices: list[int]
    kept_records: list[dict[str, Any]]
    report: dict[str, Any]


def _sequence_sha1(sequence: np.ndarray) -> str:
    """Perceptual hash: resample, quantize, then SHA-1."""
    sampled = resample_sequence(sequence, 16)
    sampled = np.nan_to_num(sampled, nan=0.0)
    quantized = np.round(sampled * 1000.0).astype(np.int16)
    return hashlib.sha1(quantized.tobytes()).hexdigest()


def _dedupe_vector(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized flat keypoints and an L2-normalized descriptor."""
    sampled = resample_sequence(sequence, 16)
    sampled = np.nan_to_num(normalize_sequence(sampled), nan=0.0)
    flat = sampled.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(flat))
    descriptor = flat / norm if norm > 1e-8 else flat
    return flat, descriptor


def dedupe(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    *,
    cosine_threshold: float = 0.995,
    max_relative_rmse: float = 0.02,
) -> DedupeReport:
    """Detect exact and near-duplicate takes within each gloss.

    A pair is a near duplicate when its 16-frame perceptual descriptor cosine is
    above ``cosine_threshold`` **and** its relative RMSE is below
    ``max_relative_rmse``. The second check is important: two clean takes of the
    same gloss already share most of their body pose, so cosine alone would
    collapse an entire class. Earlier clips are always kept.
    """
    n_input = len(records)
    by_label: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_label.setdefault(int(record["label_id"]), []).append(index)

    kept: set[int] = set()
    dropped: set[int] = set()
    duplicate_of: dict[str, str] = {}
    groups: list[list[str]] = []
    hashes: dict[str, int] = {}

    for label_indices in by_label.values():
        local: list[int] = []
        descriptors: list[np.ndarray] = []
        flat_vectors: list[np.ndarray] = []
        for index in label_indices:
            record = records[index]
            sequence = np.asarray(clips[index]).astype(np.float32)
            digest = _sequence_sha1(sequence)
            duplicate_index = None
            if digest in hashes and hashes[digest] in local:
                duplicate_index = hashes[digest]
            flat, descriptor = _dedupe_vector(sequence)
            if duplicate_index is None and descriptors:
                matrix = np.stack(descriptors, axis=0)
                similarities = matrix @ descriptor
                best = int(np.argmax(similarities))
                best_similarity = float(similarities[best])
                reference = flat_vectors[best]
                denominator = float(np.sqrt(np.mean(reference**2))) + 1e-8
                relative_rmse = float(
                    np.sqrt(np.mean((flat - reference) ** 2)) / denominator
                )
                if best_similarity > cosine_threshold and relative_rmse < max_relative_rmse:
                    duplicate_index = local[best]

            if duplicate_index is not None:
                dropped.add(index)
                kept.discard(index)
                duplicate_of[record["clip_id"]] = records[duplicate_index]["clip_id"]
                keep_id = records[duplicate_index]["clip_id"]
                for group in groups:
                    if keep_id in group:
                        group.append(record["clip_id"])
                        break
                else:
                    groups.append([keep_id, record["clip_id"]])
            else:
                local.append(index)
                descriptors.append(descriptor)
                flat_vectors.append(flat)
                kept.add(index)
                hashes[digest] = index

    dropped_ids = [records[i]["clip_id"] for i in sorted(dropped)]
    kept_ids = [records[i]["clip_id"] for i in sorted(kept)]
    LOGGER.info("Deduplication: %d/%d clips dropped", len(dropped_ids), n_input)
    return DedupeReport(
        n_input=n_input,
        n_dropped=len(dropped_ids),
        dropped_ids=dropped_ids,
        duplicate_of=duplicate_of,
        groups=groups,
        kept_ids=kept_ids,
    )


def _valid_sequence(clip: np.ndarray, valid_length: int) -> np.ndarray:
    length = int(np.clip(valid_length, 0, clip.shape[0]))
    return np.asarray(clip[:length], dtype=np.float32)


def quality_filter(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    *,
    min_valid_frames: int,
    max_nan_fraction: float,
    allowed_indices: Iterable[int] | None = None,
) -> QualityReport:
    """Drop clips with too few frames, excessive NaNs, or no motion energy."""
    candidates = (
        list(range(len(records))) if allowed_indices is None else sorted(set(allowed_indices))
    )
    reasons: dict[str, list[str]] = {
        "too_few_frames": [],
        "excess_nan": [],
        "no_motion": [],
    }
    kept: list[int] = []
    for index in candidates:
        record = records[index]
        clip = np.asarray(clips[index], dtype=np.float32)
        valid_length = int(record.get("valid_length", clip.shape[0]))
        sequence = _valid_sequence(clip, valid_length)
        if sequence.shape[0] < min_valid_frames:
            reasons["too_few_frames"].append(record["clip_id"])
            continue
        missing_fraction = float(np.isnan(sequence).mean())
        if missing_fraction > max_nan_fraction:
            reasons["excess_nan"].append(record["clip_id"])
            continue
        filled = np.nan_to_num(sequence, nan=0.0)
        if sequence.shape[0] > 1:
            motion = float(np.mean(np.abs(np.diff(filled, axis=0))))
        else:
            motion = 0.0
        if motion < 1e-6:
            reasons["no_motion"].append(record["clip_id"])
            continue
        kept.append(index)

    dropped_ids = [record_id for ids in reasons.values() for record_id in ids]
    LOGGER.info(
        "Quality filter: %d/%d clips dropped (%s)",
        len(dropped_ids),
        len(candidates),
        {name: len(ids) for name, ids in reasons.items() if ids},
    )
    return QualityReport(
        n_input=len(candidates),
        n_dropped=len(dropped_ids),
        dropped_ids=dropped_ids,
        reasons=reasons,
        kept_ids=[records[i]["clip_id"] for i in kept],
    )


def _audit_descriptor(clip: np.ndarray, valid_length: int) -> np.ndarray:
    sequence = _valid_sequence(clip, valid_length)
    sampled = resample_sequence(sequence, 48)
    sampled = np.nan_to_num(sampled, nan=0.0)
    sampled = normalize_sequence(sampled)
    vector = np.nan_to_num(sampled, nan=0.0).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _knn_suggestions(
    descriptors: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out k-NN vote fraction for each clip's own label and suggestions."""
    n = descriptors.shape[0]
    own_fraction = np.zeros(n, dtype=np.float32)
    suggested = labels.copy()
    k_eff = max(1, min(k, n - 1))
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        similarities = descriptors[start:stop] @ descriptors.T
        for row in range(stop - start):
            index = start + row
            similarities[row, index] = -np.inf
            neighbors = np.argpartition(similarities[row], -k_eff)[-k_eff:]
            votes = labels[neighbors]
            counts = np.bincount(votes, minlength=len(GLOSSES))
            suggested[index] = int(np.argmax(counts))
            own_fraction[index] = counts[labels[index]] / k_eff
    return own_fraction, suggested


def audit_labels(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    *,
    top_fraction: float = 0.02,
    model_free: bool = True,
    allowed_indices: Iterable[int] | None = None,
) -> LabelAuditReport:
    """Flag likely label errors using leave-one-out nearest-centroid and k-NN.

    The audit never uses labels at inference time beyond comparing a clip to
    class statistics. When injected noise is present, precision and recall are
    measured against that known ground truth.
    """
    if not model_free:
        raise ValueError("Only the model-free audit is implemented")
    candidates = (
        list(range(len(records))) if allowed_indices is None else sorted(set(allowed_indices))
    )
    if not candidates:
        return LabelAuditReport(n_input=0, n_flagged=0, top_fraction=top_fraction)

    descriptors = np.stack(
        [
            _audit_descriptor(clips[index], int(records[index].get("valid_length", 48)))
            for index in candidates
        ],
        axis=0,
    )
    labels = np.asarray([int(records[index]["label_id"]) for index in candidates], dtype=np.int64)

    sums = np.zeros((len(GLOSSES), descriptors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(GLOSSES), dtype=np.float32)
    for descriptor, label in zip(descriptors, labels, strict=True):
        sums[label] += descriptor
        counts[label] += 1
    centroids = sums / np.maximum(counts[:, None], 1.0)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    class_similarities = descriptors @ centroids.T
    own_similarity = class_similarities[np.arange(len(candidates)), labels]
    knn_fraction, knn_suggested = _knn_suggestions(descriptors, labels, k=7)
    # The leave-one-out k-NN vote is the primary signal. The centroid margin is
    # retained as a tiny tie-breaker so visually near-identical failures do not
    # get arbitrary ordering.
    disagreement = (1.0 - knn_fraction) + 0.001 * (1.0 - own_similarity)
    n_flagged = max(1, int(round(top_fraction * len(candidates))))
    flagged_rows = np.argsort(disagreement)[-n_flagged:][::-1]
    flagged_indices = [candidates[int(row)] for row in flagged_rows]
    flagged_ids = [records[index]["clip_id"] for index in flagged_indices]
    suggested_labels = {
        records[index]["clip_id"]: GLOSSES[int(knn_suggested[row])]
        for row, index in zip(flagged_rows, flagged_indices, strict=True)
    }
    disagreement_map = {
        records[index]["clip_id"]: float(disagreement[row])
        for row, index in zip(flagged_rows, flagged_indices, strict=True)
    }

    injected_ids = {
        records[index]["clip_id"] for index in candidates if records[index].get("injected_mislabel")
    }
    flagged_set = set(flagged_ids)
    true_positives = len(flagged_set & injected_ids)
    precision = true_positives / len(flagged_ids) if flagged_ids else 0.0
    recall = true_positives / len(injected_ids) if injected_ids else 0.0
    LOGGER.info(
        "Label audit: flagged %d clips; precision=%.3f recall=%.3f against %d injected mislabels",
        len(flagged_ids),
        precision,
        recall,
        len(injected_ids),
    )
    return LabelAuditReport(
        n_input=len(candidates),
        n_flagged=len(flagged_ids),
        top_fraction=top_fraction,
        flagged_ids=flagged_ids,
        suggested_labels=suggested_labels,
        disagreement=disagreement_map,
        injected_total=len(injected_ids),
        true_positives=true_positives,
        precision=float(precision),
        recall=float(recall),
        flagged_injected_ids=sorted(flagged_set & injected_ids),
        missed_injected_ids=sorted(injected_ids - flagged_set),
    )


def curate_all(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    config: DataConfig,
    *,
    audit: bool = True,
) -> CurationResult:
    """Run the complete curation pipeline and write the audit report."""
    dedupe_report = dedupe(
        clips, records, cosine_threshold=config.dedupe_cosine_threshold
    )
    dropped_duplicates = set(dedupe_report.dropped_ids)
    allowed = [
        index for index, record in enumerate(records)
        if record["clip_id"] not in dropped_duplicates
    ]
    quality_report = quality_filter(
        clips,
        records,
        min_valid_frames=config.min_valid_frames,
        max_nan_fraction=config.max_nan_fraction,
        allowed_indices=allowed,
    )
    quality_dropped = set(quality_report.dropped_ids)
    audit_candidates = [
        index for index in allowed if records[index]["clip_id"] not in quality_dropped
    ]
    label_report = (
        audit_labels(
            clips,
            records,
            top_fraction=config.curation_top_fraction,
            allowed_indices=audit_candidates,
        )
        if audit
        else LabelAuditReport(n_input=len(audit_candidates), n_flagged=0, top_fraction=0.0)
    )
    flagged = set(label_report.flagged_ids)
    kept_indices = [
        index for index in audit_candidates if records[index]["clip_id"] not in flagged
    ]
    kept_records = [records[index] for index in kept_indices]

    counts_by_reason = {
        "duplicate": dedupe_report.n_dropped,
        **quality_report.counts_by_reason,
        "suspected_mislabel": label_report.n_flagged if audit else 0,
    }
    report = {
        "schema": "signstream-curation-v1",
        "input_clips": len(records),
        "kept_clips": len(kept_records),
        "dropped_clips": len(records) - len(kept_records),
        "counts_by_reason": counts_by_reason,
        "dedupe": {
            "n_dropped": dedupe_report.n_dropped,
            "n_groups": len(dedupe_report.groups),
            "dropped_ids": dedupe_report.dropped_ids,
            "duplicate_of": dedupe_report.duplicate_of,
            "groups": dedupe_report.groups[:50],
        },
        "quality": {
            "n_dropped": quality_report.n_dropped,
            "counts_by_reason": quality_report.counts_by_reason,
            "dropped_ids": quality_report.dropped_ids,
        },
        "label_audit": {
            "n_input": label_report.n_input,
            "n_flagged": label_report.n_flagged,
            "top_fraction": label_report.top_fraction,
            "injected_total": label_report.injected_total,
            "true_positives": label_report.true_positives,
            "precision": label_report.precision,
            "recall": label_report.recall,
            "flagged_injected_ids": label_report.flagged_injected_ids,
            "missed_injected_ids": label_report.missed_injected_ids,
            "flagged_samples": label_report.flagged_ids[:20],
            "suggested_labels": dict(list(label_report.suggested_labels.items())[:20]),
        },
        "kept_ids": [record["clip_id"] for record in kept_records],
        "feature_dim": FEATURE_DIM,
        "landmarks": LANDMARK_COUNT,
    }
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (Path(config.data_dir) / "curation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (reports_dir / "curation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info(
        "Curation kept %d/%d clips; drops=%s; audit precision=%.3f recall=%.3f",
        len(kept_records),
        len(records),
        counts_by_reason,
        label_report.precision,
        label_report.recall,
    )
    return CurationResult(kept_indices=kept_indices, kept_records=kept_records, report=report)
