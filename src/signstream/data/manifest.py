"""Versioned dataset manifest generation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from signstream.config import DataConfig
from signstream.data.corpus import content_hash_ids, corpus_content_hash
from signstream.data.keypoints import FEATURE_DIM, LANDMARK_COUNT

LOGGER = logging.getLogger(__name__)


def _manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_manifest(
    version: str,
    config: DataConfig,
    kept_records: list[dict[str, Any]],
    curation_report: dict[str, Any],
    split_payloads: dict[str, dict[str, Any]],
    *,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``manifest_<version>.json`` describing the curated dataset."""
    directory = Path(data_dir or config.data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    per_gloss = Counter(str(record["label"]) for record in kept_records)
    per_signer = Counter(str(record["signer"]) for record in kept_records)
    quality_by_signer: dict[str, str] = {}
    handedness_by_signer: dict[str, str] = {}
    for record in kept_records:
        quality_by_signer[str(record["signer"])] = str(record["quality_tier"])
        handedness_by_signer[str(record["signer"])] = str(record["handedness"])

    payload: dict[str, Any] = {
        "dataset_version": version,
        "schema_version": config.schema_version,
        "seed": config.seed,
        "feature_dim": FEATURE_DIM,
        "landmark_count": LANDMARK_COUNT,
        "sequence_frames": config.seq_len,
        "content_hash": content_hash_ids(str(record["clip_id"]) for record in kept_records),
        "label_content_hash": corpus_content_hash(kept_records),
        "num_clips": len(kept_records),
        "num_glosses": len(per_gloss),
        "num_signers": len(per_signer),
        "per_gloss_counts": dict(sorted(per_gloss.items())),
        "per_signer_counts": dict(sorted(per_signer.items())),
        "per_signer_quality_tier": dict(sorted(quality_by_signer.items())),
        "per_signer_handedness": dict(sorted(handedness_by_signer.items())),
        "curation_drop_summary": curation_report.get("counts_by_reason", {}),
        "label_audit": {
            "precision": curation_report.get("label_audit", {}).get("precision", 0.0),
            "recall": curation_report.get("label_audit", {}).get("recall", 0.0),
            "flagged": curation_report.get("label_audit", {}).get("n_flagged", 0),
        },
        "split_definitions": {
            name: {
                "path": f"{config.data_dir}/splits/{name}.json",
                "strategy": split["strategy"],
                "counts": {
                    "train": len(split["train"]),
                    "val": len(split["val"]),
                    "test": len(split["test"]),
                },
                "signers": split.get("signer_lists", {}),
            }
            for name, split in split_payloads.items()
        },
    }
    payload["dataset_manifest_hash"] = _manifest_hash(payload)
    path = directory / f"manifest_{version}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "Wrote manifest %s (%d clips, hash=%s)",
        path,
        len(kept_records),
        payload["dataset_manifest_hash"][:12],
    )
    return payload


def load_manifest(path: str | Path = "artifacts/data/manifest_v1.json") -> dict[str, Any]:
    """Load a manifest JSON file."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
