"""Deterministic, ASL-structured procedural signing corpus.

Real ASL corpora (WLASL, How2Sign) require multi-gigabyte downloads and manual
license acceptance. To make the entire product reproducible from a clean clone,
this module synthesizes physically structured upper-body signing trajectories
from 60 gloss prototypes, 14 persistent signer styles, and six takes per pair.
It intentionally injects the defects a real curation pipeline must catch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from signstream.config import DataConfig
from signstream.data.keypoints import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    sequence_descriptor,
)

LOGGER = logging.getLogger(__name__)

GLOSSES: tuple[str, ...] = (
    "HELLO", "THANK-YOU", "HELP", "WATER", "NAME", "LEARN", "YES", "NO",
    "PLEASE", "SORRY", "MORE", "FINISH", "EAT", "DRINK", "GO", "COME",
    "WANT", "NEED", "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER",
    "SISTER", "HOME", "SCHOOL", "WORK", "PLAY", "READ", "WRITE", "TALK",
    "SEE", "LOOK", "WAIT", "STOP", "AGAIN", "UNDERSTAND", "KNOW", "THINK",
    "FEEL", "HAPPY", "SAD", "ANGRY", "LOVE", "LIKE", "GOOD", "BAD", "BIG",
    "SMALL", "HOT", "COLD", "DAY", "NIGHT", "TIME", "NOW", "LATER",
    "MORNING", "EVENING", "WHERE", "WHO",
)
GLOSS_TO_ID = {gloss: index for index, gloss in enumerate(GLOSSES)}

# Pairs a handshape/place-confusable neighbor for injected label noise.
CONFUSABLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("HELLO", "THANK-YOU"), ("HELP", "PLEASE"), ("WATER", "DRINK"),
    ("NAME", "WHO"), ("LEARN", "SCHOOL"), ("YES", "GOOD"), ("NO", "BAD"),
    ("FINISH", "STOP"), ("EAT", "HUNGRY" if "HUNGRY" in GLOSSES else "MORE"),
    ("GO", "COME"), ("WANT", "NEED"), ("FRIEND", "FAMILY"),
    ("MOTHER", "FATHER"), ("BROTHER", "SISTER"), ("HOME", "WORK"),
    ("PLAY", "READ"), ("WRITE", "TALK"), ("SEE", "LOOK"), ("WAIT", "LATER"),
    ("AGAIN", "MORE"), ("UNDERSTAND", "KNOW"), ("THINK", "FEEL"),
    ("HAPPY", "SAD"), ("ANGRY", "SAD"), ("LOVE", "LIKE"), ("BIG", "SMALL"),
    ("HOT", "COLD"), ("DAY", "NIGHT"), ("TIME", "NOW"), ("MORNING", "EVENING"),
    ("WHERE", "WHO"), ("SORRY", "PLEASE"),
)
CONFUSABLE_NEIGHBOR: dict[str, str] = {}
for _left, _right in CONFUSABLE_PAIRS:
    if _left in GLOSS_TO_ID and _right in GLOSS_TO_ID:
        CONFUSABLE_NEIGHBOR.setdefault(_left, _right)
        CONFUSABLE_NEIGHBOR.setdefault(_right, _left)


@dataclass(frozen=True)
class SignerStyle:
    """Persistent, signer-specific capture and articulation style."""

    signer_id: str
    index: int
    scale: float
    speed: float
    amplitude: float
    noise_sigma: float
    left_dominant: bool
    quality_tier: str
    frame_dropout_rate: float
    hand_loss_rate: float
    persistent_offset: np.ndarray

    @property
    def handedness(self) -> str:
        return "left-dominant" if self.left_dominant else "right-dominant"


def signer_styles(config: DataConfig) -> list[SignerStyle]:
    """Build deterministic signer identities and persistent capture styles."""
    styles: list[SignerStyle] = []
    for index in range(config.num_signers):
        signer_id = f"S{index + 1:02d}"
        rng = np.random.default_rng(config.seed + 1009 * (index + 1))
        low_quality = signer_id in config.low_quality_signers
        styles.append(
            SignerStyle(
                signer_id=signer_id,
                index=index,
                scale=float(rng.uniform(0.92, 1.08)),
                speed=float(rng.uniform(0.75, 1.30)),
                amplitude=float(rng.uniform(0.82, 1.16)),
                noise_sigma=float(rng.uniform(0.003, 0.006))
                * (config.low_quality_noise_multiplier if low_quality else 1.0),
                left_dominant=signer_id in config.left_dominant_signers,
                quality_tier="LOW" if low_quality else "HIGH",
                frame_dropout_rate=config.frame_dropout_rate if low_quality else 0.0,
                hand_loss_rate=config.hand_loss_rate if low_quality else 0.01,
                persistent_offset=rng.normal(0.0, 0.006, size=(LANDMARK_COUNT, 3)).astype(
                    np.float32
                ),
            )
        )
    return styles


def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _hand_landmarks(
    wrist: np.ndarray,
    curls: np.ndarray,
    direction: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Build a 21-point hand from a wrist, arm direction, and five curl values."""
    wrist = np.asarray(wrist, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction[:2]))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / max(norm, 1e-6)
    perpendicular = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
    points = [wrist]
    finger_lengths = (0.050, 0.058, 0.055, 0.048, 0.038)
    for finger in range(5):
        curl = float(np.clip(curls[finger], 0.0, 1.0))
        side = (finger - 2.0) * 0.016
        base = wrist + direction * (0.030 * scale) + perpendicular * side * scale
        points.append(base)
        length = finger_lengths[finger] * scale
        for joint in range(1, 4):
            progress = joint / 3.0
            extension = 1.0 - 0.68 * curl * progress
            bend = -0.048 * curl * progress * scale
            point = (
                base
                + direction * length * progress * extension
                + perpendicular * side * 0.18 * progress
                + np.array([0.0, 0.0, bend], dtype=np.float32)
            )
            points.append(point)
    return np.stack(points, axis=0).astype(np.float32)


def _pose_landmarks(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    body_phase: float,
    amplitude: float,
) -> np.ndarray:
    """Build the 19 upper-body pose landmarks around the two wrists."""
    sway = 0.025 * np.sin(body_phase) * amplitude
    pose = np.zeros((19, 3), dtype=np.float32)
    pose[0] = (0.020 + sway, 0.72, 0.0)       # nose
    pose[1] = (-0.045 + sway, 0.76, -0.01)    # left eye
    pose[2] = (0.070 + sway, 0.76, -0.01)     # right eye
    pose[3] = (-0.130 + sway, 0.72, 0.0)      # left ear
    pose[4] = (0.155 + sway, 0.72, 0.0)       # right ear
    pose[5] = (-0.50, 0.17, 0.0)              # left shoulder
    pose[6] = (0.50, 0.17, 0.0)               # right shoulder
    pose[7] = (-0.61 + sway, -0.15, -0.02)    # left elbow
    pose[8] = (0.61 + sway, -0.15, -0.02)     # right elbow
    pose[9] = left_wrist                       # left wrist
    pose[10] = right_wrist                     # right wrist
    pose[11] = (-0.78, -0.36, 0.0)            # left pinky
    pose[12] = (0.78, -0.36, 0.0)             # right pinky
    pose[13] = (-0.76, -0.34, 0.0)            # left index
    pose[14] = (0.76, -0.34, 0.0)             # right index
    pose[15] = (-0.74, -0.30, 0.0)            # left thumb
    pose[16] = (0.74, -0.30, 0.0)             # right thumb
    pose[17] = (-0.34, -0.66, 0.0)            # left hip
    pose[18] = (0.34, -0.66, 0.0)             # right hip
    return pose


def _smooth_center(
    start: np.ndarray,
    end: np.ndarray,
    phase: np.ndarray,
    path: str,
    amplitude: float,
) -> np.ndarray:
    """Interpolate a hand center along one of four ASL-like movement paths."""
    blend = phase[:, None]
    center = start[None, :] * (1.0 - blend) + end[None, :] * blend
    if path == "arc":
        center[:, 1] += np.sin(np.pi * phase) * 0.18 * amplitude
        center[:, 0] += np.sin(np.pi * phase) * 0.05
    elif path == "circular":
        center[:, 0] += np.sin(2.0 * np.pi * phase) * 0.10 * amplitude
        center[:, 1] += (1.0 - np.cos(2.0 * np.pi * phase)) * 0.07 * amplitude
    elif path == "repeated-contact":
        center[:, 0] += np.sin(4.0 * np.pi * phase) * 0.07
        center[:, 1] += np.abs(np.sin(2.0 * np.pi * phase)) * 0.10
    return center.astype(np.float32)


def _prototype(gloss: str, config: DataConfig) -> dict[str, Any]:
    """Derive a deterministic gloss prototype from a SHA-256 hash of its label."""
    digest = hashlib.sha256(gloss.encode("utf-8")).digest()
    rng = np.random.default_rng(_hash_int(gloss) ^ config.seed)
    duration = 25 + digest[0] % 36
    n_keyframes = 3 + digest[1] % 4
    path = ("arc", "straight", "circular", "repeated-contact")[digest[2] % 4]
    two_handed = bool(digest[3] % 3 == 0)
    end_x = 0.04 + (digest[4] / 255.0) * 0.58
    end_y = -0.20 + (digest[5] / 255.0) * 0.68
    start = np.array([0.30 + 0.08 * (digest[6] / 255.0), -0.22], dtype=np.float32)
    end = np.array([end_x, end_y], dtype=np.float32)
    if digest[7] % 2:
        start, end = end, start
    curl_base = rng.uniform(0.05, 0.88, size=(n_keyframes, 5)).astype(np.float32)
    left_curl = rng.uniform(0.05, 0.85, size=(n_keyframes, 5)).astype(np.float32)
    if not two_handed:
        left_curl = np.clip(left_curl * 0.35, 0.0, 1.0)
    return {
        "gloss": gloss,
        "label_id": GLOSS_TO_ID[gloss],
        "duration": duration,
        "path": path,
        "two_handed": two_handed,
        "start": start,
        "end": end,
        "curls": curl_base,
        "left_curls": left_curl,
        "vertical_wave": float((digest[8] / 255.0) * 2.0 - 1.0),
        "body_phase": float((digest[9] / 255.0) * 2.0 * np.pi),
    }


def _render_prototype(prototype: dict[str, Any], style: SignerStyle, take: int) -> np.ndarray:
    """Render one take of a prototype through a signer's persistent style."""
    duration = int(round(prototype["duration"] / style.speed))
    duration = int(np.clip(duration + (take % 3) - 1, 20, 72))
    take_rng = np.random.default_rng(
        _hash_int(f"{prototype['gloss']}:{style.signer_id}:{take}") ^ 17
    )
    phase = np.linspace(0.0, 1.0, duration, dtype=np.float32)
    take_amplitude = style.amplitude * float(take_rng.uniform(0.90, 1.10))
    take_offset = take_rng.normal(0.0, 0.035, size=2).astype(np.float32)
    take_phase = float(take_rng.uniform(0.0, 2.0 * np.pi))
    start = prototype["start"] * take_amplitude + take_offset
    end = prototype["end"] * take_amplitude + take_offset
    center = _smooth_center(start, end, phase, prototype["path"], take_amplitude)
    center[:, 1] += np.sin(2.0 * np.pi * phase + prototype["body_phase"] + take_phase) * 0.025
    center[:, 0] += prototype["vertical_wave"] * 0.018
    center[:, 1] += take_rng.normal(0.0, 0.012, size=duration).astype(np.float32)

    left_center = -center.copy()
    if not prototype["two_handed"]:
        rest = np.array([-0.34, -0.20], dtype=np.float32)
        left_center = rest + (left_center - rest) * 0.12

    key_positions = np.linspace(0.0, 1.0, len(prototype["curls"]))
    left_key_positions = np.linspace(0.0, 1.0, len(prototype["left_curls"]))
    curls = np.stack(
        [np.interp(phase, key_positions, prototype["curls"][:, f]) for f in range(5)],
        axis=1,
    ).astype(np.float32)
    left_curls = np.stack(
        [np.interp(phase, left_key_positions, prototype["left_curls"][:, f]) for f in range(5)],
        axis=1,
    ).astype(np.float32)
    curl_gain = take_rng.uniform(0.88, 1.12, size=(1, 5)).astype(np.float32)
    curls = np.clip(curls * curl_gain + take_rng.normal(0.0, 0.035, size=curls.shape), 0.0, 1.0)
    left_curls = np.clip(
        left_curls * curl_gain + take_rng.normal(0.0, 0.035, size=left_curls.shape), 0.0, 1.0
    )

    sequence = np.full((duration, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    for t in range(duration):
        right_wrist = np.array([center[t, 0], center[t, 1], 0.0], dtype=np.float32)
        left_wrist = np.array([left_center[t, 0], left_center[t, 1], 0.0], dtype=np.float32)
        right_dir = np.array(
            [right_wrist[0] - 0.58, right_wrist[1] + 0.18, 0.0], dtype=np.float32
        )
        left_dir = np.array(
            [left_wrist[0] + 0.58, left_wrist[1] + 0.18, 0.0], dtype=np.float32
        )
        sequence[t, :21] = _hand_landmarks(
            left_wrist, left_curls[t], left_dir, style.scale
        )
        sequence[t, 21:42] = _hand_landmarks(
            right_wrist, curls[t], right_dir, style.scale
        )
        sequence[t, 42:] = _pose_landmarks(
            left_wrist, right_wrist, float(2.0 * np.pi * phase[t] + prototype["body_phase"]),
            style.amplitude,
        )

    sequence = sequence * style.scale
    if style.left_dominant:
        sequence[..., 0] *= -1.0
    # A small per-take camera/body-angle change makes repeated takes realistically
    # non-identical while leaving re-encoded duplicates exactly comparable.
    angle = float(take_rng.uniform(-0.10, 0.10))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated_x = cos_a * sequence[..., 0] - sin_a * sequence[..., 1]
    rotated_y = sin_a * sequence[..., 0] + cos_a * sequence[..., 1]
    sequence[..., 0] = rotated_x
    sequence[..., 1] = rotated_y
    sequence += style.persistent_offset[None, :, :]
    sequence += take_rng.normal(0.0, style.noise_sigma, size=sequence.shape).astype(np.float32)

    if style.frame_dropout_rate > 0.0:
        drop = take_rng.random(duration) < style.frame_dropout_rate
        sequence[drop] = np.nan
    if take_rng.random() < style.hand_loss_rate:
        hand = int(take_rng.integers(0, 2))
        length = int(take_rng.integers(5, 12))
        start_frame = int(take_rng.integers(0, max(1, duration - length)))
        sequence[start_frame : start_frame + length, hand * 21 : (hand + 1) * 21] = np.nan
    return sequence


def content_hash_ids(clip_ids: Iterable[str]) -> str:
    """Stable SHA-256 content hash of a sorted collection of clip identifiers."""
    joined = "\n".join(sorted(clip_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _clip_id(label_id: int, signer: SignerStyle, take: int) -> str:
    return f"g{label_id:02d}_{signer.signer_id}_t{take:02d}"


def generate_corpus(config: DataConfig, *, quick: bool | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate the full procedural corpus.

    Args:
        config: Data-generation configuration.
        quick: Generate a small deterministic slice for CI. Defaults to
            ``config.quick``.

    Returns:
        ``(clips, metadata)`` where ``clips`` has shape
        ``[N, 48, 61, 3]`` and metadata is a parallel list of clip records.
    """
    use_quick = config.quick if quick is None else quick
    glosses = GLOSSES[: config.quick_glosses] if use_quick else GLOSSES[: config.num_glosses]
    takes = config.quick_takes if use_quick else config.takes_per_pair
    styles = signer_styles(config)
    if use_quick:
        # Keep all signers so signer-holdout validation remains meaningful.
        styles = styles
        takes = min(takes, config.takes_per_pair)

    records: list[dict[str, Any]] = []
    sequences: list[np.ndarray] = []
    prototypes = {gloss: _prototype(gloss, config) for gloss in glosses}
    for gloss in glosses:
        prototype = prototypes[gloss]
        for style in styles:
            for take in range(takes):
                clip_id = _clip_id(prototype["label_id"], style, take)
                raw = _render_prototype(prototype, style, take)
                frames = np.full((config.clip_frames, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
                keep = min(raw.shape[0], config.clip_frames)
                frames[:keep] = raw[:keep]
                sequences.append(frames.astype(np.float16))
                records.append(
                    {
                        "clip_id": clip_id,
                        "label_id": int(prototype["label_id"]),
                        "label": gloss,
                        "true_label_id": int(prototype["label_id"]),
                        "true_label": gloss,
                        "signer": style.signer_id,
                        "signer_index": style.index,
                        "take": take,
                        "valid_length": int(keep),
                        "duration_frames": int(raw.shape[0]),
                        "path": prototype["path"],
                        "two_handed": bool(prototype["two_handed"]),
                        "quality_tier": style.quality_tier,
                        "handedness": style.handedness,
                        "injected_defect": None,
                        "duplicate_of": None,
                        "injected_mislabel": False,
                    }
                )

    clips = np.stack(sequences, axis=0).astype(np.float16)
    records = _inject_defects(clips, records, config, quick=use_quick)
    return clips, records


def _inject_defects(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    config: DataConfig,
    *,
    quick: bool,
) -> list[dict[str, Any]]:
    """Inject reproducible duplicates, mislabels, truncations, and track failures."""
    n_clips = len(records)
    rng = np.random.default_rng(config.seed + 7_919)
    scale = 0.5 if quick else 1.0
    duplicate_count = max(1, int(round(config.duplicate_rate * n_clips * scale)))
    mislabel_count = max(1, int(round(config.mislabel_rate * n_clips * scale)))
    truncation_count = max(1, int(round(config.truncation_rate * n_clips * scale)))
    failure_count = max(1, int(round(config.tracking_failure_rate * n_clips * scale)))

    all_indices = np.arange(n_clips)
    duplicate_targets = rng.choice(all_indices, size=duplicate_count, replace=False)
    duplicate_sources: set[int] = set()
    by_pair: dict[tuple[int, str], list[int]] = {}
    for index, record in enumerate(records):
        by_pair.setdefault((int(record["label_id"]), str(record["signer"])), []).append(index)

    for target in duplicate_targets.tolist():
        record = records[target]
        candidates = by_pair[(int(record["label_id"]), str(record["signer"]))]
        source = next(
            (
                candidate
                for candidate in candidates
                if candidate != target and candidate not in duplicate_targets.tolist()
            ),
            candidates[0],
        )
        duplicate_sources.add(source)
        noise = rng.normal(0.0, config.duplicate_noise, size=clips[source].shape)
        copied = clips[source].astype(np.float32) + noise.astype(np.float32)
        clips[target] = copied.astype(np.float16)
        record["injected_defect"] = "duplicate"
        record["duplicate_of"] = records[source]["clip_id"]
        record["valid_length"] = int(records[source]["valid_length"])
        record["duration_frames"] = int(records[source]["duration_frames"])

    excluded = set(duplicate_targets.tolist()) | duplicate_sources
    candidates = np.array([i for i in range(n_clips) if i not in excluded], dtype=np.int64)
    mislabelled = rng.choice(candidates, size=mislabel_count, replace=False)
    excluded |= set(mislabelled.tolist())
    for index in mislabelled.tolist():
        record = records[index]
        true_gloss = str(record["true_label"])
        neighbor = CONFUSABLE_NEIGHBOR.get(true_gloss, GLOSSES[(int(record["true_label_id"]) + 1) % 60])
        record["label"] = neighbor
        record["label_id"] = GLOSS_TO_ID[neighbor]
        record["injected_defect"] = "mislabel"
        record["injected_mislabel"] = True

    candidates = np.array([i for i in range(n_clips) if i not in excluded], dtype=np.int64)
    truncated = rng.choice(candidates, size=truncation_count, replace=False)
    excluded |= set(truncated.tolist())
    for index in truncated.tolist():
        length = int(rng.integers(5, 12))
        clips[index, length:] = np.nan
        records[index]["valid_length"] = length
        records[index]["injected_defect"] = "truncated"

    candidates = np.array([i for i in range(n_clips) if i not in excluded], dtype=np.int64)
    failures = rng.choice(candidates, size=failure_count, replace=False)
    for index in failures.tolist():
        clips[index] = np.nan
        records[index]["valid_length"] = int(config.clip_frames)
        records[index]["injected_defect"] = "tracking_failure"

    LOGGER.info(
        "Injected defects: duplicates=%d mislabels=%d truncated=%d tracking_failures=%d",
        duplicate_count,
        mislabel_count,
        truncation_count,
        failure_count,
    )
    return records


def save_corpus(
    clips: np.ndarray,
    records: list[dict[str, Any]],
    config: DataConfig,
    *,
    write_individual: bool = True,
) -> dict[str, Any]:
    """Persist consolidated arrays, per-clip ``.npz`` files, and a JSON index."""
    data_dir = Path(config.data_dir)
    clips_dir = Path(config.clips_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "clips.npy", clips.astype(np.float16))
    valid_lengths = np.asarray([int(record["valid_length"]) for record in records], dtype=np.int16)
    np.save(data_dir / "valid_lengths.npy", valid_lengths)
    (data_dir / "index.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )

    if write_individual:
        for record, clip in zip(records, clips, strict=True):
            np.savez_compressed(
                clips_dir / f"{record['clip_id']}.npz",
                keypoints=clip.astype(np.float16),
                label_id=np.int32(record["label_id"]),
                signer=np.str_(record["signer"]),
                valid_length=np.int16(record["valid_length"]),
            )

    defect_summary: dict[str, int] = {}
    for record in records:
        defect = record.get("injected_defect")
        if defect:
            defect_summary[str(defect)] = defect_summary.get(str(defect), 0) + 1
    injected = {
        "total_clips": len(records),
        "features": FEATURE_DIM,
        "defects": defect_summary,
        "mislabel_clip_ids": [
            record["clip_id"] for record in records if record.get("injected_mislabel")
        ],
        "content_hash": content_hash_ids(record["clip_id"] for record in records),
    }
    (data_dir / "injected_defects.json").write_text(
        json.dumps(injected, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "clips": str(data_dir / "clips.npy"),
        "index": str(data_dir / "index.json"),
        "count": len(records),
        "defects": defect_summary,
    }


def load_corpus(config: DataConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load a previously generated consolidated corpus."""
    data_dir = Path(config.data_dir)
    index_path = data_dir / "index.json"
    clips_path = data_dir / "clips.npy"
    if not index_path.exists() or not clips_path.exists():
        raise FileNotFoundError("Corpus not found; run `signstream build-data` first")
    clips = np.load(clips_path, mmap_mode="r")
    records = json.loads(index_path.read_text(encoding="utf-8"))
    return clips, records


def corpus_content_hash(records: list[dict[str, Any]]) -> str:
    """Content hash over clip identifiers and their curated labels."""
    payload = "\n".join(
        f"{record['clip_id']}:{record['label_id']}" for record in sorted(records, key=lambda r: r["clip_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def nearest_confusable(gloss: str) -> str:
    """Return the deterministic confusable neighbor used for label-noise injection."""
    return CONFUSABLE_NEIGHBOR.get(gloss, GLOSSES[(GLOSS_TO_ID[gloss] + 1) % 60])


def descriptor_for(sequence: np.ndarray, length: int = 16) -> np.ndarray:
    """Compatibility helper used by tests and curation."""
    return sequence_descriptor(sequence, length)
