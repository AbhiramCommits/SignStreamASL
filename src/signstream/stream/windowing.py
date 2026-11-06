"""Sliding-window construction and synthetic continuous-signing evaluation data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from signstream.config import DataConfig, EvalConfig, data_config, eval_config
from signstream.data.keypoints import normalize_sequence, resample_sequence

LOGGER = logging.getLogger(__name__)


def iter_windows(
    sequence: np.ndarray,
    *,
    window: int = 48,
    stride: int = 8,
) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` windows covering a sequence."""
    array = np.asarray(sequence)
    if array.ndim not in (2, 3):
        raise ValueError("sequence must be [T, 183] or [T, 61, 3]")
    total = array.shape[0]
    if total == 0:
        return []
    if total <= window:
        return [(0, total)]
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] + window < total:
        starts.append(total - window)
    return [(start, start + window) for start in starts]


def window_array(
    sequence: np.ndarray,
    starts: list[tuple[int, int]],
    *,
    window: int = 48,
) -> np.ndarray:
    """Extract fixed-length windows, zero-padding a short tail."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim == 3:
        array = array.reshape(array.shape[0], -1)
    output = np.zeros((len(starts), window, array.shape[1]), dtype=np.float32)
    for row, (start, end) in enumerate(starts):
        chunk = array[start:end]
        output[row, : chunk.shape[0]] = chunk
    return output


def _load_normalized_clip(clips: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    clip = np.asarray(clips[int(record["_row_index"])], dtype=np.float32)
    valid_length = int(record.get("valid_length", clip.shape[0]))
    clip = clip[: max(1, min(valid_length, clip.shape[0]))]
    clip = resample_sequence(clip, 48)
    clip = normalize_sequence(clip)
    return np.nan_to_num(clip, nan=0.0)


def _rest_frames(reference: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    frames = np.repeat(reference[None, :, :], count, axis=0)
    return frames + rng.normal(0.0, 0.002, size=frames.shape).astype(np.float32)


def _transition_frames(
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    count: int,
) -> np.ndarray:
    blend = np.linspace(0.0, 1.0, count + 2, dtype=np.float32)[1:-1, None, None]
    return (start_frame[None, :, :] * (1.0 - blend) + end_frame[None, :, :] * blend).astype(
        np.float32
    )


def build_continuous_dataset(
    *,
    config: DataConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Concatenate held-out isolated signs into 150 timestamped sentences."""
    data_cfg = config or data_config()
    eval_cfg_ = eval_cfg or eval_config()
    data_dir = Path(data_cfg.data_dir)
    output_dir = Path(data_cfg.continuous_dir)
    index_path = output_dir / "index.json"
    if (
        not force
        and index_path.exists()
    ):
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if len(existing.get("sentences", [])) == data_cfg.continuous_sentences:
            LOGGER.info("Reusing continuous dataset at %s", output_dir)
            return existing

    clips = np.load(data_dir / "clips.npy", mmap_mode="r")
    records = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    for row, record in enumerate(records):
        record["_row_index"] = row
    split = json.loads((data_dir / "splits" / "signer_holdout.json").read_text(encoding="utf-8"))
    record_map = {record["clip_id"]: record for record in records}
    test_records = [record_map[clip_id] for clip_id in split["test"]]
    rng = np.random.default_rng(eval_cfg_.seed + 431)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_metadata: list[dict[str, Any]] = []
    for sentence_index in range(data_cfg.continuous_sentences):
        n_signs = int(
            rng.integers(data_cfg.continuous_min_signs, data_cfg.continuous_max_signs + 1)
        )
        chosen = rng.choice(len(test_records), size=n_signs, replace=False)
        parts: list[np.ndarray] = []
        gold_ids: list[int] = []
        gold_spans: list[list[float]] = []
        signers: list[str] = []
        cursor = 0
        first_clip = _load_normalized_clip(clips, test_records[int(chosen[0])])
        lead = _rest_frames(first_clip[0], data_cfg.continuous_rest_frames, rng)
        parts.append(lead)
        cursor += len(lead)
        previous = first_clip[0]

        for choice in chosen.tolist():
            record = test_records[int(choice)]
            clip = _load_normalized_clip(clips, record)
            transition = _transition_frames(previous, clip[0], data_cfg.continuous_rest_frames)
            parts.append(transition)
            cursor += len(transition)
            span_start = cursor / data_cfg.continuous_fps
            parts.append(clip)
            cursor += clip.shape[0]
            span_end = cursor / data_cfg.continuous_fps
            gold_ids.append(int(record["label_id"]))
            gold_spans.append([float(span_start), float(span_end)])
            signers.append(str(record["signer"]))
            previous = clip[-1]

        sentence = np.concatenate(parts, axis=0).astype(np.float16)
        filename = f"sentence_{sentence_index:03d}.npz"
        np.savez_compressed(
            output_dir / filename,
            keypoints=sentence,
            gold_ids=np.asarray(gold_ids, dtype=np.int16),
            gold_spans=np.asarray(gold_spans, dtype=np.float32),
            fps=np.int32(data_cfg.continuous_fps),
        )
        sentence_metadata.append(
            {
                "sentence_id": f"sentence_{sentence_index:03d}",
                "file": filename,
                "num_frames": int(sentence.shape[0]),
                "duration_s": float(sentence.shape[0] / data_cfg.continuous_fps),
                "gold_ids": gold_ids,
                "gold_spans": gold_spans,
                "signers": signers,
            }
        )

    index = {
        "schema": "signstream-continuous-v1",
        "split": "signer_holdout",
        "fps": data_cfg.continuous_fps,
        "seed": eval_cfg_.seed,
        "sentences": sentence_metadata,
        "total_sentences": len(sentence_metadata),
        "total_signs": sum(len(item["gold_ids"]) for item in sentence_metadata),
        "total_frames": sum(item["num_frames"] for item in sentence_metadata),
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "Built continuous dataset: %d sentences, %d signs, %d frames",
        index["total_sentences"],
        index["total_signs"],
        index["total_frames"],
    )
    return index


def load_continuous_sentence(path: str | Path) -> dict[str, Any]:
    """Load one continuous sentence archive."""
    payload = np.load(Path(path), allow_pickle=False)
    return {
        "keypoints": np.asarray(payload["keypoints"], dtype=np.float32),
        "gold_ids": np.asarray(payload["gold_ids"], dtype=np.int64),
        "gold_spans": np.asarray(payload["gold_spans"], dtype=np.float32),
        "fps": int(payload["fps"]),
    }
