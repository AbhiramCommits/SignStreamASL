"""PyTorch dataset for isolated signing clips with train-time augmentation."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from signstream.config import DataConfig
from signstream.data.keypoints import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    normalize_sequence,
    resample_sequence,
)

LOGGER = logging.getLogger(__name__)

_POSE_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
)


def mirror_sequence(sequence: np.ndarray) -> np.ndarray:
    """Mirror a signing sequence in x and swap left/right body parts."""
    mirrored = np.asarray(sequence, dtype=np.float32).copy()
    mirrored[..., 0] *= -1.0
    mirrored[:, :21], mirrored[:, 21:42] = (
        mirrored[:, 21:42].copy(),
        mirrored[:, :21].copy(),
    )
    for left, right in _POSE_SWAP_PAIRS:
        mirrored[:, 42 + left], mirrored[:, 42 + right] = (
            mirrored[:, 42 + right].copy(),
            mirrored[:, 42 + left].copy(),
        )
    return mirrored


class SignClipDataset(Dataset):
    """Return ``(x[48,183], y)`` tensors for a named split."""

    def __init__(
        self,
        clips: np.ndarray,
        records: list[dict[str, Any]],
        clip_ids: Sequence[str],
        *,
        target_len: int = 48,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.clips = clips
        self.records_by_id = {str(record["clip_id"]): record for record in records}
        self.clip_ids = list(clip_ids)
        self.target_len = int(target_len)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.labels = [int(self.records_by_id[clip_id]["label_id"]) for clip_id in self.clip_ids]
        if not self.clip_ids:
            raise ValueError("Dataset split is empty")
        # Index JSON preserves the row order of clips.npy.
        self._row_index = {str(record["clip_id"]): row for row, record in enumerate(records)}

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _load_raw(self, index: int) -> tuple[np.ndarray, int]:
        clip_id = self.clip_ids[index]
        record = self.records_by_id[clip_id]
        source_index = self._row_index[clip_id]
        clip = np.asarray(self.clips[source_index], dtype=np.float32)
        valid_length = int(record.get("valid_length", clip.shape[0]))
        sequence = clip[: max(1, min(valid_length, clip.shape[0]))]
        return sequence, valid_length

    def _augment(self, sequence: np.ndarray, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 7919 * index)
        if rng.random() < 0.5:
            sequence = mirror_sequence(sequence)
        if rng.random() < 0.8:
            span = int(rng.integers(max(36, self.target_len - 8), self.target_len + 1))
            start = int(rng.integers(0, max(1, sequence.shape[0] - span + 1)))
            sequence = resample_sequence(sequence[start : start + span], self.target_len)
        angle = float(rng.uniform(-0.14, 0.14))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = sequence[..., 0].copy()
        y = sequence[..., 1].copy()
        sequence[..., 0] = cos_a * x - sin_a * y
        sequence[..., 1] = sin_a * x + cos_a * y
        sequence = sequence * float(rng.uniform(0.96, 1.04))
        sequence = sequence + rng.normal(0.0, 0.004, size=sequence.shape).astype(np.float32)
        if rng.random() < 0.5:
            landmarks = rng.choice(LANDMARK_COUNT, size=6, replace=False)
            sequence[:, landmarks, :] = 0.0
        return sequence.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _ = self._load_raw(index)
        sequence = resample_sequence(sequence, self.target_len)
        sequence = normalize_sequence(sequence)
        if self.augment:
            sequence = self._augment(sequence, index)
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        features = sequence.reshape(self.target_len, FEATURE_DIM).astype(np.float32)
        label = int(self.records_by_id[self.clip_ids[index]]["label_id"])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    @classmethod
    def for_split(
        cls,
        config: DataConfig,
        split: str,
        subset: str = "train",
        *,
        augment: bool = False,
        target_len: int | None = None,
    ) -> SignClipDataset:
        """Load one partition (``train``/``val``/``test``) of a split."""
        directory = Path(config.data_dir)
        clips_path = directory / "clips.npy"
        index_path = directory / "index.json"
        split_path = directory / "splits" / f"{split}.json"
        for path in (clips_path, index_path, split_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact {path}; run `signstream build-data`")
        clips = np.load(clips_path, mmap_mode="r")
        records = json.loads(index_path.read_text(encoding="utf-8"))
        for row, record in enumerate(records):
            record["_row_index"] = row
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        dataset = cls(
            clips,
            records,
            split_payload[subset],
            target_len=target_len or config.seq_len,
            augment=augment,
            seed=config.seed,
        )
        return dataset


def collate_fn(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-free fixed-size examples into a batch."""
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return inputs, targets


def load_records(config: DataConfig) -> list[dict[str, Any]]:
    """Load the corpus metadata index."""
    path = Path(config.data_dir) / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing index {path}; run `signstream build-data`")
    return json.loads(path.read_text(encoding="utf-8"))


def load_clip_ids(config: DataConfig, split: str, subset: str) -> list[str]:
    """Load one partition's clip identifiers."""
    path = Path(config.data_dir) / "splits" / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload[subset])
