"""Tests for the PyTorch clip dataset and augmentation."""

from __future__ import annotations

import numpy as np
import torch

from signstream.data.dataset import SignClipDataset, collate_fn, mirror_sequence


def _make_clip(offset: float) -> np.ndarray:
    frames = 48
    clip = np.zeros((frames, 61, 3), dtype=np.float16)
    clip[:, :, 0] = np.linspace(0, 1, frames)[:, None] + offset
    clip[:, 47, 0] = -0.5
    clip[:, 48, 0] = 0.5
    return clip


def _records() -> list[dict]:
    return [
        {
            "clip_id": "a",
            "label_id": 0,
            "valid_length": 48,
            "signer": "S01",
        },
        {
            "clip_id": "b",
            "label_id": 1,
            "valid_length": 48,
            "signer": "S02",
        },
    ]


def test_dataset_shapes_and_labels() -> None:
    clips = np.stack([_make_clip(0.0), _make_clip(0.5)], axis=0)
    dataset = SignClipDataset(clips, _records(), ["a", "b"])
    features, label = dataset[0]
    assert features.shape == (48, 183)
    assert features.dtype == torch.float32
    assert label.item() == 0
    assert len(dataset) == 2


def test_dataloader_collate() -> None:
    clips = np.stack([_make_clip(0.0), _make_clip(0.5)], axis=0)
    dataset = SignClipDataset(clips, _records(), ["a", "b"])
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch[0].shape == (2, 48, 183)
    assert batch[1].tolist() == [0, 1]


def test_augmentation_is_deterministic() -> None:
    clips = np.stack([_make_clip(0.0), _make_clip(0.2)], axis=0)
    first = SignClipDataset(clips, _records(), ["a", "b"], augment=True, seed=5)
    second = SignClipDataset(clips, _records(), ["a", "b"], augment=True, seed=5)
    np.testing.assert_allclose(first[0][0].numpy(), second[0][0].numpy(), atol=1e-6)


def test_mirror_swaps_hands_and_flips_x() -> None:
    sequence = np.zeros((2, 61, 3), dtype=np.float32)
    sequence[:, 0] = [0.1, 0.2, 0.3]
    sequence[:, 21] = [0.4, 0.5, 0.6]
    mirrored = mirror_sequence(sequence)
    np.testing.assert_allclose(mirrored[0, 0], [-0.4, 0.5, 0.6])
    np.testing.assert_allclose(mirrored[0, 21], [-0.1, 0.2, 0.3])


def test_empty_dataset_rejected() -> None:
    clips = np.stack([_make_clip(0.0)], axis=0)
    try:
        SignClipDataset(clips, _records(), [])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
