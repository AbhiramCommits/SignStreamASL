"""Tests for the 61-landmark schema and optional MediaPipe ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signstream.data.keypoints import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    LANDMARK_NAMES,
    MediaPipeExtractor,
    extract_keypoints_directory,
    normalize_sequence,
    resample_sequence,
    sequence_descriptor,
    validate_landmark_shape,
)


def _sequence(frames: int = 20) -> np.ndarray:
    rng = np.random.default_rng(4)
    seq = rng.normal(0.0, 0.2, size=(frames, LANDMARK_COUNT, 3)).astype(np.float32)
    left = np.array([-0.5, 0.1, 0.0], dtype=np.float32)
    right = np.array([0.5, 0.1, 0.0], dtype=np.float32)
    seq[:, 47] = left
    seq[:, 48] = right
    return seq


def test_schema_constants() -> None:
    assert len(LANDMARK_NAMES) == 61
    assert len(set(LANDMARK_NAMES)) == 61
    assert LANDMARK_COUNT == 61
    assert FEATURE_DIM == 183


def test_normalize_is_translation_and_scale_invariant() -> None:
    seq = _sequence()
    translated = seq + np.array([4.0, -3.0, 2.0], dtype=np.float32)
    scaled = seq * 2.5
    np.testing.assert_allclose(normalize_sequence(seq), normalize_sequence(translated), atol=1e-5)
    np.testing.assert_allclose(normalize_sequence(seq), normalize_sequence(scaled), atol=1e-5)
    normalized = normalize_sequence(seq)
    midpoint = (normalized[:, 47] + normalized[:, 48]) / 2
    np.testing.assert_allclose(midpoint, 0.0, atol=1e-5)


def test_resample_preserves_endpoints_and_length() -> None:
    seq = _sequence(13)
    target = resample_sequence(seq, 48)
    assert target.shape == (48, 61, 3)
    np.testing.assert_allclose(target[0], seq[0], atol=1e-6)
    np.testing.assert_allclose(target[-1], seq[-1], atol=1e-6)


def test_resample_all_nan_stays_nan() -> None:
    seq = np.full((5, 61, 3), np.nan, dtype=np.float32)
    result = resample_sequence(seq, 12)
    assert result.shape == (12, 61, 3)
    assert np.isnan(result).all()


def test_resample_constant_sequence() -> None:
    seq = np.ones((1, 61, 3), dtype=np.float32)
    result = resample_sequence(seq, 7)
    assert result.shape == (7, 61, 3)
    np.testing.assert_allclose(result, 1.0)


def test_resample_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        resample_sequence(np.zeros((4, 61, 2)), 8)
    with pytest.raises(ValueError):
        resample_sequence(_sequence(), 0)


def test_descriptor_is_unit_norm() -> None:
    descriptor = sequence_descriptor(_sequence(), 16)
    assert descriptor.shape == (16 * 183,)
    assert np.isclose(np.linalg.norm(descriptor), 1.0, atol=1e-6)


def test_validate_landmark_shape() -> None:
    validate_landmark_shape(_sequence(2))
    with pytest.raises(ValueError):
        validate_landmark_shape(np.zeros((2, 60, 3)))


class _FakeExtractor:
    def extract(self, path: Path) -> np.ndarray:
        return np.zeros((5, 61, 3), dtype=np.float32)


def test_extract_keypoints_directory_writes_ingest_log(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").write_bytes(b"fake")
    (videos / "b.mov").write_bytes(b"fake")
    output = tmp_path / "out"
    log = extract_keypoints_directory(videos, output, extractor=_FakeExtractor())
    assert log["count"] == 2
    assert log["ok"] == 2
    assert (output / "a.npz").exists()
    assert (output / "ingest_log.json").exists()


@pytest.mark.slow
def test_mediapipe_extractor_optional() -> None:
    mp = pytest.importorskip("mediapipe")
    if not hasattr(mp, "solutions"):
        pytest.skip("MediaPipe solutions API unavailable")
    extractor = MediaPipeExtractor()
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        keypoints = extractor.process_frame(frame)
        assert keypoints.shape == (61, 3)
    finally:
        extractor.close()
