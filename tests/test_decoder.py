"""Windowing, NMS, abstention, and continuous-decoding tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signstream.stream.decoder import (
    ABSTAIN_TOKEN,
    GlossEvent,
    SlidingWindowDecoder,
    edit_distance,
    evaluate_continuous,
    match_events,
)
from signstream.stream.windowing import iter_windows, load_continuous_sentence, window_array


def _windows(count: int, stride: int = 8, window: int = 48) -> list[tuple[int, int]]:
    return [(index * stride, index * stride + window) for index in range(count)]


def _probabilities(count: int, segments: dict[int, int], classes: int = 4) -> np.ndarray:
    probabilities = np.full((count, classes), 0.03, dtype=np.float32)
    probabilities[:, 3] = 0.40
    for start, (end, label) in segments.items():
        probabilities[start:end, :] = 0.03
        probabilities[start:end, label] = 0.91
    return probabilities


def test_window_helpers() -> None:
    windows = iter_windows(np.zeros((120, 61, 3)), window=48, stride=8)
    assert windows[0] == (0, 48)
    assert windows[-1] == (72, 120)
    arrays = window_array(np.ones((120, 61, 3)), windows)
    assert arrays.shape == (len(windows), 48, 183)
    assert iter_windows(np.zeros((10, 61, 3)), window=48) == [(0, 10)]


def test_decoder_emits_ordered_glosses() -> None:
    decoder = SlidingWindowDecoder(
        lambda batch: np.zeros((len(batch), 4)), ["A", "B", "C", "REST"], threshold=0.6
    )
    probabilities = _probabilities(
        30,
        {2: (7, 0), 10: (15, 1), 18: (23, 2)},
    )
    result = decoder._decode_probabilities(probabilities, _windows(30), 0.6)
    assert [event.gloss for event in result.events] == ["A", "B", "C"]
    assert result.abstain_rate > 0
    assert result.to_dict()["abstain_token"] == ABSTAIN_TOKEN


def test_decoder_abstains_below_threshold() -> None:
    decoder = SlidingWindowDecoder(
        lambda batch: np.zeros((len(batch), 4)), ["A", "B", "C", "D"], threshold=0.6
    )
    probabilities = np.full((10, 4), 0.25, dtype=np.float32)
    result = decoder._decode_probabilities(probabilities, _windows(10), 0.6)
    assert result.events == []
    assert result.abstain_rate == 1.0
    assert result.abstain_segments_s


def test_nms_and_cooldown_suppress_duplicate_gloss() -> None:
    decoder = SlidingWindowDecoder(
        lambda batch: np.zeros((len(batch), 4)), ["A", "B", "C", "D"], threshold=0.6
    )
    probabilities = _probabilities(30, {1: (5, 0), 7: (11, 0)})
    result = decoder._decode_probabilities(probabilities, _windows(30), 0.6)
    assert len([event for event in result.events if event.gloss == "A"]) == 1


def test_public_decode_wrapper() -> None:
    def predictor(batch: np.ndarray) -> np.ndarray:
        output = np.zeros((len(batch), 3), dtype=np.float32)
        output[:, 0] = 0.95
        output[:, 1] = 0.03
        output[:, 2] = 0.02
        return output

    decoder = SlidingWindowDecoder(predictor, ["A", "B", "C"], threshold=0.6)
    result = decoder.decode(np.zeros((100, 61, 3), dtype=np.float32))
    assert len(result.events) >= 1
    assert result.events[0].gloss == "A"
    no_windows = decoder.decode(np.zeros((0, 61, 3)), threshold=0.6)
    assert no_windows.events == []
    assert no_windows.to_dict()["abstain_token"] == ABSTAIN_TOKEN


def test_match_events_and_edit_distance() -> None:
    events = [
        GlossEvent("A", 0, 0.2, 0.7, 0.9),
        GlossEvent("B", 1, 5.0, 5.5, 0.8),
    ]
    matches, predicted_total, gold_total = match_events(
        events, [0, 1], [[0.0, 1.0], [4.0, 6.0]], class_names=["A", "B"]
    )
    assert (matches, predicted_total, gold_total) == (2, 2, 2)
    assert edit_distance([0, 1], [0, 1]) == 0
    assert edit_distance([0], [0, 1]) == 1
    assert edit_distance([], [1, 2]) == 2


def test_continuous_evaluation_with_tiny_fixture(tmp_path: Path) -> None:
    directory = tmp_path / "continuous"
    directory.mkdir()
    sentence = np.zeros((180, 61, 3), dtype=np.float16)
    np.savez_compressed(
        directory / "sentence_000.npz",
        keypoints=sentence,
        gold_ids=np.asarray([0, 1], dtype=np.int16),
        gold_spans=np.asarray([[0.0, 1.6], [2.0, 3.6]], dtype=np.float32),
        fps=np.int32(30),
    )
    index = {
        "sentences": [
            {
                "sentence_id": "sentence_000",
                "file": "sentence_000.npz",
                "num_frames": 180,
                "gold_ids": [0, 1],
                "gold_spans": [[0.0, 1.6], [2.0, 3.6]],
                "signers": ["S12", "S14"],
            }
        ]
    }
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")

    def predictor(batch: np.ndarray) -> np.ndarray:
        output = np.zeros((len(batch), 2), dtype=np.float32)
        output[:, 0] = 0.9
        output[:, 1] = 0.1
        return output

    decoder = SlidingWindowDecoder(predictor, ["A", "B"], threshold=0.6)
    payload = evaluate_continuous(
        decoder, continuous_dir=directory, thresholds=[0.3, 0.6], primary_threshold=0.6
    )
    assert payload["n_sentences"] == 1
    assert len(payload["threshold_sweep"]) == 2
    assert "f1" in payload["primary"]
    loaded = load_continuous_sentence(directory / "sentence_000.npz")
    assert loaded["keypoints"].shape == (180, 61, 3)
