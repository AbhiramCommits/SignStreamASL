"""Per-signer fairness and abstention tests."""

from __future__ import annotations

import numpy as np
import pytest

from signstream.eval.fairness import per_signer_report


def _fixture():
    records = [
        {"signer": "A", "quality_tier": "LOW", "handedness": "left-dominant"},
        {"signer": "A", "quality_tier": "LOW", "handedness": "left-dominant"},
        {"signer": "B", "quality_tier": "HIGH", "handedness": "right-dominant"},
        {"signer": "B", "quality_tier": "HIGH", "handedness": "right-dominant"},
        {"signer": "C", "quality_tier": "HIGH", "handedness": "right-dominant"},
        {"signer": "C", "quality_tier": "HIGH", "handedness": "right-dominant"},
    ]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    probabilities = np.full((6, 3), 0.05, dtype=np.float32)
    probabilities[0] = [0.9, 0.05, 0.05]
    probabilities[1] = [0.4, 0.5, 0.1]
    probabilities[2] = [0.05, 0.9, 0.05]
    probabilities[3] = [0.05, 0.9, 0.05]
    probabilities[4] = [0.05, 0.05, 0.9]
    probabilities[5] = [0.4, 0.2, 0.4]
    return records, y_true, y_pred, probabilities


def test_per_signer_gaps_and_abstention() -> None:
    records, y_true, y_pred, probabilities = _fixture()
    report = per_signer_report(records, y_true, y_pred, probabilities, threshold=0.6)
    by_signer = {row["signer"]: row for row in report["per_signer"]}
    assert by_signer["A"]["top1"] == 0.5
    assert by_signer["B"]["top1"] == 1.0
    assert by_signer["C"]["top1"] == 0.5
    assert by_signer["A"]["abstain_rate"] == 0.5
    assert report["fairness_gap"] == 0.5
    assert report["best_signer"] == "B"
    assert report["high_low_top1_gap"] == pytest.approx(0.25)
    assert report["handedness_gap"] == pytest.approx(0.25)
    assert report["mean_abstain_rate"] == pytest.approx(1.0 / 3.0)


def test_mismatched_lengths_rejected() -> None:
    records, y_true, y_pred, probabilities = _fixture()
    with pytest.raises(ValueError):
        per_signer_report(records[:-1], y_true, y_pred, probabilities)
