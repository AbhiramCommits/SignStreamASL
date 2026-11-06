"""Hand-computed metric and calibration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signstream.eval.metrics import (
    classification_bundle,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    micro_f1,
    per_class_report,
    top_confusions,
    top_k_accuracy,
    write_confusion_csv,
    write_json,
)


def test_top_k_accuracy_fixture() -> None:
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )
    targets = np.array([0, 1, 2, 0])
    assert top_k_accuracy(probabilities, targets, 1) == 0.75
    assert top_k_accuracy(probabilities, targets, 3) == 1.0
    assert top_k_accuracy(probabilities[:0], targets[:0], 1) == 0.0


def test_f1_and_confusion_fixture() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    assert np.isclose(macro_f1(y_true, y_pred), 5.0 / 9.0)
    assert np.isclose(micro_f1(y_true, y_pred), 2.0 / 3.0)
    matrix = confusion_matrix(y_true, y_pred, 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    report = per_class_report(y_true, y_pred, ["A", "B", "C"])
    assert report[1]["f1"] == 2.0 / 3.0
    assert report[2]["support"] == 1


def test_top_confusions_fixture() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    pairs = top_confusions(y_true, y_pred, ["A", "B"], limit=2)
    assert pairs[0] == {"true_gloss": "A", "predicted_gloss": "B", "count": 2}


def test_calibration_perfect_and_overconfident() -> None:
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    ece, rows = expected_calibration_error(perfect, np.array([0, 1]), n_bins=10)
    assert ece == 0.0
    assert len(rows) == 10
    overconfident = np.array([[0.9, 0.1], [0.9, 0.1]])
    ece, _ = expected_calibration_error(overconfident, np.array([1, 1]), n_bins=10)
    assert np.isclose(ece, 0.9)


def test_bundle_and_artifact_writes(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet(np.ones(4), size=20)
    targets = rng.integers(0, 4, size=20)
    bundle = classification_bundle(probabilities, targets, ["A", "B", "C", "D"], n_bins=5)
    assert set(["top1", "top5", "macro_f1", "ece", "per_class", "top_confusions"]) <= set(bundle)
    csv_path = write_confusion_csv(np.asarray(bundle["confusion_matrix"]), ["A", "B", "C", "D"], tmp_path / "c.csv")
    assert csv_path.exists()
    json_path = write_json({"ok": True}, tmp_path / "x.json")
    assert json_path.exists()
