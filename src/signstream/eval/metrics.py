"""Classification metrics, calibration, and confusion diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def top_k_accuracy(logits: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Return top-k accuracy for a probability or logit matrix ``[N, C]``."""
    scores = np.asarray(logits)
    targets = np.asarray(y)
    if scores.ndim != 2 or targets.ndim != 1 or scores.shape[0] != targets.shape[0]:
        raise ValueError("logits must be [N, C] and y must be [N]")
    if scores.shape[0] == 0:
        return 0.0
    k = max(1, min(k, scores.shape[1]))
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean((top == targets[:, None]).any(axis=1)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1."""
    return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Micro-averaged F1."""
    return float(precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)[2])


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Return precision, recall, F1, and support for every class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return [
        {
            "class_id": index,
            "gloss": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    ]


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute an integer confusion matrix with a fixed class range."""
    return sklearn_confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    ).astype(np.int64)


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the most frequent off-diagonal confusion pairs."""
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    pairs = []
    for true_index in range(len(class_names)):
        for pred_index in range(len(class_names)):
            count = int(matrix[true_index, pred_index])
            if true_index != pred_index and count > 0:
                pairs.append(
                    {
                        "true_gloss": class_names[true_index],
                        "predicted_gloss": class_names[pred_index],
                        "count": count,
                    }
                )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute ECE and a reliability-curve table using confidence bins."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(y_true)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        avg_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(avg_confidence - accuracy)
        ece += (count / len(targets)) * gap
        rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return float(ece), rows


def classification_bundle(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Assemble all isolated-sign metrics into one JSON-ready dictionary."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(y_true)
    predictions = probabilities.argmax(axis=1)
    ece, reliability = expected_calibration_error(probabilities, targets, n_bins=n_bins)
    return {
        "n": int(len(targets)),
        "top1": top_k_accuracy(probabilities, targets, 1),
        "top5": top_k_accuracy(probabilities, targets, 5),
        "macro_f1": macro_f1(targets, predictions),
        "micro_f1": micro_f1(targets, predictions),
        "ece": ece,
        "reliability": reliability,
        "per_class": per_class_report(targets, predictions, class_names),
        "top_confusions": top_confusions(targets, predictions, class_names),
        "confusion_matrix": confusion_matrix(targets, predictions, len(class_names)).tolist(),
    }


def write_confusion_csv(
    matrix: np.ndarray,
    class_names: list[str],
    path: str | Path,
) -> Path:
    """Write a square confusion matrix CSV with class-name headers."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for index, name in enumerate(class_names):
            writer.writerow([name, *[int(value) for value in matrix[index]]])
    return output


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact with deterministic formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
