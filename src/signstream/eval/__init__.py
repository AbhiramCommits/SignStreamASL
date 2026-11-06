"""Evaluation, calibration, fairness, and gold-set regression utilities."""

from __future__ import annotations

from signstream.eval.fairness import per_signer_report
from signstream.eval.gold import build_gold_set, evaluate_gold, load_gold_set, predict_for_ids
from signstream.eval.metrics import (
    classification_bundle,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    micro_f1,
    per_class_report,
    top_confusions,
    top_k_accuracy,
)
from signstream.eval.report import evaluate_checkpoint

__all__ = [
    "build_gold_set",
    "classification_bundle",
    "confusion_matrix",
    "evaluate_checkpoint",
    "evaluate_gold",
    "expected_calibration_error",
    "load_gold_set",
    "macro_f1",
    "micro_f1",
    "per_class_report",
    "per_signer_report",
    "predict_for_ids",
    "top_confusions",
    "top_k_accuracy",
]
