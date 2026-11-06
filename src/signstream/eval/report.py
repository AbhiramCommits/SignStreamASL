"""Evaluation reports for isolated-sign checkpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from signstream.config import EvalConfig, data_config, eval_config
from signstream.eval.fairness import per_signer_report
from signstream.eval.gold import predict_for_ids
from signstream.eval.metrics import classification_bundle, write_confusion_csv, write_json

LOGGER = logging.getLogger(__name__)


def _markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    fairness = report["fairness"]
    lines = [
        f"# Evaluation: {report['model']} on `{report['split']}`",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test clips | {metrics['n']} |",
        f"| Top-1 accuracy | {metrics['top1']:.3f} |",
        f"| Top-5 accuracy | {metrics['top5']:.3f} |",
        f"| Macro F1 | {metrics['macro_f1']:.3f} |",
        f"| Micro F1 | {metrics['micro_f1']:.3f} |",
        f"| ECE (10 bins) | {metrics['ece']:.3f} |",
        f"| Fairness gap (best - worst signer top-1) | {fairness['fairness_gap']:.3f} |",
        f"| HIGH - LOW capture-quality top-1 gap | {fairness['high_low_top1_gap']:.3f} |",
        f"| Handedness top-1 gap | {fairness['handedness_gap']:.3f} |",
        "",
        "## Per-Signer Fairness",
        "",
        "| Signer | Quality | Handedness | N | Top-1 | Top-5 | Abstain | Mean confidence |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fairness["per_signer"]:
        lines.append(
            f"| {row['signer']} | {row['quality_tier']} | {row['handedness']} | {row['n']} | "
            f"{row['top1']:.3f} | {row['top5']:.3f} | {row['abstain_rate']:.3f} | "
            f"{row['mean_confidence']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Capture-Quality Tiers",
            "",
            "| Tier | Signers | N | Top-1 |",
            "|---|---|---:|---:|",
        ]
    )
    for tier, row in fairness["quality_tiers"].items():
        lines.append(
            f"| {tier} | {', '.join(row['signers'])} | {row['n']} | {row['top1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Confidence bin | Count | Avg confidence | Accuracy | Gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["reliability"]:
        lines.append(
            f"| {row['bin']} | {row['count']} | {row['avg_confidence']:.3f} | "
            f"{row['accuracy']:.3f} | {row['gap']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top-20 Confused Gloss Pairs",
            "",
            "| True | Predicted | Count |",
            "|---|---|---:|",
        ]
    )
    for row in metrics["top_confusions"]:
        lines.append(
            f"| {row['true_gloss']} | {row['predicted_gloss']} | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Class Metrics",
            "",
            "| Gloss | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics["per_class"]:
        lines.append(
            f"| {row['gloss']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['support']} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    split: str = "signer_holdout",
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a split and write JSON, Markdown, and CSV reports."""
    eval_cfg = config or eval_config()
    data_cfg = data_config()
    data_dir = Path(data_cfg.data_dir)
    split_payload = json.loads((data_dir / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    clip_ids = list(split_payload["test"])
    probabilities, y_true, records, payload = predict_for_ids(
        checkpoint, clip_ids, config=eval_cfg
    )
    class_names = list(payload.get("class_names", []))
    if len(class_names) != probabilities.shape[1]:
        raise ValueError("Checkpoint class list does not match model output width")
    bundle = classification_bundle(
        probabilities, y_true, class_names, n_bins=eval_cfg.n_bins
    )
    predictions = probabilities.argmax(axis=1)
    fairness = per_signer_report(
        records,
        y_true,
        predictions,
        probabilities,
        threshold=eval_cfg.confidence_threshold,
    )
    model_name = str(payload.get("model_name", Path(checkpoint).stem))
    reports = Path(eval_cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{model_name}_{split}"
    confusion_path = reports / f"confusion_{model_name}_{split}.csv"
    write_confusion_csv(
        np.asarray(bundle["confusion_matrix"], dtype=np.int64), class_names, confusion_path
    )

    report: dict[str, Any] = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "split": split,
        "manifest_hash": payload.get("manifest_hash"),
        "gold_content_hash": None,
        "n_test": int(len(y_true)),
        "metrics": {
            key: value
            for key, value in bundle.items()
            if key != "confusion_matrix"
        },
        "confusion_matrix_csv": str(confusion_path),
        "fairness": fairness,
        "class_names": class_names,
        "config": eval_cfg.to_dict(),
    }
    json_path = reports / f"{stem}.json"
    markdown_path = reports / f"{stem}.md"
    write_json(report, json_path)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    report["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
    LOGGER.info(
        "Evaluation: %s on %s top1=%.3f top5=%.3f macro_f1=%.3f fairness_gap=%.3f",
        model_name,
        split,
        bundle["top1"],
        bundle["top5"],
        bundle["macro_f1"],
        fairness["fairness_gap"],
    )
    return report
