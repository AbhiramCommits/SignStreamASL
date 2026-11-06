"""Signer-level fairness and accessibility analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def _top_k(probabilities: np.ndarray, target: int, k: int) -> bool:
    k = min(k, probabilities.shape[0])
    return bool(target in np.argsort(-probabilities)[:k])


def per_signer_report(
    records: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.60,
) -> dict[str, Any]:
    """Compute per-signer top-1/top-5, abstention, and fairness gaps.

    Args:
        records: Metadata parallel to ``y_true``.
        y_true: Gold class ids.
        y_pred: Predicted class ids.
        probabilities: Class probabilities ``[N, C]``.
        threshold: Confidence below this counts as an abstention.

    Returns:
        JSON-ready dict with per-signer metrics and tier/handedness gaps.
    """
    if not (len(records) == len(y_true) == len(y_pred) == len(probabilities)):
        raise ValueError("records, y_true, y_pred, and probabilities must align")
    groups: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        signer = str(record["signer"])
        group = groups.setdefault(
            signer,
            {
                "signer": signer,
                "quality_tier": str(record.get("quality_tier", "UNKNOWN")),
                "handedness": str(record.get("handedness", "UNKNOWN")),
                "indices": [],
            },
        )
        group["indices"].append(index)

    per_signer: list[dict[str, Any]] = []
    for signer in sorted(groups):
        group = groups[signer]
        indices = np.asarray(group["indices"], dtype=np.int64)
        targets = np.asarray(y_true)[indices]
        predictions = np.asarray(y_pred)[indices]
        probs = np.asarray(probabilities)[indices]
        top5 = float(np.mean([_top_k(row, int(target), 5) for row, target in zip(probs, targets, strict=False)]))
        confidence = probs.max(axis=1)
        per_signer.append(
            {
                "signer": signer,
                "quality_tier": group["quality_tier"],
                "handedness": group["handedness"],
                "n": int(len(indices)),
                "top1": float(np.mean(predictions == targets)),
                "top5": top5,
                "abstain_rate": float(np.mean(confidence < threshold)),
                "mean_confidence": float(np.mean(confidence)),
            }
        )

    top1_values = [row["top1"] for row in per_signer if row["n"] > 0]
    fairness_gap = float(max(top1_values) - min(top1_values)) if top1_values else 0.0
    best_signer = max(per_signer, key=lambda row: row["top1"])["signer"] if per_signer else None
    worst_signer = min(per_signer, key=lambda row: row["top1"])["signer"] if per_signer else None

    tier_rows: dict[str, dict[str, Any]] = {}
    for tier in sorted({row["quality_tier"] for row in per_signer}):
        rows = [row for row in per_signer if row["quality_tier"] == tier]
        tier_rows[tier] = {
            "signers": [row["signer"] for row in rows],
            "n": int(sum(row["n"] for row in rows)),
            "top1": float(
                sum(row["top1"] * row["n"] for row in rows) / max(1, sum(row["n"] for row in rows))
            ),
            "min_signer_top1": float(min(row["top1"] for row in rows)),
            "max_signer_top1": float(max(row["top1"] for row in rows)),
        }
    high_low_gap = (
        float(tier_rows["HIGH"]["top1"] - tier_rows["LOW"]["top1"])
        if "HIGH" in tier_rows and "LOW" in tier_rows
        else 0.0
    )

    handedness_rows: dict[str, dict[str, Any]] = {}
    for handedness in sorted({row["handedness"] for row in per_signer}):
        rows = [row for row in per_signer if row["handedness"] == handedness]
        handedness_rows[handedness] = {
            "signers": [row["signer"] for row in rows],
            "n": int(sum(row["n"] for row in rows)),
            "top1": float(
                sum(row["top1"] * row["n"] for row in rows) / max(1, sum(row["n"] for row in rows))
            ),
        }
    handedness_gap = (
        abs(
            handedness_rows["left-dominant"]["top1"]
            - handedness_rows["right-dominant"]["top1"]
        )
        if "left-dominant" in handedness_rows and "right-dominant" in handedness_rows
        else 0.0
    )

    return {
        "per_signer": per_signer,
        "fairness_gap": fairness_gap,
        "best_signer": best_signer,
        "worst_signer": worst_signer,
        "quality_tiers": tier_rows,
        "high_low_top1_gap": high_low_gap,
        "handedness": handedness_rows,
        "handedness_gap": float(handedness_gap),
        "abstain_threshold": float(threshold),
        "mean_abstain_rate": float(np.mean([row["abstain_rate"] for row in per_signer]))
        if per_signer
        else 0.0,
    }
