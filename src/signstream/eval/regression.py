"""Gold-set regression gate comparison logic."""

from __future__ import annotations

from typing import Any


def evaluate_gate(
    actual: dict[str, float],
    baseline: dict[str, Any],
    tolerance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare actual metrics to committed baselines and tolerances.

    Accuracy metrics use a minimum direction; the fairness gap uses a maximum
    direction so a regression toward inequity fails the gate.
    """
    directions = {
        "top1": "min",
        "top5": "min",
        "macro_f1": "min",
        "max_fairness_gap": "max",
    }
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for key, direction in directions.items():
        expected = float(baseline.get(key, 0.0))
        tol = float(tolerance.get(f"{key}_absolute", 0.02))
        value = float(actual[key])
        passed = value >= expected - tol if direction == "min" else value <= expected + tol
        if not passed:
            failures.append(key)
        checks.append(
            {
                "metric": key,
                "actual": value,
                "baseline": expected,
                "tolerance": tol,
                "direction": direction,
                "passed": bool(passed),
            }
        )
    return checks, failures


def gate_table(checks: list[dict[str, Any]]) -> str:
    """Render the gate as a Markdown table."""
    lines = [
        "| Metric | Actual | Baseline | Tolerance | Direction | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['metric']} | {check['actual']:.4f} | {check['baseline']:.4f} | "
            f"{check['tolerance']:.4f} | {check['direction']} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines)
