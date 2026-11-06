"""Gold-set integrity and regression-gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from signstream.cli import main
from signstream.config import load_config
from signstream.eval.gold import load_gold_set
from signstream.eval.regression import evaluate_gate, gate_table


def test_gold_set_is_frozen_and_balanced() -> None:
    gold = load_gold_set()
    assert gold["n_clips"] == 300
    assert len(gold["content_hash"]) == 64
    assert gold["signer_distribution"] == {"S12": 120, "S13": 119, "S14": 61}
    assert gold["quality_tier_distribution"] == {"HIGH": 180, "LOW": 120}


def test_threshold_config_is_populated() -> None:
    config = load_config("configs/thresholds.yaml")
    baseline = config["gold"]["baseline"]
    assert baseline["top1"] > 0.5
    assert baseline["max_fairness_gap"] > 0.0
    assert config["gold"]["tolerance"]["top1_absolute"] == 0.02


def test_gate_passes_and_fails() -> None:
    actual = {"top1": 0.65, "top5": 0.90, "macro_f1": 0.64, "max_fairness_gap": 0.60}
    baseline = {"top1": 0.66, "top5": 0.91, "macro_f1": 0.65, "max_fairness_gap": 0.60}
    tolerance = {
        "top1_absolute": 0.02,
        "top5_absolute": 0.02,
        "macro_f1_absolute": 0.02,
        "max_fairness_gap_absolute": 0.05,
    }
    checks, failures = evaluate_gate(actual, baseline, tolerance)
    assert not failures
    assert all(check["passed"] for check in checks)
    assert "PASS" in gate_table(checks)

    bad = {**actual, "top1": 0.4, "max_fairness_gap": 0.9}
    checks, failures = evaluate_gate(bad, baseline, tolerance)
    assert set(failures) == {"top1", "max_fairness_gap"}
    assert "FAIL" in gate_table(checks)


def test_regress_missing_checkpoint_exits_2() -> None:
    assert main(["regress", "--checkpoint", "/definitely/missing.pt"]) == 2


@pytest.mark.slow
def test_regress_passes_with_fixture() -> None:
    if not Path("artifacts/data/manifest_v1.json").exists():
        pytest.skip("training data not built")
    checkpoint = Path("eval/fixtures/pose_transformer_signer_holdout.pt")
    if not checkpoint.exists():
        pytest.skip("fixture checkpoint missing")
    assert main(["regress", "--checkpoint", str(checkpoint)]) == 0
