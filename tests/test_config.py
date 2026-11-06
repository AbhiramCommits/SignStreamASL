"""Tests for typed YAML configuration and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from signstream.config import (
    ConfigError,
    DataConfig,
    EvalConfig,
    ModelConfig,
    data_config,
    eval_config,
    load_config,
    model_config,
)
from signstream.logging_utils import get_logger, setup_logging


def test_load_shipped_configs() -> None:
    assert data_config().num_glosses == 60
    assert model_config().d_model == 128
    assert len(eval_config().threshold_sweep) == 5


def test_unknown_keys_raise() -> None:
    with pytest.raises(ConfigError):
        DataConfig.from_dict({"seed": 1, "typo": 2})
    with pytest.raises(ConfigError):
        ModelConfig.from_dict({"num_layers": 2, "unknown": True})


def test_invalid_data_config() -> None:
    with pytest.raises(ConfigError):
        DataConfig(feature_dim=100)
    with pytest.raises(ConfigError):
        DataConfig(random_train=0.9)
    with pytest.raises(ConfigError):
        DataConfig(test_signers=["S12"], val_signers=["S12"])


def test_invalid_model_config() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(d_model=100, nhead=8)
    with pytest.raises(ConfigError):
        ModelConfig(label_budget_fractions=[0.1], label_budget_epochs=[1, 2])


def test_eval_config_validation() -> None:
    with pytest.raises(ConfigError):
        EvalConfig(n_bins=1)
    with pytest.raises(ConfigError):
        EvalConfig(window=0)
    with pytest.raises(ConfigError):
        EvalConfig(threshold_sweep=[])


def test_load_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("seed: 3\n", encoding="utf-8")
    assert load_config(path) == {"seed": 3}
    assert DataConfig.from_dict({"seed": 3}).to_dict()["seed"] == 3


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_setup_logging_is_idempotent(caplog: pytest.LogCaptureFixture) -> None:
    setup_logging("DEBUG", force=True)
    logger = get_logger("signstream.test")
    logger.debug("hello")
    assert logging.getLogger().level == logging.DEBUG
    assert any("hello" in record.message for record in caplog.records)
    with pytest.raises(ValueError):
        setup_logging("NOT-A-LEVEL")
