"""Fast integration tests for data loading, training, and continuous data."""

from __future__ import annotations

from pathlib import Path

import torch

from signstream.config import EvalConfig, ModelConfig
from signstream.data.dataset import SignClipDataset
from signstream.stream.windowing import build_continuous_dataset, load_continuous_sentence
from signstream.train import train_model


def test_dataset_for_split_and_augmentation(tiny_data, monkeypatch) -> None:
    config, _ = tiny_data
    dataset = SignClipDataset.for_split(config, "signer_holdout", "train", augment=True)
    features, label = dataset[0]
    assert features.shape == (48, 183)
    assert 0 <= label.item() < 60
    assert len(dataset) == 60


def test_train_model_tiny_end_to_end(tiny_data, monkeypatch, tmp_path: Path) -> None:
    data_cfg, _ = tiny_data
    monkeypatch.setattr("signstream.config.data_config", lambda: data_cfg)
    model_cfg = ModelConfig(
        seed=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
        batch_size=64,
        epochs=2,
        patience=2,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        reports_dir=str(tmp_path / "reports"),
        runs_dir=str(tmp_path / "runs"),
        mlflow_uri=str(tmp_path / "mlruns"),
    )
    report = train_model(
        model_cfg,
        model_name="pose_transformer",
        split="signer_holdout",
        epochs=2,
        seed=4,
        compute_test=True,
        progress=False,
        enable_mlflow=False,
    )
    assert report["history"]
    assert report["train_examples"] == 60
    assert "test" in report
    assert Path(report["checkpoint"]).exists()
    assert (tmp_path / "reports" / "train_pose_transformer_signer_holdout.json").exists()


def test_train_linear_probe_tiny(tiny_data, monkeypatch, tmp_path: Path) -> None:
    data_cfg, _ = tiny_data
    monkeypatch.setattr("signstream.config.data_config", lambda: data_cfg)
    model_cfg = ModelConfig(
        seed=2,
        batch_size=64,
        epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        reports_dir=str(tmp_path / "reports"),
        runs_dir=str(tmp_path / "runs"),
        mlflow_uri=str(tmp_path / "mlruns"),
    )
    report = train_model(
        model_cfg,
        model_name="linear_probe",
        split="signer_holdout",
        epochs=1,
        compute_test=True,
        progress=False,
        enable_mlflow=False,
    )
    assert report["best_val"]["top1"] >= 0.0
    assert Path(report["checkpoint"]).exists()


def test_build_continuous_dataset_tiny(tiny_data, tmp_path: Path) -> None:
    data_cfg, _ = tiny_data
    eval_cfg = EvalConfig(seed=2, reports_dir=str(tmp_path / "reports"))
    index = build_continuous_dataset(config=data_cfg, eval_cfg=eval_cfg, force=True)
    assert index["total_sentences"] == 3
    assert index["total_signs"] >= 6
    assert (Path(data_cfg.continuous_dir) / "index.json").exists()
    first = index["sentences"][0]
    sentence = load_continuous_sentence(Path(data_cfg.continuous_dir) / first["file"])
    assert sentence["keypoints"].ndim == 3
    assert sentence["gold_spans"].shape[1] == 2
    reused = build_continuous_dataset(config=data_cfg, eval_cfg=eval_cfg)
    assert reused["total_sentences"] == 3


def test_torch_fixture_attention_export_shape() -> None:
    path = Path("eval/fixtures/pose_transformer_signer_holdout.pt")
    if not path.exists():
        import pytest

        pytest.skip("fixture missing")
    from signstream.train import load_checkpoint_model

    model, _ = load_checkpoint_model(path)
    attention = model.temporal_attention_weights(torch.randn(2, 48, 183))
    assert attention.shape == (2, 48, 48)
