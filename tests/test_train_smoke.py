"""Training-loop smoke tests and utility checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from signstream.config import ModelConfig
from signstream.models import PoseSequenceTransformer
from signstream.tracking import RunTracker
from signstream.train import (
    _stratified_subset,
    evaluate_logits,
    load_checkpoint_model,
    prepare_model_config,
    set_seed,
    top_k_correct,
    torch_predict,
)


def test_top_k_correct() -> None:
    logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 4.0, 3.0]])
    targets = torch.tensor([0, 2])
    assert top_k_correct(logits, targets, 1) == 1
    assert top_k_correct(logits, targets, 2) == 2


def test_stratified_subset_balanced() -> None:
    labels = [0] * 100 + [1] * 20
    subset = _stratified_subset(labels, 0.10, seed=2)
    assert len(subset) == 12
    assert sum(labels[index] == 0 for index in subset) == 10
    assert len(_stratified_subset(labels, 1.0, seed=2)) == len(labels)


def test_two_epoch_smoke_drives_loss_down() -> None:
    set_seed(5)
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(5), 30)
    features = rng.normal(0, 0.02, size=(len(labels), 16, 183)).astype(np.float32)
    features[:, :, 0] += labels[:, None] * 1.5
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels).long())
    loader = DataLoader(dataset, batch_size=50, shuffle=True)
    model = PoseSequenceTransformer(
        d_model=32, nhead=4, num_layers=1, ffn_dim=64, max_len=16, num_classes=5
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(2):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / len(loader))
    assert losses[-1] < losses[0] * 0.95


def test_evaluate_logits_and_predict() -> None:
    model = PoseSequenceTransformer(
        d_model=32, nhead=4, num_layers=1, ffn_dim=64, max_len=16, num_classes=5
    )
    features = torch.randn(6, 16, 183)
    targets = torch.randint(0, 5, (6,))
    loader = DataLoader(TensorDataset(features, targets), batch_size=3)
    metrics = evaluate_logits(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert metrics["n"] == 6
    assert 0.0 <= metrics["top1"] <= 1.0
    probabilities = torch_predict(model, features.numpy())
    assert probabilities.shape == (6, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_tracker_jsonl_fallback(tmp_path: Path) -> None:
    tracker = RunTracker("run", run_dir=tmp_path, enabled=False)
    tracker.start_run({"x": 1})
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.log_artifact(tmp_path / "missing")
    tracker.end_run()
    text = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"loss": 1.0' in text


def test_prepare_model_config_override() -> None:
    config = prepare_model_config(epochs=3, seed=9)
    assert config.epochs == 3
    assert config.seed == 9
    assert isinstance(config, ModelConfig)


def test_load_fixture_checkpoint() -> None:
    path = Path("eval/fixtures/pose_transformer_signer_holdout.pt")
    if not path.exists():
        import pytest

        pytest.skip("fixture checkpoint unavailable")
    model, payload = load_checkpoint_model(path)
    assert payload["class_names"][0] == "HELLO"
    assert model(torch.randn(1, 48, 183)).shape == (1, 60)


def test_tracker_mlflow_local_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    tracker = RunTracker(
        "mlflow_run", tracking_uri=str(tmp_path / "mlruns"), run_dir=tmp_path / "runs"
    )
    tracker.start_run({"lr": 0.1})
    tracker.log_params({"model": "tiny"})
    tracker.log_metrics({"top1": 0.5}, step=1)
    tracker.log_artifact(artifact)
    tracker.end_run()
    assert (tmp_path / "mlruns").exists()
    assert (tmp_path / "runs" / "mlflow_run" / "metrics.jsonl").exists()
