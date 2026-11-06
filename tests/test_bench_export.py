"""Fast tests for benchmarking and ONNX export orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signstream.bench.latency import benchmark_latency
from signstream.export.onnx_export import export_onnx
from signstream.serve.runtime import OnnxBackend, TorchBackend

CHECKPOINT = Path("eval/fixtures/pose_transformer_signer_holdout.pt")
ONNX = Path("eval/fixtures/pose_transformer.onnx")


def test_latency_benchmark_small(tmp_path: Path) -> None:
    if not CHECKPOINT.exists() or not ONNX.exists():
        import pytest

        pytest.skip("fixtures unavailable")
    payload = benchmark_latency(
        str(CHECKPOINT),
        str(ONNX),
        batch_sizes=[1],
        warmup=1,
        iterations=3,
        reports_dir=tmp_path,
    )
    assert payload["results"]["torch"][0]["batch_size"] == 1
    assert payload["results"]["streaming"]["realtime_viable_at_30fps"] is True
    assert (tmp_path / "latency.json").exists()
    assert (tmp_path / "latency.md").exists()


def test_export_onnx_with_tiny_data(tiny_data, monkeypatch, tmp_path: Path) -> None:
    if not CHECKPOINT.exists():
        import pytest

        pytest.skip("fixture checkpoint unavailable")
    data_cfg, _ = tiny_data
    monkeypatch.setattr("signstream.config.data_config", lambda: data_cfg)
    result = export_onnx(
        CHECKPOINT,
        tmp_path / "model.onnx",
        n_parity=2,
        report_path=tmp_path / "parity.json",
    )
    assert result["argmax_agreement"] == 1.0
    assert result["n_parity_clips"] == 2
    assert result["max_abs_logit_delta"] < 1e-3
    assert (tmp_path / "parity.json").exists()


def test_backends_accept_landmark_tensors() -> None:
    if not CHECKPOINT.exists() or not ONNX.exists():
        import pytest

        pytest.skip("fixtures unavailable")
    batch = np.zeros((2, 48, 61, 3), dtype=np.float32)
    torch_backend = TorchBackend(CHECKPOINT)
    onnx_backend = OnnxBackend(ONNX, CHECKPOINT)
    assert torch_backend.predict(batch).shape == (2, 60)
    assert onnx_backend.predict(batch).shape == (2, 60)
