"""ONNX Runtime parity tests using small committed fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from signstream.serve.runtime import OnnxBackend, TorchBackend, build_backend

CHECKPOINT = Path("eval/fixtures/pose_transformer_signer_holdout.pt")
ONNX = Path("eval/fixtures/pose_transformer.onnx")


def test_onnx_and_torch_parity() -> None:
    if not CHECKPOINT.exists() or not ONNX.exists():
        pytest.skip("ONNX fixtures unavailable")
    torch_backend = TorchBackend(CHECKPOINT)
    onnx_backend = OnnxBackend(ONNX, CHECKPOINT)
    features = np.random.default_rng(4).normal(size=(4, 48, 183)).astype(np.float32)
    torch_output = torch_backend.predict(features)
    onnx_output = onnx_backend.predict(features)
    assert np.max(np.abs(torch_output - onnx_output)) < 1e-4
    np.testing.assert_array_equal(torch_output.argmax(axis=1), onnx_output.argmax(axis=1))


def test_backend_factory() -> None:
    if not CHECKPOINT.exists() or not ONNX.exists():
        pytest.skip("ONNX fixtures unavailable")
    assert build_backend("torch", checkpoint=CHECKPOINT).name == "torch"
    assert build_backend("onnx", checkpoint=CHECKPOINT, onnx_path=ONNX).name == "onnx"
    with pytest.raises(ValueError):
        build_backend("bogus", checkpoint=CHECKPOINT, onnx_path=ONNX)


def test_onnx_parity_report_if_present() -> None:
    path = Path("artifacts/reports/onnx_parity.json")
    if not path.exists():
        pytest.skip("parity report absent before export")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["argmax_agreement"] == 1.0
    assert report["max_abs_logit_delta"] < 1e-3
    assert report["n_parity_clips"] == 500
