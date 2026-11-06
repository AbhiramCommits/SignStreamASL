"""ONNX export with dynamic axes and numerical parity verification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from signstream.config import eval_config
from signstream.data.corpus import GLOSSES
from signstream.data.dataset import SignClipDataset, collate_fn
from signstream.train import load_checkpoint_model

LOGGER = logging.getLogger(__name__)


def export_onnx(
    checkpoint: str | Path,
    output_path: str | Path = "artifacts/onnx/pose_transformer.onnx",
    *,
    n_parity: int = 500,
    max_delta: float = 1e-3,
) -> dict[str, Any]:
    """Export a checkpoint with dynamic batch/sequence axes and check parity."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint_model(checkpoint)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Only torch checkpoints can be exported to ONNX")
    model.eval()
    example = torch.randn(1, 48, 183, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["pose_features"],
        output_names=["logits"],
        dynamic_axes={
            "pose_features": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )

    from signstream.config import data_config

    data_cfg = data_config()
    eval_cfg = eval_config()
    split = json.loads(
        (Path(data_cfg.data_dir) / "splits" / "signer_holdout.json").read_text(encoding="utf-8")
    )
    dataset = SignClipDataset.for_split(data_cfg, "signer_holdout", "test")
    selected = split["test"][:n_parity]
    dataset = SignClipDataset(
        np.load(Path(data_cfg.data_dir) / "clips.npy", mmap_mode="r"),
        json.loads((Path(data_cfg.data_dir) / "index.json").read_text(encoding="utf-8")),
        selected,
        target_len=48,
        augment=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    deltas: list[np.ndarray] = []
    torch_predictions: list[np.ndarray] = []
    onnx_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            reference = model(features).cpu().numpy()
            ort_output = session.run(
                None, {session.get_inputs()[0].name: features.numpy()}
            )[0]
            deltas.append(np.abs(reference - ort_output).reshape(-1))
            torch_predictions.append(reference.argmax(axis=1))
            onnx_predictions.append(ort_output.argmax(axis=1))
    delta = np.concatenate(deltas)
    torch_argmax = np.concatenate(torch_predictions)
    onnx_argmax = np.concatenate(onnx_predictions)
    agreement = float(np.mean(torch_argmax == onnx_argmax))
    result = {
        "checkpoint": str(checkpoint),
        "onnx_path": str(output),
        "n_parity_clips": int(len(selected)),
        "max_abs_logit_delta": float(delta.max()),
        "mean_abs_logit_delta": float(delta.mean()),
        "argmax_agreement": agreement,
        "opset": 17,
        "dynamic_axes": {"batch": True, "sequence": True},
        "onnx_size_bytes": int(output.stat().st_size),
        "checkpoint_size_bytes": int(Path(checkpoint).stat().st_size),
        "class_count": len(payload.get("class_names", GLOSSES)),
        "manifest_hash": payload.get("manifest_hash"),
    }
    report_path = Path(eval_cfg.reports_dir) / "onnx_parity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "ONNX parity: max_delta=%.2e mean_delta=%.2e argmax_agreement=%.4f",
        result["max_abs_logit_delta"],
        result["mean_abs_logit_delta"],
        agreement,
    )
    assert result["max_abs_logit_delta"] < max_delta, "ONNX logit delta exceeds tolerance"
    assert agreement == 1.0, "ONNX argmax agreement is not 100%"
    return result
