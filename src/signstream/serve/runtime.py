"""Torch and ONNX Runtime inference backends with one predict contract."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from signstream.data.keypoints import normalize_sequence, resample_sequence
from signstream.train import load_checkpoint_model

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "artifacts/checkpoints/pose_transformer_signer_holdout.pt"
DEFAULT_ONNX = "artifacts/onnx/pose_transformer.onnx"


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal contract shared by eager Torch and ONNX Runtime."""

    name: str
    class_names: list[str]
    manifest_hash: str
    model_name: str

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch."""


def _prepare_batch(batch: np.ndarray, *, target_len: int = 48) -> np.ndarray:
    array = np.asarray(batch, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected [B, T, 61, 3] or [B, T, 183], got {array.shape}")
    prepared = []
    for sequence in array:
        if sequence.shape[-1] == 3:
            sequence = normalize_sequence(sequence)
            if sequence.shape[0] != target_len:
                sequence = resample_sequence(sequence, target_len)
            sequence = np.nan_to_num(sequence, nan=0.0)
            prepared.append(sequence.reshape(target_len, -1))
        else:  # pragma: no cover - guarded by shape check
            raise ValueError("Expected 61 landmarks with 3 coordinates")
    return np.stack(prepared, axis=0).astype(np.float32)


class TorchBackend:
    """Eager PyTorch CPU backend."""

    name = "torch"

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint = str(checkpoint)
        self.model, self.metadata = load_checkpoint_model(checkpoint)
        self.class_names = list(self.metadata.get("class_names", []))
        self.manifest_hash = str(self.metadata.get("manifest_hash", ""))
        self.model_name = str(self.metadata.get("model_name", "pose_transformer"))
        if not self.class_names:
            raise ValueError("Checkpoint does not contain a class list")
        try:
            torch.set_num_threads(min(8, torch.get_num_threads()))
        except RuntimeError:  # pragma: no cover
            pass

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return probabilities for prepared feature batches."""
        features = np.asarray(batch, dtype=np.float32)
        if features.ndim == 4:
            features = _prepare_batch(features)
        tensor = torch.from_numpy(features)
        with torch.no_grad():
            if hasattr(self.model, "predict_proba") and not isinstance(self.model, torch.nn.Module):
                return np.asarray(self.model.predict_proba(features), dtype=np.float32)
            logits = self.model(tensor)
            return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)


class OnnxBackend:
    """ONNX Runtime CPU backend."""

    name = "onnx"

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run export-onnx first.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(8, os.cpu_count() or 4)
        self.session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        _, payload = load_checkpoint_model(checkpoint)
        self.class_names = list(payload.get("class_names", []))
        self.manifest_hash = str(payload.get("manifest_hash", ""))
        self.model_name = str(payload.get("model_name", "pose_transformer"))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Return probabilities from ONNX Runtime."""
        features = np.asarray(batch, dtype=np.float32)
        if features.ndim == 4:
            features = _prepare_batch(features)
        logits = self.session.run(None, {self.input_name: features})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def build_backend(
    name: str | None = None,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    onnx_path: str | Path = DEFAULT_ONNX,
) -> InferenceBackend:
    """Build the backend selected by ``SIGNSTREAM_BACKEND`` or an explicit name."""
    selected = (name or os.environ.get("SIGNSTREAM_BACKEND", "torch")).lower()
    if selected == "torch":
        return TorchBackend(checkpoint)
    if selected == "onnx":
        return OnnxBackend(onnx_path, checkpoint)
    raise ValueError(f"Unknown SIGNSTREAM_BACKEND={selected!r}; expected 'torch' or 'onnx'")
