"""Ablation baselines: a two-layer BiGRU and a pooled logistic probe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn


def _masked_pool(hidden: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    else:
        valid = ~padding_mask.to(dtype=torch.bool)
    weights = valid.unsqueeze(-1).to(hidden.dtype)
    mean_pool = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    masked = hidden.masked_fill(~valid.unsqueeze(-1), torch.finfo(hidden.dtype).min)
    max_pool = masked.max(dim=1).values
    return torch.cat([mean_pool, max_pool], dim=-1)


class BiGRUClassifier(nn.Module):
    """Two-layer bidirectional GRU ablation baseline."""

    def __init__(
        self,
        *,
        feature_dim: int = 183,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 60,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 4)
        self.classifier = nn.Linear(hidden_size * 4, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        hidden, _ = self.gru(x)
        return self.classifier(self.norm(_masked_pool(hidden, padding_mask)))


def pooled_features(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return mean, standard deviation, and per-landmark velocity magnitude.

    Args:
        x: Features shaped ``[B, T, 183]`` (or a single ``[T, 183]`` sequence).

    Returns:
        Array shaped ``[B, 427]``: 183 mean coordinates + 183 standard
        deviations + 61 velocity magnitudes.
    """
    array = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != 183:
        raise ValueError(f"Expected [B, T, 183], got {array.shape}")
    points = array.reshape(array.shape[0], array.shape[1], 61, 3)
    mean = points.mean(axis=1).reshape(array.shape[0], -1)
    std = points.std(axis=1).reshape(array.shape[0], -1)
    velocity = np.linalg.norm(np.diff(points, axis=1), axis=-1).mean(axis=1)
    return np.concatenate([mean, std, velocity], axis=1).astype(np.float32)


class PooledLinearProbe:
    """Scikit-learn logistic regression over pooled statistical features."""

    def __init__(self, *, max_iter: int = 1000, random_state: int = 7) -> None:
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray) -> PooledLinearProbe:
        """Fit the scaler and multinomial logistic regression."""
        features = pooled_features(x)
        targets = np.asarray(y, dtype=np.int64)
        scaled = self.scaler.fit_transform(features)
        self.classifier.fit(scaled, targets)
        self.classes_ = self.classifier.classes_.copy()
        return self

    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return class probabilities aligned with ``classes_``."""
        if self.classes_ is None:
            raise RuntimeError("PooledLinearProbe must be fitted before prediction")
        features = pooled_features(x)
        return self.classifier.predict_proba(self.scaler.transform(features))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return predicted class ids."""
        probabilities = self.predict_proba(x)
        assert self.classes_ is not None
        return self.classes_[np.argmax(probabilities, axis=1)]

    def state_dict(self) -> dict[str, Any]:
        """Serialize learned parameters without pickle-dependent sklearn state."""
        if self.classes_ is None:
            raise RuntimeError("Cannot serialize an unfitted probe")
        return {
            "kind": "linear_probe",
            "classes": self.classes_.tolist(),
            "coef": self.classifier.coef_.astype(np.float32),
            "intercept": self.classifier.intercept_.astype(np.float32),
            "scaler_mean": self.scaler.mean_.astype(np.float32),
            "scaler_scale": self.scaler.scale_.astype(np.float32),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> PooledLinearProbe:
        """Restore a probe serialized by :meth:`state_dict`."""
        probe = cls()
        probe.classes_ = np.asarray(state["classes"], dtype=np.int64)
        probe.classifier.classes_ = probe.classes_
        probe.classifier.coef_ = np.asarray(state["coef"], dtype=np.float32)
        probe.classifier.intercept_ = np.asarray(state["intercept"], dtype=np.float32)
        probe.classifier.n_features_in_ = int(probe.classifier.coef_.shape[1])
        probe.scaler.mean_ = np.asarray(state["scaler_mean"], dtype=np.float32)
        probe.scaler.scale_ = np.asarray(state["scaler_scale"], dtype=np.float32)
        probe.scaler.var_ = probe.scaler.scale_**2
        probe.scaler.n_features_in_ = int(probe.scaler.mean_.shape[0])
        return probe
