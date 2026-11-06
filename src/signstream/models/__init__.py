"""Temporal sequence models for isolated sign recognition."""

from __future__ import annotations

from signstream.models.baselines import BiGRUClassifier, PooledLinearProbe, pooled_features
from signstream.models.pose_transformer import PoseSequenceTransformer

__all__ = [
    "BiGRUClassifier",
    "PooledLinearProbe",
    "PoseSequenceTransformer",
    "build_model",
    "pooled_features",
]


def build_model(name: str, config: object) -> PoseSequenceTransformer | BiGRUClassifier | PooledLinearProbe:
    """Construct a model from its CLI name and a :class:`ModelConfig`-like object."""
    if name == "pose_transformer":
        return PoseSequenceTransformer(
            feature_dim=getattr(config, "feature_dim", 183),
            d_model=getattr(config, "d_model", 128),
            nhead=getattr(config, "nhead", 4),
            num_layers=getattr(config, "num_layers", 4),
            ffn_dim=getattr(config, "ffn_dim", 256),
            dropout=getattr(config, "dropout", 0.1),
            num_classes=getattr(config, "num_classes", 60),
            max_len=getattr(config, "seq_len", 48),
        )
    if name == "bigru":
        return BiGRUClassifier(
            feature_dim=getattr(config, "feature_dim", 183),
            hidden_size=getattr(config, "hidden_size", 128),
            num_layers=2,
            dropout=getattr(config, "dropout", 0.1),
            num_classes=getattr(config, "num_classes", 60),
        )
    if name == "linear_probe":
        return PooledLinearProbe()
    raise ValueError(f"Unknown model: {name}")
