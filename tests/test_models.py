"""Model shape, masking, attention, and baseline tests."""

from __future__ import annotations

import numpy as np
import torch

from signstream.config import ModelConfig
from signstream.models import (
    BiGRUClassifier,
    PooledLinearProbe,
    PoseSequenceTransformer,
    build_model,
    pooled_features,
)


def test_transformer_forward_and_parameter_budget() -> None:
    config = ModelConfig()
    model = PoseSequenceTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )
    logits = model(torch.randn(3, 48, 183))
    assert logits.shape == (3, 60)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_500_000


def test_transformer_accepts_landmark_tensor() -> None:
    model = PoseSequenceTransformer()
    logits = model(torch.randn(2, 48, 61, 3))
    assert logits.shape == (2, 60)


def test_transformer_masking_and_attention() -> None:
    model = PoseSequenceTransformer()
    features = torch.randn(2, 48, 183)
    mask = torch.zeros(2, 48, dtype=torch.bool)
    mask[:, 30:] = True
    logits = model(features, padding_mask=mask)
    assert torch.isfinite(logits).all()
    attention = model.temporal_attention_weights(features, padding_mask=mask)
    assert attention.shape == (2, 48, 48)
    assert torch.isfinite(attention).all()


def test_transformer_validation() -> None:
    model = PoseSequenceTransformer(max_len=48)
    try:
        model(torch.randn(1, 32, 100))
    except ValueError as exc:
        assert "183" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
    try:
        model(torch.randn(1, 50, 183))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_bigru_forward_and_mask() -> None:
    model = BiGRUClassifier()
    features = torch.randn(4, 48, 183)
    mask = torch.zeros(4, 48, dtype=torch.bool)
    mask[:, 40:] = True
    assert model(features).shape == (4, 60)
    assert model(features, padding_mask=mask).shape == (4, 60)


def test_pooled_features_fixture() -> None:
    features = np.zeros((2, 10, 183), dtype=np.float32)
    result = pooled_features(features)
    assert result.shape == (2, 427)
    assert np.isfinite(result).all()


def test_probe_fit_and_state_roundtrip() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(0, 0.2, size=(90, 12, 183)).astype(np.float32)
    labels = np.repeat(np.arange(3), 30)
    features[:, :, 0] += labels[:, None] * 2.0
    probe = PooledLinearProbe(max_iter=500, random_state=2)
    probe.fit(features, labels)
    assert probe.predict(features).shape == (90,)
    assert probe.predict_proba(features).shape == (90, 3)
    state = probe.state_dict()
    restored = PooledLinearProbe.from_state(state)
    np.testing.assert_allclose(probe.predict_proba(features), restored.predict_proba(features), atol=1e-5)


def test_build_model_dispatch() -> None:
    config = ModelConfig()
    assert isinstance(build_model("pose_transformer", config), PoseSequenceTransformer)
    assert isinstance(build_model("bigru", config), BiGRUClassifier)
    assert isinstance(build_model("linear_probe", config), PooledLinearProbe)
    try:
        build_model("unknown", config)
    except ValueError as exc:
        assert "Unknown" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
