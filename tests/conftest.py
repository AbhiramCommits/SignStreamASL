"""Shared tiny artifacts for fast integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from signstream.config import DataConfig


@pytest.fixture()
def tiny_data(tmp_path: Path) -> tuple[DataConfig, list[dict]]:
    """Create a 120-clip, 60-class fixture with train/val/test splits."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(8)
    clips = rng.normal(0.0, 0.02, size=(120, 48, 61, 3)).astype(np.float16)
    records = []
    for index in range(120):
        label = index % 60
        signer = ["S12", "S13", "S14"][index % 3]
        clips[index, :, :, 0] += label * 0.01
        records.append(
            {
                "clip_id": f"clip_{index:03d}",
                "label_id": label,
                "label": f"L{label}",
                "true_label_id": label,
                "true_label": f"L{label}",
                "signer": signer,
                "quality_tier": "LOW" if signer == "S12" else "HIGH",
                "handedness": "right-dominant",
                "valid_length": 48,
                "injected_defect": None,
                "injected_mislabel": False,
            }
        )
    np.save(data_dir / "clips.npy", clips)
    (data_dir / "index.json").write_text(json.dumps(records), encoding="utf-8")
    splits_dir = data_dir / "splits"
    splits_dir.mkdir()
    ids = [record["clip_id"] for record in records]
    payload = {
        "schema": "signstream-split-v1",
        "strategy": "signer_holdout",
        "train": ids[:60],
        "val": ids[60:80],
        "test": ids[80:],
        "signer_lists": {"train": ["S01"], "val": ["S11"], "test": ["S12", "S13", "S14"]},
    }
    for name in ("random", "signer_holdout"):
        (splits_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "dataset_version": "v1",
        "dataset_manifest_hash": "fixture-manifest-hash",
        "num_clips": 120,
        "num_glosses": 60,
        "num_signers": 3,
    }
    (data_dir / "manifest_v1.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = DataConfig(
        data_dir=str(data_dir),
        clips_dir=str(tmp_path / "clips"),
        keypoints_dir=str(tmp_path / "keypoints"),
        continuous_dir=str(tmp_path / "continuous"),
        reports_dir=str(tmp_path / "reports"),
        continuous_sentences=3,
        continuous_min_signs=2,
        continuous_max_signs=3,
    )
    return config, records
