"""Tests for deduplication, quality filtering, and label auditing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signstream.config import DataConfig
from signstream.data.curate import audit_labels, curate_all, dedupe, quality_filter


def _clip(offset: float = 0.0, frames: int = 48) -> np.ndarray:
    t = np.linspace(0, 1, frames, dtype=np.float32)[:, None, None]
    base = np.zeros((frames, 61, 3), dtype=np.float32)
    base[:, :, 0] = t[:, 0, 0, None] * 0.3 + offset
    base[:, :, 1] = np.sin(t[:, 0, 0, None] * 6.28) * 0.2
    base[:, 47, :] = [-0.5, 0.0, 0.0]
    base[:, 48, :] = [0.5, 0.0, 0.0]
    return base


def _record(clip_id: str, label: int = 0, signer: str = "S01", length: int = 48) -> dict:
    return {
        "clip_id": clip_id,
        "label_id": label,
        "label": f"L{label}",
        "true_label_id": label,
        "true_label": f"L{label}",
        "signer": signer,
        "quality_tier": "HIGH",
        "handedness": "right-dominant",
        "valid_length": length,
        "injected_mislabel": False,
        "injected_defect": None,
    }


def test_dedupe_drops_exact_and_near_copies() -> None:
    base = _clip()
    clips = np.stack([base, base, _clip(offset=0.5)], axis=0).astype(np.float16)
    records = [_record("a"), _record("b"), _record("c")]
    report = dedupe(clips, records)
    assert report.n_dropped == 1
    assert report.dropped_ids == ["b"]
    assert report.duplicate_of["b"] == "a"


def test_quality_filter_reasons() -> None:
    base = _clip()
    truncated = base.copy()
    truncated[6:] = np.nan
    missing = np.full_like(base, np.nan)
    static = np.zeros_like(base)
    clips = np.stack([base, truncated, missing, static], axis=0).astype(np.float16)
    records = [
        _record("ok"),
        _record("short", length=6),
        _record("nan"),
        _record("static"),
    ]
    report = quality_filter(clips, records, min_valid_frames=12, max_nan_fraction=0.4)
    assert report.kept_ids == ["ok"]
    assert report.counts_by_reason["too_few_frames"] == 1
    assert report.counts_by_reason["excess_nan"] == 1
    assert report.counts_by_reason["no_motion"] == 1


def test_label_audit_flags_mislabel() -> None:
    clips = []
    records = []
    for label, offset in [(0, -0.8), (1, 0.0), (2, 0.8)]:
        for index in range(20):
            clips.append(_clip(offset=offset))
            records.append(_record(f"l{label}_{index}", label=label, signer=f"S{label + 1:02d}"))
    # Inject a clip whose features belong to class 0 but whose label says class 1.
    clips.append(_clip(offset=-0.8))
    records.append(
        {
            **_record("mislabelled", label=1, signer="S02"),
            "injected_mislabel": True,
            "true_label_id": 0,
        }
    )
    report = audit_labels(
        np.stack(clips, axis=0).astype(np.float16),
        records,
        top_fraction=0.02,
    )
    assert report.n_flagged >= 1
    assert report.injected_total == 1
    assert "mislabelled" in report.flagged_ids
    assert report.precision == 1.0
    assert report.recall == 1.0


def test_curate_all_writes_report(tmp_path: Path) -> None:
    config = DataConfig(
        seed=1,
        num_glosses=3,
        reports_dir=str(tmp_path / "reports"),
        data_dir=str(tmp_path / "data"),
        curation_top_fraction=0.02,
    )
    clips = np.stack([_clip(), _clip(), _clip(offset=1.0)], axis=0).astype(np.float16)
    records = [_record("a"), _record("b"), _record("c", label=1)]
    result = curate_all(clips, records, config)
    assert result.report["input_clips"] == 3
    assert (tmp_path / "data" / "curation_report.json").exists()
    assert (tmp_path / "reports" / "curation_report.json").exists()
