"""Procedural corpus determinism and injected-defect detection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signstream.config import DataConfig
from signstream.data.corpus import (
    GLOSSES,
    content_hash_ids,
    generate_corpus,
    load_corpus,
    save_corpus,
    signer_styles,
)
from signstream.data.curate import curate_all, dedupe, quality_filter


def _small_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        seed=12,
        num_glosses=3,
        num_signers=4,
        takes_per_pair=2,
        duplicate_rate=0.10,
        mislabel_rate=0.10,
        truncation_rate=0.10,
        tracking_failure_rate=0.10,
        curation_top_fraction=0.05,
        data_dir=str(tmp_path / "data"),
        clips_dir=str(tmp_path / "clips"),
        keypoints_dir=str(tmp_path / "keypoints"),
        continuous_dir=str(tmp_path / "continuous"),
        reports_dir=str(tmp_path / "reports"),
    )


def test_gloss_vocabulary_is_60_unique() -> None:
    assert len(GLOSSES) == 60
    assert len(set(GLOSSES)) == 60


def test_signer_quality_and_handedness() -> None:
    styles = signer_styles(DataConfig())
    assert len(styles) == 14
    low = [style.signer_id for style in styles if style.quality_tier == "LOW"]
    left = [style.signer_id for style in styles if style.left_dominant]
    assert low == ["S11", "S12"]
    assert left == ["S04", "S13"]
    low_style = next(style for style in styles if style.signer_id == "S12")
    high_style = next(style for style in styles if style.signer_id == "S01")
    assert low_style.noise_sigma > high_style.noise_sigma


def test_generation_is_deterministic(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    clips_a, records_a = generate_corpus(config)
    clips_b, records_b = generate_corpus(config)
    assert clips_a.shape == (24, 48, 61, 3)
    assert clips_a.dtype == np.float16
    assert content_hash_ids(record["clip_id"] for record in records_a) == content_hash_ids(
        record["clip_id"] for record in records_b
    )
    np.testing.assert_array_equal(clips_a, clips_b)


def test_all_injected_defect_types_present(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    _, records = generate_corpus(config)
    defects = {record["injected_defect"] for record in records if record["injected_defect"]}
    assert {"duplicate", "mislabel", "truncated", "tracking_failure"} <= defects


def test_curation_catches_injected_defects(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    clips, records = generate_corpus(config)
    duplicate_report = dedupe(clips, records, cosine_threshold=0.995)
    assert duplicate_report.n_dropped >= 1
    allowed = [index for index, record in enumerate(records) if not record["injected_defect"] == "duplicate"]
    quality = quality_filter(
        clips,
        records,
        min_valid_frames=12,
        max_nan_fraction=0.4,
        allowed_indices=allowed,
    )
    assert quality.counts_by_reason["too_few_frames"] >= 1
    assert quality.counts_by_reason["excess_nan"] >= 1
    result = curate_all(clips, records, config)
    assert result.report["dropped_clips"] > 0
    assert result.report["counts_by_reason"]["duplicate"] >= 1
    assert Path(config.data_dir, "curation_report.json").exists()


def test_save_and_load_corpus(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    clips, records = generate_corpus(config)
    save_corpus(clips, records, config, write_individual=False)
    loaded, loaded_records = load_corpus(config)
    assert loaded.shape == clips.shape
    assert len(loaded_records) == len(records)
    assert (Path(config.data_dir) / "injected_defects.json").exists()
