"""Split isolation, stratification, coverage, and manifest tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from signstream.config import DataConfig
from signstream.data.manifest import load_manifest, write_manifest
from signstream.data.splits import (
    coverage_stats,
    random_stratified_split,
    signer_holdout_split,
    write_split_files,
)


def _records(classes: int = 5, signers: int = 14, takes: int = 5) -> list[dict]:
    records = []
    for label in range(classes):
        for signer_index in range(signers):
            for take in range(takes):
                signer = f"S{signer_index + 1:02d}"
                records.append(
                    {
                        "clip_id": f"g{label:02d}_{signer}_t{take:02d}",
                        "label_id": label,
                        "label": f"L{label}",
                        "signer": signer,
                        "quality_tier": "LOW" if signer in {"S11", "S12"} else "HIGH",
                        "handedness": "left-dominant" if signer in {"S04", "S13"} else "right-dominant",
                        "valid_length": 48,
                    }
                )
    return records


def test_signer_holdout_has_zero_overlap(tmp_path: Path) -> None:
    config = DataConfig(data_dir=str(tmp_path / "data"))
    records = _records()
    split = signer_holdout_split(records, config)
    assert set(split["signer_lists"]["test"]) == {"S12", "S13", "S14"}
    assert not set(split["signer_lists"]["train"]) & set(split["signer_lists"]["test"])
    assert not set(split["signer_lists"]["val"]) & set(split["signer_lists"]["test"])
    assert not set(split["signer_lists"]["train"]) & set(split["signer_lists"]["val"])


def test_random_split_is_stratified(tmp_path: Path) -> None:
    config = DataConfig(seed=3, data_dir=str(tmp_path / "data"))
    records = _records(classes=4, takes=10)
    split = random_stratified_split(records, config)
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    label_map = {record["clip_id"]: record["label_id"] for record in records}
    train_counts = Counter(label_map[clip] for clip in split["train"])
    for label in range(4):
        fraction = train_counts[label] / 140
        assert abs(fraction - 0.70) < 0.08


def test_write_splits_and_coverage(tmp_path: Path) -> None:
    config = DataConfig(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"))
    records = _records(classes=3)
    payloads = write_split_files(records, config)
    assert (tmp_path / "data" / "splits" / "random.json").exists()
    assert (tmp_path / "data" / "splits" / "signer_holdout.json").exists()
    assert payloads["signer_holdout"]["signer_lists"]["test"] == ["S12", "S13", "S14"]
    coverage = coverage_stats(records, reports_dir=tmp_path / "reports")
    assert coverage["total_clips"] == len(records)
    assert coverage["clips_per_gloss"]["min"] == 70
    assert (tmp_path / "reports" / "coverage.md").exists()


def test_manifest_roundtrip_and_hash(tmp_path: Path) -> None:
    config = DataConfig(data_dir=str(tmp_path / "data"))
    records = _records(classes=3)
    splits = write_split_files(records, config)
    curation = {"counts_by_reason": {"duplicate": 1}, "label_audit": {"precision": 0.5, "recall": 0.4}}
    manifest = write_manifest("v1", config, records, curation, splits)
    assert len(manifest["dataset_manifest_hash"]) == 64
    assert manifest["num_clips"] == len(records)
    assert manifest["per_signer_counts"]["S12"] == 15
    loaded = load_manifest(tmp_path / "data" / "manifest_v1.json")
    assert loaded["dataset_version"] == "v1"
    assert loaded["split_definitions"]["signer_holdout"]["counts"]["test"] == 45
