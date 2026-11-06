"""Split construction with signer-level isolation and coverage diagnostics."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from signstream.config import DataConfig

LOGGER = logging.getLogger(__name__)


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["clip_id"]): record for record in records}


def random_stratified_split(
    records: list[dict[str, Any]],
    config: DataConfig,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Stratified 70/15/15 split, sampled independently within each gloss."""
    rng = np.random.default_rng(config.seed if seed is None else seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for record in records:
        by_label[int(record["label_id"])].append(str(record["clip_id"]))

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * config.random_train))
        n_val = int(round(n * config.random_val))
        n_train = min(n_train, max(0, n - 2)) if n >= 3 else n
        n_val = min(n_val, max(0, n - n_train - 1)) if n >= 3 else 0
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val :])

    records_by_id = _records_by_id(records)
    payload = {
        "schema": "signstream-split-v1",
        "strategy": "random",
        "seed": int(config.seed if seed is None else seed),
        "fractions": {
            "train": config.random_train,
            "val": config.random_val,
            "test": config.random_test,
        },
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "signer_lists": {
            "train": sorted({records_by_id[clip]["signer"] for clip in train}),
            "val": sorted({records_by_id[clip]["signer"] for clip in val}),
            "test": sorted({records_by_id[clip]["signer"] for clip in test}),
        },
    }
    return payload


def signer_holdout_split(
    records: list[dict[str, Any]],
    config: DataConfig,
) -> dict[str, Any]:
    """Split by signer so held-out test signers never appear during training."""
    available = sorted({str(record["signer"]) for record in records})
    missing_test = sorted(set(config.test_signers) - set(available))
    missing_val = sorted(set(config.val_signers) - set(available))
    if missing_test:
        raise ValueError(f"Held-out test signers absent from corpus: {missing_test}")
    if missing_val:
        raise ValueError(f"Validation signers absent from corpus: {missing_val}")

    test_signers = set(config.test_signers)
    val_signers = set(config.val_signers)
    train_signers = set(available) - test_signers - val_signers
    if not train_signers:
        raise ValueError("No signers remain for training")

    assert test_signers.isdisjoint(val_signers)
    assert test_signers.isdisjoint(train_signers)
    assert val_signers.isdisjoint(train_signers)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for record in records:
        signer = str(record["signer"])
        clip_id = str(record["clip_id"])
        if signer in test_signers:
            test.append(clip_id)
        elif signer in val_signers:
            val.append(clip_id)
        else:
            assert signer in train_signers
            train.append(clip_id)

    payload = {
        "schema": "signstream-split-v1",
        "strategy": "signer_holdout",
        "seed": int(config.seed),
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "signer_lists": {
            "train": sorted(train_signers),
            "val": sorted(val_signers),
            "test": sorted(test_signers),
        },
    }
    assert not (set(payload["signer_lists"]["train"]) & set(payload["signer_lists"]["test"]))
    assert not (set(payload["signer_lists"]["val"]) & set(payload["signer_lists"]["test"]))
    assert not (set(payload["signer_lists"]["train"]) & set(payload["signer_lists"]["val"]))
    return payload


def write_split_files(
    records: list[dict[str, Any]],
    config: DataConfig,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Write both split strategies and assert signer isolation for holdout."""
    directory = Path(data_dir or config.data_dir) / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "random": random_stratified_split(records, config),
        "signer_holdout": signer_holdout_split(records, config),
    }
    for name, payload in payloads.items():
        (directory / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    holdout = payloads["signer_holdout"]
    assert set(holdout["signer_lists"]["train"]).isdisjoint(holdout["signer_lists"]["test"])
    LOGGER.info(
        "Splits written: random train/val/test=%d/%d/%d; signer_holdout=%d/%d/%d",
        len(payloads["random"]["train"]),
        len(payloads["random"]["val"]),
        len(payloads["random"]["test"]),
        len(payloads["signer_holdout"]["train"]),
        len(payloads["signer_holdout"]["val"]),
        len(payloads["signer_holdout"]["test"]),
    )
    return payloads


def coverage_stats(
    records: list[dict[str, Any]],
    *,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute and optionally write per-gloss and per-signer coverage tables."""
    by_gloss: Counter[str] = Counter(str(record["gloss"] if "gloss" in record else record["label"]) for record in records)
    by_signer: Counter[str] = Counter(str(record["signer"]) for record in records)
    counts = np.asarray(list(by_gloss.values()), dtype=np.float32)
    summary = {
        "total_clips": len(records),
        "num_glosses": len(by_gloss),
        "num_signers": len(by_signer),
        "clips_per_gloss": {
            "min": int(counts.min()) if counts.size else 0,
            "median": float(np.median(counts)) if counts.size else 0.0,
            "max": int(counts.max()) if counts.size else 0,
        },
        "per_gloss": dict(sorted(by_gloss.items())),
        "per_signer": dict(sorted(by_signer.items())),
    }
    lines = [
        "# Dataset Coverage",
        "",
        f"Total clips: **{summary['total_clips']}** across **{summary['num_glosses']}** glosses "
        f"and **{summary['num_signers']}** signers.",
        "",
        "| Gloss | Clips |",
        "|---|---:|",
    ]
    lines.extend(f"| {gloss} | {count} |" for gloss, count in sorted(by_gloss.items()))
    lines.extend(
        [
            "",
            "| Signer | Clips |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {signer} | {count} |" for signer, count in sorted(by_signer.items()))
    lines.extend(
        [
            "",
            f"Clips per gloss: min **{summary['clips_per_gloss']['min']}**, "
            f"median **{summary['clips_per_gloss']['median']:.1f}**, "
            f"max **{summary['clips_per_gloss']['max']}**.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    if reports_dir is not None:
        path = Path(reports_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "coverage.md").write_text(markdown, encoding="utf-8")
    summary["markdown"] = markdown
    LOGGER.info(
        "Coverage: %d clips, %d glosses, %d signers, clips/gloss min/median/max=%s/%s/%s",
        summary["total_clips"],
        summary["num_glosses"],
        summary["num_signers"],
        summary["clips_per_gloss"]["min"],
        summary["clips_per_gloss"]["median"],
        summary["clips_per_gloss"]["max"],
    )
    return summary
