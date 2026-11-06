"""Frozen gold-set construction, hashing, and checkpoint inference helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from signstream.config import EvalConfig, data_config, eval_config
from signstream.data.corpus import GLOSSES
from signstream.data.dataset import SignClipDataset, collate_fn
from signstream.eval.metrics import classification_bundle
from signstream.models.baselines import PooledLinearProbe
from signstream.train import load_checkpoint_model

LOGGER = logging.getLogger(__name__)


def _gold_hash(gold: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"clips": gold["clips"], "split": gold["split"]}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_gold_set(
    *,
    config: EvalConfig | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Create a frozen 300-clip gold set balanced across all 60 glosses.

    Five clips per gloss are chosen round-robin from the held-out signers,
    guaranteeing both LOW and HIGH capture-quality tiers are represented.
    """
    eval_cfg = config or eval_config()
    data_cfg = data_config()
    data_dir = Path(data_cfg.data_dir)
    records = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    split = json.loads((data_dir / "splits" / "signer_holdout.json").read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    by_gloss: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["clip_id"] in test_ids:
            by_gloss[int(record["label_id"])].append(record)

    selected: list[dict[str, Any]] = []
    for label_id in range(len(GLOSSES)):
        candidates = sorted(by_gloss[label_id], key=lambda item: (item["signer"], item["clip_id"]))
        by_signer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            by_signer[str(record["signer"])].append(record)
        signers = sorted(by_signer)
        chosen: list[dict[str, Any]] = []
        cursor = 0
        while len(chosen) < 5 and signers:
            signer = signers[cursor % len(signers)]
            pool = by_signer[signer]
            if pool:
                chosen.append(pool.pop(0))
            cursor += 1
            if all(not pool for pool in by_signer.values()):
                break
        if len(chosen) != 5:
            raise RuntimeError(f"Gloss {GLOSSES[label_id]} has fewer than 5 held-out clips")
        selected.extend(chosen)

    clips = [
        {"clip_id": record["clip_id"], "label_id": int(record["label_id"]), "gloss": GLOSSES[int(record["label_id"])]}
        for record in sorted(selected, key=lambda item: item["clip_id"])
    ]
    if len(clips) != eval_cfg.gold_size:
        raise RuntimeError(f"Expected {eval_cfg.gold_size} gold clips, got {len(clips)}")
    gold: dict[str, Any] = {
        "schema": "signstream-gold-v1",
        "split": "signer_holdout",
        "n_clips": len(clips),
        "clips": clips,
        "signer_distribution": dict(
            sorted(Counter(str(record["signer"]) for record in selected).items())
        ),
        "quality_tier_distribution": dict(
            sorted(Counter(str(record["quality_tier"]) for record in selected).items())
        ),
        "gloss_distribution": dict(
            sorted(Counter(GLOSSES[int(record["label_id"])] for record in selected).items())
        ),
    }
    gold["content_hash"] = _gold_hash(gold)
    if write:
        path = Path(eval_cfg.gold_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(gold, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Wrote frozen gold set with %d clips to %s", len(clips), path)
    return gold


def load_gold_set(path: str | Path | None = None) -> dict[str, Any]:
    """Load and integrity-check the frozen gold set."""
    config = eval_config()
    gold_path = Path(path or config.gold_path)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold set missing at {gold_path}; run gold construction first")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    expected = _gold_hash(gold)
    if gold.get("content_hash") != expected:
        raise ValueError(
            f"Gold-set content hash mismatch: stored={gold.get('content_hash')} computed={expected}"
        )
    return gold


def predict_for_ids(
    checkpoint: str | Path,
    clip_ids: list[str],
    *,
    config: EvalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Run a checkpoint over a list of clips and return probabilities and labels."""
    eval_cfg = config or eval_config()
    data_cfg = data_config()
    data_dir = Path(data_cfg.data_dir)
    clips = np.load(data_dir / "clips.npy", mmap_mode="r")
    records = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    record_map = {record["clip_id"]: record for record in records}
    ordered_records = [record_map[clip_id] for clip_id in clip_ids]
    dataset = SignClipDataset(
        clips,
        records,
        clip_ids,
        target_len=48,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model, payload = load_checkpoint_model(checkpoint)
    probability_rows: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for features, targets in loader:
            if isinstance(model, PooledLinearProbe):
                raw = model.predict_proba(features.numpy())
                classes = model.classes_ if model.classes_ is not None else np.arange(raw.shape[1])
                aligned = np.zeros((raw.shape[0], len(GLOSSES)), dtype=np.float32)
                aligned[:, classes] = raw
                probability_rows.append(aligned)
            else:
                logits = model(features)
                probability_rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
            labels.extend(int(value) for value in targets.numpy())
    return (
        np.concatenate(probability_rows, axis=0),
        np.asarray(labels, dtype=np.int64),
        ordered_records,
        payload,
    )


def evaluate_gold(
    checkpoint: str | Path,
    *,
    gold_path: str | Path | None = None,
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on the frozen gold set."""
    eval_cfg = config or eval_config()
    gold = load_gold_set(gold_path or eval_cfg.gold_path)
    clip_ids = [clip["clip_id"] for clip in gold["clips"]]
    probabilities, y_true, _, payload = predict_for_ids(checkpoint, clip_ids, config=eval_cfg)
    bundle = classification_bundle(
        probabilities, y_true, list(payload.get("class_names", GLOSSES)), n_bins=eval_cfg.n_bins
    )
    return {
        "checkpoint": str(checkpoint),
        "model": payload.get("model_name"),
        "split": gold["split"],
        "gold_content_hash": gold["content_hash"],
        "n_clips": len(clip_ids),
        "top1": bundle["top1"],
        "top5": bundle["top5"],
        "macro_f1": bundle["macro_f1"],
        "ece": bundle["ece"],
    }
