"""Training loops, checkpointing, MLflow tracking, and label-budget study."""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from signstream.config import ModelConfig, model_config
from signstream.data.corpus import GLOSSES
from signstream.data.dataset import SignClipDataset, collate_fn
from signstream.data.manifest import load_manifest
from signstream.models import PooledLinearProbe, build_model
from signstream.tracking import RunTracker

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds and deterministic CPU flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def top_k_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Count examples whose target is in the top-k logits."""
    k = min(k, logits.shape[1])
    _, predictions = torch.topk(logits, k=k, dim=1)
    return int(predictions.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


@torch.no_grad()
def evaluate_logits(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a torch classifier on a loader."""
    model.eval()
    total = 0
    total_loss = 0.0
    top1 = 0
    top5 = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, targets).item()) * targets.numel()
        top1 += top_k_correct(logits, targets, 1)
        top5 += top_k_correct(logits, targets, 5)
        total += targets.numel()
    return {
        "loss": total_loss / max(total, 1),
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "n": float(total),
    }


def _warmup_cosine(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, schedule)


def _stratified_subset(labels: list[int], fraction: float, seed: int) -> list[int]:
    if fraction >= 1.0:
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    selected: list[int] = []
    for label in sorted(by_label):
        indices = by_label[label]
        count = max(1, int(round(len(indices) * fraction)))
        chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
        selected.extend(int(index) for index in chosen)
    return sorted(selected)


def _run_id(model_name: str, split: str) -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}_{model_name}_{split}_{uuid.uuid4().hex[:6]}"


def _save_checkpoint(
    path: Path,
    *,
    model_name: str,
    model_state: dict[str, Any] | None,
    probe_state: dict[str, Any] | None,
    config: ModelConfig,
    split: str,
    manifest_hash: str,
    seed: int,
    best_val_top1: float,
    best_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "linear_probe" if probe_state is not None else "torch",
        "model_name": model_name,
        "model_state": model_state,
        "probe_state": probe_state,
        "config": config.to_dict(),
        "class_names": list(GLOSSES),
        "manifest_hash": manifest_hash,
        "split": split,
        "seed": seed,
        "best_val_top1": best_val_top1,
        "best_epoch": best_epoch,
    }
    torch.save(payload, path)


def _probe_metrics(
    probe: PooledLinearProbe,
    loader: DataLoader,
) -> dict[str, float]:
    probabilities = []
    targets_all = []
    for features, targets in loader:
        probabilities.append(probe.predict_proba(features))
        targets_all.append(targets.numpy())
    probs = np.concatenate(probabilities, axis=0)
    targets_array = np.concatenate(targets_all, axis=0)
    top1 = float(np.mean(probs.argmax(axis=1) == targets_array))
    top5_indices = np.argsort(-probs, axis=1)[:, :5]
    top5 = float(np.mean((top5_indices == targets_array[:, None]).any(axis=1)))
    loss = float(-np.log(np.clip(probs[np.arange(len(targets_array)), targets_array], 1e-9, 1.0)).mean())
    return {"loss": loss, "top1": top1, "top5": top5, "n": float(len(targets_array))}


def train_model(
    config: ModelConfig,
    *,
    model_name: str = "pose_transformer",
    split: str = "signer_holdout",
    epochs: int | None = None,
    seed: int | None = None,
    train_fraction: float = 1.0,
    checkpoint_name: str | None = None,
    run_name: str | None = None,
    compute_test: bool = False,
    progress: bool = True,
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    """Train one model on one split and persist checkpoint + metrics.

    Args:
        config: Model and training configuration.
        model_name: ``pose_transformer``, ``bigru``, or ``linear_probe``.
        split: ``random`` or ``signer_holdout``.
        epochs: Optional epoch override.
        seed: Optional seed override.
        train_fraction: Stratified fraction of training labels to use.
        checkpoint_name: Optional output checkpoint filename.
        run_name: Optional tracking run name.
        compute_test: Evaluate the test partition after training.
        progress: Log epoch summaries.

    Returns:
        A report dictionary with history, best validation metrics, run paths,
        and (when requested) test metrics.
    """
    from signstream.config import data_config

    data_cfg = data_config()
    run_seed = config.seed if seed is None else int(seed)
    run_epochs = config.epochs if epochs is None else int(epochs)
    set_seed(run_seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
    manifest = load_manifest(Path(data_cfg.data_dir) / f"manifest_{data_cfg.manifest_version}.json")

    train_dataset = SignClipDataset.for_split(
        data_cfg, split, "train", augment=True, target_len=config.seq_len
    )
    val_dataset = SignClipDataset.for_split(
        data_cfg, split, "val", augment=False, target_len=config.seq_len
    )
    train_indices = _stratified_subset(train_dataset.labels, train_fraction, run_seed + 31)
    train_subset = Subset(train_dataset, train_indices)
    generator = torch.Generator().manual_seed(run_seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    test_loader = None
    if compute_test:
        test_loader = DataLoader(
            SignClipDataset.for_split(data_cfg, split, "test", augment=False, target_len=config.seq_len),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )

    effective_run = run_name or model_name
    run_id = _run_id(effective_run, split)
    run_dir = Path(config.runs_dir) / run_id
    checkpoint_path = Path(config.checkpoint_dir) / (
        checkpoint_name or f"{model_name}_{split}.pt"
    )
    report: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "seed": run_seed,
        "epochs": run_epochs,
        "train_fraction": float(train_fraction),
        "train_examples": len(train_indices),
        "val_examples": len(val_dataset),
        "manifest_hash": manifest["dataset_manifest_hash"],
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "history": [],
    }

    tracker: RunTracker = RunTracker(
        run_id,
        tracking_uri=config.mlflow_uri,
        run_dir=config.runs_dir,
        enabled=enable_mlflow,
    ).__enter__()
    tracker.log_params(
        {
            **config.to_dict(),
            "model_name": model_name,
            "split": split,
            "run_seed": run_seed,
            "run_epochs": run_epochs,
            "train_fraction": train_fraction,
            "train_examples": len(train_indices),
            "manifest_hash": manifest["dataset_manifest_hash"],
        }
    )

    try:
        if model_name == "linear_probe":
            probe = PooledLinearProbe(max_iter=1000, random_state=run_seed)
            all_features: list[np.ndarray] = []
            all_targets: list[np.ndarray] = []
            for features, targets in train_loader:
                all_features.append(features.numpy())
                all_targets.append(targets.numpy())
            probe.fit(np.concatenate(all_features, axis=0), np.concatenate(all_targets, axis=0))
            val_metrics = _probe_metrics(probe, val_loader)
            tracker.log_metrics(val_metrics, step=1)
            report["history"].append({"epoch": 1, **val_metrics})
            report["best_val"] = val_metrics
            report["best_epoch"] = 1
            report["parameters"] = int(np.prod(probe.classifier.coef_.shape))
            _save_checkpoint(
                checkpoint_path,
                model_name=model_name,
                model_state=None,
                probe_state=probe.state_dict(),
                config=config,
                split=split,
                manifest_hash=manifest["dataset_manifest_hash"],
                seed=run_seed,
                best_val_top1=val_metrics["top1"],
                best_epoch=1,
            )
            if test_loader is not None:
                report["test"] = _probe_metrics(probe, test_loader)
        else:
            model = build_model(model_name, config)
            assert isinstance(model, nn.Module)
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            report["parameters"] = int(parameters)
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            total_steps = max(1, len(train_loader) * run_epochs)
            scheduler = _warmup_cosine(optimizer, total_steps, config.warmup_ratio)
            criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            best_top1 = -1.0
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = 0
            stale = 0
            for epoch in range(1, run_epochs + 1):
                model.train()
                total = 0
                running_loss = 0.0
                running_top1 = 0
                for features, targets in train_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    loss = criterion(logits, targets)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    count = targets.numel()
                    running_loss += float(loss.item()) * count
                    running_top1 += top_k_correct(logits.detach(), targets, 1)
                    total += count
                train_metrics = {
                    "loss": running_loss / max(total, 1),
                    "top1": running_top1 / max(total, 1),
                }
                val_metrics = evaluate_logits(model, val_loader, criterion, device)
                record = {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": train_metrics["loss"],
                    "train_top1": train_metrics["top1"],
                    "val_loss": val_metrics["loss"],
                    "val_top1": val_metrics["top1"],
                    "val_top5": val_metrics["top5"],
                }
                report["history"].append(record)
                tracker.log_metrics(record, step=epoch)
                if progress:
                    LOGGER.info(
                        "%s epoch %d/%d train_loss=%.4f train_top1=%.3f val_top1=%.3f val_top5=%.3f",
                        model_name,
                        epoch,
                        run_epochs,
                        record["train_loss"],
                        record["train_top1"],
                        record["val_top1"],
                        record["val_top5"],
                    )
                if val_metrics["top1"] > best_top1:
                    best_top1 = val_metrics["top1"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= config.patience:
                        LOGGER.info("Early stopping at epoch %d", epoch)
                        break

            assert best_state is not None
            model.load_state_dict(best_state)
            report["best_epoch"] = best_epoch
            report["best_val"] = {
                key: value
                for key, value in report["history"][best_epoch - 1].items()
                if key.startswith("val_")
            }
            _save_checkpoint(
                checkpoint_path,
                model_name=model_name,
                model_state=best_state,
                probe_state=None,
                config=config,
                split=split,
                manifest_hash=manifest["dataset_manifest_hash"],
                seed=run_seed,
                best_val_top1=float(best_top1),
                best_epoch=best_epoch,
            )
            if test_loader is not None:
                report["test"] = evaluate_logits(model, test_loader, criterion, device)

        report_path = Path(config.reports_dir) / f"train_{model_name}_{split}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        tracker.log_artifact(report_path)
        best_value = report["best_val"].get("top1", report["best_val"].get("val_top1", 0.0))
        LOGGER.info(
            "Saved %s checkpoint at %s (best val top1=%.3f)",
            model_name,
            checkpoint_path,
            best_value,
        )
        return report
    finally:
        tracker.end_run()


def run_label_budget(config: ModelConfig) -> dict[str, Any]:
    """Retrain the transformer across the configured label fractions.

    Results are checkpointed after each fraction so an interrupted study can
    resume without retraining completed points.
    """
    if config.seq_len != 48:
        raise ValueError("Label-budget study expects 48-frame sequences")
    reports = Path(config.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    partial_path = reports / "label_budget_partial.json"
    existing: dict[float, dict[str, Any]] = {}
    if partial_path.exists():
        try:
            previous = json.loads(partial_path.read_text(encoding="utf-8"))
            existing = {
                float(point["label_fraction"]): point for point in previous.get("points", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}

    points: list[dict[str, Any]] = []
    for fraction, epochs in zip(
        config.label_budget_fractions, config.label_budget_epochs, strict=True
    ):
        if fraction in existing:
            LOGGER.info("Reusing completed label-budget point at %.0f%%", fraction * 100)
            points.append(existing[fraction])
            continue
        LOGGER.info("Label budget %.0f%% (%d epochs)", fraction * 100, epochs)
        result = train_model(
            config,
            model_name="pose_transformer",
            split="signer_holdout",
            epochs=epochs,
            seed=config.label_budget_seed,
            train_fraction=fraction,
            run_name=f"label_budget_{int(fraction * 100):03d}",
            checkpoint_name=f"label_budget_{int(fraction * 100):03d}.pt",
            compute_test=True,
            progress=False,
            enable_mlflow=False,
        )
        test_metrics = result.get("test", {})
        point = {
            "label_fraction": fraction,
            "labels": result["train_examples"],
            "epochs": epochs,
            "test_top1": float(test_metrics.get("top1", 0.0)),
            "test_top5": float(test_metrics.get("top5", 0.0)),
            "checkpoint": result["checkpoint"],
        }
        points.append(point)
        partial = {
            "model": "pose_transformer",
            "split": "signer_holdout",
            "seed": config.label_budget_seed,
            "points": points,
        }
        partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8")

    payload = {
        "model": "pose_transformer",
        "split": "signer_holdout",
        "seed": config.label_budget_seed,
        "points": points,
    }
    (reports / "label_budget.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Label-Budget Study",
        "",
        "| Label fraction | Training clips | Epochs | Test top-1 | Test top-5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        lines.append(
            f"| {point['label_fraction'] * 100:.0f}% | {point['labels']} | {point['epochs']} | "
            f"{point['test_top1']:.3f} | {point['test_top5']:.3f} |"
        )
    (reports / "label_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a torch checkpoint or linear probe from disk."""
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_config_obj = SimpleNamespace(**payload["config"])
    if payload.get("kind") == "linear_probe":
        probe = PooledLinearProbe.from_state(payload["probe_state"])
        return probe, payload
    model = build_model(payload["model_name"], model_config_obj)
    assert isinstance(model, nn.Module)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def torch_predict(model: nn.Module, features: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return class probabilities from a torch model."""
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def prepare_model_config(
    path: str | Path = "configs/model.yaml",
    *,
    epochs: int | None = None,
    seed: int | None = None,
) -> ModelConfig:
    """Load model config with optional CLI overrides."""
    config = model_config(path)
    if epochs is not None:
        config.epochs = int(epochs)
    if seed is not None:
        config.seed = int(seed)
    return config
