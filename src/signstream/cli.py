"""Command-line interface for the SignStreamASL pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from signstream import __version__
from signstream.logging_utils import setup_logging


def _print_args(command: str, args: argparse.Namespace) -> int:
    payload = {k: v for k, v in vars(args).items() if k not in {"func", "command"}}
    print(f"{command}: {json.dumps(payload, indent=2, sort_keys=True, default=str)}")
    return 0


def _stub(command: str):
    def run(args: argparse.Namespace) -> int:
        return _print_args(command, args)

    return run


def _cmd_build_data(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from signstream.config import data_config
    from signstream.data.corpus import generate_corpus, save_corpus
    from signstream.data.curate import curate_all
    from signstream.data.dataset import load_records
    from signstream.data.manifest import write_manifest
    from signstream.data.splits import coverage_stats, write_split_files

    config = data_config(args.config)
    if args.coverage_only:
        records = load_records(config)
        curation_path = Path(config.data_dir) / "curation_report.json"
        if curation_path.exists():
            kept = set(json.loads(curation_path.read_text(encoding="utf-8")).get("kept_ids", []))
            records = [record for record in records if record["clip_id"] in kept]
        summary = coverage_stats(records, reports_dir=config.reports_dir)
        print(summary["markdown"])
        return 0

    quick = bool(args.quick or config.quick)
    clips, records = generate_corpus(config, quick=quick)
    save_corpus(clips, records, config, write_individual=not quick)
    curation = curate_all(clips, records, config)
    splits = write_split_files(curation.kept_records, config)
    manifest = write_manifest(
        config.manifest_version,
        config,
        curation.kept_records,
        curation.report,
        splits,
    )
    coverage = coverage_stats(curation.kept_records, reports_dir=config.reports_dir)
    summary = {
        "clips_generated": len(records),
        "clips_kept": len(curation.kept_records),
        "counts_by_reason": curation.report["counts_by_reason"],
        "label_audit": {
            key: curation.report["label_audit"][key]
            for key in ("precision", "recall", "n_flagged", "injected_total", "true_positives")
        },
        "manifest_hash": manifest["dataset_manifest_hash"],
        "coverage": coverage["clips_per_gloss"],
        "splits": {
            name: {key: len(payload[key]) for key in ("train", "val", "test")}
            for name, payload in splits.items()
        },
        "quick": quick,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_extract_keypoints(args: argparse.Namespace) -> int:
    import json

    from signstream.data.keypoints import (
        MissingDependencyError,
        extract_keypoints_directory,
    )

    try:
        log = extract_keypoints_directory(args.videos_dir, args.out)
    except MissingDependencyError as exc:
        print(f"Capture dependencies unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: log[key] for key in ("count", "ok", "failed")}, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import json

    from signstream.train import prepare_model_config, train_model

    config = prepare_model_config(args.config, epochs=args.epochs, seed=args.seed)
    report = train_model(
        config,
        model_name=args.model,
        split=args.split,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "model": report["model"],
                "split": report["split"],
                "checkpoint": report["checkpoint"],
                "parameters": report["parameters"],
                "best_epoch": report["best_epoch"],
                "best_val": report["best_val"],
                "train_examples": report["train_examples"],
                "run_id": report["run_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_label_budget(args: argparse.Namespace) -> int:
    import json

    from signstream.train import prepare_model_config, run_label_budget

    config = prepare_model_config(args.config)
    payload = run_label_budget(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    from signstream.config import eval_config
    from signstream.eval.report import evaluate_checkpoint

    config = eval_config(args.config)
    report = evaluate_checkpoint(args.checkpoint, split=args.split, config=config)
    print(
        json.dumps(
            {
                "model": report["model"],
                "split": report["split"],
                "top1": report["metrics"]["top1"],
                "top5": report["metrics"]["top5"],
                "macro_f1": report["metrics"]["macro_f1"],
                "ece": report["metrics"]["ece"],
                "fairness_gap": report["fairness"]["fairness_gap"],
                "high_low_top1_gap": report["fairness"]["high_low_top1_gap"],
                "artifacts": report["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _print_events(title: str, events: list[object]) -> None:
    print(title)
    for event in events:
        print(
            f"  {event.start_s:6.3f}s - {event.end_s:6.3f}s  {event.gloss:<14s} "
            f"confidence={event.confidence:.3f}"
        )


def _cmd_decode(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from signstream.config import eval_config
    from signstream.data.keypoints import MissingDependencyError
    from signstream.stream.decoder import (
        decode_video,
        decode_webcam,
        decoder_from_checkpoint,
        evaluate_continuous,
    )
    from signstream.stream.windowing import build_continuous_dataset

    config = eval_config(args.config)
    try:
        decoder, payload = decoder_from_checkpoint(
            args.checkpoint, config=config, threshold=args.threshold
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.video:
            result, _ = decode_video(args.video, decoder)
            print(f"Decoded video: {args.video}")
            _print_events("Gloss events:", result.events)
            print(f"Abstain rate: {result.abstain_rate:.3f}")
            return 0
        if args.webcam:
            result, _ = decode_webcam(decoder)
            _print_events("Gloss events:", result.events)
            print(f"Abstain rate: {result.abstain_rate:.3f}")
            return 0
    except MissingDependencyError as exc:
        print(f"Capture dependencies unavailable: {exc}", file=sys.stderr)
        return 2

    directory = Path(args.input)
    if not (directory / "index.json").exists():
        build_continuous_dataset(eval_cfg=config)
    payload = evaluate_continuous(
        decoder,
        continuous_dir=directory,
        thresholds=config.threshold_sweep,
        primary_threshold=args.threshold,
    )
    reports = Path(config.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "continuous_eval.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Continuous Decoding",
        "",
        f"Sentences: {payload['n_sentences']}, gold signs: {payload['total_gold_signs']}.",
        "",
        "| Threshold | Precision | Recall | F1 | Gloss error rate | Abstain rate | Events |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.1f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['gloss_error_rate']:.3f} | "
            f"{row['abstain_rate']:.3f} | {row['predicted_events']} |"
        )
    (reports / "continuous_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for example in payload["examples"]:
        print(
            f"{example['sentence_id']} gold={' '.join(example['gold_glosses'])}"
        )
        for event in example["events"]:
            print(
                f"  {event['start_s']:6.3f}s - {event['end_s']:6.3f}s  "
                f"{event['gloss']:<14s} confidence={event['confidence']:.3f}"
            )
    primary = payload["primary"]
    print(
        f"Primary threshold={primary['threshold']:.2f}: P={primary['precision']:.3f} "
        f"R={primary['recall']:.3f} F1={primary['f1']:.3f} "
        f"GER={primary['gloss_error_rate']:.3f} abstain={primary['abstain_rate']:.3f}"
    )
    print(f"Wrote {report_path}")
    return 0


def _cmd_regress(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import yaml

    from signstream.config import eval_config
    from signstream.eval.gold import build_gold_set, evaluate_gold
    from signstream.eval.report import evaluate_checkpoint

    thresholds = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gold_cfg = thresholds.get("gold", {})
    checkpoint = args.checkpoint or gold_cfg.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        print(
            f"Regression checkpoint missing: {checkpoint}. Train it or pass --checkpoint.",
            file=sys.stderr,
        )
        return 2
    gold_path = Path(eval_config().gold_path)
    if not gold_path.exists():
        build_gold_set()
    gold_metrics = evaluate_gold(checkpoint, gold_path=gold_path)
    split = str(gold_cfg.get("split", "signer_holdout"))
    evaluation = evaluate_checkpoint(checkpoint, split=split)
    fairness_gap = float(evaluation["fairness"]["fairness_gap"])
    actual = {
        "top1": float(gold_metrics["top1"]),
        "top5": float(gold_metrics["top5"]),
        "macro_f1": float(gold_metrics["macro_f1"]),
        "max_fairness_gap": fairness_gap,
    }
    baseline = gold_cfg.get("baseline", {})
    tolerance = gold_cfg.get("tolerance", {})
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    directions = {
        "top1": "min",
        "top5": "min",
        "macro_f1": "min",
        "max_fairness_gap": "max",
    }
    for key, direction in directions.items():
        expected = float(baseline.get(key, 0.0))
        tol = float(tolerance.get(f"{key}_absolute", 0.02))
        value = actual[key]
        passed = value >= expected - tol if direction == "min" else value <= expected + tol
        if not passed:
            failures.append(key)
        checks.append(
            {
                "metric": key,
                "actual": value,
                "baseline": expected,
                "tolerance": tol,
                "direction": direction,
                "passed": bool(passed),
            }
        )
    payload = {
        "checkpoint": str(checkpoint),
        "gold_metrics": gold_metrics,
        "actual": actual,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }
    reports = Path("artifacts/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "regression.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("| Metric | Actual | Baseline | Tolerance | Direction | Status |")
    print("|---|---:|---:|---:|---|---|")
    for check in checks:
        print(
            f"| {check['metric']} | {check['actual']:.4f} | {check['baseline']:.4f} | "
            f"{check['tolerance']:.4f} | {check['direction']} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    if failures:
        print(f"Regression gate failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Regression gate passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser with all pipeline stages."""
    parser = argparse.ArgumentParser(
        prog="signstream",
        description="End-to-end ASL recognition: data, models, fairness, decoding, serving.",
    )
    parser.add_argument("--version", action="version", version=f"signstream {__version__}")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    build_data = subparsers.add_parser("build-data", help="Generate and curate the signing corpus")
    build_data.add_argument("--config", default="configs/data.yaml")
    build_data.add_argument("--coverage-only", action="store_true")
    build_data.add_argument("--quick", action="store_true", help="Generate a small CI data slice")
    build_data.set_defaults(func=_cmd_build_data)

    extract = subparsers.add_parser(
        "extract-keypoints", help="Extract 61-landmark sequences from video files"
    )
    extract.add_argument("--videos-dir", required=True)
    extract.add_argument("--out", default="artifacts/data/keypoints")
    extract.add_argument("--config", default="configs/data.yaml")
    extract.set_defaults(func=_cmd_extract_keypoints)

    train = subparsers.add_parser("train", help="Train a temporal recognizer")
    train.add_argument("--config", default="configs/model.yaml")
    train.add_argument(
        "--model", default="pose_transformer", choices=["pose_transformer", "bigru", "linear_probe"]
    )
    train.add_argument("--split", default="signer_holdout", choices=["random", "signer_holdout"])
    train.add_argument("--epochs", type=int, default=None, help="Override configured epochs")
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint with fairness analysis")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", default="signer_holdout", choices=["random", "signer_holdout"])
    evaluate.add_argument("--config", default="configs/eval.yaml")
    evaluate.set_defaults(func=_cmd_evaluate)

    label_budget = subparsers.add_parser("label-budget", help="Measure test accuracy vs label budget")
    label_budget.add_argument("--model", default="pose_transformer")
    label_budget.add_argument("--config", default="configs/model.yaml")
    label_budget.set_defaults(func=_cmd_label_budget)

    decode = subparsers.add_parser("decode", help="Decode continuous signing into gloss events")
    decode.add_argument("--input", default="artifacts/data/continuous/")
    decode.add_argument("--video", default=None)
    decode.add_argument("--webcam", action="store_true")
    decode.add_argument("--threshold", type=float, default=0.60)
    decode.add_argument("--checkpoint", default="artifacts/checkpoints/pose_transformer_signer_holdout.pt")
    decode.add_argument("--config", default="configs/eval.yaml")
    decode.set_defaults(func=_cmd_decode)

    export = subparsers.add_parser("export-onnx", help="Export a checkpoint and verify ONNX parity")
    export.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/pose_transformer_signer_holdout.pt",
    )
    export.add_argument("--out", default="artifacts/onnx/pose_transformer.onnx")
    export.set_defaults(func=_stub("export-onnx"))

    bench = subparsers.add_parser("bench", help="Benchmark Torch and ONNX Runtime latency")
    bench.add_argument("--all", action="store_true")
    bench.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/pose_transformer_signer_holdout.pt",
    )
    bench.add_argument("--onnx", default="artifacts/onnx/pose_transformer.onnx")
    bench.add_argument("--batch-sizes", default="1,8,32")
    bench.set_defaults(func=_stub("bench"))

    regress = subparsers.add_parser("regress", help="Run the gold-set regression gate")
    regress.add_argument("--config", default="configs/thresholds.yaml")
    regress.add_argument("--checkpoint", default=None)
    regress.set_defaults(func=_cmd_regress)

    serve = subparsers.add_parser("serve", help="Serve the inference API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--backend", default=None, choices=["torch", "onnx"])
    serve.set_defaults(func=_stub("serve"))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, configure logging, and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
