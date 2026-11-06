"""Command-line interface for the SignStreamASL pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

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
    build_data.set_defaults(func=_stub("build-data"))

    extract = subparsers.add_parser(
        "extract-keypoints", help="Extract 61-landmark sequences from video files"
    )
    extract.add_argument("--videos-dir", required=True)
    extract.add_argument("--out", default="artifacts/data/keypoints")
    extract.add_argument("--config", default="configs/data.yaml")
    extract.set_defaults(func=_stub("extract-keypoints"))

    train = subparsers.add_parser("train", help="Train a temporal recognizer")
    train.add_argument("--config", default="configs/model.yaml")
    train.add_argument(
        "--model", default="pose_transformer", choices=["pose_transformer", "bigru", "linear_probe"]
    )
    train.add_argument("--split", default="signer_holdout", choices=["random", "signer_holdout"])
    train.add_argument("--epochs", type=int, default=None, help="Override configured epochs")
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(func=_stub("train"))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint with fairness analysis")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", default="signer_holdout", choices=["random", "signer_holdout"])
    evaluate.add_argument("--config", default="configs/eval.yaml")
    evaluate.set_defaults(func=_stub("evaluate"))

    label_budget = subparsers.add_parser("label-budget", help="Measure test accuracy vs label budget")
    label_budget.add_argument("--model", default="pose_transformer")
    label_budget.add_argument("--config", default="configs/model.yaml")
    label_budget.set_defaults(func=_stub("label-budget"))

    decode = subparsers.add_parser("decode", help="Decode continuous signing into gloss events")
    decode.add_argument("--input", default="artifacts/data/continuous/")
    decode.add_argument("--video", default=None)
    decode.add_argument("--webcam", action="store_true")
    decode.add_argument("--threshold", type=float, default=0.60)
    decode.add_argument("--checkpoint", default="artifacts/checkpoints/pose_transformer_signer_holdout.pt")
    decode.add_argument("--config", default="configs/eval.yaml")
    decode.set_defaults(func=_stub("decode"))

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
    regress.set_defaults(func=_stub("regress"))

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
