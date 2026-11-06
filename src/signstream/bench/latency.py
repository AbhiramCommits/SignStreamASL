"""Latency benchmarks for eager Torch, ONNX Runtime, and streaming decode."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signstream.config import eval_config
from signstream.serve.runtime import OnnxBackend, TorchBackend

LOGGER = logging.getLogger(__name__)


def _benchmark_callable(
    function,
    batch_size: int,
    *,
    warmup: int = 50,
    iterations: int = 300,
) -> dict[str, float]:
    features = np.random.default_rng(123).normal(size=(batch_size, 48, 183)).astype(np.float32)
    for _ in range(warmup):
        function(features)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        function(features)
        timings.append((time.perf_counter() - start) * 1000.0)
    array = np.asarray(timings, dtype=np.float64)
    p50 = float(np.percentile(array, 50))
    return {
        "p50_ms": p50,
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "throughput_clips_s": float(batch_size / (p50 / 1000.0)),
        "per_clip_ms": float(p50 / batch_size),
    }


def benchmark_latency(
    checkpoint: str = "artifacts/checkpoints/pose_transformer_signer_holdout.pt",
    onnx_path: str = "artifacts/onnx/pose_transformer.onnx",
    *,
    batch_sizes: list[int] | None = None,
    warmup: int = 50,
    iterations: int = 300,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark both backends and the streaming window budget."""
    sizes = batch_sizes or [1, 8, 32]
    torch_backend = TorchBackend(checkpoint)
    onnx_backend = OnnxBackend(onnx_path, checkpoint)
    results: dict[str, Any] = {}
    for name, backend in (("torch", torch_backend), ("onnx", onnx_backend)):
        backend_rows = []
        for batch_size in sizes:
            row = {
                "batch_size": batch_size,
                **_benchmark_callable(
                    backend.predict,
                    batch_size,
                    warmup=warmup,
                    iterations=iterations,
                ),
            }
            backend_rows.append(row)
            LOGGER.info(
                "%s batch=%d p50=%.2fms p95=%.2fms throughput=%.1f clips/s",
                name,
                batch_size,
                row["p50_ms"],
                row["p95_ms"],
                row["throughput_clips_s"],
            )
        results[name] = backend_rows

    streaming = _benchmark_callable(
        onnx_backend.predict, 1, warmup=warmup, iterations=iterations
    )
    stride = eval_config().stride
    window_throughput = streaming["throughput_clips_s"]
    max_realtime_fps = window_throughput * stride
    results["streaming"] = {
        "backend": "onnx",
        "window_frames": 48,
        "stride_frames": stride,
        "ms_per_window_p50": streaming["p50_ms"],
        "ms_per_window_p95": streaming["p95_ms"],
        "windows_per_second": window_throughput,
        "required_windows_per_second_at_30fps": 30.0 / stride,
        "implied_max_realtime_fps": max_realtime_fps,
        "webcam_requirement_fps": 30.0,
        "realtime_viable_at_30fps": bool(max_realtime_fps >= 30.0),
    }
    payload = {
        "checkpoint": str(checkpoint),
        "onnx_path": str(onnx_path),
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "batch_sizes": sizes,
        "results": results,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "numpy_version": np.__version__,
        },
    }
    reports = Path(reports_dir) if reports_dir is not None else Path(eval_config().reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "latency.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Latency Benchmarks",
        "",
        f"{warmup} warmup + {iterations} timed iterations per configuration.",
        "",
        "| Backend | Batch | p50 ms | p95 ms | p99 ms | Clips/s | Per-clip ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in ("torch", "onnx"):
        for row in results[backend]:
            lines.append(
                f"| {backend} | {row['batch_size']} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | "
                f"{row['p99_ms']:.3f} | {row['throughput_clips_s']:.1f} | {row['per_clip_ms']:.3f} |"
            )
    stream = results["streaming"]
    lines.extend(
        [
            "",
            "## Streaming (48-frame window, stride 8)",
            "",
            f"- p50 per window: **{stream['ms_per_window_p50']:.3f} ms**",
            f"- p95 per window: **{stream['ms_per_window_p95']:.3f} ms**",
            f"- Implied maximum real-time frame rate: **{stream['implied_max_realtime_fps']:.1f} FPS**",
            f"- 30 FPS webcam viable: **{'yes' if stream['realtime_viable_at_30fps'] else 'no'}**",
            "",
        ]
    )
    (reports / "latency.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
