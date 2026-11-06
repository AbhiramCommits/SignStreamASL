"""Thin MLflow tracking wrapper with a JSONL-only degradation path."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class RunTracker:
    """Log params, metrics, and artifacts to local MLflow and JSONL.

    MLflow is optional at runtime: if it is missing or raises, metrics still
    land in ``<run_dir>/metrics.jsonl`` so results are never lost.
    """

    def __init__(
        self,
        run_name: str,
        *,
        tracking_uri: str = "mlruns",
        run_dir: str | Path = "artifacts/runs",
        enabled: bool = True,
    ) -> None:
        self.run_name = run_name
        self.run_dir = Path(run_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.tracking_uri = tracking_uri
        self.enabled = bool(enabled)
        self._mlflow: Any = None
        self._active = False
        self._run_id: str | None = None

    def start_run(self, params: dict[str, Any] | None = None) -> str:
        """Start a local MLflow run, or quietly continue JSONL-only."""
        if not self.enabled:
            (self.run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_name": self.run_name,
                        "mlflow_run_id": None,
                        "mlflow_active": False,
                        "tracking_uri": self.tracking_uri,
                        "disabled": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return self.run_name
        try:
            import mlflow

            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            uri = self.tracking_uri
            if "://" not in uri:
                uri = Path(uri).resolve().as_uri()
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("signstream")
            run = mlflow.start_run(run_name=self.run_name)
            self._mlflow = mlflow
            self._active = True
            self._run_id = run.info.run_id
            mlflow.set_tags({"pipeline": "signstream", "run_name": self.run_name})
            if params:
                self.log_params(params)
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            LOGGER.warning("MLflow unavailable; using JSONL only: %s", exc)
        (self.run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_name": self.run_name,
                    "mlflow_run_id": self._run_id,
                    "mlflow_active": self._active,
                    "tracking_uri": self.tracking_uri,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return self._run_id or self.run_name

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters to MLflow and a local JSON file."""
        (self.run_dir / "params.json").write_text(
            json.dumps(params, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        if self._active and self._mlflow is not None:
            try:
                self._mlflow.log_params({key: str(value) for key, value in params.items()})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append metrics to JSONL and forward them to MLflow."""
        record = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self._active and self._mlflow is not None:
            try:
                self._mlflow.log_metrics(
                    {key: float(value) for key, value in metrics.items()}, step=step
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MLflow log_metrics failed: %s", exc)

    def log_artifact(self, path: str | Path) -> None:
        """Copy an artifact into the MLflow run when available."""
        artifact_path = Path(path)
        if self._active and self._mlflow is not None and artifact_path.exists():
            try:
                self._mlflow.log_artifact(str(artifact_path))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MLflow log_artifact failed: %s", exc)

    def end_run(self) -> None:
        """Finish the MLflow run."""
        if self._active and self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MLflow end_run failed: %s", exc)
        self._active = False

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.end_run()


def start_run(
    run_name: str,
    *,
    tracking_uri: str = "mlruns",
    run_dir: str | Path = "artifacts/runs",
    params: dict[str, Any] | None = None,
) -> RunTracker:
    """Create and start a tracker."""
    tracker = RunTracker(run_name, tracking_uri=tracking_uri, run_dir=run_dir)
    tracker.start_run(params)
    return tracker


def log_params(tracker: RunTracker, params: dict[str, Any]) -> None:
    """Log parameters through a tracker."""
    tracker.log_params(params)


def log_metrics(tracker: RunTracker, metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics through a tracker."""
    tracker.log_metrics(metrics, step)


def log_artifact(tracker: RunTracker, path: str | Path) -> None:
    """Log an artifact through a tracker."""
    tracker.log_artifact(path)
