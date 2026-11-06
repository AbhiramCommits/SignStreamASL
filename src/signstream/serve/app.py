"""FastAPI inference service with micro-batching and Prometheus-style metrics."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from signstream import __version__
from signstream.config import eval_config
from signstream.data.keypoints import (
    MediaPipeExtractor,
    MissingDependencyError,
    normalize_sequence,
    resample_sequence,
)
from signstream.serve.batcher import MicroBatcher
from signstream.serve.runtime import InferenceBackend, build_backend

LOGGER = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """A single keypoint sequence shaped ``[T, 61, 3]``."""

    sequence: list[list[list[float]]] = Field(..., description="Keypoints [T, 61, 3]")

    @field_validator("sequence")
    @classmethod
    def validate_shape(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        if not value:
            raise ValueError("sequence must contain at least one frame")
        for frame in value:
            if len(frame) != 61:
                raise ValueError("each frame must contain exactly 61 landmarks")
            for point in frame:
                if len(point) != 3:
                    raise ValueError("each landmark must contain exactly 3 coordinates")
        return value


class BatchPredictRequest(BaseModel):
    """A batch of keypoint sequences."""

    sequences: list[list[list[list[float]]]] = Field(..., description="List of [T, 61, 3]")

    @field_validator("sequences")
    @classmethod
    def validate_batch(cls, value: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
        if not value:
            raise ValueError("sequences must not be empty")
        if len(value) > 32:
            raise ValueError("at most 32 sequences per batch")
        for sequence in value:
            PredictRequest.validate_shape(sequence)
        return value


class TopKPrediction(BaseModel):
    gloss: str
    confidence: float


class PredictResponse(BaseModel):
    top5: list[TopKPrediction]
    abstain: bool
    threshold: float
    model: str
    backend: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    batch_size: int


def _classify(
    probabilities: np.ndarray,
    class_names: list[str],
    *,
    threshold: float,
    model: str,
    backend: str,
) -> PredictResponse:
    order = np.argsort(-probabilities)[:5]
    return PredictResponse(
        top5=[
            TopKPrediction(gloss=class_names[int(index)], confidence=float(probabilities[index]))
            for index in order
        ],
        abstain=bool(float(probabilities.max()) < threshold),
        threshold=float(threshold),
        model=model,
        backend=backend,
    )


def _prepare_sequence(sequence: list[list[list[float]]]) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (61, 3):
        raise ValueError("sequence must be [T, 61, 3]")
    array = normalize_sequence(array)
    array = resample_sequence(array, 48)
    return np.nan_to_num(array, nan=0.0).astype(np.float32)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100.0))


def create_app(
    backend: InferenceBackend | None = None,
    *,
    checkpoint: str | None = None,
    device: str = "cpu",
) -> FastAPI:
    """Create the inference app, optionally injecting a backend for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected = backend
        if selected is None:
            selected = build_backend(checkpoint=checkpoint) if checkpoint else build_backend()
        app.state.backend = selected
        app.state.started_at = time.time()
        app.state.request_count = 0
        app.state.predict_count = 0
        app.state.abstain_count = 0
        app.state.latencies_ms = []
        app.state.batch_sizes = []
        app.state.threshold = eval_config().confidence_threshold
        app.state.batcher = MicroBatcher(
            selected,
            max_batch=32,
            max_wait_ms=10.0,
            on_batch=lambda size: app.state.batch_sizes.append(int(size)),
        )
        app.state.batcher.start()
        LOGGER.info("Inference service ready with %s backend", selected.name)
        try:
            yield
        finally:
            await app.state.batcher.close()

    application = FastAPI(
        title="SignStreamASL",
        version=__version__,
        description="ASL pose-sequence recognition and continuous gloss decoding",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        application.state.request_count = getattr(application.state, "request_count", 0) + 1
        LOGGER.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["x-request-id"] = request_id
        return response

    @application.get("/health")
    async def health() -> dict[str, Any]:
        state = application.state
        return {
            "status": "ok",
            "model": state.backend.model_name,
            "backend": state.backend.name,
            "class_count": len(state.backend.class_names),
            "manifest_hash": state.backend.manifest_hash,
            "uptime_s": round(time.time() - state.started_at, 3),
            "version": __version__,
        }

    async def _predict_one(sequence: list[list[list[float]]]) -> PredictResponse:
        state = application.state
        features = _prepare_sequence(sequence)
        start = time.perf_counter()
        probabilities = await state.batcher.submit(features)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        state.latencies_ms.append(elapsed_ms)
        state.predict_count += 1
        response = _classify(
            probabilities,
            state.backend.class_names,
            threshold=state.threshold,
            model=state.backend.model_name,
            backend=state.backend.name,
        )
        if response.abstain:
            state.abstain_count += 1
        return response

    @application.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        return await _predict_one(request.sequence)

    @application.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
        responses = await asyncio.gather(
            *[_predict_one(sequence) for sequence in request.sequences]
        )
        return BatchPredictResponse(predictions=list(responses), batch_size=len(responses))

    @application.post("/predict/video")
    async def predict_video(file: UploadFile = File(...)) -> dict[str, Any]:
        from signstream.stream.decoder import (
            SlidingWindowDecoder,
            evaluate_continuous,  # noqa: F401
        )

        config = eval_config()
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temporary_path = Path(temporary.name)
        try:
            temporary.write(await file.read())
            temporary.close()
            with MediaPipeExtractor() as extractor:
                sequence = extractor.extract(temporary_path)
            decoder = SlidingWindowDecoder(
                application.state.backend.predict,
                application.state.backend.class_names,
                window=config.window,
                stride=config.stride,
                ema_windows=config.ema_windows,
                threshold=config.confidence_threshold,
                min_duration_s=config.min_duration_s,
                cooldown_s=config.cooldown_s,
                fps=config.fps,
            )
            result = decoder.decode(sequence)
            return {
                "filename": file.filename,
                "events": [event.to_dict() for event in result.events],
                "abstain_rate": result.abstain_rate,
                "n_windows": result.n_windows,
                "abstain_token": "<ABSTAIN>",
            }
        except MissingDependencyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Video prediction failed")
            raise HTTPException(status_code=400, detail=f"Video decode failed: {exc}") from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    @application.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        state = application.state
        latency = state.latencies_ms[-1000:]
        batches = state.batch_sizes[-1000:]
        lines = [
            "# HELP signstream_requests_total Total HTTP requests received.",
            "# TYPE signstream_requests_total counter",
            f"signstream_requests_total {state.request_count}",
            "# HELP signstream_predictions_total Total single-clip predictions.",
            "# TYPE signstream_predictions_total counter",
            f"signstream_predictions_total {state.predict_count}",
            "# TYPE signstream_abstain_total counter",
            f"signstream_abstain_total {state.abstain_count}",
            "# TYPE signstream_latency_ms summary",
            f'signstream_latency_ms{{quantile="0.5"}} {_percentile(latency, 0.50):.4f}',
            f'signstream_latency_ms{{quantile="0.95"}} {_percentile(latency, 0.95):.4f}',
            f'signstream_latency_ms{{quantile="0.99"}} {_percentile(latency, 0.99):.4f}',
            "# TYPE signstream_batch_size histogram",
        ]
        edges = [1, 2, 4, 8, 16, 32]
        for edge in edges:
            count = sum(1 for size in batches if size <= edge)
            lines.append(f'signstream_batch_size_bucket{{le="{edge}"}} {count}')
        lines.append(f'signstream_batch_size_bucket{{le="+Inf"}} {len(batches)}')
        lines.append(f"signstream_batch_size_count {len(batches)}")
        lines.append(
            f"signstream_abstain_rate {state.abstain_count / max(1, state.predict_count):.6f}"
        )
        return "\n".join(lines) + "\n"

    return application


app = create_app()
