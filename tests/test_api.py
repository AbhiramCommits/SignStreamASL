"""FastAPI endpoint and micro-batcher tests."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signstream.serve.app import create_app
from signstream.serve.batcher import MicroBatcher


class FakeBackend:
    name = "fake"
    model_name = "fake-model"
    manifest_hash = "abc123"
    class_names = [f"G{index:02d}" for index in range(60)]

    def predict(self, batch: np.ndarray) -> np.ndarray:
        probabilities = np.full((len(batch), 60), 0.001, dtype=np.float32)
        probabilities[:, 0] = 0.95
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


@pytest.fixture()
def client():
    app = create_app(FakeBackend())
    with TestClient(app) as test_client:
        yield test_client


def _sequence() -> list[list[list[float]]]:
    return np.zeros((48, 61, 3), dtype=float).tolist()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "fake"
    assert body["class_count"] == 60


def test_predict_valid(client: TestClient) -> None:
    response = client.post("/predict", json={"sequence": _sequence()})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top5"]) == 5
    assert body["top5"][0]["gloss"] == "G00"
    assert body["abstain"] is False


def test_predict_rejects_bad_shape(client: TestClient) -> None:
    sequence = _sequence()
    sequence[0][0] = [0.1, 0.2]
    response = client.post("/predict", json={"sequence": sequence})
    assert response.status_code == 422
    empty = client.post("/predict", json={"sequence": []})
    assert empty.status_code == 422


def test_batch_endpoint(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"sequences": [_sequence(), _sequence()]})
    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert len(body["predictions"]) == 2


def test_metrics_endpoint(client: TestClient) -> None:
    client.post("/predict", json={"sequence": _sequence()})
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "signstream_requests_total" in text
    assert 'signstream_latency_ms{quantile="0.95"}' in text
    assert "signstream_batch_size_bucket" in text


def test_video_endpoint_handles_bad_upload(client: TestClient) -> None:
    response = client.post(
        "/predict/video", files={"file": ("bad.mp4", b"not-a-video", "video/mp4")}
    )
    assert response.status_code in {400, 503}


class _SlowBackend(FakeBackend):
    def __init__(self) -> None:
        self.batches: list[int] = []

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.batches.append(len(batch))
        time.sleep(0.02)
        return super().predict(batch)


def test_micro_batcher_coalesces_concurrent_requests() -> None:
    backend = _SlowBackend()
    batch_sizes: list[int] = []

    async def scenario() -> list[np.ndarray]:
        batcher = MicroBatcher(
            backend, max_batch=32, max_wait_ms=10, on_batch=batch_sizes.append
        )
        batcher.start()
        features = np.zeros((48, 183), dtype=np.float32)
        results = await asyncio.gather(*(batcher.submit(features) for _ in range(10)))
        await batcher.close()
        return results

    results = asyncio.run(scenario())
    assert len(results) == 10
    assert max(batch_sizes) > 1
    assert max(backend.batches) > 1
