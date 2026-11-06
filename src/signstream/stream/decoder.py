"""Sliding-window continuous ASL gloss decoding."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from signstream.stream.windowing import iter_windows, window_array

LOGGER = logging.getLogger(__name__)

Predictor = Callable[[np.ndarray], np.ndarray]
ABSTAIN_TOKEN = "<ABSTAIN>"


@dataclass
class GlossEvent:
    """One emitted gloss with timestamped span and confidence."""

    gloss: str
    gloss_id: int
    start_s: float
    end_s: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gloss": self.gloss,
            "gloss_id": int(self.gloss_id),
            "start_s": round(float(self.start_s), 3),
            "end_s": round(float(self.end_s), 3),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class DecodeResult:
    """Events and diagnostics for one continuous sequence."""

    events: list[GlossEvent]
    abstain_rate: float
    n_windows: int
    mean_confidence: float
    abstain_segments_s: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "abstain_token": ABSTAIN_TOKEN,
            "abstain_segments_s": self.abstain_segments_s or [],
            "abstain_rate": float(self.abstain_rate),
            "n_windows": int(self.n_windows),
            "mean_confidence": float(self.mean_confidence),
        }


def _temporal_overlap(event_a: GlossEvent, event_b: GlossEvent) -> float:
    overlap = max(0.0, min(event_a.end_s, event_b.end_s) - max(event_a.start_s, event_b.start_s))
    shorter = max(1e-6, min(event_a.end_s - event_a.start_s, event_b.end_s - event_b.start_s))
    return overlap / shorter


class SlidingWindowDecoder:
    """Decode a continuous pose stream into one gloss per detected segment."""

    def __init__(
        self,
        predictor: Predictor,
        class_names: Sequence[str],
        *,
        window: int = 48,
        stride: int = 8,
        ema_windows: int = 3,
        threshold: float = 0.60,
        min_duration_s: float = 0.35,
        cooldown_s: float = 0.35,
        fps: int = 30,
        nms_overlap: float = 0.50,
    ) -> None:
        self.predictor = predictor
        self.class_names = list(class_names)
        self.window = int(window)
        self.stride = int(stride)
        self.ema_windows = max(1, int(ema_windows))
        self.threshold = float(threshold)
        self.min_duration_s = float(min_duration_s)
        self.cooldown_s = float(cooldown_s)
        self.fps = int(fps)
        self.nms_overlap = float(nms_overlap)

    def _predict_windows(self, windows: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.predictor(windows), dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.class_names):
            raise ValueError(
                f"Predictor returned {probabilities.shape}; expected [N, {len(self.class_names)}]"
            )
        return probabilities

    def _decode_probabilities(
        self,
        probabilities: np.ndarray,
        windows: list[tuple[int, int]],
        threshold: float,
    ) -> DecodeResult:
        alpha = 2.0 / (self.ema_windows + 1.0)
        smoothed = np.empty_like(probabilities)
        running = probabilities[0]
        for index, row in enumerate(probabilities):
            running = alpha * row + (1.0 - alpha) * running if index else row
            smoothed[index] = running

        predictions = smoothed.argmax(axis=1)
        confidences = smoothed.max(axis=1)
        abstentions = confidences < threshold
        runs: list[GlossEvent] = []
        run_start: int | None = None
        run_label = -1
        run_peak = 0.0

        def finalize(end_index: int) -> None:
            nonlocal run_start, run_label, run_peak
            if run_start is None:
                return
            start_s = windows[run_start][0] / self.fps
            end_s = windows[end_index][1] / self.fps
            duration = end_s - start_s
            if duration >= self.min_duration_s:
                runs.append(
                    GlossEvent(
                        gloss=self.class_names[int(run_label)],
                        gloss_id=int(run_label),
                        start_s=float(start_s),
                        end_s=float(end_s),
                        confidence=float(run_peak),
                    )
                )
            run_start = None
            run_label = -1
            run_peak = 0.0

        for index, label in enumerate(predictions):
            if abstentions[index]:
                finalize(index - 1)
                continue
            if run_start is None or int(label) != run_label:
                finalize(index - 1)
                run_start = index
                run_label = int(label)
                run_peak = float(confidences[index])
            else:
                run_peak = max(run_peak, float(confidences[index]))
        finalize(len(predictions) - 1)

        # Non-maximum suppression: overlapping detections compete by confidence.
        kept: list[GlossEvent] = []
        for candidate in sorted(runs, key=lambda event: event.confidence, reverse=True):
            if all(
                candidate.gloss_id != existing.gloss_id
                or _temporal_overlap(candidate, existing) <= self.nms_overlap
                for existing in kept
            ):
                kept.append(candidate)
        kept.sort(key=lambda event: event.start_s)

        # Cooldown suppresses duplicate emissions of the same gloss close together.
        events: list[GlossEvent] = []
        for event in kept:
            duplicate = any(
                previous.gloss_id == event.gloss_id
                and event.start_s - previous.end_s < self.cooldown_s
                for previous in events
            )
            if not duplicate:
                events.append(event)

        abstain_rate = float(abstentions.mean()) if len(abstentions) else 1.0
        abstain_segments: list[list[float]] = []
        abstain_start: int | None = None
        for index, flag in enumerate(abstentions):
            if flag and abstain_start is None:
                abstain_start = index
            elif not flag and abstain_start is not None:
                abstain_segments.append(
                    [windows[abstain_start][0] / self.fps, windows[index][1] / self.fps]
                )
                abstain_start = None
        if abstain_start is not None:
            abstain_segments.append(
                [windows[abstain_start][0] / self.fps, windows[-1][1] / self.fps]
            )
        return DecodeResult(
            events=events,
            abstain_rate=abstain_rate,
            n_windows=len(windows),
            mean_confidence=float(confidences.mean()) if len(confidences) else 0.0,
            abstain_segments_s=abstain_segments,
        )

    def decode(
        self,
        sequence: np.ndarray,
        *,
        threshold: float | None = None,
        class_offset: int = 0,
    ) -> DecodeResult:
        """Decode a sequence shaped ``[T, 61, 3]`` or ``[T, 183]``.

        ``class_offset`` maps a subset-classifier's probabilities into the full
        class list. It is mostly useful for tests.
        """
        array = np.asarray(sequence, dtype=np.float32)
        starts = iter_windows(array, window=self.window, stride=self.stride)
        if not starts:
            return DecodeResult(
                events=[],
                abstain_rate=1.0,
                n_windows=0,
                mean_confidence=0.0,
                abstain_segments_s=[[0.0, array.shape[0] / self.fps]],
            )
        windows = window_array(array, starts, window=self.window)
        probabilities = self._predict_windows(windows)
        if class_offset > 0:
            shifted = np.zeros((probabilities.shape[0], len(self.class_names)), dtype=np.float32)
            shifted[:, class_offset : class_offset + probabilities.shape[1]] = probabilities
            probabilities = shifted
        return self._decode_probabilities(
            probabilities,
            starts,
            self.threshold if threshold is None else float(threshold),
        )


def torch_predictor(model: Any, *, batch_size: int = 256) -> Predictor:
    """Wrap a torch model or linear probe as a numpy predictor."""

    def predict(windows: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        if hasattr(model, "predict_proba") and not isinstance(model, torch.nn.Module):
            for start in range(0, len(windows), batch_size):
                outputs.append(model.predict_proba(windows[start : start + batch_size]))
            return np.concatenate(outputs, axis=0)
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.as_tensor(windows[start : start + batch_size], dtype=torch.float32)
                logits = model(batch)
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    return predict


def decoder_from_checkpoint(
    checkpoint: str | Any,
    *,
    config: Any | None = None,
    threshold: float | None = None,
) -> tuple[SlidingWindowDecoder, dict[str, Any]]:
    """Load a checkpoint and build a configured decoder."""
    from signstream.config import eval_config
    from signstream.train import load_checkpoint_model

    eval_cfg = config or eval_config()
    model, payload = load_checkpoint_model(checkpoint)
    class_names = list(payload.get("class_names", []))
    decoder = SlidingWindowDecoder(
        torch_predictor(model),
        class_names,
        window=eval_cfg.window,
        stride=eval_cfg.stride,
        ema_windows=eval_cfg.ema_windows,
        threshold=eval_cfg.confidence_threshold if threshold is None else threshold,
        min_duration_s=eval_cfg.min_duration_s,
        cooldown_s=eval_cfg.cooldown_s,
        fps=eval_cfg.fps,
    )
    return decoder, payload


def match_events(
    predicted: list[GlossEvent],
    gold_ids: Sequence[int],
    gold_spans: Sequence[Sequence[float]],
    *,
    class_names: Sequence[str],
) -> tuple[int, int, int]:
    """Greedy match predicted gloss events to gold spans by gloss and midpoint."""
    used: set[int] = set()
    matches = 0
    for event in predicted:
        midpoint = (event.start_s + event.end_s) / 2.0
        for index, (gold_id, span) in enumerate(zip(gold_ids, gold_spans, strict=True)):
            if index in used or int(gold_id) != event.gloss_id:
                continue
            if float(span[0]) <= midpoint <= float(span[1]):
                used.add(index)
                matches += 1
                break
    return matches, len(predicted), len(gold_ids)


def edit_distance(predicted: Sequence[int], gold: Sequence[int]) -> int:
    """Levenshtein distance between two gloss-id sequences."""
    if not predicted:
        return len(gold)
    if not gold:
        return len(predicted)
    previous = list(range(len(gold) + 1))
    for row, pred in enumerate(predicted, start=1):
        current = [row]
        for column, target in enumerate(gold, start=1):
            substitution = previous[column - 1] + (pred != target)
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, substitution)
            )
        previous = current
    return int(previous[-1])


def evaluate_continuous(
    decoder: SlidingWindowDecoder,
    *,
    continuous_dir: str | Any = "artifacts/data/continuous",
    thresholds: Sequence[float] = (0.3, 0.5, 0.6, 0.7, 0.8),
    primary_threshold: float = 0.6,
) -> dict[str, Any]:
    """Decode every synthetic sentence and evaluate a threshold sweep."""
    from signstream.stream.windowing import load_continuous_sentence

    directory = Path(continuous_dir)
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    sentences = [
        (metadata, load_continuous_sentence(directory / metadata["file"]))
        for metadata in index["sentences"]
    ]
    class_names = decoder.class_names
    sweep: list[dict[str, Any]] = []
    primary_events: list[dict[str, Any]] = []
    for threshold in thresholds:
        matches = predicted_total = gold_total = 0
        abstain_rates: list[float] = []
        event_counts: list[int] = []
        edit_total = 0
        ordered_predictions: list[list[dict[str, Any]]] = []
        for sentence_index, (metadata, sentence) in enumerate(sentences):
            result = decoder.decode(sentence["keypoints"], threshold=threshold)
            sentence_matches, sentence_predicted, sentence_gold = match_events(
                result.events,
                sentence["gold_ids"],
                sentence["gold_spans"],
                class_names=class_names,
            )
            matches += sentence_matches
            predicted_total += sentence_predicted
            gold_total += sentence_gold
            abstain_rates.append(result.abstain_rate)
            event_counts.append(len(result.events))
            predicted_ids = [event.gloss_id for event in result.events]
            edit_total += edit_distance(predicted_ids, sentence["gold_ids"].tolist())
            if abs(threshold - primary_threshold) < 1e-9 and sentence_index < 5:
                ordered_predictions.append(
                    {
                        "sentence_id": metadata["sentence_id"],
                        "gold_glosses": [class_names[int(i)] for i in sentence["gold_ids"]],
                        "events": [event.to_dict() for event in result.events],
                    }
                )
        precision = matches / predicted_total if predicted_total else 0.0
        recall = matches / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "gloss_error_rate": float(edit_total / max(1, gold_total)),
            "abstain_rate": float(np.mean(abstain_rates)) if abstain_rates else 1.0,
            "matches": int(matches),
            "predicted_events": int(predicted_total),
            "gold_events": int(gold_total),
            "mean_events_per_sentence": float(np.mean(event_counts)) if event_counts else 0.0,
        }
        sweep.append(row)
        if abs(threshold - primary_threshold) < 1e-9:
            primary_events = ordered_predictions

    payload = {
        "split": "signer_holdout",
        "n_sentences": len(sentences),
        "total_gold_signs": int(sum(len(item["gold_ids"]) for item in index["sentences"])),
        "total_frames": int(sum(item["num_frames"] for item in index["sentences"])),
        "primary_threshold": float(primary_threshold),
        "primary": next(
            (row for row in sweep if abs(row["threshold"] - primary_threshold) < 1e-9), None
        ),
        "threshold_sweep": sweep,
        "examples": primary_events,
        "decoder": {
            "window": decoder.window,
            "stride": decoder.stride,
            "ema_windows": decoder.ema_windows,
            "min_duration_s": decoder.min_duration_s,
            "cooldown_s": decoder.cooldown_s,
            "fps": decoder.fps,
        },
    }
    return payload


def decode_video(
    path: str | Any,
    decoder: SlidingWindowDecoder,
) -> tuple[DecodeResult, np.ndarray]:
    """Extract MediaPipe keypoints from a video and decode gloss events."""
    from signstream.data.keypoints import MediaPipeExtractor, normalize_sequence

    with MediaPipeExtractor() as extractor:
        sequence = extractor.extract(path)
    sequence = normalize_sequence(sequence)
    return decoder.decode(sequence), sequence


def decode_webcam(
    decoder: SlidingWindowDecoder,
    *,
    max_frames: int = 300,
) -> tuple[DecodeResult, np.ndarray]:
    """Capture keypoints from a webcam via OpenCV, then decode gloss events."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        from signstream.data.keypoints import MissingDependencyError

        raise MissingDependencyError(
            'Webcam decoding requires opencv and mediapipe: pip install "signstream[capture]"'
        ) from exc
    from signstream.data.keypoints import MediaPipeExtractor, normalize_sequence

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open webcam at index 0")
    frames: list[np.ndarray] = []
    with MediaPipeExtractor() as extractor:
        try:
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(extractor.process_frame(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
    if not frames:
        raise RuntimeError("No webcam frames captured")
    sequence = normalize_sequence(np.stack(frames, axis=0))
    return decoder.decode(sequence), sequence
