"""Fixed 61-landmark signing skeleton and MediaPipe video ingestion.

The schema is deliberately small and body-relative: 21 left-hand landmarks, 21
right-hand landmarks, and 19 upper-body pose landmarks. Coordinates are
normalized to a shoulder-midpoint origin and shoulder-width scale so camera
distance and body size do not leak into the classifier.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    """Raised when an optional capture dependency is requested but unavailable."""


_HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

# MediaPipe Pose landmark ids, chosen to cover the head, shoulders, arms, and hips.
POSE_MEDIAPIPE_INDICES: tuple[int, ...] = (
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    17,  # left pinky
    18,  # right pinky
    19,  # left index
    20,  # right index
    21,  # left thumb
    22,  # right thumb
    23,  # left hip
    24,  # right hip
)
POSE_NAMES = [
    "pose_nose",
    "pose_left_eye",
    "pose_right_eye",
    "pose_left_ear",
    "pose_right_ear",
    "pose_left_shoulder",
    "pose_right_shoulder",
    "pose_left_elbow",
    "pose_right_elbow",
    "pose_left_wrist",
    "pose_right_wrist",
    "pose_left_pinky",
    "pose_right_pinky",
    "pose_left_index",
    "pose_right_index",
    "pose_left_thumb",
    "pose_right_thumb",
    "pose_left_hip",
    "pose_right_hip",
]

LANDMARK_NAMES: tuple[str, ...] = tuple(
    [f"left_hand_{name}" for name in _HAND_NAMES]
    + [f"right_hand_{name}" for name in _HAND_NAMES]
    + POSE_NAMES
)
LANDMARK_COUNT = 61
FEATURE_DIM = LANDMARK_COUNT * 3

# Local indices of the pose block and the shoulders used for normalization.
POSE_OFFSET = 42
LEFT_SHOULDER_INDEX = POSE_OFFSET + 5
RIGHT_SHOULDER_INDEX = POSE_OFFSET + 6
DEFAULT_SEQUENCE_LENGTH = 48


def _as_float_sequence(seq: np.ndarray) -> np.ndarray:
    array = np.asarray(seq, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected shape [T, {LANDMARK_COUNT}, 3], got {array.shape}")
    return array


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Translate to shoulder midpoint and scale by shoulder width.

    Args:
        seq: Array with shape ``[T, 61, 3]``.

    Returns:
        A float32 array of the same shape. Translation and uniform scaling of
        the input leave the output unchanged.
    """
    array = _as_float_sequence(seq).copy()
    left = array[:, LEFT_SHOULDER_INDEX, :]
    right = array[:, RIGHT_SHOULDER_INDEX, :]
    midpoint = (left + right) / 2.0
    scale = np.linalg.norm(left - right, axis=-1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    # If the midpoint is not finite, leave the frame translation-free.
    midpoint = np.where(np.isfinite(midpoint), midpoint, 0.0)
    array = (array - midpoint[:, None, :]) / scale[:, None, None]
    return array.astype(np.float32, copy=False)


def _fill_nan_time(array: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaNs along the time axis, preserving all-NaN rows."""
    filled = array.copy()
    n_frames = filled.shape[0]
    if n_frames == 0:
        return filled
    for t in range(1, n_frames):
        missing = ~np.isfinite(filled[t])
        filled[t, missing] = filled[t - 1, missing]
    for t in range(n_frames - 2, -1, -1):
        missing = ~np.isfinite(filled[t])
        filled[t, missing] = filled[t + 1, missing]
    return filled


def resample_sequence(seq: np.ndarray, target_len: int = DEFAULT_SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample a sequence along time while preserving its endpoints.

    Missing values are forward/backward filled before interpolation. Entirely
    missing sequences remain NaN, which keeps tracking failures visible to the
    curation filter rather than silently turning them into valid-looking data.

    Args:
        seq: Array with shape ``[T, 61, 3]``.
        target_len: Number of output frames.

    Returns:
        Array with shape ``[target_len, 61, 3]``.
    """
    array = _as_float_sequence(seq)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if array.shape[0] == 0:
        raise ValueError("Cannot resample an empty sequence")

    filled = _fill_nan_time(array)
    if np.isnan(filled).all():
        return np.full((target_len, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    if filled.shape[0] == 1:
        return np.repeat(filled, target_len, axis=0).astype(np.float32)

    source = np.linspace(0.0, filled.shape[0] - 1, target_len)
    flat = filled.reshape(filled.shape[0], -1)
    indices = np.arange(filled.shape[0], dtype=np.float32)
    output = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for feature in range(flat.shape[1]):
        output[:, feature] = np.interp(source, indices, flat[:, feature])
    output[0] = flat[0]
    output[-1] = flat[-1]
    return output.reshape(target_len, LANDMARK_COUNT, 3)


def sequence_descriptor(seq: np.ndarray, length: int = 16) -> np.ndarray:
    """Return a compact, L2-normalized descriptor for deduplication."""
    sampled = resample_sequence(seq, length)
    if np.isnan(sampled).all():
        return np.zeros(length * FEATURE_DIM, dtype=np.float32)
    sampled = np.nan_to_num(sampled, nan=0.0)
    vector = sampled.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


class MediaPipeExtractor:
    """Extract the SignStream 61-landmark schema from a video using MediaPipe.

    MediaPipe is imported lazily so the rest of the project runs with only the
    core dependencies installed. Missing hands are inserted as NaN and then
    forward/backward filled when possible.
    """

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError(
                "Video extraction requires the optional capture dependencies. "
                'Install them with: pip install "signstream[capture]"'
            ) from exc

        try:
            holistic_module = mp.solutions.holistic
        except (AttributeError, ImportError) as exc:  # pragma: no cover
            raise MissingDependencyError(
                "This MediaPipe build does not expose mediapipe.solutions.holistic. "
                "Install mediapipe>=0.10,<0.11 for the legacy Holistic solution, "
                'or use the procedural corpus path: pip install "signstream[capture]"'
            ) from exc

        self._cv2 = cv2
        self._mp = mp
        self._holistic = holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._holistic.close()

    def __enter__(self) -> MediaPipeExtractor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _landmarks_to_array(landmarks: Any, count: int) -> np.ndarray:
        if landmarks is None:
            return np.full((count, 3), np.nan, dtype=np.float32)
        return np.asarray(
            [(point.x, point.y, point.z) for point in landmarks.landmark[:count]],
            dtype=np.float32,
        )


    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract one 61-landmark frame from a BGR webcam/video frame."""
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        result = self._holistic.process(frame_rgb)
        left = self._landmarks_to_array(result.left_hand_landmarks, 21)
        right = self._landmarks_to_array(result.right_hand_landmarks, 21)
        pose = np.full((19, 3), np.nan, dtype=np.float32)
        if result.pose_landmarks is not None:
            source = np.asarray(
                [(p.x, p.y, p.z) for p in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
            pose = source[list(POSE_MEDIAPIPE_INDICES)]
        return np.concatenate([left, right, pose], axis=0).astype(np.float32)

    def extract(self, video_path: str | Path, *, normalize: bool = True) -> np.ndarray:
        """Extract one normalized keypoint sequence from a video.

        Args:
            video_path: Path to an ``.mp4`` or ``.mov`` clip.
            normalize: Apply :func:`normalize_sequence` before returning.

        Returns:
            Float32 array with shape ``[T, 61, 3]``.

        Raises:
            FileNotFoundError: If the video does not exist.
            RuntimeError: If no frames can be read.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        capture = self._cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(self.process_frame(frame_bgr))
        finally:
            capture.release()

        if not frames:
            raise RuntimeError(f"No readable frames in {path}")
        sequence = _fill_nan_time(np.stack(frames, axis=0).astype(np.float32))
        if normalize:
            sequence = normalize_sequence(sequence)
        return sequence.astype(np.float32, copy=False)


def extract_keypoints_directory(
    videos_dir: str | Path,
    out_dir: str | Path,
    *,
    extractor: MediaPipeExtractor | None = None,
) -> dict[str, Any]:
    """Extract every ``.mp4``/``.mov`` clip in a directory into ``.npz`` files."""
    source = Path(videos_dir)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in {".mp4", ".mov"}
    )
    if not videos:
        raise FileNotFoundError(f"No .mp4 or .mov files found under {source}")

    owns_extractor = extractor is None
    active = extractor or MediaPipeExtractor()
    records: list[dict[str, Any]] = []
    try:
        for video in videos:
            clip_id = video.stem
            output_path = destination / f"{clip_id}.npz"
            try:
                sequence = active.extract(video)
                np.savez_compressed(
                    output_path,
                    keypoints=sequence.astype(np.float16),
                    valid_length=np.int32(sequence.shape[0]),
                )
                records.append(
                    {
                        "clip_id": clip_id,
                        "source": str(video),
                        "output": str(output_path),
                        "frames": int(sequence.shape[0]),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - ingest should continue per clip
                LOGGER.exception("Failed to extract %s", video)
                records.append(
                    {
                        "clip_id": clip_id,
                        "source": str(video),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    finally:
        if owns_extractor:
            active.close()

    log = {
        "videos_dir": str(source),
        "out_dir": str(destination),
        "count": len(records),
        "ok": sum(record["status"] == "ok" for record in records),
        "failed": sum(record["status"] != "ok" for record in records),
        "records": records,
        "schema": {
            "shape": ["T", LANDMARK_COUNT, 3],
            "feature_dim": FEATURE_DIM,
            "normalized": True,
        },
    }
    import json

    (destination / "ingest_log.json").write_text(
        json.dumps(log, indent=2, sort_keys=True), encoding="utf-8"
    )
    return log


def validate_landmark_shape(seq: Sequence[Sequence[Sequence[float]]] | np.ndarray) -> None:
    """Raise ``ValueError`` unless a sequence matches the 61-landmark schema."""
    _as_float_sequence(np.asarray(seq, dtype=np.float32))
