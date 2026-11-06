"""Continuous signing windowing and gloss decoding."""

from __future__ import annotations

from signstream.stream.decoder import (
    ABSTAIN_TOKEN,
    DecodeResult,
    GlossEvent,
    SlidingWindowDecoder,
    decode_video,
    decode_webcam,
    decoder_from_checkpoint,
    edit_distance,
    evaluate_continuous,
    match_events,
    torch_predictor,
)
from signstream.stream.windowing import (
    build_continuous_dataset,
    iter_windows,
    load_continuous_sentence,
    window_array,
)

__all__ = [
    "ABSTAIN_TOKEN",
    "DecodeResult",
    "GlossEvent",
    "SlidingWindowDecoder",
    "build_continuous_dataset",
    "decode_video",
    "decode_webcam",
    "decoder_from_checkpoint",
    "edit_distance",
    "evaluate_continuous",
    "iter_windows",
    "load_continuous_sentence",
    "match_events",
    "torch_predictor",
    "window_array",
]
