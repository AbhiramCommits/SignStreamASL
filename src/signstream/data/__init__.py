"""Signing data ingestion, curation, manifest, and dataset utilities."""

from __future__ import annotations

from signstream.data.corpus import (
    GLOSSES,
    content_hash_ids,
    generate_corpus,
    load_corpus,
    save_corpus,
)
from signstream.data.curate import CurationResult, audit_labels, curate_all, dedupe, quality_filter
from signstream.data.dataset import SignClipDataset, collate_fn, mirror_sequence
from signstream.data.keypoints import (
    FEATURE_DIM,
    LANDMARK_COUNT,
    LANDMARK_NAMES,
    MediaPipeExtractor,
    MissingDependencyError,
    extract_keypoints_directory,
    normalize_sequence,
    resample_sequence,
)
from signstream.data.manifest import load_manifest, write_manifest
from signstream.data.splits import (
    coverage_stats,
    random_stratified_split,
    signer_holdout_split,
    write_split_files,
)

__all__ = [
    "GLOSSES",
    "FEATURE_DIM",
    "LANDMARK_COUNT",
    "LANDMARK_NAMES",
    "CurationResult",
    "MediaPipeExtractor",
    "MissingDependencyError",
    "SignClipDataset",
    "audit_labels",
    "collate_fn",
    "content_hash_ids",
    "coverage_stats",
    "curate_all",
    "dedupe",
    "extract_keypoints_directory",
    "generate_corpus",
    "load_corpus",
    "load_manifest",
    "mirror_sequence",
    "normalize_sequence",
    "quality_filter",
    "random_stratified_split",
    "resample_sequence",
    "save_corpus",
    "signer_holdout_split",
    "write_manifest",
    "write_split_files",
]
