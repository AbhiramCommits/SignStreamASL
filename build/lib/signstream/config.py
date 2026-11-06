"""Typed configuration loading with strict unknown-key validation.

All tunables live in YAML under ``configs/``. The dataclasses below are the
contract between the YAML files and the pipeline. A typo in a config file fails
loudly rather than silently falling back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or has unknown keys."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        The parsed mapping. An empty YAML file becomes ``{}``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If the YAML document is not a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(payload).__name__}")
    return payload


T = TypeVar("T", bound="_ValidatedConfig")


class _ValidatedConfig:
    """Mixin that constructs a dataclass from a mapping with strict keys."""

    CONFIG_NAME: ClassVar[str] = "config"

    @classmethod
    def from_dict(cls: type[T], values: dict[str, Any] | None) -> T:
        values = dict(values or {})
        known = {item.name for item in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigError(
                f"Unknown keys in {cls.CONFIG_NAME} config: {', '.join(unknown)}. "
                f"Allowed keys: {', '.join(sorted(known))}"
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dictionary."""
        return asdict(self)  # type: ignore[arg-type]


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {value!r}")


@dataclass
class DataConfig(_ValidatedConfig):
    """Configuration for procedural corpus generation and curation."""

    CONFIG_NAME: ClassVar[str] = "data"

    seed: int = 42
    seq_len: int = 48
    clip_frames: int = 48
    landmark_count: int = 61
    feature_dim: int = 183
    num_glosses: int = 60
    num_signers: int = 14
    takes_per_pair: int = 6
    low_quality_signers: list[str] = field(default_factory=lambda: ["S11", "S12"])
    left_dominant_signers: list[str] = field(default_factory=lambda: ["S04", "S13"])
    duplicate_rate: float = 0.02
    mislabel_rate: float = 0.015
    truncation_rate: float = 0.01
    tracking_failure_rate: float = 0.005
    duplicate_noise: float = 0.0015
    low_quality_noise_multiplier: float = 3.0
    frame_dropout_rate: float = 0.15
    hand_loss_rate: float = 0.04
    dedupe_cosine_threshold: float = 0.995
    curation_top_fraction: float = 0.02
    min_valid_frames: int = 12
    max_nan_fraction: float = 0.40
    random_train: float = 0.70
    random_val: float = 0.15
    random_test: float = 0.15
    test_signers: list[str] = field(default_factory=lambda: ["S12", "S13", "S14"])
    val_signers: list[str] = field(default_factory=lambda: ["S11", "S10"])
    splits: list[str] = field(default_factory=lambda: ["random", "signer_holdout"])
    manifest_version: str = "v1"
    schema_version: str = "asl61-v1"
    data_dir: str = "artifacts/data"
    clips_dir: str = "artifacts/data/clips"
    keypoints_dir: str = "artifacts/data/keypoints"
    continuous_dir: str = "artifacts/data/continuous"
    reports_dir: str = "artifacts/reports"
    continuous_sentences: int = 150
    continuous_min_signs: int = 4
    continuous_max_signs: int = 8
    continuous_fps: int = 30
    continuous_rest_frames: int = 12
    quick: bool = False
    quick_glosses: int = 8
    quick_signers: int = 5
    quick_takes: int = 2

    def __post_init__(self) -> None:
        if self.seq_len <= 0 or self.clip_frames <= 0:
            raise ConfigError("seq_len and clip_frames must be positive")
        if self.landmark_count != 61 or self.feature_dim != 183:
            raise ConfigError("SignStream requires the fixed 61-landmark / 183-feature schema")
        if self.num_glosses <= 0 or self.num_signers <= 0 or self.takes_per_pair <= 0:
            raise ConfigError("num_glosses, num_signers and takes_per_pair must be positive")
        for name in (
            "duplicate_rate",
            "mislabel_rate",
            "truncation_rate",
            "tracking_failure_rate",
            "curation_top_fraction",
            "max_nan_fraction",
        ):
            _validate_fraction(name, getattr(self, name))
        total_split = self.random_train + self.random_val + self.random_test
        if abs(total_split - 1.0) > 1e-6:
            raise ConfigError("random split fractions must sum to 1.0")
        overlap = set(self.test_signers) & set(self.val_signers)
        if overlap:
            raise ConfigError(f"test and validation signers overlap: {sorted(overlap)}")
        if not 0.0 < self.curation_top_fraction < 1.0:
            raise ConfigError("curation_top_fraction must be strictly between 0 and 1")


@dataclass
class ModelConfig(_ValidatedConfig):
    """Configuration shared by the transformer, BiGRU, and training loop."""

    CONFIG_NAME: ClassVar[str] = "model"

    seed: int = 7
    seq_len: int = 48
    feature_dim: int = 183
    num_classes: int = 60
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    warmup_ratio: float = 0.10
    patience: int = 4
    grad_clip: float = 1.0
    num_workers: int = 0
    hidden_size: int = 128
    checkpoint_dir: str = "artifacts/checkpoints"
    reports_dir: str = "artifacts/reports"
    runs_dir: str = "artifacts/runs"
    mlflow_uri: str = "mlruns"
    label_budget_fractions: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 1.00]
    )
    label_budget_epochs: list[int] = field(default_factory=lambda: [5, 5, 5, 5, 6])
    label_budget_seed: int = 11

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ConfigError("d_model must be divisible by nhead")
        if self.feature_dim != 183 or self.num_classes != 60:
            raise ConfigError("model feature_dim and num_classes must match the 60-class schema")
        for name in ("seed", "epochs", "batch_size", "d_model", "ffn_dim"):
            if int(getattr(self, name)) <= 0:
                raise ConfigError(f"{name} must be positive")
        _validate_fraction("dropout", self.dropout)
        _validate_fraction("label_smoothing", self.label_smoothing)
        _validate_fraction("warmup_ratio", self.warmup_ratio)
        if len(self.label_budget_fractions) != len(self.label_budget_epochs):
            raise ConfigError("label_budget_fractions and label_budget_epochs must align")


@dataclass
class EvalConfig(_ValidatedConfig):
    """Evaluation, fairness, decoding, and benchmarking settings."""

    CONFIG_NAME: ClassVar[str] = "eval"

    seed: int = 99
    batch_size: int = 256
    gold_size: int = 300
    n_bins: int = 10
    top_k: int = 5
    window: int = 48
    stride: int = 8
    ema_windows: int = 3
    min_duration_s: float = 0.35
    cooldown_s: float = 0.35
    fps: int = 30
    confidence_threshold: float = 0.60
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.6, 0.7, 0.8]
    )
    reports_dir: str = "artifacts/reports"
    gold_path: str = "eval/gold_set.json"
    parquet_index: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.gold_size <= 0:
            raise ConfigError("batch_size and gold_size must be positive")
        if self.n_bins < 2:
            raise ConfigError("n_bins must be at least 2")
        if self.window <= 0 or self.stride <= 0:
            raise ConfigError("window and stride must be positive")
        if not self.threshold_sweep:
            raise ConfigError("threshold_sweep must not be empty")
        for threshold in self.threshold_sweep:
            _validate_fraction("threshold", threshold)


def data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    """Load and validate :class:`DataConfig` from YAML."""
    return DataConfig.from_dict(load_config(path))


def model_config(path: str | Path = "configs/model.yaml") -> ModelConfig:
    """Load and validate :class:`ModelConfig` from YAML."""
    return ModelConfig.from_dict(load_config(path))


def eval_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    """Load and validate :class:`EvalConfig` from YAML."""
    return EvalConfig.from_dict(load_config(path))
