"""Consistent logging setup for SignStream commands and libraries."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure root logging with a timestamped, single-line format.

    Args:
        level: Logging level name, case-insensitive (for example ``"DEBUG"``).
        force: Replace handlers previously installed by this function.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level!r}")

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            if getattr(handler, "_signstream_handler", False):
                root.removeHandler(handler)
    if not any(getattr(h, "_signstream_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        handler._signstream_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, preserving the package hierarchy."""
    return logging.getLogger(name)
