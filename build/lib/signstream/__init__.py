"""SignStreamASL: an end-to-end ASL recognition pipeline.

The package is intentionally CPU-friendly and fully reproducible from a clean
checkout. Public modules are grouped by pipeline stage: ``data``, ``models``,
``eval``, ``stream``, ``serve``, ``export``, and ``bench``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
