"""Model interfaces.

A minimal, swappable ``fit``/``predict`` contract (§9 requires components be swappable).
Concrete models: :mod:`bsealpha.models.gbm` (LightGBM primary/meta, sklearn fallback),
:mod:`bsealpha.models.tcn` (optional PyTorch trunk), assembled by
:mod:`bsealpha.models.ensemble`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Estimator(Protocol):
    """Anything with a scikit-style fit/predict used by the pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "Estimator": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def to_numpy(X) -> np.ndarray:
    """Coerce a polars/pandas/ndarray feature matrix to a float ``ndarray``."""
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    return np.ascontiguousarray(np.asarray(X, dtype=np.float64))
