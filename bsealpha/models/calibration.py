"""Probability calibration for the meta-model (§7.1, §4.5).

The meta-model's output probability maps directly onto position size, so it must be
*calibrated*: predicted win-rate should equal realized win-rate. We fit isotonic
regression on **out-of-fold** meta predictions (never in-sample) and refit it more often
than the model itself (weekly isotonic recalibration corrects probability->size drift
without retraining the learned structure, §4.5).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Monotone probability calibration ``score -> P(act)``.

    Fit on OOF ``(score, binary_label)`` pairs. Clipped to ``[eps, 1-eps]`` so downstream
    sizing never sees a degenerate 0/1.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, float)
        if len(np.unique(labels)) < 2:
            # degenerate fold: fall back to the base rate
            self._base = float(labels.mean()) if labels.size else 0.5
            self._fitted = False
            return self
        self.iso.fit(scores, labels)
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, float)
        if not self._fitted:
            return np.full(scores.shape, getattr(self, "_base", 0.5))
        p = self.iso.predict(scores)
        return np.clip(p, self.eps, 1 - self.eps)

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(scores, labels).transform(scores)
