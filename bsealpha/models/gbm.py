"""Gradient-boosted models: LightGBM primary (LambdaRank / regression) + meta, with a
scikit-learn fallback (§4.2).

LightGBM is the production workhorse here. The cross-sectional **LambdaRank** objective
(group = one ``(date, minute)`` cross-section) directly optimizes the relative ordering
you actually trade and is naturally immune to the market factor (§4.2). Regression on the
cross-sectional rank gives a calibrated continuous score that is easier to size on -- we
build both.

If LightGBM cannot be imported (e.g. missing OpenMP), we transparently fall back to
scikit-learn ``HistGradientBoosting`` so the whole pipeline still runs. LambdaRank has no
sklearn analogue, so the fallback ranker is a regressor on the (float) relevance label --
a graceful, clearly-labeled degradation, not a silent one.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import to_numpy

try:  # pragma: no cover - availability depends on the environment
    import lightgbm as lgb

    _HAVE_LGB = True
except Exception:  # pragma: no cover
    _HAVE_LGB = False

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


def lightgbm_available() -> bool:
    """True if LightGBM imported successfully (incl. its native library)."""
    return _HAVE_LGB


def _monotone_vector(feature_cols: list[str], monotone_features: list[str]) -> list[int]:
    signs = {f: 1 for f in monotone_features}
    return [signs.get(c, 0) for c in feature_cols]


class GBM:
    """Unified GBDT wrapper over LightGBM with a sklearn fallback.

    Parameters
    ----------
    task
        One of ``"lambdarank"``, ``"regression"``, ``"binary"``.
    params
        LightGBM parameter dict (from the ``model:`` config block).
    feature_cols
        Column names, used to build monotone-constraint vectors.
    monotone_features
        Feature names constrained to a positive monotone relationship (§4.2).
    """

    def __init__(self, task: str, params: dict, feature_cols: list[str] | None = None,
                 monotone_features: list[str] | None = None) -> None:
        self.task = task
        self.params = dict(params)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.monotone_features = list(monotone_features) if monotone_features else []
        self.model = None
        self.used_fallback = False

    # ------------------------------------------------------------------
    def fit(self, X, y, *, group: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None) -> "GBM":
        X = to_numpy(X)
        y = np.asarray(y)
        n_estimators = int(self.params.get("n_estimators", 200))
        if _HAVE_LGB:
            self._fit_lgb(X, y, group, sample_weight, n_estimators)
        else:  # pragma: no cover - exercised only without LightGBM
            self._fit_fallback(X, y, sample_weight, n_estimators)
        return self

    def _fit_lgb(self, X, y, group, sample_weight, n_estimators) -> None:
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        if self.feature_cols and self.monotone_features:
            params["monotone_constraints"] = _monotone_vector(
                self.feature_cols, self.monotone_features
            )
        if self.task == "lambdarank":
            dset = lgb.Dataset(X, label=y, group=group, weight=sample_weight)
        else:
            dset = lgb.Dataset(X, label=y, weight=sample_weight)
        self.model = lgb.train(params, dset, num_boost_round=n_estimators)

    def _fit_fallback(self, X, y, sample_weight, n_estimators) -> None:  # pragma: no cover
        self.used_fallback = True
        warnings.warn("LightGBM unavailable; using sklearn HistGradientBoosting fallback.")
        common = dict(max_iter=n_estimators,
                      learning_rate=float(self.params.get("learning_rate", 0.05)),
                      max_leaf_nodes=int(self.params.get("num_leaves", 31)),
                      min_samples_leaf=int(self.params.get("min_data_in_leaf", 100)),
                      l2_regularization=float(self.params.get("lambda_l2", 0.0)))
        if self.task == "binary":
            self.model = HistGradientBoostingClassifier(**common)
            self.model.fit(X, y.astype(int), sample_weight=sample_weight)
        else:  # lambdarank falls back to regression on the (float) relevance label
            self.model = HistGradientBoostingRegressor(**common)
            self.model.fit(X, y.astype(float), sample_weight=sample_weight)

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X = to_numpy(X)
        if self.model is None:
            raise RuntimeError("GBM must be fit before predict")
        if _HAVE_LGB and not self.used_fallback:
            return np.asarray(self.model.predict(X))
        if self.task == "binary":  # pragma: no cover
            return self.model.predict_proba(X)[:, 1]
        return np.asarray(self.model.predict(X))  # pragma: no cover

    def predict_proba(self, X) -> np.ndarray:
        """Probability for the binary (meta) task; raw score otherwise."""
        if self.task != "binary":
            raise ValueError("predict_proba is only defined for the binary task")
        return self.predict(X)


def make_group_array(date_minute_keys) -> np.ndarray:
    """Build the LightGBM ``group`` array: contiguous row counts per cross-section (§4.2).

    ``date_minute_keys`` must be row-ordered so each ``(date, minute)`` group is a run of
    contiguous rows -- the CV split must never cut a group (§5.5). Returns the run-length
    of each group in order.
    """
    import polars as pl

    if isinstance(date_minute_keys, pl.DataFrame):
        keys = list(zip(*[date_minute_keys[c].to_list() for c in date_minute_keys.columns]))
    else:
        keys = list(date_minute_keys)
    counts: list[int] = []
    prev = object()
    for k in keys:
        if k != prev:
            counts.append(0)
            prev = k
        counts[-1] += 1
    return np.asarray(counts, dtype=np.int64)
