"""The pooled cross-sectional ensemble (§3.3, §4.1-§4.3).

One model across all names (Sirignano-Cont universality, §4.1). Structure:

    features ─▶ LightGBM PRIMARY (LambdaRank/regression) ─▶ score (side)
                                                              │
    OOF primary score + execution context ─▶ LightGBM META ─▶ p(act) ─▶ isotonic ─▶ size

The meta-model trains on **out-of-fold** primary predictions (§3.3). For the final,
deployable model we generate those OOF scores with an internal day-grouped K-fold so the
ensemble is self-contained; the CPCV harness (:mod:`bsealpha.validation`) uses its own
purged OOF for the reported metrics. An optional TCN embedding (§4.3) is appended when
enabled.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Config
from ..labeling.meta import META_CONTEXT_COLS, make_meta_labels
from .calibration import IsotonicCalibrator
from .gbm import GBM, make_group_array


def _day_kfold(dates: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Contiguous day-block folds (test masks). Labels never cross days, so day grouping
    is a sufficient split for generating OOF primary scores (§5.2)."""
    uniq = np.array(sorted(np.unique(dates)))
    blocks = np.array_split(uniq, min(n_folds, len(uniq)))
    return [np.isin(dates, b) for b in blocks if len(b)]


class PooledEnsemble:
    """Fit/predict the primary + meta + calibration stack on a labeled feature panel."""

    def __init__(self, cfg: Config, feature_cols: list[str]) -> None:
        self.cfg = cfg
        self.feature_cols = list(feature_cols)
        self.primary: GBM | None = None
        self.meta: GBM | None = None
        self.calibrator = IsotonicCalibrator()
        self.meta_cols: list[str] = []

    # ------------------------------------------------------------------
    def _primary_task(self) -> str:
        return self.cfg.model.primary

    def _primary_target(self, panel: pl.DataFrame) -> np.ndarray:
        if self._primary_task() == "lambdarank":
            return panel["y_bucket"].to_numpy()
        return panel["y_rank"].to_numpy()

    def _primary_params(self) -> dict:
        key = "lambdarank" if self._primary_task() == "lambdarank" else "regression"
        return self.cfg.model[key].to_dict()

    def _X(self, panel: pl.DataFrame) -> np.ndarray:
        return panel.select(self.feature_cols).to_numpy()

    def _fit_primary(self, panel: pl.DataFrame) -> GBM:
        gbm = GBM(self._primary_task(), self._primary_params(),
                  feature_cols=self.feature_cols,
                  monotone_features=list(self.cfg.model.monotone_features))
        group = None
        if self._primary_task() == "lambdarank":
            group = make_group_array(panel.select(["date", "minute"]))
        gbm.fit(self._X(panel), self._primary_target(panel), group=group,
                sample_weight=panel["weight"].to_numpy() if "weight" in panel.columns else None)
        return gbm

    # ------------------------------------------------------------------
    def fit(self, panel: pl.DataFrame, *, oof_primary: np.ndarray | None = None) -> "PooledEnsemble":
        """Fit the full stack on a labeled, weighted feature panel.

        ``panel`` must carry ``y_bucket``/``y_rank``, ``ret_resid``, ``weight``, and the
        meta-context columns. If ``oof_primary`` is not supplied, it is generated with an
        internal day K-fold.
        """
        panel = panel.sort(["date", "minute", "scrip_code"])

        # 1) OOF primary scores for meta training (§3.3)
        if oof_primary is None:
            oof_primary = self._generate_oof_primary(panel)

        # 2) final primary on all data
        self.primary = self._fit_primary(panel)

        # 3) meta-model on OOF primary scores + execution context
        side = np.sign(oof_primary)
        side = np.where(side == 0, 1, side)
        meta_label = make_meta_labels(panel["ret_resid"].to_numpy(), side,
                                      float(self.cfg.labeling.meta_cost_bps))
        self.meta_cols = ["primary_score", "primary_conf"] + [
            c for c in META_CONTEXT_COLS if c in panel.columns
        ]
        meta_X = self._meta_matrix(panel, oof_primary)
        self.meta = GBM("binary", self.cfg.model.meta.to_dict())
        w = panel["weight"].to_numpy() if "weight" in panel.columns else None
        self.meta.fit(meta_X, meta_label, sample_weight=w)

        # 4) calibrate on OOF meta predictions (internal split for honesty)
        oof_meta = self._generate_oof_meta(panel, oof_primary, meta_label)
        self.calibrator.fit(oof_meta, meta_label)
        return self

    def _meta_matrix(self, panel: pl.DataFrame, primary_score: np.ndarray) -> np.ndarray:
        ctx = [c for c in self.meta_cols if c not in ("primary_score", "primary_conf")]
        base = np.column_stack([primary_score, np.abs(primary_score)])
        if ctx:
            n = len(primary_score)
            present = set(panel.columns)
            # Columns absent at predict time (e.g. the label-derived 'truncated' in the live
            # loop, which predicts on features only) default to 0, preserving the exact column
            # layout the meta-model was fit with.
            cols = np.column_stack([
                panel[c].to_numpy() if c in present else np.zeros(n) for c in ctx
            ])
            base = np.column_stack([base, cols])
        return base

    def _generate_oof_primary(self, panel: pl.DataFrame) -> np.ndarray:
        dates = panel["date"].to_numpy()
        oof = np.zeros(panel.height)
        for test_mask in _day_kfold(dates, n_folds=3):
            train = panel.filter(pl.Series(~test_mask))
            test = panel.filter(pl.Series(test_mask))
            gbm = GBM(self._primary_task(), self._primary_params(),
                      feature_cols=self.feature_cols,
                      monotone_features=list(self.cfg.model.monotone_features))
            group = (make_group_array(train.select(["date", "minute"]))
                     if self._primary_task() == "lambdarank" else None)
            gbm.fit(self._X(train), self._primary_target(train), group=group,
                    sample_weight=train["weight"].to_numpy() if "weight" in train.columns else None)
            oof[test_mask] = gbm.predict(self._X(test))
        return oof

    def _generate_oof_meta(self, panel: pl.DataFrame, oof_primary: np.ndarray,
                           meta_label: np.ndarray) -> np.ndarray:
        dates = panel["date"].to_numpy()
        oof = np.zeros(panel.height)
        meta_X = self._meta_matrix(panel, oof_primary)
        for test_mask in _day_kfold(dates, n_folds=3):
            tr = ~test_mask
            g = GBM("binary", self.cfg.model.meta.to_dict())
            g.fit(meta_X[tr], meta_label[tr])
            oof[test_mask] = g.predict(meta_X[test_mask])
        return oof

    # ------------------------------------------------------------------
    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Return ``panel`` with ``primary_score``, ``primary_side``, ``p_act`` columns."""
        if self.primary is None or self.meta is None:
            raise RuntimeError("PooledEnsemble must be fit before predict")
        score = self.primary.predict(self._X(panel))
        side = np.sign(score)
        side = np.where(side == 0, 1, side)
        meta_X = self._meta_matrix(panel, score)
        p_raw = self.meta.predict_proba(meta_X)
        p_act = self.calibrator.transform(p_raw)
        return panel.with_columns(
            pl.Series("primary_score", score),
            pl.Series("primary_side", side.astype(np.int8)),
            pl.Series("p_act", p_act),
        )
