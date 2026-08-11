"""Validation runner: OOF prediction, CPCV Sharpe distribution, and the report card.

Ties CV + the pooled ensemble into the numbers §11.1 gates on:

* **OOF pass** (purged day-blocks): primary OOF score, then meta OOF *on top of the OOF
  primary* (so the meta-model never sees in-sample primary predictions, §3.3), giving
  meta-AUC, IC, a daily P&L series, effective breadth, and DSR.
* **CPCV** distribution (§5.3): reconstruct ``phi`` out-of-sample paths and report the
  **5th percentile** Sharpe, not the mean.
* **PBO** (CSCV) over a few candidate sizings.
* **Tripwires** (§0.5, §11.3): warn when Sharpe > 4, meta-AUC > 0.62, IC > 0.08, or
  effective breadth < the gate -- each is a bug report, not a triumph.

The book here is an idealized market-/sector-neutral cross-sectional book sampled at the
holding horizon (non-overlapping), net of a flat cost -- the right instrument for the
"is there a stable *relationship*" question. Execution realism (queues, circuits, forced
flatten, per-order brokerage) lives in :mod:`bsealpha.backtest`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from .. import market
from ..config import Config
from ..labeling.meta import META_CONTEXT_COLS, make_meta_labels
from ..models.calibration import IsotonicCalibrator
from ..models.gbm import GBM, make_group_array
from .breadth import effective_breadth
from .cv import CombinatorialPurgedCV, PurgedDayGroupCV
from .metrics import deflated_sharpe, pbo_cscv, sharpe_ratio


@dataclass
class ValidationReport:
    """Container for the validation numbers gated on in §11.1."""

    sharpe_oos: float = 0.0
    cpcv_sharpe_median: float = 0.0
    cpcv_sharpe_5pct: float = 0.0
    dsr: float = 0.0
    dsr_sr: float = 0.0
    pbo: float = float("nan")
    meta_auc: float = 0.0
    ic: float = 0.0
    effective_breadth: float = 0.0
    n_trials: int = 0
    tripwires: list[str] = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["daily_returns"] = list(map(float, self.daily_returns))
        return d


def _primary_params(cfg: Config) -> tuple[str, dict]:
    task = cfg.model.primary
    key = "lambdarank" if task == "lambdarank" else "regression"
    return task, cfg.model[key].to_dict()


def _fit_primary(train: pl.DataFrame, feature_cols: list[str], cfg: Config) -> GBM:
    task, params = _primary_params(cfg)
    target = "y_bucket" if task == "lambdarank" else "y_rank"
    g = GBM(task, params, feature_cols=feature_cols,
            monotone_features=list(cfg.model.monotone_features))
    group = make_group_array(train.select(["date", "minute"])) if task == "lambdarank" else None
    g.fit(train.select(feature_cols).to_numpy(), train[target].to_numpy(), group=group,
          sample_weight=train["weight"].to_numpy() if "weight" in train.columns else None)
    return g


def oof_primary(panel: pl.DataFrame, feature_cols: list[str], cfg: Config,
                n_splits: int) -> np.ndarray:
    """Out-of-fold primary scores via purged day-block CV."""
    panel = panel.sort(["date", "minute", "scrip_code"])
    dates = panel["date"].to_numpy()
    oof = np.full(panel.height, np.nan)
    cv = PurgedDayGroupCV(n_splits=n_splits, embargo_days=int(cfg.validation.embargo_days))
    for tr, te in cv.split(dates):
        g = _fit_primary(panel[tr], feature_cols, cfg)
        oof[te] = g.predict(panel[te].select(feature_cols).to_numpy())
    oof[np.isnan(oof)] = 0.0
    return oof


def oof_meta(panel: pl.DataFrame, oof_primary_score: np.ndarray, cfg: Config,
             n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    """OOF meta probabilities + labels, meta trained on the OOF primary score (§3.3)."""
    side = np.where(np.sign(oof_primary_score) == 0, 1, np.sign(oof_primary_score))
    meta_label = make_meta_labels(panel["ret_resid"].to_numpy(), side,
                                  float(cfg.labeling.meta_cost_bps))
    ctx = [c for c in META_CONTEXT_COLS if c in panel.columns]
    meta_X = np.column_stack(
        [oof_primary_score, np.abs(oof_primary_score)]
        + ([panel.select(ctx).to_numpy()] if ctx else [])
    )
    dates = panel["date"].to_numpy()
    oof = np.full(panel.height, np.nan)
    cv = PurgedDayGroupCV(n_splits=n_splits, embargo_days=int(cfg.validation.embargo_days))
    for tr, te in cv.split(dates):
        if len(np.unique(meta_label[tr])) < 2:
            oof[te] = meta_label[tr].mean() if len(tr) else 0.5
            continue
        g = GBM("binary", cfg.model.meta.to_dict())
        g.fit(meta_X[tr], meta_label[tr])
        oof[te] = g.predict_proba(meta_X[te])
    oof[np.isnan(oof)] = np.nanmean(oof[~np.isnan(oof)]) if np.isfinite(oof).any() else 0.5
    return oof, meta_label


def cross_sectional_daily_returns(panel: pl.DataFrame, score: np.ndarray, cfg: Config, *,
                                  p_act: np.ndarray | None = None,
                                  cost_bps: float = 0.0
                                  ) -> tuple[np.ndarray, pl.DataFrame]:
    """Idealized neutral cross-sectional book sampled at the holding horizon (§5.3).

    Positions = meta-gated, market-neutral (demeaned) score, unit gross per decision
    cross-section; P&L uses the realized residual return to the label exit. This answers
    the *relationship* question ("is there stable signal") and defaults to **gross**
    (``cost_bps=0``); execution cost/turnover realism lives in
    :mod:`bsealpha.backtest`. Returns ``(daily_returns, per_name)`` where ``per_name`` is a
    long frame of ``(date, minute, scrip_code, pnl)`` for breadth.
    """
    h = int(cfg.labeling.horizon_min)
    cost = float(cost_bps) * 1e-4
    df = panel.select(["date", "minute", "scrip_code", "ret_resid"]).with_columns(
        pl.Series("score", score),
        pl.Series("p_act", p_act if p_act is not None else np.ones(panel.height)),
    )
    # decision minutes: non-overlapping holding, before forced flatten
    df = df.filter((pl.col("minute") % h == 0)
                   & (pl.col("minute") < market.flatten_session_min()))
    if df.height == 0:
        return np.array([]), df

    # neutralize within each (date, minute) cross-section and gate by p_act
    df = df.with_columns(
        ((pl.col("score") - pl.col("score").mean().over(["date", "minute"]))
         * pl.col("p_act")).alias("raw_pos")
    )
    gross = pl.col("raw_pos").abs().sum().over(["date", "minute"]).clip(lower_bound=1e-12)
    df = df.with_columns((pl.col("raw_pos") / gross).alias("pos"))
    df = df.with_columns((pl.col("pos") * pl.col("ret_resid")).alias("pnl"))

    per_decision = df.group_by(["date", "minute"], maintain_order=True).agg(
        ret=pl.col("pnl").sum(), turn=pl.col("pos").abs().sum()
    )
    per_decision = per_decision.with_columns(
        (pl.col("ret") - 2.0 * cost * pl.col("turn")).alias("net_ret")
    )
    daily = (per_decision.group_by("date", maintain_order=True)
             .agg(day_ret=pl.col("net_ret").sum())
             .sort("date"))
    return daily["day_ret"].to_numpy(), df.select(["date", "minute", "scrip_code", "pnl"])


def cpcv_sharpe_distribution(panel: pl.DataFrame, feature_cols: list[str],
                             cfg: Config) -> np.ndarray:
    """Reconstruct CPCV out-of-sample paths and return their Sharpes (§5.3).

    Each of the ``N`` day-groups is tested in ``C(N-1, k-1)`` splits; the i-th occurrence
    of a group feeds path i. A path's daily returns are assembled from each group's
    predictions in that path, then a Sharpe is computed. Uses primary-only fits (the
    relationship question) to keep the 66-fit sweep tractable.
    """
    panel = panel.sort(["date", "minute", "scrip_code"])
    dates = panel["date"].to_numpy()
    cpcv = CombinatorialPurgedCV(int(cfg.validation.cpcv_n_groups),
                                 int(cfg.validation.cpcv_k_test),
                                 int(cfg.validation.embargo_days))
    n_paths = max(cpcv.n_paths, 1)
    # per-group daily returns for each path occurrence
    group_occurrence: dict[int, int] = {}
    path_daily: list[dict] = [dict() for _ in range(n_paths)]

    for tr, te, combo in cpcv.split(dates):
        g = _fit_primary(panel[tr], feature_cols, cfg)
        test_panel = panel[te]
        score = g.predict(test_panel.select(feature_cols).to_numpy())
        daily, _ = cross_sectional_daily_returns(test_panel, score, cfg)
        day_vals = (test_panel.select("date").unique().sort("date")["date"].to_list())
        # assign this split's test groups to paths by occurrence order
        for grp in combo:
            occ = group_occurrence.get(grp, 0)
            group_occurrence[grp] = occ + 1
            if occ < n_paths and len(daily):
                # distribute daily returns to the path keyed by date
                for dval, r in zip(day_vals, daily):
                    path_daily[occ][dval] = r

    sharpes = []
    for pd_map in path_daily:
        if len(pd_map) >= 3:
            r = np.array([pd_map[k] for k in sorted(pd_map)])
            sharpes.append(sharpe_ratio(r))
    return np.array(sharpes) if sharpes else np.array([0.0])


def oof_predicted_panel(panel: pl.DataFrame, feature_cols: list[str],
                        cfg: Config) -> pl.DataFrame:
    """Attach OUT-OF-FOLD ``primary_score`` / ``primary_side`` / ``p_act`` to the panel.

    The honest input to the execution backtest: predictions for each day come from a model
    that never trained on that day (§5.2). Avoids the in-sample optimism of predicting with
    a model fit on all data.
    """
    panel = panel.sort(["date", "minute", "scrip_code"])
    n_splits = int(cfg.validation.cpcv_n_groups)
    prim = oof_primary(panel, feature_cols, cfg, n_splits)
    meta_p, meta_label = oof_meta(panel, prim, cfg, n_splits)
    p_act = IsotonicCalibrator().fit_transform(meta_p, meta_label)
    side = np.where(np.sign(prim) == 0, 1, np.sign(prim)).astype(np.int8)
    return panel.with_columns(
        pl.Series("primary_score", prim),
        pl.Series("primary_side", side),
        pl.Series("p_act", p_act),
    )


def evaluate(panel: pl.DataFrame, feature_cols: list[str], cfg: Config, *,
             trial_sharpes: list[float] | None = None,
             run_cpcv: bool = True, trial_log=None) -> ValidationReport:
    """Run the full validation and return a :class:`ValidationReport` with tripwires.

    If a :class:`~bsealpha.validation.trials.TrialLog` is supplied, this run is logged and
    the honest trial count ``N`` (including all prior logged runs) drives the Deflated
    Sharpe (§5.4) -- self-reported ``N`` is always low by 5-10x.
    """
    rep = ValidationReport()
    n_splits = int(cfg.validation.cpcv_n_groups)
    panel = panel.sort(["date", "minute", "scrip_code"])

    # -- OOF primary + meta -----------------------------------------------
    prim = oof_primary(panel, feature_cols, cfg, n_splits)
    meta_p, meta_label = oof_meta(panel, prim, cfg, n_splits)
    p_act = IsotonicCalibrator().fit_transform(meta_p, meta_label)

    # meta-AUC (leak tripwire)
    if len(np.unique(meta_label)) == 2:
        rep.meta_auc = float(roc_auc_score(meta_label, meta_p))
    # IC of primary score vs the vol-adjusted residual target
    yv = panel["y_voladj"].to_numpy()
    if np.std(prim) > 0:
        rep.ic = float(np.corrcoef(prim, yv)[0, 1])

    # -- idealized book: daily returns, breadth ---------------------------
    daily, per_name = cross_sectional_daily_returns(panel, prim, cfg, p_act=p_act)
    rep.daily_returns = daily
    rep.sharpe_oos = sharpe_ratio(daily)
    if per_name.height:
        wide = (per_name.pivot(values="pnl", index=["date", "minute"], on="scrip_code")
                .drop(["date", "minute"]).fill_null(0.0))
        rep.effective_breadth = effective_breadth(wide.to_numpy())

    # -- DSR (T = days!) with the honest trial count (§5.4) ---------------
    if trial_log is not None:
        # log THIS run, then deflate against every configuration ever logged
        trial_log.log(cfg.to_dict(), sharpe_ratio(daily), note="evaluate")
        logged = trial_log.sharpes()
        trials = logged if len(logged) >= 2 else _null_trials(cfg)
    else:
        trials = trial_sharpes or _null_trials(cfg)
    rep.n_trials = len(trials)
    rep.dsr, rep.dsr_sr, _ = deflated_sharpe(daily, np.asarray(trials))

    # -- CPCV distribution -------------------------------------------------
    if run_cpcv:
        dist = cpcv_sharpe_distribution(panel, feature_cols, cfg)
        rep.cpcv_sharpe_median = float(np.median(dist))
        rep.cpcv_sharpe_5pct = float(np.percentile(dist, 5))

    # -- PBO over candidate sizings ---------------------------------------
    rep.pbo = _pbo_over_sizings(panel, prim, p_act, cfg)

    # -- tripwires (§0.5, §11.3) ------------------------------------------
    rep.tripwires = _check_tripwires(rep, cfg)
    for w in rep.tripwires:
        warnings.warn(w)
    return rep


def _null_trials(cfg: Config, n: int = 200) -> list[float]:
    """A default null trial set for the DSR when no trial log is supplied (§5.4).

    Draws N per-day Sharpes with the configured trial std around zero, so DSR still
    deflates for multiple testing even before a real trial log exists.
    """
    rng = np.random.default_rng(0)
    return list(rng.normal(0.0, float(cfg.validation.dsr_trial_sr_std), n))


def _pbo_over_sizings(panel: pl.DataFrame, score: np.ndarray, p_act: np.ndarray,
                      cfg: Config) -> float:
    """Build a small config matrix (different sizings) and compute PBO via CSCV (§5.4)."""
    configs = {
        "primary_only": (score, None),
        "meta_gated": (score, p_act),
        "meta_sq": (score, p_act ** 2),
        "sign_only": (np.sign(score), p_act),
    }
    cols = []
    idx = None
    for _, (sc, pa) in configs.items():
        daily, _ = cross_sectional_daily_returns(panel, np.asarray(sc, float), cfg, p_act=pa)
        cols.append(daily)
    m = min(len(c) for c in cols)
    if m < 4:
        return float("nan")
    mat = np.column_stack([c[:m] for c in cols])
    pbo, _ = pbo_cscv(mat, S=min(int(cfg.validation.pbo_blocks), m))
    return pbo


def _check_tripwires(rep: ValidationReport, cfg: Config) -> list[str]:
    v = cfg.validation
    out = []
    # NB: the OOF/CPCV book Sharpe here is GROSS/frictionless (relationship, §5.3) and is
    # high by design -- the deploy Sharpe tripwire (>4) applies to the NET event-driven
    # backtest and is checked there. Validation tripwires target leak signatures:
    if rep.meta_auc > float(v.tripwire_meta_auc):
        out.append(f"TRIPWIRE: meta-AUC {rep.meta_auc:.3f} > {v.tripwire_meta_auc} -- suspect leak (§11.3).")
    if abs(rep.ic) > float(v.tripwire_ic):
        out.append(f"TRIPWIRE: |IC| {rep.ic:.3f} > {v.tripwire_ic} -- suspect leak (§11.3).")
    if 0 < rep.effective_breadth < float(v.breadth_gate):
        out.append(f"NOTE: effective breadth {rep.effective_breadth:.2f} < gate {v.breadth_gate} "
                   "-- residualization may be collapsing onto the market factor (§0.5).")
    return out
