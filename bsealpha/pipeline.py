"""End-to-end research pipeline orchestration (§9).

Wires the modules into the flow of §9's diagram, on synthetic data by default:

    generate/load panel
      -> point-in-time universe screen
      -> features (one event-driven engine, residualized, cross-sectionally ranked)
      -> residual-path triple-barrier labels + meta + sample weights
      -> validation (OOF/CPCV, DSR, PBO, effective breadth, tripwires)
      -> final ensemble -> OOF-predicted panel
      -> event-driven backtest (Indian constraint set) -> metrics

Returns a :class:`ResearchResult` bundling the universe size, validation report, and
backtest metrics. Each stage is a plain function so components stay swappable (§9).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .backtest.engine import run_backtest
from .backtest.metrics import BacktestMetrics
from .config import Config
from .data import generate_panel
from .features import build_features
from .labeling import (
    add_cross_sectional_targets,
    compute_weights,
    triple_barrier_labels,
)
from .live.reconcile import ReconciliationReport
from .universe import build_universe
from .validation import (
    CeilingReport,
    Lockbox,
    TrialLog,
    ValidationReport,
    date_split,
    evaluate,
    perfect_foresight_ceiling,
)
from .validation.runner import oof_predicted_panel


@dataclass
class ResearchResult:
    n_universe: int
    n_labeled_rows: int
    validation: ValidationReport
    backtest: BacktestMetrics
    feature_cols: list[str]
    ceiling: CeilingReport | None = None
    lockbox_rows: int = 0
    reconciliation: "ReconciliationReport | None" = None


def run_pipeline(cfg: Config, *, seed: int | None = None,
                 run_cpcv: bool = True, run_backtest_engine: bool = True,
                 lockbox_days: int = 0, trial_db: str | None = None,
                 run_paper: bool = False) -> ResearchResult:
    """Run the full research pipeline on a freshly generated synthetic panel.

    ``lockbox_days`` holds out the most recent N sessions untouched (§5.4). ``trial_db``
    enables automatic trial logging so the DSR uses the honest cumulative trial count.
    """
    panel_data = generate_panel(cfg, seed=seed)

    # point-in-time universe (screens the synthetic names; §1.2)
    dates = panel_data.daily["date"].unique().sort().to_list()
    asof = dates[min(int(cfg.universe.lookback_days), len(dates) - 1)]
    universe = build_universe(panel_data.daily, asof, cfg)
    n_universe = universe.height

    # restrict the tradable panel to the screened names (fall back to all if empty)
    depth, trades = panel_data.depth, panel_data.trades
    if n_universe > 0:
        keep = universe["scrip_code"]
        depth = depth.filter(pl.col("scrip_code").is_in(keep))
        trades = trades.filter(pl.col("scrip_code").is_in(keep))

    # features -> labels -> weights
    feats, feature_cols = build_features(depth, trades, panel_data.meta, cfg)
    labels = triple_barrier_labels(feats, cfg)
    labels = add_cross_sectional_targets(labels, cfg)
    labels = compute_weights(labels, cfg)

    # LOCKBOX: physically hold out the most recent sessions, untouched (§5.4)
    lockbox_rows = 0
    if lockbox_days > 0:
        labels, held = date_split(labels, lockbox_days)
        lockbox = Lockbox(held, name="research-lockbox")
        lockbox_rows = lockbox.n_rows()   # metadata only; never opened here

    # perfect-foresight ceiling gate (§11.2 wk3): is the LABEL design even viable?
    ceiling = perfect_foresight_ceiling(labels, cfg)

    # trial logging -> honest DSR trial count N (§5.4)
    trial_log = TrialLog(trial_db) if trial_db else None

    # validation (honest, out-of-fold)
    report = evaluate(labels, feature_cols, cfg, run_cpcv=run_cpcv, trial_log=trial_log)
    if trial_log is not None:
        trial_log.close()

    # execution backtest on OUT-OF-FOLD predictions (§5.2) with the Indian constraint set
    bt = BacktestMetrics()
    recon = None
    if run_backtest_engine:
        betas = {int(r["scrip_code"]): float(r["beta"])
                 for r in panel_data.meta.select(["scrip_code", "beta"]).iter_rows(named=True)}
        pred = oof_predicted_panel(labels, feature_cols, cfg)
        bt = run_backtest(pred, cfg, betas=betas)

        # PAPER SESSION: run the live stack on the SAME predictions and reconcile (§11.2 wk6)
        if run_paper:
            from .live import reconcile, run_paper_session

            paper = run_paper_session(pred, cfg, betas=betas)
            recon = reconcile(bt, paper)

    return ResearchResult(
        n_universe=n_universe,
        n_labeled_rows=labels.height,
        validation=report,
        backtest=bt,
        feature_cols=feature_cols,
        ceiling=ceiling,
        lockbox_rows=lockbox_rows,
        reconciliation=recon,
    )
