"""Research diagnostics: the perfect-foresight label ceiling (§11.2, week 3).

The single most important sanity check before trusting any model: **what would perfect
foresight of your labels earn, after costs and a realistic book?** If a perfect-foresight
strategy on your labels does not clear roughly 3x your target Sharpe, the *label design*
is wrong and no model will rescue it. This runs the idealized cross-sectional book with the
model replaced by the true label side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import Config
from .metrics import sharpe_ratio


@dataclass
class CeilingReport:
    gross_sharpe: float
    net_sharpe: float
    gross_bps_per_trade: float
    net_bps_per_trade: float
    n_traded: int
    passes_gate: bool
    gate: float

    def summary(self) -> str:
        return (f"perfect-foresight ceiling: gross {self.gross_bps_per_trade:.2f} bps/trade, "
                f"net {self.net_bps_per_trade:.2f} bps/trade, net Sharpe {self.net_sharpe:.2f} "
                f"(gate > {self.gate:.2f}) -> {'PASS' if self.passes_gate else 'FAIL'}")


def perfect_foresight_ceiling(labels: pl.DataFrame, cfg: Config,
                              gate_multiple: float = 3.0) -> CeilingReport:
    """Compute the perfect-foresight ceiling on the residual labels (§11.2 wk3).

    Uses the *true* side (``sign(y_voladj)``) in the idealized market-neutral book, gross and
    net of the 4.18 bps round-trip cost. The gate is ``gate_multiple x`` the deploy Sharpe
    gate (0.5); below it, the label design cannot support a deployable strategy.
    """
    from .runner import cross_sectional_daily_returns

    perfect_side = np.sign(labels["y_voladj"].to_numpy())
    cost = float(cfg.labeling.meta_cost_bps)

    daily_gross, _ = cross_sectional_daily_returns(labels, perfect_side, cfg, cost_bps=0.0)
    daily_net, _ = cross_sectional_daily_returns(labels, perfect_side, cfg, cost_bps=cost)

    traded = labels.filter((pl.col("tb_label") != 0) & (~pl.col("truncated")))
    if traded.height:
        gross_bps = float((np.sign(traded["y_voladj"].to_numpy())
                           * traded["ret_resid"].to_numpy()).mean() * 1e4)
    else:
        gross_bps = 0.0
    net_bps = gross_bps - cost

    net_sharpe = sharpe_ratio(daily_net)
    gate = gate_multiple * 0.5     # 3x the 5th-pct CPCV deploy gate
    return CeilingReport(
        gross_sharpe=sharpe_ratio(daily_gross),
        net_sharpe=net_sharpe,
        gross_bps_per_trade=gross_bps,
        net_bps_per_trade=net_bps,
        n_traded=traded.height,
        passes_gate=net_sharpe > gate,
        gate=gate,
    )
