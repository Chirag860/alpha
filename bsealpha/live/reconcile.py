"""Live-vs-backtest reconciliation (§11.2 wk6, §5.6, §8.3).

"The number that matters is the gap, not the P&L." A short-horizon strategy's live results
are typically **40-60% of the backtest** (§11.3), driven by fill-model optimism, latency
tails, and your own footprint. This compares the paper session (realistic passive fills) to
the idealized event-driven backtest on the *same* predictions and reports the haircut,
fill ratio, turnover ratio, and markout deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..backtest.metrics import BacktestMetrics
from .session import PaperSessionResult


@dataclass
class ReconciliationReport:
    backtest_net_sharpe: float
    paper_net_sharpe: float
    sharpe_gap: float
    haircut: float                 # paper / backtest (the §11.3 40-60% number)
    fill_ratio: float
    taker_fraction: float
    turnover_backtest: float
    turnover_paper: float
    markout_delta_bps: dict

    def summary(self) -> str:
        lines = [
            f"backtest net Sharpe : {self.backtest_net_sharpe:+.2f}",
            f"paper net Sharpe    : {self.paper_net_sharpe:+.2f}",
            f"gap (bt - paper)    : {self.sharpe_gap:+.2f}",
            f"haircut (paper/bt)  : {self.haircut:.2f}  (§11.3 expects ~0.4-0.6)",
            f"maker fill ratio    : {self.fill_ratio:.2f}  (model-dependent; measure live, §6.3)",
            f"taker fraction      : {self.taker_fraction:.2f}",
            f"turnover bt / paper : {self.turnover_backtest:.1f}x / {self.turnover_paper:.1f}x",
            "markout delta (paper - bt) bps: "
            + ", ".join(f"{h}m={self.markout_delta_bps.get(h, 0.0):+.2f}"
                        for h in sorted(self.markout_delta_bps)),
        ]
        return "\n".join(lines)


def reconcile(backtest: BacktestMetrics, paper: PaperSessionResult) -> ReconciliationReport:
    """Compare a :class:`BacktestMetrics` and a :class:`PaperSessionResult` on the same data."""
    bt_sr = backtest.sharpe
    pp_sr = paper.net_sharpe
    haircut = (pp_sr / bt_sr) if abs(bt_sr) > 1e-9 else 0.0
    horizons = set(backtest.markout_bps) | set(paper.markout_bps)
    delta = {int(h): float(paper.markout_bps.get(h, 0.0) - backtest.markout_bps.get(h, 0.0))
             for h in horizons}
    return ReconciliationReport(
        backtest_net_sharpe=bt_sr,
        paper_net_sharpe=pp_sr,
        sharpe_gap=bt_sr - pp_sr,
        haircut=haircut,
        fill_ratio=paper.fill_ratio,
        taker_fraction=paper.taker_fraction,
        turnover_backtest=backtest.turnover_x,
        turnover_paper=paper.turnover_x,
        markout_delta_bps=delta,
    )
