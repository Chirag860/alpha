"""Paper-trading session: the live path, minute by minute (§11.2 wk6).

Replays a predicted panel through the *live* stack -- :class:`ExecutionManager` routing a
market/sector-neutral book to a :class:`PaperBroker` with **passive maker fills** -- and books
the same Indian cost stack on every fill. Unlike the idealized backtest book, orders here
**may not fill**, and they fill preferentially when the market comes to you (adverse
selection, §5.1). The gap between this and the backtest is the number that matters (§11.2).

This is the same code shape a real deployment runs; swap ``PaperBroker`` for a live
``BrokerAdapter`` and the loop is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..market import flatten_session_min, session_len_min, session_open_min
from ..backtest.costs import CostParams, impact_rupees, leg_cost_rupees
from ..backtest.metrics import sharpe as _sharpe
from ..config import Config
from ..execution import ExecutionManager, PaperBroker
from ..portfolio.construct import build_book
from ..validation.breadth import effective_breadth


@dataclass
class PaperSessionResult:
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    net_sharpe: float = 0.0
    fill_ratio: float = 0.0
    taker_fraction: float = 0.0
    n_orders: int = 0
    n_fills: int = 0
    turnover_x: float = 0.0
    total_cost_rupees: float = 0.0
    total_impact_rupees: float = 0.0
    effective_breadth: float = 0.0
    markout_bps: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["daily_returns"] = list(map(float, self.daily_returns))
        return d


def run_paper_session(panel: pl.DataFrame, cfg: Config, *,
                      betas: dict[int, float] | None = None) -> PaperSessionResult:
    """Run the paper session over a predicted panel and return execution metrics.

    ``panel`` needs ``primary_score``, ``p_act``, ``mid``, ``high``, ``low``, ``sector``,
    ``scrip_code``, ``date``, ``minute``. Fills come from the paper broker; costs from the
    Indian stack (§0.1, §6.4).
    """
    bt = cfg.backtest
    p = CostParams.from_config(cfg)
    decision_interval = int(getattr(bt, "decision_interval_min", cfg.labeling.horizon_min))
    smooth = int(getattr(bt, "score_smooth_min", 0))
    gross_target = float(cfg.portfolio.gross_target)
    equity = float(bt.equity)
    betas = betas or {}

    panel = panel.sort(["date", "minute", "scrip_code"])
    if smooth > 1:
        panel = panel.with_columns(
            pl.col("primary_score").rolling_mean(smooth, min_samples=1)
            .over(["scrip_code", "date"]).alias("primary_score")
        )

    # per-name daily vol for the impact model + adv from turnover
    adv = {int(k): float(v) for k, v in zip(
        *panel.group_by(["scrip_code", "date"]).agg(t=pl.col("turnover").sum())
        .group_by("scrip_code").agg(m=pl.col("t").median())[["scrip_code", "m"]])}
    sig = {int(k): float(v or 0.0) * np.sqrt(float(session_len_min())) for k, v in zip(
        *panel.with_columns(pl.col("mid").pct_change().over(["scrip_code", "date"]).alias("_r"))
        .group_by("scrip_code").agg(v=pl.col("_r").std())[["scrip_code", "v"]])}

    daily_returns: list[float] = []
    per_name_daily: list[dict] = []
    fills_all: list[dict] = []
    total_cost = total_impact = total_turnover = 0.0
    n_orders = n_fills = 0
    n_maker_placed = n_maker_filled = 0.0

    mid_lookup = {(r["date"], int(r["scrip_code"]), int(r["minute"])): r["mid"]
                  for r in panel.select(["date", "scrip_code", "minute", "mid"]).iter_rows(named=True)}

    for (date,), day in panel.group_by("date", maintain_order=True):
        broker = PaperBroker(
            taker_slippage_bps=float(cfg.execution.taker_slippage_bps),
            maker_fill_prob=float(getattr(cfg.execution, "maker_fill_prob", 1.0)),
            seed=int(cfg.synthetic.seed))
        mgr = ExecutionManager(broker, cfg)
        prev_mid: dict[int, float] = {}
        day_pnl = day_cost = day_turnover = 0.0
        name_pnl: dict[int, float] = {}
        processed_costs: set[str] = set()

        for (minute,), g in day.group_by("minute", maintain_order=True):
            minute = int(minute)
            mod = session_open_min() + minute
            scrip = g["scrip_code"].to_numpy()
            mid = g["mid"].to_numpy()
            high = g["high"].to_numpy()
            low = g["low"].to_numpy()
            hs_bps = float(getattr(cfg.execution, "half_spread_bps", 1.0)) * 1e-4
            market = {}
            for i, sc in enumerate(scrip):
                sci = int(sc)
                hs = max(mid[i] * hs_bps, 0.05)        # passive quote offset from mid (§0.3)
                broker.update_market(sci, mid[i] - hs, mid[i] + hs, mid[i])
                broker.fill_on_range(sci, float(low[i]), float(high[i]))   # maker fill on touch
                market[sci] = (mid[i] - hs, mid[i] + hs, mid[i])

            # mark existing positions to this minute's mid
            for sci, pos in broker.positions().items():
                if pos.qty != 0.0 and sci in prev_mid:
                    pnl = pos.qty * (market.get(sci, (0, 0, prev_mid[sci]))[2] - prev_mid[sci])
                    day_pnl += pnl
                    name_pnl[sci] = name_pnl.get(sci, 0.0) + pnl
            for sci in market:
                prev_mid[sci] = market[sci][2]

            # book new fills -> costs
            for f in broker.poll_fills():
                n_fills += 1
                cost = leg_cost_rupees(f.qty * f.price, f.side, p)
                imp = impact_rupees(f.qty * f.price, adv.get(f.scrip_code, 1e7),
                                    sig.get(f.scrip_code, 0.02), p)
                if f.is_taker:
                    imp += 0.25e-4 * f.qty * f.price
                day_cost += (cost + imp)
                total_cost += cost
                total_impact += imp
                day_turnover += f.qty * f.price
                if not f.is_taker:
                    n_maker_filled += f.qty
                fills_all.append(dict(date=date, scrip=f.scrip_code, minute=minute,
                                      side=f.side, price=f.price, taker=f.is_taker))

            # decision / flatten
            is_decision = (minute % decision_interval == 0
                           and minute < flatten_session_min())
            if is_decision or mod >= cfg.execution.flatten_start_min:
                if is_decision and mod < cfg.execution.flatten_start_min:
                    beta = np.array([betas.get(int(s), 1.0) for s in scrip])
                    adv_arr = np.array([adv.get(int(s), 1e7) for s in scrip])
                    targets_arr = build_book(
                        g["primary_score"].to_numpy(), beta, g["sector"].to_numpy(), adv_arr,
                        gross_target=gross_target,
                        max_participation=float(cfg.portfolio.max_participation),
                        min_clip=float(cfg.portfolio.min_clip),
                        max_names=int(cfg.portfolio.max_names),
                        sector_cap=float(cfg.portfolio.sector_cap),
                        p_act=g["p_act"].to_numpy())
                    targets = {int(scrip[i]): float(targets_arr[i]) for i in range(len(scrip))}
                else:
                    targets = {}
                res = mgr.step(mod, mod * 60.0, targets, market)
                n_orders += len(res.orders)
                n_maker_placed += sum(o.qty for o in res.orders if o.order_type == "LIMIT")

        daily_returns.append((day_pnl - day_cost) / equity)
        total_turnover += day_turnover
        for sc, pnl in name_pnl.items():
            per_name_daily.append(dict(date=date, scrip_code=sc, pnl=pnl / equity))

    res = PaperSessionResult()
    res.daily_returns = np.array(daily_returns)
    res.net_sharpe = _sharpe(res.daily_returns)
    res.n_orders = n_orders
    res.n_fills = n_fills
    res.fill_ratio = float(n_maker_filled / n_maker_placed) if n_maker_placed > 0 else 0.0
    res.taker_fraction = float(sum(f["taker"] for f in fills_all) / max(len(fills_all), 1))
    res.turnover_x = float(total_turnover / max(panel["date"].n_unique(), 1) / max(gross_target, 1.0))
    res.total_cost_rupees = total_cost
    res.total_impact_rupees = total_impact
    if per_name_daily:
        pn = pl.DataFrame(per_name_daily)
        wide = pn.pivot(values="pnl", index="date", on="scrip_code").drop("date").fill_null(0.0)
        res.effective_breadth = effective_breadth(wide.to_numpy())
    res.markout_bps = _markouts(fills_all, mid_lookup, cfg)
    return res


def _markouts(fills, mid_lookup, cfg) -> dict:
    if not fills:
        return {int(h): 0.0 for h in cfg.backtest.markout_horizons_min}
    out = {}
    for h in cfg.backtest.markout_horizons_min:
        vals = []
        for f in fills:
            fut = mid_lookup.get((f["date"], f["scrip"], f["minute"] + int(h)))
            if fut is not None and f["price"] > 0:
                vals.append(f["side"] * (fut - f["price"]) / f["price"] * 1e4)
        out[int(h)] = float(np.mean(vals)) if vals else 0.0
    return out
