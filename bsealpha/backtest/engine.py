"""Event-driven cross-sectional backtest with the Indian constraint set (§6, §7).

A custom day-by-day, minute-by-minute loop (§6.5 recommends exactly this -- no off-the-
shelf framework knows the Indian constraints). At each decision minute it builds a
market/sector-neutral, participation-capped book (§7.1), applies a no-trade band, charges
the real cost stack **per order** (§0.1, §6.2) plus square-root impact against BSE ADV
(§6.4), and enforces:

* **circuit proximity** -- no new positions within ``circuit_no_new_pct`` of a band; forced
  flat within ``circuit_flatten_pct`` (the circuit trap, §1.3);
* **forced flatten** from 15:15 -- everything liquidated as a taker in an adversely-
  selected window (§0.4, §7.3);
* **≤10 orders/second** SEBI ceiling via a per-minute token budget (§8.1).

P&L is marked on **raw** mid returns (what you actually hold); the neutral construction is
what makes the net residual-like. Outputs :class:`~bsealpha.backtest.metrics.BacktestMetrics`
including markouts and effective breadth.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .. import market
from ..config import Config
from ..portfolio.construct import build_book, rebalance
from ..validation.breadth import effective_breadth
from .costs import CostParams, impact_rupees, leg_cost_rupees
from .metrics import (
    BacktestMetrics,
    capacity_estimate,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
)


def _daily_vol_by_scrip(panel: pl.DataFrame) -> dict[int, float]:
    """Per-name daily vol = std(minute raw returns) * sqrt(session_len), for the impact model."""
    scale = np.sqrt(float(market.session_len_min()))
    r = (panel.sort(["scrip_code", "date", "minute"])
         .with_columns((pl.col("mid").pct_change().over(["scrip_code", "date"])).alias("_r"))
         .group_by("scrip_code").agg(v=pl.col("_r").std()))
    return {int(k): float(val or 0.0) * scale
            for k, val in zip(r["scrip_code"], r["v"])}


def run_backtest(panel: pl.DataFrame, cfg: Config, *,
                 betas: dict[int, float] | None = None,
                 adv_by_scrip: dict[int, float] | None = None,
                 cancel_model: str | None = None) -> BacktestMetrics:
    """Run the event-driven backtest over a panel carrying predictions.

    ``panel`` must have ``primary_score``, ``p_act``, ``mid``, ``sector``, ``scrip_code``,
    ``date``, ``minute``, ``circuit_dist_upper``, ``circuit_dist_lower``. ``betas`` /
    ``adv_by_scrip`` default to 1.0 / per-name median turnover.
    """
    bt = cfg.backtest
    p = CostParams.from_config(cfg)
    horizon = int(cfg.labeling.horizon_min)
    equity = float(bt.equity)
    no_new = float(bt.circuit_no_new_pct)
    flat_pct = float(bt.circuit_flatten_pct)
    ops_budget = int(bt.max_ops) * 60          # per-minute order budget (10/s)
    gross_target = float(cfg.portfolio.gross_target)

    sigma_daily = _daily_vol_by_scrip(panel)
    if not adv_by_scrip:
        # median per-day turnover per name = the participation base (§6.4)
        tmp = (panel.group_by(["scrip_code", "date"]).agg(t=pl.col("turnover").sum())
               .group_by("scrip_code").agg(m=pl.col("t").median()))
        adv_by_scrip = {int(k): float(v) for k, v in zip(tmp["scrip_code"], tmp["m"])}
    betas = betas or {}

    decision_interval = int(getattr(bt, "decision_interval_min", horizon))
    smooth = int(getattr(bt, "score_smooth_min", 0))
    panel = panel.sort(["date", "minute", "scrip_code"]).with_columns(
        pl.col("mid").pct_change().over(["scrip_code", "date"]).fill_null(0.0).alias("ret_raw")
    )
    if smooth > 1:
        # signal persistence: average the score over the recent window so the neutral
        # book does not sign-flip on per-minute noise (the dominant turnover source, §7.1)
        panel = panel.with_columns(
            pl.col("primary_score").rolling_mean(smooth, min_samples=1)
            .over(["scrip_code", "date"]).alias("primary_score")
        )

    daily_returns: list[float] = []
    gross_daily: list[float] = []             # marking P&L before costs
    per_name_daily: list[dict] = []           # {date, scrip, pnl} for breadth
    fills: list[dict] = []
    total_cost = 0.0
    total_impact = 0.0
    total_turnover = 0.0
    n_trades = 0
    n_days = 0

    for (date,), day in panel.group_by("date", maintain_order=True):
        n_days += 1
        positions: dict[int, float] = {}
        entry_price: dict[int, float] = {}
        day_pnl = 0.0        # marking P&L before costs (gross)
        day_cost = 0.0       # fees + impact charged during the day
        day_gross_traded = 0.0
        name_pnl: dict[int, float] = {}

        for (minute,), g in day.group_by("minute", maintain_order=True):
            minute = int(minute)
            scrip = g["scrip_code"].to_numpy()
            ret_raw = g["ret_raw"].to_numpy()
            mid = g["mid"].to_numpy()
            score = g["primary_score"].to_numpy()
            p_act = g["p_act"].to_numpy()
            cu = g["circuit_dist_upper"].to_numpy()
            cl = g["circuit_dist_lower"].to_numpy()

            # (1) mark existing positions to this minute's move
            for i, sc in enumerate(scrip):
                pos = positions.get(int(sc), 0.0)
                if pos != 0.0:
                    pnl = pos * ret_raw[i]
                    day_pnl += pnl
                    name_pnl[int(sc)] = name_pnl.get(int(sc), 0.0) + pnl

            # (2) forced flatten from the deadline -- taker liquidation (adverse window)
            if minute >= market.flatten_session_min():
                for i, sc in enumerate(scrip):
                    sci = int(sc)
                    pos = positions.get(sci, 0.0)
                    if pos != 0.0:
                        cost, imp, _ = _close_position(
                            pos, mid[i], adv_by_scrip.get(sci, 1e7),
                            sigma_daily.get(sci, 0.02), p, entry_price.get(sci, mid[i]),
                            taker=True)
                        day_cost += (cost + imp)
                        total_cost += cost
                        total_impact += imp
                        day_gross_traded += abs(pos)
                        n_trades += 1
                        fills.append(dict(date=date, scrip=sci, minute=minute,
                                          side=int(-np.sign(pos)), price=float(mid[i])))
                        positions[sci] = 0.0
                continue

            # (3) decision minute: rebuild the neutral, capped book
            if minute % decision_interval != 0:
                continue
            adv = np.array([adv_by_scrip.get(int(s), 1e7) for s in scrip])
            beta = np.array([betas.get(int(s), 1.0) for s in scrip])
            sectors = g["sector"].to_numpy()
            target = build_book(
                score, beta, sectors, adv, gross_target=gross_target,
                max_participation=float(bt.circuit_no_new_pct) * 0 + float(cfg.portfolio.max_participation),
                min_clip=float(cfg.portfolio.min_clip),
                max_names=int(cfg.portfolio.max_names),
                sector_cap=float(cfg.portfolio.sector_cap), p_act=p_act,
            )
            current = np.array([positions.get(int(s), 0.0) for s in scrip])

            # circuit constraints (feature + hard constraint, §1.3)
            band_dist = np.minimum(cu, cl)
            target = np.where(band_dist < flat_pct, 0.0, target)          # force flat
            no_new_mask = band_dist < no_new                              # no new/increase
            target = np.where(no_new_mask & (np.abs(target) > np.abs(current)),
                              current, target)

            trades = rebalance(current, target, band_frac=float(cfg.portfolio.no_trade_band_frac),
                               min_clip=float(cfg.portfolio.min_clip))

            # SEBI OPS cap: keep the largest trades within the per-minute budget (§8.1)
            nz = np.flatnonzero(trades != 0.0)
            if len(nz) > ops_budget:
                keep = nz[np.argsort(-np.abs(trades[nz]))[:ops_budget]]
                mask = np.zeros(len(trades), bool)
                mask[keep] = True
                trades = np.where(mask, trades, 0.0)

            for i in np.flatnonzero(trades != 0.0):
                sci = int(scrip[i])
                tr = float(trades[i])
                side = int(np.sign(tr))
                notional = abs(tr)
                cost = leg_cost_rupees(notional, side, p)
                imp = impact_rupees(notional, adv[i], sigma_daily.get(sci, 0.02), p)
                day_cost += (cost + imp)
                total_cost += cost
                total_impact += imp
                day_gross_traded += notional
                n_trades += 1
                new_pos = positions.get(sci, 0.0) + tr
                positions[sci] = new_pos
                entry_price[sci] = float(mid[i])
                fills.append(dict(date=date, scrip=sci, minute=minute, side=side,
                                  price=float(mid[i])))

        daily_returns.append((day_pnl - day_cost) / equity)
        gross_daily.append(day_pnl / equity)
        total_turnover += day_gross_traded
        for sc, pnl in name_pnl.items():
            per_name_daily.append(dict(date=date, scrip_code=sc, pnl=pnl / equity))

    return _assemble_metrics(
        cfg, np.array(daily_returns), np.array(gross_daily), per_name_daily,
        fills, panel, total_cost, total_impact, total_turnover, n_trades, n_days, adv_by_scrip,
    )


def _close_position(pos: float, mid: float, adv: float, sigma_daily: float,
                    p: CostParams, entry: float, taker: bool) -> tuple[float, float, float]:
    """Cost/impact/pnl of closing a position (used by the forced flatten)."""
    side = int(-np.sign(pos))
    notional = abs(pos)
    cost = leg_cost_rupees(notional, side, p)
    imp = impact_rupees(notional, adv, sigma_daily, p)
    # taker exits in the flatten window pay an extra half-spread proxy (adverse selection)
    if taker:
        imp += 0.25e-4 * notional
    trade_pnl = 0.0                       # marking already captured price P&L
    return cost, imp, trade_pnl


def _assemble_metrics(cfg, daily_returns, gross_daily, per_name_daily, fills,
                      panel, total_cost, total_impact, total_turnover, n_trades, n_days,
                      adv_by_scrip) -> BacktestMetrics:
    m = BacktestMetrics()
    m.daily_returns = daily_returns
    m.gross_daily_returns = gross_daily
    m.sharpe = sharpe(daily_returns)
    m.gross_sharpe = sharpe(gross_daily)
    m.sortino = sortino(daily_returns)
    m.max_drawdown = max_drawdown(daily_returns)
    # hit rate = fraction of profitable name-days (per-position outcomes)
    m.hit_rate = hit_rate(np.array([r["pnl"] for r in per_name_daily])) if per_name_daily else 0.0
    m.n_trades = n_trades
    m.total_cost_rupees = float(total_cost)
    m.total_impact_rupees = float(total_impact)
    gross = float(cfg.portfolio.gross_target)
    m.turnover_x = float(total_turnover / max(n_days, 1) / max(gross, 1.0))
    m.capacity_rupees = capacity_estimate(
        np.array(list(adv_by_scrip.values())),
        float(cfg.portfolio.max_participation),
        turns_per_day=max(m.turnover_x, 1.0),
    )

    # effective breadth from per-name daily P&L (§0.5)
    if per_name_daily:
        pn = pl.DataFrame(per_name_daily)
        wide = (pn.pivot(values="pnl", index="date", on="scrip_code")
                .drop("date").fill_null(0.0))
        m.effective_breadth = effective_breadth(wide.to_numpy())

    # markouts from executed fills (§6.3)
    m.markout_bps = _compute_markouts(fills, panel, cfg)
    return m


def _compute_markouts(fills, panel, cfg) -> dict:
    if not fills:
        return {int(h): 0.0 for h in cfg.backtest.markout_horizons_min}
    mid_lookup = {(r["date"], int(r["scrip_code"]), int(r["minute"])): r["mid"]
                  for r in panel.select(["date", "scrip_code", "minute", "mid"]).iter_rows(named=True)}
    out = {}
    for h in cfg.backtest.markout_horizons_min:
        vals = []
        for f in fills:
            fut = mid_lookup.get((f["date"], f["scrip"], f["minute"] + int(h)))
            if fut is not None and f["price"] > 0:
                vals.append(f["side"] * (fut - f["price"]) / f["price"] * 1e4)
        out[int(h)] = float(np.mean(vals)) if vals else 0.0
    return out
