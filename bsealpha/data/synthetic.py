"""Synthetic BSE multi-name panel generator.

Produces a self-consistent cross-sectional panel so the *entire* pipeline runs with
zero paid data (§10, §11 make "runnable on sample data" a hard deliverable):

* per-name **5-level depth snapshots** and **trade prints** (event streams),
* a **daily EOD reference panel** with surveillance flags and corporate actions,
* **static metadata** (sector, intraday beta/gamma, liquidity tier).

Design choices that make the synthetic data *useful for research*, not just runnable:

1. **Factor structure** (§0.5). Every name's return = ``beta*market + gamma*sector +
   residual``. This is what makes effective-breadth < N and forces residualization to
   matter -- exactly the illusion the report is built around. A model on raw returns
   will look like an index-timer here, by construction.
2. **A genuine, small residual edge.** A latent mean-reverting signal ``s`` drives both
   the observable depth imbalance *and* the next residual return, with a deliberately
   tiny coefficient. So OFI / micro-price carry real, thin predictive content -- the
   pipeline should recover a modest IC, never a huge one (a huge one would be a bug per
   §11.3).
3. **U-shaped intraday vol** (§2.1) so the time-of-day normalization has something to do.
4. **Realistic tick bands, spreads, and a liquidity spread across names** so the
   universe screen (§1.2) admits some names and rejects others.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import Config
from .. import market
from ..market import round_to_tick, tick_size

_SECTORS = ["FIN", "IT", "ENERGY", "PHARMA", "AUTO", "FMCG", "METAL", "INFRA"]


@dataclass
class SyntheticPanel:
    """Container for a generated panel."""

    depth: pl.DataFrame        # 5-level snapshots (DEPTH_SCHEMA + mid/imbalance truth)
    trades: pl.DataFrame       # trade prints (TRADE_SCHEMA)
    daily: pl.DataFrame        # EOD reference panel (DAILY_SCHEMA)
    meta: pl.DataFrame         # static per-name metadata (sector, beta, gamma, tier)


def _u_shape(session_min: np.ndarray) -> np.ndarray:
    """Intraday vol multiplier: high at open/close, lull in the middle (§2.1)."""
    x = session_min / market.session_len_min()   # 0..1
    return 0.7 + 1.8 * (np.exp(-x / 0.12) + np.exp(-(1.0 - x) / 0.15))


def generate_panel(cfg: Config, *, seed: int | None = None) -> SyntheticPanel:
    """Generate a :class:`SyntheticPanel` from the ``synthetic:`` config block.

    Parameters
    ----------
    cfg
        Loaded :class:`~bsealpha.config.Config`.
    seed
        Optional override of ``cfg.synthetic.seed`` for reproducible variation.
    """
    sc = cfg.synthetic
    rng = np.random.default_rng(sc.seed if seed is None else seed)

    n_names = int(sc.n_names)
    n_days = int(sc.n_days)
    n_sectors = min(int(sc.n_sectors), len(_SECTORS))
    dt_sec = float(sc.bar_seconds)
    steps = int(market.session_len_min() * 60 / dt_sec)        # snapshots per day
    annual_steps = steps * 250

    # -- static per-name metadata -----------------------------------------
    scrip_codes = 500000 + np.arange(n_names)
    sectors = np.array([_SECTORS[i % n_sectors] for i in range(n_names)])
    base_price = np.exp(rng.uniform(np.log(sc.base_price_range[0]),
                                    np.log(sc.base_price_range[1]), n_names))
    ann_vol = rng.uniform(sc.ann_vol_range[0], sc.ann_vol_range[1], n_names)
    beta = rng.uniform(sc.market_beta_range[0], sc.market_beta_range[1], n_names)
    gamma = rng.uniform(sc.sector_gamma_range[0], sc.sector_gamma_range[1], n_names)
    # liquidity tier spreads turnover across ~Rs 1cr .. Rs 60cr/day so the screen bites.
    tier = np.exp(rng.uniform(np.log(1e7), np.log(6e8), n_names))   # target daily turnover

    # surveillance / exclusion flags assigned at name level (§1.3)
    def _flag(frac: float) -> np.ndarray:
        return rng.random(n_names) < frac

    t2t = _flag(sc.frac_t2t)
    asm = _flag(sc.frac_asm)
    gsm = _flag(sc.frac_gsm)
    suspended = _flag(sc.frac_suspended)
    series = np.where(t2t, "T", np.where(rng.random(n_names) < 0.5, "A", "B"))
    circuit_band = np.where(rng.random(n_names) < 0.15,
                            rng.choice([5.0, 10.0], n_names), 20.0)

    # per-step vol contributions
    mkt_step_vol = sc.market_vol_ann / np.sqrt(annual_steps)
    sec_step_vol = sc.sector_vol_ann / np.sqrt(annual_steps)
    idio_step_vol = ann_vol / np.sqrt(annual_steps)            # per-name

    session_min = (np.arange(steps) + 1) * dt_sec / 60.0        # 0 < .. <= session_len
    ushape = _u_shape(session_min)
    minute_of_day = market.session_open_min() + session_min

    base_date = dt.date(2026, 1, 5)                             # a Monday
    kappa = 0.06                                                 # residual-edge strength
    trades_per_step = 8.0                                        # ~liquid-name trade rate

    depth_rows: list[pl.DataFrame] = []
    trade_rows: list[pl.DataFrame] = []
    daily_records: list[dict] = []

    n_levels = 5
    for d in range(n_days):
        date = base_date + dt.timedelta(days=d)
        # shared factor paths for the day
        m_inc = rng.normal(0, mkt_step_vol, steps) * ushape
        sec_inc = {s: rng.normal(0, sec_step_vol, steps) * ushape for s in range(n_sectors)}

        for i in range(n_names):
            sec_id = i % n_sectors
            # latent mean-reverting signal driving both imbalance and next residual
            s = np.zeros(steps)
            eps_s = rng.normal(0, 1.0, steps)
            for k in range(1, steps):
                s[k] = 0.92 * s[k - 1] + 0.39 * eps_s[k]        # AR(1), unit-ish variance

            idio = rng.normal(0, idio_step_vol[i], steps) * ushape
            # predictable part: next residual return loads on lagged signal s[k-1]
            resid_ret = idio.copy()
            resid_ret[1:] += kappa * idio_step_vol[i] * s[:-1] * ushape[1:]

            logret = beta[i] * m_inc + gamma[i] * sec_inc[sec_id] + resid_ret
            mid = base_price[i] * np.exp(np.cumsum(logret))

            # ---- depth snapshots -------------------------------------
            ts = tick_size(mid)
            imb = 1.0 / (1.0 + np.exp(-(1.4 * s + rng.normal(0, 0.4, steps))))  # in (0,1)
            spread_ticks = np.where(rng.random(steps) < 0.7, 1.0, 2.0)
            half = spread_ticks * ts / 2.0
            bid0 = round_to_tick(mid - half)
            ask0 = round_to_tick(mid + half)
            ask0 = np.where(ask0 <= bid0, bid0 + ts, ask0)      # keep book uncrossed

            # base displayed size scaled so per-day turnover ~ tier[i]
            avg_trade_val = tier[i] / (steps * trades_per_step)
            base_qty = np.maximum(avg_trade_val / np.maximum(mid, 1.0) * 6.0, 1.0)

            level_decay = np.array([1.0, 0.7, 0.5, 0.35, 0.25])
            depth = {}
            for lv in range(n_levels):
                depth[f"bid_px_{lv}"] = bid0 - lv * ts
                depth[f"ask_px_{lv}"] = ask0 + lv * ts
                bq = base_qty * level_decay[lv] * (0.5 + imb) * (1 + rng.normal(0, 0.1, steps))
                aq = base_qty * level_decay[lv] * (1.5 - imb) * (1 + rng.normal(0, 0.1, steps))
                depth[f"bid_qty_{lv}"] = np.maximum(bq, 1.0)
                depth[f"ask_qty_{lv}"] = np.maximum(aq, 1.0)

            ts_ns = (session_min * 60 * 1e9).astype(np.int64) + i  # +i keeps ties distinct
            frame = {
                "scrip_code": np.full(steps, scrip_codes[i], dtype=np.int64),
                "ts_ns": ts_ns,
                "session_min": session_min,
                "mid": mid,
                "imbalance_true": imb,
                **depth,
            }
            depth_rows.append(
                pl.DataFrame(frame).with_columns(pl.lit(date).cast(pl.Date).alias("date"))
            )

            # ---- trades ----------------------------------------------
            n_tr = rng.poisson(trades_per_step, steps)
            total = int(n_tr.sum())
            if total > 0:
                rep = np.repeat(np.arange(steps), n_tr)
                buy = rng.random(total) < imb[rep]              # buyer-initiated ~ imbalance
                tprice = np.where(buy, ask0[rep], bid0[rep])
                tqty = np.maximum(rng.lognormal(np.log(np.maximum(base_qty[rep] * 0.15, 1.0)),
                                                0.5), 1.0)
                trade_rows.append(pl.DataFrame({
                    "scrip_code": np.full(total, scrip_codes[i], dtype=np.int64),
                    "ts_ns": ts_ns[rep] + rng.integers(0, int(dt_sec * 1e9), total),
                    "session_min": session_min[rep],
                    "price": tprice,
                    "qty": tqty,
                    "true_sign": np.where(buy, 1, -1).astype(np.int8),
                }).with_columns(pl.lit(date).cast(pl.Date).alias("date")))
                day_turnover = float((tprice * tqty).sum())
                day_trades = total
            else:  # pragma: no cover - trades_per_step keeps this rare
                day_turnover, day_trades = 0.0, 0

            med_spread_bps = float(np.median((ask0 - bid0) / mid) * 1e4)
            daily_records.append({
                "date": date.isoformat(),
                "scrip_code": int(scrip_codes[i]),
                "symbol": f"SCRIP{scrip_codes[i]}",
                "sector": str(sectors[i]),
                "close": float(mid[-1]),
                "bse_turnover": day_turnover,
                "bse_trades": int(day_trades),
                "median_spread_bps": med_spread_bps,
                "series": str(series[i]),
                "asm_flag": bool(asm[i]),
                "gsm_flag": bool(gsm[i]),
                "t2t_flag": bool(t2t[i]),
                "is_suspended": bool(suspended[i]),
                "circuit_band_pct": float(circuit_band[i]),
                "adj_factor": 1.0,
            })

    depth_df = pl.concat(depth_rows, how="vertical").sort(["date", "scrip_code", "ts_ns"])
    trades_df = (pl.concat(trade_rows, how="vertical").sort(["date", "scrip_code", "ts_ns"])
                 if trade_rows else pl.DataFrame())
    daily_df = pl.DataFrame(daily_records).with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )

    meta_df = pl.DataFrame({
        "scrip_code": scrip_codes.astype(np.int64),
        "symbol": [f"SCRIP{c}" for c in scrip_codes],
        "sector": sectors,
        "base_price": base_price,
        "ann_vol": ann_vol,
        "beta": beta,
        "gamma": gamma,
        "target_turnover": tier,
        "series": series,
        "t2t_flag": t2t,
        "asm_flag": asm,
        "gsm_flag": gsm,
        "is_suspended": suspended,
        "circuit_band_pct": circuit_band,
    })

    return SyntheticPanel(depth=depth_df, trades=trades_df, daily=daily_df, meta=meta_df)
