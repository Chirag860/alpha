#!/usr/bin/env python3
"""Trend + carry system runner — backtest/validate (anywhere) and live (Windows/VM).

    python3 run_trend.py --backtest              # backtest+validate on data/trend (needs export)
    python3 run_trend.py --backtest --synthetic  # demo on synthetic trending data (no data needed)
    python  run_trend.py --live                  # daily rebalance to the MT5 demo (VM only)
    python  run_trend.py --live --dry-run        # compute the target book, print, don't send

Data-heavy work (backtest/validate) runs anywhere. `--live` needs the MT5 terminal (Windows/VM).
Export daily history first with:  python -m bsealpha.trend.mt5_daily_export --config config/trend.yaml
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import pathlib

import yaml


def _flatten_universe(cfg_mt5) -> list[tuple[str, str]]:
    """Return [(symbol, asset_class)] from the config's mt5.universe mapping."""
    uni = getattr(cfg_mt5, "universe", None)
    out = []
    if uni is not None:
        for cls, syms in (uni.items() if hasattr(uni, "items") else dict(uni).items()):
            for s in syms:
                out.append((str(s), str(cls)))
    return out


def do_backtest(config: str, synthetic: bool, overrides: dict) -> int:
    from bsealpha.trend import (
        load_daily_panel, load_trend_params, run_trend_backtest,
        validate_book, format_report,
    )
    params = load_trend_params(config, overrides=overrides)
    if synthetic:
        from bsealpha.trend.synthetic import generate_daily_panel
        print("Synthetic trending demo (no real data).")
        grid, meta = generate_daily_panel(n_inst=28, n_days=2200, seed=3)
    else:
        cfg = yaml.safe_load(pathlib.Path(config).read_text()) or {}
        data_dir = cfg.get("mt5", {}).get("data_dir", "data/trend")
        grid, meta = load_daily_panel(data_dir)
        print(f"Loaded daily panel: {grid['symbol'].n_unique()} instruments | "
              f"{grid['date'].n_unique()} days")
    res, book = run_trend_backtest(grid, meta, params)
    v = validate_book(book["dates"], res, params, n_trials=1)
    print(format_report(res, v, book))
    return 0


def do_live(config: str, dry_run: bool) -> int:  # pragma: no cover - VM only
    import numpy as np

    from bsealpha.config import load_config
    from bsealpha.execution import MT5BrokerAdapter, connect_mt5
    from bsealpha.execution.broker import Order
    from bsealpha.trend import compute_book, load_trend_params
    from bsealpha.trend.data import load_daily_panel

    params = load_trend_params(config)
    cfg = load_config(overrides=yaml.safe_load(pathlib.Path(config).read_text()) or {})
    grid, meta = load_daily_panel(str(cfg.mt5.data_dir))     # trailing daily history (re-export daily)
    book = compute_book(grid, meta, params, include_carry=True)
    symbols = book["symbols"]
    w_today = book["weights"][-1]                            # target weights for today's rebalance

    mt5 = connect_mt5(cfg)
    try:
        equity = float(mt5.account_info().equity)
        symbol_map = {i: s for i, s in enumerate(symbols)}
        adapter = MT5BrokerAdapter(mt5, symbol_map, magic=770001, force_market=True)
        # current signed notional per symbol (from live positions), and targets
        pos = adapter.positions()
        cur_notional = {i: (pos[i].qty * float(mt5.symbol_info_tick(symbols[i]).bid))
                        for i in pos}
        n_orders = 0
        for i, s in enumerate(symbols):
            tick = mt5.symbol_info_tick(s)
            px = float(tick.bid or tick.ask or 0.0)
            if px <= 0:
                continue
            tgt_notional = float(w_today[i]) * equity
            gap = tgt_notional - cur_notional.get(i, 0.0)
            shares = abs(gap) / px
            if shares <= 0:
                continue
            side = 1 if gap > 0 else -1
            order = Order(scrip_code=i, side=side, qty=shares, order_type="MARKET",
                          price=None, algo_id="TREND-0001", ts_ns=0)
            print(f"  {s:8s} w={w_today[i]:+.3f} target=${tgt_notional:+,.0f} "
                  f"gap=${gap:+,.0f} -> {'BUY' if side>0 else 'SELL'} {shares:.3f}")
            if not dry_run:
                adapter.place_order(order)
                n_orders += 1
        print(f"\nEquity ${equity:,.0f} | {'DRY-RUN, no orders sent' if dry_run else f'{n_orders} orders sent'} "
              f"| rejects {adapter.reject_rate():.0%}")
    finally:
        mt5.shutdown()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/trend.yaml")
    ap.add_argument("--backtest", action="store_true", help="run backtest + validation")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (with --backtest)")
    ap.add_argument("--live", action="store_true", help="daily rebalance to the MT5 demo (VM only)")
    ap.add_argument("--dry-run", action="store_true", help="with --live: compute + print, don't send")
    ap.add_argument("--lookbacks", default=None, help="override signal lookbacks, e.g. 63,126,252")
    ap.add_argument("--band", type=float, default=None, help="override no-trade band (cut turnover)")
    ap.add_argument("--target-vol", type=float, default=None, help="override annual vol target")
    args = ap.parse_args()
    if args.live:
        return do_live(args.config, args.dry_run)
    overrides: dict = {}
    if args.lookbacks:
        overrides["lookbacks"] = [int(x) for x in args.lookbacks.split(",")]
    if args.band is not None:
        overrides["no_trade_band"] = args.band
    if args.target_vol is not None:
        overrides["target_ann_vol"] = args.target_vol
    return do_backtest(args.config, args.synthetic, overrides)


if __name__ == "__main__":
    raise SystemExit(main())
