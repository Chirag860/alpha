#!/usr/bin/env python3
"""Export DAILY bars + swap/carry metadata for the trend universe — **Windows/VM only**.

Pulls D1 OHLC for every symbol in ``config/trend.yaml``'s ``mt5.universe`` and each symbol's
current spread / swap rates, writing the canonical daily layout that :mod:`bsealpha.trend.data`
reads:

    <data_dir>/daily_grid.parquet   date, symbol, asset_class, open, high, low, close
    <data_dir>/daily_meta.parquet   symbol, asset_class, spread_bps, contract_size,
                                    swap_long, swap_short, currency

    python -m bsealpha.trend.mt5_daily_export --config config/trend.yaml --discover
    python -m bsealpha.trend.mt5_daily_export --config config/trend.yaml
"""

from __future__ import annotations

import argparse
import pathlib


def _universe(cfg_mt5: dict) -> list[tuple[str, str]]:
    out = []
    for cls, syms in (cfg_mt5.get("universe", {}) or {}).items():
        for s in syms:
            out.append((str(s), str(cls)))
    return out


# asset-class label -> case-insensitive substring of the symbol's MT5 path
_AUTO_GROUPS = {"FX": "forex", "INDEX": "index", "METAL": "metal",
                "ENERGY": "energ", "CRYPTO": "crypto", "COMMODITY": "commodit"}


def _auto_universe(mt5, *, max_per_class: int = 40) -> list[tuple[str, str]]:  # pragma: no cover - VM
    """Discover a diversified universe from the terminal by symbol PATH (capped per class).

    (mt5.symbols_get(group=...) matches the symbol NAME, not the folder — so we scan paths.)
    """
    per: dict[str, list[str]] = {k: [] for k in _AUTO_GROUPS}
    for s in (mt5.symbols_get() or ()):
        path = (getattr(s, "path", "") or "").lower()
        if getattr(s, "trade_mode", None) == getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
            continue
        for cls, needle in _AUTO_GROUPS.items():
            if needle in path:
                per[cls].append(s.name)
                break
    out = []
    for cls, names in per.items():
        for n in names[:max_per_class]:
            out.append((n, cls))
    return out


def export(config_path: str, *, discover: bool = False, auto: bool = False,
           max_per_class: int = 40) -> int:  # pragma: no cover - VM only
    import datetime as dt

    import pandas as pd
    import polars as pl
    import yaml

    from ..execution.mt5_broker import connect_mt5
    from ..config import load_config

    raw = yaml.safe_load(pathlib.Path(config_path).read_text()) or {}
    cfg = load_config(overrides=raw)
    m = raw.get("mt5", {})
    mt5 = connect_mt5(cfg)
    try:
        universe = _auto_universe(mt5, max_per_class=max_per_class) if auto else _universe(m)
        if auto:
            by_cls: dict[str, int] = {}
            for _, c in universe:
                by_cls[c] = by_cls.get(c, 0) + 1
            print(f"Auto-discovered {len(universe)} instruments: "
                  + ", ".join(f"{k}:{v}" for k, v in by_cls.items()))
        tf = getattr(mt5, "TIMEFRAME_" + str(m.get("timeframe", "D1")))
        start = dt.datetime.fromisoformat(str(m.get("history_start", "2015-01-01")))

        grid_rows, meta_rows, missing = [], [], []
        for sym, cls in universe:
            info = mt5.symbol_info(sym)
            if info is None or not mt5.symbol_select(sym, True):
                missing.append(sym)
                continue
            if discover:
                print(f"  OK  {sym:10s} [{cls}]  spread={getattr(info,'spread',0)} "
                      f"swap L/S={getattr(info,'swap_long',0)}/{getattr(info,'swap_short',0)}")
                continue
            rates = mt5.copy_rates_range(sym, tf, start, dt.datetime.now())
            if rates is None or len(rates) == 0:
                missing.append(sym)
                continue
            df = pd.DataFrame(rates)
            ts = pd.to_datetime(df["time"], unit="s", utc=True)
            for i in range(len(df)):
                grid_rows.append({"date": ts.iloc[i].strftime("%Y-%m-%d"), "symbol": sym,
                                  "asset_class": cls, "open": float(df["open"].iloc[i]),
                                  "high": float(df["high"].iloc[i]), "low": float(df["low"].iloc[i]),
                                  "close": float(df["close"].iloc[i])})
            point = float(getattr(info, "point", 0.0) or 0.0)
            last = float(df["close"].iloc[-1]) or 1.0
            spread_bps = (float(getattr(info, "spread", 0)) * point / last) * 1e4 if last else 1.0
            meta_rows.append({"symbol": sym, "asset_class": cls,
                              "spread_bps": max(spread_bps, 0.1),
                              "contract_size": float(getattr(info, "trade_contract_size", 1.0) or 1.0),
                              "swap_long": float(getattr(info, "swap_long", 0.0) or 0.0),
                              "swap_short": float(getattr(info, "swap_short", 0.0) or 0.0),
                              "currency": str(getattr(info, "currency_profit", "USD") or "USD")})

        if missing:
            print(f"  NOTE: {len(missing)} symbol(s) not found/available: {missing}\n"
                  "        edit config/trend.yaml mt5.universe to your broker's exact names (Ctrl+U).")
        if discover:
            print(f"Discovery done: {len(universe) - len(missing)}/{len(universe)} symbols available.")
            return 0
        if not grid_rows:
            print("No bars exported. Check universe symbol names and history_start.")
            return 1

        grid = (pl.DataFrame(grid_rows).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
                .sort(["date", "symbol"]))
        meta = pl.DataFrame(meta_rows)
        out = pathlib.Path(str(m.get("data_dir", "data/trend")))
        out.mkdir(parents=True, exist_ok=True)
        grid.write_parquet(out / "daily_grid.parquet")
        meta.write_parquet(out / "daily_meta.parquet")
        print(f"Exported {grid.height:,} daily bars | {meta.height} instruments | "
              f"{grid['date'].n_unique()} days -> {out}/")
        return 0
    finally:
        mt5.shutdown()


def main() -> int:  # pragma: no cover - VM only
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/trend.yaml")
    ap.add_argument("--discover", action="store_true", help="list which universe symbols exist, then exit")
    ap.add_argument("--auto", action="store_true",
                    help="auto-discover a diversified universe from the terminal (ignores config universe)")
    ap.add_argument("--max-per-class", type=int, default=40, help="cap instruments per asset class (--auto)")
    args = ap.parse_args()
    return export(args.config, discover=args.discover, auto=args.auto, max_per_class=args.max_per_class)


if __name__ == "__main__":
    raise SystemExit(main())
