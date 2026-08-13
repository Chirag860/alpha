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


# Ordered (first match wins) asset-class -> substrings to look for in the lowercased symbol path.
# Covers MetaQuotes-Demo ("Forex"/"Indexes"/"Metals") and Pepperstone/IC
# ("Markets\\Forex\\...", "Markets\\Commodities\\Energies", "Markets\\Forwards\\Treasuries", ...).
_AUTO_GROUPS = {
    "FX":        ("forex",),
    "INDEX":     ("indic",),                         # matches "Indices" / "Indexes"
    "BOND":      ("treasur", "bond", "bund", "gilt"),
    "METAL":     ("metal", "gold", "silver", "platin", "pallad"),
    "ENERGY":    ("energ", "oil", "gas"),
    "CRYPTO":    ("crypto",),
    "COMMODITY": ("commodit", "soft", "agri"),
}
# per-class spread ceiling (bps): natural spreads vary hugely, so one number can't fit all.
_SPREAD_LIMIT = {"FX": 4.0, "INDEX": 8.0, "BOND": 8.0, "METAL": 12.0,
                 "ENERGY": 15.0, "CRYPTO": 80.0, "COMMODITY": 25.0}
# never treat these as macro trend instruments (single names / duplicates / exotics)
_EXCLUDE = ("stock", "etf", "warrant", "exotic", "ndf", "perp", "world asset")


def _spread_bps(mt5, name: str) -> float:  # pragma: no cover - VM
    info = mt5.symbol_info(name)
    tick = mt5.symbol_info_tick(name)
    if info is None or tick is None:
        return 1e9
    px = float(getattr(tick, "ask", 0.0) or getattr(tick, "bid", 0.0) or getattr(info, "bid", 0.0) or 0.0)
    point = float(getattr(info, "point", 0.0) or 0.0)
    spr = float(getattr(info, "spread", 0.0) or 0.0)
    if px <= 0:
        return 1e9
    return spr * point / px * 1e4


def _auto_universe(mt5, *, max_per_class: int = 15,
                   max_spread_bps: float | None = None) -> list[tuple[str, str]]:  # pragma: no cover - VM
    """Discover a diversified, LIQUID universe by symbol PATH, per-class spread-filtered.

    Excludes single stocks/ETFs/warrants/exotics/perps; keeps the tightest-spread instruments per
    asset class up to ``max_per_class``. ``max_spread_bps`` overrides the per-class limits.
    """
    cand: dict[str, list[tuple[str, float]]] = {k: [] for k in _AUTO_GROUPS}
    for s in (mt5.symbols_get() or ()):
        path = (getattr(s, "path", "") or "").lower()
        if getattr(s, "trade_mode", None) == getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
            continue
        if any(x in path for x in _EXCLUDE):
            continue
        for cls, needles in _AUTO_GROUPS.items():
            if any(n in path for n in needles):
                mt5.symbol_select(s.name, True)
                sb = _spread_bps(mt5, s.name)         # may be unreadable (closed market)
                # FAIL OPEN: only exclude on an EXPLICIT override with a readable spread.
                # A closed market (no live tick) must NOT drop the instrument — that silently
                # deleted whole asset classes (bonds/commodities/crypto) that were shut at export.
                if max_spread_bps is not None and sb < 1e8 and sb > max_spread_bps:
                    break
                cand[cls].append((s.name, sb))
                break
    out = []
    for cls, lst in cand.items():
        lst.sort(key=lambda x: x[1])                 # readable+tight first; unreadable last
        for n, _ in lst[:max_per_class]:
            out.append((n, cls))
    return out


def export(config_path: str, *, discover: bool = False, auto: bool = False,
           max_per_class: int = 15, max_spread_bps: float | None = None) -> int:  # pragma: no cover - VM only
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
        if auto:
            universe = _auto_universe(mt5, max_per_class=max_per_class, max_spread_bps=max_spread_bps)
            # merge in the config's explicit names (e.g. known-good indices) that path-scan missed
            have = {s for s, _ in universe}
            for s, c in _universe(m):
                if s not in have and mt5.symbol_info(s) is not None and mt5.symbol_select(s, True):
                    universe.append((s, c))
                    have.add(s)
            by_cls: dict[str, int] = {}
            for _, c in universe:
                by_cls[c] = by_cls.get(c, 0) + 1
            print(f"Auto-discovered {len(universe)} instruments: "
                  + ", ".join(f"{k}:{v}" for k, v in by_cls.items()))
        else:
            universe = _universe(m)
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
            # copy_rates_from_pos FORCES the terminal to download history on demand -- far more
            # reliable than copy_rates_range, which returns empty for symbols (futures/forwards/
            # less-active CFDs) whose history isn't cached yet. Retry once, then fall back.
            rates = mt5.copy_rates_from_pos(sym, tf, 0, 8000)
            if rates is None or len(rates) == 0:
                import time as _t
                _t.sleep(0.3)
                rates = mt5.copy_rates_from_pos(sym, tf, 0, 8000)
            if rates is None or len(rates) == 0:
                rates = mt5.copy_rates_range(sym, tf, start, dt.datetime.now())
            if rates is None or len(rates) == 0:
                missing.append(sym)
                continue
            df = pd.DataFrame(rates)
            ts = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[(ts >= pd.Timestamp(start, tz="UTC")).values].reset_index(drop=True)
            ts = pd.to_datetime(df["time"], unit="s", utc=True)
            if len(df) == 0:
                missing.append(sym)
                continue
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
    ap.add_argument("--max-per-class", type=int, default=15,
                    help="cap instruments per asset class (--auto) — keeps the book balanced across classes")
    ap.add_argument("--max-spread-bps", type=float, default=None,
                    help="override the per-class spread ceiling (--auto); default uses per-class limits")
    args = ap.parse_args()
    return export(args.config, discover=args.discover, auto=args.auto,
                  max_per_class=args.max_per_class, max_spread_bps=args.max_spread_bps)


if __name__ == "__main__":
    raise SystemExit(main())
