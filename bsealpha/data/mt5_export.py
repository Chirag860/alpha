#!/usr/bin/env python3
"""Export MT5 stock-CFD history to the canonical (grid, meta) Parquet -- **Windows/VM only**.

Runs on the machine with the MetaTrader 5 terminal. Discovers the tradable stock-CFD
universe, pulls M1 bars per symbol, filters to the venue's regular session, and writes
``grid.parquet`` + ``meta.parquet`` + ``symbol_map.json`` under ``mt5.data_dir`` -- the exact
schema :func:`bsealpha.data.load_mt5_panel` reads on the research/training (Mac) side.

    python -m bsealpha.data.mt5_export --config config/mt5.yaml --discover-only
    python -m bsealpha.data.mt5_export --config config/mt5.yaml

``--discover-only`` connects, lists the stock CFDs it finds, and prints the count -- use it to
confirm the account has enough names (100+) for the cross-section BEFORE a full export.

**Timezone caveat (verify on your broker):** MT5 bar times are in the *broker server* zone,
not UTC. Set ``mt5.server_tz_offset_hours`` (server minus UTC) so bars map to the correct ET
session minute. MetaQuotes-Demo is typically UTC+2 (or +3 in DST). Sanity-check that the first
bar of a US day lands at session minute 0 (09:30 ET) after export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from .. import market


def _load_mt5_config(config_path: str):
    """default.yaml deep-merged with the mt5 overlay, and the active profile set."""
    import yaml
    overlay = yaml.safe_load(Path(config_path).read_text()) or {}
    cfg = load_config(overrides=overlay)
    market.set_active_profile_from_config(cfg)
    return cfg


def discover_symbols(mt5, cfg) -> list[str]:
    """Return tradable stock-CFD symbols matching ``mt5.symbol_group`` (or the explicit list)."""
    m = cfg.mt5
    if not bool(getattr(m, "discover", True)):
        return list(getattr(m, "symbols", []) or [])
    group = str(getattr(m, "symbol_group", "*Stock*") or "*")
    syms = mt5.symbols_get(group) or ()
    names = []
    for s in syms:
        # keep only symbols currently tradable (full trade mode)
        trade_mode = getattr(s, "trade_mode", None)
        if trade_mode == getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
            continue
        names.append(s.name)
    cap = int(getattr(m, "max_symbols", 0) or 0)
    return names[:cap] if cap else names


def _sector_map(cfg) -> dict[str, str]:
    import yaml
    path = getattr(cfg.mt5, "sector_map_path", "") or ""
    if path and Path(path).exists():
        return {str(k): str(v) for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items()}
    return {}


def rates_to_grid(rates, code: int, cfg) -> "object | None":  # pragma: no cover - VM only
    """Convert an MT5 rates array for one symbol to a canonical grid frame (or None).

    Shared by the batch exporter and the live runner so the server-tz -> session-minute
    mapping is defined exactly once.
    """
    import pandas as pd
    import polars as pl

    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    offset_h = float(getattr(cfg.mt5, "server_tz_offset_hours", 0.0) or 0.0)
    open_min, close_min = market.session_open_min(), market.session_close_min()
    ts = pd.to_datetime(df["time"], unit="s", utc=True) - pd.Timedelta(hours=offset_h)
    ts = ts.dt.tz_convert(market.active_profile().tz)
    mod = ts.dt.hour * 60 + ts.dt.minute
    minute = (mod - open_min).astype(int)
    keep = ((mod >= open_min) & (mod < close_min)).values
    if not keep.any():
        return None
    sub = df[keep]
    close = sub["close"].astype(float)
    vol = sub["tick_volume"].astype(float)
    return pl.DataFrame({
        "scrip_code": code,
        "date": ts[keep].dt.strftime("%Y-%m-%d").values,
        "minute": minute[keep].values,
        "session_min": minute[keep].astype(float).values,
        "open": sub["open"].astype(float).values,
        "high": sub["high"].astype(float).values,
        "low": sub["low"].astype(float).values,
        "close": close.values, "vwap": close.values,
        "turnover": (close * vol).values, "n_trades": 0,
        "mid": close.values, "micro": close.values,
    })


def export(config_path: str, *, discover_only: bool = False) -> int:  # pragma: no cover - VM only
    import polars as pl

    from .mt5_data import GRID_COLUMNS
    from ..execution.mt5_broker import connect_mt5

    cfg = _load_mt5_config(config_path)
    mt5 = connect_mt5(cfg)
    try:
        symbols = discover_symbols(mt5, cfg)
        suffix = str(getattr(cfg.mt5, "symbol_suffix", "") or "")
        print(f"Discovered {len(symbols)} tradable stock-CFD symbols "
              f"(group={getattr(cfg.mt5, 'symbol_group', '*')!r}).")
        if len(symbols) < 100:
            print("  WARNING: fewer than 100 names. The cross-sectional book needs breadth; "
                  "a thin universe (e.g. MetaQuotes-Demo) will make the strategy near-degenerate. "
                  "Consider a broker demo with a broad stock-CFD list.")
        if discover_only:
            for s in symbols:
                print("   ", s)
            return 0

        import datetime as _dt

        sectors = _sector_map(cfg)
        tf = getattr(mt5, "TIMEFRAME_" + str(getattr(cfg.mt5, "timeframe", "M1")), None)
        start = str(getattr(cfg.mt5, "history_start", "2024-01-01"))
        start_dt = _dt.datetime.fromisoformat(start)

        grid_rows: list[pl.DataFrame] = []
        meta_rows: list[dict] = []
        symbol_map: dict[int, str] = {}
        for i, sym in enumerate(sorted(symbols)):
            code = 900000 + i                       # synthetic stable scrip_code
            symbol_map[code] = sym
            mt5.symbol_select(sym, True)
            rates = mt5.copy_rates_range(sym, tf, start_dt, _dt.datetime.now())
            g = rates_to_grid(rates, code, cfg)
            if g is None:
                continue
            base = sym[:-len(suffix)] if suffix and sym.endswith(suffix) else sym
            meta_rows.append({"scrip_code": code, "symbol": sym,
                              "sector": sectors.get(base, "UNKNOWN"),
                              "beta": 1.0, "circuit_band_pct": 20.0})
            grid_rows.append(g)

        if not grid_rows:
            print("No bars exported (empty history / wrong session filter). "
                  "Check mt5.server_tz_offset_hours and history_start.")
            return 1

        grid = (pl.concat(grid_rows, how="vertical_relaxed")
                .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
                .select(GRID_COLUMNS)
                .sort(["date", "minute", "scrip_code"]))
        meta = pl.DataFrame(meta_rows)

        out = Path(str(cfg.mt5.data_dir))
        out.mkdir(parents=True, exist_ok=True)
        grid.write_parquet(out / "grid.parquet")
        meta.write_parquet(out / "meta.parquet")
        (out / "symbol_map.json").write_text(json.dumps(symbol_map, indent=2))
        print(f"Exported {grid.height:,} bars | {meta.height} names | "
              f"{grid['date'].n_unique()} sessions -> {out}/")
        return 0
    finally:
        mt5.shutdown()


def main() -> int:  # pragma: no cover - VM only
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/mt5.yaml", help="MT5 overlay config path")
    ap.add_argument("--discover-only", action="store_true",
                    help="List the tradable stock-CFD universe and exit (no export)")
    args = ap.parse_args()
    return export(args.config, discover_only=args.discover_only)


if __name__ == "__main__":
    raise SystemExit(main())
