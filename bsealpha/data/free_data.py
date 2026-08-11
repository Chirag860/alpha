"""Free real-data loaders (yfinance) -- for a zero-cost smoke test only.

yfinance serves **1-minute OHLCV for roughly the last week** on BSE tickers (``.BO``) and
long daily history. That is enough to prove the pipeline ingests *real* BSE data and to run
a reduced (bars-only) model end to end for free -- but it is **not** research-grade:

* **no order-book depth and no trade tape** -> OFI / micro-price / book / signed-flow
  features are unavailable (use :func:`~bsealpha.features.build_features_bars_only`);
* **~1 week of history** -> far too short for CPCV / DSR / a lockbox (§5.1 needs years).

Treat this as a mechanics demo. Real research needs a paid depth vendor (TrueData/GDFL).
"""

from __future__ import annotations

import polars as pl

from .. import market

# A small liquid, dual-listed BSE universe with scrip codes + sectors (verify codes live).
DEFAULT_UNIVERSE: dict[str, tuple[int, str]] = {
    "RELIANCE.BO": (500325, "ENERGY"),
    "TCS.BO": (532540, "IT"),
    "INFY.BO": (500209, "IT"),
    "HDFCBANK.BO": (500180, "FIN"),
    "ICICIBANK.BO": (532174, "FIN"),
    "SBIN.BO": (500112, "FIN"),
    "AXISBANK.BO": (532215, "FIN"),
    "KOTAKBANK.BO": (500247, "FIN"),
    "ITC.BO": (500875, "FMCG"),
    "HINDUNILVR.BO": (500696, "FMCG"),
    "LT.BO": (500510, "INFRA"),
    "BHARTIARTL.BO": (532454, "TELECOM"),
}


def load_yfinance_panel(universe: dict[str, tuple[int, str]] | None = None,
                        period: str = "5d", interval: str = "1m"
                        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Download free 1-minute BSE bars and return ``(grid, meta)`` in canonical shape.

    ``grid`` is a common-minute grid (``mid = close``; no depth); ``meta`` carries sector and
    a default circuit band. Requires network + ``yfinance``. Raises if nothing loads.
    """
    import pandas as pd
    import yfinance as yf

    universe = universe or DEFAULT_UNIVERSE
    rows: list[dict] = []
    meta_rows: list[dict] = []
    for sym, (code, sector) in universe.items():
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False,
                             auto_adjust=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
        meta_rows.append({"scrip_code": code, "symbol": sym, "sector": sector,
                          "beta": 1.0, "circuit_band_pct": 20.0})
        for _, r in df.iterrows():
            ts = pd.Timestamp(r[tcol])
            mod = ts.hour * 60 + ts.minute
            minute = mod - market.session_open_min()
            if minute < 0 or minute >= market.session_len_min():
                continue
            close = float(r["Close"])
            vol = float(r["Volume"])
            rows.append({
                "scrip_code": code, "date": ts.strftime("%Y-%m-%d"), "minute": int(minute),
                "session_min": float(minute), "open": float(r["Open"]),
                "high": float(r["High"]), "low": float(r["Low"]), "close": close,
                "vwap": close, "turnover": close * vol, "n_trades": 0,
                "mid": close, "micro": close,
            })
    if not rows:
        raise RuntimeError("yfinance returned no intraday rows (market closed / throttled / "
                           "no network). Try a different period or run the synthetic demo.")
    grid = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date("%Y-%m-%d")).sort(
        ["date", "minute", "scrip_code"])
    meta = pl.DataFrame(meta_rows)
    return grid, meta
