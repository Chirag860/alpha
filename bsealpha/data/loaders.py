"""Data loaders: local historical (Parquet) and a live broker-API interface.

Three sources, one canonical output (§9 INGEST):

* :class:`ParquetLoader`  -- local historical depth/trade/daily Parquet files, the
  path used for research and CI.
* :class:`BrokerFeed`     -- an *interface* for a SEBI-compliant broker websocket
  (Kite / Dhan / Fyers shape). Not exercised offline; documented so the live wiring
  is obvious and the same downstream code consumes it.
* :func:`panel_to_parquet` / :func:`SyntheticLoader` -- persist / replay a generated
  :class:`~bsealpha.data.synthetic.SyntheticPanel`.

Note we deliberately do **not** use ``ccxt``: that library is crypto-exchange shaped.
Indian equities go through a registered broker under the SEBI algo framework (§8.1);
there is no direct-exchange retail path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Protocol, runtime_checkable

import polars as pl

from .schema import BAR_SCHEMA, DAILY_SCHEMA, DEPTH_SCHEMA, TRADE_SCHEMA, validate_frame
from .synthetic import SyntheticPanel


class ParquetLoader:
    """Load a historical panel from a directory of Parquet files.

    Expected layout::

        root/
          depth.parquet   (DEPTH_SCHEMA)
          trades.parquet  (TRADE_SCHEMA)
          daily.parquet   (DAILY_SCHEMA)

    Uses ``scan_parquet`` (lazy) so a multi-GB, multi-year panel is not pulled into
    memory until collected (§10 recommends polars streaming for exactly this).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _scan(self, name: str) -> pl.LazyFrame:
        path = self.root / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"expected {path}")
        return pl.scan_parquet(path)

    def depth(self) -> pl.LazyFrame:
        return self._scan("depth")

    def trades(self) -> pl.LazyFrame:
        return self._scan("trades")

    def daily(self) -> pl.DataFrame:
        df = self._scan("daily").collect()
        validate_frame(df, DAILY_SCHEMA, name="daily")
        return df

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Collect (depth, trades, daily) eagerly. Prefer the lazy scanners for scale."""
        depth = self.depth().collect()
        trades = self.trades().collect()
        validate_frame(depth, DEPTH_SCHEMA, name="depth")
        validate_frame(trades, TRADE_SCHEMA, name="trades")
        return depth, trades, self.daily()


def panel_to_parquet(panel: SyntheticPanel, root: str | Path) -> Path:
    """Persist a :class:`SyntheticPanel` to Parquet for later :class:`ParquetLoader` use."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    panel.depth.write_parquet(root / "depth.parquet")
    panel.trades.write_parquet(root / "trades.parquet")
    panel.daily.write_parquet(root / "daily.parquet")
    panel.meta.write_parquet(root / "meta.parquet")
    return root


class SyntheticLoader:
    """Adapt an in-memory :class:`SyntheticPanel` to the loader interface."""

    def __init__(self, panel: SyntheticPanel) -> None:
        self.panel = panel

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        return self.panel.depth, self.panel.trades, self.panel.daily


@runtime_checkable
class DepthEvent(Protocol):
    """Minimal shape of a live depth snapshot pushed by a broker websocket."""

    scrip_code: int
    ts_ns: int
    bid_px: list[float]
    bid_qty: list[float]
    ask_px: list[float]
    ask_qty: list[float]


class BrokerFeed:
    """Interface for a live broker websocket feed (Kite/Dhan/Fyers).

    This is a documented stub, not a working client -- the offline pipeline never
    calls it. A real implementation would:

    * subscribe to L1 + 5-level depth for the screened universe plus index/VIX,
    * stamp each message with a **local-receipt** ``ts_ns`` (never the vendor clock),
    * enforce the ≤10 orders/second SEBI ceiling on the *order* path (§8.1),
    * assert snapshot continuity and flatten on any gap (§8.4),

    and yield the *same* canonical dicts the offline replayer produces, so a single
    feature engine serves both paths (§8.3 parity).
    """

    def __init__(self, scrip_codes: Iterable[int]) -> None:
        self.scrip_codes = list(scrip_codes)

    def stream(self) -> Iterator[DepthEvent]:  # pragma: no cover - live only
        raise NotImplementedError(
            "BrokerFeed is a live-only interface; use ParquetLoader/SyntheticLoader offline."
        )


__all__ = [
    "ParquetLoader",
    "SyntheticLoader",
    "BrokerFeed",
    "panel_to_parquet",
    "BAR_SCHEMA",
]
