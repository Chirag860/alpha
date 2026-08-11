"""Data layer: canonical schema, synthetic panel generator, loaders, and real-data ingest."""

from __future__ import annotations

from .corporate_actions import (
    CORP_ACTION_SCHEMA,
    adjust_prices,
    adjusted_returns,
    cumulative_factor,
)
from .free_data import DEFAULT_UNIVERSE, load_yfinance_panel
from .mt5_data import GRID_COLUMNS, META_COLUMNS, load_mt5_panel
from .hygiene import HygieneReport, run_hygiene
from .loaders import BrokerFeed, ParquetLoader, SyntheticLoader, panel_to_parquet
from .vendor import (
    TRUEDATA_BARS_SPEC,
    VendorSpec,
    load_vendor_daily,
    load_vendor_depth,
    minute_bars_to_grid,
)
from .schema import (
    BAR_SCHEMA,
    DAILY_SCHEMA,
    DEPTH_SCHEMA,
    N_DEPTH_LEVELS,
    TRADE_SCHEMA,
    empty_frame,
    validate_frame,
)
from .synthetic import SyntheticPanel, generate_panel

__all__ = [
    "BAR_SCHEMA",
    "DAILY_SCHEMA",
    "DEPTH_SCHEMA",
    "TRADE_SCHEMA",
    "N_DEPTH_LEVELS",
    "empty_frame",
    "validate_frame",
    "SyntheticPanel",
    "generate_panel",
    "ParquetLoader",
    "SyntheticLoader",
    "BrokerFeed",
    "panel_to_parquet",
    # real-data ingest (Phase 1)
    "VendorSpec",
    "TRUEDATA_BARS_SPEC",
    "minute_bars_to_grid",
    "load_vendor_depth",
    "load_vendor_daily",
    "CORP_ACTION_SCHEMA",
    "cumulative_factor",
    "adjust_prices",
    "adjusted_returns",
    "HygieneReport",
    "run_hygiene",
    "load_yfinance_panel",
    "DEFAULT_UNIVERSE",
    "load_mt5_panel",
    "GRID_COLUMNS",
    "META_COLUMNS",
]
