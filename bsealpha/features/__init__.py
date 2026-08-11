"""Feature engine and its component feature families."""

from __future__ import annotations

from .cross_sectional import cross_sectional_rank
from .engine import (
    RAW_MICRO_COLS,
    build_features,
    build_features_bars_only,
    build_raw_grid,
    finalize_features,
)
from .streaming import ScripState, StreamingFeatureEngine
from .index_factor import attach_factors, compute_returns
from .ofi import OFI5, IntegratedOFI, ofi_frame
from .residualize import add_residual_vol, fit_betas, fit_betas_trailing, residualize
from .tradeflow import lee_ready_sign, sign_trades, tick_rule_sign
from .volatility import TodVolProfile, bipower_variation, jump_component, realized_variance

__all__ = [
    "build_features",
    "build_features_bars_only",
    "build_raw_grid",
    "finalize_features",
    "RAW_MICRO_COLS",
    "StreamingFeatureEngine",
    "ScripState",
    "OFI5",
    "IntegratedOFI",
    "ofi_frame",
    "compute_returns",
    "attach_factors",
    "residualize",
    "fit_betas",
    "fit_betas_trailing",
    "add_residual_vol",
    "cross_sectional_rank",
    "tick_rule_sign",
    "lee_ready_sign",
    "sign_trades",
    "TodVolProfile",
    "bipower_variation",
    "realized_variance",
    "jump_component",
]
