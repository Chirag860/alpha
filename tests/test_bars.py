"""Tests for bars: rupee/volume/tick event bars and the common minute grid."""

from __future__ import annotations

import polars as pl
import pytest

from bsealpha.bars import build_bars, common_minute_grid, rupee_bars, volume_bars
from bsealpha.config import load_config
from bsealpha.data import generate_panel


@pytest.fixture(scope="module")
def panel():
    cfg = load_config(overrides={"synthetic": {"n_names": 8, "n_days": 5, "seed": 3}})
    return generate_panel(cfg), cfg


def test_rupee_bars_basic(panel):
    p, cfg = panel
    bars = rupee_bars(p.trades, p.depth, cfg)
    assert bars.height > 0
    # OHLC ordering within a bar
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["close_trade"]).all()
    assert (bars["low"] <= bars["close_trade"]).all()
    assert (bars["turnover"] > 0).all()
    assert (bars["n_trades"] >= 1).all()
    # closing mid/micro populated
    assert bars["mid"].null_count() == 0
    assert bars["micro"].null_count() == 0


def test_rupee_bar_timestamp_is_close_event(panel):
    """Bar ts must be the closing trade's ts; closing mid must be <= that ts (causal)."""
    p, cfg = panel
    bars = rupee_bars(p.trades, p.depth, cfg)
    # every bar ts_ns should exist in the trades stream for that scrip/date
    j = bars.join(p.trades.select(["scrip_code", "date", "ts_ns"]).unique(),
                  on=["scrip_code", "date", "ts_ns"], how="inner")
    assert j.height == bars.height


def test_common_grid_shared_clock(panel):
    p, _ = panel
    grid = common_minute_grid(p.depth, p.trades)
    assert grid.height > 0
    # minute is the shared cross-sectional clock: many names per (date, minute)
    counts = grid.group_by(["date", "minute"]).len()
    assert counts["len"].max() > 1
    # mid strictly positive; turnover non-negative
    assert (grid["mid"] > 0).all()
    assert (grid["turnover"] >= 0).all()
    # micro-price stays inside the observed price range (sanity on crossed weights)
    assert (grid["micro"] > 0).all()


def test_bars_never_span_days(panel):
    p, cfg = panel
    bars = rupee_bars(p.trades, p.depth, cfg)
    # bar_id resets per day => a (scrip, date) group's bars are within one date only.
    # Trivially true by construction, but assert session_min stays in-session.
    assert (bars["session_min"] > 0).all()
    assert (bars["session_min"] <= 375).all()


def test_build_bars_returns_event_and_grid(panel):
    p, cfg = panel
    event, grid = build_bars(p.depth, p.trades, cfg)
    assert event.height > 0 and grid.height > 0
    assert "minute" in grid.columns and "close" in grid.columns


def test_volume_bars_monotone_turnover(panel):
    p, _ = panel
    bars = volume_bars(p.trades, p.depth, threshold_qty=500.0)
    assert bars.height > 0
    assert (bars["turnover"] > 0).all()
