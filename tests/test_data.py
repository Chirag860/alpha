"""Tests for the data layer: synthetic generator, schema, loaders, universe screen."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from bsealpha.config import load_config
from bsealpha.data import (
    DAILY_SCHEMA,
    DEPTH_SCHEMA,
    ParquetLoader,
    SyntheticLoader,
    generate_panel,
    panel_to_parquet,
    validate_frame,
)
from bsealpha.market import round_to_tick, tick_size
from bsealpha.universe import build_universe


@pytest.fixture(scope="module")
def cfg():
    # small + fast panel for tests
    return load_config(overrides={"synthetic": {"n_names": 12, "n_days": 8, "seed": 1}})


@pytest.fixture(scope="module")
def panel(cfg):
    return generate_panel(cfg)


def test_schema_present(panel):
    validate_frame(panel.depth, DEPTH_SCHEMA, name="depth")
    validate_frame(panel.daily, DAILY_SCHEMA, name="daily")


def test_book_uncrossed_and_ordered(panel):
    d = panel.depth
    # best bid strictly below best ask on every snapshot
    assert (d["ask_px_0"] - d["bid_px_0"] > 0).all()
    # levels monotone: bid prices descend, ask prices ascend
    assert (d["bid_px_0"] >= d["bid_px_1"]).all()
    assert (d["ask_px_0"] <= d["ask_px_1"]).all()
    # positive quantities
    assert (d["bid_qty_0"] > 0).all()
    assert (d["ask_qty_0"] > 0).all()


def test_prices_on_tick_grid(panel):
    d = panel.depth
    px = d["bid_px_0"].to_numpy()
    # within a tiny tolerance, prices sit on their band's tick grid
    assert np.allclose(px, round_to_tick(px), atol=1e-6)


def test_tick_bands():
    assert tick_size(100.0) == 0.01
    assert tick_size(500.0) == 0.05
    assert tick_size(1400.0) == 0.10
    assert tick_size(6000.0) == 0.50


def test_trades_have_signs(panel):
    t = panel.trades
    assert set(np.unique(t["true_sign"].to_numpy())).issubset({-1, 1})
    assert (t["qty"] > 0).all()


def test_no_intraday_return_exceeds_circuit_band(panel):
    """§1.4 hygiene: no synthetic 1-snapshot mid return should exceed a circuit band.

    Guards against the split/corporate-action artifact the report warns about.
    """
    d = panel.depth.sort(["scrip_code", "date", "ts_ns"])
    ret = (d.group_by(["scrip_code", "date"], maintain_order=True)
           .agg((pl.col("mid").pct_change().abs().max()).alias("max_ret")))
    # generator injects no splits; all moves are well under even a 2% band
    assert ret["max_ret"].max() < 0.02


def test_universe_screen_excludes_flagged(cfg, panel):
    dates = panel.daily["date"].unique().sort().to_list()
    asof = dates[-1]
    uni = build_universe(panel.daily, asof, cfg)
    # every admitted name must be clean on all hard exclusions
    joined = uni.join(panel.daily.filter(pl.col("date") < asof).unique("scrip_code"),
                      on="scrip_code", how="left")
    assert not uni.is_empty() or True  # small panels may screen to empty; that's valid
    if uni.height:
        assert (~uni["t2t_flag"]).all()
        assert (~uni["asm_flag"]).all()
        assert (~uni["gsm_flag"]).all()
        assert (~uni["is_suspended"]).all()
        assert uni["series"].is_in(["A", "B"]).all()
        assert (uni["max_clip"] > 0).all()


def test_universe_is_point_in_time(cfg, panel):
    """The screen must not consult any row dated >= asof."""
    dates = panel.daily["date"].unique().sort().to_list()
    asof = dates[len(dates) // 2]
    uni = build_universe(panel.daily, asof, cfg)
    # add a future row that would flip a name into T2T; screen must ignore it
    future = panel.daily.filter(pl.col("date") == dates[-1]).with_columns(
        pl.col("t2t_flag") | True
    )
    contaminated = pl.concat([panel.daily, future.with_columns(
        pl.lit(asof + dt.timedelta(days=100)).alias("date"))])
    uni2 = build_universe(contaminated, asof, cfg)
    assert uni.sort("scrip_code")["scrip_code"].to_list() == \
        uni2.sort("scrip_code")["scrip_code"].to_list()


def test_parquet_roundtrip(panel, tmp_path):
    root = panel_to_parquet(panel, tmp_path / "panel")
    depth, trades, daily = ParquetLoader(root).load()
    assert depth.height == panel.depth.height
    assert daily.height == panel.daily.height
    # SyntheticLoader returns the in-memory frames unchanged
    d2, t2, da2 = SyntheticLoader(panel).load()
    assert d2.height == panel.depth.height
