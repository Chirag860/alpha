"""Phase 1 tests: vendor adapter, corporate actions, data hygiene (§1.4)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from bsealpha.data import (
    VendorSpec,
    adjust_prices,
    adjusted_returns,
    cumulative_factor,
    load_vendor_daily,
    load_vendor_depth,
    minute_bars_to_grid,
    run_hygiene,
)
from bsealpha.data.corporate_actions import CORP_ACTION_SCHEMA
from bsealpha.data.hygiene import circuit_band_violations, survivorship_report


# --------------------------------------------------------------- corporate actions
def _split_bars() -> pl.DataFrame:
    """Two sessions of one name; a 1:10 split on day 2 makes day-2 prices 1/10 of day-1."""
    d1 = dt.date(2026, 1, 5)
    d2 = dt.date(2026, 1, 6)
    rows = []
    for m in range(5):
        rows.append({"scrip_code": 500001, "date": d1, "minute": m, "close": 1000.0 + m})
    for m in range(5):
        rows.append({"scrip_code": 500001, "date": d2, "minute": m, "close": 100.0 + m * 0.1})
    return pl.DataFrame(rows)


def _split_action() -> pl.DataFrame:
    return pl.DataFrame(
        {"scrip_code": [500001], "ex_date": [dt.date(2026, 1, 6)],
         "action_type": ["split"], "price_ratio": [0.1], "qty_ratio": [10.0]},
        schema_overrides=CORP_ACTION_SCHEMA,
    )


def test_cumulative_factor_removes_jump():
    ex = np.array([dt.date(2026, 1, 6)])
    r = np.array([0.1])
    bar_dates = np.array([dt.date(2026, 1, 5), dt.date(2026, 1, 6)])
    f = cumulative_factor(ex, r, bar_dates)
    # day before the split is scaled by 0.1 onto the post-split grid; ex-day onward = 1.0
    assert f[0] == pytest.approx(0.1)
    assert f[1] == pytest.approx(1.0)


def test_adjust_prices_no_return_exceeds_band():
    bars = _split_bars()
    adj = adjust_prices(bars, _split_action(), price_cols=("close",))
    adj = adjusted_returns(adj, price_col="adj_close")
    # the raw -90% ex-date jump must vanish after adjustment
    ret = adj.filter(pl.col("adj_ret").is_not_null())["adj_ret"].abs().max()
    assert ret < 0.10                    # well within any circuit band
    # raw close still present and unadjusted (levels use as-traded price)
    assert "close" in adj.columns
    assert adj.filter(pl.col("date") == dt.date(2026, 1, 5))["close"].max() > 900


def test_point_in_time_ignores_future_action():
    """PIT mode on an as-of date before the ex-date must not apply the split (§1.4)."""
    bars = _split_bars()
    adj = adjust_prices(bars, _split_action(), price_cols=("close",),
                        asof=dt.date(2026, 1, 5), point_in_time=True)
    # no future action known on day 1 => factor 1 everywhere => adj_close == close
    assert (adj["adj_close"] == adj["close"]).all()


# --------------------------------------------------------------- hygiene
def test_hygiene_flags_unadjusted_split():
    bars = _split_bars()
    bars = adjusted_returns(bars.with_columns(pl.col("close").alias("adj_close")),
                            price_col="close").rename({"adj_ret": "ret"})
    viol = circuit_band_violations(bars, {500001: 20.0}, ret_col="ret")
    assert viol.height >= 1              # the -90% jump is flagged
    # after adjustment it disappears
    adj = adjusted_returns(adjust_prices(bars, _split_action(), price_cols=("close",)),
                           price_col="adj_close")
    viol2 = circuit_band_violations(adj, {500001: 20.0}, ret_col="adj_ret")
    assert viol2.height == 0


def test_survivorship_detects_delisting():
    d = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(4)]
    rows = []
    for day in d:
        rows.append({"date": day, "scrip_code": 1})       # survives all days
        if day < d[-1]:
            rows.append({"date": day, "scrip_code": 2})    # disappears before last day
    daily = pl.DataFrame(rows)
    assert survivorship_report(daily) == 1


def test_run_hygiene_report():
    bars = _split_bars()
    adj = adjusted_returns(adjust_prices(bars, _split_action(), price_cols=("close",)),
                           price_col="adj_close")
    daily = pl.DataFrame({"date": [dt.date(2026, 1, 6)], "scrip_code": [500001]})
    rep = run_hygiene(adj, daily, {500001: 20.0})
    assert rep.circuit_violations == 0
    assert rep.n_names == 1
    assert "rows=" in rep.summary()


# --------------------------------------------------------------- vendor adapter
def test_minute_bars_adapter(tmp_path):
    """A generic vendor CSV maps into the common-grid schema (§10)."""
    csv = tmp_path / "bars.csv"
    csv.write_text(
        "timestamp,symbol,open,high,low,close,value,trades\n"
        "2026-01-05 09:16:00,ACME,100.0,100.5,99.8,100.2,1000000,50\n"
        "2026-01-05 09:17:00,ACME,100.2,100.6,100.0,100.4,1200000,60\n"
    )
    spec = VendorSpec(
        column_map={"symbol": "symbol", "open": "open", "high": "high", "low": "low",
                    "close": "close", "turnover": "value", "n_trades": "trades"},
        timestamp_col="timestamp", timestamp_fmt="%Y-%m-%d %H:%M:%S",
        symbol_to_scrip={"ACME": 500999},
    )
    grid = minute_bars_to_grid(csv, spec)
    assert grid.height == 2
    assert grid["scrip_code"][0] == 500999
    assert grid["minute"].to_list() == [1, 2]        # 09:16, 09:17 => session-min 1, 2
    assert (grid["mid"] == grid["close"]).all()      # reduced fidelity without depth


def test_vendor_daily_defaults(tmp_path):
    csv = tmp_path / "eod.csv"
    csv.write_text("date,symbol,close,bse_turnover\n2026-01-05,ACME,100.2,50000000\n")
    spec = VendorSpec(column_map={"symbol": "symbol", "close": "close",
                                  "bse_turnover": "bse_turnover"},
                      symbol_to_scrip={"ACME": 500999})
    daily = load_vendor_daily(csv, spec)
    # missing surveillance flags filled with safe defaults so the screen can run
    assert "t2t_flag" in daily.columns and not daily["t2t_flag"][0]
    assert daily["series"][0] == "A"
