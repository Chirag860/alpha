"""Market-profile refactor tests: BSE stays the default; US-equity is selectable.

Critical invariant: switching the active profile must not leak into other tests, so every
test here restores BSE via the ``restore_bse`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from bsealpha import market
from bsealpha.config import load_config


@pytest.fixture(autouse=True)
def restore_bse():
    yield
    market.set_active_profile("bse")


def test_default_profile_is_bse():
    assert market.active_profile().name == "bse"
    assert market.session_open_min() == 555        # 09:15 IST
    assert market.session_len_min() == 375
    assert market.flatten_session_min() == 360     # 15:15 session-relative


def test_us_profile_values():
    market.set_active_profile("us_equity")
    assert market.session_open_min() == 570        # 09:30 ET
    assert market.session_close_min() == 960       # 16:00 ET
    assert market.session_len_min() == 390
    assert market.flatten_session_min() == 385     # 15:55 session-relative


def test_tick_size_is_profile_dependent():
    # BSE: price-banded
    assert market.tick_size(123.45) == 0.01
    assert market.tick_size(600.0) == 0.05
    assert market.tick_size(30000.0) == 5.00
    # US: flat penny tick regardless of price
    market.set_active_profile("us_equity")
    assert market.tick_size(123.45) == 0.01
    assert market.tick_size(30000.0) == 0.01
    arr = market.tick_size(np.array([50.0, 5000.0]))
    assert np.allclose(arr, [0.01, 0.01])


def test_session_helpers_track_profile():
    market.set_active_profile("us_equity")
    # 09:30 ET open -> session_minute 0; 15:55 -> minutes_to_flatten 0 at the deadline
    assert market.session_minute(570) == 0.0
    assert market.is_tradable_minute(571) and not market.is_tradable_minute(956)
    assert market.minutes_to_close(960) == 0.0


def test_set_from_config():
    cfg = load_config(overrides={"market": {"profile": "us_equity"}})
    market.set_active_profile_from_config(cfg)
    assert market.active_profile().name == "us_equity"
    # a config without a market block defaults to BSE
    market.set_active_profile_from_config(load_config())
    assert market.active_profile().name == "bse"


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        market.set_active_profile("forex_ecn")
