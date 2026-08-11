"""Phase 3: offline/online feature parity (§8.3).

The mandatory CI test: replay the same session through the batch engine and the
event-driven streaming engine and assert every raw microstructure feature matches. Because
the cross-sectional/residual layer (`finalize_features`) is shared code applied to whichever
raw grid you have, matching the raw grid guarantees full-feature parity -- this is the
architecture that prevents training/serving skew.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bsealpha.config import load_config
from bsealpha.data import generate_panel
from bsealpha.features import (
    RAW_MICRO_COLS,
    StreamingFeatureEngine,
    build_raw_grid,
    finalize_features,
)


@pytest.fixture(scope="module")
def panels():
    cfg = load_config(overrides={"synthetic": {"n_names": 8, "n_days": 4, "seed": 4}})
    panel = generate_panel(cfg)
    batch = build_raw_grid(panel.depth, panel.trades, panel.meta, cfg)
    stream = StreamingFeatureEngine(cfg).run(panel.depth, panel.trades)
    return cfg, panel, batch, stream


def test_streaming_emits_same_rows(panels):
    _, _, batch, stream = panels
    keyb = batch.select(["scrip_code", "date", "minute"]).sort(["scrip_code", "date", "minute"])
    keys = stream.select(["scrip_code", "date", "minute"]).sort(["scrip_code", "date", "minute"])
    assert keyb.equals(keys)          # exact same (scrip, date, minute) grid


def test_raw_feature_parity(panels):
    """Every raw microstructure feature matches batch to 1e-6 (§8.3)."""
    _, _, batch, stream = panels
    keys = ["scrip_code", "date", "minute"]
    b = batch.select(keys + RAW_MICRO_COLS).sort(keys)
    s = stream.select(keys + RAW_MICRO_COLS).sort(keys)
    assert b.height == s.height
    mism = {}
    for col in RAW_MICRO_COLS:
        # null == null (e.g. warm-up rows for spread_rel / trade cols); else compare values
        bv = b[col].fill_null(0.0).fill_nan(0.0).to_numpy()
        sv = s[col].fill_null(0.0).fill_nan(0.0).to_numpy()
        if not np.allclose(bv, sv, rtol=1e-6, atol=1e-9):
            mism[col] = float(np.max(np.abs(bv - sv)))
    assert not mism, f"features diverged between batch and streaming: {mism}"


def test_full_feature_parity_after_finalize(panels):
    """Running the shared finalize layer on each raw grid yields identical features."""
    cfg, panel, batch, stream = panels
    fb, cols = finalize_features(batch, panel.meta, cfg)
    # feed the streaming raw grid (needs sector) through the SAME finalize code
    stream2 = stream.join(panel.meta.select(["scrip_code", "sector"]), on="scrip_code")
    fs, _ = finalize_features(stream2, panel.meta, cfg)
    keys = ["scrip_code", "date", "minute"]
    fb = fb.select(keys + cols).sort(keys)
    fs = fs.select(keys + cols).sort(keys)
    bad = {}
    for c in cols:
        bv = fb[c].fill_null(0.0).fill_nan(0.0).to_numpy()
        sv = fs[c].fill_null(0.0).fill_nan(0.0).to_numpy()
        if not np.allclose(bv, sv, rtol=1e-5, atol=1e-8):
            bad[c] = float(np.max(np.abs(bv - sv)))
    assert not bad, f"full features diverged after finalize: {bad}"
