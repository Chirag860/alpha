"""Tests for the Phase-2 extractor: identity stripping, stub determinism, cache, ranges.

All offline (no Anthropic API). The live path is a thin cached wrapper over the same
schema exercised here; what must be pinned is the leak-control (identity stripping) and
the reproducibility (deterministic cache), both of which are testable without a network.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from bsealpha.text import (
    AnnouncementExtractor,
    ExtractionCache,
    sanitize_text,
    stub_extract_frame,
    synth_announcements,
)

_SCORES = ("direction", "materiality", "surprise", "novelty")


def test_sanitize_strips_company_and_scrip():
    # BSE-style "Name - scrip - subject": identity must be gone, subject kept
    out = sanitize_text("Jindal Poly Films Ltd - 500227 - Board Meeting Outcome", "", 500227)
    assert "500227" not in out
    assert "Jindal" not in out
    assert "Board Meeting Outcome" in out


def test_sanitize_removes_bare_scrip_anywhere():
    out = sanitize_text("Result: revenue up for scrip 500001", "", 500001)
    assert "500001" not in out
    assert "revenue up" in out


def test_stub_is_deterministic_and_identity_free():
    df = synth_announcements([500001, 500002], [dt.date(2024, 1, 2)], seed=5,
                             rate_per_name_day=1.0, dup_frac=0.0)
    a = stub_extract_frame(df, seed=1)
    b = stub_extract_frame(df, seed=1)
    assert a.equals(b)  # same input + seed => identical scores (reproducible feature)
    # different seed generally differs
    c = stub_extract_frame(df, seed=2)
    assert not a.equals(c)


def test_stub_scores_in_range_and_aligned():
    df = synth_announcements(list(range(500001, 500011)),
                             [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(20)],
                             seed=9)
    scores = stub_extract_frame(df)
    assert set(scores["ann_id"].to_list()) == set(df["ann_id"].to_list())
    assert (scores["direction"] >= -1.0).all() and (scores["direction"] <= 1.0).all()
    for c in ("materiality", "surprise", "novelty"):
        assert (scores[c] >= 0.0).all() and (scores[c] <= 1.0).all()


def test_extraction_cache_roundtrip(tmp_path):
    cache = ExtractionCache(tmp_path / "c")
    assert cache.get("k") is None
    payload = {"direction": 0.3, "materiality": 0.2, "surprise": 0.1, "novelty": 0.4}
    cache.put("k", payload)
    assert cache.get("k") == payload


def test_extractor_uses_cache_without_api(tmp_path):
    # pre-seed the cache with the exact key the extractor will compute, then confirm
    # extract_frame returns those numbers without ever constructing an API client.
    from bsealpha.text.extract import _cache_key, DEFAULT_MODEL

    df = synth_announcements([500001], [dt.date(2024, 1, 2)], seed=1,
                             rate_per_name_day=1.0, dup_frac=0.0)
    row = df.row(0, named=True)
    sanitized = sanitize_text(row["headline"], row["body"], row["scrip_code"])
    key = _cache_key(DEFAULT_MODEL, sanitized)

    ext = AnnouncementExtractor(tmp_path / "cache")
    ext.cache.put(key, {"direction": -0.5, "materiality": 0.9,
                        "surprise": 0.8, "novelty": 0.7})
    out = ext.extract_frame(df)  # must hit cache; _api() would raise if anthropic missing
    assert out.height == 1
    assert out.row(0, named=True)["materiality"] == 0.9
    assert ext._client is None  # never touched the network
