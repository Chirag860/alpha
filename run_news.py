#!/usr/bin/env python3
"""Announcement ingest runner (Phase 1 of the news-signal build) — fetch, cache, audit.

    python3 run_news.py --synthetic                         # offline demo: synth panel + audit
    python3 run_news.py --fetch --start 2024-01-01 --end 2024-03-29 \
                        --universe data/universe_scrips.txt  # live BSE pull -> parquet + audit
    python3 run_news.py --audit data/news/announcements.parquet   # re-audit a saved panel

Phase 1 is data engineering only: pull BSE corporate announcements with trustworthy
disclosure timestamps, persist them, and print the coverage/timestamp audit that gates
whether Phases 2-5 (LLM extract -> align -> validate -> backtest) are worth building.

--fetch caches raw JSON per day under <out-dir>/raw/ (an immutable point-in-time archive)
and needs network access to api.bseindia.com. --synthetic and --audit run anywhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib

os.environ.setdefault("POLARS_MAX_THREADS", "1")


def _load_universe(path: str | None) -> set[int] | None:
    """Read a universe of scrip codes (one int per line) if a path is given."""
    if not path:
        return None
    text = pathlib.Path(path).read_text().split()
    return {int(x) for x in text if x.strip().isdigit()}


def _run_audit(df, universe: set[int] | None) -> int:
    from bsealpha.text import audit_announcements
    print(audit_announcements(df, universe=universe).report())
    return 0


def _print_score_summary(scores) -> None:
    import polars as pl
    stats = scores.select([
        pl.col(c).mean().alias(f"{c}_mean") for c in
        ("direction", "materiality", "surprise", "novelty")
    ])
    print("Extracted score means:", {k: round(v, 3) for k, v in stats.row(0, named=True).items()})


def do_synthetic(universe_path: str | None, extract: bool) -> int:
    from bsealpha.text import synth_announcements, stub_extract_frame
    # a small self-contained universe + one quarter of trading days
    scrips = list(range(500001, 500041))
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(90)]
    df = synth_announcements(scrips, dates, seed=7)
    print(f"Synthetic announcements: {df.height} rows over {len(scrips)} scrips.\n")
    uni = _load_universe(universe_path) or set(scrips)
    rc = _run_audit(df, uni)
    if extract:
        print()
        from bsealpha.text import (attach_scores, stub_extract_frame,
                                   synthetic_grid, align_announcement_features)
        scores = stub_extract_frame(df)  # offline deterministic stub (no API)
        print(f"Stub-extracted {scores.height} rows (offline, no API call).")
        _print_score_summary(scores)
        # Phase 3: align onto a synthetic grid and report sparsity (point-in-time)
        scored = attach_scores(df, scores)
        grid = synthetic_grid(scrips, dates, step_min=15)
        aligned, cols = align_announcement_features(grid, scored)
        import polars as pl
        frac = float(aligned["has_news"].mean())
        print(f"Aligned onto {aligned.height:,} grid cells; "
              f"{frac:.1%} carry live news (features: {', '.join(cols)}).")
    return rc


def do_phase4(n_names: int, n_days: int, seed: int, run_cpcv: bool) -> int:
    from bsealpha.config import load_config
    from bsealpha.text import evaluate_news_feature
    cfg = load_config(overrides={"synthetic": {"n_names": n_names, "n_days": n_days,
                                               "seed": seed}})
    print(f"Phase 4 verdict on synthetic panel ({n_names} names x {n_days} days, "
          f"stub extractor = no-signal null)...\n")
    res = evaluate_news_feature(cfg, seed=seed, run_cpcv=run_cpcv)
    print(res.report())
    return 0


def do_extract_file(path: str, out_dir: str, model: str) -> int:
    from bsealpha.text import ParquetAnnouncementLoader, AnnouncementExtractor
    import polars as pl
    anns = ParquetAnnouncementLoader(path).load()
    out = pathlib.Path(out_dir)
    extractor = AnnouncementExtractor(out / "extract_cache", model=model)
    print(f"Extracting {anns.height} announcements with {model} (cached)...")
    scores = extractor.extract_frame(anns, progress=True)
    scores.write_parquet(out / "scores.parquet")
    print(f"Wrote {scores.height} scores -> {out / 'scores.parquet'}")
    _print_score_summary(scores)
    return 0


def do_fetch(start: str, end: str, out_dir: str, universe_path: str | None) -> int:
    from bsealpha.text import BseAnnouncementsClient, announcements_to_parquet
    out = pathlib.Path(out_dir)
    universe = _load_universe(universe_path)
    client = BseAnnouncementsClient(out / "raw")
    df = client.fetch(dt.date.fromisoformat(start), dt.date.fromisoformat(end),
                      scrip_codes=universe)
    parquet = announcements_to_parquet(df, out / "announcements.parquet")
    print(f"Fetched {df.height} announcements -> {parquet}\n")
    return _run_audit(df, universe)


def do_audit(path: str, universe_path: str | None) -> int:
    from bsealpha.text import ParquetAnnouncementLoader
    df = ParquetAnnouncementLoader(path).load()
    return _run_audit(df, _load_universe(universe_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="offline synth panel + audit (no network)")
    ap.add_argument("--fetch", action="store_true", help="live BSE pull for --start..--end")
    ap.add_argument("--audit", metavar="PARQUET", default=None, help="re-audit a saved panel")
    ap.add_argument("--start", default=None, help="fetch start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="fetch end date YYYY-MM-DD")
    ap.add_argument("--out-dir", default="data/news", help="output dir for --fetch/--extract-file")
    ap.add_argument("--universe", default=None, help="file of scrip codes (one per line)")
    ap.add_argument("--extract", action="store_true",
                    help="with --synthetic: run the offline stub extractor + score summary")
    ap.add_argument("--extract-file", metavar="PARQUET", default=None,
                    help="run the LIVE Claude extractor over a saved panel (needs API)")
    ap.add_argument("--model", default="claude-opus-5", help="model for --extract-file")
    ap.add_argument("--phase4", action="store_true",
                    help="run the with/without validation verdict on a synthetic panel")
    ap.add_argument("--n-names", type=int, default=40, help="synthetic universe size (--phase4)")
    ap.add_argument("--n-days", type=int, default=40, help="synthetic session count (--phase4)")
    ap.add_argument("--seed", type=int, default=1, help="synthetic seed (--phase4)")
    ap.add_argument("--fast", action="store_true", help="skip the CPCV sweep (--phase4)")
    args = ap.parse_args()

    if args.phase4:
        return do_phase4(args.n_names, args.n_days, args.seed, run_cpcv=not args.fast)
    if args.synthetic:
        return do_synthetic(args.universe, args.extract)
    if args.audit:
        return do_audit(args.audit, args.universe)
    if args.extract_file:
        return do_extract_file(args.extract_file, args.out_dir, args.model)
    if args.fetch:
        if not (args.start and args.end):
            ap.error("--fetch requires --start and --end")
        return do_fetch(args.start, args.end, args.out_dir, args.universe)
    ap.error("choose one of --synthetic / --fetch / --audit / --extract-file")


if __name__ == "__main__":
    raise SystemExit(main())
