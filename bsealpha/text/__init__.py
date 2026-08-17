"""Text layer: point-in-time corporate-announcement ingest (Phase 1).

The alpha thesis (see the news-signal design note): a disclosure agent turns each
BSE filing into a small numeric feature panel that the *existing* validated model
may consume -- the LLM never sizes a trade, it only produces a feature that must
survive :mod:`bsealpha.validation` (OOF IC, CPCV 5th-percentile Sharpe, leak
tripwires). None of that matters until the raw text arrives with a **trustworthy
disclosure timestamp**, which is what this phase delivers and audits.

Everything keys on ``scrip_code`` (an int), exactly like the price panel, and the
disclosure instant is stored as an **absolute** integer-nanosecond epoch stamp to
keep the same "integers, never floats/datetimes" discipline that the price schema
uses to foreclose look-ahead (§ data.schema, §3.3).

Phase 1 is data engineering only -- no LLM, no modeling. Phases 2-5 (extract /
align / judge / backtest) build on this.
"""

from __future__ import annotations

from .schema import ANNOUNCEMENT_SCHEMA, empty_announcements, validate_announcements
from .loaders import (
    BseAnnouncementsClient,
    ParquetAnnouncementLoader,
    announcements_to_parquet,
    synth_announcements,
)
from .audit import AnnouncementAudit, audit_announcements
from .extract import (
    AnnouncementExtractor,
    ExtractionCache,
    sanitize_text,
    stub_extract_frame,
)
from .align import (
    NEWS_FEATURES,
    align_announcement_features,
    attach_scores,
    synthetic_grid,
)
from .evaluate_feature import (
    NewsEvalResult,
    add_news_features,
    evaluate_news_feature,
)

__all__ = [
    "ANNOUNCEMENT_SCHEMA",
    "empty_announcements",
    "validate_announcements",
    "BseAnnouncementsClient",
    "ParquetAnnouncementLoader",
    "announcements_to_parquet",
    "synth_announcements",
    "AnnouncementAudit",
    "audit_announcements",
    # Phase 2: LLM extractor
    "AnnouncementExtractor",
    "ExtractionCache",
    "sanitize_text",
    "stub_extract_frame",
    # Phase 3: point-in-time alignment onto the grid
    "NEWS_FEATURES",
    "align_announcement_features",
    "attach_scores",
    "synthetic_grid",
    # Phase 4: with/without validation verdict
    "NewsEvalResult",
    "add_news_features",
    "evaluate_news_feature",
]
