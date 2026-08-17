"""Phase 4: the verdict -- does the news feature actually add out-of-fold edge?

This is the whole project's go/no-go, and it reuses the user's *own* validation machine
(:func:`bsealpha.validation.evaluate` -- OOF IC, CPCV 5th-percentile Sharpe, leak
tripwires) rather than inventing a new scorecard. The test is brutally simple:

1. build the labeled cross-sectional panel;
2. score + align the announcements into :data:`NEWS_FEATURES` on that exact grid;
3. run ``evaluate`` **without** the news columns (baseline) and **with** them (augmented);
4. compare -- ship only if augmented raises ``cpcv_sharpe_5pct`` **and** ``ic`` with **no new
   leak tripwire** and PBO no worse.

On the synthetic panel the extractor is the *no-signal stub*, so the honest expected result
is **no improvement** -- that is a passing harness, not a disappointment. Real edge can only
appear when this same function is pointed at real announcements + real returns with the live
Claude extractor (pass ``score_fn=AnnouncementExtractor(...).extract_frame``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import polars as pl

from ..config import Config
from ..pipeline import build_labeled_panel
from ..validation import evaluate
from ..validation.runner import ValidationReport
from .align import NEWS_FEATURES, align_announcement_features, attach_scores
from .extract import stub_extract_frame
from .loaders import synth_announcements

# score_fn: announcement panel -> DataFrame(ann_id, direction, materiality, surprise, novelty)
ScoreFn = Callable[[pl.DataFrame], pl.DataFrame]


def _tripwire_key(msg: str) -> str:
    """Category of a tripwire message: the text with numeric tokens removed."""
    return re.sub(r"[-+]?\d*\.?\d+", "#", msg)


@dataclass
class NewsEvalResult:
    """Baseline vs news-augmented validation, with the deltas the decision turns on."""

    baseline: ValidationReport
    augmented: ValidationReport
    news_cols: list[str]
    news_coverage: float          # fraction of panel rows carrying live news

    @property
    def d_ic(self) -> float:
        return self.augmented.ic - self.baseline.ic

    @property
    def d_cpcv_5pct(self) -> float:
        return self.augmented.cpcv_sharpe_5pct - self.baseline.cpcv_sharpe_5pct

    @property
    def new_tripwires(self) -> list[str]:
        """Tripwires the news feature *introduced* -- compared by category, not exact text.

        A tripwire string embeds the metric's value ("meta-AUC 0.672 > 0.62"), so a
        pre-existing tripwire whose number merely wobbles between the two runs would look
        "new" under a string diff. We key on the category (the text with numbers stripped)
        so only a genuinely new *kind* of leak counts.
        """
        base_keys = {_tripwire_key(t) for t in self.baseline.tripwires}
        return [t for t in self.augmented.tripwires if _tripwire_key(t) not in base_keys]

    @property
    def adds_edge(self) -> bool:
        """The ship rule: OOF IC *and* 5th-pct CPCV Sharpe both up, no new leak tripwire."""
        return (self.d_ic > 0 and self.d_cpcv_5pct > 0
                and not self.new_tripwires
                and self.augmented.pbo <= self.baseline.pbo + 1e-9)

    def report(self) -> str:
        L = [
            "=== Phase 4: news-feature validation (with vs without) ===",
            f"news features : {', '.join(self.news_cols)}",
            f"panel coverage: {self.news_coverage:.1%} of rows carry live news",
            f"{'metric':<26}{'baseline':>12}{'+news':>12}{'delta':>12}",
            f"{'OOF IC':<26}{self.baseline.ic:>12.4f}{self.augmented.ic:>12.4f}{self.d_ic:>+12.4f}",
            f"{'CPCV Sharpe 5th-pct':<26}{self.baseline.cpcv_sharpe_5pct:>12.3f}"
            f"{self.augmented.cpcv_sharpe_5pct:>12.3f}{self.d_cpcv_5pct:>+12.3f}",
            f"{'meta-AUC':<26}{self.baseline.meta_auc:>12.3f}{self.augmented.meta_auc:>12.3f}"
            f"{self.augmented.meta_auc - self.baseline.meta_auc:>+12.3f}",
            f"{'PBO':<26}{self.baseline.pbo:>12.3f}{self.augmented.pbo:>12.3f}"
            f"{self.augmented.pbo - self.baseline.pbo:>+12.3f}",
        ]
        if self.new_tripwires:
            L.append("NEW TRIPWIRES (suspect leak):")
            L += [f"  ! {t}" for t in self.new_tripwires]
        verdict = "ADDS EDGE — ship" if self.adds_edge else "NO EDGE — do not ship"
        L.append(f"verdict: {verdict}")
        return "\n".join(L)


def add_news_features(labels: pl.DataFrame, cfg: Config, *,
                      score_fn: ScoreFn = stub_extract_frame,
                      ann_seed: int = 0, rate_per_name_day: float = 0.12,
                      lag_minutes: int = 1) -> tuple[pl.DataFrame, list[str], float]:
    """Score + align synthetic announcements onto ``labels``' grid; left-join the features.

    Returns ``(labels_with_news, NEWS_FEATURES, coverage)``. Announcements are generated over
    the panel's own scrips/dates so the alignment lands on real cells; the join is 1:1
    because the panel has one row per ``(scrip_code, date, minute)``.
    """
    scrips = labels["scrip_code"].unique().to_list()
    dates = labels["date"].unique().sort().to_list()
    anns = synth_announcements(scrips, dates, seed=ann_seed, rate_per_name_day=rate_per_name_day)
    scored = attach_scores(anns, score_fn(anns))

    grid = labels.select(["scrip_code", "date", "minute"]).unique()
    aligned, cols = align_announcement_features(grid, scored, lag_minutes=lag_minutes)

    out = labels.join(
        aligned.select(["scrip_code", "date", "minute", *cols]),
        on=["scrip_code", "date", "minute"], how="left",
    ).with_columns([pl.col(c).fill_null(0.0) for c in cols])
    coverage = float(out["has_news"].mean()) if out.height else 0.0
    return out, cols, coverage


def evaluate_news_feature(cfg: Config, *, seed: int | None = None, run_cpcv: bool = True,
                          score_fn: ScoreFn = stub_extract_frame,
                          ann_seed: int = 0) -> NewsEvalResult:
    """Run the baseline vs news-augmented validation and return the comparison.

    ``score_fn`` defaults to the offline no-signal stub (synthetic go/no-go); pass a live
    extractor's ``extract_frame`` to test real announcements against real returns.
    """
    labels, feature_cols, _panel, _n = build_labeled_panel(cfg, seed=seed)
    baseline = evaluate(labels, feature_cols, cfg, run_cpcv=run_cpcv)

    labels_news, news_cols, coverage = add_news_features(
        labels, cfg, score_fn=score_fn, ann_seed=ann_seed)
    augmented = evaluate(labels_news, feature_cols + news_cols, cfg, run_cpcv=run_cpcv)

    return NewsEvalResult(baseline=baseline, augmented=augmented,
                          news_cols=news_cols, news_coverage=coverage)
