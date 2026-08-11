"""Cross-sectional rank normalization (§3.2, §3.3).

Cross-sectional ranks are usually stronger inputs than raw levels here, and rank-
normalizing aggressively makes a ₹80 mid-cap and a ₹3,000 large-cap comparable.

**The leak this module exists to prevent (§3.3):** computing a cross-sectional rank at
minute ``t`` requires every name's value at ``t`` -- which a live system only has *after*
``t``. So we shift each feature by ``lag_minutes`` (per name) *before* ranking, enforcing
that the cross-section used at ``t`` is the one completed at ``t-lag``. This is enforced
in code, not by convention.
"""

from __future__ import annotations

import polars as pl


def cross_sectional_rank(df: pl.DataFrame, feature_cols: list[str], *,
                         by: tuple[str, ...] = ("date", "minute"),
                         sector_col: str = "sector",
                         lag_minutes: int = 1) -> pl.DataFrame:
    """Add ``{col}_xs`` (within cross-section) and ``{col}_xsec`` (within sector) ranks.

    Ranks are centered to ``[-0.5, 0.5]``. Each source feature is first shifted by
    ``lag_minutes`` within ``scrip_code`` so the rank at minute ``t`` uses only
    information available at ``t - lag_minutes`` (§3.3).
    """
    df = df.sort(["scrip_code", "date", "minute"])
    # lag the raw features (in place, name-wise) so ranks are built from t-lag
    lagged_names = {c: f"__lag_{c}" for c in feature_cols}
    df = df.with_columns([
        pl.col(c).shift(lag_minutes).over("scrip_code").alias(lagged_names[c])
        for c in feature_cols
    ])
    out_exprs: list[pl.Expr] = []
    for c in feature_cols:
        lc = lagged_names[c]
        n = pl.len().over(list(by))
        n_sec = pl.len().over(list(by) + [sector_col])
        out_exprs.append(
            ((pl.col(lc).rank("average").over(list(by)) - 0.5) / n - 0.5).alias(f"{c}_xs")
        )
        out_exprs.append(
            ((pl.col(lc).rank("average").over(list(by) + [sector_col]) - 0.5) / n_sec - 0.5)
            .alias(f"{c}_xsec")
        )
    df = df.with_columns(out_exprs)
    return df.drop(list(lagged_names.values()))
