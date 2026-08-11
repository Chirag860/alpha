"""Clustered feature importance (MDA) with purged CV (§5.6).

Microstructure features are heavily collinear, so shuffling one at a time understates
importance via substitution. We shuffle in **economic clusters** (order flow, book shape,
vol, session, cross-sectional, constraint, index/factor) and measure the drop in a rank
correlation score on the purged test fold (Clustered MDA, AFML Ch. 8).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

FEATURE_CLUSTERS: dict[str, tuple[str, ...]] = {
    "order_flow": ("ofi", "signed_vol", "large_print"),
    "book_shape": ("imb", "depth", "wmid", "micro", "spread"),
    "volatility": ("rv", "jump", "vix", "dispersion"),
    "session": ("tod", "mins_to", "opening", "squareoff", "expiry", "sess"),
    "momentum": ("resid_mom", "vwap_minus_mid"),
    "constraint": ("circuit",),
    "index_factor": ("index_ret", "sector_ret"),
}


def assign_cluster(col: str) -> str:
    for name, keys in FEATURE_CLUSTERS.items():
        if any(k in col for k in keys):
            return name
    return "other"


def clustered_mda(model, X: np.ndarray, y: np.ndarray, feature_cols: list[str], *,
                  n_repeats: int = 3, seed: int = 0) -> dict[str, float]:
    """Mean-decrease-accuracy by feature cluster on a held-out fold.

    ``model`` must expose ``predict``. Score is Spearman(pred, y); importance is the mean
    score drop when a cluster's columns are jointly permuted. Positive => the cluster
    carries signal.
    """
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X)
    base = _score(base_pred, y)

    clusters: dict[str, list[int]] = {}
    for j, c in enumerate(feature_cols):
        clusters.setdefault(assign_cluster(c), []).append(j)

    out: dict[str, float] = {}
    for name, idxs in clusters.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            for j in idxs:
                Xp[:, j] = Xp[perm, j]
            drops.append(base - _score(model.predict(Xp), y))
        out[name] = float(np.mean(drops))
    return out


def _score(pred: np.ndarray, y: np.ndarray) -> float:
    if np.std(pred) < 1e-12:
        return 0.0
    rho, _ = spearmanr(pred, y)
    return 0.0 if np.isnan(rho) else float(rho)
