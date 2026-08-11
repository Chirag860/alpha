"""Labeling: residual-path triple barrier, meta-labeling, and sample weights."""

from __future__ import annotations

from .meta import META_CONTEXT_COLS, build_meta_frame, make_meta_labels
from .triple_barrier import add_cross_sectional_targets, triple_barrier_labels
from .weights import compute_weights

__all__ = [
    "triple_barrier_labels",
    "add_cross_sectional_targets",
    "make_meta_labels",
    "build_meta_frame",
    "META_CONTEXT_COLS",
    "compute_weights",
]
