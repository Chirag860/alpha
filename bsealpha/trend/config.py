"""Parameters for the trend + carry system (a small, explicit set — that's the point)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TrendParams:
    # -- signal --
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)   # trailing-return horizons (trading days)
    vol_halflife: int = 33                            # EWMA halflife for risk estimation
    signal_cap: float = 1.5                           # clip combined signal to +/- this
    carry_weight: float = 0.3                         # weight on the (live) carry tilt

    # -- sizing / risk --
    target_ann_vol: float = 0.15                      # portfolio volatility target (annualized)
    max_weight_per_instrument: float = 0.5            # cap |notional weight| per instrument (x NAV)
    max_gross_leverage: float = 4.0                   # cap sum |weights|
    vol_target_overlay: bool = True                   # tighten to realized vol on top of analytic

    # -- execution / costs --
    cost_bps_per_side: float = 1.0                    # default per-side cost if meta lacks spread
    no_trade_band: float = 0.05                       # skip rebalances smaller than this (x weight)
    min_active_frac: float = 0.4                      # drop instruments trading < this fraction of days

    # -- validation --
    dsr_trial_sr_std: float = 0.5                     # spread of the null-trial Sharpes for DSR

    def to_dict(self) -> dict:
        return asdict(self)


def load_trend_params(path: str | Path | None = None, overrides: dict | None = None) -> TrendParams:
    """Load :class:`TrendParams` from a YAML file's ``trend:`` block, with optional overrides."""
    data: dict = {}
    if path is not None and Path(path).exists():
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        data = dict(raw.get("trend", raw))
    if overrides:
        data.update(overrides)
    # tuples survive YAML as lists
    if "lookbacks" in data:
        data["lookbacks"] = tuple(int(x) for x in data["lookbacks"])
    known = {f for f in TrendParams().__dataclass_fields__}  # type: ignore[attr-defined]
    return TrendParams(**{k: v for k, v in data.items() if k in known})
