"""bsealpha: intraday cross-sectional ML alpha research on BSE cash equities.

A modular research pipeline implementing the program in
``bse_intraday_ml_research_report.md``:

* :mod:`bsealpha.data`        - panel schema, synthetic generator, loaders
* :mod:`bsealpha.universe`    - point-in-time liquidity screen, clip caps
* :mod:`bsealpha.bars`        - per-name rupee bars + common 1-min grid
* :mod:`bsealpha.features`    - microstructure + cross-sectional feature engine
* :mod:`bsealpha.labeling`    - residual-path triple barrier, meta-labels, weights
* :mod:`bsealpha.models`      - pooled LightGBM (LambdaRank/regression) + meta + TCN
* :mod:`bsealpha.validation`  - purged CV, CPCV, DSR, PBO, effective breadth
* :mod:`bsealpha.portfolio`   - market/sector-neutral book construction
* :mod:`bsealpha.backtest`    - event-driven backtest with the Indian constraint set

Everything runs offline on synthetically generated data; paid vendor / live broker
paths are provided as typed interfaces only.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
