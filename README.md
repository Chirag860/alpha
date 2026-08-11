# bsealpha — Intraday Cross-Sectional ML Alpha on BSE Cash Equities

A modular, tested Python research pipeline implementing the program in
[`bse_intraday_ml_research_report.md`](bse_intraday_ml_research_report.md): a pooled,
cross-sectional machine-learning strategy on BSE cash equities at the **1–30 minute**
horizon, with the full Indian constraint set (STT/brokerage/impact, circuits, T2T,
forced flatten, SEBI ≤10 OPS).

It runs **end to end on synthetically generated data** with zero paid feeds. Paid vendor
(TrueData/GDFL) and live broker (Kite/Dhan/Fyers) paths are provided as typed interfaces.

> Research and education only. Not investment advice. Every fee, tick band, and regulatory
> claim must be verified against current SEBI/BSE circulars before use.

---

## Why this is different from single-instrument (crypto) research

The report's central insight (§0.5): in an equity panel, **cross-sectional breadth is
largely illusory** — every Indian name loads on the market factor, so 30 positions are
mostly *one* bet on the index. Everything downstream follows:

- **Residualize first, then label** (§2.3): model the residual `r − β·index − γ·sector`,
  never the raw return. Otherwise you build an index-timing model with effective breadth ≈ 1.
- **Impact dominates fees** (§6.4) — the opposite of the crypto conclusion. Participation
  caps, not fee optimization, are the binding constraint.
- **Overnight is forbidden** (§0.4): STT on delivery is 5.4× intraday, so the book is
  structurally **flat by 15:15** — a hard vertical barrier on every label.

## Architecture (maps to §9 of the report)

```
generate/load panel  ─▶  point-in-time universe screen  ─▶  feature engine
   (5-level depth,          (T2T/ASM/GSM/circuit/          (OFI, micro-price, book,
    trades, EOD flags)       series/liquidity, §1.2)        signed flow, tod-vol,
                                                            session, index/factor)
        │                                                          │
        │                          residualize (β/γ per fold) + cross-sectional ranks (t−1)
        ▼                                                          ▼
  triple-barrier on the RESIDUAL path (±σ, h=15m, hard 15:15 vertical) + meta-labels
        │                          + sample weights (uniqueness × attribution × decay
        │                            × liquidity × xs-concurrency)
        ▼
  pooled LightGBM  (LambdaRank / regression)  ─▶  LightGBM meta  ─▶  isotonic calibration
        │                                          (trained on OOF primary preds)
        ▼
  validation:  purged day-block CV · CPCV (66 splits/11 paths) · DSR (T=days) · PBO ·
               effective breadth · clustered MDA · tripwires
        ▼
  event-driven backtest (Indian constraint set) ─▶ Sharpe/Sortino/DD/turnover/hit/
                                                     capacity/markouts/breadth
```

### Package layout

| Module | What it does | Report §|
|---|---|---|
| `bsealpha.data` | Panel schema, synthetic multi-name generator, Parquet + broker-stub loaders | §3.1, §10 |
| `bsealpha.universe` | Point-in-time liquidity screen, clip caps | §1.2–1.3 |
| `bsealpha.bars` | Per-name rupee/volume/tick bars + common 1-min grid | §2.2 |
| `bsealpha.features` | OFI(5)+integrated, micro-price, book, tick-rule flow, tod-vol/bipower, session, index/dispersion, **residualize**, cross-sectional ranks, `build_features` engine | §2.3–§3.4 |
| `bsealpha.labeling` | Residual-path triple barrier (15:15 vertical), meta-labels, sample weights | §2.5, §3.6, §4.4 |
| `bsealpha.models` | Pooled LightGBM (LambdaRank/regression) + meta + isotonic; optional TCN trunk; sklearn fallback; `PooledEnsemble` | §4.1–§4.3 |
| `bsealpha.validation` | PurgedDayGroupCV, CPCV, DSR, PBO, effective breadth, clustered MDA, trial log, `evaluate` | §5 |
| `bsealpha.portfolio` | `build_book` (neutral, participation-capped), no-trade band | §7.1 |
| `bsealpha.backtest` | Indian cost stack, queue fill sim (3 cancel models), event-driven engine, metrics/markouts | §0.1, §6 |

---

## Setup

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt        # numpy, polars, scipy, lightgbm, scikit-learn, torch, pyyaml, pytest
# or: pip install -e .
```

**macOS LightGBM note:** LightGBM needs the OpenMP runtime. If `import lightgbm` fails with
a `libomp.dylib` error:

```bash
brew install libomp
```

If LightGBM still cannot load, the pipeline **degrades gracefully** to a scikit-learn
`HistGradientBoosting` fallback (LambdaRank falls back to rank-regression) — everything
still runs. PyTorch is optional (only the TCN trunk needs it; it is off by default; its
test is gated behind `BSEALPHA_RUN_TCN=1` since some torch builds are unstable on the very
newest Python releases).

---

## Run

```bash
python run_research.py                       # full pipeline on synthetic data (default 30×30)
python run_research.py --fast                # skip the 66-fit CPCV sweep (quicker)
python run_research.py --n-names 60 --n-days 60
python run_research.py --config config/default.yaml --seed 11
```

`--fast` finishes in ~1 minute; the full CPCV run does 66 model fits and takes a few
minutes. All knobs live in [`config/default.yaml`](config/default.yaml).

### Reading the output

The report prints two blocks:

- **VALIDATION** — out-of-fold, the honest numbers. IC, meta-AUC, effective breadth, the
  CPCV **gross-relationship** Sharpe distribution (§5.3: "is there a stable relationship?"),
  DSR (with **T = days**, §5.4), PBO, and tripwires.
- **BACKTEST** — the event-driven engine with the Indian constraint set on **out-of-fold**
  predictions: gross vs **net** Sharpe, turnover, hit rate, fees vs impact, capacity, and
  the markout curve.

**What you should see, and why it is honest.** On synthetic data the machinery recovers the
planted residual signal (positive gross Sharpe, healthy breadth ≫ 1, in-range IC/AUC/DSR/PBO),
but **net is impact-dominated**: square-root impact against thin BSE ADV (~10–16 bps/clip,
§6.4) exceeds the thin edge, so net is negative. That is the report's central lesson and its
**modal outcome** — no deployable edge on thin BSE books, tiny capacity (§7.4). The pipeline is
built to *tell you that clearly* rather than to manufacture a Sharpe.

The built-in **tripwires** (§0.5, §11.3) fire on implausibly good results — Sharpe > 4,
meta-AUC > 0.62, IC > 0.08, effective breadth ≈ 1 — because those are bug reports, not wins.

---

## Run on FREE real BSE data (zero-cost smoke test)

```bash
python run_free.py --paper        # downloads free 1-min BSE bars (yfinance), full pipeline
python run_live.py                # SHADOW paper session on the live stack, replayed
python run_live.py --synthetic    # shadow session on the synthetic panel
```

`run_free.py` pulls ~1 week of free 1-minute BSE bars via `yfinance`, builds the **reduced
bars-only feature set** (`build_features_bars_only` — no depth, so no OFI/micro-price/book/
flow), and runs training → backtest → paper end to end. It proves the pipeline ingests and
trades **real** BSE data — but it is a *mechanics check only*: ~1 week of history and no
order-book depth (the caveats print at the end). Real research needs a paid depth vendor
(TrueData/GDFL) and years of data. `run_live.py`'s `live_loop()` is the real-time shadow
skeleton — wire `connect_broker_feed()` to a free broker websocket (Fyers / Angel One
SmartAPI / Dhan) and it runs live against `PaperBroker` with no capital at risk.

---

## Run on an MT5 stock-CFD demo account (Phase 1: plumbing)

The strategy is a **cross-sectional** relative-value model — it ranks many correlated names
against each other. MT5 brokers don't list BSE stocks, but some list **hundreds of US/EU stock
CFDs**, which form the coherent, factor-sharing, many-name universe the cross-section needs. So
the framework can be *repointed* onto an MT5 stock-CFD basket. This is selected entirely by
config: a **market profile** (`bsealpha.market`, session hours + tick bands) and a **cost
profile** — see [`config/mt5.yaml`](config/mt5.yaml), which deep-merges over `default.yaml`.

**Requires a Windows machine/VM** — the `MetaTrader5` Python package is Windows-only and drives
a locally-running MT5 terminal. All MT5-touching code lazily imports it, so the rest of the
package still installs and tests on macOS/Linux.

```bash
# On the Windows VM (MT5 terminal installed + logged into your demo account):
python -m bsealpha.data.mt5_export --config config/mt5.yaml --discover-only  # 1. count stock CFDs
python -m bsealpha.data.mt5_export --config config/mt5.yaml                   # 2. export M1 history
python run_mt5.py --config config/mt5.yaml --dry-run                          # 3a. build orders, don't send
python run_mt5.py --config config/mt5.yaml                                    # 3b. route to the demo
```

Fill in your `login`/`password`/`server` and verify `server_tz_offset_hours` in
`config/mt5.yaml` first. The adapter routes through the same `ExecutionManager` gates as BSE;
`deploy.mode: demo` sends orders to the demo account **without** the live-capital confirmation.

> **Breadth caveat (run step 1 first).** The generic **MetaQuotes-Demo** server is mostly
> forex/metals/indices with a *thin* stock-CFD list. The cross-section needs **100+** names; a
> thin universe makes the neutral book degenerate. If `--discover-only` reports few stocks, open
> a demo with a stock-heavy broker (e.g. Admirals ~3000, Pepperstone ~1000 stock CFDs).

**Phase 1 is mechanics only.** `run_mt5.py` trains the model *in-sample* on the exported history
at startup (like `run_free.py`) — enough to see real orders flow to the demo and flatten at the
session deadline. Retraining/validating the alpha on the new universe (OOF, CPCV/DSR, a
serialized model artifact, beta estimation) is **Phase 2**.

## Testing

```bash
python -m pytest -q
```

Every module has unit tests; **leakage/causality tests are mandatory** for features and
labels and include:

- OFI sign conventions and streaming↔vectorized parity;
- micro-price crossed-weight sign; tick-rule vs. true-sign correlation;
- the **t−1 cross-sectional lag** is enforced (a rank at `t` cannot use any name's value at `t`);
- no feature column is forward-looking;
- **no label crosses the 15:15 forced-flatten**; late-session signals are truncated;
- purged-CV / CPCV produce **zero train/test day overlap**;
- DSR deflates a best-of-noise winner; PBO ≈ 0.5 on noise, low with one real strategy;
- effective breadth collapses toward 1 under a strong market factor (§0.5 calibration).

---

## Extending to real data & live trading

- **Historical:** implement a loader that emits the canonical `DEPTH_SCHEMA` / `TRADE_SCHEMA`
  / `DAILY_SCHEMA` frames (see `bsealpha.data.schema`) from your vendor, then use
  `ParquetLoader`. Everything downstream is unchanged.
- **Live:** implement `bsealpha.data.loaders.BrokerFeed.stream()` against your broker's
  websocket, stamping **local-receipt** timestamps, enforcing the ≤10 OPS token bucket and
  Algo-ID (§8.1), and yielding the same canonical events. The **streaming feature engine**
  (`bsealpha.features.StreamingFeatureEngine`) already computes the raw microstructure
  features event-by-event and is **parity-tested against the batch engine to ~1e-12**
  (`tests/test_parity.py`), so offline == online by construction (the §8.3 requirement); the
  cross-sectional/residual layer (`finalize_features`) is shared code on top.
- **Execution scaffold** (`bsealpha.execution`): the order/position lifecycle is code-complete
  and testable against a `PaperBroker` — `ExecutionManager` drives a target rupee book through
  risk gates → forced-flatten → min-clip filter → ≤10 OPS token bucket → Algo-ID tagging →
  broker. Wiring a real broker = implementing the `BrokerAdapter` interface (`place_order`,
  `cancel_order`, `positions`, `poll_fills`).
- **Paper trading + reconciliation** (`bsealpha.live`): `run_paper_session` replays predictions
  through the live stack (`ExecutionManager` → `PaperBroker` with queue-bounded passive fills +
  the Indian cost stack); `reconcile()` produces the **live-vs-backtest report** — net-Sharpe
  gap, haircut, fill ratio, turnover ratio, markout deltas (`run_research.py --paper`). The gap
  is the number that matters, not the P&L (§11.2 wk6). Then a broker sandbox for plumbing, then
  **tiny real capital** (₹5k–25k clips) — sandboxes reveal nothing about fills or adverse
  selection; only real orders do (§10).
- **Compliance is a gate, not an afterthought** (§8.1): SEBI's retail algo framework is
  mandatory since 1 April 2026. Confirm Algo-ID and OPS handling with your broker before
  routing a single order.

## Deliberate non-goals (per the report)

NSE→BSE latency arbitrage (resolves in ms, §0.6); RL for alpha generation (§4.6); a queue-
position-obsessed market-making system (execution style is second-order here, §0.3). The TCN
trunk and Almgren–Chriss forced-flatten execution are scaffolded but off by default (§4.3, §4.6).
```
