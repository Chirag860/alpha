# Crash strategy — QuantConnect validation pass

Free, survivorship-bias-free backtest of the Connors "Crash" short strategy. This is the
**validation gate**, not a deployment. The point is to find out whether the edge survives borrow,
fills, and delistings *before* building anything heavier.

## Run it

1. quantconnect.com → **Create New Algorithm** (Python).
2. Replace `main.py` with `crash_algorithm.py` from this folder.
3. **Backtest.**

Universe = US common stock (ETFs excluded via `HasFundamentalData`), price > $5, 21-day avg share
volume ≥ 1M, HV(100) > 100%, ConnorsRSI ≥ 90. Covers on CRSI < 30. Dates 2016–2025 so every stress
regime is in-sample: 2018 vol, 2020 COVID, **2021 meme squeeze** (the make-or-break period), 2022
bear, 2023–25.

## The two checkpoints — do them in order

**1. Faithfulness (gross).** Set `BORROW_ANNUAL = 0.0`, `AVAILABILITY_HAIRCUT = 0.0`,
`COVER_SLIPPAGE_BPS = 0.0`. Run. In QC's Statistics compare against EdgeRater's published gross
numbers:

| Metric | EdgeRater (gross) | QC output |
|---|---|---|
| Win Rate | ~70% | ? |
| Average Win | ~+14% | ? |
| Average Loss | ~-15% | ? |
| Profit-Loss Ratio | ~1.9 | ? |

If these are roughly in the ballpark, the port is faithful and you can trust the net runs. **If
they're way off, the port is wrong — debug before reading any Sharpe.** (Note: exact match is not
expected — different universe/data than EdgeRater's AmiBroker run — but 70%-ish win rate and PF ~2
should reproduce. A win rate of 55% or 85% means something is broken.)

**2. Honesty (net).** Sweep `BORROW_ANNUAL` over **0.00 → 0.15 → 0.50 → 1.00** (four runs). Watch
Sharpe, max drawdown, and the **2021 drawdown** specifically. The brief's expectation: net Sharpe
lands **0.3–0.8** and craters somewhere in the borrow sweep; anything > 1.0 net is more likely a
modelling error than an edge.

## What this run models honestly

- **Survivorship / delistings** — free from QC's security master (the big win).
- **Borrow** — QC charges none by default, so the algo deducts it daily (flat band, or an HV-based
  proxy via `USE_HV_BORROW_PROXY`). The HV>100% screen *is* a hard-to-borrow screen; this is the
  dominant cost.
- **Shorts pay dividends** — QC default.
- **Availability** — `AVAILABILITY_HAIRCUT` randomly drops a fraction of setups as "no locate"
  (seeded, reproducible).
- **Day-limit orders** — unfilled sell-limits expire at the close, not carried forward.
- **Slot accounting** — subtracts both held positions *and* working orders from the cap.
- **Forced liquidation** only on structural universe removal (delist/halt), never on the screen —
  so HV/price/volume can't act as a hidden second exit.

## What is still optimistic (second-pass work, don't trust net numbers past this without it)

- **Fills.** LEAN fills a sell-limit when the daily **high ≥ limit** ("touched"), not "traded
  *through*". Real fills on squeezing microcaps are adverse-selected. A custom fill model requiring
  the trade to clear the limit + a volume cap (≤1–2% of day volume) is the next honesty upgrade.
- **Borrow is a flat band / crude proxy**, not real per-name S3/Ortex fees.
- **No real shortable-quantity data** on the free tier — the haircut is an approximation.

## Go / no-go before spending more effort

Proceed to a local production engine + Alpaca paper only if the QC pass clears **all** of:

- gross per-trade stats reproduce EdgeRater (checkpoint 1),
- **net Sharpe still positive at the 50% borrow band**,
- **2021 drawdown is survivable** (not a book-ender),
- results don't collapse the moment the availability haircut is on.

If it dies here, it died for free — which, given the last three pivots, is exactly the outcome this
step is designed to buy cheaply.
