# Intraday ML Alpha on BSE Cash Equities
### A research program for the 1–30 minute horizon, executing on the Bombay Stock Exchange

**Scope:** BSE cash equity segment, intraday (MIS), 1–30 min prediction horizon, retail/prop infrastructure via broker APIs. Cross-sectional panel across a liquidity-screened BSE universe.
**Date basis:** August 2026. SEBI's retail algo-trading framework is **already mandatory** (since 1 April 2026) — §8 is not optional reading.
**Status:** research/education only. Not investment advice. Numbers marked *[est]* are order-of-magnitude estimates you must re-measure. Verify every fee, tick size, and regulatory claim against current SEBI/BSE circulars before relying on it — they change frequently and several changed within the last 18 months.

**On the code and the numbers:** every arithmetic claim and code snippet here was executed before publication. Verified: the full cost stack and its clip-size sensitivity; per-minute σ across vol regimes on a 375-minute session; relative tick sizes against the SEBI April-2025 bands; micro-price crossed-weight sign; OFI signs on bid-uptick and ask-downtick per Cont–Kukanov–Stoikov; the session-end vertical barrier (no label's exit crosses the 15:15 flatten, late-session bars correctly flagged); effective-breadth participation ratio against a known market-factor loading; the naive-vs-breadth-adjusted Sharpe annualization; Sharpe standard errors by history length; and square-root-law impact against BSE ADV. Run them yourself before trusting any of it.

---

## 0. The arithmetic that governs everything else

### 0.1 The cost stack

Intraday (squared-off same day) equity on BSE, expressed in **bps of one-side notional, per round trip**:

| Component | Rate | Applied | bps |
|---|---|---|---|
| STT | 0.025% | **sell side only** | 2.500 |
| BSE transaction charge (Group A/B) | ₹375 per crore = 0.00375% | both sides | 0.750 |
| Stamp duty | 0.003% | buy side only | 0.300 |
| SEBI turnover fee | ₹10 per crore = 0.0001% | both sides | 0.020 |
| GST @18% on (txn + SEBI) | — | — | 0.139 |
| **Regulatory floor** | | | **3.709** |
| Brokerage ₹20/order × 2, +18% GST | — | — | **size-dependent** |

Sources: [STT rates](https://lakshmishree.com/blog/securities-transaction-tax/), [BSE transaction charges](https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/articles/exchange-transaction-charges), [broker charge sheet](https://zerodha.com/charges/). Note BSE's ₹375/crore is ~26% above NSE's ₹297/crore — a real, if second-order, penalty for the venue constraint.

Because brokerage is a **flat ₹20 per order**, not a percentage, clip size is a first-order design variable:

| Clip size (one side) | Brokerage+GST | **Total round-trip cost** |
|---|---|---|
| ₹1 lakh | 4.72 bps | **8.43 bps** |
| ₹2.5 lakh | 1.89 bps | **5.60 bps** |
| ₹5 lakh | 0.94 bps | **4.65 bps** |
| **₹10 lakh** | 0.47 bps | **4.18 bps** |
| ₹25 lakh | 0.19 bps | **3.90 bps** |
| ₹50 lakh | 0.09 bps | **3.80 bps** |

**Conclusion 1: your minimum viable clip is ₹5–10 lakh.** Below ₹2.5 lakh, brokerage alone doubles your cost base. This collides head-on with BSE's thin books (§1) and is the central tension of the entire project.

### 0.2 The volatility budget

The Indian session is 375 minutes (09:15–15:30 IST) × ~250 days = **93,750 trading minutes/year**, versus 525,600 for a 24/7 crypto perp. Volatility is compressed into 18% as much clock time, which raises per-minute σ substantially:

| Stock annualized vol | σ(1 min) | σ(5 min) | σ(15 min) | σ(30 min) |
|---|---|---|---|---|
| 20% | 6.5 bps | 14.6 bps | 25.3 bps | 35.8 bps |
| 25% | 8.2 bps | 18.3 bps | 31.6 bps | 44.7 bps |
| **30%** | **9.8 bps** | **21.9 bps** | **38.0 bps** | **53.7 bps** |
| 40% | 13.1 bps | 29.2 bps | 50.6 bps | 71.6 bps |

*(Adjust down ~10–20% for the overnight-gap share of total variance: a 30%-annualized-vol stock whose overnight moves carry ~35% of variance has intraday σ ≈ 7.9 bps/min. Measure this per name; it varies a lot between index heavyweights and mid caps.)*

**Conclusion 2: the edge-to-cost ratio here is decisively better than crypto perps.** For a 30%-vol stock at ₹10 lakh clips:

| Horizon | σ | Cost as multiple of σ |
|---|---|---|
| 1 min | 9.8 bps | 0.43 σ |
| 5 min | 21.9 bps | 0.19 σ |
| **15 min** | **38.0 bps** | **0.11 σ** |
| **30 min** | **53.7 bps** | **0.08 σ** |

For reference, the best case in a BTC perp at a 1-minute horizon was **0.65 σ** (maker/maker at VIP 0), and taker/taker was 1.61 σ — arithmetically dead. Here, at 15–30 minutes, costs consume only 8–11% of a one-σ move. **This is a fundamentally more hospitable cost environment, and it is the main reason this pivot is a good idea.**

### 0.3 The tick size regime — a genuinely large-tick market

Following SEBI's April 2025 revision, BSE cash tick sizes are price-banded: **₹0.01 below ₹250, ₹0.05 for ₹250–1,000**, then ₹0.10 / ₹0.50 / ₹1 / ₹5 for successively higher bands ([reference](https://www.business-standard.com/markets/capital-market-news/nse-lowers-tick-size-for-stocks-below-rs-250-124052700731_1.html) — verify current bands against the live BSE circular, this changed recently and may change again).

| Price | Tick | Relative tick | vs BTC perp (0.0091 bps) |
|---|---|---|---|
| ₹50 | ₹0.01 | 2.00 bps | 220× |
| ₹100 | ₹0.01 | 1.00 bps | 110× |
| ₹250 | ₹0.05 | 2.00 bps | 220× |
| ₹500 | ₹0.05 | 1.00 bps | 110× |
| ₹1,400 | ₹0.10 | 0.71 bps | 78× |
| ₹3,000 | ₹0.10 | 0.33 bps | 37× |

Relative ticks are **37–220× larger than a BTC perp**. This is a genuinely large-tick market: spreads are pinned at 1–2 ticks in liquid names, queues at the touch are long, and queue position is a real state variable rather than noise.

**But note the trap, because it inverts the crypto conclusion.** In crypto, fees dwarfed the spread by ~440×, so execution style was everything. Here the 1-tick spread (0.3–2.0 bps) is *smaller* than the round-trip cost (4.18 bps), so:

- Pure passive spread capture is **still not viable** — you would need to clear ~6 ticks of spread on a ₹1,400 stock just to cover fees, and there is **no maker/taker fee distinction on BSE** (transaction charges are flat regardless of whether you add or remove liquidity), so passivity earns you no rebate.
- Being maker rather than taker saves you the spread — roughly 0.7 bps out of a ~4.9 bps taker round trip, a **~14% cost reduction**. Meaningful, not decisive.

**Conclusion 3: execution style is a second-order optimization here, not the strategy.** Do not build a queue-position-obsessed market-making system. Build a **directional cross-sectional model at 5–30 minutes** and use passive placement opportunistically to shave the spread. This is the opposite of the correct conclusion for crypto perps, and getting it backwards will cost you the project.

### 0.4 The overnight prohibition

STT on **delivery** trades is 0.1% on *both* buy and sell = **20 bps round trip**, versus 3.71 bps for an intraday square-off — a **5.4× penalty**. Anything you carry overnight retroactively converts the buy leg into a delivery trade.

**Conclusion 4: the strategy is structurally forced to be flat by 15:30 every day.** This is not a risk-management preference; it is a tax rule. It simplifies the design enormously (no overnight gap risk, no funding, no borrow) and it hard-caps your holding period. Build the forced-liquidation logic into the backtest from day one, including its cost — the 15:15–15:25 unwind is a taker-heavy, adversely-selected window and you will pay for it.

### 0.5 The falsification test — and the breadth illusion

This is the most important paragraph in the report, and it is the one that differs most from single-instrument crypto research.

Suppose a net edge of 2 bps per round trip, per-trade noise σ(15 min) = 38 bps, and 180 round trips per day (30 names × 6 trades). Per-trade Sharpe = 2/38 = 0.053. Naive annualization:

```
0.053 × √(180 × 250)  =  0.053 × 212  =  Sharpe 11.2      ← absurd
```

The error is treating 180 daily trades as 180 independent bets. **They are not.** Every Indian equity loads heavily on the market factor; on any given 15-minute window your 30 positions are largely one bet on Nifty/Sensex direction plus a small residual. Realistic effective breadth for an intraday cross-sectional equity book is **3–8 independent bets per day** *[est — measure it as the participation ratio of the eigenvalues of your position-return covariance]*. With 5:

```
0.053 × √(5 × 250)  =  0.053 × 35.4  =  Sharpe 1.87       ← plausible
```

**Conclusion 5:** cross-sectional breadth in intraday equities is largely illusory unless you explicitly neutralize the market and sector factors. Two consequences that shape everything downstream:

1. **Model and trade the residual, not the raw return.** Beta-hedge to the index (or at least demean cross-sectionally within sector) at both the label stage and the position stage. Otherwise you have built an expensive intraday index-timing model wearing a stock-selection costume.
2. **Any backtest implying Sharpe > ~4 has a breadth bug**, a leak, or a fill fantasy. Treat it as a bug report. This tripwire replaces the "Sharpe > 5" rule used for single-instrument crypto and is, if anything, tighter — because the correlation structure gives you more ways to fool yourself.

### 0.6 What this scope can and cannot capture

**Can:** intraday cross-sectional mean reversion and momentum in index/sector-neutral residuals; order-flow-driven drift over 2–30 min; opening-auction imbalance and the first-30-minute drift it predicts; closing-window flow; volatility regime forecasting; event-driven reaction decay (results, block deals, index-inclusion news); circuit-proximity dynamics.

**Cannot:** NSE→BSE latency arbitrage (price discovery between the venues resolves in **milliseconds**, and the desks doing it are colocated — at 1–30 min the two prices are identical); anything requiring sub-second reaction, given broker-API rate limits of ~10 requests/sec ([Kite Connect](https://kite.trade/forum/discussion/13397/rate-limits)); anything holding overnight (§0.4); anything in a stock under circuit, T2T, ASM or GSM restriction (§1.3).

**Where retail infra caps you:** broker REST/WS APIs, ~10 req/s, 5-level depth as standard with 20-level (L3) available on some platforms but with patchy BSE coverage; no colocation; no tick-by-tick full order book without institutional licensing. Realistic decision-to-ack is **150–400 ms**. Design only for signals with a half-life ≥ 30 s, and preferably ≥ 5 min.

---

## 1. Universe construction — and the honest BSE problem

### 1.1 The liquidity gap, stated plainly

NSE holds roughly **93% of Indian cash-equity turnover**; BSE has the remainder ([market share comparison](https://investwithmithun.com/bse-vs-nse/)). BSE lists ~5,400 companies against NSE's ~2,500, of which ~2,200 are dual-listed — so BSE's ~3,000 exclusive names are overwhelmingly small, illiquid, or SME-platform.

This creates a hard fork, and you should understand which side of it you are on:

- **Dual-listed large/mid caps (Reliance, HDFC Bank, TCS, …).** Deep, well-behaved, and *the reference price is set on NSE.* BSE's book for these names is a thin echo — typically single-digit percent of NSE depth *[est, measure per name]*. Your ₹5–10 lakh clip may be 20–100% of BSE's touch depth in a name where the same clip would be 2–5% of NSE's.
- **BSE-exclusive names.** No NSE competition for the flow, but almost all are small caps with wide spreads, frequent circuit hits, and daily turnover too low to absorb ₹5 lakh clips. Many sit in T2T/ASM/GSM restriction.

**There is no comfortable middle.** The realistic tradable BSE universe for a ₹5–10 lakh clip is, I would estimate, **150–350 names** *[est — this is the single most important number to measure in week 1]*. That is enough for a cross-sectional model, but only just.

### 1.2 A concrete liquidity screen

Rebuild this **weekly, point-in-time**, and store the resulting universe with its as-of date. Never screen using data from after the trade date.

```python
import polars as pl

def bse_universe(daily: pl.DataFrame, asof, lookback_days=20,
                 min_bse_turnover=5e7,      # ₹5 crore/day median BSE turnover
                 min_bse_trades=3000,       # trades/day: proxy for continuous quoting
                 min_price=50, max_price=20000,
                 max_spread_bps=15,
                 min_participation_pct=1.5):  # your clip as % of ADV, capped
    """
    daily : point-in-time BSE EOD panel with columns
            [date, scrip_code, symbol, close, bse_turnover, bse_trades,
             median_spread_bps, series, asm_flag, gsm_flag, t2t_flag,
             circuit_band_pct, is_suspended]
    Returns the tradable universe as of `asof`. EVERY exclusion below is a
    real, tradability-destroying constraint, not a cosmetic filter.
    """
    win = daily.filter(
        (pl.col("date") < asof) &                       # STRICTLY before: no same-day info
        (pl.col("date") >= asof - pl.duration(days=lookback_days * 2))
    )
    agg = (win.group_by("scrip_code")
              .agg(turnover=pl.col("bse_turnover").median(),
                   trades=pl.col("bse_trades").median(),
                   spread=pl.col("median_spread_bps").median(),
                   price=pl.col("close").last(),
                   band=pl.col("circuit_band_pct").min(),
                   n_obs=pl.len()))
    latest = daily.filter(pl.col("date") == win["date"].max())
    return (agg.join(latest.select(["scrip_code", "series", "asm_flag", "gsm_flag",
                                    "t2t_flag", "is_suspended"]), on="scrip_code")
        .filter(
            (pl.col("n_obs") >= lookback_days) &        # enough history
            (pl.col("turnover") >= min_bse_turnover) &
            (pl.col("trades")   >= min_bse_trades) &
            (pl.col("spread")   <= max_spread_bps) &
            (pl.col("price").is_between(min_price, max_price)) &
            (pl.col("band")     >= 20) &                # exclude 2/5/10% band names
            (~pl.col("t2t_flag")) &                     # T2T: NO intraday netting possible
            (~pl.col("asm_flag")) & (~pl.col("gsm_flag")) &
            (~pl.col("is_suspended")) &
            (pl.col("series").is_in(["A", "B"]))        # exclude Z, SME, XC/XD/XT
        ))

def max_clip(turnover_median, participation_pct=1.5):
    """Cap each clip at a fixed % of that name's own median BSE turnover."""
    return turnover_median * participation_pct / 100.0
```

### 1.3 The exclusions are not optional — they are tradability constraints

| Restriction | What it does | Why it kills you |
|---|---|---|
| **T2T / BE series** | Trade-to-trade: every trade must result in delivery | **Intraday netting is prohibited.** You cannot square off. Your position becomes a delivery trade at 20 bps STT plus overnight risk. Automatic, non-negotiable exclusion. |
| **ASM (Additional Surveillance Measure)** | 100% margin, often periodic call auction, reduced price band | Continuous trading may not exist; margin makes the strategy uneconomic |
| **GSM (Graded Surveillance Measure)** | Escalating tiers up to trade-once-a-week with 5% band and 100% deposit | Effectively untradeable |
| **Circuit bands (2/5/10/20%)** | Price frozen at band | Position becomes unexitable *precisely when you most want out*. A 2% or 5% band name can gap to the band and trap you into a delivery position. |
| **Suspension / corporate-action halt** | No trading | Same |
| **Freeze quantity** | Single-order size cap per scrip | Forces splitting; interacts with order-to-trade-ratio penalties |

**The circuit-band interaction with §0.4 is the sharpest risk in the whole strategy.** If a name hits the upper circuit while you are short intraday, you cannot cover, you go to short delivery, you face the exchange auction settlement with a penalty, and your "intraday" trade becomes a 20 bps delivery trade with an auction markup on top. Model circuit proximity as both a **feature** and a **hard position constraint**: no new positions when the price is within, say, 1.5% of a band, and forced flatten at 1.0%.

### 1.4 Survivorship, corporate actions, and the point-in-time universe

With one crypto perp, these were non-issues. With a 300-name equity panel they are first-order and they will silently inflate your backtest by a large factor if handled naively.

- **Survivorship.** Build the universe from a point-in-time listing table that includes delisted, suspended, merged, and renamed scrips. BSE scrip *codes* are more stable than symbols — key on `scrip_code`, never on the ticker string. A universe screened on today's listed set will exclude every name that blew up, which is exactly the population your model most needs to have seen.
- **Corporate actions.** Splits, bonuses, rights, dividends, and consolidations must be applied with a point-in-time adjustment factor. A 1:10 split shows up as a −90% one-minute return in raw data, which will dominate any vol estimate and generate spectacular fake signals. Reconcile your adjustment factors against BSE's corporate-action file and assert that no intraday bar return exceeds the circuit band.
- **Name/series changes.** Movement in and out of T2T, ASM, and GSM happens on published effective dates. Your universe screen must reflect the flag *as it was known on the morning of the trade date*, not as restated later.
- **The test:** compute the distribution of your 1-minute returns and inspect every observation beyond ±5σ by hand for the first month. Essentially all of them will be data errors or unhandled corporate actions, not alpha.

---

## 2. Bars & target design

### 2.1 Session structure is a feature, not a nuisance

The Indian session has hard structural boundaries that a 24/7 crypto model never had to think about:

| Window (IST) | What happens | Modeling treatment |
|---|---|---|
| 09:00–09:08 | Pre-open order entry (call auction) | **Indicative price/quantity is published and is a genuine predictor of opening drift.** Capture it. |
| 09:08–09:12 | Pre-open matching | No continuous trading |
| 09:12–09:15 | Buffer | — |
| 09:15–09:45 | Opening drift; overnight information gets impounded | Highest vol and volume of the day. Either model separately or include a strong session-time feature. |
| 09:45–14:45 | Continuous, U-shaped lull in the middle | The bulk of your tradable opportunity |
| 14:45–15:20 | Closing ramp; MIS square-off pressure from the whole market | **Predictable, mechanical flow.** Brokers force-close MIS positions in this window. Real signal, and real danger. |
| 15:20–15:30 | Your own forced flatten | Cost centre. Budget for taker exits. |
| 15:40–16:00 | Closing session (closing-price orders) | Not usable for this strategy |

**The intraday volume/volatility curve is far more pronounced than in crypto** — the first and last 30 minutes typically carry a multiple of the midday rate. Every volatility estimate, every barrier width, and every position limit must be **session-time-relative**, not absolute. A 30-bps move at 09:20 and a 30-bps move at 13:00 are completely different events.

### 2.2 Bars

| Bar type | Verdict for BSE intraday |
|---|---|
| **Dollar (rupee) bars, per-name calibrated** | **Primary.** Sample every ₹X of BSE turnover in that name, with X set so the *median* bar duration is ~30–60 s. Returns are closest to IID; the U-shaped intensity curve is absorbed automatically, which is exactly what you want. |
| Volume bars | Acceptable, but the threshold drifts as price moves; recalibrate monthly |
| Time bars (1 min) | Keep a parallel grid for cross-sectional alignment, factor computation, and reporting. **You need a common clock to compute cross-sectional residuals** — that is what the time grid is for. Do not train on it directly. |
| Tick bars | Distorted by order splitting; low information per bar in thin BSE names |
| Imbalance/run bars | Fragile under the session's intensity swings. Skip for v1. |

**The dual-clock architecture matters here in a way it did not for a single crypto perp:** you need *event bars per stock* for feature/label construction and a *common 1-minute grid* for cross-sectional operations (residualization, ranking, portfolio construction). Build both, and be explicit about which clock every quantity lives on.

**Per-name calibration is not optional.** A single rupee threshold across a 300-name universe gives you 400 bars/day in Reliance and 6 in a mid cap. Set X_i = (median daily BSE turnover of name i) / 400.

### 2.3 Targets: residualize first, then label

This is the single most consequential design decision in the report, and it follows directly from §0.5.

**Step 1 — residualize.** On the common 1-minute grid, strip the market and sector factors:

```
r_resid[i,t] = r[i,t] − β_i · r_index[t] − γ_i · r_sector(i)[t]
```

with β and γ estimated on a **trailing** window (60 sessions, EWMA-weighted) and refreshed weekly. Use the Sensex/BSE 500 for the market factor and BSE sector indices for the sector factor. Estimate β on the *intraday* return series, not on daily closes — intraday beta and daily beta differ materially, and the intraday one is what you are hedging.

Alternatively, and more robustly for a first pass: **cross-sectional demeaning within sector at each minute**. It is non-parametric, has no estimation error, and captures most of the benefit.

**Step 2 — label the residual.** Options, in order of preference:

**(a) Volatility-adjusted forward residual return.**
```
y[i,t] = r_resid[i, t→t+h] / (σ_resid[i,t] · √h)
```
with σ_resid a point-in-time EWMA of residual bar returns, **conditioned on session time** (use a time-of-day volatility profile estimated on training data only). h ∈ {5, 15, 30} minutes.

**(b) Cross-sectional rank.** Convert (a) to a within-cross-section percentile rank at each minute. This makes the target scale-free across names and regimes and pairs naturally with a ranking objective (§4.2). For a first cross-sectional intraday model this is often the strongest choice.

**(c) Triple barrier on the residual path**, with barriers at ±k·σ_resid[i,t], a vertical barrier at h, and — critically — **a hard vertical barrier at the forced-flatten time**. A signal firing at 15:10 with a 30-minute horizon does not have 30 minutes. Every label must respect the session end.

**(d) Meta-labeling** on top of any of the above: primary gives the side, secondary decides whether the trade clears the 4.18 bps cost. Same logic as in any López de Prado pipeline, and just as valuable here.

**What not to do:** do not label raw (unresidualized) returns. You will build an intraday index-timing model, your effective breadth will be ~1, your backtest Sharpe will look fine on a trending sample, and it will not survive.

### 2.4 The bid-ask bounce, recalibrated

The Roll (1984) MA(1) bounce in trade-price returns has magnitude ~s/2 per observation. Here s ≈ 0.3–2.0 bps, so the bounce contributes ~0.15–1.0 bps of spurious negative autocorrelation. Against σ(1 min) ≈ 10 bps that is a **1.5–10% contamination** — noticeable but not catastrophic; against σ(15 min) ≈ 38 bps it is negligible.

Contrast with crypto at a 5-second horizon, where the bounce could exceed the true innovation. **So the bounce is a second-order problem at this horizon.** Still: label on **mid-price**, and compute the micro-price as a feature. In a large-tick market the micro-price carries more information than in a small-tick one, because queue imbalance genuinely predicts which side the next tick goes to.

```python
def micro_price(bid_px, bid_qty, ask_px, ask_qty):
    """
    Crossed weights: the ASK price is weighted by the BID quantity.
    Thick bid (imb -> 1) = upward pressure -> micro -> ask_px.
    Getting this backwards is a silent, common sign error.
    In a large-tick market this is a strong feature; verify the sign empirically
    by regressing the next mid change on (micro - mid) and checking beta > 0.
    """
    imb  = bid_qty / (bid_qty + ask_qty)
    return ask_px * imb + bid_px * (1.0 - imb), imb
```

### 2.5 Reference implementation — session-aware labels with forced flatten

```python
import numpy as np, polars as pl

SESSION_OPEN_MIN  = 9 * 60 + 15      # 09:15
SESSION_CLOSE_MIN = 15 * 60 + 30     # 15:30
FLATTEN_MIN       = 15 * 60 + 15     # start forced unwind 15:15
SESSION_LEN       = SESSION_CLOSE_MIN - SESSION_OPEN_MIN   # 375

def session_minute(ts_minutes_of_day):
    """0 at 09:15, 374 at 15:29. Negative/oversized => outside continuous session."""
    return ts_minutes_of_day - SESSION_OPEN_MIN

def tod_vol_profile(resid_ret, sess_min, n_bins=25, clip=(0.4, 3.0)):
    """
    Time-of-day volatility multiplier. FIT ON TRAINING DATA ONLY.
    Indian intraday vol is strongly U-shaped; failing to normalize for it makes
    every model an opening-auction detector.
    """
    b = np.clip((sess_min / SESSION_LEN * n_bins).astype(int), 0, n_bins - 1)
    overall = np.nanstd(resid_ret)
    prof = np.array([np.nanstd(resid_ret[b == k]) / overall if (b == k).sum() > 200 else 1.0
                     for k in range(n_bins)])
    return np.clip(prof, *clip)

def label_residual(resid_px, sess_min, sigma_resid, horizon_min=15,
                   u=1.0, l=1.0, bars_per_min=1.0):
    """
    Triple barrier on the RESIDUAL price path with a hard session-end vertical barrier.
    resid_px    : cumulative residual log-price (index/sector stripped)
    sigma_resid : point-in-time residual vol per bar, ALREADY tod-normalized
    Returns (label, realized_resid_ret, exit_bar, truncated_flag).
    """
    n = len(resid_px)
    h_bars = int(horizon_min * bars_per_min)
    y   = np.zeros(n, np.int8); r = np.zeros(n); ex = np.zeros(n, np.int64)
    trunc = np.zeros(n, bool)
    for t in range(n):
        if sess_min[t] >= FLATTEN_MIN - SESSION_OPEN_MIN:
            ex[t] = t; trunc[t] = True; continue          # too late to open anything
        # vertical barrier = min(horizon, forced flatten)
        bars_to_flatten = int((FLATTEN_MIN - SESSION_OPEN_MIN - sess_min[t]) * bars_per_min)
        t_end = min(t + h_bars, t + bars_to_flatten, n - 1)
        trunc[t] = (t + h_bars) > (t + bars_to_flatten)
        up = resid_px[t] + u * sigma_resid[t] * np.sqrt(h_bars)
        dn = resid_px[t] - l * sigma_resid[t] * np.sqrt(h_bars)
        j = t + 1; hit = 0
        while j <= t_end:
            if resid_px[j] >= up: hit = 1; break
            if resid_px[j] <= dn: hit = -1; break
            j += 1
        j = min(j, t_end)
        y[t], r[t], ex[t] = hit, resid_px[j] - resid_px[t], j
    return y, r, ex, trunc
```

The `truncated` flag is not cosmetic. Signals fired late in the session have systematically shorter effective horizons and therefore lower realized edge relative to a fixed cost. If you do not track truncation, your model will happily learn to fire at 15:10 and your backtest will grant it a full 30-minute move that could never have been realized.

---

## 3. Microstructure features for Indian cash equities

### 3.1 What depth you actually get

| Feed level | Content | Availability |
|---|---|---|
| L1 | Best bid/ask + LTP | Universal; all broker APIs; authorized vendors (TrueData, GDFL) |
| **L2 (5 levels)** | Standard exchange depth | **Your working assumption.** Available via broker websockets. |
| L3 (20 levels) | Extended depth | Available on some platforms — [Kite's 20-depth](https://zerodha.com/z-connect/kite/introducing-20-depth-or-level-3-data-beta-on-kite) is currently NSE-weighted with patchy BSE coverage. Verify per-scrip before designing features that need it. |
| TBT (tick-by-tick, full order book) | Order-by-order | Institutional licensing + colocation. **Not available to you.** |

Consequences: you get 5-level *aggregated* depth. You cannot see order counts, individual order sizes, or your own queue rank. Depth updates are snapshot-based, not order-by-order, so cancel/trade attribution is inferred, not observed (§6.3).

**Use NSE data as a measurement instrument, even though you execute on BSE.** For a dual-listed name, NSE carries ~93% of the flow, which means NSE's book and tape give you a **far better-conditioned estimate of the same underlying order flow** — more trades, deeper book, less noise. Since price discovery between the venues resolves in milliseconds, NSE and BSE prices at a 5-minute horizon are the same number; NSE simply measures it better. This is legitimate public-data feature engineering and it is one of the few genuine advantages available in this setup. It is *not* a latency arbitrage, and you should not build one.

### 3.2 Feature set

| Family | Features | India-specific notes |
|---|---|---|
| **Order flow** | OFI at L1 and across the 5 available levels; integrated OFI (PC1); multiple lookbacks (10 s, 1 min, 5 min, 30 min) | Compute on **NSE** for dual-listed names, and separately on BSE; the NSE-BSE OFI divergence is itself a feature |
| **Book shape** | Queue imbalance L1–L5; log depth by level; depth slope; depth at ±5/10/25 bps; BSE-depth / NSE-depth ratio | Large tick ⇒ imbalance is unusually informative here |
| **Micro-price** | micro − mid; its change; sign persistence | Stronger signal in large-tick markets than in crypto |
| **Spread** | spread in ticks; fraction of session at 1 tick; spread vol; **spread relative to that name's own trailing median** | Cross-sectional spread comparisons are meaningless without normalizing by the name's own tick regime |
| **Trade flow** | Signed volume (tick rule / Lee-Ready — **India does not publish an aggressor flag, unlike crypto**, so signing is estimated and noisy); large-print indicator; trade-size distribution; VWAP − mid | This is a real information loss versus the crypto setup. Validate your sign estimator against L1 quote changes. |
| **Volatility** | Realized vol at 1/5/15 min; bipower variation; jump component; **time-of-day-normalized vol**; vol relative to own trailing profile | The U-shaped profile must be divided out before any cross-sectional comparison |
| **Session structure** | Session minute (cyclical + raw); minutes to close; minutes since open; opening-30-min flag; MIS-square-off-window flag (14:45+) | Among the most reliably useful features in this market |
| **Pre-open auction** | Indicative open vs previous close; indicative matched quantity vs ADV; unmatched imbalance and its side | Only usable for the first ~30–60 min, but genuinely predictive there |
| **Cross-sectional** | Rank of every above feature within the day-minute cross-section; rank within sector; residual return over trailing 5/15/30/60 min; distance from VWAP; distance from opening range | **Cross-sectional ranks are usually stronger inputs than raw levels.** Rank-normalize aggressively. |
| **Index / factor** | Sensex and BSE-500 returns over matching windows; sector index return; India VIX level and change; index-futures basis; realized market-wide dispersion | Dispersion (cross-sectional σ of residuals) is a strong regime variable: high dispersion = stock-picking regime = your model works |
| **Constraint state** | Distance to upper/lower circuit (in σ and %); ASM/GSM tier; T2T flag; days since flag change; freeze-quantity headroom | Both a **feature** and a **hard constraint**. Proximity to a band changes the return distribution violently. |
| **Slow / EOD (lagged)** | Delivery percentage (T−1); FII/DII net flow (T−1); bulk/block deal flags (T−1); index-inclusion/exclusion announcements; results calendar; expiry-day flag | All strictly lagged by at least one full session. **Expiry days (weekly Sensex options, monthly F&O) have materially different intraday dynamics — flag them.** |

### 3.3 Causality and data hygiene

Less treacherous than crypto's asynchronous multi-venue websockets, but three hazards remain:

1. **Broker API snapshots are not event streams.** Most retail depth feeds push snapshots at an interval (often ~1 s or on-change with throttling), not every book event. You therefore never see the intermediate states. Your features are inherently coarse — accept it, and make sure your backtest replays the *same* coarse snapshots rather than a finer historical reconstruction. Backtesting on 1-second data and deploying on throttled snapshots is a guaranteed live/backtest divergence.
2. **Cross-sectional features leak trivially.** Computing a cross-sectional rank at minute *t* requires all names at minute *t*. In live trading you have all names at minute *t* only after every name's snapshot has arrived — which is *after* t. Build the cross-section from the **last completed minute**, and enforce it in code, not by convention.
3. **Corporate-action timing.** Adjustment factors are known as of the ex-date morning. Applying an adjustment retroactively to intraday bars before the ex-date is a look-ahead. Store factors with effective dates and apply forward only.

### 3.4 Reference implementation — OFI and cross-sectional normalization

```python
import numpy as np, polars as pl
from dataclasses import dataclass, field

@dataclass
class OFI5:
    """
    Cont-Kukanov-Stoikov OFI over the 5 levels available on Indian exchange depth.
    Feed snapshots in receipt order. State is O(1); no look-ahead is possible.
    """
    M: int = 5
    pb: np.ndarray = field(default=None); qb: np.ndarray = field(default=None)
    pa: np.ndarray = field(default=None); qa: np.ndarray = field(default=None)
    depth_ewma: np.ndarray = field(default=None); alpha: float = 1e-3

    def update(self, bid_px, bid_qty, ask_px, ask_qty):
        if self.pb is None:
            self._init(bid_px, bid_qty, ask_px, ask_qty); return np.zeros(self.M)
        # bid price up  -> whole new queue is fresh liquidity  -> +qty
        # bid price down-> whole old queue gone                -> -prev_qty
        # unchanged     -> the delta
        e_bid = np.where(bid_px > self.pb, bid_qty,
                np.where(bid_px < self.pb, -self.qb, bid_qty - self.qb))
        e_ask = np.where(ask_px < self.pa, ask_qty,
                np.where(ask_px > self.pa, -self.qa, ask_qty - self.qa))
        ofi = e_bid - e_ask
        avg = 0.5 * (bid_qty + ask_qty)
        self.depth_ewma = (1 - self.alpha) * self.depth_ewma + self.alpha * avg
        out = ofi / np.maximum(self.depth_ewma, 1e-12)
        self.pb, self.qb = bid_px.copy(), bid_qty.copy()
        self.pa, self.qa = ask_px.copy(), ask_qty.copy()
        return out

    def _init(self, bp, bq, ap, aq):
        self.pb, self.qb, self.pa, self.qa = bp.copy(), bq.copy(), ap.copy(), aq.copy()
        self.depth_ewma = 0.5 * (bq + aq)


def cross_sectional_normalize(df: pl.DataFrame, feature_cols, by=("date", "minute"),
                              sector_col="sector", lag_minutes=1):
    """
    Rank-normalize features within each (date, minute) cross-section, and within sector.
    CRITICAL: shift by `lag_minutes` first. The cross-section at minute t is only
    complete AFTER minute t, so a live system can never use it at t.
    """
    df = df.sort(["scrip_code", "date", "minute"])
    lagged = [pl.col(c).shift(lag_minutes).over("scrip_code").alias(c) for c in feature_cols]
    df = df.with_columns(lagged)
    out = []
    for c in feature_cols:
        out.append(((pl.col(c).rank("average").over(list(by)) - 0.5)
                    / pl.len().over(list(by)) - 0.5).alias(f"{c}_xs"))
        out.append(((pl.col(c).rank("average").over(list(by) + [sector_col]) - 0.5)
                    / pl.len().over(list(by) + [sector_col]) - 0.5).alias(f"{c}_xsec"))
    return df.with_columns(out)


def effective_breadth(position_returns: np.ndarray):
    """
    THE diagnostic from section 0.5. position_returns: (T, N) matrix of per-period
    P&L contributions by name. Returns the participation ratio of the eigenvalue
    spectrum = the number of INDEPENDENT bets you are actually making.
    If this is ~1, you have built an index-timing model. If it is 3-8, that is normal.
    Use THIS number, not N, when you annualize a Sharpe.
    """
    X = position_returns - position_returns.mean(0)
    C = np.cov(X, rowvar=False)
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-14]
    return float(w.sum() ** 2 / (w ** 2).sum())
```

**Calibration for `effective_breadth`,** from a 60-name simulation with a common market factor of varying strength (residual market loading measured relative to idiosyncratic vol):

| Market loading | Effective breadth |
|---|---|
| 1.0× idio | **1.22** ← an index bet wearing 60 tickers |
| 0.5× | 2.02 |
| 0.2× | 11.24 |
| 0.0 (pure idiosyncratic) | 56.5 ← theoretical ceiling for N=60 |

So the "3–8 independent bets" target in §0.5 corresponds to leaving roughly **0.2–0.5× residual market loading** after neutralization. If you measure breadth ≈ 1–2, your residualization has failed and everything downstream is measuring the wrong thing. Compute this in week 4, not week 6.

---

## 4. Modeling a cross-sectional intraday panel

### 4.1 One pooled model, not 300 models

The single most important architectural decision: **train one model across all names**, with stock identity entering only through normalized features and (optionally) a learned embedding. This is the Sirignano–Cont universality result applied to an Indian panel, and here the argument is stronger than it was for crypto because you have no choice — a mid-cap BSE name gives you perhaps 200 usable bars/day × 250 days = 50,000 observations per year, which is nowhere near enough for a per-stock model of any complexity.

Pooled training gives you 300 names × 200 bars × 250 days ≈ **15M observations/year**. That is a real dataset. But remember §0.5: those 15M rows contain only ~5 independent bets per day, so **your effective sample size for evaluating the strategy is ~1,250 independent observations per year, not 15M.** Rows buy you model-fitting capacity; days buy you evidence. Do not confuse the two.

**Requirements for pooling to work:**

- **Normalize everything per-name and cross-sectionally.** Raw ₹ volumes, raw spreads, and raw volatilities are not comparable between a ₹80 mid cap and a ₹3,000 large cap. Convert to: z-scores against the name's own trailing distribution, cross-sectional ranks at each minute, and ratios to the name's own median.
- **Weight by liquidity, not equally.** An observation in a name where you can actually deploy ₹10 lakh is worth more than one in a name capped at ₹1 lakh. Weight training samples by min(clip_cap, target_clip).
- **Include a small learned stock embedding (8–16 dims)** if you use a neural trunk. It lets the model capture idiosyncratic behaviour without fragmenting the sample. Regularize it hard, and check that it has not simply memorized which stocks trended during the training period — a good test is whether embedding-nearest-neighbours correspond to sectors and liquidity tiers rather than to past returns.

### 4.2 Gradient-boosted trees with a ranking objective

LightGBM remains the workhorse. The change from the single-instrument case is the **objective**:

```python
import lightgbm as lgb

# Cross-sectional ranking: group = one (date, minute) cross-section.
# This directly optimizes what you actually trade -- relative ordering within the
# cross-section -- and is naturally immune to the market factor.
rank_params = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10, 30],
    lambdarank_truncation_level=30,
    learning_rate=0.02, num_leaves=31, max_depth=6,
    min_data_in_leaf=1000, feature_fraction=0.6,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, max_bin=127, verbosity=-1,
)
# group array: number of rows in each (date, minute) cross-section, in row order.
# Labels must be non-negative integers -> bucket the residual return into 5 quantiles.

# Alternative, often better calibrated for position sizing:
reg_params = dict(objective="regression", metric="l2", learning_rate=0.02,
                  num_leaves=31, min_data_in_leaf=1000, lambda_l2=5.0)
# target = cross-sectional rank of the vol-adjusted forward residual return, in [-0.5, 0.5]
```

Practical notes that differ from the single-instrument setup:

- **`num_leaves` can be larger here (31–63) than in the crypto case (15)** because you have far more rows and a better SNR at a 15-minute horizon. But keep `min_data_in_leaf` ≥ 1,000 so each leaf is backed by many *days*, not many rows from one afternoon.
- **Ranking vs regression:** LambdaRank optimizes ordering, which maps cleanly to a long/short cross-sectional book. Regression on cross-sectional ranks gives you a calibrated continuous score that is easier to size on. Build both; they usually agree on the top features and disagree on the tails.
- **Group construction is a leak surface.** The LightGBM `group` array must be built from *contiguous rows within one cross-section*, and your CV split must never cut a group in half.
- **Monotone constraints** on features with a known sign (residual OFI → residual return positive; distance-to-upper-circuit → negative) are cheap regularization. Use them.

### 4.3 Sequence models

Lower priority here than in crypto, for a concrete reason: at a 15-minute horizon on 30–60 second bars, your sequence has ~30 steps of moderately informative history, versus 256 steps of very fine microstructure in the crypto setup. There is less for a sequence model to do.

If you build one:

- **Shared-weight TCN** over per-stock sequences (30–60 steps of the normalized feature vector), plus the stock embedding, trained on the cross-sectional rank target. Strictly causal; explicit receptive field.
- Feed its penultimate embedding (16–32 dims) into the LightGBM as features, rather than trusting its own output head. Extract embeddings **out-of-fold** — the embedding for fold *k* must come from a trunk that never saw fold *k*. This is the most commonly committed leak in stacked pipelines.
- Transformers: not justified at this sequence length and sample size. Skip.

### 4.4 Sample weighting and overlapping labels

Same machinery as any López de Prado pipeline (average uniqueness × return attribution × time decay), with one addition specific to panels:

**Cross-sectional concurrency.** Two observations from *different stocks at the same minute* are not independent either — they share the market factor. Standard uniqueness weighting only accounts for temporal overlap within a name. A defensible practical fix: multiply the temporal uniqueness weight by `1/√(n_names_active_at_t)`, or simply cap the number of names sampled per minute during bagging. Neither is theoretically clean; both are far better than ignoring it.

### 4.5 Non-stationarity: the Indian event calendar

Regimes here are more calendar-driven and more predictable than in crypto, which is an advantage — you can *anticipate* them:

| Event | Effect | Handling |
|---|---|---|
| **Weekly Sensex options expiry** (BSE's flagship product) | Distorted intraday flow, especially in index heavyweights, in the final hours | Flag as a feature; consider a separate model or reduced size |
| **Monthly F&O expiry** | Large mechanical rebalancing flow | Flag; expect your model's IC to differ materially |
| **Results season** (Jan, Apr, Jul, Oct) | Idiosyncratic vol spikes; dispersion rises | Dispersion feature captures much of it; exclude names on their results day |
| **Union Budget (Feb 1)**, RBI MPC | Market-wide vol events | Exclude or size down |
| **Index rebalances** (Sensex/BSE-500 reconstitution) | Predictable flow but crowded | Exclude affected names |
| **Circuit-band or surveillance-tier changes** | Discrete regime change per name | Universe screen catches it; also feed days-since-change |

**Retrain cadence:** monthly full retrain on a trailing 2–3 years with exponentially decayed weights (half-life ~6 months). Weekly is unnecessary here — the microstructure is more stable than crypto's, and you do not have enough new *days* per week to justify it. Recalibrate the probability→size mapping more often (weekly isotonic refit) than you retrain the model.

**Online learning:** not recommended. You accumulate ~250 new independent observations per year. There is nothing to learn online.

### 4.6 Reinforcement learning

Even less justified than in the crypto case. Alpha generation via RL is a supervised problem in costume; the sample here is smaller and the simulator less faithful. The one defensible application is **the forced-flatten execution problem in the 14:45–15:20 window** — a genuine finite-horizon optimal-liquidation problem with a hard deadline, known inventory, and predictable market-wide MIS square-off pressure. That is textbook Almgren–Chriss territory, and AC (or a simple heuristic schedule) will get you most of the value. Do not put RL in the first six weeks.

---

## 5. Validation & overfitting defense

### 5.1 The uncomfortable arithmetic of evidence

Before any technique: understand how little evidence you have.

The standard error of an estimated annualized Sharpe over *Y* years of daily returns is approximately `√((1 + SR²/(2·252)) / Y)`, which for realistic Sharpes is ≈ **1/√Y**:

| History | SE of annualized Sharpe | 95% CI around a measured 1.5 |
|---|---|---|
| 2 years | ±0.71 | **[0.11, 2.89]** |
| 3 years | ±0.58 | [0.36, 2.64] |
| 5 years | ±0.45 | [0.62, 2.38] |
| 10 years | ±0.32 | [0.88, 2.12] |

**With two years of data you cannot distinguish a Sharpe-1.5 strategy from a Sharpe-0.2 strategy.** This is before any adjustment for the hundreds of configurations you will try. It is the strongest possible argument for (a) acquiring as much history as you can afford, (b) using CPCV to get a *distribution* rather than a point estimate, and (c) supplementing daily-P&L evidence with per-trade and per-name evidence, which has more observations even if they are correlated.

Get **at least 3 years, target 5** of intraday BSE data before you draw conclusions. This is the main budget item of the project.

### 5.2 Purging and embargoing a panel

The panel structure changes the mechanics:

- **Split by calendar day, always.** Every name shares the time index, so an observation-level split leaks across the entire cross-section instantly. Groups = trading days. Never split within a day.
- **Purge on label spans.** Drop training days whose labels extend into the test window. With a 15–30 min horizon and forced flatten, labels never cross a session boundary — a genuine simplification versus crypto. But **features** do: a 60-session beta estimate or a 20-day liquidity screen reaches back far.
- **Embargo must cover your longest feature lookback.** If you use 60-session trailing betas, embargo ≥ 60 sessions on the training side preceding a test block. This is expensive and it is correct. Alternatively, recompute betas within each fold using only that fold's available history — more code, less data loss, and the version I would write.
- **Embargo after the test block too**, to prevent the model learning regime information that bleeds backward through volatility clustering.

### 5.3 Combinatorial Purged CV at monthly granularity

With ~3 years = ~750 sessions, use **N = 12–18 monthly groups, k = 2**, giving C(12,2) = 66 splits and 11 reconstructed paths (or C(18,2) = 153 splits, 17 paths). Report the **5th percentile** of the resulting Sharpe distribution as your headline figure.

Two India-specific cautions:

1. **Monthly blocks align with the F&O expiry cycle.** If your blocks are calendar months, every test block contains exactly one monthly expiry, which is fine — but do not use blocks of ~20 sessions offset arbitrarily, or some folds will have two expiries and some none. Align blocks to calendar months deliberately.
2. **Seasonality across the year is real** (results seasons, budget, monsoon-linked sectors). With only 3 years you have 3 observations of each calendar month. Do not read too much into month-specific performance; it is noise.

### 5.4 Deflated Sharpe with the right T

The DSR formula (Bailey & López de Prado) requires the number of return observations *T* and the number of trials *N*. Two errors are easy here:

- **T is the number of days in your P&L series, not the number of trades.** Using 45,000 trades as T instead of 750 days inflates the significance by √60. This is a mechanical, catastrophic error and it is common.
- **N must include every configuration ever evaluated** — hyperparameters, feature sets, horizons, barrier widths, universe screens, residualization choices, and every trial from earlier abandoned versions on the same data.

```python
import numpy as np, pandas as pd
from scipy.stats import norm
EULER = 0.5772156649015329

def expected_max_sharpe(sr_trials):
    N = len(sr_trials); v = np.var(sr_trials, ddof=1)
    return np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N)
                         + EULER * norm.ppf(1 - 1 / (N * np.e)))

def deflated_sharpe(daily_returns, sr_trials):
    """
    daily_returns : DAILY strategy P&L series (one obs per trading session).
    sr_trials     : per-DAY Sharpe of every configuration you evaluated.
    Returns P(true SR > 0). Gate at 0.95.
    """
    r = np.asarray(daily_returns, float); T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew(); g4 = pd.Series(r).kurtosis() + 3.0
    sr_star = expected_max_sharpe(sr_trials)
    num = (sr - sr_star) * np.sqrt(T - 1)
    den = np.sqrt(1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr ** 2)
    return norm.cdf(num / den), sr, sr_star
```

**Calibration to internalize:** with per-trial daily-Sharpe standard deviation of 0.5, the expected maximum Sharpe under the null (every trial has zero true edge) is 1.27 at N=100 trials, **1.63 at N=1,000**, and 1.93 at N=10,000 — in per-period units. After a thousand honest experiments, a Sharpe of 1.6 *is the expected result of pure noise*. And the reverse also bites: a strategy with a genuine per-period edge of 0.05σ, tested against 500 trials over 2,000 periods, scores DSR ≈ 0.79 and **fails a 0.95 gate**. Multiple testing does not just create false positives; past a few hundred trials it destroys your ability to recognize a true positive. Try fewer things, for reasons you can state in advance.

**Instrument your trial count automatically** — log every fit (config hash, CV Sharpe, timestamp) to SQLite or MLflow from day one. Self-reported N is always low by 5–10×.

**And keep a lockbox.** Physically separate the most recent 6 months before you start. Touch it once, at the end. If it fails, the project fails; you do not get to iterate. This is the only defense that does not rely on your own honesty about N.

### 5.5 Panel-specific leaks to hunt

Beyond the standard list, these are the ones that bite in a cross-sectional intraday equity setup:

| Leak | Mechanism | Test |
|---|---|---|
| **Survivorship in the universe** | Screening on today's listed set | Rebuild the universe point-in-time; count how many names in your 2023 universe no longer exist |
| **Restated fundamentals / classifications** | Current sector mapping applied to old data | Store sector with an effective date |
| **Cross-sectional rank computed at t using t** | Needs the whole cross-section, which arrives after t | Assert every `_xs` feature is built from t−1 (§3.4) |
| **Corporate action applied retroactively** | Adjustment factor known only on ex-date | Assert no intraday return exceeds the circuit band |
| **Universe screen using future liquidity** | ADV computed over a window including test days | Screen strictly on data before `asof` (§1.2) |
| **Beta/residualization fit on the full sample** | β estimated using test-period returns | Fit β per fold, or embargo the estimation window |
| **T2T/ASM flags as-restated** | Flag applied before its effective date | Store with effective dates |
| **Group leakage in LambdaRank** | A cross-section split across train and test | Assert every group is wholly in one side |

### 5.6 Feature importance and decay

- **Clustered MDA with purged CV.** Microstructure features are heavily collinear; shuffle them in economic blocks (order flow, book shape, vol, session, cross-sectional, constraint state, index/factor), not one at a time.
- **SHAP stability across CPCV paths.** Spearman ρ < 0.5 between paths means you have a different model in each regime.
- **Stability across the cross-section, not just across time.** Does the model rely on the same features in large caps as in mid caps? If the top features differ completely by liquidity tier, you have two strategies and should either split them or admit that one tier is carrying everything.
- **Decay monitors:** rolling 20-session IC of prediction vs realized residual return; realized vs. predicted bps per trade; effective breadth (§3.4) — a *falling* breadth number means the model is collapsing onto the market factor; fill ratio and its trend; feature PSI.

**The quarterly Sharpe trend matters more than the mean.** A strategy at 2.2 → 1.3 → 0.5 → 0.1 across quarters averages 1.0 and has a forward expectation near zero. This is the normal way intraday equity alpha dies in India, as it is everywhere.

---

## 6. Backtesting with execution realism

### 6.1 What changes versus the crypto case

A vectorized bar-level backtest was catastrophically misleading at a 5-second crypto horizon. At a 15-minute horizon on Indian equities it is **merely inadequate**, which is a meaningful difference — you can get useful triage signal from a well-built bar backtest here. But four things still require an event-driven treatment:

1. **Your clip is a large fraction of BSE's book.** On a dual-listed name, BSE depth is a thin echo of NSE's. A ₹10 lakh clip that is 3% of NSE's touch may be 40% of BSE's. Bar-level backtests assume you transact at the bar price; you will move it.
2. **Circuit halts and surveillance states make positions unexitable.** A vectorized backtest silently exits at the next bar. Reality freezes you at the band and converts your intraday trade into a 20 bps delivery trade plus, if short, an auction penalty.
3. **The forced flatten is a real, expensive, adversely-selected event.** Everyone's MIS positions square off in the same window. Backtests that mark the exit at the 15:30 close are pricing an execution that competes with the entire retail market doing the same thing.
4. **Fill uncertainty on passive orders.** You save ~0.7 bps by posting rather than crossing, but you may not fill, and you fill preferentially when wrong.

### 6.2 The Indian-specific realism checklist

| Item | Treatment |
|---|---|
| **No maker/taker fee distinction** | BSE transaction charges are flat both ways. Passivity saves only the spread. Model it as such — do not import crypto rebate logic. |
| **STT asymmetry** | 0.025% on the **sell leg only**. A long round trip and a short round trip cost the same in total, but the timing of the charge differs. Book it on the sell. |
| **Brokerage as a step function** | ₹20 per *order*, not per fill. **Splitting one clip into five child orders quintuples brokerage.** This is a genuine and counterintuitive constraint: order splitting is expensive in India in a way it is not on crypto venues. Model brokerage per order sent, and let the optimizer see that cost. |
| **Order-to-trade ratio (OTR) penalties** | Exchanges levy charges for excessive order modification/cancellation relative to trades. A quote-heavy strategy accrues real cost. Track OTR in the backtest and price it. |
| **Freeze quantity** | Per-scrip cap on single-order size; forces splitting (and thus more brokerage). Load the freeze-qty table. |
| **Circuit bands** | Hard constraint: no fills beyond the band; positions frozen. Simulate the trap. |
| **T2T / ASM / GSM** | Hard exclusion at universe level, re-checked daily (§1.3). |
| **Short delivery** | If you fail to cover an intraday short, exchange auction settlement with penalty. Model as a large, fat-tailed cost on any unclosed short. |
| **Peak margin / MIS leverage** | SEBI peak-margin rules cap intraday leverage. Verify current broker MIS multipliers; do not assume the pre-2021 numbers. |
| **Pre-open and closing sessions** | Different matching mechanics; exclude from continuous-trading assumptions. |

### 6.3 Fill modeling with snapshot data

You have 5-level aggregated snapshots, not order-by-order events. So the honest queue model is coarser than the crypto one, and you should bound rather than pretend:

- **Taker fills:** walk the visible 5 levels; if your clip exhausts them, assume the remainder fills at the 5th level price plus a penalty drawn from your impact model. Log how often this happens — in thin BSE names it will be common, and it is a capacity signal.
- **Maker fills:** on placement, `queue_ahead = displayed depth at that price`. Decrement by traded volume at that price. For depth reductions not explained by trades, run **three bounds** — all-cancels-ahead (optimistic), proportional (default), all-cancels-behind (pessimistic) — and report Sharpe under each. **If the strategy is only profitable under the optimistic bound, it does not exist.**
- **Do not model queue position more precisely than your data supports.** With ~1 s throttled snapshots you cannot resolve intra-second queue dynamics. Pretending otherwise produces false precision that flatters the backtest.
- **Markouts remain the key diagnostic:** side × (mid[t+h] − fill_price)/fill_price at h = 1, 5, 15, 30 min. Healthy passive execution shows mildly negative markouts at 1 min (adverse selection paid) turning positive by your holding horizon. Monotonically negative means you are someone else's flow.

### 6.4 Impact and the participation cap

Use the square-root law against **BSE turnover in that name**, not consolidated turnover — you are executing on BSE, so BSE's liquidity is what absorbs you:

```
impact ≈ Y · σ_daily · √(Q / ADV_bse),   Y ≈ 0.5–1.0
```

For a name with ₹8 crore/day BSE turnover, σ_daily = 30%/√250 = 1.90%, and a ₹10 lakh clip:
`0.7 × 1.90% × √(10L/8cr) = 0.7 × 1.90% × 0.112 = 14.9 bps`

**That is 3.5× your entire round-trip fee stack.** Read that again — in a thin BSE name, market impact dominates fees by a wide margin, which is the exact opposite of the crypto conclusion where fees dominated everything. It means:

- **Participation caps are the real constraint**, not fee optimization.
- Cap each clip at a fixed fraction of that name's own ADV (start at 1.5–3%), and cap *daily* participation per name similarly.
- Prefer more names with smaller clips over fewer names with larger ones — but remember the ₹20/order floor pushes the other way. The optimum is a genuine two-sided trade-off, and it is worth solving numerically rather than by intuition.

### 6.5 Frameworks

| Option | Verdict |
|---|---|
| **Custom event-driven loop (polars + numba)** | **Recommended for this project.** At a 15-minute horizon on 1-second snapshots, a well-written custom loop is tractable, fully controllable, and lets you encode the India-specific constraints (circuits, T2T, freeze qty, per-order brokerage, OTR) that no off-the-shelf framework knows about. |
| **nautilus_trader** | Excellent engine, deterministic, same code backtest→live. But it has no Indian exchange adapter out of the box and no model of circuits/T2T/STT — you would be writing all the India logic anyway. Consider it if you plan to scale into a multi-market operation. |
| **backtrader / zipline-reloaded / vectorbt** | Bar-level. Fine for **feature triage only** (§6.1). None model the Indian constraint set. |
| **Broker-provided backtesters** | Convenient, opaque, and typically model fills optimistically. Use for sanity checks, never for a go/no-go decision. |

**Recommended split:** vectorized polars pass for feature triage → custom event-driven loop with the full constraint set for validation → broker paper/live for reconciliation. Require that the event-driven loop and the live system produce the same P&L on a recorded day, to a tolerance you specify in advance.

---

## 7. Portfolio, risk & execution

### 7.1 Signal → position, cross-sectionally

The output of §4 is a cross-sectional score. Convert it to a market- and sector-neutral book:

```python
import numpy as np

def build_book(scores, betas, sectors, adv_bse, equity,
               gross_target, max_participation=0.03, min_clip=5e5,
               max_names=60, sector_cap=0.25):
    """
    scores  : cross-sectional model score per name (higher = more attractive long)
    betas   : intraday beta to the market factor
    adv_bse : that name's median BSE daily turnover
    Returns target rupee positions, market- and sector-neutral, participation-capped.
    """
    s = scores - np.mean(scores)                      # market-neutral in score space
    for sec in np.unique(sectors):                     # sector-neutral
        m = sectors == sec
        s[m] -= s[m].mean()

    keep = np.argsort(-np.abs(s))[:max_names]          # trade only the strongest names
    w = np.zeros_like(s); w[keep] = s[keep]
    if np.abs(w).sum() == 0:
        return w
    w = w / np.abs(w).sum() * gross_target             # scale to gross target

    # participation cap: never exceed x% of that name's own BSE ADV
    cap = max_participation * adv_bse
    w = np.sign(w) * np.minimum(np.abs(w), cap)

    # sector gross cap
    for sec in np.unique(sectors):
        m = sectors == sec
        g = np.abs(w[m]).sum()
        if g > sector_cap * gross_target:
            w[m] *= sector_cap * gross_target / g

    # beta neutralize residually (scores are already demeaned, betas are not uniform)
    if np.abs(w).sum() > 0:
        w -= betas * (w @ betas) / max(betas @ betas, 1e-12)

    # brokerage floor: a position below min_clip is not worth the Rs20/order
    w[np.abs(w) < min_clip] = 0.0
    return w
```

Then apply a **no-trade band** rather than tracking the target continuously. With 4.18 bps round-trip cost and σ(15 min) ≈ 38 bps, the optimal policy under quadratic costs is a band around a moving target. Rebalance a name only when the gap exceeds a threshold *and* clears `min_clip`. Grid-search the band on **net** Sharpe — and count that search in your trial count.

### 7.2 The turnover reality check

```
daily fee drag (₹) = (round trips) × 2 × clip × cost_bps × 1e-4
```

60 names × 2 RT/day × ₹7 lakh × 4.18 bps ≈ **₹7,000/day ≈ ₹17.5 lakh/year** in costs alone on a book of that size. Your gross edge must clear that before anything reaches you. At a 15-minute horizon with σ = 38 bps, a 4.18 bps cost is 11% of a one-σ move — comfortable relative to crypto, but it still means **roughly one trade in nine has to be "free" just to pay the toll.**

### 7.3 Risk limits

- **Market and sector neutrality as a hard constraint**, monitored live, not just at construction. Drift in realized beta is the most common way a "neutral" intraday book becomes an accidental index bet.
- **Intraday vol targeting** scaled by a session-time-aware vol forecast, floored/capped at [0.3×, 2.0×]. Size down in the opening 15 minutes until your vol estimate has data.
- **Per-name limits:** participation cap (§6.4), absolute rupee cap, and a hard exclusion within 1.5% of a circuit band (no new positions) / 1.0% (forced flatten).
- **Drawdown governor:** intraday soft stop at −1.5× daily P&L vol → halve gross; hard stop at −3× → flat for the day; rolling 20-session stop at −2× monthly vol → flat, manual restart, written post-mortem.
- **Forced flatten schedule:** begin at 15:15, escalate to market orders by 15:25, guaranteed flat by 15:28. **Never** let an intraday position become a delivery position — the 5.4× STT penalty plus overnight gap risk plus, for shorts, auction settlement, will dwarf a day's P&L.
- **Short-side care:** verify per-scrip intraday shorting is permitted with your broker before including a name; some are restricted, and the restriction list changes.
- **Infrastructure kill switches:** feed staleness > 5 s, snapshot gap, order reject rate > 2%, position mismatch vs broker, API rate-limit breach → flatten and alert. Do not auto-restart.

### 7.4 Capacity — the honest section

The binding constraint is BSE turnover, and it is severe.

Take a working universe of ~60 tradable names averaging ₹8 crore/day BSE turnover, at 3% participation:

```
daily one-way turnover budget = 60 × ₹8cr × 3%  ≈  ₹14.4 crore/day
```

With an average 15-minute holding period, the book turns over ~25× per session, so:

```
sustainable gross book ≈ ₹14.4cr / 25  ≈  ₹58 lakh
```

Lengthening the horizon to 30 minutes roughly doubles it. Realistic ranges *[est — measure your own universe]*:

| Quantity | Range |
|---|---|
| Tradable BSE universe | 150–350 names screened; 40–80 traded on a given day |
| Clip size | ₹5–10 lakh |
| Gross book | ₹50 lakh – ₹2 crore |
| Daily one-way turnover | ₹10–25 crore |
| Deployable capital | ₹1–3 crore |
| Marginal capacity ceiling | ~₹5 crore book before impact eats the edge |

**And the honest comparison you asked me not to make but should hear anyway:** routing the same BSE-listed universe to NSE would raise these numbers roughly an order of magnitude, because NSE carries ~93% of the flow. The BSE-only constraint costs you most of your capacity. If the constraint is regulatory, structural, or a deliberate choice about competition, it is a defensible trade. If it is habit, it is expensive.

**Base rates.** For a well-executed six-week project: ~50% no deployable edge found; ~30% marginal (net Sharpe 0.3–0.8, not worth the operational and regulatory overhead); ~17% modest and real (0.8–1.8); ~3% strong (>1.8, expect decay within 12–24 months). Design the project so the 50% outcome still leaves you reusable infrastructure and a validated research process — that is the modal result, and it is not a failure.

---

## 8. Infrastructure, production & SEBI compliance

### 8.1 Compliance is a gate, not an afterthought

SEBI's retail algorithmic-trading framework became **fully mandatory on 1 April 2026** ([Business Standard](https://www.business-standard.com/markets/news/sebi-extends-retail-algo-trading-framework-rollout-to-2026-125093000956_1.html), [QuantInsti overview](https://www.quantinsti.com/articles/algorithmic-trading-india/)). What this means concretely:

| Requirement | Detail | Your position |
|---|---|---|
| **Algo-ID tagging** | Every algo-originated order must carry an exchange-assigned identifier, so exchanges can trace automated orders to source | Your broker supplies this. Confirm before you write a line of order-routing code. |
| **Broker responsibility** | Brokers are accountable for every algo running through their platform and must conduct due diligence on providers | You will go through a broker's onboarding process. Budget weeks, not days. |
| **No direct exchange connectivity** | Algo providers must partner with a registered broker; they cannot connect directly | Rules out any DIY exchange link. |
| **Order-per-second threshold** | Individual traders whose algos run **under 10 OPS** are treated as regular API users and do not need separate SEBI/exchange strategy registration; above that, registration applies | **Design to stay under 10 OPS.** At a 15-minute horizon across 60 names this is comfortable — but forced-flatten bursts and rebalance storms can spike you over. Rate-limit in code, with a hard token bucket. |
| **Strategy registration** | Strategies offered to others, or above the OPS threshold, must be registered with exchanges | Relevant only if you productize. |

**Two design implications you should internalize now.** First, the 10 OPS ceiling means order splitting is doubly penalized in India — once by the ₹20-per-order brokerage, once by your rate budget. Second, verify all of this directly with your chosen broker and against the current SEBI circular. This framework has already been extended once and the detail may have moved since August 2026.

### 8.2 Latency budget

Target class: **150–400 ms decision-to-ack**, dominated by broker API constraints rather than by your code.

| Stage | Typical | Notes |
|---|---|---|
| Exchange → vendor/broker → your process | 50–250 ms | Snapshot throttling dominates; ~1 s cadence is common on retail depth feeds |
| Snapshot parse + book apply | < 1 ms | Use `orjson`/msgpack, not stdlib `json` |
| Feature update (incremental, per name) | 50–500 µs | Must be O(1) per event across ~300 names |
| Cross-sectional assembly | 1–5 ms | Runs once per minute on the common grid, not per event |
| Model inference (LightGBM, 300 rows) | 1–3 ms | Batch the whole cross-section; trivially fast |
| Risk checks + order build | < 1 ms | |
| Order → broker → exchange ack | 40–200 ms | REST is slower than WS order entry where available |

Implication: **your compute budget is enormous relative to the data cadence.** Spend it on better features and stronger validation, not on micro-optimization. Nothing about this strategy is latency-competitive, and it should not try to be. Colocation exists at Indian exchanges but is institutional; you are not in that race and your alpha should not depend on winning it.

### 8.3 Feature parity — the same warning, with an Indian twist

The offline/online skew problem is identical to any other market: write the feature engine **once**, as an event-driven class, and run the same code over historical replay and live streams. Add a CI test that replays one recorded session offline and asserts every feature matches the live-logged value to 1e-9.

The Indian-specific twist: **backtest on the same throttled snapshot cadence you will receive live.** If your historical vendor provides 1-second snapshots but your broker feed throttles to 2 seconds under load, your backtest is systematically better-informed than production. Record your own live feed from week one and periodically re-run the backtest on *your* captures rather than the vendor's.

### 8.4 Monitoring and retraining

| Category | Metric | Alert |
|---|---|---|
| Data | Snapshot gaps, feed staleness, missing names in the cross-section, corporate-action file freshness | any gap → flatten |
| Universe | Names newly in T2T/ASM/GSM; circuit-band changes; suspensions | daily pre-open check, hard |
| Feature | PSI vs training distribution; NaN rate; cross-section completeness at each minute | PSI > 0.25 on a top-10 feature |
| Model | Prediction distribution vs OOF; calibration of predicted vs realized win rate | KS p < 0.01 |
| Alpha | Rolling 20-session IC; realized vs predicted bps/trade; **effective breadth (§3.4)** | IC < 50% of backtest mean for 5 sessions; breadth trending toward 1 |
| Execution | Fill ratio; markouts at 1/5/15/30 min; realized vs modeled impact; taker fraction; **OTR** | markout at 15 min turns negative |
| Risk | Realized beta and sector exposure; participation vs cap; drawdown tiers; **positions open after 15:25** | any breach |
| Compliance | OPS rate vs the 10/s ceiling; Algo-ID present on every order; reject reasons | OPS > 7/s sustained |

**Retrain monthly** on a trailing 2–3 years with decayed weights. Force an off-cycle retrain on a PSI or IC breach, a tick-size or fee-schedule change, a surveillance-regime change affecting many names, or a market structural break. Every retrain passes the same CPCV + lockbox gates as the original, must beat the incumbent on *fresh* data rather than on the CV that selected it, and shadow-runs for 5 sessions before switching. Never hot-swap on a CV score.

---

## 9. Deliverable: full pipeline

**Latency class: 150–400 ms decision-to-ack. Signal half-life must exceed ~30 s; design target 5–30 min.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INGEST                                                                       │
│  Live:  broker WS (Kite/Dhan/Fyers) L1+5-level depth, ~300 BSE scrips        │
│         + NSE depth for dual-listed names (better-conditioned measurement)   │
│         + Sensex / BSE-500 / sector indices, India VIX                       │
│         + pre-open indicative price & matched qty (09:00-09:08)              │
│  Hist:  TrueData / GDFL authorized tick+depth history (target 3-5 years)     │
│         + BSE bhavcopy, corporate actions, T2T/ASM/GSM flags, circuit bands  │
│  Own capture from day 1 -> NVMe -> Parquet (YOUR receipt timestamps)         │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓ snapshot stream, receipt-ordered, per scrip
┌─────────────────────────────────────────────────────────────────────────────┐
│ UNIVERSE (rebuilt weekly, POINT-IN-TIME)                                     │
│  liquidity screen -> exclude T2T / ASM / GSM / Z / SME / suspended /         │
│  narrow-band names -> per-name clip cap = 1.5-3% of BSE ADV                  │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE ENGINE — ONE codebase, event-driven, CI parity-tested                │
│  per-scrip: OFI (5 lvl), micro-price, book shape, spread, signed flow,       │
│             tod-normalized vol, bipower/jump, circuit distance               │
│  session:   session-minute, minutes-to-close, MIS-squareoff flag,            │
│             pre-open imbalance, expiry flag                                  │
│  index:     Sensex/sector returns, India VIX, dispersion                     │
│  ── on the COMMON 1-MIN GRID, lagged 1 min ──                                │
│  residualize: r - beta*index - gamma*sector   (beta fit per CV fold)         │
│  cross-sectional ranks, within-sector ranks                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓ offline                                        ↓ online
┌────────────────────────────────────┐    ┌──────────────────────────────────┐
│ LABELING                           │    │ INFERENCE (batch the whole        │
│  triple barrier on RESIDUAL path   │    │  cross-section, once per minute)  │
│  +/-1.0 sigma_resid, h=15 min,     │    │   LGBM rank/regression → score    │
│  HARD vertical barrier at 15:15    │    │        ↓                          │
│  truncation flag                   │    │   LGBM meta → p(clears 4.18bps)   │
│  meta-labels net of 4.18 bps       │    │        ↓ isotonic calibration     │
│  weights: uniqueness × attribution │    │   build_book(): mkt+sector neutral│
│           × decay × liquidity      │    │   participation-capped, min-clip  │
└────────────────────────────────────┘    └──────────────────────────────────┘
    ↓                                                ↓
┌────────────────────────────────────┐    ┌──────────────────────────────────┐
│ TRAINING (pooled across all names) │    │ EXECUTION                        │
│  LGBM LambdaRank (group=date,min)  │    │  no-trade band; passive at touch  │
│  + optional shared-weight TCN      │    │  escalate → cross after T1        │
│    embedding (extracted OOF)       │    │  ≤10 OPS token bucket (SEBI)      │
│  + stock embedding 8-16d           │    │  Algo-ID on every order           │
│  monotone constraints on OFI etc.  │    │  FORCED FLATTEN 15:15 → 15:28     │
└────────────────────────────────────┘    └──────────────────────────────────┘
    ↓                                                ↓
┌────────────────────────────────────┐    ┌──────────────────────────────────┐
│ VALIDATION                         │    │ MONITORING                       │
│  purged CV, DAY blocks, embargo ≥  │    │  IC, effective breadth, markouts │
│    longest feature lookback        │    │  realized beta & sector exposure │
│  CPCV N=12-18, k=2 → 11-17 paths   │    │  participation vs cap, OTR, OPS  │
│  report 5th percentile Sharpe      │    │  positions open after 15:25 (!)  │
│  DSR with T = DAYS (not trades)    │    │  → retrain triggers              │
│  PBO < 0.2; clustered MDA; SHAP    │    └──────────────────────────────────┘
│  event-driven backtest × 3 cancel  │                ↓
│    models, with circuits/T2T/OTR   │    ┌──────────────────────────────────┐
│  → LOCKBOX: last 6 months, once    │    │ MONTHLY RETRAIN → shadow 5 days  │
└────────────────────────────────────┘    │ → CPCV + lockbox gate → swap     │
                                          └──────────────────────────────────┘
```

---

## 10. Deliverable: recommended stack

| Layer | Choice | Why / caveat |
|---|---|---|
| **Historical intraday + depth** | **TrueData** or **Global Datafeeds (GDFL)** — both SEBI-authorized NSE/BSE/MCX vendors ([TrueData](https://www.truedata.in/), [levels explained](https://www.truedata.in/blog/levels-real-time-data-nse-bse-mcx)). Target **3–5 years**, 1-min bars minimum, tick+depth where affordable. | This is the main budget line and the main risk. **Confirm BSE-specific depth coverage before paying** — vendor coverage is often NSE-first. Feed levels: L1 / L2 (5) / L3 (20) / TBT; TBT needs institutional licensing you will not get. |
| **Reference / EOD data** | BSE bhavcopy, corporate actions, T2T-ASM-GSM circulars, circuit bands, freeze quantities — all from BSE directly | Free, authoritative, and non-negotiable for point-in-time correctness (§1.4). Automate the daily pull. |
| **Live feed + execution** | **Zerodha Kite Connect** (₹500/mo, 10 req/s, 3,000-instrument WS, [20-depth in beta but NSE-weighted](https://zerodha.com/z-connect/kite/introducing-20-depth-or-level-3-data-beta-on-kite)), or **Dhan** / **Fyers** | Choose primarily on **BSE depth quality and Algo-ID readiness under the SEBI framework**, not on price. Ask each broker directly about BSE 5-level depth latency and their algo onboarding process. |
| **Own capture** | Record every live snapshot to Parquet from day one | Your receipt timestamps; irreplaceable; the only honest basis for §8.3 parity testing |
| **Data processing** | **polars** (lazy/streaming) + **numba** for event loops | A 300-name × 5-year intraday panel is tens of GB; pandas will not cope gracefully |
| **Models** | **LightGBM** (primary, LambdaRank + regression), **PyTorch** only if you add the TCN trunk | GBDT is the production model. Resist the urge to lead with deep learning here. |
| **Backtester** | **Custom event-driven loop** encoding circuits, T2T, freeze qty, per-order brokerage, OTR, STT asymmetry, forced flatten | No off-the-shelf framework knows the Indian constraint set (§6.5) |
| **Fin-ML utilities** | Implement from *AFML* directly, or **`mlfinpy`** | **`mlfinlab` is proprietary, all-rights-reserved** — read the terms before depending on it |
| **Experiment tracking** | **MLflow** or plain SQLite — log **every** fit | This is your trial counter N for the DSR. Non-negotiable. |
| **Paper trading** | Broker paper/sandbox for plumbing; then **small real capital** (₹5,000–25,000 clips) for anything about fills | Sandboxes tell you nothing about fills or adverse selection. Only real orders do. |
| **Compute** | A single well-specced box or a cloud VM in **Mumbai region** (AWS `ap-south-1`) | Proximity helps modestly; nothing here is latency-critical |

---

## 11. Deliverable: the concrete first project (6 weeks)

### 11.1 Specification

| Item | Value |
|---|---|
| Universe | BSE cash equities, weekly point-in-time screen: ≥ ₹5 cr median BSE turnover, ≥ 3,000 trades/day, ₹50–20,000 price, ≤ 15 bps median spread, ≥ 20% circuit band, Series A/B only, no T2T/ASM/GSM/suspended. Expect 150–350 names. |
| Data | 3 years research + **6 months lockbox**, 1-min bars + 5-level depth snapshots; BSE bhavcopy, corporate actions, surveillance flags. Own live capture from week 1. |
| Bars | Per-name rupee bars (median ~45 s) for features/labels; common 1-min grid for cross-sectional operations |
| Residualization | r − β·Sensex − γ·sector, β/γ fit on trailing 60 sessions **within each CV fold**; fallback = cross-sectional demeaning within sector |
| Target | Triple barrier on the residual path, ±1.0·σ_resid (tod-normalized), h = 15 min, **hard vertical barrier at 15:15**, truncation flag |
| Meta target | Binary: did the primary-side trade clear **4.18 bps** round trip? |
| Features | ~70: OFI (5 levels, 4 lookbacks), micro-price, book shape, spread-vs-own-median, signed flow, tod-normalized vol + bipower/jump, session-time set, pre-open imbalance, circuit distance, index/sector/VIX/dispersion, and cross-sectional ranks of all of the above (lagged 1 min) |
| Models | LGBM LambdaRank (group = date-minute) **and** LGBM regression on cross-sectional rank; LGBM meta-model; isotonic calibration. TCN embedding optional, week 5 only if time permits. |
| Validation | Purged day-block CV with embargo ≥ longest feature lookback; **CPCV N=12, k=2 → 66 splits, 11 paths**; DSR with **T = days**; PBO (CSCV, S=16); clustered MDA; SHAP stability across paths and across liquidity tiers |
| Backtest | Custom event-driven, 3 cancel models, full Indian constraint set, participation caps, forced flatten with escalation |
| Deploy gate | 5th-pct CPCV net Sharpe > 0.5; PBO < 0.2; DSR > 0.95; profitable under the **proportional** cancel model; **effective breadth ≥ 3**; markout positive at 15 min; lockbox passes on first and only touch |

### 11.2 Week by week

**Week 1 — Data, universe, and the honest liquidity answer.** Vendor onboarding; pull 3 years; build the point-in-time universe with all surveillance flags and corporate actions. **Deliverable: the number of BSE names that actually support a ₹5 lakh clip, by month, for 3 years.** If that number is under ~80, stop and reconsider the venue constraint before spending another week.

**Week 2 — Feature engine and residualization.** One event-driven class, dual-clock (per-name bars + common minute grid), CI parity test. Residualization and the time-of-day vol profile. *Deliverable: feature panel + a plot of the intraday vol/volume curve you will be normalizing by.*

**Week 3 — Labels and the perfect-foresight ceiling.** Triple barrier on residuals with session-end truncation; sample weights. Then the critical sanity check: **what would perfect foresight of your labels earn, after 4.18 bps costs, participation caps, and a realistic fill model?** If that ceiling is not ≥ 3× your target Sharpe, the label design is wrong and no model will rescue it. *Deliverable: labeled panel + perfect-foresight ceiling.*

**Week 4 — Baselines, in order.** (i) flat; (ii) a single-feature cross-sectional rule (e.g. rank of 15-min residual reversal); (iii) LGBM on engineered features with purged day-block CV. Score everything through the event-driven backtest with the full constraint set. Log every fit. **Compute effective breadth (§3.4) at this stage, not later** — if it is ~1, your residualization is broken and everything downstream is wasted. *Deliverable: LGBM baseline with net Sharpe, breadth, and markouts.*

**Week 5 — Meta-labeling and adversarial validation.** Meta-model on out-of-fold primary predictions; calibration. Then CPCV (66 fits) → Sharpe distribution → 5th percentile; DSR with the logged trial count and T = days; PBO; clustered MDA; SHAP stability across paths and liquidity tiers. *Deliverable: a kill-or-proceed decision. Expect to kill — that is a successful week, not a wasted one.*

**Week 6 — Lockbox, compliance, paper.** One run on the 6-month lockbox: no tuning, no second attempt. In parallel (start earlier — it has lead time): broker algo onboarding, Algo-ID confirmation, OPS rate limiting. If the lockbox passes, paper-trade for plumbing, then live at ₹5,000–25,000 clips. *Deliverable: live-vs-backtest reconciliation. The gap is the number that matters, not the P&L.*

### 11.3 Realistic expectations

| Metric | Realistic range *[est]* | Red flag |
|---|---|---|
| Tradable universe (₹5L clip) | 150–350 screened, 40–80 traded/day | < 80 screened → reconsider venue |
| Cross-sectional IC (15-min residual) | 0.01–0.04 | **> 0.08 ⇒ leak** |
| Meta-model OOS AUC | 0.52–0.58 | > 0.62 ⇒ leak |
| Gross edge per trade | 6–14 bps | > 25 bps |
| Round-trip cost (fees, ₹7L clips) | 4.2–4.7 bps | — |
| Impact per clip (thin BSE names) | 5–20 bps | this usually dominates fees |
| **Net edge per trade** | **1–4 bps** | > 8 bps |
| Round trips / name / day | 1–3 | > 6 (participation-capped) |
| Avg holding period | 10–30 min | < 5 min |
| **Effective breadth** | **3–8 independent bets/day** | **≈ 1 ⇒ index-timing model** |
| **CPCV net Sharpe, median** | **0.8–1.8** | > 4 |
| **CPCV net Sharpe, 5th pct** | **0.0–0.8** | — |
| **Walk-forward net Sharpe** | **0.5–1.3** | > 2.5 |
| **Live Sharpe after decay** | **backtest × 0.4–0.6** | — |
| Max drawdown | 3–8× daily P&L vol | < 2× ⇒ too smooth to be real |
| Gross book | ₹50 lakh – ₹2 crore | — |
| Deployable capital | ₹1–3 crore | — |

Two calibrations to keep in mind. First, remember §5.1: a measured Sharpe of 1.5 over two years carries a standard error of ±0.71, so the honest statement is "somewhere between 0.1 and 2.9." Second, the 40–60% live haircut is the empirical norm, driven by fill optimism, impact underestimation in thin BSE books, and your own presence changing the book.

---

## 12. The references that matter, and why

**Books**

1. **López de Prado, *Advances in Financial Machine Learning* (2018).** Ch. 2–8 and 11–14 are the operational spine: event bars, triple-barrier labeling, meta-labeling, sample uniqueness, purged/embargoed CV, CPCV, clustered feature importance, DSR/PBO. Read it assuming daily-frequency equities examples and make the intraday adaptations yourself (§5.2).
2. **Grinold & Kahn, *Active Portfolio Management*.** The Fundamental Law of Active Management — `IR ≈ IC × √breadth` — and, more importantly, the correlation-adjusted version. **This is the theory behind §0.5**, which is the most important section of this report. If you read one thing about why your 180 daily trades are really 5 bets, read this.
3. **Bouchaud, Bonart, Donier & Gould, *Trades, Quotes and Prices* (2018).** Empirical order-book physics: long-memory order flow, the square-root impact law (§6.4), why linear-impact models are wrong. Ch. 3–5 and 11–13.
4. **Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading* (2015).** Stochastic-control solutions for market making and optimal execution with inventory penalties — directly relevant to the forced-flatten problem (§7.3). Ch. 8–10.

**Papers**

5. **Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events."** Mid-price changes are ~linear in OFI with slope inversely proportional to depth. **If you implement one microstructure feature, implement OFI.** Highest signal per unit of effort in the literature.
6. **Cont, Cucuringu & Zhang (2023), "Cross-Impact of Order Flow Imbalance in Equity Markets."** Multi-level OFI and PCA integration across levels. Matters here because you have exactly 5 levels and want to use all of them.
7. **Sirignano & Cont (2019), "Universal features of price formation in financial markets."** Price formation is universal across instruments, nonlinear, and stationary over time. **This is the justification for pooled cross-sectional training (§4.1)** rather than 300 per-stock models.
8. **Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions."** The mean-variance optimal liquidation schedule. Your forced-flatten window is exactly this problem with a hard deadline. Required as the baseline any smarter scheme must beat.
9. **Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"** + **Bailey, Borwein, López de Prado & Zhu (2014), "The Probability of Backtest Overfitting."** The only rigorous defenses against multiple-testing bias. Implemented in §5.4 — note the T-is-days warning.
10. **Prata et al. (2024/25), "Deep limit order book forecasting: a microstructural guide"** (with the open-source [LOBFrame](https://github.com/FinancialComputingUCL/LOBFrame)). Shows that DL forecasting accuracy depends on microstructural characteristics and that **high accuracy does not imply actionable signals**. Read before writing model code, especially if tempted toward deep learning.
11. **Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread."** The MA(1) bounce in trade prices — recalibrated for this market in §2.4, where it is a second-order rather than dominant effect.

**India-specific primary sources (read the originals, not summaries)**

12. **SEBI circulars on the retail algorithmic trading framework** (2025 issuance, mandatory 1 April 2026), plus your broker's implementation notes. Determines whether you can deploy at all. [Overview](https://www.quantinsti.com/articles/algorithmic-trading-india/); [timeline](https://www.business-standard.com/markets/news/sebi-extends-retail-algo-trading-framework-rollout-to-2026-125093000956_1.html).
13. **BSE circulars**: tick-size bands, transaction charges, T2T/ASM/GSM frameworks, circuit-band and freeze-quantity tables, corporate-action files. These *are* your constraint set (§1.3, §6.2) and they change. Automate the daily pull and diff them.
14. **STT and stamp-duty schedules** ([current rates](https://lakshmishree.com/blog/securities-transaction-tax/); [broker charge sheet](https://zerodha.com/charges/)). The 0.025%-sell-side intraday vs 0.1%-both-sides delivery asymmetry (§0.4) is the single most consequential number in this report.

---

## 13. Standing checklist: how you will know you are fooling yourself

- [ ] Is the universe rebuilt **point-in-time**, including delisted and renamed scrips, keyed on scrip_code? (§1.4)
- [ ] Are T2T, ASM, GSM, suspension, and narrow-band names excluded **as flagged on the morning of the trade date**? (§1.3)
- [ ] Are corporate actions applied forward-only, with no intraday return exceeding the circuit band? (§1.4)
- [ ] Are labels computed on the **residual** path, not raw returns? (§2.3)
- [ ] Does every label respect the **15:15 forced-flatten** vertical barrier, with truncation tracked? (§2.5)
- [ ] Are cross-sectional features built from **t−1**, never from the cross-section at t? (§3.3)
- [ ] Is β/γ residualization fit **within each CV fold**, not on the full sample? (§5.5)
- [ ] Are CV splits by **calendar day**, with embargo ≥ the longest feature lookback? (§5.2)
- [ ] Is every LambdaRank group wholly inside one side of the split? (§5.5)
- [ ] Is the reported Sharpe the **5th percentile** of the CPCV distribution? (§5.3)
- [ ] Is DSR computed with **T = number of days**, not number of trades? (§5.4)
- [ ] Is the trial count N logged automatically, including abandoned experiments? (§5.4)
- [ ] Is the lockbox still untouched — and touched exactly once, at the end?
- [ ] Is **effective breadth ≥ 3**? If it is ~1, you have built an index-timing model. (§3.4, §0.5)
- [ ] Does the backtest charge brokerage **per order sent**, so order splitting is correctly penalized? (§6.2)
- [ ] Does it model circuit traps, short-delivery auction risk, freeze quantity, and OTR? (§6.2)
- [ ] Is impact modeled against **BSE** ADV, not consolidated ADV? (§6.4)
- [ ] Is the strategy profitable under the **proportional** (not optimistic) cancel model? (§6.3)
- [ ] Are markouts negative at 1 min and positive at 15 min? (§6.3)
- [ ] Does the forced-flatten window carry a realistic taker cost? (§7.3)
- [ ] Is the design under **10 orders/second**, with a hard token bucket, and does every order carry an Algo-ID? (§8.1)
- [ ] Is the walk-forward Sharpe **stable across quarters**, or decaying? (§5.6)
- [ ] **Is the backtest Sharpe below 4?** If not, you have a bug. (§0.5)

---

*Research and education only. Nothing here is investment advice, and nothing here claims the described strategy will be profitable. Intraday leveraged equity trading can lose more than the capital deployed. All figures marked [est] are order-of-magnitude estimates requiring re-measurement on your own data. Fee schedules, tick-size bands, surveillance frameworks, STT rates, and the SEBI algorithmic-trading framework all change — verify every one against current SEBI and BSE circulars before relying on it.*
