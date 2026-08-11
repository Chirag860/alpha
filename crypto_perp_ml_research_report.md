# Short-Horizon ML Alpha on a Crypto Perpetual Future
### A research program for the 5s–5min horizon on non-colocated retail infrastructure

**Scope:** single top-tier USDⓈ-M perpetual (BTCUSDT, Binance primary). Prediction horizon 5s–5min. Public websocket/REST access only.
**Status:** research/education. Not financial advice. Every number marked *[est]* is an order-of-magnitude estimate you must re-measure on your own data.

**On the code:** every snippet in this report was executed against synthetic data before publication. The unit tests verified: micro-price stays inside the book and moves toward the correct side; OFI signs match Cont-Kukanov-Stoikov on bid-uptick / ask-downtick; TSRV recovers true integrated variance under microstructure noise where naive RV is off by 10×; purged CV and CPCV produce zero label-span intersection between train and test (66 splits, 11 paths); DSR correctly deflates a best-of-500 noise winner; PBO returns ≈0.5 on pure noise and ≈0.14 when one real strategy is present; the fill simulator's fill ratio is monotone in cancel-model optimism. **Two real bugs were found and fixed this way** — a reversed weight in the micro-price and a false-fill condition in the queue simulator (both flagged inline). Run the tests yourself before trusting any of it.

---

## 0. The arithmetic that governs everything else

Before any modeling discussion, fix the cost/noise ratio. This single table determines what is and is not reachable, and most of the report is downstream of it.

Take BTC annualized realized vol σ_ann ≈ 45% (2025–26 regime; re-measure). With 525,600 minutes/year:

| Horizon | σ of forward return |
|---|---|
| 5 s | ≈ 1.8 bps |
| 30 s | ≈ 4.4 bps |
| 1 min | ≈ 6.2 bps |
| 5 min | ≈ 13.9 bps |

Now the costs. Binance USDⓈ-M VIP 0 is **0.0200% maker / 0.0500% taker**; Bybit is **0.0200% / 0.0550%**; both fall toward 0.000% maker at top VIP tiers ([Binance fee refs](https://tradersunion.com/brokers/crypto/view/binance/futures-fees/), [Bybit fee refs](https://www.bitdegree.org/crypto/tutorials/bybit-fees)). Round-trip explicit cost:

| Execution mode | Round-trip fee | As multiple of σ(1 min) | As multiple of σ(5 s) |
|---|---|---|---|
| taker / taker, VIP 0 | 10.0 bps | **1.61 σ** | 5.6 σ |
| maker in / taker out, VIP 0 | 7.0 bps | 1.13 σ | 3.9 σ |
| maker / maker, VIP 0 | 4.0 bps | 0.65 σ | 2.2 σ |
| maker / maker, maker-free tier | ~0 bps + spread + adverse selection | — | — |

**Conclusion 1: taker-in/taker-out is arithmetically dead at every horizon in scope.** You would need a directional edge exceeding 1.6 standard deviations of the 1-minute return distribution, per trade, net. Nothing in the public literature or in any honest backtest produces that.

**Conclusion 2: the spread is not the prize — it is a rounding error.** BTCUSDT perp tick size is 0.10 USDT; at BTC ≈ 110,000 that is **0.0091 bps**. A 1-tick spread means a half-spread of 0.0045 bps, so the VIP 0 maker fee (2 bps) is **≈ 440× the half-spread**. Even at a wide 5-tick (0.50 USDT) spread the fee is still ~88× the half-spread. To earn back a 4 bps maker round trip from the spread alone you would need the book to be **$44 wide**; it is typically $0.10–$1.00. Classic passive spread capture on this instrument is not a retail strategy at any fee tier you can reach. Verify current `tickSize` from `GET /fapi/v1/exchangeInfo`; exchanges change it.

**Conclusion 2b: the tiny relative tick also destroys queue priority as a defense.** With σ(1 min) = 6.2 bps and a tick of 0.0091 bps, price traverses ~680 ticks per one-minute standard deviation. Undercutting you by one tick costs a competitor essentially nothing. So BTCUSDT perp is emphatically a **small-tick** market: queues at each level are short, price levels are numerous, and the dominant competitive dimension is *speed of repricing* — which is precisely the dimension on which you lose to colocated desks. Queue-position modeling still matters for backtest honesty (§5.3), but do not build a thesis on queue priority.

**Conclusion 3: the only viable retail game in this scope is short-horizon *directional* alpha executed passively.** You need a signal worth ~4–10 bps over 30 s–2 min, and you must capture it with maker fills so the fee line is 4 bps or less, while surviving the adverse selection that comes with being filled precisely when you are wrong.

**Conclusion 4 (the falsification test).** Suppose a genuine net edge of 1.5 bps/trade with 8 bps per-trade noise, 100 trades/day, independent. Annualized Sharpe = 0.19 × √36,500 ≈ **36**. This obviously does not exist. Therefore, if your backtest implies Sharpe > ~5, you have a leak, a fill-model fantasy, or an overlapping-label artifact — not an edge. Treat Sharpe > 5 as a bug report, not a result. Use this as a standing tripwire throughout the project.

### Instrument choice: BTCUSDT perpetual, Binance USDⓈ-M

- **Depth and continuity.** Binance remains the largest single venue for BTC perp OI (roughly 19–30% share depending on the aggregator and date; [Loris](https://loris.tools/oi/coin/btc), [CoinGlass](https://www.coinglass.com/open-interest/BTC)). Depth continuity matters more than headline volume for a queue-sensitive strategy.
- **Data.** Tardis.dev has the deepest retail history for exactly this instrument (tick-level book updates, trades, funding, liquidations, OI), which is what makes leakage-free replay possible at all. Cross-venue reference points (Bybit, OKX, Coinbase spot, CME) are also densest for BTC.
- **Honest counterpoint.** BTCUSDT perp is the single most competed instrument in crypto. Every professional market maker and every stat-arb desk is in this book. Your edge, if any, will live where their infrastructure makes chasing uneconomic — which is *not* the 5–50 ms band, and is plausibly the 10 s–3 min band where the signal is weaker but the competition is thinner and the capacity is small enough to be beneath their attention.

**The case for ETHUSDT is stronger than it first looks, and you should know why.** ETH's relative tick is 0.01/3,500 ≈ **0.0286 bps — about 3× larger than BTC's 0.0091 bps** — so ETH is the more "large-tick" of the two, with relatively more queue structure and a book that is pinned at 1 tick more often. ETH realized vol also typically runs 1.1–1.4× BTC's *[est, measure]*, so σ(1 min) ≈ 7–9 bps against the *same* 4 bps fee floor. On pure edge-to-cost grounds **ETH is the better target.**

**Recommendation, stated honestly:** run BTCUSDT as primary for a first project — deeper book, cleaner data, more cross-venue reference, fewer idiosyncratic data pathologies — and run the identical, untuned pipeline on ETHUSDT as a mandatory out-of-instrument test. If ETH comes out *stronger*, that is not a surprise and not a red flag; it is what the tick/vol arithmetic predicts. If a feature works on BTC and fails on ETH, assume BTC was fitted.

### Perp-specific mechanics you must model

- **Funding.** Binance BTCUSDT settles every 8 h (00/08/16 UTC). Rate = Premium Index P + clamp(interest − P, ±0.05%), interest defaulting to 0.01%/8h; the premium index is a time-weighted average over the interval built from impact bid/ask vs the price index ([Binance docs](https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031)). Consequences for a seconds-horizon strategy: (a) funding is a *cost only if you hold inventory across the stamp* — at a 2-minute horizon you can simply be flat at the boundary; (b) the accumulating premium index in the minutes before settlement generates **predictable order flow** (funding-sniping and pre-funding position adjustment) which is a legitimate, exploitable feature; (c) the sign and magnitude of funding is a good regime/positioning proxy.
- **Basis vs. spot index.** Perp − index basis is a crowding/leverage indicator and mean-reverts at a horizon compatible with 1–5 min.
- **Liquidations.** Binance publishes a `forceOrder` stream. Liquidation cascades are the single highest-signal, lowest-frequency event in the crypto microstructure. They are also when your maker orders get run over. Model them explicitly, both as feature and as risk.
- **Mark price vs. last price.** PnL, liquidation, and funding use mark price (index-anchored), not last traded price. A backtest that marks to last price will misstate margin and can silently permit positions that would have been liquidated.

### What this horizon can and cannot capture

**Can:** order-flow-driven drift over 10 s–3 min; cross-venue lead-lag at 50 ms–2 s granularity (Binance generally leads, but the informative lag is often longer than your latency budget on the *slower* venues); short-horizon volatility and regime forecasting; funding/basis pressure; liquidation-cascade continuation and reversal; queue-imbalance-conditioned drift.

**Cannot:** anything requiring sub-100 ms reaction, because Binance's diff-depth stream is batched at 100 ms cadence — your *information* is already stale by ~50 ms on average regardless of network. Cannot capture true queue-position alpha, because Binance publishes L2 aggregated depth, **not order-by-order (MBO)** data: you cannot observe your own queue rank. Cannot capture latency arbitrage, quote-fade, or any of the sub-millisecond family.

**Where retail infra caps you:** measured AWS `ap-northeast-1` → Binance latency is ≈ 4 ms median / 13 ms p99 ([Deltix measurement](https://deltixworld.com/measuring-websocket-data-feed-latency.html)), and Binance offers no colocation, so this is roughly the floor. Add ~50 ms mean depth-batching delay and 10–40 ms order round trip and your realistic decision-to-ack cycle is **~120–200 ms**. That is the hard boundary. Design only for signals with a half-life ≥ 10× that, i.e. ≥ 2 s, and preferably ≥ 20 s.

---

## 1. Bars & target design

### 1.1 Event bars vs. time bars

Time bars have two known defects: returns are non-normal and heteroskedastic because information arrives in clusters, and sampling is oversampled in quiet periods and undersampled in bursts (López de Prado, *AFML* Ch. 2). At the 5 s–5 min horizon in crypto these defects are severe: BTC perp trade intensity varies by more than an order of magnitude between the Asian lunch lull and a US CPI print.

Practical ranking for this scope:

| Bar type | Verdict |
|---|---|
| **Dollar bars** | **Primary recommendation.** Sample every $X of traded notional. Returns are closest to IID-normal; bar count is stable across price regimes (unlike volume bars, which drift as price moves 3× in a year). Set X so median bar duration ≈ 2–5 s. |
| **Volume bars** | Fine intraday, but need periodic recalibration of the threshold as price level moves. |
| **Tick bars** | Distorted by crypto trade-splitting; one taker order becomes N prints. Use `aggTrade` (aggregated by price/side/ms) rather than raw trades, which partially fixes it. |
| **Imbalance / run bars (TIB, VIB, DIB, DRB)** | Theoretically attractive — they sample when order flow becomes informative. In practice they are **fragile**: the EWMA of expected imbalance is a tuned recursion that can explode or collapse under regime shifts, generating bars of wildly varying duration and silently changing your effective horizon. If you use them, hard-clamp min/max bar duration and treat the clamp rate as a monitored statistic. Do not build v1 on them. |
| **Time bars** | Keep a parallel 1 s time-bar grid as a *reference clock* for evaluation, cross-venue alignment, and reporting. Do not train on them. |

**Crypto-specific caveat:** dollar bars built from a single venue's trade tape inherit that venue's wash/self-trade characteristics. Binance perp is comparatively clean, but check for repeated identical-size prints.

**Critical implementation point:** the bar's timestamp must be the **local receipt time of the event that closed the bar**, not the exchange event time. Backtest causality is about when *you* could have acted.

### 1.2 Why raw short-horizon returns are noise-dominated

At 5 s, σ ≈ 1.8 bps and the tick is 0.0091 bps — so a 5-second return spans **~200 ticks**, almost all of it microstructure noise. Decompose the observed mid return as signal + noise; the efficient-price innovation over 5 s is small relative to (a) the discreteness of the grid, (b) transient impact of individual large child orders that reverts within seconds, and (c) quote flicker from cancel/replace churn. Empirically, R² of any honest model on raw 5 s mid returns will be in the **0.1%–1.5%** range. That is *not* a failure — a 1% R² on a 4.4 bps σ implies a conditional edge of 0.1×4.4 ≈ 0.44 bps, which is below your fee floor. This is precisely why you do not trade the raw return forecast.

### 1.3 The bid-ask bounce corrupts labels

Label on trade prices and you inherit Roll (1984): observed trade-price changes contain an MA(1) component with negative autocorrelation of magnitude −s²/4, purely from buyer/seller-initiated alternation. At the 5 s horizon this bounce component can *exceed* the true price innovation. A model trained on trade-price labels will learn to predict the bounce — i.e. it will learn "the last trade was a buy, so the next print is more likely a sell" — which is real, unarbitrageable (you pay the spread to harvest it), and will produce a beautiful, untradeable backtest.

**Fix:** label on **mid-price** at minimum, and preferably on **micro-price**.

### 1.4 Micro-price vs. mid-price

The volume-weighted mid,

```
P_wmid = (P_ask · V_bid + P_bid · V_ask) / (V_bid + V_ask)
```

is a better instantaneous estimate of the efficient price than the mid, because it moves toward the side with the thin queue. But it is **not a martingale** — it systematically overshoots. Stoikov (2018) defines the *micro-price* as the limit of E[mid_{t+τ}] as τ→∞ conditional on the current (imbalance, spread) state, estimated via a Markov chain on discretized imbalance/spread. The result is a shrunk version of the weighted mid, and the shrinkage factor is asset- and regime-specific.

**Recommendation:** compute all three (mid, wmid, Stoikov micro-price). Label on micro-price. Use `micro − mid` as a *feature* (it is a clean, bounded imbalance signal). Predicting micro-price change rather than mid change materially reduces label noise because the micro-price absorbs the mechanical component of the next mid move.

### 1.5 Targets

**(a) Volatility-adjusted forward return.** The base target:

```
y_t = (P_micro[t+h] − P_micro[t]) / (σ_t · √h)
```

with σ_t an EWMA of bar-level returns computed **strictly from data ≤ t**. Normalizing by σ_t is not cosmetic: without it, the loss function is dominated by the 3% of bars in high-vol regimes and the model becomes a volatility detector wearing a direction-prediction costume.

**(b) Triple-barrier with dynamic barriers.** (*AFML* Ch. 3.) From each event t, set an upper barrier at +u·σ_t, lower at −l·σ_t, vertical at t+T. Label by which is touched first.

- Barrier width must be **at least the round-trip cost**. With maker/maker at 4 bps and σ(1min) = 6.2 bps, a barrier of 1.0σ at h = 1 min is 6.2 bps gross — 2.2 bps net. Use u = l ∈ [1.0, 2.0] and T ∈ [30 s, 2 min]. Anything tighter is fee-dominated by construction.
- Barrier touch must be evaluated on the **tick/quote path**, not on bar closes. Using bar closes systematically understates touch frequency and is a well-known source of optimistic bias.
- Touch should be evaluated against the price you could actually *transact* at (bid for a long exit, ask for a short exit), not the mid. This alone typically removes 15–30% *[est]* of apparent barrier hits.

**(c) Meta-labeling.** (*AFML* Ch. 3.6.) Primary model decides *side*; secondary model decides *whether to act*, trained on binary labels {primary trade was profitable net of cost, or not}. This is the highest-leverage single technique in this report, for three reasons: (i) it converts an F1-hostile 3-class problem into a well-posed binary problem with a natural sample weight; (ii) it lets you use a simple, robust, low-capacity primary (even a linear OFI rule) and put your model complexity where the data supports it; (iii) the secondary model's output probability maps directly onto position size, which is where the money actually is. Expect the secondary model to gate away 60–85% of primary signals.

**(d) What not to do.** Do not use fixed-horizon ±k-bps labels (they encode a volatility forecast, not a direction forecast). Do not use the FI-2010 smoothed-mid labeling scheme (mean of next k mids vs mean of previous k mids) — the backward-looking window overlaps the present and it is a known source of inflated accuracy in the LOB-DL literature.

### 1.6 Reference implementation — micro-price, triple barrier, uniqueness

```python
import numpy as np, polars as pl

# ---------- micro-price family ----------
def price_family(bid_px, bid_qty, ask_px, ask_qty):
    """
    Vectorized mid / weighted-mid / imbalance. All inputs = np arrays at L1.
    NOTE THE CROSSED WEIGHTS: the ASK price is weighted by the BID quantity.
    A thick bid (imb -> 1) means upward pressure, so wmid -> ask_px.
    Getting this backwards is a common and silent sign error.
    """
    mid  = 0.5 * (bid_px + ask_px)
    imb  = bid_qty / (bid_qty + ask_qty)          # in [0,1]; 1 = all depth on the bid
    wmid = ask_px * imb + bid_px * (1.0 - imb)
    assert np.all((wmid >= bid_px - 1e-9) & (wmid <= ask_px + 1e-9))
    return mid, wmid, imb

def stoikov_microprice(mid, imb, spread, n_imb=10, n_spread=3, horizon=200):
    """
    Empirical Stoikov micro-price adjustment g(I, S) = E[mid_{t+H} - mid_t | I_t, S_t].
    Fit on TRAIN ONLY, then apply forward. horizon in bars.
    Returns lookup table; apply as micro = mid + g[i_bin, s_bin].
    """
    fwd = np.full_like(mid, np.nan)
    fwd[:-horizon] = mid[horizon:] - mid[:-horizon]
    i_bin = np.clip((imb * n_imb).astype(int), 0, n_imb - 1)
    s_bin = np.clip(np.searchsorted(np.quantile(spread, np.linspace(0, 1, n_spread + 1)[1:-1]),
                                    spread), 0, n_spread - 1)
    g = np.zeros((n_imb, n_spread))
    ok = ~np.isnan(fwd)
    for a in range(n_imb):
        for b in range(n_spread):
            m = ok & (i_bin == a) & (s_bin == b)
            if m.sum() > 500:
                g[a, b] = fwd[m].mean()
    return g, i_bin, s_bin

# ---------- triple barrier on the TICK PATH ----------
def triple_barrier(event_idx, path_ts, path_bid, path_ask, sigma, u=1.5, l=1.5,
                   vert_ns=60_000_000_000, side=None):
    """
    event_idx : indices into the tick path where signals fire
    path_*    : full quote path (ns timestamps, best bid, best ask)
    sigma     : per-event vol estimate (fractional), computed from data <= event
    side      : +1/-1 per event (for meta-labeling); None => label the raw move
    Exits are evaluated at TRADEABLE prices: long exits on bid, short exits on ask.
    Returns (touch_idx, label, realized_ret).
    """
    n = len(event_idx)
    out_i = np.empty(n, dtype=np.int64); out_y = np.zeros(n, dtype=np.int8)
    out_r = np.zeros(n, dtype=np.float64)
    for k in range(n):
        i0 = event_idx[k]
        s  = 1 if side is None else side[k]
        entry = path_ask[i0] if s > 0 else path_bid[i0]     # cross to enter (pessimistic)
        up, dn = entry * (1 + u * sigma[k]), entry * (1 - l * sigma[k])
        t_end  = path_ts[i0] + vert_ns
        j = i0 + 1; hit = 0; ret = 0.0
        while j < len(path_ts) and path_ts[j] <= t_end:
            exit_px = path_bid[j] if s > 0 else path_ask[j]  # cross to exit
            if exit_px >= up: hit, ret = 1, (exit_px / entry - 1.0) * s; break
            if exit_px <= dn: hit, ret = -1, (exit_px / entry - 1.0) * s; break
            j += 1
        if hit == 0:
            j = min(j, len(path_ts) - 1)
            exit_px = path_bid[j] if s > 0 else path_ask[j]
            ret = (exit_px / entry - 1.0) * s
        out_i[k], out_y[k], out_r[k] = j, hit, ret
    if side is not None:                    # meta-label: did the primary side pay?
        out_y = (out_r > 0).astype(np.int8)
    return out_i, out_y, out_r

# ---------- sample uniqueness & return-attribution weights (AFML Ch.4) ----------
def concurrency(n_bars, t0, t1):
    c = np.zeros(n_bars, dtype=np.int32)
    for a, b in zip(t0, t1):
        c[a:b + 1] += 1
    return c

def avg_uniqueness(c, t0, t1):
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1), 0.0)
    return np.array([inv[a:b + 1].mean() if b >= a else 0.0 for a, b in zip(t0, t1)])

def return_attribution_weights(c, t0, t1, log_ret_by_bar):
    """w_i = |sum_{t in [t0_i,t1_i]} r_t / c_t|, normalized to mean 1."""
    r_over_c = np.where(c > 0, log_ret_by_bar / np.maximum(c, 1), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(r_over_c)])
    w = np.abs(cum[t1 + 1] - cum[t0])
    return w * (len(w) / w.sum())
```

Two things this code makes explicit that most implementations get wrong: barriers are checked against **bid/ask, not mid**, and uniqueness weights are computed from the **actual label spans** (which vary per event under a triple barrier), not from a fixed horizon.

---

## 2. Microstructure features on crypto L2

### 2.0 The causality problem comes first

Everything in this section is worthless if the timestamps are wrong. Three hazards, in order of how often they silently destroy research:

**(a) Exchange time vs. local receipt time.** Every Binance stream carries `E` (event time) and often `T` (transaction time). Neither is when you *knew*. The only legitimate clock for backtest causality is **local receipt timestamp**, because that is the earliest moment the information could enter your decision. Tardis.dev records both `timestamp` (exchange) and `local_timestamp` (their collector's receipt) — **use `local_timestamp` for feature computation and `timestamp` only for intra-venue event ordering.** Note the residual bias: Tardis's collector latency is not your collector's latency; if you deploy from a different region, add a constant offset and re-run sensitivity.

**(b) Cross-venue clock skew.** Binance, Bybit, and OKX clocks are independently synced and drift by single-digit to tens of milliseconds. Aligning two venues by their own event timestamps produces a lead-lag estimate contaminated by clock offset — and since the effect you're measuring is also in the tens of milliseconds, you can manufacture a lead-lag signal out of pure drift. **Align cross-venue features on a single collector's local receipt clock.** Estimate the residual offset by cross-correlating a common reference (e.g. large trade prints on the same underlying) and monitor it; if it moves, your lead-lag features have changed meaning.

**(c) Asynchronous, batched feeds.** Binance's diff-depth stream is batched (100 ms is the fastest generally available depth cadence; partial-depth streams offer 100/250/500 ms — [WS docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams)), while `aggTrade` and `forceOrder` push per event. So trades arrive *before* the book update that reflects them. If you join trades to the "current" book naively you will see the trade and the post-trade book simultaneously — a direct look-ahead of up to 100 ms. **Rule: every feature must be computed by an event-driven state machine that processes events in strict local-receipt order and emits a snapshot only on demand.** Never `merge_asof` a trades frame onto a book frame without a deliberate, signed tolerance. This is the single most common leak in crypto microstructure research.

**(d) Book reconstruction correctness.** Diff-depth streams require the documented snapshot+buffer resync procedure (`pu`/`U`/`u` sequence checks on futures). A dropped update silently corrupts depth for hours. Assert sequence continuity on every event; count and log gaps; drop any period with gaps from training rather than interpolating.

### 2.1 Order Flow Imbalance (OFI)

The single most reliable microstructure feature in any market. Cont, Kukanov & Stoikov (2014) show that contemporaneous mid-price changes are approximately **linear in OFI**, with slope inversely proportional to depth — and that OFI subsumes trade imbalance, because it counts limit-order placements and cancellations at the BBO alongside executions.

Level-1 event contribution between consecutive book states n−1 → n:

```
e_n = 1{P^b_n ≥ P^b_{n−1}}·q^b_n − 1{P^b_n ≤ P^b_{n−1}}·q^b_{n−1}
    − 1{P^a_n ≤ P^a_{n−1}}·q^a_n + 1{P^a_n ≥ P^a_{n−1}}·q^a_{n−1}
```

OFI over a bar = Σ e_n. Cont, Cucuringu & Zhang (2023) extend this to **multi-level OFI**: compute OFI_m at each of the first M levels, scale each by average depth at that level, then reduce the M-vector via its first principal component ("integrated OFI"). Empirically the deeper levels add real information beyond L1 — and in crypto, where BBO is thin relative to L2–L10, this matters more than in equities.

### 2.2 The feature set

| Family | Features | Notes / hazards |
|---|---|---|
| **Order flow** | OFI at L1; OFI_m for m=1..10; integrated OFI (PC1); OFI over multiple lookbacks (100 ms, 1 s, 5 s, 30 s, 2 min) | Multi-scale is essential — the horizon of the OFI window should bracket your prediction horizon. |
| **Book shape** | queue imbalance at L1..L5; log depth by level; slope of the depth curve; depth-weighted price at ±k bps; book "convexity"; ratio of BBO depth to L2–10 depth | Aggregated L2 hides order count. You cannot compute #orders or avg order size — a genuine information deficit vs. MBO desks. |
| **Micro-price** | micro − mid; wmid − mid; Stoikov g(I,S); their first differences | `micro − mid` is bounded and near-stationary — unusually well-behaved as an ML input. |
| **Queue dynamics** | BBO refill rate after depletion; cancel-to-add ratio per level; time since last BBO price change; depth-decay half-life; count of price-level changes per second | Cancel/add decomposition requires diffing consecutive book states; it is approximate under 100 ms batching. |
| **Spread** | spread in ticks; time-weighted spread; fraction of time spread > 1 tick; spread volatility | Spread widening is one of the few clean regime indicators available at this frequency. |
| **Trade flow** | signed volume (use the exchange's `m` maker-side flag on `aggTrade` — **do not use Lee-Ready or tick rule when the true sign is published**); trade-size distribution moments; large-print indicator; run length of same-signed trades; VWAP − mid | Free true trade signs is a real advantage crypto has over equity TAQ. Use it. |
| **Toxicity** | VPIN over volume buckets; order-flow autocorrelation; realized-spread vs. effective-spread decomposition | See caveat below. |
| **Impact / illiquidity** | Kyle's λ (rolling regression of Δmid on signed volume per bucket); Amihud ILLIQ = mean(\|r\|/dollar-vol); depth-normalized λ | λ is your *own* impact estimate and feeds directly into capacity. Estimate it, don't assume it. |
| **Volatility** | realized variance at 1 s/10 s/1 min; **bipower variation** (jump-robust, Barndorff-Nielsen & Shephard); **two-scale RV** (noise-robust, Zhang–Mykland–Aït-Sahalia); jump component RV − BV; Parkinson/Garman-Klass on bars; vol-of-vol | Never use naive RV at tick frequency — it estimates microstructure noise, not volatility. TSRV or a 5–10 s subsampling grid is mandatory. |
| **Perp-specific** | current funding rate; accumulating premium index; **time to next funding stamp** (cyclical encoding); realized funding over trailing 1/3/9 stamps; perp − index basis and its z-score; open interest level and Δ; OI/volume ratio; liquidation volume by side over 1 s/10 s/1 min; time since last liquidation ≥ $1M | The pre-funding window (T−5 min to T) has structurally different flow. Either add a "minutes-to-funding" feature or fit a separate model for that window. |
| **Cross-venue** | Bybit/OKX mid − Binance mid (in bps, lag-aligned); cross-venue OFI; Coinbase/Binance spot vs. perp basis; leader-follower correlation over rolling windows | See 2.4. |
| **Calendar/regime** | hour-of-day (cyclical), day-of-week, minutes to/from major macro prints, session flags (Asia/EU/US) | Crypto has strong and *stable* intraday seasonality in volume and spread. This is real, but it is also the easiest thing to overfit — cap the number of calendar features at ~4. |

### 2.3 Caveats on the famous ones

**VPIN.** Easley, López de Prado & O'Hara (2012) propose VPIN as a real-time flow-toxicity measure using bulk volume classification over equal-volume buckets. It is worth computing. But Andersen & Bondarenko (2014) show that VPIN's apparent predictive power for volatility is **largely mechanical** — it is a noisy transform of trailing volatility and trade intensity, and its forecasting performance does not survive controlling for those. In crypto you additionally have true trade signs published by the exchange, which makes BVC's Gaussian approximation unnecessary. **Recommendation:** compute a true-sign VPIN analogue, but always include trailing RV and trade-count in the same model so that any incremental credit VPIN receives is genuinely incremental. Watch its SHAP value: if VPIN's importance collapses when RV is present, it was never a feature.

**Kyle's λ.** Regress Δmid on signed volume within fixed-volume buckets. λ is not a stable constant — it varies by 3–5× intraday. Use a rolling estimate with an explicit window, and feed *both* λ and its recent change. λ is also the input to your capacity calculation (§6.4), so a biased λ propagates into a biased size limit.

**Amihud.** Cheap, robust, low-resolution. Good as a slow regime variable at 1–5 min aggregation; near-useless below 30 s.

**Realized vol estimators.** At tick frequency, standard RV diverges as sampling frequency increases (it converges to 2·n·(noise variance)). Two-scale RV corrects this by combining a fast and slow subsampling grid. Bipower variation is robust to jumps; the difference RV − BV isolates the jump component, which in crypto is a strong, distinct feature — jumps and continuous vol have different subsequent order-flow signatures.

### 2.4 Cross-exchange lead-lag: a warning

Binance is generally the price-discovery leader for BTC. The problem is arithmetic: if Binance leads Bybit by 20–80 ms, and your own information latency is ~120–200 ms, then **you cannot trade the lag on Bybit** — the professional desks with 1 ms links have already closed it before your packet arrives. What you *can* use cross-venue data for:

1. **As a filter/state variable, not as a trigger.** "Bybit and OKX are both offered below Binance's mid" is a 1–5 s persistent state, not a 50 ms event. That persistence is within your reach.
2. **Divergence magnitude as a toxicity flag.** Wide cross-venue dislocation ⇒ your maker quotes on Binance are about to be adversely selected. Widen or pull.
3. **Aggregate cross-venue OFI** as a stronger version of single-venue OFI, on 5 s–1 min windows.

Do not build a cross-exchange latency-arb backtest. It will look spectacular and it is not yours to have.

### 2.5 Reference implementation — multi-level OFI, streaming

```python
import numpy as np
from dataclasses import dataclass, field

@dataclass
class MultiLevelOFI:
    """
    Streaming multi-level OFI (Cont-Kukanov-Stoikov L1; Cont-Cucuringu-Zhang multi-level).
    Feed strictly in LOCAL-RECEIPT order. State is O(M); no look-ahead possible by construction.
    """
    M: int = 10
    prev_bid_px: np.ndarray = field(default=None)
    prev_bid_qty: np.ndarray = field(default=None)
    prev_ask_px: np.ndarray = field(default=None)
    prev_ask_qty: np.ndarray = field(default=None)
    _depth_ewma: np.ndarray = field(default=None)
    alpha: float = 1e-4                       # depth EWMA decay

    def update(self, bid_px, bid_qty, ask_px, ask_qty):
        """bid_px etc: length-M arrays, level 0 = best. Returns length-M OFI increments."""
        if self.prev_bid_px is None:
            self._init(bid_px, bid_qty, ask_px, ask_qty)
            return np.zeros(self.M)

        bp, bq = self.prev_bid_px, self.prev_bid_qty
        ap, aq = self.prev_ask_px, self.prev_ask_qty

        # bid side: price up => full new qty is an addition; price down => full old qty removed;
        # price unchanged => the delta.
        e_bid = np.where(bid_px > bp, bid_qty,
                np.where(bid_px < bp, -bq, bid_qty - bq))
        # ask side sign-flipped
        e_ask = np.where(ask_px < ap, ask_qty,
                np.where(ask_px > ap, -aq, ask_qty - aq))
        ofi = e_bid - e_ask

        # scale by average depth so levels are comparable (CCZ 2023)
        avg_depth = 0.5 * (bid_qty + ask_qty)
        self._depth_ewma = (1 - self.alpha) * self._depth_ewma + self.alpha * avg_depth
        ofi_scaled = ofi / np.maximum(self._depth_ewma, 1e-12)

        self.prev_bid_px, self.prev_bid_qty = bid_px.copy(), bid_qty.copy()
        self.prev_ask_px, self.prev_ask_qty = ask_px.copy(), ask_qty.copy()
        return ofi_scaled

    def _init(self, bid_px, bid_qty, ask_px, ask_qty):
        self.prev_bid_px, self.prev_bid_qty = bid_px.copy(), bid_qty.copy()
        self.prev_ask_px, self.prev_ask_qty = ask_px.copy(), ask_qty.copy()
        self._depth_ewma = 0.5 * (bid_qty + ask_qty)


class IntegratedOFI:
    """PC1 of the M-vector of scaled OFIs. Fit PCA on TRAIN ONLY; transform forward."""
    def __init__(self): self.w = None
    def fit(self, ofi_matrix):                      # (N, M)
        X = ofi_matrix - ofi_matrix.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        w = Vt[0]
        self.w = w * np.sign(w.sum())               # orient so +ve = buy pressure
        return self
    def transform(self, ofi_matrix): return ofi_matrix @ self.w


def two_scale_rv(log_px, K=30):
    """
    Zhang-Mykland-Ait-Sahalia TSRV. log_px sampled at the FASTEST available grid.
    K = slow-grid subsampling factor. Returns noise-robust integrated variance.
    """
    n = len(log_px) - 1
    rv_fast = np.sum(np.diff(log_px) ** 2)                       # noise-dominated
    rv_slow = np.mean([np.sum(np.diff(log_px[k::K]) ** 2) for k in range(K)])
    n_bar = (n - K + 1) / K
    return rv_slow - (n_bar / n) * rv_fast


def bipower_variation(log_px):
    r = np.abs(np.diff(log_px))
    mu1 = np.sqrt(2.0 / np.pi)
    return (mu1 ** -2) * np.sum(r[1:] * r[:-1])                  # jump-robust
```

**Note on the OFI sign convention** in `update`: when the best bid *price* rises, the entire new queue is fresh liquidity added on the bid side (the old level is no longer best), hence `+bid_qty`; when it falls, the entire old queue was consumed or cancelled, hence `−prev_bid_qty`. This is exactly the CKS indicator written branchlessly, and it generalizes level-by-level.

---

## 3. Advanced modeling

### 3.1 Sequence models on the LOB — what the literature actually supports

**DeepLOB** (Zhang, Zohren & Roberts, 2019) is the reference architecture: a CNN block that convolves across (price, volume) pairs and then across levels, an Inception module for multi-scale temporal filters, and an LSTM head, applied to 100 timesteps × 40 features (10 levels × {bid px, bid qty, ask px, ask qty}). Its contribution is the **inductive bias**: the convolution structure encodes the fact that adjacent LOB levels are related and that price/volume pairs at a level belong together. That bias is genuinely useful and transfers to crypto.

**Sirignano & Cont (2019)** is the more important paper conceptually. Trained on a large cross-section of US equities, they find (a) a **universal** price-formation relationship — a single model trained on all stocks outperforms stock-specific models, (b) the mapping from order-flow history to price moves is **nonlinear** and depends on history beyond the current state (i.e. the LOB is not Markov in its own state), and (c) the relationship is **stationary over time** in a way individual stocks' idiosyncrasies are not. The operational lesson for you: **train across instruments and across time, then specialize** — a model trained on BTC+ETH+SOL perps and fine-tuned on BTC will generalize better than a BTC-only model, and it is a strong regularizer against instrument-specific overfitting.

**Successors (2020–2026).** TransLOB (CNN feature extractor + transformer), TLOB (dual attention, 2025), LiT (LOB transformer), HLOB (using information-persistence structure), and ViT-LOB. The benchmark studies are the useful part, not the architectures:

- Prata et al., *Deep limit order book forecasting: a microstructural guide* (2024/2025, Quantitative Finance) with the open-source **LOBFrame** pipeline: the headline finding is that **stocks' microstructural characteristics determine whether DL works at all**, and that **high forecasting accuracy does not correspond to actionable trading signals**. They propose evaluating on the probability of correctly forecasting *complete transactions* rather than on F1. This is the most important recent paper for your purposes and you should read it before writing model code.
- Across benchmark studies, DL over classical ML buys **1–3%**, and SOTA over vanilla DL buys another **1–2%**, with **all models under ~60% accuracy** on 3-class mid-move prediction. That is the ceiling. Any crypto result far above it is a labeling or leakage artifact.
- FI-2010, the standard benchmark, is 10 days of 5 Nordic small-cap stocks with smoothed labels. **Use it once, to verify your DeepLOB implementation reproduces published numbers, and then never again.** Its conclusions do not transfer to BTC perp.

**Practical architecture ranking for this project:**

1. **TCN** (dilated causal convolutions, WaveNet-style) — first choice. Strictly causal by construction (no accidental future leakage through padding, which is a real bug in careless LSTM/transformer implementations), receptive field is explicit and tunable, trains 5–10× faster than an LSTM of comparable capacity, and parallelizes. For a 100–400 step input window it is at least as accurate as DeepLOB in most replications.
2. **DeepLOB** — worth building as the literature baseline and for its LOB-specific inductive bias. Slow to train (the LSTM head is sequential).
3. **Transformer** — only with strict causal masking, learned or rotary positional encoding, and a sequence length ≤ 512. Attention over LOB sequences is data-hungry; with < ~10M labeled events you will overfit. Reserve for later.
4. **GRU/LSTM** — fine, unremarkable, mainly useful as a head on top of a CNN/TCN trunk.

**Use the sequence model as a representation learner, not as the final predictor.** Train the TCN/DeepLOB on a proxy task (3-class micro-price move, or vol-normalized forward return), then **extract the penultimate-layer embedding (16–32 dims) as features for the GBDT**. This is more robust than trusting the DL model's own output head, because the GBDT can then arbitrate between the learned representation and the hand-engineered features, and you get SHAP attributions on the combination.

### 3.2 Gradient-boosted ensembles — the actual workhorse

On engineered tabular microstructure features, LightGBM will beat every deep architecture in this report, on this data volume, at this signal-to-noise ratio. Plan for it to be the production model.

Configuration that matters at this SNR:

```python
params = dict(
    objective="binary",              # meta-label: act / don't act
    metric="auc",
    learning_rate=0.01,              # low; you have samples but little signal
    num_leaves=15,                   # SMALL. 31+ overfits microstructure noise immediately
    max_depth=5,
    min_data_in_leaf=2000,           # large; each leaf must be statistically real
    feature_fraction=0.5,
    bagging_fraction=0.6, bagging_freq=1,
    lambda_l2=10.0,
    min_gain_to_split=0.01,
    max_bin=127,                     # coarse bins = implicit regularization + speed
    verbosity=-1,
)
# ALWAYS pass sample weights: w = uniqueness * return_attribution * time_decay
```

Non-obvious points:

- **`num_leaves` is your main overfitting knob**, not `n_estimators`. At 0.5% R², 15 leaves is already a lot of capacity.
- **Early stopping must use a *purged* validation fold**, not a random split, or you will stop at the wrong iteration and never notice.
- **Categorical time-of-day** as a native LightGBM categorical is a leak magnet — it lets the model memorize specific historical minutes. Use cyclical sin/cos encoding instead.
- **Monotonic constraints** on features with known sign (OFI → return should be monotone increasing) are a cheap, effective prior that costs almost nothing in-sample and buys real robustness out-of-sample. Use them.
- CatBoost's ordered boosting is a genuine plus for small-sample regimes; XGBoost with `tree_method="hist"` is equivalent to LightGBM for practical purposes. Pick one and stop.

### 3.3 Stacking, blending, and the ensemble that is actually recommended

```
                 ┌──────────────────────────────────────────┐
   raw L2/trades │ TCN trunk (causal, 256-step window)       │──► embedding e_t ∈ R^32
                 └──────────────────────────────────────────┘          │
                 ┌──────────────────────────────────────────┐          │
   engineered    │ ~80 microstructure features (§2.2)        │──────────┼──► LightGBM #1
                 └──────────────────────────────────────────┘          │   (PRIMARY: side)
                                                                       │        │
                                                                       │        ▼
                 ┌──────────────────────────────────────────┐   secondary features:
                 │ meta-features: primary prob, |prob−0.5|, │◄──  primary output +
                 │ current vol, spread, toxicity, inventory │     execution context
                 └──────────────────────────────────────────┘
                                    │
                                    ▼
                          LightGBM #2 (META: act / don't act)  ──►  size = f(p_meta)
```

Rules for stacking without fooling yourself:

- The primary model's out-of-fold predictions that feed the meta-model **must come from purged, embargoed CV**. Using in-sample primary predictions to train the meta-model is the classic stacking leak and it inflates meta-model AUC by 5–15 points.
- Keep the primary **deliberately simple and stable**. A linear or shallow-tree primary on integrated OFI + micro-price is often better than a complex one, because the meta-model's job is easier when the primary's errors are structured rather than idiosyncratic.
- Blend weights (if you blend several primaries) must themselves be fit out-of-fold, and should be constrained (non-negative, sum-to-one) — unconstrained stacking regressions at this SNR produce weights of +14 and −13.

### 3.4 Non-stationarity, online learning, and regime shift

Crypto microstructure regimes change on a timescale of **weeks**, sometimes days: exchange fee-tier changes, a new dominant market maker, a listing on a competing venue, a change in tick size or in the depth-stream cadence, a shift in the retail/institutional mix. Your model's shelf life is short.

Options, ranked by effort-adjusted value:

1. **Frequent full retrain with exponentially-decayed sample weights.** Retrain weekly on a trailing 3–6 months with a half-life of ~3–6 weeks. Boring, robust, and captures ~80% of the available adaptation. Do this first.
2. **Drift detection as a *trigger*, not as a model.** Run ADWIN or Page-Hinkley on (a) rolling feature PSI, (b) rolling prediction distribution, (c) rolling realized IC. Fire a retrain when any breaches. `river` implements these cleanly.
3. **Regime-conditional models.** Fit separate models for {low-vol, high-vol} × {pre-funding, normal}, or add regime as a feature with monotonic constraints. Splitting the data 4 ways quarters your effective sample — only do this if each bucket still has > ~500k unique-weighted samples.
4. **True online/incremental learning** (SGD-based linear, or `river`'s Hoeffding trees). Attractive in principle. In practice: online models at this SNR chase noise, are extremely hard to validate (you cannot do CPCV on a model that has already seen everything), and give you no reproducible artifact to audit after a bad day. **Recommendation: do not run an online-updating model in production until you have a stable batch-retrained model with 3+ months of live track record.** The exception worth making is an online *calibration* layer — a small isotonic or Platt recalibration of the meta-model's probability, refit daily, which corrects drift in the probability→size mapping without touching the learned structure.

### 3.5 Class imbalance — mostly a solved problem, by construction

With a triple barrier sized in units of σ_t, you *control* the class distribution: symmetric barriers at ±1.5σ with a 60 s vertical barrier will produce roughly balanced up/down classes and a large "timeout" class. So the imbalance problem in this setup is not up-vs-down; it is:

- **Timeout dominance.** If most events hit the vertical barrier, your barriers are too wide relative to the horizon. Tune u, l, T so that the timeout class is 30–50%, not 85%.
- **Meta-label imbalance.** After costs, the "act" class will be a small minority (10–30%). Handle with `scale_pos_weight` or `is_unbalance`, **not** with resampling.
- **Never use SMOTE or any synthetic oversampler.** Interpolating between two microstructure states produces states that cannot occur (fractional queue sizes, crossed books) and, worse, interpolates across time, which is leakage. This is a genuine footgun that appears in a great deal of published crypto-ML work.
- **Never random-undersample the majority class** without also recomputing uniqueness weights — you change the effective concurrency structure.

### 3.6 Sample weighting (López de Prado) — do not skip this

Labels from overlapping horizons are not IID. With a 60 s vertical barrier and 2 s bars, each label overlaps ~30 others. Standard bootstrap/bagging therefore massively oversamples redundant information and every OOB estimate is optimistic.

Three weights, multiplied together:

```
w_i = ū_i  ×  |Σ_{t∈span_i} r_t / c_t|  ×  exp(−λ · age_i)
      ^avg uniqueness  ^return attribution     ^time decay
```

Plus **sequential bootstrap** for bagged models: draw samples with probability proportional to their uniqueness *given what has already been drawn*, which yields far more independent bags than uniform bootstrap. Implementation is in *AFML* Ch. 4; note that `mlfinlab` is **not open source** — it is licensed all-rights-reserved by Hudson & Thames ([license](https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt)). For a self-funded project, implement these from the book (they are ~200 lines total, and the versions in §1.6 and §4.3 here cover most of it) or use an open reimplementation such as `mlfinpy`. Do not build a production dependency on a proprietary library whose terms you have not read.

### 3.7 Reinforcement learning: where it earns its keep, and where it is hype

**Hype:** RL for alpha generation. Framing "predict the market" as an MDP with the market as environment gives you an off-policy learning problem with a non-stationary, adversarial, partially-observed environment, reward SNR of ~0.5%, and no simulator that is faithful enough to train in. Every published "RL beats buy-and-hold on BTC" result you will encounter uses a backtest environment where the agent's actions do not affect fills. It is a supervised learning problem wearing a costume, and the supervised version is better-validated and easier to debug.

**Justified:** RL for **execution and inventory management**, given an exogenous alpha signal. Here the setup is actually sound: the state (inventory, time remaining, spread, queue state, signal) is low-dimensional and observable; the action space (place/cancel/cross, at which offset) is small and discrete; the reward (implementation shortfall) is well-defined and immediate; and — crucially — you can build a **faithful simulator**, because the counterfactual "what if I had quoted 1 tick deeper" is answerable from historical book data with a queue model, in a way that "what if I had traded 100 BTC" is not.

**The baseline you must beat: Almgren–Chriss (2000).** Closed-form optimal liquidation trajectory under linear temporary and permanent impact with a mean-variance objective. Two things to keep straight: (a) AC gives you a *schedule* (how much to trade in each interval), not a *placement policy* (which price to quote at), and the placement problem is where crypto perp execution actually lives; (b) AC's linear permanent impact is empirically wrong — the square-root law (Bouchaud et al., *Trades, Quotes and Prices*) fits better. Use AC as the schedule skeleton and RL (or a simpler bandit/heuristic) for the placement layer on top.

**If you do RL:** use `mbt_gym` (market-making/execution gym environments built on the Cartea–Jaimungal framework) or ABIDES for a multi-agent simulator. Cartea, Jaimungal & Penalva give you the stochastic-control solutions that your RL agent must beat before you believe it. Expect PPO/SAC to need 10⁶–10⁷ simulator steps and to be highly sensitive to reward shaping. **Budget: this is a 3-month project on its own. Do not put it in the first 6 weeks.**

---

## 4. Validation & overfitting defense

This is the section that decides whether the project produces knowledge or a very expensive random number. Assume, as a prior, that **your strategy does not work** and that your job is to fail to reject that.

### 4.1 Purging and embargoing at intraday scale

Standard k-fold CV on time series leaks in two directions. Purging and embargoing (*AFML* Ch. 7) fix both:

- **Purge:** drop from the training set every observation whose *label span* [t0_i, t1_i] overlaps the test set's time range. With a triple barrier, t1_i is data-dependent — you must track it per sample, not assume a fixed horizon.
- **Embargo:** additionally drop training observations in a window immediately *after* the test set, because features have lookbacks and serial correlation carries information backward across the boundary.

**Where intraday practice diverges from the textbook.** The book's guidance (embargo ≈ 1% of the sample) is calibrated to daily bars. At 2-second bars with 60-second labels, a naive embargo of "60 seconds" is far too short, because:

1. **Feature lookbacks are long.** If your slowest feature is a 30-minute EWMA, an observation 20 minutes after the test fold still shares state with it. **Embargo ≥ max(label horizon) + max(feature lookback), with a safety multiple of 2–3.**
2. **Volatility clustering spans hours to days.** Two observations 4 hours apart in the same volatility regime are not independent; a model that has memorized "what happens in a high-vol Tuesday" will score well on the adjacent fold. The practical fix is not a longer embargo but **block CV at the day or half-day level**: your CV groups should be contiguous calendar days, not arbitrary index ranges.
3. **Intraday seasonality is a shortcut.** If folds are contiguous *hours*, the model can learn hour-of-day effects that trivially transfer. Contiguous *days* mostly avoids this.

**Recommendation:** groups = calendar days; embargo = 1 full day on each side of each test block. This is aggressive, costs you ~15% of training data, and is the correct level of paranoia.

### 4.2 Combinatorial Purged CV (CPCV)

Walk-forward gives you exactly **one** path through history, so its Sharpe has enormous sampling error and no distribution. CPCV (*AFML* Ch. 12) fixes this: split into N groups, use k groups as test in every combination, and reassemble the out-of-sample predictions into **φ = C(N,k)·k/N** distinct backtest paths. With N = 12 (months) and k = 2, that is C(12,2)=66 splits and 11 paths — so you get a *distribution* of Sharpes rather than a point estimate.

Report the **5th percentile** of the CPCV Sharpe distribution as your headline number, not the mean. If the 5th percentile is below zero, you have not demonstrated anything.

Caveats, honestly stated:

- CPCV trains on data that is chronologically *after* some test data. That is defensible for testing whether a *relationship* exists, but it does **not** test deployability. You still need a strictly forward walk-forward as the final gate. CPCV answers "is there a stable relationship here?"; walk-forward answers "would I have made money?" You need both, and they answer different questions.
- CPCV multiplies your compute by C(N,k). With N=12, k=2 and a TCN, that is 66 model fits. Budget for it; use the GBDT for the CPCV sweep and reserve the TCN for the final configuration.

### 4.3 Reference implementation — purged group CV, CPCV, DSR, PBO

```python
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import norm

# ---------- 1. Purged, embargoed CV over contiguous day-groups ----------
class PurgedGroupTimeSeriesSplit:
    """
    groups     : array of group ids (use calendar day) aligned with X rows, monotone non-decreasing
    t1         : pd.Series indexed like X, value = label END time  (from triple_barrier)
    embargo_td : pd.Timedelta applied on BOTH sides of each test block
    """
    def __init__(self, n_splits=6, embargo_td=pd.Timedelta("1D")):
        self.n_splits, self.embargo_td = n_splits, embargo_td

    def split(self, X, t0, t1, groups):
        uniq = np.unique(groups)
        folds = np.array_split(uniq, self.n_splits)
        for f in folds:
            test_mask = np.isin(groups, f)
            test_idx = np.flatnonzero(test_mask)
            lo = t0[test_idx].min() - self.embargo_td
            hi = t1[test_idx].max() + self.embargo_td
            # purge+embargo: drop any train sample whose LABEL SPAN intersects [lo, hi]
            overlap = (t1 >= lo) & (t0 <= hi)
            train_idx = np.flatnonzero(~overlap & ~test_mask)
            yield train_idx, test_idx


# ---------- 2. Combinatorial Purged CV ----------
class CombinatorialPurgedCV:
    def __init__(self, n_groups=12, k_test=2, embargo_td=pd.Timedelta("1D")):
        self.N, self.k, self.embargo_td = n_groups, k_test, embargo_td

    @property
    def n_paths(self):
        from math import comb
        return comb(self.N, self.k) * self.k // self.N

    def split(self, t0, t1, groups):
        uniq = np.unique(groups)
        blocks = np.array_split(uniq, self.N)
        for combo in combinations(range(self.N), self.k):
            test_groups = np.concatenate([blocks[c] for c in combo])
            test_mask = np.isin(groups, test_groups)
            test_idx = np.flatnonzero(test_mask)
            keep = np.ones(len(t0), dtype=bool)
            # purge around EACH contiguous test block independently
            for c in combo:
                g = blocks[c]
                bi = np.flatnonzero(np.isin(groups, g))
                lo, hi = t0[bi].min() - self.embargo_td, t1[bi].max() + self.embargo_td
                keep &= ~((t1 >= lo) & (t0 <= hi))
            train_idx = np.flatnonzero(keep & ~test_mask)
            yield train_idx, test_idx, combo


# ---------- 3. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ----------
EULER = 0.5772156649015329

def expected_max_sharpe(sr_trials):
    """E[max SR] under the null that all N trials have true SR = 0."""
    N = len(sr_trials)
    v = np.var(sr_trials, ddof=1)
    return np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N)
                         + EULER * norm.ppf(1 - 1 / (N * np.e)))

def deflated_sharpe(returns, sr_trials):
    """
    returns   : per-period strategy returns (NOT annualized)
    sr_trials : per-period Sharpe of EVERY configuration you evaluated (all trials!)
    Returns P(true SR > 0 | observed SR, N trials, non-normality). Want > 0.95.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = pd.Series(r).skew()
    g4 = pd.Series(r).kurtosis() + 3.0          # non-excess kurtosis
    sr_star = expected_max_sharpe(sr_trials)
    num = (sr - sr_star) * np.sqrt(T - 1)
    den = np.sqrt(1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr ** 2)
    return norm.cdf(num / den), sr, sr_star


# ---------- 4. Probability of Backtest Overfitting (CSCV) ----------
def pbo_cscv(perf_matrix, S=16):
    """
    perf_matrix : (T, n_configs) matrix of per-period returns, one column per configuration.
    Splits time into S blocks, forms all C(S, S/2) train/test partitions,
    picks the in-sample best config, and measures its out-of-sample rank.
    PBO = P(best-in-sample config ranks below median out-of-sample). Want < 0.2.
    """
    T, n = perf_matrix.shape
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for tr in combinations(range(S), S // 2):
        te = [b for b in range(S) if b not in tr]
        tr_i = np.concatenate([blocks[b] for b in tr])
        te_i = np.concatenate([blocks[b] for b in te])
        sr_is = perf_matrix[tr_i].mean(0) / (perf_matrix[tr_i].std(0, ddof=1) + 1e-12)
        sr_oos = perf_matrix[te_i].mean(0) / (perf_matrix[te_i].std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (np.argsort(np.argsort(sr_oos))[best] + 1) / (n + 1)   # relative rank in (0,1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.asarray(logits)
    return float((logits <= 0).mean()), logits
```

### 4.4 The trial count is the number you will be tempted to lie about

DSR requires N = the number of configurations you *evaluated*, and it must include:

- every hyperparameter setting you tried,
- every feature-set variation,
- every barrier width, horizon, and bar threshold,
- every "let me just check if it works better without the weekend data,"
- **and every trial from previous, abandoned versions of the project on the same data.**

In practice N reaches 10³–10⁴ within a month of serious work. With per-trial Sharpe std of 0.5, the expected maximum Sharpe **under the null that every trial has zero true edge** is:

| N trials | E[max SR] (per-period units) |
|---|---|
| 100 | 1.27 |
| 1,000 | **1.63** |
| 10,000 | 1.93 |

So after a thousand honest experiments, a Sharpe of 1.6 is the *expected* result of pure noise. This is why a backtest Sharpe of 2 with a thousand trials behind it is worth approximately nothing.

**A calibration exercise worth running yourself** (the §4.3 code, on synthetic data): draw 500 pure-noise strategies of 2,000 periods each, take the best, and compute its DSR — you get ≈ 0.29, correctly rejected. Now take a strategy with a **genuine** per-period edge of 0.05σ and compute its DSR against those same 500 trials — you get ≈ **0.79, which fails a 0.95 gate.** Read that twice. A real edge, correctly measured, is *rejected* because of how many things you tried. Multiple testing does not merely inflate false positives; it destroys your ability to recognize true positives. The only fix is to try fewer things, and to try them for reasons you can articulate in advance.

**Operational defense: instrument your trial count automatically.** Log every model fit — config hash, CV Sharpe, timestamp — to a SQLite/MLflow store from day one. You cannot reconstruct N honestly from memory, and self-reported N is always low by 5–10×.

**The stronger defense is a held-out lockbox.** Physically separate the most recent 2 months of data before you begin. Do not look at it. Do not compute a single statistic on it. Touch it exactly once, at the end, and if it fails, the project fails — you do not get to iterate. This is the only defense that does not depend on your own honesty about N.

### 4.5 Feature-importance stability

Point-in-time feature importance is nearly useless; **stability of importance across regimes** is the diagnostic that matters.

- **MDA (mean decrease accuracy) with purged CV**, not MDI. MDI is biased toward high-cardinality continuous features and is computed in-sample. MDA must shuffle a feature *within* the purged test fold and re-score.
- **Shuffle correlated features as blocks.** OFI at 10 levels is one piece of information; shuffling them one at a time will show all ten as unimportant (substitution effect). Group features into ~10 economic clusters (order flow, book shape, vol, toxicity, funding, cross-venue, calendar, …) and run clustered MDA. *AFML* Ch. 8 calls this Clustered MDA; it is the correct default for microstructure feature sets, which are always heavily collinear.
- **SHAP across regimes.** Compute SHAP on each CPCV path separately, then measure the rank correlation of feature importance between paths. **Spearman ρ < 0.5 between regimes means your model is a different model in each regime** — either add the regime as an explicit feature, split the model, or drop the unstable features.
- **The stability test that actually predicts live performance:** does the *sign* of the marginal effect of your top-5 features stay constant across all CPCV paths? A feature whose SHAP direction flips between paths is noise you have named.

### 4.6 Alpha decay detection

Short-horizon crypto alpha decays. Instrument it from day one:

| Monitor | Definition | Action threshold *[est]* |
|---|---|---|
| Rolling IC | Spearman(prediction, realized vol-normalized return) over trailing 5 days | IC below 50% of its backtest mean for 3 consecutive days → halve size |
| IC half-life | Fit exponential to IC vs. calendar time in walk-forward | If half-life < 8 weeks, your retrain cadence must be ≤ half of it |
| Realized vs. expected edge | (realized net bps/trade) − (model-predicted bps/trade) | Persistent negative gap → adverse selection or fill-model error, not alpha decay. Diagnose separately. |
| Fill ratio | filled maker orders / placed maker orders | Sharp *rise* is a warning, not good news: it means you're being filled more often, usually because you're being picked off |
| Feature PSI | Population stability index per feature vs. train distribution | PSI > 0.25 on any top-10 feature → retrain |

**The decay test in backtest:** plot walk-forward net Sharpe by quarter. A strategy whose Sharpe is 2.5 in 2023, 1.4 in 2024, 0.6 in 2025 and 0.1 in 2026 has an average Sharpe of 1.15 and an expected forward Sharpe of approximately zero. The trend matters more than the mean. This pattern is extremely common in intraday crypto and is the honest reason most such strategies are retired rather than "discovered to have never worked."

---

## 5. Backtesting with execution realism

### 5.1 Why a vectorized bar-level backtest is not merely inaccurate — it is inverted

A vectorized backtest computes `pnl = signal.shift(1) * returns - costs` on a bar grid. At a 5 s–5 min horizon on a perp, it is wrong in ways that all point the same direction:

1. **It assumes you get filled.** At the bar close, at the mid or the close price. A passive order at the top of the book fills roughly 20–50% of the time *[est, measure it]*, and — critically — the fills are not random. You are filled when the market comes to you, which is when you were wrong.
2. **It inverts the sign of the selection.** This is the deep problem, not a rounding error. Unconditional 30 s forward return after a signal might be +5 bps. Forward return *conditional on your passive buy having been filled* might be −2 bps. The vectorized backtest reports the former. Your account experiences the latter. The gap is **adverse selection**, and on BTC perp it is comparable in magnitude to the entire fee line.
3. **It ignores queue position.** Two orders at the same price with the same signal have completely different economics depending on whether they are 5% or 95% back in the queue. You cannot even observe this on Binance L2 (§7.4), which caps how well you can model it.
4. **It ignores your own footprint.** Your order changes the book others see and react to.
5. **It uses the wrong price for exits.** Marking at mid rather than at the tradeable side overstates every exit by half the spread — which on BTC perp is only ~0.005–0.025 bps, so this one is genuinely minor here (unlike in equities). One of the few places crypto perps are *easier*: the spread is negligible, and the entire cost is fees plus adverse selection.
6. **Bar-close alignment is a look-ahead.** The bar closes at time t; you knew its close only *at* t; a fill at the close price at t is a fill at a price you learned simultaneously with acting on it.

Vectorized backtests are fine for one purpose: **fast, coarse screening of whether a feature has any predictive content at all**, before you spend a week on the event-driven version. Use them for feature triage. Never use them for a go/no-go decision.

### 5.2 Event-driven backtest at quote/tick resolution

The correct architecture is a single event loop that processes a merged, local-receipt-ordered stream of `{book_update, trade, liquidation, funding, own_order_ack, own_fill, timer}` events, where your strategy is a callback that can only see state built from events already processed.

Latency must be injected in **both** directions:

```
market event happens on exchange
   → +L_feed  (feed latency: network + 100 ms batching)      → your handler sees it
   → +L_compute (feature update + inference)                  → decision made
   → +L_order  (order network latency)                        → order reaches matching engine
   → order joins queue BEHIND everything that arrived earlier
```

Model L_feed and L_order as **distributions, not constants** — lognormal fits well, and the tail is what kills you. Run the whole backtest at p50, p90, and p99 latency and report all three. A strategy whose Sharpe collapses from 1.8 at p50 to 0.3 at p90 is a strategy you cannot deploy, because you will spend 10% of your time at p90.

### 5.3 Queue position and fill probability — the core model

Binance publishes aggregated L2 depth, so you know total quantity at a price level but not the order-by-order composition. Given that, a defensible queue model:

- On order placement at price P: `queue_ahead = depth_at_P` (observed after L_order). Optimistic variant: `0.9 × depth`; pessimistic: `depth + expected_new_arrivals`.
- **Trades at P** consume from the front. Decrement `queue_ahead` by traded volume; once `queue_ahead ≤ 0`, the remainder fills your order.
- **Cancels at P** are ambiguous. Three bounds you must run:
  - *Optimistic:* all cancels are ahead of you → `queue_ahead` decreases fully. Your fill rate is overstated.
  - *Proportional:* cancels are distributed uniformly over the queue → `queue_ahead` decreases by `cancel_vol × queue_ahead / total_depth`. This is the realistic default.
  - *Pessimistic:* all cancels are behind you → `queue_ahead` unchanged.
  Report Sharpe under all three. **If the strategy is only profitable under the optimistic cancel assumption, it does not exist.** This single test kills the majority of retail market-making backtests.
- **Price moves through your level** → you are fully filled (and it is precisely the bad case).
- **Price moves away** → you are no longer at BBO; your order rests, and you must decide whether to cancel-replace (costing you queue position and adding message-rate load) or to wait.

### 5.4 Reference implementation — passive fill simulator with latency and markouts

```python
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class PassiveOrder:
    side: int          # +1 buy, -1 sell
    price: float
    qty: float
    ts_sent: int       # ns, local decision time
    ts_live: int       # ns, ts_sent + L_order  -> when it joins the queue
    queue_ahead: float = np.nan
    filled: float = 0.0
    alive: bool = True
    observable: bool = True   # False when our level drops out of the visible depth window


class PassiveFillSimulator:
    """
    Queue-aware maker fill simulator for aggregated L2 feeds.
    cancel_model in {"optimistic", "proportional", "pessimistic"}.
    Feed events in strict LOCAL-RECEIPT order.
    """
    def __init__(self, cancel_model="proportional", maker_fee_bps=2.0, taker_fee_bps=5.0):
        self.cancel_model = cancel_model
        self.maker_fee = maker_fee_bps * 1e-4
        self.taker_fee = taker_fee_bps * 1e-4
        self.pending = deque()      # orders not yet live (in flight)
        self.live = []              # orders resting at the exchange
        self.fills = []             # (ts, side, price, qty, fee, mid_at_fill)
        self.depth_at = {}          # price -> aggregated qty, current book state
        self.unobservable_events = 0

    def send(self, order: PassiveOrder):
        self.pending.append(order)

    def _activate(self, ts):
        while self.pending and self.pending[0].ts_live <= ts:
            o = self.pending.popleft()
            o.queue_ahead = self.depth_at.get(o.price, 0.0)   # join the back of the queue
            self.live.append(o)

    def on_book(self, ts, book_levels, best_bid, best_ask):
        """book_levels: dict price -> aggregated qty (visible window, both sides), AFTER update."""
        self._activate(ts)
        mid = 0.5 * (best_bid + best_ask)
        for o in self.live:
            if not o.alive:
                continue
            # (1) Did the market trade THROUGH our level? A resting BUY at P is swept when
            #     someone is offering at or below P. Testing `price not in book_levels` here
            #     is WRONG: a level also disappears merely by falling out of the visible
            #     depth window, which must not be treated as a fill.
            through = (best_ask <= o.price) if o.side > 0 else (best_bid >= o.price)
            if through and o.filled < o.qty:
                self._fill(ts, o, o.qty - o.filled, mid, through=True)
                continue

            # (2) Level outside the visible window -> queue state genuinely unknown. Freeze it
            #     and COUNT it: a high unobservable rate means your fill model is guessing.
            if o.price not in book_levels:
                o.observable = False
                self.unobservable_events += 1
                continue
            o.observable = True

            new_d = book_levels[o.price]
            old_d = self.depth_at.get(o.price, new_d)
            delta = new_d - old_d
            if delta < 0:
                # removals not explained by trades (handled in on_trade) => cancels
                cancel_vol = -delta
                if self.cancel_model == "optimistic":
                    o.queue_ahead -= cancel_vol
                elif self.cancel_model == "proportional":
                    frac = o.queue_ahead / max(old_d, 1e-12)
                    o.queue_ahead -= cancel_vol * min(frac, 1.0)
                # pessimistic: no change
                o.queue_ahead = max(o.queue_ahead, 0.0)
        self.depth_at = dict(book_levels)

    def on_trade(self, ts, price, qty, aggressor_side, mid):
        """aggressor_side: +1 buyer-initiated (lifts ask), -1 seller-initiated (hits bid)."""
        self._activate(ts)
        for o in self.live:
            if not o.alive or o.price != price:
                continue
            # a resting BUY is filled by SELLER-initiated trades, and vice versa
            if aggressor_side == o.side:
                continue
            consumed = min(qty, o.queue_ahead)
            o.queue_ahead -= consumed
            residual = qty - consumed
            if residual > 0 and o.filled < o.qty:
                self._fill(ts, o, min(residual, o.qty - o.filled), mid)

    def _fill(self, ts, o, q, mid, through=False):
        o.filled += q
        fee = q * o.price * (self.taker_fee if through else self.maker_fee)
        self.fills.append(dict(ts=ts, side=o.side, price=o.price, qty=q,
                               fee=fee, mid=mid, through=through))
        if o.filled >= o.qty - 1e-12:
            o.alive = False

    def cancel(self, order):  order.alive = False


def markouts(fills, mid_series_ts, mid_series_px, horizons_ns=(1e9, 5e9, 30e9, 120e9)):
    """
    THE diagnostic for any maker strategy. markout(h) = side * (mid[t+h] - fill_px) / fill_px.
    A healthy passive strategy has markouts that are NEGATIVE at short h (you pay adverse
    selection) and turn POSITIVE by your holding horizon. If they are negative at every h,
    you are pure toxic flow to someone else and no amount of modeling will fix it.
    """
    out = {}
    ts = np.asarray([f["ts"] for f in fills]); side = np.asarray([f["side"] for f in fills])
    px = np.asarray([f["price"] for f in fills])
    for h in horizons_ns:
        j = np.searchsorted(mid_series_ts, ts + h)
        j = np.clip(j, 0, len(mid_series_px) - 1)
        out[int(h / 1e9)] = float(np.mean(side * (mid_series_px[j] - px) / px) * 1e4)  # bps
    return out
```

**Read the markout docstring twice.** The markout curve is the single most informative plot in short-horizon maker research, and it is the thing that vectorized backtests structurally cannot produce. Typical healthy shape for a signal-driven maker on BTC perp: markout at +1 s ≈ −0.5 to −1.5 bps (you were adversely selected on entry), recovering to +2 to +5 bps by +60 s if your alpha is real. If the curve is monotonically decreasing, stop.

### 5.5 Market impact

For passive fills at retail size, temporary impact is second-order; the binding cost is adverse selection, which is already in the markouts. For any taker leg and for capacity work, use the **square-root law**:

```
impact ≈ Y · σ_daily · √(Q / V_daily),   Y ≈ 0.5–1.0
```

(Bouchaud et al., *Trades, Quotes and Prices*, Ch. 12). Fit Y on your own fills as soon as you have live data; do not trust a literature constant on a different market. Note that the square-root law applies to *metaorders* — the aggregate of your child orders in one direction — not to individual clips.

### 5.6 Frameworks

| Option | Verdict |
|---|---|
| **nautilus_trader** | **Recommended.** Rust core with nanosecond-resolution deterministic event-driven simulation, configurable fill/fee/latency/order-book models, crypto-native adapters, and the same strategy code runs in backtest and live — which eliminates the largest class of production bugs. Actively developed (1.228.0, June 2026: [releases](https://github.com/nautechsystems/nautilus_trader/blob/develop/RELEASES.md)). **Limits:** the built-in L2 fill model is simpler than the queue model above; you will need to implement your own `FillModel` for realistic queue dynamics, and the learning curve is real (2–3 weeks to competence). |
| **Custom asyncio loop** | Justified only for the *research* backtest, where you want total control over the queue model and can run it as a tight NumPy/Numba loop over a memory-mapped event array. **Limits:** you will end up rewriting order lifecycle, position/margin accounting, and reconnect logic — and, crucially, your research loop and your live loop will diverge. If you go this route, enforce that feature computation is shared code between the two. |
| **vectorbt / backtesting.py / bt** | Bar-level and vectorized. Useful for feature triage (§5.1). Not for this. |
| **ABIDES / mbt_gym** | Multi-agent / RL simulation. Right tool for the execution sub-problem, wrong tool for alpha validation. |

**Recommended split:** custom Numba event loop for research iteration speed, `nautilus_trader` for the final validation backtest and for live/paper deployment, with a hard requirement that both produce the same P&L on the same day of data to within a tolerance you specify in advance. That reconciliation test is worth a week of work and will find at least three bugs.

---

## 6. Portfolio, risk & execution

### 6.1 Signal → position

Do not map probability to position with a threshold. Use a continuous, cost-aware map:

```python
def target_position(p_meta, side, vol_t, target_vol_ann, equity, max_notional,
                    kelly_frac=0.25):
    """
    p_meta   : meta-model P(trade is profitable), calibrated (isotonic on OOF preds)
    vol_t    : point-in-time forecast of return vol over the holding horizon
    Half-Kelly-ish sizing, then vol-targeted, then hard-capped.
    """
    edge = 2.0 * p_meta - 1.0                       # in [-1, 1]
    raw  = side * kelly_frac * edge / max(vol_t, 1e-6)
    scale = (target_vol_ann / np.sqrt(365 * 24 * 60)) / max(vol_t, 1e-6)
    return float(np.clip(raw * scale * equity, -max_notional, max_notional))
```

Then apply a **no-trade band** rather than trading to target continuously:

```python
def rebalance(current, target, band_bps, price, min_clip):
    gap = target - current
    if abs(gap) < band_bps * 1e-4 * abs(target or 1.0) or abs(gap) < min_clip * price:
        return 0.0                                  # do nothing; turnover is the enemy
    return gap
```

The no-trade band is not a detail. With a 4 bps round-trip cost and a 6 bps σ, the optimal policy under quadratic transaction costs is a **band around a moving target**, not tracking. Set the band width by grid-searching *net* Sharpe — and count that grid search in your trial count N.

### 6.2 Turnover and the fee-drag reality check

Compute this before you write any model code, with your own assumptions:

```
annual fee drag (bps of capital) = trades_per_day × 365 × cost_per_RT_bps × (avg_notional / equity)
```

At 100 RT/day, 4 bps maker/maker, and notional = 1× equity:
`100 × 365 × 4 = 146,000 bps = 1,460% of capital per year in fees.`

You are not paying that on capital, you are paying it on turnover — but the point stands starkly: **at 100 round trips a day you are churning 100× your capital daily, and your gross edge must exceed 4 bps per trade just to break even.** Since σ(1 min) is 6.2 bps, that means your model must capture ~65% of a one-minute standard deviation, per trade, on average. Write that sentence on a wall. It is the reason most of these projects fail, and it is why the honest design target is *fewer, better trades* (20–60/day at 1–3 min holding) rather than more.

### 6.3 Intraday vol targeting and risk limits

- **Vol target:** scale gross exposure by `target_vol / forecast_vol` using an intraday vol forecast (EWMA of TSRV with a half-life of ~30 min), floored and capped at [0.25×, 2.5×]. Crypto vol moves fast enough that a daily-updated vol target is too slow.
- **Inventory limits:** hard cap on absolute position, plus a *time-weighted* cap — being flat is the default state, and any position held > 3× the intended horizon should be closed at market. Stale inventory is where seconds-horizon strategies die.
- **Drawdown governor:** three tiers — (i) intraday soft stop at −1.5× daily vol → halve size; (ii) intraday hard stop at −3× → flat and stop for the day; (iii) rolling 20-day stop at −2× monthly vol → flat and require manual restart with a written post-mortem. Automate all three; do not leave them to judgment at 3 a.m.
- **Funding-stamp policy:** be flat, or deliberately positioned, at 00/08/16 UTC. Never accidentally positioned.
- **Liquidation risk:** use ≤ 3× effective leverage even though the exchange permits far more, and monitor against **mark price**, not last price. A 5-minute-horizon strategy has no business being anywhere near a liquidation price.
- **Kill switches on infrastructure, not just P&L:** websocket gap detected, sequence number break, clock drift > 50 ms, order reject rate > 1%, feed staleness > 2 s, position mismatch between local and exchange state → flat immediately, alert, do not auto-restart.

### 6.4 Maker vs. taker placement

| Situation | Placement |
|---|---|
| Signal strong, vol low, spread 1 tick | Post at BBO. Accept queue risk. |
| Signal strong, adverse-selection flags hot (cross-venue divergence, high VPIN, recent liquidation) | Post 1–3 ticks *inside your favor* (deeper), or don't trade |
| Signal moderate, need to exit inventory | Post at BBO with a time-based escalation: after T₁ reprice, after T₂ cross |
| Emergency (risk limit, kill switch) | Cross immediately. Never optimize execution during a risk event. |
| Entry always | **Maker.** If you must be taker on entry, the signal is not strong enough to be worth 5 bps. |
| Exit | Maker with taker escalation. Budget ~20–35% taker exits *[est]* and put that in the cost model from day one. |

**Order splitting:** at retail size on BTC perp you are rarely large relative to BBO depth, so splitting is mostly about *queue* management, not impact — placing several smaller orders across 2–3 price levels gives you a fill-probability profile instead of a binary outcome, at the cost of more messages. Watch Binance's order rate limits; hitting them mid-session is a real operational failure mode.

### 6.5 Capacity — the honest section

Estimate with the square-root law. Take BTCUSDT perp daily notional volume ≈ $15B *[est — measure it]* and σ_daily = 45%/√365 = 2.36%. A $10M metaorder in one direction over a day implies impact ≈ 0.7 × 2.36% × √(10M/15B) ≈ **4.3 bps** — i.e. it consumes your entire gross edge.

For a strategy at this horizon the binding constraints are, in order:
1. **BBO depth per clip.** BTCUSDT perp BBO depth is roughly 2–10 BTC ($200k–$1M) *[est]*. Clips above ~15% of BBO depth start visibly changing the book and inviting reaction. → **max clip ≈ $30k–$150k.**
2. **Daily gross turnover before impact bites.** Roughly **$2M–$10M/day** *[est]*.
3. **Deployed capital.** At 10–30× daily turnover, that is **$100k–$1M** of capital.

So the honest capacity conclusion: **if this works, it produces a high percentage return on a small absolute capital base and does not scale past low single-digit millions.** That is not a bug — it is precisely why the opportunity is available to you at all. A $500M fund cannot deploy meaningfully into it, so they don't compete for it. It is also why you should be suspicious of anyone selling you a scalable version.

And the base rate: **most projects in this scope produce a strategy with a true net Sharpe indistinguishable from zero.** The realistic distribution of outcomes for a well-executed 6-week project is roughly: 55% no deployable edge found *[est]*, 30% marginal edge (net Sharpe 0.3–0.8, not worth the operational risk), 12% modest real edge (0.8–1.8), 3% strong (>1.8, and expect it to decay within 6–18 months). Plan the project so that the 55% outcome still leaves you with reusable infrastructure and a validated research process — because that is the modal result.

---

## 7. Infrastructure & production

### 7.1 Latency budget

Target class: **decision-to-ack ≈ 120–200 ms**, dominated by the exchange's own feed batching, not by your code.

| Stage | p50 | p99 | Notes |
|---|---|---|---|
| Exchange event → depth stream emission | ~50 ms | 100 ms | Mean wait for the 100 ms diff-depth batch. **Irreducible.** `aggTrade` is per-event and much faster — exploit the asymmetry. |
| Network exchange → your VM | 4 ms | 13 ms | AWS `ap-northeast-1`; no Binance colocation exists ([measurement](https://deltixworld.com/measuring-websocket-data-feed-latency.html)) |
| WS parse + book apply | 20 µs | 200 µs | Rust/C++ or numba; naive Python `json.loads` will cost 200 µs+ — use `orjson` or a Rust parser |
| Feature update (incremental) | 50 µs | 500 µs | Must be O(1) per event. Any O(window) recomputation is a design bug. |
| Model inference | 0.3 ms (LGBM) / 2 ms (TCN, CPU) | 1 / 8 ms | Batch size 1. GPU is *slower* than CPU here due to transfer overhead. |
| Risk checks + order build | 50 µs | 300 µs | |
| Order → exchange, ack | 15 ms | 60 ms | WS order entry < REST. Use it. |
| **Total decision loop** | **~70 ms** | **~180 ms** | |

Design implications: (a) your compute budget is generous relative to the feed batching, so **spend it on better features rather than on micro-optimization** — going from 2 ms to 200 µs inference changes nothing; (b) the one place latency genuinely matters is the trade tape, which is unbatched, so trade-reactive features are worth more than book-reactive ones; (c) p99 matters much more than p50 — a 180 ms tail during a volatility burst is exactly when your orders are stale.

**Do not** attempt to run this from a laptop on residential internet. 50–150 ms of jitter to Tokyo will dominate everything and make live/backtest reconciliation impossible.

### 7.2 Online vs. offline features — the parity problem

The #1 production bug in this domain is **training/serving skew**: the feature vector computed offline from historical files differs subtly from the one computed online from the live stream. Sources: different event ordering, different NaN handling at warm-up, different clock, a `.shift(1)` in pandas that has no analogue in the streaming path.

**Architecture that prevents it:**

```
                    ┌─────────────────────────────┐
  historical replay │                             │  live websocket
  (Tardis CSV/API)  │      SAME feature engine    │  (Binance/Bybit)
        │           │   (one codebase, one class, │        │
        └──────────►│    event-driven, O(1)/event)│◄───────┘
                    └──────────────┬──────────────┘
                                   │  same FeatureVector struct
                    ┌──────────────┴──────────────┐
              training parquet              live inference
```

Write the feature engine **once**, as an event-driven class with `on_book()`, `on_trade()`, `on_funding()`, `snapshot()` (see §2.5). Historical mode is just replaying the same events from files. Never write a "vectorized version for training" — that is where skew is born.

**Parity test as CI:** replay one live-recorded session through the offline path, and assert that every feature matches the values logged live to within 1e-9. Run it on every commit. This test will pay for itself in the first month.

**Do not build a feature store** (Feast et al.) for this. Feature stores solve point-in-time correctness for *batch* features shared across many models with day-scale latency. Your features are millisecond-scale and single-model. A feature store adds a network hop, a serialization boundary, and a new class of skew bug. Use in-process state.

### 7.3 Model serving, monitoring, and retraining

**Serving.** In-process, no network hop. LightGBM: use the C API or `treelite`/compile to shared object (~10× faster than the Python predict path). PyTorch: TorchScript or ONNX Runtime, CPU, `torch.set_num_threads(1)` to avoid thread-pool jitter. Load the model once at start; version it; **log the model hash with every prediction** so you can attribute live P&L to a specific artifact.

**Monitoring** (all of these, from day one — retrofitting monitoring after a bad week is how people lose money twice):

| Category | Metric | Alert |
|---|---|---|
| Data | WS gap count, sequence breaks, feed staleness, clock drift vs NTP | any gap → flatten |
| Feature | PSI per feature vs. training distribution; NaN rate; feature latency p99 | PSI > 0.25 on a top-10 feature |
| Model | prediction distribution KS vs. OOF; calibration (predicted vs realized win rate, bucketed) | KS p < 0.01 |
| Alpha | rolling 5-day IC; realized bps/trade vs. predicted | IC < 50% of backtest mean, 3 days |
| Execution | fill ratio; **markout curve at 1/5/30/120 s**; realized vs. modeled slippage; taker fraction | markout at 30 s turns negative |
| Risk | position vs. limit; leverage vs. mark price; drawdown tiers; funding paid | any tier breach |
| Ops | order reject rate, rate-limit headroom, reconnect count, event-loop lag | reject rate > 1% |

**Retrain triggers.** Scheduled weekly retrain as the baseline; force an off-cycle retrain on: PSI breach, IC breach, an exchange-side change (fee tier, tick size, stream cadence, contract spec), or a market structural break (a > 6σ day). **Every retrain must pass the same CPCV + lockbox gates as the original model**, and the new model must beat the incumbent on a *fresh* holdout, not on the CV that selected it. Shadow-run the new model alongside the old for 3–5 days before switching. Never hot-swap a model into live trading on the strength of a CV score alone.

### 7.4 Region, connectivity, and the retail/pro divide

- **Region:** AWS `ap-northeast-1` (Tokyo) for Binance; verify current best region for Bybit/OKX by measuring, not by reading. Expect 4–13 ms. There is no colocation product for Binance, so this is the floor available to anyone.
- **Redundancy:** two websocket connections to different endpoints, deduplicated by sequence number. Reconnect logic that resyncs the book from a REST snapshot per the documented procedure, and that **marks the strategy flat-only until the book is verified**.
- **Time:** chrony against a local stratum-1; alert on drift > 5 ms. Log both exchange and local timestamps for every event, forever — you will need them to diagnose lead-lag questions later.
- **Storage:** write every raw event to local NVMe as length-prefixed binary or Parquet, then ship to S3. Do not rely on being able to re-buy history; and your own capture is the only source with *your* receipt timestamps.

**Where you structurally cannot compete (state it plainly):**

| Capability | Pro desk | You |
|---|---|---|
| Market data | Colocated / direct cross-connect; some venues offer order-by-order | Public WS, 100 ms batched L2 aggregate |
| Queue position | Inferable from MBO + own-order acks | **Unobservable.** Estimated at best. |
| Fees | Negotiated / market-maker programs, maker rebates possible | VIP 0 unless volume is very high |
| Latency | sub-ms | 70–200 ms |
| Order rate | Elevated limits via MM agreements | Standard rate limits |
| Signal | Cross-venue, cross-asset, client flow, options surface | Public data only |

The honest reading: you are locked out of the 0–2 s band entirely and you cannot make money from spread capture. What remains is the **10 s–3 min directional band with passive execution**, at small capacity, where the alpha is real but thin and the operational discipline required is disproportionate to the money involved. Go in knowing that.

---

## 8. Deliverable: full pipeline

**Latency class: ~70 ms p50 / ~180 ms p99 decision loop. Signal half-life must exceed ~2 s; design target 20 s–2 min.**

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ INGEST (live)                              INGEST (historical)                 │
│  Binance USDⓈ-M WS:                        Tardis.dev:                         │
│   btcusdt@depth@100ms  (L2 diff)            incremental_book_L2, trades,       │
│   btcusdt@aggTrade     (per-event)          derivative_ticker (funding/OI/     │
│   btcusdt@forceOrder   (liquidations)       index/mark), liquidations           │
│   btcusdt@markPrice@1s (funding/index)      → local_timestamp is the clock      │
│  + Bybit / OKX equivalents (cross-venue)                                        │
│  ↓ sequence-gap assertion, book resync, raw capture → NVMe → S3 (Parquet)       │
└────────────────────────────────────────────────────────────────────────────────┘
                    ↓  merged, LOCAL-RECEIPT-ORDERED event stream
┌────────────────────────────────────────────────────────────────────────────────┐
│ FEATURE ENGINE  — ONE codebase, event-driven, O(1)/event, CI parity-tested      │
│   book state machine → MultiLevelOFI, integrated OFI (PC1), micro-price,        │
│   queue dynamics, spread, TSRV/bipower vol, true-sign trade flow, VPIN,         │
│   Kyle λ, Amihud, funding/basis/OI, liquidation flow, cross-venue spread,       │
│   cyclical calendar.  snapshot() → FeatureVector                                │
└────────────────────────────────────────────────────────────────────────────────┘
       ↓ (offline: dollar bars @ ~2–5 s median)          ↓ (online: on demand)
┌───────────────────────────────────────┐        ┌──────────────────────────────┐
│ LABELING (offline only)               │        │ INFERENCE (in-process)       │
│  micro-price path → triple barrier    │        │  TCN embedding (TorchScript) │
│  ±1.5σ_t, T=60s, exits at bid/ask     │        │       ↓                      │
│  → primary labels                     │        │  LGBM primary → side, p      │
│  → meta-labels (net of 4 bps)         │        │       ↓                      │
│  → t0, t1 spans                       │        │  LGBM meta → p_act           │
│  → uniqueness × return-attr × decay   │        │       ↓ isotonic calibration │
│    sample weights                     │        │  size = f(p_act, σ_t, inv)   │
└───────────────────────────────────────┘        └──────────────────────────────┘
       ↓                                                   ↓
┌───────────────────────────────────────┐        ┌──────────────────────────────┐
│ TRAINING                              │        │ EXECUTION                    │
│  TCN trunk (causal, 256 steps)        │        │  no-trade band → clip sizing │
│    → 32-d embedding                   │        │  maker at BBO; escalate:     │
│  + 80 engineered features             │        │   T1 reprice → T2 cross      │
│  → LGBM primary (num_leaves=15)       │        │  risk gates, kill switches   │
│  → LGBM meta (act/don't act)          │        │  WS order entry              │
│  weights = uniqueness×attr×decay      │        └──────────────────────────────┘
└───────────────────────────────────────┘                  ↓
       ↓                                          ┌──────────────────────────────┐
┌───────────────────────────────────────┐         │ MONITORING                   │
│ VALIDATION                            │         │  markouts 1/5/30/120s        │
│  PurgedGroupCV (day blocks, 1d embargo│         │  rolling IC, PSI, calibration│
│  CPCV N=12 k=2 → 11 paths → SR dist   │         │  fill ratio, reject rate     │
│  report 5th percentile SR             │         │  drawdown tiers              │
│  DSR (log ALL trials), PBO < 0.2      │         │  → retrain triggers          │
│  clustered MDA + SHAP stability       │         └──────────────────────────────┘
│  event-driven backtest × 3 cancel     │                   ↓
│    models × 3 latency percentiles     │         ┌──────────────────────────────┐
│  → LOCKBOX: last 2 months, touch once │         │ WEEKLY RETRAIN → shadow 3–5d │
└───────────────────────────────────────┘         │ → CPCV+lockbox gate → swap   │
                                                  └──────────────────────────────┘
```

---

## 9. Deliverable: recommended stack

| Layer | Choice | Why / caveat |
|---|---|---|
| **Historical LOB** | **Tardis.dev** — `incremental_book_L2`, `trades`, `derivative_ticker`, `liquidations` for `binance-futures:BTCUSDT`. Budget 6–12 months. | Deepest retail LOB history; provides `local_timestamp` (their receipt) alongside exchange time, which is what makes causal replay possible. Subscription tiers gate how much history you can access — quarterly billing gives 12 months, monthly gives 4 ([docs](https://docs.tardis.dev/faq/billing-and-subscriptions)). Alternatives: Kaiko and Amberdata (institutional pricing, better coverage/support); Binance's own free `data.binance.vision` dumps (trades and 1s klines, **no book depth** — not sufficient). |
| **Live feed** | Native Binance USDⓈ-M websockets (`@depth@100ms`, `@aggTrade`, `@forceOrder`, `@markPrice@1s`) | Use native SDKs or a thin Rust/Python client. **Avoid `ccxt` in the hot path** — it normalizes across exchanges at the cost of latency and hides the sequence-number semantics you need for correct book reconstruction. `ccxt` is fine for account/REST operations. |
| **Own capture** | Length-prefixed binary or Parquet on NVMe → S3, from day one | Your receipt timestamps; irreplaceable later. |
| **Tick processing** | **polars** (lazy, streaming, Arrow-backed) + **numba** for the event loops | polars handles 10⁹-row tick files that pandas cannot. Use `scan_parquet` + streaming collect. |
| **Streaming bus** | **Redpanda** (Kafka API, single binary, no ZooKeeper) — *only if* you split processes | For a single-instrument, single-model system, in-process is faster and simpler. Add the bus when you have 3+ instruments or need to fan out to multiple consumers. Do not add it in week 1. |
| **DL** | **PyTorch** + TorchScript/ONNX Runtime for serving | TCN first, DeepLOB as literature baseline. CPU inference; batch size 1. |
| **GBDT** | **LightGBM** (+ `treelite` compilation for serving) | The production model. CatBoost as a cross-check. |
| **Fin-ML utilities** | Implement from *AFML* directly (§1.6, §4.3 cover most of it), or **`mlfinpy`** | **`mlfinlab` is proprietary, all-rights-reserved** ([license](https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt)) — do not build a dependency on it without reading the terms. |
| **Backtester** | **nautilus_trader** for final validation + live, with a **custom `FillModel`** implementing §5.3; custom numba loop for research iteration | Same strategy code backtest→live is the killer feature. Reconcile the two engines on one day of data as a gate. |
| **Drift detection** | **`river`** (ADWIN, Page-Hinkley, PSI) | Use as retrain *trigger*, not as the model. |
| **Experiment tracking** | **MLflow** or a plain SQLite table — but log **every** fit | This is your trial counter N for the DSR. Non-negotiable. |
| **RL (later)** | **`mbt_gym`** for execution; **ABIDES** for multi-agent | Only after a stable supervised strategy exists. |
| **Paper trading** | **Binance USDⓈ-M Testnet** for plumbing correctness; then **small real capital on Binance mainnet** for anything about P&L | Testnet's book is not the real book — it validates order lifecycle, reconnects, rate limits, and position accounting, and tells you **nothing** about fills or adverse selection. The only honest paper trade is a tiny real one ($100–500 clips). Budget for it. |
| **Compute** | AWS `ap-northeast-1`: `c7i.2xlarge` for live; a single `g5.xlarge` or local RTX for TCN training | Training is not the bottleneck; CPCV sweeps are. Parallelize the 66 GBDT fits across cores. |

---

## 10. Deliverable: the concrete first project (6 weeks)

### 10.1 Specification

| Item | Value |
|---|---|
| Instrument | `binance-futures:BTCUSDT` perpetual (ETHUSDT as out-of-instrument robustness check) |
| Data | Tardis.dev `incremental_book_L2` (10 levels), `trades`, `derivative_ticker`, `liquidations`. **8 months total: 6 months research + 2 months LOCKBOX (untouched).** Plus own live capture from week 1. |
| Bars | Dollar bars, threshold set so median duration ≈ 3 s; parallel 1 s time grid for evaluation only |
| Label price | Stoikov micro-price (g(I,S) fit on train folds only) |
| Primary target | Triple barrier, u = l = 1.5·σ_t (σ_t = EWMA of bar returns, half-life 500 bars), vertical barrier T = 60 s, evaluated on the tick path at **bid/ask**, not mid |
| Meta target | Binary: did the primary-side trade clear **4 bps** round-trip (maker/maker) net? |
| Features | ~80: multi-level OFI + integrated OFI at 5 lookbacks; micro-price family; book shape L1–L10; queue dynamics; spread; true-sign trade flow; TSRV + bipower + jump; VPIN; Kyle λ; Amihud; funding/premium/time-to-stamp; basis z-score; OI Δ; liquidation flow; Bybit+OKX cross-venue spread; 4 cyclical calendar features. **Plus a 32-d TCN embedding.** |
| Sample weights | avg uniqueness × return attribution × exp time decay (half-life 6 weeks) |
| Models | TCN trunk (256 steps, dilations 1–64, 64 channels) → embedding; LGBM primary (num_leaves=15, lr=0.01, min_data_in_leaf=2000, monotone constraints on OFI); LGBM meta; isotonic calibration on OOF |
| Validation | PurgedGroupCV (day blocks, 1-day embargo) for tuning; **CPCV N=12, k=2 → 66 splits, 11 paths**; DSR with logged trial count; PBO via CSCV (S=16); clustered MDA + cross-path SHAP stability |
| Backtest | Event-driven, tick resolution, 3 cancel models × 3 latency percentiles = 9 runs; markout curves at 1/5/30/120 s mandatory |
| Deploy gate | 5th-percentile CPCV net Sharpe > 0.5, PBO < 0.2, DSR > 0.95, profitable under the **proportional** cancel model at **p90** latency, markout positive at 60 s, and lockbox passes on first and only touch |

### 10.2 Week-by-week

**Week 1 — Data & the feature engine.** Tardis subscription; download 8 months; build the event-driven book state machine with sequence-gap assertions; stand up own live capture in `ap-northeast-1` from day 1 (you'll want 6 weeks of your own timestamps by the end). Write the feature engine as one class. Write the offline/online parity CI test. *Deliverable: replay 1 month, assert zero sequence gaps, feature engine runs at > 200k events/sec.*

**Week 2 — Bars, labels, and the noise floor.** Dollar bars; micro-price fit; triple barrier on the tick path; uniqueness/attribution weights. Then the single most important sanity check of the project: **compute the empirical distribution of barrier outcomes and the gross edge available under perfect foresight of the label.** If a perfect-foresight strategy on your labels, after 4 bps costs and a realistic fill model, yields less than ~3× your target Sharpe, your label design is wrong and no model will save it. *Deliverable: labeled dataset + perfect-foresight ceiling.*

**Week 3 — Baselines, honestly.** In order: (i) always-flat; (ii) a single-feature linear rule on integrated OFI; (iii) LGBM on engineered features only, purged CV. Each must be scored through the **event-driven** backtest, not vectorized. Most of your final performance will come from step (iii); the deep model in week 4 will add less than you expect. Log every fit to the trial store. *Deliverable: LGBM baseline with purged-CV net Sharpe and a full markout curve.*

**Week 4 — TCN + meta-labeling.** Train the TCN trunk on the 3-class micro-price target; extract embeddings via purged CV (embeddings for fold k must come from a model trained without fold k — this is a leak most people commit); concatenate to engineered features; refit primary; build the meta-model on out-of-fold primary predictions; calibrate. *Deliverable: full ensemble, OOF meta-AUC, calibration plot.*

**Week 5 — Adversarial validation.** CPCV (66 fits) → Sharpe distribution → report the 5th percentile. DSR with the true trial count from your log. PBO via CSCV. Clustered MDA. SHAP rank correlation across the 11 CPCV paths. Backtest across 3 cancel models × 3 latency percentiles. Repeat the entire pipeline **unchanged** on ETHUSDT. *Deliverable: a decision — kill or proceed. Expect to kill. That is a successful week.*

**Week 6 — Lockbox, then paper.** Run once on the 2 held-out months. No tuning, no second attempt. If it passes: deploy to Binance testnet for order-lifecycle correctness (reconnects, rate limits, position reconciliation, kill switches), then to mainnet at $100–500 clips. Measure live markouts and fill ratios against the backtest's. *Deliverable: a live/backtest reconciliation report — the number that matters is the gap, not the P&L.*

### 10.3 Realistic expectations

Anchor on these. If you beat them substantially, look for the bug first.

| Metric | Realistic range *[est]* | Red flag |
|---|---|---|
| Primary 3-class accuracy (balanced barriers) | 38–48% | > 55% |
| Meta-model OOS AUC | 0.52–0.58 | **> 0.62 ⇒ leak** |
| Gross edge per trade | 3–7 bps | > 12 bps |
| Round-trip cost (maker/maker + ~30% taker exits) | 4.5–5.5 bps | — |
| **Net edge per trade** | **0 – 2 bps** | > 4 bps |
| Trades/day | 20–60 | > 200 (fee-dominated by construction) |
| Avg holding time | 45 s – 3 min | < 20 s |
| Fill ratio (maker) | 25–45% | > 70% ⇒ you're being picked off |
| Markout at +1 s | −0.5 to −1.5 bps | positive ⇒ fill model is wrong |
| Markout at +60 s | +1 to +4 bps | > +8 bps |
| Gross daily turnover | 20–60× deployed capital | — |
| **CPCV net Sharpe, median** | **0.7 – 1.6** | > 3 |
| **CPCV net Sharpe, 5th percentile** | **0.0 – 0.7** | — |
| **Walk-forward net Sharpe (the number to believe)** | **0.4 – 1.2** | > 2.5 |
| **Live Sharpe after decay** | **backtest × 0.4–0.6** | — |
| Max drawdown | 3–8× daily P&L vol | < 2× ⇒ backtest is too smooth to be real |
| Capacity (gross notional/day) | $2M – $10M | — |
| Deployable capital | $100k – $1M | — |

Two calibration notes. First, the **live-vs-backtest haircut of 40–60%** is not pessimism, it is the empirical norm for short-horizon strategies; it comes from fill-model optimism, latency tails, and the fact that your presence changes the book. Second, a **backtest Sharpe of 6 is not six times better than a Sharpe of 1** — it is a different kind of object, namely evidence of a bug. Re-read §0, Conclusion 4.

---

## 11. The 12 references that matter, and why

**Books (read in this order)**

1. **López de Prado, *Advances in Financial Machine Learning* (2018).** Chapters 2–8 and 11–14 are the operational spine of this entire report: event bars, triple-barrier labeling, meta-labeling, sample uniqueness and return-attribution weights, sequential bootstrap, purged/embargoed CV, CPCV, clustered feature importance, DSR/PBO. Everything else you read will assume you have internalized this. *Its weakness:* the examples are daily-frequency equities and the intraday adaptations (§4.1 here) are yours to make.

2. **Bouchaud, Bonart, Donier & Gould, *Trades, Quotes and Prices* (2018).** The empirical physics of order books: long-memory in order flow, the square-root law of impact, the order-flow/price-impact relationship, why linear-impact models (including Almgren–Chriss) are wrong in a specific and quantifiable way. This is the book that tells you *why* OFI works and what the capacity limit really is. Read Ch. 3–5 (LOB statistics) and Ch. 11–13 (impact).

3. **Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading* (2015).** The stochastic-control counterpart: closed-form optimal market-making with inventory penalties (Avellaneda–Stoikov and successors), optimal execution, and the correct formulation of the inventory-risk trade-off. These closed forms are the baselines any RL execution agent must beat before you believe it. Ch. 8–10.

**Papers — features and price formation**

4. **Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events," *Journal of Financial Econometrics*.** Establishes that mid-price changes are approximately linear in OFI with slope inversely proportional to depth, and that OFI dominates trade imbalance because it captures placements and cancellations. **If you implement one feature from this report, implement OFI.** It is the highest signal-to-implementation-effort item in the entire microstructure literature.

5. **Cont, Cucuringu & Zhang (2023), "Cross-Impact of Order Flow Imbalance in Equity Markets."** Extends OFI to multiple book levels and shows that integrating levels via PCA ("integrated OFI") materially improves explanatory power over L1 OFI. In crypto, where BBO depth is thin relative to L2–L10, this matters more than it does in equities. Directly implemented in §2.5.

6. **Sirignano & Cont (2019), "Universal features of price formation in financial markets: perspectives from deep learning," *Quantitative Finance*.** Three findings that shape your architecture: the price-formation relationship is *universal* across instruments (so train across instruments and fine-tune), *nonlinear* (so the LOB is not adequately summarized by linear OFI alone), and *stationary over time* in a way individual instruments are not. This is the strongest published justification for using deep learning on LOB data at all.

7. **Stoikov (2018), "The micro-price: a high-frequency estimator of future prices," *Quantitative Finance*.** Defines the micro-price as a martingale-corrected weighted mid, estimated via a Markov chain on (imbalance, spread). Fixes the naive weighted-mid's overshoot. Matters because **labeling on the wrong price is the most common silent error in short-horizon research** (§1.3–1.4).

**Papers — architectures and their limits**

8. **Zhang, Zohren & Roberts (2019), "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books," *IEEE Trans. Signal Processing*.** The canonical LOB architecture and the FI-2010 benchmark result. Read it for the *inductive bias* (convolve within a level, then across levels, then in time), which transfers to crypto. Use FI-2010 exactly once, to verify your reimplementation, and then discard it — 10 days of 5 Nordic small caps with smoothed labels tells you nothing about BTC perp.

9. **Prata et al. (2024/2025), "Deep limit order book forecasting: a microstructural guide," *Quantitative Finance* — with the open-source [LOBFrame](https://github.com/FinancialComputingUCL/LOBFrame) pipeline.** The adversarial paper, and the one most likely to save you money. Shows that DL forecasting power depends heavily on an instrument's microstructural characteristics, that standard ML metrics (accuracy, F1) badly misrepresent forecast quality in the LOB setting, and that **high forecasting accuracy does not translate into actionable trading signals**. Read it *before* you write model code, not after.

**Papers — toxicity, and its critique**

10. **Easley, López de Prado & O'Hara (2012), "Flow Toxicity and Liquidity in a High-Frequency World," *Review of Financial Studies* — read together with Andersen & Bondarenko (2014), "VPIN and the Flash Crash," *Journal of Financial Markets*.** The first proposes VPIN as a real-time order-flow-toxicity measure; the second shows its predictive power is largely mechanical, driven by trailing volatility and trade intensity. **Read them as a pair.** The lesson generalizes far beyond VPIN: a feature that correlates with your target because it is a noisy transform of another feature you already have is not a feature. Always include the obvious control.

**Papers — validation and execution**

11. **Bailey & López de Prado (2014), "The Deflated Sharpe Ratio," *Journal of Portfolio Management* — with Bailey, Borwein, López de Prado & Zhu (2014), "The Probability of Backtest Overfitting," *Journal of Computational Finance*.** DSR corrects an observed Sharpe for the number of trials, the non-normality of returns, and sample length; PBO/CSCV estimates the probability that your in-sample-best configuration underperforms out-of-sample. Together they are the only rigorous defense against multiple-testing bias, and they are the reason §4.4 insists you instrument your trial count automatically. Implemented in §4.3.

12. **Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions," *Journal of Risk*.** The mean-variance optimal trading trajectory under linear temporary and permanent impact. Required as the baseline any execution model must beat, and required for understanding what a *schedule* is versus what a *placement policy* is. Its linear-impact assumption is empirically wrong (see #2) — knowing precisely how it is wrong is more valuable than the closed form itself.

**Supporting (short, worth an hour each):** Roll (1984) on the bid-ask bounce and the MA(1) structure it induces in trade-price returns (§1.3); Zhang, Mykland & Aït-Sahalia (2005) on two-scale realized volatility (§2.3); Barndorff-Nielsen & Shephard (2004) on bipower variation and jump separation.

---

## 12. Standing checklist: how you will know you are fooling yourself

Run through this before believing any result.

- [ ] Are labels computed on micro-price or mid, never on trade prices? (§1.3)
- [ ] Are barrier touches evaluated on the tick path at **bid/ask**, not on bar closes at mid? (§1.5)
- [ ] Is every feature computed by an event-driven state machine in **local-receipt order**, with no `merge_asof` between trades and book? (§2.0)
- [ ] Are cross-venue features aligned on a single collector's clock, not on exchange timestamps? (§2.0b)
- [ ] Do TCN embeddings used in fold *k* come from a model that never saw fold *k*? (§10.2 wk4)
- [ ] Is the meta-model trained on **out-of-fold** primary predictions? (§3.3)
- [ ] Are sample weights (uniqueness × attribution × decay) actually passed to the fitter? (§3.6)
- [ ] Is the CV embargo ≥ max label horizon + max feature lookback, with day-level blocks? (§4.1)
- [ ] Is the reported Sharpe the **5th percentile** of the CPCV distribution, not the mean? (§4.2)
- [ ] Is the trial count N logged automatically, including abandoned experiments? (§4.4)
- [ ] Is the lockbox still untouched? Has it been touched exactly once, at the end?
- [ ] Is the strategy profitable under the **proportional** (not optimistic) cancel model? (§5.3)
- [ ] Is it profitable at **p90** latency, not just p50? (§5.2)
- [ ] Is the markout curve negative at +1 s and positive at +60 s? (§5.4)
- [ ] Does the net Sharpe survive when you add 30% taker exits to the cost model? (§6.4)
- [ ] Is the walk-forward Sharpe **stable across quarters**, or is it decaying? (§4.6)
- [ ] Does the identical, untuned pipeline produce a positive result on ETHUSDT? (§0)
- [ ] **Is the backtest Sharpe below 5?** If not, you have a bug. (§0, Conclusion 4)

---

*Research and education only. Nothing here is financial advice, and nothing here should be taken as a claim that the described strategy will be profitable. Trading leveraged perpetual futures can lose more than the capital deployed. All figures marked [est] are the author's order-of-magnitude estimates and must be re-measured on your own data before they are relied upon. Exchange fee schedules, tick sizes, stream cadences, and funding mechanics change — verify every one against current exchange documentation before use.*
