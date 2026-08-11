# Deployment Runbook & Go-Live Gates

**Read this before pointing anything at real money.** This system has **never traded a live
market**. It is production-*shaped* (streaming feature parity, execution manager with kill
switches / OPS limit / Algo-ID / forced flatten, reconciliation, and a hard safety guard),
but "deployment ready" is **not a state the code alone can reach** — it is gated on data, a
broker, regulatory onboarding, testing, and capital that are yours to supply.

The default deployment mode is `paper`. Real order routing is **physically blocked** unless
you set `deploy.mode: live` *and* export `BSEALPHA_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK`. Do
not remove that guard.

---

## Readiness status

| Layer | Status | Blocker |
|---|---|---|
| Feature engine (batch + streaming, parity 1e-12) | ✅ code-complete, tested | — |
| Labeling / weights / models / validation | ✅ code-complete, tested | — |
| Backtest (Indian constraints) + reconciliation | ✅ code-complete, tested | — |
| Execution manager (OPS / Algo-ID / kill switches / flatten) | ✅ code-complete, tested vs PaperBroker | — |
| Deployment safety guard (paper/dry_run/live) | ✅ code-complete, tested | — |
| **Real historical depth data (train the model)** | ❌ | **paid vendor (TrueData/GDFL)** |
| **A model with a validated edge** | ❌ | **research gates must pass on real data** |
| **Live broker adapter (order routing)** | ❌ interface only | **your account + SDK + testing** |
| **SEBI algo onboarding + Algo-ID** | ❌ | **broker onboarding, weeks** |
| **Capital** | ❌ | **you** |

You are **not** deployment ready until every ❌ is ✅. The green rows are necessary, not
sufficient.

---

## The hard prerequisites (none are code)

1. **Data** — 3–5 years of BSE 1-min bars **+ 5-level depth** from TrueData/GDFL. Free
   sources (yfinance) give ~1 week of bars and **no depth** — enough for a mechanics demo,
   not a strategy.
2. **A validated model** — run the research pipeline on real data and pass the gates:
   universe ≥ ~80 names, perfect-foresight ceiling PASS, effective breadth ≥ 3, CPCV 5th-pct
   Sharpe > 0.5, DSR > 0.95, PBO < 0.2, and the **lockbox passes on its single touch**. The
   base rate is ~50% that no edge is found — that is a valid outcome, not a failure (§11.3).
3. **Broker + API + SEBI onboarding** — a broker account, an implemented `BrokerAdapter`,
   Algo-ID provisioning, and confirmation your design stays under **10 orders/second** (§8.1).
4. **Infrastructure** — an always-on VM (AWS `ap-south-1`), `chrony` clock sync (<5 ms
   drift), own-capture of every event to Parquet from day one, monitoring + alerting.
5. **Capital** — start at ₹5k–25k clips. Only real orders reveal fills and adverse
   selection (§10).

---

## Go-live sequence (do not skip a stage)

1. **Backtest + reconcile** on real data (`run_research.py` on your ingested panel). Confirm
   the deploy gates above. **Seal the lockbox.**
2. **Dry-run** (`deploy.mode: dry_run`): run the live loop against the live feed; the manager
   computes and logs the orders it *would* send but routes nothing. Run for days. Verify the
   **offline/online feature parity test** on your own capture (features must match live to
   1e-9, §8.3).
3. **Broker sandbox**: validate plumbing only — order acks, reconnect/resync, position
   reconciliation, ≤10 OPS, kill switches, forced flatten. Sandbox fills are fiction (§10).
4. **Preflight**: `DeploymentGuard.preflight(...)` must return `passed=True` (model loaded,
   universe screened today, Algo-ID present, kill switches armed, clock synced, lockbox
   passed).
5. **Tiny real capital** (`deploy.mode: live` + env confirmation): ₹5k–25k clips. Measure
   **real fill ratio, real markouts, real slippage**; produce the live-vs-backtest
   reconciliation — the **gap** is the number, not the P&L. Expect a 40–60% haircut.
6. **Scale only after** a live track record and a stable gap. Capacity ceiling is ~₹5 cr
   book before impact eats the edge (§7.4).

---

## Daily operations

- **Pre-open (09:00–09:12)**: rebuild the point-in-time universe; hard-check today's
  T2T/ASM/GSM/suspension/circuit changes; confirm feed + clock + Algo-ID; size down.
- **Session**: monitor OPS < 7/s, realized beta/sector exposure, PSI, 15-min markout, rolling
  IC, effective breadth, and **positions-open-after-15:25 (must be zero)**.
- **15:15 → 15:28**: forced flatten (LIMIT → MARKET escalation) — automatic; verify flat.
- **Any kill-switch breach** (feed staleness / gap / reject rate / clock drift / position
  mismatch): flatten + halt, **no auto-restart**; restart only via `RiskMonitor.reset(...)`
  with a written post-mortem.
- **Post-session**: reconcile live vs backtest; log the trial; **never touch the lockbox**.

---

## Retraining & monitoring

- Retrain monthly on a trailing 2–3 years with decayed weights; force an off-cycle retrain on
  a PSI/IC breach, a fee/tick/surveillance change, or a >6σ day. Every retrain passes the
  **same CPCV + lockbox gates** and beats the incumbent on *fresh* data, then shadow-runs 5
  sessions before switching. Never hot-swap on a CV score (§8.4).
- Recalibrate the probability→size mapping (isotonic) weekly; retrain the model monthly.

---

## The honest bottom line

The engineering is done and tested to the boundary of what code can verify. Everything past
that boundary — data, edge, broker, onboarding, capital — is external and is where the real
risk and cost live. Treat a green test suite as permission to *start* the go-live sequence,
not as permission to trade.
