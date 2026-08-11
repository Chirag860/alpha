# MT5 Setup Guide — running `bsealpha` on a MetaTrader 5 demo account

This guide takes you from a fresh Mac to the strategy placing (paper) orders on an MT5 demo
account, step by step, with nothing assumed. Read it top to bottom the first time.

> **What you are setting up.** `bsealpha` is a cross-sectional stock strategy. It was built for
> BSE (India) but has been *repointed* onto a basket of **stock CFDs** traded through MetaTrader 5.
> Phase 1 (this guide) gets the machinery running and routing orders to a **demo** account —
> **no real money, ever**. Tuning the model to actually make money is Phase 2 and is out of scope
> here (see [What Phase 1 does and does not do](#12-what-phase-1-does-and-does-not-do)).

---

## Table of contents

1. [How it works (and why you need Windows)](#1-how-it-works-and-why-you-need-windows)
2. [Prerequisites checklist](#2-prerequisites-checklist)
3. [Part A — Create the Windows environment](#3-part-a--create-the-windows-environment)
4. [Part B — Install & open MetaTrader 5](#4-part-b--install--open-metatrader-5)
5. [Part C — Install Python and the code on the VM](#5-part-c--install-python-and-the-code-on-the-vm)
6. [Part D — Confirm MetaTrader5 connects](#6-part-d--confirm-metatrader5-connects)
7. [Part E — Configure `config/mt5.yaml`](#7-part-e--configure-configmt5yaml)
8. [Part F — Find your server timezone offset (critical)](#8-part-f--find-your-server-timezone-offset-critical)
9. [Part G — Run it](#9-part-g--run-it)
10. [Troubleshooting (every failure mode)](#10-troubleshooting-every-failure-mode)
11. [Config field reference](#11-config-field-reference)
12. [What Phase 1 does and does not do](#12-what-phase-1-does-and-does-not-do)
13. [Daily operation](#13-daily-operation)

---

## 1. How it works (and why you need Windows)

```
   ┌──────────────────────── Windows VM (or VPS) ────────────────────────┐
   │                                                                      │
   │   MetaTrader 5 terminal  ◄── IPC ──►  Python  (bsealpha + your code) │
   │   (logged into demo)                    │                            │
   │        ▲                                 │  1. mt5_export.py  → Parquet history
   │        │ orders / fills / bars           │  2. run_mt5.py     → train + trade loop
   │        └─────────────────────────────────┘                          │
   └──────────────────────────────────────────────────────────────────────┘
```

* The official **`MetaTrader5` Python package is Windows-only.** It talks to a **running MT5
  terminal** on the same machine over local IPC. There is no macOS/Linux build. That is why
  everything that touches MT5 must run on Windows.
* In this project, **all MT5-touching code lazily imports `MetaTrader5`** — so the rest of the
  package still installs and its tests still pass on macOS/Linux. But the two runnable scripts
  (`bsealpha/data/mt5_export.py` and `run_mt5.py`) **must be run on the Windows machine.**
* For Phase 1, **everything runs on the VM**: the export *and* the training+trading loop. Your
  Mac is only needed if you also want to run the unit tests or develop the code.

---

## 2. Prerequisites checklist

Before starting you need:

- [ ] A Mac (Apple Silicon or Intel) with **~15 GB free disk** and **8 GB+ RAM** (the VM needs
      at least 4 GB allocated).
- [ ] A way to run Windows 11 (see Part A). Options: **Parallels Desktop** (easiest on Apple
      Silicon, paid), **VMware Fusion** (free for personal use), **UTM** (free), or a cheap
      **Windows VPS** (fallback if the VM is painful).
- [ ] Internet on the VM.
- [ ] ~1 hour for first-time setup.

You do **not** need: a broker account application, any payment, an API key, or SEBI/Algo-ID
onboarding. A MetaQuotes demo account is created for free inside the terminal in 2 minutes.

---

## 3. Part A — Create the Windows environment

You need a working **Windows 11 (x64 apps)** desktop. Pick one path.

### Option 1 — Parallels Desktop (recommended on Apple Silicon)
1. Install **Parallels Desktop** from parallels.com.
2. Parallels → **File → New… → “Get Windows 11 from Microsoft”**. It downloads and installs
   Windows 11 automatically. On Apple Silicon this is **Windows 11 ARM**, which runs x64 apps
   (including MT5) through built-in x64 emulation — this works fine for our purposes.
3. Give the VM **4 GB+ RAM** and **2+ CPUs** (Parallels → VM → Configure → Hardware).
4. Enable a **Shared Folder** if you want to edit code from the Mac (Configure → Options →
   Sharing → “Share Mac folders with Windows”). Optional — you can also just `git clone` on the VM.

### Option 2 — VMware Fusion (free) / UTM (free)
1. Install VMware Fusion (free personal license) or UTM.
2. Create a **Windows 11** VM. On Apple Silicon, download the **Windows 11 ARM** ISO from
   Microsoft; on Intel Macs use the **Windows 11 x64** ISO.
3. Allocate 4 GB+ RAM, 2+ CPUs.

### Option 3 — Windows VPS (fallback, always-on)
If a local VM is troublesome, rent a small **Windows Server / Windows 10-11 x64 VPS**
(any low-cost provider). This is also the better choice if you later want the loop to run 24/5
without keeping your Mac awake. Everything below is identical once you can RDP into Windows.

> **Apple Silicon note (important):** later, install the **x64 build of Python** on the VM (Part C),
> not the ARM64 build. The `MetaTrader5` package only ships **x64 wheels**, so it must match an
> x64 Python interpreter. Windows 11 ARM runs x64 Python transparently under emulation.

---

## 4. Part B — Install & open MetaTrader 5

Do all of this **inside Windows**.

### B1. Install the terminal
1. In the Windows browser go to **https://www.metatrader5.com/en/download** and download the
   Windows installer.
2. Run it, accept defaults. It installs to
   `C:\Program Files\MetaTrader 5\terminal64.exe` (remember this path).
3. Launch **MetaTrader 5**.

### B2. Open a free demo account (choose NETTING)
1. On first launch it may prompt to open an account. Otherwise: **File → Open an Account**.
2. Broker/server: search and select **“MetaQuotes-Demo”** (the built-in free demo). Click Next.
3. Choose **“Open a demo account”**. Fill in name/email.
4. **Account type — pick a NETTING account if offered.** Netting = one position per symbol,
   which maps cleanly to the strategy’s signed positions. (If only *Hedging* is available, it
   still works — the adapter sums hedged legs defensively — but netting is cleaner.)
5. Set a comfortable **deposit** (e.g. **50,000 USD**) and **leverage** (1:20 or 1:30 is fine
   for stock CFDs). Click Next.
6. It shows your **Login**, **Password**, and **Server** (e.g. `MetaQuotes-Demo`). **Write all
   three down** — you will paste them into `config/mt5.yaml`. (You can re-open them via
   **File → Login to Trade Account**.)

### B3. Enable algorithmic trading (Python cannot place orders without this)
1. **Tools → Options → Expert Advisors** tab → tick **“Allow algorithmic trading”** → OK.
2. On the main toolbar, the **“Algo Trading”** button must be **ON** (it turns green / shows a
   play icon). Click it once if it’s off.

### B4. Make sure history is deep enough
1. **Tools → Options → Charts** tab → set **“Max bars in chart”** to **Unlimited** (or a very
   large number). This lets Python pull months of M1 bars. → OK.

### B5. Add stock CFDs to Market Watch
1. Right-click the **Market Watch** panel → **Symbols** (or press **Ctrl+U**).
2. Expand the tree and find a **“Stocks”** / **“Shares”** / **“US Stocks”** group. If there is
   no such group, this server has few/no stock CFDs — see the breadth warning in Part G / the
   note in the README (you may need a stock-heavy broker demo).
3. Select a bunch of stocks and click **Show**, or just **Show All** in the stocks group. (The
   export script also enables symbols programmatically, but showing them here lets you eyeball
   what exists.)

> **Leave the terminal running and logged in** the entire time you run the Python scripts.

---

## 5. Part C — Install Python and the code on the VM

Do all of this **inside Windows**.

### C1. Install Python 3.11 (x64)
1. Download **Python 3.11 (Windows installer, 64-bit)** from python.org.
   - Use **3.11** (not 3.13/3.14): it has prebuilt wheels for `MetaTrader5`, `lightgbm`,
     `polars`, and `torch`, so nothing needs a compiler.
   - On Apple-Silicon VMs make sure this is the **x64** installer (default from python.org).
2. In the installer, tick **“Add python.exe to PATH”**, then Install.
3. Verify in **Command Prompt (cmd)** or **PowerShell**:
   ```
   python --version
   ```
   It should print `Python 3.11.x`.

### C2. Get the project onto the VM
Pick one:
- **Git:** install Git for Windows, then in a folder of your choice:
  ```
  git clone <your-repo-url> alpha
  cd alpha
  ```
- **Shared folder / copy:** copy the whole `alpha` project folder into Windows (via the
  Parallels shared folder, a USB, or a zip). Open a terminal and `cd` into it.

You should now be inside the project folder (the one containing `run_mt5.py`, `bsealpha\`,
`config\`).

### C3. Create a virtual environment and install dependencies
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install MetaTrader5
```

> **PowerShell blocks `Activate.ps1`?** If `.venv\Scripts\activate` errors with *“running
> scripts is disabled on this system”* (`UnauthorizedAccess`), that’s PowerShell’s execution
> policy, not a venv problem — the `.venv` was created fine. Either allow scripts once:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # answer Y, then re-run activate
> ```
> **or** skip activation entirely and call the venv’s Python directly (most robust):
> ```powershell
> .venv\Scripts\python.exe -m pip install --upgrade pip
> .venv\Scripts\python.exe -m pip install -r requirements.txt
> .venv\Scripts\python.exe -m pip install MetaTrader5
> ```
> Then run every later command the same way, e.g.
> `.venv\Scripts\python.exe run_mt5.py --config config/mt5.yaml`.
> (The `create` command needs a space: `python -m venv .venv`, not `venv.venv`.)

Notes:
- `requirements.txt` already includes `numpy, pandas, scipy, polars, pyyaml, lightgbm,
  scikit-learn, torch, pytest`. `MetaTrader5` is installed separately (it is Windows-only, so it
  is intentionally **not** in `requirements.txt`).
- **Apple-Silicon VM (x64 emulation) — swap in `polars-lts-cpu`.** Because Windows-on-ARM runs
  x64 Python under emulation, the standard `polars` wheel (built for modern Intel SIMD) crashes
  on import with `RuntimeError: unknown feature flag: 'sse3'` (or a SIGILL). Replace it with the
  LTS-CPU build, which is compiled for older/emulated CPUs and imports identically:
  ```powershell
  .venv\Scripts\python.exe -m pip uninstall -y polars
  .venv\Scripts\python.exe -m pip install polars-lts-cpu
  ```
  (If `lightgbm` later throws an *illegal instruction* for the same reason, it will simply fall
  back to scikit-learn — the pipeline still runs.)
- **Windows does NOT need `libomp`/`brew`** — the `lightgbm` Windows wheel bundles OpenMP. (The
  `brew install libomp` note in the README is macOS-only.)
- If `pip install MetaTrader5` fails, your Python is almost certainly the wrong version/arch —
  see [Troubleshooting](#10-troubleshooting-every-failure-mode).

---

## 6. Part D — Confirm MetaTrader5 connects

Before configuring anything, prove Python can see the terminal. With the **terminal open and
logged into your demo**, run this one-liner in the activated venv:

```
python -c "import MetaTrader5 as m; print('init', m.initialize()); print(m.version()); print(m.account_info()); m.shutdown()"
```

Expected: `init True`, a version tuple, and your account info (login, balance, server). If you
see `init False`, or `None` for account info, jump to
[Troubleshooting → initialize/login](#a-initialize-or-login-fails). Do not proceed until this
works.

---

## 7. Part E — Configure `config/mt5.yaml`

Open `config\mt5.yaml` in a text editor (Notepad works). This file is a **partial config** that
is deep-merged on top of `config/default.yaml` by the MT5 scripts. Fill in the `mt5:` block:

```yaml
mt5:
  server: "MetaQuotes-Demo"     # EXACTLY the server name from Part B2 (case-sensitive)
  login: 12345678               # <-- your demo LOGIN number (an integer, no quotes)
  password: "YourDemoPassword"  # <-- your demo password
  terminal_path: ""             # usually leave blank; if init fails, set the full path:
                                #   "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
  timeframe: "M1"               # keep M1 (1-minute bars)
  discover: true                # auto-find stock CFDs from the terminal
  symbol_group: "*Stock*"       # discovery filter (see note below)
  symbols: []                   # only used if discover: false
  symbol_suffix: ""             # broker suffix on tickers, if any (e.g. ".NAS", "#")
  max_symbols: 200              # cap the universe size
  data_dir: "data/mt5"          # where exported Parquet is written/read
  sector_map_path: "config/us_sectors.yaml"
  history_start: "2024-01-01"   # earliest date to export
  server_tz_offset_hours: 3     # SERVER time minus UTC — SET THIS (see Part F)
  netting: true
```

Key points:
- **`login`** is an integer (the number MT5 gave you). **`password`** and **`server`** are
  strings. If your password contains special characters, keep the quotes.
- **`symbol_group`** is a substring/wildcard used by `symbols_get`. `"*Stock*"` matches paths
  like `Stocks\AAPL`. If your broker groups them differently (e.g. `Shares\...` or
  `Equities\...`), change it to `"*Share*"` / `"*Equit*"` / `"*"` (everything — then rely on
  `max_symbols`). You’ll confirm what matches in Part G step 1.
- **`symbol_suffix`**: many brokers append a suffix to the ticker (e.g. `AAPL.NAS`, `#AAPL`).
  If yours does, put it here so the sector map (keyed by the bare ticker) still matches.
  MetaQuotes-Demo usually has **no** suffix — leave `""`.
- **`server_tz_offset_hours`** is the single most error-prone field. **Do Part F to set it.**

> **Keep your credentials out of git.** `config/mt5.yaml` now holds your login/password. It’s a
> throwaway demo (low risk), but add the file to `.gitignore` — or keep a credential-free
> `config/mt5.example.yaml` and pass your real one with `--config` — before committing anything.

You normally do **not** edit the other blocks in `config/mt5.yaml` (they already switch the
market profile to `us_equity`, set `deploy.mode: demo`, zero the Indian taxes, and scale the
book to USD). See [Config field reference](#11-config-field-reference) if you want to.

---

## 8. Part F — Find your server timezone offset (critical)

MT5 timestamps its bars in the **broker’s server timezone**, not UTC. The exporter converts
those timestamps to **US Eastern** to compute the session-relative minute (0 = 09:30 ET). If the
offset is wrong, every bar lands in the wrong minute and the session filter throws away your data
(or keeps the wrong bars). MetaQuotes-Demo is usually **UTC+2 in winter, UTC+3 in summer (DST)**.

**Measure it precisely — don’t guess:**

1. In the terminal, hover the **Market Watch** clock, or look at the time of the newest tick of a
   liquid symbol. That time is **server time**.
2. Compare it to the **current UTC time** (search “UTC time now”).
3. `server_tz_offset_hours = server_time − UTC_time` (in hours). E.g. if the server clock reads
   18:00 and UTC is 15:00, the offset is **3**.

**Then verify against the export (Part G step 2):** after exporting, the first bar of any US
trading day should be at **minute 0** (09:30 ET). The export/verify snippet below checks this. If
the first minute is not ~0, adjust `server_tz_offset_hours` by the difference (in units of 60
minutes) and re-export.

---

## 9. Part G — Run it

All commands run **inside the activated venv, in the project folder, with the terminal open**.

### Step 1 — Discover the universe (breadth check). Do this first.
```
python -m bsealpha.data.mt5_export --config config/mt5.yaml --discover-only
```
- It connects, lists the matched stock CFDs, and prints the count.
- **Read the count.** The cross-sectional strategy needs **100+** correlated names to be
  meaningful. If it prints a small number (MetaQuotes-Demo often has few stock CFDs), the
  strategy will be near-degenerate. Options:
  - broaden `symbol_group` (try `"*Share*"`, `"*Equit*"`, or `"*"`), or
  - open a demo with a **stock-heavy broker** (e.g. **Admirals** ~3000, **Pepperstone** ~1000
    stock CFDs), then update `server`/`login`/`password` and re-run.
- If it lists **zero**, your `symbol_group` filter doesn’t match — inspect the symbol tree
  (Ctrl+U in the terminal) to see the real group name and fix the filter.

### Step 2 — Export history, then verify the timezone
```
python -m bsealpha.data.mt5_export --config config/mt5.yaml
```
This writes `data\mt5\grid.parquet`, `data\mt5\meta.parquet`, and `data\mt5\symbol_map.json`,
and prints how many bars/names/sessions it exported.

Now **verify the session mapping** (paste into `python`):
```python
import polars as pl
g = pl.read_parquet("data/mt5/grid.parquet")
# first minute of each day should be ~0 (09:30 ET) and the max ~389 (15:59)
print(g.group_by("date").agg(pl.col("minute").min().alias("min"),
                             pl.col("minute").max().alias("max")).sort("date").head())
```
- If `min ≈ 0` and `max ≈ 389`, your `server_tz_offset_hours` is correct.
- If `min` is a constant nonzero (say 60), your offset is off by `60/60 = 1` hour — adjust
  `server_tz_offset_hours` and re-export.
- If a day has very few rows, the market may have been closed/half-day, or history is shallow
  (raise “Max bars in chart”, Part B4).

### Step 3 — Dry run (build orders, send nothing)
```
python run_mt5.py --config config/mt5.yaml --dry-run
```
- Trains the model in-sample on the exported history (takes a minute), connects, and enters the
  loop. With `--dry-run` (`deploy.mode: dry_run`) it **computes and logs orders but never sends
  them**. Each completed minute it prints a line like:
  ```
  [min  42] phase=normal orders=18 rejects=0.00% halted=False
  ```
- Let it run for a few minutes **during US market hours** (09:30–16:00 ET) so new bars arrive.
  Outside those hours there are no new bars and it will idle — that’s expected.
- Stop with **Ctrl+C**.

### Step 4 — Live demo run (routes to the demo account)
```
python run_mt5.py --config config/mt5.yaml
```
- Same as above but `deploy.mode: demo` — it **sends orders to your demo account**. You’ll see
  positions and orders appear in the terminal’s **Toolbox → Trade** tab.
- It builds a market-/sector-/beta-neutral book each decision minute, and **flattens everything
  at 15:55 ET** (the forced-flatten deadline), fully flat by 15:59.
- **This is a demo — no real money.** Watch the printed `orders`/`rejects`/`halted` and the
  terminal’s Trade tab. Stop anytime with **Ctrl+C**.

**Reading the loop output:**
| Field | Meaning |
|---|---|
| `min` | session-relative minute (0 = 09:30 ET, 385 = 15:55 flatten) |
| `phase` | `normal` → trading; `flatten`/`escalate`/`hard` → forced end-of-day unwind |
| `orders` | orders routed this minute |
| `rejects` | broker reject rate so far (should stay ~0%; high = a config/symbol problem) |
| `halted` | a risk kill-switch tripped (feed stale / too many rejects / position mismatch) → the loop flattens and stops |

---

## 10. Troubleshooting (every failure mode)

### A. `initialize` or `login` fails
- **Terminal not running / not logged in.** Open MT5 and log into the demo first.
- **`initialize()` returns False:** set `mt5.terminal_path` in `config/mt5.yaml` to the full
  path, e.g. `"C:\\Program Files\\MetaTrader 5\\terminal64.exe"` (double backslashes).
- **`login` fails:** re-check `login` (integer, no quotes), `password`, and `server` spelled
  **exactly** as shown in **File → Login to Trade Account**. Server names are case-sensitive.
- **Multiple terminals installed:** `initialize` may attach to the wrong one — set
  `terminal_path` explicitly.

### B. `pip install MetaTrader5` fails / `import MetaTrader5` errors
- You’re on the **wrong Python version or architecture.** Use **x64 Python 3.11**. On Apple
  Silicon VMs, an **ARM64** Python has no MetaTrader5 wheel — reinstall the **x64** build from
  python.org. Verify: `python -c "import platform; print(platform.architecture(), platform.python_version())"`
  → should show `('64bit', ...)` and `3.11.x`.

### C. `--discover-only` lists zero or very few symbols
- **Filter mismatch:** open the symbol tree (**Ctrl+U**) and read the real group name; set
  `symbol_group` to match (`"*Stock*"`, `"*Share*"`, `"*Equit*"`, or `"*"`).
- **Thin server:** MetaQuotes-Demo genuinely has few stock CFDs. Broaden the filter or switch to
  a stock-heavy broker demo (Admirals/Pepperstone). This is the expected weak point — the
  cross-section needs breadth.

### D. Export runs but writes 0 bars, or the verify shows wrong minutes
- **Timezone offset wrong:** re-do [Part F](#8-part-f--find-your-server-timezone-offset-critical).
  If the first minute of a day is a constant `N`, change `server_tz_offset_hours` by `round(N/60)`
  hours and re-export.
- **Shallow history:** raise **Max bars in chart** to Unlimited (Part B4) and re-export; MT5
  downloads M1 history on demand, so also open a chart of a few symbols once to prime it.
- **`history_start` in the future / too recent:** set it to an older date like `2024-01-01`.

### E. Orders are rejected (high `rejects` in the loop)
Check the order’s `retcode` (the adapter counts non-DONE/PLACED as rejects). Common MT5 retcodes:
| retcode | meaning | fix |
|---|---|---|
| 10027 | AutoTrading disabled by client | turn ON the **Algo Trading** toolbar button (Part B3) |
| 10017 | Trade disabled | the account/symbol can’t trade now |
| 10018 | Market closed | run during US market hours (09:30–16:00 ET) |
| 10014 | Invalid volume | lot below the symbol’s min or off the step — the adapter drops sub-min clips as `TOO_SMALL`; if you still see this, raise `portfolio.min_clip` or `gross_target` |
| 10019 | No money | raise the demo deposit, or lower `portfolio.gross_target` |
| 10015/10016 | Invalid price/stops | usually transient; check `symbol_suffix`/price digits |

### F. `TOO_SMALL` / no orders placed
- Your clips round below the symbol’s **minimum lot**. Increase `portfolio.gross_target` and/or
  `portfolio.min_clip` in `config/mt5.yaml`, or reduce `portfolio.max_names` so each name gets a
  bigger clip.

### G. Positions look doubled / don’t net out
- Your demo is a **hedging** account. The adapter sums hedged legs, but for clean behavior open a
  **netting** demo account (Part B2, step 4).

### H. The loop idles and prints nothing
- It’s **outside US market hours** — no new completed minutes. Run 09:30–16:00 ET on a weekday.

### I. `lightgbm` errors on the VM
- The Windows wheel bundles OpenMP; you should not need anything extra. If import fails, the
  pipeline **falls back to scikit-learn automatically** — training still runs, just a bit weaker.

### K. `polars ... RuntimeError: unknown feature flag: 'sse3'` (or illegal instruction on import)
- x64-on-ARM emulation (Apple-Silicon VM) doesn’t expose the SIMD flags the default `polars`
  wheel needs. Swap it for the LTS-CPU build:
  ```powershell
  .venv\Scripts\python.exe -m pip uninstall -y polars
  .venv\Scripts\python.exe -m pip install polars-lts-cpu
  ```

### J. `Activate.ps1 cannot be loaded because running scripts is disabled`
- PowerShell’s execution policy, not a venv error. Fix once with
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (answer **Y**), then
  re-run activate — **or** don’t activate at all and call `.venv\Scripts\python.exe` directly for
  every command. See the boxed note in [Part C step C3](#c3-create-a-virtual-environment-and-install-dependencies).

---

## 11. Config field reference

Everything the MT5 path reads from `config/mt5.yaml` (merged over `config/default.yaml`):

| Key | Meaning |
|---|---|
| `market.profile` | `us_equity` — selects US session hours (09:30–16:00 ET) & penny ticks. Do not change for US stocks. |
| `deploy.mode` | `demo` routes to the demo account; `dry_run` computes but never sends; `paper`/`live` are the BSE modes. `--dry-run` on `run_mt5.py` forces `dry_run`. |
| `execution.flatten_start_min` / `escalate_min` / `hard_flat_min` | Forced-unwind schedule in ET minutes-of-day (955/957/959 = 15:55/15:57/15:59). |
| `execution.min_clip` | Skip trades below this notional (USD). |
| `execution.algo_id` | Stamped on orders (becomes the MT5 “magic number”); MT5 accepts and ignores it. |
| `costs.*` | US-CFD cost profile: STT/stamp/GST/brokerage zeroed; `bse_txn_bps` = round-trip commission proxy; `impact_Y` keeps the square-root impact model. |
| `portfolio.gross_target` | Total gross book size (USD). Tune to your demo balance. |
| `portfolio.min_clip` / `max_names` / `sector_cap` / `max_participation` | Book construction limits. |
| `backtest.equity` | Notional base for return scaling (USD). |
| `mt5.server` / `login` / `password` / `terminal_path` | Connection. |
| `mt5.discover` / `symbol_group` / `symbols` / `symbol_suffix` / `max_symbols` | Universe selection. |
| `mt5.timeframe` | Bar timeframe (keep `M1`). |
| `mt5.data_dir` | Where Parquet + `symbol_map.json` live. |
| `mt5.sector_map_path` | Ticker→sector YAML (`config/us_sectors.yaml`); unknown tickers get `UNKNOWN`. |
| `mt5.history_start` | First date to export. |
| `mt5.server_tz_offset_hours` | Broker server tz minus UTC. **Set per Part F.** |
| `mt5.netting` | Documents the netting-account requirement (informational). |

To grow sector coverage, add `TICKER: SECTOR` lines to `config/us_sectors.yaml` (keys are the
**bare** ticker, without any `symbol_suffix`).

---

## 12. What Phase 1 does and does not do

**Phase 1 (this guide) proves the mechanics:**
- Ingests real MT5 stock-CFD bars into the pipeline’s canonical format.
- Runs the full US-session engine (features → labeling → model → neutral book).
- Routes lot-rounded orders to the demo account through all the risk/rate-limit/forced-flatten
  gates, and flattens at the ET deadline.

**Phase 1 does NOT (this is Phase 2):**
- **Make money.** `run_mt5.py` trains the model **in-sample** on the exported history at startup
  (like `run_free.py`) — good enough to see orders flow, **not** a validated edge.
- **Validate the alpha** on the new universe (out-of-fold training, CPCV/DSR, a research lockbox,
  a serialized model artifact, per-name beta estimation). Treat any P&L you see as noise.

So: use Phase 1 to confirm the plumbing end-to-end on your demo. Don’t read anything into the
returns.

---

## 13. Daily operation

- **Start:** open MT5 (logged into the demo, Algo Trading ON), activate the venv, run
  `python run_mt5.py --config config/mt5.yaml` during US market hours.
- **Refresh history** periodically by re-running `python -m bsealpha.data.mt5_export
  --config config/mt5.yaml` (e.g. daily before the open).
- **Stop:** Ctrl+C. The strategy also self-flattens at 15:55 ET each day.
- **Keep the terminal open** while the loop runs; if the terminal closes or the feed goes stale,
  the risk kill-switch halts the loop and flattens.
- For always-on running, use a **Windows VPS** (Part A, Option 3) so you don’t have to keep the
  Mac/VM awake.

---

*Questions this guide can’t answer usually come down to: is the terminal open and logged in, is
Algo Trading ON, is Python x64 3.11, and is `server_tz_offset_hours` right? Check those four
first.*
