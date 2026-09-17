# 📈 LLM Portfolio Manager

An automated, end-of-day **stock screener** that blends classic quantitative
technical analysis, news sentiment, and **LLM reasoning** (NVIDIA NIM) to produce
short-term **BUY / WATCH / NO PICK** decisions — with stop zones, price targets,
risk/reward, and a "devil's advocate" for every pick.

> ⚠️ **Not financial advice.** This is a personal learning project. Past screener
> performance does not guarantee future results. Always do your own research.

---

## 🧠 The Core Idea

The design philosophy is **"math filters, the LLM judges."**

1. Cheap, deterministic scoring runs over **375+ stocks** to rank them objectively.
2. Only the **top 30 candidates** are sent to the (expensive) LLM for reasoning.

This keeps the run essentially free and reproducible, while still getting
qualitative judgment on the finalists.

```
375+ stocks ──► Deterministic pre-score (0–100) ──► Top 30 ──► LLM reasoning ──► BUY/WATCH/NO PICK
                 (technicals + news, in code)                   (NVIDIA NIM)      (+ stops, targets, R:R)
```

---

## ✨ What It Does (Daily)

Every evening after US market close, the screener:

- Downloads OHLCV for **375+ stocks** in a single batched API call.
- Runs **bidirectional screening** — technical filters **and** news-catalyst rescue in parallel.
- Computes a **deterministic pre-score** from: RSI, MACD, ADX, CMF, StochRSI, VWAP, OBV,
  options put/call ratio, insider flow, and VADER NLP sentiment.
- Feeds the **top 30 candidates** to the LLM with full context: macro headlines,
  sector rotation, earnings risk, and self-calibration from past picks.
- Returns **BUY / WATCH / NO PICK** with stop zones, price targets, R:R ratio,
  a devil's advocate counter-argument, and a full score breakdown.
- Saves HTML / CSV / JSON reports and **auto-tracks 10-day and 30-day returns** over time.

**Runtime:** ~7–9 minutes &nbsp;|&nbsp; **Cost per run:** ~$0.00 (NVIDIA NIM free tier)

---

## 🏗️ How the Scoring Works

### Total Pre-Score = Technical (0–60) + News (0–40) = **0–100**

The 60/40 split is deliberate: **technicals dominate** (objective, predictive for
short-term trades), while **news/sentiment modifies**.

#### Technical Score — 0–60 (five equal 12-point buckets + sector bonus)

| Component | Max | Rewards |
|-----------|----:|---------|
| RSI regime fit | 12 | RSI in the ideal band *for the current market regime* (bull vs bear differ) |
| Volume conviction | 12 | Volume ratio, acceleration, OBV rising, positive CMF |
| Momentum alignment | 12 | 5d/20d momentum, relative strength vs SPY, higher-highs |
| Trend quality | 12 | MACD, ADX strength, Bollinger %B, StochRSI |
| MA + 52w position | 12 | Price vs 20/50/200 MAs, proximity to 52w high, VWAP |
| Sector bonus | +2 | Top-3 sector = +2, top-6 = +1 |

Equal buckets mean **no single indicator can dominate** — a stock must be broadly healthy.
Thresholds are **regime-aware** (e.g. the ideal RSI band shifts down in a bear market).

#### News Score — 0–40 (weighted by signal strength)

| Component | Max | Rationale |
|-----------|----:|-----------|
| VADER sentiment | 15 | Strongest news signal — headline tone |
| News significance | 10 | Catalyst keyword hits / headline count |
| Macro alignment | 10 | Alignment with overall market sentiment |
| Analyst consensus | 3 | Rating + upside %, minor confirmation |
| Options signal | ±2 | Put/call ratio (bullish/bearish flow) |
| Insider signal | ±2 | Insider BUYING +2 / SELLING −1 |

Negative weights actively **penalize red flags** rather than only adding points.

#### LLM Layer (on top of the pre-score)

- Assigns a `catalyst_score` (1–10) and macro `score_adjustment`s
  (Fed / geopolitical / sector signals).
- Final **BUY** requires clearing a threshold of **80**; **WATCH** at **70**.

> The weights are **hand-tuned heuristics** (hard-coded constants encoding trading
> priors), not learned or backtested-optimized — a good area for feedback.

---

## 🔌 Data Sources & APIs

Only **NVIDIA NIM requires a key** (free tier). Everything else is keyless public data.
Full inventory in [`docs/reference/api_inventory.csv`](docs/reference/api_inventory.csv).

| Source | Purpose | Key? |
|--------|---------|:----:|
| **NVIDIA NIM API** | LLM reasoning / ranking / commentary | ✅ Free key |
| **yfinance (Yahoo Finance)** | OHLCV, fundamentals, per-ticker news | — |
| **RSS feeds** (Yahoo, CNBC, MarketWatch, BBC, NYTimes, EIA, OilPrice, Fed) | Macro & sector news | — |
| **SEC EDGAR** | Ticker→CIK mapping & filings | — (User-Agent required) |
| **House / Senate Stock Watcher** | Congressional trading disclosures | — |
| **Wikipedia** | S&P 500 / 400 constituent lists | — |
| **VADER** (local) | NLP sentiment scoring | — |

---

## 🚀 Quick Start (Local)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env        # then edit .env and set NVIDIA_API_KEY

# 4. Run
python LLM_Portfolio_Manager.py
```

Get a free NVIDIA NIM API key at **build.nvidia.com → sign up → "Get API Key"**.

### Environment Variables

See [`.env.example`](.env.example) for the full list.

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `NVIDIA_API_KEY` | ✅ | LLM analysis (free key from build.nvidia.com) |
| `OPENROUTER_API_KEY` | — | Backup LLM provider, used only if every NVIDIA model fails |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | — | Primary market data; required for any broker feature |
| `SCREENER_LIVE_BROKER` | — | `1` mirrors orders to Alpaca and makes it the source of truth |
| `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` | — | Discord alerts (primary channel) |
| `WHATSAPP_PHONE` / `CALLMEBOT_API_KEY` | — | WhatsApp alerts (via CallMeBot) |
| `SCREENER_DISABLE_ALERTS` | — | `1` silences every notification channel |
| `SCREENER_ALERT_TEST` | — | `1` marks every alert as a TEST (messages only; does **not** stop orders) |

---

## 🤖 Automated Daily Run (GitHub Actions)

The workflow at `.github/workflows/daily-screener.yml` runs the screener every day
at **21:30 UTC** (after US close) and on manual dispatch.

- Self-skips on weekends and US market holidays, so the schedule is safe.
- Add the variables above as repository **Secrets**
  (`Settings → Secrets and variables → Actions`). Only `NVIDIA_API_KEY` is required.
- Set `Settings → Actions → General → Workflow permissions` to
  **Read and write** so the run can commit updated state and upload reports as artifacts.

---

## 🔔 Alerts — Discord & WhatsApp (optional)

Both channels are independent: each activates only when its own variables are
set, and a failure in one never suppresses the other or the trading path.
Discord is the primary channel and receives the full message (it splits at
Discord's 2000-char limit); WhatsApp keeps its 1600-char cap.

**Discord** (`DISCORD_BOT_TOKEN` + `DISCORD_CHANNEL_ID`) posts four things:

1. **Execution events** — every order outcome confirmed against Alpaca, executed
   or not: filled, partially filled, still waiting, rejected, cancelled, never
   received, plus any share-count or price correction. See *Broker-authoritative
   execution* below. Set `SCREENER_ALERT_TEST=1` while testing and every message
   is retitled `🧪 TEST`, greyed out and footnoted, so a rehearsal can never be
   mistaken for a real fill.
2. **Daily decision** — market read (QQQ/VIX/SPY), the action taken, entry/stop/target
   and risk:reward, reasoning and key risk, positions closed today, next watchlist.
3. **Portfolio** — total value, P&L, cash, each holding worst-first with a health tag.
4. **Degraded runs** — which stage failed and which trade blockers fired whenever a
   run finishes not trade-ready, plus the weekly / market-closed summary.

Setup: discord.com/developers → New Application → Bot → copy token; invite the bot
to your server with **Send Messages**; enable Developer Mode in Discord and
right-click the channel → **Copy Channel ID**. Add both as repository Secrets.

**WhatsApp** (`WHATSAPP_PHONE` + `CALLMEBOT_API_KEY`) sends messages 2 and 3 via CallMeBot.

---

## 🏦 Broker-Authoritative Execution (Alpaca)

Alpaca serves two independent purposes, each behind its own switch:

| Switch | Effect |
|---|---|
| `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | Alpaca becomes the **primary market-data** source, with yfinance as automatic fallback. No orders are placed. |
| `SCREENER_LIVE_BROKER=1` (plus the keys) | Orders are mirrored to the Alpaca account **and Alpaca becomes the source of truth** for fills, share counts and cash. |

With live-broker mode on, the ledger no longer simulates a fill at the next
session's opening print. Each run reads the account, positions and orders, then
reconciles:

- **Filled** → booked at Alpaca's real `filled_avg_price`, with the stop and
  target re-anchored to the actual fill rather than the estimate.
- **Partially filled** → only the executed shares are booked.
- **Rejected / canceled / missing** → no position is opened, and the run is
  marked degraded so no new order is queued until it is resolved.
- **Share or cost-basis drift** → corrected to the broker's numbers.
- **Held at Alpaca but absent from the ledger** → adopted, once its sector
  resolves (an unknown sector means unknown exposure, so it fails closed).
- **Cash** → taken from the account, never re-derived locally.

Exits work the same way: a stop, target or hold-period exit becomes a *sell
request*, and the position stays open in the ledger until Alpaca confirms the
sale. Outgoing orders carry the ledger id in their `client_order_id`, so a fill
is matched back to the exact record that requested it.

**Reconciliation runs on every invocation, session or not.** Screening, model
probes and new orders require a completed session, but an execution at Alpaca is
a fact regardless — so a weekend, holiday or pre-close run still reads the
account, reconciles the ledger and alerts you, then stops without screening or
trading. Fills are dated by Alpaca's own `filled_at` timestamp rather than the
local calendar date, so a Friday fill reconciled on Saturday is still recorded as
Friday.

> The ledger is only ever rewritten from a snapshot Alpaca actually answered.
> If the account, positions or orders cannot be read, **nothing is changed** and
> the run degrades — a transport failure must never be read as "you hold nothing."

---

## 📁 Project Layout

```
LLM_Portfolio_Manager.py   # Entry point: config, data, indicators, LLM rounds, reporting
screener_safety.py         # Pure primitives: sizing/risk caps, atomic writes, exits
screener_portfolio.py      # Canonical session ledger: queue -> fill -> replay -> close
screener_contracts.py      # Fail-closed LLM + config schema validation, RunHealth
screener_alpaca.py         # Alpaca market data + paper execution + authoritative reads
screener_broker_sync.py    # Pure ledger-vs-broker reconciliation planner
screener_discord.py        # Discord delivery (never raises into the trading path)
qa_validate.py             # Non-mutating ledger checks + isolated live-provider QA
requirements.txt           # Python dependencies
.env.example               # Environment variable template
docs/reference/            # PROJECT_STRUCTURE, api_inventory.csv
architecture/              # Architecture diagram
tests/                     # Offline regression suite (no network, no credentials)
StockScreener/             # Portfolio state + generated reports
```

See [`docs/reference/PROJECT_STRUCTURE.md`](docs/reference/PROJECT_STRUCTURE.md) for details and refactor plans.

---

## 💬 Feedback Welcome

This is a personal project and I'd love suggestions on:

- **Scoring weights** — is the 60/40 tech/news split and the per-bucket weighting sensible?
- **Indicator choice** — anything redundant or missing?
- **The LLM layer** — is 30 candidates the right cut-off? Better prompt structure?
- **Validation** — how would you backtest / measure whether the picks actually work?
- **Architecture** — the script is currently one big file; how would you modularize it?

---

## 📝 Disclaimer

This is a personal learning project built out of curiosity. It is **NOT financial
advice**. Past performance does not guarantee future results. Always do your own research.

---

## Notes

- The current script is notebook-style and runs end-to-end when executed.
- Runtime output is written under `StockScreener/` by default when not in Colab.
- This repository structure is prepared so the script can be split into modules incrementally.
