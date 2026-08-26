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

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `NVIDIA_API_KEY` | ✅ | LLM analysis (free key from build.nvidia.com) |
| `WHATSAPP_PHONE` | — | WhatsApp alerts (E.164, e.g. `+64211234567`) |
| `CALLMEBOT_API_KEY` | — | WhatsApp alerts (via CallMeBot) |

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

## 📲 WhatsApp Alerts (optional)

When configured, each run sends two concise messages:

1. **Daily decision** — market read (QQQ/VIX/SPY), the action taken
   (bought / picked but not opened / watch only / no buy), entry/stop/target and
   risk:reward, reasoning and key risk, positions closed today, and the next watchlist.
2. **Portfolio** — total value, P&L, cash, and each holding worst-first with a health tag
   (near stop / near target / hold done / ok).

---

## 📁 Project Layout

```
LLM_Portfolio_Manager.py   # Current all-in-one script (entry point)
requirements.txt           # Python dependencies
.env.example               # Environment variable template
config/                    # Config files (future split from script constants)
data/                      # Local intermediate data (git-ignored)
reports/                   # Generated HTML/CSV/JSON reports (git-ignored)
docs/
  guides/                  # RUNBOOK, NVIDIA endpoint setup guide
  reference/               # PROJECT_STRUCTURE, api_inventory.csv, CODE_AUDIT_REPORT
tests/                     # Test suite and notes
StockScreener/             # Portfolio state + latest report output
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
