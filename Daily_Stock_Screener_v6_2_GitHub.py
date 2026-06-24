# -*- coding: utf-8 -*-
"""
Daily_Stock_Screener_v6_2.py
======================================
Version 6.2 - Runtime Fixes

Built by Vishvesh Trivedi
OSS Architect | AI/ML Automation | 12 Patents
LinkedIn: https://www.linkedin.com/in/vishvesh-trivedi

─────────────────────────────────────────────────────────────
⚠️  IMPORTANT: HOW TO SET YOUR API KEY (READ THIS FIRST)
─────────────────────────────────────────────────────────────

Option A - Google Colab (Recommended):
  1. Click the 🔑 Secrets icon in the left sidebar
  2. Add a new secret:
       Name:  NVIDIA_API_KEY
       Value: your-key-here (get it free from build.nvidia.com → sign up → "Get API Key")
  3. Enable the secret for this notebook
  4. The code below will read it automatically

Option B - Local Python:
  1. Create a file called .env in the same folder as this script
  2. Add this line:  NVIDIA_API_KEY=your-key-here
  3. Install python-dotenv:  pip install python-dotenv
  4. The code below will read it automatically

Option C - Paste directly (Colab only, NOT for GitHub):
  Find the line:  NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
  Replace with:   NVIDIA_API_KEY = "your-key-here"
  ⚠️  Never upload this version to GitHub!

─────────────────────────────────────────────────────────────
WHAT THIS SCREENER DOES
─────────────────────────────────────────────────────────────

Every evening after market close:
  → Downloads OHLCV for 375+ stocks in a single API call
  → Runs bidirectional screening — technical filters AND news rescue in parallel
  → Computes deterministic pre-scores:
       RSI, MACD, ADX, CMF, StochRSI, VWAP, OBV,
       Options P/C ratio, Insider flow, VADER NLP sentiment
  → Feeds top 30 candidates to Claude with full context:
       Macro headlines, sector rotation, earnings risk,
       self-calibration from past picks
  → Gets back BUY / WATCH / NO PICK with:
       Stop zones, price targets, R:R ratio,
       devil's advocate, full score breakdown
  → Saves everything to Google Drive
  → Auto-updates 10-day and 30-day returns over time

Tech Stack (100% free):
  • yfinance       — market data
  • VADER          — free NLP sentiment (no API key needed)
  • 16 RSS feeds   — macro news (Reuters, CNBC, MarketWatch, BBC...)
  • NVIDIA NIM API — AI reasoning layer (free tier at build.nvidia.com)
  • Google Colab + Drive — zero-infrastructure deployment

Cost per run:  ~$0.00 (NVIDIA NIM free tier)
Runtime:       ~7 - 9 minutes

─────────────────────────────────────────────────────────────
DAILY ROUTINE
─────────────────────────────────────────────────────────────
  Cell 1  Mount Google Drive        → run every session
  Cell 2  Install dependencies      → first time only
  Cell 3  API key + config          → first time only
  Cell 4  Load functions            → run every session
  Cell 5  Run screener              → run every day after market close

─────────────────────────────────────────────────────────────
DISCLAIMER
─────────────────────────────────────────────────────────────
This is a personal learning project built out of curiosity.
It is NOT financial advice. Past screener performance does
not guarantee future results. Always do your own research.

─────────────────────────────────────────────────────────────
Version History:
  FIX 1  enrich_with_scores() is now actually called in run_screener
  FIX 2  load_performance_history() called and passed to analyze_with_nvidia
  FIX 3  Stream B now runs BEFORE options/insider fetch
  FIX 4  Hard caps clamped post-hoc instead of trusted to the LLM
  FIX 5  FI ticker removed (Yahoo Finance delisted - was FISV)
  FIX 6  Sector bonus reduced from 3 to 2 (sum cleanly to 60 max)
  FIX 7  Model names unified to claude-sonnet-4-5 in all 3 Claude calls
  FIX 8  compute_indicators logs exception when SCREENER_DEBUG env set
  FIX 9  ETFs included in batch_download so sector ranks + ETF news work
  FIX 10 Candidates trimmed to top 30 by pre_score before Claude call
  FIX 11 Claude output tokens raised from 1200 to 2000
"""

# ============================================================
# CELL 1 - MOUNT GOOGLE DRIVE (run every session)
# ============================================================
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    DRIVE_FOLDER = '/content/drive/MyDrive/StockScreener'
    print('✅ Google Drive mounted')
except ImportError:
    IN_COLAB = False
    DRIVE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'StockScreener')

os.makedirs(DRIVE_FOLDER, exist_ok=True)
print(f'📁 Output folder: {DRIVE_FOLDER}')
if os.listdir(DRIVE_FOLDER):
    print(f'📄 Files:  {os.listdir(DRIVE_FOLDER)}')

# ============================================================
# CELL 2 - INSTALL DEPENDENCIES (first time only)
# ============================================================
# In Colab: uncomment the line below and run it once
# !pip install yfinance pandas openai requests vaderSentiment python-dotenv --quiet
#
# Locally: run this once in your terminal instead:
#   pip install yfinance pandas openai requests vaderSentiment python-dotenv
print('✅ Dependencies assumed installed')

# ============================================================
# CELL 3 - CONFIGURATION
# ============================================================

# ── API KEY (reads from Colab Secrets or .env file) ────────
# Follow the instructions at the top of this file to set your key safely.
# Get your free NVIDIA NIM API key at: build.nvidia.com → sign up → "Get API Key"
# Never paste your real key here if you plan to share or upload this file.

import os

# 1. Try Colab Secrets (userdata API — works in newer Colab)
NVIDIA_API_KEY = ""
try:
    from google.colab import userdata
    NVIDIA_API_KEY = (userdata.get("NVIDIA_API_KEY") or "").strip()
except Exception:
    pass

# 2. Fall back to os.environ (older Colab / local)
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()

# 3. Fall back to .env file (local only)
if not NVIDIA_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
    except ImportError:
        pass

if not NVIDIA_API_KEY:
    print("WARNING: No API key found!")
    print("   Get your FREE key at: build.nvidia.com -> sign up -> 'Get API Key'")
    print("   In Colab: use the Secrets panel on the left sidebar.")
else:
    print("✅ API key loaded successfully")

# ── WHATSAPP (CallMeBot) ────────────────────────────────────
# Store WHATSAPP_PHONE and CALLMEBOT_API_KEY in Colab Secrets (same panel as NVIDIA_API_KEY)
# WHATSAPP_PHONE: your number in international format WITHOUT +, e.g. 447911123456 or 919876543210
WHATSAPP_PHONE      = ""
CALLMEBOT_API_KEY   = ""
for _secret in ["WHATSAPP_PHONE", "CALLMEBOT_API_KEY"]:
    try:
        from google.colab import userdata as _ud
        _val = (_ud.get(_secret) or "").strip()
    except Exception:
        _val = os.environ.get(_secret, "").strip()
    if _secret == "WHATSAPP_PHONE":
        WHATSAPP_PHONE = _val
    else:
        CALLMEBOT_API_KEY = _val

if WHATSAPP_PHONE and CALLMEBOT_API_KEY:
    print(f"✅ WhatsApp configured (phone ...{WHATSAPP_PHONE[-4:]})")
else:
    missing = []
    if not WHATSAPP_PHONE:    missing.append("WHATSAPP_PHONE")
    if not CALLMEBOT_API_KEY: missing.append("CALLMEBOT_API_KEY")
    print(f"⚠️  WhatsApp disabled — missing Colab Secrets: {', '.join(missing)}")
    print("   Add them in the Secrets panel (key icon, left sidebar) then re-run Cell 1.")

# ── SCREENER SETTINGS ──────────────────────────────────────
BUY_THRESHOLD    = 80    # minimum confidence score to generate a BUY signal
WATCH_THRESHOLD  = 70    # minimum confidence score for WATCH list
VOLUME_MIN_RATIO = 1.2   # stock must trade at 1.2x its average volume
RSI_MIN          = 35    # minimum RSI (avoid oversold)
RSI_MAX          = 75    # maximum RSI (avoid overbought)
MIN_PRICE        = 5.0   # minimum stock price in USD
SAMPLE_SIZE      = 900   # full dynamic universe — LLM can lower via config_overrides.json

PICKS_CSV        = f'{DRIVE_FOLDER}/stock_picks.csv'
WATCH_CSV        = f'{DRIVE_FOLDER}/watch_list.csv'
PORTFOLIO_JSON   = f'{DRIVE_FOLDER}/portfolio.json'
STARTING_CAPITAL = 10_000.00

# ── SHARESIES $15/MONTH PLAN (NZ broker — buys NYSE/NASDAQ in USD) ──────────
# Plan:  $5,000 NZD free buys + $5,000 NZD free sells per month
# Rate:  fetched live via NZDUSD=X each run (stored in ctx['global_macro']['nzdusd'])
# Fee when OVER coverage: 0.5% of order value, capped at $5.00 USD per trade
# BROKERAGE_FEE = 0 while positions stay within monthly coverage.
# Change to 5.00 if you regularly go over.
SHARESIES_COVERAGE_NZD = 5_000.0   # NZD of free buys per month under $15 plan
BROKERAGE_FEE          = 0.00      # $0 within coverage, max $5 USD over

# Optional: set to '1' to see why tickers fail compute_indicators
# os.environ['SCREENER_DEBUG'] = '1'

print('\n✅ Configuration ready')
print(f'   BUY threshold:   {BUY_THRESHOLD}')
print(f'   WATCH threshold: {WATCH_THRESHOLD}')
print(f'   Sample size:     {SAMPLE_SIZE}')

# ============================================================
# CELL 4 - ALL FUNCTIONS (run once per session)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _VADER = _VaderAnalyzer()
except ImportError:
    _VADER = None


# ── TUNING CONSTANTS ───────────────────────────────────────
MIN_DOLLAR_VOLUME_M  = 20      # minimum $20M/day - ensures liquidity
ATR_STOP_MULT        = 1.5     # stop = entry - 1.5xATR
ATR_TARGET_MULT      = 3.0     # target = entry + 3xATR → R:R 1:2
ADX_MIN              = 20      # minimum trend strength
VIX_LOW_PCTILE       = 25      # below this = risk-on regime
VIX_HIGH_PCTILE      = 75      # above this = risk-off regime
SECTOR_CONC_LOOKBACK = 5       # look back N picks for sector concentration
SECTOR_CONC_MAX      = 3       # max same-sector picks in lookback window
SECTOR_CONC_PENALTY  = 10      # confidence penalty if concentrated

# ── LLM-CONTROLLED TUNING CONSTANTS (override via config_overrides.json) ────
_CFG_VIX_LOW_PCTILE      = VIX_LOW_PCTILE    # VIX below this = risk-on (1.1x multiplier)
_CFG_VIX_HIGH_PCTILE     = VIX_HIGH_PCTILE   # VIX above this = risk-off (0.7x multiplier)
_CFG_SECTOR_CONC_LOOKBACK= SECTOR_CONC_LOOKBACK  # rolling window for sector concentration
_CFG_SECTOR_CONC_PENALTY = SECTOR_CONC_PENALTY   # confidence docked when over sector limit
_CFG_CONGRESS_DAYS       = 60    # how many days back to fetch congress trades
_CFG_SEC_8K_DAYS         = 7     # how many days back to fetch SEC 8-K filings
_CFG_RSI_EXIT            = 78.0  # RSI level that triggers overbought exit (when profitable)
_CFG_RSI_EXIT_MIN_PROFIT = 0.0   # min unrealized % profit before RSI exit fires (0 = no gate, LLM controls)
_CFG_MACD_EXIT_MIN_PROFIT= 0.0   # min unrealized % profit before MACD bearish cross exit fires (0 = no gate)
_CFG_ENTRY_SLIPPAGE_PCT  = 0.3   # estimated gap from last close to next-day open (buy at open, not close)
_CFG_FINAL_CANDIDATES    = 30    # how many top candidates pass to LLM final round
_CFG_PRE_EARNINGS_DAYS   = 0     # days before earnings to auto-exit (0 = only on earnings day; LLM sees warning via portfolio_summary_str)
_CFG_SQUEEZE_FLOAT_PCT   = 20.0  # min short % of float to flag a squeeze setup
_CFG_SQUEEZE_DAYS_COVER  = 5.0   # min days-to-cover to flag a squeeze setup
_CFG_TRAIL_ATR_MULT      = 1.5   # trailing stop ratchet multiplier (can differ from ATR_STOP_MULT)
_CFG_VOLUME_MIN_RATIO    = VOLUME_MIN_RATIO  # min volume vs average to pass universe screen
NEWS_WORKERS         = 20      # parallel workers for news/fundamentals fetch

# ── LLM-CONTROLLED CRITERIA (overridden by config_overrides.json) ──────────
# The screener writes these after each run so the LLM can change its own rules.
# LLM has FULL AUTHORITY — it can change any of these values or add new ones.
_CFG_SECTOR_BLACKLIST   = []      # sectors the LLM has decided to avoid
_CFG_SECTOR_WHITELIST   = []      # sectors the LLM is favouring (gets +5 boost)
_CFG_SOURCE_PREFERENCE  = 'ANY'   # ANY | TECHNICAL | NEWS | BOTH
_CFG_REQUIRE_CONGRESS   = False   # if True, only tickers with congress purchases pass
_CFG_MIN_CATALYST_SCORE = 0       # minimum catalyst score (0-100) after LLM scoring
_CFG_MIN_ADX_BUY        = ADX_MIN # stricter ADX needed for a BUY (vs passing the pool)
_CFG_AVOID_EARNINGS     = False   # if True, auto-drop any earnings-risk ticker
_CFG_MAX_VIX            = 999     # if VIX > this, skip all BUY signals (go to cash)
_CFG_MIN_PRICE          = MIN_PRICE   # override minimum price floor
_CFG_ONLY_PROFITABLE    = False   # only stocks with positive trailing EPS/revenue
_CFG_REQUIRE_ABOVE_MA   = True    # if False, disable MA20/MA50 filter (catch breakouts)
_CFG_MIN_DOLLAR_VOL_M   = MIN_DOLLAR_VOLUME_M  # minimum daily dollar volume ($M)
_CFG_HOLD_DAYS          = 10      # days before position auto-closes and Win/Loss is set
_CFG_SECTOR_CONC_MAX    = SECTOR_CONC_MAX  # max same-sector picks in rolling window
_CFG_SAMPLE_SIZE        = SAMPLE_SIZE      # how many stocks to scan each run
_CFG_ADDITIONAL_TICKERS = []      # LLM can add tickers outside the default universe
_CFG_BROKERAGE_FEE      = BROKERAGE_FEE  # per-trade flat fee (both buy and sell sides)
_CFG_MIN_CASH_FLOOR     = 500.0   # cash below this = fully deployed, no new buys
_CFG_DD_CAUTION_PCT     = -10.0   # portfolio drawdown % that triggers caution mode
_CFG_DD_SEVERE_PCT      = -20.0   # portfolio drawdown % that triggers severe mode
_CFG_DD_CRITICAL_PCT    = -30.0   # portfolio drawdown % that triggers capital preservation
_CFG_WIN_THRESHOLD_PCT  = 2.0     # vs-QQQ % at 10d to count as Win
_CFG_LOSS_THRESHOLD_PCT = -2.0    # vs-QQQ % at 10d to count as Loss
_CFG_MIN_PICKS_TO_LEARN = 5       # closed picks required before LLM writes config
_CFG_RSI_HARD_CAP       = 999     # RSI above this clamps confidence (999 = disabled)
_CFG_RSI_CAP_CONF       = 70      # confidence ceiling when RSI_HARD_CAP is breached
_CFG_UPSIDE_HARD_CAP    = -999    # analyst upside below this clamps confidence (-999 = disabled)
_CFG_UPSIDE_CAP_CONF    = 65      # confidence ceiling when UPSIDE_HARD_CAP is breached
_CFG_MAX_POSITIONS      = 5       # max simultaneous open positions (LLM configurable)

# ── NVIDIA MODEL SELECTION ─────────────────────────────────
# Pick any one — all are free on NVIDIA NIM (build.nvidia.com)
#
# RECOMMENDED FOR THIS SCREENER (best JSON + financial reasoning):
#   'deepseek-ai/deepseek-v4-pro'           ← newest DeepSeek, free on NVIDIA NIM
#   'meta/llama-3.3-70b-instruct'           ← solid all-rounder
#   'nvidia/llama-3.1-nemotron-70b-instruct'← NVIDIA-tuned, very strong reasoning
#   'qwen/qwen2.5-72b-instruct'             ← excellent structured JSON output
#   'mistralai/mixtral-8x22b-instruct-v0.1' ← fast, good for JSON
#
# DEEPSEEK REASONING (chain-of-thought — <think> block stripped automatically):
#   'deepseek-ai/deepseek-r1'              ← strongest reasoning, but slowest
#
# LARGER / MORE POWERFUL (slower, may hit free tier limits):
#   'meta/llama-3.1-405b-instruct'           ← biggest Llama, best reasoning
#   'mistralai/mistral-large-latest'         ← strong general reasoning
#
# SMALLER / FASTER (lower quality but instant):
#   'meta/llama-3.2-3b-instruct'             ← very fast, lower quality
#   'microsoft/phi-3-mini-128k-instruct'     ← lightweight

NVIDIA_MODEL = 'meta/llama-3.3-70b-instruct'  # ← change this line to switch model


_LLM_CALL_COUNT = [0]
_LLM_LAST_CALL  = [0.0]
_LLM_MIN_GAP    = 1.6   # 40/min = 1 call per 1.5s, +0.1s buffer

def call_llm(system, user, max_tokens=2000):
    """Call NVIDIA NIM. Built-in rate limiter (40/min). Strips DeepSeek <think> blocks."""
    gap = time.time() - _LLM_LAST_CALL[0]
    if gap < _LLM_MIN_GAP:
        time.sleep(_LLM_MIN_GAP - gap)
    _LLM_LAST_CALL[0] = time.time()
    _LLM_CALL_COUNT[0] += 1

    _is_r1 = 'deepseek-r1' in NVIDIA_MODEL.lower()
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": NVIDIA_MODEL,
        "max_tokens": max_tokens + (2000 if _is_r1 else 0),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    }
    for attempt in range(3):
        try:
            resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
                                 headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json()['choices'][0]['message']['content'].strip()
            if '</think>' in raw:
                raw = raw[raw.index('</think>') + len('</think>'):].strip()
            return raw
        except Exception as e:
            print(f'  LLM attempt {attempt+1}/3 failed: {type(e).__name__}: {e}')
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

# ── TWO-TIER RESCUE KEYWORDS ───────────────────────────────
RESCUE_TIER1 = [
    'acquisition', 'merger', 'buyout', 'takeover',
    'fda', 'clearance',
    'bankruptcy', 'restructur',
    'lawsuit', 'doj', 'ftc', 'fraud', 'indicted',
    'tariff', 'sanction', 'delisting', 'halted',
    'resign', 'fired', 'ousted',
]

RESCUE_TIER2_REACTION = [
    'beats', 'misses', 'exceeds', 'warning',
    'raised', 'lowered', 'rejected',
    'approved', 'upgraded', 'downgraded',
    'investigation', 'subpoena',
]

RESCUE_TIER2_CONTEXT = [
    'earnings', 'revenue', 'guidance',
    'profit', 'outlook', 'quarterly',
]


def has_significant_news(news_titles):
    """
    Two-tier rescue logic.
    Returns (should_rescue, tier1_hits, tier2_reaction_hits, tier2_context_hits)
    """
    text  = ' '.join(news_titles).lower()
    t1    = [k for k in RESCUE_TIER1 if k in text]
    t2r   = [k for k in RESCUE_TIER2_REACTION if k in text]
    t2c   = [k for k in RESCUE_TIER2_CONTEXT if k in text]
    rescue = len(t1) >= 1 or (len(t2r) >= 1 and len(t2c) >= 1)
    return rescue, t1, t2r, t2c


def compute_vader_sentiment(news_titles):
    """
    NLP sentiment on news headlines. Free - no API key needed.
    Returns (compound, label) where compound is -1 to +1.
    BULLISH >= +0.05  |  BEARISH <= -0.05  |  NEUTRAL otherwise
    """
    if not news_titles or _VADER is None:
        return 0.0, 'NEUTRAL'
    scores = [_VADER.polarity_scores(t)['compound'] for t in news_titles]
    avg = sum(scores) / len(scores)
    label = 'BULLISH' if avg >= 0.05 else 'BEARISH' if avg <= -0.05 else 'NEUTRAL'
    return round(avg, 3), label


def compute_tech_score(ind, ctx, sector_rank=6):
    """
    Deterministic technical score 0-60. Runs BEFORE Claude.

    Components:
      RSI regime fit      0-12
      Volume conviction   0-12
      Momentum alignment  0-12
      Trend quality       0-12
      MA + 52w position   0-12
    Sector bonus          0-2
    """
    if not ind:
        return 0, {}
    bd = {}
    is_bull = ctx.get('qqq_trend', 'BULLISH') == 'BULLISH'
    rsi = ind.get('rsi', 50)

    # 1. RSI regime fit (0-12)
    if is_bull:
        if   50 <= rsi <= 65: rsi_pts = 12
        elif 45 <= rsi <  50: rsi_pts = 8
        elif 65 <  rsi <= 72: rsi_pts = 7
        elif 40 <= rsi <  45: rsi_pts = 4
        elif 72 <  rsi <= 78: rsi_pts = 3
        else:                 rsi_pts = 0
    else:
        if   40 <= rsi <= 55: rsi_pts = 12
        elif 35 <= rsi <  40: rsi_pts = 8
        elif 55 <  rsi <= 62: rsi_pts = 7
        elif 30 <= rsi <  35: rsi_pts = 4
        elif 62 <  rsi <= 68: rsi_pts = 2
        else:                 rsi_pts = 0
    bd['rsi_fit'] = rsi_pts

    # 2. Volume conviction (0-12)
    vr  = ind.get('vol_ratio', 0.0)
    va  = ind.get('vol_accel', False)
    obv = ind.get('obv_rising', False)
    cmf = ind.get('cmf', 0.0)
    if   vr >= 3.0: vol_pts = 8
    elif vr >= 2.5: vol_pts = 7
    elif vr >= 2.0: vol_pts = 6
    elif vr >= 1.5: vol_pts = 4
    elif vr >= 1.2: vol_pts = 2
    else:           vol_pts = 0
    if va:        vol_pts = min(12, vol_pts + 2)
    if obv:       vol_pts = min(12, vol_pts + 1)
    if cmf > 0.1: vol_pts = min(12, vol_pts + 1)
    bd['volume'] = vol_pts

    # 3. Momentum alignment (0-12)
    m5   = ind.get('momentum_5d',  0.0)
    m20  = ind.get('momentum_20d', 0.0)
    rs   = ind.get('rs_vs_spy',    0.0)
    hh   = ind.get('hh_hl', False)
    if is_bull:
        m5_pts  = 5 if m5 > 2 else 4 if m5 > 0 else 1 if m5 > -2 else 0
        m20_pts = 4 if m20 > 4 else 3 if m20 > 1 else 1 if m20 > 0 else 0
    else:
        m5_pts  = 5 if 0 < m5 <= 3 else 2 if m5 > 3 else 2 if m5 > -2 else 0
        m20_pts = 4 if 0 < m20 <= 5 else 1 if m20 > 5 else 2 if m20 > -3 else 0
    rs_pts  = 3 if rs > 1.5 else 2 if rs > 0.5 else 1 if rs > 0 else 0
    hh_pts  = 1 if hh else 0
    mom_pts = min(12, m5_pts + m20_pts + rs_pts + hh_pts)
    bd['momentum'] = mom_pts

    # 4. Trend quality - MACD + ADX + BB%B + StochRSI (0-12)
    macd      = ind.get('macd_bullish', False)
    adx       = ind.get('adx', 0.0)
    bb        = ind.get('bb_pct_b', 0.5)
    stoch_rsi = ind.get('stoch_rsi', 0.5)
    tr_pts = 0
    if macd:        tr_pts += 4
    if   adx >= 35: tr_pts += 4
    elif adx >= 28: tr_pts += 3
    elif adx >= 22: tr_pts += 1
    elif adx <  18: tr_pts -= 1
    if bb >= 0.8:         tr_pts += 2
    elif bb >= 0.6:       tr_pts += 1
    elif bb <= 0.2:       tr_pts -= 1
    if stoch_rsi > 0.8:   tr_pts += 1
    elif stoch_rsi < 0.2: tr_pts -= 1
    bd['trend_quality'] = max(0, min(12, tr_pts))

    # 5. MA alignment + 52w + VWAP (0-12)
    vs20  = ind.get('vs_ma20_pct', 0.0)
    vs50  = ind.get('vs_ma50_pct', 0.0)
    vs200 = ind.get('vs_ma200_pct', 0.0)
    h52   = ind.get('pct_from_52h', -20.0)
    vwap  = ind.get('vs_vwap_pct', 0.0)
    ma_pts = 0
    if vs20 > 0 and vs50 > 0 and vs200 > 0: ma_pts += 5
    elif vs20 > 0 and vs50 > 0:              ma_pts += 3
    elif vs20 > 0:                            ma_pts += 1
    if   h52 >= -3:  ma_pts += 4
    elif h52 >= -8:  ma_pts += 3
    elif h52 >= -15: ma_pts += 2
    elif h52 >= -25: ma_pts += 1
    elif h52 < -40:  ma_pts -= 1
    if vwap > 0.5:   ma_pts = min(12, ma_pts + 1)
    bd['ma_52w'] = max(0, min(12, ma_pts))

    # Sector bonus (0-2)
    sec_bonus = 2 if sector_rank <= 3 else 1 if sector_rank <= 6 else 0

    total = rsi_pts + vol_pts + mom_pts + bd['trend_quality'] + bd['ma_52w'] + sec_bonus
    total = max(0, min(60, total))
    bd['sector_bonus'] = sec_bonus
    bd['total'] = total
    return total, bd


def compute_news_score(news_titles, rescue_keywords, analyst_rating,
                       upside_pct, market_sentiment, pc_ratio=None,
                       insider_label='NEUTRAL'):
    """
    News-driven score 0-40. Runs BEFORE Claude.
    """
    vader_compound, vader_label = compute_vader_sentiment(news_titles)

    # 1. VADER sentiment (0-15)
    if   vader_compound >= 0.35: v_pts = 15
    elif vader_compound >= 0.20: v_pts = 13
    elif vader_compound >= 0.10: v_pts = 11
    elif vader_compound >= 0.05: v_pts =  9
    elif vader_compound >= -0.05:v_pts =  7
    elif vader_compound >= -0.15:v_pts =  4
    elif vader_compound >= -0.25:v_pts =  2
    else:                        v_pts =  0

    # 2. News significance (0-10)
    n_hits   = len(rescue_keywords) if rescue_keywords else 0
    n_titles = len(news_titles)     if news_titles     else 0
    if   n_hits >= 4: kw_pts = 10
    elif n_hits >= 3: kw_pts = 8
    elif n_hits >= 2: kw_pts = 6
    elif n_hits >= 1: kw_pts = 4
    elif n_titles >= 5: kw_pts = 3
    elif n_titles >= 2: kw_pts = 2
    else:               kw_pts = 0

    # 3. Macro alignment (0-10)
    ms = (market_sentiment or 'NEUTRAL').upper()
    if   ms == 'BULLISH':  mac_pts = 10
    elif ms == 'NEUTRAL':  mac_pts =  6
    elif ms == 'CAUTIOUS': mac_pts =  3
    else:                  mac_pts =  1

    # 4. Analyst consensus (0-3)
    a_pts = 0
    if analyst_rating is not None:
        if   analyst_rating <= 1.5: a_pts = 3
        elif analyst_rating <= 2.0: a_pts = 2
        elif analyst_rating <= 2.5: a_pts = 1
    if upside_pct is not None:
        if   upside_pct >= 30: a_pts = min(3, a_pts + 1)
        elif upside_pct <= -10: a_pts = max(0, a_pts - 1)

    # 5. Options signal (0-2)
    opt_pts = 0
    if pc_ratio is not None:
        if   pc_ratio < 0.6: opt_pts = 2
        elif pc_ratio < 0.9: opt_pts = 1
        elif pc_ratio > 1.5: opt_pts = -1

    # 6. Insider signal (0-2)
    ins_pts = 2 if insider_label == 'BUYING' else -1 if insider_label == 'SELLING' else 0

    total = v_pts + kw_pts + mac_pts + a_pts + opt_pts + ins_pts
    bd = {
        'vader_sentiment':    v_pts,
        'news_significance':  kw_pts,
        'macro_alignment':    mac_pts,
        'analyst_consensus':  a_pts,
        'options_signal':     opt_pts,
        'insider_signal':     ins_pts,
        'total':              total,
        'vader_label':        vader_label,
    }
    return min(40, max(0, total)), bd, vader_label


def _fetch_options_single(ticker):
    """Fetch put/call ratio + unusual call activity from nearest expiry options chain."""
    try:
        tk   = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return ticker, None, 'NEUTRAL', False
        opt        = tk.option_chain(exps[0])
        calls_vol  = float(opt.calls['volume'].fillna(0).sum())
        puts_vol   = float(opt.puts['volume'].fillna(0).sum())
        if calls_vol + puts_vol < 200:
            return ticker, None, 'NEUTRAL', False
        pc = round(puts_vol / calls_vol, 2) if calls_vol > 0 else None
        if   pc is None:  label = 'NEUTRAL'
        elif pc < 0.7:    label = 'BULLISH'
        elif pc > 1.3:    label = 'BEARISH'
        else:             label = 'NEUTRAL'
        # Unusual call activity: any strike where volume > 3× open interest (smart money positioning)
        unusual_calls = False
        try:
            calls = opt.calls.copy()
            calls = calls[(calls['openInterest'] > 50) & (calls['volume'].fillna(0) > 0)]
            if not calls.empty:
                calls['vol_oi'] = calls['volume'].fillna(0) / calls['openInterest']
                if calls['vol_oi'].max() > 3:
                    unusual_calls = True
        except:
            pass
        return ticker, pc, label, unusual_calls
    except:
        return ticker, None, 'NEUTRAL', False


_SEC_CIK_CACHE = {}

def _load_sec_cik_map():
    global _SEC_CIK_CACHE
    if _SEC_CIK_CACHE: return _SEC_CIK_CACHE
    try:
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                         headers={'User-Agent': 'StockScreener research@example.com'}, timeout=15)
        _SEC_CIK_CACHE = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in r.json().values()}
    except: pass
    return _SEC_CIK_CACHE

def _fetch_insider_single(ticker):
    """Insider signals: try yfinance (3 attribute variations) then SEC EDGAR Form 4."""
    # --- Try yfinance first ---
    try:
        tk = yf.Ticker(ticker)
        ins = None
        for attr in ('insider_transactions', 'insider_purchases', 'insider_roster_holders'):
            try:
                candidate = getattr(tk, attr, None)
                if candidate is not None and not getattr(candidate, 'empty', True):
                    ins = candidate.copy(); break
            except: pass
        if ins is not None:
            date_col = next((c for c in ['Start Date','Date','Transaction Date','startDate'] if c in ins.columns), None)
            if date_col:
                ins['_dt'] = pd.to_datetime(ins[date_col], errors='coerce')
                recent = ins[ins['_dt'] >= pd.Timestamp.now() - pd.Timedelta(days=90)]
                tx_col = next((c for c in ['Transaction','Type','transactionType','Acquisition or Disposal'] if c in recent.columns), None)
                if tx_col and not recent.empty:
                    buys  = recent[recent[tx_col].astype(str).str.contains('Buy|Purchase|^P$|^A$', case=False, na=False, regex=True)]
                    sells = recent[recent[tx_col].astype(str).str.contains('Sale|Sell|^S$|^D$',    case=False, na=False, regex=True)]
                    if len(buys) >= 2 and len(buys) > len(sells):
                        return ticker, len(buys) * 10000, 'BUYING'
                    if len(sells) >= 3 and len(sells) > len(buys) * 2:
                        return ticker, -(len(sells) * 10000), 'SELLING'
                    return ticker, 0, 'NEUTRAL'
    except: pass

    # --- Fallback: SEC EDGAR Form 4 count (open govt API, always works) ---
    try:
        cik = _load_sec_cik_map().get(ticker.upper())
        if not cik: return ticker, 0, 'NEUTRAL'
        r = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json',
                         headers={'User-Agent': 'StockScreener research@example.com'}, timeout=10)
        recent_filings = r.json().get('filings', {}).get('recent', {})
        forms = recent_filings.get('form', [])
        dates = recent_filings.get('filingDate', [])
        cutoff = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        form4_count = sum(1 for f, d in zip(forms, dates) if f == '4' and d >= cutoff)
        if form4_count >= 3:
            return ticker, form4_count * 5000, 'BUYING'
    except: pass
    return ticker, 0, 'NEUTRAL'


def fetch_congress_trades(days=60):
    """
    Fetch recent US congressional stock PURCHASES (House + Senate STOCK Act disclosures).
    Tries 4 sources in order — stops at first success:
      1. House S3 bucket  (bypasses DNS issues in Colab)
      2. Senate S3 bucket
      3. housestockwatcher.com API  (fallback)
      4. senate-stock-watcher API   (fallback)
    Returns {ticker: {'count': N, 'names': ['Smith (R-TX) $15K-50K', ...], 'chamber': ['House'/'Senate']}}
    """
    _SOURCES = [
        # chamber, url, name_field, date_field
        ('House',  'https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json',
                   'representative', 'disclosure_date'),
        ('Senate', 'https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json',
                   'senator', 'transaction_date'),
        ('House',  'https://housestockwatcher.com/api/transactions',
                   'representative', 'disclosure_date'),
        ('Senate', 'https://efts.us/s3/senate-stock-watcher-data/aggregate/all_transactions.json',
                   'senator', 'transaction_date'),
    ]

    cutoff   = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    result   = {}
    loaded   = []

    for chamber, url, name_field, date_field in _SOURCES:
        if chamber in loaded:
            continue   # already got this chamber from an earlier source
        try:
            r = requests.get(url, timeout=25, headers={'User-Agent': 'StockScreener vishvesh.niyati@gmail.com'})
            if r.status_code != 200:
                continue
            data = r.json()
            if not isinstance(data, list) or not data:
                continue
            loaded.append(chamber)
        except Exception:
            continue

        for rec in data:
            if 'purchase' not in str(rec.get('type', '')).lower():
                continue
            raw_ticker = str(rec.get('ticker', '')).strip().upper().replace('$', '')
            if not raw_ticker or len(raw_ticker) > 5 or not raw_ticker.isalpha():
                continue
            date_str = str(rec.get(date_field, '') or '')[:10]
            if date_str < cutoff:
                continue

            # Build rich name string: "Smith (R-TX) $15K-50K"
            full_name  = str(rec.get(name_field, '') or '').strip()
            last_name  = full_name.split()[-1] if full_name else '?'
            party      = str(rec.get('party', '') or '').strip()
            state      = str(rec.get('state', '') or '').strip()
            amount_raw = str(rec.get('amount', '') or '').strip()
            amount_str = (amount_raw
                          .replace('$1,001 - $15,000', '$1K-15K')
                          .replace('$15,001 - $50,000', '$15K-50K')
                          .replace('$50,001 - $100,000', '$50K-100K')
                          .replace('$100,001 - $250,000', '$100K-250K')
                          .replace('$250,001 - $500,000', '$250K-500K')
                          .replace('$500,001 - $1,000,000', '$500K-1M')
                          .replace('$1,000,001 - $5,000,000', '$1M-5M')
                          .replace('Over $5,000,000', '>$5M'))
            party_state = f' ({party}-{state})' if party and state else ''
            name_tag    = f'{last_name}{party_state} {amount_str} [{chamber[0]}]'

            if raw_ticker not in result:
                result[raw_ticker] = {'count': 0, 'names': [], 'chamber': [], 'dates': []}
            result[raw_ticker]['count'] += 1
            if name_tag not in result[raw_ticker]['names']:
                result[raw_ticker]['names'].append(name_tag)
            if chamber not in result[raw_ticker]['chamber']:
                result[raw_ticker]['chamber'].append(chamber)
            if date_str and date_str not in result[raw_ticker]['dates']:
                result[raw_ticker]['dates'].append(date_str)

        if len(loaded) == 2:
            break  # got both House and Senate, done

    _cache_path = os.path.join(DRIVE_FOLDER, 'congress_cache.json')

    if not result:
        # Try Drive cache from last successful fetch (survives Colab network blocks)
        try:
            if os.path.exists(_cache_path):
                with open(_cache_path) as f:
                    cached = json.load(f)
                cache_age = (datetime.now() - datetime.fromisoformat(cached.get('fetched_at', '2000-01-01'))).days
                if cache_age <= 7:  # use cache if < 7 days old
                    print(f'  Congress trades: live sources unavailable — using Drive cache ({cache_age}d old)')
                    return cached.get('data', {})
                else:
                    print(f'  Congress trades: all sources unavailable (cache is {cache_age}d old — too stale)')
            else:
                print(f'  Congress trades: all sources unavailable (no cache)')
        except Exception as ce:
            print(f'  Congress trades: all sources unavailable ({ce})')
        return {}

    # Save successful fetch to Drive cache for future fallback
    try:
        with open(_cache_path, 'w') as f:
            json.dump({'fetched_at': datetime.now().isoformat(), 'data': result}, f)
    except Exception:
        pass

    total = sum(v['count'] for v in result.values())
    print(f'  Congress buys (last {days}d): {total} purchases | {len(result)} stocks | sources: {", ".join(loaded)}')
    return result


def fetch_options_and_insider_parallel(candidates):
    """Fetch options P/C + insider sentiment for ALL candidates in parallel."""
    tickers = [c['ticker'] for c in candidates]
    print(f'  Fetching options + insider data for {len(tickers)} candidates...')
    options_data = {}
    insider_data = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        opt_futures = {ex.submit(_fetch_options_single, t): t for t in tickers}
        ins_futures = {ex.submit(_fetch_insider_single, t): t for t in tickers}
        for f in as_completed(opt_futures):
            t, pc, lbl, unusual = f.result()
            options_data[t] = {'pc_ratio': pc, 'label': lbl, 'unusual_calls': unusual}
        for f in as_completed(ins_futures):
            t, net, lbl = f.result()
            insider_data[t] = {'net_shares': net, 'label': lbl}
    opt_signals = sum(1 for v in options_data.values() if v['pc_ratio'] is not None)
    ins_signals = sum(1 for v in insider_data.values()  if v['label'] != 'NEUTRAL')
    print(f'  Options data: {opt_signals}/{len(tickers)} | Insider signals: {ins_signals}/{len(tickers)}')
    return options_data, insider_data


def compute_sector_ranks(batch_data):
    """Rank all 11 sectors by 5-day ETF performance."""
    perf = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        df = batch_data.get(etf)
        if df is None or len(df) < 5:
            perf[sector] = 0.0
            continue
        try:
            p_now = float(df['Close'].iloc[-1])
            p_5d  = float(df['Close'].iloc[-5])
            perf[sector] = round(((p_now - p_5d) / p_5d) * 100, 2) if p_5d > 0 else 0.0
        except:
            perf[sector] = 0.0
    sorted_s = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    ranks    = {s: i + 1 for i, (s, _) in enumerate(sorted_s)}
    top3     = [s for s, _ in sorted_s[:3]]
    print(f'  Sector ranks - Top 3: {top3} | Perf: '
          + ' | '.join(f'{s}:{v:+.1f}%' for s, v in sorted_s[:5]))
    return ranks, perf


def enrich_with_scores(candidates, ctx, market_sentiment,
                       sector_ranks, options_data, insider_data, congress_data=None):
    """Attach tech_score and news_score to every candidate."""
    print(f'\nPhase 5.5 - Computing score breakdown ({len(candidates)} candidates)...')
    for c in candidates:
        t = c['ticker']
        sec_rank = sector_ranks.get(c.get('sector', 'Unknown'), 6)
        opt  = options_data.get(t, {})
        ins  = insider_data.get(t, {})

        ts, ts_bd = compute_tech_score(c, ctx, sector_rank=sec_rank)
        c['tech_score']           = ts
        c['tech_score_breakdown'] = ts_bd

        ns, ns_bd, vader_lbl = compute_news_score(
            news_titles      = c.get('stock_news', []),
            rescue_keywords  = c.get('rescue_keywords', []),
            analyst_rating   = c.get('analyst_rating'),
            upside_pct       = c.get('upside_pct'),
            market_sentiment = market_sentiment,
            pc_ratio         = opt.get('pc_ratio'),
            insider_label    = ins.get('label', 'NEUTRAL'),
        )
        c['news_score']           = ns
        c['news_score_breakdown'] = ns_bd
        c['vader_label']          = vader_lbl
        c['options_pc']             = opt.get('pc_ratio')
        c['options_label']          = opt.get('label', 'NEUTRAL')
        c['unusual_call_activity']  = opt.get('unusual_calls', False)
        c['insider_label']        = ins.get('label', 'NEUTRAL')
        cg = (congress_data or {}).get(t, {})
        c['congress_count']   = cg.get('count', 0)
        c['congress_label']   = 'BUYING' if cg.get('count', 0) >= 1 else 'NEUTRAL'
        c['congress_notes']   = ', '.join(cg.get('names', [])[:4])   # e.g. "Smith (R-TX) $15K-50K [H], Tuberville (R-AL) $100K-250K [S]"
        c['congress_chamber'] = ', '.join(cg.get('chamber', []))
        cg_dates = cg.get('dates', [])
        try:
            c['congress_days_ago'] = min(
                (datetime.now() - datetime.strptime(d, '%Y-%m-%d')).days for d in cg_dates if d
            ) if cg_dates else None
        except Exception:
            c['congress_days_ago'] = None
        c['pre_score']            = ts + ns

    cg_hits = [c['ticker'] for c in candidates if c.get('congress_label') == 'BUYING']
    if cg_hits:
        print(f'  Congress buys in pool: {cg_hits}')
    top5 = sorted(candidates, key=lambda x: x['pre_score'], reverse=True)[:5]
    print('  Pre-score top 5: '
          + '  '.join(f'{c["ticker"]}({c["pre_score"]}='
                      f'{c["tech_score"]}T+{c["news_score"]}N [vader:{c["vader_label"]}])'
                      for c in top5))
    return candidates


def load_performance_history(fp):
    """Load ALL evaluated picks with full indicator context for self-calibration."""
    if not os.path.exists(fp):
        return []
    try:
        df   = pd.read_csv(fp)
        done = df[df['Result'].isin(['Win', 'Loss', 'Neutral'])]
        if done.empty:
            return []
        hist = []
        for _, row in done.iterrows():
            hist.append({
                'ticker':           str(row.get('Ticker', '')),
                'date':             str(row.get('Date', '')),
                'confidence':       str(row.get('Confidence', '')),
                'source':           str(row.get('Source', '')),
                'sector':           str(row.get('Sector', '')),
                'result':           str(row.get('Result', '')),
                'return_pct':       str(row.get('Return_Pct', '')),
                'vs_qqq_10d':       str(row.get('vs_QQQ_10d', '')),
                'close_reason':     str(row.get('Close_Reason', '')),
                'rsi':              str(row.get('RSI', '')),
                'vix':              str(row.get('VIX', '')),
                'vix_regime':       str(row.get('VIX_Regime', '')),
                'qqq_trend':        str(row.get('QQQ_Trend', '')),
                'earnings_days':    str(row.get('Earnings_Days_Away', '')),
                'congress':         str(row.get('Congress', '')),
                'insider':          str(row.get('Insider', '')),
                'position_size_pct':str(row.get('Position_Size_Pct', '')),
                'rr':               str(row.get('RR', '')),
                'reasoning':        str(row.get('Reasoning', ''))[:120],
            })
        return hist
    except:
        return []


def build_learning_insights(pick_history):
    """Analyze patterns in wins vs losses so the LLM can adjust its scoring."""
    if not pick_history or len(pick_history) < 3:
        return ''

    wins   = [h for h in pick_history if h['result'] == 'Win']
    losses = [h for h in pick_history if h['result'] == 'Loss']

    def wr_str(subset):
        if not subset: return 'no data'
        w = sum(1 for h in subset if h['result'] == 'Win')
        return f'{w}/{len(subset)} wins ({w/len(subset)*100:.0f}%)'

    def avg_field(subset, key):
        vals = []
        for h in subset:
            try:
                v = float(h.get(key, ''))
                vals.append(v)
            except (ValueError, TypeError):
                pass
        return f'{sum(vals)/len(vals):.1f}' if vals else 'n/a'

    lines = ['PATTERN ANALYSIS FROM PAST PICKS:']

    # Win rate by source
    for src in ['TECHNICAL', 'NEWS', 'BOTH']:
        sub = [h for h in pick_history if h.get('source', '').upper() == src]
        if sub:
            lines.append(f'  Source {src}: {wr_str(sub)}')

    # Win rate by sector (only sectors with 2+ picks)
    sectors = {}
    for h in pick_history:
        s = h.get('sector', 'Unknown')
        sectors.setdefault(s, []).append(h)
    for s, sub in sectors.items():
        if len(sub) >= 2:
            lines.append(f'  Sector {s}: {wr_str(sub)}')

    # Avg indicator scores: winners vs losers
    if wins and losses:
        lines.append(f'  Avg Tech_Score  — wins: {avg_field(wins,"tech_score")}  losses: {avg_field(losses,"tech_score")}')
        lines.append(f'  Avg News_Score  — wins: {avg_field(wins,"news_score")}  losses: {avg_field(losses,"news_score")}')
        lines.append(f'  Avg Confidence  — wins: {avg_field(wins,"confidence")}  losses: {avg_field(losses,"confidence")}')
        lines.append(f'  Avg RSI         — wins: {avg_field(wins,"rsi")}  losses: {avg_field(losses,"rsi")}')

    # Recent losses with full context so LLM can reason about what went wrong
    recent_losses = [h for h in pick_history[-15:] if h['result'] == 'Loss']
    if recent_losses:
        lines.append('RECENT LOSSES — what went wrong:')
        for h in recent_losses:
            lines.append(
                f'  {h["date"]} {h["ticker"]} | conf={h["confidence"]} src={h["source"]} sector={h["sector"]}'
                f' | Tech={h["tech_score"]} News={h["news_score"]} RSI={h["rsi"]} VIX={h["vix"]} QQQ={h["qqq_trend"]}'
                f' | return={h["return_30d"]}%30d vsQQQ={h["vs_qqq_30d"]}%'
                f' | was: {h["reasoning"]}'
            )

    return '\n'.join(lines)


# ── TICKER UNIVERSE ────────────────────────────────────────
NASDAQ_100 = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AVGO','COST',
    'NFLX','ASML','TMUS','AMD','ADBE','CSCO','PEP','INTU','CMCSA','HON',
    'AMGN','SBUX','QCOM','AMAT','ISRG','ARM','BKNG','TXN','PANW','VRTX',
    'ADI','REGN','LRCX','MDLZ','MU','KLAC','CDNS','SNPS','CEG','CTAS',
    'FTNT','MELI','ABNB','KDP','PYPL','MAR','ORLY','MNST','PCAR','MRNA',
    'CRWD','NXPI','TEAM','DXCM','BIIB','IDXX','PAYX','ROST','ODFL','FAST',
    'EXC','FANG','CTSH','AEP','GEHC','VRSK','MCHP','XEL','ON','TTWO',
    'ZS','DLTR','EA','CCEP','CDW','GFS','ILMN','BKR','DDOG',
    'EBAY','ENPH','ALGN','AZN','INTC','SMCI','LULU','WDAY','DASH','ROP',
    'CPRT','CSGP','WBD','SIRI','GEN','APP','AXON','TTD','MSTR','COIN',
]

SP500_STOCKS = [
    # Financials
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','CB','AON',
    'ICE','CME','SPGI','MCO','TRV','PGR','AFL','MET','PRU','AIG','ALL',
    'USB','PNC','TFC','COF','SYF','AMP','BK','STT','NTRS','RF',
    'CFG','HBAN','KEY','MTB','FITB','ZION','FOUR','HOOD',
    # Healthcare
    'UNH','JNJ','LLY','MRK','ABT','TMO','DHR','BSX','EW','SYK','BDX',
    'ZTS','GILD','HCA','CI','ELV','HUM','MOH','CNC','CVS','MCK',
    'CAH','COR','HOLX','RMD','BAX','HSIC','VTRS','OGN','SOLV',
    # Energy
    'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','DVN','APA',
    'OXY','HAL','OVV','CTRA','RRC','EQT','KMI','WMB','OKE','ET','TRGP','LNG',
    # Nuclear and Clean Energy
    'OKLO','SMR','IMSR','NNE','CCJ','LEU',
    # Industrials
    'GE','GEV','MMM','CAT','DE','RTX','LMT','NOC','GD','LHX','TDG',
    'BA','UPS','FDX','NSC','CSX','UNP','EMR','ETN','ITW','PH','ROK',
    'AME','CARR','OTIS','GWW','SWK','IR','XYL','GNRC',
    # Consumer Discretionary
    'WMT','HD','MCD','NKE','LOW','TGT','TJX','ROST',
    'GM','F','APTV','BWA','LEA','MGA','GNTX',
    'CMG','YUM','QSR','DPZ','TXRH','EAT','DRI',
    # Consumer Staples
    'PG','KO','MO','PM','KHC','GIS','CPB','HRL',
    'SJM','MKC','CLX','CHD','CL','EL','PPC',
    # Technology
    'IBM','ORCL','CRM','NOW','SNOW','PLTR','DELL','HPQ','HPE','NTAP',
    'PSTG','NTNX','MANH','VEEV','HUBS','NET','MDB','OKTA',
    'SHOP','SOFI','AFRM','ADSK','ROKU','KVUE',
    'MRVL','QRVO','SWKS','CRUS','TER','ENTG','ACLS','VLTO',
    # Crypto and Digital Assets
    'MARA','RIOT','IREN','CIFR','WULF','SBET','ZETA','CRCL',
    # AI and Emerging Tech
    'NBIS','CRWV',
    # Real Estate
    'AMT','PLD','CCI','EQIX','SPG','O','WELL','VTR','EQR','AVB',
    'ESS','MAA','UDR','CPT',
    # Utilities
    'NEE','DUK','SO','D','SRE','PCG','EIX','ES',
    'WEC','DTE','CMS','AES','NRG','VST','ETR','PPL',
    # Materials
    'LIN','APD','ECL','SHW','NEM','FCX','NUE','STLD','CLF','AA',
    'CF','MOS','FMC','ALB','SQM','MP','SCCO',
    # Communications and Media
    'VZ','T','LUMN','NWSA','NWS','LYV',
    'MGM','WYNN','LVS','CZR','DKNG',
    'CCL','RCL','NCLH','AAL','DAL','UAL','LUV','ALK',
    'DIS','CHTR','OMC',
]

KEY_ETFS = [
    'SPY','QQQ','IWM','DIA','VTI','VOO',
    'XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLB','XLRE','XLC',
    'ARKK','SOXX','SMH','IBB','GLD','SLV','USO','VXX','UVXY',
]

SECTOR_ETF_MAP = {
    'Technology':             'XLK',
    'Financial Services':     'XLF',
    'Energy':                 'XLE',
    'Healthcare':             'XLV',
    'Industrials':            'XLI',
    'Consumer Cyclical':      'XLY',
    'Consumer Defensive':     'XLP',
    'Utilities':              'XLU',
    'Basic Materials':        'XLB',
    'Real Estate':            'XLRE',
    'Communication Services': 'XLC',
}

# ── YOUR PRIORITY STOCKS ──────────────────────────────────────────────────────
# These are ALWAYS evaluated by the LLM regardless of pre-score.
# Add stocks you'd actually consider buying. Broad scan still runs for discovery.
MY_STOCKS = [
    # Add your tickers here, e.g.:
    # 'AAPL', 'NVDA', 'TSLA', 'MSFT',
]
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_dynamic_universe():
    """
    Fetch S&P 500 + S&P 400 MidCap from Wikipedia (~900 stocks, no API key needed).
    Falls back to the hardcoded NASDAQ_100 + SP500_STOCKS if fetch fails.
    """
    tickers = []
    sources = []

    import io as _io

    def _wiki_table(url):
        """Fetch Wikipedia page via requests (timeout-safe) then parse with read_html."""
        r = requests.get(url, timeout=20, headers={'User-Agent': 'StockScreener/1.0'})
        r.raise_for_status()
        return pd.read_html(_io.StringIO(r.text))[0]

    # S&P 500 — large cap
    try:
        df    = _wiki_table('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = [str(t).replace('.', '-') for t in df['Symbol'].tolist()]  # BRK.B → BRK-B
        tickers.extend(sp500)
        sources.append(f'S&P 500 ({len(sp500)})')
    except Exception as e:
        print(f'  S&P 500 Wikipedia fetch failed ({e}) — using hardcoded fallback')
        tickers.extend(NASDAQ_100 + SP500_STOCKS)
        sources.append(f'hardcoded fallback ({len(set(NASDAQ_100 + SP500_STOCKS))})')

    # S&P 400 MidCap — expands into mid-cap opportunities
    try:
        df    = _wiki_table('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
        col   = next((c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()), df.columns[1])
        sp400 = [str(t).replace('.', '-') for t in df[col].tolist() if isinstance(t, str) and t.isascii()]
        tickers.extend(sp400)
        sources.append(f'S&P 400 MidCap ({len(sp400)})')
    except Exception as e:
        print(f'  S&P 400 Wikipedia fetch failed ({e}) — skipping midcap expansion')

    unique = list(dict.fromkeys(tickers))
    print(f'  Dynamic universe: {" + ".join(sources)} = {len(unique)} unique tickers')
    return unique

_dynamic_stocks  = _fetch_dynamic_universe()
TICKER_UNIVERSE  = list(dict.fromkeys(_dynamic_stocks + KEY_ETFS + MY_STOCKS))
UNIVERSE_SET     = set(TICKER_UNIVERSE)
ETF_SET          = set(KEY_ETFS)
STOCK_UNIVERSE   = [t for t in TICKER_UNIVERSE if t not in ETF_SET]

print('✅ Universe loaded:')
print(f'   Stocks:    {len(STOCK_UNIVERSE)}')
print(f'   ETFs:      {len(ETF_SET)}')
print(f'   TOTAL:     {len(TICKER_UNIVERSE)}')


def get_market_context():
    print('\nMarket context...')
    try:
        vix_hist  = yf.Ticker('^VIX').history(period='1y')['Close']
        vix_l     = round(float(vix_hist.iloc[-1]), 2)
        vix_pct   = round(float(vix_hist.rank(pct=True).iloc[-1]) * 100, 1)
    except:
        vix_l, vix_pct = 20.0, 50.0

    if   vix_pct < _CFG_VIX_LOW_PCTILE:  vr, vm = f'LOW (p{vix_pct:.0f} risk-on)',       1.10
    elif vix_pct < _CFG_VIX_HIGH_PCTILE: vr, vm = f'MODERATE (p{vix_pct:.0f} neutral)',  1.00
    elif vix_pct < 90:                    vr, vm = f'ELEVATED (p{vix_pct:.0f} cautious)', 0.85
    else:                                 vr, vm = f'HIGH (p{vix_pct:.0f} risk-off)',      0.70

    try:
        qh  = yf.Ticker('QQQ').history(period='60d')
        qc  = float(qh['Close'].iloc[-1])
        q50 = float(qh['Close'].rolling(50).mean().iloc[-1])
        qt  = 'BULLISH' if qc > q50 else 'BEARISH'
        qv  = round(((qc - q50) / q50) * 100, 2)
        qp  = round(qc, 2)
    except:
        qt, qv, qp = 'UNKNOWN', 0.0, 0.0

    try:
        spy_hist = yf.Ticker('SPY').history(period='5d')
        spy_ret  = round(((float(spy_hist['Close'].iloc[-1]) -
                           float(spy_hist['Close'].iloc[-2])) /
                           float(spy_hist['Close'].iloc[-2])) * 100, 2)
    except:
        spy_ret = 0.0

    # Global macro — 10Y yield, dollar, overnight markets
    _GLOBAL = {
        'yield_10y':  '^TNX',      # US 10-Year Treasury yield
        'dxy':        'DX-Y.NYB',  # Dollar index
        'es_futures': 'ES=F',      # S&P 500 e-mini futures (pre-market direction)
        'nq_futures': 'NQ=F',      # Nasdaq 100 e-mini futures (pre-market direction)
        'nzdusd':     'NZDUSD=X',  # NZD/USD live rate (Sharesies coverage calc)
        'nikkei':     '^N225',     # Japan (overnight)
        'dax':        '^GDAXI',    # Germany (overnight)
        'ftse':       '^FTSE',     # UK (overnight)
    }
    global_macro = {}
    for name, sym in _GLOBAL.items():
        try:
            h = yf.Ticker(sym).history(period='5d')
            if not h.empty and len(h) >= 2:
                latest = round(float(h['Close'].iloc[-1]), 2)
                prev   = float(h['Close'].iloc[-2])
                chg    = round((latest - prev) / prev * 100, 2) if prev else 0.0
                global_macro[name] = {'price': latest, 'chg_pct': chg}
        except:
            pass

    ctx = {
        'vix_level':        vix_l,
        'vix_percentile':   vix_pct,
        'vix_regime':       vr,
        'vix_multiplier':   vm,
        'qqq_trend':        qt,
        'qqq_vs_ma50':      qv,
        'qqq_price':        qp,
        'spy_return_today': spy_ret,
        'defensive_mode':   vix_pct > 90 and qt == 'BEARISH',
        'global_macro':     global_macro,
        'sector_1d':        {},   # filled after batch_download in run_screener
    }
    print(f'  VIX:  {vix_l} (p{vix_pct}) -> {vr} ({vm}x)')
    print(f'  QQQ:  ${qp} | {qt} ({qv:+.2f}% vs 50MA)')
    print(f'  SPY today: {spy_ret:+.2f}%')
    if global_macro:
        tnx = global_macro.get('yield_10y', {})
        dxy = global_macro.get('dxy', {})
        nk  = global_macro.get('nikkei', {})
        es  = global_macro.get('es_futures', {})
        nq  = global_macro.get('nq_futures', {})
        print(f'  10Y yield: {tnx.get("price","?")}% ({tnx.get("chg_pct",0):+.2f}%)  '
              f'DXY: {dxy.get("price","?")} ({dxy.get("chg_pct",0):+.2f}%)  '
              f'Nikkei: {nk.get("chg_pct",0):+.2f}%')
        if es or nq:
            print(f'  Futures:  ES={es.get("chg_pct",0):+.2f}%  NQ={nq.get("chg_pct",0):+.2f}%')
    if ctx['defensive_mode']:
        print('  *** DEFENSIVE MODE ACTIVE ***')
    return ctx


def batch_download(tickers):
    """Download OHLCV for all tickers in ONE yf.download() call."""
    print(f'\nBatch downloading {len(tickers)} tickers (1 API call)...')
    try:
        raw = yf.download(
            tickers, period='65d',
            auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            print('  Batch returned empty'); return {}

        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    df = raw.xs(t, axis=1, level=1).dropna(how='all')
                    if not df.empty and len(df) >= 20:
                        result[t] = df
                except:
                    continue
        else:
            if len(tickers) == 1 and not raw.empty:
                result[tickers[0]] = raw

        print(f'  Downloaded: {len(result)}/{len(tickers)} tickers')
        return result
    except Exception as e:
        print(f'  Batch error: {e}'); return {}


def _fetch_stock_news_single(ticker):
    """Fetch news for one ticker — title + summary for richer LLM context."""
    try:
        news   = yf.Ticker(ticker).news or []
        items  = []
        for n in news[:8]:
            title   = (n.get('content', {}).get('title',   '') or n.get('title',   '')).strip()
            summary = (n.get('content', {}).get('summary', '') or n.get('summary', '')).strip()
            if not title:
                continue
            text = title.lower()
            if summary and len(summary) > 20:
                text += ' — ' + summary[:160].lower()
            items.append(text)
        return ticker, items
    except:
        return ticker, []


def fetch_all_stock_news_parallel(tickers):
    """Fetch yfinance .news for ALL tickers in parallel."""
    print(f'  Fetching news for all {len(tickers)} stocks (parallel)...')
    results = {}
    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as executor:
        futures = {executor.submit(_fetch_stock_news_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, news = future.result()
            results[ticker] = news
    has_news = sum(1 for v in results.values() if v)
    print(f'  News found: {has_news}/{len(tickers)} stocks have news today')
    return results


def _fetch_fundamentals_single(ticker):
    """Fetch fundamentals for one ticker — reuses same Ticker object for all calls."""
    result = {
        'earnings_date':      'Unknown',
        'earnings_days_away': -1,
        'earnings_risk':      False,
        'analyst_rating':     None,
        'analyst_target':     None,
        'analyst_actions':    [],
        'short_ratio':        None,
        'short_pct_float':    None,  # % of float sold short (squeeze signal)
        'upside_pct':         None,
        'revenue_growth':     None,  # YoY revenue growth %
        'earnings_growth':    None,  # YoY earnings growth %
        'open_gap_pct':       None,  # today's open vs yesterday's close
        'premarket_gap_pct':  None,  # pre-market price vs yesterday's close
        'sector':             'Unknown',
        'mkt_cap_b':          0.0,
    }
    tk = yf.Ticker(ticker)  # one object, reused for all property calls below

    try:
        info  = tk.info
        rec   = info.get('recommendationMean')
        tgt   = info.get('targetMeanPrice')
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        sr    = info.get('shortRatio')
        sf    = info.get('shortPercentOfFloat')
        sec   = info.get('sector', 'Unknown')
        mc    = info.get('marketCap', 0)
        rg    = info.get('revenueGrowth')
        eg    = info.get('earningsGrowth')

        if rec:   result['analyst_rating']  = round(float(rec), 1)
        if tgt:   result['analyst_target']  = round(float(tgt), 2)
        if tgt and price and float(price) > 0:
            result['upside_pct'] = round(((float(tgt) - float(price)) / float(price)) * 100, 1)
        if sr:    result['short_ratio']     = round(float(sr), 1)
        if sf:    result['short_pct_float'] = round(float(sf) * 100, 1)
        # Short squeeze setup: heavy short load + high days-to-cover = rocket fuel if catalyst hits
        if sf and sr:
            sf_val = float(sf) * 100
            sr_val = float(sr)
            if sf_val > _CFG_SQUEEZE_FLOAT_PCT and sr_val > _CFG_SQUEEZE_DAYS_COVER:
                result['short_squeeze_setup'] = True
                result['short_squeeze_label'] = f'SQUEEZE SETUP: {sf_val:.0f}% float short, {sr_val:.1f}d to cover'
        if rg:    result['revenue_growth']  = round(float(rg) * 100, 1)
        if eg:    result['earnings_growth'] = round(float(eg) * 100, 1)
        result['sector']    = sec or 'Unknown'
        result['mkt_cap_b'] = round((mc or 0) / 1e9, 1)

        # Gap: today's open vs yesterday's close
        reg_open  = info.get('regularMarketOpen')
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
        if reg_open and prev_close and float(prev_close) > 0:
            result['open_gap_pct'] = round((float(reg_open) - float(prev_close)) / float(prev_close) * 100, 2)

        # Pre-market price vs yesterday's close (only meaningful before market open)
        pre_price = info.get('preMarketPrice')
        if pre_price and prev_close and float(prev_close) > 0:
            result['premarket_gap_pct'] = round((float(pre_price) - float(prev_close)) / float(prev_close) * 100, 2)
    except:
        pass

    try:
        cal   = tk.calendar
        today = datetime.now().date()
        ed_raw = None

        if isinstance(cal, dict):
            ed_raw = cal.get('Earnings Date')
            if isinstance(ed_raw, list) and ed_raw:
                ed_raw = ed_raw[0]
        elif cal is not None and hasattr(cal, 'empty') and not cal.empty:
            if 'Earnings Date' in cal.index:
                vals = cal.loc['Earnings Date'].values
                if len(vals) > 0:
                    ed_raw = vals[0]

        if ed_raw is not None:
            ed        = pd.to_datetime(ed_raw).date()
            days_away = (ed - today).days
            result['earnings_date']      = str(ed)
            result['earnings_days_away'] = days_away
            result['earnings_risk']      = 0 <= days_away <= 5
    except:
        pass

    try:
        ud = tk.upgrades_downgrades
        if ud is not None and not ud.empty:
            cutoff  = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=14)
            ud.index = pd.to_datetime(ud.index, utc=True)
            recent  = ud[ud.index >= cutoff].head(4)
            actions = []
            for idx, row in recent.iterrows():
                firm    = str(row.get('Firm', '?'))
                action  = str(row.get('Action', ''))
                to_g    = str(row.get('To Grade', ''))
                from_g  = str(row.get('From Grade', ''))
                date_s  = idx.strftime('%m/%d')
                if to_g or from_g:
                    actions.append(f'{date_s} {firm}: {from_g}→{to_g} ({action})')
            result['analyst_actions'] = actions
    except:
        pass

    return ticker, result


def fetch_all_fundamentals_parallel(tickers):
    """Fetch fundamentals for ALL tickers in parallel."""
    print(f'  Fetching fundamentals for all {len(tickers)} stocks (parallel)...')
    results = {}
    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as executor:
        futures = {executor.submit(_fetch_fundamentals_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            results[ticker] = data
    full = sum(1 for v in results.values()
               if v['analyst_rating'] is not None and v['earnings_date'] != 'Unknown')
    print(f'  Full fundamentals: {full}/{len(tickers)} stocks')
    return results


def fetch_sec_8k(tickers, days=7):
    """
    Fetch recent 8-K filings from SEC EDGAR for a list of tickers.
    Free, no API key. Only call for candidates (~20 stocks), not the full universe.
    8-K = material corporate event: FDA approval, contract win, guidance change, M&A.
    """
    from datetime import timedelta
    import xml.etree.ElementTree as ET_sec
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    results = {}
    headers = {'User-Agent': 'StockScreener vishvesh.niyati@gmail.com'}

    def fetch_one(ticker):
        try:
            url = (
                f'https://www.sec.gov/cgi-bin/browse-edgar'
                f'?action=getcompany&CIK={ticker}&type=8-K'
                f'&dateb=&owner=include&count=5&search_text=&output=atom'
            )
            r = requests.get(url, timeout=10, headers=headers)
            if r.status_code != 200:
                return ticker, []
            root = ET_sec.fromstring(r.content)
            ns   = {'atom': 'http://www.w3.org/2005/Atom'}
            filings = []
            for entry in root.findall('atom:entry', ns):
                title   = (entry.findtext('atom:title',   '', ns) or '').strip()
                updated = (entry.findtext('atom:updated', '', ns) or '')[:10]
                summary = (entry.findtext('atom:summary', '', ns) or '').strip()[:200]
                if updated >= start and title:
                    filings.append(f'{updated}: {title}' + (f' — {summary}' if summary else ''))
            return ticker, filings
        except:
            return ticker, []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for f in as_completed(futures):
            t, filings = f.result()
            results[t] = filings

    hits = sum(1 for v in results.values() if v)
    if hits:
        print(f'  SEC EDGAR 8-K: {hits}/{len(tickers)} candidates have recent filings')
    return results


def fetch_macro_news():
    """RSS feeds — Reuters removed (shut down 2020), replaced with reliable alternatives."""
    feeds = [
        ('US_MARKET',    'https://finance.yahoo.com/rss/topstories'),
        ('US_MARKET',    'https://finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US'),
        ('US_MARKET',    'https://www.cnbc.com/id/100003114/device/rss/rss.html'),
        ('US_MARKET',    'https://feeds.marketwatch.com/marketwatch/topstories/'),
        ('US_MARKET',    'https://feeds.marketwatch.com/marketwatch/bulletins/'),
        ('GEOPOLITICAL', 'https://feeds.bbci.co.uk/news/world/rss.xml'),
        ('GEOPOLITICAL', 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'),
        ('US_POLICY',    'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml'),
        ('US_POLICY',    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml'),
        ('ENERGY',       'https://www.eia.gov/rss/news.xml'),           # US Energy Info Admin (official)
        ('ENERGY',       'https://oilprice.com/rss/main'),
        ('FED',          'https://www.federalreserve.gov/feeds/press_all.xml'),  # Fed Reserve (official)
        ('FED',          'https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml'),
        ('EARNINGS',     'https://finance.yahoo.com/rss/2.0/headline?s=earnings&region=US&lang=en-US'),
        ('TECH',         'https://www.cnbc.com/id/19854910/device/rss/rss.html'),
        ('SECTOR',       'https://feeds.marketwatch.com/marketwatch/marketpulse/'),
    ]
    headlines = []
    active = 0
    for category, url in feeds:
        try:
            r    = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(r.content)
            ct   = 0
            for item in root.iter('item'):
                title = item.findtext('title', '').strip()
                if title and len(title) > 10:
                    headlines.append(f'[{category}] {title}')
                    ct += 1
                    if ct >= 6: break
            if ct > 0:
                active += 1
        except:
            continue
    print(f'  RSS feeds active: {active}/{len(feeds)} | Headlines: {len(headlines)}')
    return headlines[:60]


def derive_sector_sentiment(candidates):
    """Derive sector context from already-fetched stock news/scores — no ETF API needed."""
    from collections import defaultdict
    buckets = defaultdict(lambda: {'news_scores': [], 'tickers': [], 'momentum': []})
    for c in candidates:
        sec = c.get('sector', '')
        if not sec or sec in ('', 'Unknown', 'N/A'): continue
        buckets[sec]['tickers'].append(c['ticker'])
        buckets[sec]['news_scores'].append(c.get('news_score', 0))
        buckets[sec]['momentum'].append(c.get('momentum_5d', 0))
    result = {}
    for sec, data in buckets.items():
        n = len(data['tickers'])
        if n == 0: continue
        avg_news = sum(data['news_scores']) / n
        avg_mom  = sum(data['momentum']) / n
        sentiment = ('POSITIVE' if avg_news > 20 or avg_mom > 1.5
                     else 'NEGATIVE' if avg_news < 10 and avg_mom < -1.5
                     else 'NEUTRAL')
        result[sec] = {'sentiment': sentiment, 'n': n,
                       'avg_news': round(avg_news, 1), 'avg_mom': round(avg_mom, 2),
                       'tickers': data['tickers'][:4]}
    print(f'  Sector sentiment derived: {len(result)}/{len(SECTOR_ETF_MAP)} sectors '
          f'from {sum(d["n"] for d in result.values())} stocks')
    return result


def compute_indicators(df, spy_return_today=0.0):
    """Compute all technical indicators from OHLCV DataFrame."""
    if df is None or len(df) < 20:
        return None
    try:
        hi = df['High']
        lo = df['Low']
        cl = df['Close'].dropna()
        vo = df['Volume']
        if len(cl) < 20:
            return None
        p  = float(cl.iloc[-1])

        if not (p > 0):  # catches NaN and non-positive
            return None

        m20  = float(cl.rolling(20).mean().iloc[-1])
        m50  = float(cl.rolling(50).mean().iloc[-1]) if len(cl) >= 50 else p
        m200 = float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else p

        av     = float(vo.rolling(30).mean().iloc[-1]) if len(vo) >= 30 else float(vo.mean())
        vr     = float(vo.iloc[-1]) / av if av > 0 else 0.0
        dvol_m = round((p * av) / 1e6, 1)

        if len(vo) >= 3:
            v1, v2, v3 = float(vo.iloc[-3]), float(vo.iloc[-2]), float(vo.iloc[-1])
            vol_accel = v3 > v2 > v1
        else:
            vol_accel = False

        delta     = cl.diff()
        gain      = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss      = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        loss_safe = loss.replace(0, 1e-10)
        rsi_val   = float((100 - (100 / (1 + gain / loss_safe))).iloc[-1]) if len(cl) >= 14 else 50.0

        p5  = float(cl.iloc[-5])  if len(cl) >= 5  else p
        p20 = float(cl.iloc[-20]) if len(cl) >= 20 else p

        w52_high  = float(cl.rolling(min(252, len(cl))).max().iloc[-1])
        w52_low   = float(cl.rolling(min(252, len(cl))).min().iloc[-1])
        pct_from_52h = round(((p - w52_high) / w52_high) * 100, 1) if w52_high > 0 else 0.0

        prev_cl = cl.shift(1)
        tr = pd.concat([hi - lo,
                        (hi - prev_cl).abs(),
                        (lo - prev_cl).abs()], axis=1).max(axis=1)
        atr     = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())
        atr_pct = round((atr / p) * 100, 2) if p > 0 else 0.0

        ema12     = cl.ewm(span=12, adjust=False).mean()
        ema26     = cl.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        sig_line  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - sig_line
        macd_bull = bool(macd_line.iloc[-1] > sig_line.iloc[-1] and float(macd_hist.iloc[-1]) > 0)

        up_move  = hi.diff()
        dn_move  = -lo.diff()
        plus_dm  = pd.Series(
            np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
            index=cl.index)
        minus_dm = pd.Series(
            np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
            index=cl.index)
        atr_w    = tr.ewm(alpha=1/14, adjust=False).mean()
        safe_atr = atr_w.replace(0, 1e-10)
        plus_di  = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / safe_atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / safe_atr)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        adx_val  = float(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]) if len(dx) >= 14 else 0.0

        bb_mid   = cl.rolling(20).mean()
        bb_std   = cl.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = float((bb_upper - bb_lower).iloc[-1])
        bb_pct_b = float((p - float(bb_lower.iloc[-1])) / bb_range) if bb_range > 0 else 0.5

        obv = (np.sign(cl.diff()) * vo).fillna(0).cumsum()
        obv_trend = float(obv.iloc[-1]) > float(obv.iloc[-10]) if len(obv) >= 10 else False

        mfv = ((cl - lo) - (hi - cl)) / (hi - lo).replace(0, 1e-10) * vo
        cmf_val = (float(mfv.rolling(20).sum().iloc[-1]) /
                   float(vo.rolling(20).sum().replace(0, 1e-10).iloc[-1])) if len(cl) >= 20 else 0.0
        cmf_val = round(max(-1.0, min(1.0, cmf_val)), 3)

        rsi_series = 100 - (100 / (1 + gain / loss_safe))
        if len(rsi_series) >= 28:
            rsi_min14 = float(rsi_series.rolling(14).min().iloc[-1])
            rsi_max14 = float(rsi_series.rolling(14).max().iloc[-1])
            rng = rsi_max14 - rsi_min14
            stoch_rsi_val = round((float(rsi_series.iloc[-1]) - rsi_min14) / (rng + 1e-10), 3) if rng > 1 else 0.5
        else:
            stoch_rsi_val = 0.5

        if len(hi) >= 5:
            h_vals = [float(hi.iloc[-i]) for i in range(1, 6)]
            l_vals = [float(lo.iloc[-i]) for i in range(1, 6)]
            hh_hl_val = (h_vals[0] > h_vals[1] > h_vals[2] and
                         l_vals[0] > l_vals[1] > l_vals[2])
        else:
            hh_hl_val = False

        tp      = (hi + lo + cl) / 3
        vwap_20 = float((tp * vo).rolling(20).sum().iloc[-1] /
                        vo.rolling(20).sum().replace(0, 1e-10).iloc[-1]) if len(cl) >= 20 else p
        vs_vwap = round(((p - vwap_20) / vwap_20) * 100, 2) if vwap_20 > 0 else 0.0

        if len(cl) >= 2:
            stock_ret_today = ((p - float(cl.iloc[-2])) / float(cl.iloc[-2])) * 100
            rs_vs_spy       = round(stock_ret_today - spy_return_today, 2)
        else:
            rs_vs_spy = 0.0

        return {
            'price':          round(p, 2),
            'ma20':           round(m20, 2),
            'ma50':           round(m50, 2),
            'ma200':          round(m200, 2),
            'vol_ratio':      round(vr, 2),
            'dollar_vol_m':   dvol_m,
            'vol_accel':      vol_accel,
            'rsi':            round(rsi_val, 1),
            'vs_ma20_pct':    round(((p - m20) / m20) * 100, 2) if m20 > 0 else 0.0,
            'vs_ma50_pct':    round(((p - m50) / m50) * 100, 2) if m50 > 0 else 0.0,
            'vs_ma200_pct':   round(((p - m200) / m200) * 100, 2) if m200 > 0 else 0.0,
            'momentum_5d':    round(((p - p5) / p5) * 100, 2) if p5 > 0 else 0.0,
            'momentum_20d':   round(((p - p20) / p20) * 100, 2) if p20 > 0 else 0.0,
            'pct_from_52h':   pct_from_52h,
            'w52_high':       round(w52_high, 2),
            'atr':            round(atr, 2),
            'atr_pct':        atr_pct,
            'macd_bullish':   macd_bull,
            'adx':            round(adx_val, 1),
            'bb_pct_b':       round(bb_pct_b, 2),
            'obv_rising':     obv_trend,
            'rs_vs_spy':      rs_vs_spy,
            'cmf':            cmf_val,
            'stoch_rsi':      stoch_rsi_val,
            'hh_hl':          hh_hl_val,
            'vs_vwap_pct':    vs_vwap,
        }
    except Exception as e:
        if os.environ.get('SCREENER_DEBUG'):
            print(f'  [debug] compute_indicators failed: {type(e).__name__}: {e}')
        return None


def screen_technical(batch_data, ctx):
    """Screen ALL tickers using batch data."""
    is_weekend    = datetime.now().weekday() >= 5
    vol_threshold = 1.0 if is_weekend else _CFG_VOLUME_MIN_RATIO

    print(f'\nPhase 2 - Technical screening ({len(batch_data)} tickers)...')
    passed   = {}
    rejects  = {'price': 0, 'ma': 0, 'vol': 0, 'rsi': 0, 'liquidity': 0, 'adx': 0}

    for t, df in batch_data.items():
        if t in ETF_SET:
            continue
        ind = compute_indicators(df, ctx['spy_return_today'])
        if ind is None:
            rejects['price'] += 1; continue
        if _CFG_REQUIRE_ABOVE_MA and (ind['vs_ma20_pct'] < 0 or ind['vs_ma50_pct'] < 0):
            rejects['ma'] += 1; continue
        if ind['vol_ratio'] < vol_threshold:
            rejects['vol'] += 1; continue
        if not (RSI_MIN <= ind['rsi'] <= RSI_MAX):
            rejects['rsi'] += 1; continue
        if ind['dollar_vol_m'] < MIN_DOLLAR_VOLUME_M:
            rejects['liquidity'] += 1; continue
        if ind['adx'] < ADX_MIN:
            rejects['adx'] += 1; continue
        passed[t] = ind

    print(f'  Passed: {len(passed)} | Rejected: '
          f'Price:{rejects["price"]} MA:{rejects["ma"]} '
          f'Vol:{rejects["vol"]} RSI:{rejects["rsi"]} '
          f'Liq:{rejects["liquidity"]} ADX:{rejects["adx"]}')
    return passed


def screen_news(batch_data, all_stock_news, technical_passed, ctx):
    """Check ALL stocks for significant news, rescue failed ones with major news."""
    print(f'\nPhase 3 - News screening all {len(batch_data)} stocks...')
    rescued = {}
    checked = rescued_count = 0

    for t, news_titles in all_stock_news.items():
        if t in ETF_SET or t in technical_passed or not news_titles:
            continue
        checked += 1
        rescue, t1, t2r, t2c = has_significant_news(news_titles)
        keyword_hits = t1 + t2r + t2c
        if not rescue:
            continue
        df  = batch_data.get(t)
        ind = compute_indicators(df, ctx['spy_return_today']) if df is not None else None
        if ind is None or ind['price'] <= MIN_PRICE or ind['dollar_vol_m'] < MIN_DOLLAR_VOLUME_M:
            continue
        rescued[t] = ind
        rescued[t]['rescue_keywords'] = keyword_hits[:5]
        rescued_count += 1

    rescued_tickers = list(rescued.keys())
    print(f'  News rescued: {rescued_count} stocks → {rescued_tickers[:10]}{"..." if rescued_count>10 else ""}')
    return rescued


def merge_candidates(technical_passed, news_rescued, all_stock_news, fundamentals):
    """Merge technical and news pools."""
    print(f'\nPhase 4 - Merging candidate pools...')
    candidates = []

    for t, ind in technical_passed.items():
        news = all_stock_news.get(t, [])
        has_news, t1, t2r, t2c = has_significant_news(news)
        keyword_hits = t1 + t2r + t2c
        source = 'BOTH' if has_news else 'TECHNICAL'
        fund   = fundamentals.get(t, {})
        candidates.append({
            'ticker': t, 'source': source, 'news_sourced': source in ('NEWS','BOTH'),
            **ind,
            'sector':             fund.get('sector', 'Unknown'),
            'mkt_cap_b':          fund.get('mkt_cap_b', 0.0),
            'analyst_rating':     fund.get('analyst_rating'),
            'analyst_target':     fund.get('analyst_target'),
            'upside_pct':         fund.get('upside_pct'),
            'short_ratio':        fund.get('short_ratio'),
            'earnings_date':      fund.get('earnings_date', 'Unknown'),
            'earnings_days_away': fund.get('earnings_days_away', -1),
            'earnings_risk':      fund.get('earnings_risk', False),
            'stock_news':         news[:5],
            'analyst_actions':    fund.get('analyst_actions', []),
            'short_pct_float':    fund.get('short_pct_float'),
            'revenue_growth':     fund.get('revenue_growth'),
            'earnings_growth':    fund.get('earnings_growth'),
            'open_gap_pct':       fund.get('open_gap_pct'),
            'premarket_gap_pct':  fund.get('premarket_gap_pct'),
            'rescue_keywords':    keyword_hits[:5] if keyword_hits else [],
        })

    for t, ind in news_rescued.items():
        fund = fundamentals.get(t, {})
        news = all_stock_news.get(t, [])
        candidates.append({
            'ticker': t, 'source': 'NEWS', 'news_sourced': True,
            **ind,
            'sector':             fund.get('sector', 'Unknown'),
            'mkt_cap_b':          fund.get('mkt_cap_b', 0.0),
            'analyst_rating':     fund.get('analyst_rating'),
            'analyst_target':     fund.get('analyst_target'),
            'upside_pct':         fund.get('upside_pct'),
            'short_ratio':        fund.get('short_ratio'),
            'earnings_date':      fund.get('earnings_date', 'Unknown'),
            'earnings_days_away': fund.get('earnings_days_away', -1),
            'earnings_risk':      fund.get('earnings_risk', False),
            'stock_news':         news[:5],
            'analyst_actions':    fund.get('analyst_actions', []),
            'short_pct_float':    fund.get('short_pct_float'),
            'revenue_growth':     fund.get('revenue_growth'),
            'earnings_growth':    fund.get('earnings_growth'),
            'open_gap_pct':       fund.get('open_gap_pct'),
            'premarket_gap_pct':  fund.get('premarket_gap_pct'),
            'rescue_keywords':    ind.get('rescue_keywords', []),
        })

    both = sum(1 for c in candidates if c['source'] == 'BOTH')
    tech = sum(1 for c in candidates if c['source'] == 'TECHNICAL')
    news = sum(1 for c in candidates if c['source'] == 'NEWS')
    print(f'  Total candidates: {len(candidates)}')
    print(f'    BOTH (tech+news): {both}  TECHNICAL: {tech}  NEWS: {news}')
    er = [c['ticker'] for c in candidates if c.get('earnings_risk')]
    if er: print(f'  Earnings risk flags: {er}')
    return candidates


def get_news_intelligence(candidates, ctx, headlines, sector_news, all_stock_news, sec_filings=None):
    """3-layer news intelligence via LLM."""
    print(f'\nPhase 5 - News intelligence ({len(candidates)} candidates, 3 layers)...')

    sectors_in_pool  = list(set([c['sector'] for c in candidates if c['sector'] != 'Unknown']))
    macro_text       = '\n'.join(headlines[:20])
    sector_text      = '\n'.join([
        f'[{sec}] {v["sentiment"]} avg_news={v["avg_news"]} mom={v["avg_mom"]:+.2f}% tickers={",".join(v["tickers"][:3])}'
        for sec, v in sector_news.items() if sec in sectors_in_pool
    ]) or 'No sector news'
    stock_news_pool  = {c['ticker']: all_stock_news.get(c['ticker'], []) for c in candidates}
    # Show top 3 headlines+summaries per stock (was just 1 title before)
    stock_text       = '\n'.join([
        f'{t}: {" | ".join(h[:3])}'
        for t, h in stock_news_pool.items() if h
    ]) or 'No individual stock news'

    # Analyst upgrades/downgrades for candidates (last 14 days)
    action_lines = []
    for c in candidates:
        acts = c.get('analyst_actions', [])
        if acts:
            action_lines.append(f'{c["ticker"]}: {" | ".join(acts)}')
    analyst_actions_text = '\n'.join(action_lines) or 'No recent analyst actions'

    # SEC EDGAR 8-K filings for candidates
    edgar_lines = []
    for c in candidates:
        filings = (sec_filings or {}).get(c['ticker'], [])
        if filings:
            edgar_lines.append(f'{c["ticker"]}: {" | ".join(filings[:2])}')
    edgar_text = '\n'.join(edgar_lines) or 'No recent 8-K filings'

    all_tickers    = [c['ticker'] for c in candidates]
    both_tickers   = [c['ticker'] for c in candidates if c['source'] == 'BOTH']
    news_tickers   = [c['ticker'] for c in candidates if c['source'] in ('NEWS','BOTH')]
    earnings_risks = [c['ticker'] for c in candidates if c.get('earnings_risk')]
    analyst_lines  = [f'{c["ticker"]}:rating={c["analyst_rating"]},upside={c["upside_pct"]}%' for c in candidates if c.get('analyst_rating') is not None]
    today          = datetime.now().strftime('%B %d %Y')

    both_note     = f'\nSTRONGEST (passed tech AND have news): {both_tickers}' if both_tickers else ''
    news_note     = f'\nNEWS-SOURCED (relaxed tech filters): {news_tickers}'   if news_tickers else ''
    earnings_2w   = [c['ticker'] for c in candidates if 0 <= c.get('earnings_days_away', 999) <= 14]
    earnings_note = f'\nEARNINGS WITHIN 2 WEEKS (elevated risk for short-term trades): {earnings_2w}' if earnings_2w else ''

    system = 'You are a financial analyst. Respond with ONLY a valid JSON object. Start with { and end with }. No markdown, no explanation.'
    user = (
        f'Today: {today}\n'
        f'VIX={ctx["vix_level"]} (p{ctx["vix_percentile"]}) | QQQ={ctx["qqq_trend"]} ({ctx["qqq_vs_ma50"]:+.2f}% vs 50MA) | SPY today={ctx["spy_return_today"]:+.2f}%\n\n'
        f'LAYER 1 - MACRO HEADLINES:\n{macro_text}\n\n'
        f'LAYER 2 - SECTOR NEWS:\n{sector_text}\n\n'
        f'LAYER 3 - STOCK NEWS (title + summary):\n{stock_text}\n\n'
        f'ANALYST UPGRADES/DOWNGRADES (last 14 days):\n{analyst_actions_text}\n\n'
        f'SEC EDGAR 8-K FILINGS (last 7 days — material corporate events):\n{edgar_text}\n\n'
        f'ANALYST CONSENSUS (mean rating 1=Strong Buy, 5=Sell | price target upside):\n{chr(10).join(analyst_lines) if analyst_lines else "No data"}\n\n'
        f'ALL CANDIDATES: {all_tickers}{both_note}{news_note}{earnings_note}\n'
        f'SECTORS: {sectors_in_pool}\n\n'
        'Return this JSON:\n'
        '{"macro_summary":"2-3 sentences",'
        '"trump_signal":{"detected":false,"detail":"none","affected_sectors":[],"score_adjustment":0},'
        '"fed_signal":{"detected":false,"detail":"none","tone":"neutral","score_adjustment":0},'
        '"macro_data_signal":{"detected":false,"detail":"none","score_adjustment":0},'
        '"geopolitical_signal":{"detected":false,"detail":"none","score_adjustment":0},'
        '"stock_signals":[{"ticker":"X","news":"summary","auto_drop":false,"score_adjustment":0}],'
        '"sector_signals":[{"sector":"X","news":"summary","score_adjustment":0}],'
        '"overall_market_adjustment":0,'
        '"market_sentiment":"NEUTRAL"}'
    )

    try:
        raw = call_llm(system, user, max_tokens=2000)
        if '```' in raw:
            for part in raw.split('```'):
                part = part.strip()
                if part.startswith('json'): part = part[4:].strip()
                if part.startswith('{'): raw = part; break
        if '{' in raw and '}' in raw:
            raw = raw[raw.index('{'):raw.rindex('}')+1]
        nd = json.loads(raw)
        print(f'  Sentiment: {nd.get("market_sentiment","?")} | Adj: {int(nd.get("overall_market_adjustment",0)):+d}')
        return nd
    except Exception as e:
        print(f'  News intelligence failed ({e}) - neutral baseline')
        return {'macro_summary':'Unavailable','trump_signal':{'detected':False,'score_adjustment':0,'affected_sectors':[]},'fed_signal':{'detected':False,'score_adjustment':0,'tone':'neutral'},'macro_data_signal':{'detected':False,'score_adjustment':0},'geopolitical_signal':{'detected':False,'score_adjustment':0},'stock_signals':[],'sector_signals':[],'overall_market_adjustment':0,'market_sentiment':'NEUTRAL'}


def apply_news(candidates, nd):
    """Apply news adjustments including auto-drops and earnings penalties."""
    sm   = {s['ticker']: s for s in nd.get('stock_signals', [])}
    secm = {s['sector']:  s for s in nd.get('sector_signals', [])}
    oadj = nd.get('overall_market_adjustment', 0)
    tr   = nd.get('trump_signal', {})

    enriched = []
    for c in candidates:
        adj, drop, notes = oadj, False, []
        if tr.get('detected') and c['sector'] in tr.get('affected_sectors', []):
            ta = tr.get('score_adjustment', 0)
            if ta <= -20: drop = True; notes.append('AUTO DROP: Trump targeting sector')
            else: adj += ta; notes.append(f'Trump:{int(ta):+d}')
        if c['ticker'] in sm:
            sn = sm[c['ticker']]
            if sn.get('auto_drop'): drop = True; notes.append('AUTO DROP: negative stock news')
            else:
                sa = sn.get('score_adjustment', 0); adj += sa
                if sa: notes.append(f'News:{int(sa):+d}')
        if c['sector'] in secm:
            sa = secm[c['sector']].get('score_adjustment', 0); adj += sa
            if sa: notes.append(f'Sector:{int(sa):+d}')
        if c.get('earnings_risk'):
            adj -= 15; notes.append('Earnings:-15')
        c['news_adjustment'] = adj
        c['auto_drop']       = drop
        c['news_notes']      = ' | '.join(notes) if notes else 'No major news'
        enriched.append(c)

    dropped = [c for c in enriched if c['auto_drop']]
    remain  = [c for c in enriched if not c['auto_drop']]
    if dropped: print(f'  AUTO DROPPED: {", ".join([c["ticker"] for c in dropped])}')
    print(f'  {len(remain)} candidates after news filter')
    return remain


def stream_b_from_headlines(headlines, batch_data, technical_passed, all_stock_news, fundamentals, ctx):
    """Extract tickers specifically mentioned in macro headlines."""
    if not headlines: return []
    try:
        raw = call_llm(
            system='Extract US stock tickers from headlines. Return ONLY a JSON array like ["AAPL","GOOGL"]. No explanation.',
            user=f'HEADLINES:\n{chr(10).join(headlines)}\n\nRules: US stocks only, no ETFs, no indices, max 15 tickers, [] if none.',
            max_tokens=200
        )
        if '[' in raw:
            raw = raw[raw.index('['):]
            depth = 0
            for i, ch in enumerate(raw):
                if ch=='[': depth+=1
                elif ch==']':
                    depth-=1
                    if depth==0: raw=raw[:i+1]; break
        news_tickers = [t.upper().strip() for t in json.loads(raw) if isinstance(t,str)]
    except:
        return []

    existing   = set(technical_passed.keys())
    new_tickers = [t for t in news_tickers if t in UNIVERSE_SET and t not in existing and t not in ETF_SET]
    if not new_tickers: return []

    print(f'  Stream B: Processing headline mentions: {new_tickers}')
    b_cands = []
    for t in new_tickers:
        df  = batch_data.get(t)
        ind = compute_indicators(df, ctx['spy_return_today']) if df is not None else None
        if ind is None or ind['price'] <= MIN_PRICE: continue
        fund = fundamentals.get(t, {})
        news = all_stock_news.get(t, [])
        b_cands.append({
            'ticker':t,'source':'NEWS','news_sourced':True,**ind,
            'sector':fund.get('sector','Unknown'),'mkt_cap_b':fund.get('mkt_cap_b',0.0),
            'analyst_rating':fund.get('analyst_rating'),'analyst_target':fund.get('analyst_target'),
            'upside_pct':fund.get('upside_pct'),'short_ratio':fund.get('short_ratio'),
            'earnings_date':fund.get('earnings_date','Unknown'),'earnings_days_away':fund.get('earnings_days_away',-1),
            'earnings_risk':fund.get('earnings_risk',False),'stock_news':news[:5],
            'analyst_actions':fund.get('analyst_actions',[]),
            'short_pct_float':fund.get('short_pct_float'),'revenue_growth':fund.get('revenue_growth'),
            'earnings_growth':fund.get('earnings_growth'),'open_gap_pct':fund.get('open_gap_pct'),
            'premarket_gap_pct':fund.get('premarket_gap_pct'),'rescue_keywords':[],
        })
        print(f'  Stream B added: {t} | ${ind["price"]} | RSI {ind["rsi"]}')
    return b_cands


def batch_catalyst_score(candidates, ctx, all_stock_news):
    """LLM scores every candidate's short-term catalyst quality. 8 stocks per call."""
    BATCH = 8
    batches = [candidates[i:i+BATCH] for i in range(0, len(candidates), BATCH)]
    n_calls = len(batches)
    print(f'  Catalyst scoring: {len(candidates)} stocks → {n_calls} LLM calls...')
    sys_msg = ('You are a short-to-medium term equity trader (1-4 week holds). '
               'Rate each stock purely on near-term tradability. Respond ONLY with valid JSON.')
    for bi, batch in enumerate(batches):
        lines = []
        for c in batch:
            news = ' | '.join((all_stock_news.get(c['ticker'], []))[:3]) or 'No news'
            lines.append(
                f'{c["ticker"]} [{c["sector"]}] RSI={c["rsi"]:.0f} ADX={c["adx"]:.0f} '
                f'mom5d={c["momentum_5d"]:+.1f}% tech={c.get("tech_score",0)} '
                f'earnings_in={c.get("earnings_days_away","?")}d '
                f'news="{news}"'
            )
        user_msg = (
            f'Market: VIX={ctx["vix_level"]:.1f} ({ctx["vix_regime"][:10]}) | '
            f'QQQ={ctx["qqq_trend"]} | SPY={ctx["spy_return_today"]:+.2f}%\n\n'
            f'Rate each for SHORT-TERM trading (1-4 weeks):\n' + '\n'.join(lines) + '\n\n'
            f'For each:\n'
            f'- catalyst_score: 1-10 (10=strong specific near-term catalyst, 1=no reason to buy now)\n'
            f'- catalyst_type: EARNINGS_CATALYST|UPGRADE|BREAKOUT|SECTOR_ROTATION|MOMENTUM|NEWS_HYPE|NONE\n'
            f'- auto_drop: true if news is clearly negative or there is zero short-term reason to buy\n'
            f'- reason: one sentence — the specific 1-4 week thesis or why dropping\n\n'
            f'Return ONLY: {{"ratings":[{{"ticker":"X","catalyst_score":7,"catalyst_type":"BREAKOUT",'
            f'"auto_drop":false,"reason":"..."}}]}}'
        )
        try:
            raw = call_llm(sys_msg, user_msg, max_tokens=900)
            if '{' in raw and '}' in raw:
                raw = raw[raw.index('{'):raw.rindex('}')+1]
            for r in json.loads(raw).get('ratings', []):
                t = r.get('ticker', '')
                match = next((c for c in candidates if c['ticker'] == t), None)
                if match:
                    match['catalyst_score']  = r.get('catalyst_score', 5)
                    match['catalyst_type']   = r.get('catalyst_type', 'NONE')
                    match['short_term_reason'] = r.get('reason', '')
                    if r.get('auto_drop'):
                        match['auto_drop'] = True
                        match['news_notes'] = f'AUTO DROP: {r.get("reason","no short-term catalyst")}'
            print(f'    Batch {bi+1}/{n_calls} ✓')
        except Exception as e:
            print(f'    Batch {bi+1}/{n_calls} failed: {e}')
    dropped = sum(1 for c in candidates if c.get('auto_drop'))
    print(f'  Catalyst scoring done — {dropped} auto-dropped, {len(candidates)-dropped} remain')
    return candidates


def analyze_exit_signals(ctx, all_stock_news, portfolio=None):
    """Check every pending pick — LLM says HOLD / EXIT / ADD for each."""
    for fp, label in [(PICKS_CSV, 'BUY'), (WATCH_CSV, 'WATCH')]:
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp)
        pending = df[df['Result'] == 'Pending']
        if pending.empty: continue
        print(f'  Exit check: {len(pending)} pending {label} picks...')
        for _, row in pending.iterrows():
            ticker = str(row.get('Ticker', '')).strip()
            if not ticker or ticker in ('', 'nan', 'NONE'): continue
            entry_str = str(row.get('Entry_Price', 'N/A')).strip()
            pick_date = str(row.get('Date', '')).strip()
            try:
                days_in = (datetime.now() - datetime.strptime(pick_date, '%Y-%m-%d')).days
            except: days_in = 0
            try:
                curr = round(float(yf.Ticker(ticker).history(period='2d')['Close'].iloc[-1]), 2)
                entry = float(entry_str)
                unreal = round((curr - entry) / entry * 100, 1)
                price_line = f'Entry=${entry} Current=${curr} Unrealized={unreal:+.1f}%'
            except:
                price_line = f'Entry={entry_str} (current price unavailable)'
                unreal = None
            news = ' | '.join((all_stock_news or {}).get(ticker, [])[:3]) or 'No fresh news'
            sys_msg = 'You are managing an open short-to-medium term position (1-4 week horizon). Respond ONLY with valid JSON. Never use double-quote characters inside string values.'
            user_msg = (
                f'OPEN {label}: {ticker} | {price_line} | Days held: {days_in}/30\n'
                f'Original thesis: {str(row.get("Reasoning",""))[:120]}\n'
                f'Stop zone: {row.get("Stop_Zone","N/A")} | Target: {row.get("Target_Zone","N/A")}\n'
                f'Today\'s news: {news}\n'
                f'Market: VIX={ctx["vix_level"]:.1f} | QQQ={ctx["qqq_trend"]}\n\n'
                f'Decision for a 1-4 week trade:\n'
                f'- action: EXIT (take profit or cut loss now) | HOLD | ADD (strong setup, consider adding)\n'
                f'- urgency: HIGH | MEDIUM | LOW\n'
                f'- reason: one sentence, no double-quote characters inside\n'
                f'- exit_price: suggested exit price or null\n\n'
                f'Return ONLY: {{"ticker":"{ticker}","action":"HOLD","urgency":"LOW","reason":"...","exit_price":null}}'
            )
            try:
                raw = call_llm(sys_msg, user_msg, max_tokens=400)
                rec = None
                # Try proper JSON first
                raw_json = raw
                if '{' in raw_json and '}' in raw_json:
                    raw_json = raw_json[raw_json.index('{'):raw_json.rindex('}')+1]
                try:
                    rec = json.loads(raw_json)
                except json.JSONDecodeError:
                    # Fallback: regex-extract individual fields (handles unescaped quotes in reason)
                    import re as _re
                    _a = _re.search(r'"action"\s*:\s*"([^"]+)"', raw)
                    _u = _re.search(r'"urgency"\s*:\s*"([^"]+)"', raw)
                    _r = _re.search(r'"reason"\s*:\s*"(.+?)(?:","exit_price|"\s*\})', raw, _re.DOTALL)
                    rec = {
                        'action':  _a.group(1) if _a else 'HOLD',
                        'urgency': _u.group(1) if _u else 'LOW',
                        'reason':  _r.group(1) if _r else '',
                    }
                action = rec.get('action', 'HOLD')
                urgency = rec.get('urgency', 'LOW')
                reason = rec.get('reason', '')
                unreal_str = f'{unreal:+.1f}%' if unreal is not None else '?'
                if action == 'EXIT':
                    icon = '!'
                    # Close position in portfolio if held there
                    if portfolio and curr and curr > 0:
                        portfolio = close_position(portfolio, ticker, curr, reason=f'LLM EXIT: {reason[:60]}')
                elif action == 'ADD' and urgency == 'HIGH':
                    icon = '+'
                else:
                    icon = '-'
                print(f'  [{icon}] {action} [{urgency}] {ticker} ({unreal_str} in {days_in}d): {reason}')
            except Exception as e:
                print(f'  {ticker}: exit check failed ({e})')
    return portfolio


def analyze_with_nvidia(candidates, ctx, nd, pick_history=None, portfolio=None):
    """3-round LLM deliberation: rank → deep-dive → final pick. No 1-shot guessing."""
    if not candidates: return None
    print(f'\nPhase 6 - NVIDIA final scoring ({len(candidates)} candidates)...')

    if len(candidates) > _CFG_FINAL_CANDIDATES:
        my_set = set(t.upper() for t in MY_STOCKS)
        priority = [c for c in candidates if c['ticker'].upper() in my_set]
        rest     = [c for c in candidates if c['ticker'].upper() not in my_set]
        rest_sorted = sorted(rest, key=lambda x: x.get('pre_score',0), reverse=True)
        slots = max(0, _CFG_FINAL_CANDIDATES - len(priority))
        candidates = priority + rest_sorted[:slots]
        if my_set:
            print(f'  Trimmed to top {_CFG_FINAL_CANDIDATES}: {len(priority)} priority + {len(candidates)-len(priority)} broad scan')
        else:
            print(f'  Trimmed to top {_CFG_FINAL_CANDIDATES} by pre_score')

    mult   = ctx['vix_multiplier']

    hard_rule_flags = {}
    filtered = []
    for c in candidates:
        rsi=c.get('rsi',50); upside=c.get('upside_pct'); t=c['ticker']; flags=[]
        rsi_cap_on    = _CFG_RSI_HARD_CAP    < 999
        upside_cap_on = _CFG_UPSIDE_HARD_CAP > -999
        if rsi_cap_on and upside_cap_on and rsi > _CFG_RSI_HARD_CAP and upside is not None and upside < _CFG_UPSIDE_HARD_CAP:
            print(f'  DISQUALIFIED: {t} | RSI={rsi}>{_CFG_RSI_HARD_CAP} AND analyst target {upside}% below {_CFG_UPSIDE_HARD_CAP}%'); continue
        if rsi_cap_on and rsi > _CFG_RSI_HARD_CAP:
            flags.append(f'RSI_CAP(rsi={rsi}>{_CFG_RSI_HARD_CAP},max_conf={_CFG_RSI_CAP_CONF})')
        if upside_cap_on and upside is not None and upside < _CFG_UPSIDE_HARD_CAP:
            flags.append(f'ANALYST_CAP(upside={upside}%<{_CFG_UPSIDE_HARD_CAP}%,max_conf={_CFG_UPSIDE_CAP_CONF})')
        if flags: hard_rule_flags[t]=flags
        filtered.append(c)

    if not filtered:
        return {'top_pick':{'ticker':'NONE','confidence':0,'signal':'NO PICK','reasoning':'All disqualified.','key_risk':'N/A','sector':'N/A','source':'N/A'},'watch_candidates':[]}
    candidates = filtered

    compact = [{
        'ticker':c['ticker'],'source':c['source'],'price':c['price'],'sector':c['sector'],
        'tech_score':c.get('tech_score',0),'news_score':c.get('news_score',0),'pre_score':c.get('pre_score',0),
        'tech_bd':c.get('tech_score_breakdown',{}),'news_bd':c.get('news_score_breakdown',{}),
        'rsi':c['rsi'],'adx':c['adx'],'vol_ratio':c['vol_ratio'],'macd_bull':c['macd_bullish'],
        'momentum_5d':c['momentum_5d'],'rs_vs_spy':c['rs_vs_spy'],'pct_from_52h':c['pct_from_52h'],
        'cmf':c.get('cmf',0.0),'stoch_rsi':c.get('stoch_rsi',0.5),'hh_hl':c.get('hh_hl',False),
        'vader_label':c.get('vader_label','NEUTRAL'),'options_label':c.get('options_label','NEUTRAL'),
        'insider_label':c.get('insider_label','NEUTRAL'),'news_adjustment':c.get('news_adjustment',0),
        'congress_label':c.get('congress_label','NEUTRAL'),'congress_notes':c.get('congress_notes',''),
        'congress_days_ago':c.get('congress_days_ago'),
        'news_notes':c.get('news_notes',''),'mkt_cap_b':c['mkt_cap_b'],
        **({k:c[k] for k in ['analyst_rating','upside_pct','short_ratio','short_pct_float','short_squeeze_label','earnings_risk','rescue_keywords','options_pc','unusual_call_activity'] if c.get(k) is not None})
    } for c in candidates]

    hist_block = ''
    if not pick_history:
        hist_block = (
            '\nNO TRADING HISTORY YET.\n'
            'derived_rules MUST be exactly: ["No history yet (n=0) — using baseline judgment. Rules will emerge after first picks are evaluated."]\n'
            'Do not invent sample sizes or win rates. There is no data to derive rules from.\n'
        )
    if pick_history:
        wins  = sum(1 for h in pick_history if h['result'] == 'Win')
        total = len(pick_history)
        wr    = round(wins / total * 100, 1) if total else 0

        # Group history by market regime so LLM applies regime-specific rules
        def _regime(h):
            vix = h.get('vix', '')
            qqq = h.get('qqq_trend', '')
            try: v = float(vix)
            except: v = 20
            vix_bucket = 'HIGH_VIX' if v > 25 else 'LOW_VIX' if v < 15 else 'MID_VIX'
            qqq_bucket = 'BULL' if 'bull' in str(qqq).lower() else 'BEAR' if 'bear' in str(qqq).lower() else 'NEUTRAL'
            return f'{vix_bucket}_{qqq_bucket}'

        from collections import defaultdict
        regime_groups = defaultdict(list)
        for h in pick_history:
            regime_groups[_regime(h)].append(h)

        def _fmt(h):
            r30 = h.get('return_30d', '')
            mag = f'{float(r30):+.1f}%' if r30 and r30 not in ('nan','None','') else '?'
            return (f'    {h["date"]} {h["ticker"]} src={h["source"]} sector={h["sector"]} '
                    f'conf={h["confidence"]} Tech={h["tech_score"]} News={h["news_score"]} '
                    f'RSI={h["rsi"]} ADX={h["adx"]} -> {h["result"]} 30d={mag} vsQQQ={h["vs_qqq_30d"]}%'
                    f' | {h["reasoning"][:80]}')

        # Current regime for today
        curr_vix = ctx.get('vix_level', 20)
        curr_qqq = ctx.get('qqq_trend', 'NEUTRAL')
        curr_regime = _regime({'vix': curr_vix, 'qqq_trend': curr_qqq})

        regime_lines = ''
        for reg, picks in sorted(regime_groups.items()):
            reg_wins = sum(1 for p in picks if p['result'] == 'Win')
            reg_wr = round(reg_wins / len(picks) * 100) if picks else 0
            r30_vals = []
            for p in picks:
                try: r30_vals.append(float(p['return_30d']))
                except: pass
            avg_ret = f'{sum(r30_vals)/len(r30_vals):+.1f}%' if r30_vals else '?'
            marker = ' ← TODAY\'S REGIME' if reg == curr_regime else ''
            regime_lines += f'\n  REGIME: {reg} | {len(picks)} picks | {reg_wr}% win rate | avg 30d return: {avg_ret}{marker}\n'
            regime_lines += '\n'.join(_fmt(h) for h in picks) + '\n'

        hist_block = (
            f'\n── YOUR FULL TRADING HISTORY ({total} evaluated picks | {wr}% win rate) ──\n'
            f'TODAY\'S REGIME: {curr_regime} (VIX={curr_vix:.1f}, QQQ={curr_qqq})\n'
            f'{regime_lines}\n'
            f'SELF-OPTIMIZATION INSTRUCTIONS:\n'
            f'Step 1 — Study the regime groups above. Focus MOST on the group marked TODAY\'S REGIME\n'
            f'         since that\'s the market condition you are picking in right now.\n'
            f'Step 2 — Derive your own rules from the data. For each rule, note: how many picks\n'
            f'         support it (n=X), the win rate (wr=Y%), and the avg magnitude of wins/losses.\n'
            f'         A rule based on n<3 picks should be flagged as LOW CONFIDENCE.\n'
            f'Step 3 — Weight rules by: (a) regime match to today, (b) sample size, (c) magnitude.\n'
            f'         A rule with n=10, wr=80%, avg+12% beats a rule with n=2, wr=100%, avg+0.5%.\n'
            f'Step 4 — Apply your rules when scoring today\'s candidates. Reference them in reasoning.\n'
            f'Rules must be derived from the data. Do not invent rules not supported by history.\n'
        )

    hard_note = ('\nHARD CAPS APPLIED:\n' + '\n'.join([f'{t}: {" ".join(fs)}' for t,fs in hard_rule_flags.items()]) + '\nRSI_CAP=max 70 | ANALYST_CAP=max 65\n') if hard_rule_flags else ''
    both_t=[c['ticker'] for c in candidates if c['source']=='BOTH']
    news_t=[c['ticker'] for c in candidates if c['source']=='NEWS']
    src_note = (f'\nBOTH (strongest): {both_t}' if both_t else '') + (f'\nNEWS-SOURCED: {news_t}' if news_t else '')
    my_set = set(t.upper() for t in MY_STOCKS)
    priority_present = [c['ticker'] for c in candidates if c['ticker'].upper() in my_set]
    priority_note = f'\nUSER PRIORITY STOCKS (always evaluate these, even if scores are modest): {priority_present}\n' if priority_present else ''

    SYS = ('You are a short-to-medium term equity trader (1-4 week holds). '
           'NOT a long-term investor. Momentum, catalysts, and near-term price action matter most. '
           'Analyst 12-month targets are nearly irrelevant — focus on what moves in 1-4 weeks. '
           'Earnings within 2 weeks = elevated risk. Earnings within 5 days = near-disqualifier. '
           'CRITICAL RULE: Before committing to any BUY, you MUST articulate a credible bear case — '
           'specific reasons the trade could fail in the next 10 days. '
           'If you cannot name at least 2 concrete failure scenarios, the pick is not ready. '
           'High conviction with a weak bear case is overconfidence, not edge. '
           'Respond ONLY with valid JSON. Start with { end with }. No markdown.')

    gm     = ctx.get('global_macro', {})
    tnx    = gm.get('yield_10y', {})
    dxy    = gm.get('dxy', {})
    esf    = gm.get('es_futures', {})
    nqf    = gm.get('nq_futures', {})
    nk     = gm.get('nikkei', {})
    dax    = gm.get('dax', {})
    nzdusd_llm = gm.get('nzdusd', {})
    nzd_rate_llm = nzdusd_llm.get('price')
    s1d  = ctx.get('sector_1d', {})
    top_s = sorted(s1d.items(), key=lambda x: x[1], reverse=True)
    sector_flow_str = (
        ' | '.join(f'{s}:{v:+.1f}%' for s, v in top_s[:3]) +
        '  WORST: ' + ' | '.join(f'{s}:{v:+.1f}%' for s, v in top_s[-2:])
    ) if top_s else 'N/A'

    market_ctx = (
        f'MARKET: VIX={ctx["vix_level"]:.1f} (p{ctx["vix_percentile"]}) {ctx["vix_regime"]} | '
        f'VIX_mult={mult}x | QQQ={ctx["qqq_trend"]} {ctx["qqq_vs_ma50"]:+.2f}% vs 50MA | '
        f'SPY today={ctx["spy_return_today"]:+.2f}% | Defensive={"YES" if ctx["defensive_mode"] else "NO"}\n'
        f'FUTURES: ES={esf.get("chg_pct",0):+.2f}% | NQ={nqf.get("chg_pct",0):+.2f}%\n'
        f'GLOBAL: 10Y={tnx.get("price","?")}% ({tnx.get("chg_pct",0):+.2f}%) | '
        f'DXY={dxy.get("price","?")} ({dxy.get("chg_pct",0):+.2f}%) | '
        f'Nikkei={nk.get("chg_pct",0):+.2f}% | DAX={dax.get("chg_pct",0):+.2f}%\n'
        f'SECTOR FLOWS TODAY: {sector_flow_str}\n'
        f'MACRO: {nd.get("macro_summary","N/A")[:200]} | Sentiment={nd.get("market_sentiment","NEUTRAL")}\n'
        f'FED: {nd.get("fed_signal",{}).get("detail","none")} | '
        f'TRUMP: {nd.get("trump_signal",{}).get("detail","none")}'
    )

    # ── ROUND 1: Rank all candidates → surface top 10 ───────────────────────
    print(f'  Round 1/3: Ranking {len(candidates)} candidates...')
    compact_brief = [{
        'ticker': c['ticker'], 'sector': c['sector'], 'source': c['source'],
        'pre_score': c.get('pre_score', 0),
        'catalyst_score': c.get('catalyst_score', 5),
        'catalyst_type': c.get('catalyst_type', 'NONE'),
        'short_term_reason': c.get('short_term_reason', ''),
        'rsi': round(c['rsi'], 1), 'adx': round(c['adx'], 1),
        'momentum_5d': round(c['momentum_5d'], 2),
        'vol_ratio': round(c['vol_ratio'], 2),
        'open_gap_pct': c.get('open_gap_pct'),        # gap at today's open
        'premarket_gap_pct': c.get('premarket_gap_pct'),  # pre-market move
        'earnings_days_away': c.get('earnings_days_away', 'N/A'),
        'insider': c.get('insider_label', 'NEUTRAL'),
        'options': c.get('options_label', 'NEUTRAL'),
        'news_notes': c.get('news_notes', '')[:80],
    } for c in candidates]

    portfolio_block = f'\n{portfolio_summary_str(portfolio)}\n' if portfolio else ''
    open_tickers = {p['ticker'] for p in (portfolio or {}).get('positions', [])}
    cash_avail   = (portfolio or {}).get('cash', STARTING_CAPITAL)

    # Pre-compute sector exposure + week P&L for Round 3 sizing context
    _pf_positions = (portfolio or {}).get('positions', [])
    _pf_total_val = (portfolio or {}).get('cash', STARTING_CAPITAL) + sum(
        p.get('current_value', p.get('cost_basis', 0)) for p in _pf_positions)
    _sec_exp = {}
    for _p in _pf_positions:
        _s = _p.get('sector', 'Unknown')
        _sec_exp[_s] = _sec_exp.get(_s, 0) + _p.get('current_value', _p.get('cost_basis', 0))
    sector_str = '  '.join(
        f'{s}: {round(v/_pf_total_val*100)}%' for s, v in sorted(_sec_exp.items(), key=lambda x: -x[1])
    ) if _sec_exp else 'none (no open positions)'
    _week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    _week_trades = [t for t in (portfolio or {}).get('closed_trades', [])
                    if str(t.get('close_date', ''))[:10] >= _week_ago]
    week_str = (f'${sum(t.get("realized_pnl",0) for t in _week_trades):+,.2f} from {len(_week_trades)} trades'
                if _week_trades else 'no closed trades this week')

    r1_user = (
        f'{market_ctx}\n{hard_note}{priority_note}{portfolio_block}\n'
        f'CANDIDATES:\n{json.dumps(compact_brief, indent=1)}\n\n'
        f'TASK: As portfolio manager with ${cash_avail:,.0f} available cash, rank these for a 1-4 week trade.\n'
        f'Already held (do NOT pick again): {list(open_tickers) or "none"}\n'
        f'Max concurrent positions: {_CFG_MAX_POSITIONS} (currently {len(open_tickers)}/{_CFG_MAX_POSITIONS})\n'
        f'Identify:\n'
        f'1. Top 10 to deep-analyse (best short-term setups)\n'
        f'2. Immediate drops (no short-term catalyst, negative news, earnings too soon)\n'
        f'Return ONLY: {{"top10":["TICK1","TICK2"...],"drop":["TICK"],"r1_notes":"2 sentences on what you see"}}'
    )
    top10 = [c['ticker'] for c in candidates[:10]]
    r1_notes = ''
    try:
        raw1 = call_llm(SYS, r1_user, max_tokens=400)
        if '{' in raw1: raw1 = raw1[raw1.index('{'):]
        r1 = json.loads(raw1)
        top10 = r1.get('top10', top10)[:10]
        r1_notes = r1.get('r1_notes', '')
        for t in r1.get('drop', []):
            m = next((c for c in candidates if c['ticker'] == t), None)
            if m: m['auto_drop'] = True; m['news_notes'] = 'R1 drop: no short-term catalyst'
        print(f'  Round 1 → Top 10: {top10}')
    except Exception as e:
        print(f'  Round 1 failed ({e}) — using pre_score top 10')

    # ── ROUND 2: Deep-dive top 10 → bull/bear for each ──────────────────────
    print(f'  Round 2/3: Deep-diving top 10...')
    top10_candidates = [c for c in candidates if c['ticker'] in top10 and not c.get('auto_drop')]
    compact_deep = [{
        'ticker': c['ticker'], 'sector': c['sector'], 'source': c['source'],
        'price': c['price'], 'mkt_cap_b': c['mkt_cap_b'],
        'pre_score': c.get('pre_score', 0), 'tech_score': c.get('tech_score', 0),
        'news_score': c.get('news_score', 0), 'catalyst_score': c.get('catalyst_score', 5),
        'catalyst_type': c.get('catalyst_type', 'NONE'),
        'short_term_reason': c.get('short_term_reason', ''),
        'rsi': round(c['rsi'], 1), 'adx': round(c['adx'], 1),
        'vol_ratio': round(c['vol_ratio'], 2), 'momentum_5d': round(c['momentum_5d'], 2),
        'macd_bull': c['macd_bullish'], 'hh_hl': c.get('hh_hl', False),
        'rs_vs_spy': round(c['rs_vs_spy'], 2), 'pct_from_52h': round(c['pct_from_52h'], 1),
        'cmf': round(c.get('cmf', 0), 3), 'stoch_rsi': round(c.get('stoch_rsi', 0.5), 2),
        'short_ratio': c.get('short_ratio'), 'short_pct_float': c.get('short_pct_float'),
        'earnings_days_away': c.get('earnings_days_away', 'N/A'),
        'open_gap_pct': c.get('open_gap_pct'), 'premarket_gap_pct': c.get('premarket_gap_pct'),
        'upside_pct': c.get('upside_pct'), 'analyst_rating': c.get('analyst_rating'),
        'revenue_growth': c.get('revenue_growth'), 'earnings_growth': c.get('earnings_growth'),
        'analyst_actions': c.get('analyst_actions', []),
        'insider': c.get('insider_label', 'NEUTRAL'), 'options': c.get('options_label', 'NEUTRAL'),
        'congress': c.get('congress_label', 'NEUTRAL'), 'congress_who': c.get('congress_notes', ''),
        'congress_days_ago': c.get('congress_days_ago'),
        'short_squeeze': c.get('short_squeeze_label', ''), 'unusual_calls': c.get('unusual_call_activity', False),
        'news_notes': c.get('news_notes', '')[:120],
    } for c in top10_candidates]

    r2_user = (
        f'{market_ctx}\n\n'
        f'These are your top 10 candidates for a 1-4 week trade.\n'
        f'For each, give a BULL case and BEAR case specific to the next 4 weeks:\n'
        f'{json.dumps(compact_deep, indent=1)}\n\n'
        f'Return ONLY: {{"analyses":[{{"ticker":"X","bull":"specific 1-4 week bull case",'
        f'"bear":"specific 1-4 week risk","short_term_edge":"what makes this better than holding cash","rank":1}}]}}'
    )
    r2_analyses = {}
    try:
        raw2 = call_llm(SYS, r2_user, max_tokens=2000)
        if '{' in raw2 and '}' in raw2:
            raw2 = raw2[raw2.index('{'):raw2.rindex('}')+1]
        r2 = json.loads(raw2)
        for a in r2.get('analyses', []):
            r2_analyses[a['ticker']] = a
        print(f'  Round 2 → analyses for {list(r2_analyses.keys())}')
    except Exception as e:
        print(f'  Round 2 failed ({e})')

    # ── ROUND 3: Final pick — uses rounds 1+2 + full history + regime ────────
    print(f'  Round 3/3: Final deliberation...')
    r2_summary = '\n'.join([
        f'  {t}: BULL={a.get("bull","")} | BEAR={a.get("bear","")} | EDGE={a.get("short_term_edge","")}'
        for t, a in r2_analyses.items()
    ]) or 'Round 2 unavailable'

    r3_user = (
        f'{market_ctx}\n\n'
        f'ROUND 1 NOTES: {r1_notes}\n\n'
        f'ROUND 2 BULL/BEAR ANALYSIS:\n{r2_summary}\n\n'
        f'{src_note}{hist_block}\n\n'
        f'NOW: Make your final decision for a 1-4 week trade.\n'
        f'Apply your self-derived rules from history. Consider regime, catalyst quality, risk/reward.\n'
        f'VIX multiplier: {mult}x. BUY threshold: {BUY_THRESHOLD}. Watch: {WATCH_THRESHOLD}-{BUY_THRESHOLD-1}.\n\n'
        f'Return ONLY:\n'
        f'{{"derived_rules":["Rule (n=X, wr=Y%, HIGH/LOW confidence): <data-backed pattern>"],'
        f'"learning_summary":"One sentence on what history taught you this run.",'
        f'"top_pick":{{"ticker":"X","confidence":85,"signal":"BUY","tech_score":48,"news_score":32,'
        f'"pre_score":80,"catalyst_score":8,"catalyst_type":"BREAKOUT",'
        f'"score_breakdown":"Tech:48 | News:32 | Catalyst:8 | VIX:{mult}x = 85",'
        f'"position_size_pct":0,'
        f'"reasoning":"2 sentences — specific 1-4 week thesis citing the bull case.",'
        f'"devils_advocate":"2 specific risks in the next 4 weeks.",'
        f'"key_risk":"One sentence.","sector":"X","source":"TECHNICAL"}},'
        f'"watch_candidates":[{{"ticker":"X","confidence":74,"signal":"WATCH","reasoning":"1 sentence.","key_risk":"1 sentence.","sector":"X","source":"TECHNICAL"}}]}}\n'
        f'If no stock clears {BUY_THRESHOLD} confidence, signal=NO PICK.\n'
        f'position_size_pct: you decide what % of ${cash_avail:,.0f} available cash to deploy.\n'
        f'Context: {len(open_tickers)} positions currently open. VIX {mult}x regime.\n'
        f'Sector exposure in current portfolio: {sector_str}\n'
        f'Realized P&L this week: {week_str}\n'
        f'There are no rules — size based purely on your conviction, portfolio heat, and risk assessment.'
    )

    try:
        raw3 = call_llm(SYS, r3_user, max_tokens=2000)
        if '```' in raw3:
            for part in raw3.split('```'):
                part = part.strip()
                if part.startswith('json'): part = part[4:].strip()
                if part.startswith('{'): raw3 = part; break
        if '{' in raw3:
            start = raw3.index('{'); depth = 0; end = start
            for i, ch in enumerate(raw3[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0: end = i; break
            raw3 = raw3[start:end+1]
        result = json.loads(raw3)
        pick = result.get('top_pick', {})
        print(f'  Final pick: {pick.get("ticker")} | pre={pick.get("pre_score","?")} → {pick.get("confidence")}/100 | {pick.get("signal")} | {_LLM_CALL_COUNT[0]} LLM calls this run')

        rules = result.get('derived_rules', [])
        summary = result.get('learning_summary', '')
        if rules:
            print(f'\n┌─ LLM SELF-DERIVED RULES {"─"*44}')
            for i, r in enumerate(rules, 1):
                print(f'│ {i}. {r}')
            if summary: print(f'│ → {summary}')
            print(f'└{"─"*62}')
            rules_log = os.path.join(DRIVE_FOLDER, 'rules_log.csv')
            today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
            wr_now = round(sum(1 for h in (pick_history or []) if h['result'] == 'Win') / max(len(pick_history or []), 1) * 100, 1)
            pd.DataFrame([{'Date': today_str, 'Rule': r, 'Learning_Summary': summary, 'Win_Rate': wr_now} for r in rules]).to_csv(
                rules_log, mode='a', header=not os.path.exists(rules_log), index=False)

        return result
    except Exception as e:
        print(f'  Round 3 failed: {e}'); return None


PICK_COLS = [
    # Identity
    'Date','Ticker','Signal','Confidence','Sector','Source',
    # Entry details
    'Entry_Price','Stop_Price','Target_Price','Shares','Cost_Basis','RR',
    # Portfolio context at entry
    'Portfolio_Value','Cash_At_Entry','Position_Size_Pct','Open_Positions',
    # Market context at entry
    'VIX','VIX_Regime','QQQ_Trend','SPY_Day_Pct',
    # Key signals
    'RSI','Earnings_Days_Away','Congress','Congress_Days_Ago','Insider','Options',
    'Analyst_Rating','Upside_Pct','Short_Pct_Float',
    # LLM reasoning
    'Reasoning','Key_Risk','Bear_Case',
    # Outcome
    'Close_Date','Close_Price','Close_Reason',
    'Return_Pct','vs_QQQ_10d','Result'
]
WATCH_COLS = PICK_COLS + ['Watch_Score']


def load_csv(fp, cols):
    if os.path.exists(fp):
        df=pd.read_csv(fp)
        for col in cols:
            if col not in df.columns: df[col]=''
        return df
    return pd.DataFrame(columns=cols)


def check_sector_concentration(sector, picks_csv_path):
    if not os.path.exists(picks_csv_path): return False, 0, []
    try:
        df=pd.read_csv(picks_csv_path)
        if df.empty or 'Sector' not in df.columns: return False, 0, []
        recent=df.tail(_CFG_SECTOR_CONC_LOOKBACK)
        last_sects=recent['Sector'].tolist()
        same_count=sum(1 for s in last_sects if s==sector)
        return same_count>=_CFG_SECTOR_CONC_MAX, same_count, last_sects
    except:
        return False, 0, []


def save_pick(pick_data, ctx, price, fp, cols, all_candidates=None, watch_score=None, portfolio=None, stop_price=None, target_price=None):
    df=load_csv(fp,cols); today=datetime.now().strftime('%Y-%m-%d'); ticker=pick_data['ticker']
    if ((df['Date']==today)&(df['Ticker']==ticker)).any():
        print(f'  Already have {ticker} on {today} - skipping'); return
    match   = next((c for c in (all_candidates or []) if c['ticker']==ticker), {})
    sector  = pick_data.get('sector', 'Unknown')
    if fp==PICKS_CSV:
        is_conc,cnt,_ = check_sector_concentration(sector, PICKS_CSV)
        if is_conc:
            pick_data['confidence'] = max(0, pick_data['confidence'] - _CFG_SECTOR_CONC_PENALTY)

    # Portfolio context at entry
    pf_positions = (portfolio or {}).get('positions', [])
    pf_cash      = (portfolio or {}).get('cash', STARTING_CAPITAL)
    pf_total_val = pf_cash + sum(p.get('current_value', p.get('cost_basis', 0)) for p in pf_positions)
    pos_match    = next((p for p in pf_positions if p['ticker'] == ticker), {})
    shares       = pos_match.get('shares', '')
    cost_basis   = pos_match.get('cost_basis', '')
    # Use explicitly passed stop/target (computed before open_position), fall back to portfolio position
    stop_p = stop_price if stop_price is not None else pos_match.get('stop_price', '')
    tgt_p  = target_price if target_price is not None else pos_match.get('target_price', '')
    rr = ''
    if isinstance(stop_p,(int,float)) and isinstance(tgt_p,(int,float)) and isinstance(price,(int,float)) and price-stop_p>0:
        rr = round((tgt_p-price)/(price-stop_p), 1)
    size_pct = round(float(cost_basis)/pf_total_val*100, 1) if cost_basis and pf_total_val else ''

    # Congress signal
    cg_label    = match.get('congress_label', 'NEUTRAL')
    cg_days_ago = match.get('congress_days_ago')
    cg_str = f'{cg_label} ({cg_days_ago}d ago)' if cg_label == 'BUYING' and cg_days_ago is not None else cg_label

    row = {
        'Date': today, 'Ticker': ticker, 'Signal': pick_data['signal'],
        'Confidence': pick_data['confidence'], 'Sector': sector,
        'Source': pick_data.get('source', 'TECHNICAL'),
        'Entry_Price': price, 'Stop_Price': stop_p, 'Target_Price': tgt_p,
        'Shares': shares, 'Cost_Basis': cost_basis, 'RR': rr,
        'Portfolio_Value': round(pf_total_val, 2), 'Cash_At_Entry': round(pf_cash, 2),
        'Position_Size_Pct': size_pct, 'Open_Positions': len(pf_positions),
        'VIX': ctx['vix_level'], 'VIX_Regime': ctx.get('vix_regime', ''),
        'QQQ_Trend': ctx['qqq_trend'], 'SPY_Day_Pct': ctx.get('spy_return_today', ''),
        'RSI': match.get('rsi', ''),
        'Earnings_Days_Away': match.get('earnings_days_away', ''),
        'Congress': cg_str,
        'Congress_Days_Ago': cg_days_ago,
        'Insider': match.get('insider_label', ''),
        'Options': match.get('options_label', ''),
        'Analyst_Rating': match.get('analyst_rating', ''),
        'Upside_Pct': match.get('upside_pct', ''),
        'Short_Pct_Float': match.get('short_pct_float', ''),
        'Reasoning': pick_data.get('reasoning', ''),
        'Key_Risk': pick_data.get('key_risk', ''),
        'Bear_Case': pick_data.get('bear_case', pick_data.get('devils_advocate', '')),
        'Close_Date': '', 'Close_Price': '', 'Close_Reason': '',
        'Return_Pct': '', 'vs_QQQ_10d': '', 'Result': 'Pending',
    }
    if watch_score is not None: row['Watch_Score'] = watch_score
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(fp, index=False)
    print(f'  ✅ Saved {ticker} | Stop:{stop_p} Target:{tgt_p}')


def update_results(fp, cols):
    if not os.path.exists(fp): return
    df=pd.read_csv(fp); today=datetime.now(); updated=False
    for idx,row in df.iterrows():
        if str(row.get('Ticker','')).strip() in ['NONE','nan','']: continue
        if str(row.get('Entry_Price','')).strip() in ['N/A','nan','']: continue
        try:
            pd_=datetime.strptime(str(row['Date']),'%Y-%m-%d'); el=(today-pd_).days; ep=float(row['Entry_Price'])
            st=yf.Ticker(str(row['Ticker'])); qq=yf.Ticker('QQQ')
            if str(row.get('Realistic_Entry','')).strip() in ['','nan','NaN']:
                ns=pd_+timedelta(days=1)
                if ns.date()<=today.date():
                    h=st.history(start=ns,end=pd_+timedelta(days=5))
                    if not h.empty: df.at[idx,'Realistic_Entry']=round(float(h['Open'].iloc[0]),2); updated=True
            def gp(t):
                h=st.history(start=t-timedelta(days=4),end=t+timedelta(days=4))
                return round(float(h['Close'].iloc[0]),2) if not h.empty else None
            def gq(s,e):
                h=qq.history(start=s-timedelta(days=2),end=e+timedelta(days=4))
                if h.empty or len(h)<2: return None
                return ((float(h['Close'].iloc[-1])-float(h['Close'].iloc[0]))/float(h['Close'].iloc[0]))*100
            hd = _CFG_HOLD_DAYS  # LLM-controlled hold period (default 10)
            if el>=hd and str(row.get('Return_Pct','')).strip() in ['','nan','NaN']:
                p10=gp(pd_+timedelta(days=hd))
                if p10:
                    r10=round(((p10-ep)/ep)*100,2); qr=gq(pd_,pd_+timedelta(days=hd))
                    vs10=round(r10-qr,2) if qr else ''
                    df.at[idx,'Close_Price']=p10
                    df.at[idx,'Return_Pct']=r10
                    df.at[idx,'vs_QQQ_10d']=vs10
                    if not str(row.get('Close_Date','')).strip() or str(row.get('Close_Date','')).strip() in ('','nan','NaN'):
                        df.at[idx,'Close_Date']=(pd_+timedelta(days=hd)).strftime('%Y-%m-%d')
                        df.at[idx,'Close_Reason']='hold_days'
                    # Set Win/Loss using LLM-configurable QQQ-relative thresholds
                    if vs10 != '' and str(row.get('Result','')).strip() in ('','nan','NaN','Pending'):
                        vs10f = float(vs10)
                        if vs10f >= _CFG_WIN_THRESHOLD_PCT:    df.at[idx,'Result'] = 'Win'
                        elif vs10f <= _CFG_LOSS_THRESHOLD_PCT: df.at[idx,'Result'] = 'Loss'
                        else:                                   df.at[idx,'Result'] = 'Neutral'
                    updated=True
        except: continue
    if updated: df.to_csv(fp,index=False); print(f'  {os.path.basename(fp)} updated')


def display_result(result, ctx, nd, ep, wl, all_candidates=None):
    pk   = result.get('top_pick', {})
    conf = pk.get('confidence', 0)
    sig  = pk.get('signal', 'NO PICK')
    src  = pk.get('source', 'TECHNICAL')
    ticker = pk.get('ticker', '')
    fund = next((c for c in (all_candidates or []) if c['ticker'] == ticker), {})
    W = 65

    print('\n' + '█' * W)
    print(f'  TODAY\'S PICK  —  {datetime.now().strftime("%A, %b %d %Y")}')
    print('█' * W)
    print(f'  Market:  VIX {ctx["vix_level"]:.1f} ({ctx["vix_regime"]})  |  QQQ {ctx["qqq_trend"]}  |  SPY {ctx["spy_return_today"]:+.2f}%')
    print(f'  Macro:   {nd.get("macro_summary","Unavailable")[:80]}')
    print('─' * W)

    if sig == 'NO PICK' or conf < BUY_THRESHOLD:
        print('  NO PICK TODAY')
        print(f'  {pk.get("reasoning","Nothing cleared the confidence threshold.")}')
    else:
        atr        = fund.get('atr', 0)
        stop_price = round(ep - ATR_STOP_MULT  * atr, 2) if atr and isinstance(ep, (int, float)) else 'N/A'
        tgt_price  = round(ep + ATR_TARGET_MULT * atr, 2) if atr and isinstance(ep, (int, float)) else 'N/A'
        er_warn    = '  ⚠️  EARNINGS THIS WEEK' if fund.get('earnings_risk') else ''
        conf_bar   = '█' * int(conf / 5) + '░' * (20 - int(conf / 5))

        print(f'  BUY  {ticker}  [{src}]{er_warn}')
        print(f'  Sector:     {pk.get("sector","")}')
        print(f'  Confidence: {conf}/100  {conf_bar}')
        print(f'  Entry:  ~${ep}   Stop: ~${stop_price}   Target: ~${tgt_price}   R:R 1:2')
        print(f'  Score:  {pk.get("score_breakdown","")}')
        print('─' * W)
        print(f'  Why:      {pk.get("reasoning","")}')
        print(f'  Key Risk: {pk.get("key_risk","")}')
        if pk.get('devils_advocate'):
            print(f'  Bear case: {pk.get("devils_advocate","")}')

    if wl:
        print('\n  WATCH LIST:')
        for w in wl:
            print(f'    {w.get("ticker"):<6}  {w.get("confidence")}/100  [{w.get("sector","")}]  {w.get("reasoning","")[:70]}')
    print('█' * W)


def _build_rules_html(rules, summary):
    """Render the LLM self-derived rules block for the HTML report."""
    if not rules:
        return ''
    rule_items = ''.join(
        f'<li style="padding:6px 0;border-bottom:1px solid #f0f0f0;color:#333">'
        f'<span style="color:#1565c0;font-weight:600">Rule {i}:</span> {r}</li>'
        for i, r in enumerate(rules, 1)
    )
    # Load historical rules to show evolution
    rules_log = os.path.join(DRIVE_FOLDER, 'rules_log.csv')
    history_html = ''
    if os.path.exists(rules_log):
        try:
            rdf = pd.read_csv(rules_log)
            past = rdf[~rdf['Date'].str.startswith(datetime.now().strftime('%Y-%m-%d'), na=False)]
            if not past.empty:
                dates = past['Date'].unique()[-5:]
                history_html = '<details style="margin-top:12px"><summary style="cursor:pointer;color:#888;font-size:12px">▶ Show rule evolution (last 5 runs)</summary>'
                for d in reversed(dates):
                    day_rules = past[past['Date']==d]['Rule'].tolist()
                    history_html += f'<div style="margin-top:8px;padding:8px;background:#f8f9fa;border-radius:4px"><b style="font-size:11px;color:#888">{d}</b><ul style="margin-top:4px;padding-left:16px;font-size:12px">'
                    for r in day_rules:
                        history_html += f'<li style="padding:2px 0">{r}</li>'
                    history_html += '</ul></div>'
                history_html += '</details>'
        except: pass
    return f'''<div class="section" style="border-left:4px solid #1565c0">
    <h2>🧠 What the AI Learned from Your Trade History</h2>
    {f'<p style="color:#555;font-size:13px;margin-bottom:12px;font-style:italic">{summary}</p>' if summary else ''}
    <ul style="list-style:none;padding:0">{rule_items}</ul>
    {history_html}
  </div>'''


def save_html_report(result, ctx, nd, ep, wl, derived_rules=None, learning_summary='',
                     stop_price='N/A', target_price='N/A'):
    """Save a formatted HTML report to Google Drive after each run."""
    pk   = result.get('top_pick', {}) if result else {}
    sig  = pk.get('signal', 'NO PICK')
    conf = pk.get('confidence', 0)
    ticker = pk.get('ticker', 'N/A')
    today  = datetime.now().strftime('%Y-%m-%d')
    stop = f'${stop_price}' if isinstance(stop_price, (int, float)) else stop_price
    tgt  = f'${target_price}' if isinstance(target_price, (int, float)) else target_price

    pick_color  = '#00c853' if sig == 'BUY' else '#ff1744'
    pick_label  = f'BUY {ticker}' if sig == 'BUY' else 'NO PICK TODAY'
    conf_bar    = int(conf / 100 * 200)

    # Load CSVs
    def load(fp):
        if not os.path.exists(fp): return pd.DataFrame()
        df = pd.read_csv(fp)
        for c in ['Return_10d_pct','Return_30d_pct','vs_QQQ_30d','Confidence']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    picks_df = load(PICKS_CSV)
    watch_df = load(WATCH_CSV)

    def stats(df):
        done = df[df['Result'].isin(['Win','Loss','Neutral'])] if 'Result' in df.columns else pd.DataFrame()
        if done.empty: return {'total':0,'wins':0,'losses':0,'neutral':0,'wr':0,'avg30':0}
        w=len(done[done['Result']=='Win']); l=len(done[done['Result']=='Loss']); n=len(done[done['Result']=='Neutral'])
        return {'total':len(done),'wins':w,'losses':l,'neutral':n,
                'wr':round(w/len(done)*100,1) if len(done) else 0,
                'avg30':round(done['Return_30d_pct'].mean(),2)}

    ps = stats(picks_df)
    ws = stats(watch_df)

    def result_badge(r):
        if r == 'Win':     return '<span style="color:#00c853;font-weight:bold">✅ Win</span>'
        if r == 'Loss':    return '<span style="color:#ff1744;font-weight:bold">❌ Loss</span>'
        if r == 'Neutral': return '<span style="color:#ff9800;font-weight:bold">➖ Neutral</span>'
        return '<span style="color:#999">⏳ Pending</span>'

    def pct(v):
        try:
            f = float(v)
            c = '#00c853' if f > 0 else '#ff1744' if f < 0 else '#999'
            return f'<span style="color:{c}">{f:+.1f}%</span>'
        except: return '<span style="color:#999">—</span>'

    def table_rows(df, max_rows=50):
        if df.empty: return '<tr><td colspan="9">No data yet</td></tr>'
        rows = ''
        for _, r in df.sort_values('Date', ascending=False).head(max_rows).iterrows():
            res = str(r.get('Result','Pending'))
            bg  = '#f1fff6' if res=='Win' else '#fff1f1' if res=='Loss' else ''
            rows += f'''<tr style="background:{bg}">
                <td>{r.get("Date","")}</td>
                <td><b>{r.get("Ticker","")}</b></td>
                <td>{r.get("Confidence","")}</td>
                <td>{r.get("Source","")}</td>
                <td>{r.get("Sector","")}</td>
                <td>{pct(r.get("Return_10d_pct",""))}</td>
                <td>{pct(r.get("Return_30d_pct",""))}</td>
                <td>{pct(r.get("vs_QQQ_30d",""))}</td>
                <td>{result_badge(res)}</td>
                <td style="font-size:11px;color:#555;max-width:200px">{str(r.get("Reasoning",""))[:80]}</td>
            </tr>'''
        return rows

    watch_list_html = ''
    for w in (wl or []):
        watch_list_html += f'''
        <div style="background:#fff8e1;border-left:4px solid #ffc107;padding:12px;margin:8px 0;border-radius:4px">
            <b>{w.get("ticker")}</b> &nbsp;{w.get("confidence")}/100 &nbsp;·&nbsp;
            {w.get("sector","")} &nbsp;·&nbsp;
            <span style="color:#555">{w.get("reasoning","")[:120]}</span>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stock Screener — {today}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
         background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; color: #444; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .pick-card {{ background: #1a1a2e; color: white; border-radius: 14px;
                padding: 28px; margin-bottom: 24px; border-left: 8px solid {pick_color}; }}
  .pick-title {{ font-size: 32px; font-weight: 800; color: {pick_color}; margin-bottom: 8px; }}
  .pick-meta {{ font-size: 13px; color: #aaa; margin-bottom: 20px; }}
  .pick-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 20px 0; }}
  .pick-metric {{ background: rgba(255,255,255,0.07); border-radius: 8px; padding: 14px; }}
  .pick-metric .label {{ font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }}
  .pick-metric .value {{ font-size: 22px; font-weight: 700; color: white; margin-top: 4px; }}
  .conf-bar {{ background: rgba(255,255,255,0.1); border-radius: 99px; height: 8px; margin-top: 8px; }}
  .conf-fill {{ background: {pick_color}; border-radius: 99px; height: 8px; width: {conf_bar}px; max-width:200px; }}
  .reasoning-box {{ background: rgba(255,255,255,0.05); border-radius: 8px; padding: 14px; margin-top: 16px; font-size: 14px; line-height: 1.6; }}
  .risk-box {{ background: rgba(255,68,68,0.1); border: 1px solid rgba(255,68,68,0.3);
              border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 13px; color: #ffaaaa; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .stat-card .val {{ font-size: 28px; font-weight: 700; }}
  .stat-card .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .section {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #333;
                 border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fa; padding: 10px 8px; text-align: left; font-weight: 600;
        color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .market-strip {{ background: white; border-radius: 10px; padding: 14px 20px;
                   margin-bottom: 20px; display: flex; gap: 24px; align-items: center;
                   box-shadow: 0 1px 4px rgba(0,0,0,0.08); font-size: 13px; }}
  .market-item .ml {{ color: #888; font-size: 11px; }}
  .market-item .mv {{ font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <h1>📈 Daily Stock Screener</h1>
  <p class="subtitle">{datetime.now().strftime("%A, %B %d %Y  %H:%M")} &nbsp;·&nbsp; Powered by NVIDIA {NVIDIA_MODEL}</p>

  <div class="market-strip">
    <div class="market-item"><div class="ml">VIX</div><div class="mv">{ctx.get("vix_level","-"):.1f} ({ctx.get("vix_regime","")[:12]})</div></div>
    <div class="market-item"><div class="ml">QQQ</div><div class="mv">{ctx.get("qqq_trend","")} (+{ctx.get("qqq_vs_ma50",0):.1f}% vs 50MA)</div></div>
    <div class="market-item"><div class="ml">SPY Today</div><div class="mv">{ctx.get("spy_return_today",0):+.2f}%</div></div>
    <div class="market-item"><div class="ml">Sentiment</div><div class="mv">{nd.get("market_sentiment","NEUTRAL") if nd else "N/A"}</div></div>
    <div style="flex:1;color:#666;font-size:12px">{(nd.get("macro_summary","") if nd else "")[:120]}</div>
  </div>

  <!-- LEARNING RULES -->
  {_build_rules_html(derived_rules, learning_summary)}

  <!-- TODAY'S PICK -->
  <div class="pick-card">
    <div class="pick-title">{pick_label}</div>
    <div class="pick-meta">{today} &nbsp;·&nbsp; Source: {pk.get("source","")} &nbsp;·&nbsp; Sector: {pk.get("sector","")}</div>
    <div class="pick-grid">
      <div class="pick-metric">
        <div class="label">Confidence</div>
        <div class="value">{conf}/100</div>
        <div class="conf-bar"><div class="conf-fill"></div></div>
      </div>
      <div class="pick-metric">
        <div class="label">Entry Price</div>
        <div class="value">${ep if isinstance(ep,(int,float)) else "N/A"}</div>
      </div>
      <div class="pick-metric">
        <div class="label">Stop → Target</div>
        <div class="value" style="font-size:15px">{stop} → {tgt}</div>
      </div>
    </div>
    <div class="reasoning-box">{pk.get("reasoning","")}</div>
    <div class="risk-box">⚠️ Key Risk: {pk.get("key_risk","")}</div>
    {"<div class='risk-box' style='margin-top:8px;color:#aaa'>🐻 Bear case: "+pk.get("devils_advocate","")+"</div>" if pk.get("devils_advocate") else ""}
  </div>

  {"<div class='section'><h2>👀 Watch List</h2>" + watch_list_html + "</div>" if wl else ""}

  <!-- SUMMARY STATS -->
  <div class="grid4">
    <div class="stat-card"><div class="val" style="color:#00c853">{ps["wins"]}</div><div class="lbl">BUY Wins</div></div>
    <div class="stat-card"><div class="val" style="color:#ff1744">{ps["losses"]}</div><div class="lbl">BUY Losses</div></div>
    <div class="stat-card"><div class="val">{ps["wr"]}%</div><div class="lbl">BUY Win Rate</div></div>
    <div class="stat-card"><div class="val" style="color:{"#00c853" if ps["avg30"]>=0 else "#ff1744"}">{ps["avg30"]:+.1f}%</div><div class="lbl">Avg 30d Return</div></div>
  </div>

  <!-- CONFIRMED PICKS TABLE -->
  <div class="section">
    <h2>✅ Confirmed BUY History</h2>
    <table>
      <tr><th>Date</th><th>Ticker</th><th>Conf</th><th>Source</th><th>Sector</th>
          <th>10d</th><th>30d</th><th>vs QQQ</th><th>Result</th><th>Reasoning</th></tr>
      {table_rows(picks_df)}
    </table>
  </div>

  <!-- WATCH LIST TABLE -->
  <div class="section">
    <h2>👀 Watch List History</h2>
    <table>
      <tr><th>Date</th><th>Ticker</th><th>Conf</th><th>Source</th><th>Sector</th>
          <th>10d</th><th>30d</th><th>vs QQQ</th><th>Result</th><th>Reasoning</th></tr>
      {table_rows(watch_df)}
    </table>
  </div>

  <p style="text-align:center;color:#aaa;font-size:11px;margin-top:20px">
    Not financial advice. Personal learning project. Always do your own research.
  </p>
</div>
</body>
</html>'''

    # Save dated copy + rolling latest to Drive
    report_path = os.path.join(DRIVE_FOLDER, f'report_{today}.html')
    latest_path = os.path.join(DRIVE_FOLDER, 'report_latest.html')
    for fp in (report_path, latest_path):
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(html)

    # Render directly in Colab via iframe (base64 data URI bypasses Colab's style sandbox)
    try:
        import base64
        from IPython.display import display as _ipy_display, HTML as _IPyHTML
        b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
        iframe = (
            f'<iframe src="data:text/html;base64,{b64}" '
            f'width="100%" height="900" style="border:none;border-radius:8px;'
            f'box-shadow:0 2px 12px rgba(0,0,0,0.1)"></iframe>'
        )
        _ipy_display(_IPyHTML(iframe))
    except Exception:
        print(f'📊 Report saved → {latest_path}')


def display_scorecard():
    """Minimal terminal summary — full detail is in the HTML report rendered above."""
    print('\n' + '─' * 60)
    for label, fp in [('CONFIRMED PICKS', PICKS_CSV), ('WATCH LIST', WATCH_CSV)]:
        if not os.path.exists(fp): continue
        df   = pd.read_csv(fp)
        done = df[df['Result'].isin(['Win','Loss','Neutral'])]
        pend = df[df['Result'] == 'Pending']
        if done.empty:
            print(f'  {label}: 0 evaluated yet | {len(pend)} pending')
            continue
        wins = len(done[done['Result']=='Win'])
        wr   = round(wins / len(done) * 100, 1)
        r30  = pd.to_numeric(done['Return_30d_pct'], errors='coerce').mean()
        print(f'  {label}: {wins}/{len(done)} wins ({wr}%) | avg 30d {r30:+.1f}% | {len(pend)} pending')
    print(f'  📁 Full report → {DRIVE_FOLDER}/report_latest.html')
    print('─' * 60)


print('\n✅ All functions loaded - v6.2')
print('▶  Run Cell 5 to start the screener')


def _recent_picks_summary(days=10):
    """Read PICKS_CSV and return last N days of BUY picks with live P&L for pending ones."""
    lines = []
    if not os.path.exists(PICKS_CSV):
        return lines
    try:
        df = pd.read_csv(PICKS_CSV)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        cutoff = datetime.now() - pd.Timedelta(days=days)
        recent = df[df['Date'] >= cutoff].sort_values('Date', ascending=False)
        # Fetch live prices in one batch for pending tickers
        pending_tickers = recent[recent['Result'] == 'Pending']['Ticker'].dropna().unique().tolist()
        live_prices = {}
        if pending_tickers:
            try:
                raw = yf.download(pending_tickers, period='2d', auto_adjust=True,
                                  progress=False, threads=True)
                closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
                for t in pending_tickers:
                    try:
                        col = closes[t] if t in closes.columns else closes.iloc[:, 0]
                        live_prices[t] = float(col.dropna().iloc[-1])
                    except:
                        pass
            except:
                pass
        for _, row in recent.iterrows():
            t      = str(row.get('Ticker', '')).strip()
            d      = row['Date'].strftime('%b %d')
            entry  = row.get('Entry_Price', '')
            result = str(row.get('Result', 'Pending')).strip()
            if result == 'Pending':
                curr = live_prices.get(t)
                if curr and entry and str(entry) not in ('', 'nan', 'N/A'):
                    pct = (curr - float(entry)) / float(entry) * 100
                    status = f'Open {pct:+.1f}%'
                else:
                    status = 'Open'
            else:
                ret = row.get('Return_30d_pct', '')
                try:
                    status = f'{result} ({float(ret):+.1f}%)'
                except:
                    status = result
            lines.append(f'  {d}: {t} @ {entry} - {status}')
    except Exception:
        pass
    return lines


def _wa_send(text, label=''):
    """Send one WhatsApp message via CallMeBot. Waits 3s between calls to avoid rate limits."""
    if not WHATSAPP_PHONE or not CALLMEBOT_API_KEY:
        return False
    import urllib.parse
    try:
        url = (f'https://api.callmebot.com/whatsapp.php'
               f'?phone={WHATSAPP_PHONE}&text={urllib.parse.quote(text)}&apikey={CALLMEBOT_API_KEY}')
        r = requests.get(url, timeout=15)
        ok = r.status_code == 200
        print(f'  WhatsApp {label}: {"sent" if ok else f"failed HTTP {r.status_code}"}')
        time.sleep(3)
        return ok
    except Exception as e:
        print(f'  WhatsApp {label} error: {e}')
        return False


def send_weekly_summary():
    """Send portfolio weekly review — called on US Saturday (= Sunday NZT)."""
    if not WHATSAPP_PHONE or not CALLMEBOT_API_KEY:
        return
    pf = load_portfolio()
    pf = update_portfolio_prices(pf)

    total_val = round(pf['cash'] + sum(p.get('current_value', p['cost_basis']) for p in pf['positions']), 2)
    total_pnl = round(total_val - pf['starting_capital'], 2)
    total_pct = round(total_pnl / pf['starting_capital'] * 100, 2)

    # QQQ comparison — only from portfolio creation date, not a fixed 7-day window.
    # Avoids false alpha when portfolio started mid-week.
    created_str = pf.get('created', '')
    qqq_week = None
    qqq_label = 'QQQ: unavailable'
    alpha_str = ''
    try:
        qqq_h = yf.Ticker('QQQ').history(period='14d')
        if not qqq_h.empty:
            if created_str:
                # Find first QQQ close on or after portfolio creation date
                qqq_h.index = pd.to_datetime(qqq_h.index).tz_localize(None) if qqq_h.index.tz else pd.to_datetime(qqq_h.index)
                created_dt  = pd.to_datetime(created_str)
                qqq_since   = qqq_h[qqq_h.index >= created_dt]
                if len(qqq_since) >= 2:
                    qqq_week = round((float(qqq_since['Close'].iloc[-1]) - float(qqq_since['Close'].iloc[0])) /
                                      float(qqq_since['Close'].iloc[0]) * 100, 2)
                    days_tracked = (qqq_since.index[-1] - qqq_since.index[0]).days
                    qqq_label = f'QQQ since start ({days_tracked}d): {qqq_week:+.2f}%'
                    alpha = round(total_pct - qqq_week, 2)
                    alpha_str = f'You vs QQQ: {alpha:+.2f}% alpha'
                else:
                    qqq_label = 'QQQ: portfolio too new for comparison'
            else:
                # No creation date — use last 7 days but flag it
                qqq_week = round((float(qqq_h['Close'].iloc[-1]) - float(qqq_h['Close'].iloc[-5])) /
                                  float(qqq_h['Close'].iloc[-5]) * 100, 2)
                qqq_label = f'QQQ this week: {qqq_week:+.2f}%'
    except:
        pass

    week_ago    = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    week_closed = [t for t in pf.get('closed_trades', []) if t.get('exit_date', '') >= week_ago]
    week_wins   = sum(1 for t in week_closed if t.get('realized_pnl', 0) > 0)
    week_pnl    = sum(t.get('realized_pnl', 0) for t in week_closed)

    open_lines = '\n'.join(
        f'  {p["ticker"]} {p["shares"]}sh | {p.get("unrealized_pnl_pct",0):+.1f}% | '
        f'{p.get("hold_days",0)}d | Stop:{p.get("stop_price","?")} Tgt:{p.get("target_price","?")}'
        for p in pf['positions']
    ) or '  None'

    closed_lines = '\n'.join(
        f'  {t["ticker"]} {t.get("realized_pnl",0):+.0f} ({t.get("realized_pnl_pct",0):+.1f}%) [{t.get("reason","")}]'
        for t in week_closed
    ) or '  None closed this week'

    # Buy history — prefer CSV, fall back to open positions in portfolio.json
    history_lines = _recent_picks_summary(days=14)
    if history_lines:
        hist_str = '\n'.join(history_lines)
    elif pf['positions']:
        hist_str = '\n'.join(
            f'  {p["ticker"]} @ {p["entry_price"]} | entered {p["entry_date"]} | '
            f'{p.get("unrealized_pnl_pct",0):+.1f}% | {p.get("hold_days",0)}d held'
            for p in pf['positions']
        )
    else:
        hist_str = '  No picks yet'

    date_str = datetime.now().strftime('%b %d %Y')
    msg = (
        f'WEEKLY SUMMARY - {date_str}\n'
        f'{"="*30}\n'
        f'PORTFOLIO: {total_pnl:+,.0f} ({total_pct:+.1f}%) | Cash: {pf["cash"]:,.0f}\n'
        f'{qqq_label}\n'
        f'{alpha_str}\n'
        f'{"="*30}\n'
        f'OPEN POSITIONS:\n{open_lines}\n'
        f'{"="*30}\n'
        f'THIS WEEK CLOSED ({len(week_closed)} trades | {week_wins}W/{len(week_closed)-week_wins}L | PnL: {week_pnl:+.0f}):\n'
        f'{closed_lines}\n'
        f'{"="*30}\n'
        f'POSITIONS / HISTORY:\n{hist_str}\n'
        f'{"="*30}\n'
        f'Next run: Tuesday 9AM NZT'
    )
    _wa_send(msg, 'weekly-summary')


def send_whatsapp(pick, ctx, ep, wl, stop_price, target_price, candidates=None, portfolio=None):
    """Send daily pick as 3 WhatsApp messages via CallMeBot (uses ~3 of 10 daily limit)."""
    if not WHATSAPP_PHONE or not CALLMEBOT_API_KEY:
        return
    import urllib.parse

    ticker   = pick.get('ticker', 'NONE')
    sig      = pick.get('signal', 'NO PICK')
    conf     = pick.get('confidence', 0)
    sector   = pick.get('sector', '')
    why      = str(pick.get('reasoning', ''))[:120]
    risk     = str(pick.get('key_risk', ''))[:100]
    date_str = datetime.now().strftime('%b %d %Y')

    # ── Market context line ───────────────────────────────────
    gm     = ctx.get('global_macro', {})
    esf    = gm.get('es_futures', {})
    nqf    = gm.get('nq_futures', {})
    tnx    = gm.get('yield_10y', {})
    dxy    = gm.get('dxy', {})
    nzdusd = gm.get('nzdusd', {})
    s1d    = ctx.get('sector_1d', {})
    top_s  = sorted(s1d.items(), key=lambda x: x[1], reverse=True)
    top_s_str  = ' | '.join(f'{s} {v:+.1f}%' for s, v in top_s[:3])  if top_s else 'N/A'
    bot_s_str  = ' | '.join(f'{s} {v:+.1f}%' for s, v in top_s[-2:]) if top_s else 'N/A'

    fut_str = ''
    if esf or nqf:
        fut_str = f'Futures: ES {esf.get("chg_pct",0):+.2f}% | NQ {nqf.get("chg_pct",0):+.2f}%\n'
    macro_str = ''
    if tnx.get('price') or dxy.get('price'):
        macro_str = f'10Y: {tnx.get("price","?")}% ({tnx.get("chg_pct",0):+.2f}%) | DXY: {dxy.get("price","?")} ({dxy.get("chg_pct",0):+.2f}%)\n'
    nzd_rate = nzdusd.get('price')
    sharesies_str = ''
    if nzd_rate:
        coverage_usd = round(SHARESIES_COVERAGE_NZD * float(nzd_rate), 0)
        sharesies_str = f'NZD/USD: {nzd_rate} | Sharesies free coverage: ~{coverage_usd:,.0f} USD/month\n'

    # ── Portfolio block ───────────────────────────────────────
    pf_block = ''
    if portfolio:
        total_val = round(portfolio['cash'] + sum(p.get('current_value', p['cost_basis']) for p in portfolio['positions']), 2)
        total_pnl = round(total_val - portfolio['starting_capital'], 2)
        total_pct = round(total_pnl / portfolio['starting_capital'] * 100, 2)
        pos_lines = '\n'.join(
            f'  {p["ticker"]} {p["shares"]}sh @ {p["entry_price"]} '
            f'({p.get("unrealized_pnl_pct",0):+.1f}%) | '
            f'Stop: {p.get("stop_price","?")} | Target: {p.get("target_price","?")} | '
            f'{p.get("hold_days",0)}d held'
            for p in portfolio['positions']
        ) if portfolio['positions'] else '  No open positions'
        pf_block = (
            f'PORTFOLIO: {total_pnl:+,.0f} ({total_pct:+.1f}%) | Cash: {portfolio["cash"]:,.0f}\n'
            f'{pos_lines}\n'
            f'---\n'
        )

    # ── Congress + candidate detail ───────────────────────────
    pick_match = next((c for c in (candidates or []) if c['ticker'] == ticker), {})
    cg_pick    = pick_match.get('congress_label') == 'BUYING'
    cg_who     = pick_match.get('congress_notes', '')
    cg_others  = [
        f'{c["ticker"]}: {c.get("congress_notes","")}'
        for c in (candidates or [])
        if c.get('congress_label') == 'BUYING' and c['ticker'] != ticker
    ][:3]
    congress_line = f'Congress also buying:\n' + '\n'.join(f'  {o}' for o in cg_others) + '\n' if cg_others else ''
    cg_tag = f'\nCONGRESS: {cg_who}' if cg_pick else ''

    # ── Technicals from candidate ─────────────────────────────
    rsi     = pick_match.get('rsi')
    adx     = pick_match.get('adx')
    mom5    = pick_match.get('momentum_5d')
    vr      = pick_match.get('vol_ratio')
    analyst = pick_match.get('analyst_rating', '')
    upside  = pick_match.get('upside_pct')
    earn_d  = pick_match.get('earnings_days_away', 'N/A')
    short_f = pick_match.get('short_pct_float')
    sec_8k  = pick_match.get('sec_8k_filings', [])
    insider = pick_match.get('insider_label', '')
    options = pick_match.get('options_label', '')
    tech_bd = pick_match.get('tech_score_breakdown', {})

    tech_line = ''
    if rsi is not None:
        tech_line = f'RSI {rsi:.1f} | ADX {adx:.1f} | Mom5d {mom5:+.1f}% | Vol {vr:.1f}x' if all(v is not None for v in [adx, mom5, vr]) else f'RSI {rsi:.1f}'
    analyst_line = ''
    if analyst or upside is not None:
        up_str = f' | Upside {upside:+.1f}%' if upside is not None else ''
        analyst_line = f'Analyst: {analyst}{up_str}\n'
    earnings_line = f'Earnings: {earn_d}d away\n' if earn_d not in ('N/A', None) else ''
    short_line    = f'Short float: {short_f:.1f}%\n' if short_f is not None else ''
    flow_line     = ''
    if insider != 'NEUTRAL' or options != 'NEUTRAL':
        flow_line = f'Insider: {insider} | Options P/C: {options}\n'
    sec_line = ''
    if sec_8k:
        sec_line = f'SEC 8-K: {sec_8k[0][:80]}\n' if isinstance(sec_8k[0], str) else ''

    # ── R:R ──────────────────────────────────────────────────
    rr_str = 'N/A'
    if isinstance(stop_price, (int,float)) and isinstance(target_price, (int,float)) and isinstance(ep, (int,float)) and ep - stop_price > 0:
        rr = round((target_price - ep) / (ep - stop_price), 1)
        rr_str = f'1:{rr}'

    # ── Actual position from portfolio ────────────────────────
    pos = next((p for p in (portfolio or {}).get('positions', []) if p['ticker'] == ticker), None)
    shares_str = f'{pos["shares"]} shares | Spent: {pos["cost_basis"]:,.0f}' if pos else ''

    # ── Watch list ────────────────────────────────────────────
    today_str_short = datetime.now().strftime('%b %d')
    # History = prior picks only (exclude today's pick)
    history_lines = [l for l in _recent_picks_summary(days=14)
                     if ticker not in l or today_str_short not in l]
    history_block = ('HISTORY:\n' + '\n'.join(history_lines)) if history_lines else ''

    # Open positions excluding today's new pick
    other_positions = [p for p in (portfolio or {}).get('positions', []) if p['ticker'] != ticker]
    if other_positions:
        open_block = 'OPEN:\n' + '\n'.join(
            f'  {p["ticker"]} {p["shares"]}sh {p.get("unrealized_pnl_pct",0):+.1f}% | '
            f'Stop {p.get("stop_price","?")} | {p.get("hold_days",0)}d'
            for p in other_positions
        )
    else:
        open_block = ''

    watch_lines = ''
    for w in wl[:2]:
        wc = next((c for c in (candidates or []) if c['ticker'] == w.get('ticker','')), {})
        wr_rsi = wc.get('rsi')
        wr_mom = wc.get('momentum_5d')
        wr_tag = f' RSI {wr_rsi:.0f} Mom {wr_mom:+.1f}%' if wr_rsi is not None and wr_mom is not None else ''
        watch_lines += f'  {w.get("ticker","")} {w.get("confidence",0)}/100{wr_tag}\n'

    # ── 2 messages ────────────────────────────────────────────
    stop_pct     = round((stop_price - ep) / ep * 100, 1) if isinstance(stop_price,(int,float)) and isinstance(ep,(int,float)) and ep>0 else ''
    tgt_pct      = round((target_price - ep) / ep * 100, 1) if isinstance(target_price,(int,float)) and isinstance(ep,(int,float)) and ep>0 else ''
    stop_pct_str = f' ({stop_pct:+.1f}%)' if stop_pct != '' else ''
    tgt_pct_str  = f' ({tgt_pct:+.1f}%)'  if tgt_pct  != '' else ''

    # All open positions (including today's new pick) formatted the same way
    all_positions = (portfolio or {}).get('positions', [])
    def _pos_line(p):
        upc = p.get('unrealized_pnl_pct', 0)
        return (f'{p["ticker"]}  {p["shares"]}sh @ {p["entry_price"]} | '
                f'{upc:+.1f}% | Stop {p.get("stop_price","?")} | '
                f'Target {p.get("target_price","?")} | Day {p.get("hold_days",0)}')
    positions_block = '\n'.join(_pos_line(p) for p in all_positions) if all_positions else '  None'

    pf_header = (f'PORTFOLIO | Cash {portfolio["cash"]:,.0f} | P&L {total_pnl:+,.0f} ({total_pct:+.1f}%)'
                 if portfolio else 'PORTFOLIO')

    if sig == 'BUY':
        msg = '\n'.join(filter(None, [
            f'SCREENER {date_str} | SPY {ctx["spy_return_today"]:+.2f}% | VIX {ctx["vix_level"]:.1f}',
            '',
            f'TODAY: {ticker} [{sector}] {conf}/100',
            f'Buy    {ep}',
            f'Stop   {stop_price}{stop_pct_str}',
            f'Target {target_price}{tgt_pct_str}',
            f'R:R {rr_str} | {shares_str}',
            '',
            '---',
            pf_header,
            '',
            positions_block,
        ]))
    else:
        msg = '\n'.join(filter(None, [
            f'SCREENER {date_str} | SPY {ctx["spy_return_today"]:+.2f}% | VIX {ctx["vix_level"]:.1f}',
            f'NO BUY today',
            '',
            '---',
            pf_header,
            '',
            positions_block,
        ]))
    _wa_send(msg, 'daily update')


# ============================================================
# LLM SELF-ADAPTATION  —  config overrides + criteria engine
# ============================================================

_CFG_PATH = f'{DRIVE_FOLDER}/config_overrides.json'


def load_config_overrides():
    """Read config_overrides.json (written by LLM) and apply to global screening vars."""
    global RSI_MIN, RSI_MAX, ADX_MIN, BUY_THRESHOLD, WATCH_THRESHOLD
    global ATR_STOP_MULT, ATR_TARGET_MULT, VOLUME_MIN_RATIO, MIN_PRICE
    global MIN_DOLLAR_VOLUME_M, SECTOR_CONC_MAX, SAMPLE_SIZE
    global _CFG_SECTOR_BLACKLIST, _CFG_SECTOR_WHITELIST, _CFG_SOURCE_PREFERENCE
    global _CFG_REQUIRE_CONGRESS, _CFG_MIN_CATALYST_SCORE, _CFG_MIN_ADX_BUY, _CFG_AVOID_EARNINGS
    global _CFG_MAX_VIX, _CFG_MIN_PRICE, _CFG_ONLY_PROFITABLE
    global _CFG_REQUIRE_ABOVE_MA, _CFG_MIN_DOLLAR_VOL_M, _CFG_HOLD_DAYS
    global _CFG_SECTOR_CONC_MAX, _CFG_SAMPLE_SIZE, _CFG_ADDITIONAL_TICKERS, _CFG_BROKERAGE_FEE
    global _CFG_MIN_CASH_FLOOR, _CFG_DD_CAUTION_PCT, _CFG_DD_SEVERE_PCT, _CFG_DD_CRITICAL_PCT
    global _CFG_WIN_THRESHOLD_PCT, _CFG_LOSS_THRESHOLD_PCT, _CFG_MIN_PICKS_TO_LEARN
    global _CFG_RSI_HARD_CAP, _CFG_RSI_CAP_CONF, _CFG_UPSIDE_HARD_CAP, _CFG_UPSIDE_CAP_CONF
    global _CFG_MAX_POSITIONS
    global _CFG_VIX_LOW_PCTILE, _CFG_VIX_HIGH_PCTILE
    global _CFG_SECTOR_CONC_LOOKBACK, _CFG_SECTOR_CONC_PENALTY
    global _CFG_CONGRESS_DAYS, _CFG_SEC_8K_DAYS
    global _CFG_RSI_EXIT, _CFG_RSI_EXIT_MIN_PROFIT, _CFG_MACD_EXIT_MIN_PROFIT, _CFG_ENTRY_SLIPPAGE_PCT
    global _CFG_FINAL_CANDIDATES, _CFG_PRE_EARNINGS_DAYS, _CFG_SQUEEZE_FLOAT_PCT, _CFG_SQUEEZE_DAYS_COVER
    global _CFG_TRAIL_ATR_MULT, _CFG_VOLUME_MIN_RATIO
    global BROKERAGE_FEE

    paths = [_CFG_PATH, 'config_overrides.json']
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                ov = json.load(f)

            # Numeric thresholds — LLM has full authority, no clamping
            RSI_MIN           = float(ov.get('RSI_MIN',          RSI_MIN))
            RSI_MAX           = float(ov.get('RSI_MAX',          RSI_MAX))
            ADX_MIN           = float(ov.get('ADX_MIN',          ADX_MIN))
            BUY_THRESHOLD     = float(ov.get('BUY_THRESHOLD',    BUY_THRESHOLD))
            WATCH_THRESHOLD   = float(ov.get('WATCH_THRESHOLD',  WATCH_THRESHOLD))
            ATR_STOP_MULT     = float(ov.get('ATR_STOP_MULT',    ATR_STOP_MULT))
            ATR_TARGET_MULT   = float(ov.get('ATR_TARGET_MULT',  ATR_TARGET_MULT))
            VOLUME_MIN_RATIO  = float(ov.get('VOLUME_MIN_RATIO', VOLUME_MIN_RATIO))

            # Criteria — LLM full authority
            _CFG_SECTOR_BLACKLIST   = [s.strip() for s in ov.get('sector_blacklist',   [])]
            _CFG_SECTOR_WHITELIST   = [s.strip() for s in ov.get('sector_whitelist',   [])]
            _CFG_SOURCE_PREFERENCE  = str(ov.get('source_preference', 'ANY')).upper()
            _CFG_REQUIRE_CONGRESS   = bool(ov.get('require_congress',   False))
            _CFG_MIN_CATALYST_SCORE = float(ov.get('min_catalyst_score', 0))
            _CFG_MIN_ADX_BUY        = float(ov.get('min_adx_buy', ADX_MIN))
            _CFG_AVOID_EARNINGS     = bool(ov.get('avoid_earnings_week', False))
            _CFG_MAX_VIX            = float(ov.get('max_vix', 999))
            _CFG_MIN_PRICE          = float(ov.get('min_price', MIN_PRICE))
            _CFG_ONLY_PROFITABLE    = bool(ov.get('only_profitable',    False))
            _CFG_REQUIRE_ABOVE_MA   = bool(ov.get('require_above_ma',   True))
            _CFG_MIN_DOLLAR_VOL_M   = float(ov.get('min_dollar_volume_m', MIN_DOLLAR_VOLUME_M))
            _CFG_HOLD_DAYS          = int(ov.get('hold_days',            _CFG_HOLD_DAYS))
            _CFG_SECTOR_CONC_MAX    = int(ov.get('sector_conc_max',      SECTOR_CONC_MAX))
            _CFG_SAMPLE_SIZE        = int(ov.get('sample_size',          SAMPLE_SIZE))
            _CFG_ADDITIONAL_TICKERS = [t.strip().upper() for t in ov.get('additional_tickers', [])]
            _CFG_BROKERAGE_FEE      = float(ov.get('brokerage_fee',       BROKERAGE_FEE))
            _CFG_MIN_CASH_FLOOR     = float(ov.get('min_cash_floor',        _CFG_MIN_CASH_FLOOR))
            _CFG_DD_CAUTION_PCT     = float(ov.get('dd_caution_pct',        _CFG_DD_CAUTION_PCT))
            _CFG_DD_SEVERE_PCT      = float(ov.get('dd_severe_pct',         _CFG_DD_SEVERE_PCT))
            _CFG_DD_CRITICAL_PCT    = float(ov.get('dd_critical_pct',       _CFG_DD_CRITICAL_PCT))
            _CFG_WIN_THRESHOLD_PCT  = float(ov.get('win_threshold_pct',     _CFG_WIN_THRESHOLD_PCT))
            _CFG_LOSS_THRESHOLD_PCT = float(ov.get('loss_threshold_pct',    _CFG_LOSS_THRESHOLD_PCT))
            _CFG_MIN_PICKS_TO_LEARN = int(ov.get('min_picks_to_learn',      _CFG_MIN_PICKS_TO_LEARN))
            _CFG_RSI_HARD_CAP       = float(ov.get('rsi_hard_cap',          _CFG_RSI_HARD_CAP))
            _CFG_RSI_CAP_CONF       = float(ov.get('rsi_cap_conf',          _CFG_RSI_CAP_CONF))
            _CFG_UPSIDE_HARD_CAP    = float(ov.get('upside_hard_cap',       _CFG_UPSIDE_HARD_CAP))
            _CFG_UPSIDE_CAP_CONF    = float(ov.get('upside_cap_conf',       _CFG_UPSIDE_CAP_CONF))
            _CFG_MAX_POSITIONS       = int(ov.get('max_positions',              _CFG_MAX_POSITIONS))
            _CFG_VIX_LOW_PCTILE      = float(ov.get('vix_low_pctile',           _CFG_VIX_LOW_PCTILE))
            _CFG_VIX_HIGH_PCTILE     = float(ov.get('vix_high_pctile',          _CFG_VIX_HIGH_PCTILE))
            _CFG_SECTOR_CONC_LOOKBACK= int(ov.get('sector_conc_lookback',       _CFG_SECTOR_CONC_LOOKBACK))
            _CFG_SECTOR_CONC_PENALTY = float(ov.get('sector_conc_penalty',      _CFG_SECTOR_CONC_PENALTY))
            _CFG_CONGRESS_DAYS       = int(ov.get('congress_days',              _CFG_CONGRESS_DAYS))
            _CFG_SEC_8K_DAYS         = int(ov.get('sec_8k_days',               _CFG_SEC_8K_DAYS))
            _CFG_RSI_EXIT            = float(ov.get('rsi_exit',                _CFG_RSI_EXIT))
            _CFG_RSI_EXIT_MIN_PROFIT = float(ov.get('rsi_exit_min_profit',     _CFG_RSI_EXIT_MIN_PROFIT))
            _CFG_MACD_EXIT_MIN_PROFIT= float(ov.get('macd_exit_min_profit',    _CFG_MACD_EXIT_MIN_PROFIT))
            _CFG_ENTRY_SLIPPAGE_PCT  = float(ov.get('entry_slippage_pct',      _CFG_ENTRY_SLIPPAGE_PCT))
            _CFG_FINAL_CANDIDATES    = int(ov.get('final_candidates',          _CFG_FINAL_CANDIDATES))
            _CFG_PRE_EARNINGS_DAYS   = int(ov.get('pre_earnings_exit_days',    _CFG_PRE_EARNINGS_DAYS))
            _CFG_SQUEEZE_FLOAT_PCT   = float(ov.get('squeeze_float_pct',       _CFG_SQUEEZE_FLOAT_PCT))
            _CFG_SQUEEZE_DAYS_COVER  = float(ov.get('squeeze_days_to_cover',   _CFG_SQUEEZE_DAYS_COVER))
            _CFG_TRAIL_ATR_MULT      = float(ov.get('trail_atr_mult',          _CFG_TRAIL_ATR_MULT))
            _CFG_VOLUME_MIN_RATIO    = float(ov.get('volume_min_ratio',        _CFG_VOLUME_MIN_RATIO))
            VOLUME_MIN_RATIO         = _CFG_VOLUME_MIN_RATIO

            # Sync module-level globals so existing code picks up the new values
            MIN_PRICE           = _CFG_MIN_PRICE
            MIN_DOLLAR_VOLUME_M = _CFG_MIN_DOLLAR_VOL_M
            SECTOR_CONC_MAX     = _CFG_SECTOR_CONC_MAX
            SAMPLE_SIZE         = _CFG_SAMPLE_SIZE
            BROKERAGE_FEE       = _CFG_BROKERAGE_FEE

            print(f'✅ Config overrides loaded  RSI {RSI_MIN}-{RSI_MAX}  ADX≥{ADX_MIN}  BUY≥{BUY_THRESHOLD}')
            if _CFG_SECTOR_BLACKLIST:
                print(f'   Sector blacklist: {", ".join(_CFG_SECTOR_BLACKLIST)}')
            if _CFG_SECTOR_WHITELIST:
                print(f'   Sector whitelist: {", ".join(_CFG_SECTOR_WHITELIST)}')
            if _CFG_SOURCE_PREFERENCE != 'ANY':
                print(f'   Source preference: {_CFG_SOURCE_PREFERENCE} only')
            if _CFG_REQUIRE_CONGRESS:
                print(f'   Congress filter: ON — only tickers with congressional buys')
            if _CFG_AVOID_EARNINGS:
                print(f'   Earnings filter: ON — auto-dropping earnings-risk tickers')
            if ov.get('reasoning'):
                print(f'   LLM note: {str(ov["reasoning"])[:120]}')
            return
        except Exception as e:
            print(f'⚠️  config_overrides.json load error: {e}')
    print('ℹ️  No config_overrides.json — using defaults')


def save_config_overrides(cfg: dict):
    """Persist LLM-proposed config to Drive so next run picks it up."""
    try:
        with open(_CFG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'  Config saved → {_CFG_PATH}')
    except Exception as e:
        print(f'  Could not save config: {e}')


def apply_config_criteria(candidates, ctx=None):
    """
    Post-enrichment filter enforcing LLM-written criteria.
    LLM has full authority — any of these flags can be set.
    Runs after enrich_with_scores so sector/source/congress fields are populated.
    """
    kept, dropped = [], []

    # VIX gate — if market too wild, skip everything
    vix = (ctx or {}).get('vix_level', 0)
    if _CFG_MAX_VIX < 999 and vix and float(vix) > _CFG_MAX_VIX:
        print(f'  VIX {vix:.1f} > {_CFG_MAX_VIX} cap — LLM says go to cash, no BUY signals')
        return []

    for c in candidates:
        sector = c.get('sector', '')
        source = c.get('source', 'TECHNICAL').upper()
        ticker = c.get('ticker', '')

        # Sector blacklist
        if _CFG_SECTOR_BLACKLIST and sector in _CFG_SECTOR_BLACKLIST:
            dropped.append(f'{ticker}(blacklisted sector: {sector})')
            continue

        # Source preference (LLM can restrict to only TECHNICAL or only NEWS etc.)
        if _CFG_SOURCE_PREFERENCE not in ('ANY', '') and source != _CFG_SOURCE_PREFERENCE:
            dropped.append(f'{ticker}(source={source}, want {_CFG_SOURCE_PREFERENCE})')
            continue

        # Congress filter — only stocks bought by congress members
        if _CFG_REQUIRE_CONGRESS and c.get('congress_label', 'NEUTRAL') != 'BUYING':
            dropped.append(f'{ticker}(no congress buy)')
            continue

        # Earnings avoidance
        if _CFG_AVOID_EARNINGS and c.get('earnings_risk'):
            dropped.append(f'{ticker}(earnings risk)')
            continue

        # Minimum catalyst score
        cs = c.get('catalyst_score', 0)
        if cs and float(cs) < _CFG_MIN_CATALYST_SCORE:
            dropped.append(f'{ticker}(catalyst={cs:.0f}<{_CFG_MIN_CATALYST_SCORE:.0f})')
            continue

        # Profitable companies only (EPS > 0 if data available)
        if _CFG_ONLY_PROFITABLE:
            eps = c.get('eps_ttm', None)
            if eps is not None and float(eps) <= 0:
                dropped.append(f'{ticker}(unprofitable EPS={eps})')
                continue

        # Price floor override (screen_technicals already used updated MIN_PRICE,
        # but Stream B news rescues can bypass it — check here as a backstop)
        if c.get('price', 0) < _CFG_MIN_PRICE:
            dropped.append(f'{ticker}(price<{_CFG_MIN_PRICE})')
            continue

        # Stricter ADX for BUY pool
        if _CFG_MIN_ADX_BUY > ADX_MIN and c.get('adx', 0) < _CFG_MIN_ADX_BUY:
            dropped.append(f'{ticker}(ADX={c.get("adx",0):.0f}<{_CFG_MIN_ADX_BUY:.0f})')
            continue

        # Sector whitelist — boost score for preferred sectors
        if _CFG_SECTOR_WHITELIST and sector in _CFG_SECTOR_WHITELIST:
            c['news_adjustment'] = c.get('news_adjustment', 0) + 5

        kept.append(c)

    if dropped:
        print(f'  Config criteria dropped: {", ".join(dropped[:8])}{"..." if len(dropped)>8 else ""}')
    print(f'  {len(kept)} candidates after config criteria')
    return kept


def update_config_from_llm(pick_history):
    """
    Ask the LLM to review recent pick performance and propose new screening
    thresholds AND criteria. Saves result to config_overrides.json on Drive.
    Only runs when there are enough closed picks to learn from.
    """
    closed = [h for h in (pick_history or []) if h.get('result') in ('Win','Loss','Neutral')]
    if len(closed) < _CFG_MIN_PICKS_TO_LEARN:
        print(f'  Config update skipped — need {_CFG_MIN_PICKS_TO_LEARN} closed picks, have {len(closed)}')
        return

    wins = sum(1 for h in closed if h['result'] == 'Win')
    wr   = wins / len(closed) * 100

    history_lines = []
    for h in closed[-20:]:
        line = (f"  {h.get('date','?')}: {h.get('ticker','?')} [{h.get('sector','?')}/{h.get('source','?')}]"
                f" @ {h.get('price','?')} → {h.get('result','?')}"
                f" (vs QQQ: {h.get('vs_qqq_10d', h.get('vs_qqq_30d','?'))}%)")
        history_lines.append(line)

    current_cfg = {
        'RSI_MIN': RSI_MIN, 'RSI_MAX': RSI_MAX, 'ADX_MIN': ADX_MIN,
        'BUY_THRESHOLD': BUY_THRESHOLD, 'WATCH_THRESHOLD': WATCH_THRESHOLD,
        'VOLUME_MIN_RATIO': VOLUME_MIN_RATIO,
        'ATR_STOP_MULT': ATR_STOP_MULT, 'ATR_TARGET_MULT': ATR_TARGET_MULT,
        'sector_blacklist': _CFG_SECTOR_BLACKLIST,
        'sector_whitelist': _CFG_SECTOR_WHITELIST,
        'source_preference': _CFG_SOURCE_PREFERENCE,
        'require_congress': _CFG_REQUIRE_CONGRESS,
        'min_catalyst_score': _CFG_MIN_CATALYST_SCORE,
        'min_adx_buy': _CFG_MIN_ADX_BUY,
        'avoid_earnings_week': _CFG_AVOID_EARNINGS,
        'max_vix': _CFG_MAX_VIX,
        'min_price': _CFG_MIN_PRICE,
        'only_profitable': _CFG_ONLY_PROFITABLE,
        'require_above_ma': _CFG_REQUIRE_ABOVE_MA,
        'min_dollar_volume_m': _CFG_MIN_DOLLAR_VOL_M,
        'hold_days': _CFG_HOLD_DAYS,
        'sector_conc_max': _CFG_SECTOR_CONC_MAX,
        'sample_size': _CFG_SAMPLE_SIZE,
        'additional_tickers': _CFG_ADDITIONAL_TICKERS,
        'min_cash_floor': _CFG_MIN_CASH_FLOOR,
        'dd_caution_pct': _CFG_DD_CAUTION_PCT,
        'dd_severe_pct': _CFG_DD_SEVERE_PCT,
        'dd_critical_pct': _CFG_DD_CRITICAL_PCT,
        'win_threshold_pct': _CFG_WIN_THRESHOLD_PCT,
        'loss_threshold_pct': _CFG_LOSS_THRESHOLD_PCT,
        'min_picks_to_learn': _CFG_MIN_PICKS_TO_LEARN,
        'rsi_hard_cap': _CFG_RSI_HARD_CAP,
        'rsi_cap_conf': _CFG_RSI_CAP_CONF,
        'upside_hard_cap': _CFG_UPSIDE_HARD_CAP,
        'upside_cap_conf': _CFG_UPSIDE_CAP_CONF,
        'max_positions': _CFG_MAX_POSITIONS,
        'vix_low_pctile': _CFG_VIX_LOW_PCTILE,
        'vix_high_pctile': _CFG_VIX_HIGH_PCTILE,
        'sector_conc_lookback': _CFG_SECTOR_CONC_LOOKBACK,
        'sector_conc_penalty': _CFG_SECTOR_CONC_PENALTY,
        'congress_days': _CFG_CONGRESS_DAYS,
        'sec_8k_days': _CFG_SEC_8K_DAYS,
        'rsi_exit': _CFG_RSI_EXIT,
        'rsi_exit_min_profit': _CFG_RSI_EXIT_MIN_PROFIT,
        'macd_exit_min_profit': _CFG_MACD_EXIT_MIN_PROFIT,
        'entry_slippage_pct': _CFG_ENTRY_SLIPPAGE_PCT,
        'final_candidates': _CFG_FINAL_CANDIDATES,
        'pre_earnings_exit_days': _CFG_PRE_EARNINGS_DAYS,
        'squeeze_float_pct': _CFG_SQUEEZE_FLOAT_PCT,
        'squeeze_days_to_cover': _CFG_SQUEEZE_DAYS_COVER,
        'trail_atr_mult': _CFG_TRAIL_ATR_MULT,
        'volume_min_ratio': _CFG_VOLUME_MIN_RATIO,
    }

    sys_msg = (
        f'You are the autonomous portfolio manager for a short-term ({_CFG_HOLD_DAYS}-day) equity screener. '
        'You have FULL AUTHORITY to change any screening parameter, including hold_days itself. '
        'Your goal: maximize QQQ-relative returns across all picks.'
    )
    user_msg = f"""PORTFOLIO PERFORMANCE: {wr:.0f}% win rate ({wins}/{len(closed)} closed picks, Win = beat QQQ by >{_CFG_WIN_THRESHOLD_PCT}% in {_CFG_HOLD_DAYS} days)

CURRENT SCREENING CONFIG:
{json.dumps(current_cfg, indent=2)}

PICK HISTORY (last 20 closed picks, sorted oldest→newest):
{chr(10).join(history_lines)}

YOUR JOB:
1. Find the pattern. What sectors, sources, VIX regimes, or signal types are winning vs losing?
2. Change the config to capitalise on what's working and eliminate what's failing.
3. You have FULL AUTHORITY — no restrictions. You can:
   - Change any numeric threshold to any value that makes sense
   - Blacklist entire sectors that keep losing
   - Whitelist sectors that keep winning
   - Set require_congress=true if congress picks outperform
   - Set max_vix to go to cash when markets are too volatile (e.g. 25)
   - Restrict source_preference to TECHNICAL/NEWS/BOTH if one clearly outperforms
   - Raise BUY_THRESHOLD to 85-90 to be more selective when losing
   - Lower WATCH_THRESHOLD to 60 to see more ideas when confident
   - Set avoid_earnings_week=true if earnings plays keep failing
   - Set only_profitable=true if unprofitable companies are underperforming
   - Raise min_adx_buy to 30+ if weak-trend stocks keep losing
   - Set require_above_ma=false to catch early breakouts below MA (risky but sometimes right)
   - Raise min_dollar_volume_m to 50+ for large-cap only if small/mid keeps losing
   - Change hold_days to 7 or 14 if 10-day results are inconsistent
   - Add additional_tickers (e.g. ["PLTR","ARM","RDDT"]) to expand the universe
   - Raise sector_conc_max if diversification is hurting returns, lower it to force diversity
   - Lower rsi_exit (e.g. 75) to take profits earlier; raise it (e.g. 82) to let winners run further
   - Lower rsi_exit_min_profit / macd_exit_min_profit if momentum reversals are costing unrealised gains
4. When win rate >65%: fine-tune only. When win rate <50%: make meaningful changes.
5. If no clear pattern: keep current config unchanged.

Return ONLY valid JSON — no markdown fences, no text outside the JSON.
You can change ANY value. Keep unchanged values as-is.
{{
  "RSI_MIN": {RSI_MIN},
  "RSI_MAX": {RSI_MAX},
  "ADX_MIN": {ADX_MIN},
  "BUY_THRESHOLD": {BUY_THRESHOLD},
  "WATCH_THRESHOLD": {WATCH_THRESHOLD},
  "VOLUME_MIN_RATIO": {VOLUME_MIN_RATIO},
  "ATR_STOP_MULT": {ATR_STOP_MULT},
  "ATR_TARGET_MULT": {ATR_TARGET_MULT},
  "sector_blacklist": {json.dumps(_CFG_SECTOR_BLACKLIST)},
  "sector_whitelist": {json.dumps(_CFG_SECTOR_WHITELIST)},
  "source_preference": "{_CFG_SOURCE_PREFERENCE}",
  "require_congress": {str(_CFG_REQUIRE_CONGRESS).lower()},
  "min_catalyst_score": {_CFG_MIN_CATALYST_SCORE},
  "min_adx_buy": {_CFG_MIN_ADX_BUY},
  "avoid_earnings_week": {str(_CFG_AVOID_EARNINGS).lower()},
  "max_vix": {_CFG_MAX_VIX},
  "min_price": {_CFG_MIN_PRICE},
  "only_profitable": {str(_CFG_ONLY_PROFITABLE).lower()},
  "require_above_ma": {str(_CFG_REQUIRE_ABOVE_MA).lower()},
  "min_dollar_volume_m": {_CFG_MIN_DOLLAR_VOL_M},
  "hold_days": {_CFG_HOLD_DAYS},
  "sector_conc_max": {_CFG_SECTOR_CONC_MAX},
  "sample_size": {_CFG_SAMPLE_SIZE},
  "additional_tickers": {json.dumps(_CFG_ADDITIONAL_TICKERS)},
  "brokerage_fee": {_CFG_BROKERAGE_FEE},
  "min_cash_floor": {_CFG_MIN_CASH_FLOOR},
  "dd_caution_pct": {_CFG_DD_CAUTION_PCT},
  "dd_severe_pct": {_CFG_DD_SEVERE_PCT},
  "dd_critical_pct": {_CFG_DD_CRITICAL_PCT},
  "win_threshold_pct": {_CFG_WIN_THRESHOLD_PCT},
  "loss_threshold_pct": {_CFG_LOSS_THRESHOLD_PCT},
  "min_picks_to_learn": {_CFG_MIN_PICKS_TO_LEARN},
  "rsi_hard_cap": {_CFG_RSI_HARD_CAP},
  "rsi_cap_conf": {_CFG_RSI_CAP_CONF},
  "upside_hard_cap": {_CFG_UPSIDE_HARD_CAP},
  "upside_cap_conf": {_CFG_UPSIDE_CAP_CONF},
  "max_positions": {_CFG_MAX_POSITIONS},
  "vix_low_pctile": {_CFG_VIX_LOW_PCTILE},
  "vix_high_pctile": {_CFG_VIX_HIGH_PCTILE},
  "sector_conc_lookback": {_CFG_SECTOR_CONC_LOOKBACK},
  "sector_conc_penalty": {_CFG_SECTOR_CONC_PENALTY},
  "congress_days": {_CFG_CONGRESS_DAYS},
  "sec_8k_days": {_CFG_SEC_8K_DAYS},
  "rsi_exit": {_CFG_RSI_EXIT},
  "rsi_exit_min_profit": {_CFG_RSI_EXIT_MIN_PROFIT},
  "macd_exit_min_profit": {_CFG_MACD_EXIT_MIN_PROFIT},
  "entry_slippage_pct": {_CFG_ENTRY_SLIPPAGE_PCT},
  "final_candidates": {_CFG_FINAL_CANDIDATES},
  "pre_earnings_exit_days": {_CFG_PRE_EARNINGS_DAYS},
  "squeeze_float_pct": {_CFG_SQUEEZE_FLOAT_PCT},
  "squeeze_days_to_cover": {_CFG_SQUEEZE_DAYS_COVER},
  "trail_atr_mult": {_CFG_TRAIL_ATR_MULT},
  "volume_min_ratio": {_CFG_VOLUME_MIN_RATIO},
  "reasoning": "one clear sentence: what changed and why the data supports it"
}}"""

    print('\n  Asking LLM to update screening config based on pick history...')
    try:
        raw = call_llm(sys_msg, user_msg, max_tokens=600)
        if '{' in raw and '}' in raw:
            raw = raw[raw.index('{'):raw.rindex('}')+1]
        cfg = json.loads(raw)

        # Validate core keys present
        required = ['RSI_MIN','RSI_MAX','ADX_MIN','BUY_THRESHOLD','WATCH_THRESHOLD',
                    'sector_blacklist','source_preference','reasoning']
        for k in required:
            if k not in cfg:
                print(f'  Config update skipped — LLM response missing key: {k}')
                return

        print(f'  LLM config update: {cfg.get("reasoning","")[:120]}')
        save_config_overrides(cfg)
    except Exception as e:
        print(f'  Config update failed: {e}')


# ============================================================
# PORTFOLIO  —  paper trading engine ($10,000 starting capital)
# ============================================================

def load_portfolio():
    """Load portfolio.json from Drive, or create fresh $10k portfolio."""
    paths = [PORTFOLIO_JSON, 'portfolio.json']
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    pf = json.load(f)
                # Backfill any missing keys (in case schema evolved)
                pf.setdefault('cash', STARTING_CAPITAL)
                pf.setdefault('starting_capital', STARTING_CAPITAL)
                pf.setdefault('positions', [])
                pf.setdefault('closed_trades', [])
                pf.setdefault('total_realized_pnl', 0.0)
                return pf
            except Exception as e:
                print(f'  portfolio.json load error: {e}')
    # First run — create fresh portfolio
    pf = {
        'cash': STARTING_CAPITAL,
        'starting_capital': STARTING_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'total_realized_pnl': 0.0,
        'created': datetime.now().strftime('%Y-%m-%d'),
    }
    print(f'  New portfolio created — starting capital: ${STARTING_CAPITAL:,.0f}')
    return pf


def save_portfolio(pf):
    pf['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    try:
        with open(PORTFOLIO_JSON, 'w') as f:
            json.dump(pf, f, indent=2)
    except Exception as e:
        print(f'  portfolio save error: {e}')


def update_portfolio_prices(pf):
    """
    Fetch live prices for all open positions, update P&L, then auto-close any
    position that has hit its stop-loss, profit target, or is 1 day before earnings.
    These are the three mechanical exit rules for a 10-day swing trade.
    """
    if not pf['positions']:
        return pf
    tickers = [p['ticker'] for p in pf['positions']]
    try:
        raw = yf.download(tickers, period='2d', auto_adjust=True,
                          progress=False, threads=True)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
        for pos in pf['positions']:
            try:
                t = pos['ticker']
                col = closes[t] if t in closes.columns else closes.iloc[:, 0]
                curr = float(col.dropna().iloc[-1])
                pos['current_price']      = round(curr, 2)
                pos['current_value']      = round(curr * pos['shares'], 2)
                pos['unrealized_pnl']     = round(pos['current_value'] - pos['cost_basis'], 2)
                pos['unrealized_pnl_pct'] = round((pos['current_value'] / pos['cost_basis'] - 1) * 100, 2)
                entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
                pos['hold_days'] = (datetime.now() - entry_date).days
            except Exception:
                pass
    except Exception as e:
        print(f'  price update error: {e}')

    # ── Trailing stop — raise stop as price rises, never lower it ───────────
    for pos in pf['positions']:
        curr = pos.get('current_price')
        if curr is None:
            continue
        hwm = float(pos.get('high_watermark', pos['entry_price']))
        if float(curr) > hwm:
            pos['high_watermark'] = round(float(curr), 2)
            atr = float(pos.get('atr_at_entry', 0))
            if atr > 0:
                new_stop = round(float(curr) - _CFG_TRAIL_ATR_MULT * atr, 2)
                old_stop = float(pos.get('stop_price') or 0)
                if new_stop > old_stop:
                    pos['stop_price'] = new_stop
                    print(f'  Trailing stop ratcheted: {pos["ticker"]} stop {old_stop} → {new_stop} (new high {curr})')

    # ── Mechanical exit rules (run after prices are updated) ─────────────────
    to_close = []   # (ticker, reason)

    for pos in pf['positions']:
        curr  = pos.get('current_price')
        stop  = pos.get('stop_price')
        tgt   = pos.get('target_price')
        t     = pos['ticker']
        if curr is None:
            continue

        # Rule 1 — Stop-loss breached
        if stop and float(curr) <= float(stop):
            to_close.append((t, f'stop_loss (curr={curr} ≤ stop={stop})'))
            continue

        # Rule 2 — Profit target reached
        if tgt and float(curr) >= float(tgt):
            to_close.append((t, f'profit_target (curr={curr} ≥ target={tgt})'))
            continue

        # Rule 3 — Earnings in ≤1 day: close to avoid binary event risk
        try:
            cal    = yf.Ticker(t).calendar
            ed_raw = None
            if isinstance(cal, dict):
                ed_raw = cal.get('Earnings Date')
                if isinstance(ed_raw, list) and ed_raw:
                    ed_raw = ed_raw[0]
            elif cal is not None and hasattr(cal, 'empty') and not cal.empty:
                if 'Earnings Date' in cal.index:
                    vals = cal.loc['Earnings Date'].values
                    if len(vals) > 0:
                        ed_raw = vals[0]
            if ed_raw is not None:
                days_to_earnings = (pd.to_datetime(ed_raw).date() - datetime.now().date()).days
                pos['earnings_days_away'] = days_to_earnings  # keep fresh for portfolio_summary_str
                if 0 <= days_to_earnings <= _CFG_PRE_EARNINGS_DAYS:
                    to_close.append((t, f'pre_earnings (earnings in {days_to_earnings}d)'))
        except:
            pass

        # Rule 4 — RSI overbought exit: price extended, lock in gains
        # Only fires when position is already profitable past LLM-set threshold
        pnl_pct = pos.get('unrealized_pnl_pct', 0)
        try:
            _h = yf.Ticker(t).history(period='21d')
            if len(_h) >= 14:
                _delta = _h['Close'].diff()
                _gain  = _delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                _loss  = (-_delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                _rs    = _gain / _loss.replace(0, float('nan'))
                _rsi   = (100 - (100 / (1 + _rs))).iloc[-1]
                if _rsi > _CFG_RSI_EXIT and pnl_pct >= _CFG_RSI_EXIT_MIN_PROFIT:
                    to_close.append((t, f'rsi_overbought (RSI={_rsi:.0f} > {_CFG_RSI_EXIT}, up {pnl_pct:.1f}%)'))
        except:
            pass

        # Rule 5 — MACD bearish cross after a profitable run: momentum has turned
        try:
            if pnl_pct >= _CFG_MACD_EXIT_MIN_PROFIT and t not in [x[0] for x in to_close]:
                _h2   = yf.Ticker(t).history(period='40d') if '_h' not in dir() or len(_h) < 26 else _h
                if len(_h2) >= 26:
                    _close      = _h2['Close']
                    _macd       = _close.ewm(span=12).mean() - _close.ewm(span=26).mean()
                    _signal     = _macd.ewm(span=9).mean()
                    _cross_now  = _macd.iloc[-1] < _signal.iloc[-1]
                    _cross_prev = _macd.iloc[-2] >= _signal.iloc[-2]
                    if _cross_now and _cross_prev:
                        to_close.append((t, f'macd_bearish_cross (up {pnl_pct:.1f}%, momentum turned)'))
        except:
            pass

    for ticker, reason in to_close:
        pos = next((p for p in pf['positions'] if p['ticker'] == ticker), None)
        if pos:
            pf = close_position(pf, ticker, pos['current_price'], reason=reason)

    return pf


def _sharesies_fee(amount_usd, pf, nzdusd_rate=None, side='buy'):
    """
    Calculate actual Sharesies brokerage fee for this trade.
    $15/month plan: $5,000 NZD free buys + $5,000 NZD free sells per month.
    Over the limit: 0.5% of trade value, capped at $5 USD.
    Tracks usage in portfolio.json so the free tier is consumed correctly.
    nzdusd_rate falls back to pf['last_nzdusd_rate'] if not supplied.
    """
    if not nzdusd_rate:
        nzdusd_rate = pf.get('last_nzdusd_rate', 0)
    if not nzdusd_rate or float(nzdusd_rate) <= 0:
        return 0.0  # rate unavailable — assume free

    month_key = datetime.now().strftime('%Y-%m')
    if pf.get('sharesies_month') != month_key:
        pf['sharesies_month']      = month_key
        pf['sharesies_bought_usd'] = 0.0
        pf['sharesies_sold_usd']   = 0.0

    coverage_usd  = SHARESIES_COVERAGE_NZD * float(nzdusd_rate)
    used_key      = 'sharesies_bought_usd' if side == 'buy' else 'sharesies_sold_usd'
    already_used  = float(pf.get(used_key, 0.0))
    remaining_free = max(0.0, coverage_usd - already_used)

    if amount_usd <= remaining_free:
        fee = 0.0
    else:
        over_amount = amount_usd - remaining_free
        fee = min(over_amount * 0.005, 5.0)  # 0.5%, max $5 USD

    pf[used_key] = round(already_used + amount_usd, 2)
    return round(fee, 2)


def open_position(pf, ticker, entry_price, amount_usd, stop, target, sector='', atr=0, nzdusd_rate=None):
    """Deploy cash into a new position. LLM controls amount_usd."""
    # Guard: already holding this ticker
    if any(p['ticker'] == ticker for p in pf['positions']):
        print(f'  Portfolio: already holding {ticker} — skipping')
        return pf
    # Guard: not enough cash (floor is LLM-configurable via min_cash_floor)
    if pf['cash'] < amount_usd or amount_usd < _CFG_MIN_CASH_FLOOR:
        print(f'  Portfolio: insufficient cash (${pf["cash"]:,.0f}) for ${amount_usd:,.0f} position')
        return pf
    shares     = int(amount_usd / entry_price)          # whole shares only — remainder stays as cash
    if shares < 1:
        print(f'  Portfolio: cannot afford even 1 share of {ticker} @ ${entry_price} with ${amount_usd:,.0f}')
        return pf
    stock_cost = round(shares * entry_price, 2)         # actual dollars spent on shares
    brokerage  = _sharesies_fee(stock_cost, pf, nzdusd_rate, side='buy')
    total_cost = round(stock_cost + brokerage, 2)
    pf['positions'].append({
        'ticker':              ticker,
        'shares':              shares,
        'entry_price':         round(entry_price, 2),
        'cost_basis':          total_cost,   # actual shares cost + brokerage
        'brokerage_in':        brokerage,
        'entry_date':          datetime.now().strftime('%Y-%m-%d'),
        'stop_price':          round(stop, 2) if isinstance(stop, (int, float)) else None,
        'target_price':        round(target, 2) if isinstance(target, (int, float)) else None,
        'sector':              sector,
        'current_price':       round(entry_price, 2),
        'current_value':       round(stock_cost, 2),
        'unrealized_pnl':      round(-brokerage, 2),
        'unrealized_pnl_pct':  round(-brokerage / total_cost * 100, 2) if total_cost else 0.0,
        'hold_days':           0,
        'high_watermark':      round(entry_price, 2),   # trailing stop pivot
        'atr_at_entry':        round(float(atr), 4) if atr else 0.0,
    })
    pf['cash'] = round(pf['cash'] - total_cost, 2)
    leftover   = round(amount_usd - total_cost, 2)
    fee_note   = f' + ${brokerage:.2f} brokerage' if brokerage else ' (free — within Sharesies coverage)'
    print(f'  Portfolio: OPENED {ticker}  {shares} shares @ ${entry_price}  spent ${total_cost:,.2f}{fee_note}  leftover ${leftover:,.2f}  cash ${pf["cash"]:,.2f}')
    return pf


def close_position(pf, ticker, exit_price, reason='hold_period', nzdusd_rate=None):
    """Close an open position and return cash + P&L to portfolio."""
    pos = next((p for p in pf['positions'] if p['ticker'] == ticker), None)
    if not pos:
        return pf
    gross_proceeds  = round(exit_price * pos['shares'], 2)
    sell_brokerage  = _sharesies_fee(gross_proceeds, pf, nzdusd_rate, side='sell')
    exit_value      = round(gross_proceeds - sell_brokerage, 2)  # net after sell-side fee
    realized_pnl   = round(exit_value - pos['cost_basis'], 2)   # cost_basis already includes buy-side fee
    realized_pct   = round((exit_value / pos['cost_basis'] - 1) * 100, 2)
    pf['closed_trades'].append({
        'ticker':           ticker,
        'entry_price':      pos['entry_price'],
        'exit_price':       round(exit_price, 2),
        'shares':           pos['shares'],
        'cost_basis':       pos['cost_basis'],
        'exit_value':       exit_value,
        'realized_pnl':     realized_pnl,
        'realized_pnl_pct': realized_pct,
        'entry_date':       pos['entry_date'],
        'exit_date':        datetime.now().strftime('%Y-%m-%d'),
        'hold_days':        pos.get('hold_days', 0),
        'reason':           reason,
    })
    pf['cash']                = round(pf['cash'] + exit_value, 2)
    pf['total_realized_pnl']  = round(pf.get('total_realized_pnl', 0) + realized_pnl, 2)
    pf['positions']           = [p for p in pf['positions'] if p['ticker'] != ticker]
    sign = '+' if realized_pnl >= 0 else ''
    print(f'  Portfolio: CLOSED {ticker} @ ${exit_price}  P&L {sign}${realized_pnl:,.2f} ({sign}{realized_pct:.1f}%)  [{reason}]')
    return pf


def reconcile_closed_picks(pf):
    """
    After update_results runs, check PICKS_CSV for any Win/Loss/Neutral entries
    that still have an open position in the portfolio — and close them.
    """
    if not os.path.exists(PICKS_CSV):
        return pf
    try:
        df = pd.read_csv(PICKS_CSV)
        closed = df[df['Result'].isin(['Win', 'Loss', 'Neutral'])]
        open_tickers = {p['ticker'] for p in pf['positions']}
        for _, row in closed.iterrows():
            t = str(row.get('Ticker', '')).strip()
            if t not in open_tickers:
                continue
            # Use 10-day close price from CSV; fall back to live price
            price = None
            try:    price = float(row['Price_10d'])
            except: pass
            if not price or not (price > 0):
                try:    price = float(yf.Ticker(t).history(period='2d')['Close'].dropna().iloc[-1])
                except: pass
            if price and price > 0:
                pf = close_position(pf, t, price, reason=str(row.get('Result', 'hold_period')))
    except Exception as e:
        print(f'  reconcile error: {e}')
    return pf


def portfolio_summary_str(pf):
    """One-block text summary for the LLM — shown before Round 1."""
    total_value = round(pf['cash'] + sum(p.get('current_value', p['cost_basis']) for p in pf['positions']), 2)
    total_pnl   = round(total_value - pf['starting_capital'], 2)
    total_pct   = round(total_pnl / pf['starting_capital'] * 100, 2)

    # Drawdown severity — inform the LLM; it decides how to respond
    if total_pct <= _CFG_DD_CRITICAL_PCT:
        dd_warn = (f'CRITICAL DRAWDOWN: Portfolio is {total_pct:.1f}% from starting capital '
                   f'(your critical threshold is {_CFG_DD_CRITICAL_PCT:.0f}%). You decide how to respond.')
    elif total_pct <= _CFG_DD_SEVERE_PCT:
        dd_warn = (f'SEVERE DRAWDOWN: Portfolio is {total_pct:.1f}% from starting capital '
                   f'(your severe threshold is {_CFG_DD_SEVERE_PCT:.0f}%). You decide how to respond.')
    elif total_pct <= _CFG_DD_CAUTION_PCT:
        dd_warn = (f'DRAWDOWN NOTICE: Portfolio is {total_pct:.1f}% from starting capital '
                   f'(your caution threshold is {_CFG_DD_CAUTION_PCT:.0f}%). You decide how to respond.')
    else:
        dd_warn = ''

    if pf['cash'] <= _CFG_MIN_CASH_FLOOR:
        cash_warn = (f'FULLY DEPLOYED: ${pf["cash"]:,.2f} cash remaining (your floor is ${_CFG_MIN_CASH_FLOOR:,.0f}). '
                     f'You decide whether to output a BUY or hold.')
    else:
        cash_warn = ''

    invested   = total_value - pf['cash']
    deploy_pct = round(invested / total_value * 100, 0) if total_value > 0 else 0
    idle_pct   = 100 - deploy_pct

    # ── Sector exposure breakdown ─────────────────────────────────────────────
    sector_exposure = {}
    for p in pf['positions']:
        sec = p.get('sector', 'Unknown')
        val = p.get('current_value', p.get('cost_basis', 0))
        sector_exposure[sec] = sector_exposure.get(sec, 0) + val
    sector_str = '  '.join(
        f'{sec}: {round(val/total_value*100)}%'
        for sec, val in sorted(sector_exposure.items(), key=lambda x: -x[1])
    ) if sector_exposure else 'none'

    # ── Realized P&L this week ────────────────────────────────────────────────
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    week_trades = [t for t in pf.get('closed_trades', [])
                   if str(t.get('close_date', ''))[:10] >= week_ago]
    week_pnl = sum(t.get('realized_pnl', 0) for t in week_trades)
    week_str = f'${week_pnl:+,.2f} from {len(week_trades)} trades this week' if week_trades else 'no closed trades this week'

    lines = [
        f'PORTFOLIO: ${total_value:,.2f} ({total_pnl:+,.2f} / {total_pct:+.1f}% vs ${pf["starting_capital"]:,.0f} starting)',
        f'Cash: ${pf["cash"]:,.2f}  |  Positions: {len(pf["positions"])}/{_CFG_MAX_POSITIONS}  |  Deployed: {deploy_pct:.0f}%  |  Idle: {idle_pct:.0f}%',
        f'Sector exposure: {sector_str}',
        f'Realized P&L: ${pf.get("total_realized_pnl", 0):+,.2f} all-time  |  {week_str}',
    ]

    # ── Open positions — full detail ──────────────────────────────────────────
    for p in pf['positions']:
        upl = p.get('unrealized_pnl', 0)
        upc = p.get('unrealized_pnl_pct', 0)
        stop  = p.get('stop_price', '?')
        tgt   = p.get('target_price', '?')
        earn  = p.get('earnings_days_away')
        earn_str = f' | EARNINGS IN {earn}d ⚠' if earn is not None and isinstance(earn, (int,float)) and earn <= 10 else ''
        gap   = p.get('open_gap_pct')
        gap_str = f' | gap {gap:+.1f}% at entry' if gap is not None else ''
        lines.append(
            f'  {p["ticker"]} [{p.get("sector","?")}]: {p["shares"]}sh @ ${p["entry_price"]} → ${p.get("current_price","?")} '
            f'({upl:+.0f} / {upc:+.1f}%) | {p.get("hold_days",0)}d | Stop ${stop} | Target ${tgt}{earn_str}{gap_str}'
        )

    if dd_warn:
        lines.append(f'*** {dd_warn} ***')
    if cash_warn:
        lines.append(f'*** {cash_warn} ***')
    return '\n'.join(lines)


# ============================================================
# CELL 5 - RUN DAILY SCREENER
# ============================================================

def run_screener():
    print('🚀 DAILY STOCK SCREENER v6.2')
    print(f'   Time:   {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'   Folder: {DRIVE_FOLDER}')
    print(f'   Stocks: {len(STOCK_UNIVERSE)} | ETFs: {len(KEY_ETFS)}')
    print('='*65)

    load_config_overrides()

    # Weekend check — always use US Eastern time, not local time.
    # A user in NZ running Saturday 9 AM NZST is actually Friday 5 PM ET — valid trading day.
    try:
        from datetime import timezone, timedelta
        try:
            from zoneinfo import ZoneInfo
            _et_now = datetime.now(ZoneInfo('America/New_York'))
        except ImportError:
            # Fallback: approximate EDT/EST offset without zoneinfo
            _month = datetime.now(timezone.utc).month
            _et_offset = -4 if 3 <= _month <= 11 else -5  # EDT Mar-Nov, EST Dec-Feb
            _et_now = datetime.now(timezone(timedelta(hours=_et_offset)))
    except Exception:
        _et_now = datetime.now()  # last resort: local time

    if _et_now.weekday() >= 5:
        if _et_now.weekday() == 5:  # US Saturday = Sunday NZT → send weekly summary
            print(f'\nMarket closed (US Saturday {_et_now.strftime("%Y-%m-%d %H:%M")} ET) — sending weekly summary...')
            send_weekly_summary()
        else:
            print(f'\nMarket closed (US Sunday {_et_now.strftime("%Y-%m-%d %H:%M")} ET) — nothing to do. See you Monday.')
        return None
    print(f'   US ET:  {_et_now.strftime("%Y-%m-%d %H:%M %Z")} (weekday {_et_now.weekday()}, markets open)')

    # Apply LLM-controlled universe expansions and sample size
    scan_universe = list(STOCK_UNIVERSE)
    if _CFG_ADDITIONAL_TICKERS:
        extra = [t for t in _CFG_ADDITIONAL_TICKERS if t not in UNIVERSE_SET and t not in ETF_SET]
        if extra:
            print(f'  LLM added tickers: {", ".join(extra)}')
            scan_universe = extra + scan_universe  # LLM picks come first
    if _CFG_SAMPLE_SIZE < len(scan_universe):
        scan_universe = scan_universe[:_CFG_SAMPLE_SIZE]

    print('\nStep 0/8: Loading portfolio...')
    portfolio = load_portfolio()
    portfolio = update_portfolio_prices(portfolio)
    print(portfolio_summary_str(portfolio))

    # Drawdown & deployment status (human-readable console summary)
    _total_val = round(portfolio['cash'] + sum(p.get('current_value', p['cost_basis']) for p in portfolio['positions']), 2)
    _dd_pct    = round((_total_val - portfolio['starting_capital']) / portfolio['starting_capital'] * 100, 2)
    if _dd_pct <= _CFG_DD_CRITICAL_PCT:
        print(f'  *** CRITICAL DRAWDOWN {_dd_pct:.1f}% — portfolio ${_total_val:,.0f} — LLM in capital preservation mode ***')
    elif _dd_pct <= _CFG_DD_SEVERE_PCT:
        print(f'  *** SEVERE DRAWDOWN {_dd_pct:.1f}% — portfolio ${_total_val:,.0f} — LLM using high-conviction-only mode ***')
    elif _dd_pct <= _CFG_DD_CAUTION_PCT:
        print(f'  CAUTION: Portfolio down {abs(_dd_pct):.1f}% (${_total_val:,.0f})')
    if portfolio['cash'] <= _CFG_MIN_CASH_FLOOR:
        if portfolio['positions']:
            print(f'  FULLY DEPLOYED: ${portfolio["cash"]:,.2f} cash left — {len(portfolio["positions"])} open position(s) still running — no new buys today')
        else:
            print(f'  PORTFOLIO DEPLETED: ${portfolio["cash"]:,.2f} cash, no open positions — LLM will output NO PICK')

    print('\nStep 1/8: Updating past results + exit signals...')
    update_results(PICKS_CSV, PICK_COLS)
    update_results(WATCH_CSV, WATCH_COLS)
    portfolio = reconcile_closed_picks(portfolio)
    _LLM_CALL_COUNT[0] = 0  # reset call counter for this run

    print('\nStep 2/8: Market context...')
    ctx = get_market_context()

    # Store live NZD/USD in portfolio so _sharesies_fee() can use it without re-fetching
    _nzdusd = ctx.get('global_macro', {}).get('nzdusd', {}).get('price')
    if _nzdusd:
        portfolio['last_nzdusd_rate'] = float(_nzdusd)
        save_portfolio(portfolio)

    print('\nStep 3/8: Fetching all data (parallel)...')
    batch_data = batch_download(scan_universe + KEY_ETFS)
    if not batch_data:
        print('  Batch download failed - cannot proceed')
        display_scorecard(); return None

    headlines        = fetch_macro_news()
    all_stock_news   = fetch_all_stock_news_parallel(STOCK_UNIVERSE)
    all_fundamentals = fetch_all_fundamentals_parallel(STOCK_UNIVERSE)
    sector_ranks, sector_perf = compute_sector_ranks(batch_data)

    # Compute sector 1-day returns from batch data and add to ctx
    sector_1d = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        df = batch_data.get(etf)
        if df is not None and len(df) >= 2:
            try:
                r1d = round((float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])) /
                             float(df['Close'].iloc[-2]) * 100, 2)
                sector_1d[sector] = r1d
            except:
                pass
    ctx['sector_1d'] = sector_1d
    top_sectors    = sorted(sector_1d.items(), key=lambda x: x[1], reverse=True)
    if top_sectors:
        print(f'  Sector flows: TOP {top_sectors[0][0]} {top_sectors[0][1]:+.1f}% | '
              f'BOT {top_sectors[-1][0]} {top_sectors[-1][1]:+.1f}%')

    print('\nStep 4/8: Bidirectional screening...')
    technical_passed = screen_technical(batch_data, ctx)
    news_rescued     = screen_news(batch_data, all_stock_news, technical_passed, ctx)
    candidates       = merge_candidates(technical_passed, news_rescued, all_stock_news, all_fundamentals)
    sector_news      = derive_sector_sentiment(candidates)  # derived from candidates, not ETF API

    print('\nStep 4.5/8: Stream B - headline ticker extraction...')
    b_cands = stream_b_from_headlines(headlines, batch_data, technical_passed, all_stock_news, all_fundamentals, ctx)
    existing = {c['ticker'] for c in candidates}
    for c in b_cands:
        if c['ticker'] not in existing:
            candidates.append(c); existing.add(c['ticker'])

    # Exit analysis here — has both market context AND fresh news
    portfolio = analyze_exit_signals(ctx, all_stock_news, portfolio=portfolio) or portfolio

    if not candidates:
        print('\nNo candidates from any stream today - NO PICK')
        display_scorecard(); return None

    print('\nStep 4.6/8: Options P/C + insider + congress + SEC 8-K...')
    options_data, insider_data = fetch_options_and_insider_parallel(candidates)
    congress_data = fetch_congress_trades(days=_CFG_CONGRESS_DAYS)
    candidate_tickers = [c['ticker'] for c in candidates]
    sec_filings = fetch_sec_8k(candidate_tickers, days=_CFG_SEC_8K_DAYS)

    print('\nStep 4.7/8: LLM catalyst scoring (every candidate)...')
    candidates = batch_catalyst_score(candidates, ctx, all_stock_news)
    candidates = [c for c in candidates if not c.get('auto_drop')]

    print('\nStep 5/8: News intelligence (3 layers + analyst actions + SEC filings)...')
    nd         = get_news_intelligence(candidates, ctx, headlines, sector_news, all_stock_news, sec_filings=sec_filings)
    candidates = apply_news(candidates, nd)

    if not candidates:
        print('All candidates dropped by news filter')
        display_scorecard(); return None

    market_sentiment = nd.get('market_sentiment','NEUTRAL')
    candidates = enrich_with_scores(candidates, ctx, market_sentiment, sector_ranks, options_data, insider_data, congress_data)

    print('\nStep 5.5/8: Applying LLM config criteria...')
    candidates = apply_config_criteria(candidates, ctx=ctx)
    if not candidates:
        print('All candidates dropped by config criteria')
        display_scorecard(); return None

    pick_history = load_performance_history(PICKS_CSV)
    if pick_history:
        wins=sum(1 for h in pick_history if h['result']=='Win')
        print(f'  Self-calibration: {len(pick_history)} prior picks, {wins/len(pick_history)*100:.0f}% win rate')

    print('\nStep 6/8: Claude final scoring...')
    result = analyze_with_nvidia(candidates, ctx, nd, pick_history=pick_history, portfolio=portfolio)
    if not result:
        print('NVIDIA analysis failed - no result returned'); return None

    # Post-hoc cap enforcement — only active if LLM has enabled caps via config
    pick = result.get('top_pick',{})
    if pick and pick.get('ticker') not in (None,'','NONE'):
        match=next((c for c in candidates if c['ticker']==pick['ticker']),{})
        rsi=match.get('rsi',50); upside=match.get('upside_pct'); orig=pick.get('confidence',0)
        if _CFG_RSI_HARD_CAP < 999 and rsi > _CFG_RSI_HARD_CAP and pick.get('confidence',0) > _CFG_RSI_CAP_CONF:
            pick['confidence'] = _CFG_RSI_CAP_CONF
            print(f'  Cap: RSI={rsi}>{_CFG_RSI_HARD_CAP}, confidence {orig}->{_CFG_RSI_CAP_CONF}')
        if _CFG_UPSIDE_HARD_CAP > -999 and upside is not None and upside < _CFG_UPSIDE_HARD_CAP and pick.get('confidence',0) > _CFG_UPSIDE_CAP_CONF:
            pick['confidence'] = _CFG_UPSIDE_CAP_CONF
            print(f'  Cap: analyst upside={upside}%<{_CFG_UPSIDE_HARD_CAP}%, confidence {orig}->{_CFG_UPSIDE_CAP_CONF}')
        if pick['confidence']<BUY_THRESHOLD:
            pick['signal']='WATCH' if pick['confidence']>=WATCH_THRESHOLD else 'NO PICK'

    wl=result.get('watch_candidates',[]); conf=pick.get('confidence',0); sig=pick.get('signal','NO PICK')
    ep='N/A'
    ep_close='N/A'   # last close — used for CSV display reference
    if sig=='BUY' and conf>=BUY_THRESHOLD:
        match=next((c for c in candidates if c['ticker']==pick['ticker']),{})
        _p = match.get('price')
        if _p is not None and float(_p) > 0:
            ep_close=round(float(_p),2)
        else:
            try: ep_close=round(float(yf.Ticker(pick['ticker']).history(period='2d')['Close'].dropna().iloc[-1]),2)
            except: ep_close='N/A'
        # Realistic entry = estimated next-day open (screener runs after close; actual buy is at next open)
        if isinstance(ep_close, float):
            ep = round(ep_close * (1 + _CFG_ENTRY_SLIPPAGE_PCT / 100), 2)
            print(f'  Entry estimate: close=${ep_close} + {_CFG_ENTRY_SLIPPAGE_PCT}% slippage = ${ep} (stop/target anchored here)')
        else:
            ep = ep_close

    display_result(result, ctx, nd, ep, wl, all_candidates=candidates)

    _rules = (result or {}).get('derived_rules', [])
    _summary = (result or {}).get('learning_summary', '')
    _fund = next((c for c in candidates if c['ticker'] == pick.get('ticker', '')), {})
    _atr  = _fund.get('atr', 0)
    _stop = round(ep - ATR_STOP_MULT  * _atr, 2) if _atr and isinstance(ep, (int, float)) else 'N/A'
    _tgt  = round(ep + ATR_TARGET_MULT * _atr, 2) if _atr and isinstance(ep, (int, float)) else 'N/A'

    # Open portfolio position — LLM chose position_size_pct
    if sig == 'BUY' and conf >= BUY_THRESHOLD and isinstance(ep, (int, float)) and ep > 0:
        pct    = min(float(pick.get('position_size_pct', 20)), 100)  # LLM sets this; 100% max is physics
        amount = round(portfolio['cash'] * pct / 100, 2)
        portfolio = open_position(portfolio, pick.get('ticker',''), ep, amount,
                                  _stop if isinstance(_stop, float) else 0,
                                  _tgt  if isinstance(_tgt,  float) else 0,
                                  pick.get('sector',''), atr=_atr or 0,
                                  nzdusd_rate=portfolio.get('last_nzdusd_rate'))
        save_portfolio(portfolio)

    print('\nStep 8/8: Saving results...')
    if sig=='BUY' and conf>=BUY_THRESHOLD:
        save_pick(pick, ctx, ep, PICKS_CSV, PICK_COLS, all_candidates=candidates,
                  portfolio=portfolio, stop_price=_stop, target_price=_tgt)
    for w in wl:
        if w.get('confidence',0)>=WATCH_THRESHOLD:
            wmatch = next((c for c in candidates if c['ticker']==w['ticker']),{})
            _wp = wmatch.get('price')
            if _wp is not None and float(_wp) > 0:
                wp = round(float(_wp),2)
            else:
                try: wp=round(float(yf.Ticker(w['ticker']).history(period='2d')['Close'].dropna().iloc[-1]),2)
                except: wp='N/A'
            wfund = next((c for c in candidates if c['ticker']==w['ticker']),{})
            watr  = wfund.get('atr',0)
            wstop = round(wp - ATR_STOP_MULT * watr, 2) if watr and isinstance(wp,(int,float)) else None
            wtgt  = round(wp + ATR_TARGET_MULT * watr, 2) if watr and isinstance(wp,(int,float)) else None
            save_pick(w, ctx, wp, WATCH_CSV, WATCH_COLS, all_candidates=candidates,
                      watch_score=w.get('confidence'), portfolio=portfolio, stop_price=wstop, target_price=wtgt)

    save_html_report(result, ctx, nd, ep, wl, derived_rules=_rules, learning_summary=_summary,
                     stop_price=_stop, target_price=_tgt)
    display_scorecard()
    send_whatsapp(pick, ctx, ep, wl, _stop, _tgt, candidates=candidates, portfolio=portfolio)

    print('\nStep 9/8: LLM self-adaptation (updating config for next run)...')
    update_config_from_llm(pick_history)

    return result


result = run_screener()
