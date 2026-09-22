"""Deterministic scoring: what a stock is worth before any LLM sees it.

Split out of LLM_Portfolio_Manager because this layer is genuinely separate -
it reads no shared state, calls nothing in the parent, and touches no network.
Everything here is arithmetic on a price frame and a list of headlines, which
is what makes it testable in isolation and safe to move.

The parent re-imports these names, so app.compute_indicators still resolves.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _VADER = _VaderAnalyzer()
except ImportError:          # scoring still works; headline sentiment is skipped
    _VADER = None

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
    Deterministic technical score 0-60. Runs BEFORE the LLM.

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
    News-driven score 0-40. Runs BEFORE the LLM.
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
    # The news contract only ever emits BULLISH/NEUTRAL/BEARISH, so the old
    # CAUTIOUS branch was unreachable and BEARISH fell through to the catch-all
    # worth 1 point. BEARISH now takes the cautious tier it was written for;
    # the catch-all is reserved for a genuinely unrecognised value.
    ms = (market_sentiment or 'NEUTRAL').upper()
    if   ms == 'BULLISH':                mac_pts = 10
    elif ms == 'NEUTRAL':                mac_pts =  6
    elif ms in ('BEARISH', 'CAUTIOUS'):  mac_pts =  3
    else:                                mac_pts =  1

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

def _clean_ohlcv(df):
    """Return aligned, numeric OHLCV rows and discard incomplete Yahoo rows.

    Yahoo can append an in-progress row whose Close/High/Low values are NaN.
    Keeping that row while dropping NaNs from Close alone gives indicators
    unequal indexes and makes every ticker fail the technical screen.
    """
    required = ['High', 'Low', 'Close', 'Volume']
    if df is None or not all(col in df.columns for col in required):
        return None
    clean = df.copy()
    for col in required + (['Open'] if 'Open' in clean.columns else []):
        clean[col] = pd.to_numeric(clean[col], errors='coerce')
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=['High', 'Low', 'Close'])
    if clean.empty:
        return None
    clean['Volume'] = clean['Volume'].fillna(0.0)
    return clean

def compute_indicators(df, spy_return_today=0.0):
    """Compute all technical indicators from OHLCV DataFrame."""
    df = _clean_ohlcv(df)
    if df is None or len(df) < 20:
        return None
    try:
        hi = df['High']
        lo = df['Low']
        cl = df['Close']
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

        # Wilder's smoothing (com=13 == alpha 1/14). The exit rule in
        # screener_portfolio._indicator_exit already uses Wilder; a simple
        # rolling mean here read up to ~12 RSI points differently on the same
        # data, so "don't buy over RSI 75" and "sell over RSI 78" were measuring
        # different things.
        delta     = cl.diff()
        gain      = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss      = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
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
