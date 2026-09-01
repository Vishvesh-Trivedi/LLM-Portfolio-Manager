# --- restored compatibility helpers for the GitHub Action test suite ---
from collections import deque
import threading
import time

_LLM_LAST_CALL = [0.0]
_LLM_REQUEST_TIMESTAMPS = deque()
_LLM_RATE_LOCK = threading.Lock()


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


def _valid_closes(df):
    """Return finite Close values from a Yahoo history frame."""
    if df is None or 'Close' not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df['Close'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


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
        p = float(cl.iloc[-1])
        if not (p > 0):
            return None

        m20 = float(cl.rolling(20).mean().iloc[-1])
        m50 = float(cl.rolling(50).mean().iloc[-1]) if len(cl) >= 50 else p
        m200 = float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else p

        av = float(vo.rolling(30).mean().iloc[-1]) if len(vo) >= 30 else float(vo.mean())
        vr = float(vo.iloc[-1]) / av if av > 0 else 0.0
        dvol_m = round((p * av) / 1e6, 1)

        if len(vo) >= 3:
            v1, v2, v3 = float(vo.iloc[-3]), float(vo.iloc[-2]), float(vo.iloc[-1])
            vol_accel = v3 > v2 > v1
        else:
            vol_accel = False

        delta = cl.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        loss_safe = loss.replace(0, 1e-10)
        rsi_val = float((100 - (100 / (1 + gain / loss_safe))).iloc[-1]) if len(cl) >= 14 else 50.0

        p5 = float(cl.iloc[-5]) if len(cl) >= 5 else p
        p20 = float(cl.iloc[-20]) if len(cl) >= 20 else p

        w52_high = float(cl.rolling(min(252, len(cl))).max().iloc[-1])
        pct_from_52h = round(((p - w52_high) / w52_high) * 100, 1) if w52_high > 0 else 0.0

        prev_cl = cl.shift(1)
        tr = pd.concat([
            hi - lo,
            (hi - prev_cl).abs(),
            (lo - prev_cl).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())
        atr_pct = round((atr / p) * 100, 2) if p > 0 else 0.0

        ema12 = cl.ewm(span=12, adjust=False).mean()
        ema26 = cl.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        sig_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - sig_line
        macd_bull = bool(macd_line.iloc[-1] > sig_line.iloc[-1] and float(macd_hist.iloc[-1]) > 0)

        up_move = hi.diff()
        dn_move = -lo.diff()
        plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=cl.index)
        minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=cl.index)
        atr_w = tr.ewm(alpha=1/14, adjust=False).mean()
        safe_atr = atr_w.replace(0, 1e-10)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / safe_atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / safe_atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        adx_val = float(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]) if len(dx) >= 14 else 0.0

        bb_mid = cl.rolling(20).mean()
        bb_std = cl.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = float((bb_upper - bb_lower).iloc[-1])
        bb_pct_b = float((p - float(bb_lower.iloc[-1])) / bb_range) if bb_range > 0 else 0.5

        obv = (np.sign(cl.diff()) * vo).fillna(0).cumsum()
        obv_trend = float(obv.iloc[-1]) > float(obv.iloc[-10]) if len(obv) >= 10 else False

        mfv = ((cl - lo) - (hi - cl)) / (hi - lo).replace(0, 1e-10) * vo
        cmf_val = (float(mfv.rolling(20).sum().iloc[-1]) / float(vo.rolling(20).sum().replace(0, 1e-10).iloc[-1])) if len(cl) >= 20 else 0.0
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
            hh_hl_val = (h_vals[0] > h_vals[1] > h_vals[2] and l_vals[0] > l_vals[1] > l_vals[2])
        else:
            hh_hl_val = False

        tp = (hi + lo + cl) / 3
        vwap_20 = float((tp * vo).rolling(20).sum().iloc[-1] / vo.rolling(20).sum().replace(0, 1e-10).iloc[-1]) if len(cl) >= 20 else p
        vs_vwap = round(((p - vwap_20) / vwap_20) * 100, 2) if vwap_20 > 0 else 0.0

        if len(cl) >= 2:
            stock_ret_today = ((p - float(cl.iloc[-2])) / float(cl.iloc[-2])) * 100
            rs_vs_spy = round(stock_ret_today - spy_return_today, 2)
        else:
            rs_vs_spy = 0.0

        return {
            'price': round(p, 2),
            'ma20': round(m20, 2),
            'ma50': round(m50, 2),
            'ma200': round(m200, 2),
            'vol_ratio': round(vr, 2),
            'dollar_vol_m': dvol_m,
            'vol_accel': vol_accel,
            'rsi': round(rsi_val, 1),
            'vs_ma20_pct': round(((p - m20) / m20) * 100, 2) if m20 > 0 else 0.0,
            'vs_ma50_pct': round(((p - m50) / m50) * 100, 2) if m50 > 0 else 0.0,
            'vs_ma200_pct': round(((p - m200) / m200) * 100, 2) if m200 > 0 else 0.0,
            'momentum_5d': round(((p - p5) / p5) * 100, 2) if p5 > 0 else 0.0,
            'momentum_20d': round(((p - p20) / p20) * 100, 2) if p20 > 0 else 0.0,
            'pct_from_52h': pct_from_52h,
            'w52_high': round(w52_high, 2),
            'atr': round(atr, 2),
            'atr_pct': atr_pct,
            'macd_bullish': macd_bull,
            'adx': round(adx_val, 1),
            'bb_pct_b': round(bb_pct_b, 2),
            'obv_rising': obv_trend,
            'rs_vs_spy': rs_vs_spy,
            'cmf': cmf_val,
            'stoch_rsi': stoch_rsi_val,
            'hh_hl': hh_hl_val,
            'vs_vwap_pct': vs_vwap,
        }
    except Exception as e:
        if os.environ.get('SCREENER_DEBUG'):
            print(f'  [debug] compute_indicators failed: {type(e).__name__}: {e}')
        return None
