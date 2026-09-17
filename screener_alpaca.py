"""Alpaca integration: reliable market data + paper-trade execution.

This module is deliberately dependency-free beyond ``requests``/``pandas`` (both
already required by the screener) so it works under restrictive environments and
from GitHub Actions datacenter IPs where Yahoo Finance is routinely throttled.

Two independent capabilities, each gated by its own switch:

* Market data (``data_enabled``): reads daily OHLCV bars from the Alpaca Market
  Data API. Used as the PRIMARY source in ``batch_download`` with yfinance as an
  automatic fallback, so nothing changes when Alpaca keys are absent.
* Paper execution (``trading_enabled``): mirrors the validated paper ledger onto
  an Alpaca (paper) brokerage account. OFF unless ``SCREENER_LIVE_BROKER=1`` so
  the default screener behaviour and the offline test-suite are never affected.

Environment variables
    ALPACA_API_KEY / ALPACA_SECRET_KEY   credentials (never logged)
    ALPACA_DATA_FEED                     'iex' (free, default) or 'sip' (paid)
    ALPACA_PAPER                         '1' (default) paper endpoint, '0' live
    SCREENER_LIVE_BROKER                 '1' enables order submission/reconcile
"""

import os
import time
import threading
import uuid

import numpy as np
import pandas as pd
import requests
from alpaca.trading.stream import TradingStream

_DATA_BASE = 'https://data.alpaca.markets'
_PAPER_TRADE_BASE = 'https://paper-api.alpaca.markets'
_LIVE_TRADE_BASE = 'https://api.alpaca.markets'

_OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 4
_STREAM_THREAD = None
_STREAM_LOCK = threading.Lock()
_TRADE_UPDATES = []
_ORDER_LEDGER = []
ALPACA_STREAM_UPDATES = os.environ.get('ALPACA_STREAM_UPDATES', '0').strip().lower() in ('1', 'true', 'yes', 'on')

_SYMBOLS_PER_REQUEST = 100


def _key():
    return os.environ.get('ALPACA_API_KEY', '').strip()


def _secret():
    return os.environ.get('ALPACA_SECRET_KEY', '').strip()


def data_enabled():
    """True when Alpaca credentials are present (market-data reads allowed)."""
    return bool(_key() and _secret())


def trading_enabled():
    """True only when credentials exist AND live-broker mirroring is opted in."""
    return data_enabled() and os.environ.get('SCREENER_LIVE_BROKER', '').strip() == '1'


def _feed():
    feed = os.environ.get('ALPACA_DATA_FEED', 'iex').strip().lower()
    return feed or 'iex'


def _trade_base():
    return _LIVE_TRADE_BASE if os.environ.get('ALPACA_PAPER', '1').strip() == '0' else _PAPER_TRADE_BASE


def _headers():
    return {'APCA-API-KEY-ID': _key(), 'APCA-API-SECRET-KEY': _secret(),
            'accept': 'application/json'}


def _redact(text):
    """Strip credentials from any string before it can reach a log."""
    detail = str(text)
    for token in (_key(), _secret()):
        if token:
            detail = detail.replace(token, '[redacted]')
    return detail


def _request(method, url, params=None, body=None):
    """HTTP with bounded retries on 429/5xx; returns parsed JSON or None.

    Never raises: a broker/data outage must degrade gracefully, not abort the run.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=_headers(), params=params,
                                    json=body, timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                print('  Alpaca ' + method + ' transport error: ' + type(exc).__name__)
                return None
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == _MAX_RETRIES - 1:
                print('  Alpaca ' + method + ' HTTP ' + str(resp.status_code))
                return None
            time.sleep(min(2 ** attempt, 8))
            continue
        if resp.status_code >= 400:
            print('  Alpaca ' + method + ' HTTP ' + str(resp.status_code) + ': '
                  + _redact(resp.text)[:200])
            return None
        try:
            return resp.json()
        except ValueError:
            return None
    return None


def _get(url, params=None):
    return _request('GET', url, params=params)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _iso(value):
    """Accept a date/datetime/str and return an ISO date string (UTC-safe)."""
    stamp = pd.Timestamp(value)
    return stamp.date().isoformat()


def _bars_to_frame(bars):
    """Convert Alpaca daily bar dicts into a yfinance-shaped OHLCV frame.

    The result matches what ``_clean_ohlcv`` produces: a tz-naive DatetimeIndex
    keyed on the New York session date, numeric Open/High/Low/Close/Volume, and
    no incomplete rows — so ``fresh_bar`` and the safety validators accept it.
    """
    if not bars:
        return None
    index = []
    rows = []
    for bar in bars:
        try:
            stamp = pd.Timestamp(bar['t'])
        except (KeyError, ValueError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize('UTC')
        session_day = stamp.tz_convert('America/New_York').date()
        index.append(pd.Timestamp(session_day))
        rows.append({'Open': bar.get('o'), 'High': bar.get('h'), 'Low': bar.get('l'),
                     'Close': bar.get('c'), 'Volume': bar.get('v', 0)})
    if not rows:
        return None
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    for col in _OHLCV_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=['Open', 'High', 'Low', 'Close'])
    if frame.empty:
        return None
    frame['Volume'] = frame['Volume'].fillna(0.0)
    return frame


def daily_bars(symbols, start, end=None, feed=None):
    """Return ``{symbol: OHLCV DataFrame}`` for the given symbols.

    Missing/failed symbols are simply omitted so the caller can fall back to
    another provider. Never raises on network/credential problems.
    """
    if not data_enabled():
        return {}
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return {}
    feed = (feed or _feed())
    accumulated = {}
    for chunk in _chunks(symbols, _SYMBOLS_PER_REQUEST):
        page_token = None
        while True:
            params = {'symbols': ','.join(chunk), 'timeframe': '1Day',
                      'start': _iso(start), 'limit': 10000, 'adjustment': 'raw',
                      'feed': feed}
            if end is not None:
                params['end'] = _iso(end)
            if page_token:
                params['page_token'] = page_token
            data = _get(_DATA_BASE + '/v2/stocks/bars', params)
            if not data:
                break
            for sym, blist in (data.get('bars') or {}).items():
                accumulated.setdefault(sym, []).extend(blist)
            page_token = data.get('next_page_token')
            if not page_token:
                break
    result = {}
    for sym, blist in accumulated.items():
        frame = _bars_to_frame(blist)
        if frame is not None:
            result[sym] = frame
    return result


# ── Trading (paper) ────────────────────────────────────────────────────────

def get_account():
    """Return the Alpaca account dict, or None on failure."""
    return _get(_trade_base() + '/v2/account')


def list_positions():
    """Return open Alpaca positions as a list of dicts (empty on failure)."""
    data = _get(_trade_base() + '/v2/positions')
    return data if isinstance(data, list) else []


def list_orders(status='all', limit=100):
    data = _get(_trade_base() + '/v2/orders', {'status': status, 'limit': limit})
    return data if isinstance(data, list) else []


def open_order_shares_by_symbol():
    """Map of open-order share deltas by symbol (buy positive, sell negative)."""
    committed = {}
    for order in list_orders(status='open'):
        try:
            symbol = str(order['symbol']).strip().upper()
            side = str(order.get('side', '')).strip().lower()
            qty = int(float(order.get('qty', 0) or 0))
            filled = int(float(order.get('filled_qty', 0) or 0))
        except (KeyError, TypeError, ValueError):
            continue
        remaining = max(0, qty - filled)
        if remaining == 0:
            continue
        if side == 'buy':
            committed[symbol] = committed.get(symbol, 0) + remaining
        elif side == 'sell':
            committed[symbol] = committed.get(symbol, 0) - remaining
    return committed


def effective_shares_by_symbol():
    """Broker positions plus outstanding open-order share deltas."""
    effective = dict(positions_by_symbol())
    for symbol, delta in open_order_shares_by_symbol().items():
        effective[symbol] = effective.get(symbol, 0) + delta
        if effective[symbol] == 0:
            effective.pop(symbol, None)
    return effective


def _client_order_id(symbol, side):
    cleaned = ''.join(ch for ch in str(symbol).upper() if ch.isalnum() or ch in ('-', '_'))[:12] or 'UNK'
    prefix = 'B' if str(side).lower() == 'buy' else 'S'
    return f'lpm-{prefix}-{cleaned}-{int(time.time())}-{uuid.uuid4().hex[:8]}'


def _order_key(snapshot):
    return snapshot.get("client_order_id") or snapshot.get("order_id") or snapshot.get("symbol") or ""


def _order_snapshot(order, source):
    return {
        "source": source,
        "event": str(order.get("status", "") or order.get("event", "") or "").lower(),
        "timestamp": order.get("updated_at") or order.get("submitted_at") or order.get("timestamp"),
        "order_id": str(order.get("id", "") or ""),
        "client_order_id": order.get("client_order_id"),
        "symbol": order.get("symbol"),
        "status": str(order.get("status", "") or "").lower(),
        "filled_qty": str(order.get("filled_qty", "") or ""),
        "filled_avg_price": str(order.get("filled_avg_price", "") or ""),
        "qty": str(order.get("qty", "") or ""),
        "side": str(order.get("side", "") or "").lower(),
    }


def _upsert_order_ledger(snapshot):
    key = _order_key(snapshot)
    if not key:
        return snapshot
    for idx, existing in enumerate(_ORDER_LEDGER):
        if _order_key(existing) == key:
            merged = dict(existing)
            merged.update(snapshot)
            _ORDER_LEDGER[idx] = merged
            return merged
    _ORDER_LEDGER.append(dict(snapshot))
    return snapshot


def order_ledger():
    """Return a copy of the last known order ledger."""
    return [dict(item) for item in _ORDER_LEDGER]


def sync_order_statuses(existing_ledger=None):
    """Poll Alpaca for current order states and merge them into the ledger."""
    if isinstance(existing_ledger, list):
        for item in existing_ledger:
            if isinstance(item, dict):
                _upsert_order_ledger(item)
    for order in list_orders(status="all", limit=500):
        try:
            snapshot = _order_snapshot(order, "poll")
        except Exception:
            continue
        _upsert_order_ledger(snapshot)
    return order_ledger()


def trade_updates():
    """Return a copy of recent Alpaca trade update snapshots."""
    return list(_TRADE_UPDATES)


async def _trade_update_handler(update):
    order = getattr(update, "order", None)
    snapshot = {
        "event": str(getattr(update, "event", "") or "").lower(),
        "timestamp": getattr(update, "timestamp", None).isoformat() if getattr(update, "timestamp", None) else None,
        "order_id": str(getattr(order, "id", "") or ""),
        "client_order_id": getattr(order, "client_order_id", None),
        "symbol": getattr(order, "symbol", None),
        "status": str(getattr(order, "status", "") or "").lower(),
        "filled_qty": str(getattr(order, "filled_qty", "") or ""),
        "filled_avg_price": str(getattr(order, "filled_avg_price", "") or ""),
        "qty": str(getattr(order, "qty", "") or ""),
    }
    _TRADE_UPDATES.append(snapshot)
    del _TRADE_UPDATES[:-50]
    _upsert_order_ledger(snapshot)
    print(f"  Alpaca update: {snapshot['symbol']} {snapshot['event']} status={snapshot['status'] or 'unknown'} qty={snapshot['filled_qty'] or snapshot['qty'] or '?'}")


def _start_trade_updates_stream():
    global _STREAM_THREAD
    if not ALPACA_STREAM_UPDATES or not trading_enabled() or TradingStream is None:
        return False
    with _STREAM_LOCK:
        if _STREAM_THREAD and _STREAM_THREAD.is_alive():
            return True

        def _runner():
            try:
                stream = TradingStream(api_key=_key(), secret_key=_secret(), paper=os.environ.get("ALPACA_PAPER", "1").strip() != "0")
                stream.subscribe_trade_updates(_trade_update_handler)
                print('  Alpaca trade_updates stream started')
                stream.run()
            except Exception as exc:
                print(f'  Alpaca trade_updates stream error: {exc}')

        _STREAM_THREAD = threading.Thread(target=_runner, name='alpaca-trade-updates', daemon=True)
        _STREAM_THREAD.start()
    return True


def submit_market_order(symbol, qty, side):
    """Submit a market DAY order. Returns the order dict or None.

    Submitted after the close, Alpaca accepts the order and queues it for the
    next session open — matching the screener's next-open fill model.
    """
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        return None
    if qty <= 0 or side not in ('buy', 'sell'):
        return None
    _start_trade_updates_stream()
    body = {'symbol': str(symbol).strip().upper(), 'qty': str(qty),
            'side': side, 'type': 'market', 'time_in_force': 'day',
            'client_order_id': _client_order_id(symbol, side)}
    return _request('POST', _trade_base() + '/v2/orders', body=body)


def close_position(symbol):
    """Liquidate an Alpaca position entirely. Returns the order dict or None."""
    return _request('DELETE', _trade_base() + '/v2/positions/' + str(symbol).strip().upper())


def positions_by_symbol():
    """Map of ``{SYMBOL: integer share qty}`` currently held at the broker."""
    held = {}
    for pos in list_positions():
        try:
            held[str(pos['symbol']).strip().upper()] = int(float(pos.get('qty', 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return held


def plan_reconciliation(ledger_shares, broker_shares):
    """Pure desired-state diff: return the buy/sell actions to align the broker.

    ``ledger_shares`` / ``broker_shares`` are ``{SYMBOL: int}`` maps. Returns a
    list of ``(side, symbol, qty)`` tuples. Kept side-effect free so it can be
    unit-tested without any network access.
    """
    actions = []
    symbols = set(ledger_shares) | set(broker_shares)
    for symbol in sorted(symbols):
        want = int(ledger_shares.get(symbol, 0))
        have = int(broker_shares.get(symbol, 0))
        if want > have:
            actions.append(('buy', symbol, want - have))
        elif have > want:
            actions.append(('sell', symbol, have - want))
    return actions
