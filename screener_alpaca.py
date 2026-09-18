"""Alpaca integration: reliable market data + paper-trade execution.

This module is deliberately dependency-free beyond ``requests``/``pandas`` (both
already required by the screener) so it works under restrictive environments and
from GitHub Actions datacenter IPs where Yahoo Finance is routinely throttled.

Two independent capabilities, each gated by its own switch:

* Market data (``data_enabled``): reads daily OHLCV bars from the Alpaca Market
  Data API. Used as the PRIMARY source in ``batch_download`` with yfinance as an
  automatic fallback, so nothing changes when Alpaca keys are absent.
* Paper execution (``trading_enabled``): mirrors the validated paper ledger onto
  an Alpaca (paper) brokerage account, and makes Alpaca the source of truth for
  fills, share counts and cash. OFF unless ``SCREENER_LIVE_BROKER`` is set, so
  the default screener behaviour and the offline test-suite are never affected.

Environment variables
    ALPACA_API_KEY / ALPACA_SECRET_KEY   credentials (never logged)
    ALPACA_DATA_FEED                     'iex' (free, default) or 'sip' (paid)
    ALPACA_PAPER                         '1' (default) paper endpoint, '0' live
    SCREENER_LIVE_BROKER                 1/true/yes/on enables order submission
                                         and broker-authoritative reconciliation
"""

import os
import time
import threading
import uuid

import numpy as np
import pandas as pd
import requests
try:
    from alpaca.trading.stream import TradingStream
except Exception:
    TradingStream = None

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
    """True only when credentials exist AND live-broker mirroring is opted in.

    Accepts 1/true/yes/on, matching every other switch in this project. A strict
    '1' compare made SCREENER_LIVE_BROKER=true read as OFF, which is the single
    most expensive way to misconfigure this system: the run would not read the
    Alpaca account and would sell holdings its ledger had lost track of.
    """
    return data_enabled() and os.environ.get(
        'SCREENER_LIVE_BROKER', '').strip().lower() in ('1', 'true', 'yes', 'on')


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


def list_orders(status='all', limit=100, nested=False):
    params = {'status': status, 'limit': limit}
    if nested:
        # Return OCO/bracket legs nested under their parent instead of as
        # separate rows, so a single top-level check can identify them.
        params['nested'] = 'true'
    data = _get(_trade_base() + '/v2/orders', params)
    return data if isinstance(data, list) else []


# ── Authoritative reads ────────────────────────────────────────────────────
# The helpers above flatten a transport failure into an empty list, which is
# indistinguishable from a genuinely empty account. That is safe for the
# best-effort mirror but NOT for broker-authoritative reconciliation: rewriting
# a ledger from a failed read would silently delete real positions. The fetch_*
# helpers below return None on failure and only ever return a container when
# Alpaca actually answered.

def _number(value, default=None):
    try:
        if value is None or value == '':
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _whole(value, default=0):
    number = _number(value)
    return default if number is None else int(round(number))


def fetch_account():
    """Account dict, or None when Alpaca could not be read."""
    data = _get(_trade_base() + '/v2/account')
    return data if isinstance(data, dict) else None


def fetch_positions():
    """Open positions list, or None when Alpaca could not be read."""
    data = _get(_trade_base() + '/v2/positions')
    return data if isinstance(data, list) else None


def fetch_orders(status='all', limit=500):
    """Recent orders list, or None when Alpaca could not be read."""
    data = _get(_trade_base() + '/v2/orders',
                {'status': status, 'limit': limit, 'direction': 'desc'})
    return data if isinstance(data, list) else None


def normalize_position(position):
    """Reduce an Alpaca position payload to the fields reconciliation needs."""
    symbol = str(position.get('symbol', '') or '').strip().upper()
    if not symbol:
        return None
    qty = _whole(position.get('qty'))
    if qty == 0:
        return None
    return {
        'symbol': symbol,
        'qty': qty,
        'avg_entry_price': _number(position.get('avg_entry_price')),
        'market_value': _number(position.get('market_value')),
        'current_price': _number(position.get('current_price')),
    }


def normalize_order(order):
    """Reduce an Alpaca order payload to the fields reconciliation needs."""
    symbol = str(order.get('symbol', '') or '').strip().upper()
    if not symbol:
        return None
    return {
        'order_id': str(order.get('id', '') or ''),
        'client_order_id': str(order.get('client_order_id', '') or ''),
        'symbol': symbol,
        'side': str(order.get('side', '') or '').strip().lower(),
        'status': str(order.get('status', '') or '').strip().lower(),
        'qty': _whole(order.get('qty')),
        'filled_qty': _whole(order.get('filled_qty')),
        'filled_avg_price': _number(order.get('filled_avg_price')),
        'submitted_at': order.get('submitted_at'),
        'filled_at': order.get('filled_at'),
        'updated_at': order.get('updated_at'),
        'canceled_at': order.get('canceled_at'),
        'expired_at': order.get('expired_at'),
    }


def broker_snapshot(order_limit=500):
    """Authoritative account/positions/orders read for reconciliation.

    ``ok`` is True only when ALL THREE reads succeeded. A partial read is
    reported as a failure rather than a half-truth, because a caller that
    rewrites a ledger from it would corrupt real holdings. Never raises.
    """
    try:
        if not trading_enabled():
            return {'ok': False, 'error': 'trading_disabled', 'positions': {}, 'orders': []}
        account = fetch_account()
        positions = fetch_positions()
        orders = fetch_orders(limit=order_limit)
        missing = [name for name, value in (('account', account), ('positions', positions),
                                            ('orders', orders)) if value is None]
        if missing:
            return {'ok': False, 'error': 'unreadable: ' + ', '.join(missing),
                    'positions': {}, 'orders': []}
        held = {}
        for raw in positions:
            normalized = normalize_position(raw) if isinstance(raw, dict) else None
            if normalized is not None:
                held[normalized['symbol']] = normalized
        recent = []
        for raw in orders:
            normalized = normalize_order(raw) if isinstance(raw, dict) else None
            if normalized is not None:
                recent.append(normalized)
        return {
            'ok': True,
            'cash': _number(account.get('cash')),
            'equity': _number(account.get('equity')),
            'buying_power': _number(account.get('buying_power')),
            'account_blocked': bool(account.get('account_blocked')
                                    or account.get('trading_blocked')),
            'positions': held,
            'orders': recent,
        }
    except Exception as exc:  # defensive: a read must never abort the run
        return {'ok': False, 'error': 'snapshot error: ' + type(exc).__name__,
                'positions': {}, 'orders': []}


_PROTECTIVE_CLASSES = frozenset({'oco', 'bracket', 'oto'})


def is_protective(order):
    """True for a resting stop/target order rather than a sizing instruction.

    A protective sell sits open for the whole life of a position. Counting it as
    a negative share delta makes a fully protected holding look like zero shares
    held, and the desired-state mirror then BUYS the position a second time.
    """
    if str(order.get('order_class', '') or '').strip().lower() in _PROTECTIVE_CLASSES:
        return True
    if order.get('legs'):
        return True
    parsed = parse_client_order_id(order.get('client_order_id'))
    return bool(parsed and parsed.get('protective'))


def open_order_shares_by_symbol():
    """Map of open-order share deltas by symbol (buy positive, sell negative).

    Resting protective orders are excluded: they express where to exit, not how
    much to hold.
    """
    committed = {}
    for order in list_orders(status='open', nested=True):
        if is_protective(order):
            continue
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


_CLIENT_ORDER_PREFIX = 'lpm'


def _client_order_id(symbol, side, ref='', protective=False):
    """Build a client_order_id that carries the ledger identity of the order.

    ``ref`` is the ledger's pending-order id (buys) or the position trade_id
    (sells). Embedding it makes broker→ledger matching exact instead of guessing
    from symbol and side, which is ambiguous as soon as a symbol is re-entered.
    Alpaca allows 128 characters; this stays well under that.

    ``protective`` marks a resting stop/target order with a 'P' prefix. Those
    must never be read as an intent to change the position size - see
    open_order_shares_by_symbol.
    """
    cleaned = ''.join(ch for ch in str(symbol).upper() if ch.isalnum() or ch in ('-', '_'))[:12] or 'UNK'
    prefix = 'P' if protective else ('B' if str(side).lower() == 'buy' else 'S')
    tag = ''.join(ch for ch in str(ref or '') if ch.isalnum())[:32]
    if not tag:
        tag = f'{int(time.time())}{uuid.uuid4().hex[:8]}'
    return f'{_CLIENT_ORDER_PREFIX}-{prefix}-{cleaned}-{tag}'


def parse_client_order_id(client_order_id):
    """Return ``{'side', 'symbol', 'ref'}`` for our own ids, else None.

    Orders placed by hand or by an older build simply do not parse; callers fall
    back to symbol/side matching for those rather than assuming ownership.
    """
    parts = str(client_order_id or '').split('-')
    if len(parts) != 4 or parts[0] != _CLIENT_ORDER_PREFIX or parts[1] not in ('B', 'S', 'P'):
        return None
    return {'side': 'buy' if parts[1] == 'B' else 'sell',
            'symbol': parts[2].upper(), 'ref': parts[3],
            'protective': parts[1] == 'P'}


def ledger_ref(order):
    """Ledger id embedded in a (normalized or raw) order, or '' when absent."""
    parsed = parse_client_order_id(order.get('client_order_id'))
    return parsed['ref'] if parsed else ''


def _order_key(snapshot):
    return snapshot.get('client_order_id') or snapshot.get('order_id') or snapshot.get('symbol') or ''


def _order_snapshot(order, source):
    return {
        'source': source,
        'event': str(order.get('status', '') or order.get('event', '') or '').lower(),
        'timestamp': order.get('updated_at') or order.get('submitted_at') or order.get('timestamp'),
        'order_id': str(order.get('id', '') or ''),
        'client_order_id': order.get('client_order_id'),
        'symbol': order.get('symbol'),
        'status': str(order.get('status', '') or '').lower(),
        'filled_qty': str(order.get('filled_qty', '') or ''),
        'filled_avg_price': str(order.get('filled_avg_price', '') or ''),
        'qty': str(order.get('qty', '') or ''),
        'side': str(order.get('side', '') or '').lower(),
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
    for order in list_orders(status='all', limit=500):
        try:
            snapshot = _order_snapshot(order, 'poll')
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


def submit_market_order(symbol, qty, side, ref=''):
    """Submit a market DAY order. Returns the order dict or None.

    Submitted after the close, Alpaca accepts the order and queues it for the
    next session open — matching the screener's next-open fill model. ``ref``
    carries the ledger id so the next run can match this order back to the
    record that requested it.
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
            'client_order_id': _client_order_id(symbol, side, ref)}
    order = _request('POST', _trade_base() + '/v2/orders', body=body)
    if isinstance(order, dict):
        try:
            _upsert_order_ledger(_order_snapshot(order, 'submit'))
        except Exception:
            pass
    return order


def _price(value):
    return f'{round(float(value), 2):.2f}'


def submit_protective_oco(symbol, qty, stop_price, limit_price, ref=''):
    """Rest a GTC one-cancels-other stop-loss / take-profit on a held long.

    Whichever leg triggers cancels the other. Good-till-cancelled so it protects
    the position continuously, not only when this program happens to be running.
    Returns the order dict or None; never raises.
    """
    try:
        qty = int(qty)
        stop = round(float(stop_price), 2)
        limit = round(float(limit_price), 2)
    except (TypeError, ValueError):
        return None
    if qty <= 0 or not 0 < stop < limit:
        return None
    body = {
        'symbol': str(symbol).strip().upper(), 'qty': str(qty), 'side': 'sell',
        'type': 'limit', 'time_in_force': 'gtc', 'order_class': 'oco',
        'limit_price': _price(limit),
        'take_profit': {'limit_price': _price(limit)},
        'stop_loss': {'stop_price': _price(stop)},
        'client_order_id': _client_order_id(symbol, 'sell', ref, protective=True),
    }
    return _request('POST', _trade_base() + '/v2/orders', body=body)


def protective_orders_by_symbol():
    """Resting protection per symbol: ``{SYMBOL: {order_id, qty, stop, limit}}``."""
    found = {}
    for order in list_orders(status='open', limit=500, nested=True):
        if not is_protective(order):
            continue
        symbol = str(order.get('symbol', '') or '').strip().upper()
        if not symbol:
            continue
        stop = limit = None
        for leg in list(order.get('legs') or []) + [order]:
            if leg.get('stop_price') is not None:
                stop = _number(leg.get('stop_price'), stop)
            elif leg.get('limit_price') is not None:
                limit = _number(leg.get('limit_price'), limit)
        found[symbol] = {'order_id': str(order.get('id', '') or ''),
                         'qty': _whole(order.get('qty')),
                         'stop': stop, 'limit': limit}
    return found


def cancel_order(order_id):
    """Cancel one order. True when Alpaca accepted it."""
    order_id = str(order_id or '').strip()
    if not order_id:
        return False
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.delete(_trade_base() + '/v2/orders/' + order_id,
                                   headers=_headers(), timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException:
            if attempt == _MAX_RETRIES - 1:
                return False
            time.sleep(min(2 ** attempt, 8))
            continue
        # 204 accepted; 404 means it is already gone, which is the same outcome.
        return resp.status_code in (200, 204, 404)
    return False


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
