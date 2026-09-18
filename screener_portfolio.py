"""Canonical, session-based paper ledger (Python 3.11+).

All entry points take the parent module as ``app``; this module never imports
the main program or a network client. Required hooks are PORTFOLIO_JSON,
STARTING_CAPITAL, _session_date(), _sharesies_fee(), _ORDER_REASON (one-item
list), _degrade(reason), and yf.Ticker. The parent's NYSE gate must supply a
completed session. Optional _CFG_* settings retain the parent's names.

open_position/close_position mutate only after approval and do not save.
queue_position and update_portfolio_prices save atomically before publishing.
Optional app._ORDER_CONTEXT is a dict of signal/learning metadata, hold_sessions,
trade_id and filled_at_open. Internal pending fills set this temporarily and
restore it. A pending order's id is its eventual trade_id; amount_usd includes
fees and shares caps its eventual quantity. No cash or fee allowance is reserved.

Execution sessions come from _next_session_date(date), or weekdays excluding
_us_market_holiday(date) when supplied. Without either hook the fallback knows
weekends only; the parent owns exchange holidays and exceptional closures.
Internal replay uses a scoped _CLOSE_CONTEXT and temporarily dates the fee hook
to the simulated exit session. These hooks require a serial runner, not concurrent
calls on the same app. Fees use the runner's available allowance/FX estimates,
not reconstructed historical broker statements. Legacy cost bases stay estimates.

Unadjusted OHLC is intentional: actual share/cost records must not be mixed
with retrospectively adjusted prices. Corporate-action reconciliation is a
separate responsibility. CSVs are reports, never execution instructions.
"""

import copy
import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd

from screener_safety import (
    atomic_csv, atomic_json, evaluate_horizon, finite_number, fresh_bar,
    mechanical_exit, plan_order, validate_portfolio,
    _sector as _canonical_sector,
)


__all__ = [
    'load_portfolio', 'save_portfolio', 'update_portfolio_prices',
    'open_position', 'close_position', 'reconcile_closed_picks',
    'load_performance_history', 'update_results', 'queue_position',
    'apply_broker_state', 'broker_authoritative',
]

_LEARNING_FIELDS = (
    'source', 'reasoning', 'confidence', 'rsi', 'vix', 'vix_regime',
    'qqq_trend', 'earnings_days', 'congress', 'insider', 'position_size_pct',
    'rr', 'tech_score', 'news_score',
)
_LEARNING_ALIASES = {
    'earnings_days': 'earnings_days_away',
    'congress': 'congress_label', 'insider': 'insider_label',
}
_CLOSE_TOKEN = object()


class _CorporateActionReview(ValueError):
    """Raw prices cannot safely update unreconciled share/cost records."""


def _learning_metadata(record):
    values = {key: copy.deepcopy(record[key]) for key in _LEARNING_FIELDS if key in record}
    for key, alias in _LEARNING_ALIASES.items():
        if key not in values and alias in record:
            values[key] = copy.deepcopy(record[alias])
    return values


def _reason(app, text):
    app._ORDER_REASON[0] = text


def _positive(value, name):
    value = finite_number(value, name, minimum=0)
    if value == 0:
        raise ValueError(name + ' must be positive')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value.strip()


def _date(value):
    if not isinstance(value, (str, date, datetime)) or value == '':
        raise ValueError('session date must be a date or ISO date string')
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('missing session date')
    return stamp.date()


def _session(app):
    return _date(app._session_date()).isoformat()


def expected_execution_session(app, signal):
    """Return the intended next session, independently of ticker availability."""
    signal = _date(signal)
    next_session = getattr(app, '_next_session_date', None)
    if next_session is not None:
        expected = _date(next_session(signal))
        if expected <= signal:
            raise ValueError('next session must be after signal date')
        return expected.isoformat()
    holiday = getattr(app, '_us_market_holiday', None)
    for offset in range(1, 367):
        expected = signal + timedelta(days=offset)
        if expected.weekday() < 5 and (holiday is None or not holiday(expected)):
            return expected.isoformat()
    raise ValueError('calendar has no next session within a year')


def _sessions_after(app, start, asof):
    current, cutoff = _date(start).isoformat(), _date(asof).isoformat()
    sessions = []
    while current < cutoff:
        current = expected_execution_session(app, current)
        if current <= cutoff:
            sessions.append(current)
    return sessions


def _hold(value):
    return max(1, min(30, int(finite_number(value, 'hold_sessions'))))


def _price_offset(price, distance, subtract=False):
    """Avoid binary subtraction noise changing an exact 1.5 reward/risk ratio."""
    base = Decimal(str(finite_number(price, 'price')))
    offset = Decimal(str(finite_number(distance, 'distance')))
    return float(base - offset if subtract else base + offset)


def _result(pct):
    return 'Win' if pct > 0 else 'Loss' if pct < 0 else 'Neutral'


def _net_pct(trade):
    # These are legacy *net* fields, not the CSV's benchmark-relative outcome.
    for key in ('net_realized_pct', 'realized_pnl_pct', 'realized_pct'):
        if key in trade:
            return finite_number(trade[key], key)
    pnl = trade['realized_pnl'] if 'realized_pnl' in trade else trade['pnl']
    return finite_number(pnl / trade['cost_basis'] * 100, 'net realized percent')


def _validated(pf):
    """Validate a private copy, including the schema extensions safety lacks."""
    work = copy.deepcopy(pf)
    validate_portfolio(work)
    work.setdefault('pending_orders', [])
    work.setdefault('processed_sessions', [])
    for key in ('pending_orders', 'processed_sessions'):
        if not isinstance(work[key], list):
            raise ValueError(key + ' must be a list')
    dates = [_date(day).isoformat() for day in work['processed_sessions']]
    if len(set(dates)) != len(dates):
        raise ValueError('duplicate processed session')
    tickers = set()
    ids = {p['trade_id'] for p in work['positions'] + work['closed_trades']}
    for pos in work['positions']:
        ticker = pos['ticker'].strip().upper()
        if ticker in tickers:
            raise ValueError('duplicate open ticker')
        tickers.add(ticker)
    for trade in work['positions'] + work['closed_trades']:
        _date(trade['entry_date'])
        for key in ('signal_date', 'exit_date', 'close_date', 'last_evaluated_session',
                    'execution_session'):
            if key in trade:
                _date(trade[key])
        # Label old accounting without changing cash, shares or historical cost.
        trade.setdefault('cost_basis_basis', 'legacy_estimate')
        for key in ('stop_price', 'target_price', 'current_price'):
            if trade.get(key) is not None:
                _positive(trade[key], key)
        for key in ('atr_at_entry', 'brokerage_in', 'brokerage_out'):
            if key in trade:
                finite_number(trade[key], key, minimum=0)
        if 'hold_sessions' in trade:
            finite_number(trade['hold_sessions'], 'hold_sessions', 1, 30)
            if not isinstance(trade['hold_sessions'], int):
                raise ValueError('hold_sessions must be an integer')
        if 'filled_at_open' in trade and not isinstance(trade['filled_at_open'], bool):
            raise ValueError('filled_at_open must be a boolean')
    pending_dates = set()
    for order in work['pending_orders']:
        if not isinstance(order, dict):
            raise ValueError('pending order must be a dict')
        oid = _text(order['id'], 'pending id')
        if oid in ids:
            raise ValueError('duplicate pending/trade id')
        ids.add(oid)
        if order.get('trade_id', oid) != oid:
            raise ValueError('pending trade_id must equal id')
        ticker = _text(order['ticker'], 'pending ticker').upper()
        if ticker in tickers:
            raise ValueError('duplicate pending/open ticker')
        tickers.add(ticker)
        signal = _date(order['signal_date']).isoformat()
        if ('execution_session' in order
                and _date(order['execution_session']) <= _date(signal)):
            raise ValueError('execution_session must be after signal_date')
        if signal in pending_dates:
            raise ValueError('duplicate pending signal session')
        pending_dates.add(signal)
        entry = _positive(order['estimated_entry'], 'estimated_entry')
        distance = _positive(order['stop_distance'], 'stop_distance')
        reward = _positive(order['target_distance'], 'target_distance')
        if distance >= entry or reward < 1.5 * distance - 1e-10:
            raise ValueError('invalid pending stop/reward distances')
        _positive(order['amount_usd'], 'amount_usd')
        finite_number(order['shares'], 'pending shares', minimum=1)
        if not isinstance(order['shares'], int):
            raise ValueError('pending shares must be an integer')
        _text(order['sector'], 'pending sector')
        finite_number(order['atr'], 'atr', minimum=0)
        if _hold(order['hold_sessions']) != order['hold_sessions']:
            raise ValueError('invalid pending hold_sessions')
    for trade in work['closed_trades']:
        _net_pct(trade)
    work.setdefault('total_realized_pnl', sum(
        t['realized_pnl'] if 'realized_pnl' in t else t['pnl']
        for t in work['closed_trades']))
    finite_number(work['total_realized_pnl'], 'total_realized_pnl')
    finite_number(work.get('equity_peak', 0), 'equity_peak', minimum=0)
    # Also reject nonfinite values in unknown metadata instead of emitting NaN JSON.
    json.dumps(work, allow_nan=False)
    return work


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_ledger(path):
    with open(path, encoding='utf-8') as stream:
        return _validated(json.load(stream, object_pairs_hook=_unique_object))


def _publish(pf, work):
    pf.clear()
    pf.update(work)
    return pf


def load_portfolio(app):
    """Read only the canonical path; a corrupt existing file is a fatal error."""
    path = Path(app.PORTFOLIO_JSON)
    try:
        return _read_ledger(path)
    except FileNotFoundError:
        if os.path.lexists(path):
            raise
    capital = _positive(app.STARTING_CAPITAL, 'starting_capital')
    pf = {
        'cash': capital, 'starting_capital': capital, 'positions': [],
        'closed_trades': [], 'pending_orders': [], 'processed_sessions': [],
        'total_realized_pnl': 0.0, 'equity_peak': capital,
        'created': _session(app),
    }
    save_portfolio(app, pf)
    return pf


def save_portfolio(app, pf):
    """Atomic strict JSON; never lower a previously persisted equity peak."""
    work = _validated(pf)
    path = Path(app.PORTFOLIO_JSON)
    old_peak = 0
    if os.path.lexists(path):
        previous = _read_ledger(path)  # A save cannot silently reset corruption.
        old_peak = previous.get('equity_peak', previous['starting_capital'])
    equity = work['cash'] + sum(
        p.get('current_value', p['cost_basis']) for p in work['positions'])
    work['equity_peak'] = max(work['starting_capital'], old_peak,
                              work.get('equity_peak', 0), equity)
    work['last_updated'] = _session(app)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, work)
    return _publish(pf, work)


def _rate(pf, supplied):
    rate = supplied if supplied is not None else pf.get('last_nzdusd_rate')
    if rate is not None:
        _positive(rate, 'nzdusd_rate')
    return rate


def _fee_quote(app, pf, rate):
    def quote(notional):
        # Each side sees an independent snapshot, including month/allowance state.
        return max(finite_number(app._sharesies_fee(
            notional, copy.deepcopy(pf), nzdusd_rate=rate, side=side),
            side + ' fee', minimum=0) for side in ('buy', 'sell'))
    return quote


def _plan(app, pf, ticker, entry, amount, stop, target, sector, quote):
    return plan_order(
        pf, ticker, entry, amount, stop, target, sector, quote,
        max_positions=getattr(app, '_CFG_MAX_POSITIONS', 5),
        cash_floor=getattr(app, '_CFG_MIN_CASH_FLOOR', 500.0))


def _context(app):
    context = getattr(app, '_ORDER_CONTEXT', {})
    if not isinstance(context, dict):
        raise ValueError('_ORDER_CONTEXT must be a dict')
    return copy.deepcopy(context)


@contextmanager
def _fill_context(app, metadata):
    missing = object()
    previous = getattr(app, '_ORDER_CONTEXT', missing)
    app._ORDER_CONTEXT = metadata
    try:
        yield
    finally:
        if previous is missing:
            delattr(app, '_ORDER_CONTEXT')
        else:
            app._ORDER_CONTEXT = previous


@contextmanager
def _close_context(app, exit_date):
    """Private replay capability; restore even when staging/fees fail."""
    missing = object()
    previous = getattr(app, '_CLOSE_CONTEXT', missing)
    app._CLOSE_CONTEXT = {'exit_date': exit_date, '_token': _CLOSE_TOKEN}
    try:
        yield
    finally:
        if previous is missing:
            delattr(app, '_CLOSE_CONTEXT')
        else:
            app._CLOSE_CONTEXT = previous


@contextmanager
def _fee_session(app, exit_date):
    # The parent fee hook reads _session_date for its monthly allowance bucket.
    # Only that synchronous call sees this override; never patch the run globally.
    previous = app._session_date
    app._session_date = lambda: exit_date
    try:
        yield
    finally:
        app._session_date = previous


def open_position(app, pf, ticker, entry_price, amount_usd, stop, target,
                  sector='', atr=0, nzdusd_rate=None):
    """Fill a safety-approved order, charging the committed buy allowance once."""
    try:
        work = _validated(pf)
        ticker = _text(ticker, 'ticker').upper()
        finite_number(atr, 'atr', minimum=0)
        rate = _rate(work, nzdusd_rate)
        context = _context(app)
        hold = _hold(context.get('hold_sessions', getattr(app, '_CFG_HOLD_DAYS', 10)))
        fill_date = _session(app)
        if context.get('signal_date') is not None:
            signal = _date(context['signal_date']).isoformat()
            if signal > fill_date:
                raise ValueError('signal date is after fill date')
        quote = _fee_quote(app, work, rate)
        # A gap down must not increase the quantity authorized at signal time.
        cap = context.get('max_shares')
        if cap is not None:
            finite_number(cap, 'max_shares', minimum=1)
            if not isinstance(cap, int):
                raise ValueError('max_shares must be an integer')
            _positive(entry_price, 'entry_price')
            finite_number(amount_usd, 'amount_usd', minimum=0)
            amount_usd = min(amount_usd, cap * entry_price + quote(round(cap * entry_price, 2)))
        plan = _plan(app, work, ticker, entry_price, amount_usd, stop, target, sector, quote)
        if cap is not None and plan['shares'] > cap:
            raise ValueError('planned quantity exceeds pending quantity')
        trade_id = _text(context.get('trade_id', str(uuid4())), 'trade_id')
        if any(p['trade_id'] == trade_id for p in work['positions'] + work['closed_trades']):
            raise ValueError('duplicate trade_id')
        pos = _learning_metadata(context)
        if context.get('signal_date') is not None:
            pos['signal_date'] = signal
        if 'execution_session' in context:
            pos['execution_session'] = _date(context['execution_session']).isoformat()
            if pos['execution_session'] != fill_date:
                raise ValueError('fill must match execution_session')
        pos.update({
            'trade_id': trade_id, 'ticker': ticker, 'shares': plan['shares'],
            'entry_price': entry_price, 'entry_date': fill_date, 'fill_date': fill_date,
            'stop_price': stop, 'target_price': target, 'sector': sector,
            'current_price': entry_price, 'current_value': plan['stock_cost'],
            'hold_days': 0, 'held_sessions': 0, 'hold_sessions': hold,
            'high_watermark': entry_price, 'atr_at_entry': atr,
            'filled_at_open': context.get('filled_at_open', False),
            'cost_basis_basis': ('paper_open_fill_plus_fees' if context.get('filled_at_open', False)
                                 else 'after_close_estimate'),
            'quote_stale': False,
        })
        # Stage the actual fee: a fee error or postcondition failure cannot spend
        # cash or monthly coverage in the caller's portfolio.
        fee = finite_number(app._sharesies_fee(
            plan['stock_cost'], work, nzdusd_rate=rate, side='buy'), 'buy fee', minimum=0)
        debit = round(plan['stock_cost'] + fee, 2)
        if Decimal(str(plan['stock_cost'])) + Decimal(str(fee)) > Decimal(str(plan['total_cost'])):
            raise ValueError('actual buy fee exceeds approved conservative quote')
        pos.update({'brokerage_in': round(fee, 2), 'cost_basis': debit,
                    'unrealized_pnl': round(plan['stock_cost'] - debit, 2),
                    'unrealized_pnl_pct': (plan['stock_cost'] / debit - 1) * 100})
        work['cash'] = round(work['cash'] - debit, 2)
        work['positions'].append(pos)
        work = _validated(work)
    except Exception as exc:
        _reason(app, 'Order rejected: ' + str(exc))
        return pf
    _reason(app, 'Filled ' + ticker + ': ' + str(plan['shares']) + ' shares')
    return _publish(pf, work)


def close_position(app, pf, ticker, exit_price, reason='hold_period', nzdusd_rate=None):
    """Close only a ledger position; preserve identity, metadata and entry dates."""
    try:
        work = _validated(pf)
        ticker = _text(ticker, 'ticker').upper()
        pos = next((p for p in work['positions'] if p['ticker'].strip().upper() == ticker), None)
        if pos is None:
            raise ValueError('no open position for ' + ticker)
        _positive(exit_price, 'exit_price')
        rate = _rate(work, nzdusd_rate)
        asof = _session(app)
        context = getattr(app, '_CLOSE_CONTEXT', None)
        if context is not None:
            if not isinstance(context, dict) or context.get('_token') is not _CLOSE_TOKEN:
                raise ValueError('_CLOSE_CONTEXT is internal-only')
            exit_date = _date(context['exit_date']).isoformat()
        else:
            exit_date = asof
        if exit_date > asof:
            raise ValueError('exit date is after current as-of session')
        if exit_date < _date(pos['entry_date']).isoformat():
            raise ValueError('exit date precedes entry date')
        gross = round(exit_price * pos['shares'], 2)
        with _fee_session(app, exit_date):
            fee = finite_number(app._sharesies_fee(
                gross, work, nzdusd_rate=rate, side='sell'), 'sell fee', minimum=0)
        net = round(gross - fee, 2)
        if net < 0:
            raise ValueError('sell fee exceeds proceeds')
        pnl = round(net - pos['cost_basis'], 2)
        pct = pnl / pos['cost_basis'] * 100
        closed = copy.deepcopy(pos)
        closed.update({
            'exit_price': exit_price, 'exit_date': exit_date, 'exit_value': net,
            'brokerage_out': round(fee, 2), 'realized_pnl': pnl,
            'realized_pnl_pct': pct, 'realized_pct': pct, 'net_realized_pct': pct,
            'reason': reason, 'result': _result(pct), 'Result': _result(pct),
            'outcome_basis': 'net_realized', 'last_evaluated_session': exit_date,
            'fee_basis': 'runner_estimate_at_exit_session',
        })
        if exit_date < asof:
            closed['replayed_asof'] = asof
        work['closed_trades'].append(closed)
        work['positions'].remove(pos)
        work['cash'] = round(work['cash'] + net, 2)
        work['total_realized_pnl'] = round(work['total_realized_pnl'] + pnl, 2)
        work = _validated(work)
    except Exception as exc:
        _reason(app, 'Close rejected: ' + str(exc))
        return pf
    _reason(app, 'Closed ' + ticker + ': ' + str(reason))
    return _publish(pf, work)


def queue_position(app, pf, pick, entry, stop, target, candidate):
    """Persist a next-session order, never a pretend after-close execution."""
    try:
        work = _validated(pf)
        signal = _session(app)
        processed = {_date(day).isoformat() for day in work['processed_sessions']}
        records = work['pending_orders'] + work['positions'] + work['closed_trades']
        if signal in processed or any(
            _date(p['signal_date']).isoformat() == signal for p in records if 'signal_date' in p):
            raise ValueError('session already has a decision/order')
        pct = finite_number(pick.get('position_size_pct'), 'position_size_pct', 0, 100)
        if pct == 0:
            raise ValueError('position_size_pct must be positive; no minimum-size uplift')
        ticker = _text(pick['ticker'], 'ticker').upper()
        if any(p['ticker'].strip().upper() == ticker for p in work['pending_orders']):
            raise ValueError('ticker already has a pending order')
        sector = candidate.get('sector', '')  # Do not trust a hallucinated pick sector.
        atr = finite_number(candidate.get('atr', 0), 'atr', minimum=0)
        amount = work['cash'] * pct / 100  # Absolute requested cash budget, including fees.
        rate = _rate(work, None)
        plan = _plan(app, work, ticker, entry, amount, stop, target, sector,
                     _fee_quote(app, work, rate))
        metadata = _context(app)
        for source in (candidate, pick):
            metadata.update(_learning_metadata(source))
        hold = _hold(pick.get('hold_sessions', metadata.get(
            'hold_sessions', getattr(app, '_CFG_HOLD_DAYS', 10))))
        oid = str(uuid4())
        order = _learning_metadata(metadata)
        order.update({
            'id': oid, 'trade_id': oid, 'ticker': ticker, 'signal_date': signal,
            'execution_session': expected_execution_session(app, signal),
            'estimated_entry': entry, 'stop_distance': _price_offset(entry, stop, subtract=True),
            'target_distance': _price_offset(target, entry, subtract=True), 'sector': sector, 'atr': atr,
            'amount_usd': amount, 'shares': plan['shares'], 'hold_sessions': hold,
            'source': metadata.get('source', ''), 'reasoning': metadata.get('reasoning', ''),
            'status': 'PENDING',
        })
        work['pending_orders'].append(order)
        work = _validated(work)
    except Exception as exc:
        _reason(app, 'Order not queued: ' + str(exc))
        return False
    save_portfolio(app, work)
    _publish(pf, work)
    _reason(app, 'Queued ' + ticker + ' for the next session Open; not filled')
    return True


# ── Broker-authoritative reconciliation ────────────────────────────────────
# In broker mode the Alpaca account is the source of truth: the ledger does not
# simulate a fill at the opening print, it books what actually executed. These
# writers deliberately bypass plan_order — its risk caps decide whether an order
# MAY be placed, and re-running them here would refuse to record a trade that
# has already happened in the real account. Cash is never adjusted arithmetically
# either; the 'set_cash' action carries the broker's own balance.


def broker_authoritative(app):
    """True when the parent has opted this run into broker-authoritative mode."""
    hook = getattr(app, '_broker_authoritative', None)
    try:
        return bool(hook()) if callable(hook) else bool(hook)
    except Exception:
        return False


def _find(records, key, value):
    return next((r for r in records if r.get(key) == value), None)


def _apply_fill_pending(app, work, action):
    order = _find(work['pending_orders'], 'id', action['order_id'])
    if order is None:
        raise ValueError('pending order not found: ' + str(action['order_id']))
    shares = action['shares']
    if not isinstance(shares, int) or shares < 1:
        raise ValueError('broker fill quantity must be a positive integer')
    entry = _positive(action['price'], 'fill price')
    work['pending_orders'].remove(order)
    stop = _price_offset(entry, order['stop_distance'], subtract=True)
    target = _price_offset(entry, order['target_distance'])
    if not 0 < stop < entry < target:
        raise ValueError('broker fill price invalidates the stored stop/target')
    cost = round(entry * shares, 2)
    pos = _learning_metadata(order)
    pos.update({
        'trade_id': order['id'], 'ticker': _text(order['ticker'], 'ticker').upper(),
        'shares': shares, 'entry_price': entry, 'entry_date': action['session'],
        'fill_date': action['session'], 'signal_date': order['signal_date'],
        'execution_session': action['session'],
        'stop_price': stop, 'target_price': target, 'sector': order['sector'],
        'current_price': entry, 'current_value': cost, 'cost_basis': cost,
        'brokerage_in': 0.0, 'unrealized_pnl': 0.0, 'unrealized_pnl_pct': 0.0,
        'hold_days': 0, 'held_sessions': 0, 'hold_sessions': order['hold_sessions'],
        'high_watermark': entry, 'atr_at_entry': order['atr'],
        'filled_at_open': True, 'quote_stale': False,
        'cost_basis_basis': 'broker_confirmed_fill',
        'broker_order_id': action.get('broker_order_id', ''),
        'broker_partial_fill': bool(action.get('partial')),
        'requested_shares': order['shares'],
    })
    work['positions'].append(pos)
    return ('Filled ' + pos['ticker'] + ': ' + str(shares) + ' @ ' + str(entry)
            + ' (broker confirmed)')


def _apply_expire_pending(app, work, action):
    order = _find(work['pending_orders'], 'id', action['order_id'])
    if order is None:
        raise ValueError('pending order not found: ' + str(action['order_id']))
    _expire(app, work, order, action['reason'])
    return 'Expired ' + str(action['symbol']) + ': ' + action['reason']


def _apply_close_position(app, work, action):
    pos = _find(work['positions'], 'trade_id', action['trade_id'])
    if pos is None:
        raise ValueError('position not found: ' + str(action['trade_id']))
    exit_price = _positive(action['price'], 'exit price')
    exit_date = _date(action['session']).isoformat()
    if exit_date < _date(pos['entry_date']).isoformat():
        raise ValueError('broker exit date precedes entry date')
    gross = round(exit_price * pos['shares'], 2)
    pnl = round(gross - pos['cost_basis'], 2)
    pct = pnl / pos['cost_basis'] * 100
    closed = copy.deepcopy(pos)
    closed.update({
        'exit_price': exit_price, 'exit_date': exit_date, 'exit_value': gross,
        'brokerage_out': 0.0, 'realized_pnl': pnl, 'realized_pnl_pct': pct,
        'realized_pct': pct, 'net_realized_pct': pct,
        'reason': action.get('reason', 'broker_confirmed_exit'),
        'result': _result(pct), 'Result': _result(pct),
        'outcome_basis': 'net_realized', 'last_evaluated_session': exit_date,
        'fee_basis': 'broker_actual', 'broker_order_id': action.get('broker_order_id', ''),
    })
    closed.pop('exit_requested', None)
    work['closed_trades'].append(closed)
    work['positions'].remove(pos)
    work['total_realized_pnl'] = round(work['total_realized_pnl'] + pnl, 2)
    return ('Closed ' + closed['ticker'] + ': ' + str(pos['shares']) + ' @ '
            + str(exit_price) + ' (broker confirmed)')


def _apply_resize_position(app, work, action):
    pos = _find(work['positions'], 'trade_id', action['trade_id'])
    if pos is None:
        raise ValueError('position not found: ' + str(action['trade_id']))
    shares = action['shares']
    if not isinstance(shares, int) or shares < 1:
        raise ValueError('broker share count must be a positive integer')
    previous = pos['shares']
    entry = _positive(pos['entry_price'], 'entry_price')
    pos['shares'] = shares
    pos['cost_basis'] = round(entry * shares, 2)
    pos['current_value'] = round(_positive(pos.get('current_price', entry),
                                           'current_price') * shares, 2)
    pos['unrealized_pnl'] = round(pos['current_value'] - pos['cost_basis'], 2)
    pos['unrealized_pnl_pct'] = (pos['current_value'] / pos['cost_basis'] - 1) * 100
    pos['cost_basis_basis'] = 'broker_reconciled'
    return ('Resized ' + pos['ticker'] + ': ' + str(previous) + ' -> ' + str(shares))


def _apply_reprice_position(app, work, action):
    pos = _find(work['positions'], 'trade_id', action['trade_id'])
    if pos is None:
        raise ValueError('position not found: ' + str(action['trade_id']))
    entry = _positive(action['entry_price'], 'entry_price')
    previous = pos['entry_price']
    pos['entry_price'] = entry
    pos['cost_basis'] = round(entry * pos['shares'], 2)
    current = _positive(pos.get('current_price', entry), 'current_price')
    pos['current_value'] = round(current * pos['shares'], 2)
    pos['unrealized_pnl'] = round(pos['current_value'] - pos['cost_basis'], 2)
    pos['unrealized_pnl_pct'] = (pos['current_value'] / pos['cost_basis'] - 1) * 100
    pos['cost_basis_basis'] = 'broker_reconciled'
    return ('Repriced ' + pos['ticker'] + ': ' + str(previous) + ' -> ' + str(entry))


def _apply_adopt_position(app, work, action):
    symbol = _text(action['symbol'], 'symbol').upper()
    if any(p['ticker'].strip().upper() == symbol for p in work['positions']):
        raise ValueError('ticker already held: ' + symbol)
    shares = action['shares']
    if not isinstance(shares, int) or shares < 1:
        raise ValueError('adopted share count must be a positive integer')
    entry = _positive(action['entry_price'], 'entry_price')
    # plan_order calls _sector() on EVERY open position when sizing a new order,
    # so a sector it cannot map raises there and permanently blocks every future
    # order until the ledger is hand-edited. Validate against the same mapping
    # now and refuse the adoption, rather than poisoning the ledger with it.
    sector = _text(action.get('sector') or '', 'sector')
    _canonical_sector(sector)
    current = _positive(action.get('current_price') or entry, 'current_price')
    cost = round(entry * shares, 2)
    value = round(current * shares, 2)
    work['positions'].append({
        'trade_id': str(uuid4()), 'ticker': symbol, 'shares': shares,
        'entry_price': entry, 'entry_date': action['session'],
        'signal_date': action['session'], 'sector': sector,
        'cost_basis': cost, 'brokerage_in': 0.0,
        'current_price': current, 'current_value': value,
        'unrealized_pnl': round(value - cost, 2),
        'unrealized_pnl_pct': (value / cost - 1) * 100,
        'hold_days': 0, 'held_sessions': 0,
        'hold_sessions': _hold(getattr(app, '_CFG_HOLD_DAYS', 10)),
        'high_watermark': current, 'atr_at_entry': 0, 'filled_at_open': False,
        'quote_stale': False, 'cost_basis_basis': 'broker_adopted',
        'source': 'BROKER_ADOPTED',
        'reasoning': 'Adopted from the Alpaca account; not originated by the screener.',
        # No stop/target is known for a position the screener did not plan, so
        # mechanical_exit cannot fire until a human sets them.
        'needs_risk_levels': True,
    })
    return 'Adopted ' + symbol + ': ' + str(shares) + ' @ ' + str(entry)


def _apply_set_cash(app, work, action):
    cash = finite_number(action['cash'], 'cash', minimum=0)
    previous = work['cash']
    work['cash'] = round(cash, 2)
    equity = action.get('equity')
    if equity is not None:
        work['broker_equity'] = round(finite_number(equity, 'equity'), 2)
    work['cash_basis'] = 'broker_actual'
    return 'Cash set from broker: ' + str(previous) + ' -> ' + str(work['cash'])


_BROKER_OPS = {
    'fill_pending': _apply_fill_pending,
    'expire_pending': _apply_expire_pending,
    'close_position': _apply_close_position,
    'resize_position': _apply_resize_position,
    'reprice_position': _apply_reprice_position,
    'adopt_position': _apply_adopt_position,
    'set_cash': _apply_set_cash,
}
# Positions settle before cash so the broker balance is written last and is
# never re-derived from local arithmetic.
_BROKER_OP_ORDER = ('fill_pending', 'expire_pending', 'close_position',
                    'resize_position', 'reprice_position', 'adopt_position',
                    'set_cash')


def apply_broker_state(app, pf, plan):
    """Apply a broker reconciliation plan to the ledger, then save atomically.

    Each action is staged against a private copy and validated on its own, so a
    single unusable action is recorded as a failure instead of discarding the
    rest of the reconciliation. Returns ``{'applied', 'failed', 'notes'}``.
    A plan the broker could not substantiate (``ok`` false) applies nothing.
    """
    result = {'applied': [], 'failed': [], 'notes': []}
    if not isinstance(plan, dict) or not plan.get('ok'):
        result['notes'].append('broker plan unavailable; ledger untouched')
        return result
    work = _validated(pf)
    actions = sorted(plan.get('actions', []),
                     key=lambda a: _BROKER_OP_ORDER.index(a['op'])
                     if a.get('op') in _BROKER_OP_ORDER else len(_BROKER_OP_ORDER))
    for action in actions:
        handler = _BROKER_OPS.get(action.get('op'))
        if handler is None:
            result['failed'].append({'action': action, 'error': 'unknown op'})
            continue
        try:
            stage = copy.deepcopy(work)
            note = handler(app, stage, action)
            work = _validated(stage)
            result['applied'].append(action['op'] + ': ' + str(action.get('symbol', '')))
            result['notes'].append(note)
        except Exception as exc:
            result['failed'].append({'action': action, 'error': str(exc)})
            app._degrade('broker_sync_failed:' + str(action.get('op'))
                         + ':' + str(action.get('symbol', '')))
    save_portfolio(app, work)
    _publish(pf, work)
    return result


def _frame(frame, start, asof):
    """Keep only this ticker's requested dates; reject ambiguous daily indexes."""
    if (not isinstance(frame, pd.DataFrame)
            or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans or isinstance(frame.columns, pd.MultiIndex)
            or not frame.columns.is_unique
            or not {'Open', 'High', 'Low', 'Close'}.issubset(frame.columns)):
        raise ValueError('invalid single-ticker OHLC history')
    days = frame.index.date
    selected = frame.loc[(days >= _date(start)) & (days <= _date(asof))].sort_index().copy()
    if len(set(selected.index.date)) != len(selected):
        raise ValueError('ambiguous duplicate market session')
    return selected


def _fetch(app, ticker, start, asof):
    instrument = app.yf.Ticker(ticker)
    history = instrument.history(start=_date(start).isoformat(),
                                 end=(_date(asof) + timedelta(days=1)).isoformat(),
                                 auto_adjust=False)
    return instrument, _frame(history, start, asof)


def _earnings_exit(app, instrument, pos, asof):
    try:
        calendar = instrument.calendar
        raw = None
        if isinstance(calendar, dict):
            raw = calendar.get('Earnings Date')
        elif isinstance(calendar, pd.DataFrame) and 'Earnings Date' in calendar.index:
            raw = calendar.loc['Earnings Date'].tolist()
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if raw is not None:
            days = (_date(raw) - _date(asof)).days
            pos['earnings_days_away'] = days
            if 0 <= days <= finite_number(getattr(app, '_CFG_PRE_EARNINGS_DAYS', 0),
                                          'pre_earnings_days', minimum=0):
                return 'pre_earnings (earnings in ' + str(days) + 'd)'
    except Exception as exc:
        pos['earnings_status'] = 'unavailable: ' + str(exc)
        app._degrade('earnings_unavailable:' + pos['ticker'])
    return None


def _indicator_exit(app, pos, history):
    # No second request, cross-ticker fallback or future bars. Insufficient
    # pre-entry history explicitly disables only the indicators lacking warmup.
    if 'Stock Splits' in history:
        splits = history['Stock Splits'].fillna(0).ne(0)
        if splits.any():
            # A pre-entry split need not block ownership, but raw prices across
            # it are not a comparable indicator series.
            history = history.loc[history.index >= history.index[splits][-1]]
    closes = history['Close'].astype(float)
    note = []
    if len(closes) < 14:
        note.append('RSI omitted: fewer than 14 same-ticker sessions')
    else:
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean().iloc[-1]
        rsi = 100.0 if loss == 0 and gain > 0 else 50.0 if loss == gain == 0 else 100 - 100 / (1 + gain / loss)
        if (rsi > finite_number(getattr(app, '_CFG_RSI_EXIT', 78.0), 'rsi_exit')
                and pos['unrealized_pnl_pct'] >= finite_number(
                    getattr(app, '_CFG_RSI_EXIT_MIN_PROFIT', 0.0), 'rsi_exit_min_profit')):
            return 'rsi_overbought'
    if len(closes) < 26:
        note.append('MACD omitted: fewer than 26 same-ticker sessions')
    elif pos['unrealized_pnl_pct'] >= finite_number(
            getattr(app, '_CFG_MACD_EXIT_MIN_PROFIT', 0.0), 'macd_exit_min_profit'):
        macd = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return 'macd_bearish_cross'
    pos['indicator_exit_note'] = '; '.join(note)
    return None


def _expire(app, work, order, reason):
    work['pending_orders'].remove(order)
    record = copy.deepcopy(order)
    record.update({'status': 'EXPIRED', 'reason': reason, 'expired_date': _session(app)})
    work.setdefault('expired_orders', []).append(record)
    _reason(app, 'Pending ' + order['ticker'] + ' expired: ' + reason)


def _check_splits(history, first, asof):
    if 'Stock Splits' not in history:
        return
    days = history.index.date
    interval = history.loc[(days >= _date(first)) & (days <= _date(asof)), 'Stock Splits']
    for stamp, value in interval.items():
        try:
            split = finite_number(value, 'Stock Splits', minimum=0)
        except ValueError as exc:
            raise _CorporateActionReview('corporate action review required: invalid Stock Splits') from exc
        if split != 0:
            raise _CorporateActionReview(
                'corporate action review required: Stock Splits on ' + _date(stamp).isoformat())


def _stale(app, record, exc):
    record['quote_stale'] = True
    record['update_error'] = str(exc)
    if isinstance(exc, _CorporateActionReview):
        record['corporate_action_review_required'] = True
        app._degrade('corporate_action_review_required:' + record['ticker'])
    app._degrade('stale_quote:' + record['ticker'])


def _replay_position(app, original, instrument, history, asof):
    """Plan a whole position update privately; a hole rolls back every bar."""
    pos = copy.deepcopy(original)
    entry = _date(pos['entry_date']).isoformat()
    actual_open = pos.get('filled_at_open', False)
    owned = ([entry] if actual_open else []) + _sessions_after(app, entry, asof)
    last = _date(pos['last_evaluated_session']).isoformat() if 'last_evaluated_session' in pos else None
    pending = [day for day in owned if last is None or day > last]
    if pos.get('corporate_action_review_required'):
        raise _CorporateActionReview('corporate action review required: reconcile ledger before resuming')
    if pending:
        _check_splits(history, pending[0], asof)
    # Even an earlier replay exit requires complete coverage through current asof.
    # A partial response must never silently advance the last-evaluated cursor.
    valid = {}
    for day in dict.fromkeys(pending + [asof]):
        bar = fresh_bar(history, day)
        if bar is None:
            raise ValueError('quote gap: missing/invalid exact session ' + day)
        valid[day] = bar
    if asof > entry and (not owned or owned[-1] != asof):
        raise ValueError('as-of is not an expected completed market session')
    if any(fresh_bar(history, day) is None for day in history.index.date):
        raise ValueError('invalid historical bar; session/indicator evaluation unsafe')
    hold = pos.setdefault('hold_sessions', _hold(getattr(app, '_CFG_HOLD_DAYS', 10)))
    # An after-close legacy entry can be marked on entry day but never owns its
    # intraday stop/target/indicator path. Actual opening fills own that day.
    for day in pending or [asof]:
        bar = valid[day]
        pos.update({'current_price': bar['Close'],
                    'current_value': round(bar['Close'] * pos['shares'], 2),
                    'quote_stale': False, 'quote_date': day})
        pos.pop('update_error', None)
        pos['unrealized_pnl'] = round(pos['current_value'] - pos['cost_basis'], 2)
        pos['unrealized_pnl_pct'] = (pos['current_value'] / pos['cost_basis'] - 1) * 100
        count = sum(session <= day for session in owned)
        pos['hold_days'] = pos['held_sessions'] = count
        pos['last_evaluated_session'] = day
        if day not in owned:
            continue
        exit_order = mechanical_exit(pos, bar)
        if exit_order is None:
            # Today's earnings calendar is not historical point-in-time evidence.
            reason = _earnings_exit(app, instrument, pos, asof) if day == asof else None
            if reason is None and count >= hold:
                reason = 'hold_period'
            if reason is None:
                reason = _indicator_exit(app, pos, history.loc[history.index.date <= _date(day)])
            if reason is not None:
                exit_order = (bar['Close'], reason)
        if exit_order is not None:
            return pos, exit_order
        # Ratchet only after this bar's original stop/target has been tested.
        hwm = finite_number(pos.get('high_watermark', pos['entry_price']), 'high_watermark', minimum=0)
        if bar['Close'] > hwm:
            pos['high_watermark'] = bar['Close']
            atr = finite_number(pos.get('atr_at_entry', 0), 'atr', minimum=0)
            multiplier = _positive(getattr(app, '_CFG_TRAIL_ATR_MULT', 1.5), 'trail_atr_mult')
            new_stop = round(bar['Close'] - multiplier * atr, 2)
            if atr > 0 and new_stop > 0 and new_stop > (pos.get('stop_price') or 0):
                pos['stop_price'] = new_stop
    return pos, None


def update_portfolio_prices(app, pf):
    """Replay unevaluated completed sessions, then atomically publish the run."""
    work = _validated(pf)
    asof = _session(app)
    plans = []
    starts = {}
    for record in work['positions'] + work['pending_orders']:
        ticker = record['ticker']
        dates = [record[key] for key in ('signal_date', 'entry_date') if key in record]
        start = min(_date(day).isoformat() for day in dates)
        starts[ticker] = min(starts.get(ticker, start), start)
    cache = {}

    def history_for(ticker):
        if ticker not in cache:
            try:
                # Keep starts as the earliest ledger record, NOT the warmup date.
                warmup = _date(starts[ticker]) - timedelta(days=60)
                cache[ticker] = _fetch(app, ticker, warmup, asof)
            except Exception as exc:
                cache[ticker] = exc
        if isinstance(cache[ticker], Exception):
            raise cache[ticker]
        return cache[ticker]

    # In broker mode a pending order is resolved by apply_broker_state from
    # Alpaca's actual execution. Simulating a fill at the opening print here
    # would be exactly the phantom-position bug broker authority exists to end.
    for order in ([] if broker_authoritative(app) else list(work['pending_orders'])):
        ticker = order['ticker']
        try:
            if 'execution_session' not in order:
                order['execution_session'] = expected_execution_session(app, order['signal_date'])
            expected = _date(order['execution_session']).isoformat()
            if asof > expected:
                _expire(app, work, order, 'missed next session; never backdate a fill')
                continue  # Expiry is unconditional and precedes any ticker request.
            if asof < expected:
                continue
            _, history = history_for(ticker)
            if order.get('corporate_action_review_required'):
                raise _CorporateActionReview('corporate action review required before execution')
            _check_splits(history, _date(order['signal_date']) + timedelta(days=1), asof)
            bar = fresh_bar(history, asof)
            if bar is None:
                raise ValueError('no valid exact-session opening bar')
            metadata = _learning_metadata(order)
            metadata.update({'signal_date': order['signal_date'], 'trade_id': order['id'],
                             'execution_session': expected,
                             'filled_at_open': True, 'hold_sessions': order['hold_sessions'],
                             'max_shares': order['shares']})
            # Remove the pending identity before turning it into a trade identity.
            work['pending_orders'].remove(order)
            with _fill_context(app, metadata):
                open_position(app, work, ticker, bar['Open'], order['amount_usd'],
                              _price_offset(bar['Open'], order['stop_distance'], subtract=True),
                              _price_offset(bar['Open'], order['target_distance']), order['sector'],
                              order['atr'], work.get('last_nzdusd_rate'))
            if not any(p['trade_id'] == order['id'] for p in work['positions']):
                detail = app._ORDER_REASON[0]
                work['pending_orders'].append(order)
                _expire(app, work, order, 'execution rejected: ' + detail)
        except Exception as exc:
            # On unexpected execution failure retain the unfilled order, so a
            # later session can expire it rather than silently losing its audit.
            _stale(app, order, exc)
            if not any(p['id'] == order['id'] for p in work['pending_orders']) and not any(
                    p['trade_id'] == order['id'] for p in work['positions']):
                work['pending_orders'].append(order)
            _reason(app, 'Pending ' + ticker + ' unavailable: ' + str(exc))

    for original in list(work['positions']):
        ticker = original['ticker']
        if ('last_evaluated_session' in original
                and _date(original['last_evaluated_session']) >= _date(asof)):
            continue
        if _date(original['entry_date']) > _date(asof):
            app._degrade('future_entry:' + ticker)
            continue
        try:
            instrument, history = history_for(ticker)
            pos, exit_order = _replay_position(app, original, instrument, history, asof)
            plans.append((pos, exit_order))
        except Exception as exc:
            _stale(app, original, exc)
    # Commit historical sells in session order even if ledger positions were not
    # ordered that way. Every close has an independent rollback-safe fee stage.
    for pos, exit_order in sorted(plans, key=lambda plan: plan[0]['last_evaluated_session']):
        ticker = pos['ticker']
        try:
            stage = copy.deepcopy(work)
            index = next(i for i, p in enumerate(stage['positions']) if p['trade_id'] == pos['trade_id'])
            stage['positions'][index] = pos
            if exit_order is not None and broker_authoritative(app):
                # An exit is a REQUEST, not a fact. The position stays open until
                # Alpaca confirms the sale; _ledger_share_map drops flagged
                # positions so this run's mirror submits the sell.
                stage['positions'][index]['exit_requested'] = {
                    'reason': str(exit_order[1]),
                    'session': _date(pos['last_evaluated_session']).isoformat()}
                _reason(app, 'Exit requested for ' + ticker + ': ' + str(exit_order[1]))
            elif exit_order is not None:
                with _close_context(app, pos['last_evaluated_session']):
                    close_position(app, stage, ticker, exit_order[0], exit_order[1],
                                   stage.get('last_nzdusd_rate'))
                if any(p['trade_id'] == pos['trade_id'] for p in stage['positions']):
                    raise ValueError(app._ORDER_REASON[0])
            work = _validated(stage)
        except Exception as exc:
            current = next(p for p in work['positions'] if p['trade_id'] == pos['trade_id'])
            _stale(app, current, exc)
    save_portfolio(app, work)
    return _publish(pf, work)


def _string(value):
    return '' if value is None else str(value)


def load_performance_history(app, fp):
    """The compatibility fp argument is ignored: learning requires no CSV."""
    history = []
    for trade in load_portfolio(app)['closed_trades']:
        pct = _net_pct(trade)
        metadata = _learning_metadata(trade)
        row = {key: _string(metadata.get(key)) for key in _LEARNING_FIELDS}
        excess = _string(trade.get('excess_return_pct'))
        row.update({
            'trade_id': trade['trade_id'], 'ticker': trade['ticker'],
            'date': _string(trade.get('signal_date', trade['entry_date'])),
            'entry_date': trade['entry_date'],
            'exit_date': _string(trade.get('exit_date') or trade.get('close_date')),
            'sector': _string(trade.get('sector')), 'result': _result(pct),
            'return_pct': str(pct), 'net_realized_pct': str(pct),
            'vs_qqq_10d': excess, 'close_reason': _string(trade.get('reason')),
            'outcome_basis': 'net_realized',
            'benchmark_return_pct': _string(trade.get('benchmark_return_pct')),
            # Old build_learning_insights indexes these directly. They are aliases
            # of the actual ledger interval, NOT fabricated 30-day observations.
            'return_30d': str(pct), 'vs_qqq_30d': excess,
            'holding_period_basis': 'actual_ledger_interval',
        })
        row['reasoning'] = row['reasoning'][:120]
        history.append(row)
    return history


def _csv(path):
    # Object columns avoid dtype guessing and preserve old report text verbatim.
    return pd.read_csv(path, dtype=str, keep_default_na=False).astype(object)


def _set_row(df, index, values):
    for key, value in values.items():
        if key not in df:
            df[key] = ''
        df.at[index, key] = _string(value)


def reconcile_closed_picks(app, pf):
    """Project known trade IDs to CSV; NEVER use CSV prices to close positions."""
    work = _validated(pf)
    path = getattr(app, 'PICKS_CSV', None)
    _reason(app, 'Ledger is the source of truth; CSV cannot close positions')
    if path is None or not Path(path).exists():
        return pf
    df = _csv(path)
    trades = {p['trade_id']: p for p in work['positions'] + work['closed_trades']}
    closed = {p['trade_id'] for p in work['closed_trades']}
    changed = False
    for index, row in df.iterrows():
        tid = row.get('Trade_ID', '')
        if tid not in trades:
            continue  # No ticker-only or guessed legacy identity matching.
        trade = trades[tid]
        values = {'Execution_Status': 'FILLED', 'Outcome_Basis': 'net_realized',
                  'Realistic_Entry': trade['entry_price'], 'Entry_Date': trade['entry_date'],
                  'Shares': trade['shares'], 'Cost_Basis': trade['cost_basis'],
                  'Return_Pct': '', 'vs_QQQ_10d': '', 'Benchmark_Return_Pct': '',
                  'Result': 'Pending', 'Close_Date': '', 'Close_Price': '', 'Close_Reason': ''}
        if tid in closed:
            pct = _net_pct(trade)
            values.update({'Result': _result(pct), 'Return_Pct': pct,
                           'Close_Date': trade.get('exit_date', ''),
                           'Close_Price': trade['exit_price'], 'Close_Reason': trade.get('reason', ''),
                           'vs_QQQ_10d': trade.get('excess_return_pct', ''),
                           'Benchmark_Return_Pct': trade.get('benchmark_return_pct', '')})
        if any(_string(row.get(key, '')) != _string(value) for key, value in values.items()):
            _set_row(df, index, values)
            changed = True
    if changed:
        atomic_csv(path, df)
    return pf


def update_results(app, fp, cols):
    """Annotate unexecuted paper signals; never horizon-close a filled trade."""
    if not Path(fp).exists():
        return
    pf = load_portfolio(app)
    executed = {p['trade_id'] for p in pf['positions'] + pf['closed_trades']}
    df = _csv(fp)
    asof = _session(app)
    changed = False
    for index, row in df.iterrows():
        ticker = row.get('Ticker', '').strip().upper()
        if (ticker in ('', 'NONE', 'NAN', 'NO PICK')
                or row.get('Signal', '').strip().upper() == 'NO PICK'
                or row.get('Execution_Status', '').strip().upper() == 'FILLED'
                or row.get('Trade_ID', '') in executed):
            continue
        # Prospectively annotate pending outcomes, not rewrite legacy CSV history.
        if row.get('Result', '').strip() not in ('', 'Pending'):
            continue
        try:
            signal = _date(row['Date']).isoformat()
            if signal >= asof:
                continue
            raw_hold = row.get('Hold_Sessions', '')
            hold = _hold(float(raw_hold) if raw_hold else getattr(app, '_CFG_HOLD_DAYS', 10))
            _, stock = _fetch(app, ticker, signal, asof)
            observed = stock.loc[(stock.index.date > _date(signal)) & (stock.index.date <= _date(asof))]
            if len(observed) < hold:
                continue  # Normal immature WATCH: not an operational degradation.
            _, benchmark = _fetch(app, 'QQQ', signal, asof)
            result = evaluate_horizon(stock, benchmark, signal, hold, asof)
            if result is None:
                # Advisory namespace: the parent must not gate core orders on
                # mature paper-report failures (separate from stale held quotes).
                app._degrade('paper_horizon_unavailable:' + ticker)
                continue
            values = {
                'Realistic_Entry': result['entry_price'], 'Entry_Date': result['entry_date'],
                'Close_Date': result['exit_date'], 'Close_Price': result['exit_price'],
                'Close_Reason': 'paper_hold_sessions', 'Hold_Sessions': hold,
                'Return_Pct': result['return_pct'], 'vs_QQQ_10d': result['excess_return_pct'],
                'Benchmark_Return_Pct': result['benchmark_return_pct'],
                'Outcome_Basis': 'paper_horizon', 'Result': _result(result['return_pct']),
            }
            _set_row(df, index, values)
            changed = True
        except Exception as exc:
            app._degrade('paper_horizon_error:' + ticker + ':' + str(exc))
    if changed:
        for column in cols:
            if column not in df:
                df[column] = ''
        atomic_csv(fp, df)