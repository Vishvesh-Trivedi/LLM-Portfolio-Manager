"""Offline safety primitives. Invalid portfolios/orders raise ValueError.

Market-data helpers fail closed with None. Daily dates retain their local
calendar date. Callers must supply single-ticker, consistently adjusted OHLC
frames and a pure, nonnegative fee quote (also used to estimate a stop sale).
"""

import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import contextlib


def finite_number(value, name='value', minimum=None, maximum=None):
    """Return an int/float unchanged; reject booleans, coercion and nonfinite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be an int or float')
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    if not finite:
        raise ValueError(f'{name} must be finite')
    if minimum is not None:
        finite_number(minimum, 'minimum')
        if value < minimum:
            raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None:
        finite_number(maximum, 'maximum')
        if value > maximum:
            raise ValueError(f'{name} must be <= {maximum}')
    return value


def _atomic_write(path, writer):
    destination = Path(path)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', delete=False,
            dir=destination.parent, prefix='.' + destination.name + '.',
            suffix='.tmp',
        ) as stream:
            staged = stream.name
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    finally:
        if staged is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staged)


def atomic_json(path, data):
    """Serialize strictly before staging; never truncate an existing destination."""
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2)
    _atomic_write(path, lambda stream: stream.write(payload))


def atomic_csv(path, dataframe):
    """Atomically write a pandas CSV without its index; preserve pandas NA semantics."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('dataframe must be a pandas DataFrame')
    _atomic_write(path, lambda stream: dataframe.to_csv(stream, index=False))


def _positive(value, name):
    value = finite_number(value, name, minimum=0)
    if value == 0:
        raise ValueError(f'{name} must be positive')
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _required(record, key):
    if key not in record:
        raise ValueError(f'missing required field: {key}')
    return record[key]


def validate_portfolio(pf):
    """Validate without defaults; commit missing-ID migrations only after success.

    IDs are unique across open and closed trades. A missing ID requires entry_date;
    identical legacy identities are ambiguous and rejected, not given random IDs.
    Closed P&L uses realized_pnl (legacy source) or pnl; if both exist validate both.
    """
    if not isinstance(pf, dict):
        raise ValueError('portfolio must be a dict')
    finite_number(_required(pf, 'cash'), 'cash', minimum=0)
    _positive(_required(pf, 'starting_capital'), 'starting_capital')
    migrations, seen = [], set()
    for collection in ('positions', 'closed_trades'):
        trades = _required(pf, collection)
        if not isinstance(trades, list):
            raise ValueError(f'{collection} must be a list')
        for trade in trades:
            if not isinstance(trade, dict):
                raise ValueError('trade must be a dict')
            ticker = _text(_required(trade, 'ticker'), 'ticker')
            shares = _integer(_required(trade, 'shares'), 'shares')
            entry = _positive(_required(trade, 'entry_price'), 'entry_price')
            _positive(_required(trade, 'cost_basis'), 'cost_basis')
            if 'current_value' in trade:
                finite_number(trade['current_value'], 'current_value', minimum=0)
            if collection == 'closed_trades':
                _positive(_required(trade, 'exit_price'), 'exit_price')
                pnl_fields = [key for key in ('realized_pnl', 'pnl') if key in trade]
                if not pnl_fields:
                    raise ValueError('closed trade requires realized_pnl or pnl')
                for key in pnl_fields:
                    finite_number(trade[key], key)
            if 'trade_id' in trade:
                trade_id = _text(trade['trade_id'], 'trade_id')
            else:
                entry_date = _text(_required(trade, 'entry_date'), 'entry_date')
                identity = [ticker.upper(), entry_date, float(entry).hex(), shares]
                encoded = json.dumps(identity, separators=(',', ':')).encode('utf-8')
                trade_id = hashlib.sha256(encoded).hexdigest()
                migrations.append((trade, trade_id))
            if trade_id in seen:
                raise ValueError('duplicate trade_id or ambiguous legacy trade identity')
            seen.add(trade_id)
    for trade, trade_id in migrations:
        trade['trade_id'] = trade_id
    return pf


_SECTORS = {
    'technology': 'technology', 'information technology': 'technology',
    'financial services': 'financials', 'financials': 'financials',
    'healthcare': 'healthcare', 'health care': 'healthcare',
    'consumer cyclical': 'consumer cyclical',
    'consumer discretionary': 'consumer cyclical',
    'consumer defensive': 'consumer defensive',
    'consumer staples': 'consumer defensive',
    'basic materials': 'materials', 'materials': 'materials',
    'communication services': 'communication services',
    'industrials': 'industrials', 'real estate': 'real estate',
    'energy': 'energy', 'utilities': 'utilities',
}


def _sector(value):
    sector = _text(value, 'sector').casefold()
    if sector not in _SECTORS:
        raise ValueError('unknown sector')
    return _SECTORS[sector]


def _decimal(value):
    return Decimal(str(value))


def _cents(value):
    return _decimal(round(finite_number(float(value)), 2))


def plan_order(pf, ticker, entry_price, amount_usd, stop, target, sector, fee_quote,
               max_positions=5, cash_floor=500.0):
    """Return the largest feasible whole-share order or raise ValueError.

    amount_usd caps the total debit, including brokerage. Quotes receive only a
    float notional, never portfolio objects; their purity is a caller contract.
    Descending search deliberately permits nonmonotonic (e.g. tiered) fees.
    All limits include unrounded amounts and any upward cent-rounding effects.
    """
    portfolio = validate_portfolio(copy.deepcopy(pf))
    ticker = _text(ticker, 'ticker').upper()
    sector = _sector(sector)
    entry = _decimal(_positive(entry_price, 'entry_price'))
    stop_value = _decimal(_positive(stop, 'stop'))
    target_value = _decimal(_positive(target, 'target'))
    amount = _decimal(finite_number(amount_usd, 'amount_usd', minimum=0))
    floor = max(_decimal(finite_number(cash_floor, 'cash_floor', minimum=0)),
                Decimal('500'))
    limit = min(_integer(max_positions, 'max_positions'), 5)
    if not callable(fee_quote):
        raise ValueError('fee_quote must be callable')
    if not stop_value < entry < target_value:
        raise ValueError('require 0 < stop < entry_price < target')
    if target_value - entry < Decimal('1.5') * (entry - stop_value):
        raise ValueError('reward/risk must be >= 1.5')
    positions = portfolio['positions']
    if len(positions) >= limit:
        raise ValueError('maximum positions reached')
    if any(p['ticker'].strip().upper() == ticker for p in positions):
        raise ValueError('duplicate ticker')
    cash = _decimal(portfolio['cash'])
    equity, sector_value = cash, Decimal('0')
    for position in positions:
        value = _decimal(position.get('current_value', position['cost_basis']))
        equity += value
        if _sector(position.get('sector')) == sector:
            sector_value += value
    peak = max(_decimal(portfolio['starting_capital']),
               _decimal(finite_number(portfolio.get('equity_peak', 0),
                                      'equity_peak', minimum=0)))
    if equity <= peak * Decimal('0.8'):
        raise ValueError('equity drawdown is at least 20%')
    budget = min(amount, cash - floor)
    stock_cap = equity * Decimal('0.25')
    sector_cap = equity * Decimal('0.40') - sector_value
    risk_cap = equity * Decimal('0.01')
    share_cap = min(budget / entry, stock_cap / entry, sector_cap / entry,
                    risk_cap / (entry - stop_value))
    for shares in range(max(0, int(share_cap)), 0, -1):
        raw_stock = shares * entry
        stop_proceeds = shares * stop_value
        stock_cost = _cents(raw_stock)
        buy_fee = _decimal(finite_number(fee_quote(float(stock_cost)),
                                         'buy fee', minimum=0))
        sell_fee = _decimal(finite_number(fee_quote(float(_cents(stop_proceeds))),
                                          'sell fee', minimum=0))
        brokerage = _cents(buy_fee)
        total_cost = _cents(stock_cost + brokerage)
        exposure = max(raw_stock, stock_cost)
        debit = max(raw_stock + buy_fee, total_cost)
        risk = (exposure - min(stop_proceeds, _cents(stop_proceeds))
                + max(buy_fee, brokerage) + max(sell_fee, _cents(sell_fee)))
        if (stock_cost > 0 and debit <= budget and exposure <= stock_cap
                and exposure <= sector_cap and risk <= risk_cap):
            return {'shares': shares, 'stock_cost': float(stock_cost),
                    'brokerage': float(brokerage), 'total_cost': float(total_cost),
                    'amount_usd': amount_usd}
    raise ValueError('cannot buy one share within the safety limits')


def _day(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('date must not be missing')
    return stamp.date()


def _daily_frame(frame, first_day, last_day):
    if (not isinstance(frame, pd.DataFrame)
            or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans or isinstance(frame.columns, pd.MultiIndex)
            or not frame.columns.is_unique
            or not {'Open', 'High', 'Low', 'Close'}.issubset(frame.columns)):
        return None
    dates = frame.index.date
    selected = frame.loc[(dates >= first_day) & (dates <= last_day)].copy()
    selected.index = selected.index.date
    if not selected.index.is_unique:
        return None
    return selected.sort_index()


def _ohlc(bar):
    try:
        prices = {key: _positive(bar[key], key)
                  for key in ('Open', 'High', 'Low', 'Close')}
        opening, high, low, close = (prices[key] for key in
                                     ('Open', 'High', 'Low', 'Close'))
        if high < max(opening, close, low) or low > min(opening, close, high):
            return None
        return prices
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_horizon(stock, benchmark, signal_date, hold_sessions, as_of):
    """Use the next stock session's Open through the nth session's Close.

    Every held stock date must exist exactly once in the benchmark. Invalid or
    incomplete data returns None; invalid hold_sessions raises ValueError.
    """
    _integer(hold_sessions, 'hold_sessions')
    try:
        signal, cutoff = _day(signal_date), _day(as_of)
        stock_days = _daily_frame(stock, signal + timedelta(days=1), cutoff)
        if stock_days is None or len(stock_days) < hold_sessions:
            return None
        held = stock_days.iloc[:hold_sessions]
        benchmark_days = _daily_frame(benchmark, held.index[0], held.index[-1])
        if benchmark_days is None or not held.index.isin(benchmark_days.index).all():
            return None
        matched = benchmark_days.loc[held.index]
        stock_bars = [_ohlc(row.to_dict()) for _, row in held.iterrows()]
        benchmark_bars = [_ohlc(row.to_dict()) for _, row in matched.iterrows()]
        if any(bar is None for bar in stock_bars + benchmark_bars):
            return None
        entry, exit_price = stock_bars[0]['Open'], stock_bars[-1]['Close']
        benchmark_entry = benchmark_bars[0]['Open']
        benchmark_exit = benchmark_bars[-1]['Close']
        stock_return = finite_number((exit_price / entry - 1) * 100)
        benchmark_return = finite_number((benchmark_exit / benchmark_entry - 1) * 100)
        excess = finite_number(stock_return - benchmark_return)
        return {'entry_date': held.index[0].isoformat(),
                'exit_date': held.index[-1].isoformat(),
                'entry_price': entry, 'exit_price': exit_price,
                'return_pct': stock_return, 'benchmark_return_pct': benchmark_return,
                'excess_return_pct': excess}
    except (TypeError, ValueError, OverflowError):
        return None


def fresh_bar(frame, as_of):
    """Return only exact-day OHLC, rejecting ambiguous/multi-ticker frames."""
    try:
        day = _day(as_of)
        selected = _daily_frame(frame, day, day)
        if selected is None or len(selected) != 1:
            return None
        bar = _ohlc(selected.iloc[0].to_dict())
        return dict(bar, date=day.isoformat()) if bar is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def mechanical_exit(position, bar):
    """Use existing stop_price/target_price; no trailing-stop ratchet.

    Opening gaps precede intrabar touches; otherwise a dual touch is stop-first.
    Reasons match the parent source: stop_loss or profit_target (including gaps).
    """
    try:
        stop = _positive(position['stop_price'], 'stop_price')
        target = _positive(position['target_price'], 'target_price')
        prices = _ohlc(bar)
        if stop >= target or prices is None:
            return None
        if prices['Open'] <= stop:
            return prices['Open'], 'stop_loss'
        if prices['Open'] >= target:
            return prices['Open'], 'profit_target'
        if prices['Low'] <= stop:
            return stop, 'stop_loss'
        if prices['High'] >= target:
            return target, 'profit_target'
        return None
    except (KeyError, TypeError, ValueError):
        return None