"""Broker-authoritative reconciliation planning. Pure logic, no I/O.

The screener queues an order in its own ledger and mirrors it onto Alpaca, but
until now nothing ever read back what Alpaca actually did: the ledger booked a
fill at the daily bar's opening print and assumed the broker complied. A
rejected order became a phantom position, a partial fill was booked whole, and
real slippage never reached the cost basis.

This module answers "what actually happened at the broker" by diffing the
ledger against an authoritative snapshot. It is deliberately free of network,
file and app access so every branch is unit-testable offline.

``plan_broker_sync`` returns declarative ACTIONS (applied by the ledger) and
EVENTS (rendered to Discord and the run report). It never mutates its inputs.

Authority rule: the broker wins, but only when the broker actually spoke. A
snapshot with ``ok=False`` yields no actions at all — a transport failure must
never be read as "the account holds nothing" and delete live positions. For the
same reason a record the broker reports incoherently (a filled order with no
fill price) is reported as blocked rather than guessed at.
"""

import math


# Alpaca order lifecycle. Anything unrecognized is treated as still working, so
# an unknown future status can never silently cancel a ledger order.
FILLED = 'filled'
PARTIAL = 'partially_filled'
DEAD_STATUSES = frozenset({
    'canceled', 'cancelled', 'expired', 'rejected', 'done_for_day',
    'replaced', 'stopped', 'suspended',
})
WORKING_STATUSES = frozenset({
    'new', 'accepted', 'pending_new', 'accepted_for_bidding', 'calculated',
    'held', 'pending_cancel', 'pending_replace', 'partially_filled',
})

# Event severities drive Discord colour and whether the run is degraded.
INFO, WARN, ERROR = 'info', 'warning', 'error'


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value):
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _whole(value):
    number = _finite(value)
    if number is None:
        return None
    rounded = int(round(number))
    return rounded if abs(number - rounded) < 1e-6 else None


def _symbol(value):
    return str(value or '').strip().upper()


def _norm_ref(value):
    """Normalize a ledger id the same way the client_order_id tag is built."""
    return ''.join(ch for ch in str(value or '') if ch.isalnum())[:32]


def _event(kind, severity, symbol, summary, **fields):
    event = {'kind': kind, 'severity': severity, 'symbol': symbol, 'summary': summary}
    event.update(fields)
    return event


def _index_orders(orders):
    """Index broker orders by ledger ref and by (symbol, side).

    Newest-first ordering is preserved so a re-entered symbol resolves to its
    most recent order when only the weak (symbol, side) fallback is available.
    """
    by_ref, by_symbol_side = {}, {}
    for order in orders or []:
        if not isinstance(order, dict):
            continue
        symbol = _symbol(order.get('symbol'))
        if not symbol:
            continue
        side = str(order.get('side', '') or '').strip().lower()
        ref = _norm_ref(_parse_ref(order.get('client_order_id')))
        if ref and ref not in by_ref:
            by_ref[ref] = order
        by_symbol_side.setdefault((symbol, side), []).append(order)
    return by_ref, by_symbol_side


def _parse_ref(client_order_id):
    """Extract our embedded ledger id from a client_order_id, else ''."""
    parts = str(client_order_id or '').split('-')
    if len(parts) != 4 or parts[0] != 'lpm' or parts[1] not in ('B', 'S'):
        return ''
    return parts[3]


def _match_order(record_id, symbol, side, by_ref, by_symbol_side, used):
    """Exact ref match first; fall back to the newest unused symbol/side order.

    The fallback exists only for orders placed before ids were tagged. It is
    reported to the caller so a weak match can be surfaced rather than trusted
    as silently as an exact one.
    """
    ref = _norm_ref(record_id)
    order = by_ref.get(ref)
    if order is not None and id(order) not in used:
        used.add(id(order))
        return order, True
    for candidate in by_symbol_side.get((symbol, side), []):
        if id(candidate) in used:
            continue
        if _parse_ref(candidate.get('client_order_id')):
            continue  # tagged for a different ledger record; not ours to claim
        used.add(id(candidate))
        return candidate, False
    return None, False


def _fill_of(order):
    """Return (filled_qty, filled_avg_price) when the order really filled."""
    qty = _whole(order.get('filled_qty')) or 0
    price = _positive(order.get('filled_avg_price'))
    return qty, price


def _plan_pending(ledger, by_ref, by_symbol_side, used, session, actions, events):
    for order in ledger.get('pending_orders', []) or []:
        if not isinstance(order, dict):
            continue
        symbol = _symbol(order.get('ticker'))
        order_id = order.get('id')
        broker_order, exact = _match_order(order_id, symbol, 'buy', by_ref,
                                           by_symbol_side, used)
        if broker_order is None:
            # Never submitted, or submitted and already purged from the window.
            # Either way there is no evidence of a fill, so do not invent one.
            actions.append({'op': 'expire_pending', 'order_id': order_id,
                            'symbol': symbol,
                            'reason': 'no matching Alpaca order found'})
            events.append(_event(
                'order_missing', WARN, symbol,
                f'{symbol}: queued order has no matching Alpaca order — expired, not filled',
                requested_shares=order.get('shares')))
            continue

        status = str(broker_order.get('status', '') or '').strip().lower()
        filled_qty, fill_price = _fill_of(broker_order)
        requested = _whole(order.get('shares')) or 0
        broker_order_id = str(broker_order.get('order_id', '') or '')

        if filled_qty > 0:
            if fill_price is None:
                actions.append({'op': 'expire_pending', 'order_id': order_id,
                                'symbol': symbol,
                                'reason': 'broker reported a fill without a price'})
                events.append(_event(
                    'fill_unpriced', ERROR, symbol,
                    f'{symbol}: Alpaca reports {filled_qty} filled but no fill price — not booked',
                    status=status, broker_order_id=broker_order_id))
                continue
            partial = status != FILLED or filled_qty < requested
            actions.append({'op': 'fill_pending', 'order_id': order_id,
                            'symbol': symbol, 'shares': filled_qty,
                            'price': round(fill_price, 4),
                            'broker_order_id': broker_order_id,
                            'partial': partial, 'session': session})
            expected = _positive(order.get('estimated_entry'))
            slippage = ((fill_price / expected - 1) * 100) if expected else None
            # Display context so the alert can read like a broker fill notice
            # rather than a diff: what it cost, and where protection now sits.
            stop_distance = _positive(order.get('stop_distance'))
            target_distance = _positive(order.get('target_distance'))
            events.append(_event(
                'partial_fill' if partial else 'fill',
                WARN if partial else INFO, symbol,
                (f'{symbol}: PARTIALLY FILLED {filled_qty}/{requested} @ ${fill_price:,.2f}'
                 if partial else
                 f'{symbol}: FILLED {filled_qty} @ ${fill_price:,.2f}'),
                side='buy', shares=filled_qty, requested_shares=requested,
                unfilled=max(0, requested - filled_qty), price=fill_price,
                notional=round(fill_price * filled_qty, 2),
                stop=(round(fill_price - stop_distance, 2) if stop_distance else None),
                target=(round(fill_price + target_distance, 2) if target_distance else None),
                expected_price=expected, slippage_pct=slippage,
                status=status, broker_order_id=broker_order_id,
                exact_match=exact))
            continue

        if status in DEAD_STATUSES:
            actions.append({'op': 'expire_pending', 'order_id': order_id,
                            'symbol': symbol, 'reason': 'broker status ' + status})
            events.append(_event(
                'rejected' if status == 'rejected' else 'canceled',
                ERROR if status == 'rejected' else WARN, symbol,
                f'{symbol}: order {status.upper()} at Alpaca — no position opened',
                status=status, requested_shares=requested,
                broker_order_id=broker_order_id))
            continue

        # Still working (queued for the next open, or held pre-market).
        events.append(_event(
            'working', INFO, symbol,
            f'{symbol}: order still {status or "open"} at Alpaca — awaiting fill',
            status=status, requested_shares=requested,
            broker_order_id=broker_order_id))


def _sell_fill_for(trade_id, symbol, by_ref, by_symbol_side, used):
    order, _ = _match_order(trade_id, symbol, 'sell', by_ref, by_symbol_side, used)
    if order is None:
        return None, None, ''
    qty, price = _fill_of(order)
    if qty <= 0 or price is None:
        return None, None, str(order.get('order_id', '') or '')
    return qty, price, str(order.get('order_id', '') or '')


def _plan_positions(ledger, broker, by_ref, by_symbol_side, used, session,
                    actions, events):
    held = broker.get('positions') or {}
    seen = set()
    for position in ledger.get('positions', []) or []:
        if not isinstance(position, dict):
            continue
        symbol = _symbol(position.get('ticker'))
        trade_id = position.get('trade_id')
        seen.add(symbol)
        ledger_qty = _whole(position.get('shares')) or 0
        broker_position = held.get(symbol)

        if broker_position is None:
            qty, price, broker_order_id = _sell_fill_for(
                trade_id, symbol, by_ref, by_symbol_side, used)
            if price is None:
                # Gone from the account with no sell fill we can price. Closing
                # at a stale quote would fabricate a P&L number, so this is
                # surfaced for a human instead of guessed.
                events.append(_event(
                    'position_vanished', ERROR, symbol,
                    f'{symbol}: held in the ledger but absent at Alpaca, with no priced sell fill',
                    ledger_shares=ledger_qty))
                continue
            actions.append({'op': 'close_position', 'trade_id': trade_id,
                            'symbol': symbol, 'price': round(price, 4),
                            'shares': qty, 'reason': 'broker_confirmed_exit',
                            'broker_order_id': broker_order_id, 'session': session})
            proceeds = round(price * qty, 2)
            basis = _positive(position.get('cost_basis'))
            pnl = round(proceeds - basis, 2) if basis else None
            events.append(_event(
                'exit_filled', INFO, symbol,
                f'{symbol}: SOLD {qty} @ ${price:,.2f} — confirmed by Alpaca',
                side='sell', shares=qty, price=price, notional=proceeds,
                cost_basis=basis, pnl=pnl,
                pnl_pct=(round(pnl / basis * 100, 2) if basis and pnl is not None else None),
                held_sessions=position.get('held_sessions', position.get('hold_days')),
                exit_reason=(position.get('exit_requested') or {}).get('reason', ''),
                broker_order_id=broker_order_id))
            continue

        broker_qty = broker_position.get('qty') or 0
        if broker_qty != ledger_qty:
            actions.append({'op': 'resize_position', 'trade_id': trade_id,
                            'symbol': symbol, 'shares': broker_qty,
                            'session': session})
            events.append(_event(
                'qty_drift', WARN, symbol,
                f'{symbol}: share count corrected {ledger_qty} → {broker_qty} from Alpaca',
                ledger_shares=ledger_qty, broker_shares=broker_qty))

        broker_entry = _positive(broker_position.get('avg_entry_price'))
        ledger_entry = _positive(position.get('entry_price'))
        if broker_entry is not None and ledger_entry is not None and (
                abs(broker_entry - ledger_entry) >= 0.005):
            actions.append({'op': 'reprice_position', 'trade_id': trade_id,
                            'symbol': symbol, 'entry_price': round(broker_entry, 4),
                            'session': session})
            events.append(_event(
                'price_drift', WARN, symbol,
                f'{symbol}: cost basis corrected ${ledger_entry:,.2f} → ${broker_entry:,.2f} from Alpaca',
                ledger_price=ledger_entry, broker_price=broker_entry))

    pending_symbols = {_symbol(o.get('ticker'))
                       for o in ledger.get('pending_orders', []) or []
                       if isinstance(o, dict)}
    for symbol, broker_position in sorted(held.items()):
        if symbol in seen or symbol in pending_symbols:
            continue
        # A holding the ledger has never heard of: a manual trade, or drift from
        # an earlier run. Adopting it keeps risk limits honest, but it needs a
        # sector before the order planner can price exposure, so the caller
        # resolves that and drops the action if it cannot.
        actions.append({'op': 'adopt_position', 'symbol': symbol,
                        'shares': broker_position.get('qty'),
                        'entry_price': broker_position.get('avg_entry_price'),
                        'current_price': broker_position.get('current_price'),
                        'sector': None, 'session': session})
        events.append(_event(
            'adopted', WARN, symbol,
            f'{symbol}: {broker_position.get("qty")} shares held at Alpaca but missing from the ledger — adopting',
            broker_shares=broker_position.get('qty'),
            broker_price=broker_position.get('avg_entry_price')))


def plan_broker_sync(ledger, broker, session):
    """Diff a ledger against an authoritative broker snapshot.

    Returns ``{'ok', 'actions', 'events', 'blocked', 'summary'}``. ``ok`` is
    False when the snapshot could not be trusted, in which case no actions are
    produced. ``blocked`` lists conditions a human must resolve; the caller
    degrades the run on those so no new order is queued while the ledger and the
    account disagree.
    """
    if not isinstance(ledger, dict):
        raise ValueError('ledger must be a dict')
    if not isinstance(broker, dict):
        raise ValueError('broker snapshot must be a dict')

    if not broker.get('ok'):
        reason = str(broker.get('error') or 'broker snapshot unavailable')
        return {'ok': False, 'actions': [], 'blocked': [reason],
                'events': [_event('snapshot_failed', ERROR, '',
                                  'Alpaca state could not be read — ledger left untouched: ' + reason)],
                'summary': 'broker unreadable: ' + reason}

    actions, events = [], []
    used = set()
    by_ref, by_symbol_side = _index_orders(broker.get('orders'))

    _plan_pending(ledger, by_ref, by_symbol_side, used, session, actions, events)
    _plan_positions(ledger, broker, by_ref, by_symbol_side, used, session,
                    actions, events)

    cash = _finite(broker.get('cash'))
    if cash is not None and cash >= 0:
        ledger_cash = _finite(ledger.get('cash'))
        actions.append({'op': 'set_cash', 'cash': round(cash, 2),
                        'equity': _finite(broker.get('equity'))})
        if ledger_cash is not None and abs(ledger_cash - cash) >= 0.01:
            events.append(_event(
                'cash_drift', INFO, '',
                f'Cash reset from Alpaca: ${ledger_cash:,.2f} → ${cash:,.2f}',
                ledger_cash=ledger_cash, broker_cash=cash))
    else:
        events.append(_event('cash_unreadable', WARN, '',
                             'Alpaca cash was missing or invalid — ledger cash left as is'))

    if broker.get('account_blocked'):
        events.append(_event('account_blocked', ERROR, '',
                             'Alpaca reports the account as blocked/restricted'))

    blocked = [event['summary'] for event in events if event['severity'] == ERROR]
    counts = {}
    for event in events:
        counts[event['kind']] = counts.get(event['kind'], 0) + 1
    summary = ', '.join(f'{kind}={count}' for kind, count in sorted(counts.items())) or 'no changes'
    return {'ok': True, 'actions': actions, 'events': events,
            'blocked': blocked, 'summary': summary}


__all__ = ['plan_broker_sync', 'FILLED', 'PARTIAL', 'DEAD_STATUSES',
           'WORKING_STATUSES', 'INFO', 'WARN', 'ERROR']
