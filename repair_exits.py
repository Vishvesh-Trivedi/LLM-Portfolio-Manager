"""Rebuild closed trades that were recorded at their own entry price.

One ledger record has two orders at Alpaca: the buy that opened the position
carries the pending-order id, and the sell that closed it carries the trade_id,
which a filled pending order hands over unchanged. Both ids are therefore the
same string, and until 5db8c32 the order index kept only one order per id. When
it kept the buy, closing a position priced the exit at the entry:

    Closed GILD: 98 @ 149.5798   entry 149.5798, exit_date == entry_date
    Closed CDNS: 54 @ 322.1806   entry 322.1806
    Closed DHR:  59 @ 226.78     entry 226.78

Cash comes from the broker, so the account value stayed correct throughout and
nothing flagged it. What was lost is the trade record: 874.98 of realised profit
across those three, three winners filed as neutral, and a win rate computed from
all of it. The strategy reads its own history to calibrate, so a wrong record is
not only a reporting problem.

The sync fix stops this happening again. It does not revisit trades already
closed, which is what this does, by asking Alpaca what the sells actually filled
at. Alpaca is the authority; nothing here is inferred from a quote or a
difference in cash, because a repair that guesses is worse than a record that is
visibly wrong.

Dry run by default: it prints what it would change and touches nothing. Pass
--write to apply. Needs ALPACA_API_KEY and ALPACA_SECRET_KEY, so in practice it
runs through the workflow's repair_exits input where the secrets live.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import screener_alpaca as _alpaca
import screener_portfolio as _portfolio

# A trade is a candidate only when all of these hold, which is the signature the
# bug left and not something a real trade produces: a broker-confirmed exit
# priced to the cent at its own entry, for exactly nothing.
CENT = 0.005


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def suspect(trade):
    """True when this closed trade carries the mispriced-exit signature."""
    if str(trade.get('reason', '')) != 'broker_confirmed_exit':
        return False
    entry, exit_price = _float(trade.get('entry_price')), _float(trade.get('exit_price'))
    if not entry or not exit_price:
        return False
    return (abs(entry - exit_price) < CENT
            and abs(_float(trade.get('realized_pnl'))) < 0.01)


def _fill_date(order, fallback=''):
    raw = order.get('filled_at') or order.get('updated_at')
    if not raw:
        return fallback
    try:
        text = str(raw).replace('Z', '+00:00')
        stamp = datetime.fromisoformat(text)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.date().isoformat()
    except ValueError:
        return fallback


def closing_sell(trade, orders):
    """The filled sell that closed this trade, or None.

    Matched on symbol, side, a fill after the entry date, and a filled quantity
    that matches the position. The ledger's own broker_order_id is useless here:
    it is the id the buggy match recorded, which is the buy's.
    """
    ticker = str(trade.get('ticker', '')).strip().upper()
    shares = int(_float(trade.get('shares')))
    entered = str(trade.get('entry_date', ''))[:10]
    candidates = []
    for order in orders or []:
        if str(order.get('symbol', '')).strip().upper() != ticker:
            continue
        if str(order.get('side', '')).strip().lower() != 'sell':
            continue
        filled_qty = int(_float(order.get('filled_qty')))
        price = _float(order.get('filled_avg_price'))
        if filled_qty != shares or price <= 0:
            continue
        when = _fill_date(order)
        if entered and when and when < entered:
            continue
        candidates.append((when, order, price))
    if not candidates:
        return None
    # Earliest qualifying fill: the sell that closed this position, not a later
    # one for a symbol that was re-entered.
    candidates.sort(key=lambda item: item[0] or '9999')
    return candidates[0][1]


def repair(ledger, orders):
    """Return (fixes, unresolved) without mutating the ledger."""
    fixes, unresolved = [], []
    for trade in ledger.get('closed_trades', []) or []:
        if not isinstance(trade, dict) or not suspect(trade):
            continue
        sell = closing_sell(trade, orders)
        if sell is None:
            unresolved.append(trade)
            continue
        price = _float(sell.get('filled_avg_price'))
        shares = int(_float(trade.get('shares')))
        basis = _float(trade.get('cost_basis'))
        gross = round(price * shares, 2)
        pnl = round(gross - basis, 2)
        fixes.append({
            'trade': trade,
            'order_id': str(sell.get('order_id', '') or sell.get('id', '') or ''),
            'exit_price': round(price, 4),
            'exit_date': _fill_date(sell, str(trade.get('exit_date', ''))[:10]),
            'exit_value': gross,
            'realized_pnl': pnl,
            'realized_pnl_pct': (pnl / basis * 100) if basis else 0.0,
            'was_pnl': _float(trade.get('realized_pnl')),
            'was_price': _float(trade.get('exit_price')),
            'was_date': str(trade.get('exit_date', ''))[:10],
        })
    return fixes, unresolved


def apply(ledger, fixes):
    """Write the corrected exits into the ledger and re-derive the realised total."""
    for fix in fixes:
        trade = fix['trade']
        pct = fix['realized_pnl_pct']
        trade.update({
            'exit_price': fix['exit_price'], 'exit_date': fix['exit_date'],
            'exit_value': fix['exit_value'], 'realized_pnl': fix['realized_pnl'],
            'realized_pnl_pct': pct, 'realized_pct': pct, 'net_realized_pct': pct,
            'result': _portfolio._result(pct), 'Result': _portfolio._result(pct),
            'last_evaluated_session': fix['exit_date'],
            'broker_order_id': fix['order_id'],
            'exit_repaired_from': 'alpaca_sell_fill',
        })
    # Re-derived rather than adjusted, so the total cannot drift from the trades
    # it is supposed to summarise - which is the invariant check_arithmetic
    # enforces on every run.
    ledger['total_realized_pnl'] = round(
        sum(_float(t.get('realized_pnl')) for t in ledger.get('closed_trades', []) or []), 2)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ledger', default='StockScreener/portfolio.json')
    parser.add_argument('--write', action='store_true',
                        help='apply the corrections; without it, nothing is touched')
    args = parser.parse_args(argv)

    app = SimpleNamespace(PORTFOLIO_JSON=Path(args.ledger).resolve(strict=True))
    ledger = _portfolio.load_portfolio(app)

    candidates = [t for t in ledger.get('closed_trades', []) or []
                  if isinstance(t, dict) and suspect(t)]
    if not candidates:
        print('Nothing to repair: no closed trade is priced at its own entry.')
        return 0
    print(f'{len(candidates)} closed trade(s) carry the mispriced-exit signature: '
          + ', '.join(str(t.get('ticker')) for t in candidates))

    if not _alpaca.trading_enabled():
        print('Alpaca is not configured, so the real sell fills cannot be read. '
              'Run this where ALPACA_API_KEY and ALPACA_SECRET_KEY are set '
              '(the workflow\'s repair_exits input).')
        return 1
    raw = _alpaca.fetch_orders(status='all', limit=500)
    if raw is None:
        print('Alpaca order history could not be read; nothing changed.')
        return 1
    orders = [o for o in (_alpaca.normalize_order(o) for o in raw) if o]
    print(f'Read {len(orders)} order(s) from Alpaca.')

    fixes, unresolved = repair(ledger, orders)
    for fix in fixes:
        trade = fix['trade']
        print(f'  {trade.get("ticker"):<6} {fix["was_price"]:.4f} on {fix["was_date"]}'
              f' -> {fix["exit_price"]:.4f} on {fix["exit_date"]}'
              f'   pnl {fix["was_pnl"]:+.2f} -> {fix["realized_pnl"]:+.2f}'
              f' ({fix["realized_pnl_pct"]:+.2f}%)')
    for trade in unresolved:
        print(f'  {trade.get("ticker"):<6} no matching filled sell at Alpaca; '
              f'left alone for a human')
    if not fixes:
        print('No correction could be sourced from Alpaca; nothing changed.')
        return 1
    delta = sum(f['realized_pnl'] - f['was_pnl'] for f in fixes)
    print(f'Realised total would move by {delta:+,.2f}.')

    if not args.write:
        print('Dry run. Nothing written. Pass --write to apply.')
        return 0

    _portfolio.save_portfolio(app, apply(ledger, fixes))
    print(f'Wrote {len(fixes)} corrected exit(s) to {app.PORTFOLIO_JSON}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
