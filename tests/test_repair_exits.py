"""Tests for rebuilding exits that were recorded at their own entry price.

A repair that guesses is worse than a record that is visibly wrong, so these
pin both halves: that the three real cases are found and priced from Alpaca's
actual sell fill, and that nothing else is touched.
"""

import copy
import socket
import unittest
from unittest.mock import patch

import repair_exits


def order(**kwargs):
    base = {'order_id': 'o-sell', 'client_order_id': 'lpm-S-x-abc', 'symbol': 'GILD',
            'side': 'sell', 'status': 'filled', 'qty': 98, 'filled_qty': 98,
            'filled_avg_price': 158.0, 'filled_at': '2026-10-01T19:55:00Z'}
    base.update(kwargs)
    return base


def mispriced(**kwargs):
    """A closed trade carrying the exact signature the bug left behind."""
    base = {'ticker': 'GILD', 'shares': 98, 'entry_price': 149.5798,
            'cost_basis': 14658.82, 'entry_date': '2026-09-21',
            'exit_price': 149.5798, 'exit_date': '2026-09-21',
            'exit_value': 14658.82, 'realized_pnl': 0.0, 'realized_pnl_pct': 0.0,
            'reason': 'broker_confirmed_exit', 'result': 'Neutral',
            'broker_order_id': 'ded5428d'}
    base.update(kwargs)
    return base


class TheSignatureIsNarrow(unittest.TestCase):
    """It must not sweep up trades that merely finished flat."""

    def test_the_three_live_records_are_recognised(self):
        for ticker, entry, basis, shares in (('GILD', 149.5798, 14658.82, 98),
                                             ('CDNS', 322.1806, 17397.75, 54),
                                             ('DHR', 226.78, 13380.02, 59)):
            with self.subTest(ticker=ticker):
                self.assertTrue(repair_exits.suspect(
                    mispriced(ticker=ticker, entry_price=entry, exit_price=entry,
                              cost_basis=basis, shares=shares)))

    def test_a_genuine_winner_is_not_touched(self):
        self.assertFalse(repair_exits.suspect(
            mispriced(exit_price=158.0, realized_pnl=825.18, result='Win')))

    def test_a_genuine_loser_is_not_touched(self):
        self.assertFalse(repair_exits.suspect(
            mispriced(exit_price=140.0, realized_pnl=-938.0, result='Loss')))

    def test_a_trade_that_really_closed_flat_by_another_route_is_not_touched(self):
        """Only a broker-confirmed exit can carry this bug."""
        self.assertFalse(repair_exits.suspect(mispriced(reason='stop_loss')))

    def test_a_sold_at_exactly_the_entry_but_nonzero_pnl_is_not_touched(self):
        """Fees or a partial would make the P&L nonzero; leave it alone."""
        self.assertFalse(repair_exits.suspect(mispriced(realized_pnl=-3.20)))


class TheExitComesFromAlpaca(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def ledger(self, trades):
        return {'closed_trades': copy.deepcopy(trades), 'positions': [],
                'pending_orders': [], 'cash': 79526.76,
                'total_realized_pnl': sum(t.get('realized_pnl', 0) for t in trades)}

    def test_it_prices_the_exit_from_the_filled_sell(self):
        book = self.ledger([mispriced()])
        fixes, unresolved = repair_exits.repair(book, [order()])
        self.assertEqual(unresolved, [])
        self.assertEqual(len(fixes), 1)
        fix = fixes[0]
        self.assertAlmostEqual(fix['exit_price'], 158.0, places=2)
        self.assertEqual(fix['exit_date'], '2026-10-01')
        # 98 x 158.00 = 15,484.00 against a 14,658.82 basis.
        self.assertAlmostEqual(fix['realized_pnl'], 825.18, places=2)

    def test_it_ignores_the_buy_that_caused_the_bug(self):
        """The buy sits in the same list and must never be matched."""
        buy = order(order_id='o-buy', client_order_id='lpm-B-x-abc', side='buy',
                    filled_avg_price=149.5798, filled_at='2026-09-21T13:30:05Z')
        fixes, _ = repair_exits.repair(self.ledger([mispriced()]), [buy, order()])
        self.assertAlmostEqual(fixes[0]['exit_price'], 158.0, places=2)

    def test_a_sell_before_the_entry_belongs_to_an_earlier_holding(self):
        earlier = order(order_id='o-old', filled_avg_price=120.0,
                        filled_at='2026-08-10T19:55:00Z')
        fixes, _ = repair_exits.repair(self.ledger([mispriced()]), [earlier, order()])
        self.assertAlmostEqual(fixes[0]['exit_price'], 158.0, places=2)

    def test_a_sell_for_a_different_share_count_is_not_this_position(self):
        other = order(order_id='o-other', filled_qty=40, qty=40, filled_avg_price=99.0)
        fixes, unresolved = repair_exits.repair(self.ledger([mispriced()]), [other])
        self.assertEqual(fixes, [])
        self.assertEqual(len(unresolved), 1)

    def test_with_no_matching_sell_it_refuses_to_invent_one(self):
        fixes, unresolved = repair_exits.repair(self.ledger([mispriced()]), [])
        self.assertEqual(fixes, [])
        self.assertEqual([t['ticker'] for t in unresolved], ['GILD'])

    def test_applying_rewrites_the_trade_and_rederives_the_total(self):
        book = self.ledger([mispriced(),
                            mispriced(ticker='MTD', exit_price=1502.0,
                                      realized_pnl=1792.58, result='Win',
                                      reason='take_profit')])
        fixes, _ = repair_exits.repair(book, [order()])
        repair_exits.apply(book, fixes)
        gild = next(t for t in book['closed_trades'] if t['ticker'] == 'GILD')
        self.assertAlmostEqual(gild['realized_pnl'], 825.18, places=2)
        self.assertEqual(gild['exit_date'], '2026-10-01')
        self.assertEqual(gild['result'], 'Win')
        self.assertEqual(gild['broker_order_id'], 'o-sell')
        self.assertEqual(gild['exit_repaired_from'], 'alpaca_sell_fill')
        # Re-derived from the trades, not adjusted, so it cannot drift from them.
        self.assertAlmostEqual(book['total_realized_pnl'], 825.18 + 1792.58, places=2)

    def test_repair_does_not_mutate_until_apply_is_called(self):
        book = self.ledger([mispriced()])
        before = copy.deepcopy(book)
        repair_exits.repair(book, [order()])
        self.assertEqual(book, before)

    def test_a_dry_run_writes_nothing(self):
        saved = []
        book = self.ledger([mispriced()])
        with patch.object(repair_exits._portfolio, 'load_portfolio', return_value=book), \
                patch.object(repair_exits._portfolio, 'save_portfolio',
                             side_effect=lambda app, pf: saved.append(pf)), \
                patch.object(repair_exits._alpaca, 'trading_enabled', return_value=True), \
                patch.object(repair_exits._alpaca, 'fetch_orders', return_value=[order()]), \
                patch.object(repair_exits._alpaca, 'normalize_order', side_effect=lambda o: o), \
                patch.object(repair_exits.Path, 'resolve', lambda self, strict=False: self):
            code = repair_exits.main([])
        self.assertEqual(code, 0)
        self.assertEqual(saved, [], 'a dry run must not save')

    def test_it_refuses_rather_than_guess_when_alpaca_is_unreadable(self):
        book = self.ledger([mispriced()])
        with patch.object(repair_exits._portfolio, 'load_portfolio', return_value=book), \
                patch.object(repair_exits._portfolio, 'save_portfolio',
                             side_effect=AssertionError('must not save')), \
                patch.object(repair_exits._alpaca, 'trading_enabled', return_value=True), \
                patch.object(repair_exits._alpaca, 'fetch_orders', return_value=None), \
                patch.object(repair_exits.Path, 'resolve', lambda self, strict=False: self):
            self.assertEqual(repair_exits.main(['--write']), 1)


if __name__ == '__main__':
    unittest.main()
