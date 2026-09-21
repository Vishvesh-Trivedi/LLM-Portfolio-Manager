"""A position must never be held at Alpaca with nothing protecting it.

GILD filled at Monday's open and sat all day with no stop and no target. The
entry was a plain market order, and protection was only ever attached by
protect_positions, which runs once after the close. Nothing reported it,
because protect_positions walks the ledger's positions and the ledger still
recorded GILD as a pending order rather than a holding.

Two changes, covered here:
  * a buy is submitted as a bracket, so Alpaca attaches the stop and target to
    the fill itself and the naked window never opens;
  * protection is audited against what Alpaca actually holds, so a holding the
    ledger has not booked can no longer hide.
"""

import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

import screener_alpaca as alpaca


class BuyIsNeverProtection(unittest.TestCase):
    """The reverse of the bug that once doubled a position.

    A bracket entry carries order_class 'bracket'. Counting it as protection
    hides it from open_order_shares_by_symbol, the incoming shares go
    uncounted, and the mirror buys the position a second time.
    """

    def test_a_bracket_buy_is_an_entry_not_protection(self):
        order = {'side': 'buy', 'order_class': 'bracket', 'symbol': 'GILD',
                 'legs': [{'stop_price': '145.00'}]}
        self.assertFalse(alpaca.is_protective(order))

    def test_a_protective_sell_is_still_protection(self):
        for order in ({'side': 'sell', 'order_class': 'oco'},
                      {'side': 'sell', 'order_class': 'bracket'},
                      {'side': 'sell', 'legs': [{'stop_price': '1.0'}]}):
            with self.subTest(order=order):
                self.assertTrue(alpaca.is_protective(order))

    def test_a_pending_bracket_buy_is_counted_as_incoming_shares(self):
        orders = [{'symbol': 'GILD', 'side': 'buy', 'qty': '98', 'filled_qty': '0',
                   'order_class': 'bracket', 'legs': [{'stop_price': '145.73'}]}]
        with patch.object(alpaca, 'list_orders', return_value=orders):
            self.assertEqual(alpaca.open_order_shares_by_symbol(), {'GILD': 98})

    def test_a_resting_protective_sell_is_not_counted_as_shares_leaving(self):
        orders = [{'symbol': 'MTD', 'side': 'sell', 'qty': '18', 'filled_qty': '0',
                   'order_class': 'oco'}]
        with patch.object(alpaca, 'list_orders', return_value=orders):
            self.assertEqual(alpaca.open_order_shares_by_symbol(), {})


class EntryCarriesItsProtection(unittest.TestCase):

    def submit(self, side='buy', **kwargs):
        sent = {}

        def request(method, url, body=None, **_):
            sent.update(body or {})
            return {'id': 'order-1'}

        with patch.object(alpaca, '_request', side_effect=request), \
                patch.object(alpaca, '_start_trade_updates_stream'), \
                patch.object(alpaca, '_upsert_order_ledger'):
            alpaca.submit_market_order('GILD', 98, side, ref='r1', **kwargs)
        return sent

    def test_a_buy_with_levels_becomes_a_bracket(self):
        body = self.submit(stop_price=145.73, target_price=159.86)
        self.assertEqual(body['order_class'], 'bracket')
        self.assertEqual(body['stop_loss'], {'stop_price': '145.73'})
        self.assertEqual(body['take_profit'], {'limit_price': '159.86'})
        self.assertEqual(body['side'], 'buy')
        self.assertEqual(body['qty'], '98')

    def test_a_buy_without_levels_stays_a_plain_order(self):
        self.assertNotIn('order_class', self.submit())

    def test_an_unusable_level_sends_no_bracket_rather_than_a_rejected_one(self):
        """Better naked and reported than an order Alpaca refuses outright."""
        for stop, target in ((0, 10), (10, 10), (20, 10), (-1, 10),
                             (None, 10), (5, None), ('x', 10), (5, 'y')):
            with self.subTest(stop=stop, target=target):
                self.assertNotIn('order_class',
                                 self.submit(stop_price=stop, target_price=target))

    def test_a_sell_never_carries_a_bracket(self):
        body = self.submit(side='sell', stop_price=145.73, target_price=159.86)
        self.assertNotIn('order_class', body)


class EntryLevels(unittest.TestCase):
    """Where the bracket's prices come from before a fill price exists."""

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    def test_a_pending_order_supplies_levels_from_its_estimate(self):
        book = {'pending_orders': [{'ticker': 'GILD', 'estimated_entry': 150.44,
                                    'stop_distance': 4.71, 'target_distance': 9.42}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book),
                         {'GILD': (145.73, 159.86)})

    def test_an_open_position_supplies_its_recorded_levels(self):
        book = {'positions': [{'ticker': 'MTD', 'stop_price': 1353.69,
                               'target_price': 1502.46}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book),
                         {'MTD': (1353.69, 1502.46)})

    def test_a_pending_order_wins_over_a_stale_position_record(self):
        book = {'pending_orders': [{'ticker': 'X', 'estimated_entry': 100.0,
                                    'stop_distance': 5.0, 'target_distance': 10.0}],
                'positions': [{'ticker': 'X', 'stop_price': 1.0, 'target_price': 2.0}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book), {'X': (95.0, 110.0)})

    def test_unusable_records_are_skipped_rather_than_guessed(self):
        book = {'pending_orders': [
            {'ticker': 'A'},
            {'ticker': 'B', 'estimated_entry': 'x', 'stop_distance': 1, 'target_distance': 2},
            {'ticker': 'C', 'estimated_entry': 10, 'stop_distance': 20, 'target_distance': 2},
        ], 'positions': [{'ticker': 'D', 'stop_price': None, 'target_price': 5}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book), {})


class BrokerHoldingsAreAudited(unittest.TestCase):
    """Protection is checked against Alpaca's holdings, not the ledger's."""

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    def protect(self, ledger_positions, held, existing=None):
        events = []
        # protect_positions refuses to act unless trading is on and the
        # ledger has been reconciled: a stop derived from a ledger known to
        # be wrong could sell at the wrong level.
        broker = self.app._alpaca
        with ExitStack() as stack:
            for target, kwargs in (
                ('trading_enabled', {'return_value': True}),
                ('protective_orders_by_symbol', {'return_value': dict(existing or {})}),
                ('positions_by_symbol', {'return_value': held}),
                ('submit_protective_oco', {'return_value': {'id': 'p1'}}),
                ('cancel_order', {'return_value': True}),
            ):
                stack.enter_context(patch.object(broker, target, **kwargs))
            stack.enter_context(patch.object(self.app, '_BROKER_SYNC_OK', [True]))
            stack.enter_context(patch.object(self.app, '_degrade',
                                             side_effect=events.append))
            stack.enter_context(redirect_stdout(io.StringIO()))
            produced = self.app.protect_positions({'positions': ledger_positions})
        return events, produced

    def test_a_holding_the_ledger_has_not_booked_is_reported(self):
        """Exactly GILD's state: filled at the broker, still 'pending' here."""
        reasons, events = self.protect([], {'GILD': 98})
        self.assertIn('broker_holding_unprotected:GILD', reasons)
        self.assertTrue(any(e['kind'] == 'unprotected' and e['symbol'] == 'GILD'
                            for e in events))

    def test_the_run_is_not_called_healthy_while_something_is_exposed(self):
        with patch.object(self.app._HEALTH, 'stage') as stage:
            self.protect([], {'GILD': 98})
        success, detail = stage.call_args.args[1], stage.call_args.args[2]
        self.assertFalse(success)
        self.assertIn('GILD', detail)

    def test_a_holding_that_is_already_protected_is_not_reported(self):
        reasons, _ = self.protect(
            [], {'MTD': 18}, existing={'MTD': {'order_id': 'o1', 'qty': 18,
                                               'stop': 1353.69, 'limit': 1502.46}})
        self.assertEqual(reasons, [])

    def test_a_position_protected_during_this_run_is_not_reported(self):
        reasons, _ = self.protect(
            [{'ticker': 'MTD', 'stop_price': 1353.69, 'target_price': 1502.46}],
            {'MTD': 18})
        self.assertEqual(reasons, [])

    def test_nothing_held_means_nothing_to_report(self):
        reasons, _ = self.protect([], {})
        self.assertEqual(reasons, [])


if __name__ == '__main__':
    unittest.main()
