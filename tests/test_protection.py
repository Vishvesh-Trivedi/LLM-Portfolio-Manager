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
import shutil
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
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


class MissedSessionsAreNoticed(unittest.TestCase):
    """A trading day that was never screened must not vanish quietly.

    processed_sessions is only ever asked whether today is done. When GitHub
    dropped the whole schedule on 2026-09-18 that session was skipped in
    silence, and the next run simply moved on to the next day.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        self.friday = datetime(2026, 9, 25, 17, tzinfo=ZoneInfo('America/New_York'))

    def missed(self, processed, now=None):
        with patch.object(self.app, '_session_date', return_value='2026-09-25'):
            return self.app._missed_sessions({'processed_sessions': processed},
                                             now=now or self.friday)

    def test_a_weekday_that_was_never_screened_is_reported(self):
        self.assertEqual(self.missed(['2026-09-21', '2026-09-22']),
                         ['2026-09-23', '2026-09-24'])

    def test_an_unbroken_run_of_sessions_reports_nothing(self):
        self.assertEqual(
            self.missed(['2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24']), [])

    def test_a_weekend_is_not_a_gap(self):
        """Saturday and Sunday were never sessions, so they cannot be missed."""
        monday = datetime(2026, 9, 21, 17, tzinfo=ZoneInfo('America/New_York'))
        with patch.object(self.app, '_session_date', return_value='2026-09-21'):
            self.assertEqual(
                self.app._missed_sessions({'processed_sessions': ['2026-09-18']},
                                          now=monday), [])

    def test_a_new_ledger_does_not_report_the_days_before_it_existed(self):
        self.assertEqual(self.missed([]), [])
        # Nothing at or before the first processed session is history, not a gap.
        self.assertEqual(self.missed(['2026-09-24']), [])

    def test_today_is_not_reported_as_missed_by_the_run_processing_it(self):
        self.assertNotIn('2026-09-25', self.missed(['2026-09-21']))

    def test_a_holiday_is_not_a_gap(self):
        """2026-09-07 is Labor Day; the Friday before it is a real session."""
        tuesday = datetime(2026, 9, 8, 17, tzinfo=ZoneInfo('America/New_York'))
        with patch.object(self.app, '_session_date', return_value='2026-09-08'):
            gaps = self.app._missed_sessions({'processed_sessions': ['2026-09-04']},
                                             now=tuesday)
        self.assertNotIn('2026-09-07', gaps)
        self.assertNotIn('2026-09-05', gaps)
        self.assertNotIn('2026-09-06', gaps)


class ReplacingProtectionNeverLeavesItNaked(unittest.TestCase):
    """Run 102: MTD's trailing stop moved up, and MTD ended up with nothing.

    Alpaca accepts a cancel immediately but settles it asynchronously. Until it
    settles the old order still reserves the shares, so the replacement is
    rejected for insufficient quantity - and the holding, which had protection
    a moment earlier, now has none.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        self.position = {'trade_id': 't1', 'ticker': 'MTD', 'shares': 18,
                         'stop_price': 1380.85, 'target_price': 1502.46}
        # Protection exists, but at the old stop - so it must be replaced.
        self.existing = {'MTD': {'order_id': 'old-1', 'qty': 18,
                                 'stop': 1353.69, 'limit': 1502.46}}

    def run_protect(self, submit, released=True, cancelled=True):
        calls = {'await': [], 'submit': 0}

        def _submit(*args, **kwargs):
            calls['submit'] += 1
            return submit(calls['submit'])

        broker = self.app._alpaca
        with ExitStack() as stack:
            for name, kw in (
                ('trading_enabled', {'return_value': True}),
                ('protective_orders_by_symbol', {'return_value': dict(self.existing)}),
                ('positions_by_symbol', {'return_value': {'MTD': 18}}),
                ('cancel_order', {'return_value': cancelled}),
            ):
                stack.enter_context(patch.object(broker, name, **kw))
            stack.enter_context(patch.object(
                broker, 'await_order_released',
                side_effect=lambda oid, *a, **k: calls['await'].append(oid) or released))
            stack.enter_context(patch.object(broker, 'submit_protective_oco',
                                             side_effect=_submit))
            stack.enter_context(patch.object(self.app, '_BROKER_SYNC_OK', [True]))
            stack.enter_context(patch.object(self.app, 'time'))  # no real sleeping
            degraded = []
            stack.enter_context(patch.object(self.app, '_degrade',
                                             side_effect=degraded.append))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.app.protect_positions({'positions': [self.position]})
        return calls, degraded

    def test_the_cancel_is_confirmed_before_the_replacement_is_sent(self):
        calls, degraded = self.run_protect(lambda n: {'id': 'new-1'})
        self.assertEqual(calls['await'], ['old-1'])
        self.assertEqual(degraded, [])

    def test_a_transient_refusal_is_retried_rather_than_left_naked(self):
        """One rejection used to mean the position stayed unprotected."""
        calls, degraded = self.run_protect(
            lambda n: None if n == 1 else {'id': 'new-1'})
        self.assertEqual(calls['submit'], 2)
        self.assertEqual(degraded, [])

    def test_a_persistent_refusal_is_still_reported(self):
        """Retrying must not hide a real refusal."""
        calls, degraded = self.run_protect(lambda n: None)
        self.assertEqual(calls['submit'], 3)
        self.assertIn('protection_rejected:MTD', degraded)


class OneMessagePerRun(unittest.TestCase):

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        # write_run_health writes a file; the import-time temp dir is long gone.
        self.output = tempfile.mkdtemp(prefix='health-')
        self.addCleanup(shutil.rmtree, self.output, ignore_errors=True)
        patcher = patch.object(app, 'DRIVE_FOLDER', self.output)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_health_embed_is_not_sent_alongside_the_digest(self):
        """Run 102 posted the digest and then a health embed saying the same."""
        with patch.object(self.app, '_CAPTURE_EVENTS', [True]),                 patch.object(self.app, 'send_health_alert') as health,                 redirect_stdout(io.StringIO()):
            self.app.write_run_health(None)
        health.assert_not_called()

    def test_an_explicit_caller_outside_a_run_still_gets_it(self):
        with patch.object(self.app, '_CAPTURE_EVENTS', [False]),                 patch.object(self.app, 'send_health_alert') as health,                 redirect_stdout(io.StringIO()):
            self.app.write_run_health(None)
        health.assert_called_once()


if __name__ == '__main__':
    unittest.main()
