"""Offline tests for screener_alpaca — pure logic only, no network access."""

import os
import unittest
from unittest import mock

import pandas as pd

import screener_alpaca as alpaca


class EnableFlagsTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'SCREENER_LIVE_BROKER')}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_data_disabled_without_credentials(self):
        self.assertFalse(alpaca.data_enabled())
        self.assertFalse(alpaca.trading_enabled())

    def test_data_enabled_with_credentials(self):
        os.environ['ALPACA_API_KEY'] = 'k'
        os.environ['ALPACA_SECRET_KEY'] = 's'
        self.assertTrue(alpaca.data_enabled())
        # Trading still requires the explicit opt-in flag.
        self.assertFalse(alpaca.trading_enabled())

    def test_trading_requires_opt_in_flag(self):
        os.environ['ALPACA_API_KEY'] = 'k'
        os.environ['ALPACA_SECRET_KEY'] = 's'
        os.environ['SCREENER_LIVE_BROKER'] = '1'
        self.assertTrue(alpaca.trading_enabled())

    def test_daily_bars_noop_without_credentials(self):
        self.assertEqual(alpaca.daily_bars(['AAPL'], start='2026-01-01'), {})


class BarParsingTest(unittest.TestCase):
    def test_bars_to_frame_shape_and_session_dates(self):
        bars = [
            {'t': '2026-09-15T04:00:00Z', 'o': 10.0, 'h': 11.0, 'l': 9.5, 'c': 10.5, 'v': 1000},
            {'t': '2026-09-16T04:00:00Z', 'o': 10.5, 'h': 12.0, 'l': 10.0, 'c': 11.8, 'v': 2000},
        ]
        frame = alpaca._bars_to_frame(bars)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertIsInstance(frame.index, pd.DatetimeIndex)
        self.assertIsNone(frame.index.tz)
        self.assertListEqual(list(frame.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertEqual([d.isoformat() for d in frame.index.date],
                         ['2026-09-15', '2026-09-16'])
        self.assertAlmostEqual(frame['Close'].iloc[-1], 11.8)

    def test_bars_to_frame_drops_incomplete_rows(self):
        bars = [
            {'t': '2026-09-16T04:00:00Z', 'o': 10.0, 'h': 11.0, 'l': 9.5, 'c': None, 'v': 1000},
            {'t': '2026-09-17T04:00:00Z', 'o': 10.0, 'h': 11.0, 'l': 9.5, 'c': 10.5, 'v': 1000},
        ]
        frame = alpaca._bars_to_frame(bars)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.index.date[0].isoformat(), '2026-09-17')

    def test_bars_to_frame_empty(self):
        self.assertIsNone(alpaca._bars_to_frame([]))
        self.assertIsNone(alpaca._bars_to_frame(None))


class ReconciliationPlanTest(unittest.TestCase):
    def test_buy_when_broker_short(self):
        actions = alpaca.plan_reconciliation({'AAPL': 10}, {})
        self.assertEqual(actions, [('buy', 'AAPL', 10)])

    def test_sell_when_ledger_exited(self):
        actions = alpaca.plan_reconciliation({}, {'MSFT': 5})
        self.assertEqual(actions, [('sell', 'MSFT', 5)])

    def test_partial_adjustments_both_directions(self):
        actions = alpaca.plan_reconciliation({'AAPL': 10, 'MSFT': 3}, {'AAPL': 4, 'MSFT': 8})
        self.assertIn(('buy', 'AAPL', 6), actions)
        self.assertIn(('sell', 'MSFT', 5), actions)

    def test_in_sync_produces_no_actions(self):
        self.assertEqual(alpaca.plan_reconciliation({'AAPL': 10}, {'AAPL': 10}), [])


class RedactionTest(unittest.TestCase):
    def test_credentials_never_leak(self):
        os.environ['ALPACA_API_KEY'] = 'SECRET_ID'
        os.environ['ALPACA_SECRET_KEY'] = 'SECRET_VALUE'
        try:
            cleaned = alpaca._redact('error for SECRET_ID using SECRET_VALUE')
            self.assertNotIn('SECRET_ID', cleaned)
            self.assertNotIn('SECRET_VALUE', cleaned)
        finally:
            os.environ.pop('ALPACA_API_KEY', None)
            os.environ.pop('ALPACA_SECRET_KEY', None)


class LiveBrokerFlagTest(unittest.TestCase):
    """The costliest misconfiguration in the system deserves a forgiving parse."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'SCREENER_LIVE_BROKER')}
        os.environ.update(ALPACA_API_KEY='k', ALPACA_SECRET_KEY='s')

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_every_affirmative_spelling_enables_broker_mode(self):
        for value in ('1', 'true', 'TRUE', 'True', 'yes', 'YES', 'on', ' on '):
            os.environ['SCREENER_LIVE_BROKER'] = value
            self.assertTrue(alpaca.trading_enabled(), repr(value))

    def test_anything_else_leaves_broker_mode_off(self):
        for value in ('0', 'false', 'no', 'off', '', '   ', 'maybe', '2'):
            os.environ['SCREENER_LIVE_BROKER'] = value
            self.assertFalse(alpaca.trading_enabled(), repr(value))

    def test_flag_alone_is_not_enough_without_credentials(self):
        os.environ['SCREENER_LIVE_BROKER'] = '1'
        for missing in ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY'):
            saved = os.environ.pop(missing)
            self.assertFalse(alpaca.trading_enabled())
            os.environ[missing] = saved


class ClientOrderIdTest(unittest.TestCase):
    """The id carries the ledger record so a fill can be matched back exactly."""

    def test_ref_is_embedded_and_round_trips(self):
        coid = alpaca._client_order_id('MTD', 'buy',
                                       ref='e904bd8b-ef79-4a81-bfd2-a61eb4a563d9')
        parsed = alpaca.parse_client_order_id(coid)
        self.assertEqual(parsed['side'], 'buy')
        self.assertEqual(parsed['symbol'], 'MTD')
        self.assertEqual(parsed['ref'], 'e904bd8bef794a81bfd2a61eb4a563d9')
        self.assertLessEqual(len(coid), 128)

    def test_sell_side_is_distinguished(self):
        coid = alpaca._client_order_id('ABC', 'sell', ref='tradeid1')
        self.assertEqual(alpaca.parse_client_order_id(coid)['side'], 'sell')

    def test_missing_ref_still_produces_a_unique_id(self):
        first = alpaca._client_order_id('ABC', 'buy')
        second = alpaca._client_order_id('ABC', 'buy')
        self.assertNotEqual(first, second)

    def test_legacy_and_foreign_ids_do_not_parse(self):
        # Orders from an older build or placed by hand must not be mistaken for
        # ours; the caller falls back to weak symbol/side matching for those.
        for value in ('lpm-B-MTD-1758134400-a1b2c3d4', 'manual-order-1', '', None):
            self.assertIsNone(alpaca.parse_client_order_id(value))

    def test_ledger_ref_reads_a_normalized_order(self):
        self.assertEqual(alpaca.ledger_ref({'client_order_id': 'lpm-S-ABC-tid42'}), 'tid42')
        self.assertEqual(alpaca.ledger_ref({'client_order_id': 'manual'}), '')


class SnapshotTest(unittest.TestCase):
    """broker_snapshot must never report a partial read as usable state."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'SCREENER_LIVE_BROKER')}
        os.environ.update(ALPACA_API_KEY='k', ALPACA_SECRET_KEY='s',
                          SCREENER_LIVE_BROKER='1')

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def patch(self, account, positions, orders):
        return (mock.patch.object(alpaca, 'fetch_account', return_value=account),
                mock.patch.object(alpaca, 'fetch_positions', return_value=positions),
                mock.patch.object(alpaca, 'fetch_orders', return_value=orders))

    def snapshot(self, account, positions, orders):
        a, p, o = self.patch(account, positions, orders)
        with a, p, o:
            return alpaca.broker_snapshot()

    def test_complete_read_is_ok_and_normalized(self):
        snap = self.snapshot(
            {'cash': '74740.92', 'equity': '100128.84'},
            [{'symbol': 'mtd', 'qty': '18', 'avg_entry_price': '1403.28'}],
            [{'id': 'o1', 'symbol': 'MTD', 'side': 'buy', 'status': 'filled',
              'qty': '18', 'filled_qty': '18', 'filled_avg_price': '1403.28'}])
        self.assertTrue(snap['ok'])
        self.assertEqual(snap['cash'], 74740.92)
        self.assertEqual(snap['positions']['MTD']['qty'], 18)
        self.assertEqual(snap['orders'][0]['filled_avg_price'], 1403.28)

    def test_empty_account_is_readable_and_ok(self):
        snap = self.snapshot({'cash': '100000', 'equity': '100000'}, [], [])
        self.assertTrue(snap['ok'])
        self.assertEqual(snap['positions'], {})

    def test_any_failed_read_makes_the_whole_snapshot_unusable(self):
        for account, positions, orders in (
            (None, [], []), ({'cash': '1'}, None, []), ({'cash': '1'}, [], None),
        ):
            snap = self.snapshot(account, positions, orders)
            self.assertFalse(snap['ok'])
            self.assertIn('unreadable', snap['error'])

    def test_zero_quantity_position_is_dropped(self):
        snap = self.snapshot({'cash': '1', 'equity': '1'},
                             [{'symbol': 'ABC', 'qty': '0'}], [])
        self.assertEqual(snap['positions'], {})

    def test_disabled_trading_yields_not_ok(self):
        os.environ['SCREENER_LIVE_BROKER'] = '0'
        snap = alpaca.broker_snapshot()
        self.assertFalse(snap['ok'])
        self.assertEqual(snap['error'], 'trading_disabled')

    def test_unexpected_exception_is_contained(self):
        with mock.patch.object(alpaca, 'fetch_account', side_effect=RuntimeError('x')):
            snap = alpaca.broker_snapshot()
        self.assertFalse(snap['ok'])


class SubmitRefTest(unittest.TestCase):
    def test_submit_passes_the_ref_into_the_client_order_id(self):
        captured = {}

        def fake_request(method, url, params=None, body=None):
            captured.update(body or {})
            return {'id': 'o1'}

        with mock.patch.object(alpaca, '_request', fake_request),                 mock.patch.object(alpaca, '_start_trade_updates_stream', return_value=False):
            alpaca.submit_market_order('MTD', 18, 'buy', ref='abc-123')
        self.assertEqual(alpaca.parse_client_order_id(captured['client_order_id'])['ref'],
                         'abc123')


if __name__ == '__main__':
    unittest.main()
