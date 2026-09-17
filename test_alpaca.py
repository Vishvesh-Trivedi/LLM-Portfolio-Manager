"""Offline tests for screener_alpaca — pure logic only, no network access."""

import os
import unittest

import numpy as np
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


if __name__ == '__main__':
    unittest.main()
