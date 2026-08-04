import os
import unittest

os.environ['SCREENER_SKIP_UNIVERSE_FETCH'] = '1'

import numpy as np
import pandas as pd

import LLM_Portfolio_Manager as screener


class MarketDataCleaningTests(unittest.TestCase):
    @staticmethod
    def _history_with_incomplete_tail():
        dates = pd.date_range('2026-01-01', periods=66, freq='B')
        closes = np.linspace(100.0, 125.0, 66)
        frame = pd.DataFrame(
            {
                'Open': closes - 0.4,
                'High': closes + 1.0,
                'Low': closes - 1.0,
                'Close': closes,
                'Volume': np.linspace(1_000_000, 1_500_000, 66),
            },
            index=dates,
        )
        frame.loc[dates[-1], ['Open', 'High', 'Low', 'Close', 'Volume']] = np.nan
        return frame

    def test_clean_ohlcv_drops_incomplete_yahoo_tail(self):
        frame = self._history_with_incomplete_tail()

        cleaned = screener._clean_ohlcv(frame)

        self.assertIsNotNone(cleaned)
        self.assertEqual(len(cleaned), 65)
        self.assertFalse(cleaned[['High', 'Low', 'Close', 'Volume']].isna().any().any())

    def test_indicators_survive_incomplete_yahoo_tail(self):
        indicators = screener.compute_indicators(self._history_with_incomplete_tail())

        self.assertIsNotNone(indicators)
        self.assertGreater(indicators['price'], 0)
        self.assertTrue(np.isfinite(indicators['adx']))
        self.assertTrue(np.isfinite(indicators['vol_ratio']))

    def test_valid_closes_ignores_nan_and_infinity(self):
        frame = pd.DataFrame({'Close': [100.0, np.nan, np.inf, 101.0]})

        closes = screener._valid_closes(frame)

        self.assertEqual(closes.tolist(), [100.0, 101.0])


if __name__ == '__main__':
    unittest.main()
