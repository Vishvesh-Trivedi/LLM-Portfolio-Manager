import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['SCREENER_SKIP_UNIVERSE_FETCH'] = '1'

import LLM_Portfolio_Manager as screener


class AlpacaRecoveryTests(unittest.TestCase):
    def test_sync_order_statuses_updates_order_ledger_and_pending_status(self):
        portfolio = {
            'pending_orders': [
                {'ticker': 'MTD', 'client_order_id': 'cid-1', 'status': 'PENDING'}
            ]
        }
        orders = [
            {
                'id': 'oid-1',
                'client_order_id': 'cid-1',
                'symbol': 'MTD',
                'status': 'filled',
                'filled_qty': '3',
                'filled_avg_price': '12.34',
                'qty': '3',
                'side': 'buy',
            }
        ]

        with patch.object(screener, '_alpaca_trading_enabled', return_value=True), \
             patch.object(screener, '_alpaca_list_orders', return_value=orders):
            synced, changed = screener._alpaca_sync_order_statuses(portfolio)

        self.assertTrue(changed)
        self.assertEqual(synced['alpaca_order_updates']['cid-1']['source'], 'poll')
        self.assertEqual(synced['alpaca_order_updates']['cid-1']['status'], 'filled')
        self.assertEqual(synced['pending_orders'][0]['status'], 'FILLED')
        self.assertEqual(synced['pending_orders'][0]['broker_filled_qty'], 3.0)
        self.assertEqual(synced['pending_orders'][0]['broker_filled_avg_price'], 12.34)

    def test_open_position_records_submit_snapshot_before_save(self):
        portfolio = {
            'cash': 10_000.0,
            'starting_capital': 10_000.0,
            'positions': [],
            'closed_trades': [],
            'total_realized_pnl': 0.0,
            'created': '2026-09-17',
        }
        broker_order = SimpleNamespace(
            id='oid-2',
            client_order_id='cid-2',
            symbol='NVDA',
            status='accepted',
            filled_qty='0',
            filled_avg_price=None,
            qty='10',
            side='buy',
            updated_at=None,
            submitted_at=None,
        )

        with patch.object(screener, '_alpaca_trading_enabled', return_value=True), \
             patch.object(screener, '_alpaca_submit_market_order', return_value=broker_order), \
             patch.object(screener, '_sharesies_fee', return_value=0.0):
            updated = screener.open_position(portfolio, 'NVDA', 100.0, 1_000.0, 90.0, 120.0)

        self.assertIn('cid-2', updated['alpaca_order_updates'])
        self.assertEqual(updated['alpaca_order_updates']['cid-2']['source'], 'submit')
        self.assertEqual(updated['alpaca_order_updates']['cid-2']['status'], 'accepted')
        self.assertEqual(updated['positions'][0]['ticker'], 'NVDA')


if __name__ == '__main__':
    unittest.main()