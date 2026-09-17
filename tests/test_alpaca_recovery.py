import unittest
from unittest.mock import patch

import screener_alpaca as alpaca


class AlpacaRecoveryTests(unittest.TestCase):
    def setUp(self):
        alpaca._ORDER_LEDGER[:] = []
        alpaca._TRADE_UPDATES[:] = []

    def tearDown(self):
        alpaca._ORDER_LEDGER[:] = []
        alpaca._TRADE_UPDATES[:] = []

    def test_sync_order_statuses_merges_existing_and_polled_snapshots(self):
        existing = [{'client_order_id': 'cid-1', 'symbol': 'MTD', 'status': 'accepted'}]
        orders = [{
            'id': 'oid-1',
            'client_order_id': 'cid-1',
            'symbol': 'MTD',
            'status': 'filled',
            'filled_qty': '3',
            'filled_avg_price': '12.34',
            'qty': '3',
            'side': 'buy',
        }]

        with patch.object(alpaca, 'list_orders', return_value=orders):
            ledger = alpaca.sync_order_statuses(existing)

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]['client_order_id'], 'cid-1')
        self.assertEqual(ledger[0]['status'], 'filled')
        self.assertEqual(ledger[0]['filled_qty'], '3')
        self.assertEqual(ledger[0]['source'], 'poll')

    def test_submit_market_order_records_submit_snapshot(self):
        order = {
            'id': 'oid-2',
            'client_order_id': 'cid-2',
            'symbol': 'NVDA',
            'status': 'accepted',
            'filled_qty': '0',
            'filled_avg_price': '',
            'qty': '10',
            'side': 'buy',
        }

        with patch.object(alpaca, '_request', return_value=order), \
             patch.object(alpaca, '_start_trade_updates_stream', return_value=True):
            result = alpaca.submit_market_order('NVDA', 10, 'buy')

        self.assertEqual(result, order)
        ledger = alpaca.order_ledger()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]['client_order_id'], 'cid-2')
        self.assertEqual(ledger[0]['status'], 'accepted')
        self.assertEqual(ledger[0]['source'], 'submit')


if __name__ == '__main__':
    unittest.main()
