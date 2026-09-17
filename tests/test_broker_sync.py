"""Offline tests for broker-authoritative reconciliation planning.

Pure logic only: no network, no app, no files. Every branch that can change a
ledger is pinned here, because this is the module that decides whether a real
position is booked, corrected or deleted.
"""

import copy
import socket
import unittest
from unittest.mock import patch

from screener_broker_sync import plan_broker_sync


def order(**kwargs):
    base = {'order_id': 'o1', 'client_order_id': '', 'symbol': 'ABC', 'side': 'buy',
            'status': 'filled', 'qty': 10, 'filled_qty': 10,
            'filled_avg_price': 100.0}
    base.update(kwargs)
    return base


def pending(**kwargs):
    base = {'id': 'e904bd8b-ef79-4a81-bfd2-a61eb4a563d9', 'ticker': 'ABC',
            'signal_date': '2026-09-16', 'execution_session': '2026-09-17',
            'shares': 10, 'amount_usd': 1000.0, 'estimated_entry': 99.0,
            'stop_distance': 4.0, 'target_distance': 8.0, 'sector': 'Technology',
            'atr': 1.0, 'hold_sessions': 10, 'status': 'PENDING'}
    base.update(kwargs)
    return base


def position(**kwargs):
    base = {'trade_id': 't-1', 'ticker': 'ABC', 'shares': 10, 'entry_price': 100.0,
            'cost_basis': 1000.0, 'entry_date': '2026-09-17'}
    base.update(kwargs)
    return base


def snapshot(**kwargs):
    base = {'ok': True, 'cash': 50000.0, 'equity': 51000.0, 'positions': {},
            'orders': [], 'account_blocked': False}
    base.update(kwargs)
    return base


def ops(plan):
    return [a['op'] for a in plan['actions']]


def kinds(plan):
    return [e['kind'] for e in plan['events']]


def only(plan, op):
    return next(a for a in plan['actions'] if a['op'] == op)


class BrokerSyncTests(unittest.TestCase):
    SESSION = '2026-09-18'

    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def plan(self, ledger, broker):
        frozen = copy.deepcopy(ledger)
        result = plan_broker_sync(ledger, broker, self.SESSION)
        self.assertEqual(ledger, frozen, 'planner must not mutate its inputs')
        return result

    # ── Authority requires the broker to actually answer ──────────────────

    def test_unreadable_snapshot_produces_no_actions(self):
        ledger = {'positions': [position()], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, {'ok': False, 'error': 'unreadable: positions'})
        self.assertFalse(plan['ok'])
        self.assertEqual(plan['actions'], [])
        self.assertEqual(kinds(plan), ['snapshot_failed'])
        self.assertTrue(plan['blocked'])

    def test_empty_account_is_not_confused_with_a_failed_read(self):
        # An genuinely empty, readable account DOES close a ledger position when
        # a priced sell exists — the distinction from a failed read is the point.
        ledger = {'positions': [position()], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[order(side='sell', status='filled',
                                                        filled_qty=10,
                                                        filled_avg_price=110.0)]))
        self.assertTrue(plan['ok'])
        self.assertIn('close_position', ops(plan))

    # ── Pending orders ────────────────────────────────────────────────────

    def test_filled_order_books_broker_price_not_the_estimate(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[order(filled_avg_price=103.25)]))
        action = only(plan, 'fill_pending')
        self.assertEqual(action['shares'], 10)
        self.assertEqual(action['price'], 103.25)
        self.assertFalse(action['partial'])
        event = next(e for e in plan['events'] if e['kind'] == 'fill')
        self.assertAlmostEqual(event['slippage_pct'], (103.25 / 99.0 - 1) * 100)

    def test_partial_fill_books_only_what_executed(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='partially_filled', filled_qty=4, filled_avg_price=101.0)]))
        action = only(plan, 'fill_pending')
        self.assertEqual(action['shares'], 4)
        self.assertTrue(action['partial'])
        self.assertIn('partial_fill', kinds(plan))

    def test_partially_filled_then_canceled_keeps_the_executed_shares(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='canceled', filled_qty=3, filled_avg_price=100.5)]))
        action = only(plan, 'fill_pending')
        self.assertEqual((action['shares'], action['partial']), (3, True))

    def test_rejected_order_opens_no_position_and_blocks_the_run(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='rejected', filled_qty=0, filled_avg_price=None)]))
        self.assertEqual(ops(plan), ['expire_pending', 'set_cash'])
        self.assertIn('rejected', kinds(plan))
        self.assertTrue(plan['blocked'], 'a rejected order must block new trading')

    def test_canceled_without_fill_expires_without_blocking(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='expired', filled_qty=0, filled_avg_price=None)]))
        self.assertIn('expire_pending', ops(plan))
        self.assertFalse(plan['blocked'])

    def test_working_order_is_left_pending(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='accepted', filled_qty=0, filled_avg_price=None)]))
        self.assertEqual(ops(plan), ['set_cash'])
        self.assertIn('working', kinds(plan))

    def test_unknown_future_status_is_treated_as_working_not_canceled(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='some_new_alpaca_state', filled_qty=0, filled_avg_price=None)]))
        self.assertNotIn('expire_pending', ops(plan))
        self.assertIn('working', kinds(plan))

    def test_missing_broker_order_expires_rather_than_inventing_a_fill(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[]))
        self.assertIn('expire_pending', ops(plan))
        self.assertIn('order_missing', kinds(plan))

    def test_fill_without_a_price_is_blocked_not_guessed(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(status='filled', filled_qty=10, filled_avg_price=None)]))
        self.assertNotIn('fill_pending', ops(plan))
        self.assertIn('fill_unpriced', kinds(plan))
        self.assertTrue(plan['blocked'])

    # ── Order matching ────────────────────────────────────────────────────

    def test_tagged_order_matches_its_exact_ledger_record(self):
        first = pending(id='aaaaaaaabbbbccccddddeeeeeeeeeeee', ticker='ABC')
        ledger = {'positions': [], 'pending_orders': [first], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(client_order_id='lpm-B-ABC-ffffffffffffffffffffffffffffffff',
                  filled_avg_price=200.0, order_id='wrong'),
            order(client_order_id='lpm-B-ABC-aaaaaaaabbbbccccddddeeeeeeeeeeee',
                  filled_avg_price=150.0, order_id='right'),
        ]))
        self.assertEqual(only(plan, 'fill_pending')['broker_order_id'], 'right')
        self.assertTrue(next(e for e in plan['events'] if e['kind'] == 'fill')['exact_match'])

    def test_legacy_untagged_order_matches_by_symbol_and_side(self):
        # Orders submitted before ids were embedded still have to be recognized,
        # or a real fill would be expired and then sold back out.
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(client_order_id='lpm-B-ABC-1758134400-a1b2c3d4',
                  filled_avg_price=105.0)]))
        self.assertEqual(only(plan, 'fill_pending')['price'], 105.0)
        self.assertFalse(next(e for e in plan['events'] if e['kind'] == 'fill')['exact_match'])

    def test_order_tagged_for_another_record_is_not_claimed(self):
        ledger = {'positions': [], 'pending_orders': [pending(id='mine')], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(client_order_id='lpm-B-ABC-someoneelsesrecordid')]))
        self.assertIn('expire_pending', ops(plan))
        self.assertIn('order_missing', kinds(plan))

    # ── Held positions ────────────────────────────────────────────────────

    def test_position_absent_at_broker_closes_at_the_real_sell_price(self):
        ledger = {'positions': [position()], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[
            order(side='sell', status='filled', filled_qty=10,
                  filled_avg_price=112.5, order_id='sell-1')]))
        action = only(plan, 'close_position')
        self.assertEqual((action['price'], action['shares']), (112.5, 10))
        self.assertIn('exit_filled', kinds(plan))

    def test_vanished_position_without_a_priced_sell_is_blocked(self):
        ledger = {'positions': [position()], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(orders=[]))
        self.assertEqual(ops(plan), ['set_cash'])
        self.assertIn('position_vanished', kinds(plan))
        self.assertTrue(plan['blocked'])

    def test_share_count_drift_is_corrected_to_the_broker(self):
        ledger = {'positions': [position(shares=10)], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(positions={
            'ABC': {'symbol': 'ABC', 'qty': 7, 'avg_entry_price': 100.0}}))
        self.assertEqual(only(plan, 'resize_position')['shares'], 7)
        self.assertIn('qty_drift', kinds(plan))

    def test_cost_basis_drift_is_corrected_to_the_broker(self):
        ledger = {'positions': [position(entry_price=100.0)], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(positions={
            'ABC': {'symbol': 'ABC', 'qty': 10, 'avg_entry_price': 103.4}}))
        self.assertEqual(only(plan, 'reprice_position')['entry_price'], 103.4)

    def test_sub_cent_price_difference_is_not_churned(self):
        ledger = {'positions': [position(entry_price=100.0)], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(positions={
            'ABC': {'symbol': 'ABC', 'qty': 10, 'avg_entry_price': 100.004}}))
        self.assertNotIn('reprice_position', ops(plan))

    def test_unknown_broker_holding_is_adopted_without_a_sector(self):
        ledger = {'positions': [], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(positions={
            'XYZ': {'symbol': 'XYZ', 'qty': 5, 'avg_entry_price': 50.0,
                    'current_price': 52.0}}))
        action = only(plan, 'adopt_position')
        self.assertEqual((action['symbol'], action['shares']), ('XYZ', 5))
        self.assertIsNone(action['sector'], 'caller must resolve the sector')
        self.assertIn('adopted', kinds(plan))

    def test_symbol_with_a_pending_order_is_not_also_adopted(self):
        ledger = {'positions': [], 'pending_orders': [pending()], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(
            positions={'ABC': {'symbol': 'ABC', 'qty': 10, 'avg_entry_price': 100.0}},
            orders=[order()]))
        self.assertNotIn('adopt_position', ops(plan))

    # ── Cash ──────────────────────────────────────────────────────────────

    def test_cash_is_taken_from_the_broker(self):
        ledger = {'positions': [], 'pending_orders': [], 'cash': 100000.0}
        plan = self.plan(ledger, snapshot(cash=74740.92))
        self.assertEqual(only(plan, 'set_cash')['cash'], 74740.92)
        self.assertIn('cash_drift', kinds(plan))

    def test_invalid_broker_cash_leaves_ledger_cash_alone(self):
        ledger = {'positions': [], 'pending_orders': [], 'cash': 100000.0}
        plan = self.plan(ledger, snapshot(cash=None))
        self.assertNotIn('set_cash', ops(plan))
        self.assertIn('cash_unreadable', kinds(plan))

    def test_blocked_account_is_reported_as_an_error(self):
        ledger = {'positions': [], 'pending_orders': [], 'cash': 1.0}
        plan = self.plan(ledger, snapshot(account_blocked=True))
        self.assertIn('account_blocked', kinds(plan))
        self.assertTrue(plan['blocked'])

    # ── The real 2026-09-17 MTD divergence this feature was built for ─────

    def test_real_mtd_fill_missed_by_the_ledger_is_recovered(self):
        """Ledger still shows MTD pending; Alpaca filled 18 @ ~1403.28.

        Under the old code the stale pending order would be expired and the
        broker position sold back out. The plan must instead book the fill.
        """
        ledger = {
            'cash': 100000.0, 'positions': [],
            'pending_orders': [pending(ticker='MTD', shares=18, amount_usd=25000.0,
                                       estimated_entry=1382.83, stop_distance=54.93,
                                       target_distance=109.86, sector='Healthcare')],
        }
        broker = snapshot(
            cash=74740.92,
            positions={'MTD': {'symbol': 'MTD', 'qty': 18,
                               'avg_entry_price': 1403.28, 'current_price': 1410.44}},
            orders=[order(symbol='MTD', side='buy', status='filled', qty=18,
                          filled_qty=18, filled_avg_price=1403.28,
                          client_order_id='lpm-B-MTD-1758134400-a1b2c3d4')])
        plan = self.plan(ledger, broker)
        self.assertEqual(sorted(ops(plan)), ['fill_pending', 'set_cash'])
        fill = only(plan, 'fill_pending')
        self.assertEqual((fill['shares'], fill['price']), (18, 1403.28))
        self.assertFalse(fill['partial'])
        # The stop/target must re-anchor to the real fill, not the estimate.
        self.assertAlmostEqual(fill['price'] - 54.93, 1348.35, places=2)
        self.assertAlmostEqual(fill['price'] + 109.86, 1513.14, places=2)
        event = next(e for e in plan['events'] if e['kind'] == 'fill')
        self.assertAlmostEqual(event['slippage_pct'], 1.4788, places=3)
        self.assertFalse(plan['blocked'])


if __name__ == '__main__':
    unittest.main()
