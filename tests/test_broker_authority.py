"""Alpaca is the source of truth. These assert that, rather than assume it.

The rule is easy to state and easy to lose: where the ledger and the broker
disagree about what is owned, what it cost or how much cash there is, the
broker wins - and where the broker cannot be read at all, nothing trades.

Each test below names the way that rule could quietly stop holding.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from screener_broker_sync import plan_broker_sync


SESSION = '2026-09-21'


def ledger(**overrides):
    book = {
        'cash': 10000.0, 'starting_capital': 10000.0,
        'positions': [], 'pending_orders': [], 'closed_trades': [],
        'processed_sessions': [], 'equity_peak': 10000.0,
    }
    book.update(overrides)
    return book


def position(symbol='GILD', shares=98, entry=150.44, **extra):
    record = {'trade_id': 't-' + symbol, 'ticker': symbol, 'shares': shares,
              'entry_price': entry, 'cost_basis': round(shares * entry, 2),
              'entry_date': '2026-09-18', 'sector': 'Healthcare',
              'stop_price': 145.73, 'target_price': 159.86}
    record.update(extra)
    return record


def broker(cash=10000.0, positions=None, orders=None, **extra):
    snapshot = {'ok': True, 'cash': cash, 'equity': cash,
                'positions': dict(positions or {}), 'orders': list(orders or []),
                'account_blocked': False}
    snapshot.update(extra)
    return snapshot


def ops(plan):
    return [action['op'] for action in plan['actions']]


class TheBrokerWinsEveryDisagreement(unittest.TestCase):

    def test_a_share_count_is_corrected_to_the_brokers(self):
        """The ledger thinks 98, Alpaca says 50. Alpaca is right."""
        plan = plan_broker_sync(
            ledger(positions=[position(shares=98)]),
            broker(positions={'GILD': {'symbol': 'GILD', 'qty': 50,
                                       'avg_entry_price': 150.44}}),
            SESSION)
        self.assertIn('resize_position', ops(plan))
        resize = next(a for a in plan['actions'] if a['op'] == 'resize_position')
        self.assertEqual(resize['shares'], 50)

    def test_cash_is_taken_from_the_broker_not_computed_locally(self):
        plan = plan_broker_sync(ledger(cash=10000.0), broker(cash=7431.08), SESSION)
        self.assertIn('set_cash', ops(plan))
        self.assertEqual(
            next(a for a in plan['actions'] if a['op'] == 'set_cash')['cash'], 7431.08)

    def test_a_holding_the_ledger_never_recorded_is_adopted_not_ignored(self):
        """The MTD case: 18 real shares the ledger had no record of."""
        plan = plan_broker_sync(
            ledger(),
            broker(positions={'MTD': {'symbol': 'MTD', 'qty': 18,
                                      'avg_entry_price': 1403.28}}),
            SESSION)
        self.assertIn('adopt_position', ops(plan))


class ItRefusesToGuessRatherThanFabricate(unittest.TestCase):

    def test_a_vanished_position_with_no_priced_fill_blocks_instead_of_closing(self):
        """Closing at a stale quote would invent a P&L number.

        This is the one disagreement the broker cannot settle on its own, so
        it must stop the run rather than be papered over.
        """
        plan = plan_broker_sync(
            ledger(positions=[position()]), broker(), SESSION)
        self.assertNotIn('close_position', ops(plan))
        self.assertTrue(plan['blocked'], 'an unpriceable disagreement must block')
        self.assertTrue(any('absent at Alpaca' in reason for reason in plan['blocked']))

    def test_an_unreadable_broker_produces_no_actions_at_all(self):
        plan = plan_broker_sync(ledger(positions=[position()]),
                                {'ok': False, 'error': 'timeout'}, SESSION)
        self.assertFalse(plan['ok'])
        self.assertEqual(plan['actions'], [])
        self.assertTrue(plan['blocked'])

    def test_a_restricted_account_is_reported_as_blocking(self):
        plan = plan_broker_sync(ledger(), broker(account_blocked=True), SESSION)
        self.assertTrue(any('blocked' in reason.lower() for reason in plan['blocked']))


class NothingTradesOnAnUnverifiedLedger(unittest.TestCase):
    """The consequences of the above, at the application level."""

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    def test_a_broker_discrepancy_makes_the_run_refuse_to_order(self):
        with patch.object(self.app, '_blocking_degraded_reasons',
                          return_value=['broker_discrepancy:GILD absent at Alpaca']):
            readiness = self.app._trade_readiness()
        self.assertFalse(readiness['trade_ready'])
        self.assertTrue(readiness['trade_blockers'])

    def test_an_unreadable_broker_is_a_blocking_degradation(self):
        """It must not be filed alongside the reporting-only degradations."""
        for reason in ('broker_state_unreadable',
                       'broker_discrepancy:anything',
                       'broker_holding_unprotected:GILD',
                       'session_never_screened:2026-09-18'):
            with self.subTest(reason=reason):
                self.assertFalse(reason.startswith(self.app._NON_BLOCKING_DEGRADATIONS),
                                 f'{reason} must block trading')

    def mirror(self, sync_ok):
        """Run the mirror with a ledger the broker does not match.

        The broker reads are stubbed so that WITHOUT the gate there is a real
        order to send - the ledger holds 98 GILD and Alpaca holds nothing, so
        the mirror would buy. Otherwise the call dies on an unmocked network
        read and "no orders sent" proves nothing, which is exactly how the
        first version of this test passed while testing nothing.
        """
        sent = []
        stub = lambda *a, **k: sent.append(a) or {'id': '1'}
        with patch.object(self.app, '_BROKER_SYNC_OK', [sync_ok]),                 patch.object(self.app._alpaca, 'trading_enabled', return_value=True),                 patch.object(self.app._alpaca, 'get_account',
                             return_value={'equity': 10000.0}),                 patch.object(self.app._alpaca, 'effective_shares_by_symbol',
                             return_value={}),                 patch.object(self.app._alpaca, 'submit_market_order', side_effect=stub),                 patch.object(self.app._alpaca, 'close_position', side_effect=stub),                 redirect_stdout(io.StringIO()):
            self.app.reconcile_broker(ledger(positions=[position()]))
        return sent

    def test_the_mirror_sends_nothing_while_reconciliation_is_incomplete(self):
        """Doing nothing is recoverable; liquidating on a bad read is not."""
        self.assertEqual(self.mirror(sync_ok=False), [])

    def test_the_gate_is_what_stops_it_not_a_broken_mirror(self):
        """Same setup, reconciliation clean: it must actually send the order."""
        self.assertTrue(self.mirror(sync_ok=True),
                        'the mirror should order once the ledger is trusted')

    def test_protection_is_not_placed_from_a_ledger_known_to_be_wrong(self):
        """A stop derived from bad share counts could sell the wrong amount."""
        with patch.object(self.app, '_BROKER_SYNC_OK', [False]), \
                patch.object(self.app._alpaca, 'trading_enabled', return_value=True), \
                patch.object(self.app._alpaca, 'submit_protective_oco',
                             side_effect=AssertionError('must not place protection')), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(
                self.app.protect_positions(ledger(positions=[position()])), [])

    def test_a_broker_holding_the_ledger_does_not_know_is_never_sold(self):
        """Selling it would destroy what broker-authoritative sync protects."""
        sent = []
        with patch.object(self.app, '_BROKER_SYNC_OK', [True]), \
                patch.object(self.app._alpaca, 'trading_enabled', return_value=True), \
                patch.object(self.app._alpaca, 'get_account',
                             return_value={'equity': 10000.0}), \
                patch.object(self.app._alpaca, 'effective_shares_by_symbol',
                             return_value={'SURPRISE': 40}), \
                patch.object(self.app._alpaca, 'submit_market_order',
                             side_effect=lambda *a, **k: sent.append(a)), \
                patch.object(self.app._alpaca, 'close_position',
                             side_effect=lambda *a, **k: sent.append(a)), \
                redirect_stdout(io.StringIO()):
            self.app.reconcile_broker(ledger())
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main()
