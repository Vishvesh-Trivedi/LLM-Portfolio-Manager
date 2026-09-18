"""Offline tests for applying a broker plan to the canonical ledger.

Injected app only — no main import, no network, no credentials. These pin the
writers that bypass plan_order: they record trades that ALREADY happened at the
broker, so they must never re-plan, never silently drop a record, and never
leave the ledger in a state the validator would reject.
"""

import copy
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import screener_portfolio as engine


def plan(*actions, ok=True, blocked=()):
    return {'ok': ok, 'actions': list(actions), 'events': [], 'blocked': list(blocked),
            'summary': 'test'}


class _LedgerFixture:
    SESSION = '2026-09-18'

    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.temp = tempfile.TemporaryDirectory(prefix='broker_apply_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.degraded = []
        self.app = SimpleNamespace(
            PORTFOLIO_JSON=self.root / 'portfolio.json',
            STARTING_CAPITAL=100000.0,
            _session_date=lambda: self.SESSION,
            _sharesies_fee=lambda amount, pf, nzdusd_rate=None, side='buy': 0.0,
            _ORDER_REASON=[''], _degrade=self.degraded.append,
            _CFG_MAX_POSITIONS=5, _CFG_MIN_CASH_FLOOR=500.0, _CFG_HOLD_DAYS=10,
            _broker_authoritative=lambda: True,
        )

    def ledger(self, **kwargs):
        base = {'cash': 100000.0, 'starting_capital': 100000.0, 'positions': [],
                'closed_trades': [], 'pending_orders': [], 'processed_sessions': [],
                'total_realized_pnl': 0.0, 'equity_peak': 100000.0}
        base.update(kwargs)
        return base

    def pending(self, **kwargs):
        base = {'id': 'e904bd8b-ef79-4a81-bfd2-a61eb4a563d9',
                'trade_id': 'e904bd8b-ef79-4a81-bfd2-a61eb4a563d9',
                'ticker': 'MTD', 'signal_date': '2026-09-16',
                'execution_session': '2026-09-17', 'estimated_entry': 1382.83,
                'stop_distance': 54.93, 'target_distance': 109.86,
                'amount_usd': 25000.0, 'shares': 18, 'sector': 'Healthcare',
                'atr': 36.62, 'hold_sessions': 10, 'status': 'PENDING',
                'source': 'BOTH', 'reasoning': 'catalyst', 'confidence': 85}
        base.update(kwargs)
        return base

    def position(self, **kwargs):
        base = {'trade_id': 't-1', 'ticker': 'ABC', 'shares': 10,
                'entry_price': 100.0, 'cost_basis': 1000.0,
                'entry_date': '2026-09-17', 'current_price': 100.0,
                'current_value': 1000.0, 'sector': 'Technology',
                'hold_sessions': 10, 'stop_price': 96.0, 'target_price': 110.0}
        base.update(kwargs)
        return base


class BrokerApplyTests(_LedgerFixture, unittest.TestCase):
    # ── Fills ─────────────────────────────────────────────────────────────

    def test_fill_books_the_broker_price_and_reanchors_stop_and_target(self):
        pf = self.ledger(pending_orders=[self.pending()])
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 18, 'price': 1403.28, 'broker_order_id': 'o-9',
             'partial': False, 'session': self.SESSION}))
        self.assertEqual(result['failed'], [])
        self.assertEqual(pf['pending_orders'], [])
        pos = pf['positions'][0]
        self.assertEqual((pos['ticker'], pos['shares'], pos['entry_price']),
                         ('MTD', 18, 1403.28))
        self.assertAlmostEqual(pos['stop_price'], 1348.35, places=2)
        self.assertAlmostEqual(pos['target_price'], 1513.14, places=2)
        self.assertEqual(pos['cost_basis'], round(1403.28 * 18, 2))
        self.assertEqual(pos['brokerage_in'], 0.0)
        self.assertEqual(pos['cost_basis_basis'], 'broker_confirmed_fill')
        self.assertEqual(pos['broker_order_id'], 'o-9')
        self.assertTrue(pos['filled_at_open'])
        # The pending id becomes the trade identity so the audit trail survives.
        self.assertEqual(pos['trade_id'], self.pending()['id'])
        self.assertEqual(pos['requested_shares'], 18)
        # Learning metadata must carry through to closed-trade history.
        self.assertEqual(pos['source'], 'BOTH')
        self.assertEqual(pos['confidence'], 85)

    def test_partial_fill_books_only_the_executed_shares(self):
        pf = self.ledger(pending_orders=[self.pending()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 5, 'price': 1403.28, 'partial': True, 'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertEqual(pos['shares'], 5)
        self.assertTrue(pos['broker_partial_fill'])
        self.assertEqual(pos['cost_basis'], round(1403.28 * 5, 2))

    def test_fill_price_that_invalidates_the_stop_is_refused(self):
        # A fill far below the stored stop distance would produce stop <= 0.
        pf = self.ledger(pending_orders=[self.pending(stop_distance=100.0,
                                                      target_distance=150.0)])
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 18, 'price': 50.0, 'partial': False, 'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(len(pf['pending_orders']), 1, 'order must survive a refusal')
        self.assertTrue(any('broker_sync_failed' in d for d in self.degraded))

    def test_fractional_broker_quantity_is_refused(self):
        pf = self.ledger(pending_orders=[self.pending()])
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 2.5, 'price': 1403.28, 'partial': True, 'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(pf['positions'], [])

    # ── Expiry and closes ─────────────────────────────────────────────────

    def test_expire_moves_the_order_into_the_expired_audit_list(self):
        pf = self.ledger(pending_orders=[self.pending()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'expire_pending', 'order_id': self.pending()['id'],
             'symbol': 'MTD', 'reason': 'broker status rejected'}))
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(pf['expired_orders'][0]['status'], 'EXPIRED')
        self.assertIn('rejected', pf['expired_orders'][0]['reason'])

    def test_close_books_realized_pnl_from_the_broker_exit_price(self):
        pf = self.ledger(positions=[self.position()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'close_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'price': 112.5, 'shares': 10, 'reason': 'broker_confirmed_exit',
             'session': self.SESSION}))
        self.assertEqual(pf['positions'], [])
        closed = pf['closed_trades'][0]
        self.assertEqual(closed['exit_price'], 112.5)
        self.assertEqual(closed['realized_pnl'], 125.0)   # 1125 gross - 1000 basis
        self.assertEqual(closed['result'], 'Win')
        self.assertEqual(closed['brokerage_out'], 0.0)
        self.assertEqual(closed['fee_basis'], 'broker_actual')
        self.assertEqual(pf['total_realized_pnl'], 125.0)

    def test_close_before_entry_date_is_refused(self):
        pf = self.ledger(positions=[self.position(entry_date='2026-09-30')])
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'close_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'price': 112.5, 'shares': 10, 'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(len(pf['positions']), 1)

    # ── Drift corrections ─────────────────────────────────────────────────

    def test_resize_rewrites_shares_and_cost_basis(self):
        pf = self.ledger(positions=[self.position(shares=10)])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'resize_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'shares': 7, 'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertEqual((pos['shares'], pos['cost_basis']), (7, 700.0))
        self.assertEqual(pos['cost_basis_basis'], 'broker_reconciled')

    def test_reprice_rewrites_entry_and_cost_basis(self):
        pf = self.ledger(positions=[self.position()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'reprice_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'entry_price': 103.4, 'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertEqual((pos['entry_price'], pos['cost_basis']), (103.4, 1034.0))

    # ── Adoption ──────────────────────────────────────────────────────────

    def test_adopt_requires_a_resolved_sector(self):
        # plan_order calls _sector() on every held position; an unknown sector
        # there would raise and block every future order.
        pf = self.ledger()
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 50.0, 'current_price': 52.0, 'sector': None,
             'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(pf['positions'], [])

    def test_an_adoption_does_not_consume_the_sessions_order_slot(self):
        """Run #94: 'Order not queued: session already has a decision/order'.

        Adopting MTD stamped the position with signal_date = today, so
        queue_position saw the session as already decided and refused the day's
        real trade. An adoption records something that already happened at the
        broker; it is not this session's decision.
        """
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 102.0, 'sector': 'Healthcare',
             'stop_price': 94.0, 'target_price': 112.0, 'atr': 4.0,
             'session': self.SESSION}))
        adopted = pf['positions'][0]
        self.assertNotIn('signal_date', adopted,
                         'an adoption must not claim this session as its signal')

        # The day's real order must still be able to go through.
        queued = engine.queue_position(
            self.app, pf,
            {'ticker': 'AAA', 'position_size_pct': 10, 'source': 'TECHNICAL',
             'reasoning': 'setup', 'hold_sessions': 10},
            100.0, 96.0, 110.0, {'sector': 'Technology', 'atr': 1.0})
        self.assertTrue(queued, self.app._ORDER_REASON[0])
        self.assertEqual(pf['pending_orders'][0]['ticker'], 'AAA')

    def test_adopt_refuses_a_sector_the_order_planner_cannot_map(self):
        # An unmappable sector is accepted by the ledger but then raises inside
        # plan_order on EVERY later order, silently blocking all trading.
        pf = self.ledger()
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 50.0, 'current_price': 52.0, 'sector': 'Biotechnology',
             'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(pf['positions'], [])

    def test_an_adopted_position_never_blocks_future_orders(self):
        from screener_safety import plan_order
        pf = self.ledger()  # full cash: keeps the drawdown guard out of this test
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 50.0, 'current_price': 52.0, 'sector': 'Healthcare',
             'session': self.SESSION}))
        self.assertEqual(len(pf['positions']), 1)
        # The whole point: sizing a new order must still work afterwards.
        order = plan_order(pf, 'AAA', 100.0, 5000.0, 96.0, 110.0, 'Technology',
                           lambda n: 0.0, max_positions=5, cash_floor=500.0)
        self.assertGreater(order['shares'], 0)

    def test_adopt_sets_atr_derived_sell_prices_when_they_are_safe(self):
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 102.0, 'sector': 'Technology',
             'stop_price': 94.0, 'target_price': 112.0, 'atr': 4.0,
             'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertEqual((pos['stop_price'], pos['target_price']), (94.0, 112.0))
        self.assertFalse(pos['needs_risk_levels'])
        self.assertEqual(pos['risk_levels_basis'], 'atr_derived_at_adoption')
        self.assertEqual(pos['atr_at_entry'], 4.0)

    def test_adopt_refuses_a_stop_that_would_fire_immediately(self):
        # A stop already above the market would liquidate, at market, a position
        # the owner never asked this system to manage.
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 90.0, 'sector': 'Technology',
             'stop_price': 94.0, 'target_price': 112.0, 'atr': 4.0,
             'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertNotIn('stop_price', pos)
        self.assertTrue(pos['needs_risk_levels'], 'must be flagged, not auto-sold')

    def test_adopt_refuses_a_target_already_reached(self):
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 120.0, 'sector': 'Technology',
             'stop_price': 94.0, 'target_price': 112.0, 'atr': 4.0,
             'session': self.SESSION}))
        self.assertTrue(pf['positions'][0]['needs_risk_levels'])

    def test_adopt_without_levels_still_succeeds_but_is_flagged(self):
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 102.0, 'sector': 'Technology',
             'stop_price': None, 'target_price': None, 'atr': 0.0,
             'session': self.SESSION}))
        self.assertTrue(pf['positions'][0]['needs_risk_levels'])

    def test_a_protected_adopted_position_can_actually_exit(self):
        from screener_safety import mechanical_exit
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 100.0, 'current_price': 102.0, 'sector': 'Technology',
             'stop_price': 94.0, 'target_price': 112.0, 'atr': 4.0,
             'session': self.SESSION}))
        pos = pf['positions'][0]
        bar = {'Open': 96.0, 'High': 97.0, 'Low': 93.0, 'Close': 95.0}
        self.assertEqual(mechanical_exit(pos, bar), (94.0, 'stop_loss'))

    def test_adopt_with_a_sector_creates_a_flagged_position(self):
        pf = self.ledger()
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'adopt_position', 'symbol': 'XYZ', 'shares': 5,
             'entry_price': 50.0, 'current_price': 52.0, 'sector': 'Technology',
             'session': self.SESSION}))
        pos = pf['positions'][0]
        self.assertEqual((pos['ticker'], pos['shares']), ('XYZ', 5))
        self.assertEqual(pos['source'], 'BROKER_ADOPTED')
        self.assertTrue(pos['needs_risk_levels'])
        self.assertNotIn('stop_price', pos, 'no stop may be invented for it')

    # ── Cash and ordering ─────────────────────────────────────────────────

    def test_cash_is_written_last_and_not_re_derived_locally(self):
        pf = self.ledger(pending_orders=[self.pending()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'set_cash', 'cash': 74740.92, 'equity': 100128.84},
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 18, 'price': 1403.28, 'partial': False, 'session': self.SESSION}))
        # set_cash is listed first but must be applied after the fill, so the
        # broker balance is the final word rather than a locally debited one.
        self.assertEqual(pf['cash'], 74740.92)
        self.assertEqual(pf['cash_basis'], 'broker_actual')
        self.assertEqual(pf['broker_equity'], 100128.84)
        self.assertEqual(len(pf['positions']), 1)

    def test_unreadable_plan_changes_nothing(self):
        pf = self.ledger(positions=[self.position()], cash=123.45)
        before = copy.deepcopy(pf)
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'close_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'price': 1.0, 'shares': 10, 'session': self.SESSION}, ok=False))
        self.assertEqual(pf, before)
        self.assertEqual(result['applied'], [])

    def test_one_bad_action_does_not_discard_the_others(self):
        pf = self.ledger(positions=[self.position()], pending_orders=[self.pending()])
        result = engine.apply_broker_state(self.app, pf, plan(
            {'op': 'close_position', 'trade_id': 'does-not-exist', 'symbol': 'ZZZ',
             'price': 1.0, 'shares': 1, 'session': self.SESSION},
            {'op': 'resize_position', 'trade_id': 't-1', 'symbol': 'ABC',
             'shares': 4, 'session': self.SESSION}))
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual(pf['positions'][0]['shares'], 4)

    def test_applied_state_is_persisted_and_reloads(self):
        pf = self.ledger(pending_orders=[self.pending()])
        engine.apply_broker_state(self.app, pf, plan(
            {'op': 'fill_pending', 'order_id': self.pending()['id'], 'symbol': 'MTD',
             'shares': 18, 'price': 1403.28, 'partial': False, 'session': self.SESSION}))
        reloaded = engine.load_portfolio(self.app)
        self.assertEqual(reloaded['positions'][0]['entry_price'], 1403.28)
        self.assertEqual(reloaded['pending_orders'], [])


class BrokerModeReplayTests(_LedgerFixture, unittest.TestCase):
    """Broker mode must stop the ledger simulating fills and local exits."""

    def test_broker_authoritative_reads_the_app_hook(self):
        self.assertTrue(engine.broker_authoritative(self.app))
        self.app._broker_authoritative = lambda: False
        self.assertFalse(engine.broker_authoritative(self.app))
        del self.app._broker_authoritative
        self.assertFalse(engine.broker_authoritative(self.app),
                         'absent hook must default to the old local behaviour')

    def test_hook_raising_does_not_enable_broker_mode(self):
        def boom():
            raise RuntimeError('nope')
        self.app._broker_authoritative = boom
        self.assertFalse(engine.broker_authoritative(self.app))

    def test_pending_order_is_not_simulated_or_expired_in_broker_mode(self):
        # The stale pending order below is PAST its execution session. The old
        # path would expire it, which is what would have sold the real MTD
        # position back out. Broker mode must leave it for the broker sync.
        pf = self.ledger(pending_orders=[self.pending()])
        engine.save_portfolio(self.app, pf)
        self.app.yf = SimpleNamespace(Ticker=lambda t: (_ for _ in ()).throw(
            AssertionError('no market data may be requested for a pending order')))
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['pending_orders']), 1)
        self.assertEqual(pf['positions'], [])
        self.assertNotIn('expired_orders', pf)


if __name__ == '__main__':
    unittest.main()
