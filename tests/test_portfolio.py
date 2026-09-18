"""Offline regression tests: injected app/yf only, no main import or credentials."""

import ast
import copy
import inspect
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pandas as pd

import screener_portfolio as engine
from screener_safety import atomic_json, validate_portfolio


def bars(days, prices=None):
    if prices is None:
        prices = [(100.0, 101.0, 99.0, 100.0)] * len(days)
    return pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close'],
                        index=pd.to_datetime(days))


class FakeTicker:
    def __init__(self, owner, ticker):
        self.owner = owner
        self.ticker = ticker

    def history(self, **kwargs):
        self.owner.calls.append((self.ticker, kwargs))
        value = self.owner.frames.get(self.ticker)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return bars([])
        # Deliberately return future/out-of-range data to test engine cutoffs.
        return value.copy(deep=True)

    @property
    def calendar(self):
        value = self.owner.calendars.get(self.ticker, {})
        if isinstance(value, Exception):
            raise value
        return value


class FakeYF:
    def __init__(self):
        self.frames = {}
        self.calendars = {}
        self.calls = []

    def Ticker(self, ticker):
        return FakeTicker(self, ticker)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        # Even an accidental library network request must fail the test offline.
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.temp = tempfile.TemporaryDirectory(prefix='portfolio_tests_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.today = '2026-09-08'
        self.degraded = []
        self.fee_calls = []
        self.fees = {'buy': 1.0, 'sell': 5.0}
        self.yf = FakeYF()
        self.app = SimpleNamespace(
            PORTFOLIO_JSON=self.root / 'portfolio.json',
            PICKS_CSV=self.root / 'picks.csv',
            STARTING_CAPITAL=10000.0,
            _session_date=lambda: self.today,
            _sharesies_fee=self.fee,
            _ORDER_REASON=[''], _degrade=self.degraded.append,
            _CFG_MAX_POSITIONS=5, _CFG_MIN_CASH_FLOOR=500.0,
            _CFG_HOLD_DAYS=10, _CFG_TRAIL_ATR_MULT=1.5,
            _CFG_RSI_EXIT=101.0, _CFG_RSI_EXIT_MIN_PROFIT=0.0,
            _CFG_MACD_EXIT_MIN_PROFIT=10000.0, _CFG_PRE_EARNINGS_DAYS=0,
            yf=self.yf,
        )

    def fee(self, amount, pf, nzdusd_rate=None, side='buy'):
        self.fee_calls.append((amount, side, nzdusd_rate))
        key = 'sharesies_bought_usd' if side == 'buy' else 'sharesies_sold_usd'
        pf[key] = pf.get(key, 0.0) + amount
        return self.fees[side]

    def new(self):
        return engine.load_portfolio(self.app)

    def open(self, pf, ticker='ABC', entry=100.0, stop=96.0, target=110.0,
             sector='Technology', amount=2000.0, atr=0.0, context=None):
        if context is not None:
            self.app._ORDER_CONTEXT = context
        result = engine.open_position(self.app, pf, ticker, entry, amount, stop, target,
                                      sector, atr=atr)
        self.assertIs(result, pf)
        self.assertTrue(any(p['ticker'] == ticker for p in pf['positions']), self.app._ORDER_REASON[0])
        return next(p for p in pf['positions'] if p['ticker'] == ticker)

    def queue(self, pf, ticker='ABC', pct=20, entry=100.0, stop=96.0,
              target=110.0, hold=10):
        return engine.queue_position(
            self.app, pf,
            {'ticker': ticker, 'position_size_pct': pct, 'source': 'NEWS',
             'reasoning': 'Known catalyst', 'hold_sessions': hold, 'sector': 'Wrong'},
            entry, stop, target, {'sector': 'Technology', 'atr': 1.0, 'rsi': 55})

    def test_fee_inclusive_fill_is_not_rejected_by_binary_rounding(self):
        self.fees = {'buy': 4.96, 'sell': 4.96}
        pf = self.new()
        pos = self.open(pf, entry=198.27, stop=194.0, target=210.0,
                        amount=996.31)
        self.assertEqual(pos['shares'], 5)
        self.assertEqual(pos['cost_basis'], 996.31)
        self.assertEqual(pf['cash'], 9003.69)

    def test_exact_public_wrapper_signatures(self):
        expected = {
            'load_portfolio': ['app'], 'save_portfolio': ['app', 'pf'],
            'update_portfolio_prices': ['app', 'pf'],
            'open_position': ['app', 'pf', 'ticker', 'entry_price', 'amount_usd', 'stop',
                              'target', 'sector', 'atr', 'nzdusd_rate'],
            'close_position': ['app', 'pf', 'ticker', 'exit_price', 'reason', 'nzdusd_rate'],
            'reconcile_closed_picks': ['app', 'pf'],
            'load_performance_history': ['app', 'fp'],
            'update_results': ['app', 'fp', 'cols'],
            'queue_position': ['app', 'pf', 'pick', 'entry', 'stop', 'target', 'candidate'],
            'apply_broker_state': ['app', 'pf', 'plan'],
            'broker_authoritative': ['app'],
        }
        self.assertEqual(set(engine.__all__), set(expected))
        for name, parameters in expected.items():
            self.assertEqual(list(inspect.signature(getattr(engine, name)).parameters), parameters)

    def test_source_parses_with_python_311_grammar(self):
        # Real runtime availability is reported separately; no 3.12-only syntax.
        ast.parse(Path(engine.__file__).read_text(encoding='utf-8'), feature_version=(3, 11))

    def test_new_canonical_file_initializes_and_roundtrips(self):
        pf = self.new()
        self.assertEqual(pf['cash'], 10000)
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(pf['processed_sessions'], [])
        self.assertTrue(self.app.PORTFOLIO_JSON.exists())
        self.assertEqual(engine.load_portfolio(self.app), pf)
        self.assertEqual(self.yf.calls, [])

    def test_corrupt_existing_file_never_resets_or_falls_back(self):
        atomic_json(self.root / 'fallback.json', {'cash': 10000})
        for payload in ('{broken', '{}', 'null', '{"cash": NaN}',
                        '{"cash": 1, "cash": 2}'):
            with self.subTest(payload=payload):
                self.app.PORTFOLIO_JSON.write_text(payload, encoding='utf-8')
                before = self.app.PORTFOLIO_JSON.read_bytes()
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.new()
                self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), before)

    def test_missing_core_fields_fail_even_if_defaults_exist_elsewhere(self):
        base = self.new()
        for key in ('cash', 'starting_capital', 'positions', 'closed_trades'):
            bad = copy.deepcopy(base)
            del bad[key]
            atomic_json(self.app.PORTFOLIO_JSON, bad)
            with self.assertRaises((ValueError, KeyError)):
                engine.load_portfolio(self.app)

    def test_legacy_defaults_and_deterministic_ids_do_not_change_dates(self):
        pf = self.new()
        pos = self.open(pf)
        del pos['trade_id']
        del pf['pending_orders']
        del pf['processed_sessions']
        atomic_json(self.app.PORTFOLIO_JSON, pf)
        first = engine.load_portfolio(self.app)
        second = engine.load_portfolio(self.app)
        self.assertEqual(first['positions'][0]['trade_id'], second['positions'][0]['trade_id'])
        self.assertEqual(first['positions'][0]['entry_date'], '2026-09-08')
        self.assertEqual(first['pending_orders'], [])
        self.assertEqual(first['processed_sessions'], [])

    def test_pending_and_processed_schema_validation(self):
        base = self.new()
        for key, value in [('pending_orders', {}), ('pending_orders', [None]),
                           ('processed_sessions', {}), ('processed_sessions', ['not-a-date']),
                           ('processed_sessions', ['2026-09-08', '2026-09-08'])]:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(base)
                bad[key] = value
                atomic_json(self.app.PORTFOLIO_JSON, bad)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    engine.load_portfolio(self.app)

    def test_malformed_pending_cannot_be_saved(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        before = self.app.PORTFOLIO_JSON.read_bytes()
        for key, value in [('stop_distance', 100), ('shares', 1.5), ('estimated_entry', 0),
                           ('amount_usd', -1), ('signal_date', 'bad'), ('atr', float('nan'))]:
            with self.subTest(key=key):
                bad = copy.deepcopy(pf)
                bad['pending_orders'][0][key] = value
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    engine.save_portfolio(self.app, bad)
                self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), before)

    def test_atomic_save_failure_leaves_memory_and_file_unchanged(self):
        pf = self.new()
        pf['cash'] = 12000.0
        snapshot = copy.deepcopy(pf)
        before = self.app.PORTFOLIO_JSON.read_bytes()
        with patch('screener_safety.os.replace', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                engine.save_portfolio(self.app, pf)
        self.assertEqual(pf, snapshot)
        self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_save_equity_peak_never_decreases(self):
        pf = self.new()
        pf['cash'] = 12000.0
        engine.save_portfolio(self.app, pf)
        self.assertEqual(pf['equity_peak'], 12000)
        pf.update(cash=9000.0, equity_peak=0.0)
        engine.save_portfolio(self.app, pf)
        self.assertEqual(pf['equity_peak'], 12000)

    def test_existing_corrupt_ledger_cannot_be_overwritten_by_save(self):
        pf = self.new()
        self.app.PORTFOLIO_JSON.write_text('corrupt', encoding='utf-8')
        with self.assertRaises(ValueError):
            engine.save_portfolio(self.app, pf)
        self.assertEqual(self.app.PORTFOLIO_JSON.read_text(), 'corrupt')

    def test_fee_quotes_are_pure_and_conservative_and_actual_fee_commits_once(self):
        pf = self.new()
        pf['last_nzdusd_rate'] = 0.61
        self.fees.update(buy=0.0, sell=5.0)
        pos = self.open(pf, amount=10000.0)
        self.assertEqual(pos['shares'], 22)  # floor((100 risk - 5 - 5) / 4)
        self.assertEqual(pf['sharesies_bought_usd'], pos['shares'] * 100)
        self.assertNotIn('sharesies_sold_usd', pf)
        self.assertEqual(pos['brokerage_in'], 0)
        self.assertLessEqual(pos['shares'] * 4 + 10, 100)
        self.assertTrue(all(call[2] == 0.61 for call in self.fee_calls))

    def test_open_rejections_do_not_mutate_portfolio_or_fee_usage(self):
        pf = self.new()
        base = dict(ticker='ABC', entry_price=100.0, amount_usd=2000.0,
                    stop=96.0, target=110.0, sector='Technology', atr=1.0)
        cases = [dict(stop=0), dict(stop=-1), dict(stop=101), dict(target=102),
                 dict(entry_price=float('nan')), dict(amount_usd=float('inf')),
                 dict(amount_usd=0), dict(amount_usd=True), dict(entry_price='100'),
                 dict(atr=-1), dict(atr=float('nan')), dict(nzdusd_rate=float('nan')),
                 dict(sector='Unknown'), dict(ticker='')]
        for override in cases:
            with self.subTest(override=override):
                before = copy.deepcopy(pf)
                self.assertIs(engine.open_position(self.app, pf, **(base | override)), pf)
                self.assertEqual(pf, before)
                self.assertIn('rejected', self.app._ORDER_REASON[0])

    def test_rejected_metadata_does_not_consume_fee_coverage(self):
        pf = self.new()
        self.app._ORDER_CONTEXT = {'confidence': float('nan')}
        before = copy.deepcopy(pf)
        engine.open_position(self.app, pf, 'ABC', 100, 2000, 96, 110, 'Technology')
        self.assertEqual(pf, before)
        self.assertIn('rejected', self.app._ORDER_REASON[0])

    def test_distinct_trade_ids_when_ticker_reopens_same_day(self):
        pf = self.new()
        first = self.open(pf)['trade_id']
        UUID(first)
        engine.close_position(self.app, pf, 'ABC', 101.0)
        second = self.open(pf)['trade_id']
        UUID(second)
        self.assertNotEqual(first, second)
        validate_portfolio(pf)

    def test_hold_metadata_clamped_and_session_override_used(self):
        for hold, expected in [(-5, 1), (0, 1), (2, 2), (900, 30)]:
            with self.subTest(hold=hold):
                pf = self.new()
                self.today = '2026-09-09'
                pos = self.open(pf, context={'hold_sessions': hold, 'signal_date': '2026-09-08'})
                self.assertEqual(pos['hold_sessions'], expected)
                self.assertEqual(pos['entry_date'], '2026-09-09')
                self.assertEqual(pos['fill_date'], '2026-09-09')
                self.assertEqual(pos['signal_date'], '2026-09-08')

    def test_close_preserves_metadata_dates_and_net_not_excess_result(self):
        pf = self.new()
        self.open(pf, context={'source': 'BOTH', 'signal_date': '2026-09-07', 'reasoning': 'Evidence'})
        pos = pf['positions'][0]
        pos['benchmark_return_pct'] = 20.0
        pos['excess_return_pct'] = -19.0
        tid = pos['trade_id']
        self.today = '2026-09-10'
        engine.close_position(self.app, pf, 'ABC', 101.0, 'profit_test', nzdusd_rate=0.63)
        trade = pf['closed_trades'][0]
        self.assertEqual(trade['trade_id'], tid)
        for key, expected in [('entry_date', '2026-09-08'), ('exit_date', '2026-09-10'),
                              ('signal_date', '2026-09-07'), ('source', 'BOTH'),
                              ('sector', 'Technology'), ('reason', 'profit_test'),
                              ('outcome_basis', 'net_realized'), ('Result', 'Win')]:
            self.assertEqual(trade[key], expected)
        self.assertEqual(trade['stop_price'], 96)
        self.assertEqual(trade['target_price'], 110)
        self.assertEqual(trade['benchmark_return_pct'], 20)
        self.assertEqual(trade['realized_pnl'], trade['exit_value'] - trade['cost_basis'])
        self.assertEqual(self.fee_calls[-1][1:], ('sell', 0.63))
        self.assertEqual(pf['sharesies_sold_usd'], trade['shares'] * 101)

    def test_close_invalid_price_is_no_mutation(self):
        pf = self.new()
        self.open(pf)
        for value in (float('nan'), float('inf'), 0, -1, True, '100'):
            before = copy.deepcopy(pf)
            engine.close_position(self.app, pf, 'ABC', value)
            self.assertEqual(pf, before)
            self.assertIn('rejected', self.app._ORDER_REASON[0])

    def test_queue_persists_without_cash_or_fee_debit(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        order = pf['pending_orders'][0]
        UUID(order['id'])
        self.assertEqual(order['estimated_entry'], 100)
        self.assertEqual(order['stop_distance'], 4)
        self.assertEqual(order['target_distance'], 10)
        self.assertEqual(order['sector'], 'Technology')
        self.assertEqual(order['amount_usd'], 2000)
        self.assertEqual(order['source'], 'NEWS')
        self.assertEqual(order['signal_date'], self.today)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['cash'], 10000)
        self.assertNotIn('sharesies_bought_usd', pf)
        self.assertEqual(engine.load_portfolio(self.app), pf)

    def test_queue_invalid_zero_and_tiny_size_never_uplifted(self):
        pf = self.new()
        for size in (0, -1, 101, float('nan'), float('inf'), True, None, '20', 0.4):
            with self.subTest(size=size):
                before = copy.deepcopy(pf)
                disk = self.app.PORTFOLIO_JSON.read_bytes()
                self.assertFalse(self.queue(pf, pct=size))
                self.assertEqual(pf, before)
                self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), disk)
                self.assertTrue(self.app._ORDER_REASON[0])

    def test_queue_small_valid_budget_is_absolute_not_floor_or_target(self):
        pf = self.new()
        self.assertTrue(self.queue(pf, pct=2))
        order = pf['pending_orders'][0]
        self.assertEqual(order['amount_usd'], 200)
        self.assertEqual(order['shares'], 1)

    def test_queue_uses_current_cash_snapshot(self):
        pf = self.new()
        pf['cash'] = 100000.0
        self.assertTrue(self.queue(pf, pct=25, entry=1382.83,
                                   stop=1327.9, target=1492.69))
        order = pf['pending_orders'][0]
        self.assertEqual(order['amount_usd'], 25000.0)
        self.assertEqual(order['shares'], 18)

    def test_one_queue_per_session_including_other_ticker(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        before = copy.deepcopy(pf)
        self.assertFalse(self.queue(pf, ticker='XYZ'))
        self.assertEqual(pf, before)
        self.assertIn('session', self.app._ORDER_REASON[0])

    def test_processed_and_closed_signal_dates_prevent_duplicate_queue(self):
        for source in ('processed', 'closed'):
            with self.subTest(source=source):
                pf = self.new()
                if source == 'processed':
                    pf['processed_sessions'] = [self.today]
                else:
                    self.open(pf, context={'signal_date': self.today})
                    engine.close_position(self.app, pf, 'ABC', 100)
                before = copy.deepcopy(pf)
                self.assertFalse(self.queue(pf, ticker='XYZ'))
                self.assertEqual(pf, before)

    def test_same_signal_day_does_not_fill_or_fetch(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.yf.frames['ABC'] = bars([self.today], [(100, 111, 90, 100)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['pending_orders']), 1)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(self.yf.calls, [])

    def test_next_session_open_fill_and_same_day_idempotency(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        order = copy.deepcopy(pf['pending_orders'][0])
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today, '2026-09-10'],
                                     [(100, 101, 99, 100), (102, 104, 100, 103),
                                      (300, 301, 299, 300)])
        parent_context = {'source': 'parent', 'nested': {'x': 1}}
        self.app._ORDER_CONTEXT = parent_context
        engine.update_portfolio_prices(self.app, pf)
        self.assertIs(self.app._ORDER_CONTEXT, parent_context)
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(len(pf['positions']), 1)
        pos = pf['positions'][0]
        self.assertEqual(pos['trade_id'], order['id'])
        self.assertEqual(pos['entry_price'], 102)
        self.assertEqual(pos['entry_date'], self.today)
        self.assertEqual(pos['signal_date'], '2026-09-08')
        self.assertTrue(pos['filled_at_open'])
        self.assertEqual(pos['target_price'], 112)
        self.assertEqual(pos['current_price'], 103)
        self.assertEqual(pos['held_sessions'], 1)
        self.assertLessEqual(pos['shares'], order['shares'])
        self.assertLessEqual(pos['cost_basis'], order['amount_usd'])
        self.assertEqual(pf['sharesies_bought_usd'], pos['shares'] * 102)
        self.assertEqual(pos['last_evaluated_session'], self.today)
        self.assertIn('RSI omitted', pos['indicator_exit_note'])
        before = copy.deepcopy(pf)
        fee_calls = len(self.fee_calls)
        requests = len(self.yf.calls)
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf, before)
        self.assertEqual(len(self.fee_calls), fee_calls)
        self.assertEqual(len(self.yf.calls), requests)
        kwargs = self.yf.calls[0][1]
        self.assertFalse(kwargs['auto_adjust'])
        self.assertEqual(kwargs['start'], '2026-07-10')  # 60 calendar days of warmup.
        self.assertEqual(kwargs['end'], '2026-09-10')

    def test_actual_open_stop_target_evaluated_before_close_trailing(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (100, 102, 95, 101)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['closed_trades'][0]['exit_price'], 96)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'stop_loss')
        self.assertEqual(pf['closed_trades'][0]['exit_date'], self.today)

    def test_expired_order_never_backdated(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['cash'], 10000)
        self.assertNotIn('sharesies_bought_usd', pf)
        self.assertIn('missed next session', pf['expired_orders'][0]['reason'])
        self.assertIn('expired', self.app._ORDER_REASON[0])

    def test_pending_missing_quote_then_later_session_expires(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08'])
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['pending_orders'][0]['quote_stale'])
        self.assertIn('stale_quote:ABC', self.degraded)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'])
        self.assertFalse(pf['pending_orders'])

    def test_pending_execution_rechecks_safety_and_does_not_spend_on_rejection(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        pf['cash'] = 500.0  # Adverse cash change after authorization.
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['cash'], 500)
        self.assertNotIn('sharesies_bought_usd', pf)
        self.assertEqual(pf['positions'], [])
        self.assertIn('execution rejected', pf['expired_orders'][0]['reason'])

    def test_gap_down_cannot_increase_preflight_share_quantity(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        authorized = pf['pending_orders'][0]['shares']
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (50, 51, 49, 50)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'][0]['shares'], authorized)
        self.assertEqual(pf['positions'][0]['stop_price'], 46)

    def test_legacy_entry_day_cannot_stop_or_ratchet_on_preentry_bar(self):
        pf = self.new()
        self.open(pf, atr=1)
        self.yf.frames['ABC'] = bars([self.today], [(100, 130, 90, 105)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['positions']), 1)
        self.assertEqual(pf['positions'][0]['stop_price'], 96)
        self.assertEqual(pf['positions'][0]['held_sessions'], 0)
        self.assertEqual(pf['positions'][0]['high_watermark'], 100)

    def test_exit_uses_old_stop_before_trailing_ratchet_and_gap(self):
        pf = self.new()
        self.open(pf, atr=1)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (100, 106, 97, 105)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['positions']), 1)
        self.assertEqual(pf['positions'][0]['stop_price'], 103.5)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today],
                                     [(100, 101, 99, 100), (100, 106, 97, 105),
                                      (102, 105, 101, 104)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'])
        self.assertEqual(pf['closed_trades'][0]['exit_price'], 102)

    def test_missing_quote_no_exit_ratchet_or_other_ticker_fallback(self):
        pf = self.new()
        self.open(pf, 'ABC', atr=1)
        self.open(pf, 'XYZ', sector='Energy')
        first = copy.deepcopy(pf['positions'][0])
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08'])
        self.yf.frames['XYZ'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (100, 105, 99, 104)])
        engine.update_portfolio_prices(self.app, pf)
        pos = pf['positions'][0]
        self.assertTrue(pos['quote_stale'])
        self.assertEqual(pos['current_price'], first['current_price'])
        self.assertEqual(pos['stop_price'], first['stop_price'])
        self.assertNotIn('last_evaluated_session', pos)
        self.assertIn('stale_quote:ABC', self.degraded)
        self.assertEqual(pf['positions'][1]['current_price'], 104)
        self.assertEqual({call[0] for call in self.yf.calls}, {'ABC', 'XYZ'})
        self.assertEqual(len(pf['positions']), 2)

    def test_quote_failure_is_degraded_and_same_session_recovery_allowed(self):
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = RuntimeError('offline provider failure')
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['positions'][0]['quote_stale'])
        self.assertIn('offline provider failure', pf['positions'][0]['update_error'])
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'][0]['quote_stale'])
        self.assertEqual(pf['positions'][0]['last_evaluated_session'], self.today)

    def test_invalid_exact_bar_degrades_without_exiting(self):
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (100, 90, 99, 100)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['positions'][0]['quote_stale'])
        self.assertFalse(pf['closed_trades'])

    def test_hold_exit_counts_sessions_not_weekend_days_actual_open(self):
        pf = self.new()
        self.today = '2026-09-10'
        self.assertTrue(self.queue(pf, hold=2))
        self.today = '2026-09-11'
        self.yf.frames['ABC'] = bars(['2026-09-10', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'][0]['held_sessions'], 1)
        self.today = '2026-09-14'
        self.yf.frames['ABC'] = bars(['2026-09-10', '2026-09-11', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'])
        trade = pf['closed_trades'][0]
        self.assertEqual(trade['reason'], 'hold_period')
        self.assertEqual(trade['held_sessions'], 2)
        self.assertEqual(trade['exit_date'], self.today)
        before = copy.deepcopy(pf)
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf, before)

    def test_hold_exit_legacy_excludes_entry_session(self):
        pf = self.new()
        self.open(pf, context={'hold_sessions': 2})
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'][0]['held_sessions'], 1)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'hold_period')

    def test_earnings_exit_and_unavailable_calendar_are_explicit(self):
        pf = self.new()
        self.open(pf)
        self.open(pf, ticker='XYZ', sector='Energy')
        self.today = '2026-09-09'
        self.yf.frames.update(ABC=bars(['2026-09-08', self.today]),
                              XYZ=bars(['2026-09-08', self.today]))
        self.yf.calendars['ABC'] = {'Earnings Date': [pd.Timestamp(self.today)]}
        self.yf.calendars['XYZ'] = RuntimeError('calendar offline')
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['closed_trades'][0]['reason'].startswith('pre_earnings'))
        self.assertEqual(pf['positions'][0]['ticker'], 'XYZ')
        self.assertIn('earnings_unavailable:XYZ', self.degraded)

    def test_rsi_exit_uses_same_ticker_history_and_no_future_bars(self):
        pf = self.new()
        self.app._CFG_RSI_EXIT = 78
        self.today = '2026-08-20'
        days = pd.bdate_range(self.today, periods=14)
        self.open(pf, stop=95, target=140, context={'hold_sessions': 30})
        self.today = days[-1].date().isoformat()
        prices = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(14)]
        self.yf.frames['ABC'] = bars(days, prices)
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'rsi_overbought')
        self.assertEqual(len(self.yf.calls), 1)

    def test_macd_bearish_cross_is_preserved(self):
        pf = self.new()
        self.app._CFG_MACD_EXIT_MIN_PROFIT = 0
        self.today = '2026-08-03'
        self.open(pf, stop=95, target=160, context={'hold_sessions': 30})
        prices = [100 + i * 0.8 for i in range(26)] + [120 - i * 0.7 for i in range(1, 5)]
        closes = pd.Series(prices)
        macd = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        cross = next(i for i in range(26, len(prices))
                     if macd[i] < signal[i] and macd[i - 1] >= signal[i - 1])
        days = pd.bdate_range(self.today, periods=cross + 1)
        self.today = days[-1].date().isoformat()
        self.yf.frames['ABC'] = bars(days, [(p, p + 0.1, p - 0.1, p) for p in prices[:cross + 1]])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'macd_bearish_cross')
        self.assertEqual(len(self.yf.calls), 1)

    def test_learning_restores_all_eleven_legacy_trades_without_csv(self):
        pf = self.new()
        original_dates = []
        for i in range(11):
            pct = [-2.0, 0.0, 3.0][i % 3]
            entry_date = '2026-08-' + str(i + 1).zfill(2)
            exit_date = '2026-08-' + str(i + 12).zfill(2)
            trade = {'ticker': 'LEG' + str(i), 'shares': 10, 'entry_price': 100.0,
                     'exit_price': 100 + pct, 'cost_basis': 1000.0,
                     'realized_pnl': pct * 10, 'realized_pnl_pct': pct,
                     'entry_date': entry_date, 'exit_date': exit_date,
                     'reason': 'legacy close', 'excess_return_pct': -50.0}
            if i == 0:
                trade.update(source='BOTH', sector='Energy', tech_score=5, news_score=2)
            pf['closed_trades'].append(trade)
            original_dates.append((entry_date, exit_date))
        atomic_json(self.app.PORTFOLIO_JSON, pf)
        self.assertFalse(self.app.PICKS_CSV.exists())
        history = engine.load_performance_history(self.app, self.app.PICKS_CSV)
        self.assertEqual(len(history), 11)
        self.assertEqual([h['result'] for h in history[:3]], ['Loss', 'Neutral', 'Win'])
        self.assertEqual(history[2]['vs_qqq_10d'], '-50.0')
        self.assertTrue(all(h['outcome_basis'] == 'net_realized' for h in history))
        self.assertEqual(history[0]['source'], 'BOTH')
        self.assertEqual(history[1]['source'], '')
        self.assertEqual(history[1]['sector'], '')
        self.assertEqual([(h['entry_date'], h['exit_date']) for h in history], original_dates)
        self.assertEqual(len({h['trade_id'] for h in history}), 11)
        for h in history:
            for key in ('tech_score', 'news_score', 'confidence', 'rsi', 'vix', 'qqq_trend',
                        'return_30d', 'vs_qqq_30d', 'reasoning', 'source', 'sector'):
                self.assertIsInstance(h[key], str)
        # Exercise the actual legacy consumer without importing its network-heavy main.
        main_path = Path(engine.__file__).with_name('LLM_Portfolio_Manager.py')
        if main_path.exists():
            source = main_path.read_text(encoding='utf-8-sig')
            module = ast.parse(source)
            fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'build_learning_insights')
            namespace = {}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(main_path), 'exec'), namespace)
            self.assertIn('PATTERN ANALYSIS', namespace['build_learning_insights'](history))
        self.assertEqual(self.yf.calls, [])

    def test_learning_falls_back_to_net_pnl_not_benchmark(self):
        pf = self.new()
        self.open(pf)
        engine.close_position(self.app, pf, 'ABC', 100.0)
        trade = pf['closed_trades'][0]
        del trade['net_realized_pct']
        del trade['realized_pnl_pct']
        del trade['realized_pct']
        trade['excess_return_pct'] = 100.0
        engine.save_portfolio(self.app, pf)
        history = engine.load_performance_history(self.app, 'ignored.csv')
        self.assertEqual(history[0]['result'], 'Loss')
        self.assertLess(float(history[0]['return_pct']), 0)

    def test_stale_csv_never_closes_reopened_ticker(self):
        pf = self.new()
        first = self.open(pf)['trade_id']
        engine.close_position(self.app, pf, 'ABC', 101.0)
        self.open(pf)
        pd.DataFrame([{'Ticker': 'ABC', 'Result': 'Win', 'Close_Price': 999,
                       'Date': '2026-08-01', 'Trade_ID': first},
                      {'Ticker': 'ABC', 'Result': 'Loss', 'Close_Price': 1,
                       'Date': '2026-08-01', 'Trade_ID': ''}]).to_csv(self.app.PICKS_CSV, index=False)
        before = copy.deepcopy(pf)
        engine.reconcile_closed_picks(self.app, pf)
        self.assertEqual(pf, before)
        self.assertEqual(len(pf['positions']), 1)
        df = pd.read_csv(self.app.PICKS_CSV)
        self.assertEqual(df.iloc[0]['Close_Price'], 101)
        self.assertEqual(df.iloc[1]['Close_Price'], 1)
        self.assertEqual(self.yf.calls, [])

    def test_csv_sync_known_open_trade_clears_false_horizon_outcome(self):
        pf = self.new()
        tid = self.open(pf)['trade_id']
        pd.DataFrame([{'Ticker': 'ABC', 'Trade_ID': tid, 'Result': 'Loss',
                       'Close_Price': 1, 'Return_Pct': -99}]).to_csv(self.app.PICKS_CSV, index=False)
        engine.reconcile_closed_picks(self.app, pf)
        row = pd.read_csv(self.app.PICKS_CSV, keep_default_na=False).iloc[0]
        self.assertEqual(row['Result'], 'Pending')
        self.assertEqual(row['Execution_Status'], 'FILLED')
        self.assertEqual(row['Outcome_Basis'], 'net_realized')
        self.assertEqual(row['Close_Price'], '')
        self.assertEqual(row['Return_Pct'], '')

    def test_paper_horizon_exact_entry_exit_matching_benchmark_and_cutoff(self):
        self.new()
        self.today = '2026-09-11'
        self.app._CFG_HOLD_DAYS = 2
        pd.DataFrame([{'Date': '2026-09-08', 'Ticker': 'ABC', 'Signal': 'WATCH',
                       'Entry_Price': 1, 'Result': 'Pending', 'Execution_Status': 'SIGNAL'}]).to_csv(
                           self.app.PICKS_CSV, index=False)
        dates = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14']
        self.yf.frames['ABC'] = bars(dates, [(90, 91, 89, 90), (100, 111, 99, 110),
                                            (110, 121, 109, 120), (120, 131, 119, 130),
                                            (130, 301, 129, 300)])
        self.yf.frames['QQQ'] = bars(dates, [(190, 191, 189, 190), (200, 221, 199, 220),
                                            (220, 231, 219, 230), (230, 241, 229, 240),
                                            (240, 901, 239, 900)])
        engine.update_results(self.app, self.app.PICKS_CSV, ['Trade_ID'])
        row = pd.read_csv(self.app.PICKS_CSV, keep_default_na=False).iloc[0]
        self.assertEqual(row['Realistic_Entry'], 100)
        self.assertEqual(row['Entry_Date'], '2026-09-09')
        self.assertEqual(row['Close_Date'], '2026-09-10')
        self.assertEqual(row['Close_Price'], 120)
        self.assertAlmostEqual(row['Return_Pct'], 20)
        self.assertAlmostEqual(row['Benchmark_Return_Pct'], 15)
        self.assertAlmostEqual(row['vs_QQQ_10d'], 5)
        self.assertEqual(row['Outcome_Basis'], 'paper_horizon')
        self.assertEqual(row['Execution_Status'], 'SIGNAL')
        before = self.app.PICKS_CSV.read_bytes()
        requests = len(self.yf.calls)
        engine.update_results(self.app, self.app.PICKS_CSV, [])
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), before)
        self.assertEqual(len(self.yf.calls), requests)

    def test_update_results_skips_filled_ids_filled_status_no_pick_and_old_outcomes(self):
        pf = self.new()
        tid = self.open(pf)['trade_id']
        engine.save_portfolio(self.app, pf)
        self.today = '2026-09-11'
        records = [dict(Trade_ID=tid), dict(Execution_Status='FILLED'),
                   dict(Signal='NO PICK'), dict(Result='Win')]
        rows = [dict({'Date': '2026-09-01', 'Ticker': 'ABC', 'Result': 'Pending'}, **record)
                for record in records]
        pd.DataFrame(rows).fillna('').to_csv(self.app.PICKS_CSV, index=False)
        before = self.app.PICKS_CSV.read_bytes()
        engine.update_results(self.app, self.app.PICKS_CSV, [])
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), before)
        self.assertEqual(self.yf.calls, [])
        self.assertEqual(len(engine.load_portfolio(self.app)['positions']), 1)

    def test_missing_benchmark_session_cannot_annotate_paper_result(self):
        self.new()
        self.today = '2026-09-11'
        self.app._CFG_HOLD_DAYS = 2
        pd.DataFrame([{'Date': '2026-09-08', 'Ticker': 'ABC', 'Result': 'Pending'}]).to_csv(
            self.app.PICKS_CSV, index=False)
        self.yf.frames['ABC'] = bars(['2026-09-09', '2026-09-10'])
        self.yf.frames['QQQ'] = bars(['2026-09-09', '2026-09-11'])
        before = self.app.PICKS_CSV.read_bytes()
        engine.update_results(self.app, self.app.PICKS_CSV, [])
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), before)
        self.assertIn('paper_horizon_unavailable:ABC', self.degraded)

    def test_csv_atomic_failure_never_closes_or_mutates_ledger(self):
        pf = self.new()
        tid = self.open(pf)['trade_id']
        engine.close_position(self.app, pf, 'ABC', 100)
        pd.DataFrame([{'Ticker': 'ABC', 'Trade_ID': tid}]).to_csv(self.app.PICKS_CSV, index=False)
        before = copy.deepcopy(pf)
        disk = self.app.PICKS_CSV.read_bytes()
        with patch('screener_safety.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                engine.reconcile_closed_picks(self.app, pf)
        self.assertEqual(pf, before)
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), disk)

    def test_decimal_distances_and_exact_minimum_reward_risk_survive_gap(self):
        pf = self.new()
        self.assertTrue(self.queue(pf, entry=100.3, stop=100.1, target=100.6))
        order = pf['pending_orders'][0]
        self.assertEqual(order['stop_distance'], 0.2)
        self.assertEqual(order['target_distance'], 0.3)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100.3, 100.4, 100.2, 100.3),
                                      (101.3, 101.4, 101.2, 101.3)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['positions']), 1, self.app._ORDER_REASON[0])
        self.assertEqual(pf['positions'][0]['stop_price'], 101.1)
        self.assertEqual(pf['positions'][0]['target_price'], 101.6)

    def test_fee_errors_are_readable_rejections_without_mutation(self):
        pf = self.new()
        self.open(pf, 'XYZ', sector='Energy')
        before = copy.deepcopy(pf)

        def broken_fee(amount, state, nzdusd_rate=None, side='buy'):
            state['sharesies_bought_usd'] = 999999
            raise RuntimeError('fee calculation unavailable')

        self.app._sharesies_fee = broken_fee
        engine.open_position(self.app, pf, 'ABC', 100, 2000, 96, 110, 'Technology')
        self.assertEqual(pf, before)
        self.assertIn('fee calculation unavailable', self.app._ORDER_REASON[0])
        self.assertFalse(self.queue(pf))
        self.assertEqual(pf, before)
        engine.close_position(self.app, pf, 'XYZ', 100)
        self.assertEqual(pf, before)
        self.assertIn('fee calculation unavailable', self.app._ORDER_REASON[0])

    def test_duplicate_daily_bars_fail_closed_and_do_not_cross_ticker(self):
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today, self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['positions'][0]['quote_stale'])
        self.assertIn('duplicate', pf['positions'][0]['update_error'])
        self.assertNotIn('last_evaluated_session', pf['positions'][0])
        self.assertFalse(pf['closed_trades'])

    def test_pending_fill_day_one_hold_exits_once_and_persists(self):
        pf = self.new()
        self.assertTrue(self.queue(pf, hold=1))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'])
        self.assertFalse(pf['pending_orders'])
        self.assertEqual(len(pf['closed_trades']), 1)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'hold_period')
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 1)
        self.assertEqual(pf['closed_trades'][0]['signal_date'], '2026-09-08')
        reloaded = engine.load_portfolio(self.app)
        before = copy.deepcopy(reloaded)
        engine.update_portfolio_prices(self.app, reloaded)
        self.assertEqual(reloaded, before)

    def test_gap_below_stop_distance_expires_instead_of_creating_zero_stop(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today],
                                     [(100, 101, 99, 100), (4, 5, 3, 4)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertFalse(pf['positions'])
        self.assertFalse(pf['pending_orders'])
        self.assertIn('stop', pf['expired_orders'][0]['reason'])
        self.assertEqual(pf['cash'], 10000)
        self.assertNotIn('sharesies_bought_usd', pf)

    def test_update_save_failure_preserves_unfilled_order_for_recovery(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today])
        before = copy.deepcopy(pf)
        disk = self.app.PORTFOLIO_JSON.read_bytes()
        with patch('screener_safety.os.replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf, before)
        self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), disk)


    def test_immature_watch_is_not_degraded_or_benchmark_fetched(self):
        self.new()
        self.today = '2026-09-09'
        pd.DataFrame([{'Date': '2026-09-08', 'Ticker': 'ABC', 'Signal': 'WATCH',
                       'Result': 'Pending'}]).to_csv(self.app.PICKS_CSV, index=False)
        self.yf.frames['ABC'] = bars(['2026-09-08', self.today, '2026-09-10'])
        self.yf.frames['QQQ'] = RuntimeError('not needed for an immature signal')
        before = self.app.PICKS_CSV.read_bytes()
        with patch.object(engine, 'evaluate_horizon', wraps=engine.evaluate_horizon) as evaluate:
            engine.update_results(self.app, self.app.PICKS_CSV, [])
        self.assertEqual(self.degraded, [])
        evaluate.assert_not_called()
        self.assertEqual([call[0] for call in self.yf.calls], ['ABC'])
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), before)

    def test_monday_signal_missing_tuesday_cannot_fill_wednesday_only(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars([self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(pf['expired_orders'][0]['execution_session'], '2026-09-15')
        self.assertEqual(self.yf.calls, [])
        self.assertEqual(pf['cash'], 10000)

    def test_permanent_quote_outage_still_expires_without_second_request(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.yf.frames['ABC'] = RuntimeError('permanently unavailable')
        self.today = '2026-09-09'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['pending_orders']), 1)
        self.today = '2026-09-10'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(len(pf['expired_orders']), 1)
        self.assertEqual(len(self.yf.calls), 1)

    def test_pending_execution_session_is_migrated_once_and_persisted(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        pf['pending_orders'][0].pop('execution_session', None)
        self.app._next_session_date = lambda day: pd.Timestamp('2026-09-10').date()
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['pending_orders'][0]['execution_session'], '2026-09-10')
        self.app._next_session_date = lambda day: self.fail('must reuse persisted session')
        self.today = '2026-09-09'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(self.yf.calls, [])
        self.assertEqual(engine.load_portfolio(self.app), pf)

    def test_calendar_hooks_choose_holiday_execution_not_first_ticker_bar(self):
        self.today = '2026-09-04'
        self.app._us_market_holiday = lambda day: day.isoformat() == '2026-09-07'
        self.assertEqual(engine.expected_execution_session(self.app, self.today), '2026-09-08')
        self.app._next_session_date = lambda day: pd.Timestamp('2026-09-09').date()
        self.assertEqual(engine.expected_execution_session(self.app, self.today), '2026-09-09')
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-08'
        self.yf.frames['ABC'] = bars([self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(self.yf.calls, [])
        self.assertEqual(len(pf['pending_orders']), 1)

    def test_default_hold_rsi_uses_thirty_preentry_bars(self):
        self.today = '2026-09-08'
        pf = self.new()
        self.open(pf, stop=95, target=140)
        self.app._CFG_RSI_EXIT = 78
        days = pd.bdate_range(end=self.today, periods=31)
        prices = [70 + i for i in range(31)]
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(list(days) + [pd.Timestamp(self.today)],
                                    [(p, p + 1, p - 1, p) for p in prices + [101]])
        engine.update_portfolio_prices(self.app, pf)
        trade = pf['closed_trades'][0]
        self.assertEqual(trade['reason'], 'rsi_overbought')
        self.assertEqual(trade['held_sessions'], 1)
        self.assertEqual(trade['hold_sessions'], 10)

    def test_missed_tuesday_stop_cannot_be_hidden_by_wednesday_recovery(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 99),
                                     (100, 105, 99, 104)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['closed_trades'][0]['exit_date'], '2026-09-15')
        self.assertEqual(pf['closed_trades'][0]['exit_price'], 96)
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 1)

    def test_tuesday_hole_blocks_all_staged_progress_until_repaired(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf, atr=1)
        before = copy.deepcopy(pf['positions'][0])
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars(['2026-09-14', self.today],
                                    [(100, 101, 99, 100), (100, 120, 90, 105)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'], [])
        pos = pf['positions'][0]
        self.assertTrue(pos['quote_stale'])
        self.assertIn('2026-09-15', pos['update_error'])
        self.assertIn('gap', pos['update_error'])
        for key, value in before.items():
            if key != 'quote_stale':
                self.assertEqual(pos[key], value, key)
        self.assertNotIn('last_evaluated_session', pos)
        self.assertIn('stale_quote:ABC', self.degraded)
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 99),
                                     (100, 120, 90, 105)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['exit_date'], '2026-09-15')

    def test_candidate_learning_aliases_survive_queue_fill_and_close(self):
        pf = self.new()
        self.assertTrue(engine.queue_position(
            self.app, pf, {'ticker': 'ABC', 'position_size_pct': 20}, 100, 96, 110,
            {'sector': 'Technology', 'earnings_days_away': 7,
             'congress_label': 'BUY', 'insider_label': 'NET BUY'}))
        expected = {'earnings_days': 7, 'congress': 'BUY', 'insider': 'NET BUY'}
        for key, value in expected.items():
            self.assertEqual(pf['pending_orders'][0][key], value)
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars([self.today])
        engine.update_portfolio_prices(self.app, pf)
        engine.close_position(self.app, pf, 'ABC', 101)
        engine.save_portfolio(self.app, pf)
        row = engine.load_performance_history(self.app, 'ignored.csv')[0]
        for key, value in expected.items():
            self.assertEqual(row[key], str(value))

    def test_learning_legacy_close_date_is_not_lost_or_invented(self):
        pf = self.new()
        self.open(pf)
        engine.close_position(self.app, pf, 'ABC', 100)
        trade = pf['closed_trades'][0]
        trade['close_date'] = trade.pop('exit_date')
        engine.save_portfolio(self.app, pf)
        self.assertEqual(engine.load_performance_history(self.app, 'ignored.csv')[0]['exit_date'], self.today)
        # save_portfolio publishes a validated copy, so edit the current record.
        pf['closed_trades'][0].pop('close_date', None)
        engine.save_portfolio(self.app, pf)
        self.assertEqual(engine.load_performance_history(self.app, 'ignored.csv')[0]['exit_date'], '')

    def test_unevaluated_split_requires_review_not_a_spurious_stop(self):
        pf = self.new()
        self.open(pf)
        before = copy.deepcopy(pf['positions'][0])
        self.today = '2026-09-10'
        frame = bars(['2026-09-08', '2026-09-09', self.today],
                     [(100, 101, 99, 100), (50, 51, 49, 50), (50, 51, 49, 50)])
        frame['Stock Splits'] = [0.0, 2.0, 0.0]
        self.yf.frames['ABC'] = frame
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'], [])
        pos = pf['positions'][0]
        self.assertTrue(pos['corporate_action_review_required'])
        self.assertTrue(pos['quote_stale'])
        self.assertEqual(pos['shares'], before['shares'])
        self.assertEqual(pos['cost_basis'], before['cost_basis'])
        self.assertNotIn('last_evaluated_session', pos)
        self.assertIn('corporate_action_review_required:ABC', self.degraded)


    def test_pending_single_old_quote_never_substitutes_for_expected_open(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-08-12'])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'], [])
        self.assertTrue(pf['pending_orders'][0]['quote_stale'])
        self.assertEqual(pf['cash'], 10000)
        self.today = '2026-09-10'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['expired_orders']), 1)
        self.assertEqual(len(self.yf.calls), 1)

    def test_calendar_weekend_and_new_year_saturday_hook(self):
        # NYSE remains open Fri 2021-12-31 when New Year falls on Saturday.
        self.app._us_market_holiday = lambda day: day.isoformat() in {
            '2022-01-01', '2023-01-02', '2026-12-25'}
        for signal, expected in [('2021-12-30', '2021-12-31'),
                                 ('2021-12-31', '2022-01-03'),
                                 ('2022-12-30', '2023-01-03'),
                                 ('2026-12-24', '2026-12-28')]:
            with self.subTest(signal=signal):
                self.assertEqual(engine.expected_execution_session(self.app, signal), expected)
        del self.app._us_market_holiday
        self.assertEqual(engine.expected_execution_session(self.app, '2026-09-11'), '2026-09-14')

    def test_holiday_pending_fills_exact_tuesday_only(self):
        self.today = '2026-09-04'
        self.app._us_market_holiday = lambda day: day.isoformat() == '2026-09-07'
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.assertEqual(pf['pending_orders'][0]['execution_session'], '2026-09-08')
        self.today = '2026-09-07'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(self.yf.calls, [])
        self.today = '2026-09-08'
        self.yf.frames['ABC'] = bars([self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['positions'][0]['execution_session'], self.today)
        self.assertEqual(pf['positions'][0]['held_sessions'], 1)

    def test_invalid_calendar_and_pending_session_fail_closed(self):
        pf = self.new()
        self.app._next_session_date = lambda day: day
        before = copy.deepcopy(pf)
        self.assertFalse(self.queue(pf))
        self.assertEqual(pf, before)
        del self.app._next_session_date
        self.assertTrue(self.queue(pf))
        for value in ('2026-09-07', '2026-09-08', 'not-a-date', None):
            with self.subTest(value=value):
                bad = copy.deepcopy(pf)
                bad['pending_orders'][0]['execution_session'] = value
                with self.assertRaises(ValueError):
                    engine.save_portfolio(self.app, bad)

    def test_mature_invalid_paper_data_degrades_only_in_advisory_namespace(self):
        self.new()
        self.today = '2026-09-10'
        pd.DataFrame([{'Date': '2026-09-08', 'Ticker': 'ABC', 'Signal': 'WATCH',
                       'Hold_Sessions': 2, 'Result': 'Pending'}]).to_csv(self.app.PICKS_CSV, index=False)
        self.yf.frames['ABC'] = bars(['2026-09-09', self.today],
                                    [(100, 90, 99, 100), (100, 101, 99, 100)])
        self.yf.frames['QQQ'] = bars(['2026-09-09', self.today])
        before = self.app.PICKS_CSV.read_bytes()
        engine.update_results(self.app, self.app.PICKS_CSV, [])
        self.assertEqual(self.degraded, ['paper_horizon_unavailable:ABC'])
        self.assertEqual(self.app.PICKS_CSV.read_bytes(), before)

    def test_default_hold_macd_has_preentry_warmup(self):
        prices = [75 + i * .8 for i in range(36)] + [103 - i * .8 for i in range(1, 9)]
        closes = pd.Series(prices)
        macd = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        cross = next(i for i in range(36, len(prices))
                     if macd[i] < signal[i] and macd[i - 1] >= signal[i - 1])
        days = pd.bdate_range(end='2026-09-09', periods=cross + 1)
        self.today = days[-2].date().isoformat()
        pf = self.new()
        entry = prices[cross - 1]
        self.open(pf, entry=entry, stop=entry - 20, target=entry + 40)
        self.app._CFG_MACD_EXIT_MIN_PROFIT = -100
        self.today = days[-1].date().isoformat()
        self.yf.frames['ABC'] = bars(days, [(p, p + .1, p - .1, p) for p in prices[:cross + 1]])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'macd_bearish_cross')
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 1)
        self.assertEqual(pf['closed_trades'][0]['hold_sessions'], 10)

    def test_replay_includes_actual_open_day_but_not_legacy_entry_day(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf, 'ABC', context={'filled_at_open': True})
        self.open(pf, 'XYZ', sector='Energy', context={})
        self.today = '2026-09-16'
        history = bars(['2026-09-14', '2026-09-15', self.today],
                       [(100, 101, 95, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
        self.yf.frames.update(ABC=history, XYZ=history)
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['exit_date'], '2026-09-14')
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 1)
        self.assertEqual(pf['positions'][0]['ticker'], 'XYZ')
        self.assertEqual(pf['positions'][0]['held_sessions'], 2)

    def test_replay_ratchets_after_each_bar_before_next_day_gap(self):
        pf = self.new()
        self.open(pf, atr=1)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today],
                                    [(100, 101, 99, 100), (100, 106, 97, 105),
                                     (102, 109, 101, 108)]).iloc[::-1]
        engine.update_portfolio_prices(self.app, pf)
        trade = pf['closed_trades'][0]
        self.assertEqual(trade['stop_price'], 103.5)
        self.assertEqual(trade['exit_price'], 102)
        self.assertEqual(trade['reason'], 'stop_loss')
        self.assertEqual(trade['held_sessions'], 2)

    def test_replay_hold_exits_on_first_due_session_not_runner_date(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf, context={'hold_sessions': 1})
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['reason'], 'hold_period')
        self.assertEqual(pf['closed_trades'][0]['exit_date'], '2026-09-15')
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 1)

    def test_replay_indicators_see_only_each_bar_and_earnings_only_asof(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today, '2026-09-17'])
        seen = []

        def indicator(app, pos, history):
            day = pos['last_evaluated_session']
            self.assertEqual(history.index.max().date().isoformat(), day)
            seen.append(day)
            return None

        with patch.object(engine, '_indicator_exit', side_effect=indicator), patch.object(
                engine, '_earnings_exit', return_value=None) as earnings:
            engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(seen, ['2026-09-15', '2026-09-16'])
        earnings.assert_called_once()
        self.assertEqual(earnings.call_args.args[-1], self.today)

    def test_replay_ignores_breaches_before_last_evaluated_cursor(self):
        pf = self.new()
        self.open(pf)
        pf['positions'][0].update(last_evaluated_session='2026-09-09', held_sessions=1, hold_days=1)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 100),
                                     (100, 101, 99, 100)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'], [])
        self.assertEqual(pf['positions'][0]['held_sessions'], 2)
        self.assertEqual(pf['positions'][0]['last_evaluated_session'], self.today)

    def test_replay_holiday_is_not_a_quote_gap_or_hold_session(self):
        self.today = '2026-09-04'
        self.app._us_market_holiday = lambda day: day.isoformat() == '2026-09-07'
        pf = self.new()
        self.open(pf, context={'hold_sessions': 2})
        self.today = '2026-09-09'
        self.yf.frames['ABC'] = bars(['2026-09-04', '2026-09-08', self.today])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'][0]['held_sessions'], 2)
        self.assertEqual(pf['closed_trades'][0]['exit_date'], self.today)
        self.assertEqual(self.degraded, [])

    def test_gap_after_a_good_bar_rolls_back_ratchet_and_hold_count(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf, atr=1)
        before = copy.deepcopy(pf['positions'][0])
        self.today = '2026-09-17'
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today],
                                    [(100, 101, 99, 100), (100, 106, 99, 105),
                                     (100, 101, 90, 99)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'], [])
        pos = pf['positions'][0]
        for key in ('current_price', 'current_value', 'stop_price', 'high_watermark', 'held_sessions'):
            self.assertEqual(pos[key], before[key])
        self.assertIn('2026-09-16', pos['update_error'])
        self.assertNotIn('last_evaluated_session', pos)

    def test_historical_stop_still_requires_valid_exact_asof_quote(self):
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09'],
                                    [(100, 101, 99, 100), (100, 101, 95, 100)])
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf['closed_trades'], [])
        self.assertTrue(pf['positions'][0]['quote_stale'])
        self.assertIn(self.today, pf['positions'][0]['update_error'])

    def test_replayed_fee_uses_exit_month_and_restores_parent_hooks(self):
        self.today = '2026-08-28'
        pf = self.new()
        self.open(pf)
        opening_cash = pf['cash']
        self.today = '2026-09-01'
        self.yf.frames['ABC'] = bars(['2026-08-28', '2026-08-31', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 99),
                                     (100, 105, 99, 104)])
        session_hook = self.app._session_date
        prior_context = {'unrelated': 'parent state'}
        self.app._CLOSE_CONTEXT = prior_context
        seen = []

        def monthly_fee(amount, state, nzdusd_rate=None, side='buy'):
            day = self.app._session_date()
            seen.append((day, side))
            month = day[:7]
            state.setdefault('fee_months', {})[month] = amount
            return 3.0 if month == '2026-08' else 8.0

        self.app._sharesies_fee = monthly_fee
        engine.update_portfolio_prices(self.app, pf)
        trade = pf['closed_trades'][0]
        self.assertEqual(seen, [('2026-08-31', 'sell')])
        self.assertEqual(trade['brokerage_out'], 3)
        self.assertEqual(trade['exit_date'], '2026-08-31')
        self.assertEqual(trade['replayed_asof'], self.today)
        self.assertEqual(trade['fee_basis'], 'runner_estimate_at_exit_session')
        self.assertEqual(pf['cash'], opening_cash + trade['exit_value'])
        self.assertEqual(pf['last_updated'], self.today)
        self.assertIs(self.app._session_date, session_hook)
        self.assertIs(self.app._CLOSE_CONTEXT, prior_context)

    def test_replay_fee_failure_restores_context_and_discards_position_stage(self):
        pf = self.new()
        self.open(pf, atr=1)
        before = copy.deepcopy(pf)
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 100),
                                     (100, 105, 99, 104)])
        original_hook = self.app._session_date

        def broken(amount, state, nzdusd_rate=None, side='sell'):
            self.assertEqual(self.app._session_date(), '2026-09-09')
            state['cash'] = 1
            raise RuntimeError('historical fee failed')

        self.app._sharesies_fee = broken
        engine.update_portfolio_prices(self.app, pf)
        self.assertIs(self.app._session_date, original_hook)
        self.assertFalse(hasattr(self.app, '_CLOSE_CONTEXT'))
        self.assertEqual(pf['cash'], before['cash'])
        self.assertEqual(pf['closed_trades'], [])
        self.assertEqual(pf['positions'][0]['current_price'], before['positions'][0]['current_price'])
        self.assertNotIn('last_evaluated_session', pf['positions'][0])
        self.assertIn('historical fee failed', pf['positions'][0]['update_error'])

    def test_replay_save_failure_is_atomic_and_restores_session_hook(self):
        pf = self.new()
        self.open(pf)
        engine.save_portfolio(self.app, pf)
        before, disk = copy.deepcopy(pf), self.app.PORTFOLIO_JSON.read_bytes()
        original_hook = self.app._session_date
        self.today = '2026-09-10'
        self.yf.frames['ABC'] = bars(['2026-09-08', '2026-09-09', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 100),
                                     (100, 105, 99, 104)])
        with patch('screener_safety.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(pf, before)
        self.assertEqual(self.app.PORTFOLIO_JSON.read_bytes(), disk)
        self.assertIs(self.app._session_date, original_hook)
        self.assertFalse(hasattr(self.app, '_CLOSE_CONTEXT'))

    def test_internal_close_date_rejects_future_preentry_and_external_context(self):
        pf = self.new()
        self.open(pf)
        before = copy.deepcopy(pf)
        calls = len(self.fee_calls)
        for day in ('2026-09-07', '2026-09-09'):
            with self.subTest(day=day), engine._close_context(self.app, day):
                engine.close_position(self.app, pf, 'ABC', 100)
                self.assertEqual(pf, before)
        self.assertFalse(hasattr(self.app, '_CLOSE_CONTEXT'))
        self.app._CLOSE_CONTEXT = {'exit_date': '2026-09-08'}
        engine.close_position(self.app, pf, 'ABC', 100)
        self.assertEqual(pf, before)
        self.assertIn('internal-only', self.app._ORDER_REASON[0])
        self.assertEqual(len(self.fee_calls), calls)

    def test_multiple_replay_closes_commit_in_exit_date_order(self):
        self.today = '2026-09-14'
        pf = self.new()
        self.open(pf, 'ABC')
        self.open(pf, 'XYZ', sector='Energy')
        self.today = '2026-09-16'
        self.yf.frames['ABC'] = bars(['2026-09-14', '2026-09-15', self.today],
                                    [(100, 101, 99, 100), (100, 101, 99, 100),
                                     (100, 101, 95, 100)])
        self.yf.frames['XYZ'] = bars(['2026-09-14', '2026-09-15', self.today],
                                    [(100, 101, 99, 100), (100, 101, 95, 100),
                                     (100, 101, 99, 100)])
        seen = []

        def dated_fee(amount, state, nzdusd_rate=None, side='sell'):
            seen.append(self.app._session_date())
            return self.fee(amount, state, nzdusd_rate, side)

        self.app._sharesies_fee = dated_fee
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual([t['ticker'] for t in pf['closed_trades']], ['XYZ', 'ABC'])
        self.assertEqual(seen, ['2026-09-15', '2026-09-16'])

    def test_pending_split_blocks_fill_then_expires_without_cash_adjustment(self):
        pf = self.new()
        self.assertTrue(self.queue(pf))
        self.today = '2026-09-09'
        frame = bars(['2026-09-08', self.today], [(100, 101, 99, 100), (50, 51, 49, 50)])
        frame['Stock Splits'] = [0.0, 2.0]
        self.yf.frames['ABC'] = frame
        engine.update_portfolio_prices(self.app, pf)
        self.assertTrue(pf['pending_orders'][0]['corporate_action_review_required'])
        self.assertEqual(pf['positions'], [])
        self.assertEqual(pf['cash'], 10000)
        self.today = '2026-09-10'
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(len(pf['expired_orders']), 1)
        self.assertEqual(len(self.yf.calls), 1)

    def test_split_before_owned_interval_or_after_asof_does_not_block(self):
        pf = self.new()
        self.open(pf)
        self.today = '2026-09-09'
        frame = bars(['2026-09-07', '2026-09-08', self.today, '2026-09-10'])
        frame['Stock Splits'] = [2.0, 0.0, 0.0, 2.0]
        self.yf.frames['ABC'] = frame
        engine.update_portfolio_prices(self.app, pf)
        self.assertEqual(self.degraded, [])
        self.assertEqual(pf['positions'][0]['held_sessions'], 1)
        self.assertFalse(pf['positions'][0]['quote_stale'])

    def test_legacy_cost_estimates_and_aliases_preserved_without_reaccounting(self):
        pf = self.new()
        self.open(pf)
        engine.close_position(self.app, pf, 'ABC', 101)
        trade = pf['closed_trades'][0]
        del trade['cost_basis_basis']
        trade.update(earnings_days_away=4, congress_label='BUY', insider_label='SELL')
        cost, cash = trade['cost_basis'], pf['cash']
        atomic_json(self.app.PORTFOLIO_JSON, pf)
        loaded = engine.load_portfolio(self.app)
        self.assertEqual(loaded['closed_trades'][0]['cost_basis_basis'], 'legacy_estimate')
        self.assertEqual(loaded['closed_trades'][0]['cost_basis'], cost)
        self.assertEqual(loaded['cash'], cash)
        row = engine.load_performance_history(self.app, 'ignored.csv')[0]
        self.assertEqual((row['earnings_days'], row['congress'], row['insider']), ('4', 'BUY', 'SELL'))

    def test_canonical_metadata_wins_over_alias_in_same_record(self):
        values = engine._learning_metadata({'earnings_days': 0, 'earnings_days_away': 7,
                                            'congress': 'NONE', 'congress_label': 'BUY',
                                            'insider': 'NONE', 'insider_label': 'SELL'})
        self.assertEqual(values, {'earnings_days': 0, 'congress': 'NONE', 'insider': 'NONE'})


if __name__ == '__main__':
    unittest.main()