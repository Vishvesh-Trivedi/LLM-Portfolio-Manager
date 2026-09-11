"""Offline unittest regressions; all transient I/O stays beneath lpm_fix."""

import ast
import copy
import inspect
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import screener_safety as safety


ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    # A regression must never accidentally turn these isolated tests into live I/O.
    for target in ('socket.socket', 'socket.create_connection', 'socket.getaddrinfo'):
        guard = patch(target, side_effect=AssertionError('network forbidden'))
        guard.start()
        unittest.addModuleCleanup(guard.stop)


def portfolio(cash=10000, positions=None, closed=None, capital=10000):
    return {'cash': cash, 'starting_capital': capital,
            'positions': positions or [], 'closed_trades': closed or []}


def position(ticker='AAA', **changes):
    result = {'ticker': ticker, 'shares': 10, 'entry_price': 100,
              'cost_basis': 1000, 'entry_date': '2026-08-03',
              'sector': 'Technology', 'stop_price': 95, 'target_price': 110}
    result.update(changes)
    return result


def closed_trade(**changes):
    result = position(exit_price=110, realized_pnl=100)
    result.update(changes)
    return result


def bars(dates, opening=100, close=105, high=120, low=90):
    return pd.DataFrame({'Open': opening, 'High': high, 'Low': low, 'Close': close},
                        index=pd.to_datetime(dates))


def order(pf=None, **changes):
    arguments = dict(ticker='NEW', entry_price=100, amount_usd=2500,
                     stop=99, target=102, sector='Technology', fee_quote=lambda _: 0.0)
    arguments.update(changes)
    return safety.plan_order(portfolio() if pf is None else pf, **arguments)


class NumberTests(unittest.TestCase):
    def test_finite_number_accepts_only_finite_ints_floats_and_inclusive_bounds(self):
        for value in (0, 1, -1, 1.25):
            self.assertEqual(safety.finite_number(value), value)
        self.assertEqual(safety.finite_number(5, minimum=5, maximum=5), 5)

    def test_finite_number_rejects_malformed_numbers_nan_and_infinity(self):
        for value in (True, False, '1', None, [], complex(1), float('nan'),
                      float('inf'), -float('inf'), 10 ** 1000):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                safety.finite_number(value, 'test')

    def test_finite_number_rejects_out_of_range_and_invalid_bounds(self):
        for kwargs in ({'minimum': 2}, {'maximum': 0}, {'minimum': float('nan')},
                       {'maximum': '2'}, {'minimum': 2, 'maximum': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                safety.finite_number(1, **kwargs)


class AtomicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.destination = self.folder / 'state'
        self.destination.write_bytes(b'original bytes')

    def assert_original_and_no_staging(self):
        self.assertEqual(self.destination.read_bytes(), b'original bytes')
        self.assertEqual(list(self.folder.iterdir()), [self.destination])

    def test_atomic_json_roundtrip_unicode(self):
        data = {'cash': 100, 'label': '\u03bb', 'positions': []}
        safety.atomic_json(self.destination, data)
        self.assertEqual(json.loads(self.destination.read_text(encoding='utf-8')), data)
        self.assertEqual(list(self.folder.iterdir()), [self.destination])

    def test_failed_atomic_json_save_retains_original_on_nan_corruption(self):
        for data in ({'x': float('nan')}, {'x': [float('inf')]}, {'x': object()}):
            with self.subTest(data=data):
                with patch.object(safety.tempfile, 'NamedTemporaryFile') as stage:
                    with self.assertRaises((ValueError, TypeError)):
                        safety.atomic_json(self.destination, data)
                    stage.assert_not_called()
                self.assert_original_and_no_staging()

    def test_atomic_json_circular_data_retains_original(self):
        data = []
        data.append(data)
        with self.assertRaises(ValueError):
            safety.atomic_json(self.destination, data)
        self.assert_original_and_no_staging()

    def test_atomic_json_fsync_failure_retains_original_and_cleans_temp(self):
        with patch.object(safety.os, 'fsync', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                safety.atomic_json(self.destination, {'ok': 1})
        self.assert_original_and_no_staging()

    def test_atomic_json_replace_failure_retains_original_and_cleans_temp(self):
        with patch.object(safety.os, 'replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                safety.atomic_json(self.destination, {'ok': 1})
        self.assert_original_and_no_staging()

    def test_atomic_json_encoding_failure_retains_original_and_cleans_temp(self):
        with self.assertRaises(UnicodeEncodeError):
            safety.atomic_json(self.destination, {'invalid_surrogate': '\ud800'})
        self.assert_original_and_no_staging()

    def test_atomic_json_flush_failure_retains_original_and_cleans_temp(self):
        real_factory = safety.tempfile.NamedTemporaryFile

        def failing_stream(*args, **kwargs):
            stream = real_factory(*args, **kwargs)
            stream.flush = lambda: (_ for _ in ()).throw(OSError('flush failed'))
            return stream

        with patch.object(safety.tempfile, 'NamedTemporaryFile', side_effect=failing_stream):
            with self.assertRaises(OSError):
                safety.atomic_json(self.destination, {'ok': 1})
        self.assert_original_and_no_staging()

    def test_atomic_helpers_stage_sibling_flush_fsync_before_replace(self):
        real_fsync, real_replace = os.fsync, os.replace
        for save, data in ((safety.atomic_json, {'ok': 1}),
                           (safety.atomic_csv, pd.DataFrame({'ok': [1]}))):
            events = []

            def sync(fd):
                self.assertGreater(os.fstat(fd).st_size, 0)  # flushed before fsync
                events.append('fsync')
                real_fsync(fd)

            def replace(source, target):
                self.assertEqual(events, ['fsync'])
                self.assertEqual(Path(source).parent, self.destination.parent)
                self.assertNotEqual(Path(source), Path(target))
                events.append('replace')
                real_replace(source, target)

            with patch.object(safety.os, 'fsync', side_effect=sync), \
                    patch.object(safety.os, 'replace', side_effect=replace):
                save(self.destination, data)
            self.assertEqual(events, ['fsync', 'replace'])

    def test_atomic_csv_writes_no_index_and_does_not_mutate_dataframe(self):
        frame = pd.DataFrame({'ticker': ['A,B'], 'shares': [2]}, index=[99])
        before = frame.copy(deep=True)
        safety.atomic_csv(self.destination, frame)
        self.assertEqual(pd.read_csv(self.destination).to_dict('list'),
                         {'ticker': ['A,B'], 'shares': [2]})
        pd.testing.assert_frame_equal(frame, before)

    def test_atomic_csv_partial_write_failure_retains_original(self):
        def fail(frame, stream, index):
            self.assertFalse(index)
            stream.write('partial CSV')
            raise OSError('disk full')

        with patch.object(pd.DataFrame, 'to_csv', fail):
            with self.assertRaises(OSError):
                safety.atomic_csv(self.destination, pd.DataFrame({'x': [1]}))
        self.assert_original_and_no_staging()

    def test_atomic_csv_fsync_and_replace_failures_retain_original(self):
        for method in ('fsync', 'replace'):
            with self.subTest(method=method):
                with patch.object(safety.os, method, side_effect=OSError('failure')):
                    with self.assertRaises(OSError):
                        safety.atomic_csv(self.destination, pd.DataFrame({'x': [1]}))
                self.assert_original_and_no_staging()

    def test_atomic_helpers_can_create_destination_and_reject_missing_parent(self):
        new = self.folder / 'new'
        safety.atomic_json(new, {'ok': 1})
        self.assertTrue(new.exists())
        with self.assertRaises(FileNotFoundError):
            safety.atomic_json(self.folder / 'missing' / 'new', {})


class PortfolioTests(unittest.TestCase):
    def test_validate_portfolio_returns_same_object_and_ignores_extra_keys(self):
        pf = portfolio(positions=[position(current_value=0)], closed=[closed_trade(ticker='OLD')])
        pf['future_schema'] = {'opaque': 'kept'}
        self.assertIs(safety.validate_portfolio(pf), pf)
        self.assertEqual(pf['positions'][0]['current_value'], 0)
        self.assertEqual(pf['future_schema'], {'opaque': 'kept'})

    def test_validate_portfolio_requires_all_root_fields_without_reset(self):
        for key in portfolio():
            pf = portfolio()
            del pf[key]
            before = copy.deepcopy(pf)
            with self.subTest(key=key), self.assertRaises(ValueError):
                safety.validate_portfolio(pf)
            self.assertEqual(pf, before)

    def test_validate_portfolio_rejects_root_nan_corruption_and_wrong_types(self):
        cases = [('cash', value) for value in (-1, True, '100', math.nan, math.inf)]
        cases += [('starting_capital', value) for value in (0, -1, False, math.nan)]
        cases += [(key, value) for key in ('positions', 'closed_trades')
                  for value in (None, {}, ())]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pf = portfolio()
                pf[key] = value
                safety.validate_portfolio(pf)

    def test_validate_portfolio_rejects_invalid_position_numbers_and_missing_fields(self):
        for key in ('ticker', 'shares', 'entry_price', 'cost_basis'):
            record = position()
            del record[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                safety.validate_portfolio(portfolio(positions=[record]))
        cases = [('ticker', v) for v in ('', '  ', None, 12)]
        cases += [('shares', v) for v in (0, -1, 1.0, True, '1', math.nan)]
        cases += [(key, v) for key in ('entry_price', 'cost_basis')
                  for v in (0, -1, True, '1', math.nan, math.inf)]
        cases += [('current_value', v) for v in (-1, None, '0', False, math.nan)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                safety.validate_portfolio(portfolio(positions=[position(**{key: value})]))

    def test_closed_trades_require_positive_prices_shares_cost_and_finite_pnl(self):
        for key in ('entry_price', 'exit_price', 'shares', 'cost_basis', 'realized_pnl'):
            record = closed_trade()
            del record[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                safety.validate_portfolio(portfolio(closed=[record]))
            for value in ((math.nan, math.inf, '1', True) if key == 'realized_pnl'
                          else (0, -1, math.nan, True)):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    safety.validate_portfolio(portfolio(closed=[closed_trade(**{key: value})]))
        safety.validate_portfolio(portfolio(closed=[closed_trade(realized_pnl=-50)]))

    def test_closed_trade_pnl_alias_supported_and_both_fields_checked(self):
        trade = closed_trade(pnl=-100)
        del trade['realized_pnl']
        safety.validate_portfolio(portfolio(closed=[trade]))
        trade['realized_pnl'] = math.nan
        with self.assertRaises(ValueError):
            safety.validate_portfolio(portfolio(closed=[trade]))

    def test_same_ticker_reentry_id_differences_for_date_price_and_shares(self):
        trades = [position(), position(entry_date='2026-08-04'),
                  position(entry_price=101), position(shares=11)]
        pf = portfolio(positions=trades)
        safety.validate_portfolio(pf)
        self.assertEqual(len({p['trade_id'] for p in trades}), 4)
        clone = portfolio(positions=[position(), position(entry_date='2026-08-04'),
                                     position(entry_price=101), position(shares=11)])
        safety.validate_portfolio(clone)
        self.assertEqual([p['trade_id'] for p in trades],
                         [p['trade_id'] for p in clone['positions']])

    def test_closed_and_open_migrations_use_same_identity_not_exit_details(self):
        opened = portfolio(positions=[position()])
        closed = portfolio(closed=[closed_trade(exit_price=150, realized_pnl=500)])
        safety.validate_portfolio(opened)
        safety.validate_portfolio(closed)
        self.assertEqual(opened['positions'][0]['trade_id'], closed['closed_trades'][0]['trade_id'])

    def test_migration_normalizes_int_float_price_and_ticker_case(self):
        a = portfolio(positions=[position()])
        b = portfolio(positions=[position(ticker=' aaa ', entry_price=100.0)])
        safety.validate_portfolio(a)
        safety.validate_portfolio(b)
        self.assertEqual(a['positions'][0]['trade_id'], b['positions'][0]['trade_id'])

    def test_existing_trade_id_is_never_recomputed(self):
        trade = position(trade_id='stable-id', entry_price=120, shares=3)
        del trade['entry_date']
        pf = portfolio(positions=[trade])
        safety.validate_portfolio(pf)
        safety.validate_portfolio(pf)
        self.assertEqual(trade['trade_id'], 'stable-id')

    def test_duplicate_ids_across_open_closed_and_ambiguous_migrations_rejected(self):
        for pf in (portfolio(positions=[position(trade_id='same')],
                             closed=[closed_trade(trade_id='same')]),
                   portfolio(positions=[position()], closed=[closed_trade()])):
            with self.assertRaises(ValueError):
                safety.validate_portfolio(pf)

    def test_invalid_ids_and_missing_migration_identity_do_not_reset(self):
        for value in ('', None, 1, '  '):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safety.validate_portfolio(portfolio(positions=[position(trade_id=value)]))
        trade = position()
        del trade['entry_date']
        with self.assertRaises(ValueError):
            safety.validate_portfolio(portfolio(positions=[trade]))

    def test_failed_validation_commits_no_partial_id_migrations(self):
        pf = portfolio(positions=[position(), position(ticker='BAD', shares=0)])
        before = copy.deepcopy(pf)
        with self.assertRaises(ValueError):
            safety.validate_portfolio(pf)
        self.assertEqual(pf, before)


class OrderTests(unittest.TestCase):
    def test_plan_order_exact_output_signature_and_nonmutation(self):
        pf = portfolio(cash=9000, positions=[position(sector='Energy')])
        before = copy.deepcopy(pf)
        calls = []

        def quote(notional):
            self.assertIs(type(notional), float)
            self.assertEqual(pf, before)
            calls.append(notional)
            return 1.0

        result = order(pf, fee_quote=quote)
        self.assertEqual(result, {'shares': 24, 'stock_cost': 2400.0,
                                  'brokerage': 1.0, 'total_cost': 2401.0, 'amount_usd': 2500})
        self.assertIs(type(result['shares']), int)
        self.assertEqual(pf, before)
        self.assertIn(24 * 99.0, calls)

    def test_cash_buffer_plus_fee_exact_boundary_and_one_cent_below(self):
        for cash, succeeds in ((605, True), (604.99, False)):
            pf = portfolio(cash=cash, capital=cash)
            if succeeds:
                result = order(pf, amount_usd=200, fee_quote=lambda _: 5 if _ >= 100 else 0)
                self.assertEqual(result['shares'], 1)
                self.assertEqual(cash - result['total_cost'], 500)
            else:
                with self.assertRaises(ValueError):
                    order(pf, amount_usd=200, fee_quote=lambda _: 5 if _ >= 100 else 0)

    def test_hard_cash_floor_cannot_be_relaxed_and_stricter_config_respected(self):
        with self.assertRaises(ValueError):
            order(portfolio(cash=599, capital=599), cash_floor=0)
        result = order(portfolio(cash=700, capital=700), cash_floor=600)
        self.assertEqual(result['total_cost'], 100)

    def test_amount_is_fee_inclusive_cap_never_forced_minimum(self):
        for amount in (0, 0.1, 99, 100):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                order(amount_usd=amount, fee_quote=lambda _: 1.0)
        result = order(amount_usd=101, fee_quote=lambda _: 1.0)
        self.assertEqual(result['shares'], 1)
        self.assertEqual(result['amount_usd'], 101)

    def test_per_stock_cap_is_25_percent_equity(self):
        self.assertEqual(order(amount_usd=9000)['shares'], 25)

    def test_sector_cap_uses_existing_market_values_not_costs(self):
        pf = portfolio(cash=7000, positions=[position(current_value=3000)])
        self.assertEqual(order(pf)['shares'], 10)

    def test_sector_cap_aliases_match_and_cost_basis_fallback(self):
        pf = portfolio(cash=7000, positions=[position(cost_basis=3000, sector='Information Technology')])
        self.assertEqual(order(pf, sector=' technology ')['shares'], 10)
        pf['positions'][0]['cost_basis'] = 5000
        with self.assertRaises(ValueError):
            order(pf)

    def test_zero_current_value_is_used_not_replaced_with_cost_basis(self):
        pf = portfolio(cash=10000, positions=[position(current_value=0, cost_basis=9000)])
        self.assertEqual(order(pf, amount_usd=9000)['shares'], 25)

    def test_risk_cap_includes_buy_and_estimated_stop_sale_fees(self):
        result = order(stop=90, target=115, fee_quote=lambda _: 5.0)
        self.assertEqual(result['shares'], 9)
        self.assertEqual(result['shares'] * 10 + 5 + 5, 100)

    def test_risk_cap_quotes_sell_notional_at_stop(self):
        result = order(stop=90, target=115, fee_quote=lambda cost: cost * 0.01)
        self.assertEqual(result['shares'], 8)
        self.assertLessEqual(8 * 10 + 8 + 7.2, 100)

    def test_quotes_receive_rounded_stock_cost_and_rounded_stop_proceeds(self):
        calls = []

        def quote(cost):
            calls.append(cost)
            return 0.0

        result = order(entry_price=100.1234, stop=99.1234, target=102.1234,
                       amount_usd=101, fee_quote=quote)
        self.assertEqual(calls, [result['stock_cost'], 99.12])
        self.assertEqual(result['stock_cost'], 100.12)

    def test_fee_cent_rounding_does_not_allow_risk_or_cash_overrun(self):
        with self.assertRaises(ValueError):
            order(portfolio(cash=601.004, capital=601.004), amount_usd=101.004,
                  fee_quote=lambda _: 1.006)
        result = order(stop=90, target=115, fee_quote=lambda _: 5.001)
        self.assertEqual(result['shares'], 8)

    def test_fee_search_does_not_assume_monotonic_quotes(self):
        def quote(cost):
            return 0.0 if cost in (2300.0, 2277.0) else 10000.0
        self.assertEqual(order(fee_quote=quote)['shares'], 23)

    def test_max_positions_hard_ceiling_five_and_lower_config(self):
        for count, limit in ((5, 100), (3, 3)):
            positions = [position(ticker=f'T{i}', current_value=100) for i in range(count)]
            with self.subTest(count=count), self.assertRaises(ValueError):
                order(portfolio(positions=positions), max_positions=limit)
        positions = [position(ticker=f'T{i}', current_value=100) for i in range(4)]
        self.assertGreater(order(portfolio(positions=positions), max_positions=100)['shares'], 0)

    def test_duplicate_ticker_case_and_whitespace_rejected(self):
        with self.assertRaises(ValueError):
            order(portfolio(positions=[position(ticker='new')]), ticker=' NEW ')

    def test_unknown_candidate_or_existing_sector_rejected(self):
        for sector in ('', 'Unknown', 'N/A', None, 'fictional sector'):
            with self.subTest(sector=sector), self.assertRaises(ValueError):
                order(sector=sector)
        with self.assertRaises(ValueError):
            order(portfolio(positions=[position(sector='Unknown')]))

    def test_drawdown_at_twenty_percent_starting_capital_blocks(self):
        for cash in (8000, 7999):
            with self.subTest(cash=cash), self.assertRaises(ValueError):
                order(portfolio(cash=cash))
        self.assertGreater(order(portfolio(cash=8000.01))['shares'], 0)

    def test_drawdown_uses_higher_equity_peak_and_includes_positions(self):
        pf = portfolio()
        pf['equity_peak'] = 12500
        with self.assertRaises(ValueError):
            order(pf)
        pf = portfolio(cash=7500, positions=[position(current_value=600, sector='Energy')])
        self.assertGreater(order(pf)['shares'], 0)
        pf['equity_peak'] = math.nan
        with self.assertRaises(ValueError):
            order(pf)

    def test_malformed_order_prices_amounts_config_and_fees_rejected(self):
        for key in ('entry_price', 'stop', 'target', 'amount_usd', 'cash_floor'):
            for value in (True, '100', None, math.nan, math.inf, -1):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    order(**{key: value})
        for key in ('entry_price', 'stop', 'target'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                order(**{key: 0})
        for value in (True, 1.5, 0, -1, '5'):
            with self.subTest(max_positions=value), self.assertRaises(ValueError):
                order(max_positions=value)
        for fee in (-1, math.nan, math.inf, True, '0', None):
            with self.subTest(fee=fee), self.assertRaises(ValueError):
                order(fee_quote=lambda _, value=fee: value)
        with self.assertRaises(ValueError):
            order(fee_quote=None)

    def test_sell_fee_alone_is_validated(self):
        with self.assertRaises(ValueError):
            order(amount_usd=100, fee_quote=lambda cost: 0 if cost == 100 else math.nan)

    def test_stop_entry_target_order_and_reward_risk_boundary(self):
        for stop, target in ((100, 110), (101, 120), (90, 100), (90, 114.99)):
            with self.subTest(stop=stop, target=target), self.assertRaises(ValueError):
                order(stop=stop, target=target)
        self.assertEqual(order(stop=90, target=115)['shares'], 10)

    def test_failed_order_and_fee_exception_do_not_migrate_or_mutate_portfolio(self):
        pf = portfolio(positions=[position(sector='Energy')])
        before = copy.deepcopy(pf)
        with self.assertRaises(ValueError):
            order(pf, amount_usd=1)
        self.assertEqual(pf, before)
        with self.assertRaises(RuntimeError):
            order(pf, fee_quote=lambda _: (_ for _ in ()).throw(RuntimeError('quote failed')))
        self.assertEqual(pf, before)

    def test_deterministic_orders_match_independent_exhaustive_cap_checks(self):
        for cash in (599, 605, 700, 8000, 10000):
            for existing in (0, 1000, 3000):
                for amount in (99, 101, 2500, 9000):
                    equity = cash + existing
                    pf = portfolio(cash=cash, capital=equity,
                                   positions=[position(current_value=existing)])
                    allowed = [n for n in range(1, 101)
                               if n * 100 + 1 <= min(amount, cash - 500)
                               and n * 100 <= equity * 0.25
                               and existing + n * 100 <= equity * 0.40
                               and n + 2 <= equity * 0.01]
                    with self.subTest(cash=cash, existing=existing, amount=amount):
                        if not allowed:
                            with self.assertRaises(ValueError):
                                order(pf, amount_usd=amount, fee_quote=lambda _: 1.0)
                        else:
                            first = order(pf, amount_usd=amount, fee_quote=lambda _: 1.0)
                            second = order(pf, amount_usd=amount, fee_quote=lambda _: 1.0)
                            self.assertEqual(first, second)
                            self.assertEqual(first['shares'], max(allowed))


class HorizonTests(unittest.TestCase):
    def setUp(self):
        # Friday before US Labor Day, then Tuesday/Wednesday/Thursday.
        self.dates = ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10']
        self.stock = bars(self.dates)
        self.benchmark = bars(self.dates, opening=200, close=204, high=210, low=195)

    def evaluate(self, **changes):
        arguments = dict(stock=self.stock, benchmark=self.benchmark,
                         signal_date='2026-09-04', hold_sessions=2, as_of='2026-09-09')
        arguments.update(changes)
        return safety.evaluate_horizon(**arguments)

    def test_horizon_holiday_weekend_next_session_and_nth_session_close(self):
        result = self.evaluate()
        self.assertEqual(result['entry_date'], '2026-09-08')
        self.assertEqual(result['exit_date'], '2026-09-09')
        self.assertEqual(result['entry_price'], 100)
        self.assertEqual(result['exit_price'], 105)
        self.assertAlmostEqual(result['return_pct'], 5)
        self.assertAlmostEqual(result['benchmark_return_pct'], 2)
        self.assertAlmostEqual(result['excess_return_pct'], 3)

    def test_weekend_signal_uses_next_stock_session(self):
        self.assertEqual(self.evaluate(signal_date='2026-09-05')['entry_date'], '2026-09-08')

    def test_single_session_opens_and_closes_same_day(self):
        result = self.evaluate(hold_sessions=1, as_of='2026-09-08')
        self.assertEqual(result['entry_date'], result['exit_date'])
        self.assertAlmostEqual(result['return_pct'], 5)

    def test_horizon_never_uses_signal_day_or_forward_bars(self):
        self.assertIsNone(self.evaluate(as_of='2026-09-08'))
        self.assertIsNone(self.evaluate(as_of='2026-09-04', hold_sessions=1))
        expected = self.evaluate()
        self.stock = self.stock.astype(float)
        self.benchmark = self.benchmark.astype(float)
        self.stock.loc['2026-09-10', :] = math.nan
        self.benchmark.loc['2026-09-10', :] = math.inf
        self.assertEqual(self.evaluate(), expected)

    def test_horizon_requires_positive_integer_hold_sessions(self):
        for value in (0, -1, 1.0, True, '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(hold_sessions=value)

    def test_missing_benchmark_entry_exit_or_middle_date_returns_none(self):
        for date in self.dates[1:]:
            with self.subTest(date=date):
                benchmark = self.benchmark.drop(pd.Timestamp(date))
                self.assertIsNone(self.evaluate(benchmark=benchmark, hold_sessions=3,
                                               as_of='2026-09-10'))

    def test_exact_benchmark_dates_not_same_row_offsets(self):
        benchmark = bars(['2026-09-07', '2026-09-08', '2026-09-09'],
                         opening=200, close=204, high=210, low=195)
        benchmark.loc['2026-09-07', 'Open'] = 199
        self.assertAlmostEqual(self.evaluate(benchmark=benchmark)['benchmark_return_pct'], 2)
        self.assertIsNone(self.evaluate(benchmark=benchmark.drop(pd.Timestamp('2026-09-08'))))

    def test_horizon_local_date_not_timezone_conversion(self):
        stock = bars(['2026-09-08 00:30', '2026-09-09 00:30'])
        stock.index = stock.index.tz_localize('Pacific/Auckland')
        benchmark = bars(['2026-09-08 23:00', '2026-09-09 23:00'])
        benchmark.index = benchmark.index.tz_localize('America/New_York')
        result = self.evaluate(stock=stock, benchmark=benchmark,
                               as_of=pd.Timestamp('2026-09-09 00:01', tz='Pacific/Auckland'))
        self.assertEqual(result['entry_date'], '2026-09-08')
        self.assertEqual(result['exit_date'], '2026-09-09')

    def test_horizon_rejects_nan_zero_and_incoherent_prices_in_either_frame(self):
        for source in ('stock', 'benchmark'):
            for column, value in (('Open', 0), ('Close', math.nan), ('High', math.inf),
                                  ('Low', -1), ('High', 1), ('Low', 1000)):
                frame = getattr(self, source).copy()
                frame[column] = frame[column].astype(float)
                frame.loc['2026-09-08', column] = value
                with self.subTest(source=source, column=column, value=value):
                    self.assertIsNone(self.evaluate(**{source: frame}))

    def test_horizon_sorts_sessions_without_mutating_inputs(self):
        stock, benchmark = self.stock.iloc[::-1].copy(), self.benchmark.iloc[::-1].copy()
        stock_before, benchmark_before = stock.copy(), benchmark.copy()
        self.assertEqual(self.evaluate(stock=stock, benchmark=benchmark), self.evaluate())
        pd.testing.assert_frame_equal(stock, stock_before)
        pd.testing.assert_frame_equal(benchmark, benchmark_before)

    def test_horizon_rejects_duplicate_dates_empty_frames_and_missing_columns(self):
        for frame in (pd.concat([self.stock, self.stock.iloc[[1]]]),
                      self.stock.iloc[:0], self.stock.drop(columns='Open'), None):
            with self.subTest(frame=type(frame)):
                self.assertIsNone(self.evaluate(stock=frame))
        self.assertIsNone(self.evaluate(benchmark=pd.concat([self.benchmark, self.benchmark.iloc[[1]]])))


class FreshBarTests(unittest.TestCase):
    def test_fresh_bar_exact_day_and_no_input_mutation(self):
        frame = bars(['2026-09-09', '2026-09-10'])
        before = frame.copy()
        self.assertEqual(safety.fresh_bar(frame, '2026-09-09'),
                         {'Open': 100, 'High': 120, 'Low': 90, 'Close': 105,
                          'date': '2026-09-09'})
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_quotes_stale_or_future_only_return_none(self):
        for frame in (bars(['2026-09-08']), bars(['2026-09-10']),
                      bars([]), pd.DataFrame(), None):
            self.assertIsNone(safety.fresh_bar(frame, '2026-09-09'))

    def test_missing_exact_day_quote_does_not_fall_back_to_valid_previous_day(self):
        frame = bars(['2026-09-08', '2026-09-09']).astype(float)
        frame.loc['2026-09-09', 'Close'] = math.nan
        self.assertIsNone(safety.fresh_bar(frame, '2026-09-09'))

    def test_fresh_bar_rejects_all_nonpositive_nonfinite_or_malformed_ohlc(self):
        for column in ('Open', 'High', 'Low', 'Close'):
            for value in (0, -1, math.nan, math.inf, '100', True, None):
                frame = bars(['2026-09-09']).astype(object)
                frame.loc['2026-09-09', column] = value
                with self.subTest(column=column, value=value):
                    self.assertIsNone(safety.fresh_bar(frame, '2026-09-09'))

    def test_fresh_bar_requires_consistent_high_and_low(self):
        for high, low in ((99, 90), (120, 101), (80, 90)):
            self.assertIsNone(safety.fresh_bar(bars(['2026-09-09'], high=high, low=low), '2026-09-09'))
        self.assertIsNotNone(safety.fresh_bar(bars(['2026-09-09'], opening=100,
                                                  close=100, high=100, low=100), '2026-09-09'))

    def test_fresh_bar_never_falls_back_to_another_ticker(self):
        frame = bars(['2026-09-09'])
        multi = pd.concat({'OTHER': frame}, axis=1)
        self.assertIsNone(safety.fresh_bar(multi, '2026-09-09'))
        self.assertIsNone(safety.fresh_bar(multi.swaplevel(axis=1), '2026-09-09'))
        self.assertIsNone(safety.fresh_bar(pd.DataFrame({'OTHER': [100]}, index=frame.index), '2026-09-09'))

    def test_fresh_bar_local_day_duplicate_day_and_duplicate_columns(self):
        frame = bars(['2026-09-09 00:30'])
        frame.index = frame.index.tz_localize('Pacific/Auckland')
        self.assertIsNotNone(safety.fresh_bar(frame, '2026-09-09'))
        self.assertIsNone(safety.fresh_bar(frame, '2026-09-08'))
        duplicate = bars(['2026-09-09 09:00', '2026-09-09 16:00'])
        self.assertIsNone(safety.fresh_bar(duplicate, '2026-09-09'))
        self.assertIsNone(safety.fresh_bar(pd.concat([frame, frame[['Open']]], axis=1), '2026-09-09'))

    def test_fresh_bar_rejects_non_datetime_index_and_invalid_as_of(self):
        frame = bars(['2026-09-09'])
        frame.index = ['2026-09-09']
        self.assertIsNone(safety.fresh_bar(frame, '2026-09-09'))
        self.assertIsNone(safety.fresh_bar(bars(['2026-09-09']), 'not-a-date'))
        self.assertIsNone(safety.fresh_bar(bars(['2026-09-09']), pd.NaT))


class MechanicalExitTests(unittest.TestCase):
    def exit(self, opening=100, high=109, low=96, close=100, **position_changes):
        return safety.mechanical_exit(position(**position_changes),
                                      {'Open': opening, 'High': high, 'Low': low, 'Close': close})

    def test_stop_target_dual_hit_is_stop_first(self):
        self.assertEqual(self.exit(high=115, low=90), (95, 'stop_loss'))

    def test_gap_stop_fills_open_not_stop_even_if_target_later_touched(self):
        self.assertEqual(self.exit(opening=90, low=85, high=120), (90, 'stop_loss'))

    def test_gap_target_fills_open_before_later_intrabar_stop(self):
        self.assertEqual(self.exit(opening=115, high=120, low=90), (115, 'profit_target'))

    def test_intrabar_stop_and_target_fill_levels(self):
        self.assertEqual(self.exit(low=95), (95, 'stop_loss'))
        self.assertEqual(self.exit(high=110), (110, 'profit_target'))

    def test_exact_open_stop_or_target_is_gap_fill(self):
        self.assertEqual(self.exit(opening=95, low=95), (95, 'stop_loss'))
        self.assertEqual(self.exit(opening=110, high=110), (110, 'profit_target'))

    def test_no_hit_missing_quotes_invalid_levels_return_none(self):
        self.assertIsNone(self.exit())
        for value in (None, 0, -1, math.nan, math.inf, True, '95'):
            for key in ('stop_price', 'target_price'):
                with self.subTest(key=key, value=value):
                    self.assertIsNone(self.exit(**{key: value}))
        self.assertIsNone(self.exit(stop_price=110, target_price=100))
        self.assertIsNone(safety.mechanical_exit(position(), None))
        self.assertIsNone(safety.mechanical_exit({}, {}))

    def test_invalid_bar_does_not_trigger_exit(self):
        self.assertIsNone(self.exit(high=80, low=70))
        self.assertIsNone(self.exit(low=math.nan))

    def test_no_mutation_or_same_bar_trailing_stop_lookahead(self):
        pos = position(high_watermark=100, atr_at_entry=1, target_price=140)
        bar = {'Open': 100, 'High': 130, 'Low': 96, 'Close': 100}
        before = copy.deepcopy((pos, bar))
        self.assertIsNone(safety.mechanical_exit(pos, bar))
        self.assertEqual((pos, bar), before)


class InterfaceTests(unittest.TestCase):
    def test_only_temporary_module_is_loaded_and_sources_parse_as_python_311(self):
        self.assertEqual(Path(safety.__file__).resolve(), ROOT / 'screener_safety.py')
        for source in (ROOT / 'screener_safety.py', Path(__file__)):
            ast.parse(source.read_text(encoding='utf-8'), filename=str(source),
                      feature_version=(3, 11))

    def test_exact_public_function_signatures(self):
        expected = {
            'finite_number': "(value, name='value', minimum=None, maximum=None)",
            'atomic_json': '(path, data)', 'atomic_csv': '(path, dataframe)',
            'validate_portfolio': '(pf)',
            'plan_order': '(pf, ticker, entry_price, amount_usd, stop, target, sector, fee_quote, max_positions=5, cash_floor=500.0)',
            'evaluate_horizon': '(stock, benchmark, signal_date, hold_sessions, as_of)',
            'fresh_bar': '(frame, as_of)', 'mechanical_exit': '(position, bar)',
        }
        for name, signature in expected.items():
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(getattr(safety, name))), signature)


if __name__ == '__main__':
    unittest.main()