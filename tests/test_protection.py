"""A position must never be held at Alpaca with nothing protecting it.

GILD filled at Monday's open and sat all day with no stop and no target. The
entry was a plain market order, and protection was only ever attached by
protect_positions, which runs once after the close. Nothing reported it,
because protect_positions walks the ledger's positions and the ledger still
recorded GILD as a pending order rather than a holding.

Two changes, covered here:
  * a buy is submitted as a bracket, so Alpaca attaches the stop and target to
    the fill itself and the naked window never opens;
  * protection is audited against what Alpaca actually holds, so a holding the
    ledger has not booked can no longer hide.
"""

import io
import pathlib
import shutil
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

import screener_alpaca as alpaca


class BuyIsNeverProtection(unittest.TestCase):
    """The reverse of the bug that once doubled a position.

    A bracket entry carries order_class 'bracket'. Counting it as protection
    hides it from open_order_shares_by_symbol, the incoming shares go
    uncounted, and the mirror buys the position a second time.
    """

    def test_a_bracket_buy_is_an_entry_not_protection(self):
        order = {'side': 'buy', 'order_class': 'bracket', 'symbol': 'GILD',
                 'legs': [{'stop_price': '145.00'}]}
        self.assertFalse(alpaca.is_protective(order))

    def test_a_protective_sell_is_still_protection(self):
        for order in ({'side': 'sell', 'order_class': 'oco'},
                      {'side': 'sell', 'order_class': 'bracket'},
                      {'side': 'sell', 'legs': [{'stop_price': '1.0'}]}):
            with self.subTest(order=order):
                self.assertTrue(alpaca.is_protective(order))

    def test_a_pending_bracket_buy_is_counted_as_incoming_shares(self):
        orders = [{'symbol': 'GILD', 'side': 'buy', 'qty': '98', 'filled_qty': '0',
                   'order_class': 'bracket', 'legs': [{'stop_price': '145.73'}]}]
        with patch.object(alpaca, 'list_orders', return_value=orders):
            self.assertEqual(alpaca.open_order_shares_by_symbol(), {'GILD': 98})

    def test_a_resting_protective_sell_is_not_counted_as_shares_leaving(self):
        orders = [{'symbol': 'MTD', 'side': 'sell', 'qty': '18', 'filled_qty': '0',
                   'order_class': 'oco'}]
        with patch.object(alpaca, 'list_orders', return_value=orders):
            self.assertEqual(alpaca.open_order_shares_by_symbol(), {})


class ProtectiveOrderIdsAreUniquePerSubmission(unittest.TestCase):
    """The bug that actually left MTD unprotected.

    Alpaca answers a repeated client_order_id with
    422 {"code":40010001,"message":"client_order_id must be unique"}. The
    protective id was derived only from the trade_id, so it was identical on
    every submission - meaning a resting stop could be placed once and never
    replaced. Since the stop is re-submitted every time the trailing stop moves
    up, every trailing move after the first was refused and the position was
    left with nothing.
    """

    def test_repeated_protective_submissions_get_distinct_ids(self):
        ids = {alpaca._client_order_id('MTD', 'sell', 'tid-1', protective=True,
                                       unique=True)
               for _ in range(25)}
        self.assertEqual(len(ids), 25)

    def test_a_unique_id_still_parses_and_is_still_protective(self):
        """It must keep four parts, or it stops matching back to the ledger."""
        one = alpaca._client_order_id('MTD', 'sell', 'tid-1', protective=True,
                                      unique=True)
        self.assertEqual(len(one.split('-')), 4)
        parsed = alpaca.parse_client_order_id(one)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed['protective'])
        self.assertEqual(parsed['symbol'], 'MTD')

    def test_an_ordinary_entry_id_stays_stable(self):
        """Buys are matched back to the ledger by this id; it must not drift."""
        first = alpaca._client_order_id('GILD', 'buy', 'order-9')
        self.assertEqual(first, alpaca._client_order_id('GILD', 'buy', 'order-9'))

    def test_the_real_submission_asks_for_a_unique_id(self):
        sent = {}
        with patch.object(alpaca, '_request',
                          side_effect=lambda m, u, body=None, **k:
                              sent.update(body or {}) or {'id': '1'}):
            alpaca.submit_protective_oco('MTD', 18, 1380.85, 1502.46, ref='tid-1')
        first = sent['client_order_id']
        with patch.object(alpaca, '_request',
                          side_effect=lambda m, u, body=None, **k:
                              sent.update(body or {}) or {'id': '2'}):
            alpaca.submit_protective_oco('MTD', 18, 1390.00, 1502.46, ref='tid-1')
        self.assertNotEqual(first, sent['client_order_id'])


class RiskLevelsForAnAdoptedHolding(unittest.TestCase):
    """A holding the screener never planned carries no stop or target.

    Without derived levels mechanical_exit can never fire on it and the
    position sits unguarded indefinitely - which is how MTD was first found.
    Deriving nothing is better than inventing a level, so every failure path
    has to answer (None, None, ...) rather than a guess.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    _PRESENT = object()   # a stand-in frame; created once, never mutated

    def resolve(self, entry, atr=None, frame=_PRESENT, indicators=None):
        indicators = {'atr': atr} if indicators is None else indicators
        with patch.object(self.app, 'batch_download',
                          return_value={} if frame is None else {'MTD': frame}),                 patch.object(self.app, 'compute_indicators', return_value=indicators),                 redirect_stdout(io.StringIO()):
            return self.app._resolve_risk_levels('MTD', entry)

    def test_levels_come_from_the_entry_and_the_stocks_own_atr(self):
        stop, target, atr = self.resolve(1403.28, atr=33.06)
        self.assertEqual(atr, 33.06)
        self.assertEqual(stop, round(1403.28 - 1.5 * 33.06, 2))
        self.assertEqual(target, round(1403.28 + 3.0 * 33.06, 2))
        self.assertLess(stop, 1403.28)
        self.assertGreater(target, 1403.28)

    def test_the_reward_is_twice_the_risk(self):
        stop, target, _ = self.resolve(100.0, atr=2.0)
        self.assertAlmostEqual((target - 100.0) / (100.0 - stop), 2.0, places=6)

    def test_no_price_history_invents_nothing(self):
        self.assertEqual(self.resolve(1403.28, frame=None), (None, None, 0.0))

    def test_no_usable_atr_invents_nothing(self):
        for bad in (0, None, -1):
            with self.subTest(atr=bad):
                self.assertEqual(self.resolve(1403.28, atr=bad), (None, None, 0.0))

    def test_a_nonsense_entry_price_invents_nothing(self):
        for bad in (0, -5, float('nan'), 'x', None):
            with self.subTest(entry=bad):
                self.assertEqual(self.resolve(bad, atr=3.0)[:2], (None, None))

    def test_a_stop_that_would_land_at_or_below_zero_is_refused(self):
        """A penny stock with a wide ATR must not get a negative stop."""
        stop, target, atr = self.resolve(2.0, atr=5.0)
        self.assertIsNone(stop)
        self.assertIsNone(target)
        self.assertEqual(atr, 5.0)

    def test_a_broken_indicator_call_is_reported_not_guessed(self):
        with patch.object(self.app, 'batch_download', return_value={'MTD': object()}),                 patch.object(self.app, 'compute_indicators',
                             side_effect=RuntimeError('boom')),                 redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.app._resolve_risk_levels('MTD', 100.0),
                             (None, None, 0.0))
        self.assertIn('cannot derive risk levels', out.getvalue())


class AlpacaSpellsClassSharesDifferently(unittest.TestCase):
    """From the live screening log:

        Alpaca GET HTTP 400: {"message":"invalid symbol: BRK-B"}
        Alpaca GET HTTP 400: {"message":"invalid symbol: MOG-A"}

    Yahoo and this app write BRK-B; Alpaca writes BRK.B. Symbols are sent a
    hundred at a time and a rejected request returns nothing, so one unknown
    ticker cost the bars for the ninety-nine beside it.
    """

    def bars(self, symbols, reject=(), per_request=100):
        asked = []

        def fake_get(url, params=None):
            names = params['symbols'].split(',')
            asked.append(names)
            if any(name in reject for name in names):
                return None
            return {'bars': {name: [{'t': '2026-09-21T00:00:00Z', 'o': 1, 'h': 2,
                                     'l': 1, 'c': 2, 'v': 9}] for name in names}}

        with patch.object(alpaca, '_get', side_effect=fake_get),                 patch.object(alpaca, 'data_enabled', return_value=True),                 patch.object(alpaca, '_SYMBOLS_PER_REQUEST', per_request),                 redirect_stdout(io.StringIO()):
            return alpaca.daily_bars(symbols, '2026-09-01'), asked

    def test_a_class_share_is_sent_in_alpacas_spelling(self):
        _, asked = self.bars(['BRK-B', 'AAPL'])
        self.assertEqual(asked[0], ['BRK.B', 'AAPL'])

    def test_the_answer_comes_back_in_this_apps_spelling(self):
        """Everything downstream keys off the Yahoo spelling."""
        frames, _ = self.bars(['BRK-B', 'AAPL'])
        self.assertEqual(sorted(frames), ['AAPL', 'BRK-B'])

    def test_an_order_is_placed_in_alpacas_spelling(self):
        sent = {}
        with patch.object(alpaca, '_request',
                          side_effect=lambda m, u, body=None, **k:
                              sent.update(body or {}) or {'id': '1'}),                 patch.object(alpaca, '_start_trade_updates_stream'),                 patch.object(alpaca, '_upsert_order_ledger'):
            alpaca.submit_market_order('BRK-B', 5, 'buy', ref='r1')
        self.assertEqual(sent['symbol'], 'BRK.B')

    def test_one_unknown_ticker_no_longer_costs_the_whole_batch(self):
        """A delisting or a typo should cost one symbol, not a hundred."""
        frames, _ = self.bars(
            ['AAPL', 'MSFT', 'BAD', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL'],
            reject=('BAD',), per_request=8)
        self.assertNotIn('BAD', frames)
        self.assertEqual(sorted(frames),
                         ['AAPL', 'AMD', 'CSCO', 'INTC', 'MSFT', 'NVDA', 'ORCL'])

    def test_a_single_bad_symbol_alone_is_simply_dropped(self):
        frames, asked = self.bars(['BAD'], reject=('BAD',))
        self.assertEqual(frames, {})
        self.assertEqual(len(asked), 1, 'must not retry a lone symbol forever')

    def test_a_wholly_rejecting_broker_terminates(self):
        frames, asked = self.bars(['A', 'B', 'C', 'D'], reject=('A', 'B', 'C', 'D'),
                                  per_request=4)
        self.assertEqual(frames, {})
        self.assertLess(len(asked), 20, 'splitting must not run away')


class EntryCarriesItsProtection(unittest.TestCase):

    def submit(self, side='buy', **kwargs):
        sent = {}

        def request(method, url, body=None, **_):
            sent.update(body or {})
            return {'id': 'order-1'}

        with patch.object(alpaca, '_request', side_effect=request), \
                patch.object(alpaca, '_start_trade_updates_stream'), \
                patch.object(alpaca, '_upsert_order_ledger'):
            alpaca.submit_market_order('GILD', 98, side, ref='r1', **kwargs)
        return sent

    def test_a_buy_with_levels_becomes_a_bracket(self):
        body = self.submit(stop_price=145.73, target_price=159.86)
        self.assertEqual(body['order_class'], 'bracket')
        self.assertEqual(body['stop_loss'], {'stop_price': '145.73'})
        self.assertEqual(body['take_profit'], {'limit_price': '159.86'})
        self.assertEqual(body['side'], 'buy')
        self.assertEqual(body['qty'], '98')

    def test_a_buy_without_levels_stays_a_plain_order(self):
        self.assertNotIn('order_class', self.submit())

    def test_an_unusable_level_sends_no_bracket_rather_than_a_rejected_one(self):
        """Better naked and reported than an order Alpaca refuses outright."""
        for stop, target in ((0, 10), (10, 10), (20, 10), (-1, 10),
                             (None, 10), (5, None), ('x', 10), (5, 'y')):
            with self.subTest(stop=stop, target=target):
                self.assertNotIn('order_class',
                                 self.submit(stop_price=stop, target_price=target))

    def test_a_refused_bracket_falls_back_rather_than_losing_the_trade(self):
        """Alpaca is fussier about brackets than about plain orders.

        Losing the buy entirely is the worse outcome: protect_positions
        attaches the same stop and target on this run, and the schedule
        re-checks protection several times a day.
        """
        bodies = []

        def request(method, url, body=None, **_):
            bodies.append(dict(body or {}))
            return None if len(bodies) == 1 else {'id': 'plain-1'}

        with patch.object(alpaca, '_request', side_effect=request),                 patch.object(alpaca, '_start_trade_updates_stream'),                 patch.object(alpaca, '_upsert_order_ledger'):
            order = alpaca.submit_market_order('GILD', 98, 'buy', ref='r1',
                                               stop_price=145.73,
                                               target_price=159.86)
        self.assertIsNotNone(order, 'the trade must still be placed')
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0].get('order_class'), 'bracket')
        self.assertNotIn('order_class', bodies[1])
        self.assertNotIn('stop_loss', bodies[1])
        # A distinct id: the refused submission may have registered the first.
        self.assertNotEqual(bodies[0]['client_order_id'],
                            bodies[1]['client_order_id'])

    def test_a_plain_order_that_is_refused_is_not_retried_forever(self):
        calls = []
        with patch.object(alpaca, '_request',
                          side_effect=lambda *a, **k: calls.append(1)),                 patch.object(alpaca, '_start_trade_updates_stream'),                 patch.object(alpaca, '_upsert_order_ledger'):
            self.assertIsNone(alpaca.submit_market_order('GILD', 98, 'buy', ref='r'))
        self.assertEqual(len(calls), 1)

    def test_a_sell_never_carries_a_bracket(self):
        body = self.submit(side='sell', stop_price=145.73, target_price=159.86)
        self.assertNotIn('order_class', body)


class EntryLevels(unittest.TestCase):
    """Where the bracket's prices come from before a fill price exists."""

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    def test_a_pending_order_supplies_levels_from_its_estimate(self):
        book = {'pending_orders': [{'ticker': 'GILD', 'estimated_entry': 150.44,
                                    'stop_distance': 4.71, 'target_distance': 9.42}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book),
                         {'GILD': (145.73, 159.86)})

    def test_an_open_position_supplies_its_recorded_levels(self):
        book = {'positions': [{'ticker': 'MTD', 'stop_price': 1353.69,
                               'target_price': 1502.46}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book),
                         {'MTD': (1353.69, 1502.46)})

    def test_a_pending_order_wins_over_a_stale_position_record(self):
        book = {'pending_orders': [{'ticker': 'X', 'estimated_entry': 100.0,
                                    'stop_distance': 5.0, 'target_distance': 10.0}],
                'positions': [{'ticker': 'X', 'stop_price': 1.0, 'target_price': 2.0}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book), {'X': (95.0, 110.0)})

    def test_unusable_records_are_skipped_rather_than_guessed(self):
        book = {'pending_orders': [
            {'ticker': 'A'},
            {'ticker': 'B', 'estimated_entry': 'x', 'stop_distance': 1, 'target_distance': 2},
            {'ticker': 'C', 'estimated_entry': 10, 'stop_distance': 20, 'target_distance': 2},
        ], 'positions': [{'ticker': 'D', 'stop_price': None, 'target_price': 5}]}
        self.assertEqual(self.app._entry_levels_by_symbol(book), {})


class BrokerHoldingsAreAudited(unittest.TestCase):
    """Protection is checked against Alpaca's holdings, not the ledger's."""

    def setUp(self):
        from tests.test_messages import app
        self.app = app

    def protect(self, ledger_positions, held, existing=None, accepts=True):
        """Run protect_positions against a broker that remembers what it took.

        protect_positions refuses to act unless trading is on and the ledger
        has been reconciled: a stop derived from a ledger known to be wrong
        could sell at the wrong level.

        The fake records what it accepts, because a static {} for
        protective_orders_by_symbol makes the confirmation read at the end see
        nothing - every holding then looks naked however well the run went, and
        the test measures the mock rather than the code.
        """
        events = []
        at_broker = dict(existing or {})

        def _submit(symbol, qty, stop, limit, ref=''):
            if accepts is False:
                return None
            if accepts == 'drops':
                # Alpaca answered 200 and then dropped the order, which is a
                # real behaviour and indistinguishable from success at the
                # point of submission.
                return {'id': 'p-' + symbol}
            at_broker[symbol] = {'order_id': 'p-' + symbol, 'qty': qty,
                                 'stop': stop, 'limit': limit}
            return {'id': 'p-' + symbol}

        def _cancel(order_id):
            for symbol, order in list(at_broker.items()):
                if order.get('order_id') == order_id:
                    del at_broker[symbol]
            return True

        broker = self.app._alpaca
        with ExitStack() as stack:
            for target, kwargs in (
                ('trading_enabled', {'return_value': True}),
                ('protective_orders_by_symbol', {'side_effect': lambda: dict(at_broker)}),
                ('positions_by_symbol', {'return_value': held}),
                ('submit_protective_oco', {'side_effect': _submit}),
                ('cancel_order', {'side_effect': _cancel}),
                ('await_order_released', {'return_value': True}),
            ):
                stack.enter_context(patch.object(broker, target, **kwargs))
            stack.enter_context(patch.object(self.app, '_BROKER_SYNC_OK', [True]))
            stack.enter_context(patch.object(self.app, '_degrade',
                                             side_effect=events.append))
            stack.enter_context(redirect_stdout(io.StringIO()))
            produced = self.app.protect_positions({'positions': ledger_positions})
        return events, produced

    def test_a_holding_the_ledger_has_not_booked_is_reported(self):
        """Exactly GILD's state: filled at the broker, still 'pending' here."""
        reasons, events = self.protect([], {'GILD': 98})
        self.assertIn('broker_holding_unprotected:GILD', reasons)
        self.assertTrue(any(e['kind'] == 'unprotected' and e['symbol'] == 'GILD'
                            for e in events))

    def test_the_run_is_not_called_healthy_while_something_is_exposed(self):
        with patch.object(self.app._HEALTH, 'stage') as stage:
            self.protect([], {'GILD': 98})
        success, detail = stage.call_args.args[1], stage.call_args.args[2]
        self.assertFalse(success)
        self.assertIn('GILD', detail)

    def test_protection_accepted_then_rejected_by_alpaca_is_caught(self):
        """A 200 from Alpaca is not proof the order lived.

        The run believed it had protected the position; only the confirmation
        read notices that it had not.
        """
        reasons, _ = self.protect(
            [{'ticker': 'MTD', 'stop_price': 1353.69, 'target_price': 1502.46}],
            {'MTD': 18}, accepts=False)
        self.assertTrue(any('MTD' in reason for reason in reasons), reasons)

    def test_an_order_accepted_then_dropped_by_alpaca_is_still_caught(self):
        """This is why the run re-reads instead of trusting its own submits.

        Alpaca answers 200 and the order never lives. Believing the submit
        leaves the position unprotected and the run reporting success.
        """
        reasons, events = self.protect(
            [{'ticker': 'MTD', 'stop_price': 1353.69, 'target_price': 1502.46}],
            {'MTD': 18}, accepts='drops')
        self.assertIn('broker_holding_unprotected:MTD', reasons)
        self.assertTrue(any(e['kind'] == 'unprotected' for e in events))

    def test_a_holding_that_is_already_protected_is_not_reported(self):
        reasons, _ = self.protect(
            [], {'MTD': 18}, existing={'MTD': {'order_id': 'o1', 'qty': 18,
                                               'stop': 1353.69, 'limit': 1502.46}})
        self.assertEqual(reasons, [])

    def test_a_position_protected_during_this_run_is_not_reported(self):
        reasons, _ = self.protect(
            [{'ticker': 'MTD', 'stop_price': 1353.69, 'target_price': 1502.46}],
            {'MTD': 18})
        self.assertEqual(reasons, [])

    def test_nothing_held_means_nothing_to_report(self):
        reasons, _ = self.protect([], {})
        self.assertEqual(reasons, [])


class TheEquityPeakFollowsTheBroker(unittest.TestCase):
    """The peak must never be raised from a half-updated ledger.

    load_portfolio writes Alpaca's cash into the ledger before reconciliation
    removes a position that was sold. For that moment cash already excludes the
    holding while the holding is still listed, so local arithmetic counts the
    same money twice. One such save recorded a peak of 127,636 on an account
    that has never exceeded 102,000, and the 20% drawdown guard then refused
    every order for four days while the account was up 2%.
    """

    def setUp(self):
        import screener_portfolio
        self.engine = screener_portfolio
        self.root = tempfile.mkdtemp(prefix='peak-')
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.app = SimpleNamespace(
            PORTFOLIO_JSON=pathlib.Path(self.root) / 'portfolio.json',
            STARTING_CAPITAL=100000.0, _session_date=lambda: '2026-09-24',
            _ORDER_REASON=[''], _degrade=lambda reason: None,
        )

    def ledger(self, **overrides):
        book = {'cash': 87133.76, 'starting_capital': 100000.0,
                'positions': [], 'closed_trades': [], 'pending_orders': [],
                'processed_sessions': [], 'total_realized_pnl': 0.0,
                'equity_peak': 100000.0, 'created': '2026-09-08',
                'last_updated': '2026-09-24'}
        book.update(overrides)
        return book

    def held(self, ticker, shares, price):
        return {'trade_id': 't-' + ticker, 'ticker': ticker, 'shares': shares,
                'entry_price': price, 'cost_basis': round(shares * price, 2),
                'current_price': price, 'current_value': round(shares * price, 2),
                'entry_date': '2026-09-18', 'sector': 'Healthcare',
                'stop_price': round(price * 0.95, 2),
                'target_price': round(price * 1.1, 2)}

    def test_the_brokers_equity_sets_the_peak_not_local_arithmetic(self):
        """The exact shape that broke it: broker cash plus a stale position."""
        book = self.ledger(broker_equity=101968.02,
                           positions=[self.held('MTD', 18, 1430.44),
                                      self.held('GILD', 98, 151.37)])
        saved = self.engine.save_portfolio(self.app, book)
        # Local arithmetic would say 87,134 + 25,748 + 14,834 = 127,716.
        self.assertAlmostEqual(saved['equity_peak'], 101968.02, places=2)

    def test_a_healthy_account_is_not_treated_as_a_drawdown(self):
        book = self.ledger(broker_equity=101968.02,
                           positions=[self.held('GILD', 98, 151.37)])
        saved = self.engine.save_portfolio(self.app, book)
        equity = saved['cash'] + sum(p['current_value'] for p in saved['positions'])
        self.assertGreater(equity, saved['equity_peak'] * 0.8,
                           'an account above its start must be able to trade')

    def test_a_real_high_still_raises_the_peak(self):
        """Guarding against double counting must not stop the peak rising."""
        book = self.ledger(broker_equity=118400.0,
                           positions=[self.held('GILD', 98, 151.37)])
        saved = self.engine.save_portfolio(self.app, book)
        self.assertAlmostEqual(saved['equity_peak'], 118400.0, places=2)

    def test_without_a_broker_figure_it_falls_back_to_the_ledger(self):
        book = self.ledger(positions=[self.held('GILD', 98, 151.37)])
        saved = self.engine.save_portfolio(self.app, book)
        self.assertAlmostEqual(
            saved['equity_peak'],
            max(100000.0, book['cash'] + 98 * 151.37), places=2)

    def test_a_nonsense_broker_figure_is_ignored(self):
        for bad in (0, -1, 'x', None):
            with self.subTest(value=bad):
                book = self.ledger(broker_equity=bad,
                                   positions=[self.held('GILD', 98, 151.37)])
                saved = self.engine.save_portfolio(self.app, book)
                self.assertGreaterEqual(saved['equity_peak'], 100000.0)


class ARejectedOrderSaysWhatRefusedIt(unittest.TestCase):
    """run_health is the first artefact anyone reads when a run looks wrong.

    Four days of refused orders carried order_status REJECTED with
    order_reason None. The reason reached Discord and the console, but the
    machine-readable record dropped it, so diagnosing it meant reproducing the
    order planner by hand.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        self.output = tempfile.mkdtemp(prefix='health-')
        self.addCleanup(shutil.rmtree, self.output, ignore_errors=True)
        patcher = patch.object(app, 'DRIVE_FOLDER', self.output)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, result, fallback=''):
        with patch.object(self.app, '_ORDER_REASON', [fallback]),                 redirect_stdout(io.StringIO()):
            return self.app.write_run_health(result)

    def test_the_refusal_reason_is_recorded(self):
        health = self.write({'order_status': 'REJECTED',
                             'order_reason': 'equity drawdown is at least 20%'})
        self.assertEqual(health['order_status'], 'REJECTED')
        self.assertEqual(health['order_reason'], 'equity drawdown is at least 20%')

    def test_it_falls_back_to_the_live_reason(self):
        """A run that died before building a result still knows why."""
        health = self.write({'order_status': 'REJECTED'},
                            fallback='maximum positions reached')
        self.assertEqual(health['order_reason'], 'maximum positions reached')

    def test_a_clean_run_records_no_reason(self):
        health = self.write({'order_status': 'QUEUED', 'order_reason': ''})
        self.assertIsNone(health.get('order_reason'))


class MissedSessionsAreNoticed(unittest.TestCase):
    """A trading day that was never screened must not vanish quietly.

    processed_sessions is only ever asked whether today is done. When GitHub
    dropped the whole schedule on 2026-09-18 that session was skipped in
    silence, and the next run simply moved on to the next day.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        self.friday = datetime(2026, 9, 25, 17, tzinfo=ZoneInfo('America/New_York'))

    def missed(self, processed, now=None):
        with patch.object(self.app, '_session_date', return_value='2026-09-25'):
            return self.app._missed_sessions({'processed_sessions': processed},
                                             now=now or self.friday)

    def test_a_weekday_that_was_never_screened_is_reported(self):
        self.assertEqual(self.missed(['2026-09-21', '2026-09-22']),
                         ['2026-09-23', '2026-09-24'])

    def test_an_unbroken_run_of_sessions_reports_nothing(self):
        self.assertEqual(
            self.missed(['2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24']), [])

    def test_a_weekend_is_not_a_gap(self):
        """Saturday and Sunday were never sessions, so they cannot be missed."""
        monday = datetime(2026, 9, 21, 17, tzinfo=ZoneInfo('America/New_York'))
        with patch.object(self.app, '_session_date', return_value='2026-09-21'):
            self.assertEqual(
                self.app._missed_sessions({'processed_sessions': ['2026-09-18']},
                                          now=monday), [])

    def test_a_screened_but_deliberately_open_session_is_not_a_gap(self):
        """The false alarm the live run produced.

        A blocked run leaves its session out of processed_sessions on purpose,
        so a later run can retry it. Reading that as "never screened" fired the
        missed-session alarm on a day that had in fact been screened twice.
        """
        with patch.object(self.app, '_session_date', return_value='2026-09-25'):
            gaps = self.app._missed_sessions(
                {'processed_sessions': ['2026-09-21', '2026-09-22'],
                 'screened_sessions': ['2026-09-23', '2026-09-24']},
                now=self.friday)
        self.assertEqual(gaps, [])

    def test_a_genuinely_skipped_day_is_still_reported(self):
        """Forgiving open sessions must not forgive absent ones."""
        with patch.object(self.app, '_session_date', return_value='2026-09-25'):
            gaps = self.app._missed_sessions(
                {'processed_sessions': ['2026-09-21', '2026-09-22'],
                 'screened_sessions': ['2026-09-23']},
                now=self.friday)
        self.assertEqual(gaps, ['2026-09-24'])

    def test_a_new_ledger_does_not_report_the_days_before_it_existed(self):
        self.assertEqual(self.missed([]), [])
        # Nothing at or before the first processed session is history, not a gap.
        self.assertEqual(self.missed(['2026-09-24']), [])

    def test_today_is_not_reported_as_missed_by_the_run_processing_it(self):
        self.assertNotIn('2026-09-25', self.missed(['2026-09-21']))

    def test_a_holiday_is_not_a_gap(self):
        """2026-09-07 is Labor Day; the Friday before it is a real session."""
        tuesday = datetime(2026, 9, 8, 17, tzinfo=ZoneInfo('America/New_York'))
        with patch.object(self.app, '_session_date', return_value='2026-09-08'):
            gaps = self.app._missed_sessions({'processed_sessions': ['2026-09-04']},
                                             now=tuesday)
        self.assertNotIn('2026-09-07', gaps)
        self.assertNotIn('2026-09-05', gaps)
        self.assertNotIn('2026-09-06', gaps)


class ReplacingProtectionNeverLeavesItNaked(unittest.TestCase):
    """Run 102: MTD's trailing stop moved up, and MTD ended up with nothing.

    Alpaca accepts a cancel immediately but settles it asynchronously. Until it
    settles the old order still reserves the shares, so the replacement is
    rejected for insufficient quantity - and the holding, which had protection
    a moment earlier, now has none.
    """

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        self.position = {'trade_id': 't1', 'ticker': 'MTD', 'shares': 18,
                         'stop_price': 1380.85, 'target_price': 1502.46}
        # Protection exists, but at the old stop - so it must be replaced.
        self.existing = {'MTD': {'order_id': 'old-1', 'qty': 18,
                                 'stop': 1353.69, 'limit': 1502.46}}

    def run_protect(self, submit, released=True, cancelled=True):
        calls = {'await': [], 'submit': 0}

        def _submit(*args, **kwargs):
            calls['submit'] += 1
            return submit(calls['submit'])

        broker = self.app._alpaca
        with ExitStack() as stack:
            for name, kw in (
                ('trading_enabled', {'return_value': True}),
                ('protective_orders_by_symbol', {'return_value': dict(self.existing)}),
                ('positions_by_symbol', {'return_value': {'MTD': 18}}),
                ('cancel_order', {'return_value': cancelled}),
            ):
                stack.enter_context(patch.object(broker, name, **kw))
            stack.enter_context(patch.object(
                broker, 'await_order_released',
                side_effect=lambda oid, *a, **k: calls['await'].append(oid) or released))
            stack.enter_context(patch.object(broker, 'submit_protective_oco',
                                             side_effect=_submit))
            stack.enter_context(patch.object(self.app, '_BROKER_SYNC_OK', [True]))
            stack.enter_context(patch.object(self.app, 'time'))  # no real sleeping
            degraded = []
            stack.enter_context(patch.object(self.app, '_degrade',
                                             side_effect=degraded.append))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.app.protect_positions({'positions': [self.position]})
        return calls, degraded

    def test_the_cancel_is_confirmed_before_the_replacement_is_sent(self):
        calls, degraded = self.run_protect(lambda n: {'id': 'new-1'})
        self.assertEqual(calls['await'], ['old-1'])
        self.assertEqual(degraded, [])

    def test_a_transient_refusal_is_retried_rather_than_left_naked(self):
        """One rejection used to mean the position stayed unprotected."""
        calls, degraded = self.run_protect(
            lambda n: None if n == 1 else {'id': 'new-1'})
        self.assertEqual(calls['submit'], 2)
        self.assertEqual(degraded, [])

    def test_a_persistent_refusal_is_still_reported(self):
        """Retrying must not hide a real refusal."""
        calls, degraded = self.run_protect(lambda n: None)
        self.assertEqual(calls['submit'], 3)
        self.assertIn('protection_rejected:MTD', degraded)


class OneMessagePerRun(unittest.TestCase):

    def setUp(self):
        from tests.test_messages import app
        self.app = app
        # write_run_health writes a file; the import-time temp dir is long gone.
        self.output = tempfile.mkdtemp(prefix='health-')
        self.addCleanup(shutil.rmtree, self.output, ignore_errors=True)
        patcher = patch.object(app, 'DRIVE_FOLDER', self.output)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_health_embed_is_not_sent_alongside_the_digest(self):
        """Run 102 posted the digest and then a health embed saying the same."""
        with patch.object(self.app, '_CAPTURE_EVENTS', [True]),                 patch.object(self.app, 'send_health_alert') as health,                 redirect_stdout(io.StringIO()):
            self.app.write_run_health(None)
        health.assert_not_called()

    def test_an_explicit_caller_outside_a_run_still_gets_it(self):
        with patch.object(self.app, '_CAPTURE_EVENTS', [False]),                 patch.object(self.app, 'send_health_alert') as health,                 redirect_stdout(io.StringIO()):
            self.app.write_run_health(None)
        health.assert_called_once()


if __name__ == '__main__':
    unittest.main()
