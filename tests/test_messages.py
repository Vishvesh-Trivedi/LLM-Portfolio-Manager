"""Offline tests for the words that actually reach Discord and WhatsApp.

Nothing tested this before. 503 tests passed while the weekend summary was
never sent at all - the suite mocked send_weekly_summary and then asserted it
had not been called, so the defect was written into the tests as the expected
contract. These read the rendered text instead, which is the only thing that
can tell the difference.

The contract being pinned:
  * a closed market reports on BOTH channels, and says why there is no trade;
  * every message states whether it describes a real Alpaca account or a
    simulation, and never contradicts itself about that;
  * a message about a portfolio says what is held, at what price, and where it
    will be sold;
  * nothing is re-priced on a day the market never opened.
"""

import io
import os
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import sys
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def import_isolated():
    """Import main with alerts enabled but no real credentials or network.

    patch.dict('sys.modules') restores the whole dictionary when it exits,
    which unregisters every module imported inside it. The module objects
    survive through the names bound here, but anything that later runs
    `import LLM_Portfolio_Manager` would re-execute it from source - and a
    second numpy import in one process raises "cannot load module more than
    once". So whatever the import added is put back afterwards.
    """
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = Mock()
    before = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix='messages-') as output, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': output, 'SCREENER_SKIP_UNIVERSE_FETCH': '1',
            'DISCORD_BOT_TOKEN': 'token', 'DISCORD_CHANNEL_ID': '1',
            'NVIDIA_API_KEY': '', 'OPENROUTER_API_KEY': '',
            'WHATSAPP_PHONE': 'phone', 'CALLMEBOT_API_KEY': 'key',
        }, clear=False))
        os.environ.pop('SCREENER_DISABLE_ALERTS', None)
        stack.enter_context(patch.dict('sys.modules', {'dotenv': dotenv}))
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        with redirect_stdout(io.StringIO()):
            import LLM_Portfolio_Manager as app
        added = {name: module for name, module in sys.modules.items()
                 if name not in before and name != 'dotenv'}
    sys.modules.update(added)
    return app


app = import_isolated()


def ledger(**overrides):
    book = {
        'cash': 74740.92, 'starting_capital': 100000.0, 'created': '2026-09-08',
        'positions': [{
            'ticker': 'MTD', 'shares': 18, 'entry_price': 1403.28,
            'current_price': 1392.32, 'cost_basis': 25259.08,
            'current_value': 25061.76, 'unrealized_pnl': -197.32,
            'unrealized_pnl_pct': -0.78, 'held_sessions': 3, 'hold_sessions': 10,
            'stop_price': 1353.69, 'target_price': 1502.46, 'sector': 'Technology',
        }],
        'pending_orders': [{'ticker': 'GILD', 'shares': 98, 'estimated_entry': 150.44}],
        'closed_trades': [{
            'ticker': 'FSLR', 'realized_pnl': -109.6, 'realized_pnl_pct': -5.27,
            'reason': 'stop_loss', 'exit_date': '2099-01-01',
        }],
    }
    book.update(overrides)
    return book


class ClosedMarketSummary(unittest.TestCase):
    """The message sent on a weekend or a NYSE holiday."""

    def render(self, reason='weekend', live=True, **overrides):
        sent = []
        # Both channels are enabled here rather than through os.environ.
        # WhatsApp is configured into module globals at import, and Discord
        # re-reads the environment on every call, so which of them is live
        # would otherwise depend on who imported the module first and what the
        # environment held at that moment.
        with patch.object(app, 'WHATSAPP_PHONE', 'phone'), \
                patch.object(app, 'CALLMEBOT_API_KEY', 'key'), \
                patch.object(app._alpaca, 'trading_enabled', return_value=live), \
                patch.object(app._discord, 'enabled', return_value=True), \
                patch.object(app._discord, 'send',
                             lambda text, label='': sent.append(('discord', text)) or True), \
                patch.object(app, '_wa_send',
                             lambda text, label='': sent.append(('whatsapp', text)) or True), \
                patch.object(app.yf, 'Ticker', side_effect=AssertionError('no network')), \
                redirect_stdout(io.StringIO()):
            app.send_weekly_summary(reason, portfolio=ledger(**overrides))
        return sent

    def test_a_shut_market_reports_on_both_channels(self):
        """The whole point. This is what was silently missing."""
        channels = [channel for channel, _ in self.render()]
        self.assertEqual(sorted(channels), ['discord', 'whatsapp'])

    def test_both_channels_receive_the_same_words(self):
        sent = dict(self.render())
        self.assertEqual(sent['discord'], sent['whatsapp'])

    def test_it_says_why_there_is_no_trade(self):
        text = self.render()[0][1]
        self.assertIn('WHY THERE IS NO TRADE TODAY', text)
        self.assertIn('shut', text)
        self.assertIn('Nothing was bought or sold', text)

    def test_a_holiday_is_named_rather_than_called_a_weekend(self):
        text = self.render(reason='Thanksgiving')[0][1]
        self.assertIn('Thanksgiving', text)
        self.assertNotIn('weekend', text.lower())

    def test_it_reports_money_holdings_and_the_exit_plan(self):
        text = self.render()[0][1]
        for expected in ('$99,803',            # cash plus the value of the holding
                         '$74,741',            # cash on its own
                         'MTD',
                         '18 shares',
                         '$1,403.28 -> $1,392.32',
                         'sells at $1,353.69 or $1,502.46',
                         'day 3 of 10'):
            self.assertIn(expected, text)

    def test_it_reports_what_is_waiting_to_be_bought(self):
        text = self.render()[0][1]
        self.assertIn('WAITING TO BUY (1)', text)
        self.assertIn('GILD: 98 shares', text)
        self.assertIn('$150.44', text)

    def test_an_empty_portfolio_does_not_claim_holdings_are_unchanged(self):
        text = self.render(positions=[], pending_orders=[])[0][1]
        self.assertIn('not holding anything', text)
        self.assertNotIn('holdings and their sell orders are unchanged', text)
        self.assertNotIn('WAITING TO BUY', text)

    def test_a_live_run_says_alpaca_and_never_calls_itself_simulated(self):
        text = self.render(live=True)[0][1]
        self.assertIn('Trading through your Alpaca account.', text)
        self.assertIn('Alpaca is the source of truth', text)
        self.assertNotIn('Simulated', text)

    def test_a_simulated_run_never_claims_the_numbers_came_from_alpaca(self):
        """The two halves of the message once contradicted each other."""
        text = self.render(live=False)[0][1]
        self.assertIn('Simulated only', text)
        self.assertIn('simulated numbers', text)
        self.assertNotIn('Alpaca is the source of truth', text)

    def test_it_never_reprices_on_a_day_the_market_did_not_open(self):
        """Re-pricing a closed market marks every quote stale and degrades the run."""
        with patch.object(app, 'update_portfolio_prices',
                          side_effect=AssertionError('must not re-price')), \
                patch.object(app, 'load_portfolio',
                             side_effect=AssertionError('must not reload')):
            self.render()

    def test_it_fits_in_one_message_on_both_services(self):
        sent = dict(self.render())
        self.assertLessEqual(len(sent['whatsapp']), 1600)  # CallMeBot limit
        self.assertLessEqual(len(sent['discord']), 2000)   # Discord limit

    def test_nothing_is_sent_when_no_channel_is_configured(self):
        with patch.object(app, '_alerts_configured', return_value=False), \
                patch.object(app._discord, 'send',
                             side_effect=AssertionError('must not send')), \
                redirect_stdout(io.StringIO()):
            app.send_weekly_summary('weekend', portfolio=ledger())

    def test_a_stale_closed_trade_is_not_reported_as_this_week(self):
        old = ledger()['closed_trades'][0] | {'exit_date': '2020-01-01'}
        text = self.render(closed_trades=[old])[0][1]
        self.assertIn('Nothing was sold', text)
        self.assertNotIn('FSLR', text)


class WhatsAppAllowanceIsSpentOnce(unittest.TestCase):
    """CallMeBot stops delivering when its allowance runs out.

    The schedule makes six attempts a day so a dropped cron cannot lose the
    session. Sending all six to WhatsApp exhausts the allowance in a day and
    the one message that matters never arrives.
    """

    def digest(self, mode='screening', events=()):
        sent = []
        with patch.object(app, 'WHATSAPP_PHONE', 'phone'),                 patch.object(app, 'CALLMEBOT_API_KEY', 'key'),                 patch.object(app, '_RUN_MODE', mode),                 patch.object(app, '_RUN_EVENTS', list(events)),                 patch.object(app._alpaca, 'trading_enabled', return_value=True),                 patch.object(app._discord, 'enabled', return_value=True),                 patch.object(app._discord, 'send',
                             lambda t, l='': sent.append('discord') or True),                 patch.object(app, '_wa_send',
                             lambda t, l='': sent.append('whatsapp') or True),                 redirect_stdout(io.StringIO()):
            app.send_run_digest(ledger())
        return sent

    def test_a_repeat_reconciliation_does_not_spend_a_message(self):
        self.assertNotIn('whatsapp', self.digest(mode='already_processed'))

    def test_discord_still_gets_every_run(self):
        self.assertIn('discord', self.digest(mode='already_processed'))

    def test_the_run_that_screened_does_spend_one(self):
        self.assertIn('whatsapp', self.digest(mode='screening'))

    def test_a_repeat_run_where_money_moved_still_spends_one(self):
        """A fill landing on a later attempt is worth a message."""
        sent = self.digest(mode='already_processed',
                           events=[{'kind': 'fill', 'symbol': 'GILD',
                                    'shares': 98, 'price': 150.44}])
        self.assertIn('whatsapp', sent)

    def test_a_missed_session_is_carried_in_the_digest_not_sent_separately(self):
        with patch.object(app, '_MISSED_SESSIONS', ['2026-09-21']),                 patch.object(app, 'WHATSAPP_PHONE', 'phone'),                 patch.object(app, 'CALLMEBOT_API_KEY', 'key'),                 patch.object(app, '_RUN_EVENTS', []),                 patch.object(app._alpaca, 'trading_enabled', return_value=True),                 patch.object(app._discord, 'enabled', return_value=True),                 patch.object(app._discord, 'send', lambda t, l='': True),                 patch.object(app, '_wa_send', lambda t, l='': True),                 redirect_stdout(io.StringIO()) as out:
            app.send_run_digest(ledger())
        self.assertIn('2026-09-21', out.getvalue())
        self.assertIn('without being screened', out.getvalue())


class SectorResolution(unittest.TestCase):
    """A bug in our own code must not look like a gap in the data."""

    def resolve(self, result):
        fetch = (Mock(side_effect=result) if isinstance(result, Exception)
                 else Mock(return_value=result))
        with patch.object(app, '_fetch_fundamentals_single', fetch), \
                redirect_stdout(io.StringIO()):
            return app._resolve_sector('MTD')

    def test_a_real_sector_is_returned(self):
        self.assertEqual(self.resolve(('MTD', {'sector': 'Technology'})), 'Technology')

    def test_a_failed_lookup_is_expected_and_answers_empty(self):
        self.assertEqual(self.resolve(OSError('offline')), '')

    def test_a_missing_sector_answers_empty(self):
        for payload in (('MTD', {}), ('MTD', {'sector': ''}),
                        ('MTD', {'sector': 'Unknown'}), ('MTD', None)):
            with self.subTest(payload=payload):
                self.assertEqual(self.resolve(payload), '')

    def test_a_sector_the_order_planner_cannot_map_answers_empty(self):
        """Accepting it would make plan_order raise on every later order."""
        self.assertEqual(self.resolve(('MTD', {'sector': 'Intergalactic Mining'})), '')

    def test_a_bug_in_our_parsing_raises_instead_of_answering_empty(self):
        """The regression that nearly sold 18 real shares.

        _fetch_fundamentals_single returns (ticker, data). When this called
        .get() straight on that tuple, the AttributeError was swallowed and
        returned '' - the same answer as 'yfinance has no sector for MTD'. The
        run degraded, degrading is routine, and the position went unadopted.
        """
        with self.assertRaises(AttributeError):
            self.resolve(('MTD', 'not-a-dict'))


class StrategyExplainer(unittest.TestCase):
    """The plain-English explainer people outside this repo will read.

    Its whole value is being true. A hand-written description drifts the moment
    a threshold changes, so it reads the live configuration instead - and these
    tests check it really did, rather than hard-coding the same numbers twice.
    """

    def setUp(self):
        import explain_strategy
        self.module = explain_strategy
        self.text = explain_strategy.build()

    def test_the_numbers_come_from_the_live_configuration(self):
        for value in (app.BUY_THRESHOLD, app.WATCH_THRESHOLD,
                      app._CFG_MAX_POSITIONS, app._CFG_HOLD_DAYS):
            self.assertIn(str(value), self.text)

    def test_changing_a_threshold_changes_the_explanation(self):
        """Proves it is read, not transcribed."""
        with patch.object(app, 'BUY_THRESHOLD', 91):
            self.assertIn('91', self.module.build())

    def test_it_says_the_money_is_not_real(self):
        lowered = self.text.lower()
        self.assertTrue(any(word in lowered for word in ('fake money', 'practice account')))

    def test_it_states_the_two_things_people_get_wrong(self):
        lowered = self.text.lower()
        # That it does not adapt on its own, and that it was never backtested.
        self.assertIn('nothing changes automatically', lowered)
        self.assertIn('20%', self.text)
        self.assertIn('past market data', lowered)

    def test_it_explains_both_halves_of_the_score(self):
        self.assertIn('60 points', self.text)
        self.assertIn('40 points', self.text)

    def test_it_avoids_the_jargon_it_is_meant_to_replace(self):
        """These are the words a non-trader cannot parse."""
        for jargon in ('ATR', 'RSI', 'MACD', 'R:R', 'drawdown', 'stop-loss',
                       'reward:risk', 'overbought'):
            self.assertNotIn(jargon.lower(), self.text.lower(), f'{jargon} leaked in')

    def test_printing_it_sends_nothing(self):
        with patch.object(self.module.discord, 'send',
                          side_effect=AssertionError('must not send')),                 patch.object(sys, 'argv', ['explain_strategy.py']),                 redirect_stdout(io.StringIO()):
            self.assertEqual(self.module.main(), 0)

    def test_send_reports_failure_rather_than_pretending(self):
        with patch.object(self.module.discord, 'enabled', return_value=True),                 patch.object(self.module.discord, 'send', return_value=False),                 patch.object(self.module.discord, 'last_error', return_value='401'),                 patch.object(sys, 'argv', ['explain_strategy.py', '--send']),                 redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.module.main(), 1)
        self.assertIn('401', out.getvalue())

    def test_it_survives_a_missing_ledger(self):
        with patch.object(self.module, '_record', return_value=None):
            self.assertIn('HOW MY TRADING BOT WORKS', self.module.build())


if __name__ == '__main__':
    unittest.main()
