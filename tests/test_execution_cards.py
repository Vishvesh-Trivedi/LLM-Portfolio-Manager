"""Offline tests for broker-style execution notifications.

These pin the rendering contract: the headline alone must carry action,
quantity, symbol and price, every alerting event kind must render without
raising, and a card must never invent a number the broker did not report.
"""

import io
import os
import socket
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def import_isolated():
    """Import main with alerts enabled but no real credentials or network."""
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = Mock()
    with tempfile.TemporaryDirectory(prefix='cards-') as output, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': output, 'SCREENER_SKIP_UNIVERSE_FETCH': '1',
            'DISCORD_BOT_TOKEN': 'token', 'DISCORD_CHANNEL_ID': '1',
            'NVIDIA_API_KEY': '', 'OPENROUTER_API_KEY': '',
            'WHATSAPP_PHONE': '', 'CALLMEBOT_API_KEY': '',
        }, clear=False))
        os.environ.pop('SCREENER_DISABLE_ALERTS', None)
        stack.enter_context(patch.dict('sys.modules', {'dotenv': dotenv}))
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        with redirect_stdout(io.StringIO()):
            import LLM_Portfolio_Manager as app
        return app


app = import_isolated()


FILL = {'kind': 'fill', 'severity': 'info', 'symbol': 'MTD',
        'summary': 'MTD: FILLED 18 @ $1,403.28', 'side': 'buy', 'shares': 18,
        'requested_shares': 18, 'unfilled': 0, 'price': 1403.2822,
        'notional': 25259.08, 'stop': 1348.35, 'target': 1513.14,
        'expected_price': 1382.83, 'slippage_pct': 1.4789, 'status': 'filled',
        'broker_order_id': 'abc123', 'exact_match': False}


class ExecutionCardTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def card(self, **overrides):
        event = dict(FILL, **overrides)
        return app._execution_card(event)

    def test_buy_headline_carries_action_qty_symbol_and_price(self):
        title, body, color, stats = self.card()
        self.assertIn('BOUGHT', title)
        self.assertIn('18 MTD', title)
        self.assertIn('$1,403.28', title)
        self.assertEqual(color, app._GREEN)
        self.assertEqual([name for name, _, _ in stats],
                         ['Total cost', 'Sell if it falls to', 'Sell if it rises to'])
        self.assertTrue(all(inline for _, _, inline in stats), 'stat row must be inline')

    def test_no_jargon_appears_anywhere_in_a_card(self):
        # The whole point of the rewrite: a reader should need no glossary.
        banned = ('slippage', 'notional', 'cost basis', 'expected price',
                  'proceeds', 'ledger', 'reconcil', 'broker status', 'r:r')
        for kind in sorted(app._ALERTING_EVENTS):
            title, body, _, stats = self.card(kind=kind)
            text = ' '.join([title, body] + [f'{n} {v}' for n, v, _ in stats]).lower()
            for word in banned:
                self.assertNotIn(word, text, f'{kind} still says "{word}"')

    def test_overpaying_is_explained_in_plain_words_with_a_dollar_total(self):
        _, body, _, _ = self.card()
        self.assertIn('$368 more than planned', body)
        self.assertNotIn('%', body.split('planned')[0])

    def test_underpaying_says_less_not_more(self):
        _, body, _, _ = self.card(price=1370.0, slippage_pct=-0.93)
        self.assertIn('less than planned', body)

    def test_negligible_difference_is_not_mentioned_at_all(self):
        _, body, _, _ = self.card(slippage_pct=0.02)
        self.assertNotIn('planned', body)
        self.assertTrue(body.strip())

    def test_partial_fill_explains_what_did_not_happen(self):
        title, body, color, _ = self.card(kind='partial_fill', shares=5,
                                          unfilled=13, notional=7016.41)
        self.assertIn('5 of 18 MTD', title)
        self.assertIn('the other 13 were not', body)
        self.assertEqual(color, app._AMBER)

    def test_profit_and_loss_are_stated_as_made_or_lost(self):
        base = {'kind': 'exit_filled', 'shares': 18, 'price': 1512.0,
                'notional': 27216.0, 'cost_basis': 25259.08, 'held_sessions': 8,
                'exit_reason': 'profit_target'}
        title, body, color, _ = self.card(**base, pnl=1956.92, pnl_pct=7.75)
        self.assertIn('SOLD', title)
        self.assertIn('You made $1,957', body)
        self.assertIn('Held for 8 trading days', body)
        self.assertIn('it hit the target price', body)
        self.assertEqual(color, app._GREEN)
        _, body, color, _ = self.card(**base, pnl=-500.0, pnl_pct=-1.98)
        self.assertIn('You lost $500', body)
        self.assertEqual(color, app._RED)

    def test_single_day_hold_is_not_pluralized(self):
        _, body, _, _ = self.card(kind='exit_filled', pnl=10.0, pnl_pct=1.0,
                                  held_sessions=1, exit_reason='')
        self.assertIn('Held for 1 trading day.', body)

    def test_waiting_card_says_nothing_bought_and_no_money_spent(self):
        title, body, color, _ = self.card(kind='working', status='accepted')
        self.assertIn('WAITING TO BUY', title)
        self.assertIn('has not gone through yet', body)
        self.assertIn('no money has been spent', body)
        self.assertEqual(color, app._BLUE)

    def test_rejected_card_says_nothing_was_bought_and_no_money_spent(self):
        title, body, color, _ = self.card(kind='rejected', severity='error',
                                          status='rejected', requested_shares=10)
        self.assertIn('REJECTED', title)
        self.assertIn('Alpaca refused it', body)
        self.assertIn('no money was spent', body)
        self.assertEqual(color, app._RED)

    def test_adopted_card_states_whether_sell_prices_were_set(self):
        protected = self.card(kind='adopted', symbol='XYZ', broker_shares=5,
                              broker_price=50.0, stop=47.0, target=56.0)
        self.assertIn('FOUND A POSITION YOU ALREADY OWNED', protected[0])
        self.assertIn('automatically', protected[1])
        self.assertEqual([n for n, _, _ in protected[3]],
                         ['Sell if it falls to', 'Sell if it rises to'])

    def test_unprotected_adoption_is_flagged_not_quietly_accepted(self):
        bare = self.card(kind='adopted', symbol='XYZ', broker_shares=5,
                         broker_price=50.0, stop=None, target=None)
        self.assertIn('UNPROTECTED', bare[0])
        self.assertIn('will not be sold', bare[1])
        self.assertEqual(bare[2], app._AMBER)

    def test_every_alerting_kind_renders_without_raising(self):
        for kind in sorted(app._ALERTING_EVENTS):
            title, body, color, stats = self.card(kind=kind)
            self.assertTrue(title, kind)
            self.assertTrue(body, kind)
            self.assertIsInstance(color, int)
            self.assertIsInstance(stats, list)

    def test_missing_numbers_render_as_na_not_zero(self):
        _, _, _, stats = self.card(notional=None, stop=None, target=None)
        self.assertEqual([v for _, v, _ in stats], ['n/a', 'n/a', 'n/a'])

    def test_unknown_kind_falls_back_to_its_summary(self):
        title, body, _, _ = self.card(kind='some_future_kind')
        self.assertIn('SOME FUTURE KIND', title)
        self.assertEqual(body, FILL['summary'])


class ExecutionAlertDispatchTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def dispatch(self, events):
        sent = []
        with patch.object(app._discord, 'send_embed',
                          side_effect=lambda **kw: sent.append(kw) or True), \
                patch.object(app._discord, 'send', return_value=True), \
                patch.object(app._discord, 'enabled', return_value=True), \
                redirect_stdout(io.StringIO()):
            app.send_execution_alerts(events)
        return sent

    def test_embed_carries_author_timestamp_and_order_reference(self):
        sent = self.dispatch([FILL])
        self.assertEqual(len(sent), 1)
        self.assertIn('Portfolio Manager', sent[0]['author'])
        self.assertIn('order abc123', sent[0]['footer'])
        self.assertTrue(sent[0]['timestamp'], 'native Discord timestamp expected')

    def test_errors_are_posted_before_informational_fills(self):
        reject = dict(FILL, kind='rejected', severity='error', symbol='AAA')
        sent = self.dispatch([FILL, reject])
        self.assertIn('REJECTED', sent[0]['title'])

    def test_unfilled_order_is_pushed_because_not_executed_still_counts(self):
        sent = self.dispatch([dict(FILL, kind='working', severity='info',
                                   status='accepted', requested_shares=18)])
        self.assertEqual(len(sent), 1)
        self.assertIn('WAITING TO BUY', sent[0]['title'])

    def test_only_balance_bookkeeping_stays_out_of_the_push_stream(self):
        self.assertEqual(self.dispatch([
            {'kind': 'cash_drift', 'severity': 'info', 'symbol': '', 'summary': 's'},
            {'kind': 'cash_unreadable', 'severity': 'warning', 'symbol': '', 'summary': 's'},
        ]), [])

    def test_every_order_outcome_reaches_the_channel(self):
        # Executed or not, each of these must produce a notification.
        outcomes = ('fill', 'partial_fill', 'working', 'rejected', 'canceled',
                    'order_missing', 'fill_unpriced', 'exit_filled')
        for kind in outcomes:
            self.assertEqual(len(self.dispatch([dict(FILL, kind=kind)])), 1, kind)

    def test_overflow_is_summarized_rather_than_flooding(self):
        events = [dict(FILL, symbol=f'S{i}') for i in range(app._MAX_EVENT_EMBEDS + 3)]
        self.assertEqual(len(self.dispatch(events)), app._MAX_EVENT_EMBEDS)

    def test_dispatch_never_raises_on_a_malformed_event(self):
        with redirect_stdout(io.StringIO()):
            self.assertFalse(app.send_execution_alerts([{'kind': 'fill'}]))

    def test_nothing_is_sent_when_discord_is_disabled(self):
        with patch.object(app._discord, 'enabled', return_value=False), \
                patch.object(app._discord, 'send_embed',
                             side_effect=AssertionError('must not send')), \
                redirect_stdout(io.StringIO()):
            self.assertFalse(app.send_execution_alerts([FILL]))


if __name__ == '__main__':
    unittest.main()
