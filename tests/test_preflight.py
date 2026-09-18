"""Offline tests for the pre-run readiness check.

Preflight exists to answer one question before real money is involved: will the
next run work, and what will it do to the ledger? So it must be honest about
blockers, must never write anything, and must never send a message.
"""

import hashlib
import io
import os
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('SCREENER_SKIP_UNIVERSE_FETCH', '1')
os.environ.setdefault('SCREENER_DISABLE_ALERTS', '1')
os.environ.setdefault('SCREENER_OUTPUT_DIR', tempfile.mkdtemp(prefix='preflight-import-'))

import qa_validate  # noqa: E402

with redirect_stdout(io.StringIO()):
    import LLM_Portfolio_Manager as app  # noqa: E402

LEDGER = ROOT / 'StockScreener' / 'portfolio.json'

FILLED_ORDER = {'order_id': 'o-1', 'client_order_id': 'lpm-B-MTD-1758134400-a1b2c3d4',
                'symbol': 'MTD', 'side': 'buy', 'status': 'filled', 'qty': 18,
                'filled_qty': 18, 'filled_avg_price': 1403.2822,
                'filled_at': '2026-09-17T13:30:05Z', 'submitted_at': None,
                'updated_at': None, 'canceled_at': None, 'expired_at': None}
ACCOUNT = {'cash': '74740.92', 'equity': '100128.84',
           'trading_blocked': False, 'account_blocked': False}


def ledger(**kwargs):
    base = {'cash': 74740.92, 'starting_capital': 100000.0, 'positions': [],
            'closed_trades': [], 'pending_orders': [], 'processed_sessions': [],
            'total_realized_pnl': 0.0, 'equity_peak': 100000.0}
    base.update(kwargs)
    return base


PENDING_MTD = {'id': 'e904bd8b-ef79-4a81-bfd2-a61eb4a563d9',
               'trade_id': 'e904bd8b-ef79-4a81-bfd2-a61eb4a563d9',
               'ticker': 'MTD', 'signal_date': '2026-09-16',
               'execution_session': '2026-09-17', 'estimated_entry': 1382.83,
               'stop_distance': 54.93, 'target_distance': 109.86,
               'amount_usd': 25000.0, 'shares': 18, 'sector': 'Healthcare',
               'atr': 36.62, 'hold_sessions': 10, 'status': 'PENDING'}


def snapshot(**kwargs):
    base = {'ok': True, 'cash': 74740.92, 'equity': 100128.84, 'account_blocked': False,
            'positions': {}, 'orders': []}
    base.update(kwargs)
    return base


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.temp = tempfile.TemporaryDirectory(prefix='preflight-')
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': self.temp.name, 'ALPACA_PAPER': '1',
            'DISCORD_CHANNEL_ID': '123'}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        # Module constants are read at import time, so patch the module itself.
        for name, value in (('NVIDIA_API_KEY', 'nv'), ('OPENROUTER_API_KEY', 'or'),
                            ('WHATSAPP_PHONE', ''), ('CALLMEBOT_API_KEY', '')):
            p = patch.object(app, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(app, '_fetch_served_models', return_value=['meta/llama-3.3-70b-instruct'])
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(app._discord, 'enabled', return_value=True)
        p.start()
        self.addCleanup(p.stop)
        # Read-only probe against Discord; stubbed so the suite stays offline.
        p = patch.object(app._discord, 'check_access',
                         return_value=(True, 'bot can post to #alpaca-bot-1'))
        p.start()
        self.addCleanup(p.stop)

    def run_preflight(self, *, keys=True, live=True, account=ACCOUNT, snap=None,
                      book=None):
        snap = snapshot() if snap is None else snap
        book = ledger() if book is None else book
        with patch.object(qa_validate, 'check_state', return_value=book), \
                patch.object(app._alpaca, 'data_enabled', return_value=keys), \
                patch.object(app._alpaca, 'trading_enabled', return_value=keys and live), \
                patch.object(app._alpaca, 'fetch_account', return_value=account), \
                patch.object(app._alpaca, 'broker_snapshot', return_value=snap), \
                redirect_stdout(io.StringIO()) as out:
            code = qa_validate.preflight()
        return code, out.getvalue()

    # -- Blockers ---------------------------------------------------------

    def test_missing_alpaca_keys_blocks_and_exits_nonzero(self):
        code, text = self.run_preflight(keys=False)
        self.assertEqual(code, 1)
        self.assertIn('ALPACA_API_KEY', text)
        self.assertIn('NOT READY', text)

    def test_live_broker_flag_off_is_a_blocker_with_the_real_consequence(self):
        # This is the exact misconfiguration that would sell a position the
        # ledger has lost track of, so it must be stated, not merely warned.
        code, text = self.run_preflight(live=False)
        self.assertEqual(code, 1)
        self.assertIn('SCREENER_LIVE_BROKER', text)
        self.assertIn('may sell holdings it has lost track of', text)

    def test_unreadable_account_blocks(self):
        code, text = self.run_preflight(account=None)
        self.assertEqual(code, 1)
        self.assertIn('could not be read', text)

    def test_blocked_alpaca_account_is_reported(self):
        blocked = dict(ACCOUNT, account_blocked=True)
        _, text = self.run_preflight(account=blocked)
        self.assertIn('BLOCKED', text)

    def test_unreadable_broker_snapshot_blocks(self):
        code, text = self.run_preflight(snap={'ok': False, 'error': 'unreadable: positions'})
        self.assertEqual(code, 1)
        self.assertIn('unreadable', text)

    def test_missing_both_llm_keys_blocks(self):
        with patch.object(app, 'NVIDIA_API_KEY', ''), \
                patch.object(app, 'OPENROUTER_API_KEY', ''):
            code, text = self.run_preflight()
        self.assertEqual(code, 1)
        self.assertIn('no pick can be made', text)

    # -- Warnings that are not blockers ------------------------------------

    def test_live_money_account_is_flagged_loudly_but_is_not_a_blocker(self):
        with patch.dict(os.environ, {'ALPACA_PAPER': '0'}):
            code, text = self.run_preflight()
        self.assertEqual(code, 0)
        self.assertIn('LIVE MONEY', text)

    def test_missing_backup_provider_warns_with_the_consequence(self):
        with patch.object(app, 'OPENROUTER_API_KEY', ''):
            code, text = self.run_preflight()
        self.assertEqual(code, 0)
        self.assertIn('no trade that day', text)

    def test_discord_configured_but_bot_not_invited_is_a_blocker(self):
        # A valid token with no channel access only shows up as a failed
        # send otherwise - by which point the alert is already lost.
        with patch.object(app._discord, 'check_access',
                          return_value=(False, 'bot cannot see this channel (403)')):
            code, text = self.run_preflight()
        self.assertEqual(code, 1)
        self.assertIn('403', text)

    def test_unconfigured_discord_warns_that_you_will_not_be_told(self):
        with patch.object(app._discord, 'enabled', return_value=False), \
                patch.dict(os.environ, {'SCREENER_DISABLE_ALERTS': '0'}):
            code, text = self.run_preflight()
        self.assertEqual(code, 0)
        self.assertIn('tell you nothing', text)
        self.assertIn('not a trading blocker', text)

    # -- The headline feature: what will the next run actually do? ---------

    def test_in_sync_ledger_reports_that_nothing_will_change(self):
        # Ledger holds a pending order that Alpaca is still working: nothing
        # for the next run to change yet.
        working = snapshot(orders=[dict(FILLED_ORDER, status='accepted',
                                        filled_qty=0, filled_avg_price=None)])
        code, text = self.run_preflight(snap=working,
                                        book=ledger(pending_orders=[PENDING_MTD]))
        self.assertEqual(code, 0)
        self.assertIn('in sync', text)
        self.assertIn('READY', text)

    def test_pending_fill_is_previewed_before_it_is_applied(self):
        snap = snapshot(orders=[FILLED_ORDER], positions={
            'MTD': {'symbol': 'MTD', 'qty': 18, 'avg_entry_price': 1403.2822,
                    'market_value': 25387.92, 'current_price': 1410.44}})
        code, text = self.run_preflight(snap=snap,
                                        book=ledger(pending_orders=[PENDING_MTD]))
        self.assertEqual(code, 0, 'a pending fill is expected news, not a blocker')
        self.assertIn('next run will', text)
        self.assertIn('MTD', text)
        self.assertIn('1,403.28', text)

    def test_rejected_order_surfaces_as_a_blocking_conflict(self):
        snap = snapshot(orders=[dict(FILLED_ORDER, status='rejected', filled_qty=0,
                                     filled_avg_price=None)])
        code, text = self.run_preflight(snap=snap,
                                        book=ledger(pending_orders=[PENDING_MTD]))
        self.assertEqual(code, 1)
        self.assertIn('NOT READY', text)

    # -- Safety ------------------------------------------------------------

    def test_preflight_never_modifies_the_canonical_ledger(self):
        # Deliberately reads the real file: this is the guarantee that matters.
        before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
        with patch.object(app._alpaca, 'data_enabled', return_value=True), \
                patch.object(app._alpaca, 'trading_enabled', return_value=True), \
                patch.object(app._alpaca, 'fetch_account', return_value=ACCOUNT), \
                patch.object(app._alpaca, 'broker_snapshot',
                             return_value=snapshot(orders=[FILLED_ORDER])), \
                redirect_stdout(io.StringIO()):
            qa_validate.preflight()
        self.assertEqual(hashlib.sha256(LEDGER.read_bytes()).hexdigest(), before)

    def test_a_position_held_only_at_alpaca_is_reported_for_adoption(self):
        # The live situation on 2026-09-17: Alpaca filled MTD, the ledger's
        # pending order was expired by the drawdown guard, and the ledger ended
        # up holding nothing while the broker held 18 shares.
        orphan = snapshot(positions={'MTD': {'symbol': 'MTD', 'qty': 18,
                                             'avg_entry_price': 1403.28,
                                             'current_price': 1410.44}})
        code, text = self.run_preflight(snap=orphan, book=ledger())
        self.assertIn('adopt', text)
        self.assertIn('MTD', text)
        self.assertEqual(code, 0)

    def test_preflight_never_sends_a_message_or_places_an_order(self):
        with patch.object(app._discord, 'send', side_effect=AssertionError('must not send')), \
                patch.object(app._discord, 'send_embed', side_effect=AssertionError('must not send')), \
                patch.object(app._alpaca, 'submit_market_order',
                             side_effect=AssertionError('must not order')):
            code, _ = self.run_preflight()
        self.assertEqual(code, 0)

    def test_preflight_refuses_to_use_the_canonical_output_directory(self):
        with patch.dict(os.environ, {'SCREENER_OUTPUT_DIR': str(ROOT / 'StockScreener')}):
            with self.assertRaises(ValueError):
                qa_validate.preflight()


if __name__ == '__main__':
    unittest.main()
