"""Offline tests for per-provider rate budgets and failover routing.

No network. The contract: NVIDIA and OpenRouter never consume each other's
free-tier allowance, a failing provider stops being retried into every
remaining call of the run, and the backup is used BEFORE burning a backoff
rather than after.
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
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = Mock()
    with tempfile.TemporaryDirectory(prefix='routing-') as output, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': output, 'SCREENER_SKIP_UNIVERSE_FETCH': '1',
            'SCREENER_DISABLE_ALERTS': '1', 'NVIDIA_API_KEY': '', 'OPENROUTER_API_KEY': '',
        }, clear=False))
        stack.enter_context(patch.dict('sys.modules', {'dotenv': dotenv}))
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        with redirect_stdout(io.StringIO()):
            import LLM_Portfolio_Manager as app
        return app


app = import_isolated()


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.clock = [1_000_000.0]
        self.slept = []
        self.time = patch.object(app.time, 'time', lambda: self.clock[0])
        self.sleep = patch.object(app.time, 'sleep',
                                  lambda s: (self.slept.append(s),
                                             self.clock.__setitem__(0, self.clock[0] + s)))
        self.time.start(); self.sleep.start()
        self.addCleanup(self.time.stop); self.addCleanup(self.sleep.stop)
        for budget in app._LLM_BUDGETS.values():
            budget.stamps.clear()
            budget.last_call = 0.0
            budget.strikes = 0
            budget.cooldown_until = 0.0

    def budget(self, name):
        return app._llm_budget(name)

    # ── Independent allowances ────────────────────────────────────────────

    def test_providers_do_not_consume_each_others_allowance(self):
        nvidia, router = self.budget('NVIDIA'), self.budget('OpenRouter')
        with patch.object(app, '_LLM_RATE_LIMIT_PER_MIN', 3), \
                patch.object(app, '_LLM_MIN_GAP', 0):
            for _ in range(3):
                app._llm_acquire_rate_slot('NVIDIA')
            # NVIDIA is now saturated; OpenRouter must still be free.
            self.assertIsNotNone(nvidia.wait_time())
            self.assertGreater(nvidia.wait_time(), 0)
            self.assertEqual(router.wait_time(), 0.0)
            self.assertTrue(app._llm_provider_ready('OpenRouter'))
            self.assertFalse(app._llm_provider_ready('NVIDIA'))

    def test_window_frees_up_after_sixty_seconds(self):
        with patch.object(app, '_LLM_RATE_LIMIT_PER_MIN', 2), \
                patch.object(app, '_LLM_MIN_GAP', 0):
            app._llm_acquire_rate_slot('NVIDIA')
            app._llm_acquire_rate_slot('NVIDIA')
            self.assertGreater(self.budget('NVIDIA').wait_time(), 0)
            self.clock[0] += 61
            self.assertEqual(self.budget('NVIDIA').wait_time(), 0.0)

    def test_openrouter_limit_is_lower_than_nvidias(self):
        self.assertLess(app._LLM_PROVIDER_LIMITS['OpenRouter']['per_min'],
                        app._LLM_RATE_LIMIT_PER_MIN)

    def test_minimum_gap_is_enforced_per_provider(self):
        with patch.object(app, '_LLM_MIN_GAP', 5.0):
            app._llm_acquire_rate_slot('NVIDIA')
            self.assertAlmostEqual(self.budget('NVIDIA').wait_time(), 5.0, places=1)
            # OpenRouter has its own, smaller gap and is unaffected.
            self.assertEqual(self.budget('OpenRouter').wait_time(), 0.0)

    # ── Circuit breaker ───────────────────────────────────────────────────

    def test_repeated_failures_open_the_breaker_and_success_clears_it(self):
        budget = self.budget('NVIDIA')
        with redirect_stdout(io.StringIO()):
            for _ in range(app._LLM_PROVIDER_STRIKES):
                budget.note(False)
        self.assertIsNone(budget.wait_time(), 'breaker should be open')
        self.assertFalse(app._llm_provider_ready('NVIDIA'))
        budget.note(True)
        self.assertEqual(budget.wait_time(), 0.0)

    def test_breaker_reopens_after_the_cooldown_elapses(self):
        budget = self.budget('NVIDIA')
        with redirect_stdout(io.StringIO()):
            for _ in range(app._LLM_PROVIDER_STRIKES):
                budget.note(False)
        self.assertIsNone(budget.wait_time())
        self.clock[0] += app._LLM_PROVIDER_COOLDOWN_SECONDS + 1
        self.assertEqual(budget.wait_time(), 0.0)

    def test_isolated_failures_do_not_open_the_breaker(self):
        budget = self.budget('NVIDIA')
        for _ in range(app._LLM_PROVIDER_STRIKES - 1):
            budget.note(False)
        budget.note(True)
        budget.note(False)
        self.assertIsNotNone(budget.wait_time())


class RoutingTests(BudgetTests):
    """call_llm must reach for the backup before spending time waiting."""

    def setUp(self):
        super().setUp()
        self.mock('_LLM_COOLDOWN_UNTIL', new=[0.0])
        self.mock('_LLM_LAST_CALL', new=[0.0])
        self.mock('_LLM_CALL_COUNT', new=[0])
        self.mock('_LLM_MIN_GAP', new=0)
        self.mock('OPENROUTER_API_KEY', new='test-key')
        self.router = self.mock('_call_openrouter', return_value='{"ok":true}')

    def mock(self, name, **kwargs):
        patcher = patch.object(app, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def test_saturated_nvidia_routes_straight_to_the_backup(self):
        with patch.object(app, '_LLM_RATE_LIMIT_PER_MIN', 2):
            app._llm_acquire_rate_slot('NVIDIA')
            app._llm_acquire_rate_slot('NVIDIA')
            post = self.mock('_REQUESTS_SESSION')
            with redirect_stdout(io.StringIO()):
                result = app.call_llm('s', 'u')
        self.assertEqual(result, '{"ok":true}')
        self.router.assert_called_once()
        post.post.assert_not_called()

    def test_open_breaker_routes_straight_to_the_backup(self):
        with redirect_stdout(io.StringIO()):
            for _ in range(app._LLM_PROVIDER_STRIKES):
                self.budget('NVIDIA').note(False)
            post = self.mock('_REQUESTS_SESSION')
            result = app.call_llm('s', 'u')
        self.assertEqual(result, '{"ok":true}')
        post.post.assert_not_called()

    def test_healthy_nvidia_is_still_preferred(self):
        body = Mock()
        body.json.return_value = {'model': 'm', 'choices': [{'message': {'content': 'from-nvidia'}}]}
        body.raise_for_status.return_value = None
        session = self.mock('_REQUESTS_SESSION')
        session.post.return_value = body
        with redirect_stdout(io.StringIO()):
            result = app.call_llm('s', 'u')
        self.assertEqual(result, 'from-nvidia')
        self.router.assert_not_called()

    def test_backup_is_not_used_when_fallback_is_disabled(self):
        with patch.object(app, '_LLM_RATE_LIMIT_PER_MIN', 1):
            app._llm_acquire_rate_slot('NVIDIA')
            body = Mock()
            body.json.return_value = {'model': 'm', 'choices': [{'message': {'content': 'nv'}}]}
            body.raise_for_status.return_value = None
            session = self.mock('_REQUESTS_SESSION')
            session.post.return_value = body
            with redirect_stdout(io.StringIO()):
                result = app.call_llm('s', 'u', allow_fallback=False)
        self.assertEqual(result, 'nv')
        self.router.assert_not_called()

    def test_no_backup_key_means_nvidia_is_still_attempted(self):
        self.mock('OPENROUTER_API_KEY', new='')
        with patch.object(app, '_LLM_RATE_LIMIT_PER_MIN', 1):
            app._llm_acquire_rate_slot('NVIDIA')
            body = Mock()
            body.json.return_value = {'model': 'm', 'choices': [{'message': {'content': 'nv'}}]}
            body.raise_for_status.return_value = None
            session = self.mock('_REQUESTS_SESSION')
            session.post.return_value = body
            with redirect_stdout(io.StringIO()):
                result = app.call_llm('s', 'u')
        self.assertEqual(result, 'nv')
        self.router.assert_not_called()


if __name__ == '__main__':
    unittest.main()
