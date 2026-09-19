"""Offline main-module integration: real contracts, ledger, screening and queue.

Only provider/market I/O and presentation boundaries are mocked. Import and all
test outputs are isolated before main is loaded; no credentials are discovered.
"""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = ('NVIDIA_API_KEY', 'OPENROUTER_API_KEY', 'WHATSAPP_PHONE', 'CALLMEBOT_API_KEY')


def import_isolated():
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = Mock(side_effect=AssertionError('dotenv must not load'))
    with tempfile.TemporaryDirectory(prefix='lpm-import-') as output, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': output, 'SCREENER_SKIP_UNIVERSE_FETCH': '1',
            'SCREENER_DISABLE_ALERTS': '1',
        }))
        for name in SECRET_NAMES:
            os.environ.pop(name, None)
        previous_dotenv = sys.modules.get('dotenv')
        sys.modules['dotenv'] = dotenv
        def restore_dotenv():
            if previous_dotenv is None:
                sys.modules.pop('dotenv', None)
            else:
                sys.modules['dotenv'] = previous_dotenv
        stack.callback(restore_dotenv)
        stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')))
        stack.enter_context(patch.object(requests.sessions.Session, 'request', side_effect=AssertionError('network forbidden')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        spec = importlib.util.spec_from_file_location('_lpm_integration_main', ROOT / 'LLM_Portfolio_Manager.py')
        app = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = app
        spec.loader.exec_module(app)
        assert Path(app.DRIVE_FOLDER) == Path(output)
        assert app.NVIDIA_API_KEY == app.OPENROUTER_API_KEY == ''
        dotenv.load_dotenv.assert_not_called()
    return app


APP = import_isolated()


class Clock(datetime):
    instant = datetime(2026, 9, 11, 17, 0, tzinfo=ZoneInfo('America/New_York'))

    @classmethod
    def now(cls, tz=None):
        return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)


def bars(end='2026-09-11', count=65):
    index = pd.bdate_range(end=end, periods=count)
    close = 100 + np.arange(count) * .15 + (np.arange(count) % 2) * .7
    volume = np.full(count, 1_000_000.0)
    volume[-1] = 2_000_000
    return pd.DataFrame({'Open': close - .2, 'High': close + 1,
                         'Low': close - 1, 'Close': close, 'Volume': volume}, index=index)


def news_payload():
    payload = {'macro_summary': 'No material change.', 'market_sentiment': 'NEUTRAL',
               'overall_market_adjustment': 0, 'stock_signals': [], 'sector_signals': []}
    for name in ('trump_signal', 'fed_signal', 'macro_data_signal', 'geopolitical_signal'):
        payload[name] = {'detected': False, 'score_adjustment': 0}
    payload['fed_signal']['tone'] = 'neutral'
    return payload


def decision(ticker='AAA', size=25):
    return {'top_pick': {'ticker': ticker, 'signal': 'BUY', 'confidence': 90,
                          'position_size_pct': size, 'reasoning': 'A test thesis.',
                          'key_risk': 'Demand risk.', 'devils_advocate': 'Demand and execution risks.'},
            'watch_candidates': [], 'derived_rules': [], 'learning_summary': ''}


def response(payload, model='actual-primary-model'):
    result = Mock()
    result.raise_for_status.return_value = None
    result.json.return_value = {'model': model, 'choices': [{'message': {
        'content': payload if isinstance(payload, str) else json.dumps(payload)}}]}
    return result


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='lpm-test-')))
        self.stack.enter_context(patch.dict(os.environ, {
            'SCREENER_OUTPUT_DIR': str(self.output), 'SCREENER_SKIP_UNIVERSE_FETCH': '1',
            'SCREENER_DISABLE_ALERTS': '1', 'SCREENER_ENABLE_SELF_TUNING': '0',
        }))
        for name in SECRET_NAMES + ('GITHUB_STEP_SUMMARY',):
            os.environ.pop(name, None)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch.object(requests.sessions.Session, 'request', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch.object(APP, 'datetime', Clock))
        self.stack.enter_context(patch.object(Clock, 'instant', datetime(2026, 9, 11, 17, tzinfo=ZoneInfo('America/New_York'))))
        values = {'DRIVE_FOLDER': str(self.output), 'PORTFOLIO_JSON': str(self.output / 'portfolio.json'),
                  'PICKS_CSV': str(self.output / 'stock_picks.csv'), 'WATCH_CSV': str(self.output / 'watch_list.csv'),
                  '_CFG_PATH': str(self.output / 'config_overrides.json'), '_HEALTH': APP.RunHealth(),
                  '_ORDER_REASON': [''], '_RUN_MODE': 'test', '_RUN_REPORT': None,
                  'NVIDIA_API_KEY': '', 'OPENROUTER_API_KEY': '', 'WHATSAPP_PHONE': '', 'CALLMEBOT_API_KEY': '',
                  '_LLM_LAST_CALL': [0.0], '_LLM_COOLDOWN_UNTIL': [0.0], '_LLM_CALL_COUNT': [0],
                  '_LAST_LLM_FAILURE_REASON': [''], '_NVIDIA_ACTIVE_MODEL': ['primary'],
                  '_NVIDIA_MODEL_ROTATION': ['primary'], '_OPENROUTER_RECONCILED': [False],
                  '_OPENROUTER_ACTIVE': [None], '_OPENROUTER_MODELS': ['backup:free'],
                  '_LLM_MIN_GAP': 0, '_LLM_BATCH_COOLDOWN_EVERY': 100}
        for name, value in values.items():
            self.stack.enter_context(patch.object(APP, name, value))
        self.stack.enter_context(patch.object(APP, '_llm_acquire_rate_slot'))
        self.stack.enter_context(patch.object(APP.time, 'sleep'))
        self.post = self.stack.enter_context(patch.object(APP._REQUESTS_SESSION, 'post',
                                                       side_effect=AssertionError('unmocked provider call')))

    def mock(self, name, **kwargs):
        return self.stack.enter_context(patch.object(APP, name, **kwargs))

    def pipeline(self, tickers=('AAA',), frames=None):
        self.events = []
        self.frames = frames or {ticker: bars() for ticker in tuple(tickers) + ('SPY', 'QQQ')}
        self.mock('STOCK_UNIVERSE', new=list(tickers))
        self.mock('UNIVERSE_SET', new=set(tickers))
        self.mock('KEY_ETFS', new=['SPY', 'QQQ'])
        self.mock('ETF_SET', new={'SPY', 'QQQ'})
        self.mock('_CFG_ADDITIONAL_TICKERS', new=[])
        self.mock('_CFG_SAMPLE_SIZE', new=900)
        self.mock('MY_STOCKS', new=[])
        self.ctx = {'vix_level': 18.0, 'vix_percentile': 50.0, 'vix_regime': 'MODERATE',
                    'vix_available': True, 'vix_multiplier': 1.0, 'qqq_trend': 'BULLISH',
                    'qqq_vs_ma50': 2.0, 'qqq_price': 100., 'spy_return_today': 0.0,
                    'defensive_mode': False, 'global_macro': {}, 'sector_1d': {}}
        self.context = self.mock('get_market_context', side_effect=lambda: copy.deepcopy(self.ctx))
        monitor = APP.update_portfolio_prices

        def monitoring(pf):
            self.events.append('monitor')
            return monitor(pf)

        self.monitor = self.mock('update_portfolio_prices', side_effect=monitoring)
        self.config = self.mock('load_config_overrides', wraps=APP.load_config_overrides)
        self.probe = self.mock('_reconcile_models_with_catalog', side_effect=lambda: self.events.append('probe'))

        def download(tickers, **kwargs):
            self.events.append('download')
            self.assertFalse(kwargs['auto_adjust'])
            frames = {t: self.frames[t] for t in tickers if t in self.frames}
            return pd.concat(frames, axis=1, sort=True).swaplevel(0, 1, axis=1) if frames else pd.DataFrame()

        self.download = self.stack.enter_context(patch.object(APP.yf, 'download', side_effect=download))

        def instrument(ticker):
            obj = Mock()
            obj.history.side_effect = lambda **kwargs: self.frames.get(ticker, pd.DataFrame()).copy()
            obj.calendar = {}
            return obj

        self.ticker = self.stack.enter_context(patch.object(APP.yf, 'Ticker', side_effect=instrument))
        self.mock('fetch_macro_news', return_value=[])
        self.mock('fetch_all_stock_news_parallel', side_effect=lambda ts: {t: [] for t in ts})
        self.fundamentals = self.mock('fetch_all_fundamentals_parallel',
                                     side_effect=lambda ts: {t: {'sector': 'Technology', 'mkt_cap_b': 20,
                                                                'earnings_days_away': 60} for t in ts})
        self.options = self.mock('fetch_options_and_insider_parallel', return_value=({}, {}))
        self.mock('fetch_congress_trades', return_value={})
        self.mock('fetch_sec_8k', return_value={})
        self.advisory = self.mock('analyze_exit_signals', side_effect=lambda *a, **kw: kw['portfolio'])
        self.report = self.mock('save_html_report')
        self.alert = self.mock('send_whatsapp')
        self.weekly = self.mock('send_weekly_summary')
        self.mock('display_scorecard')
        self.mock('display_result')
        self.queue = self.stack.enter_context(patch.object(APP._portfolio, 'queue_position', wraps=APP._portfolio.queue_position))
        self.immediate = self.mock('open_position', side_effect=AssertionError('signal-day fill forbidden'))
        self.pick = decision(tickers[0] if tickers else 'AAA')
        self.bad_stage = None
        self.ranking = None
        self.deep_ticker = None
        self.provider_stages = []
        self.prompts = {}
        self.catalyst_inputs = []
        self.post.side_effect = self.provider
        return self

    def provider(self, url, **kwargs):
        user = kwargs['json']['messages'][-1]['content']
        if '"ratings":' in user:
            stage = 'catalysts'
            names = re.findall(r'^([A-Z][A-Z0-9.-]*) \[', user, re.M)
            self.catalyst_inputs.extend(names)
            payload = {'ratings': [{'ticker': t, 'catalyst_score': 8, 'catalyst_type': 'BREAKOUT',
                                    'auto_drop': False, 'reason': 'Verified test setup.'} for t in names]}
        elif 'LAYER 1 - MACRO' in user:
            stage, payload = 'news', news_payload()
        elif 'ROUND 1 NOTES:' in user:
            stage, payload = 'final', self.pick
        elif 'These are your top 10 candidates' in user:
            stage, payload = 'round2', {'analyses': [{'ticker': self.deep_ticker or self.pick['top_pick']['ticker'],
                                                     'bull': 'Demand.', 'bear': 'Competition.'}]}
        elif 'rank these for a 1-4 week trade' in user:
            stage = 'round1'
            payload = self.ranking if self.ranking is not None else {
                'top10': [self.pick['top_pick']['ticker']], 'drop': [], 'r1_notes': 'Test ranking.'}
        else:
            raise AssertionError('Unexpected provider prompt')
        self.provider_stages.append(stage)
        self.prompts.setdefault(stage, []).append(user)
        if self.bad_stage == stage:
            payload = '{"incomplete":'
        return response(payload)

    def ledger(self):
        return json.loads(Path(APP.PORTFOLIO_JSON).read_text(encoding='utf-8'))

    def test_offline_import_and_alert_guard(self):
        self.assertEqual(APP.NVIDIA_API_KEY, '')
        self.assertEqual(APP.OPENROUTER_API_KEY, '')
        self.assertEqual(Path(APP.DRIVE_FOLDER), self.output)
        self.mock('WHATSAPP_PHONE', new='test-phone')
        self.mock('CALLMEBOT_API_KEY', new='test-alert-key')
        with patch.object(APP.requests, 'get') as get:
            self.assertFalse(APP._wa_send('Must not send'))
            get.assert_not_called()

    def test_primary_json_records_actual_model_and_stage(self):
        self.post.side_effect = None
        self.post.return_value = response(decision(), 'actual-served-model')
        result = APP._llm_json_with_fallback('system', 'user', max_attempts=1,
                    validator=lambda p: APP.validate_decision(p, [{'ticker': 'AAA', 'sector': 'Technology'}]), stage='final')
        self.assertEqual(result['top_pick']['signal'], 'BUY')
        self.assertEqual(APP._HEALTH.label(), 'NVIDIA:actual-served-model')
        self.assertTrue(APP._HEALTH.as_dict()['stages']['final']['success'])
        self.assertEqual(self.post.call_count, 1)

    def test_invalid_primary_recovers_through_real_backup_transport(self):
        self.mock('OPENROUTER_API_KEY', new='fake-backup-key')
        self.post.side_effect = [response('{"broken":'), response(decision(), 'actual-backup-model')]
        result = APP._llm_json_with_fallback('s', 'u', max_attempts=1,
                    validator=lambda p: APP.validate_decision(p, [{'ticker': 'AAA'}]), stage='final')
        self.assertEqual(result['top_pick']['ticker'], 'AAA')
        self.assertIn('OpenRouter:actual-backup-model', APP._HEALTH.label())
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertTrue(self.post.call_args.kwargs['json']['model'].endswith(':free'))

    def test_transport_failure_retries_without_making_stage_healthy(self):
        self.post.side_effect = requests.exceptions.Timeout('private transport detail')
        with self.assertRaises(ValueError):
            APP._llm_json_with_fallback('s', 'u', max_attempts=1, stage='final')
        health = APP._HEALTH.as_dict()
        self.assertEqual(health['status'], 'failed')
        self.assertEqual(health['failures'], 2)
        self.assertEqual(health['successes'], 0)

    def test_invalid_configuration_has_no_partial_global_update(self):
        before = copy.deepcopy(APP._current_config())
        Path(APP._CFG_PATH).write_text(json.dumps({'RSI_MIN': 40, 'BUY_THRESHOLD': 99}), encoding='utf-8')
        APP.load_config_overrides()
        self.assertEqual(APP._current_config(), before)
        self.assertIn('invalid_config', APP._HEALTH.as_dict()['degraded_reasons'])

    def test_market_gate_precedes_screening_probes_and_alerts(self):
        # Outside a completed session nothing may screen, probe a model, price a
        # replay off incomplete bars, or place an order. Broker reconciliation is
        # deliberately NOT in that list — see the weekend test below.
        self.pipeline()
        for instant in (datetime(2026, 9, 12, 17), datetime(2026, 9, 7, 17), datetime(2026, 9, 11, 16, 14)):
            with self.subTest(instant=instant):
                Clock.instant = instant.replace(tzinfo=ZoneInfo('America/New_York'))
                self.assertIsNone(APP.run_screener())
                self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        for mock in (self.monitor, self.probe, self.context, self.download,
                     self.post, self.alert, self.weekly):
            mock.assert_not_called()

    def test_weekend_run_reconciles_the_broker_but_never_trades(self):
        """A fill is a fact even when this run may not screen.

        The 2026-09-17 MTD fill went unrecorded precisely because the gate
        returned before anything asked Alpaca what had happened.
        """
        self.pipeline()
        Clock.instant = datetime(2026, 9, 12, 17, tzinfo=ZoneInfo('America/New_York'))
        events = [{'kind': 'fill', 'severity': 'info', 'symbol': 'AAA',
                   'summary': 'AAA filled', 'shares': 5, 'price': 10.0}]
        with patch.object(APP, 'sync_with_broker', return_value=events) as sync, \
                patch.object(APP, 'protect_positions', return_value=[]) as protect, \
                patch.object(APP, 'send_execution_alerts') as alerts:
            self.assertIsNone(APP.run_screener())
        sync.assert_called_once()
        # Protection runs on a gated run too: a position must not sit unguarded
        # over a weekend just because there is nothing to screen.
        protect.assert_called_once()
        self.assertIn(events, [call.args[0] for call in alerts.call_args_list])
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertEqual(APP._RUN_MODE, 'no_session')
        # Reconciled state is persisted, but nothing was screened or ordered.
        self.assertTrue(Path(APP.PORTFOLIO_JSON).exists())
        for mock in (self.monitor, self.probe, self.context, self.download,
                     self.post, self.alert, self.weekly):
            mock.assert_not_called()

    def test_already_processed_session_still_reconciles_the_broker(self):
        self.pipeline()
        APP.run_screener()
        with patch.object(APP, 'sync_with_broker', return_value=[]) as sync:
            self.assertIsNone(APP.run_screener())
        sync.assert_called_once()
        self.assertEqual(APP._RUN_MODE, 'already_processed')

    def test_healthy_pipeline_queues_and_persists_without_spending_cash(self):
        self.pipeline()
        result = APP.run_screener()
        pf = self.ledger()
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertEqual(result['top_pick']['order_status'], 'QUEUED')
        self.assertEqual(pf['cash'], APP.STARTING_CAPITAL)
        self.assertEqual(pf['positions'], [])
        self.assertEqual(len(pf['pending_orders']), 1)
        self.assertEqual(pf['processed_sessions'], ['2026-09-11'])
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertLess(self.events.index('monitor'), self.events.index('download'))
        self.assertLess(self.events.index('monitor'), self.events.index('probe'))
        self.assertEqual(self.queue.call_count, 1)
        self.immediate.assert_not_called()
        self.assertFalse(self.report.call_args.kwargs['position_opened'])
        self.assertFalse(self.alert.call_args.kwargs['position_opened'])
        order = pf['pending_orders'][0]
        self.assertEqual(order['execution_session'], '2026-09-14')
        self.assertEqual(self.download.call_args.kwargs['period'], APP._HISTORY_PERIOD)
        self.assertEqual(order['vix'], 18)
        self.assertEqual(order['qqq_trend'], 'BULLISH')
        self.assertEqual(result['top_pick']['facts_as_of'], '2026-09-11')
        csv = pd.read_csv(APP.PICKS_CSV)
        self.assertEqual(csv.iloc[0]['Trade_ID'], order['id'])
        self.assertEqual(csv.iloc[0]['Execution_Status'], 'QUEUED')
        self.assertTrue(pd.isna(csv.iloc[0]['Cost_Basis']))

    def test_repeat_processed_session_has_no_market_llm_order_or_alert(self):
        self.pipeline()
        APP.run_screener()
        before = Path(APP.PORTFOLIO_JSON).read_bytes()
        for mock in (self.monitor, self.context, self.download, self.probe, self.post, self.queue, self.alert, self.report):
            mock.reset_mock()
        self.assertIsNone(APP.run_screener())
        self.assertEqual(APP._HEALTH.as_dict()['stages']['session']['detail'], 'already processed')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertEqual(before, Path(APP.PORTFOLIO_JSON).read_bytes())
        for mock in (self.monitor, self.context, self.download, self.probe, self.post, self.queue, self.alert, self.report):
            mock.assert_not_called()

    def test_alpaca_pending_order_resizes_from_live_cash_baseline(self):
        self.pipeline()
        self.mock('_alpaca', new=SimpleNamespace(
            trading_enabled=lambda: True,
            data_enabled=lambda: True,
            get_account=lambda: {'cash': 100000.0, 'equity': 100000.0},
            effective_shares_by_symbol=lambda: {'MTD': 1},
            plan_reconciliation=lambda want, have: [('buy', 'MTD', want['MTD'] - have.get('MTD', 0))] if want.get('MTD', 0) > have.get('MTD', 0) else [],
            submit_market_order=lambda symbol, qty, side: {'symbol': symbol, 'qty': qty, 'side': side},
            close_position=lambda symbol: None,
        ))
        pf = APP.load_portfolio()
        pf['pending_orders'] = [{
            'id': 'pending-legacy-order', 'trade_id': 'pending-legacy-order',
            'ticker': 'MTD', 'signal_date': '2026-09-16', 'execution_session': '2026-09-17',
            'estimated_entry': 1382.83, 'stop_distance': 54.93, 'target_distance': 109.86,
            'sector': 'Healthcare', 'atr': 36.62, 'amount_usd': 2506.435, 'shares': 1,
            'hold_sessions': 10, 'status': 'PENDING', 'position_size_pct': 25,
        }]
        APP.save_portfolio(pf)
        resized = APP.load_portfolio()
        order = resized['pending_orders'][0]
        self.assertEqual(order['amount_usd'], 25000.0)
        self.assertEqual(order['shares'], 18)

    def test_forced_rerun_reports_existing_pending_order_without_new_pick(self):
        self.pipeline()
        first = APP.run_screener()
        self.assertEqual(first['order_status'], 'QUEUED')
        for mock in (self.download, self.probe, self.post, self.queue, self.alert, self.report):
            mock.reset_mock()
        with patch.dict(os.environ, {'SCREENER_FORCE_SESSION': '1'}):
            result = APP.run_screener()
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertEqual(result['top_pick']['ticker'], 'AAA')
        self.assertIn('Existing session order already queued', result['order_reason'])
        self.assertEqual(APP._RUN_MODE, 'existing_session_order')
        self.download.assert_not_called()
        self.probe.assert_not_called()
        self.post.assert_not_called()
        self.queue.assert_not_called()
        self.report.assert_called_once()
        self.alert.assert_called_once()

    def test_news_failure_is_loud_but_does_not_block_the_trade(self):
        # Trading continues on verified prices, verified catalysts and the final
        # decision. The loss is recorded, degrades the run and is alerted - it
        # simply no longer stops execution.
        self.pipeline()
        self.bad_stage = 'news'
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'QUEUED')
        health = APP._HEALTH.as_dict()
        self.assertFalse(health['stages']['news']['success'], 'the loss must be recorded')
        self.assertIn('advisory_news_unavailable', health['degraded_reasons'])
        self.assertEqual(health['status'], 'degraded', 'the run must not look clean')
        self.assertTrue(APP._trade_readiness()['trade_ready'])

    def test_a_rejected_news_response_is_never_applied_to_candidates(self):
        # The original hazard: apply_news ran on whatever came back, so a bad
        # response could auto-drop every candidate and the run would report
        # 'all candidates dropped by news' - a failure to evaluate disguised as
        # a decision. A rejected response must now be discarded outright.
        self.pipeline()
        self.bad_stage = 'news'
        applied = self.mock('apply_news', side_effect=AssertionError(
            'a rejected news response must never reach apply_news'))
        result = APP.run_screener()
        applied.assert_not_called()
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertFalse(APP._HEALTH.as_dict()['stages']['news']['success'])

    def test_invalid_config_degrades_full_pipeline_without_queueing(self):
        self.pipeline()
        Path(APP._CFG_PATH).write_text('{"RSI_MIN": 40, "max_positions": 100}', encoding='utf-8')
        before = copy.deepcopy(APP._current_config())
        result = APP.run_screener()
        self.assertEqual(APP._current_config(), before)
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertIn('invalid_config', APP._HEALTH.as_dict()['degraded_reasons'])
        self.queue.assert_not_called()
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_successful_catalyst_drops_are_legitimate_no_pick(self):
        self.pipeline()
        provider = self.provider

        def drop_all(url, **kwargs):
            result = provider(url, **kwargs)
            raw = result.json.return_value['choices'][0]['message']['content']
            payload = json.loads(raw)
            for rating in payload.get('ratings', []):
                rating['auto_drop'] = True
            return response(payload)

        self.post.side_effect = drop_all
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])
        self.assertEqual(self.post.call_count, 1)

    def test_missing_final_stage_cannot_queue_a_fabricated_confident_result(self):
        self.pipeline()
        self.mock('analyze_with_nvidia', return_value=decision())
        result = APP.run_screener()
        self.assertEqual(result['top_pick']['confidence'], 0)
        self.assertFalse(APP._HEALTH.as_dict()['stages']['final']['success'])
        self.queue.assert_not_called()

    def test_optional_proposal_failure_does_not_block_core_ready_order(self):
        self.pipeline()
        self.mock('update_config_from_llm', side_effect=lambda history:
                  APP._HEALTH.stage('config_proposal', False, 'invalid proposal'))
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertEqual(result['top_pick']['signal'], 'BUY')
        self.queue.assert_called_once()
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])
        health = APP.write_run_health(result)
        self.assertEqual(health['status'], 'degraded')
        self.assertTrue(health['trade_ready'])
        self.assertEqual(health['trade_blockers'], [])
        self.assertTrue(health['stages']['final']['success'])

    def test_pending_slots_respect_hard_five_even_if_runtime_limit_is_larger(self):
        self.pipeline()
        pf = APP.load_portfolio()
        for i in range(5):
            Clock.instant = datetime(2026, 9, 14 + i, 17, tzinfo=ZoneInfo('America/New_York'))
            pick = decision('P' + chr(65 + i))['top_pick']
            self.assertTrue(APP._portfolio.queue_position(APP, pf, pick, 100, 96, 108,
                                                         {'sector': 'Technology', 'atr': 2}))
        Clock.instant = datetime(2026, 9, 11, 17, tzinfo=ZoneInfo('America/New_York'))
        self.mock('_CFG_MAX_POSITIONS', new=99)
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'REJECTED')
        self.assertIn('including pending', result['order_reason'])
        self.assertEqual(len(self.ledger()['pending_orders']), 5)

    def test_existing_failed_health_resets_on_next_completed_run(self):
        self.pipeline()
        APP._HEALTH.stage('final', False, 'previous run')
        APP._HEALTH.degrade('previous error')
        self.assertEqual(APP.main(), 0)
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertEqual(APP._HEALTH.as_dict()['degraded_reasons'], [])

    def test_current_held_ticker_in_final_model_output_is_rejected(self):
        self.pipeline(('AAA', 'HELD'))
        pf = APP.load_portfolio()
        entry = float(bars()['Close'].iloc[-1])
        APP._portfolio.open_position(APP, pf, 'HELD', entry, 1000, entry - 4, entry + 8, 'Energy', 2)
        APP.save_portfolio(pf)
        self.pick = decision('HELD')
        result = APP.run_screener()
        self.assertEqual(result['top_pick']['signal'], 'NO PICK')
        self.queue.assert_not_called()
        self.assertEqual(len(self.ledger()['positions']), 1)

    def test_csv_report_history_is_not_used_as_trade_or_learning_input(self):
        self.pipeline()
        row = {name: '' for name in APP.PICK_COLS}
        row.update(Date='2026-08-01', Ticker='FAKE', Signal='BUY', Confidence=99,
                   Result='Win', Return_Pct=100, Entry_Price=1,
                   Execution_Status='SIGNAL', Outcome_Basis='hypothetical_excess_return')
        pd.DataFrame([row]).to_csv(APP.PICKS_CSV, index=False)
        analyze = self.mock('analyze_with_nvidia', wraps=APP.analyze_with_nvidia)
        self.assertEqual(APP.run_screener()['order_status'], 'QUEUED')
        self.assertEqual(analyze.call_args.kwargs['pick_history'], [])
        self.assertEqual(self.ledger()['positions'], [])
        self.assertEqual(self.ledger()['closed_trades'], [])

    def test_retry_after_data_failure_does_not_double_count_held_sessions(self):
        frames = {'AAA': bars(), 'HELD': bars(), 'SPY': bars(), 'QQQ': bars('2026-09-10')}
        self.pipeline(('AAA', 'HELD'), frames)
        Clock.instant = datetime(2026, 9, 10, 17, tzinfo=ZoneInfo('America/New_York'))
        pf = APP.load_portfolio()
        entry = float(bars()['Close'].iloc[-2])
        # Isolate replay from the portfolio's separate binary fee/debit guard.
        # Fee coverage is explicitly known here, not assumed when FX is absent.
        APP._portfolio.open_position(APP, pf, 'HELD', entry, 1000, entry - 4, entry + 8,
                         'Energy', 2, nzdusd_rate=.6)
        self.assertEqual(len(pf['positions']), 1, APP._ORDER_REASON[0])
        APP.save_portfolio(pf)
        Clock.instant = datetime(2026, 9, 11, 17, tzinfo=ZoneInfo('America/New_York'))
        APP.run_screener()
        first = self.ledger()
        APP.run_screener()
        second = self.ledger()
        self.assertEqual(first['positions'][0]['held_sessions'], 1)
        self.assertEqual(first['positions'], second['positions'])
        self.assertEqual(first['cash'], second['cash'])
        self.assertEqual(first['closed_trades'], second['closed_trades'])
        self.assertEqual(second['processed_sessions'], [])

    def test_run74_catalyst_errors_recover_without_repeating_healthy_batches(self):
        self.pipeline(tuple(f'T{i:02d}' for i in range(30)))
        self.mock('OPENROUTER_API_KEY', new='fake-backup-key')
        calls = []
        failures = iter((
            '{"ratings":[{"ticker":"T12","catalyst_score":7,"catalyst_score":8}]}',
            'No JSON available',
            {'ratings': [{'ticker': 'RNR', 'catalyst_score': 7,
                          'catalyst_type': 'BREAKOUT', 'auto_drop': False, 'reason': 'Wrong ticker.'}]},
        ))

        def provider(url, **kwargs):
            user = kwargs['json']['messages'][-1]['content']
            names = re.findall(r'^([A-Z][A-Z0-9.-]*) \[', user, re.M)
            if names:
                calls.append(names)
                if names == [f'T{i:02d}' for i in range(12, 18)]:
                    return response(next(failures))
            return self.provider(url, **kwargs)

        self.post.side_effect = provider
        result = APP.run_screener()
        health = APP.write_run_health(result)
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertTrue(health['trade_ready'])
        self.assertEqual(health['status'], 'healthy')
        self.assertEqual(health['stages']['catalysts']['detail'], '30/30 verified')
        self.assertTrue(health['stages']['catalyst_3']['success'])
        self.assertIn('Recovered', health['stages']['catalyst_3']['detail'])
        self.assertEqual([len(names) for names in calls], [6, 6, 6, 6, 6, 3, 3, 6, 6])
        self.assertEqual(calls[5], ['T12', 'T13', 'T14'])
        self.assertEqual(calls[6], ['T15', 'T16', 'T17'])
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])
        self.assertEqual(len(self.ledger()['pending_orders']), 1)

    def catalyst_fixture(self, count):
        self.pipeline()
        return [{'ticker': f'T{i:02d}', 'sector': 'Technology', 'rsi': 55.,
                 'adx': 25., 'momentum_5d': 2., 'tech_score': 40} for i in range(count)]

    def test_catalyst_prompt_names_exact_tickers_and_forbids_duplicate_keys(self):
        candidates = self.catalyst_fixture(6)
        APP.batch_catalyst_score(candidates, self.ctx, {'T00': ['RNR also appeared in this headline']})
        prompt = self.prompts['catalysts'][0]
        self.assertIn('Allowed tickers (exactly 6): T00, T01, T02, T03, T04, T05', prompt)
        self.assertIn('Never repeat a JSON key', prompt)
        self.assertNotIn('"ticker":"X"', prompt)
        self.assertEqual(self.post.call_count, 1)
        self.assertTrue(all(c['catalyst_verified'] for c in candidates))

    def test_failed_split_does_not_accept_partial_or_unknown_ratings(self):
        candidates = self.catalyst_fixture(6)

        def provider(url, **kwargs):
            user = kwargs['json']['messages'][-1]['content']
            names = re.findall(r'^([A-Z][A-Z0-9.-]*) \[', user, re.M)
            if len(names) == 6 or 'T05' in names:
                return response({'ratings': [{'ticker': 'RNR'}]})
            return self.provider(url, **kwargs)

        self.post.side_effect = provider
        APP.batch_catalyst_score(candidates, self.ctx, {})
        self.assertFalse(any(c['catalyst_verified'] for c in candidates))
        self.assertFalse(any('catalyst_score' in c for c in candidates))
        stages = APP._HEALTH.as_dict()['stages']
        self.assertFalse(stages['catalyst_1']['success'])
        self.assertFalse(stages['catalysts']['success'])
        self.assertFalse(APP._trade_readiness()['trade_ready'])

    def test_catalyst_split_recovery_budget_is_run_wide(self):
        candidates = self.catalyst_fixture(30)
        sizes = []

        def provider(url, **kwargs):
            user = kwargs['json']['messages'][-1]['content']
            names = re.findall(r'^([A-Z][A-Z0-9.-]*) \[', user, re.M)
            sizes.append(len(names))
            if len(names) == 6:
                return response('{"invalid":')
            return self.provider(url, **kwargs)

        self.post.side_effect = provider
        APP.batch_catalyst_score(candidates, self.ctx, {})
        self.assertEqual(sizes.count(3), 4)
        self.assertEqual(len(sizes), 12)  # Four primary attempts (two transports each), four recovery calls.
        self.assertEqual(sum(c['catalyst_verified'] for c in candidates), 12)
        self.assertFalse(APP._HEALTH.as_dict()['stages']['catalysts']['success'])

    def test_catalyst_recovery_handles_short_final_batch(self):
        candidates = self.catalyst_fixture(2)

        def provider(url, **kwargs):
            user = kwargs['json']['messages'][-1]['content']
            names = re.findall(r'^([A-Z][A-Z0-9.-]*) \[', user, re.M)
            if len(names) == 2:
                return response('{"invalid":')
            return self.provider(url, **kwargs)

        self.post.side_effect = provider
        APP.batch_catalyst_score(candidates, self.ctx, {})
        self.assertTrue(all(c['catalyst_verified'] for c in candidates))
        self.assertEqual(self.catalyst_inputs, ['T00', 'T01'])
        self.assertEqual(self.post.call_count, 4)

    def test_catalyst_failure_cannot_be_hidden_by_final_success(self):
        self.pipeline()
        self.bad_stage = 'catalysts'
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertFalse(APP._HEALTH.as_dict()['stages']['catalysts']['success'])
        self.queue.assert_not_called()
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_invalid_final_unknown_ticker_and_size_fail_closed(self):
        self.pipeline()
        for bad in (decision('UNKNOWN'), decision(size='25'), decision(size=0)):
            with self.subTest(pick=bad):
                self.pick = bad
                result = APP.run_screener()
                self.assertEqual(result['top_pick']['signal'], 'NO PICK')
                self.assertEqual(APP._HEALTH.as_dict()['status'], 'failed')
                self.assertEqual(self.ledger()['processed_sessions'], [])
        self.queue.assert_not_called()

    def test_undersized_buy_is_rejected_without_minimum_uplift(self):
        self.pipeline()
        self.pick = decision(size=.01)
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'REJECTED')
        self.assertIn('Order not queued:', result['order_reason'])
        self.assertEqual(result['top_pick']['position_size_pct'], .01)
        self.assertEqual(self.ledger()['cash'], APP.STARTING_CAPITAL)
        self.assertEqual(self.ledger()['pending_orders'], [])
        # The size was refused, so nothing reached Alpaca and nothing was
        # decided - the session stays open rather than being closed by a
        # rejection. The point of this test, that a tiny size is never
        # silently rounded up to something buyable, is unchanged.
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_missing_selected_sector_rejected_but_optional_coverage_not_global_failure(self):
        self.pipeline()
        self.fundamentals.side_effect = lambda ts: {}
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'REJECTED')
        self.assertIn('sector', result['order_reason'].lower())
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertIn('AAA', APP._HEALTH.as_dict()['stages']['fundamental_coverage']['detail'])

    def test_missing_unselected_fundamentals_does_not_block_known_sector(self):
        self.pipeline(('AAA', 'BBB'))
        self.fundamentals.side_effect = lambda ts: {'AAA': {'sector': 'Technology'}}
        self.assertEqual(APP.run_screener()['order_status'], 'QUEUED')

    def test_missing_benchmark_returns_after_monitoring_without_probes(self):
        self.pipeline(frames={'AAA': bars(), 'SPY': bars(), 'QQQ': bars('2026-09-10')})
        result = APP.run_screener()
        self.monitor.assert_called_once()
        self.probe.assert_not_called()
        self.post.assert_not_called()
        self.queue.assert_not_called()
        self.report.assert_called_once()
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertFalse(APP._HEALTH.as_dict()['stages']['market_data']['success'])
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_stale_candidate_excluded_and_future_bars_not_screened(self):
        frames = {t: bars() for t in ('AAA', 'SPY', 'QQQ')}
        frames['AAA'] = pd.concat([frames['AAA'], pd.DataFrame(
            {'Open': [900], 'High': [1001], 'Low': [899], 'Close': [1000], 'Volume': [9e6]},
            index=pd.DatetimeIndex(['2026-09-14']))])
        frames['OLD'] = bars('2026-09-10')
        self.pipeline(('AAA', 'OLD'), frames)
        screened = self.mock('screen_technical', wraps=APP.screen_technical)
        result = APP.run_screener()
        supplied = screened.call_args.args[0]
        self.assertNotIn('OLD', supplied)
        self.assertEqual(supplied['AAA'].index[-1].date().isoformat(), '2026-09-11')
        self.assertLess(result['top_pick']['price'], 200)
        self.assertNotIn('OLD', self.catalyst_inputs)

    def test_no_candidates_is_processed_only_for_valid_data(self):
        self.pipeline()
        self.mock('screen_technical', return_value={})
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])
        self.post.assert_not_called()
        self.options.assert_not_called()

    def test_missing_all_stock_quotes_is_not_healthy_no_candidates(self):
        self.pipeline(frames={'AAA': bars('2026-09-10'), 'SPY': bars(), 'QQQ': bars()})
        result = APP.run_screener()
        self.assertIn('failure_reason', result)
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_prescoring_once_before_catalysts_and_maximum_thirty(self):
        tickers = tuple('A' + chr(65 + i // 26) + chr(65 + i % 26) for i in range(35))
        self.pipeline(tickers)
        enrich = self.mock('enrich_with_scores', wraps=APP.enrich_with_scores)
        APP.run_screener()
        self.assertEqual(len(self.catalyst_inputs), 30)
        self.assertEqual(self.catalyst_inputs, sorted(tickers)[:30])
        self.assertEqual(enrich.call_count, 2)
        self.assertEqual(len(enrich.call_args_list[0].args[0]), 35)
        self.assertEqual(len(enrich.call_args_list[1].args[0]), 30)
        self.assertTrue(all(c['tech_score'] > 0 for c in enrich.call_args_list[0].args[0]))

    def test_held_and_pending_filtered_before_options_or_catalysts(self):
        self.pipeline(('AAA', 'HELD', 'PEND'))
        pf = APP.load_portfolio()
        entry = float(bars()['Close'].iloc[-1])
        APP._portfolio.open_position(APP, pf, 'HELD', entry, 1000, entry - 4, entry + 8, 'Technology', 2)
        APP.save_portfolio(pf)
        self.assertTrue(APP._portfolio.queue_position(APP, pf, decision('PEND')['top_pick'], entry,
                                                     entry - 4, entry + 8, {'sector': 'Technology', 'atr': 2}))
        APP.run_screener()
        names = [c['ticker'] for c in self.options.call_args.args[0]]
        self.assertEqual(names, ['AAA'])
        self.assertEqual(self.catalyst_inputs, ['AAA'])

    def test_transaction_write_failure_is_retryable_and_preserves_cash(self):
        self.pipeline()
        save = APP._portfolio.save_portfolio

        def fail_pending(app, pf):
            if pf['pending_orders']:
                raise OSError('disk failure')
            return save(app, pf)

        with patch.object(APP._portfolio, 'save_portfolio', side_effect=fail_pending):
            with self.assertRaises(OSError):
                APP.main()
        pf = self.ledger()
        self.assertEqual(pf['cash'], APP.STARTING_CAPITAL)
        self.assertEqual(pf['pending_orders'], [])
        self.assertEqual(pf['processed_sessions'], [])
        health = json.loads((self.output / 'run_health.json').read_text(encoding='utf-8'))
        self.assertEqual(health['stages']['final']['detail'], 'unhandled error:OSError')
        self.assertEqual(APP.run_screener()['order_status'], 'QUEUED')

    def test_main_exit_two_after_degraded_reports_and_zero_for_noop(self):
        self.pipeline()
        self.bad_stage = 'news'
        self.assertEqual(APP.main(), 2)
        self.report.assert_called_once()
        self.assertTrue(Path(APP.PORTFOLIO_JSON).exists())
        health = json.loads((self.output / 'run_health.json').read_text(encoding='utf-8'))
        self.assertEqual(health['status'], 'degraded')
        self.assertEqual(health['report'], str(self.output / 'report_latest.html'))
        Clock.instant = datetime(2026, 9, 12, 17, tzinfo=ZoneInfo('America/New_York'))
        self.assertEqual(APP.main(), 0)
        health = json.loads((self.output / 'run_health.json').read_text(encoding='utf-8'))
        self.assertEqual(health['mode'], 'no_session')
        self.assertIsNone(health['report'])

    def test_health_summary_counts_actual_models_and_redacts_secrets(self):
        self.mock('NVIDIA_API_KEY', new='fake-private-key')
        APP._HEALTH.provider('NVIDIA', 'actual-model', True)
        APP._HEALTH.provider('OpenRouter', 'failed-model', False)
        APP._HEALTH.stage('final', False, 'fake-private-key https://private.invalid/path Bearer token')
        APP._HEALTH.degrade('fake-private-key')
        summary = self.output / 'summary.md'
        summary.write_text('Existing summary\n', encoding='utf-8')
        os.environ['GITHUB_STEP_SUMMARY'] = str(summary)
        APP.write_run_health()
        text = summary.read_text(encoding='utf-8')
        raw = (self.output / 'run_health.json').read_text(encoding='utf-8')
        self.assertTrue(text.startswith('Existing summary'))
        self.assertIn('NVIDIA:actual-model', text)
        # The alerts stage now also reports, so the count is 1 of 2.
        self.assertIn('Stages: 1/2', text)
        self.assertIn('alerts: OK', text)
        self.assertIn('Provider attempts: 2; successes: 1; failures: 1', text)
        self.assertNotIn('fake-private-key', text + raw)
        self.assertNotIn('private.invalid', text + raw)
        self.assertNotIn('Bearer token', text + raw)

    def test_batch_window_covers_ma200_and_52_week_warmup(self):
        # A calendar-day window is not a session count: 65 calendar days is
        # commonly only 46 bars. The window must clear BOTH the 200 sessions
        # MA200 needs and the 252 a real 52-week high needs, or
        # compute_indicators silently substitutes spot price for MA200 and
        # mislabels a short-window high as a 52-week high.
        self.pipeline()

        def yahoo_window(tickers, **kwargs):
            count = 520 if kwargs['period'] == APP._HISTORY_PERIOD else 46
            frames = {ticker: bars(count=count) for ticker in tickers}
            return pd.concat(frames, axis=1).swaplevel(0, 1, axis=1)

        self.download.side_effect = yahoo_window
        self.assertEqual(APP.run_screener()['order_status'], 'QUEUED')
        self.assertEqual(self.download.call_args.kwargs['period'], APP._HISTORY_PERIOD)
        self.assertFalse(self.download.call_args.kwargs['auto_adjust'])
        self.assertTrue(APP._HEALTH.as_dict()['stages']['market_data']['success'])

    def test_configured_window_is_long_enough_for_both_indicators(self):
        sessions_per_year = 252
        years = int(APP._HISTORY_PERIOD.rstrip('y'))
        self.assertGreaterEqual(years * sessions_per_year, APP._MIN_SESSIONS_52W)
        self.assertGreaterEqual(years * sessions_per_year, APP._MIN_SESSIONS_MA200)
        # Alpaca is specified in calendar days; ~252 of every 365 are sessions.
        self.assertGreaterEqual(APP._HISTORY_CALENDAR_DAYS * 252 / 365,
                                APP._MIN_SESSIONS_52W)

    def test_ma200_and_52_week_high_are_real_values_not_fallbacks(self):
        # The regression this guards: with too little history MA200 fell back to
        # spot price, making vs_ma200_pct exactly 0.0 and locking the top tier
        # of the moving-average score out of reach for every stock, forever.
        long_history = bars(count=520)
        indicators = APP.compute_indicators(long_history)
        self.assertNotEqual(indicators['vs_ma200_pct'], 0.0)
        self.assertLess(indicators['ma200'], indicators['price'])
        short_history = bars(count=126)
        self.assertEqual(APP.compute_indicators(short_history)['vs_ma200_pct'], 0.0)

    def test_entry_rsi_matches_the_exit_rule_rsi(self):
        # The buy filter and the sell rule must measure RSI the same way, or
        # "don't buy over 75" and "sell over 78" are different scales.
        history = bars(count=300)
        entry = APP.compute_indicators(history)['rsi']
        closes = history['Close'].astype(float)
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean().iloc[-1]
        exit_rsi = 100.0 if loss == 0 and gain > 0 else 100 - 100 / (1 + gain / loss)
        self.assertAlmostEqual(entry, round(exit_rsi, 1), places=1)

    def test_bearish_market_scores_the_cautious_tier_not_the_unknown_one(self):
        scores = {}
        for sentiment in ('BULLISH', 'NEUTRAL', 'BEARISH', 'SOMETHING_ELSE'):
            _, breakdown, _ = APP.compute_news_score([], [], None, None, sentiment)
            scores[sentiment] = breakdown['macro_alignment']
        self.assertEqual(scores['BULLISH'], 10)
        self.assertEqual(scores['NEUTRAL'], 6)
        self.assertEqual(scores['BEARISH'], 3, 'BEARISH must not fall through to the catch-all')
        self.assertEqual(scores['SOMETHING_ELSE'], 1)

    def test_weekend_volume_relaxation_uses_the_new_york_session_date(self):
        # The runner's clock is UTC on CI and can be a different weekday, which
        # would relax the volume filter on a real trading session.
        with patch.object(APP, '_session_date', return_value='2026-09-12'):   # Saturday
            self.assertTrue(pd.Timestamp(APP._session_date()).weekday() >= 5)
        with patch.object(APP, '_session_date', return_value='2026-09-11'):   # Friday
            self.assertFalse(pd.Timestamp(APP._session_date()).weekday() >= 5)

    def test_unknown_fx_charges_and_tracks_each_side_normally(self):
        for rate in (None, 0, -0.6):
            with self.subTest(rate=rate):
                pf = {}
                self.assertEqual(APP._sharesies_fee(200, pf, rate), 1.0)
                self.assertEqual(APP._sharesies_fee(300, pf, rate), 1.5)
                self.assertEqual(APP._sharesies_fee(400, pf, rate, side='sell'), 2.0)
                self.assertEqual(pf['sharesies_bought_usd'], 500)
                self.assertEqual(pf['sharesies_sold_usd'], 400)
                self.assertEqual(pf['sharesies_month'], '2026-09')

    def test_unknown_fx_fee_is_capped_at_five_dollars(self):
        for amount in (1000, 1001, 10_000, 1_000_000):
            for side in ('buy', 'sell'):
                with self.subTest(amount=amount, side=side):
                    pf = {}
                    self.assertEqual(APP._sharesies_fee(amount, pf, side=side), 5.0)
                    key = 'sharesies_bought_usd' if side == 'buy' else 'sharesies_sold_usd'
                    self.assertEqual(pf[key], amount)

    def test_known_fx_preserves_coverage_and_counts_prior_unknown_usage(self):
        self.mock('SHARESIES_COVERAGE_NZD', new=5000)
        pf = {}
        self.assertEqual(APP._sharesies_fee(1000, pf), 5.0)
        pf['last_nzdusd_rate'] = .6
        self.assertEqual(APP._sharesies_fee(2000, pf), 0.0)
        self.assertEqual(APP._sharesies_fee(400, pf), 2.0)
        self.assertEqual(APP._sharesies_fee(2000, pf), 5.0)
        self.assertEqual(APP._sharesies_fee(3100, pf, side='sell'), .5)
        self.assertEqual(pf['sharesies_bought_usd'], 5400)
        self.assertEqual(pf['sharesies_sold_usd'], 3100)

    def test_fee_month_uses_session_date_and_resets_both_sides(self):
        pf = {'sharesies_month': '2026-09', 'sharesies_bought_usd': 8000,
              'sharesies_sold_usd': 4000}
        with patch.object(APP, '_session_date', return_value='2026-08-31'):
            self.assertEqual(APP._sharesies_fee(200, pf), 1.0)
        self.assertEqual(pf, {'sharesies_month': '2026-08', 'sharesies_bought_usd': 200,
                              'sharesies_sold_usd': 0})
        self.assertEqual(APP._sharesies_fee(300, pf, side='sell'), 1.5)
        self.assertEqual(pf, {'sharesies_month': '2026-09', 'sharesies_bought_usd': 0,
                              'sharesies_sold_usd': 300})

    def test_fee_rejects_invalid_numbers_before_mutating_usage(self):
        for bad in (float('nan'), float('inf'), -float('inf'), True, '0.6', [], {}):
            for field in ('amount', 'explicit_fx', 'stored_fx', 'usage'):
                with self.subTest(field=field, value=bad):
                    pf = {'sharesies_month': '2026-09', 'sharesies_bought_usd': 10,
                          'sharesies_sold_usd': 20, 'last_nzdusd_rate': .6}
                    if field == 'stored_fx':
                        pf['last_nzdusd_rate'] = bad
                    if field == 'usage':
                        pf['sharesies_bought_usd'] = bad
                    before = json.dumps(pf)
                    with self.assertRaises(ValueError):
                        APP._sharesies_fee(bad if field == 'amount' else 200, pf,
                                           bad if field == 'explicit_fx' else None)
                    self.assertEqual(json.dumps(pf), before)
        pf = {}
        with self.assertRaises(ValueError):
            APP._sharesies_fee(-1, pf)
        self.assertEqual(pf, {})

    def test_missing_or_failed_required_stage_is_not_ready(self):
        # Each of these makes a trade unsafe if it fails: prices must be real, a
        # catalyst must be verified rather than invented, and there must be a
        # decision. News is deliberately absent - see the test below.
        self.assertEqual(APP._CORE_STAGES, ('market_data', 'catalysts', 'final'))
        for required in APP._CORE_STAGES:
            for missing in (True, False):
                with self.subTest(required=required, missing=missing):
                    health = APP.RunHealth()
                    for name in APP._CORE_STAGES:
                        if name != required:
                            health.stage(name, True, 'validated')
                        elif not missing:
                            health.stage(name, False, 'validation failed')
                    with patch.object(APP, '_HEALTH', health):
                        self.assertFalse(APP._require_core_health())
                        self.assertFalse(health.as_dict()['stages'][required]['success'])
                        if required != 'final':
                            self.assertTrue(health.as_dict()['stages']['final']['success'])

    def test_failed_news_does_not_make_the_run_unready(self):
        # News enrichment only adds score adjustments; losing it leaves the
        # final model less informed, not wrong. Gating execution on it let one
        # brittle LLM response stop the screener trading indefinitely.
        health = APP.RunHealth()
        for name in APP._CORE_STAGES:
            health.stage(name, True, 'validated')
        health.stage('news', False, 'schema rejected')
        with patch.object(APP, '_HEALTH', health):
            self.assertTrue(APP._require_core_health())

    def test_optional_stage_and_provider_failures_preserve_final_validation(self):
        for name in ('market_data', 'catalysts', 'news', 'final'):
            APP._HEALTH.stage(name, True, 'validated')
        for name in ('round1', 'round2', 'exit:HELD', 'config_proposal', 'paper_horizon:OLD'):
            APP._HEALTH.stage(name, False, 'optional unavailable')
        APP._HEALTH.provider('NVIDIA', 'unavailable-optional-model', False)
        before = APP._HEALTH.as_dict()
        self.assertTrue(APP._require_core_health())
        self.assertEqual(APP._HEALTH.as_dict(), before)
        self.assertEqual(before['status'], 'degraded')

    def test_optional_round1_failure_still_queues_and_processes(self):
        self.pipeline()
        self.bad_stage = 'round1'
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'QUEUED')
        health = APP._HEALTH.as_dict()
        self.assertEqual(health['status'], 'degraded')
        self.assertFalse(health['stages']['round1']['success'])
        self.assertTrue(health['stages']['final']['success'])
        self.assertNotIn('failure_reason', result)
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])

    def test_optional_round2_failure_keeps_exit_two_but_reports_trade_ready(self):
        self.pipeline()
        self.bad_stage = 'round2'
        summary = self.output / 'summary.md'
        os.environ['GITHUB_STEP_SUMMARY'] = str(summary)
        self.assertEqual(APP.main(), 2)
        health = json.loads((self.output / 'run_health.json').read_text(encoding='utf-8'))
        self.assertEqual(health['status'], 'degraded')
        self.assertTrue(health['trade_ready'])
        self.assertEqual(health['trade_blockers'], [])
        self.assertEqual(health['order_status'], 'QUEUED')
        self.assertIn('Core trade readiness: READY', summary.read_text(encoding='utf-8'))
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])

    def test_optional_exit_and_paper_horizon_failures_do_not_block_orders(self):
        self.pipeline()

        def advice(*args, **kwargs):
            APP._HEALTH.stage('exit:HELD', False, 'advisory unavailable')
            return kwargs['portfolio']

        def paper(*args):
            APP._HEALTH.stage('paper_horizon:OLD', False, 'hypothetical data missing')
            APP._degrade('paper_horizon_unavailable:OLD')
            APP._degrade('paper_horizon_error:OLD:provider unavailable')

        self.advisory.side_effect = advice
        self.mock('update_results', side_effect=paper)
        self.assertEqual(APP.run_screener()['order_status'], 'QUEUED')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'degraded')
        self.assertTrue(APP._require_core_health())
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])

    def test_blocking_degradation_prevents_readiness_even_with_valid_core(self):
        for reason in ('invalid_config', 'stale_vix', 'stale_quote:HELD',
                       'earnings_unavailable:HELD', 'corporate_action_review_required:HELD',
                       'future_entry:HELD', 'unknown_risk'):
            with self.subTest(reason=reason), patch.object(APP, '_HEALTH', APP.RunHealth()):
                for name in ('market_data', 'catalysts', 'news', 'final'):
                    APP._HEALTH.stage(name, True, 'validated')
                APP._degrade(reason)
                self.assertFalse(APP._require_core_health())
                self.assertEqual(APP._trade_readiness()['trade_blockers'], [reason])
                self.assertTrue(APP._HEALTH.as_dict()['stages']['final']['success'])

    def test_empty_exit_cannot_invent_missing_upstream_success(self):
        self.pipeline()
        for skipped in (('catalysts', 'news', 'final'), ('news', 'final'), ('final',)):
            with self.subTest(skipped=skipped), patch.object(APP, '_HEALTH', APP.RunHealth()):
                pf = APP.load_portfolio()
                result = APP._finish_no_pick(self.ctx, pf, 'Empty candidates', not_required=skipped)
                self.assertIn('failure_reason', result)
                self.assertEqual(pf['processed_sessions'], [])
                self.assertFalse(APP._HEALTH.as_dict()['stages']['market_data']['success'])

    def test_catalyst_returning_empty_without_validation_is_not_legitimate(self):
        self.pipeline()
        self.mock('batch_catalyst_score', return_value=[])
        result = APP.run_screener()
        self.assertIn('failure_reason', result)
        self.assertFalse(APP._HEALTH.as_dict()['stages']['catalysts']['success'])
        self.assertEqual(self.ledger()['processed_sessions'], [])
        self.queue.assert_not_called()

    def test_valid_no_candidates_with_optional_failure_still_processes(self):
        self.pipeline()
        self.mock('screen_technical', return_value={})
        self.advisory.side_effect = lambda *a, **kw: APP._HEALTH.stage('exit:HELD', False, 'optional')
        result = APP.run_screener()
        self.assertNotIn('failure_reason', result)
        self.assertEqual(result['order_status'], 'NO ORDER')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'degraded')
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])

    def test_round1_drop_is_removed_from_later_prompts_and_final_membership(self):
        self.pipeline(('AAA', 'BBB'))
        self.ranking = {'top10': ['AAA', 'BBB'], 'drop': ['AAA'], 'r1_notes': 'Prefer the survivor.'}
        self.deep_ticker = 'BBB'
        self.pick = decision('AAA')  # Model ignores the exclusion: contract must reject it.
        result = APP.run_screener()
        self.assertEqual(result['top_pick']['signal'], 'NO PICK')
        self.assertFalse(APP._HEALTH.as_dict()['stages']['final']['success'])
        self.assertTrue(APP._HEALTH.as_dict()['stages']['round1']['success'])
        for prompt in self.prompts['round2'] + self.prompts['final']:
            self.assertNotIn('"ticker": "AAA"', prompt)
            self.assertIn('"ticker": "BBB"', prompt)
        self.queue.assert_not_called()
        self.assertEqual(self.ledger()['processed_sessions'], [])

    def test_round1_survivor_can_queue_when_ranking_only_named_dropped_ticker(self):
        self.pipeline(('AAA', 'BBB'))
        self.ranking = {'top10': ['AAA'], 'drop': ['AAA'], 'r1_notes': ''}
        self.pick = decision('BBB')
        result = APP.run_screener()
        self.assertEqual(result['order_status'], 'QUEUED')
        self.assertEqual(self.ledger()['pending_orders'][0]['ticker'], 'BBB')
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        for prompt in self.prompts['round2'] + self.prompts['final']:
            self.assertNotIn('"ticker": "AAA"', prompt)

    def test_round1_dropped_watch_is_also_rejected(self):
        self.pipeline(('AAA', 'BBB'))
        self.ranking = {'top10': ['BBB'], 'drop': ['AAA'], 'r1_notes': ''}
        self.pick = decision('BBB')
        watch = decision('AAA')['top_pick']
        watch['signal'] = 'WATCH'
        self.pick['watch_candidates'] = [watch]
        self.assertEqual(APP.run_screener()['order_status'], 'NO ORDER')
        self.assertFalse(APP._HEALTH.as_dict()['stages']['final']['success'])
        self.queue.assert_not_called()

    def test_round1_all_dropped_returns_validated_none_without_subsequent_calls(self):
        self.pipeline(('AAA', 'BBB'))
        self.ranking = {'top10': ['AAA', 'BBB'], 'drop': ['AAA', 'BBB'], 'r1_notes': ''}
        result = APP.run_screener()
        self.assertEqual(result['top_pick']['ticker'], 'NONE')
        self.assertEqual(result['top_pick']['signal'], 'NO PICK')
        self.assertEqual(result['top_pick']['confidence'], 0)
        self.assertIn('Round 1', result['top_pick']['reasoning'])
        self.assertEqual(result['watch_candidates'], [])
        self.assertEqual(APP._HEALTH.as_dict()['status'], 'healthy')
        self.assertTrue(APP._HEALTH.as_dict()['stages']['final']['success'])
        self.assertEqual(self.provider_stages, ['catalysts', 'news', 'round1'])
        self.assertEqual(self.ledger()['processed_sessions'], ['2026-09-11'])
        self.queue.assert_not_called()

    def test_standard_nyse_holidays_and_weekends_share_portfolio_calendar(self):
        cases = (('2025-12-31', '2026-01-02'), ('2026-01-16', '2026-01-20'),
                 ('2026-02-13', '2026-02-17'), ('2026-04-02', '2026-04-06'),
                 ('2026-05-22', '2026-05-26'), ('2026-06-18', '2026-06-22'),
                 ('2026-07-02', '2026-07-06'), ('2026-09-04', '2026-09-08'),
                 ('2026-09-11', '2026-09-14'), ('2026-11-25', '2026-11-27'),
                 ('2026-12-24', '2026-12-28'), ('2026-12-31', '2027-01-04'))
        with patch.object(APP, '_next_session_date', wraps=APP._next_session_date) as next_session:
            for signal, expected in cases:
                with self.subTest(signal=signal):
                    self.assertEqual(APP._portfolio.expected_execution_session(APP, signal), expected)
                    self.assertEqual(next_session.call_args.args[0].isoformat(), signal)
                    day = datetime.fromisoformat(expected).replace(hour=17, tzinfo=ZoneInfo('America/New_York'))
                    self.assertEqual(APP._session_gate(day), '')
        self.assertEqual(APP._portfolio._sessions_after(APP, '2026-09-04', '2026-09-09'),
                         ['2026-09-08', '2026-09-09'])

    def test_new_year_saturday_is_not_a_friday_closure_but_sunday_is_monday(self):
        self.assertEqual(APP._us_market_holiday(datetime(2021, 12, 31)), '')
        self.assertEqual(APP._next_session_date('2021-12-30'), '2021-12-31')
        self.assertEqual(APP._next_session_date('2021-12-31'), '2022-01-03')
        self.assertEqual(APP._us_market_holiday(datetime(2022, 1, 3)), '')
        self.assertEqual(APP._us_market_holiday(datetime(2023, 1, 2)), "New Year's Day")
        self.assertEqual(APP._next_session_date('2022-12-30'), '2023-01-03')

    def test_session_gate_holidays_cutoff_dst_and_conservative_half_day(self):
        for day in ('2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
                    '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
                    '2026-11-26', '2026-12-25', '2026-09-12', '2026-09-13'):
            with self.subTest(day=day):
                instant = datetime.fromisoformat(day).replace(hour=17, tzinfo=ZoneInfo('America/New_York'))
                self.assertTrue(APP._session_gate(instant))
        for day, hour in (('2026-03-06', 21), ('2026-03-09', 20), ('2026-11-02', 21)):
            with self.subTest(day=day):
                instant = datetime.fromisoformat(day).replace(hour=hour, minute=15, tzinfo=ZoneInfo('UTC'))
                self.assertEqual(APP._session_gate(instant), '')
                self.assertEqual(APP._session_gate(instant.replace(minute=14)), 'before 16:15 ET')
        half_day = datetime(2026, 11, 27, 13, 15, tzinfo=ZoneInfo('America/New_York'))
        self.assertEqual(APP._session_gate(half_day), 'before 16:15 ET')
        self.assertEqual(APP._session_gate(half_day.replace(hour=16)), '')
        with self.assertRaises(ValueError):
            APP._session_gate(datetime(2026, 9, 11, 17))

    def test_session_date_uses_new_york_even_when_utc_is_next_month(self):
        Clock.instant = datetime(2026, 10, 1, 0, 30, tzinfo=ZoneInfo('UTC'))
        self.assertEqual(APP._session_date(), '2026-09-30')
        self.assertEqual(APP._session_gate(Clock.instant), '')
        pf = {}
        APP._sharesies_fee(200, pf)
        self.assertEqual(pf['sharesies_month'], '2026-09')

    def test_vix_stale_missing_or_invalid_is_display_only_and_blocks_new_order(self):
        self.pipeline()
        self.context.side_effect = self.real_context
        invalid = bars()
        invalid.loc[invalid.index[-1], 'Close'] = float('nan')
        for frame in (bars('2026-09-10'), pd.DataFrame(), invalid):
            with self.subTest(rows=len(frame)):
                self.frames['^VIX'] = frame
                result = APP.run_screener()
                ctx = self.report.call_args.args[1]
                self.assertFalse(ctx['vix_available'])
                self.assertEqual(ctx['vix_level'], 20.0)
                self.assertEqual(result['order_status'], 'NO ORDER')
                health = APP.write_run_health(result)
                self.assertFalse(health['stages']['market_context']['success'])
                self.assertTrue(health['stages']['final']['success'])
                self.assertFalse(health['trade_ready'])
                self.assertIn('stale_vix', health['trade_blockers'])
                self.assertIn('stale_vix', health['degraded_reasons'])
                self.assertEqual(self.ledger()['processed_sessions'], [])
        self.queue.assert_not_called()

    def test_vix_exact_session_ignores_future_rows_and_fx_keeps_four_decimals(self):
        self.pipeline()
        vix = bars()
        future = vix.iloc[[-1]].copy()
        future.index = pd.DatetimeIndex(['2026-09-14'])
        future[['Open', 'High', 'Low', 'Close']] = 999
        self.frames['^VIX'] = pd.concat([future, vix.iloc[::-1]])
        self.frames['NZDUSD=X'] = pd.DataFrame({'Close': [.60123, .61234]},
                                               index=pd.to_datetime(['2026-09-10', '2026-09-11']))
        ctx = self.real_context()
        self.assertTrue(ctx['vix_available'])
        self.assertEqual(ctx['vix_level'], round(float(vix['Close'].iloc[-1]), 2))
        self.assertEqual(ctx['global_macro']['nzdusd']['price'], .6123)
        self.assertTrue(APP._HEALTH.as_dict()['stages']['market_context']['success'])
        self.assertNotIn('stale_vix', APP._HEALTH.as_dict()['degraded_reasons'])

    def test_stale_vix_cannot_certify_otherwise_legitimate_empty_screen(self):
        self.pipeline()
        self.context.side_effect = self.real_context
        self.frames['^VIX'] = bars('2026-09-10')
        self.mock('screen_technical', return_value={})
        result = APP.run_screener()
        self.assertIn('failure_reason', result)
        self.assertEqual(self.ledger()['processed_sessions'], [])
        self.queue.assert_not_called()

    def test_queue_and_rejection_visible_in_real_html_and_messages(self):
        self.pipeline()
        real_report = self.real_report
        result = APP.run_screener()
        pf = self.ledger()
        self.mock('WHATSAPP_PHONE', new='test-phone')
        self.mock('CALLMEBOT_API_KEY', new='test-alert-key')
        with patch.object(APP, '_wa_send') as send:
            for status in ('QUEUED', 'REJECTED'):
                result['order_status'] = status
                result['top_pick'].update(order_status=status, order_reason='<denied> & limit')
                real_report(result, self.ctx, {}, 110., [], portfolio=pf, position_opened=False)
                html = (self.output / 'report_latest.html').read_text(encoding='utf-8')
                self.assertIn(status, html)
                self.assertIn('&lt;denied&gt; &amp; limit', html)
                self.assertNotIn('BOUGHT AAA', html)
                self.real_whatsapp(result['top_pick'], self.ctx, 110., [], 106., 118., portfolio=pf)
            messages = '\n'.join(call.args[0] for call in send.call_args_list)
            # A queued order must read as "not yet bought", and a rejected one as
            # "no order placed" — neither may ever look like a completed purchase.
            self.assertIn('Order placed: buy', messages)
            self.assertIn('Nothing has been bought yet and no money has been spent',
                          messages)
            self.assertIn('NOTHING WAS BOUGHT TODAY', messages)
            # A refused order must not advertise sell prices or a thesis,
            # which would read as though something had been bought.
            self.assertNotIn('Sell if it falls to', messages.split('NOTHING WAS BOUGHT')[1])
            # Plain language first, but an unrecognised refusal must still
            # carry its raw wording so it stays diagnosable.
            self.assertIn('Why:', messages)
            self.assertIn('<denied> & limit', messages)
            # Precise, not a bare substring: 'WAITING TO BE BOUGHT (no money
            # spent yet)' is the OPPOSITE of a completed purchase. What must
            # never appear is the heading announcing one.
            self.assertNotIn(chr(10) + 'BOUGHT' + chr(10), chr(10) + messages + chr(10))
            self.assertNotIn('- 22 shares of AAA at', messages)
            self.assertNotIn('likely no cash', messages)


    def protect(self, positions, held=None, existing=None, submit=None, cancel=True):
        """Run protect_positions against a fake broker; return (events, calls)."""
        APP._BROKER_SYNC_OK[0] = True
        calls = []

        def _submit(symbol, qty, stop, target, ref=''):
            calls.append(('submit', symbol, qty, stop, target))
            return None if submit == 'reject' else {'id': 'oco-1'}

        with patch.object(APP._alpaca, 'trading_enabled', return_value=True), \
                patch.object(APP._alpaca, 'positions_by_symbol',
                             return_value=held if held is not None else {'MTD': 18}), \
                patch.object(APP._alpaca, 'protective_orders_by_symbol',
                             return_value=existing or {}), \
                patch.object(APP._alpaca, 'submit_protective_oco', side_effect=_submit), \
                patch.object(APP._alpaca, 'cancel_order',
                             side_effect=lambda oid: calls.append(('cancel', oid)) or cancel), \
                redirect_stdout(io.StringIO()):
            events = APP.protect_positions({'positions': positions})
        return events, calls

    POS = {'ticker': 'MTD', 'shares': 18, 'trade_id': 'tid-1',
           'stop_price': 1353.69, 'target_price': 1502.46}

    def test_an_open_position_gets_a_resting_stop_at_the_broker(self):
        # The whole point: the stop must be an instruction Alpaca acts on
        # intraday, not a number checked once a day after the close.
        events, calls = self.protect([dict(self.POS)])
        self.assertEqual(calls, [('submit', 'MTD', 18, 1353.69, 1502.46)])
        self.assertEqual([e['kind'] for e in events], ['protected'])
        self.assertFalse(events[0]['moved'])

    def test_matching_protection_is_left_alone(self):
        existing = {'MTD': {'order_id': 'o1', 'qty': 18, 'stop': 1353.69, 'limit': 1502.46}}
        events, calls = self.protect([dict(self.POS)], existing=existing)
        self.assertEqual(calls, [], 'must not churn an already-correct order')
        self.assertEqual(events, [])

    def test_a_ratcheted_trailing_stop_replaces_the_old_order(self):
        existing = {'MTD': {'order_id': 'o1', 'qty': 18, 'stop': 1300.00, 'limit': 1502.46}}
        events, calls = self.protect([dict(self.POS)], existing=existing)
        self.assertEqual(calls[0], ('cancel', 'o1'))
        self.assertEqual(calls[1], ('submit', 'MTD', 18, 1353.69, 1502.46))
        self.assertTrue(events[0]['moved'])

    def test_a_failed_cancel_does_not_leave_two_live_orders(self):
        existing = {'MTD': {'order_id': 'o1', 'qty': 18, 'stop': 1300.00, 'limit': 1502.46}}
        events, calls = self.protect([dict(self.POS)], existing=existing, cancel=False)
        self.assertNotIn('submit', [c[0] for c in calls])
        self.assertEqual([e['kind'] for e in events], ['protection_failed'])

    def test_quantity_comes_from_the_broker_not_the_ledger(self):
        # Alpaca is the source of truth; protecting 18 when 15 are held would
        # be rejected or would oversell.
        events, calls = self.protect([dict(self.POS)], held={'MTD': 15})
        self.assertEqual(calls[0][2], 15)

    def test_a_position_without_levels_is_reported_not_guessed(self):
        events, calls = self.protect([{'ticker': 'MTD', 'shares': 18,
                                       'needs_risk_levels': True}])
        self.assertEqual(calls, [])
        self.assertEqual([e['kind'] for e in events], ['unprotected'])

    def test_an_inverted_stop_and_target_is_refused(self):
        bad = dict(self.POS, stop_price=1502.46, target_price=1353.69)
        events, calls = self.protect([bad])
        self.assertEqual(calls, [])
        self.assertEqual([e['kind'] for e in events], ['unprotected'])

    def test_a_position_already_queued_for_a_market_exit_is_skipped(self):
        exiting = dict(self.POS, exit_requested={'reason': 'stop_loss',
                                                 'session': '2026-09-18'})
        events, calls = self.protect([exiting])
        self.assertEqual(calls, [])
        self.assertEqual(events, [])

    def test_nothing_held_at_the_broker_means_nothing_to_protect(self):
        events, calls = self.protect([dict(self.POS)], held={})
        self.assertEqual(calls, [])

    def test_a_rejected_order_is_reported_as_unprotected(self):
        events, calls = self.protect([dict(self.POS)], submit='reject')
        self.assertEqual([e['kind'] for e in events], ['protection_failed'])

    def test_protection_is_skipped_when_reconciliation_is_untrustworthy(self):
        APP._BROKER_SYNC_OK[0] = False
        with patch.object(APP._alpaca, 'trading_enabled', return_value=True), \
                patch.object(APP._alpaca, 'protective_orders_by_symbol',
                             side_effect=AssertionError('must not touch the broker')):
            self.assertEqual(APP.protect_positions({'positions': [dict(self.POS)]}), [])

    def test_a_rejected_order_leaves_the_session_open_for_a_retry(self):
        # Nothing reached Alpaca, so nothing was decided. Closing the day on a
        # rejection lets a local flag claim a decision the broker never saw,
        # and queue_position then refuses to revisit that session forever.
        portfolio = {'processed_sessions': [], 'positions': [], 'pending_orders': []}
        with patch.object(APP, '_require_core_health', return_value=True), \
                patch.object(APP, 'save_portfolio'), \
                patch.object(APP, 'reconcile_broker'), \
                patch.object(APP, '_session_date', return_value='2026-09-18'), \
                redirect_stdout(io.StringIO()):
            APP._persist_session(portfolio, decided=False)
        self.assertEqual(portfolio['processed_sessions'], [])

    def test_a_deliberate_no_pick_still_closes_the_session(self):
        portfolio = {'processed_sessions': [], 'positions': [], 'pending_orders': []}
        with patch.object(APP, '_require_core_health', return_value=True), \
                patch.object(APP, 'save_portfolio'), \
                patch.object(APP, 'reconcile_broker'), \
                patch.object(APP, '_session_date', return_value='2026-09-18'), \
                redirect_stdout(io.StringIO()):
            APP._persist_session(portfolio, decided=True)
        self.assertEqual(portfolio['processed_sessions'], ['2026-09-18'])

    def test_sector_resolution_unpacks_the_fundamentals_tuple(self):
        # _fetch_fundamentals_single returns (ticker, data). Calling .get() on
        # the tuple raised, the bare except swallowed it, and EVERY adoption was
        # silently skipped - which then let the mirror sell the real position.
        with patch.object(APP, '_fetch_fundamentals_single',
                          return_value=('MTD', {'sector': 'Healthcare'})):
            self.assertEqual(APP._resolve_sector('MTD'), 'Healthcare')

    def test_sector_resolution_rejects_a_sector_the_planner_cannot_map(self):
        with patch.object(APP, '_fetch_fundamentals_single',
                          return_value=('XYZ', {'sector': 'Biotechnology'})):
            self.assertEqual(APP._resolve_sector('XYZ'), '')

    def test_sector_resolution_survives_a_provider_failure(self):
        with patch.object(APP, '_fetch_fundamentals_single',
                          side_effect=RuntimeError('yahoo down')):
            self.assertEqual(APP._resolve_sector('MTD'), '')

    def test_mirror_never_sells_a_holding_the_ledger_failed_to_adopt(self):
        # plan_reconciliation({}, {'MTD': 18}) returns [('sell','MTD',18)]. When
        # adoption fails, obeying that diff liquidates a real position.
        APP._BROKER_SYNC_OK[0] = True
        submitted = []
        with patch.object(APP._alpaca, 'trading_enabled', return_value=True), \
                patch.object(APP._alpaca, 'get_account', return_value={'equity': '100000'}), \
                patch.object(APP._alpaca, 'effective_shares_by_symbol',
                             return_value={'MTD': 18}), \
                patch.object(APP._alpaca, 'submit_market_order',
                             side_effect=lambda *a, **k: submitted.append(a)), \
                patch.object(APP._alpaca, 'close_position',
                             side_effect=lambda *a: submitted.append(a)), \
                redirect_stdout(io.StringIO()):
            APP.reconcile_broker({'positions': [], 'pending_orders': []})
        self.assertEqual(submitted, [], 'an unadopted holding must never be sold')

    def test_mirror_does_nothing_at_all_when_reconciliation_did_not_complete(self):
        APP._BROKER_SYNC_OK[0] = False
        with patch.object(APP._alpaca, 'trading_enabled', return_value=True), \
                patch.object(APP._alpaca, 'get_account',
                             side_effect=AssertionError('must not touch the broker')), \
                redirect_stdout(io.StringIO()):
            APP.reconcile_broker({'positions': [], 'pending_orders': []})

    def test_mirror_still_sells_a_position_the_ledger_knows_and_has_exited(self):
        APP._BROKER_SYNC_OK[0] = True
        submitted = []
        # A real position carries a trade_id, which the mirror stamps onto the
        # sell so the next run can match the fill back to this record.
        ledger = {'positions': [{'ticker': 'MTD', 'shares': 18,
                                 'trade_id': 'tid-1',
                                 'exit_requested': {'reason': 'stop_loss',
                                                    'session': '2026-09-18'}}],
                  'pending_orders': []}
        with patch.object(APP._alpaca, 'trading_enabled', return_value=True), \
                patch.object(APP._alpaca, 'get_account', return_value={'equity': '100000'}), \
                patch.object(APP._alpaca, 'effective_shares_by_symbol',
                             return_value={'MTD': 18}), \
                patch.object(APP._alpaca, 'submit_market_order',
                             side_effect=lambda *a, **k: submitted.append(a) or {'id': 'x'}), \
                redirect_stdout(io.StringIO()):
            APP.reconcile_broker(ledger)
        self.assertEqual([(a[0], a[2]) for a in submitted], [('MTD', 'sell')])


IntegrationTests.real_report = staticmethod(APP.save_html_report)
IntegrationTests.real_whatsapp = staticmethod(APP.send_whatsapp)
IntegrationTests.real_context = staticmethod(APP.get_market_context)


if __name__ == '__main__':
    unittest.main()