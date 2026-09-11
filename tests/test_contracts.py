"""Offline contract regressions. Read staged main only; never import or edit it."""

import ast
import copy
import inspect
import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import screener_contracts as contracts


ROOT = Path(__file__).resolve().parents[1]
BAD_NUMBERS = (True, False, '7', None, float('nan'), float('inf'), -float('inf'), [], {})


def candidates():
    return [
        {'ticker': 'ABC', 'sector': 'Technology', 'source': 'BOTH',
         'rsi': 55.5, 'short_ratio': 2.25, 'short_pct_float': 4.5,
         'quote_date': '2026-09-10', 'price': 100.0, 'pre_score': 81,
         'tech_score': 49, 'news_score': 32, 'catalyst_score': 8,
         'catalyst_type': 'BREAKOUT', 'news_notes': 'Verified headline',
         'stock_news': ['Known company event'], 'atr': 2.0},
        {'ticker': 'XYZ', 'sector': 'Energy', 'source': 'TECHNICAL',
         'rsi': 48.0, 'short_ratio': None, 'short_pct_float': None,
         'quote_date': '2026-09-10', 'pre_score': 70, 'tech_score': 50, 'news_score': 20},
    ]


def catalysts():
    return {'ratings': [
        {'ticker': ticker, 'catalyst_score': 7.5, 'catalyst_type': 'BREAKOUT',
         'auto_drop': False, 'reason': 'A specific upcoming product launch.'}
        for ticker in ('ABC', 'XYZ')
    ]}


def news():
    result = {
        'macro_summary': 'Balanced outlook.', 'market_sentiment': 'NEUTRAL',
        'overall_market_adjustment': 0,
        'stock_signals': [{'ticker': 'ABC', 'news': 'Product launch',
                           'score_adjustment': 2.5, 'auto_drop': False}],
        'sector_signals': [{'sector': 'Energy', 'news': 'Supply unchanged',
                            'score_adjustment': -3}],
    }
    for key in contracts.NEWS_SIGNAL_BLOCKS:
        result[key] = {'detected': False, 'score_adjustment': 0}
    result['trump_signal'].update({'detail': 'None', 'affected_sectors': ['Technology']})
    result['fed_signal']['tone'] = 'neutral'
    return result


def decision():
    return {
        'top_pick': {'ticker': 'ABC', 'signal': 'BUY', 'confidence': 85.5,
                     'position_size_pct': 25, 'reasoning': 'Novel thesis, not a verified fact.',
                     'key_risk': 'Demand could fall.', 'devils_advocate': 'Competition could intensify.'},
        'watch_candidates': [{'ticker': 'XYZ', 'signal': 'WATCH', 'confidence': 73,
                              'reasoning': 'Wait for confirmation.', 'key_risk': 'Prices may fall.'}],
        'derived_rules': ['A model hypothesis, not a proven trading rule.'],
        'learning_summary': 'Evidence is limited.',
    }


def exit_response():
    return {'ticker': 'ABC', 'action': 'HOLD', 'urgency': 'LOW',
            'reason': 'Thesis unchanged.', 'exit_price': None}


def config():
    return {
        'RSI_MIN': 30, 'RSI_MAX': 75, 'ADX_MIN': 15,
        'BUY_THRESHOLD': 80, 'WATCH_THRESHOLD': 65, 'VOLUME_MIN_RATIO': 1.2,
        'ATR_STOP_MULT': 2, 'ATR_TARGET_MULT': 3,
        'max_positions': 5, 'min_cash_floor': 500.0, 'hold_days': 10, 'trail_atr_mult': 1.5,
    }


def full_config():
    result = config()
    result.update({
        'sector_blacklist': ['Utilities'], 'sector_whitelist': ['Technology'],
        'source_preference': 'ANY', 'require_congress': False,
        'min_catalyst_score': 0, 'min_adx_buy': 15, 'avoid_earnings_week': False,
        'max_vix': 999, 'min_price': 5, 'only_profitable': False, 'require_above_ma': True,
        'min_dollar_volume_m': 5, 'sector_conc_max': 3, 'sample_size': 900,
        'additional_tickers': ['BRK-B', 'BRK.B', 'PLTR'], 'brokerage_fee': 1.5,
        'dd_caution_pct': -10, 'dd_severe_pct': -20, 'dd_critical_pct': -30,
        'win_threshold_pct': 2, 'loss_threshold_pct': -2, 'min_picks_to_learn': 5,
        'rsi_hard_cap': 999, 'rsi_cap_conf': 70, 'upside_hard_cap': -999,
        'upside_cap_conf': 65, 'vix_low_pctile': 25, 'vix_high_pctile': 75,
        'sector_conc_lookback': 10, 'sector_conc_penalty': 10,
        'congress_days': 60, 'sec_8k_days': 7, 'rsi_exit': 78,
        'rsi_exit_min_profit': 0, 'macd_exit_min_profit': 0, 'entry_slippage_pct': 0.3,
        'final_candidates': 30, 'pre_earnings_exit_days': 0,
        'squeeze_float_pct': 20, 'squeeze_days_to_cover': 5, 'volume_min_ratio': 1.2,
        'reasoning': 'No change without sufficient evidence.',
    })
    return result


class OfflineTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        network.start()
        self.addCleanup(network.stop)


class ParserTests(OfflineTests):
    def test_complete_objects_and_surrounding_prose(self):
        payload = {'top_pick': {'ticker': 'ABC'}, 'text': 'braces { } [ ] and "quotes"'}
        encoded = json.dumps(payload)
        for raw in (
            encoded, '  ' + encoded + '\n', 'Here is the result:\n' + encoded + '\nDone.',
            'The "answer" is: ' + encoded + ' This is "provisional".',
            '```json\n' + encoded + '\n```', '```\n' + encoded + '\n```',
            'Here is the result:\n```JSON\n' + encoded + '\n```\nDone.',
            '<think>Ignore {broken JSON in reasoning} [also this]</think>\n' + encoded,
            '<think>Reason one</think><think>Reason two</think>```json\n' + encoded + '\n```',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(contracts.parse_object(raw), payload)
        self.assertEqual(contracts.parse_object('{}'), {})

    def test_nonobjects_and_bad_input_types(self):
        for raw in ('', 'nothing', 'null', 'true', '123', '[]', '[{"x":1}]',
                    '42 {"x":1}', 'null {"x":1}', '"{\\"x\\":1}"', None, {}, [], b'{}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contracts.parse_object(raw)

    def test_no_quote_substitution_control_cleanup_or_comma_repair(self):
        for raw in ("{'x':1}", '{"x":1,}', '{"x":"bad\x01value"}',
                    '{"x":1 //comment\n}', '{"x":"unterminated}', '{{"x":1}}',
                    '{note} {"x":1}', '{"x":01}', '{"x":undefined}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contracts.parse_object(raw)

    def test_truncated_outer_object_never_salvages_complete_nested_object(self):
        nested = json.dumps(decision()['top_pick'])
        for raw in (
            '{', '{"top_pick":' + nested,
            '{"top_pick":' + nested + ',"watch_candidates":[',
            '{"top_pick":' + nested + ',"watch_candidates":[{"ticker":"XYZ"}',
            '```json\n{"top_pick":' + nested + '\n```',
            '{"ratings":[{"ticker":"ABC","catalyst_score":8,"catalyst_type":"BREAKOUT",'
            '"auto_drop":false,"reason":"Complete inner rating"}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contracts.parse_object(raw)

    def test_duplicate_keys_rejected_at_every_depth(self):
        for raw in ('{"x":1,"x":2}', '{"nested":{"ticker":"ABC","ticker":"XYZ"}}',
                    '{"rows":[{"x":1,"x":2}]}', '{"x":1,"\\u0078":2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contracts.parse_object(raw)

    def test_nonfinite_json_values_rejected_at_every_depth(self):
        for number in ('NaN', 'Infinity', '-Infinity', '1e999', '-1e999'):
            for raw in ('{"n":' + number + '}', '{"n":[{"value":' + number + '}]}'):
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    contracts.parse_object(raw)

    def test_ambiguous_trailing_json_and_bad_wrappers(self):
        for raw in ('{} {}', '{} some text {"other":1}', '{} []', '{} [truncated',
                    '{} }', '{} ]', '```json\n{}', '{}\n```',
                    '<think>unfinished {}', '```python\n{}\n```',
                    '```json\n{}\n``` more ```json\n{}\n```'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contracts.parse_object(raw)

    def test_raw_decode_is_called_only_once(self):
        original = json.JSONDecoder.raw_decode
        with patch.object(json.JSONDecoder, 'raw_decode', autospec=True, side_effect=original) as decode:
            with self.assertRaises(ValueError):
                contracts.parse_object('{bad outer {"valid_inner":true}}')
            self.assertEqual(decode.call_count, 1)


class CatalystTests(OfflineTests):
    def test_exact_coverage_preserves_independent_payload(self):
        payload = catalysts()
        payload['provider_extra'] = {'nested': [1]}
        result = contracts.validate_catalysts(payload, candidates())
        self.assertEqual(result, payload)
        result['ratings'][0]['reason'] = 'Changed'
        result['provider_extra']['nested'].append(2)
        self.assertNotEqual(result, payload)
        self.assertEqual(payload['provider_extra']['nested'], [1])

    def test_missing_duplicate_unknown_or_partial_coverage(self):
        original = catalysts()['ratings']
        for ratings in ([], original[:1], original + original[:1], [original[0], original[0]],
                        original + [dict(original[0], ticker='GHOST')]):
            with self.subTest(ratings=ratings), self.assertRaises(ValueError):
                contracts.validate_catalysts({'ratings': ratings}, candidates())

    def test_all_required_rating_fields_and_outer_shape(self):
        for key in catalysts()['ratings'][0]:
            payload = catalysts()
            del payload['ratings'][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate_catalysts(payload, candidates())
        for payload in ({}, [], None, {'ratings': {}}, {'ratings': [None]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                contracts.validate_catalysts(payload, candidates())

    def test_scores_enums_booleans_and_reason_constraints(self):
        invalid = {
            'catalyst_score': BAD_NUMBERS + (0, 10.001),
            'auto_drop': (1, 0, 'false', None), 'catalyst_type': ('FAKE', 'breakout', None, []),
            'reason': ('', '  ', 'x' * 601, None, 42),
        }
        for field, values in invalid.items():
            for value in values:
                payload = catalysts()
                payload['ratings'][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    contracts.validate_catalysts(payload, candidates())
        for score in (1, 10, 1.1):
            payload = catalysts()
            payload['ratings'][0].update(catalyst_score=score, reason='x' * 600)
            contracts.validate_catalysts(payload, candidates())
        for kind in contracts.CATALYST_TYPES:
            payload = catalysts()
            payload['ratings'][0]['catalyst_type'] = kind
            contracts.validate_catalysts(payload, candidates())


class NewsTests(OfflineTests):
    def test_normalized_copy_ignores_unknown_provider_extras(self):
        payload = news()
        payload['fed_signal']['tone'] = ' HaWkIsH '
        payload['provider_extra'] = {'unvalidated': True}
        payload['stock_signals'][0]['extra'] = 'discard'
        payload['fed_signal']['extra'] = 'discard'
        original = copy.deepcopy(payload)
        result = contracts.validate_news(payload, candidates())
        self.assertEqual(result['fed_signal']['tone'], 'hawkish')
        self.assertNotIn('provider_extra', result)
        self.assertNotIn('extra', result['stock_signals'][0])
        self.assertNotIn('extra', result['fed_signal'])
        result['trump_signal']['affected_sectors'].append('Energy')
        self.assertEqual(payload, original)

    def test_no_signals_is_valid_but_absent_signals_are_not_defaulted(self):
        payload = news()
        payload['stock_signals'] = []
        payload['sector_signals'] = []
        payload['trump_signal']['affected_sectors'] = []
        result = contracts.validate_news(payload, [])
        self.assertEqual(result['stock_signals'], [])
        self.assertNotIn('detail', result['fed_signal'])
        self.assertNotIn('affected_sectors', result['fed_signal'])
        for key in news():
            payload = news()
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate_news(payload, candidates())

    def test_all_block_fields_required_and_strict(self):
        for block in contracts.NEWS_SIGNAL_BLOCKS:
            for field in ('detected', 'score_adjustment'):
                payload = news()
                del payload[block][field]
                with self.subTest(block=block, field=field), self.assertRaises(ValueError):
                    contracts.validate_news(payload, candidates())
            for field, values in {
                'detected': (1, 0, 'false', None),
                'score_adjustment': BAD_NUMBERS + (-30.01, 30.01),
                'affected_sectors': ('Technology', [1], ['GHOST'], [None]),
                'detail': (1, None, []),
            }.items():
                for value in values:
                    payload = news()
                    payload[block][field] = value
                    with self.subTest(block=block, field=field, value=value), self.assertRaises(ValueError):
                        contracts.validate_news(payload, candidates())
            for value in (None, [], 'neutral'):
                payload = news()
                payload[block] = value
                with self.subTest(block=block, value=value), self.assertRaises(ValueError):
                    contracts.validate_news(payload, candidates())

    def test_fed_tone_required_and_enum_normalized(self):
        payload = news()
        del payload['fed_signal']['tone']
        with self.assertRaises(ValueError):
            contracts.validate_news(payload, candidates())
        for tone in ('DOVISH', 'neutral', ' HaWkIsH '):
            payload['fed_signal']['tone'] = tone
            result = contracts.validate_news(payload, candidates())
            self.assertEqual(result['fed_signal']['tone'], tone.strip().lower())
        for tone in ('mixed', None, False, 42):
            payload['fed_signal']['tone'] = tone
            with self.subTest(tone=tone), self.assertRaises(ValueError):
                contracts.validate_news(payload, candidates())

    def test_stock_and_sector_signal_membership_types_and_duplicates(self):
        for key, identity in (('stock_signals', 'ticker'), ('sector_signals', 'sector')):
            for field in news()[key][0]:
                payload = news()
                del payload[key][0][field]
                with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                    contracts.validate_news(payload, candidates())
            invalid = {identity: ('GHOST', None, 1), 'news': (None, 5, []),
                       'score_adjustment': BAD_NUMBERS + (-31, 31)}
            if identity == 'ticker':
                invalid['auto_drop'] = (1, 0, 'false', None)
            for field, values in invalid.items():
                for value in values:
                    payload = news()
                    payload[key][0][field] = value
                    with self.subTest(key=key, field=field, value=value), self.assertRaises(ValueError):
                        contracts.validate_news(payload, candidates())
            payload = news()
            payload[key].append(copy.deepcopy(payload[key][0]))
            with self.assertRaises(ValueError):
                contracts.validate_news(payload, candidates())
            for value in (None, {}, 'signals', [False]):
                payload = news()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_news(payload, candidates())

    def test_macro_types_and_adjustment_boundaries(self):
        for field, values in {
            'overall_market_adjustment': BAD_NUMBERS + (-31, 31),
            'macro_summary': (None, [], 1), 'market_sentiment': ('MIXED', 'neutral', [], None),
        }.items():
            for value in values:
                payload = news()
                payload[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    contracts.validate_news(payload, candidates())
        for value in (-30, 30):
            payload = news()
            payload['overall_market_adjustment'] = value
            payload['stock_signals'][0]['score_adjustment'] = value
            payload['sector_signals'][0]['score_adjustment'] = value
            for block in contracts.NEWS_SIGNAL_BLOCKS:
                payload[block]['score_adjustment'] = value
            contracts.validate_news(payload, candidates())


class DecisionTests(OfflineTests):
    def test_candidate_facts_overwrite_model_metadata_without_rewriting_narrative(self):
        pool, payload = candidates(), decision()
        original_narrative = payload['top_pick']['reasoning']
        for key in pool[0]:
            if key != 'ticker':
                payload['top_pick'][key] = 'MODEL FABRICATION'
        payload['top_pick'].update({'facts': {'rsi': 99}, 'facts_as_of': 'tomorrow',
                                    'factual_summary': 'Guaranteed return', 'fill_price': 0.01,
                                    'confidence_label': 'Probability of success',
                                    'reasoning_label': 'Verified', 'score_breakdown': 'Wrong numbers'})
        payload['watch_candidates'][0].update(sector='Fabricated', rsi=100)
        original, original_pool = copy.deepcopy(payload), copy.deepcopy(pool)
        result = contracts.validate_decision(payload, pool)
        pick = result['top_pick']
        for key, value in pool[0].items():
            self.assertEqual(pick[key], value)
        self.assertEqual(pick['reasoning'], original_narrative)
        self.assertEqual(pick['confidence'], 85.5)
        self.assertEqual(pick['confidence_label'], 'LLM score (uncalibrated)')
        self.assertEqual(pick['reasoning_label'], contracts.REASONING_LABEL)
        self.assertEqual(pick['facts_as_of'], '2026-09-10')
        self.assertEqual(pick['facts'], {key: pool[0][key] for key in ('ticker',) + contracts.FACT_FIELDS})
        self.assertIn('rsi=55.5', pick['factual_summary'])
        self.assertNotIn('MODEL', pick['factual_summary'])
        self.assertNotIn('fill_price', pick)
        self.assertNotIn('score_breakdown', pick)
        self.assertEqual(result['watch_candidates'][0]['sector'], 'Energy')
        pick['stock_news'].append('Changed')
        pick['facts']['sector'] = 'Changed'
        result['derived_rules'].append('Changed')
        self.assertEqual(payload, original)
        self.assertEqual(pool, original_pool)

    def test_missing_candidate_facts_are_not_fabricated(self):
        payload = decision()
        payload['watch_candidates'] = []
        payload['top_pick'].update(rsi=90, short_ratio=50, sector='Fake', price=1)
        result = contracts.validate_decision(payload, [{'ticker': 'ABC'}])['top_pick']
        self.assertEqual(result['facts'], {'ticker': 'ABC'})
        self.assertIsNone(result['facts_as_of'])
        for key in ('rsi', 'short_ratio', 'sector', 'price'):
            self.assertNotIn(key, result)

    def test_buy_size_is_required_positive_finite_and_not_coerced_or_floored(self):
        for size in BAD_NUMBERS + (0, -1, 100.01):
            payload = decision()
            payload['top_pick']['position_size_pct'] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                contracts.validate_decision(payload, candidates())
        payload = decision()
        del payload['top_pick']['position_size_pct']
        with self.assertRaises(ValueError):
            contracts.validate_decision(payload, candidates())
        for size in (0.01, 0.4, 1, 100):
            payload['top_pick']['position_size_pct'] = size
            self.assertEqual(contracts.validate_decision(payload, candidates())['top_pick']['position_size_pct'], size)

    def test_membership_held_tickers_and_strict_signal(self):
        for signal, ticker in (('BUY', 'GHOST'), ('BUY', 'NONE'), ('WATCH', 'NONE'),
                               ('NO PICK', 'GHOST'), ('buy', 'ABC'), ('SELL', 'ABC'),
                               ('BUY', 'abc'), ('BUY', ' ABC'), ('BUY', None)):
            payload = decision()
            payload['top_pick'].update(signal=signal, ticker=ticker)
            with self.subTest(signal=signal, ticker=ticker), self.assertRaises(ValueError):
                contracts.validate_decision(payload, candidates())
        for held in (('ABC',), ['abc'], {'ABC'}, ('XYZ',)):
            with self.subTest(held=held), self.assertRaises(ValueError):
                contracts.validate_decision(decision(), candidates(), held)
        with self.assertRaises(ValueError):
            contracts.validate_decision(decision(), candidates(), 'ABC')

    def test_no_pick_none_and_watch_require_no_buy_only_fields(self):
        for signal, ticker, pool in (('NO PICK', 'NONE', []), ('NO PICK', 'ABC', candidates()),
                                      ('WATCH', 'ABC', candidates())):
            payload = decision()
            payload['top_pick'].update(signal=signal, ticker=ticker, confidence=0)
            del payload['top_pick']['position_size_pct']
            del payload['top_pick']['devils_advocate']
            payload['watch_candidates'] = []
            result = contracts.validate_decision(payload, pool)
            self.assertEqual(result['top_pick']['signal'], signal)
            self.assertNotIn('position_size_pct', result['top_pick'])
        payload = decision()
        payload['top_pick']['signal'] = 'WATCH'
        with self.assertRaises(ValueError):
            contracts.validate_decision(payload, candidates(), ('ABC',))

    def test_required_outer_fields_and_shapes(self):
        for key in decision():
            payload = decision()
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate_decision(payload, candidates())
        for key, values in {
            'top_pick': (None, [], ''), 'watch_candidates': (None, {}, [None]),
            'derived_rules': (None, 'rule', [1], [None]), 'learning_summary': (None, [], 1),
        }.items():
            for value in values:
                payload = decision()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_decision(payload, candidates())

    def test_required_pick_and_watch_fields(self):
        for target in ('top_pick', 'watch_candidates'):
            record = decision()[target] if target == 'top_pick' else decision()[target][0]
            for key in record:
                payload = decision()
                selected = payload[target] if target == 'top_pick' else payload[target][0]
                del selected[key]
                with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                    contracts.validate_decision(payload, candidates())

    def test_confidence_strict_on_both_pick_and_watch(self):
        for target in ('top_pick', 'watch_candidates'):
            for value in BAD_NUMBERS + (-0.1, 100.1):
                payload = decision()
                record = payload[target] if target == 'top_pick' else payload[target][0]
                record['confidence'] = value
                with self.subTest(target=target, value=value), self.assertRaises(ValueError):
                    contracts.validate_decision(payload, candidates())
            for value in (0, 100):
                payload = decision()
                record = payload[target] if target == 'top_pick' else payload[target][0]
                record['confidence'] = value
                contracts.validate_decision(payload, candidates())

    def test_narrative_types_and_buy_bear_case(self):
        for field in ('reasoning', 'key_risk', 'devils_advocate'):
            for value in (None, 1, []):
                payload = decision()
                payload['top_pick'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    contracts.validate_decision(payload, candidates())
        for value in ('', '  '):
            payload = decision()
            payload['top_pick']['devils_advocate'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.validate_decision(payload, candidates())

    def test_watch_unique_known_unheld_and_watch_only(self):
        for update in ({'ticker': 'GHOST'}, {'ticker': 'ABC'}, {'signal': 'BUY'}, {'signal': 'NO PICK'}):
            payload = decision()
            payload['watch_candidates'][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                contracts.validate_decision(payload, candidates())
        payload = decision()
        payload['watch_candidates'].append(copy.deepcopy(payload['watch_candidates'][0]))
        with self.assertRaises(ValueError):
            contracts.validate_decision(payload, candidates())

    def test_invalid_candidate_facts_and_ambiguous_candidates(self):
        for key in ('rsi', 'short_ratio', 'short_pct_float'):
            for value in (True, '55', float('nan'), float('inf'), -1):
                pool = candidates()
                pool[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_decision(decision(), pool)
        for pool in ({'ABC': {}}, ['ABC'], [None], [{'ticker': 'abc'}],
                     candidates() + candidates()[:1]):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                contracts.validate_decision(decision(), pool)


class ExitTests(OfflineTests):
    def test_valid_advisory_preserved_without_execution_or_fill_generation(self):
        for action in ('HOLD', 'EXIT', 'ADD'):
            for urgency in ('LOW', 'MEDIUM', 'HIGH'):
                payload = exit_response()
                payload.update(action=action, urgency=urgency, exit_price=120.5)
                payload['extra'] = ['opaque']
                result = contracts.validate_exit(payload, 'ABC')
                self.assertEqual(result, payload)
                self.assertIsNot(result, payload)
                result['extra'].append('changed')
                self.assertEqual(payload['extra'], ['opaque'])
        payload = exit_response()
        contracts.validate_exit(payload, 'ABC')
        del payload['exit_price']
        self.assertNotIn('exit_price', contracts.validate_exit(payload, 'ABC'))

    def test_required_fields_and_malformed_exit(self):
        for field in ('ticker', 'action', 'urgency', 'reason'):
            payload = exit_response()
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                contracts.validate_exit(payload, 'ABC')
        for field, values in {
            'ticker': ('XYZ', 'abc', None, []), 'action': ('SELL', 'hold', None, []),
            'urgency': ('NOW', 'low', None, []), 'reason': ('', '  ', None, 1),
            'exit_price': tuple(v for v in BAD_NUMBERS if v is not None) + (0, -1),
        }.items():
            for value in values:
                payload = exit_response()
                payload[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    contracts.validate_exit(payload, 'ABC')


class ConfigTests(OfflineTests):
    def test_core_and_full_config_return_independent_copies(self):
        self.assertEqual(set(config()), contracts.REQUIRED_CONFIG_KEYS)
        for payload in (config(), full_config()):
            result = contracts.validate_config(payload)
            self.assertEqual(result, payload)
            self.assertIsNot(result, payload)
        payload = full_config()
        result = contracts.validate_config(payload)
        result['additional_tickers'].append('ABC')
        self.assertNotIn('ABC', payload['additional_tickers'])

    def test_required_core_keys_never_defaulted(self):
        for key in contracts.REQUIRED_CONFIG_KEYS:
            payload = full_config()
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate_config(payload)

    def test_every_numeric_config_rejects_coercion_bool_and_nonfinite(self):
        for key in contracts.CONFIG_NUMERIC_BOUNDS:
            for value in BAD_NUMBERS:
                payload = full_config()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_config(payload)

    def test_all_integer_counts_require_actual_ints(self):
        for key in contracts.CONFIG_INTEGER_KEYS:
            for value in (1.5, 2.0):
                payload = full_config()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_config(payload)

    def test_every_declared_bound_is_enforced(self):
        for key, (low, high) in contracts.CONFIG_NUMERIC_BOUNDS.items():
            for value in ([low - 1] if low is not None else []) + ([high + 1] if high is not None else []):
                payload = full_config()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_config(payload)

    def test_required_safe_boundaries_and_atr_ratio(self):
        for stop, target in ((1, 1.5), (4, 6), (4, 8)):
            payload = config()
            payload.update(ATR_STOP_MULT=stop, ATR_TARGET_MULT=target)
            contracts.validate_config(payload)
        payload = config()
        payload.update(ATR_STOP_MULT=4, ATR_TARGET_MULT=5.99)
        with self.assertRaises(ValueError):
            contracts.validate_config(payload)
        for key, values in {
            'max_positions': (1, 5), 'min_cash_floor': (500, 10000),
            'hold_days': (1, 30), 'trail_atr_mult': (1, 4),
            'BUY_THRESHOLD': (70, 88), 'WATCH_THRESHOLD': (55, 79),
            'RSI_MIN': (10, 45), 'RSI_MAX': (55, 90), 'ADX_MIN': (5, 30),
        }.items():
            for value in values:
                payload = config()
                payload[key] = value
                with self.subTest(key=key, value=value):
                    contracts.validate_config(payload)

    def test_related_thresholds_and_conflicting_volume_alias(self):
        for update in ({'BUY_THRESHOLD': 70, 'WATCH_THRESHOLD': 70},
                       {'BUY_THRESHOLD': 70, 'WATCH_THRESHOLD': 69.5},
                       {'volume_min_ratio': 2}, {'vix_low_pctile': 80, 'vix_high_pctile': 70},
                       {'dd_critical_pct': -10, 'dd_severe_pct': -20},
                       {'dd_caution_pct': -30, 'dd_severe_pct': -20}):
            payload = full_config()
            payload.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                contracts.validate_config(payload)

    def test_boolean_and_list_types_ticker_syntax_and_unknown_keys(self):
        for key in contracts.CONFIG_BOOLEAN_KEYS:
            for value in (1, 0, 'false', None, []):
                payload = full_config()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_config(payload)
        for key in ('sector_blacklist', 'sector_whitelist', 'additional_tickers'):
            for value in ('ABC', None, [1], [''], ['  '], ['ABC', 'abc']):
                payload = full_config()
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_config(payload)
        for value in ('A B', '^GSPC', '$ABC', '../ABC', 'NONE', 'A' * 16):
            payload = config()
            payload['additional_tickers'] = [value]
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.validate_config(payload)
        for field, value in (('made_up_threshold', 1), ('source_preference', 'FAKE'), ('reasoning', 1)):
            payload = config()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contracts.validate_config(payload)

    def test_atomic_failure_never_mutates_input_or_published_bounds(self):
        payload = full_config()
        payload['additional_tickers'] = [' abc ']
        payload['ATR_TARGET_MULT'] = 1.5  # individually valid, but fails final ratio check
        original = copy.deepcopy(payload)
        bounds = dict(contracts.CONFIG_NUMERIC_BOUNDS)
        with self.assertRaises(ValueError):
            contracts.validate_config(payload)
        self.assertEqual(payload, original)
        self.assertEqual(dict(contracts.CONFIG_NUMERIC_BOUNDS), bounds)
        payload['ATR_TARGET_MULT'] = 3
        result = contracts.validate_config(payload)
        self.assertEqual(result['additional_tickers'], ['ABC'])
        self.assertEqual(payload['additional_tickers'], [' abc '])
        with self.assertRaises(TypeError):
            contracts.CONFIG_NUMERIC_BOUNDS['max_positions'] = (0, 99)

    def test_self_tuning_is_public_and_disabled_without_explicit_environment_opt_in(self):
        self.assertIs(contracts.DEFAULT_SELF_TUNING_ENABLED, False)
        self.assertFalse(contracts.self_tuning_enabled({}))
        for value in ('', '0', 'false', 'off', 'enabled', '2', True, 1):
            with self.subTest(value=value):
                self.assertFalse(contracts.self_tuning_enabled({contracts.SELF_TUNING_ENV: value}))
        for value in ('1', 'true', 'YES', ' On '):
            with self.subTest(value=value):
                self.assertTrue(contracts.self_tuning_enabled({contracts.SELF_TUNING_ENV: value}))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(contracts.self_tuning_enabled())
            os.environ[contracts.SELF_TUNING_ENV] = '1'
            self.assertTrue(contracts.self_tuning_enabled())


class HealthTests(OfflineTests):
    def test_empty_run_failed_and_no_fabricated_provider(self):
        health = contracts.RunHealth()
        self.assertEqual(health.as_dict()['status'], 'failed')
        self.assertEqual(health.label(), 'No successful LLM response')
        self.assertEqual(health.as_dict()['attempts'], 0)
        health.provider('NVIDIA', 'model', False)
        self.assertEqual(health.label(), 'No successful LLM response')

    def test_actual_provider_counts_and_recovered_timeout(self):
        health = contracts.RunHealth()
        health.provider('NVIDIA', 'primary', False)
        health.stage('final_decision', False, 'Timeout')
        self.assertEqual(health.as_dict()['status'], 'failed')
        health.provider('OpenRouter', 'backup:free')
        health.provider('OpenRouter', 'backup:free', False)
        health.provider('OpenRouter', 'backup:free', True)
        health.stage('final_decision', True, 'Validated backup response')
        result = health.as_dict()
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual((result['attempts'], result['successes'], result['failures']), (4, 2, 2))
        self.assertEqual(result['providers']['NVIDIA:primary'], {'attempts': 1, 'successes': 0, 'failures': 1})
        self.assertEqual(result['providers']['OpenRouter:backup:free'], {'attempts': 3, 'successes': 2, 'failures': 1})
        self.assertEqual(result['degraded_reasons'], [])
        self.assertEqual(health.label(), 'OpenRouter:backup:free')
        health.provider('NVIDIA', 'other-model', True)
        self.assertEqual(health.label(), 'OpenRouter:backup:free, NVIDIA:other-model')
        for counts in health.as_dict()['providers'].values():
            self.assertEqual(counts['attempts'], counts['successes'] + counts['failures'])

    def test_persistent_stale_data_and_skipped_stage_reasons_are_unique(self):
        health = contracts.RunHealth()
        health.degrade(' stale market data ')
        health.degrade('stale market data')
        health.degrade('skipped catalyst batch')
        health.stage('news', False, 'unavailable')
        health.stage('news', True, 'recovered')
        health.stage('final_decision', True)
        self.assertEqual(health.as_dict()['status'], 'degraded')
        self.assertEqual(health.as_dict()['degraded_reasons'], ['stale market data', 'skipped catalyst batch'])
        self.assertEqual(health.as_dict()['stages']['news'], {'success': True, 'detail': 'recovered'})

    def test_failed_stage_status_recovers_but_critical_final_failure_wins(self):
        health = contracts.RunHealth()
        health.stage('news', False, 'Skipped')
        self.assertEqual(health.as_dict()['status'], 'degraded')
        health.stage('news', True)
        self.assertEqual(health.as_dict()['status'], 'healthy')
        for name in contracts.CRITICAL_FINAL_STAGES:
            health.stage(name, False, 'No validated decision')
            health.stage('report', True)  # later reporting success cannot mask final failure
            self.assertEqual(health.as_dict()['status'], 'failed')
            health.stage(name, True)
            self.assertEqual(health.as_dict()['status'], 'healthy')

    def test_snapshots_are_independent_serializable_and_instances_isolated(self):
        first, second = contracts.RunHealth(), contracts.RunHealth()
        first.provider('NVIDIA', 'model')
        first.stage('final_decision', True)
        first.degrade('stale data')
        result = first.as_dict()
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        result['providers']['NVIDIA:model']['attempts'] = 999
        result['stages']['final_decision']['success'] = False
        result['degraded_reasons'].clear()
        self.assertEqual(first.as_dict()['attempts'], 1)
        self.assertEqual(first.as_dict()['status'], 'degraded')
        self.assertEqual(first.as_dict()['degraded_reasons'], ['stale data'])
        self.assertEqual(second.as_dict()['attempts'], 0)

    def test_invalid_health_inputs_do_not_record_attempts_or_stages(self):
        health = contracts.RunHealth()
        for invoke in (
            lambda: health.provider('', 'model'), lambda: health.provider('provider', ''),
            lambda: health.provider('provider', 'model', 'true'),
            lambda: health.provider('ambiguous:provider', 'model'),
            lambda: health.stage('', True), lambda: health.stage('news', 1),
            lambda: health.stage('news', True, None), lambda: health.degrade(' '),
        ):
            with self.assertRaises(ValueError):
                invoke()
        self.assertEqual(health.as_dict()['attempts'], 0)
        self.assertEqual(health.as_dict()['stages'], {})


class IntegrationTests(OfflineTests):
    def test_public_signatures(self):
        expected = {
            'parse_object': ('raw',), 'validate_catalysts': ('payload', 'candidates'),
            'validate_news': ('payload', 'candidates'),
            'validate_decision': ('payload', 'candidates', 'held_tickers'),
            'validate_exit': ('payload', 'ticker'), 'validate_config': ('cfg',),
        }
        for name, parameters in expected.items():
            self.assertEqual(tuple(inspect.signature(getattr(contracts, name)).parameters), parameters)
        health = contracts.RunHealth()
        self.assertEqual(tuple(inspect.signature(health.provider).parameters), ('provider', 'model', 'success'))
        self.assertEqual(tuple(inspect.signature(health.stage).parameters), ('name', 'success', 'detail'))
        self.assertEqual(inspect.signature(contracts.validate_decision).parameters['held_tickers'].default, ())

    def test_complete_json_to_validated_decision(self):
        encoded = '<think>Considering risks</think>```json\n' + json.dumps(decision()) + '\n```'
        result = contracts.validate_decision(contracts.parse_object(encoded), candidates())
        self.assertEqual(result['top_pick']['ticker'], 'ABC')
        json.dumps(result, allow_nan=False)

    def test_validators_reject_nonobjects(self):
        for payload in (None, [], 'object', 1):
            for validate, args in (
                (contracts.validate_catalysts, (candidates(),)),
                (contracts.validate_news, (candidates(),)),
                (contracts.validate_decision, (candidates(),)),
                (contracts.validate_exit, ('ABC',)), (contracts.validate_config, ()),
            ):
                with self.subTest(validator=validate.__name__, payload=payload), self.assertRaises(ValueError):
                    validate(payload, *args)

    def test_main_schema_parity_by_ast_read_only(self):
        main = ROOT / 'LLM_Portfolio_Manager.py'
        if not main.exists():
            self.skipTest('Optional read-only main schema parity check: staged main absent')
        source = main.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        update = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'update_config_from_llm')
        current = next(node.value for node in ast.walk(update) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'current_cfg'
                               for target in node.targets))
        main_keys = {key.value for key in current.keys if isinstance(key, ast.Constant)}
        self.assertEqual(set(full_config()), main_keys | {'brokerage_fee', 'reasoning'})
        for kind in contracts.CATALYST_TYPES:
            self.assertIn(kind, source)

    def test_python_311_grammar_and_existing_finite_number_dependency(self):
        for path in (Path(contracts.__file__), Path(__file__)):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 11))
        from screener_safety import finite_number
        self.assertIs(contracts.finite_number, finite_number)

    def test_clean_import_and_validation_do_not_write_or_import_main(self):
        code = '''
import os, sys
sys.dont_write_bytecode = True
def audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            raise AssertionError('file writes forbidden')
    if event in {'socket.connect', 'socket.connect_ex', 'socket.bind',
                 'socket.sendto', 'socket.sendmsg', 'socket.getaddrinfo',
                 'socket.gethostbyname', 'socket.gethostbyaddr'}:
        raise AssertionError('network forbidden')
sys.addaudithook(audit)
import screener_contracts as c
assert 'LLM_Portfolio_Manager' not in sys.modules
assert c.parse_object('{"ok":true}') == {'ok': True}
assert c.RunHealth().as_dict()['status'] == 'failed'
'''
        result = subprocess.run([sys.executable, '-B', '-W', 'error', '-c', code],
                                cwd=ROOT, text=True, capture_output=True, check=False,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()