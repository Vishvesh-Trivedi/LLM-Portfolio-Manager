"""Fail-closed LLM contracts, configuration validation and in-memory run health.

Python 3.11+. No main import, file writes, network calls, global configuration
mutation, or order execution. Schema errors raise ValueError. Validators return
independent dictionaries; a caller must validate an entire proposed configuration
before saving it or applying any globals. LLM narratives are NOT verified facts.
"""

import copy
import json
import os
import re
from types import MappingProxyType

from screener_safety import finite_number


CATALYST_TYPES = frozenset({
    'EARNINGS_CATALYST', 'UPGRADE', 'BREAKOUT', 'SECTOR_ROTATION',
    'MOMENTUM', 'NEWS_HYPE', 'NONE',
})
CONFIDENCE_LABEL = 'LLM score (uncalibrated)'
REASONING_LABEL = 'LLM narrative (not verified facts)'
DEFAULT_SELF_TUNING_ENABLED = False
SELF_TUNING_ENV = 'SCREENER_ENABLE_SELF_TUNING'
CRITICAL_FINAL_STAGES = frozenset({'final', 'decision', 'final_decision', 'round3'})
NEWS_SIGNAL_BLOCKS = (
    'trump_signal', 'fed_signal', 'macro_data_signal', 'geopolitical_signal',
)
FACT_FIELDS = ('sector', 'source', 'rsi', 'short_ratio', 'short_pct_float')
REQUIRED_CONFIG_KEYS = frozenset({
    'RSI_MIN', 'RSI_MAX', 'ADX_MIN', 'BUY_THRESHOLD', 'WATCH_THRESHOLD',
    'VOLUME_MIN_RATIO', 'ATR_STOP_MULT', 'ATR_TARGET_MULT',
    'max_positions', 'min_cash_floor', 'hold_days', 'trail_atr_mult',
})
# None means no bound beyond finiteness. Do not clamp or silently coerce a value.
CONFIG_NUMERIC_BOUNDS = MappingProxyType({
    'RSI_MIN': (10, 45), 'RSI_MAX': (55, 90), 'ADX_MIN': (5, 30),
    'BUY_THRESHOLD': (70, 88), 'WATCH_THRESHOLD': (55, 87),
    'VOLUME_MIN_RATIO': (0, None), 'ATR_STOP_MULT': (1, 4),
    'ATR_TARGET_MULT': (1.5, 8), 'max_positions': (1, 5),
    'min_cash_floor': (500, 10000), 'hold_days': (1, 30),
    'trail_atr_mult': (1, 4), 'min_catalyst_score': (0, 100),
    'min_adx_buy': (5, 30), 'max_vix': (0, None), 'min_price': (0, None),
    'min_dollar_volume_m': (1, 40), 'sector_conc_max': (1, None),
    'sample_size': (1, None), 'brokerage_fee': (0, None),
    'dd_caution_pct': (-100, 0), 'dd_severe_pct': (-100, 0),
    'dd_critical_pct': (-100, 0), 'win_threshold_pct': (0, None),
    'loss_threshold_pct': (None, 0), 'min_picks_to_learn': (1, None),
    'rsi_hard_cap': (0, 999), 'rsi_cap_conf': (0, 100),
    'upside_hard_cap': (None, None), 'upside_cap_conf': (0, 100),
    'vix_low_pctile': (0, 100), 'vix_high_pctile': (0, 100),
    'sector_conc_lookback': (1, None), 'sector_conc_penalty': (0, 100),
    'congress_days': (1, None), 'sec_8k_days': (1, None),
    'rsi_exit': (0, 100), 'rsi_exit_min_profit': (None, None),
    'macd_exit_min_profit': (None, None), 'entry_slippage_pct': (0, None),
    'final_candidates': (1, None), 'pre_earnings_exit_days': (0, None),
    'squeeze_float_pct': (0, None), 'squeeze_days_to_cover': (0, None),
    'volume_min_ratio': (0, None),
})
CONFIG_INTEGER_KEYS = frozenset({
    'max_positions', 'hold_days', 'sector_conc_max', 'sample_size',
    'min_picks_to_learn', 'sector_conc_lookback', 'congress_days',
    'sec_8k_days', 'final_candidates', 'pre_earnings_exit_days',
})
CONFIG_BOOLEAN_KEYS = frozenset({
    'require_congress', 'avoid_earnings_week', 'only_profitable', 'require_above_ma',
})
_CONFIG_LIST_KEYS = frozenset({
    'sector_blacklist', 'sector_whitelist', 'additional_tickers',
})
_TICKER = re.compile(r'[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)*\Z')
_MODEL_PICK_FIELDS = frozenset({
    'ticker', 'signal', 'confidence', 'position_size_pct',
    'reasoning', 'key_risk', 'devils_advocate',
})
_RESERVED_PICK_FIELDS = _MODEL_PICK_FIELDS | frozenset({
    'facts', 'facts_as_of', 'factual_summary', 'confidence_label',
    'reasoning_label', 'score_breakdown',
})


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a dict')
    return value


def _required(record, key):
    if key not in record:
        raise ValueError(f'missing required field: {key}')
    return record[key]


def _text(value, name, nonempty=False, maximum=None):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a string')
    if nonempty and not value.strip():
        raise ValueError(f'{name} must be nonempty')
    if maximum is not None and len(value) > maximum:
        raise ValueError(f'{name} must be at most {maximum} characters')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be a bool')
    return value


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    return value


def _enum(value, name, allowed, lowercase=False):
    value = _text(value, name)
    if lowercase:
        value = value.strip().lower()
    if value not in allowed:
        raise ValueError(f'{name} has an invalid value')
    return value


def _ticker(value):
    value = _text(value, 'ticker', nonempty=True)
    if len(value) > 15 or _TICKER.fullmatch(value) is None or value == 'NONE':
        raise ValueError('invalid ticker')
    return value


def _candidate_index(candidates):
    """Candidates are a list/tuple of verified records, not model output."""
    if not isinstance(candidates, (list, tuple)):
        raise ValueError('candidates must be a list or tuple of dicts')
    result = {}
    for candidate in candidates:
        _object(candidate, 'candidate')
        ticker = _ticker(_required(candidate, 'ticker'))
        if ticker in result:
            raise ValueError('duplicate candidate ticker')
        result[ticker] = candidate
    return result


def _member(value, allowed, name):
    value = _text(value, name, nonempty=True)
    if value not in allowed:
        raise ValueError(f'unknown {name}: {value}')
    return value


def _unique(value, seen, name):
    if value in seen:
        raise ValueError(f'duplicate {name}: {value}')
    seen.add(value)


def _reject_constant(value):
    raise ValueError(f'non-JSON numeric constant: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_object(raw):
    """Decode ONLY the first object, never repair it or salvage nested objects.

    Complete leading <think> blocks and a paired ```json (or ```) wrapper are
    permitted. Natural prose may surround the object, but JSON container
    delimiters outside it are rejected: arrays, quoted objects, extra objects,
    unmatched braces, and truncated outer objects must not become decisions.
    """
    text = _text(raw, 'raw', nonempty=True).strip()
    while text.startswith('<think>'):
        end = text.find('</think>', len('<think>'))
        if end < 0:
            raise ValueError('incomplete reasoning prefix')
        text = text[end + len('</think>'):].lstrip()
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object')
    prefix = text[:start]
    fenced = '```' in prefix
    if fenced:
        match = re.fullmatch(r'(.*?)```(?:json)?\s*', prefix, re.DOTALL | re.IGNORECASE)
        if match is None:
            raise ValueError('invalid JSON fence')
        prefix = match.group(1)
    if (any(char in prefix for char in '[]}`') or '<think>' in prefix
            or prefix.lstrip().startswith('"')):
        raise ValueError('non-object JSON or malformed prefix')
    if re.fullmatch(r'\s*(?:null|true|false|NaN|[-+]?Infinity|[-+]?\d[\d.eE+-]*)\s*', prefix):
        raise ValueError('non-object JSON prefix')
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        parse_float=lambda value: finite_number(float(value), 'JSON number'),
    )
    value, end = decoder.raw_decode(text, start)
    _object(value, 'JSON result')
    suffix = text[end:].strip()
    if fenced:
        if not suffix.startswith('```'):
            raise ValueError('missing closing JSON fence')
        suffix = suffix[3:].strip()
    if any(char in suffix for char in '{}[]`'):
        raise ValueError('extra JSON or malformed suffix')
    return value


def validate_catalysts(payload, candidates):
    """Require exactly one complete rating per candidate; preserve extras."""
    _object(payload, 'catalysts')
    index = _candidate_index(candidates)
    ratings = _list(_required(payload, 'ratings'), 'ratings')
    seen = set()
    for rating in ratings:
        _object(rating, 'rating')
        ticker = _member(_required(rating, 'ticker'), index, 'ticker')
        _unique(ticker, seen, 'rating ticker')
        finite_number(_required(rating, 'catalyst_score'), 'catalyst_score', 1, 10)
        _enum(_required(rating, 'catalyst_type'), 'catalyst_type', CATALYST_TYPES)
        _boolean(_required(rating, 'auto_drop'), 'auto_drop')
        _text(_required(rating, 'reason'), 'reason', nonempty=True, maximum=600)
    if seen != set(index):
        raise ValueError('ratings must cover every candidate exactly once')
    return copy.deepcopy(payload)


def validate_news(payload, candidates):
    """Return normalized known fields only; missing required data is not neutral."""
    _object(payload, 'news')
    index = _candidate_index(candidates)
    sectors = {
        c['sector'] for c in index.values()
        if isinstance(c.get('sector'), str) and c['sector'].strip()
        and c['sector'].casefold() not in {'unknown', 'n/a'}
    }
    result = {
        'macro_summary': _text(_required(payload, 'macro_summary'), 'macro_summary'),
        'market_sentiment': _enum(_required(payload, 'market_sentiment'),
                                  'market_sentiment', {'NEUTRAL', 'BULLISH', 'BEARISH'}),
        'overall_market_adjustment': finite_number(
            _required(payload, 'overall_market_adjustment'),
            'overall_market_adjustment', -30, 30),
    }
    for key, member_key, members in (
        ('stock_signals', 'ticker', index), ('sector_signals', 'sector', sectors),
    ):
        signals = _list(_required(payload, key), key)
        result[key] = []
        seen = set()
        for signal in signals:
            _object(signal, key)
            member = _member(_required(signal, member_key), members, member_key)
            _unique(member, seen, member_key)
            normalized = {
                member_key: member,
                'news': _text(_required(signal, 'news'), 'news'),
                'score_adjustment': finite_number(_required(signal, 'score_adjustment'),
                                                  'score_adjustment', -30, 30),
            }
            if member_key == 'ticker':
                normalized['auto_drop'] = _boolean(_required(signal, 'auto_drop'), 'auto_drop')
            result[key].append(normalized)
    for key in NEWS_SIGNAL_BLOCKS:
        signal = _object(_required(payload, key), key)
        normalized = {
            'detected': _boolean(_required(signal, 'detected'), 'detected'),
            'score_adjustment': finite_number(_required(signal, 'score_adjustment'),
                                              'score_adjustment', -30, 30),
        }
        if 'affected_sectors' in signal:
            normalized['affected_sectors'] = [
                _member(sector, sectors, 'sector') for sector in
                _list(signal['affected_sectors'], 'affected_sectors')
            ]
        if 'detail' in signal:
            normalized['detail'] = _text(signal['detail'], 'detail')
        if key == 'fed_signal':
            normalized['tone'] = _enum(_required(signal, 'tone'), 'tone',
                                        {'neutral', 'hawkish', 'dovish'}, lowercase=True)
        result[key] = normalized
    return copy.deepcopy(result)


def _decision_record(record, index, held, watch=False):
    _object(record, 'pick')
    signal = _enum(_required(record, 'signal'), 'signal',
                   {'WATCH'} if watch else {'BUY', 'WATCH', 'NO PICK'})
    ticker = _text(_required(record, 'ticker'), 'ticker', nonempty=True)
    if not (signal == 'NO PICK' and ticker == 'NONE'):
        _member(ticker, index, 'ticker')
    if signal in {'BUY', 'WATCH'} and ticker in held:
        raise ValueError('pick/watch ticker is already held')
    finite_number(_required(record, 'confidence'), 'confidence', 0, 100)
    _text(_required(record, 'reasoning'), 'reasoning')
    _text(_required(record, 'key_risk'), 'key_risk')
    if signal == 'BUY' or 'position_size_pct' in record:
        size = finite_number(_required(record, 'position_size_pct'), 'position_size_pct', 0, 100)
        if signal == 'BUY' and size == 0:
            raise ValueError('BUY position_size_pct must be positive')
    if signal == 'BUY' or 'devils_advocate' in record:
        _text(_required(record, 'devils_advocate'), 'devils_advocate', nonempty=signal == 'BUY')

    # Do not let model-provided metadata, facts, prices, or scores masquerade as
    # candidate observations. Missing candidate facts stay missing (never defaults).
    result = {key: copy.deepcopy(record[key]) for key in _MODEL_PICK_FIELDS if key in record}
    candidate = index.get(ticker, {})
    result.update({key: copy.deepcopy(value) for key, value in candidate.items()
                   if key not in _RESERVED_PICK_FIELDS})
    facts = {'ticker': ticker}
    for key in FACT_FIELDS:
        if key not in candidate:
            continue
        value = candidate[key]
        if value is not None:
            if key in {'sector', 'source'}:
                _text(value, key)
            else:
                finite_number(value, key, 0, 100 if key == 'rsi' else None)
        facts[key] = copy.deepcopy(value)
    result['facts'] = facts
    result['facts_as_of'] = copy.deepcopy(candidate.get('quote_date'))
    result['confidence_label'] = CONFIDENCE_LABEL
    result['reasoning_label'] = REASONING_LABEL
    # Separate display field: never append to, rewrite, or certify novel prose.
    result['factual_summary'] = '; '.join(
        f'{key}={value}' for key, value in facts.items() if value is not None
    ) if candidate else ''
    return result


def validate_decision(payload, candidates, held_tickers=()):
    """Validate picks, bind candidate facts, and label uncalibrated LLM scores.

    WATCH/BUY must be known and unheld. Watch tickers cannot duplicate an active
    top pick. Only NO PICK may use NONE. Unknown provider fields are discarded.
    Narratives remain verbatim; factual_summary/facts are separate observations.
    """
    _object(payload, 'decision')
    index = _candidate_index(candidates)
    if not isinstance(held_tickers, (list, tuple, set, frozenset)):
        raise ValueError('held_tickers must be a collection of tickers')
    held = {_ticker(_text(t, 'held ticker').strip().upper()) for t in held_tickers}
    rules = _list(_required(payload, 'derived_rules'), 'derived_rules')
    for rule in rules:
        _text(rule, 'derived rule')
    summary = _text(_required(payload, 'learning_summary'), 'learning_summary')
    pick = _decision_record(_required(payload, 'top_pick'), index, held)
    watches = _list(_required(payload, 'watch_candidates'), 'watch_candidates')
    seen = {pick['ticker']} if pick['signal'] != 'NO PICK' else set()
    normalized_watches = []
    for watch in watches:
        normalized = _decision_record(watch, index, held, watch=True)
        _unique(normalized['ticker'], seen, 'watch ticker')
        normalized_watches.append(normalized)
    return {
        'top_pick': pick, 'watch_candidates': normalized_watches,
        'derived_rules': copy.deepcopy(rules), 'learning_summary': summary,
        'confidence_label': CONFIDENCE_LABEL, 'reasoning_label': REASONING_LABEL,
    }


def validate_exit(payload, ticker):
    """Validate an advisory exit response; exit_price is NEVER an executable fill.

    An absent/null exit_price is permitted. Any suggested price must be positive
    and finite. The parent must use verified market prices for actual execution.
    """
    _object(payload, 'exit')
    _ticker(ticker)
    if _required(payload, 'ticker') != ticker:
        raise ValueError('exit ticker must match exactly')
    _enum(_required(payload, 'action'), 'action', {'HOLD', 'EXIT', 'ADD'})
    _enum(_required(payload, 'urgency'), 'urgency', {'HIGH', 'MEDIUM', 'LOW'})
    _text(_required(payload, 'reason'), 'reason', nonempty=True)
    if payload.get('exit_price') is not None:
        price = finite_number(payload['exit_price'], 'exit_price', 0)
        if price == 0:
            raise ValueError('exit_price must be positive')
    return copy.deepcopy(payload)


def validate_config(cfg):
    """Validate a complete core plus optional main-save keys, without side effects.

    Required keys and numeric bounds are exported. Unknown keys are rejected;
    optional keys are validated when present, never filled from mutable globals.
    Integer counts, booleans and numeric thresholds do not accept coercion.
    """
    _object(cfg, 'config')
    missing = REQUIRED_CONFIG_KEYS.difference(cfg)
    if missing:
        raise ValueError('missing required config keys: ' + ', '.join(sorted(missing)))
    allowed = (set(CONFIG_NUMERIC_BOUNDS) | CONFIG_BOOLEAN_KEYS | _CONFIG_LIST_KEYS
               | {'source_preference', 'reasoning'})
    if any(key not in allowed for key in cfg):
        raise ValueError('unknown config key')
    result = copy.deepcopy(cfg)
    for key, value in cfg.items():
        if key in CONFIG_NUMERIC_BOUNDS:
            low, high = CONFIG_NUMERIC_BOUNDS[key]
            finite_number(value, key, low, high)
            if key in CONFIG_INTEGER_KEYS and type(value) is not int:
                raise ValueError(f'{key} must be an integer')
        elif key in CONFIG_BOOLEAN_KEYS:
            _boolean(value, key)
        elif key in _CONFIG_LIST_KEYS:
            values = _list(value, key)
            normalized, seen = [], set()
            for item in values:
                item = _text(item, key, nonempty=True).strip()
                if key == 'additional_tickers':
                    item = _ticker(item.upper())
                _unique(item.casefold(), seen, key)
                normalized.append(item)
            result[key] = normalized
        elif key == 'source_preference':
            result[key] = _enum(value, key, {'ANY', 'TECHNICAL', 'NEWS', 'BOTH'})
        else:
            _text(value, key)
    if cfg['WATCH_THRESHOLD'] > cfg['BUY_THRESHOLD'] - 1:
        raise ValueError('WATCH_THRESHOLD must be at least one below BUY_THRESHOLD')
    if cfg['ATR_TARGET_MULT'] < 1.5 * cfg['ATR_STOP_MULT']:
        raise ValueError('ATR reward/risk must be >= 1.5')
    if 'volume_min_ratio' in cfg and cfg['volume_min_ratio'] != cfg['VOLUME_MIN_RATIO']:
        raise ValueError('volume_min_ratio conflicts with VOLUME_MIN_RATIO')
    for low, high in (
        ('vix_low_pctile', 'vix_high_pctile'),
        ('dd_critical_pct', 'dd_severe_pct'), ('dd_severe_pct', 'dd_caution_pct'),
        ('dd_critical_pct', 'dd_caution_pct'),
    ):
        if low in cfg and high in cfg and cfg[low] >= cfg[high]:
            raise ValueError(f'{low} must be below {high}')
    return result


def self_tuning_enabled(environ=None):
    """Explicit opt-in only; absent, malformed or unrecognized env values disable."""
    environment = os.environ if environ is None else environ
    value = environment.get(SELF_TUNING_ENV)
    if value is None:
        return DEFAULT_SELF_TUNING_ENABLED
    return isinstance(value, str) and value.strip().lower() in {'1', 'true', 'yes', 'on'}


class RunHealth:
    """In-memory accounting; parent serializes as_dict() and renders label().

    provider() records one actual attempt, not a configured/intended provider.
    stage() replaces the latest outcome so successful retries recover a stage.
    degrade() is persistent and deduplicated (e.g. stale data/skipped batches).
    No stages, or a failed CRITICAL_FINAL_STAGES entry, means failed. Other
    unfinished failures/reasons mean degraded. Provider failures alone do NOT.
    """

    def __init__(self):
        self._providers = {}
        self._stages = {}
        self._reasons = []

    def provider(self, provider, model, success=True):
        provider = _text(provider, 'provider', nonempty=True).strip()
        model = _text(model, 'model', nonempty=True).strip()
        _boolean(success, 'success')
        if ':' in provider:
            raise ValueError('provider must not contain a colon')
        key = provider + ':' + model
        counts = self._providers.setdefault(key, {'attempts': 0, 'successes': 0, 'failures': 0})
        counts['attempts'] += 1
        counts['successes' if success else 'failures'] += 1

    def stage(self, name, success, detail=''):
        name = _text(name, 'stage name', nonempty=True).strip()
        _boolean(success, 'success')
        _text(detail, 'detail')
        self._stages[name] = {'success': success, 'detail': detail}

    def degrade(self, reason):
        reason = _text(reason, 'degradation reason', nonempty=True).strip()
        if reason not in self._reasons:
            self._reasons.append(reason)

    def as_dict(self):
        failed = {name for name, stage in self._stages.items() if not stage['success']}
        if not self._stages or any(name.casefold() in CRITICAL_FINAL_STAGES for name in failed):
            status = 'failed'
        elif failed or self._reasons:
            status = 'degraded'
        else:
            status = 'healthy'
        result = {
            'status': status, 'providers': self._providers, 'stages': self._stages,
            'degraded_reasons': self._reasons, 'label': self.label(),
        }
        for key in ('attempts', 'successes', 'failures'):
            result[key] = sum(counts[key] for counts in self._providers.values())
        return copy.deepcopy(result)

    def label(self):
        return ', '.join(key for key, counts in self._providers.items() if counts['successes']) or 'No successful LLM response'


__all__ = [
    'parse_object', 'validate_catalysts', 'validate_news', 'validate_decision',
    'validate_exit', 'validate_config', 'self_tuning_enabled', 'RunHealth',
    'CATALYST_TYPES', 'CONFIDENCE_LABEL', 'REASONING_LABEL', 'FACT_FIELDS',
    'NEWS_SIGNAL_BLOCKS', 'REQUIRED_CONFIG_KEYS', 'CONFIG_NUMERIC_BOUNDS',
    'CONFIG_INTEGER_KEYS', 'CONFIG_BOOLEAN_KEYS', 'CRITICAL_FINAL_STAGES',
    'DEFAULT_SELF_TUNING_ENABLED', 'SELF_TUNING_ENV',
]