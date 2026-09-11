"""Non-mutating ledger checks and isolated live-provider contract QA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_state(path):
    from screener_portfolio import load_portfolio

    path = Path(path).resolve(strict=True)
    before = digest(path)
    portfolio = load_portfolio(SimpleNamespace(PORTFOLIO_JSON=path))
    assert digest(path) == before, 'Read-only ledger validation modified the source'
    print(f'Ledger valid: {len(portfolio["positions"])} open, '
          f'{len(portfolio["closed_trades"])} closed; cash ${portfolio["cash"]:.2f}')
    return portfolio


def live_qa():
    root = Path(__file__).resolve().parent
    source = root / 'StockScreener'
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    output = Path(os.environ.get('SCREENER_OUTPUT_DIR') or tempfile.mkdtemp(prefix='screener-live-qa-')).resolve()
    if output == source.resolve() or source.resolve() in output.parents:
        raise ValueError('Live QA output must be outside the canonical portfolio directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(SCREENER_SKIP_UNIVERSE_FETCH='1', SCREENER_DISABLE_ALERTS='1',
                      SCREENER_OUTPUT_DIR=str(output))
    import LLM_Portfolio_Manager as app
    from screener_contracts import validate_decision

    result = None
    app._RUN_MODE = 'isolated_live_qa'
    try:
        ledger = check_state(source / 'portfolio.json')
        shutil.copyfile(source / 'portfolio.json', output / 'portfolio.json')
        restored = app.load_portfolio()
        app.save_portfolio(restored)
        history1 = app.load_performance_history('unused.csv')
        app.save_portfolio(app.load_portfolio())
        history2 = app.load_performance_history('unused.csv')
        assert len(history1) == len(ledger['closed_trades']) == len(history2)
        assert [h['trade_id'] for h in history1] == [h['trade_id'] for h in history2]
        app._HEALTH.stage('ledger_restart', True, f'{len(history2)} closed trades survive restart')

        if not app.NVIDIA_API_KEY:
            raise RuntimeError('NVIDIA_API_KEY is required for live primary QA')
        app._reconcile_models_with_catalog()
        raw = app.call_llm(
            'This is a connectivity test. Return only the JSON object {"ok":true}.',
            'Return {"ok":true} exactly.', max_tokens=64, max_attempts=2,
            read_timeout=45, allow_fallback=False,
        )
        assert app.parse_object(raw) == {'ok': True}, 'Primary response did not meet smoke contract'
        app._HEALTH.stage('nvidia_primary', True, 'Validated JSON from NVIDIA, fallback disabled')

        ctx = {'vix_level': 18.0, 'vix_percentile': 50.0, 'vix_regime': 'MODERATE',
               'vix_multiplier': 1.0, 'qqq_trend': 'BULLISH', 'qqq_vs_ma50': 1.0,
               'spy_return_today': 0.5, 'qqq_price': 100.0, 'defensive_mode': False}
        candidates = [dict(ticker=t, sector='Technology', source='TECHNICAL',
                           rsi=55.0, adx=25.0, momentum_5d=2.0, tech_score=45,
                           news_score=20, pre_score=65, price=100.0, atr=2.0,
                           earnings_days_away=40, quote_date=app._session_date(),
                           short_ratio=2.0, short_pct_float=1.0,
                           analyst_rating=None, upside_pct=None)
                      for t in ('AAPL', 'MSFT', 'NVDA', 'AMD', 'ADBE', 'ORCL')]
        headlines = ['SYNTHETIC QA DATA: no verified company news is supplied. Do not invent a catalyst.']
        news = {c['ticker']: headlines for c in candidates}
        rated = app.batch_catalyst_score(candidates, ctx, news)
        assert all(c.get('catalyst_verified') for c in rated), 'Catalyst coverage incomplete'
        nd = app.get_news_intelligence(candidates, ctx, headlines, {}, news)
        assert not nd.get('_unavailable'), 'News schema validation failed'
        result = app._llm_json_with_fallback(
            'This is synthetic QA, not an investment recommendation. Return exactly the requested complete JSON.',
            'Return {"derived_rules":[],"learning_summary":"Synthetic QA only",'
            '"top_pick":{"ticker":"NONE","confidence":0,"signal":"NO PICK",'
            '"reasoning":"No verified catalyst in this synthetic fixture",'
            '"key_risk":"Synthetic data"},"watch_candidates":[]}',
            max_tokens=600, max_attempts=1, read_timeout=60,
            validator=lambda p: validate_decision(p, candidates), stage='final',
        )
        app.save_html_report(result, ctx, nd, 'N/A', [], portfolio=restored,
                             position_opened=False)
        app._HEALTH.stage('report', True, 'Rendered validated synthetic NO PICK')
        health = app._HEALTH.as_dict()
        assert health['status'] == 'healthy', 'Provider/schema QA is degraded'
        print('LIVE_QA_PASS: primary NVIDIA, catalyst coverage, news, final schema, report, and ledger restart')
    except Exception as exc:
        app._HEALTH.stage('final', False, 'QA failed: ' + type(exc).__name__)
        raise
    finally:
        after = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
        unchanged = before == after
        app._HEALTH.stage('canonical_state_unchanged', unchanged,
                          'All canonical output file hashes checked')
        app.write_run_health(result)
        if not unchanged:
            raise RuntimeError('QA modified canonical portfolio outputs')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-state')
    group.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.check_state:
        check_state(args.check_state)
    else:
        live_qa()


if __name__ == '__main__':
    main()