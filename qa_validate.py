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


# -- Preflight -------------------------------------------------------------
# Answers one question before real money is involved: will the next run work,
# and what exactly will it do to the ledger? Entirely read-only: no orders, no
# ledger writes, no messages sent.

OK, WARN, BAD = 'OK', 'WARN', 'FAIL'


def _row(results, status, name, detail):
    results.append((status, name, detail))


def _check_ledger(results, root):
    try:
        portfolio = check_state(root / 'StockScreener' / 'portfolio.json')
        _row(results, OK, 'Ledger file',
             f'{len(portfolio["positions"])} open, {len(portfolio["closed_trades"])} closed, '
             f'{len(portfolio.get("pending_orders", []))} pending, cash ${portfolio["cash"]:,.2f}')
        return portfolio
    except Exception as exc:
        _row(results, BAD, 'Ledger file', f'{type(exc).__name__}: {exc}')
        return None


def _check_alpaca(results, app, portfolio):
    alpaca = app._alpaca
    if not alpaca.data_enabled():
        _row(results, BAD, 'Alpaca keys', 'ALPACA_API_KEY / ALPACA_SECRET_KEY not set')
        return
    _row(results, OK, 'Alpaca keys', 'present')

    account = alpaca.fetch_account()
    if not account:
        _row(results, BAD, 'Alpaca account', 'could not be read - check keys and network')
        return
    paper = os.environ.get('ALPACA_PAPER', '1').strip() != '0'
    try:
        cash = float(account.get('cash', 0) or 0)
        equity = float(account.get('equity', 0) or 0)
    except (TypeError, ValueError):
        cash = equity = 0.0
    _row(results, OK if paper else WARN, 'Alpaca account',
         f'{"PAPER" if paper else "*** LIVE MONEY ***"} - cash ${cash:,.2f}, equity ${equity:,.2f}')
    if account.get('trading_blocked') or account.get('account_blocked'):
        _row(results, BAD, 'Alpaca status', 'account is BLOCKED - no trade can go through')

    if not alpaca.trading_enabled():
        _row(results, BAD, 'Live broker mode',
             'SCREENER_LIVE_BROKER is not 1 - the run will NOT read your Alpaca '
             'positions, and may sell holdings it has lost track of')
        return
    _row(results, OK, 'Live broker mode', 'ON - Alpaca is the source of truth')

    if portfolio is None:
        return
    snapshot = alpaca.broker_snapshot()
    if not snapshot.get('ok'):
        _row(results, BAD, 'Ledger vs Alpaca', f'unreadable: {snapshot.get("error")}')
        return
    from screener_broker_sync import plan_broker_sync
    plan = plan_broker_sync(portfolio, snapshot, app._session_date())
    changes = [a for a in plan['actions'] if a['op'] != 'set_cash']
    if not changes:
        _row(results, OK, 'Ledger vs Alpaca', 'in sync - the next run changes nothing')
    else:
        detail = '; '.join(
            a['op'].replace('_', ' ') + ' ' + str(a.get('symbol', ''))
            + (f' {a.get("shares")} @ ${a["price"]:,.2f}' if a.get('price') else '')
            for a in changes)
        _row(results, WARN, 'Ledger vs Alpaca', 'next run will: ' + detail)
    for reason in plan['blocked']:
        _row(results, BAD, 'Broker conflict', reason)


def _check_llm(results, app):
    nvidia = bool(app.NVIDIA_API_KEY)
    router = bool(app.OPENROUTER_API_KEY)
    if not nvidia and not router:
        _row(results, BAD, 'LLM providers',
             'neither NVIDIA_API_KEY nor OPENROUTER_API_KEY is set - no pick can be made')
        return
    _row(results, OK if nvidia else WARN, 'NVIDIA key',
         'present' if nvidia else 'missing - running on the backup only')
    _row(results, OK if router else WARN, 'OpenRouter key',
         'present - failover available' if router else
         'missing - an NVIDIA outage or throttle means no trade that day')
    if nvidia:
        try:
            served = app._fetch_served_models()
            chat = [m for m in served if app._is_chat_model(m)] if served else []
            _row(results, OK if chat else WARN, 'NVIDIA models',
                 f'{len(chat)} chat model(s) reachable' if chat else
                 'catalog unreadable - the run falls back to its built-in list')
        except Exception as exc:
            _row(results, WARN, 'NVIDIA models', f'probe failed: {type(exc).__name__}')


def _check_alerts(results, app):
    discord = app._discord
    if discord.enabled():
        _row(results, OK, 'Discord alerts',
             'channel ' + os.environ.get('DISCORD_CHANNEL_ID', '?')
             + (' [TEST MODE - every message marked as a test]' if discord.test_mode() else ''))
    elif os.environ.get('SCREENER_DISABLE_ALERTS') == '1':
        _row(results, WARN, 'Discord alerts', 'silenced by SCREENER_DISABLE_ALERTS=1')
    else:
        _row(results, WARN, 'Discord alerts',
             'not configured - you will not be told what executed')
    if app.WHATSAPP_PHONE and app.CALLMEBOT_API_KEY:
        _row(results, OK, 'WhatsApp alerts', 'configured')


def preflight():
    """Report whether the next live run will work, and what it will do."""
    import contextlib
    import io as _io

    root = Path(__file__).resolve().parent
    output = Path(os.environ.get('SCREENER_OUTPUT_DIR')
                  or tempfile.mkdtemp(prefix='screener-preflight-')).resolve()
    if output == (root / 'StockScreener').resolve():
        raise ValueError('Preflight must not write into the canonical portfolio directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(SCREENER_SKIP_UNIVERSE_FETCH='1', SCREENER_OUTPUT_DIR=str(output))

    with contextlib.redirect_stdout(_io.StringIO()):
        import LLM_Portfolio_Manager as app

    results = []
    portfolio = _check_ledger(results, root)
    _check_alpaca(results, app, portfolio)
    _check_llm(results, app)
    _check_alerts(results, app)

    print()
    print('PREFLIGHT - LLM Portfolio Manager')
    print('=' * 74)
    for status, name, detail in results:
        print(f'  [{status:4}] {name:<20} {detail}')
    print('=' * 74)
    failures = [r for r in results if r[0] == BAD]
    warnings = [r for r in results if r[0] == WARN]
    if failures:
        print(f'  VERDICT: NOT READY - {len(failures)} blocking issue(s) above.')
    elif warnings:
        print(f'  VERDICT: READY, with {len(warnings)} thing(s) worth a look.')
    else:
        print('  VERDICT: READY.')
    print()
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-state')
    group.add_argument('--live', action='store_true')
    group.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.check_state:
        check_state(args.check_state)
    elif args.preflight:
        raise SystemExit(preflight())
    else:
        live_qa()


if __name__ == '__main__':
    main()