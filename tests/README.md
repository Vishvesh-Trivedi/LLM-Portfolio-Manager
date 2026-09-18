# Tests

Run the regression suite with `python -m unittest discover -s tests -p "test_*.py"`.

Every test is offline: `socket.connect` is patched to fail, the main module is
imported into a temporary output directory, and no credentials are read. Set
`SCREENER_SKIP_UNIVERSE_FETCH=1`, `SCREENER_DISABLE_ALERTS=1` and
`SCREENER_OUTPUT_DIR=<temp>` as CI does.

> On Windows, also set `PYTHONIOENCODING=utf-8` — the module prints emoji at
> import and the default `cp1252` console encoding raises on them.

| File | Covers |
|---|---|
| `test_integration.py` | Main-module pipeline: session gate, health/fail-closed, screening, queueing, fees, reporting |
| `test_portfolio.py` | Ledger lifecycle: queue, fill, replay, exits, trailing stops, public API signatures |
| `test_safety.py` | Sizing and risk caps, atomic writes, horizon evaluation, mechanical exits |
| `test_contracts.py` | LLM/config schema validation and `RunHealth` |
| `test_alpaca.py` | Enable flags, bar conversion, reconciliation diff, redaction, client-order-id tagging, authoritative snapshot |
| `test_alpaca_recovery.py` | Order-ledger merge and recovery from polled snapshots |
| `test_broker_sync.py` | Broker-authoritative planning: fills, partials, rejects, drift, adoption, unreadable snapshots |
| `test_broker_apply.py` | Applying a broker plan to the ledger, and broker-mode replay behaviour |
| `test_discord.py` | Discord delivery: gating, chunking, retries, rate limits, token redaction |
| `test_llm_routing.py` | Per-provider rate budgets, circuit breaker, NVIDIA/OpenRouter failover routing |
| `test_preflight.py` | Pre-run readiness: blockers, warnings, ledger-change preview, read-only guarantees |
| `test_market_data.py` | OHLCV cleanup, indicators, bounded LLM calls |
