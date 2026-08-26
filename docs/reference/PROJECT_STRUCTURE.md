# Project Structure Plan

This repository currently uses one large script. The structure below supports both current usage and a safe migration path toward modular code.

## Implemented Structure

```text
LLM-PortfolioManager/
  LLM_Portfolio_Manager.py       # main script (entry point, required by CI)
  README.md
  requirements.txt
  .env.example
  .gitignore
  config/                        # config placeholder (future split from constants)
  data/
    .gitkeep
  reports/
    .gitkeep
  docs/
    guides/                      # how-to guides
      RUNBOOK.md
      NVIDIA_FREE_ENDPOINT_GUIDE.txt
    reference/                   # reference material
      PROJECT_STRUCTURE.md
      api_inventory.csv
      CODE_AUDIT_REPORT.md
  tests/
    README.md
    test_market_data.py
  StockScreener/                 # runtime output (git-ignored): portfolio + reports
```

> Note: `LLM_Portfolio_Manager.py`, `tests/`, and `StockScreener/` must stay at the
> repo root — the GitHub Actions workflow and the script's `__file__`-based output
> path depend on those locations.

## Recommended Next Modular Layout

```text
src/
  core/
    config.py
    paths.py
    logging_setup.py
  data/
    market_data.py
    news_feed.py
    indicators.py
  llm/
    client.py
    prompts.py
    scoring.py
    validation.py
  portfolio/
    state.py
    execution.py
    risk.py
  reporting/
    csv_writer.py
    html_report.py
    notifier.py
  app/
    run_screener.py
```

## Migration Steps

1. Extract constants and environment loading into `core/config.py`.
2. Move market/news collection logic into `data/` modules.
3. Isolate LLM request/response handling into `llm/` modules.
4. Move portfolio state transitions into `portfolio/` modules.
5. Keep `LLM_Portfolio_Manager.py` as a thin runner until migration is complete.
