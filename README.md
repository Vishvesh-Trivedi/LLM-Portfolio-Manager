# LLM Portfolio Manager

A Python-based stock screening and portfolio workflow that combines market data, technical indicators, news sentiment, and LLM reasoning to generate BUY/WATCH decisions.

## Current Entry Point

- Main script: `LLM_Portfolio_Manager.py`

## Project Layout

- `LLM_Portfolio_Manager.py` - current all-in-one script.
- `requirements.txt` - Python dependencies.
- `.env.example` - environment variable template.
- `docs/` - project documentation.
- `config/` - configuration files (future split from script constants).
- `data/` - local intermediate data (git-ignored except `.gitkeep`).
- `reports/` - generated reports and exports (git-ignored except `.gitkeep`).
- `tests/` - test suite and testing notes.

See `docs/PROJECT_STRUCTURE.md` for details and next refactor steps.

## Quick Start (Local)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set values.
4. Run:

```bash
python LLM_Portfolio_Manager.py
```

## Environment Variables

- `NVIDIA_API_KEY` - required for LLM analysis (free key from build.nvidia.com).
- `WHATSAPP_PHONE` - optional, for WhatsApp notifications (E.164 format, e.g. +64211234567).
- `CALLMEBOT_API_KEY` - optional, for WhatsApp notifications (via CallMeBot).

## Automated Daily Run (GitHub Actions)

The workflow at `.github/workflows/daily-screener.yml` runs the screener automatically
every day at 21:30 UTC (after the US market close) and on manual dispatch.

- The script self-skips on weekends and US market holidays, so the daily schedule is safe.
- Add the three variables above as repository **Secrets**
  (`Settings -> Secrets and variables -> Actions`). Only `NVIDIA_API_KEY` is required;
  the two WhatsApp values enable alerts.
- Set `Settings -> Actions -> General -> Workflow permissions` to
  **Read and write permissions** so the run can commit updated state back and upload
  the HTML/CSV/JSON reports as build artifacts.

## WhatsApp Alerts

When configured, each run sends two concise WhatsApp messages:

1. **Daily decision** - market read (QQQ/VIX/SPY), the actual action taken
   (bought / picked but not opened / watch only / no buy), entry/stop/target and
   risk:reward, the reasoning and key risk, positions closed today, and the next watchlist.
2. **Portfolio** - total value, P&L, cash, and each holding worst-first with a health tag
   (near stop / near target / hold done / ok).

## Notes

- The current script is notebook-style and runs end-to-end when executed.
- Runtime output is written under `StockScreener/` by default when not in Colab.
- This repository structure is prepared so the script can be split into modules incrementally.
