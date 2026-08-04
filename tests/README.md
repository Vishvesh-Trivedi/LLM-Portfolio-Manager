# Tests

Run the regression suite with `python -m unittest discover -s tests -p "test_*.py"`.

Current coverage:
- Yahoo Finance OHLCV cleanup when an in-progress trailing row contains NaNs.
- Technical indicator calculation after incomplete-row cleanup.
- Close-price cleanup for market context.

Suggested first tests:
- Candidate filtering rules.
- Parsing/validation of LLM responses.
- Portfolio open/close lifecycle behavior.
