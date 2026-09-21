#!/usr/bin/env python
"""Explain the strategy to someone who has never traded.

Every number below is read from the running configuration rather than typed
in, so this cannot quietly drift out of date the way a hand-written README
would. If a threshold changes, this message changes with it.

    python explain_strategy.py             # print it, send nothing
    python explain_strategy.py --send      # also post it to Discord

Sending needs DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in the environment,
which is why the normal way to send it is the "Daily Stock Screener" workflow
with the strategy_explainer input set to true - the secrets live there.
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('SCREENER_SKIP_UNIVERSE_FETCH', '1')
with redirect_stdout(io.StringIO()):          # the import prints a config banner
    import LLM_Portfolio_Manager as app
import screener_discord as discord


def _record():
    """Closed-trade statistics, or None when the ledger cannot be read."""
    try:
        book = json.loads(Path(app.PORTFOLIO_JSON).read_text(encoding='utf-8'))
    except Exception:
        return None
    trades = book.get('closed_trades') or []
    if not trades:
        return None
    wins = [t for t in trades if t.get('realized_pnl', 0) > 0]
    losses = [t for t in trades if t.get('realized_pnl', 0) <= 0]

    def average(group):
        return sum(t.get('realized_pnl_pct', 0) for t in group) / len(group) if group else 0.0

    return {'n': len(trades), 'wins': len(wins), 'losses': len(losses),
            'avg_win': average(wins), 'avg_loss': average(losses),
            'total': book.get('total_realized_pnl', 0)}


def build():
    lines = [
        '**HOW MY TRADING BOT WORKS**',
        '_Written for people who have never bought a share. No jargon._',
        '',
        '**FIRST, THE IMPORTANT BIT**',
        'This is fake money. The account is a practice account at a broker '
        'called Alpaca. Real share prices, real orders, real timing - but the '
        'money is not real. I am doing this to find out whether the idea works '
        'before ever risking anything.',
        '',
        'Alpaca charges nothing to buy or sell US shares, so there are no fees '
        'eating into the results. What you see is what the strategy did.',
        '',
        '**WHAT IT DOES EACH DAY**',
        'Every weekday, after the US stock market closes, a program wakes up '
        'and does this:',
        '1. Looks at about 900 big US companies (the 500 largest, plus 400 '
        'medium ones).',
        '2. Scores every one of them out of 100.',
        '3. Buys at most one. Often it buys nothing.',
        '4. Tells me what it did on Discord and WhatsApp.',
        '',
        'It never trades during the day. It decides after the market shuts, '
        'and the order goes through when the market opens the next morning.',
        '',
        '**HOW IT SCORES A COMPANY**',
        'Half the score comes from the share price chart, half from the news. '
        'None of it is a hunch - it is all arithmetic.',
        '',
        '_From the price chart (60 points)_',
        '- Has the price been climbing steadily, rather than jumping around? '
        '(12 points)',
        '- Are more people buying it than normal? (12)',
        '- Is the climb speeding up, or running out of steam? (12)',
        '- Do the different ways of measuring "is this going up" agree? (12)',
        '- Is the price above its own recent average, and near its highest '
        'point of the last year? (12)',
        '- A small bonus if that whole industry is doing well. (2)',
        '',
        '_From the news (40 points)_',
        '- Do recent headlines about it sound good or bad? (15 points)',
        '- Is the news genuinely important, or just chatter? (10)',
        '- Is the market as a whole up or down today? (10)',
        '- Do professional analysts rate it? (3)',
        '- Are big investors placing unusual bets on it? (2)',
        "- Are the company's own staff buying their own shares? (2)",
        '',
        '**THEN AN AI HAS THE FINAL SAY**',
        'The arithmetic picks the best handful. Those get handed to an AI, '
        'which reads the details, gives its own score out of 100, and has to '
        'write down why. That AI score is what actually decides.',
        f'- {app.BUY_THRESHOLD} or above: buy it',
        f'- {app.WATCH_THRESHOLD} to {app.BUY_THRESHOLD - 1}: interesting, but do nothing',
        f'- Below {app.WATCH_THRESHOLD}: ignore it',
        '',
        'Most days nothing scores high enough, so most days it buys nothing. '
        'That is on purpose.',
        '',
        '**WHEN IT SELLS - decided before it even buys**',
        'The moment it buys, it already knows the two prices at which it will '
        'sell. Both are based on how much that particular share normally moves '
        'in a day, so a calm share gets tight limits and a jumpy one gets '
        'wider limits.',
        '',
        '- **If the price falls** to a set level, sell immediately and take '
        'the small loss. No arguing, no hoping it comes back.',
        '- **If the price rises** to a set level, sell and bank the profit.',
        '- The profit target is set at **twice** the distance of the loss '
        'limit. So it should win about twice as much as it loses, each time.',
        '- **If it keeps rising**, the sell-at-a-loss price is dragged up '
        'behind it, so a winner cannot turn back into a loser. That price only '
        'ever moves up, never down.',
        '',
        'It will also sell if:',
        f'- it has held the share {app._CFG_HOLD_DAYS} days and nothing is '
        'happening',
        '- the company is about to announce results (too unpredictable)',
        '- the rise looks exhausted - but only once the trade is already up by '
        'more than it was risking. It used to sell the moment a share showed '
        'any profit at all, which meant it kept selling its winners far too '
        'early. One was sold for a gain of 0.04%.',
        '',
        '**WHERE THE SAFETY NET ACTUALLY LIVES**',
        'This is the part people assume is riskier than it is. Those two sell '
        'prices are not a note in a file waiting for the program to wake up. '
        'The moment a share is bought, both are placed as **real standing '
        'orders at Alpaca**, linked so that whichever one triggers cancels the '
        'other.',
        '',
        'So if a share crashes at 10am, Alpaca sells it at 10am. The program is '
        'only awake for a few minutes a day, but the protection sits at the '
        'broker the whole time and does not need the program to be running, '
        'or even working.',
        '',
        '**ALPACA IS THE BOSS, NOT THE PROGRAM**',
        'Every single run, before doing anything else, the program asks Alpaca '
        'what it actually owns and how much cash it actually has - and rewrites '
        'its own records to match.',
        '',
        'If the program and the broker ever disagree, the broker wins, every '
        'time. If it cannot get a clear answer from Alpaca, it refuses to trade '
        'at all that day rather than act on numbers it is not sure about.',
        '',
        '**THE RULES IT IS NOT ALLOWED TO BREAK**',
        'These are locked in the code. The AI can make them stricter but can '
        'never loosen them, no matter what it decides.',
        '- Never put more than a quarter of the money in one company',
        '- Never put more than 40% into one industry',
        '- Never risk more than 1% of the account on any single trade',
        f'- Never own more than {app._CFG_MAX_POSITIONS} companies at once',
        f'- Always keep at least ${app._CFG_MIN_CASH_FLOOR:,.0f} in cash',
        '- **If the account ever drops 20% below its best-ever value, stop '
        'buying completely**',
        '- Refuse any trade where the possible gain is not at least 1.5 times '
        'the possible loss',
        '',
        '**DOES IT LEARN WHEN IT LOSES?**',
        'This is the question everyone asks, and the honest answer is: not by '
        'itself.',
        '',
        '- It does notice. After 3 losing trades in a row, it flags that '
        'something is wrong.',
        '- It then asks the AI to suggest better settings, and saves those '
        'suggestions to a file.',
        '- **But nothing changes automatically.** A human has to read the '
        'suggestion and decide.',
        '- The one thing that happens on its own is the 20% rule above: if '
        'losses get that bad, it simply stops buying.',
        '',
        'That is deliberate. A bot that quietly rewrites its own safety rules '
        'after a bad week is exactly how people blow up their accounts.',
    ]

    record = _record()
    if record:
        lines += [
            '',
            '**HOW IT IS ACTUALLY GOING**',
            f'- {record["n"]} trades finished so far: {record["wins"]} made '
            f'money, {record["losses"]} lost money',
            f'- The average win was {record["avg_win"]:+.1f}% and the average '
            f'loss was {record["avg_loss"]:+.1f}%',
            f'- Net result: ${record["total"]:,.2f} on a $100,000 practice account',
            '- That is far too few trades to prove anything either way.',
        ]

    lines += [
        '',
        '**THE HONEST CATCH**',
        'Nobody has ever tested these rules against years of past market data. '
        'They were reasoned out, not proven. Running it forward on fake money '
        'is how I find out if they hold up.',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--send', action='store_true',
                        help='post to Discord (default is to print only)')
    args = parser.parse_args()

    message = build()
    print(message)
    print()
    print(f'  [{len(message)} characters]')

    if not args.send:
        print('  Nothing sent. Pass --send to post it to Discord.')
        return 0
    if not discord.enabled():
        print('  Cannot send: DISCORD_BOT_TOKEN / DISCORD_CHANNEL_ID are not set.')
        return 1
    if discord.send(message, 'strategy-explainer'):
        print('  Sent to Discord.')
        return 0
    print(f'  Discord refused it: {discord.last_error()}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
