"""Discord delivery. Small on purpose — reporting must never break a rebalance.

Every send is wrapped: a notification failure returns False and is logged by the
caller, it never raises into the trading path. A bot that refuses to trade
because Discord is down is worse than one that trades silently.

Environment variables
    DISCORD_BOT_TOKEN     bot token (never logged; redacted from every message)
    DISCORD_CHANNEL_ID    numeric channel id the bot posts into
    SCREENER_DISABLE_ALERTS  '1' silences delivery (tests, QA, dry runs)
    SCREENER_ALERT_TEST      '1' marks every message as a test (see below)

``SCREENER_ALERT_TEST`` exists because a realistic-looking alert is dangerous:
a reader cannot tell a rehearsal from a real fill, and acting on a fake one
costs real money. When set, every message carries a TEST banner and a neutral
grey colour. It is applied here, at the single delivery choke point, so no
caller can forget it. It marks MESSAGES ONLY — it does not stop the screener
from placing orders; use SCREENER_LIVE_BROKER for that.

The bot needs "Send Messages" in the target channel. Nothing here reads from
Discord, so no gateway connection, no privileged intents and no extra
dependency beyond ``requests`` are required.
"""

import os
import time

import requests


API = "https://discord.com/api/v10"

# Discord rejects a message body over 2000 characters. Long reports are split on
# line boundaries rather than truncated, so a portfolio table is never cut off
# mid-holding the way the WhatsApp path has to do it.
_CONTENT_LIMIT = 2000
_CHUNK_TARGET = 1900
_REQUEST_TIMEOUT = 15
_MAX_ATTEMPTS = 3
_MAX_CHUNKS = 6


def _token():
    return os.environ.get('DISCORD_BOT_TOKEN', '').strip()


def _channel_id():
    return os.environ.get('DISCORD_CHANNEL_ID', '').strip()


ALERT_TEST_ENV = 'SCREENER_ALERT_TEST'
TEST_TITLE_PREFIX = '🧪 TEST — '
TEST_BANNER = '🧪 **TEST MESSAGE — not a real trade**'
TEST_FOOTER = 'TEST MESSAGE — not a real trade'
TEST_COLOR = 0x8E8E93  # neutral grey: never reads as a green fill or red reject


def enabled():
    """True when a token and channel are configured and alerts are not disabled."""
    if os.environ.get('SCREENER_DISABLE_ALERTS') == '1':
        return False
    return bool(_token() and _channel_id())


def test_mode():
    """True when every outgoing message must be marked as a rehearsal."""
    return os.environ.get(ALERT_TEST_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')


_LAST_ERROR = ['']


def last_error():
    """Why the most recent send failed, for the run's health record."""
    return _LAST_ERROR[0]


def _fail(reason):
    _LAST_ERROR[0] = str(reason)[:120]
    return False


def _redact(text):
    """Strip the bot token from any string before it can reach a log."""
    detail = str(text)
    token = _token()
    if token:
        detail = detail.replace(token, '[redacted]')
    return detail


def _split(text):
    """Split into <=2000 char chunks, preferring line then word boundaries."""
    cleaned = (text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not cleaned:
        return []
    chunks = []
    remaining = cleaned
    while remaining and len(chunks) < _MAX_CHUNKS:
        if len(remaining) <= _CONTENT_LIMIT:
            chunks.append(remaining)
            break
        window = remaining[:_CHUNK_TARGET]
        cut = window.rfind('\n')
        if cut < int(_CHUNK_TARGET * 0.5):
            cut = window.rfind(' ')
        if cut < int(_CHUNK_TARGET * 0.5):
            cut = _CHUNK_TARGET
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    else:
        # Ran out of chunk budget: mark the tail as dropped rather than pretend
        # the report was delivered whole.
        if remaining:
            chunks.append('... (truncated; see the HTML report for the rest)')
    return [chunk for chunk in chunks if chunk]


def _post(payload, label):
    """One bounded POST. Returns True only on a 2xx; never raises."""
    url = API + '/channels/' + _channel_id() + '/messages'
    headers = {'Authorization': 'Bot ' + _token(),
               'Content-Type': 'application/json',
               'User-Agent': 'LLM-Portfolio-Manager (+https://github.com/Vishvesh-Trivedi/LLM-Portfolio-Manager, 1.0)'}
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = requests.post(url, headers=headers, json=payload,
                                     timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            if attempt == _MAX_ATTEMPTS - 1:
                print('  Discord ' + label + ' transport error: ' + type(exc).__name__)
                return False
            time.sleep(min(2 ** attempt, 4))
            continue
        if response.status_code == 429:
            # Honour Discord's own backoff hint; fall back to exponential.
            wait = 2.0
            try:
                wait = float((response.json() or {}).get('retry_after', wait))
            except (ValueError, AttributeError, TypeError):
                pass
            if attempt == _MAX_ATTEMPTS - 1:
                print('  Discord ' + label + ': rate limited, giving up')
                return False
            time.sleep(max(0.0, min(wait, 10.0)))
            continue
        if response.status_code in (500, 502, 503, 504):
            if attempt == _MAX_ATTEMPTS - 1:
                print('  Discord ' + label + ' HTTP ' + str(response.status_code))
                return False
            time.sleep(min(2 ** attempt, 4))
            continue
        if response.status_code >= 400:
            hint = {401: 'bad DISCORD_BOT_TOKEN', 403: 'bot lacks Send Messages here',
                    404: 'wrong DISCORD_CHANNEL_ID'}.get(response.status_code, '')
            print('  Discord ' + label + ' HTTP ' + str(response.status_code)
                  + ': ' + _redact(response.text)[:160])
            return _fail('HTTP ' + str(response.status_code)
                         + (' - ' + hint if hint else ''))
        _LAST_ERROR[0] = ''
        return True
    return _fail('gave up after ' + str(_MAX_ATTEMPTS) + ' attempts')


def check_access():
    """Verify the bot can see the channel. Read-only: posts nothing.

    Returns ``(ok, detail)``. A token can be valid while the bot has never been
    invited to the server, which only shows up as a failed send otherwise.
    """
    try:
        if not _token() or not _channel_id():
            return False, 'DISCORD_BOT_TOKEN / DISCORD_CHANNEL_ID not set'
        response = requests.get(
            API + '/channels/' + _channel_id(),
            headers={'Authorization': 'Bot ' + _token()}, timeout=_REQUEST_TIMEOUT)
        if response.status_code == 200:
            try:
                name = (response.json() or {}).get('name') or _channel_id()
            except ValueError:
                name = _channel_id()
            return True, 'bot can post to #' + str(name)
        if response.status_code == 401:
            return False, 'token rejected by Discord (401) - rotate and update the secret'
        if response.status_code == 403:
            return False, 'bot cannot see this channel (403) - invite it and grant Send Messages'
        if response.status_code == 404:
            return False, 'channel not found (404) - check DISCORD_CHANNEL_ID'
        return False, 'Discord returned HTTP ' + str(response.status_code)
    except requests.exceptions.RequestException as exc:
        return False, 'could not reach Discord: ' + type(exc).__name__
    except Exception as exc:
        return False, 'check failed: ' + _redact(type(exc).__name__)


def send(text, label=''):
    """Send a plain message, splitting when over Discord's 2000-char limit.

    Returns True only when every chunk was accepted. Never raises.
    """
    try:
        if not enabled():
            return False
        chunks = _split(text)
        if not chunks:
            return False
        if test_mode():
            # Banner every chunk: a split message must not have unmarked parts.
            chunks = [TEST_BANNER + '\n' + chunk for chunk in chunks]
        sent = 0
        for index, chunk in enumerate(chunks):
            if not _post({'content': chunk,
                          'allowed_mentions': {'parse': []}}, label):
                break
            sent += 1
            if index < len(chunks) - 1:
                time.sleep(0.5)  # stay clear of the per-channel burst limit
        ok = sent == len(chunks)
        suffix = '' if len(chunks) == 1 else ' (' + str(sent) + '/' + str(len(chunks)) + ' parts)'
        print('  Discord ' + label + ': ' + ('sent' if ok else 'failed') + suffix)
        return ok
    except Exception as exc:  # defensive: delivery must never reach the trade path
        print('  Discord ' + label + ' error: ' + _redact(type(exc).__name__))
        return False


def send_embed(title, description, color=0x5865F2, fields=None, footer='',
               author='', timestamp=None, label=''):
    """Send a single embed. Falls back to nothing (False) when disabled.

    ``fields`` is a list of ``(name, value, inline)`` tuples; inline fields lay
    out three-per-row, which is what gives an alert the stat-row look of a
    broker notification. ``timestamp`` must be ISO 8601 — Discord renders it as
    a native, locally-formatted time rather than text in the body. Values are
    trimmed to Discord's per-field limits so an over-long reason cannot 400 the
    send.
    """
    try:
        if not enabled():
            return False
        if test_mode():
            title = TEST_TITLE_PREFIX + str(title)
            color = TEST_COLOR
            footer = (str(footer) + ' · ' if footer else '') + TEST_FOOTER
        embed = {'title': str(title)[:256],
                 'description': str(description)[:4096],
                 'color': int(color)}
        if fields:
            embed['fields'] = [{'name': str(name)[:256],
                                'value': (str(value) or '—')[:1024],
                                'inline': bool(inline)}
                               for name, value, inline in list(fields)[:25]]
        if footer:
            embed['footer'] = {'text': str(footer)[:2048]}
        if author:
            embed['author'] = {'name': str(author)[:256]}
        if timestamp:
            embed['timestamp'] = str(timestamp)
        ok = _post({'embeds': [embed], 'allowed_mentions': {'parse': []}}, label)
        print('  Discord ' + label + ': ' + ('sent' if ok else 'failed'))
        return ok
    except Exception as exc:
        print('  Discord ' + label + ' error: ' + _redact(type(exc).__name__))
        return False


__all__ = ['enabled', 'test_mode', 'check_access', 'send', 'send_embed', 'API']
