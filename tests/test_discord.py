"""Offline tests for Discord delivery.

No network: requests.post is always mocked. The contract under test is that
reporting can fail in every way without raising into the trading path, and that
the bot token never reaches stdout.
"""

import io
import os
import socket
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import requests

import screener_discord as discord


TOKEN = 'MTAxMjM0NTY3ODkw.Gabcde.notarealtokenjustfortests'
CHANNEL = '1234567890'


class Response:
    def __init__(self, status_code=200, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload


class DiscordTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect',
                                    side_effect=AssertionError('network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.env = patch.dict(os.environ, {'DISCORD_BOT_TOKEN': TOKEN,
                                           'DISCORD_CHANNEL_ID': CHANNEL},
                              clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop('SCREENER_DISABLE_ALERTS', None)
        self.sleep = patch.object(discord.time, 'sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def post(self, *responses):
        mock = Mock(side_effect=list(responses))
        return patch.object(discord.requests, 'post', mock), mock

    def send(self, *responses, text='hello', label='t'):
        patcher, mock = self.post(*responses)
        with patcher, redirect_stdout(io.StringIO()) as out:
            result = discord.send(text, label)
        return result, mock, out.getvalue()

    # ── Configuration gating ─────────────────────────────────────────────

    def test_requires_both_token_and_channel(self):
        self.assertTrue(discord.enabled())
        for missing in ('DISCORD_BOT_TOKEN', 'DISCORD_CHANNEL_ID'):
            with patch.dict(os.environ, {missing: ''}):
                self.assertFalse(discord.enabled())

    def test_disable_alerts_silences_delivery(self):
        with patch.dict(os.environ, {'SCREENER_DISABLE_ALERTS': '1'}):
            self.assertFalse(discord.enabled())
            with patch.object(discord.requests, 'post',
                              side_effect=AssertionError('must not post')):
                self.assertFalse(discord.send('anything', 'x'))

    def test_empty_message_is_not_sent(self):
        with patch.object(discord.requests, 'post',
                          side_effect=AssertionError('must not post')):
            self.assertFalse(discord.send('   \n  ', 'x'))
            self.assertFalse(discord.send(None, 'x'))

    # ── Request shape ────────────────────────────────────────────────────

    def test_posts_bot_auth_to_the_channel_messages_endpoint(self):
        ok, mock, _ = self.send(Response())
        self.assertTrue(ok)
        (url,), kwargs = mock.call_args
        self.assertEqual(url, discord.API + '/channels/' + CHANNEL + '/messages')
        self.assertEqual(kwargs['headers']['Authorization'], 'Bot ' + TOKEN)
        self.assertEqual(kwargs['json']['content'], 'hello')

    def test_mentions_are_suppressed(self):
        # A pick reasoning containing @everyone must never ping a whole server.
        ok, mock, _ = self.send(Response(), text='watch @everyone and @here')
        self.assertEqual(mock.call_args.kwargs['json']['allowed_mentions'],
                         {'parse': []})

    # ── Long reports ─────────────────────────────────────────────────────

    def test_long_message_splits_on_line_boundaries(self):
        body = '\n'.join(f'LINE {i:04d} ' + 'x' * 40 for i in range(120))
        ok, mock, _ = self.send(*[Response()] * discord._MAX_CHUNKS, text=body)
        self.assertTrue(ok)
        self.assertGreater(mock.call_count, 1)
        chunks = [c.kwargs['json']['content'] for c in mock.call_args_list]
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2000)
        # Nothing may be silently dropped between the chunks.
        self.assertEqual(''.join(chunks).replace('\n', ''), body.replace('\n', ''))

    def test_partial_chunk_failure_reports_failure(self):
        body = '\n'.join(f'LINE {i:04d} ' + 'x' * 40 for i in range(120))
        ok, mock, out = self.send(Response(), Response(status_code=400, text='bad'),
                                  text=body)
        self.assertFalse(ok)
        self.assertIn('parts', out)

    # ── Failure handling ─────────────────────────────────────────────────

    def test_transport_error_returns_false_without_raising(self):
        patcher, _ = self.post(*[requests.exceptions.ConnectionError('down')] * 3)
        with patcher, redirect_stdout(io.StringIO()):
            self.assertFalse(discord.send('hi', 'x'))

    def test_server_error_is_retried_then_gives_up(self):
        ok, mock, _ = self.send(*[Response(status_code=503)] * 3)
        self.assertFalse(ok)
        self.assertEqual(mock.call_count, 3)

    def test_rate_limit_honours_retry_after_then_succeeds(self):
        ok, mock, _ = self.send(Response(status_code=429, payload={'retry_after': 1.5}),
                                Response())
        self.assertTrue(ok)
        self.assertEqual(mock.call_count, 2)
        discord.time.sleep.assert_any_call(1.5)

    def test_rate_limit_without_a_parsable_body_still_backs_off(self):
        ok, _, _ = self.send(Response(status_code=429), Response())
        self.assertTrue(ok)

    def test_unexpected_exception_is_swallowed(self):
        with patch.object(discord, '_split', side_effect=RuntimeError('boom')), \
                redirect_stdout(io.StringIO()):
            self.assertFalse(discord.send('hi', 'x'))

    # ── Secret hygiene ───────────────────────────────────────────────────

    def test_token_is_never_printed_on_an_error_response(self):
        _, _, out = self.send(Response(status_code=401, text='Unauthorized for ' + TOKEN))
        self.assertNotIn(TOKEN, out)
        self.assertIn('[redacted]', out)

    # ── Embeds ───────────────────────────────────────────────────────────

    def test_embed_carries_fields_and_trims_overlong_values(self):
        patcher, mock = self.post(Response())
        with patcher, redirect_stdout(io.StringIO()):
            ok = discord.send_embed('FILLED — MTD', 'desc', color=0x2ECC71,
                                    fields=[('Reason', 'y' * 4000, False)],
                                    footer='f', label='x')
        self.assertTrue(ok)
        embed = mock.call_args.kwargs['json']['embeds'][0]
        self.assertEqual(embed['title'], 'FILLED — MTD')
        self.assertEqual(embed['color'], 0x2ECC71)
        self.assertEqual(len(embed['fields'][0]['value']), 1024)
        self.assertEqual(embed['footer']['text'], 'f')

    def test_embed_field_count_is_capped(self):
        patcher, mock = self.post(Response())
        with patcher, redirect_stdout(io.StringIO()):
            discord.send_embed('t', 'd', fields=[(f'n{i}', 'v', True) for i in range(40)],
                               label='x')
        self.assertEqual(len(mock.call_args.kwargs['json']['embeds'][0]['fields']), 25)

    def test_embed_is_not_sent_when_disabled(self):
        with patch.dict(os.environ, {'SCREENER_DISABLE_ALERTS': '1'}), \
                patch.object(discord.requests, 'post',
                             side_effect=AssertionError('must not post')):
            self.assertFalse(discord.send_embed('t', 'd', label='x'))

    def test_embed_failure_does_not_raise(self):
        patcher, _ = self.post(Response(status_code=403, text='forbidden'))
        with patcher, redirect_stdout(io.StringIO()):
            self.assertFalse(discord.send_embed('t', 'd', label='x'))

    # ── Test-message marking ─────────────────────────────────────────────

    def test_test_mode_is_off_by_default_and_opt_in(self):
        self.assertFalse(discord.test_mode())
        for value in ('1', 'true', 'YES', 'on'):
            with patch.dict(os.environ, {discord.ALERT_TEST_ENV: value}):
                self.assertTrue(discord.test_mode())
        for value in ('0', 'false', '', 'maybe'):
            with patch.dict(os.environ, {discord.ALERT_TEST_ENV: value}):
                self.assertFalse(discord.test_mode())

    def test_plain_message_is_bannered_in_test_mode(self):
        with patch.dict(os.environ, {discord.ALERT_TEST_ENV: '1'}):
            ok, mock, _ = self.send(Response(), text='BOUGHT 18 MTD')
        self.assertTrue(ok)
        self.assertTrue(mock.call_args.kwargs['json']['content'].startswith(discord.TEST_BANNER))

    def test_every_chunk_of_a_split_message_is_bannered(self):
        body = '\n'.join(f'LINE {i:04d} ' + 'x' * 40 for i in range(120))
        with patch.dict(os.environ, {discord.ALERT_TEST_ENV: '1'}):
            ok, mock, _ = self.send(*[Response()] * discord._MAX_CHUNKS, text=body)
        self.assertTrue(ok)
        self.assertGreater(mock.call_count, 1)
        for call in mock.call_args_list:
            content = call.kwargs['json']['content']
            self.assertTrue(content.startswith(discord.TEST_BANNER))
            self.assertLessEqual(len(content), 2000)

    def test_embed_is_retitled_greyed_and_footnoted_in_test_mode(self):
        patcher, mock = self.post(Response())
        with patch.dict(os.environ, {discord.ALERT_TEST_ENV: '1'}), patcher,                 redirect_stdout(io.StringIO()):
            discord.send_embed('SOLD 18 MTD @ $1,512.00', 'You made $1,957',
                               color=0x00C805, footer='Alpaca Paper', label='x')
        embed = mock.call_args.kwargs['json']['embeds'][0]
        self.assertTrue(embed['title'].startswith(discord.TEST_TITLE_PREFIX))
        self.assertEqual(embed['color'], discord.TEST_COLOR,
                         'a test card must never keep the green fill colour')
        self.assertIn(discord.TEST_FOOTER, embed['footer']['text'])
        self.assertIn('Alpaca Paper', embed['footer']['text'])

    def test_real_messages_are_untouched_when_test_mode_is_off(self):
        patcher, mock = self.post(Response())
        with patcher, redirect_stdout(io.StringIO()):
            discord.send_embed('SOLD 18 MTD', 'd', color=0x00C805, label='x')
        embed = mock.call_args.kwargs['json']['embeds'][0]
        self.assertEqual(embed['title'], 'SOLD 18 MTD')
        self.assertEqual(embed['color'], 0x00C805)
        self.assertNotIn('footer', embed)


if __name__ == '__main__':
    unittest.main()
