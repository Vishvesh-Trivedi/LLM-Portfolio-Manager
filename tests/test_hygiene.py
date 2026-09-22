"""Guards against defect classes creeping back in.

Each of these was a real problem in this repository, found the expensive way -
from a live run, or from an audit after the fact. A test is cheaper than
finding it again.
"""

import ast
import pathlib
import re
import unittest
from typing import ClassVar

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = sorted(ROOT.glob('*.py'))


def read(path):
    return path.read_text(encoding='utf-8-sig').lstrip('﻿')


def parse(path):
    return ast.parse(read(path))


class NoBareExcept(unittest.TestCase):
    """`except:` catches BaseException - KeyboardInterrupt and SystemExit too.

    Ctrl-C during a screen would be swallowed and the run would carry on, and a
    sys.exit inside a guarded block would do nothing. Every handler in this
    repository means "this optional lookup may fail", which is
    `except Exception:`.
    """

    def test_no_module_uses_a_bare_except(self):
        offenders = []
        for path in MODULES:
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    offenders.append(f'{path.name}:{node.lineno}')
        self.assertEqual(offenders, [], 'use `except Exception:` instead')


class NoMutableDefaultArguments(unittest.TestCase):
    """A list or dict default is created once and shared by every call."""

    def test_no_function_takes_a_mutable_default(self):
        offenders = []
        for path in MODULES:
            for node in ast.walk(parse(path)):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        offenders.append(f'{path.name}:{node.lineno} {node.name}')
        self.assertEqual(offenders, [])


class NoCredentialsInSource(unittest.TestCase):
    """A token pasted into tracked source is published the moment it is pushed.

    This repository has had exactly that happen, so the guard is not theoretical.
    """

    PATTERNS: ClassVar[tuple] = (
        r'[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}',   # Discord bot
        r'\bsk-[A-Za-z0-9]{20,}',                                    # OpenAI style
        r'\bnvapi-[A-Za-z0-9]{20,}',                                 # NVIDIA
        r'AKIA[0-9A-Z]{16}',                                         # AWS
    )

    def test_no_module_contains_something_shaped_like_a_secret(self):
        offenders = []
        for path in MODULES:
            text = read(path)
            for pattern in self.PATTERNS:
                for match in re.finditer(pattern, text):
                    offenders.append(f'{path.name}: {match.group(0)[:10]}...')
        self.assertEqual(offenders, [], 'rotate it, then remove it from source')


class TheTradingPathStaysCovered(unittest.TestCase):
    """These decide trades, move money, or write the ledger.

    Every one of them has had a bug that reached production. A new one arriving
    without a test is how the next one gets there.
    """

    CRITICAL: ClassVar[set] = {
        'sync_with_broker', 'protect_positions', 'reconcile_broker',
        'send_run_digest', '_resolve_sector', '_resolve_risk_levels',
        '_persist_session', '_missed_sessions', '_entry_levels_by_symbol',
    }

    def test_every_critical_function_is_named_by_some_test(self):
        tests = ' '.join(path.read_text(encoding='utf-8', errors='ignore')
                         for path in (ROOT / 'tests').glob('test_*.py'))
        module = parse(ROOT / 'LLM_Portfolio_Manager.py')
        defined = {node.name for node in module.body
                   if isinstance(node, ast.FunctionDef)}
        expected = self.CRITICAL & defined
        self.assertTrue(expected, 'the critical set has drifted from the module')
        self.assertEqual(sorted(name for name in expected if name not in tests), [])


if __name__ == '__main__':
    unittest.main()
