# -*- coding: utf-8 -*-"""Daily_Stock_Screener_v6_2.py======================================Version 6.2 - Runtime FixesBuilt by Vishvesh TrivediOSS Architect | AI/ML Automation | 12 PatentsLinkedIn: https://www.linkedin.com/in/vishvesh-trivedi─────────────────────────────────────────────────────────────⚠️  IMPORTANT: HOW TO SET YOUR API KEY (READ THIS FIRST)─────────────────────────────────────────────────────────────Option A - Google Colab (Recommended):  1. Click the 🔑 Secrets icon in the left sidebar  2. Add a new secret:       Name:  NVIDIA_API_KEY       Value: your-key-here (get it free from build.nvidia.com → sign up → "Get API Key")  3. Enable the secret for this notebook  4. The code below will read it automaticallyOption B - Local Python:  1. Create a file called .env in the same folder as this script  2. Add this line:  NVIDIA_API_KEY=your-key-here  3. Install python-dotenv:  pip install python-dotenv  4. The code below will read it automaticallyOption C - Paste directly (Colab only, NOT for GitHub):  Find the line:  NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")  Replace with:   NVIDIA_API_KEY = "your-key-here"  ⚠️  Never upload this version to GitHub!─────────────────────────────────────────────────────────────WHAT THIS SCREENER DOES─────────────────────────────────────────────────────────────
# ── ROBUST LLM JSON PARSING ────────────────────────────────
# LLMs occasionally wrap JSON in prose/markdown, use single quotes, double the
# outer braces, emit stray control chars, or truncate at max_tokens. A single
# failed parse in the final scoring round would otherwise abort the whole buy
# decision, so extraction tolerates all of these and only gives up when nothing
# usable decodes.
def _close_truncated_json(text):
    """Best-effort close of a truncated JSON object so json can parse it.

    Terminates an unterminated string, drops a dangling trailing token, and
    appends the missing } / ] closers in reverse order of opening.
    """
    import re as _re
    stack = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in '{[':
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
    t = text
    if in_str:
        t += '"'
    t = t.rstrip()
    # Drop a dangling comma or an incomplete trailing "key": with no value.
    t = _re.sub(r',\s*$', '', t)
    t = _re.sub(r',\s*"[^"]*"\s*:\s*$', '', t)
    t = _re.sub(r'\{\s*"[^"]*"\s*:\s*$', '{', t)
    t = t.rstrip().rstrip(',')
    for opener in reversed(stack):
        t += '}' if opener == '{' else ']'
    return t


def _parse_llm_json(raw):
    """Parse a JSON object out of an LLM reply, tolerating common glitches.

    Strips ``` fences, then tries json.raw_decode at every '{' (so stray or
    doubled outer braces and trailing prose are skipped) and returns the first
    dict decoded. Falls back to control-char stripping, trailing-comma removal,
    Python-literal parsing for mixed single/double quote payloads, and truncation
    repair. Raises the last error only when nothing usable decodes.
    """
    import ast as _ast
    import re as _re
    if not raw or not raw.strip():
        raise ValueError('empty LLM response')
    s = raw.strip()
    if '```' in s:
        for part in s.split('```'):
            p = part.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('{'):
                s = p
                break

    def _clean(t):
        t = ''.join(ch for ch in t if ord(ch) >= 32 or ch in '\n\r\t')
        return _re.sub(r',(\s*[}\]])', r'\1', t)  # drop trailing commas

    def _literal_dict(t):
        try:
            obj = _ast.literal_eval(t)
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass
        return None

    decoder = json.JSONDecoder()
    positions = [i for i, ch in enumerate(s) if ch == '{']
    last = None
    # Primary: decode at each '{'; raw_decode ignores any trailing text.
    for pos in positions:
        frag = s[pos:]
        for cand in (frag, _clean(frag)):
            try:
                obj, _end = decoder.raw_decode(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last = e
            lit = _literal_dict(cand)
            if lit is not None:
                return lit
    # Recovery on the first brace: single-quoted object and truncated output.
    # Require a non-empty dict here so genuinely unusable replies still surface
    # as a diagnosable failure rather than a silent empty result.
    if positions:
        frag = _clean(s[positions[0]:])
        repairs = []
        if "'" in frag and '"' not in frag:
            repairs.append(frag.replace("'", '"'))
        repairs.append(_close_truncated_json(frag))
        for cand in repairs:
            try:
                obj, _end = decoder.raw_decode(cand)
                if isinstance(obj, dict) and obj:
                    return obj
            except Exception as e:
                last = e
            lit = _literal_dict(cand)
            if lit is not None:
                return lit
    raise last if last is not None else ValueError('no JSON object in LLM response')

