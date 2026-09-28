#!/usr/bin/env python3
"""Gate 11B: Chat reasoning + structured output."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

OK_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "ok", "strict": True,
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
               "required": ["ok"], "additionalProperties": False}}}
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=")


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def main() -> int:
    checker = Checker("server-reasoning-structured")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            prompt = "Return an object with ok set to true."
            base = {"model": "phaseshift", "temperature": 0, "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": OK_SCHEMA}

            status, body = post(url, dict(base, reasoning_effort="high"))
            msg = message(body)
            content = msg.get("content") or ""
            checker.check("high-200", status == 200, repr(body)[:200])
            checker.check("high-reasoning", bool((msg.get("reasoning") or "").strip()),
                          repr(msg.get("reasoning"))[:120])
            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                parsed = None
            checker.check("high-json", isinstance(parsed, dict) and "ok" in parsed,
                          repr(content)[:160])
            checker.check("high-bool", isinstance((parsed or {}).get("ok"), bool),
                          repr(parsed))
            checker.check("high-no-markup",
                          all(m not in content for m in MARKUP), repr(content)[:160])

            status, body = post(url, dict(base, reasoning_effort="none"))
            msg = message(body)
            content = msg.get("content") or ""
            checker.check("none-200", status == 200, repr(body)[:200])
            checker.check("none-no-reasoning", not msg.get("reasoning"), repr(msg)[:160])
            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                parsed = None
            checker.check("none-json", isinstance(parsed, dict) and "ok" in parsed,
                          repr(content)[:160])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
