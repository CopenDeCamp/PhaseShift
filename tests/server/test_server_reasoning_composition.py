#!/usr/bin/env python3
"""Gate 11B: Chat reasoning + structured output + tools composition."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
OK_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "ok", "strict": True,
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
               "required": ["ok"], "additionalProperties": False}}}
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
TEXT_PROMPT = "Do not call any tool. Return an object with ok set to true."
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


def call_names(msg):
    return [(c.get("function") or {}).get("name") for c in (msg.get("tool_calls") or [])]


def is_ok_json(text):
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool)


def main() -> int:
    checker = Checker("server-reasoning-composition")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift", "temperature": 0, "max_tokens": 1024,
                    "reasoning_effort": "high", "tools": [WEATHER],
                    "response_format": OK_SCHEMA}

            status, body = post(url, dict(base, tool_choice="none",
                                          messages=[{"role": "user", "content": TEXT_PROMPT}]))
            msg = message(body)
            checker.check("none-200", status == 200, repr(body)[:200])
            checker.check("none-reasoning", bool((msg.get("reasoning") or "").strip()),
                          repr(msg.get("reasoning"))[:120])
            checker.check("none-text", is_ok_json(msg.get("content") or ""),
                          repr(msg.get("content"))[:160])
            checker.check("none-no-call", not msg.get("tool_calls"), repr(call_names(msg)))

            status, body = post(url, dict(base, tool_choice="required",
                                          messages=[{"role": "user", "content": TOOL_PROMPT}]))
            msg = message(body)
            checker.check("required-200", status == 200, repr(body)[:200])
            checker.check("required-call", len(call_names(msg)) >= 1
                          and set(call_names(msg)) == {"get_weather"},
                          repr(call_names(msg)))
            checker.check("required-reasoning", bool((msg.get("reasoning") or "").strip()),
                          repr(msg.get("reasoning"))[:120])

            status, body = post(url, dict(
                base, tool_choice={"type": "function", "function": {"name": "get_weather"}},
                messages=[{"role": "user", "content": TOOL_PROMPT}]))
            msg = message(body)
            checker.check("named-200", status == 200, repr(body)[:200])
            checker.check("named-call", call_names(msg) == ["get_weather"],
                          repr(call_names(msg)))

            status, body = post(url, dict(base, tool_choice="auto",
                                          messages=[{"role": "user", "content": TOOL_PROMPT}]))
            msg = message(body)
            checker.check("auto-tool-200", status == 200, repr(body)[:200])
            checker.check("auto-tool-outcome",
                          (bool(call_names(msg))
                           and set(call_names(msg)) == {"get_weather"})
                          or is_ok_json(msg.get("content") or ""),
                          f"calls={call_names(msg)} content={repr(msg.get('content'))[:80]}")
            checker.check("auto-tool-no-markup",
                          all(m not in (msg.get("content") or "") for m in MARKUP),
                          repr(msg.get("content"))[:120])

            status, body = post(url, dict(base, tool_choice="auto",
                                          messages=[{"role": "user", "content": TEXT_PROMPT}]))
            msg = message(body)
            checker.check("auto-text-200", status == 200, repr(body)[:200])
            checker.check("auto-text-outcome",
                          is_ok_json(msg.get("content") or "")
                          or (bool(call_names(msg))
                              and set(call_names(msg)) == {"get_weather"}),
                          f"calls={call_names(msg)} content={repr(msg.get('content'))[:80]}")
            checker.check("auto-text-no-markup",
                          all(m not in (msg.get("content") or "") for m in MARKUP),
                          repr(msg.get("content"))[:120])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
