#!/usr/bin/env python3
"""Gate 11B: Chat reasoning + tool calling (non-stream)."""

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
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "time", "parameters": CITY, "strict": True}}
LOOSE = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY}}
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=", "</function>")


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


def call_args(msg):
    out = []
    for c in (msg.get("tool_calls") or []):
        args = (c.get("function") or {}).get("arguments")
        out.append(args if isinstance(args, dict) else json.loads(args or "{}"))
    return out


def has_markup(text):
    return any(m in (text or "") for m in MARKUP)


def main() -> int:
    checker = Checker("server-reasoning-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift", "temperature": 0, "max_tokens": 512,
                    "messages": [{"role": "user", "content": TOOL_PROMPT}],
                    "reasoning_effort": "high"}

            status, body = post(url, dict(base, tools=[WEATHER], tool_choice="required"))
            msg = message(body)
            checker.check("required-200", status == 200, repr(body)[:200])
            checker.check("required-reasoning", bool((msg.get("reasoning") or "").strip()),
                          repr(msg.get("reasoning"))[:160])
            checker.check("required-call", len(call_names(msg)) >= 1
                          and set(call_names(msg)) == {"get_weather"},
                          repr(call_names(msg)))
            args = call_args(msg)
            checker.check("required-args-object",
                          bool(args) and isinstance(args[0], dict)
                          and set(args[0].keys()) <= {"city"}, repr(args))
            checker.check("required-no-markup",
                          not has_markup(msg.get("reasoning")) and not has_markup(msg.get("content")),
                          repr((msg.get("reasoning") or "")[-80:]))

            status, body = post(url, dict(
                base, tools=[WEATHER, TIME],
                tool_choice={"type": "function", "function": {"name": "get_weather"}}))
            msg = message(body)
            checker.check("named-200", status == 200, repr(body)[:200])
            checker.check("named-call", call_names(msg) == ["get_weather"],
                          repr(call_names(msg)))
            checker.check("named-reasoning", bool((msg.get("reasoning") or "").strip()),
                          repr(msg.get("reasoning"))[:120])

            status, body = post(url, dict(base, tools=[WEATHER, TIME],
                                          tool_choice="required",
                                          parallel_tool_calls=False))
            msg = message(body)
            checker.check("parallel-false-200", status == 200, repr(body)[:200])
            checker.check("parallel-false-one", len(msg.get("tool_calls") or []) == 1,
                          repr(call_names(msg)))

            status, body = post(url, dict(base, tools=[LOOSE], tool_choice="auto"))
            checker.check("loose-auto-200", status == 200, repr(body)[:200])
            msg = message(body)
            checker.check("loose-auto-no-markup", not has_markup(msg.get("content")),
                          repr(msg.get("content"))[:120])

            none = {"model": "phaseshift", "temperature": 0, "max_tokens": 512,
                    "messages": [{"role": "user", "content": TOOL_PROMPT}],
                    "reasoning_effort": "none", "tools": [WEATHER],
                    "tool_choice": "required"}
            status, body = post(url, none)
            msg = message(body)
            checker.check("none-200", status == 200, repr(body)[:200])
            checker.check("none-no-reasoning", not msg.get("reasoning"), repr(msg)[:160])
            checker.check("none-call", bool(call_names(msg))
                          and set(call_names(msg)) == {"get_weather"},
                          repr(call_names(msg)))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
