#!/usr/bin/env python3
"""Gate 11A: plain Chat reasoning (non-stream)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

PROMPT = "What is 2+2? Answer with only the number."
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "r", "strict": True,
    "schema": {"type": "object", "properties": {"a": {"type": "string"}},
               "required": ["a"], "additionalProperties": False}}}


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def main() -> int:
    checker = Checker("server-reasoning")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift",
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0, "max_tokens": 512}

            status, absent = post(url, dict(base))
            checker.check("absent-200", status == 200, repr(absent)[:200])
            status, none = post(url, dict(base, reasoning_effort="none"))
            checker.check("none-200", status == 200, repr(none)[:200])
            checker.check("none-parity",
                          message(absent).get("content") == message(none).get("content"),
                          f"{message(absent).get('content')!r} vs "
                          f"{message(none).get('content')!r}")

            status, high = post(url, dict(base, reasoning_effort="high"))
            checker.check("high-200", status == 200, repr(high)[:200])
            hmsg = message(high)
            reasoning = hmsg.get("reasoning") or ""
            content = hmsg.get("content") or ""
            checker.check("high-reasoning-nonempty", bool(reasoning.strip()),
                          repr(hmsg)[:300])
            checker.check("high-content-nonempty", bool(content.strip()),
                          f"finish={high['choices'][0].get('finish_reason')} {content[:120]!r}")
            checker.check("reasoning-no-markup",
                          "<think>" not in reasoning and "</think>" not in reasoning,
                          repr(reasoning[:200]))
            checker.check("content-no-markup",
                          "<think>" not in content and "</think>" not in content,
                          repr(content[:200]))

            for level in ("xhigh", "max"):
                status, body = post(url, dict(base, reasoning_effort=level))
                msg = message(body)
                level_reasoning = msg.get("reasoning") or ""
                checker.check(f"{level}-200", status == 200, repr(body)[:200])
                checker.check(f"{level}-reasoning", bool(level_reasoning.strip()),
                              repr(msg)[:300])
                checker.check(f"{level}-no-markup",
                              "<think>" not in level_reasoning
                              and "</think>" not in level_reasoning,
                              repr(level_reasoning[:200]))

            status, invalid = post(url, dict(base, reasoning_effort="banana"))
            checker.check("invalid-400", status == 400, f"{status} {invalid}")

            status, tools = post(url, dict(base, max_tokens=512,
                                           reasoning_effort="high", tools=[WEATHER]))
            checker.check("tools-accepted", status == 200, f"{status} {tools}")
            status, structured = post(url, dict(base, max_tokens=512,
                                                reasoning_effort="high",
                                                response_format=SCHEMA))
            checker.check("structured-accepted", status == 200,
                          f"{status} {structured}")

            status, none_tools = post(url, dict(
                base, max_tokens=64, reasoning_effort="none", tools=[WEATHER],
                tool_choice="none"))
            checker.check("none-tools-ok", status == 200, f"{status} {none_tools}")
            status, none_structured = post(url, dict(
                base, max_tokens=64, reasoning_effort="none",
                response_format=SCHEMA))
            checker.check("none-structured-ok", status == 200,
                          f"{status} {none_structured}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
