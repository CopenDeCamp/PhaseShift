#!/usr/bin/env python3
"""Gate 11C: single-server public-surface smoke matrix.

Starts one canonical server and exercises the whole public surface. Everything
supported returns HTTP 200; only a malformed/unsupported request may return a
controlled 400.
"""

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
RESP_SCHEMA = {"type": "json_schema", "name": "ok", "strict": True,
               "schema": OK_SCHEMA["json_schema"]["schema"]}
RESP_WEATHER = {"type": "function", "name": "get_weather", "description": "weather",
                "parameters": CITY, "strict": True}
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=")


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))
    except (urllib.error.URLError, OSError):
        return 0, {}


def sse(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    text = ""
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            for choice in (json.loads(body).get("choices") or []):
                delta = choice.get("delta") or {}
                text += delta.get("content") or ""
    return text


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def main() -> int:
    checker = Checker("server-public-surface")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            chat_url = f"{server.base_url}/chat/completions"
            resp_url = f"{server.base_url}/responses"
            base = {"model": "phaseshift", "temperature": 0, "max_tokens": 512}

            status, body = post(chat_url, dict(base, reasoning_effort="none",
                                               messages=[{"role": "user",
                                                          "content": "Say hello."}]))
            checker.check("chat-plain", status == 200
                          and bool((message(body).get("content") or "").strip()),
                          repr(body)[:160])

            text = sse(chat_url, dict(base, reasoning_effort="none", stream=True,
                                      messages=[{"role": "user", "content": "Say hello."}]))
            checker.check("chat-stream", bool(text.strip()), repr(text)[:120])

            status, body = post(chat_url, dict(
                base, reasoning_effort="none", tools=[WEATHER], tool_choice="required",
                messages=[{"role": "user", "content": "Weather in Osaka?"}]))
            checker.check("chat-tool", status == 200, repr(body)[:160])

            status, body = post(chat_url, dict(
                base, reasoning_effort="none", response_format=OK_SCHEMA,
                messages=[{"role": "user", "content": "Return the JSON object."}]))
            checker.check("chat-structured", status == 200, repr(body)[:160])

            status, body = post(resp_url, {"model": "phaseshift", "temperature": 0,
                                           "max_output_tokens": 512,
                                           "input": "Say hello."})
            checker.check("responses-plain", status == 200, repr(body)[:160])

            status, body = post(resp_url, {"model": "phaseshift", "temperature": 0,
                                           "max_output_tokens": 512,
                                           "input": "Weather in Osaka?",
                                           "tools": [RESP_WEATHER], "tool_choice": "required"})
            checker.check("responses-tool", status == 200, repr(body)[:160])

            status, body = post(resp_url, {"model": "phaseshift", "temperature": 0,
                                           "max_output_tokens": 512,
                                           "input": "Return the JSON object.",
                                           "text": {"format": RESP_SCHEMA}})
            checker.check("responses-structured", status == 200, repr(body)[:160])

            status, body = post(chat_url, dict(base, reasoning_effort="none",
                                               messages=[{"role": "user",
                                                          "content": "What is 2+2?"}]))
            checker.check("reasoning-none", status == 200
                          and not message(body).get("reasoning"), repr(body)[:160])

            status, body = post(chat_url, dict(base, reasoning_effort="high",
                                               messages=[{"role": "user",
                                                          "content": "What is 2+2?"}]))
            checker.check("reasoning-high", status == 200
                          and bool((message(body).get("reasoning") or "").strip()),
                          repr(body)[:160])

            status, body = post(chat_url, dict(
                base, reasoning_effort="high", tools=[WEATHER],
                tool_choice="required", parallel_tool_calls=False,
                messages=[{"role": "user", "content": "Weather in Osaka?"}]))
            checker.check("reasoning-tool", status == 200
                          and bool(message(body).get("tool_calls")), repr(body)[:160])

            status, body = post(chat_url, dict(
                base, reasoning_effort="high", response_format=OK_SCHEMA,
                messages=[{"role": "user", "content": "Return the JSON object with ok true."}]))
            content = message(body).get("content") or ""
            checker.check("reasoning-structured", status == 200
                          and not any(m in content for m in MARKUP), repr(body)[:160])

            status, body = post(chat_url, dict(base, reasoning_effort="banana",
                                               messages=[{"role": "user",
                                                          "content": "Say hello."}]))
            checker.check("malformed-400", status == 400, f"{status} {repr(body)[:120]}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
