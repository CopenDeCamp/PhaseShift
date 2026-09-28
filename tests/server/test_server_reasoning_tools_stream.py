#!/usr/bin/env python3
"""Gate 11B: Chat reasoning + tool calling streaming."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=", "</function>")


def stream(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    deltas = []
    finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            event = json.loads(line[6:])
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if delta:
                deltas.append(delta)
    return deltas, finish


def summarize(deltas):
    reasoning = "".join(d.get("reasoning") or "" for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    calls = []
    order = []
    for delta in deltas:
        if (delta.get("reasoning") or "").strip():
            order.append("r")
        if delta.get("tool_calls"):
            order.append("t")
            for call in delta["tool_calls"]:
                calls.append(call)
        if (delta.get("content") or "").strip():
            order.append("c")
    compressed = []
    for kind in order:
        if not compressed or compressed[-1] != kind:
            compressed.append(kind)
    return reasoning, content, calls, "".join(compressed)


def main() -> int:
    checker = Checker("server-reasoning-tools-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift", "temperature": 0, "max_tokens": 512,
                    "messages": [{"role": "user", "content": TOOL_PROMPT}],
                    "stream": True, "tools": [WEATHER], "tool_choice": "required"}

            deltas, finish = stream(url, dict(base, reasoning_effort="high"))
            reasoning, content, calls, order = summarize(deltas)
            checker.check("high-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            checker.check("high-calls", len(calls) >= 1, repr(calls)[:160])
            checker.check("high-name",
                          any((c.get("function") or {}).get("name") == "get_weather"
                              for c in calls), repr(calls)[:160])
            checker.check("high-order", order in ("rt", "r", "t"),
                          repr(order))
            checker.check("high-no-markup",
                          all(m not in reasoning and m not in content for m in MARKUP),
                          repr((reasoning + content)[-80:]))
            checker.check("high-finish", finish in ("tool_calls", "stop"), repr(finish))

            deltas, finish = stream(url, dict(base, reasoning_effort="none"))
            reasoning, content, calls, order = summarize(deltas)
            checker.check("none-no-reasoning", not reasoning, repr(reasoning)[:80])
            checker.check("none-calls", len(calls) >= 1, repr(calls)[:160])
            checker.check("none-order", order in ("t", "c", "tc", "ct"), repr(order))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
