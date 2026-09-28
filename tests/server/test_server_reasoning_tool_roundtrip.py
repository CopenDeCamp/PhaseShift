#!/usr/bin/env python3
"""Gate 11B: reasoning + tool-result round-trip and prefix cache reuse."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

SYSTEM = ("You are a meticulous assistant that always verifies facts and "
          "explains each step of the reasoning in detail. ") * 8
CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
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


def turn1(url, system, parallel=True):
    payload = {"model": "phaseshift", "temperature": 0, "max_tokens": 512,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": TOOL_PROMPT}],
               "reasoning_effort": "high", "tools": [WEATHER],
               "tool_choice": "required", "parallel_tool_calls": parallel}
    return post(url, payload)


def turn2(url, system, assistant):
    calls = assistant.get("tool_calls") or []
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": TOOL_PROMPT},
                {"role": "assistant",
                 "content": assistant.get("content") or "",
                 "reasoning_content": assistant.get("reasoning") or "",
                 "tool_calls": calls}]
    for call in calls:
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "name": (call.get("function") or {}).get("name"),
                         "content": "sunny 25C"})
    return post(url, {"model": "phaseshift", "temperature": 0, "max_tokens": 512,
                      "messages": messages, "reasoning_effort": "high",
                      "tools": [WEATHER]})


def main() -> int:
    checker = Checker("server-reasoning-tool-roundtrip")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    compute = Path("/tmp") / "phaseshift-g11b-roundtrip-compute.log"
    requests = Path("/tmp") / "phaseshift-g11b-roundtrip-requests.log"
    for path in (compute, requests):
        if path.exists():
            path.unlink()
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute),
           "PHASESHIFT_BACKEND_REQUEST_LOG": str(requests)}
    try:
        with ServerHarness(max_seq_len=2048, prefix_cache_capacity_tokens=8192,
                           prefix_cache_max_entries=4, env=env) as server:
            url = f"{server.base_url}/chat/completions"
            status, first = turn1(url, SYSTEM, parallel=True)
            checker.check("turn1-200", status == 200, repr(first)[:200])
            assistant = message(first)
            calls = assistant.get("tool_calls") or []
            checker.check("turn1-call", len(calls) >= 1, repr(calls)[:160])
            checker.check("turn1-reasoning", bool((assistant.get("reasoning") or "").strip()),
                          repr(assistant.get("reasoning"))[:120])
            checker.check("turn1-call-id", bool(calls and calls[0].get("id")),
                          repr(calls[:1]))

            status, second = turn2(url, SYSTEM, assistant)
            checker.check("turn2-200", status == 200, repr(second)[:200])
            final = message(second)
            checker.check("turn2-reasoning", bool((final.get("reasoning") or "").strip()),
                          repr(final.get("reasoning"))[:120])
            checker.check("turn2-answer",
                          bool((final.get("content") or "").strip())
                          or bool(final.get("tool_calls")),
                          repr(final)[:200])
            checker.check("turn2-no-markup",
                          all(m not in (final.get("content") or "") for m in MARKUP),
                          repr(final.get("content"))[:160])
            time.sleep(0.5)

        compute_text = compute.read_text(errors="replace")
        checker.check("prefix-enabled", "PREFIX_CACHE_ENABLED=1" in compute_text,
                      compute_text[:160])
        checker.check("prefix-hit", "PREFIX_CACHE_HIT" in compute_text, compute_text[-500:])

        entries = [json.loads(line) for line in
                   requests.read_text(errors="replace").splitlines()]
        turn2_entry = entries[-1]
        assistant_messages = [m for m in turn2_entry["messages"]
                              if m.get("role") == "assistant"]
        checker.check("history-reasoning-transported",
                      any(m.get("reasoning_len", 0) > 0 for m in assistant_messages),
                      repr([m.get("reasoning_len") for m in assistant_messages]))
        checker.check("history-tool-calls-transported",
                      any(m.get("has_tool_calls") for m in assistant_messages),
                      repr(assistant_messages))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
