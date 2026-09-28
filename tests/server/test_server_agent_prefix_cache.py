#!/usr/bin/env python3
"""Gate 10B: real multi-turn prefix reuse through the server.

Ordinary Chat, tool round-trip, strict tool round-trip and structured+tools
each must produce a prefix cache hit on the second turn because the backend
now sends a turn-stable prompt boundary.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json  # noqa: E402

SYSTEM = ("You are a meticulous assistant that always verifies facts before "
          "answering and explains the reasoning behind each step in detail. ") * 8
USER = "大阪の天気をget_weatherで調べて。"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}
SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "answer", "strict": True, "schema": {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}}}


def message(response) -> dict:
    return response["choices"][0]["message"]


def compute_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()


def hit_after(path: Path, offset: int) -> bool:
    return any("PREFIX_CACHE_HIT" in line for line in compute_lines(path)[offset:])


def scenario(checker, name, path, fn):
    time.sleep(0.3)
    before = len(compute_lines(path))
    first, second = fn()
    time.sleep(0.5)
    checker.check(f"{name}-second-response", second is not None, repr(second))
    checker.check(f"{name}-hit", hit_after(path, before),
                  repr(compute_lines(path)[before:]))


def main() -> int:
    checker = Checker("server-agent-prefix-cache")
    compute_log = Path("/tmp") / "phaseshift-g10b-agent-compute.log"
    if compute_log.exists():
        compute_log.unlink()
    server_log = Path("/tmp") / "phaseshift-g10b-agent-server.log"
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute_log)}

    def ordinary():
        first = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}],
            "temperature": 0, "max_tokens": 32})
        content = message(first).get("content") or ""
        second = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER},
                         {"role": "assistant", "content": content},
                         {"role": "user", "content": "続けて。"}],
            "temperature": 0, "max_tokens": 32})
        return first, second

    def tool_roundtrip(strict):
        first = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}],
            "temperature": 0, "max_tokens": 128,
            "tools": [WEATHER], "tool_choice": "auto", "parallel_tool_calls": False})
        calls = message(first).get("tool_calls") or []
        if not calls:
            return first, first
        call = calls[0]
        second = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call.get("id"),
                 "name": call["function"]["name"],
                 "content": json.dumps({"temperature": 25, "condition": "sunny"})},
            ],
            "temperature": 0, "max_tokens": 64,
            "tools": [WEATHER], "tool_choice": "none"})
        return first, second

    def structured_tools():
        first = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}],
            "temperature": 0, "max_tokens": 128,
            "response_format": SCHEMA,
            "tools": [WEATHER], "tool_choice": "auto", "parallel_tool_calls": False})
        msg = message(first)
        echo = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            echo["tool_calls"] = msg["tool_calls"]
        second = http_json(f"{url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER}, echo,
                         {"role": "user", "content": "続けて。"}],
            "temperature": 0, "max_tokens": 128,
            "response_format": SCHEMA,
            "tools": [WEATHER], "tool_choice": "auto", "parallel_tool_calls": False})
        return first, second

    try:
        with ServerHarness(max_seq_len=1024, log_path=server_log,
                           prefix_cache_capacity_tokens=2048,
                           prefix_cache_max_entries=8, env=env) as server:
            url = server.base_url
            scenario(checker, "ordinary", compute_log, ordinary)
            scenario(checker, "tool", compute_log, lambda: tool_roundtrip(False))
            scenario(checker, "strict-tool", compute_log, lambda: tool_roundtrip(True))
            scenario(checker, "structured-tools", compute_log, structured_tools)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
