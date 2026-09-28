#!/usr/bin/env python3
"""Gate 10B: prefix reuse across Responses turns, including function outputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json, model_dir  # noqa: E402

SYSTEM = ("You are a meticulous assistant that always verifies facts before "
          "answering and explains the reasoning behind each step in detail. ") * 8
USER = "大阪の天気をget_weatherで調べて。"
WEATHER = {
    "type": "function", "name": "get_weather", "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True,
}


def function_calls(response):
    return [item for item in response.get("output", [])
            if item.get("type") == "function_call"]


def output_text(response):
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def compute_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()


def hit_after(path: Path, offset: int) -> bool:
    return any("PREFIX_CACHE_HIT" in line for line in compute_lines(path)[offset:])


def main() -> int:
    checker = Checker("server-responses-prefix-cache")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    compute_log = Path("/tmp") / "phaseshift-g10b-responses-compute.log"
    if compute_log.exists():
        compute_log.unlink()
    server_log = Path("/tmp") / "phaseshift-g10b-responses-server.log"
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute_log)}

    try:
        with ServerHarness(max_seq_len=1024, log_path=server_log,
                           prefix_cache_capacity_tokens=2048,
                           prefix_cache_max_entries=8, env=env) as server:
            url = f"{server.base_url}/responses"

            # Plain Responses multi-turn.
            time.sleep(0.3)
            before = len(compute_lines(compute_log))
            first = http_json(url, {
                "model": "phaseshift",
                "input": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER}],
                "temperature": 0, "max_output_tokens": 32,
            })
            text = output_text(first)
            second = http_json(url, {
                "model": "phaseshift",
                "input": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER},
                          {"role": "assistant", "content": text},
                          {"role": "user", "content": "続けて。"}],
                "temperature": 0, "max_output_tokens": 32,
            })
            time.sleep(0.5)
            checker.check("plain-second", bool(output_text(second).strip()),
                          repr(output_text(second)))
            checker.check("plain-hit", hit_after(compute_log, before),
                          repr(compute_lines(compute_log)[before:]))

            # function_call_output round-trip.
            time.sleep(0.3)
            before = len(compute_lines(compute_log))
            first = http_json(url, {
                "model": "phaseshift",
                "input": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER}],
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER], "tool_choice": "required",
            })
            calls = function_calls(first)
            checker.check("tool-call", bool(calls), repr(first.get("output")))
            if calls:
                call = calls[0]
                follow = http_json(url, {
                    "model": "phaseshift",
                    "input": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER},
                        {"type": "function_call", "call_id": call["call_id"],
                         "name": call["name"], "arguments": call["arguments"]},
                        {"type": "function_call_output", "call_id": call["call_id"],
                         "output": "晴れ 25C"},
                    ],
                    "temperature": 0, "max_output_tokens": 64,
                    "tools": [WEATHER],
                })
                time.sleep(0.5)
                checker.check("tool-output-hit", hit_after(compute_log, before),
                              repr(compute_lines(compute_log)[before:]))
                checker.check("tool-output-text",
                              bool(output_text(follow).strip()),
                              repr(output_text(follow)))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
