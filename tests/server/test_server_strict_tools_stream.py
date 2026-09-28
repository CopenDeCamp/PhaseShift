#!/usr/bin/env python3
"""Gate 9A: strict function tool calling over Chat streaming."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_json,
    http_post_status,
    http_sse,
    model_dir,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get time",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def reconstruct(chunks):
    calls: dict[int, dict] = {}
    raw = ""
    for chunk in chunks:
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta") or {}
        for tc in delta.get("tool_calls") or []:
            entry = calls.setdefault(tc.get("index", 0),
                                     {"id": None, "name": "", "arguments": ""})
            if tc.get("id"):
                entry["id"] = tc["id"]
            function = tc.get("function") or {}
            if function.get("name"):
                entry["name"] = function["name"]
            if function.get("arguments"):
                entry["arguments"] += function["arguments"]
        raw += json.dumps(delta) + (delta.get("content") or "")
    return [calls[i] for i in sorted(calls)], raw


def main() -> int:
    checker = Checker("server-strict-tools-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/chat/completions"

            # named: exactly one call, only the named function.
            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "get_timeを必ず呼んで。get_weatherは使わないで。"}],
                "temperature": 0, "max_tokens": 128, "stream": True,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            }))
            calls, raw = reconstruct(events)
            checker.check("named-single", len(calls) == 1, repr(calls))
            if calls:
                checker.check("named-function", calls[0]["name"] == "get_weather", repr(calls[0]))
            checker.check("named-no-other-name", "get_time" not in raw, raw[-200:])
            checker.check("named-no-markup",
                          "<tool_call>" not in raw and "<function=" not in raw, raw[-200:])

            # required + parallel false: exactly one call.
            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "Just answer in plain text. Do not call tools."}],
                "temperature": 0, "max_tokens": 128, "stream": True,
                "tools": [WEATHER, TIME], "tool_choice": "required",
                "parallel_tool_calls": False,
            }))
            calls, _ = reconstruct(events)
            checker.check("required-single", len(calls) == 1, repr(calls))

            # strict auto: arguments stay schema valid.
            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪の天気をget_weatherで調べて。"}],
                "temperature": 0, "max_tokens": 128, "stream": True,
                "tools": [WEATHER], "tool_choice": "auto",
            }))
            calls, _ = reconstruct(events)
            checker.check("auto-at-least-one", len(calls) >= 1, repr(calls))
            valid = True
            for call in calls:
                try:
                    arguments = json.loads(call["arguments"])
                except json.JSONDecodeError:
                    valid = False
                    continue
                if not set(arguments) <= {"city", "unit"}:
                    valid = False
            checker.check("auto-args-valid", valid, repr(calls))

            # Invalid strict schema is rejected before generation. Streaming
            # responses cannot change the status after SSE starts, so the
            # frontend emits an invalid_request error event instead.
            bad = json.loads(json.dumps(WEATHER))
            bad["function"]["parameters"].pop("additionalProperties")
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "stream": True,
                "tools": [bad], "tool_choice": "auto",
            })
            rejected = (400 <= status < 500) or (
                '"invalid_request"' in body and "[DONE]" in body)
            checker.check("invalid-schema-stream-rejected", rejected,
                          f"status={status} body={body[:200]}")
            checker.check("invalid-schema-stream-no-call",
                          '"tool_calls"' not in body, body[:200])

            # non-stream / stream parity for the named choice.
            non_stream = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "get_timeを必ず呼んで。get_weatherは使わないで。"}],
                "temperature": 0, "max_tokens": 128,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            })
            non_calls = non_stream["choices"][0]["message"].get("tool_calls") or []
            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "get_timeを必ず呼んで。get_weatherは使わないで。"}],
                "temperature": 0, "max_tokens": 128, "stream": True,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            }))
            stream_calls, _ = reconstruct(events)
            checker.check("parity-count", len(non_calls) == len(stream_calls),
                          f"non={len(non_calls)} stream={len(stream_calls)}")
            if non_calls and stream_calls:
                checker.check("parity-name",
                              non_calls[0]["function"]["name"] == stream_calls[0]["name"],
                              repr((non_calls[0], stream_calls[0])))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
