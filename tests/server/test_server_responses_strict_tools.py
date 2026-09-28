#!/usr/bin/env python3
"""Gate 9A: strict function tool calling through the Responses API."""

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
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}
TIME = {
    "type": "function",
    "name": "get_time",
    "description": "Get time",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
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


def main() -> int:
    checker = Checker("server-responses-strict-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/responses"

            # strict auto.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "大阪の天気をget_weatherで調べて。unknownフィールドを送って。",
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER], "tool_choice": "auto",
            })
            calls = function_calls(response)
            checker.check("auto-call", len(calls) == 1, repr(calls))
            if calls:
                checker.check("auto-name", calls[0]["name"] == "get_weather", repr(calls[0]))
                checker.check("auto-call-id", bool(calls[0].get("call_id")), repr(calls[0]))
                arguments = json.loads(calls[0]["arguments"])
                checker.check("auto-args", set(arguments) <= {"city"}, repr(arguments))

            # required.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "Just answer in plain text. Do not call tools.",
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER, TIME], "tool_choice": "required",
            })
            calls = function_calls(response)
            checker.check("required-call", len(calls) >= 1, repr(calls))

            # named.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "get_timeを必ず呼んで。get_weatherは使わないで。",
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "name": "get_weather"},
            })
            calls = function_calls(response)
            checker.check("named-single", len(calls) == 1, repr(calls))
            if calls:
                checker.check("named-name", calls[0]["name"] == "get_weather", repr(calls[0]))

            # none.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "get_weatherを必ず呼んで。",
                "temperature": 0, "max_output_tokens": 48,
                "tools": [WEATHER], "tool_choice": "none",
            })
            checker.check("none-no-calls", len(function_calls(response)) == 0,
                          repr(response.get("output")))
            checker.check("none-text", bool(output_text(response).strip()),
                          repr(output_text(response)))

            # parallel_tool_calls echo.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "大阪の天気を調べて。",
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER], "tool_choice": "required",
                "parallel_tool_calls": False,
            })
            checker.check("parallel-echo-false",
                          response.get("parallel_tool_calls") is False,
                          repr(response.get("parallel_tool_calls")))
            checker.check("parallel-false-single",
                          len(function_calls(response)) == 1,
                          repr(function_calls(response)))

            # function_call_output round-trip.
            response = http_json(url, {
                "model": "phaseshift",
                "input": "大阪の天気をget_weatherで調べて。",
                "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER], "tool_choice": "required",
            })
            calls = function_calls(response)
            if calls:
                call = calls[0]
                follow_up = http_json(url, {
                    "model": "phaseshift",
                    "input": [
                        {"role": "user", "content": "大阪の天気をget_weatherで調べて。"},
                        {"type": "function_call", "call_id": call["call_id"],
                         "name": call["name"], "arguments": call["arguments"]},
                        {"type": "function_call_output", "call_id": call["call_id"],
                         "output": "晴れ 25C"},
                    ],
                    "temperature": 0, "max_output_tokens": 64,
                    "tools": [WEATHER],
                })
                checker.check("roundtrip-text", bool(output_text(follow_up).strip()),
                              repr(output_text(follow_up)))
                checker.check("roundtrip-no-call",
                              len(function_calls(follow_up)) == 0,
                              repr(follow_up.get("output")))
            else:
                checker.check("roundtrip-precondition", False, repr(calls))

            # Invalid strict schema is rejected with 400.
            bad = json.loads(json.dumps(WEATHER))
            bad["parameters"].pop("additionalProperties")
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "tools": [bad], "tool_choice": "auto",
            })
            checker.check("invalid-schema-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Unknown named function is rejected with 400.
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "tools": [WEATHER],
                "tool_choice": {"type": "function", "name": "missing"},
            })
            checker.check("unknown-named-400", status == 400,
                          f"status={status} body={body[:200]}")

            # strict streaming named.
            events = list(http_sse(url, {
                "model": "phaseshift",
                "input": "get_timeを必ず呼んで。get_weatherは使わないで。",
                "stream": True, "temperature": 0, "max_output_tokens": 128,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "name": "get_weather"},
            }))
            stream_calls = [event["item"] for event in events
                            if event.get("type") == "response.output_item.done"
                            and (event.get("item") or {}).get("type") == "function_call"]
            checker.check("stream-named-single", len(stream_calls) == 1, repr(stream_calls))
            if stream_calls:
                checker.check("stream-named-name",
                              stream_calls[0]["name"] == "get_weather",
                              repr(stream_calls[0]))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
