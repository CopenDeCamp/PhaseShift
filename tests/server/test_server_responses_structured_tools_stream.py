#!/usr/bin/env python3
"""Gate 9B: structured text + tool calling composition over Responses streaming."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_json,
    http_sse,
    model_dir,
)

SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}
TEXT_FORMAT = {"type": "json_schema", "name": "result", "strict": True, "schema": SCHEMA}
WEATHER = {"type": "function", "name": "get_weather", "description": "weather",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"], "additionalProperties": False},
           "strict": True}


def stream_text(events):
    return "".join(event.get("delta", "") for event in events
                   if event.get("type") == "response.output_text.delta")


def stream_calls(events):
    return [event["item"] for event in events
            if event.get("type") == "response.output_item.done"
            and (event.get("item") or {}).get("type") == "function_call"]


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
    checker = Checker("server-responses-structured-tools-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/responses"

            prompt_text = "Do not use tools. Return the requested structured result."
            events = list(http_sse(url, {
                "model": "phaseshift", "input": prompt_text,
                "stream": True, "temperature": 0, "max_output_tokens": 64,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto"}))
            text = stream_text(events)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("stream-text-structured",
                          isinstance(parsed, dict) and "answer" in parsed, repr(text))
            checker.check("stream-text-no-calls", len(stream_calls(events)) == 0,
                          repr(stream_calls(events)))

            non_stream = http_json(url, {
                "model": "phaseshift", "input": prompt_text,
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto"})
            checker.check("stream-text-parity", text.strip() == output_text(non_stream).strip(),
                          f"stream={text!r} non={output_text(non_stream)!r}")
            checker.check("stream-text-json-parity",
                          json.loads(text) == json.loads(output_text(non_stream)),
                          f"stream={text!r} non={output_text(non_stream)!r}")

            events = list(http_sse(url, {
                "model": "phaseshift", "input": "必ずget_weatherを使って確認して。",
                "stream": True, "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto", "parallel_tool_calls": False}))
            calls = stream_calls(events)
            checker.check("stream-tool-calls", len(calls) == 1, repr(calls))
            checker.check("stream-tool-pure", stream_text(events).strip() == "",
                          repr(stream_text(events)))

            events = list(http_sse(url, {
                "model": "phaseshift", "input": "必ずget_weatherを呼んで。",
                "stream": True, "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "required", "parallel_tool_calls": False}))
            checker.check("stream-required-call", len(stream_calls(events)) == 1,
                          repr(stream_calls(events)))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
