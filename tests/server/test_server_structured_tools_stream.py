#!/usr/bin/env python3
"""Gate 9B: structured text + tool calling composition over Chat streaming."""

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
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "result", "strict": True, "schema": SCHEMA}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}


def reconstruct(chunks):
    calls: dict[int, dict] = {}
    content = ""
    for chunk in chunks:
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content += delta["content"]
        for tc in delta.get("tool_calls") or []:
            entry = calls.setdefault(tc.get("index", 0),
                                     {"name": "", "arguments": ""})
            function = tc.get("function") or {}
            if function.get("name"):
                entry["name"] = function["name"]
            if function.get("arguments"):
                entry["arguments"] += function["arguments"]
    return content, [calls[i] for i in sorted(calls)]


def main() -> int:
    checker = Checker("server-structured-tools-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/chat/completions"

            prompt_text = "Do not use tools. Return the requested structured result."
            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": True, "temperature": 0, "max_tokens": 64,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto"}))
            content, calls = reconstruct(events)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            checker.check("stream-text-structured",
                          isinstance(parsed, dict) and "answer" in parsed,
                          repr(content))
            checker.check("stream-text-no-calls", len(calls) == 0, repr(calls))

            # exact parity with the non-stream response.
            non_stream = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0, "max_tokens": 64,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto"})
            non_content = non_stream["choices"][0]["message"].get("content") or ""
            checker.check("stream-text-parity", content.strip() == non_content.strip(),
                          f"stream={content!r} non={non_content!r}")
            checker.check("stream-text-json-parity",
                          json.loads(content) == json.loads(non_content),
                          f"stream={content!r} non={non_content!r}")

            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "必ずget_weatherを使って確認して。"}],
                "stream": True, "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto", "parallel_tool_calls": False}))
            content, calls = reconstruct(events)
            checker.check("stream-tool-calls", len(calls) == 1, repr(calls))
            checker.check("stream-tool-pure", content.strip() == "", repr(content))
            if calls:
                checker.check("stream-tool-args-valid",
                              set(json.loads(calls[0]["arguments"])) <= {"city"},
                              repr(calls[0]))

            events = list(http_sse(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "必ずget_weatherを呼んで。"}],
                "stream": True, "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "required", "parallel_tool_calls": False}))
            content, calls = reconstruct(events)
            checker.check("stream-required-call", len(calls) == 1, repr(calls))
            checker.check("stream-required-pure", content.strip() == "", repr(content))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
