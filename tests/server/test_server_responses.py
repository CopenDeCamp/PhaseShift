#!/usr/bin/env python3
"""Gate 5 acceptance: OpenAI Responses API through PhaseShift Server."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_json,
    http_sse,
    model_dir,
)

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
USER_MESSAGE = "大阪の天気をget_weatherで調べて"


def output_text(response: dict) -> str:
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def output_function_calls(response: dict) -> list[dict]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


def stream_text(events) -> str:
    parts = []
    for event in events:
        if event.get("type") == "response.output_text.delta":
            parts.append(event.get("delta", ""))
    return "".join(parts)


def stream_function_calls(events) -> list[dict]:
    calls = []
    for event in events:
        if event.get("type") == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                calls.append(item)
    return calls


def main() -> int:
    checker = Checker("server-responses")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    logdir = Path(tempfile.mkdtemp(prefix="ps-resplog-"))
    request_log = logdir / "requests.txt"

    try:
        with ServerHarness(env={
                "PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log),
                "PHASESHIFT_BACKEND_REQUEST_LOG_FULL": "1"}) as server:
            url = f"{server.base_url}/responses"

            # Case 1: simple string input.
            basic = http_json(url, {
                "model": "phaseshift", "input": "Reply with exactly: hello",
                "stream": False, "temperature": 0, "max_output_tokens": 16})
            checker.check("case1-nonempty-output",
                          bool(output_text(basic).strip()), repr(basic.get("output")))
            checker.check("case1-status", basic.get("status") == "completed",
                          repr(basic.get("status")))
            store_off = http_json(url, {
                "model": "phaseshift", "input": "hi", "stream": False,
                "store": False, "temperature": 0, "max_output_tokens": 8})
            checker.check("case1-store-false", store_off.get("status") == "completed",
                          repr(store_off.get("status")))

            # Case 2: typed message input.
            typed = http_json(url, {
                "model": "phaseshift",
                "input": [{"type": "message", "role": "user", "content": "Hello"}],
                "stream": False, "temperature": 0, "max_output_tokens": 16})
            checker.check("case2-typed-input", bool(output_text(typed).strip()),
                          repr(typed.get("output")))

            # Case 3: instructions reach the backend as a system message.
            instructed = http_json(url, {
                "model": "phaseshift",
                "instructions": "Always answer in French.",
                "input": "Say hi", "stream": False,
                "temperature": 0, "max_output_tokens": 16})
            checker.check("case3-instructions-output",
                          bool(output_text(instructed).strip()),
                          repr(instructed.get("output")))
            log_text = request_log.read_text() if request_log.is_file() else ""
            checker.check("case3-instructions-system-message",
                          "Always answer in French." in log_text,
                          log_text[-300:])

            # Case 4: streaming text parity.
            non_stream_text = output_text(basic)
            events = list(http_sse(url, {
                "model": "phaseshift", "input": "Reply with exactly: hello",
                "stream": True, "temperature": 0, "max_output_tokens": 16}))
            checker.check("case4-stream-parity",
                          stream_text(events) == non_stream_text,
                          f"stream={stream_text(events)!r} non={non_stream_text!r}")
            event_types = {e.get("type") for e in events}
            checker.check("case4-stream-events",
                          {"response.created", "response.output_text.delta",
                           "response.completed"} <= event_types,
                          repr(sorted(event_types)))

            # Case 5: non-stream function tool.
            tool_resp = http_json(url, {
                "model": "phaseshift", "input": USER_MESSAGE,
                "tools": [WEATHER_TOOL], "tool_choice": "auto",
                "temperature": 0, "max_output_tokens": 128})
            tool_calls = output_function_calls(tool_resp)
            checker.check("case5-function-call-present", len(tool_calls) == 1,
                          repr(tool_resp.get("output")))
            non_args = None
            if tool_calls:
                call = tool_calls[0]
                checker.check("case5-name", call.get("name") == "get_weather", repr(call))
                checker.check("case5-call-id", bool(call.get("call_id")), repr(call))
                try:
                    non_args = json.loads(call.get("arguments", "{}"))
                    checker.check("case5-arguments-json",
                                  isinstance(non_args, dict) and "city" in non_args,
                                  repr(call.get("arguments")))
                except json.JSONDecodeError as exc:
                    checker.check("case5-arguments-json", False, repr(exc))

            # Case 6: streaming function tool.
            tool_events = list(http_sse(url, {
                "model": "phaseshift", "input": USER_MESSAGE,
                "tools": [WEATHER_TOOL], "tool_choice": "auto",
                "stream": True, "temperature": 0, "max_output_tokens": 128}))
            stream_calls = stream_function_calls(tool_events)
            checker.check("case6-stream-function-call", len(stream_calls) == 1,
                          repr(stream_calls))
            if stream_calls:
                stream_args = json.loads(stream_calls[0].get("arguments", "{}"))
                checker.check("case6-stream-arguments",
                              isinstance(stream_args, dict) and "city" in stream_args,
                              repr(stream_calls[0].get("arguments")))
                checker.check("case6-stream-name",
                              stream_calls[0].get("name") == "get_weather",
                              repr(stream_calls[0]))
                checker.check("case6-stream-nonstream-parity",
                              non_args == stream_args,
                              f"{non_args} vs {stream_args}")

            # Case 7: function_call_output round-trip.
            if tool_calls:
                call_id = tool_calls[0].get("call_id")
                round_trip = http_json(url, {
                    "model": "phaseshift",
                    "input": [
                        {"role": "user", "content": USER_MESSAGE},
                        {"type": "function_call", "call_id": call_id,
                         "name": "get_weather",
                         "arguments": tool_calls[0].get("arguments", "{}")},
                        {"type": "function_call_output", "call_id": call_id,
                         "output": json.dumps({"temperature": 22, "unit": "celsius"})},
                    ],
                    "tools": [WEATHER_TOOL],
                    "temperature": 0, "max_output_tokens": 64})
                final_text = output_text(round_trip)
                checker.check("case7-final-text", bool(final_text.strip()),
                              repr(round_trip.get("output")))
                checker.check("case7-no-function-call",
                              not output_function_calls(round_trip),
                              repr(round_trip.get("output")))

            # Case 8: Chat Completions parity.
            chat = http_json(f"{server.base_url}/chat/completions", {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": USER_MESSAGE}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Get weather",
                    "parameters": WEATHER_TOOL["parameters"]}}],
                "tool_choice": "auto", "temperature": 0, "max_tokens": 128})
            chat_calls = chat["choices"][0]["message"].get("tool_calls") or []
            if chat_calls and tool_calls:
                chat_fn = chat_calls[0]["function"]
                chat_args = json.loads(chat_fn["arguments"] or "{}")
                checker.check("case8-chat-parity", chat_fn["name"] == "get_weather"
                              and chat_args == non_args,
                              f"{chat_fn} vs {tool_calls[0]}")

            # Case 9/10: OpenAI Python SDK.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="unused")
                sdk = client.responses.create(
                    model="phaseshift", input="Reply with exactly: hello",
                    temperature=0, max_output_tokens=16)
                checker.check("case9-sdk-nonempty-text",
                              bool(getattr(sdk, "output_text", "").strip()),
                              repr(getattr(sdk, "output_text", None)))

                sdk_events = list(client.responses.create(
                    model="phaseshift", input="Reply with exactly: hello",
                    temperature=0, max_output_tokens=16, stream=True))
                sdk_text = "".join(
                    getattr(e, "delta", "") for e in sdk_events
                    if getattr(e, "type", None) == "response.output_text.delta")
                checker.check("case10-sdk-stream-text",
                              sdk_text == getattr(sdk, "output_text", None),
                              f"sdk_stream={sdk_text!r}")
            except ImportError as exc:
                checker.check("case9-sdk-nonempty-text", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
