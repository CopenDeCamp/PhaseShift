#!/usr/bin/env python3
"""Gate 9B: structured text + tool calling composition through Responses."""

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
TIME = {"type": "function", "name": "get_time", "description": "time",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"], "additionalProperties": False},
        "strict": True}


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


def structured(response):
    try:
        parsed = json.loads(output_text(response))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and "answer" in parsed else None


def main() -> int:
    checker = Checker("server-responses-structured-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/responses"

            response = http_json(url, {
                "model": "phaseshift", "input": "weather を調べずに answer を返して。",
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "none"})
            checker.check("none-structured", structured(response) is not None,
                          repr(output_text(response)))
            checker.check("none-no-calls", len(function_calls(response)) == 0,
                          repr(response.get("output")))

            response = http_json(url, {
                "model": "phaseshift", "input": "必ずget_weatherを呼んで。",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "required"})
            checker.check("required-calls", len(function_calls(response)) >= 1,
                          repr(response.get("output")))
            checker.check("required-no-structured", structured(response) is None,
                          repr(output_text(response)))

            response = http_json(url, {
                "model": "phaseshift", "input": "get_timeを必ず呼んで。",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "name": "get_weather"}})
            checker.check("named-single", len(function_calls(response)) == 1,
                          repr(function_calls(response)))
            if function_calls(response):
                checker.check("named-name", function_calls(response)[0]["name"] == "get_weather",
                              repr(function_calls(response)[0]))

            response = http_json(url, {
                "model": "phaseshift",
                "input": "Do not use tools. Return the requested structured result.",
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto"})
            checker.check("auto-text-structured", structured(response) is not None,
                          repr(output_text(response)))
            checker.check("auto-text-no-calls", len(function_calls(response)) == 0,
                          repr(response.get("output")))

            response = http_json(url, {
                "model": "phaseshift", "input": "必ずget_weatherを使って確認して。",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto"})
            checker.check("auto-tool-calls", len(function_calls(response)) >= 1,
                          repr(response.get("output")))
            checker.check("auto-tool-pure", output_text(response).strip() == "",
                          repr(output_text(response)))

            response = http_json(url, {
                "model": "phaseshift",
                "input": "大阪と東京をそれぞれget_weatherで調べて。",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "required", "parallel_tool_calls": False})
            checker.check("parallel-false-single", len(function_calls(response)) == 1,
                          repr(function_calls(response)))

            # function_call_output round-trip -> structured final.
            response = http_json(url, {
                "model": "phaseshift", "input": "大阪の天気をget_weatherで調べて。",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "auto", "parallel_tool_calls": False})
            calls = function_calls(response)
            checker.check("roundtrip-first-call", len(calls) == 1, repr(calls))
            if calls:
                call = calls[0]
                follow_up = http_json(url, {
                    "model": "phaseshift",
                    "input": [
                        {"role": "user", "content": "大阪の天気をget_weatherで調べて。"},
                        {"type": "function_call", "call_id": call["call_id"],
                         "name": call["name"], "arguments": call["arguments"]},
                        {"type": "function_call_output", "call_id": call["call_id"],
                         "output": "sunny 25C"},
                    ],
                    "temperature": 0, "max_output_tokens": 64,
                    "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                    "tool_choice": "none"})
                checker.check("roundtrip-structured", structured(follow_up) is not None,
                              repr(output_text(follow_up)))
                checker.check("roundtrip-no-calls", len(function_calls(follow_up)) == 0,
                              repr(follow_up.get("output")))

            # Rejection matrix.
            bad = json.loads(json.dumps(TEXT_FORMAT))
            bad["schema"]["properties"]["answer"] = {"type": "__phaseshift_invalid_type__"}
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {"format": bad}, "tools": [WEATHER], "tool_choice": "auto"})
            checker.check("invalid-response-schema-400", status == 400,
                          f"status={status} body={body[:200]}")

            bad_tool = json.loads(json.dumps(WEATHER))
            bad_tool["parameters"].pop("additionalProperties")
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {"format": TEXT_FORMAT}, "tools": [bad_tool],
                "tool_choice": "auto"})
            checker.check("invalid-tool-schema-400", status == 400,
                          f"status={status} body={body[:200]}")

            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {"format": TEXT_FORMAT}, "tools": [WEATHER],
                "tool_choice": "get_weather"})
            checker.check("invalid-choice-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Responses SDK.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="dummy")
                sdk = client.responses.create(
                    model="phaseshift", input="必ずget_weatherを呼んで。",
                    max_output_tokens=128, text={"format": TEXT_FORMAT},
                    tools=[WEATHER], tool_choice="required")
                checker.check("sdk-required-call", len(function_calls(sdk.model_dump())) >= 1,
                              repr(sdk.output))
            except ImportError:
                print("[SKIP] openai SDK not installed")
            except Exception as exc:  # noqa: BLE001
                checker.check("sdk-acceptance", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
