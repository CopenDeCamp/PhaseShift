#!/usr/bin/env python3
"""Gate 9B: structured text + tool calling composition through Chat."""

from __future__ import annotations

import json
import os
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

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "result", "strict": True, "schema": SCHEMA}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "time",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}

NO_TOOLS_INSTRUCTION = (
    "You must not call any tool or function. Reply with a short JSON object "
    "matching the required schema.")


def message(response):
    return response["choices"][0]["message"]


def calls(response):
    return message(response).get("tool_calls") or []


def content(response):
    return message(response).get("content") or ""


def structured(response):
    try:
        parsed = json.loads(content(response))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and "answer" in parsed else None


def main() -> int:
    checker = Checker("server-structured-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/chat/completions"

            # none: structured text only.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "weather を調べずに answer を返して。"}],
                "temperature": 0, "max_tokens": 64,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "none",
            })
            checker.check("none-structured", structured(response) is not None,
                          repr(content(response)))
            checker.check("none-no-calls", len(calls(response)) == 0, repr(calls(response)))

            # required: tool calls only.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "必ずget_weatherを呼んで。"}],
                "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "required",
            })
            checker.check("required-calls", len(calls(response)) >= 1, repr(calls(response)))
            checker.check("required-no-structured",
                          "answer" not in content(response), repr(content(response)))

            # named: exactly one selected call.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "get_timeを必ず呼んで。"}],
                "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            })
            checker.check("named-single", len(calls(response)) == 1, repr(calls(response)))
            if calls(response):
                checker.check("named-name",
                              calls(response)[0]["function"]["name"] == "get_weather",
                              repr(calls(response)[0]))

            # auto text branch: an explicit no-tools instruction makes the model
            # deterministically take the structured-text side of the composed
            # constraint instead of the tool branch.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [
                    {"role": "system", "content": NO_TOOLS_INSTRUCTION},
                    {"role": "user", "content":
                     "Do not use tools. Return the requested structured result."}],
                "temperature": 0, "max_tokens": 256,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto",
            })
            checker.check("auto-text-structured", structured(response) is not None,
                          repr(content(response)))
            checker.check("auto-text-no-calls", len(calls(response)) == 0,
                          repr(calls(response)))

            # auto tool branch.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "必ずget_weatherを使って確認して。"}],
                "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto",
            })
            checker.check("auto-tool-calls", len(calls(response)) >= 1, repr(calls(response)))
            checker.check("auto-tool-pure", content(response).strip() == "",
                          repr(content(response)))

            # parallel=false with required: exactly one call.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪と東京をそれぞれget_weatherで調べて。"}],
                "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "required", "parallel_tool_calls": False,
            })
            checker.check("parallel-false-single", len(calls(response)) == 1,
                          repr(calls(response)))

            # Round-trip: tool call -> structured final answer.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "大阪の天気をget_weatherで調べて。"}],
                "temperature": 0, "max_tokens": 128,
                "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                "tool_choice": "auto", "parallel_tool_calls": False,
            })
            first_calls = calls(response)
            checker.check("roundtrip-first-call", len(first_calls) == 1,
                          repr(first_calls))
            if first_calls:
                call = first_calls[0]
                follow_up = http_json(url, {
                    "model": "phaseshift",
                    "messages": [
                        {"role": "user", "content": "大阪の天気をget_weatherで調べて。"},
                        {"role": "assistant", "content": None, "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": call.get("id"),
                         "name": call["function"]["name"],
                         "content": json.dumps({"temperature": 25, "condition": "sunny"})},
                    ],
                    "temperature": 0, "max_tokens": 64,
                    "response_format": RESPONSE_FORMAT, "tools": [WEATHER],
                    "tool_choice": "none",
                })
                checker.check("roundtrip-structured", structured(follow_up) is not None,
                              repr(content(follow_up)))
                checker.check("roundtrip-no-calls", len(calls(follow_up)) == 0,
                              repr(calls(follow_up)))

            # Rejection matrix.
            bad_response_format = json.loads(json.dumps(RESPONSE_FORMAT))
            bad_response_format["json_schema"]["schema"]["properties"]["answer"] = {
                "type": "__phaseshift_invalid_type__"}
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "response_format": bad_response_format,
                "tools": [WEATHER], "tool_choice": "auto",
            })
            checker.check("invalid-response-schema-400", status == 400,
                          f"status={status} body={body[:200]}")

            bad_tool = json.loads(json.dumps(WEATHER))
            bad_tool["function"]["parameters"].pop("additionalProperties")
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "response_format": RESPONSE_FORMAT,
                "tools": [bad_tool], "tool_choice": "auto",
            })
            checker.check("invalid-tool-schema-400", status == 400,
                          f"status={status} body={body[:200]}")

            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "response_format": RESPONSE_FORMAT,
                "tools": [WEATHER], "tool_choice": "get_weather",
            })
            checker.check("invalid-choice-400", status == 400,
                          f"status={status} body={body[:200]}")

            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "response_format": RESPONSE_FORMAT,
                "tools": [WEATHER],
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            })
            checker.check("unknown-named-400", status == 400,
                          f"status={status} body={body[:200]}")

            # OpenAI SDK Chat combinations.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="dummy")
                sdk = client.chat.completions.create(
                    model="phaseshift",
                    messages=[{"role": "user", "content": "必ずget_weatherを呼んで。"}],
                    temperature=0, max_tokens=128,
                    response_format=RESPONSE_FORMAT, tools=[WEATHER],
                    tool_choice="required",
                )
                checker.check("sdk-required-call",
                              len(sdk.choices[0].message.tool_calls or []) >= 1,
                              repr(sdk.choices[0].message.tool_calls))
                sdk_none = client.chat.completions.create(
                    model="phaseshift",
                    messages=[{"role": "user", "content": "answer を返して。"}],
                    temperature=0, max_tokens=64,
                    response_format=RESPONSE_FORMAT, tools=[WEATHER],
                    tool_choice="none",
                )
                checker.check("sdk-none-structured",
                              (sdk_none.choices[0].message.content or "").strip() != "",
                              repr(sdk_none.choices[0].message.content))
            except ImportError:
                print("[SKIP] openai SDK not installed")
            except Exception as exc:  # noqa: BLE001
                checker.check("sdk-acceptance", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
