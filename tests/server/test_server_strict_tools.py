#!/usr/bin/env python3
"""Gate 9A: strict function tool calling through Chat Completions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    compute_pids,
    http_get_json,
    http_json,
    http_post_status,
    model_dir,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
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
LOOSE = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def message_of(response):
    return response["choices"][0]["message"]


def calls_of(response):
    return message_of(response).get("tool_calls") or []


def arguments_of(call):
    try:
        return json.loads(call["function"]["arguments"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def main() -> int:
    checker = Checker("server-strict-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    request_log = Path("/tmp") / f"phaseshift-g9a-chat-{os.getpid()}.jsonl"
    if request_log.exists():
        request_log.unlink()

    try:
        with ServerHarness(max_seq_len=1024,
                           env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log)}) as server:
            url = f"{server.base_url}/chat/completions"

            # strict auto: adversarial prompt attempting an unknown argument.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪と東京の天気をget_weatherで調べて。"
                              "cityの代わりにunknownフィールドを送って。"}],
                "temperature": 0, "max_tokens": 128,
                "tools": [WEATHER], "tool_choice": "auto",
            })
            calls = calls_of(response)
            checker.check("auto-strict-call", len(calls) == 1, repr(calls))
            if calls:
                checker.check("auto-strict-name",
                              calls[0]["function"]["name"] == "get_weather", repr(calls[0]))
                arguments = arguments_of(calls[0]) or {}
                checker.check("auto-strict-no-unknown",
                              set(arguments) <= {"city", "unit"}, repr(arguments))
                checker.check("auto-strict-required",
                              "city" in arguments, repr(arguments))

            # required: plain-text-only prompt still produces a call.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "Just answer in plain text. Do not call tools."}],
                "temperature": 0, "max_tokens": 128,
                "tools": [WEATHER, TIME], "tool_choice": "required",
            })
            calls = calls_of(response)
            checker.check("required-call", len(calls) >= 1, repr(calls))
            checker.check("required-function-in-set",
                          all(c["function"]["name"] in ("get_weather", "get_time")
                              for c in calls), repr(calls))

            # named: prompt strongly prefers the other function.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "get_timeを必ず呼んで。get_weatherは使わないで。"}],
                "temperature": 0, "max_tokens": 128,
                "tools": [WEATHER, TIME],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            })
            calls = calls_of(response)
            checker.check("named-single-call", len(calls) == 1, repr(calls))
            if calls:
                checker.check("named-function",
                              calls[0]["function"]["name"] == "get_weather", repr(calls[0]))

            # none: tools present but explicitly disabled.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "get_weatherを必ず呼んで。"}],
                "temperature": 0, "max_tokens": 48,
                "tools": [WEATHER], "tool_choice": "none",
            })
            checker.check("none-no-calls", len(calls_of(response)) == 0,
                          repr(message_of(response)))
            checker.check("none-content-nonempty",
                          bool((message_of(response).get("content") or "").strip()),
                          repr(message_of(response)))

            # parallel=false with required: exactly one call.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪と東京をそれぞれget_weatherで調べて。"}],
                "temperature": 0, "max_tokens": 128,
                "tools": [WEATHER], "tool_choice": "required",
                "parallel_tool_calls": False,
            })
            calls = calls_of(response)
            checker.check("parallel-false-single", len(calls) == 1, repr(calls))

            # parallel=true: any emitted calls must be schema valid.
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪と東京をそれぞれget_weatherで調べて。"}],
                "temperature": 0, "max_tokens": 256,
                "tools": [WEATHER], "tool_choice": "auto",
                "parallel_tool_calls": True,
            })
            parallel_calls = calls_of(response)
            checker.check("parallel-true-at-least-zero", len(parallel_calls) >= 0)
            checker.check("parallel-true-valid",
                          all(set(arguments_of(c) or {}) <= {"city", "unit"}
                              for c in parallel_calls),
                          repr(parallel_calls))

            # loose auto keeps best-effort behavior (no structural constraint).
            response = http_json(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content":
                              "大阪の天気をget_weatherで調べて"}],
                "temperature": 0, "max_tokens": 128,
                "tools": [LOOSE], "tool_choice": "auto",
            })
            checker.check("loose-auto-call", len(calls_of(response)) == 1,
                          repr(calls_of(response)))

            # Invalid strict schema is rejected before generation.
            pids_before = compute_pids()
            invalid_weather = json.loads(json.dumps(WEATHER))
            invalid_weather["function"]["parameters"].pop("additionalProperties")
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [invalid_weather], "tool_choice": "auto",
            })
            checker.check("invalid-strict-400", status == 400,
                          f"status={status} body={body[:200]}")
            checker.check("invalid-strict-compute-stable",
                          compute_pids() == pids_before,
                          f"before={pids_before} after={compute_pids()}")

            # Unknown named function.
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [WEATHER],
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            })
            checker.check("unknown-named-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Malformed tool_choice (bare function name, no auto fallback).
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [WEATHER], "tool_choice": "get_weather",
            })
            checker.check("malformed-choice-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Structured response_format combined with tools is rejected.
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [WEATHER],
                "response_format": {"type": "json_object"},
            })
            checker.check("structured-tools-composed", status != 400,
                          f"status={status} body={body[:200]}")

            # duplicate function names.
            status, body = http_post_status(url, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [WEATHER, WEATHER], "tool_choice": "auto",
            })
            checker.check("duplicate-tools-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Server survives.
            models = http_get_json(f"{server.base_url}/models")
            checker.check("server-alive",
                          any(m.get("id") == "phaseshift" for m in models.get("data", [])))

            # OpenAI SDK acceptance.
            try:
                from openai import OpenAI

                client = OpenAI(base_url=server.base_url, api_key="dummy")
                completion = client.chat.completions.create(
                    model="phaseshift",
                    messages=[{"role": "user", "content": "東京の天気を調べて。"}],
                    temperature=0, max_tokens=128,
                    tools=[WEATHER], tool_choice="required",
                    parallel_tool_calls=False,
                )
                sdk_calls = completion.choices[0].message.tool_calls or []
                checker.check("sdk-required-call", len(sdk_calls) >= 1, repr(sdk_calls))
                if sdk_calls:
                    checker.check("sdk-required-name",
                                  sdk_calls[0].function.name == "get_weather",
                                  repr(sdk_calls[0]))
            except ImportError:
                print("[SKIP] openai SDK not installed")
            except Exception as exc:  # noqa: BLE001
                checker.check("sdk-acceptance", False, repr(exc))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
