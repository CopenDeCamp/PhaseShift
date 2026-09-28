#!/usr/bin/env python3
"""Gate 11B: Responses reasoning + tool calling / round-trip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker, ServerHarness, http_json, http_sse, model_dir,
)

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "name": "get_weather", "description": "weather",
           "parameters": CITY, "strict": True}
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=")


def items(response, item_type):
    return [item for item in response.get("output", []) if item.get("type") == item_type]


def output_text(response):
    parts = []
    for item in items(response, "message"):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def reasoning_text(response):
    parts = []
    for item in items(response, "reasoning"):
        for content in item.get("content", []):
            parts.append(content.get("text", ""))
    return "".join(parts)


def main() -> int:
    checker = Checker("server-responses-reasoning-tools")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/responses"

            response = http_json(url, {
                "model": "phaseshift", "input": TOOL_PROMPT, "temperature": 0,
                "max_output_tokens": 512, "reasoning": {"effort": "high"},
                "tools": [WEATHER], "tool_choice": "required"})
            calls = items(response, "function_call")
            checker.check("required-reasoning-item", bool(items(response, "reasoning")),
                          repr(response.get("output"))[:300])
            checker.check("required-reasoning-text", bool(reasoning_text(response).strip()),
                          repr(reasoning_text(response))[:160])
            checker.check("required-call", len(calls) >= 1, repr(calls))
            if calls:
                checker.check("required-name",
                              all(c.get("name") == "get_weather" for c in calls),
                              repr([c.get("name") for c in calls]))
                args = json.loads(calls[0].get("arguments") or "{}")
                checker.check("required-args", set(args) <= {"city"}, repr(args))
            checker.check("required-no-markup",
                          all(m not in reasoning_text(response) for m in MARKUP),
                          repr(reasoning_text(response))[:160])

            events = list(http_sse(url, {
                "model": "phaseshift", "input": TOOL_PROMPT, "temperature": 0,
                "max_output_tokens": 1024, "reasoning": {"effort": "high"},
                "tools": [WEATHER], "tool_choice": "required", "stream": True}))
            types = [e.get("type") for e in events]
            added = [e for e in events if e.get("type") == "response.output_item.added"]
            reasoning_ids = [e.get("item", {}).get("id") for e in added
                             if e.get("item", {}).get("type") == "reasoning"]
            function_ids = [e.get("item", {}).get("id") for e in added
                            if e.get("item", {}).get("type") == "function_call"]
            checker.check("stream-reasoning-item", bool(reasoning_ids),
                          repr([e.get("item", {}).get("type") for e in added]))
            checker.check("stream-function-item", bool(function_ids),
                          repr([e.get("item", {}).get("type") for e in added]))
            added_types = [e.get("item", {}).get("type") for e in added]
            checker.check("stream-order",
                          added_types.index("reasoning") < added_types.index("function_call")
                          if ("reasoning" in added_types and "function_call" in added_types)
                          else False,
                          repr(added_types))
            delta_items = {e.get("item_id") for e in events
                           if e.get("type") == "response.output_text.delta"}
            checker.check("stream-reasoning-id-exclusive",
                          not (delta_items & set(function_ids)),
                          f"delta_items={delta_items} function={function_ids}")
            function_items = [e.get("item") for e in events
                              if e.get("type") == "response.output_item.done"
                              and e.get("item", {}).get("type") == "function_call"]
            checker.check("stream-function-args",
                          bool(function_items)
                          and all(bool(i.get("arguments")) for i in function_items),
                          repr(function_items[:2])[:200])

            if calls:
                call = calls[0]
                follow_up = http_json(url, {
                    "model": "phaseshift",
                    "input": [
                        {"role": "user", "content": TOOL_PROMPT},
                        {"type": "function_call", "call_id": call.get("call_id"),
                         "name": call.get("name"), "arguments": call.get("arguments")},
                        {"type": "function_call_output", "call_id": call.get("call_id"),
                         "output": "sunny 25C"},
                    ],
                    "temperature": 0, "max_output_tokens": 512,
                    "reasoning": {"effort": "high"}, "tools": [WEATHER]})
                checker.check("roundtrip-text",
                              bool(output_text(follow_up).strip())
                              or bool(items(follow_up, "function_call")),
                              repr(follow_up.get("output"))[:300])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
