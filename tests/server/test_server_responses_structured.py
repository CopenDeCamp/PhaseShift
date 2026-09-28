#!/usr/bin/env python3
"""Gate 8C: Responses structured output semantics.

Covers the canonical `text.format` surface, the legacy `text_format` alias, the
response echo, and the rejection matrix for unsupported or malformed formats.
"""

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

ENUM_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]},
}


def output_text(response):
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def echo_type(response):
    text = response.get("text") or {}
    fmt = text.get("format") or {}
    return fmt.get("type")


def parsed(response):
    try:
        return json.loads(output_text(response))
    except json.JSONDecodeError:
        return None


def main() -> int:
    checker = Checker("server-responses-structured")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/responses"

            # Canonical json_schema.
            response = http_json(url, {
                "model": "phaseshift", "input": "Answer yes or no.",
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": {
                    "type": "json_schema", "name": "result", "strict": True,
                    "schema": ENUM_SCHEMA}}})
            body = parsed(response)
            checker.check("canonical-schema-valid",
                          isinstance(body, dict) and body.get("answer") in ("yes", "no"),
                          repr(output_text(response)))
            checker.check("canonical-schema-echo", echo_type(response) == "json_schema",
                          repr(response.get("text")))

            # Legacy alias produces the same constrained result.
            legacy = http_json(url, {
                "model": "phaseshift", "input": "Answer yes or no.",
                "temperature": 0, "max_output_tokens": 64,
                "text_format": {
                    "type": "json_schema", "name": "result", "strict": True,
                    "schema": ENUM_SCHEMA}})
            legacy_body = parsed(legacy)
            checker.check("legacy-schema-valid",
                          isinstance(legacy_body, dict)
                          and legacy_body.get("answer") in ("yes", "no"),
                          repr(output_text(legacy)))
            checker.check("legacy-schema-echo", echo_type(legacy) == "json_schema",
                          repr(legacy.get("text")))

            # json_object.
            json_object = http_json(url, {
                "model": "phaseshift", "input": "Return a JSON object.",
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": {"type": "json_object"}}})
            checker.check("json-object-valid",
                          isinstance(parsed(json_object), (dict, list)),
                          repr(output_text(json_object)))
            checker.check("json-object-echo", echo_type(json_object) == "json_object",
                          repr(json_object.get("text")))

            # Explicit text.
            text_format = http_json(url, {
                "model": "phaseshift", "input": "Say hello in one word.",
                "temperature": 0, "max_output_tokens": 8,
                "text": {"format": {"type": "text"}}})
            checker.check("text-format-content", bool(output_text(text_format).strip()),
                          repr(output_text(text_format)))
            checker.check("text-format-echo", echo_type(text_format) == "text",
                          repr(text_format.get("text")))

            # Plain request defaults to text.
            plain = http_json(url, {
                "model": "phaseshift", "input": "Say hello in one word.",
                "temperature": 0, "max_output_tokens": 8})
            checker.check("plain-content", bool(output_text(plain).strip()),
                          repr(output_text(plain)))
            checker.check("plain-echo-text", echo_type(plain) == "text",
                          repr(plain.get("text")))

            # Rejection matrix.
            both = {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {"format": {"type": "json_object"}},
                "text_format": {"type": "json_object"}}
            status, err = http_post_status(url, both)
            checker.check("both-surfaces-400", status == 400, f"status={status} body={err[:200]}")

            for label, fmt in (
                ("unknown-type", {"type": "__invalid__"}),
                ("missing-name", {"type": "json_schema", "schema": ENUM_SCHEMA}),
                ("missing-schema", {"type": "json_schema", "name": "result"}),
                ("missing-type", {"name": "result"}),
            ):
                status, err = http_post_status(url, {
                    "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                    "text": {"format": fmt}})
                checker.check(f"reject-{label}", status == 400,
                              f"status={status} body={err[:200]}")

            status, err = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {}})
            checker.check("reject-empty-text", status == 400,
                          f"status={status} body={err[:200]}")

            # Structured output with tools is rejected.
            status, err = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "tools": [TOOL],
                "text": {"format": {"type": "json_object"}}})
            checker.check("compose-structured-tools", status == 200,
                          f"status={status} body={err[:200]}")

            # Explicit text with tools keeps tool calling working.
            tool_response = http_json(url, {
                "model": "phaseshift",
                "input": "大阪の天気をget_weatherで調べて",
                "tools": [TOOL], "tool_choice": "auto",
                "temperature": 0, "max_output_tokens": 128,
                "text": {"format": {"type": "text"}}})
            calls = [item for item in tool_response.get("output", [])
                     if item.get("type") == "function_call"]
            checker.check("text-with-tools-call", len(calls) == 1,
                          repr(tool_response.get("output")))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
