#!/usr/bin/env python3
"""Gate 8R: Chat Completions structured output transport + grammar enforcement.

This validates the Chat path only: LocalAI converts `response_format` into a
GBNF grammar, the backend forwards it, and PhaseShift enforces that grammar at
decode time. It is NOT evidence that arbitrary JSON Schema always converts;
schemas here are the ones LocalAI successfully converts. Responses Structured
Output is out of scope (BLOCKED_EXTERNAL).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json, model_dir  # noqa: E402


def chat_json(server, payload):
    return http_json(f"{server.base_url}/chat/completions", payload)


def schema_payload(schema, name="result", prompt="Produce the JSON value."):
    return {
        "model": "phaseshift",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    }


def content_of(response):
    return response["choices"][0]["message"].get("content") or ""


def main() -> int:
    checker = Checker("server-structured")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_concurrent_requests=2, max_seq_len=1024) as server:
            # Case 1: json_object.
            response = chat_json(server, {
                "model": "phaseshift",
                "messages": [{"role": "user",
                              "content": "Return a JSON object with a field ok set to true."}],
                "temperature": 0,
                "max_tokens": 128,
                "response_format": {"type": "json_object"},
            })
            text = content_of(response)
            try:
                parsed = json.loads(text)
                object_ok = isinstance(parsed, dict)
            except json.JSONDecodeError:
                object_ok = False
            checker.check("json-object-parse", object_ok, repr(text))

            # Case 2: enum object.
            enum_schema = {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["answer"],
                "additionalProperties": False,
            }
            response = chat_json(server, schema_payload(
                enum_schema, prompt="Answer yes or no."))
            text = content_of(response)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("enum-parse", isinstance(parsed, dict), repr(text))
            checker.check("enum-value",
                          isinstance(parsed, dict) and parsed.get("answer") in ("yes", "no"),
                          repr(text))

            # Case 3: integer field.
            integer_schema = {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            }
            response = chat_json(server, schema_payload(
                integer_schema, prompt="Count the number of letters in 'hello'."))
            text = content_of(response)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("integer-parse", isinstance(parsed, dict), repr(text))
            checker.check("integer-type",
                          isinstance(parsed, dict) and isinstance(parsed.get("count"), int)
                          and not isinstance(parsed.get("count"), bool), repr(text))

            # Case 4: array of integer.
            array_schema = {
                "type": "object",
                "properties": {"values": {"type": "array", "items": {"type": "integer"}}},
                "required": ["values"],
                "additionalProperties": False,
            }
            response = chat_json(server, schema_payload(
                array_schema, prompt="List three small integers."))
            text = content_of(response)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("array-parse", isinstance(parsed, dict), repr(text))
            checker.check("array-type",
                          isinstance(parsed, dict) and isinstance(parsed.get("values"), list)
                          and all(isinstance(v, int) and not isinstance(v, bool)
                                  for v in parsed.get("values", [])), repr(text))

            # Case 5: nested object.
            nested_schema = {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "additionalProperties": False,
                    }
                },
                "required": ["user"],
                "additionalProperties": False,
            }
            response = chat_json(server, schema_payload(
                nested_schema, prompt="Emit a user with a name and age."))
            text = content_of(response)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("nested-parse", isinstance(parsed, dict), repr(text))
            checker.check("nested-fields",
                          isinstance(parsed, dict) and isinstance(parsed.get("user"), dict)
                          and isinstance(parsed["user"].get("name"), str)
                          and isinstance(parsed["user"].get("age"), int),
                          repr(text))

            # Unconstrained request still works and is not forced into JSON.
            plain = chat_json(server, {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 16,
            })
            checker.check("unconstrained-content", bool(content_of(plain).strip()),
                          repr(content_of(plain)))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
