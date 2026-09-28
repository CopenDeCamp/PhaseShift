#!/usr/bin/env python3
"""Gate 8C: Responses structured output streaming.

Confirms that structured `text.format` constrains streamed output and that
invalid formats are rejected before any SSE event is emitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_post_status,
    http_sse,
    model_dir,
)

ENUM_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def stream_text(events):
    return "".join(event.get("delta", "")
                   for event in events
                   if event.get("type") == "response.output_text.delta")


def completed_response(events):
    for event in events:
        if event.get("type") == "response.completed":
            return event.get("response") or {}
    return {}


def main() -> int:
    checker = Checker("server-responses-structured-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_seq_len=1024) as server:
            url = f"{server.base_url}/responses"

            # Canonical json_schema over SSE.
            events = list(http_sse(url, {
                "model": "phaseshift", "input": "Answer yes or no.",
                "stream": True, "temperature": 0, "max_output_tokens": 64,
                "text": {"format": {
                    "type": "json_schema", "name": "result", "strict": True,
                    "schema": ENUM_SCHEMA}}}))
            text = stream_text(events)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("stream-schema-valid",
                          isinstance(parsed, dict) and parsed.get("answer") in ("yes", "no"),
                          repr(text))
            event_types = {event.get("type") for event in events}
            checker.check("stream-event-types",
                          {"response.created", "response.output_text.delta",
                           "response.completed"} <= event_types,
                          repr(sorted(event_types)))
            completed = completed_response(events)
            fmt = ((completed.get("text") or {}).get("format") or {})
            checker.check("stream-schema-echo", fmt.get("type") == "json_schema",
                          repr(completed.get("text")))

            # json_object over SSE.
            events = list(http_sse(url, {
                "model": "phaseshift", "input": "Return a JSON object.",
                "stream": True, "temperature": 0, "max_output_tokens": 64,
                "text": {"format": {"type": "json_object"}}}))
            try:
                json_object = json.loads(stream_text(events))
            except json.JSONDecodeError:
                json_object = None
            checker.check("stream-json-object-valid",
                          isinstance(json_object, (dict, list)),
                          repr(stream_text(events)))

            # Invalid schema over SSE must not open a stream.
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "stream": True,
                "max_output_tokens": 8,
                "text": {"format": {
                    "type": "json_schema", "name": "bad", "strict": True,
                    "schema": {"type": "object",
                               "properties": {"x": {"type": "__phaseshift_invalid_type__"}},
                               "required": ["x"]}}}})
            checker.check("stream-invalid-400", status == 400,
                          f"status={status} body={body[:200]}")
            checker.check("stream-invalid-no-sse",
                          "data:" not in body and "[DONE]" not in body, body[:200])

            # Ambiguous surfaces over SSE.
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "stream": True,
                "max_output_tokens": 8,
                "text": {"format": {"type": "json_object"}},
                "text_format": {"type": "json_object"}})
            checker.check("stream-both-400", status == 400,
                          f"status={status} body={body[:200]}")

            # Structured output with tools over SSE.
            status, body = http_post_status(url, {
                "model": "phaseshift", "input": "hi", "stream": True,
                "max_output_tokens": 8,
                "tools": [{"type": "function", "name": "get_weather",
                           "description": "Get weather",
                           "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}}}}],
                "text": {"format": {"type": "json_object"}}})
            checker.check("stream-compose-structured-tools", status == 200,
                          f"status={status} body={body[:200]}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
