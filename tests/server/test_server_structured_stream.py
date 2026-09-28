#!/usr/bin/env python3
"""Gate 8R: streaming structured output transport + grammar enforcement (Chat).

Confirms the Chat streaming path applies the same grammar as non-streaming and
that the final streamed text equals the non-streaming text. Not evidence for
arbitrary JSON Schema conversion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, http_json, http_sse, model_dir  # noqa: E402


def stream_content(server, payload):
    text = ""
    chunks = 0
    for event in http_sse(f"{server.base_url}/chat/completions", payload):
        chunks += 1
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                text += delta["content"]
    return chunks, text


def main() -> int:
    checker = Checker("server-structured-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    try:
        with ServerHarness(max_concurrent_requests=2, max_seq_len=1024) as server:
            enum_schema = {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["answer"],
                "additionalProperties": False,
            }
            base = {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "Answer yes or no."}],
                "temperature": 0,
                "max_tokens": 128,
            }
            nonstream = dict(base)
            nonstream["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": enum_schema},
            }
            response = http_json(f"{server.base_url}/chat/completions", nonstream)
            nonstream_text = response["choices"][0]["message"].get("content") or ""

            stream = dict(base)
            stream["stream"] = True
            stream["response_format"] = nonstream["response_format"]
            chunks, stream_text = stream_content(server, stream)
            checker.check("stream-chunks", chunks >= 1, repr(chunks))
            try:
                parsed = json.loads(stream_text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("stream-json-parse", isinstance(parsed, dict), repr(stream_text))
            checker.check("stream-schema",
                          isinstance(parsed, dict) and parsed.get("answer") in ("yes", "no"),
                          repr(stream_text))
            checker.check("stream-nonstream-exact", stream_text == nonstream_text,
                          f"stream={stream_text!r} nonstream={nonstream_text!r}")

            # Streaming json_object.
            obj_payload = dict(base)
            obj_payload["stream"] = True
            obj_payload["response_format"] = {"type": "json_object"}
            _, obj_text = stream_content(server, obj_payload)
            try:
                obj_parsed = json.loads(obj_text)
            except json.JSONDecodeError:
                obj_parsed = None
            checker.check("stream-json-object", isinstance(obj_parsed, dict), repr(obj_text))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
