#!/usr/bin/env python3
"""Gate 8C: Responses Structured Output fail-closed transport.

Requires a LocalAI runtime carrying the PhaseShift Responses fail-closed patch.
Verifies that an unconvertible `text.format` is rejected with HTTP 400 before
any generation, that the backend and compute are never invoked for such
requests, and that valid structured / plain Responses requests keep working.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    backend_request_count,
    compute_pids,
    http_get_json,
    http_json,
    http_post_status,
    model_dir,
)

INVALID_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "__phaseshift_invalid_type__"}},
    "required": ["x"],
    "additionalProperties": False,
}

VALID_SCHEMA = {
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


def format_object(schema):
    return {"type": "json_schema", "name": "result", "strict": True, "schema": schema}


def payload(schema=VALID_SCHEMA, *, mode="canonical", stream=False, tools=None):
    body = {
        "model": "phaseshift",
        "input": "Answer with a JSON object.",
        "temperature": 0,
        "max_output_tokens": 64,
    }
    if schema is not None:
        fmt = format_object(schema)
        if mode == "canonical":
            body["text"] = {"format": fmt}
        elif mode == "legacy":
            body["text_format"] = fmt
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
    return body


def output_text(response):
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def responses_url(server):
    return f"{server.base_url}/responses"


def parsed_output(response):
    text = output_text(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> int:
    checker = Checker("localai-responses-fail-closed")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    request_log = Path("/tmp") / f"phaseshift-g8c-requests-{os.getpid()}.jsonl"
    if request_log.exists():
        request_log.unlink()

    try:
        with ServerHarness(max_concurrent_requests=3, max_seq_len=1024,
                           env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log)}) as server:
            # Both startup probes (Chat and Responses) must be fail-closed: the
            # invalid schemas are rejected by the frontend and never reach the
            # backend.
            checker.check("startup-probe-backend-not-invoked",
                          backend_request_count(request_log) == 0,
                          f"count={backend_request_count(request_log)}")

            # Warm up a plain request so compute is running before we measure
            # that invalid structured requests do not spawn or restart it.
            warmup = http_json(responses_url(server), {
                "model": "phaseshift", "input": "Say hello in one word.",
                "temperature": 0, "max_output_tokens": 8})
            checker.check("warmup-content", bool(output_text(warmup).strip()),
                          repr(output_text(warmup)))

            baseline = backend_request_count(request_log)
            deadline = time.monotonic() + 60.0
            pids_before = compute_pids()
            while not pids_before and time.monotonic() < deadline:
                time.sleep(0.5)
                pids_before = compute_pids()
            checker.check("compute-running", bool(pids_before), f"pids={pids_before}")

            # Invalid json_schema (non-stream).
            status, body = http_post_status(responses_url(server), payload(INVALID_SCHEMA))
            checker.check("invalid-schema-400", status == 400, f"status={status} body={body[:200]}")
            checker.check("invalid-schema-mentions-format",
                          "text.format" in body.lower() or "json_schema" in body.lower(),
                          body[:200])
            try:
                error_obj = json.loads(body)
                has_error_json = "error" in error_obj or "message" in error_obj
            except json.JSONDecodeError:
                has_error_json = False
            checker.check("invalid-schema-error-json", has_error_json, body[:200])
            checker.check("invalid-schema-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Invalid json_schema (stream): must reject before opening SSE.
            status, body = http_post_status(responses_url(server),
                                            payload(INVALID_SCHEMA, stream=True))
            checker.check("invalid-schema-stream-400", status == 400,
                          f"status={status} body={body[:200]}")
            checker.check("invalid-schema-stream-no-sse",
                          "data:" not in body and "[DONE]" not in body, body[:200])
            checker.check("invalid-schema-stream-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Unknown text.format type.
            status, body = http_post_status(responses_url(server), {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {"format": {"type": "__invalid__"}}})
            checker.check("unknown-format-400", status == 400, f"status={status} body={body[:200]}")
            checker.check("unknown-format-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Canonical and legacy surfaces cannot be combined.
            both = payload(VALID_SCHEMA)
            both["text_format"] = format_object(VALID_SCHEMA)
            status, body = http_post_status(responses_url(server), both)
            checker.check("both-surfaces-400", status == 400,
                          f"status={status} body={body[:200]}")
            checker.check("both-surfaces-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Malformed text object without a format.
            status, body = http_post_status(responses_url(server), {
                "model": "phaseshift", "input": "hi", "max_output_tokens": 8,
                "text": {}})
            checker.check("empty-text-400", status == 400, f"status={status} body={body[:200]}")

            # Structured output combined with tools is composed (Gate 9B).
            status, body = http_post_status(responses_url(server), payload(
                VALID_SCHEMA, tools=[TOOL]))
            checker.check("structured-tools-composed", status == 200,
                          f"status={status} body={body[:200]}")

            # Server survives the rejected requests.
            try:
                models = http_get_json(f"{server.base_url}/models")
                alive = any(m.get("id") == "phaseshift" for m in models.get("data", []))
            except Exception as exc:  # noqa: BLE001
                alive = False
                body = repr(exc)
            checker.check("server-alive-after-invalid", alive, locals().get("body", ""))
            checker.check("compute-pids-unchanged", compute_pids() == pids_before,
                          f"before={pids_before} after={compute_pids()}")

            # Valid canonical json_schema still enforces structured output.
            response = http_json(responses_url(server), payload(VALID_SCHEMA))
            parsed = parsed_output(response)
            checker.check("valid-schema-200",
                          isinstance(parsed, dict) and parsed.get("answer") in ("yes", "no"),
                          repr(output_text(response)))
            checker.check("valid-schema-backend-invoked",
                          backend_request_count(request_log) > baseline,
                          f"count={backend_request_count(request_log)}")

            # Legacy text_format alias keeps working.
            legacy = http_json(responses_url(server),
                               payload(VALID_SCHEMA, mode="legacy"))
            legacy_parsed = parsed_output(legacy)
            checker.check("legacy-alias-200",
                          isinstance(legacy_parsed, dict)
                          and legacy_parsed.get("answer") in ("yes", "no"),
                          repr(output_text(legacy)))

            # json_object format produces parseable JSON.
            json_object = http_json(responses_url(server), {
                "model": "phaseshift", "input": "Return a JSON object.",
                "temperature": 0, "max_output_tokens": 64,
                "text": {"format": {"type": "json_object"}}})
            checker.check("json-object-valid",
                          isinstance(parsed_output(json_object), (dict, list)),
                          repr(output_text(json_object)))

            # Plain request after invalid ones.
            plain = http_json(responses_url(server), {
                "model": "phaseshift", "input": "Say hello in one word.",
                "temperature": 0, "max_output_tokens": 8})
            checker.check("plain-after-invalid", bool(output_text(plain).strip()),
                          repr(output_text(plain)))

            # Explicit plain text format is unconstrained.
            text_format = http_json(responses_url(server), {
                "model": "phaseshift", "input": "Say hello in one word.",
                "temperature": 0, "max_output_tokens": 8,
                "text": {"format": {"type": "text"}}})
            checker.check("text-format-content", bool(output_text(text_format).strip()),
                          repr(output_text(text_format)))

            # Concurrent isolation: invalid + plain + valid.
            results: dict[str, object] = {}

            def run_invalid():
                results["invalid"] = http_post_status(
                    responses_url(server), payload(INVALID_SCHEMA))

            def run_plain():
                try:
                    results["plain"] = http_json(responses_url(server), {
                        "model": "phaseshift", "input": "Say hello in one word.",
                        "temperature": 0, "max_output_tokens": 8})
                except Exception as exc:  # noqa: BLE001
                    results["plain"] = exc

            def run_valid():
                try:
                    results["valid"] = http_json(
                        responses_url(server), payload(VALID_SCHEMA))
                except Exception as exc:  # noqa: BLE001
                    results["valid"] = exc

            threads = [threading.Thread(target=fn)
                       for fn in (run_invalid, run_plain, run_valid)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            checker.check("concurrent-invalid-400",
                          isinstance(results.get("invalid"), tuple)
                          and results["invalid"][0] == 400,
                          repr(results.get("invalid")))
            plain_result = results.get("plain")
            checker.check("concurrent-plain-200",
                          isinstance(plain_result, dict)
                          and bool(output_text(plain_result).strip()),
                          repr(plain_result))
            valid_result = results.get("valid")
            checker.check("concurrent-valid-200",
                          isinstance(valid_result, dict)
                          and '"answer"' in output_text(valid_result),
                          repr(valid_result))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
