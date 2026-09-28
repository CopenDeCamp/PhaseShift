#!/usr/bin/env python3
"""Gate 8B: Chat Structured Output fail-closed transport.

Requires a LocalAI runtime carrying the PhaseShift fail-closed patch. Verifies
that an unconvertible `response_format` is rejected with HTTP 400 before any
generation, that the backend and compute are never invoked for such requests,
and that valid structured / plain requests keep working.
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


def schema_payload(schema, stream=False):
    payload = {
        "model": "phaseshift",
        "messages": [{"role": "user", "content": "Answer yes or no."}],
        "temperature": 0,
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        },
    }
    if stream:
        payload["stream"] = True
    return payload


def chat_url(server):
    return f"{server.base_url}/chat/completions"


def content_of(response):
    return response["choices"][0]["message"].get("content") or ""


def main() -> int:
    checker = Checker("localai-fail-closed")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    request_log = Path("/tmp") / f"phaseshift-g8b-requests-{os.getpid()}.jsonl"
    if request_log.exists():
        request_log.unlink()

    try:
        with ServerHarness(max_concurrent_requests=3, max_seq_len=1024,
                           env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log)}) as server:
            # The startup capability probe must itself be fail-closed: the invalid
            # schema is rejected by the frontend and never reaches the backend.
            checker.check("startup-probe-backend-not-invoked",
                          backend_request_count(request_log) == 0,
                          f"count={backend_request_count(request_log)}")

            # Warm up a plain request so compute is running before we measure that
            # invalid structured requests do not spawn or restart it.
            warmup = http_json(chat_url(server), {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 8,
            })
            checker.check("warmup-content", bool(content_of(warmup).strip()),
                          repr(content_of(warmup)))

            baseline = backend_request_count(request_log)
            deadline = time.monotonic() + 60.0
            pids_before = compute_pids()
            while not pids_before and time.monotonic() < deadline:
                time.sleep(0.5)
                pids_before = compute_pids()
            checker.check("compute-running", bool(pids_before), f"pids={pids_before}")

            # Invalid json_schema (non-stream).
            status, body = http_post_status(chat_url(server), schema_payload(INVALID_SCHEMA))
            checker.check("invalid-schema-400", status == 400, f"status={status} body={body[:200]}")
            checker.check("invalid-schema-mentions-format",
                          "response_format" in body.lower() or "json_schema" in body.lower(),
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
            stream_payload = schema_payload(INVALID_SCHEMA, stream=True)
            status, body = http_post_status(chat_url(server), stream_payload)
            checker.check("invalid-schema-stream-400", status == 400, f"status={status} body={body[:200]}")
            checker.check("invalid-schema-stream-no-sse",
                          "data:" not in body and "[DONE]" not in body, body[:200])
            checker.check("invalid-schema-stream-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Unknown response_format type.
            status, body = http_post_status(chat_url(server), {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "response_format": {"type": "__invalid__"},
            })
            checker.check("unknown-format-400", status == 400, f"status={status} body={body[:200]}")
            checker.check("unknown-format-backend-not-invoked",
                          backend_request_count(request_log) == baseline,
                          f"count={backend_request_count(request_log)}")

            # Missing schema payload.
            status, body = http_post_status(chat_url(server), {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "missing_schema", "strict": True},
                },
            })
            # LocalAI's converter accepts a missing schema as the empty object
            # schema, so this may legitimately succeed constrained to `{}`. It must
            # never panic or silently return free text.
            if status == 400:
                checker.check("missing-schema-handled", True)
            else:
                try:
                    parsed_missing = json.loads(body)
                    content = parsed_missing["choices"][0]["message"].get("content") or ""
                    json.loads(content)
                    constrained = True
                except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                    constrained = False
                checker.check("missing-schema-handled", status == 200 and constrained,
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

            # Valid json_schema still enforces structured output.
            response = http_json(chat_url(server), schema_payload(VALID_SCHEMA))
            text = content_of(response)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            checker.check("valid-schema-200",
                          isinstance(parsed, dict) and parsed.get("answer") in ("yes", "no"),
                          repr(text))
            checker.check("valid-schema-backend-invoked",
                          backend_request_count(request_log) > baseline,
                          f"count={backend_request_count(request_log)}")

            # Explicit text format is plain chat.
            response = http_json(chat_url(server), {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 8,
                "response_format": {"type": "text"},
            })
            checker.check("text-format-content", bool(content_of(response).strip()),
                          repr(content_of(response)))

            # Plain request after invalid ones.
            response = http_json(chat_url(server), {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 8,
            })
            checker.check("plain-after-invalid", bool(content_of(response).strip()),
                          repr(content_of(response)))

            # Concurrent isolation: invalid + plain + valid.
            results: dict[str, object] = {}

            def run_invalid():
                results["invalid"] = http_post_status(chat_url(server), schema_payload(INVALID_SCHEMA))

            def run_plain():
                try:
                    results["plain"] = http_json(chat_url(server), {
                        "model": "phaseshift",
                        "messages": [{"role": "user", "content": "Say hello in one word."}],
                        "temperature": 0,
                        "max_tokens": 8,
                    })
                except Exception as exc:  # noqa: BLE001
                    results["plain"] = exc

            def run_valid():
                try:
                    results["valid"] = http_json(chat_url(server), schema_payload(VALID_SCHEMA))
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
                          and bool(content_of(plain_result).strip()),
                          repr(plain_result))
            valid_result = results.get("valid")
            checker.check("concurrent-valid-200",
                          isinstance(valid_result, dict)
                          and '"answer"' in content_of(valid_result),
                          repr(valid_result))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
