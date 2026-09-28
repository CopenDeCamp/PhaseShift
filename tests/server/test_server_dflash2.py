#!/usr/bin/env python3
"""phaseshift-server の DFlash2 / psq4構成に対する E2E 検証.

port 8001 相当の起動、chat / streaming / responses の成功、および DFlash2 で
利用できない要件（structured output / tool calling / temperature > 0）が
fail-closed で HTTP 400 になることを確認する.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (Checker, ServerHarness, http_get_json,  # noqa: E402
                     http_json, http_post_status, localai_binary,
                     model_dir)

DFLASH2_MODEL_DIR = os.environ.get("PHASESHIFT_DFLASH2_MODEL_DIR")
KV_DTYPE = os.environ.get("PHASESHIFT_TEST_KV_DTYPE", "psq4")

STRUCTURED = {
    "type": "json_schema",
    "json_schema": {
        "name": "probe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        },
    },
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def main() -> int:
    checker = Checker("server-dflash2")
    harness = ServerHarness(max_seq_len=512, arena_gib=26, device=1,
                            max_concurrent_requests=1,
                            prefix_cache_capacity_tokens=0,
                            kv_cache_dtype=KV_DTYPE,
                            dflash2_model_dir=DFLASH2_MODEL_DIR,
                            startup_timeout=420.0)
    try:
        harness.start()

        log_text = ""
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            log_text = harness.log_path.read_text(errors="replace")
            if "Ready." in log_text or "failed" in log_text:
                break
            time.sleep(0.5)
        checker.check("server reached Ready", "Ready." in log_text, log_text[-500:])
        checker.check("log reports DFlash2",
                      "Speculative:  DFlash2" in log_text, log_text[-800:])
        checker.check("log reports kv dtype",
                      f"KV dtype:     {KV_DTYPE}" in log_text, log_text[-800:])
        checker.check("log disables tool calling",
                      "Tool calling: no (DFlash2 speculative decoding)" in log_text,
                      log_text[-800:])
        checker.check("log disables structured",
                      "Structured:   no (DFlash2 speculative decoding)" in log_text,
                      log_text[-800:])
        checker.check("log disables prefix cache",
                      "Prefix cache: off" in log_text, log_text[-800:])
        checker.check("log concurrency is 1",
                      "Concurrency:  1" in log_text, log_text[-800:])

        models = http_get_json(f"{harness.base_url}/models")
        checker.check("model listed",
                      any(m.get("id") == "phaseshift" for m in models.get("data", [])))

        chat = http_json(f"{harness.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": "Say exactly: hello"}],
            "temperature": 0,
            "max_tokens": 32,
        })
        content = chat["choices"][0]["message"]["content"]
        checker.check("chat returns content", "hello" in content, content)
        checker.check("chat finish_reason",
                      chat["choices"][0]["finish_reason"] in ("stop", "length"),
                      str(chat["choices"][0]["finish_reason"]))

        status, body = http_post_status(
            f"{harness.base_url}/chat/completions",
            {"model": "phaseshift",
             "messages": [{"role": "user", "content": "hi"}],
             "temperature": 0.7, "max_tokens": 8})
        checker.check("temperature > 0 rejected with 400", status == 400,
                      f"{status} {body[:200]}")
        checker.check("temperature rejection mentions greedy",
                      "greedy" in body, body[:200])

        status, body = http_post_status(
            f"{harness.base_url}/chat/completions",
            {"model": "phaseshift",
             "messages": [{"role": "user", "content": "json"}],
             "temperature": 0, "max_tokens": 8,
             "response_format": STRUCTURED})
        checker.check("structured rejected with 400", status == 400,
                      f"{status} {body[:200]}")

        status, body = http_post_status(
            f"{harness.base_url}/chat/completions",
            {"model": "phaseshift",
             "messages": [{"role": "user", "content": "weather"}],
             "temperature": 0, "max_tokens": 8, "tools": TOOLS})
        checker.check("tools rejected with 400", status == 400,
                      f"{status} {body[:200]}")

        responses = http_json(f"{harness.base_url}/responses", {
            "model": "phaseshift",
            "input": "Say exactly: ok",
            "temperature": 0,
            "max_output_tokens": 16,
        })
        text = "".join(
            item.get("content", [{}])[0].get("text", "")
            for item in responses.get("output", []) if item.get("type") == "message")
        checker.check("responses returns text", "ok" in text, text)

        status, body = http_post_status(
            f"{harness.base_url}/responses",
            {"model": "phaseshift", "input": "json", "temperature": 0,
             "max_output_tokens": 8, "text": {"format": STRUCTURED}})
        checker.check("responses structured rejected with 400", status == 400,
                      f"{status} {body[:200]}")

        request = urllib.request.Request(
            f"{harness.base_url}/chat/completions",
            data=json.dumps({
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "stream test"}],
                "temperature": 0, "max_tokens": 16, "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        chunks = []
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                payload = decoded[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                delta = event["choices"][0]["delta"].get("content")
                if delta:
                    chunks.append(delta)
        checker.check("stream returns content", len(chunks) > 0, str(chunks))
    finally:
        harness.close()

    return checker.done()


if __name__ == "__main__":
    if not DFLASH2_MODEL_DIR:
        print("SKIP: PHASESHIFT_DFLASH2_MODEL_DIR is not set")
        raise SystemExit(77)
    if localai_binary() is None:
        print("SKIP: LocalAI runtime not found")
        raise SystemExit(77)
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        raise SystemExit(77)
    raise SystemExit(main())
