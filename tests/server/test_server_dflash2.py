#!/usr/bin/env python3
"""phaseshift-server の DFlash2 / psq4構成に対する E2E 検証.

port 8001 相当の起動、chat / streaming / responses の成功、および DFlash2 でも
structured output / tool calling / stochastic sampling / prefix cache が
有効であることを確認する.
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
RESPONSES_STRUCTURED = {
    "type": "json_schema",
    "name": "probe",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
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
    harness = ServerHarness(max_seq_len=512, arena_gib=31, device=1,
                            max_concurrent_requests=1,
                            prefix_cache_capacity_tokens=16384,
                            prefix_cache_max_entries=4,
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
        checker.check("log enables tool calling",
                      "Tool calling: yes" in log_text, log_text[-800:])
        checker.check("log enables structured",
                      "Structured:   fail-closed (Chat + Responses)" in log_text,
                      log_text[-800:])
        checker.check("log reports prefix cache",
                      "Prefix cache: 16384 tokens / 4 entries" in log_text,
                      log_text[-800:])
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

        stochastic = http_json(f"{harness.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": "Say exactly: hello"}],
            "temperature": 0.7, "max_tokens": 32,
        })
        checker.check("temperature > 0 accepted",
                      bool(stochastic["choices"][0]["message"]["content"]),
                      str(stochastic["choices"][0]["message"])[:200])

        structured = http_json(f"{harness.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": "Produce the JSON value."}],
            "temperature": 0, "max_tokens": 256,
            "response_format": STRUCTURED,
        })
        structured_text = structured["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(structured_text, strict=False)
        except ValueError:
            parsed = None
        checker.check("structured accepted",
                      parsed is not None and parsed.get("a") is not None,
                      repr(structured_text[:400]))

        tools = http_json(f"{harness.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user",
                          "content": "大阪の現在の気温をget_weatherで確認して"}],
            "temperature": 0, "max_tokens": 64, "tools": TOOLS,
        })
        tool_calls = tools["choices"][0]["message"].get("tool_calls") or []
        checker.check("tools accepted", bool(tool_calls),
                      str(tools["choices"][0]["message"])[:300])
        if tool_calls:
            checker.check("tool call name",
                          tool_calls[0]["function"]["name"] == "get_weather",
                          str(tool_calls[0])[:300])

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

        rs_status, rs_body = http_post_status(
            f"{harness.base_url}/responses",
            {"model": "phaseshift", "input": "Produce the JSON value.",
             "temperature": 0,
             "max_output_tokens": 256, "text": {"format": RESPONSES_STRUCTURED}})
        rs_text = ""
        if rs_status == 200:
            rs_payload = json.loads(rs_body)
            rs_text = "".join(
                item.get("content", [{}])[0].get("text", "")
                for item in rs_payload.get("output", [])
                if item.get("type") == "message")
            try:
                rs_parsed = json.loads(rs_text, strict=False)
            except ValueError:
                rs_parsed = None
        else:
            rs_parsed = None
        checker.check("responses structured accepted",
                      rs_status == 200 and rs_parsed is not None
                      and rs_parsed.get("a") is not None,
                      f"status={rs_status} body={rs_body[:400]} text={rs_text[:200]}")

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
