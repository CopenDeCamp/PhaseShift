#!/usr/bin/env python3
"""Gate 9A: tool policy transport markers and runtime handshake.

Confirms the LocalAI patch forwards the reserved tool-policy metadata keys,
preserves explicit strict tools on chat and responses, and that an older
runtime without the transport marker is rejected.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_post_status,
    localai_no_tool_policy_binary,
    model_dir,
)

CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"], "additionalProperties": False},
        "strict": True,
    },
}
RESP_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True,
}


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def main() -> int:
    checker = Checker("tool-policy-transport")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    log = Path("/tmp") / f"phaseshift-g9a-transport-{os.getpid()}.jsonl"
    if log.exists():
        log.unlink()

    try:
        with ServerHarness(max_seq_len=512,
                           env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(log)}) as server:
            http_post_status(f"{server.base_url}/chat/completions", {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0, "max_tokens": 8,
                "tools": [CHAT_TOOL], "tool_choice": "auto",
                "parallel_tool_calls": False,
            })
            http_post_status(f"{server.base_url}/chat/completions", {
                "model": "phaseshift",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0, "max_tokens": 8,
                "tools": [CHAT_TOOL], "tool_choice": "required",
                "parallel_tool_calls": True,
            })
            http_post_status(f"{server.base_url}/responses", {
                "model": "phaseshift", "input": "hi",
                "max_output_tokens": 8,
                "tools": [RESP_TOOL], "tool_choice": "auto",
                "parallel_tool_calls": False,
            })

        entries = read_log(log)
        chat_entries = [e for e in entries if e.get("tool_choice")]
        checker.check("log-has-entries", len(chat_entries) >= 3, repr(len(chat_entries)))
        transports = [e.get("metadata", {}).get("phaseshift.tool_policy_transport")
                      for e in chat_entries]
        checker.check("marker-present",
                      transports and all(t == "1" for t in transports), repr(transports))
        parallels = [e.get("metadata", {}).get("phaseshift.parallel_tool_calls")
                     for e in chat_entries]
        checker.check("parallel-transported",
                      "false" in parallels and "true" in parallels, repr(parallels))

        # Explicit strict survives on both surfaces.
        chat_strict = [json.loads(e["tools"])[0]["function"].get("strict")
                       for e in chat_entries
                       if "get_weather" in (e.get("tools") or "")]
        checker.check("chat-strict-preserved",
                      chat_strict and all(s is True for s in chat_strict),
                      repr(chat_strict))
        checker.check("responses-strict-preserved",
                      any("get_weather" in (e.get("tools") or "")
                          and json.loads(e["tools"])[0]["function"].get("strict") is True
                          for e in chat_entries),
                      repr([e.get("tools") for e in chat_entries]))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))

    # Negative: a runtime without the tool-policy transport patch is rejected.
    older = localai_no_tool_policy_binary()
    if older is None:
        print("[SKIP] tool-policy-transport: set PHASESHIFT_LOCALAI_NO_TOOL_POLICY_BINARY")
    else:
        saved = os.environ.get("PHASESHIFT_LOCALAI_BINARY")
        os.environ["PHASESHIFT_LOCALAI_BINARY"] = older
        try:
            with ServerHarness(max_seq_len=512) as server:
                status, body = http_post_status(
                    f"{server.base_url}/chat/completions", {
                        "model": "phaseshift",
                        "messages": [{"role": "user", "content": "hi"}],
                        "temperature": 0, "max_tokens": 8,
                        "tools": [CHAT_TOOL], "tool_choice": "auto",
                    })
                checker.check("older-runtime-rejected", 400 <= status < 600,
                              f"status={status} body={body[:200]}")
                checker.check("older-runtime-marker-message",
                              "tool policy transport marker" in body,
                              body[:200])
        except Exception as exc:  # noqa: BLE001
            checker.check("older-runtime-release", False, repr(exc))
        finally:
            if saved is None:
                os.environ.pop("PHASESHIFT_LOCALAI_BINARY", None)
            else:
                os.environ["PHASESHIFT_LOCALAI_BINARY"] = saved

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
