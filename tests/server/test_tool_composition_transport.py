#!/usr/bin/env python3
"""Gate 9B: structured + tools transport markers and grammar preservation."""

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
    model_dir,
)

SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "result", "strict": True, "schema": SCHEMA}}
TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}
RESP_TOOL = {"type": "function", "name": "get_weather", "description": "weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"], "additionalProperties": False},
             "strict": True}

COMPOSITION = "phaseshift.constraint_composition_transport"


def read_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def main() -> int:
    checker = Checker("tool-composition-transport")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    log = Path("/tmp") / f"phaseshift-g9b-transport-{os.getpid()}.jsonl"
    if log.exists():
        log.unlink()

    try:
        with ServerHarness(max_seq_len=512,
                           env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(log),
                                "PHASESHIFT_BACKEND_REQUEST_LOG_FULL": "1"}) as server:
            chat = f"{server.base_url}/chat/completions"
            responses = f"{server.base_url}/responses"

            # Baseline structured-only grammar (Chat and Responses).
            http_post_status(chat, {"model": "phaseshift",
                                    "messages": [{"role": "user", "content": "json"}],
                                    "temperature": 0, "max_tokens": 4,
                                    "response_format": RESPONSE_FORMAT})
            http_post_status(responses, {"model": "phaseshift", "input": "json",
                                         "temperature": 0, "max_output_tokens": 4,
                                         "text": {"format": RESPONSE_FORMAT["json_schema"]
                                                  | {"type": "json_schema"}}})

            for choice in ("auto", "none", "required",
                           {"type": "function", "function": {"name": "get_weather"}}):
                body = {"model": "phaseshift",
                        "messages": [{"role": "user", "content": "hi"}],
                        "temperature": 0, "max_tokens": 16,
                        "response_format": RESPONSE_FORMAT, "tools": [TOOL],
                        "tool_choice": choice}
                http_post_status(chat, body)

            for choice in ("auto", "none", "required",
                           {"type": "function", "name": "get_weather"}):
                body = {"model": "phaseshift", "input": "hi",
                        "temperature": 0, "max_output_tokens": 16,
                        "text": {"format": RESPONSE_FORMAT["json_schema"]
                                 | {"type": "json_schema"}},
                        "tools": [RESP_TOOL], "tool_choice": choice}
                http_post_status(responses, body)

            # Plain tools request: composition marker must be absent.
            http_post_status(chat, {"model": "phaseshift",
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "temperature": 0, "max_tokens": 8,
                                    "tools": [TOOL], "tool_choice": "auto"})
            # Spoof attempt on a plain tools request.
            http_post_status(chat, {"model": "phaseshift",
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "temperature": 0, "max_tokens": 8,
                                    "tools": [TOOL], "tool_choice": "auto",
                                    "metadata": {COMPOSITION: "1"}})

        entries = read_log(log)
        # Entry 0: chat structured-only. Entry 1: responses structured-only.
        baseline_chat = entries[0].get("grammar") or ""
        baseline_resp = entries[1].get("grammar") or ""
        checker.check("baseline-chat-grammar", bool(baseline_chat))
        checker.check("baseline-responses-grammar",
                      baseline_chat == baseline_resp,
                      f"chat={len(baseline_chat)} resp={len(baseline_resp)}")

        composition_entries = entries[2:10]
        checker.check("composition-count", len(composition_entries) == 8,
                      repr(len(composition_entries)))
        checker.check("composition-marker",
                      all(e.get("metadata", {}).get(COMPOSITION) == "1"
                          for e in composition_entries),
                      repr([e.get("metadata", {}).get(COMPOSITION)
                            for e in composition_entries]))
        checker.check("composition-tool-marker",
                      all(e.get("metadata", {}).get("phaseshift.tool_policy_transport") == "1"
                          for e in composition_entries),
                      repr([e.get("metadata", {}) for e in composition_entries]))
        checker.check("composition-grammar-preserved-chat",
                      all((e.get("grammar") or "") == baseline_chat for e in entries[2:6]),
                      repr([len(e.get("grammar") or "") for e in entries[2:6]]))
        checker.check("composition-grammar-preserved-responses",
                      all((e.get("grammar") or "") == baseline_resp for e in entries[6:10]),
                      repr([len(e.get("grammar") or "") for e in entries[6:10]]))

        plain = entries[10]
        spoof = entries[11]
        checker.check("plain-tools-no-composition",
                      plain.get("metadata", {}).get(COMPOSITION) is None,
                      repr(plain.get("metadata")))
        checker.check("spoof-cleared",
                      spoof.get("metadata", {}).get(COMPOSITION) is None,
                      repr(spoof.get("metadata")))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
