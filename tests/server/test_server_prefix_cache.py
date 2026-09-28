#!/usr/bin/env python3
"""Gate 10A/10B: opt-in GPU prefix cache through the server.

Since Gate 10B the backend sends a turn-stable prompt boundary, so a
rendered multi-turn conversation does hit the prefix cache. The dedicated
agent reuse coverage lives in ``test_server_agent_prefix_cache.py``; this
test checks the server transport (CLI, announcement, insert/checkpoint,
hit path, functional round-trips).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    http_json,
    model_dir,
    server_binary,
)

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
    "strict": True}}

SYSTEM = ("You are a meticulous assistant that always verifies facts before "
          "answering and explains the reasoning behind each step in detail. ") * 12
USER = "大阪の天気をget_weatherで調べて。"


def first_message(response) -> dict:
    return response["choices"][0]["message"]


def plain_roundtrip(url: str) -> dict:
    first = http_json(url, {
        "model": "phaseshift",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER}],
        "temperature": 0, "max_tokens": 32,
    })
    content = first_message(first).get("content") or ""
    return http_json(url, {
        "model": "phaseshift",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "続けて。"},
        ],
        "temperature": 0, "max_tokens": 32,
    })


def agent_roundtrip(url: str) -> dict | None:
    first = http_json(url, {
        "model": "phaseshift",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER}],
        "temperature": 0, "max_tokens": 128,
        "tools": [WEATHER], "tool_choice": "auto", "parallel_tool_calls": False,
    })
    calls = first_message(first).get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    return http_json(url, {
        "model": "phaseshift",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call.get("id"),
             "name": call["function"]["name"],
             "content": json.dumps({"temperature": 25, "condition": "sunny"})},
        ],
        "temperature": 0, "max_tokens": 64,
        "tools": [WEATHER], "tool_choice": "none",
    })


def main() -> int:
    checker = Checker("server-prefix-cache")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    proc = subprocess.run(
        [str(server_binary()), "--model-dir", str(model_dir()),
         "--prefix-cache-capacity-tokens", "1024",
         "--prefix-cache-max-entries", "0"],
        capture_output=True, text=True, timeout=120)
    checker.check("invalid-config-rejected", proc.returncode == 2,
                  f"rc={proc.returncode} out={proc.stdout[-200:]}")

    off_log = Path("/tmp") / "phaseshift-g10a-off.log"
    try:
        with ServerHarness(max_seq_len=1024, log_path=off_log) as server:
            plain_roundtrip(f"{server.base_url}/chat/completions")
        off_text = off_log.read_text(errors="replace")
        checker.check("off-no-hit", "PREFIX_CACHE_HIT" not in off_text, off_text[-300:])
        checker.check("off-cache-off", "Prefix cache: off" in off_text, off_text[-600:])
    except Exception as exc:  # noqa: BLE001
        checker.check("off-run", False, repr(exc))

    on_log = Path("/tmp") / "phaseshift-g10a-on.log"
    on_compute_log = Path("/tmp") / "phaseshift-g10a-on-compute.log"
    if on_compute_log.exists():
        on_compute_log.unlink()
    try:
        with ServerHarness(max_seq_len=1024, log_path=on_log,
                           prefix_cache_capacity_tokens=2048,
                           prefix_cache_max_entries=8,
                           env={"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
                                "PHASESHIFT_COMPUTE_LOG": str(on_compute_log)}) as server:
            url = f"{server.base_url}/chat/completions"
            plain = plain_roundtrip(url)
            agent = agent_roundtrip(url)
        on_text = on_log.read_text(errors="replace")
        compute_text = (on_compute_log.read_text(errors="replace")
                        if on_compute_log.exists() else "")
        checker.check("on-announced",
                      "Prefix cache: 2048 tokens / 8 entries" in on_text,
                      on_text[-900:])
        checker.check("on-checkpoint", "PREFIX_CACHE_CHECKPOINT" in compute_text
                      or "PREFIX_CACHE_INSERT" in compute_text,
                      compute_text[-900:])
        checker.check("on-hit", "PREFIX_CACHE_HIT" in compute_text,
                      compute_text[-900:])
        checker.check("on-restore", "PREFIX_CACHE_RESTORE" in compute_text,
                      compute_text[-900:])
        checker.check("on-plain-roundtrip",
                      bool((first_message(plain).get("content") or "").strip()),
                      repr(first_message(plain)))
        checker.check("on-tool-roundtrip",
                      agent is not None and bool(first_message(agent)),
                      repr(agent))
    except Exception as exc:  # noqa: BLE001
        checker.check("on-run", False, repr(exc))

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
