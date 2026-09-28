#!/usr/bin/env python3
"""Gate 11A: Chat reasoning streaming separation."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

PROMPT = "What is 2+2? Answer with only the number."
MARKUP = ("<think>", "</think>")


def stream(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    deltas = []
    finish = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if delta:
                deltas.append(delta)
    return deltas, finish


def collect(deltas):
    reasoning = "".join(d.get("reasoning") or "" for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    order = []
    for delta in deltas:
        if (delta.get("reasoning") or "").strip():
            order.append("r")
        if (delta.get("content") or "").strip():
            order.append("c")
    return reasoning, content, order


def main() -> int:
    checker = Checker("server-reasoning-stream")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift",
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0, "max_tokens": 512, "stream": True}

            deltas, finish = stream(url, dict(base, reasoning_effort="high"))
            reasoning, content, order = collect(deltas)
            checker.check("high-reasoning-nonempty", bool(reasoning.strip()),
                          repr(reasoning[:200]))
            checker.check("high-content-nonempty", bool(content.strip()),
                          repr(content[:200]))
            checker.check("high-no-markup",
                          all(m not in reasoning and m not in content for m in MARKUP),
                          repr((reasoning + content)[:200]))
            compressed = []
            for kind in order:
                if not compressed or compressed[-1] != kind:
                    compressed.append(kind)
            checker.check("high-no-reopen", "".join(compressed) in ("rc", "r", "c", ""),
                          repr(compressed))
            checker.check("high-finish", finish in ("stop", "length"), repr(finish))

            none_deltas, _ = stream(url, dict(base, reasoning_effort="none"))
            none_reasoning, none_content, _ = collect(none_deltas)
            checker.check("none-no-reasoning", not none_reasoning,
                          repr(none_reasoning[:120]))
            checker.check("none-content", bool(none_content.strip()),
                          repr(none_content[:120]))
            checker.check("none-no-markup",
                          all(m not in none_content for m in MARKUP),
                          repr(none_content[:120]))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
