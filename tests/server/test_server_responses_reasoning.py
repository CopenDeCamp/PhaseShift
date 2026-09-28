#!/usr/bin/env python3
"""Gate 11A: Responses reasoning (non-stream and stream)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

PROMPT = "What is 2+2? Answer with only the number."
MARKUP = ("<think>", "</think>")


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())


def post_status(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))


def stream(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data: "):
                body = line[6:]
                if body != "[DONE]":
                    events.append(json.loads(body))
    return events


def items(body, item_type):
    return [item for item in body.get("output", []) if item.get("type") == item_type]


def item_text(item):
    return "".join(c.get("text", "") for c in item.get("content", []))


def main() -> int:
    checker = Checker("server-responses-reasoning")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/responses"
            base = {"model": "phaseshift", "input": PROMPT, "temperature": 0,
                    "max_output_tokens": 512}

            high = post(url, dict(base, reasoning={"effort": "high"}))
            reasoning_items = items(high, "reasoning")
            checker.check("nonstream-reasoning-item", bool(reasoning_items),
                          repr(high.get("output"))[:400])
            reasoning_text = "".join(item_text(i) for i in reasoning_items)
            checker.check("nonstream-reasoning-text", bool(reasoning_text.strip()),
                          repr(reasoning_text[:200]))
            checker.check("nonstream-no-markup",
                          all(m not in reasoning_text for m in MARKUP),
                          repr(reasoning_text[:200]))
            message_text = "".join(item_text(i) for i in items(high, "message"))
            checker.check("nonstream-content", bool(message_text.strip()),
                          repr(message_text[:200]))
            usage = (high.get("usage") or {}).get("output_tokens_details") or {}
            checker.check("nonstream-reasoning-tokens",
                          int(usage.get("reasoning_tokens", -1)) > 0, repr(usage))

            none = post(url, dict(base, max_output_tokens=64))
            checker.check("none-no-reasoning-item", not items(none, "reasoning"),
                          repr(none.get("output"))[:300])

            for level in ("xhigh", "max"):
                body = post(url, dict(base, reasoning={"effort": level}))
                level_items = items(body, "reasoning")
                level_text = "".join(item_text(i) for i in level_items)
                checker.check(f"{level}-reasoning-item", bool(level_items),
                              repr(body.get("output"))[:300])
                checker.check(f"{level}-reasoning-text", bool(level_text.strip()),
                              repr(level_text[:160]))
                checker.check(f"{level}-no-markup",
                              all(m not in level_text for m in MARKUP),
                              repr(level_text[:160]))

            status, invalid = post_status(
                url, dict(base, max_output_tokens=64,
                          reasoning={"effort": "banana"}))
            checker.check("invalid-400", status == 400, f"{status} {invalid}")

            events = stream(url, dict(base, reasoning={"effort": "high"},
                                      stream=True))
            added = [e for e in events if e.get("type") == "response.output_item.added"]
            reasoning_ids = [e.get("item", {}).get("id") for e in added
                             if e.get("item", {}).get("type") == "reasoning"]
            checker.check("stream-reasoning-item", bool(reasoning_ids),
                          repr([e.get("item", {}).get("type") for e in added]))
            reasoning_deltas = "".join(
                e.get("delta", "") for e in events
                if e.get("type") == "response.output_text.delta"
                and e.get("item_id") in reasoning_ids)
            checker.check("stream-reasoning-deltas", bool(reasoning_deltas.strip()),
                          repr(reasoning_deltas[:200]))
            checker.check("stream-no-markup",
                          all(m not in reasoning_deltas for m in MARKUP),
                          repr(reasoning_deltas[:200]))
            done = [e for e in events if e.get("type") == "response.completed"]
            checker.check("stream-completed", bool(done), repr(events[-3:])[:300])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
