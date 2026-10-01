#!/usr/bin/env python3
"""Gate 11A: reasoning history round-trip and prefix cache reuse."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

SYSTEM = ("You are a meticulous assistant that always verifies facts and "
          "explains each step of the reasoning in detail. ") * 6
PROMPT = "What is 2+2? Answer with only the number."
FOLLOW = "続けて、3+3もお願い。"


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def messages(system, user, assistant=None, follow=None):
    out = [{"role": "system", "content": system}]
    out.append({"role": "user", "content": user})
    if assistant is not None:
        entry = {"role": "assistant", "content": assistant.get("content") or ""}
        if assistant.get("reasoning"):
            entry["reasoning_content"] = assistant["reasoning"]
        out.append(entry)
    if follow is not None:
        out.append({"role": "user", "content": follow})
    return out


def roundtrip(url, system):
    first = post(url, {"model": "phaseshift",
                       "messages": messages(system, PROMPT),
                       "temperature": 0, "max_tokens": 512,
                       "reasoning_effort": "high"})
    asst = message(first)
    second = post(url, {"model": "phaseshift",
                        "messages": messages(system, PROMPT, asst, FOLLOW),
                        "temperature": 0, "max_tokens": 512,
                        "reasoning_effort": "high"})
    return first, second


def main() -> int:
    checker = Checker("server-reasoning-prefix-cache")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    on_compute = Path("/tmp") / "phaseshift-g11a-reasoning-on-compute.log"
    on_requests = Path("/tmp") / "phaseshift-g11a-reasoning-on-requests.log"
    for path in (on_compute, on_requests):
        if path.exists():
            path.unlink()
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(on_compute),
           "PHASESHIFT_BACKEND_REQUEST_LOG": str(on_requests)}
    on_reasoning = {}
    try:
        with ServerHarness(max_seq_len=2048, prefix_cache_capacity_tokens=4096,
                           prefix_cache_max_entries=4, env=env) as server:
            url = f"{server.base_url}/chat/completions"
            first, second = roundtrip(url, SYSTEM)
            on_reasoning["first"] = message(first).get("reasoning") or ""
            on_reasoning["first_c"] = message(first).get("content") or ""
            on_reasoning["second"] = message(second).get("reasoning") or ""
            on_reasoning["second_c"] = message(second).get("content") or ""
            time.sleep(0.5)

        compute_text = on_compute.read_text(errors="replace")
        checker.check("prefix-enabled", "PREFIX_CACHE_ENABLED=1" in compute_text,
                      compute_text[:200])
        checker.check("prefix-hit", "PREFIX_CACHE_HIT" in compute_text,
                      compute_text[-600:])
        checker.check("turn1-reasoning", bool(on_reasoning["first"].strip()),
                      repr(on_reasoning["first"][:120]))
        checker.check("turn2-reasoning", bool(on_reasoning["second"].strip()),
                      repr(on_reasoning["second"][:120]))

        entries = [json.loads(line) for line in
                   on_requests.read_text(errors="replace").splitlines()]
        turn2 = entries[-1]
        reasoning_lens = [m.get("reasoning_len", -1) for m in turn2["messages"]]
        checker.check("history-reasoning-transported", any(x > 0 for x in reasoning_lens),
                      repr(reasoning_lens))
    except Exception as exc:  # noqa: BLE001
        checker.check("on-run", False, repr(exc))

    off_reasoning = {}
    try:
        with ServerHarness(max_seq_len=2048) as server:
            url = f"{server.base_url}/chat/completions"
            first, second = roundtrip(url, SYSTEM)
            off_reasoning["first"] = message(first).get("reasoning") or ""
            off_reasoning["first_c"] = message(first).get("content") or ""
            off_reasoning["second"] = message(second).get("reasoning") or ""
            off_reasoning["second_c"] = message(second).get("content") or ""
    except Exception as exc:  # noqa: BLE001
        checker.check("off-run", False, repr(exc))

    # Turn 1 renders the same prompt with and without the cache, so its
    # reasoning and the committed answer must match exactly. Turn 2 reuses the
    # turn-1 prefix: the cached request prefills only the remainder, so the
    # attention accumulation order over the prefix differs from a full prefill
    # and the reasoning wording can differ by a few tokens. The answer content
    # is still required to match exactly on both turns.
    for key in ("first", "first_c", "second_c"):
        checker.check(f"parity-{key}", on_reasoning.get(key) == off_reasoning.get(key),
                      f"{on_reasoning.get(key)!r} != {off_reasoning.get(key)!r}")
    checker.check("turn2-reasoning-present",
                  bool(on_reasoning.get("second", "").strip())
                  and bool(off_reasoning.get("second", "").strip()),
                  f"on={on_reasoning.get('second', '')[:80]!r} "
                  f"off={off_reasoning.get('second', '')[:80]!r}")
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
