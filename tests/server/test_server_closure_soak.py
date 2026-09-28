#!/usr/bin/env python3
"""Gate 11C: long-running mixed-workload closure soak.

Manual closure test, not part of the default regression run. It drives the
canonical server with a mixed concurrent workload and asserts stability:

  * no unexpected HTTP 5xx
  * no process exit
  * no parser / tool / reasoning / constraint cross-talk
  * no <think> / </think> / tool-markup leakage
  * requested cancellations complete and the server recovers
  * repeated prefix-cache hits across a reasoning tool round-trip
  * no unbounded RSS growth

Round count can be overridden with PHASESHIFT_SOAK_ROUNDS (default 32). Each
round issues four concurrent requests (A/B/C/D); the streaming request D is
cancelled on every fourth round.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

ROUNDS = int(os.environ.get("PHASESHIFT_SOAK_ROUNDS", "32"))
CONCURRENCY = 4
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=", "</function>")
HEALTH_PROMPT = "What is 2+2? Answer with only the number."

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
OK_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "ok", "strict": True,
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
               "required": ["ok"], "additionalProperties": False}}}


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))
    except (urllib.error.URLError, OSError):
        return 0, {}


def stream(url, payload, cancel_after=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    deltas = []
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for index, raw in enumerate(resp):
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                deltas.append(json.loads(body))
                if cancel_after is not None and index >= cancel_after:
                    resp.close()
                    return 200, deltas, True
    except urllib.error.HTTPError as exc:
        return exc.code, deltas, False
    except (urllib.error.URLError, OSError):
        return 0, deltas, cancel_after is not None
    return 200, deltas, False


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def delta_text(deltas, field):
    return "".join(((d.get("choices") or [{}])[0].get("delta") or {}).get(field) or ""
                   for d in deltas)


def has_markup(*texts):
    return any(m in (t or "") for t in texts for m in MARKUP)


def children_of(root):
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                fields = handle.read().split()
            table.setdefault(int(fields[3]), []).append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    out, stack = [], [root]
    while stack:
        for child in table.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def tree_rss_kb(root):
    total = 0
    for pid in [root] + children_of(root):
        try:
            with open(f"/proc/{pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except (OSError, ValueError):
            continue
    return total


class Soak:
    def __init__(self, url):
        self.url = url
        self.records = []
        self.statuses = []
        self.cancellations = 0
        self.cancel_ok = 0

    def add(self, kind, status, ok, detail=""):
        self.records.append((kind, ok, detail))
        if status:
            self.statuses.append(status)
        if kind == "D-cancel":
            self.cancellations += 1
            if ok:
                self.cancel_ok += 1

    def run_a(self, index):
        status, body = post(self.url, {
            "model": "phaseshift", "temperature": 0, "max_tokens": 48,
            "reasoning_effort": "none",
            "messages": [{"role": "user",
                          "content": f"Reply with one short sentence about {index}."}]})
        msg = message(body)
        content = msg.get("content") or ""
        self.add("A", status, status == 200 and bool(content.strip())
                 and not has_markup(content) and not msg.get("reasoning")
                 and not msg.get("tool_calls"),
                 f"{status} reasoning={bool(msg.get('reasoning'))} {content[:50]!r}")

    def run_b(self, index):
        status, body = post(self.url, {
            "model": "phaseshift", "temperature": 0, "max_tokens": 512,
            "reasoning_effort": "high", "response_format": OK_SCHEMA,
            "messages": [{"role": "user",
                          "content": f"Return the JSON object with ok true. ({index})"}]})
        msg = message(body)
        content = msg.get("content") or ""
        parsed_ok = True
        if content.strip():
            try:
                parsed_ok = isinstance(json.loads(content), dict)
            except ValueError:
                parsed_ok = False
        self.add("B", status, status == 200 and parsed_ok and not has_markup(content)
                 and not msg.get("tool_calls"), f"{status} {content[:50]!r}")

    def run_c(self, index):
        status, body = post(self.url, {
            "model": "phaseshift", "temperature": 0, "max_tokens": 512,
            "reasoning_effort": "high", "tools": [WEATHER],
            "tool_choice": "required", "parallel_tool_calls": False,
            "messages": [{"role": "user",
                          "content": f"What is the weather in Osaka? ({index})"}]})
        msg = message(body)
        calls = msg.get("tool_calls") or []
        shape_ok = True
        for call in calls:
            fn = call.get("function") or {}
            if fn.get("name") != "get_weather":
                shape_ok = False
            try:
                args = fn.get("arguments")
                args = args if isinstance(args, dict) else json.loads(args or "{}")
                if not isinstance(args, dict):
                    shape_ok = False
            except ValueError:
                shape_ok = False
        self.add("C", status, status == 200 and shape_ok
                 and not has_markup(msg.get("content") or "")
                 and not has_markup(msg.get("reasoning") or ""),
                 f"{status} calls={len(calls)}")

    def run_d(self, index, cancel):
        status, deltas, cancelled = stream(self.url, {
            "model": "phaseshift", "temperature": 0, "max_tokens": 256,
            "reasoning_effort": "high", "stream": True,
            "messages": [{"role": "user",
                          "content": f"Count from one to twenty slowly. ({index})"}]},
            cancel_after=3 if cancel else None)
        reasoning = delta_text(deltas, "reasoning")
        content = delta_text(deltas, "content")
        clean = not has_markup(reasoning, content)
        if cancel:
            self.add("D-cancel", 0, cancelled and clean, f"cancelled={cancelled}")
        else:
            self.add("D", status, status == 200 and bool(deltas) and clean,
                     f"{status} deltas={len(deltas)}")

    def round_batch(self, index, cancel):
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(self.run_a, index), pool.submit(self.run_b, index),
                       pool.submit(self.run_c, index),
                       pool.submit(self.run_d, index, cancel)]
            for future in futures:
                future.result()

    def report(self, checker):
        by_kind = defaultdict(list)
        for kind, ok, detail in self.records:
            by_kind[kind].append((ok, detail))
        for kind in ("A", "B", "C", "D", "D-cancel"):
            entries = by_kind.get(kind, [])
            if not entries:
                continue
            failures = [d for ok, d in entries if not ok]
            checker.check(f"cross-talk-{kind}", not failures,
                          f"{len(failures)}/{len(entries)} failed: {failures[:3]}")
        checker.check("no-5xx", not [s for s in self.statuses if s >= 500],
                      repr([s for s in self.statuses if s >= 500][:5]))
        checker.check("no-transport-error", 0 not in self.statuses,
                      f"transport errors={self.statuses.count(0)}")


def tool_pair(url, nonce, checker):
    system = ("You are a meticulous weather assistant. " * 30) + f" session={nonce} "
    prompt = "What is the weather in Osaka? Use the get_weather tool."
    base = [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]
    status, first = post(url, {"model": "phaseshift", "temperature": 0,
                               "max_tokens": 512, "messages": base,
                               "reasoning_effort": "high", "tools": [WEATHER],
                               "tool_choice": "required",
                               "parallel_tool_calls": False})
    assistant = message(first)
    calls = assistant.get("tool_calls") or []
    if status != 200 or not calls:
        checker.check(f"pair{nonce}-turn1", False, f"{status} calls={len(calls)}")
        return
    messages = base + [{"role": "assistant",
                        "content": assistant.get("content") or "",
                        "reasoning_content": assistant.get("reasoning") or "",
                        "tool_calls": calls}]
    for call in calls:
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "name": (call.get("function") or {}).get("name"),
                         "content": "sunny 25C"})
    status, second = post(url, {"model": "phaseshift", "temperature": 0,
                                "max_tokens": 512, "messages": messages,
                                "reasoning_effort": "high", "tools": [WEATHER]})
    final = message(second)
    checker.check(f"pair{nonce}-turn2", status == 200
                  and (bool((final.get("content") or "").strip())
                       or bool(final.get("tool_calls"))),
                  repr(second)[:160])


def main() -> int:
    checker = Checker("server-closure-soak")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    compute_log = Path("/tmp/phaseshift-g11c-soak-compute.log")
    request_log = Path("/tmp/phaseshift-g11c-soak-requests.log")
    for path in (compute_log, request_log):
        path.unlink(missing_ok=True)
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_BATCH_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute_log),
           "PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log)}
    try:
        with ServerHarness(max_seq_len=4096, arena_gib=24,
                           max_concurrent_requests=CONCURRENCY,
                           prefix_cache_capacity_tokens=16384,
                           prefix_cache_max_entries=8, env=env) as server:
            url = f"{server.base_url}/chat/completions"
            soak = Soak(url)
            pid = server.proc.pid

            health = post(url, {"model": "phaseshift", "temperature": 0,
                                "max_tokens": 16, "reasoning_effort": "none",
                                "messages": [{"role": "user",
                                              "content": HEALTH_PROMPT}]})
            rss_samples = [tree_rss_kb(pid)]
            checker.check("warmup-ok", health[0] == 200, repr(health[1])[:120])

            for index in range(1, ROUNDS + 1):
                if index % 4 == 0:
                    tool_pair(url, index, checker)
                soak.round_batch(index, cancel=(index % 4 == 0))
                if index % 4 == 0:
                    rss_samples.append(tree_rss_kb(pid))

            soak.report(checker)

            final = post(url, {"model": "phaseshift", "temperature": 0,
                               "max_tokens": 16, "reasoning_effort": "none",
                               "messages": [{"role": "user", "content": HEALTH_PROMPT}]})
            answer = (message(final[1]).get("content") or "").strip()
            checker.check("final-health-200", final[0] == 200, repr(final[1])[:120])
            checker.check("final-health-answer", answer == "4", repr(answer))
            checker.check("cancellations-completed",
                          soak.cancellations > 0
                          and soak.cancel_ok == soak.cancellations,
                          f"{soak.cancel_ok}/{soak.cancellations}")
            checker.check("server-alive", server.proc.poll() is None, "")

            baseline = rss_samples[0]
            final_rss = rss_samples[-1]
            checker.check("rss-bounded",
                          final_rss <= baseline * 2 + 1024 * 1024,
                          f"baseline={baseline}KB final={final_rss}KB samples={rss_samples}")
            print(f"RSS samples (KB): {rss_samples}")

        compute_text = compute_log.read_text(errors="replace") if compute_log.exists() else ""
        hits = [int(m) for m in re.findall(r"PREFIX_CACHE_HIT tokens=(\d+)", compute_text)]
        checker.check("prefix-hit-repeated", len(hits) >= 8,
                      f"hits={len(hits)} tokens={hits[:10]}")
        checker.check("prefix-restored-tokens", bool(hits) and all(h > 0 for h in hits),
                      repr(hits[:10]))
        multi = [int(m) for m in re.findall(
            r"PHASESHIFT_BATCH_STEP requests=(\d+)", compute_text)]
        print(f"prefix hits={len(hits)} batch steps={len(multi)} "
              f"max_requests={max(multi) if multi else 0}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
