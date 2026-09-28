#!/usr/bin/env python3
"""Gate 8R: constrained batches use the dynamic execution path (HIP Graph bypass).

Unconstrained batches keep the existing HIP Graph path. Constrained batches must
not enter graph replay/capture, and a constrained request must not permanently
disable the graph for later unconstrained requests.

The runtime check requires a build configured with -DPHASESHIFT_HIP_GRAPH=ON.
When the graph is not compiled in, only the source-level contract is checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    REPO_ROOT,
    Checker,
    ComputeHarness,
    build_dir,
    constraint_tokenizer_info,
)

GRAMMAR = 'root ::= "AAA"'
PROMPT = [3639, 374, 220, 19, 11]


def graph_enabled() -> bool:
    cache = build_dir() / "CMakeCache.txt"
    if not cache.is_file():
        return False
    for line in cache.read_text().splitlines():
        if line.startswith("PHASESHIFT_HIP_GRAPH:BOOL="):
            return line.strip().endswith("ON")
    return False


def graph_lines(harness) -> list[str]:
    return [line for line in harness.stderr_lines if "[graph]" in line]


def send(harness, rid, grammar=None):
    message = {
        "op": "generate",
        "request_id": rid,
        "input_ids": PROMPT,
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 0,
    }
    if grammar is not None:
        message["grammar"] = grammar
    harness.send(message)


def collect(harness, target_ids):
    done = {}
    while len(done) < len(target_ids):
        event = harness.recv()
        if event.get("request_id") not in target_ids:
            continue
        if event.get("event") in ("done", "error"):
            done[event["request_id"]] = event
    return done


def main() -> int:
    checker = Checker("constraint-graph")

    source = REPO_ROOT / "src" / "phaseshift" / "models" / "qwen35" / "runtime" / "executor.hip"
    text = source.read_text() if source.is_file() else ""
    checker.check("source-graph-gate-includes-constraint",
                  "!any_constraint" in text, "executor.hip missing !any_constraint gate")

    if not graph_enabled():
        print("constraint-graph: HIP graph not compiled in (source contract checked only)")
        return checker.done()

    info = constraint_tokenizer_info()
    if info is None:
        checker.check("xgrammar-available", False, "xgrammar unavailable")
        return checker.done()

    harness = ComputeHarness(max_concurrent_requests=1, max_seq_len=256, device=1,
                             graph_debug=True, constraint_tokenizer_info=info)
    try:
        harness.start()
        harness.wait_ready()
        for i in range(4):
            send(harness, 100 + i)
            collect(harness, [100 + i])
        before = len(graph_lines(harness))
        checker.check("unconstrained-graph-active", before > 0,
                      repr(graph_lines(harness)[:5]))

        send(harness, 200, GRAMMAR)
        done = collect(harness, [200])
        after = len(graph_lines(harness))
        checker.check("constrained-completes", done[200].get("event") == "done",
                      repr(done[200]))
        checker.check("constrained-no-graph", after == before,
                      repr(graph_lines(harness)[before:after]))

        for i in range(4):
            send(harness, 300 + i)
            done = collect(harness, [300 + i])
            checker.check(f"post-constrained-{300 + i}", done[300 + i].get("event") == "done",
                          repr(done[300 + i]))
        checker.check("graph-not-permanently-failed",
                      not any("failed" in line for line in graph_lines(harness)),
                      repr([line for line in graph_lines(harness) if "failed" in line]))

        harness.send({"op": "shutdown"})
        harness.recv()
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
