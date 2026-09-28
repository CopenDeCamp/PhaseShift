#!/usr/bin/env python3
"""Gate 8A: constrained / unconstrained concurrent batching acceptance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    constraint_tokenizer_info,
    decode_ids,
    prompt_ids,
)

G_A = 'root ::= "AAA"'
G_B = 'root ::= "BBB"'
G_C = 'root ::= "CCC"'
G_D = 'root ::= "DDD"'
PROMPT = prompt_ids("Answer with a short token sequence.")


def collect(harness, target_ids):
    tokens = {i: [] for i in target_ids}
    done = {}
    while len(done) < len(target_ids):
        event = harness.recv()
        kind = event.get("event")
        rid = event.get("request_id")
        if rid not in target_ids:
            continue
        if kind == "token":
            tokens[rid].append(event["token_id"])
        elif kind == "done":
            done[rid] = event
        elif kind == "error":
            done[rid] = event
    return tokens, done


def send(harness, rid, grammar=None, max_new_tokens=6):
    message = {
        "op": "generate",
        "request_id": rid,
        "input_ids": PROMPT,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 0,
    }
    if grammar is not None:
        message["grammar"] = grammar
    harness.send(message)


def main() -> int:
    checker = Checker("constraint-concurrency")
    info = constraint_tokenizer_info()
    if info is None:
        print("constraint tokenizer info unavailable", file=sys.stderr)
        return 2

    harness = ComputeHarness(max_concurrent_requests=4, max_seq_len=512,
                             batch_trace=True, constraint_trace=True, device=1,
                             constraint_tokenizer_info=info)
    try:
        harness.start()
        harness.wait_ready()

        # Mixed constrained / unconstrained concurrent batch.
        send(harness, 1, G_A)
        send(harness, 2, None)
        send(harness, 3, G_B)
        send(harness, 4, None)
        tokens, done = collect(harness, [1, 2, 3, 4])
        checker.check("mixed-a-grammar", decode_ids(done[1].get("generated_ids", [])) == "AAA",
                      repr(done[1]))
        checker.check("mixed-c-grammar", decode_ids(done[3].get("generated_ids", [])) == "BBB",
                      repr(done[3]))
        checker.check("mixed-no-cross", decode_ids(done[1].get("generated_ids", [])) != "BBB"
                      and decode_ids(done[3].get("generated_ids", [])) != "AAA")
        checker.check("mixed-token-parity",
                      all(tokens[i] == done[i].get("generated_ids") for i in (1, 2, 3, 4)))
        checker.check("mixed-unconstrained-b",
                      bool(decode_ids(done[2].get("generated_ids", [])).strip()),
                      repr(done[2]))
        checker.check("mixed-unconstrained-d",
                      bool(decode_ids(done[4].get("generated_ids", [])).strip()),
                      repr(done[4]))

        # Same grammar concurrent: shared compiled grammar, independent matchers.
        for rid in (10, 11, 12, 13):
            send(harness, rid, G_C)
        _, done = collect(harness, [10, 11, 12, 13])
        checker.check("same-grammar-all",
                      all(decode_ids(done[i].get("generated_ids", [])) == "CCC"
                          for i in (10, 11, 12, 13)),
                      repr({i: done[i].get("generated_ids") for i in (10, 11, 12, 13)}))

        # Different grammars concurrent: mask rows must not swap.
        send(harness, 20, G_A)
        send(harness, 21, G_B)
        send(harness, 22, G_C)
        send(harness, 23, G_D)
        _, done = collect(harness, [20, 21, 22, 23])
        checker.check("different-grammar-a", decode_ids(done[20].get("generated_ids", [])) == "AAA")
        checker.check("different-grammar-b", decode_ids(done[21].get("generated_ids", [])) == "BBB")
        checker.check("different-grammar-c", decode_ids(done[22].get("generated_ids", [])) == "CCC")
        checker.check("different-grammar-d", decode_ids(done[23].get("generated_ids", [])) == "DDD")

        steps = harness.batch_steps()
        checker.check("batch-requests-ge-2", max((r for r, _ in steps), default=0) >= 2,
                      repr(steps[-8:]))
        trace_lines = [line for line in harness.stderr_lines if "CONSTRAINT" in line]
        checker.check("cache-hit-observed",
                      any("CONSTRAINT_CACHE_HIT=1" in line for line in trace_lines),
                      repr(trace_lines[:5]))

        send(harness, 0, None)
        harness.send({"op": "shutdown"})
        # Drain the unconstrained request then shutdown.
        saw_shutdown = False
        while not saw_shutdown:
            event = harness.recv()
            if event.get("event") == "shutdown":
                saw_shutdown = True
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
