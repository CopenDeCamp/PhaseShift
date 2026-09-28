#!/usr/bin/env python3
"""Gate 10B: prompt-boundary prefix checkpoints at the compute boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ComputeHarness, prompt_ids  # noqa: E402

FILLER = prompt_ids("The quick brown fox jumps over the lazy dog. " * 80)
MAX_SEQ_LEN = 1024


def run_generate(harness, request_id, input_ids, max_new_tokens=8, checkpoint=None):
    payload = {
        "op": "generate",
        "request_id": request_id,
        "input_ids": list(input_ids),
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 0,
    }
    if checkpoint is not None:
        payload["prefix_cache_checkpoint_position"] = checkpoint
    harness.send(payload)
    return read_terminal(harness, request_id)


def read_terminal(harness, request_id):
    while True:
        event = harness.recv()
        if event.get("request_id") == request_id and event.get("event") in (
                "done", "error"):
            return event


def trace_lines(harness, prefix):
    return [line for line in harness.stderr_lines if line.startswith(prefix)]


def test_protocol(checker):
    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                        prefix_cache_capacity_tokens=2048,
                        prefix_cache_max_entries=8,
                        prefix_cache_trace=True, batch_trace=True) as h:
        h.wait_ready()
        ids = FILLER[:200]

        no_field = run_generate(h, 1, ids)
        checker.check("protocol-no-field", no_field.get("event") == "done"
                      and int(no_field.get("cache_checkpoint_tokens", -1)) == 0,
                      repr(no_field))

        zero = run_generate(h, 2, ids, checkpoint=0)
        checker.check("protocol-zero", zero.get("event") == "done"
                      and int(zero.get("cache_checkpoint_tokens", -1)) == 0,
                      repr(zero))

        valid = run_generate(h, 3, ids, checkpoint=100)
        checker.check("protocol-valid", valid.get("event") == "done"
                      and int(valid.get("cache_checkpoint_tokens", -1)) == 100,
                      repr(valid))
        checker.check("scheduler-split",
                      "PHASESHIFT_BATCH_STEP requests=1 tokens=100"
                      in h.stderr_lines,
                      repr([l for l in h.stderr_lines
                            if l.startswith("PHASESHIFT_BATCH_STEP")][:4]))
        checker.check("boundary-trace",
                      any(l.startswith("PREFIX_CACHE_CHECKPOINT tokens=100")
                          for l in trace_lines(h, "PREFIX_CACHE_CHECKPOINT")),
                      repr(trace_lines(h, "PREFIX_CACHE")))

        full = run_generate(h, 4, ids, checkpoint=len(ids))
        checker.check("protocol-equal-input", full.get("event") == "done"
                      and int(full.get("cache_checkpoint_tokens", -1)) == len(ids),
                      repr(full))

        invalid = run_generate(h, 5, ids, checkpoint=len(ids) + 1)
        checker.check("protocol-out-of-range", invalid.get("event") == "error"
                      and "exceeds" in (invalid.get("message") or ""),
                      repr(invalid))

    # Terminal checkpoint is suppressed only when an explicit boundary is set.
    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                        prefix_cache_capacity_tokens=2048,
                        prefix_cache_max_entries=8,
                        prefix_cache_trace=True) as h:
        h.wait_ready()
        run_generate(h, 1, FILLER[:200], checkpoint=100)
        checker.check("terminal-suppressed",
                      not any(l.startswith("PREFIX_CACHE_INSERT")
                              for l in h.stderr_lines),
                      repr(trace_lines(h, "PREFIX_CACHE")))


def test_hit_and_parity(checker):
    on = {}
    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                        prefix_cache_capacity_tokens=2048,
                        prefix_cache_max_entries=8,
                        prefix_cache_trace=True) as h:
        h.wait_ready()
        ids = FILLER[:300]
        first = run_generate(h, 1, ids, checkpoint=200)
        checker.check("boundary-saved", int(first.get("cache_checkpoint_tokens", -1)) == 200,
                      repr(first))
        second = run_generate(h, 2, ids, checkpoint=300)
        checker.check("boundary-restored", int(second.get("restored_tokens", -1)) >= 200,
                      repr(second))
        checker.check("boundary-prefill-reduced",
                      int(second.get("prefill_tokens", -1)) <= len(ids) - 200,
                      repr(second))
        on[1] = first.get("generated_ids")
        on[2] = second.get("generated_ids")

    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0) as h:
        h.wait_ready()
        ids = FILLER[:300]
        off = {}
        off[1] = run_generate(h, 1, ids, checkpoint=200).get("generated_ids")
        off[2] = run_generate(h, 2, ids, checkpoint=300).get("generated_ids")

    checker.check("boundary-parity-1", on[1] == off[1], f"on={on[1]} off={off[1]}")
    checker.check("boundary-parity-2", on[2] == off[2], f"on={on[2]} off={off[2]}")


def test_cancel(checker):
    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                        prefix_cache_capacity_tokens=2048,
                        prefix_cache_max_entries=8,
                        prefix_cache_trace=True) as h:
        h.wait_ready()
        ids = FILLER[:200]
        h.send_batch([
            {"op": "generate", "request_id": 1, "input_ids": list(ids),
             "max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0,
             "top_k": 0, "seed": 0, "prefix_cache_checkpoint_position": 100},
            {"op": "cancel", "request_id": 1},
        ])
        first = read_terminal(h, 1)
        checker.check("cancel-before-boundary-terminal",
                      first.get("event") == "done"
                      and first.get("finish_reason") == "cancelled",
                      repr(first))
        checker.check("cancel-before-boundary-no-snapshot",
                      not any(l.startswith("PREFIX_CACHE_CHECKPOINT")
                              for l in h.stderr_lines),
                      repr(trace_lines(h, "PREFIX_CACHE")))

    with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                        prefix_cache_capacity_tokens=2048,
                        prefix_cache_max_entries=8,
                        prefix_cache_trace=True) as h:
        h.wait_ready()
        ids = FILLER[:400]
        h.send({"op": "generate", "request_id": 1, "input_ids": list(ids),
                "max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0,
                "top_k": 0, "seed": 0, "prefix_cache_checkpoint_position": 300})
        # Wait until generation starts (prefill crossed the boundary).
        while True:
            event = h.recv()
            if event.get("event") in ("token", "done", "error"):
                break
        h.send({"op": "cancel", "request_id": 1})
        terminal = read_terminal(h, 1)
        checker.check("cancel-after-boundary-terminal",
                      terminal.get("event") == "done"
                      and terminal.get("finish_reason") == "cancelled",
                      repr(terminal))
        checker.check("cancel-after-boundary-snapshot",
                      any(l.startswith("PREFIX_CACHE_CHECKPOINT tokens=300")
                          for l in h.stderr_lines),
                      repr(trace_lines(h, "PREFIX_CACHE")))
        checker.check("cancel-after-boundary-no-terminal",
                      not any(l.startswith("PREFIX_CACHE_INSERT")
                              for l in h.stderr_lines),
                      repr(trace_lines(h, "PREFIX_CACHE")))
        follow = run_generate(h, 2, ids, checkpoint=350)
        checker.check("cancel-after-boundary-reuse",
                      int(follow.get("restored_tokens", -1)) >= 300,
                      repr(follow))


def main() -> int:
    checker = Checker("compute-prefix-checkpoint")
    checker.check("filler-long-enough", len(FILLER) >= 400, str(len(FILLER)))
    try:
        test_protocol(checker)
        test_hit_and_parity(checker)
        test_cancel(checker)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
