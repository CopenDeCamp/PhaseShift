#!/usr/bin/env python3
"""Gate 10A: GPU-resident prefix cache at the compute boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    constraint_tokenizer_info,
    decode_ids,
    ensure_chat_import,
    prompt_ids,
)

codec = ensure_chat_import()

PREFIX_TEXT = "The quick brown fox jumps over the lazy dog. " * 10
SUFFIX_TEXT = "Now answer with one short sentence."
MAX_SEQ_LEN = 512
MAX_NEW = 8


def run_generate(harness, request_id, input_ids, max_new_tokens=MAX_NEW, **extra):
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
    payload.update(extra)
    harness.send(payload)
    while True:
        event = harness.recv()
        if (event.get("request_id") == request_id
                and event.get("event") in ("done", "error")):
            return event


def trace_lines(harness, prefix):
    return [line for line in harness.stderr_lines if line.startswith(prefix)]


def warm_and_hit(harness, prefix_ids, suffix_ids, tag):
    first = run_generate(harness, 1, prefix_ids)
    generated = first.get("generated_ids", [])
    follow_input = list(prefix_ids) + list(generated) + list(suffix_ids)
    second = run_generate(harness, 2, follow_input)
    return first, generated, follow_input, second


def main() -> int:
    checker = Checker("compute-prefix-cache")
    if not Path("/opt/zen/wk/PhaseNonShift/.worktrees/phaseshift-server-gate10a/models").exists():
        pass

    prefix_ids = prompt_ids(PREFIX_TEXT)
    suffix_ids = prompt_ids(SUFFIX_TEXT)
    if len(prefix_ids) < 80:
        print(f"prefix too short: {len(prefix_ids)}", file=sys.stderr)
        return 2

    try:
        with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                            prefix_cache_capacity_tokens=1024,
                            prefix_cache_max_entries=4,
                            prefix_cache_trace=True) as on:
            on.wait_ready()
            first, generated, follow_input, second = warm_and_hit(
                on, prefix_ids, suffix_ids, "bf16")
            checker.check("insert-trace", len(trace_lines(on, "PREFIX_CACHE_INSERT")) >= 1,
                          repr(on.stderr_lines[-20:]))
            checker.check("hit-trace", len(trace_lines(on, "PREFIX_CACHE_HIT")) >= 1,
                          repr(trace_lines(on, "PREFIX_CACHE_HIT")))
            restored = int(second.get("restored_tokens", -1))
            expected_restored = len(prefix_ids) + len(generated) - 1
            checker.check("restored-tokens", restored == expected_restored,
                          f"restored={restored} expected={expected_restored} "
                          f"first={first.get('event')} gen={generated}")
            checker.check("prefill-reduced",
                          int(second.get("prefill_tokens", -1)) == len(follow_input) - restored,
                          f"prefill={second.get('prefill_tokens')} "
                          f"input={len(follow_input)} restored={restored}")
            checker.check("prefill-below-input",
                          int(second.get("prefill_tokens", 0)) < len(follow_input),
                          f"prefill={second.get('prefill_tokens')}")
            on_generated = second.get("generated_ids", [])
            on_follow_input = follow_input

        with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0) as off:
            off.wait_ready()
            _, _, _, off_second = warm_and_hit(off, prefix_ids, suffix_ids, "off")
            checker.check("off-no-restore",
                          int(off_second.get("restored_tokens", -1)) == 0,
                          repr(off_second.get("restored_tokens")))
            checker.check("greedy-parity",
                          on_generated == off_second.get("generated_ids", []),
                          f"on={on_generated} off={off_second.get('generated_ids')}")

        # Constrained generation still works after a hit; the matcher is not cached.
        info = constraint_tokenizer_info()
        if info is not None:
            with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                                prefix_cache_capacity_tokens=1024,
                                prefix_cache_max_entries=4,
                                constraint_tokenizer_info=info) as cg:
                cg.wait_ready()
                grammar = 'root ::= "YES"'
                c1 = run_generate(cg, 1, prefix_ids, grammar=grammar)
                c2 = run_generate(cg, 2, list(prefix_ids) + c1.get("generated_ids", [])
                                  + suffix_ids, grammar=grammar)
                checker.check("constraint-hit",
                              int(c2.get("restored_tokens", -1)) > 0,
                              repr(c2.get("restored_tokens")))
                checker.check("constraint-output",
                              decode_ids(c2.get("generated_ids", [])).strip() == "YES",
                              repr(decode_ids(c2.get("generated_ids", []))))

        # Concurrent hits on the same immutable entry.
        with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                            kv_cache_capacity_tokens=2048,
                            max_concurrent_requests=4,
                            prefix_cache_capacity_tokens=1024,
                            prefix_cache_max_entries=4) as conc:
            conc.wait_ready()
            warm_and_hit(conc, prefix_ids, suffix_ids, "conc")
            ids = [21, 22, 23, 24]
            for rid in ids:
                conc.send({"op": "generate", "request_id": rid,
                           "input_ids": list(on_follow_input),
                           "max_new_tokens": MAX_NEW, "temperature": 0.0,
                           "top_p": 1.0, "top_k": 0, "seed": 0})
            dones = {}
            while len(dones) < len(ids):
                event = conc.recv()
                if (event.get("event") in ("done", "error")
                        and event.get("request_id") in ids):
                    dones[event["request_id"]] = event
            gens = [dones[r].get("generated_ids", []) for r in ids]
            restores = [int(dones[r].get("restored_tokens", -1)) for r in ids]
            checker.check("concurrent-hits", all(r > 0 for r in restores),
                          repr([dones[r].get("message", dones[r]) for r in ids])
                          if any(r <= 0 for r in restores) else repr(restores))
            checker.check("concurrent-parity", all(g == gens[0] for g in gens), repr(gens))
            checker.check("concurrent-parity-off", gens[0] == on_generated,
                          f"conc={gens[0]} on={on_generated}")

        with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                            kv_cache_dtype="fp8_e4m3",
                            prefix_cache_capacity_tokens=1024,
                            prefix_cache_max_entries=4) as fp8_on:
            fp8_on.wait_ready()
            _, fp8_generated, _, fp8_second = warm_and_hit(
                fp8_on, prefix_ids, suffix_ids, "fp8-on")
            checker.check("fp8-hit", int(fp8_second.get("restored_tokens", -1)) > 0,
                          repr(fp8_second.get("restored_tokens")))
            fp8_on_generated = fp8_second.get("generated_ids", [])

        with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                            kv_cache_dtype="fp8_e4m3") as fp8_off:
            fp8_off.wait_ready()
            _, _, _, fp8_off_second = warm_and_hit(
                fp8_off, prefix_ids, suffix_ids, "fp8-off")
            fp8_off_generated = fp8_off_second.get("generated_ids", [])

        checker.check("fp8-parity", fp8_on_generated == fp8_off_generated,
                      f"on={fp8_on_generated} off={fp8_off_generated}")

        for psq_dtype in ("psq4", "psq8"):
            with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                                kv_cache_dtype=psq_dtype,
                                prefix_cache_capacity_tokens=256,
                                prefix_cache_max_entries=4) as pq_on:
                pq_on.wait_ready()
                _, _, _, pq_second = warm_and_hit(
                    pq_on, prefix_ids, suffix_ids, f"{psq_dtype}-on")
                checker.check(f"{psq_dtype}-hit",
                              int(pq_second.get("restored_tokens", -1)) > 0,
                              repr(pq_second.get("restored_tokens")))
                pq_on_generated = pq_second.get("generated_ids", [])

            with ComputeHarness(max_seq_len=MAX_SEQ_LEN, device=0,
                                kv_cache_dtype=psq_dtype) as pq_off:
                pq_off.wait_ready()
                _, _, _, pq_off_second = warm_and_hit(
                    pq_off, prefix_ids, suffix_ids, f"{psq_dtype}-off")
                pq_off_generated = pq_off_second.get("generated_ids", [])

            checker.check(f"{psq_dtype}-parity", pq_on_generated == pq_off_generated,
                          f"on={pq_on_generated} off={pq_off_generated}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
