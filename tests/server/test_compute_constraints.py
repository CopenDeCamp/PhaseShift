#!/usr/bin/env python3
"""Gate 8A: raw compute grammar-constrained decoding acceptance."""

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

YESNO = 'root ::= "YES" | "NO"'
JSON_OK = 'root ::= "{\\"ok\\":true}"'
TRI = 'root ::= "YES" | "NO" | "MAYBE"'
LONG = 'root ::= "' + ("A" * 300) + '"'

IMPOSSIBLE = "You must answer MAYBE. Never answer YES or NO. Output only MAYBE."


class ComputeConstraintClient:
    def __init__(self, harness: ComputeHarness):
        self.harness = harness

    def generate(self, request_id, prompt, max_new_tokens, grammar=None,
                 temperature=0.0, top_p=1.0, top_k=0, seed=0):
        message = {
            "op": "generate",
            "request_id": request_id,
            "input_ids": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        if grammar is not None:
            message["grammar"] = grammar
        self.harness.send(message)

    def collect(self, target_ids):
        tokens = {i: [] for i in target_ids}
        done = {}
        while len(done) < len(target_ids):
            event = self.harness.recv()
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


def main() -> int:
    checker = Checker("compute-constraints")
    info = constraint_tokenizer_info()
    if info is None:
        print("constraint tokenizer info unavailable (xgrammar missing)", file=sys.stderr)
        return 2

    harness = ComputeHarness(max_concurrent_requests=4, max_seq_len=256,
                             batch_trace=True, constraint_trace=True, device=1,
                             constraint_tokenizer_info=info)
    client = ComputeConstraintClient(harness)
    try:
        harness.start()
        harness.wait_ready()

        impossible = prompt_ids(IMPOSSIBLE)

        # 1. Fixed YES/NO grammar on an adversarial prompt.
        client.generate(1, impossible, 8, YESNO)
        _, done = client.collect([1])
        yesno_text = decode_ids(done[1].get("generated_ids", []))
        checker.check("fixed-grammar-enforced", yesno_text.strip() in ("YES", "NO"),
                      repr(yesno_text))
        checker.check("fixed-grammar-terminal",
                      done[1].get("event") == "done", repr(done[1]))

        # 2. Fixed JSON grammar exact.
        client.generate(2, impossible, 16, JSON_OK)
        _, done = client.collect([2])
        checker.check("json-grammar-exact",
                      decode_ids(done[2].get("generated_ids", [])) == '{"ok":true}',
                      repr(decode_ids(done[2].get("generated_ids", []))))

        # 3. Invalid grammar is fail-closed and does not kill the process.
        pid = harness.proc.pid
        client.generate(3, impossible, 8, 'root ::= [')
        _, done = client.collect([3])
        checker.check("invalid-grammar-error",
                      done[3].get("event") == "error", repr(done[3]))
        checker.check("invalid-grammar-no-fallback",
                      "generated_ids" not in done[3], repr(done[3]))
        checker.check("invalid-grammar-pid-stable", harness.proc.pid == pid)
        client.generate(4, impossible, 8, YESNO)
        _, done = client.collect([4])
        checker.check("recovery-after-invalid",
                      decode_ids(done[4].get("generated_ids", [])).strip() in ("YES", "NO"),
                      repr(done[4]))

        # 4. Empty and oversized grammar are request errors.
        client.generate(5, impossible, 4, "")
        _, done = client.collect([5])
        checker.check("empty-grammar-error", done[5].get("event") == "error", repr(done[5]))
        client.generate(6, impossible, 4, "root ::= " + '"a"' * 600000)
        _, done = client.collect([6])
        checker.check("oversized-grammar-error", done[6].get("event") == "error",
                      repr(done[6]))

        # 5. Stochastic constrained sampling never leaves the grammar.
        stochastic_ok = True
        for index, seed in enumerate((11, 22, 33, 44, 55)):
            rid = 100 + index
            client.generate(rid, impossible, 8, YESNO, temperature=1.0,
                            top_p=0.95, top_k=40, seed=seed)
            _, done = client.collect([rid])
            text = decode_ids(done[rid].get("generated_ids", [])).strip()
            if text not in ("YES", "NO"):
                stochastic_ok = False
                checker.check(f"stochastic-{rid}", False, repr(text))
        checker.check("stochastic-constrained", stochastic_ok)

        # 6. Cancel a constrained request and reuse the process.
        client.generate(200, impossible, 200, LONG)
        cancelled = None
        seen = 0
        while cancelled is None:
            event = harness.recv()
            if event.get("request_id") != 200:
                continue
            if event.get("event") == "token":
                seen += 1
                if seen == 2:
                    harness.send({"op": "cancel", "request_id": 200})
            elif event.get("event") == "done":
                cancelled = event
        checker.check("constrained-cancel",
                      cancelled.get("finish_reason") == "cancelled", repr(cancelled))
        client.generate(201, impossible, 8, YESNO)
        _, done = client.collect([201])
        checker.check("constrained-after-cancel",
                      decode_ids(done[201].get("generated_ids", [])).strip() in ("YES", "NO"),
                      repr(done[201]))

        # 7. Retirement of many constrained requests on one process.
        for n in range(200):
            rid = 1000 + n
            client.generate(rid, impossible, 3, TRI if n % 2 == 0 else YESNO)
            _, done = client.collect([rid])
            if done[rid].get("event") != "done":
                checker.check("retirement-request", False, repr(done[rid]))
                break
        checker.check("retirement-pid-stable", harness.proc.pid == pid)
        checker.check("retirement-model-load-once", harness.count_load_events() == 1)

        harness.send({"op": "shutdown"})
        checker.check("shutdown", harness.recv().get("event") == "shutdown")
        harness.proc.wait(timeout=30)
        checker.check("shutdown-exit-zero", harness.proc.returncode == 0)

        trace_lines = [line for line in harness.stderr_lines if "CONSTRAINT" in line]
        checker.check("trace-emitted", len(trace_lines) > 0, repr(trace_lines[:3]))
        checker.check("cache-hit-observed",
                      any("CONSTRAINT_CACHE_HIT=1" in line for line in trace_lines),
                      repr(trace_lines[-3:]))
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
