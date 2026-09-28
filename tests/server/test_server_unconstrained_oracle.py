#!/usr/bin/env python3
"""Gate 8R: unconstrained greedy oracle regression.

Re-runs the committed Qwen3.5 transformers oracle for the raw_compute case and
requires exact generated token parity. This guards the unconstrained path
against accidental changes from the constraint integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import REPO_ROOT, Checker, ComputeHarness  # noqa: E402


def main() -> int:
    checker = Checker("unconstrained-oracle")
    fixture = json.loads(
        (REPO_ROOT / "tests" / "e2e" / "fixtures" / "qwen35_4b_oracle.json").read_text())
    case = next(c for c in fixture["cases"] if c["name"] == "raw_compute")
    oracle_ids = list(case["generated_ids"])

    harness = ComputeHarness(max_concurrent_requests=1, max_seq_len=64, device=1)
    try:
        harness.start()
        harness.wait_ready()
        harness.send({
            "op": "generate",
            "request_id": 1,
            "input_ids": case["input_ids"],
            "max_new_tokens": case["max_new_tokens"],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 0,
        })
        done = None
        while done is None:
            event = harness.recv()
            if event.get("request_id") != 1:
                continue
            if event.get("event") in ("done", "error"):
                done = event
        generated = list(done.get("generated_ids", []))
        checker.check("raw-compute==oracle", generated == oracle_ids,
                      f"got={generated} oracle={oracle_ids}")
        checker.check("raw-compute-terminal", done.get("event") == "done", repr(done))

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
