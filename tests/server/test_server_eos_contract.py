#!/usr/bin/env python3
"""Gate 8R: generation stop token (EOS) contract.

Confirms that PhaseShift uses the model generation stop token (config eos),
not the tokenizer EOS metadata, and that the accepted divergence does not
silently break generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    REPO_ROOT,
    Checker,
    ComputeHarness,
    generation_stop_probe,
)


def oracle_case(name: str) -> dict:
    path = REPO_ROOT / "tests" / "e2e" / "fixtures" / "qwen35_4b_oracle.json"
    fixture = json.loads(path.read_text())
    for case in fixture["cases"]:
        if case["name"] == name:
            return case
    raise AssertionError(f"oracle case missing: {name}")


def main() -> int:
    checker = Checker("eos-contract")
    probe = generation_stop_probe()
    print("generation stop probe:", json.dumps(probe, ensure_ascii=False))

    model_eos = probe["model_config_eos"]
    tokenizer_eos = probe["tokenizer_eos_metadata"]
    oracle_terminal = probe["oracle_terminal_token"]

    checker.check("model-eos-present", isinstance(model_eos, int), repr(model_eos))
    checker.check("model-eos-in-vocab", isinstance(model_eos, int) and 0 <= model_eos < 248320,
                  repr(model_eos))
    checker.check("oracle-terminal-eq-model-eos", oracle_terminal == model_eos,
                  f"oracle={oracle_terminal} model={model_eos}")
    checker.check("model-eos-is-endoftext", probe["model_eos_token"] == "<|endoftext|>",
                  repr(probe["model_eos_token"]))
    checker.check("tokenizer-eos-is-imend", probe["tokenizer_eos_token"] == "<|im_end|>",
                  repr(probe["tokenizer_eos_token"]))
    checker.check("eos-divergence-accepted", tokenizer_eos != model_eos,
                  f"tokenizer={tokenizer_eos} model={model_eos}")

    chat = oracle_case("chat_single")
    oracle_ids = list(chat["generated_ids"])
    checker.check("oracle-continues-past-tokenizer-eos",
                  tokenizer_eos in oracle_ids and oracle_ids[-1] == model_eos,
                  f"ids={oracle_ids}")

    harness = ComputeHarness(max_concurrent_requests=1, max_seq_len=256, device=1)
    try:
        harness.start()
        harness.wait_ready()
        harness.send({
            "op": "generate",
            "request_id": 1,
            "input_ids": chat["input_ids"],
            "max_new_tokens": len(oracle_ids) + 4,
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
        checker.check("runtime-reproduces-oracle", generated == oracle_ids,
                      f"got={generated} oracle={oracle_ids}")
        checker.check("runtime-terminates-with-stop",
                      done.get("finish_reason") == "stop", repr(done))
        checker.check("terminal-token-is-generation-stop",
                      bool(generated) and generated[-1] == model_eos, f"got={generated}")
        checker.check("tokenizer-eos-not-terminal",
                      tokenizer_eos in generated and generated[-1] != tokenizer_eos,
                      f"got={generated}")

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
