#!/usr/bin/env python3
"""phaseshift-compute --serve-stdio と DFlash2 speculative decoding の結合検証.

serve 経路で DFlash2 が動作すること、target-only greedy と完全一致すること、
通常の stochastic sampling が成立すること、および constraint 相当の
request field が fail-closed に拒否されることを確認する。
bf16 と psq4 の両方の KV dtype で回す。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (Checker, ComputeHarness, compute_binary,  # noqa: E402
                     model_dir, prompt_ids)

PROMPT = [248041, 77091]
DFLASH2_MODEL_DIR = os.environ.get("PHASESHIFT_DFLASH2_MODEL_DIR")
MAX_NEW_TOKENS = 64


def collect(harness, request_id):
    tokens = []
    done = None
    errors = []
    while True:
        event = harness.recv()
        kind = event.get("event")
        if kind == "token":
            tokens.append(event["token_id"])
        elif kind == "error":
            errors.append(event)
            return tokens, done, errors
        elif kind == "done":
            done = event
            return tokens, done, errors


def generated_ids(done) -> list[int]:
    return list(done.get("generated_ids", []))


def run_single_shot(input_ids, max_new_tokens, *, device, kv_cache_dtype,
                    dflash2=False):
    argv = [
        str(compute_binary()),
        "--model-dir", str(model_dir()),
        "--input-ids-file", "/dev/stdin",
        "--max-new-tokens", str(max_new_tokens),
        "--max-seq-len", "512",
        "--arena-gib", "31",
        "--device", str(device),
        "--temperature", "0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--seed", "0",
        "--kv-cache-dtype", kv_cache_dtype,
    ]
    if dflash2:
        argv += ["--dflash2-model-dir", DFLASH2_MODEL_DIR, "--dflash2-drafts", "7"]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as proc:
        stdout, stderr = proc.communicate(
            "\n".join(str(t) for t in input_ids) + "\n", timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"single-shot failed: {stderr[-2000:]}")
        for line in stdout.splitlines():
            if line.startswith("GENERATED_IDS="):
                return [int(x) for x in
                        line[len("GENERATED_IDS="):].split(",") if x.strip()]
    raise RuntimeError("GENERATED_IDS not found in single-shot output")


def run_suite(kv_cache_dtype: str, checker: Checker) -> None:
    label = f"kv={kv_cache_dtype}"
    serve_ids = None
    harness = ComputeHarness(max_seq_len=512, arena_gib=31, device=1,
                             kv_cache_dtype=kv_cache_dtype,
                             dflash2_model_dir=DFLASH2_MODEL_DIR)
    try:
        harness.start()
        harness.wait_ready()
        checker.check(f"{label} dflash2 enabled",
                      any("DFLASH2_ENABLED=1" in line for line in harness.stderr_lines),
                      str(harness.stderr_lines[-10:]))
        checker.check(f"{label} prefix cache disabled",
                      any("PREFIX_CACHE_ENABLED=0" in line for line in harness.stderr_lines),
                      str(harness.stderr_lines[-10:]))

        harness.send({
            "op": "generate", "request_id": 1, "input_ids": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
            "prefix_cache_checkpoint_position": 0,
        })
        tokens, done, errors = collect(harness, 1)
        checker.check(f"{label} no error", not errors, str(errors))
        checker.check(f"{label} done received", done is not None)
        if done is None:
            return

        ids = generated_ids(done)
        checker.check(f"{label} tokens emitted", len(tokens) == len(ids),
                      f"{len(tokens)} tokens vs {len(ids)} ids")
        checker.check(f"{label} stream matches generated_ids", tokens == ids)
        checker.check(f"{label} finish_reason", done.get("finish_reason") in
                      ("stop", "length"), str(done.get("finish_reason")))

        harness.send({
            "op": "generate", "request_id": 2, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.7, "top_p": 0.95, "top_k": 0,
            "seed": 7,
        })
        _, stochastic_done, stochastic_errors = collect(harness, 2)
        checker.check(f"{label} stochastic generation ok",
                      not stochastic_errors and stochastic_done is not None,
                      str(stochastic_errors))
        if stochastic_done is not None:
            checker.check(f"{label} stochastic tokens emitted",
                          len(generated_ids(stochastic_done)) > 0,
                          str(generated_ids(stochastic_done)))

        harness.send({
            "op": "generate", "request_id": 3, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
            "grammar": 'root ::= "YES" | "NO"',
        })
        _, grammar_done, grammar_errors = collect(harness, 3)
        checker.check(f"{label} grammar rejected",
                      grammar_done is None and len(grammar_errors) == 1,
                      str(grammar_errors))
        if grammar_errors:
            checker.check(f"{label} grammar rejection code",
                          grammar_errors[0].get("code") == "invalid_argument",
                          str(grammar_errors[0]))
            checker.check(f"{label} grammar rejection names the field",
                          "grammar" in str(grammar_errors[0].get("message", "")),
                          str(grammar_errors[0]))

        harness.send({
            "op": "generate", "request_id": 4, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
            "structural_tag": "{}",
        })
        _, tag_done, tag_errors = collect(harness, 4)
        checker.check(f"{label} structural_tag rejected",
                      tag_done is None and len(tag_errors) == 1,
                      str(tag_errors))

        harness.send({"op": "ping"})
        while True:
            if harness.recv().get("event") == "pong":
                break

        harness.send({
            "op": "generate", "request_id": 5, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
        })
        _, second_done, second_errors = collect(harness, 5)
        checker.check(f"{label} second generation ok",
                      not second_errors and second_done, str(second_errors))
        if second_done is not None:
            checker.check(f"{label} repeat deterministic",
                          generated_ids(second_done) == ids[:8],
                          f"{generated_ids(second_done)} vs {ids[:8]}")

        serve_ids = ids
    finally:
        harness.close()

    if serve_ids:
        target_only = run_single_shot(PROMPT, MAX_NEW_TOKENS, device=1,
                                      kv_cache_dtype=kv_cache_dtype,
                                      dflash2=False)
        shot_dflash = run_single_shot(PROMPT, MAX_NEW_TOKENS, device=1,
                                      kv_cache_dtype=kv_cache_dtype,
                                      dflash2=True)
        checker.check(f"{label} serve == target-only greedy",
                      serve_ids == target_only,
                      f"serve={serve_ids} target={target_only}")
        checker.check(f"{label} serve == single-shot dflash2",
                      serve_ids == shot_dflash,
                      f"serve={serve_ids} shot={shot_dflash}")


def main() -> int:
    checker = Checker("compute-dflash2")
    for kv_cache_dtype in ("bf16", "psq4"):
        run_suite(kv_cache_dtype, checker)
    return checker.done()


if __name__ == "__main__":
    if not DFLASH2_MODEL_DIR:
        print("SKIP: PHASESHIFT_DFLASH2_MODEL_DIR is not set")
        raise SystemExit(77)
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        raise SystemExit(77)
    raise SystemExit(main())
