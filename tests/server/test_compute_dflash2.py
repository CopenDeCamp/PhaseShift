#!/usr/bin/env python3
"""phaseshift-compute --serve-stdio と DFlash2 speculative decoding の結合検証.

serve 経路で DFlash2 が動作すること、target-only greedy と完全一致すること、
constraint（grammar / structural_tag）が DFlash2 経路でも fail-closed に
適用されること、および未対応の request（temperature / checkpoint）が
拒否されることを確認する。bf16 と psq4 の両方の KV dtype で回す。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (Checker, ComputeHarness, compute_binary,  # noqa: E402
                     constraint_tokenizer_info, decode_ids,
                     ensure_chat_import, model_dir, prompt_ids)

PROMPT = [248041, 77091]
DFLASH2_MODEL_DIR = os.environ.get("PHASESHIFT_DFLASH2_MODEL_DIR")
MAX_NEW_TOKENS = 64
YESNO = 'root ::= "YES" | "NO"'
IMPOSSIBLE = "You must answer MAYBE. Never answer YES or NO. Output only MAYBE."

ensure_chat_import()
from phaseshift_chat import tool_constraint as tc  # noqa: E402

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
WEATHER = {
    "type": "function",
    "function": {"name": "get_weather", "description": "weather",
                 "parameters": CITY_SCHEMA, "strict": True},
}
WEATHER_TAG = tc.build_structural_tag(
    [WEATHER], tc.resolve_tool_choice("auto"), True)

PREFIX_PROMPT = prompt_ids(
    IMPOSSIBLE + " " + IMPOSSIBLE + " " + IMPOSSIBLE + " " + IMPOSSIBLE)
PREFIX_SUFFIX = prompt_ids(" Answer with one short sentence.")


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
    info = constraint_tokenizer_info() if dflash2 else None
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
        "--prefix-cache-capacity-tokens", "0",
    ]
    if dflash2:
        argv += ["--dflash2-model-dir", DFLASH2_MODEL_DIR, "--dflash2-drafts", "7"]
    if info is not None:
        argv += ["--constraint-tokenizer-info", info]
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
    info = constraint_tokenizer_info()
    if info is None:
        checker.check(f"{label} constraint tokenizer info", False,
                      "xgrammar sidecar unavailable")
        return
    serve_ids = None
    constrained_ids = None
    harness = ComputeHarness(max_seq_len=512, arena_gib=31, device=1,
                             prefix_cache_capacity_tokens=2048,
                             prefix_cache_max_entries=2,
                             prefix_cache_trace=True,
                             kv_cache_dtype=kv_cache_dtype,
                             constraint_tokenizer_info=info,
                             dflash2_model_dir=DFLASH2_MODEL_DIR)
    try:
        harness.start()
        harness.wait_ready()
        harness.send({
            "op": "generate", "request_id": 1, "input_ids": PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
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
            "grammar": YESNO,
        })
        _, grammar_done, grammar_errors = collect(harness, 3)
        checker.check(f"{label} grammar accepted",
                      not grammar_errors and grammar_done is not None,
                      str(grammar_errors))
        if grammar_done is not None:
            grammar_text = decode_ids(generated_ids(grammar_done))
            checker.check(f"{label} grammar respected",
                          grammar_text.strip() in ("YES", "NO"),
                          repr(grammar_text))

        harness.send({
            "op": "generate", "request_id": 4, "input_ids": PROMPT,
            "max_new_tokens": 16, "temperature": 0.0,
            "structural_tag": WEATHER_TAG,
        })
        _, tag_done, tag_errors = collect(harness, 4)
        checker.check(f"{label} structural_tag accepted",
                      not tag_errors and tag_done is not None, str(tag_errors))

        harness.send({
            "op": "generate", "request_id": 5, "input_ids": PREFIX_PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
            "prefix_cache_checkpoint_position": len(PREFIX_PROMPT),
        })
        _, checkpoint_done, checkpoint_errors = collect(harness, 5)
        checker.check(f"{label} checkpoint accepted",
                      not checkpoint_errors and checkpoint_done is not None,
                      str(checkpoint_errors))
        if checkpoint_done is not None:
            checker.check(f"{label} checkpoint saved",
                          checkpoint_done.get("cache_checkpoint_tokens", 0) > 0,
                          str(checkpoint_done.get("cache_checkpoint_tokens")))

        harness.send({
            "op": "generate", "request_id": 8,
            "input_ids": PREFIX_PROMPT + PREFIX_SUFFIX,
            "max_new_tokens": 8, "temperature": 0.0,
        })
        _, hit_done, hit_errors = collect(harness, 8)
        checker.check(f"{label} prefix cache hit",
                      not hit_errors and hit_done is not None and
                      hit_done.get("restored_tokens", 0) > 0,
                      f"errors={hit_errors} restored="
                      f"{hit_done.get('restored_tokens') if hit_done else None}")

        harness.send({"op": "ping"})
        while True:
            if harness.recv().get("event") == "pong":
                break

        harness.send({
            "op": "generate", "request_id": 6, "input_ids": PROMPT,
            "max_new_tokens": 8, "temperature": 0.0,
        })
        _, second_done, second_errors = collect(harness, 6)
        checker.check(f"{label} second generation ok",
                      not second_errors and second_done, str(second_errors))
        if second_done is not None:
            checker.check(f"{label} repeat deterministic",
                          generated_ids(second_done) == ids[:8],
                          f"{generated_ids(second_done)} vs {ids[:8]}")

        harness.send({
            "op": "generate", "request_id": 7, "input_ids": prompt_ids(IMPOSSIBLE),
            "max_new_tokens": 8, "temperature": 0.0, "grammar": YESNO,
        })
        _, constrained_done, constrained_errors = collect(harness, 7)
        checker.check(f"{label} constrained generation ok",
                      not constrained_errors and constrained_done is not None,
                      str(constrained_errors))
        constrained_ids = (generated_ids(constrained_done)
                           if constrained_done is not None else None)
        if constrained_ids is not None:
            constrained_text = decode_ids(constrained_ids)
            checker.check(f"{label} constrained respected",
                          constrained_text.strip() in ("YES", "NO"),
                          repr(constrained_text))

        stochastic_ok = True
        for index, seed in enumerate((11, 22, 33)):
            rid = 20 + index
            harness.send({
                "op": "generate", "request_id": rid,
                "input_ids": prompt_ids(IMPOSSIBLE),
                "max_new_tokens": 8, "temperature": 1.0, "top_p": 0.95,
                "top_k": 0, "seed": seed, "grammar": YESNO,
            })
            _, stochastic_done, stochastic_errors = collect(harness, rid)
            if stochastic_errors or stochastic_done is None:
                stochastic_ok = False
                checker.check(f"{label} stochastic-constrained-{rid}", False,
                              str(stochastic_errors))
                continue
            text = decode_ids(generated_ids(stochastic_done)).strip()
            if text not in ("YES", "NO"):
                stochastic_ok = False
                checker.check(f"{label} stochastic-constrained-{rid}", False, repr(text))
        checker.check(f"{label} stochastic-constrained", stochastic_ok)

        serve_ids = ids
    finally:
        harness.close()

    if constrained_ids is not None:
        target = ComputeHarness(max_seq_len=512, arena_gib=31, device=1,
                                prefix_cache_capacity_tokens=0,
                                kv_cache_dtype=kv_cache_dtype,
                                constraint_tokenizer_info=info)
        try:
            target.start()
            target.wait_ready()
            target.send({
                "op": "generate", "request_id": 1,
                "input_ids": prompt_ids(IMPOSSIBLE),
                "max_new_tokens": 8, "temperature": 0.0, "grammar": YESNO,
            })
            _, target_done, target_errors = collect(target, 1)
            checker.check(f"{label} target-only constrained ok",
                          not target_errors and target_done is not None,
                          str(target_errors))
            if target_done is not None:
                target_ids = generated_ids(target_done)
                checker.check(f"{label} constrained == target-only",
                              target_ids == constrained_ids,
                              f"dflash={constrained_ids} target={target_ids}")
        finally:
            target.close()

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


def dflash_stats(harness) -> dict:
    stats = {}
    for line in harness.stderr_lines:
        if line.startswith("DFLASH2_") and "=" in line:
            key, _, value = line.partition("=")
            stats[key] = value.strip()
    return stats


def measure_acceptance(checker: Checker, info: str, proposal_mask: bool) -> dict:
    label = "proposal-on" if proposal_mask else "proposal-off"
    os.environ["PHASESHIFT_DFLASH2_SERVE_STATS"] = "1"
    if proposal_mask:
        os.environ.pop("PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL", None)
    else:
        os.environ["PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL"] = "0"
    grammar = 'root ::= "' + ("A" * 300) + '"'
    harness = ComputeHarness(max_seq_len=512, arena_gib=31, device=1,
                             prefix_cache_capacity_tokens=0,
                             kv_cache_dtype="bf16",
                             constraint_tokenizer_info=info,
                             dflash2_model_dir=DFLASH2_MODEL_DIR)
    stats = {}
    try:
        harness.start()
        harness.wait_ready()
        harness.send({
            "op": "generate", "request_id": 1,
            "input_ids": prompt_ids(IMPOSSIBLE),
            "max_new_tokens": 300, "temperature": 0.0, "grammar": grammar,
        })
        _, done, errors = collect(harness, 1)
        generated = generated_ids(done) if done is not None else []
        checker.check(f"acceptance {label} generated",
                      not errors and done is not None and len(generated) > 0,
                      f"errors={errors} generated={len(generated)}")
        if done is not None:
            checker.check(f"acceptance {label} finish reason",
                          done.get("finish_reason") == "stop",
                          str(done.get("finish_reason")))
        if generated:
            text = decode_ids(generated)
            checker.check(f"acceptance {label} grammar respected",
                          set(text.strip()) <= {"A"}, repr(text[:80]))
        stats = dflash_stats(harness)
        checker.check(f"acceptance {label} stats emitted",
                      "DFLASH2_MEAN_ACCEPTED" in stats, str(stats))
    finally:
        harness.close()
        os.environ.pop("PHASESHIFT_DFLASH2_SERVE_STATS", None)
        os.environ.pop("PHASESHIFT_DFLASH2_CONSTRAINT_PROPOSAL", None)
    return stats


def main() -> int:
    checker = Checker("compute-dflash2")
    for kv_cache_dtype in ("bf16", "psq4"):
        run_suite(kv_cache_dtype, checker)
    info = constraint_tokenizer_info()
    if info is not None:
        proposal_on = measure_acceptance(checker, info, True)
        proposal_off = measure_acceptance(checker, info, False)
        mean_on = float(proposal_on.get("DFLASH2_MEAN_ACCEPTED", "0") or 0)
        mean_off = float(proposal_off.get("DFLASH2_MEAN_ACCEPTED", "0") or 0)
        rate_on = float(proposal_on.get("DFLASH2_FULL_ACCEPT_RATE", "0") or 0)
        rate_off = float(proposal_off.get("DFLASH2_FULL_ACCEPT_RATE", "0") or 0)
        print(f"proposal-on : mean_accepted={mean_on:.3f} "
              f"full_accept_rate={rate_on:.3f} "
              f"rounds={proposal_on.get('DFLASH2_ROUNDS')}")
        print(f"proposal-off: mean_accepted={mean_off:.3f} "
              f"full_accept_rate={rate_off:.3f} "
              f"rounds={proposal_off.get('DFLASH2_ROUNDS')}")
        checker.check("proposal mask raises mean acceptance",
                      mean_on >= mean_off,
                      f"on={mean_on} off={mean_off}")
        checker.check("proposal mask raises full accept rate",
                      rate_on >= rate_off, f"on={rate_on} off={rate_off}")
    return checker.done()


if __name__ == "__main__":
    if not DFLASH2_MODEL_DIR:
        print("SKIP: PHASESHIFT_DFLASH2_MODEL_DIR is not set")
        raise SystemExit(77)
    if not model_dir().is_dir():
        print(f"SKIP: model dir not found: {model_dir()}")
        raise SystemExit(77)
    raise SystemExit(main())
