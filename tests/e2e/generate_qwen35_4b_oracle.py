#!/usr/bin/env python3
"""Generate the Qwen3.5-4B E2E oracle fixture

Independent reference: Hugging Face Transformers + PyTorch, float32,
eager attention, greedy, local files only. No PhaseShift import.
Deterministic output: no timestamps, fixed candidate lists, fixed key order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_GEOMETRY = {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "attention_head_dim": 256,
    "vocab_size": 248320,
}
EXPECTED_GDN_LAYERS = 24
EXPECTED_FULL_ATTN_LAYERS = 8
MIN_MARGIN = 0.10

RAW_CANDIDATES = [
    "The capital of Japan is",
    "2 + 2 =",
    "Write the next number: 1, 2, 3,",
    "The opposite of hot is",
    "Translate hello to Japanese.",
]
CHAT_SINGLE_CANDIDATES = [
    "Say OK.",
    "What is 1+1? Answer with just the number.",
    "Reply with a single word: yes",
]
CHAT_TURN1 = "My favorite color is blue."
CHAT_TURN2 = "What is my favorite color?"
CHAT_SINGLE_MAX_TOKENS = 8
CHAT_TURN_MAX_TOKENS = 2

FINGERPRINT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]


def fail(msg: str) -> None:
    print(f"ORACLE_GENERATION_FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fingerprint(model_dir: Path) -> list[dict]:
    files = []
    for name in FINGERPRINT_FILES:
        p = model_dir / name
        if p.is_file():
            files.append({
                "filename": name,
                "byte_size": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    for p in sorted(model_dir.glob("*.safetensors")):
        files.append({
            "filename": p.name,
            "byte_size": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    if not any(f["filename"] == "config.json" for f in files):
        fail(f"config.json not found in {model_dir}")
    if not any(f["filename"].endswith(".safetensors") for f in files):
        fail(f"no safetensors found in {model_dir}")
    return files


def validate_geometry(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    geometry = {}
    mapping = {
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "attention_head_dim": "head_dim",
        "vocab_size": "vocab_size",
    }
    for key, cfg_key in mapping.items():
        value = tc.get(cfg_key)
        if value != EXPECTED_GEOMETRY[key]:
            fail(f"geometry mismatch: {cfg_key}={value}, expected {EXPECTED_GEOMETRY[key]}")
        geometry[key] = value
    layer_types = tc.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != EXPECTED_GEOMETRY["num_hidden_layers"]:
        fail("layer_types missing or wrong length")
    gdn = sum(1 for lt in layer_types if lt != "full_attention")
    full = sum(1 for lt in layer_types if lt == "full_attention")
    if gdn != EXPECTED_GDN_LAYERS or full != EXPECTED_FULL_ATTN_LAYERS:
        fail(f"layer split mismatch: gdn={gdn} full={full}, expected {EXPECTED_GDN_LAYERS}/{EXPECTED_FULL_ATTN_LAYERS}")
    geometry["gdn_layers"] = gdn
    geometry["full_attention_layers"] = full
    return geometry


def greedy_steps(model, tokenizer, input_ids: list[int], max_new_tokens: int,
                 device: str, stop_ids: list[int]):
    import torch

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    past = None
    generated = []
    steps = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids, use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits
            if logits.dim() == 3:
                logits = logits[0, -1, :]
            elif logits.dim() == 2:
                logits = logits[-1, :]
            logits = logits.float()
            top2 = torch.topk(logits, 2, dim=-1)
            next_id = int(top2.indices[0].item())
            margin = float((top2.values[0] - top2.values[1]).item())
            generated.append(next_id)
            steps.append({
                "token_id": next_id,
                "top1_logit": float(top2.values[0].item()),
                "top2_logit": float(top2.values[1].item()),
                "margin": margin,
            })
            if next_id in stop_ids:
                break
            ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated, steps


def check_case(label: str, model, tokenizer, input_ids, max_new_tokens, device,
               stop_ids: list[int], min_tokens: int, exact_tokens: int | None):
    run1, steps1 = greedy_steps(model, tokenizer, input_ids, max_new_tokens, device, stop_ids)
    run2, _ = greedy_steps(model, tokenizer, input_ids, max_new_tokens, device, stop_ids)
    if run1 != run2:
        fail(f"{label}: non-deterministic oracle (run1={run1} run2={run2})")
    if len(run1) < min_tokens:
        fail(f"{label}: generated {len(run1)} tokens, need >= {min_tokens} (early EOS)")
    if exact_tokens is not None and len(run1) != exact_tokens:
        fail(f"{label}: generated {len(run1)} tokens, need exactly {exact_tokens}")
    min_margin = min(s["margin"] for s in steps1[:len(run1)])
    if min_margin < MIN_MARGIN:
        fail(f"{label}: min margin {min_margin:.6f} < {MIN_MARGIN}")
    return run1, steps1[:len(run1)], min_margin


def generation_stop_ids(model_dir: Path) -> list[int]:
    """Generation stop tokens from generation_config.json (text_config fallback)."""
    generation_config = model_dir / "generation_config.json"
    if generation_config.is_file():
        gen = json.loads(generation_config.read_text())
        eos = gen.get("eos_token_id") if isinstance(gen, dict) else None
        if isinstance(eos, int):
            return [int(eos)]
        if isinstance(eos, list):
            ids = [int(t) for t in eos if isinstance(t, int)]
            if ids:
                return ids
    cfg = json.loads((model_dir / "config.json").read_text())
    return [int((cfg.get("text_config", cfg))["eos_token_id"])]


def reply_text(processor, generated: list[int]) -> str:
    return processor.decode(list(generated), skip_special_tokens=True).rstrip("\n")


def chat_input_ids(processor, messages: list[dict]) -> list[int]:
    ids = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        fail(f"model dir not found: {model_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    geometry = validate_geometry(model_dir)
    files = load_fingerprint(model_dir)

    print(f"loading model on {args.device} (float32, eager)...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to(args.device).eval()
    stop_ids = generation_stop_ids(model_dir)

    cases = []

    raw_best = None
    for prompt in RAW_CANDIDATES:
        input_ids = list(processor.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        try:
            generated, steps, min_margin = check_case(
                f"raw:{prompt!r}", model, processor, input_ids, 3, args.device,
                stop_ids=stop_ids, min_tokens=3, exact_tokens=3)
        except SystemExit:
            continue
        decoded = processor.decode(list(generated), skip_special_tokens=True)
        if not decoded.strip() or "think" in decoded:
            continue
        candidate = {
            "name": "raw_compute",
            "prompt": prompt,
            "input_ids": input_ids,
            "max_new_tokens": 3,
            "generated_ids": generated,
            "steps": steps,
            "min_margin": min_margin,
        }
        if raw_best is None or min_margin > raw_best["min_margin"]:
            raw_best = candidate
    if raw_best is None:
        fail("no raw candidate satisfied the gates")
    cases.append(raw_best)


    chat_single = None
    for prompt in CHAT_SINGLE_CANDIDATES:
        messages = [{"role": "user", "content": prompt}]
        input_ids = chat_input_ids(processor, messages)
        try:
            generated, steps, min_margin = check_case(
                f"chat_single:{prompt!r}", model, processor, input_ids,
                CHAT_SINGLE_MAX_TOKENS, args.device, stop_ids=stop_ids, min_tokens=1,
                exact_tokens=None)
        except SystemExit:
            continue
        reply = reply_text(processor, generated)
        if not reply.strip() or "\n" in reply:
            continue
        chat_single = {
            "name": "chat_single",
            "user_prompt": prompt,
            "input_ids": input_ids,
            "max_new_tokens": CHAT_SINGLE_MAX_TOKENS,
            "generated_ids": generated,
            "decoded_reply": reply,
            "steps": steps,
            "min_margin": min_margin,
        }
        break
    if chat_single is None:
        fail("no chat_single candidate satisfied the gates")
    cases.append(chat_single)

    turn1_messages = [{"role": "user", "content": CHAT_TURN1}]
    turn1_ids = chat_input_ids(processor, turn1_messages)
    t1_generated, t1_steps, t1_margin = check_case(
        "chat_turn1", model, processor, turn1_ids, CHAT_TURN_MAX_TOKENS,
        args.device, stop_ids=stop_ids, min_tokens=1, exact_tokens=None)
    assistant1_raw = processor.decode(list(t1_generated), skip_special_tokens=True)
    assistant1 = reply_text(processor, t1_generated)
    if not assistant1.strip() or "\n" in assistant1:
        fail(f"chat_turn1 reply unusable: {assistant1!r}")
    cases.append({
        "name": "chat_turn1",
        "user_prompt": CHAT_TURN1,
        "input_ids": turn1_ids,
        "max_new_tokens": CHAT_TURN_MAX_TOKENS,
        "generated_ids": t1_generated,
        "decoded_reply": assistant1,
        "assistant_history_content": assistant1_raw,
        "steps": t1_steps,
        "min_margin": t1_margin,
    })

    turn2_messages = [
        {"role": "user", "content": CHAT_TURN1},
        {"role": "assistant", "content": assistant1_raw},
        {"role": "user", "content": CHAT_TURN2},
    ]
    turn2_ids = chat_input_ids(processor, turn2_messages)
    t2_generated, t2_steps, t2_margin = check_case(
        "chat_turn2", model, processor, turn2_ids, CHAT_TURN_MAX_TOKENS,
        args.device, stop_ids=stop_ids, min_tokens=1, exact_tokens=None)
    assistant2_raw = processor.decode(list(t2_generated), skip_special_tokens=True)
    assistant2 = reply_text(processor, t2_generated)
    if not assistant2.strip() or "\n" in assistant2:
        fail(f"chat_turn2 reply unusable: {assistant2!r}")
    cases.append({
        "name": "chat_turn2",
        "user_prompt": CHAT_TURN2,
        "history": [
            {"role": "user", "content": CHAT_TURN1},
            {"role": "assistant", "content": assistant1_raw},
            {"role": "user", "content": CHAT_TURN2},
        ],
        "input_ids": turn2_ids,
        "max_new_tokens": CHAT_TURN_MAX_TOKENS,
        "generated_ids": t2_generated,
        "decoded_reply": assistant2,
        "steps": t2_steps,
        "min_margin": t2_margin,
    })

    fixture = {
        "format": "phaseshift-qwen35-4b-e2e-oracle-v1",
        "model": {
            "name": "Qwen3.5-4B",
            "geometry": geometry,
            "files": files,
        },
        "oracle": {
            "implementation": "transformers",
            "dtype": "float32",
            "attention": "eager",
            "greedy": True,
            "min_margin": MIN_MARGIN,
            "torch_version": torch.__version__,
            "transformers_version": None,
        },
        "cases": cases,
    }
    try:
        import transformers
        fixture["oracle"]["transformers_version"] = transformers.__version__
    except Exception:
        pass

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"oracle fixture written: {out}")
    for c in cases:
        print(f"  {c['name']}: gen={c['generated_ids']} min_margin={c['min_margin']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
