#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 Gate 5.1 real-target fixture exporter.

実 target (Qwen3.8-27B-PSQ) の prefill から得た tap と、target の
embed_tokens から作った noise block を使い、公式 DFlash2 forward の
出力を Gate 5 fixture と同じ形式で dump する。

使い方:
  python3 tools/reference/export_qwen38_dflash2_gate51_real.py --mode prompts
  python3 tools/reference/export_qwen38_dflash2_gate51_real.py --mode fixture --case real_prose
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = {
    "real_prose": (
        "The history of the Roman Empire spans more than a thousand years, from its "
        "legendary founding on the banks of the Tiber to the fall of Constantinople."
    ),
    "real_code": (
        "def fibonacci(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
        "print([fibonacci(i) for i in range(10)])\n"
    ),
    "real_math": (
        "Solve the equation 3x + 7 = 22. First subtract 7 from both sides to obtain "
        "3x = 15, then divide both sides by 3 to get x = 5. Check: 3*5 + 7 = 22."
    ),
}


def load_exporter():
    path = os.path.join(HERE, "export_qwen38_dflash2_gate5.py")
    spec = importlib.util.spec_from_file_location("g5exporter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g5exporter"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_shard_tensor(target_dir, key):
    idx = json.load(open(os.path.join(target_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(target_dir, idx[key]), framework="pt") as f:
        return f.get_tensor(key)


def tokenize_prompts(language_dir, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(language_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for name, text in PROMPTS.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        case_dir = os.path.join(out_dir, name)
        os.makedirs(os.path.join(case_dir, "target"), exist_ok=True)
        with open(os.path.join(case_dir, "prompt.txt"), "w") as f:
            f.write(",".join(str(i) for i in ids))
        manifest[name] = ids
        print(f"[prompts] {name}: {len(ids)} tokens -> {case_dir}/prompt.txt")
    with open(os.path.join(out_dir, "prompt_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


def build_fixture(args, exporter, official, cfg, weights):
    case_dir = os.path.join(args.out, args.case)
    target_dir = os.path.join(case_dir, "target")
    meta = json.load(open(os.path.join(target_dir, "target_meta.txt")))
    context_rows = int(meta["context_rows"])
    block_rows = int(meta["block_rows"])
    context_start = int(meta["context_start"])
    block_start = int(meta["block_start"])
    anchor_token = int(meta["anchor_token_id"])
    tap_count = int(meta["tap_count"])
    hidden = int(meta["hidden_size"])

    taps = []
    for t in range(tap_count):
        raw = np.fromfile(os.path.join(target_dir, f"tap{t}.bin"), dtype=np.uint16)
        taps.append(torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).copy())
                    .view(context_rows, hidden).to(torch.bfloat16))
    concat = torch.cat(taps, dim=1).contiguous()

    draft = dict(cfg.dflash_config)
    mask_id = int(draft["mask_token_id"])
    scale = float(draft.get("input_embedding_scale", 1.0))
    embed = read_shard_tensor(args.language_dir,
                              "model.language_model.embed_tokens.weight").to(torch.bfloat16)
    ids = [anchor_token] + [mask_id] * (block_rows - 1)
    noise = (embed[torch.tensor(ids)] * scale).to(torch.bfloat16).contiguous()
    print(f"[fixture] anchor={anchor_token} mask={mask_id} scale={scale} "
          f"ctx={context_rows} blk={block_rows} start={context_start}/{block_start}")

    model = official.DFlash2DraftModel(cfg)
    model.load_state_dict(weights, strict=False)
    model.eval()
    model = model.to(torch.bfloat16)
    model.rotary_emb = official.Qwen3RotaryEmbedding(cfg)

    captured = {}
    handles = []
    handles.append(model.hidden_norm.register_forward_hook(
        lambda m, i, o: captured.__setitem__("target_feature", o.detach()[0])))
    for li, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(
            lambda m, i, o, idx=li: captured.__setitem__(f"layer{idx}_output", o.detach()[0])))
    try:
        with torch.inference_mode():
            position_ids = torch.arange(context_start, block_start + block_rows,
                                        dtype=torch.long)[None, :]
            out = model(position_ids=position_ids, attention_mask=None,
                        noise_embedding=noise.unsqueeze(0), target_hidden=concat.unsqueeze(0),
                        past_key_values=None, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    tensors = {
        "noise_embedding": noise,
        "target_concat": concat,
        "target_feature": captured["target_feature"].to(torch.bfloat16).contiguous(),
        "final_hidden": out.detach()[0].to(torch.bfloat16).contiguous(),
    }
    for i, tap in enumerate(taps):
        tensors[f"tap{i}"] = tap
    for li in range(int(cfg.num_hidden_layers)):
        tensors[f"layer{li}_output"] = captured[f"layer{li}_output"].to(torch.bfloat16).contiguous()
    save_file(tensors, os.path.join(case_dir, "fixture.safetensors"))
    fixture_meta = {
        "fixture_version": 1,
        "case": args.case,
        "context_rows": context_rows,
        "block_rows": block_rows,
        "context_start": context_start,
        "block_start": block_start,
        "hidden_size": hidden,
        "num_layers": int(cfg.num_hidden_layers),
        "block_size": int(draft["block_size"]),
        "source": "real target Qwen3.8-27B-PSQ prefill taps + target embed_tokens noise block",
        "anchor_token_id": anchor_token,
        "mask_token_id": mask_id,
        "input_embedding_scale": scale,
        "target_layer_ids": meta["target_layer_ids"],
        "reference_revision": exporter.REFERENCE_REVISION,
        "dtype": "bf16",
    }
    json.dump(fixture_meta, open(os.path.join(case_dir, "metadata.json"), "w"), indent=2)
    print(f"[fixture] wrote {case_dir}/fixture.safetensors")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prompts", choices=["prompts", "fixture"])
    ap.add_argument("--case", default="real_prose")
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B-DFlash2")
    ap.add_argument("--language-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--fixture-dir", default="build/dflash2-gate5-reference")
    ap.add_argument("--out", default="build/dflash2-gate51/real")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(16, (os.cpu_count() or 2) // 2)))
    exporter = load_exporter()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "prompts":
        return tokenize_prompts(args.language_dir, args.out)

    official = exporter.import_official(os.path.join(args.fixture_dir, "_official", "model.py"))
    from transformers.models.qwen3 import modeling_qwen3 as tq
    config, _ = exporter.read_config(args.model_dir)
    cfg = exporter.build_config(tq, config)
    weights = exporter.load_full_weights(args.model_dir)
    return build_fixture(args, exporter, official, cfg, weights)


if __name__ == "__main__":
    raise SystemExit(main())
