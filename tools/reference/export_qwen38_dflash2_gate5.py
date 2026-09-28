#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 Gate 5 golden fixture exporter.

公式 z-lab/dflash の DFlash2DraftModel.forward() をそのまま実行し、
5 layer の出力と final norm を forward hook で取得する。

出力:
  <out-dir>/<case>/fixture.safetensors
  <out-dir>/<case>/metadata.json
"""

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import urllib.request

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FIXTURE_VERSION = 1
REFERENCE_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/z-lab/dflash/{REFERENCE_REVISION}/dflash/model.py"
)

CASES = [
    {"name": "ctx1_blk2", "context_rows": 1, "block_rows": 2, "context_start": 0},
    {"name": "ctx4_blk4", "context_rows": 4, "block_rows": 4, "context_start": 100},
    {"name": "ctx8_blk8", "context_rows": 8, "block_rows": 8, "context_start": 4096},
]

TAP_COUNT = 5


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def fetch_official(out_dir: str):
    path = os.path.join(out_dir, "_official", "model.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with urllib.request.urlopen(REFERENCE_URL, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        data = f.read()
    return path, hashlib.sha256(data).hexdigest()


def import_official(path: str):
    spec = importlib.util.spec_from_file_location("dflash_official_model_g5", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dflash_official_model_g5"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_config(model_dir: str):
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def count_checkpoint_tensors(model_dir: str) -> int:
    with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return len([k for k in header.keys() if k != "__metadata__"])


def build_config(transformers, config):
    draft = config["dflash_config"]
    cfg = transformers.Qwen3Config(
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        attention_bias=bool(config["attention_bias"]),
        attention_dropout=float(config["attention_dropout"]),
        hidden_act=config["hidden_act"],
        tie_word_embeddings=bool(config["tie_word_embeddings"]),
        rope_parameters=dict(config["rope_parameters"]),
    )
    cfg.is_causal = bool(config["is_causal"])
    cfg.layer_types = list(config["layer_types"])
    cfg.sliding_window = int(config["sliding_window"])
    cfg.num_target_layers = int(config["num_target_layers"])
    cfg.dflash_config = dict(draft)
    return cfg


def load_full_weights(model_dir: str):
    weights = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith("candidate_selector."):
                continue
            weights[key] = f.get_tensor(key)
    return weights


def make_hidden(rows: int, features: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, features, generator=g, dtype=torch.float32).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="build/dflash2-gate5-reference")
    ap.add_argument("--tap-seed", type=int, default=20260923)
    ap.add_argument("--noise-seed", type=int, default=6161)
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))

    official_path, official_sha = fetch_official(args.out_dir)
    official = import_official(official_path)
    from transformers.models.qwen3 import modeling_qwen3 as tq

    config, config_sha = read_config(args.model_dir)
    tensor_count = count_checkpoint_tensors(args.model_dir)
    cfg = build_config(tq, config)

    model = official.DFlash2DraftModel(cfg)
    weights = load_full_weights(args.model_dir)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    blocking = [k for k in missing if not k.startswith("candidate_selector.")]
    if blocking:
        raise SystemExit(f"missing weights: {blocking[:5]}")
    if unexpected:
        raise SystemExit(f"unexpected weights: {unexpected[:5]}")
    model.eval()
    model = model.to(torch.bfloat16)
    model.rotary_emb = official.Qwen3RotaryEmbedding(cfg)

    for case in CASES:
        name = case["name"]
        context_rows = case["context_rows"]
        block_rows = case["block_rows"]
        context_start = case["context_start"]
        block_start = context_start + context_rows

        taps = [
            make_hidden(context_rows, cfg.hidden_size, args.tap_seed + i)
            for i in range(TAP_COUNT)
        ]
        concat = torch.cat(taps, dim=1)
        noise_embedding = make_hidden(block_rows, cfg.hidden_size, args.noise_seed + block_rows)

        captured: dict[str, torch.Tensor] = {}
        hooks = []
        hooks.append(model.hidden_norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("target_feature", o.detach()[0])))
        hooks.append(model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("final_hidden", o.detach()[0])))
        for li, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, idx=li: captured.__setitem__(f"layer{idx}_output",
                                                             o.detach()[0])))

        try:
            with torch.inference_mode():
                position_ids = torch.arange(
                    context_start, block_start + block_rows, dtype=torch.long
                )[None, :]
                out = model(
                    position_ids=position_ids,
                    attention_mask=None,
                    noise_embedding=noise_embedding.unsqueeze(0),
                    target_hidden=concat.unsqueeze(0),
                    past_key_values=None,
                    use_cache=False,
                )
        finally:
            for hook in hooks:
                hook.remove()

        tensors = {
            "noise_embedding": noise_embedding,
            "target_concat": concat,
            "target_feature": captured["target_feature"].to(torch.bfloat16),
            "final_hidden": out.detach()[0].to(torch.bfloat16),
        }
        for i, tap in enumerate(taps):
            tensors[f"tap{i}"] = tap
        for li in range(cfg.num_hidden_layers):
            tensors[f"layer{li}_output"] = captured[f"layer{li}_output"].to(torch.bfloat16)
        for key, value in tensors.items():
            tensors[key] = value.contiguous()

        case_dir = os.path.join(args.out_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        save_file(tensors, os.path.join(case_dir, "fixture.safetensors"))
        meta = {
            "fixture_version": FIXTURE_VERSION,
            "case": name,
            "context_rows": context_rows,
            "block_rows": block_rows,
            "context_start": context_start,
            "block_start": block_start,
            "hidden_size": int(cfg.hidden_size),
            "num_layers": int(cfg.num_hidden_layers),
            "block_size": int(config["dflash_config"]["block_size"]),
            "model_path": os.path.abspath(args.model_dir),
            "checkpoint_tensor_count": tensor_count,
            "config_sha256": config_sha,
            "reference_revision": REFERENCE_REVISION,
            "reference_sha256": official_sha,
            "exporter_git_sha": git_sha(),
            "dtype": "bf16",
            "seeds": {"tap_seed": args.tap_seed, "noise_seed": args.noise_seed},
        }
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[export] {name} ctx={context_rows} blk={block_rows} -> {case_dir}")

    print("DFLASH2_GATE5_EXPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
