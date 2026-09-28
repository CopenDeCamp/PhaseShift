#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 Gate 4 golden fixture exporter.

公式 z-lab/dflash の dflash/model.py をそのまま import して layer 0 を実行し、
forward hook / method patch で中間値を取得する。target 27B はロードしない。

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
    {"name": "ctx1_blk1", "context_rows": 1, "block_rows": 1, "context_start": 0},
    {"name": "ctx4_blk2", "context_rows": 4, "block_rows": 2, "context_start": 100},
    {"name": "ctx8_blk8", "context_rows": 8, "block_rows": 8, "context_start": 4096},
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def fetch_official(out_dir: str) -> tuple[str, str]:
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
    spec = importlib.util.spec_from_file_location("dflash_official_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dflash_official_model"] = mod
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
        sliding_window=int(config["sliding_window"]),
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
    cfg.dflash_config = dict(draft)
    return cfg


def load_layer_weights(model_dir: str, layer_index: int):
    prefix = f"layers.{layer_index}."
    weights = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = f.get_tensor(key)
    return weights


def add_seq(captured: dict, key: str, value):
    if not isinstance(value, torch.Tensor):
        return
    v = value.detach()
    if v.dim() >= 3 and v.shape[0] == 1:
        v = v[0].reshape(v.shape[1], -1)
    captured[key] = v.contiguous()


def add_heads(captured: dict, key: str, value):
    if not isinstance(value, torch.Tensor):
        return
    v = value.detach()
    if v.dim() == 4 and v.shape[0] == 1:
        rows = int(v.shape[2])
        v = v[0].permute(1, 0, 2).reshape(rows, -1)
    captured[key] = v.contiguous()


def make_hidden(rows: int, features: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, features, generator=g, dtype=torch.float32).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="build/dflash2-gate4-reference")
    ap.add_argument("--hidden-seed", type=int, default=20260922)
    ap.add_argument("--target-seed", type=int, default=5150)
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))

    official_path, official_sha = fetch_official(args.out_dir)
    official = import_official(official_path)
    from transformers.models.qwen3 import modeling_qwen3 as tq

    config, config_sha = read_config(args.model_dir)
    tensor_count = count_checkpoint_tensors(args.model_dir)
    cfg = build_config(tq, config)

    layer = official.Qwen3DFlashDecoderLayer(cfg, 0)
    kernel_size = int(config["dflash_config"]["conv_kernel_size"])
    group_size = int(config["dflash_config"]["conv_group_size"])
    layer.attention_conv = official.GroupedDynamicCausalConv(cfg.hidden_size, kernel_size, group_size)
    layer.mlp_conv = official.GroupedDynamicCausalConv(cfg.hidden_size, kernel_size, group_size)

    weights = load_layer_weights(args.model_dir, 0)
    missing, unexpected = layer.load_state_dict(weights, strict=False)
    if missing:
        raise SystemExit(f"missing layer weights: {missing}")
    if unexpected:
        raise SystemExit(f"unexpected layer weights: {unexpected}")
    layer.eval()
    layer = layer.to(torch.bfloat16)
    rotary_emb = official.Qwen3RotaryEmbedding(cfg)

    for case in CASES:
        name = case["name"]
        context_rows = case["context_rows"]
        block_rows = case["block_rows"]
        context_start = case["context_start"]
        block_start = context_start + context_rows

        hidden_in = make_hidden(block_rows, cfg.hidden_size, args.hidden_seed + block_rows)
        target_feature = make_hidden(
            context_rows, cfg.hidden_size, args.target_seed + context_rows
        )

        captured: dict[str, torch.Tensor] = {}

        def add(key: str, value):
            add_seq(captured, key, value)

        def add_h(key: str, value):
            add_heads(captured, key, value)

        hooks = []
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda m, i, o: add("input_norm", o)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda m, i, o: add("post_attention_norm", o)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, i: add("attention_residual", i[0])))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            lambda m, i, o: add("q_raw", o)))
        k_calls = {"n": 0}
        v_calls = {"n": 0}

        def k_hook(m, i, o):
            add("k_ctx_raw" if k_calls["n"] == 0 else "k_noise_raw", o)
            k_calls["n"] += 1

        def v_hook(m, i, o):
            add("v_ctx" if v_calls["n"] == 0 else "v_noise", o)
            v_calls["n"] += 1

        hooks.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook))
        hooks.append(layer.self_attn.q_norm.register_forward_hook(
            lambda m, i, o: add("q_norm", o)))

        def k_norm_hook(m, i, o):
            add("k_ctx_norm", o[:, :context_rows, :, :])
            add("k_noise_norm", o[:, context_rows:, :, :])

        hooks.append(layer.self_attn.k_norm.register_forward_hook(k_norm_hook))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            lambda m, i, o: add("attention_o", o)))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            lambda m, i, o: add("mlp_gate", o)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            lambda m, i, o: add("mlp_up", o)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, i: add("mlp_swiglu", i[0])))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            lambda m, i, o: add("mlp_down", o)))

        conv_hooks = []
        for kind, conv in (("attention", layer.attention_conv), ("mlp", layer.mlp_conv)):
            original_prepare = conv.prepare

            def prepare(hidden, _conv=conv, _kind=kind, _orig=original_prepare):
                out, dynamic = _orig(hidden)
                add(f"{_kind}_prepared", out)
                add(f"{_kind}_dynamic", _conv.kernel_projection(hidden))
                return out, dynamic

            conv.prepare = prepare
            original_finish = conv.finish

            def finish(hidden, dynamic, _conv=conv, _kind=kind, _orig=original_finish):
                out = _orig(hidden, dynamic)
                add(f"{_kind}_finished", out)
                return out

            conv.finish = finish
            conv_hooks.append((conv, original_prepare, original_finish))

        original_sdpa = tq.ALL_ATTENTION_FUNCTIONS["sdpa"]

        def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                 sliding_window=None, **kwargs):
            add_h("q_rope", query)
            add("attention_mask", attention_mask.to(torch.float32)[0, 0])
            add_h("k_ctx_rope", key[:, :, :context_rows, :])
            add_h("k_noise_rope", key[:, :, context_rows:, :])
            out = original_sdpa(module, query, key, value, attention_mask, dropout=dropout,
                                scaling=scaling, sliding_window=sliding_window, **kwargs)
            add("attention_context", out[0] if isinstance(out, tuple) else out)
            return out

        tq.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa

        try:
            with torch.inference_mode():
                position_ids = torch.arange(
                    context_start, block_start + block_rows, dtype=torch.long
                )[None, :]
                position_embeddings = rotary_emb(hidden_in.unsqueeze(0), position_ids)
                out = layer(
                    hidden_states=hidden_in.unsqueeze(0),
                    target_hidden=target_feature.unsqueeze(0),
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
            add("layer_output", out)
        finally:
            tq.ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
            for hook in hooks:
                hook.remove()
            for conv, prepare_orig, finish_orig in conv_hooks:
                conv.prepare = prepare_orig
                conv.finish = finish_orig

        tensors = {
            "hidden_in": hidden_in,
            "target_feature": target_feature,
        }
        for key, value in captured.items():
            tensors[key] = value.to(torch.bfloat16) if value.dtype == torch.bfloat16 else value
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
            "model_path": os.path.abspath(args.model_dir),
            "checkpoint_tensor_count": tensor_count,
            "config_sha256": config_sha,
            "hidden_size": int(cfg.hidden_size),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": int(cfg.head_dim),
            "sliding_window": int(cfg.sliding_window),
            "rms_norm_eps": float(cfg.rms_norm_eps),
            "rope_theta": float(config["rope_parameters"]["rope_theta"]),
            "reference_revision": REFERENCE_REVISION,
            "reference_sha256": official_sha,
            "exporter_git_sha": git_sha(),
            "layer_index": 0,
            "dtype": "mixed (bf16 except q_rope/k_rope/attention_mask f32)",
            "seeds": {"hidden_seed": args.hidden_seed, "target_seed": args.target_seed},
        }
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[export] {name} ctx={context_rows} blk={block_rows} -> {case_dir}")

    print("DFLASH2_GATE4_EXPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
