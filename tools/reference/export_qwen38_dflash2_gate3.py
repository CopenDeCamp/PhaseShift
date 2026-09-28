#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 Gate 3 golden fixture exporter.

target feature projection（tap concat -> fc -> hidden_norm）と
GroupedDynamicCausalConv の prepare / finish reference を CPU で作る。

公式実装の意味論は docs/references/dflash2.md の revision を参照。
target 27B は Python でロードしない。DFlash2 checkpoint だけを使う。

出力:
  <out-dir>/rows<N>/fixture.safetensors
  <out-dir>/rows<N>/metadata.json
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FIXTURE_VERSION = 1
REFERENCE_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"

CASES = [1, 2, 8]

TAP_COUNT = 5


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def count_checkpoint_tensors(model_dir):
    path = os.path.join(model_dir, "model.safetensors")
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return len([k for k in header.keys() if k != "__metadata__"])


def rmsnorm_direct(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    norm = x_f32 * torch.rsqrt(variance + eps)
    return weight * norm.to(x.dtype)


def grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.float().view(1, length, groups, group_size)
    dyn = dynamic.float().view(1, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        if offset == 0:
            values = blocks
        else:
            pad = torch.zeros_like(blocks[:, :offset])
            values = torch.cat([pad, blocks[:, :-offset]], dim=1)
        kernel = base[offset].float().view(1, 1, groups, group_size)
        output = output + (kernel + dyn[:, :, offset]) * values
    return output.to(hidden.dtype).view(length, hidden_size)


def linear_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float().t()).to(torch.bfloat16)


def make_hidden(rows: int, hidden: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=g, dtype=torch.float32).to(torch.bfloat16)


def export_case(W, case_dir, rows, seeds):
    hidden_size = W["hidden_size"]
    group_size = W["conv_group_size"]
    eps = W["rms_norm_eps"]

    tensors = {}

    taps = [make_hidden(rows, hidden_size, seeds["tap_seed"] + i) for i in range(TAP_COUNT)]
    for i, tap in enumerate(taps):
        tensors[f"tap{i}"] = tap
    concat = torch.cat(taps, dim=1)
    tensors["target_concat"] = concat
    fc_out = linear_bf16(concat, W["fc"])
    tensors["target_fc"] = fc_out
    tensors["target_feature"] = rmsnorm_direct(fc_out, W["hidden_norm"], eps)

    for layer in range(W["num_hidden_layers"]):
        lw = W["layers"][layer]
        for kind, conv in (("attn", "attention"), ("mlp", "mlp")):
            base = lw[f"{conv}_conv_base_kernel"]
            projection = lw[f"{conv}_conv_kernel_projection"]

            conv_input = make_hidden(
                rows, hidden_size, seeds["conv_seed"] + layer * 10 + (0 if kind == "attn" else 1)
            )
            dynamic = linear_bf16(conv_input, projection).view(
                rows, 2, 2, hidden_size // group_size
            )
            finish_input = make_hidden(
                rows,
                hidden_size,
                seeds["finish_seed"] + layer * 10 + (0 if kind == "attn" else 1),
            )

            prepare_out = grouped_dynamic_convolve(
                conv_input, dynamic[:, 0], base[0], group_size
            )
            finish_out = grouped_dynamic_convolve(
                finish_input, dynamic[:, 1], base[1], group_size
            )

            prefix = f"L{layer}_{kind}"
            tensors[f"{prefix}_input"] = conv_input
            tensors[f"{prefix}_dynamic"] = dynamic.reshape(rows, -1)
            tensors[f"{prefix}_prepare"] = prepare_out
            tensors[f"{prefix}_finish_input"] = finish_input
            tensors[f"{prefix}_finish"] = finish_out

    os.makedirs(case_dir, exist_ok=True)
    save_file(tensors, os.path.join(case_dir, "fixture.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="build/dflash2-gate3-reference")
    ap.add_argument("--tap-seed", type=int, default=20260921)
    ap.add_argument("--conv-seed", type=int, default=8311)
    ap.add_argument("--finish-seed", type=int, default=9411)
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))

    config, config_sha = read_config(args.model_dir)
    draft = config["dflash_config"]
    tensor_count = count_checkpoint_tensors(args.model_dir)

    W = {
        "hidden_size": int(config["hidden_size"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "conv_group_size": int(draft["conv_group_size"]),
        "conv_kernel_size": int(draft["conv_kernel_size"]),
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "target_layer_ids": list(draft["target_layer_ids"]),
        "layers": [],
    }

    with safe_open(os.path.join(args.model_dir, "model.safetensors"), framework="pt") as f:
        W["fc"] = f.get_tensor("fc.weight").to(torch.bfloat16)
        W["hidden_norm"] = f.get_tensor("hidden_norm.weight").to(torch.bfloat16)
        for layer in range(W["num_hidden_layers"]):
            lw = {}
            for conv in ("attention", "mlp"):
                lw[f"{conv}_conv_base_kernel"] = f.get_tensor(
                    f"layers.{layer}.{conv}_conv.base_kernel"
                ).to(torch.bfloat16)
                lw[f"{conv}_conv_kernel_projection"] = f.get_tensor(
                    f"layers.{layer}.{conv}_conv.kernel_projection.weight"
                ).to(torch.bfloat16)
            W["layers"].append(lw)

    seeds = {"tap_seed": args.tap_seed, "conv_seed": args.conv_seed, "finish_seed": args.finish_seed}
    sha = git_sha()
    for rows in CASES:
        case_dir = os.path.join(args.out_dir, f"rows{rows}")
        export_case(W, case_dir, rows, seeds)
        meta = {
            "fixture_version": FIXTURE_VERSION,
            "case": f"rows{rows}",
            "rows": rows,
            "model_path": os.path.abspath(args.model_dir),
            "checkpoint_tensor_count": tensor_count,
            "config_sha256": config_sha,
            "hidden_size": W["hidden_size"],
            "num_hidden_layers": W["num_hidden_layers"],
            "conv_group_size": W["conv_group_size"],
            "conv_kernel_size": W["conv_kernel_size"],
            "rms_norm_eps": W["rms_norm_eps"],
            "target_layer_ids": W["target_layer_ids"],
            "dtype": "bf16",
            "reference_revision": REFERENCE_REVISION,
            "exporter_git_sha": sha,
            "seeds": seeds,
        }
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[export] rows={rows} -> {case_dir}")

    print("DFLASH2_GATE3_EXPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
