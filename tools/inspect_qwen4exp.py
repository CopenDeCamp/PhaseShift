#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (qwen4_exp) architecture inspection / tensor manifest tool.

config.json と safetensors header だけを読み、GPU allocation を行わずに
text-model の tensor name / dtype / shape / shard を列挙する。

usage:
    inspect_qwen4exp.py manifest  [--model-dir DIR] [--repo R] [--revision REV] [--out FILE]
    inspect_qwen4exp.py inventory [--model-dir DIR] [--repo R] [--revision REV]
    inspect_qwen4exp.py budget    [--model-dir DIR] [--repo R] [--revision REV] [--out FILE]
    inspect_qwen4exp.py check     [--model-dir DIR] [--repo R] [--revision REV]

--model-dir を指定した場合はローカル snapshot を読む。
指定しない場合は Hugging Face から index / safetensors header を HTTP Range で読む。
budget は 1 次元パラメータ（norm / A_log / dt_bias 等）を BF16 固定として扱う。
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import urllib.request

DEFAULT_REPO = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

WEIGHT_FORMATS = {
    "BF16": None,
    "PSQ8": {"bits": 8, "scale_bytes": 2, "block": 32},
    "PSQ4": {"bits": 4, "scale_bytes": 2, "block": 32},
    "PSQ3": {"bits": 3, "scale_bytes": 2, "block": 32},
    "MXFP4": {"bits": 4, "scale_bytes": 1, "block": 32},
    "FP8_BLOCK128": {"bits": 8, "scale_bytes": 4, "block": 128, "matrix_block": True},
}

CATEGORY_ORDER = [
    "token_embedding",
    "lm_head",
    "routed_experts",
    "shared_expert",
    "router",
    "gdn",
    "qsa_attention",
    "qsa_indexer",
    "hyper_connection_layer",
    "hyper_connection_final",
    "ple_ngram_table",
    "ple_other",
    "mtp",
    "vision",
    "unknown",
]

DISPOSITION = {
    "token_embedding": "REUSE",
    "lm_head": "REUSE",
    "routed_experts": "NEW",
    "shared_expert": "ADAPT",
    "router": "NEW",
    "gdn": "ADAPT",
    "qsa_attention": "ADAPT",
    "qsa_indexer": "NEW",
    "hyper_connection_layer": "NEW",
    "hyper_connection_final": "NEW",
    "ple_ngram_table": "NEW",
    "ple_other": "NEW",
    "mtp": "NEW",
    "vision": "IGNORE",
    "unknown": "UNKNOWN",
}

RESIDENCY_SETS = {
    "main_backbone": [
        "token_embedding",
        "lm_head",
        "routed_experts",
        "shared_expert",
        "router",
        "gdn",
        "qsa_attention",
        "qsa_indexer",
        "hyper_connection_layer",
        "hyper_connection_final",
    ],
    "main_plus_mtp": [
        "token_embedding",
        "lm_head",
        "routed_experts",
        "shared_expert",
        "router",
        "gdn",
        "qsa_attention",
        "qsa_indexer",
        "hyper_connection_layer",
        "hyper_connection_final",
        "mtp",
    ],
    "routed_experts_only": ["routed_experts"],
    "ple_table_only": ["ple_ngram_table"],
    "ple_other_only": ["ple_other"],
}


def align_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _fetch(url, start=None, end=None, timeout=90):
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_safetensors_header(data):
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size].decode("utf-8"))
    header.pop("__metadata__", None)
    return header


class ModelSource:
    def __init__(self, model_dir=None, repo=None, revision=None):
        self.model_dir = model_dir
        self.repo = repo or DEFAULT_REPO
        self.revision = revision or DEFAULT_REVISION
        self.remote = model_dir is None
        if self.remote:
            self.base = f"https://huggingface.co/{self.repo}/resolve/{self.revision}/"
            index = json.loads(_fetch(self.base + "model.safetensors.index.json").decode("utf-8"))
        else:
            index = self._read_local_json("model.safetensors.index.json")
        self.weight_map = index["weight_map"]

    def _read_local_json(self, name):
        with open(os.path.join(self.model_dir, name), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def config(self):
        if self.remote:
            return json.loads(_fetch(self.base + "config.json").decode("utf-8"))
        return self._read_local_json("config.json")

    def headers(self):
        shards = sorted(set(self.weight_map.values()))
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for shard, header in zip(shards, executor.map(self._shard_header, shards)):
                for name, spec in header.items():
                    result[name] = {
                        "dtype": spec["dtype"],
                        "shape": list(spec["shape"]),
                        "shard": shard,
                    }
        return result

    def _shard_header(self, shard):
        if self.remote:
            header_size = int.from_bytes(_fetch(self.base + shard, 0, 7), "little")
            return _parse_safetensors_header(_fetch(self.base + shard, 0, 7 + header_size))
        with open(os.path.join(self.model_dir, shard), "rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            return _parse_safetensors_header(handle.read(8 + header_size))


def classify(name):
    if name.startswith("model.visual"):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embedding"
    if name.startswith("model.language_model.hyper_connection_mixer."):
        return "hyper_connection_final"
    match = re.match(r"model\.language_model\.layers\.(?:\d+|N)\.(.+)", name)
    if match is None:
        return "unknown"
    suffix = match.group(1)
    if suffix.startswith("ple."):
        return "ple_ngram_table" if "ngram_embedding" in suffix else "ple_other"
    if suffix.startswith("mlp.experts."):
        return "routed_experts"
    if suffix.startswith("mlp.shared_expert"):
        return "shared_expert"
    if suffix == "mlp.gate.weight":
        return "router"
    if suffix.startswith("linear_attn."):
        return "gdn"
    if suffix.startswith("self_attn.indexer."):
        return "qsa_indexer"
    if suffix.startswith("self_attn."):
        return "qsa_attention"
    if suffix.startswith("attn_hyper_connection.") or suffix.startswith("mlp_hyper_connection."):
        return "hyper_connection_layer"
    return "unknown"


def tensor_bytes(shape, fmt):
    if fmt == "BF16":
        return math.prod(shape) * 2
    if len(shape) == 1:
        return shape[0] * 2
    spec = WEIGHT_FORMATS[fmt]
    block = spec["block"]
    k = shape[-1]
    rows = math.prod(shape[:-1])
    if spec.get("matrix_block"):
        n = shape[-2]
        matrices = math.prod(shape[:-2]) if len(shape) > 2 else 1
        padded_k = align_up(k, block)
        codes = rows * padded_k
        scales = matrices * math.ceil(n / block) * (padded_k // block) * spec["scale_bytes"]
        return codes + scales
    blocks = rows * math.ceil(k / block)
    return blocks * (spec["bits"] * block // 8) + blocks * spec["scale_bytes"]


def build_expected(config):
    text = config["text_config"]
    num_layers = text["num_hidden_layers"]
    layer_types = [
        "indexed_attention" if kind == "full_attention" else kind for kind in text["layer_types"]
    ]
    ple_layers = set(text.get("ple_layer_ids") or [])
    ngram_shards = range(int(text.get("split_ngram_parts", 0)))
    expected = {
        "model.language_model.embed_tokens.weight": "required",
        "lm_head.weight": "required",
        "model.language_model.hyper_connection_mixer.hc_norm.weight": "required",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": "required",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": "required",
    }
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}."
        for suffix in (
            "attn_hyper_connection.block_inject_weight.weight",
            "attn_hyper_connection.hc_norm.weight",
            "attn_hyper_connection.input_mix_weight_down.weight",
            "attn_hyper_connection.input_mix_weight_up.weight",
            "mlp_hyper_connection.block_inject_weight.weight",
            "mlp_hyper_connection.hc_norm.weight",
            "mlp_hyper_connection.input_mix_weight_down.weight",
            "mlp_hyper_connection.input_mix_weight_up.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        ):
            expected[prefix + suffix] = "required"
        if layer_types[layer] == "linear_attention":
            for suffix in (
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ):
                expected[prefix + suffix] = "required"
        else:
            for suffix in (
                "self_attn.indexer.index_qk_proj.weight",
                "self_attn.indexer.k_layernorm.weight",
                "self_attn.indexer.q_layernorm.weight",
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
            ):
                expected[prefix + suffix] = "required"
        if (layer + 1) in ple_layers:
            for suffix in (
                "ple.conv1d.weight",
                "ple.key_proj.weight",
                "ple.norm_conv.weight",
                "ple.norm_key.weight",
                "ple.norm_query.weight",
                "ple.ple_embedding.layer_multipliers",
                "ple.ple_embedding.ngram_heads_offsets",
                "ple.ple_embedding.ngram_heads_vocab_sizes",
                "ple.value_proj.weight",
            ):
                expected[prefix + suffix] = "required"
            for shard in ngram_shards:
                expected[prefix + f"ple.ple_embedding.ngram_embedding.shard_{shard}.weight"] = "required"
    return expected


def command_manifest(source, out_path):
    headers = source.headers()
    manifest = {}
    for name, spec in sorted(headers.items()):
        manifest[name] = {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "shard": spec["shard"],
            "category": classify(name),
            "disposition": DISPOSITION[classify(name)],
        }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=0)
    print(f"tensors={len(manifest)}")
    return manifest


def command_inventory(source):
    headers = source.headers()

    def key_of(name):
        key = re.sub(r"layers\.\d+", "layers.N", name)
        key = re.sub(r"blocks\.\d+", "blocks.N", key)
        key = re.sub(r"shard_\d+", "shard_N", key)
        return key

    patterns = {}
    for name, spec in headers.items():
        patterns[key_of(name)] = (spec["dtype"], tuple(spec["shape"]))
    print(f"{'category':24s} {'disposition':12s} {'dtype':8s} {'shape':30s} pattern")
    for key in sorted(patterns, key=lambda k: (classify(k), k)):
        dtype, shape = patterns[key]
        print(
            f"{classify(key):24s} {DISPOSITION[classify(key)]:12s} {dtype:8s} "
            f"{str(list(shape)):30s} {key}"
        )


def command_budget(source, out_path):
    headers = source.headers()
    categories = {category: [] for category in CATEGORY_ORDER}
    for name, spec in headers.items():
        categories[classify(name)].append(spec["shape"])
    report = {}
    totals = {fmt: 0 for fmt in WEIGHT_FORMATS}
    for category in CATEGORY_ORDER:
        shapes = categories[category]
        entry = {"params": sum(math.prod(shape) for shape in shapes), "bytes": {}}
        for fmt in WEIGHT_FORMATS:
            value = sum(tensor_bytes(shape, fmt) for shape in shapes)
            entry["bytes"][fmt] = value
            totals[fmt] += value
        report[category] = entry
    report["_residency"] = {}
    for name, members in RESIDENCY_SETS.items():
        report["_residency"][name] = {
            "params": sum(report[member]["params"] for member in members),
            "bytes": {
                fmt: sum(report[member]["bytes"][fmt] for member in members) for fmt in WEIGHT_FORMATS
            },
        }
    report["_totals"] = totals
    print(f"{'category':24s} {'params(B)':>10s} " + " ".join(f"{fmt:>12s}" for fmt in WEIGHT_FORMATS))
    for category in CATEGORY_ORDER:
        entry = report[category]
        print(
            f"{category:24s} {entry['params'] / 1e9:10.3f} "
            + " ".join(f"{entry['bytes'][fmt] / 1e9:12.2f}" for fmt in WEIGHT_FORMATS)
        )
    print("-" * 96)
    for name, entry in report["_residency"].items():
        print(
            f"{name:24s} {entry['params'] / 1e9:10.3f} "
            + " ".join(f"{entry['bytes'][fmt] / 1e9:12.2f}" for fmt in WEIGHT_FORMATS)
        )
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def command_check(source):
    config = source.config()
    headers = source.headers()
    expected = build_expected(config)
    present = set(headers)
    required = {name for name, kind in expected.items() if kind == "required"}
    missing_required = sorted(required - present)
    optional_present = sorted(name for name in present if name.startswith("mtp."))
    ignored = sorted(name for name in present if classify(name) == "vision")
    unknown = sorted(name for name in present if classify(name) == "unknown")
    print(f"present={len(present)}")
    print(f"required={len(required)} missing_required={len(missing_required)}")
    print(f"optional_present(mtp)={len(optional_present)}")
    print(f"ignored(vision)={len(ignored)}")
    print(f"unknown={len(unknown)}")
    for label, names in (("MISSING_REQUIRED", missing_required), ("UNKNOWN", unknown)):
        if names:
            print(label + ":")
            for name in names[:20]:
                print("  " + name)
    ok = not missing_required and not unknown
    print("RESULT: " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest", "inventory", "budget", "check"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = ModelSource(model_dir=args.model_dir, repo=args.repo, revision=args.revision)
    if args.command == "manifest":
        command_manifest(source, args.out)
    elif args.command == "inventory":
        command_inventory(source)
    elif args.command == "budget":
        command_budget(source, args.out)
    elif args.command == "check":
        sys.exit(command_check(source))


if __name__ == "__main__":
    main()
