#!/usr/bin/env python3
"""Qwen3.5 MTP weight contract inspector.

Inspects a Qwen3.5 model directory without a GPU and reports:
  * config information (hidden_size, MTP layer count, ...)
  * the full inventory of `mtp.` logical tensors (name / dtype / shape /
    numel / bytes / shard), exactly as recorded in the checkpoint
  * an automatic contract check against the vLLM Qwen3.5 MTP reference
    (fc = Linear(2H -> H), i.e. `mtp.fc.weight` must be [H, 2H];
    attention shapes are compared against a target body full-attention
    layer so that gated q_proj ([2Q, H]) is handled without assumption)
  * MTP layer count (runtime currently supports exactly 1)
  * lm_head / embed_tokens contract
  * SHA-256 fingerprints of the checkpoint logical tensor payloads
  * optional floating point statistics for BF16/F32 tensors

For PhaseShift quantized checkpoints (phaseshift_quantization.json) the
logical tensors and their encodings are read from the manifest, and the
fingerprint is computed over the physical streams (data / codes /
metadata1 / ...) in a stable order.

Usage:
    python3 tools/inspect_qwen35_mtp_weights.py <model_dir> [--no-stats]

Exit codes: 0 = MTP_WEIGHT_CONTRACT PASS (or MTP absent), 1 = FAIL, 2 = usage/IO error.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

MTP_TOP_LEVEL_TENSOR_SUFFIXES = (
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.fc.weight",
)

MTP_LAYER_TEMPLATES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)

NORM_LAYER_TEMPLATES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


class TensorInfo:
    def __init__(self, name, dtype, shape, shard, data, streams):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shard = shard
        self.data = data
        self.streams = streams

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return len(self.data)

    def shape_str(self):
        return "[" + ",".join(str(d) for d in self.shape) + "]"


def read_safetensors_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def read_tensor_payload(path, data_base, name, header):
    meta = header[name]
    data_begin = meta["data_offsets"][0]
    data_end = meta["data_offsets"][1]
    with open(path, "rb") as f:
        f.seek(data_base + data_begin)
        return f.read(data_end - data_begin)


def collect_plain_safetensors(model_dir):
    """Returns (tensors, weight_map) for a plain safetensors model dir.

    tensors: logical name -> TensorInfo. Payload bytes are only read for
    `mtp.` tensors; every other tensor is spec-only (data=None) so that a
    multi-GB model directory can be inspected with a small memory footprint.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
    else:
        weight_map = {}
        candidates = sorted(model_dir.glob("*.safetensors"))
        if len(candidates) != 1:
            print(f"ERROR: cannot resolve shards (no index, {len(candidates)} shards)",
                  file=sys.stderr)
            sys.exit(2)
        shard_names = [candidates[0].name]

    tensors = {}
    for shard in shard_names:
        header, data_base = read_safetensors_header(model_dir / shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map and weight_map[name] != shard:
                continue
            data = None
            streams = None
            if name.startswith("mtp."):
                data = read_tensor_payload(model_dir / shard, data_base, name, header)
                streams = {"data": data}
            tensors[name] = TensorInfo(name, meta["dtype"], meta["shape"], shard, data, streams)
    return tensors, weight_map


def collect_quantized_safetensors(model_dir):
    """Returns (tensors, manifest) for a PhaseShift quantized model dir.

    Only `mtp.` logical tensors are materialized (payload bytes included).
    Non-MTP logical tensors are reported spec-only so the tool stays fast.
    """
    manifest = json.loads((model_dir / "phaseshift_quantization.json").read_text())
    plain, _ = collect_plain_safetensors(model_dir)
    tensors = {}
    for logical, meta in manifest.get("tensors", {}).items():
        is_mtp = logical.startswith("mtp.")
        streams = {}
        if is_mtp:
            for key in ("data", "codes", "metadata1", "metadata2", "metadata3", "metadata4"):
                ref = meta.get(key)
                if not ref:
                    continue
                physical = ref["tensor"]
                if physical not in plain or plain[physical].data is None:
                    raise KeyError(f"physical tensor missing from safetensors: {physical}")
                streams[key] = plain[physical].data
            payload = b"".join(streams[k] for k in
                               ("data", "codes", "metadata1", "metadata2", "metadata3",
                                "metadata4")
                               if k in streams)
        else:
            payload = None
        tensors[logical] = TensorInfo(
            logical,
            meta.get("encoding", "unknown"),
            [int(d) for d in meta.get("logical_shape", [])],
            "<manifest>",
            payload,
            streams,
        )
    return tensors, manifest


def read_config(model_dir):
    cfg = json.loads((model_dir / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    return cfg, text


def get_value(text, *keys):
    for k in keys:
        if k in text and isinstance(text[k], (int, float, bool, str)):
            return text[k], k
    return None, None


def fmt(value):
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def parse_layer_index(name):
    prefix = "mtp.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    idx_str = ""
    for ch in rest:
        if ch.isdigit():
            idx_str += ch
        else:
            break
    if not idx_str or rest[len(idx_str):len(idx_str) + 1] != ".":
        return None
    return int(idx_str)


def bf16_stats(data):
    n = len(data) // 2
    total = 0.0
    sq = 0.0
    mn = math.inf
    mx = -math.inf
    finite = 0
    nan_count = 0
    inf_count = 0
    for i in range(n):
        raw = int.from_bytes(data[i * 2:i * 2 + 2], "little")
        v = struct.unpack("<f", struct.pack("<I", raw << 16))[0]
        if math.isnan(v):
            nan_count += 1
            continue
        if math.isinf(v):
            inf_count += 1
            continue
        finite += 1
        total += v
        sq += v * v
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    if finite == 0:
        mn = mx = total = sq = 0.0
    return {
        "min": mn, "max": mx,
        "mean": total / finite if finite else 0.0,
        "l2": math.sqrt(sq),
        "finite": finite, "nan": nan_count, "inf": inf_count,
    }


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def dtype_key(dtype):
    return dtype.upper() if isinstance(dtype, str) else dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--no-stats", action="store_true",
                        help="skip floating point statistics")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: not a directory: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "config.json").is_file():
        print(f"ERROR: missing config.json in {model_dir}", file=sys.stderr)
        return 2

    quantized_meta = model_dir / "phaseshift_quantization.json"
    cfg, text = read_config(model_dir)

    print("=== Qwen3.5 MTP Weight Contract Inspection ===")
    print(f"model_dir: {model_dir}")
    print()
    print("--- config ---")
    checks = []
    warnings = []

    model_type, _ = get_value(cfg, "model_type")
    text_type, _ = get_value(text, "model_type")
    print(f"model_type: {fmt(model_type)}")
    print(f"text model_type: {fmt(text_type)}")

    vocab_size, _ = get_value(text, "vocab_size")
    hidden_size, _ = get_value(text, "hidden_size")
    intermediate_size, _ = get_value(text, "intermediate_size")
    num_hidden_layers, _ = get_value(text, "num_hidden_layers")
    mtp_layers_cfg, _ = get_value(text, "mtp_num_hidden_layers")
    n_heads, _ = get_value(text, "num_attention_heads")
    n_kv_heads, _ = get_value(text, "num_key_value_heads")
    head_dim, head_dim_key = get_value(text, "head_dim", "attention_head_dim")
    tie, _ = get_value(cfg, "tie_word_embeddings")
    if tie is None:
        tie, _ = get_value(text, "tie_word_embeddings")
    attn_output_gate, _ = get_value(text, "attn_output_gate")
    mtp_dedicated_emb, _ = get_value(text, "mtp_use_dedicated_embeddings")
    quant_cfg = "phaseshift_quantization.json" if quantized_meta.is_file() else (
        "phaseshift_quantization (in config.json)" if "phaseshift_quantization" in cfg else "none")

    print(f"hidden_size: {fmt(hidden_size)}")
    print(f"intermediate_size: {fmt(intermediate_size)}")
    print(f"vocab_size: {fmt(vocab_size)}")
    print(f"num_hidden_layers: {fmt(num_hidden_layers)}")
    print(f"mtp_num_hidden_layers: {fmt(mtp_layers_cfg)}")
    print(f"num_attention_heads: {fmt(n_heads)}")
    print(f"num_key_value_heads: {fmt(n_kv_heads)}")
    print(f"attention_head_dim: {fmt(head_dim)} (key: {head_dim_key or 'MISSING'})")
    print(f"tie_word_embeddings: {fmt(tie)}")
    print(f"attn_output_gate: {fmt(attn_output_gate)}")
    print(f"mtp_use_dedicated_embeddings: {fmt(mtp_dedicated_emb)}")
    print(f"quantization config: {quant_cfg}")
    print()

    H = hidden_size if isinstance(hidden_size, int) else None
    I = intermediate_size if isinstance(intermediate_size, int) else None
    Q = n_heads * head_dim if isinstance(n_heads, int) and isinstance(head_dim, int) else None
    KV = n_kv_heads * head_dim if isinstance(n_kv_heads, int) and isinstance(head_dim, int) else None
    print(f"derived: H={H} I={I} Q={Q} KV={KV} "
          f"q_proj_out={'2Q=' + str(2 * Q) if Q is not None else 'MISSING'}")
    print()

    if quantized_meta.is_file():
        tensors, manifest = collect_quantized_safetensors(model_dir)
    else:
        tensors, weight_map = collect_plain_safetensors(model_dir)

    mtp_names = sorted(n for n in tensors if n.startswith("mtp."))
    print(f"--- MTP tensor inventory ({len(mtp_names)}) ---")
    for name in mtp_names:
        t = tensors[name]
        print(f"{name} dtype={t.dtype} shape={t.shape_str()} numel={t.numel} "
              f"bytes={t.nbytes} shard={t.shard}")
    print()

    if not mtp_names:
        print("MTP_WEIGHT_CONTRACT: N/A (no MTP tensors in checkpoint)")
        return 0

    print("--- MTP tensor fingerprint (SHA-256 of logical payload) ---")
    for name in mtp_names:
        t = tensors[name]
        streams = "/".join(sorted(t.streams.keys())) if t.streams else "data"
        print(f"{name} sha256={sha256_hex(t.data)} streams={streams}")
    print()

    layer_indices = sorted({i for i in (parse_layer_index(n) for n in mtp_names) if i is not None})

    print("--- MTP layer count ---")
    print(f"checkpoint MTP layers: {len(layer_indices)} (indices: {layer_indices})")
    print(f"config mtp_num_hidden_layers: {fmt(mtp_layers_cfg)}")
    print(f"runtime supported MTP layers: 1")
    if len(layer_indices) > 1:
        checks.append((f"mtp layer count: checkpoint has {len(layer_indices)} layers, "
                       f"runtime supports 1 -> unsupported", False))
    elif len(layer_indices) != 1:
        checks.append((f"mtp layer count: no mtp.layers.0 found (indices: {layer_indices})", False))
    else:
        checks.append(("mtp layer count: 1", True))
        if isinstance(mtp_layers_cfg, int) and mtp_layers_cfg != 1:
            checks.append((f"mtp layer count: config says {mtp_layers_cfg}, checkpoint has 1", False))
    if layer_indices != list(range(len(layer_indices))):
        checks.append((f"mtp layer indices not contiguous from 0: {layer_indices}", False))
    print()

    print("--- contract check ---")

    def check_tensor(name, expected_shape, expected_dtype=None, note=""):
        if name not in tensors:
            checks.append((f"{name}: MISSING", False))
            print(f"[FAIL] {name}: MISSING")
            return
        t = tensors[name]
        problems = []
        if t.shape != expected_shape:
            problems.append(f"shape {t.shape_str()} != expected {expected_shape}")
        if expected_dtype is not None and dtype_key(t.dtype) != expected_dtype:
            problems.append(f"dtype {t.dtype} != expected {expected_dtype}")
        if problems:
            checks.append((f"{name}: " + "; ".join(problems), False))
            print(f"[FAIL] {name} {t.shape_str()} dtype={t.dtype} "
                  f"(expected {expected_shape}) {note}")
        else:
            checks.append((f"{name}: {t.shape_str()}", True))
            print(f"[OK]   {name} {t.shape_str()} dtype={t.dtype} {note}")

    if H is None or I is None or Q is None or KV is None:
        checks.append(("config: missing dimensions, cannot derive expected shapes", False))
        print("[FAIL] config: missing dimensions")
        print()

    if H is not None:
        check_tensor("mtp.pre_fc_norm_embedding.weight", [H], "BF16", "== [H]")
        check_tensor("mtp.pre_fc_norm_hidden.weight", [H], "BF16", "== [H]")
        check_tensor("mtp.norm.weight", [H], "BF16", "== [H]")

    if H is not None:
        check_tensor("mtp.fc.weight", [H, 2 * H], "BF16",
                     "== [H, 2H] (Linear(2H -> H), PyTorch [out, in])")

    # Body full-attention reference layer.
    body_full_layers = [
        int(n.split(".")[3])
        for n in tensors
        if n.startswith("model.language_model.layers.")
        and ".self_attn.q_proj.weight" in n
    ]
    body_ref = None
    if body_full_layers:
        li = sorted(body_full_layers)[0]
        prefix = f"model.language_model.layers.{li}."
        body_ref = prefix
        print(f"body full-attention reference layer: {prefix}self_attn")

    if H is not None and I is not None:
        for li in layer_indices:
            base = f"mtp.layers.{li}."
            check_tensor(base + "input_layernorm.weight", [H], "BF16", "== [H]")
            check_tensor(base + "post_attention_layernorm.weight", [H], "BF16", "== [H]")
            check_tensor(base + "mlp.gate_proj.weight", [I, H], "BF16", "== [I, H]")
            check_tensor(base + "mlp.up_proj.weight", [I, H], "BF16", "== [I, H]")
            check_tensor(base + "mlp.down_proj.weight", [H, I], "BF16", "== [H, I]")

    if H is not None and Q is not None and KV is not None:
        for li in layer_indices:
            base = f"mtp.layers.{li}."
            if body_ref is not None:
                for tpl in MTP_LAYER_TEMPLATES:
                    mtp_name = base + tpl
                    body_name = body_ref + tpl
                    if mtp_name in tensors and body_name in tensors:
                        if tensors[mtp_name].shape != tensors[body_name].shape:
                            checks.append((
                                f"{mtp_name}: differs from body {body_name} "
                                f"({tensors[mtp_name].shape_str()} vs {tensors[body_name].shape_str()})",
                                False))
                            print(f"[FAIL] {mtp_name} {tensors[mtp_name].shape_str()} "
                                  f"!= body {body_name} {tensors[body_name].shape_str()}")
                        else:
                            checks.append((f"{mtp_name}: matches body decoder layer contract", True))
                            print(f"[OK]   {mtp_name} {tensors[mtp_name].shape_str()} "
                                  f"== body {body_name}")
                    elif mtp_name in tensors:
                        checks.append((f"{mtp_name}: no body reference tensor", True))
                        print(f"[WARN] {mtp_name}: body reference not found, skipped")
                        warnings.append(f"{mtp_name}: body reference not found")
            q_out = 2 * Q if attn_output_gate is True else Q
            if body_ref is not None:
                q_out = tensors[body_ref + "self_attn.q_proj.weight"].shape[0]
            check_tensor(base + "self_attn.q_proj.weight", [q_out, H], "BF16",
                         f"== [q_proj_out, H] (attn_output_gate={attn_output_gate})")
            check_tensor(base + "self_attn.k_proj.weight", [KV, H], "BF16", "== [KV, H]")
            check_tensor(base + "self_attn.v_proj.weight", [KV, H], "BF16", "== [KV, H]")
            check_tensor(base + "self_attn.o_proj.weight", [H, Q], "BF16", "== [H, Q]")
            if isinstance(head_dim, int):
                check_tensor(base + "self_attn.q_norm.weight", [head_dim], "BF16",
                             "== [head_dim]")
                check_tensor(base + "self_attn.k_norm.weight", [head_dim], "BF16",
                             "== [head_dim]")

    known = set(MTP_TOP_LEVEL_TENSOR_SUFFIXES)
    for li in layer_indices:
        for tpl in MTP_LAYER_TEMPLATES:
            known.add(f"mtp.layers.{li}." + tpl)
    for name in mtp_names:
        if name not in known:
            warnings.append(f"unexpected MTP tensor: {name}")
            print(f"[WARN] unexpected MTP tensor: {name}")

    print()

    print("--- lm_head / embed_tokens ---")
    embed_name = None
    for cand in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
        if cand in tensors:
            embed_name = cand
            break
    lm_name = None
    for cand in ("model.language_model.lm_head.weight", "lm_head.weight"):
        if cand in tensors:
            lm_name = cand
            break
    if embed_name is None:
        checks.append(("embed_tokens.weight: MISSING", False))
        print("[FAIL] embed_tokens.weight: MISSING")
    else:
        t = tensors[embed_name]
        if isinstance(vocab_size, int) and isinstance(H, int) and t.shape == [vocab_size, H]:
            checks.append((f"{embed_name}: {t.shape_str()}", True))
            print(f"[OK]   {embed_name} {t.shape_str()} dtype={t.dtype} == [vocab, H]")
        else:
            checks.append((f"{embed_name}: {t.shape_str()} != [vocab, H]", False))
            print(f"[FAIL] {embed_name} {t.shape_str()} dtype={t.dtype} (expected [vocab, H])")
    if tie is True:
        if lm_name is not None:
            warnings.append("tie_word_embeddings=true but a separate lm_head.weight exists")
            print(f"[WARN] tie_word_embeddings=true but {lm_name} exists")
        else:
            checks.append(("lm_head: tied to embed_tokens (no separate tensor)", True))
            print("[OK]   lm_head: tied to embed_tokens (no separate tensor)")
    else:
        if lm_name is None:
            checks.append(("lm_head.weight: MISSING (tie_word_embeddings=false)", False))
            print("[FAIL] lm_head.weight: MISSING (tie_word_embeddings=false)")
        elif isinstance(vocab_size, int) and isinstance(H, int) and \
                tensors[lm_name].shape == [vocab_size, H]:
            checks.append((f"{lm_name}: {tensors[lm_name].shape_str()}", True))
            print(f"[OK]   {lm_name} {tensors[lm_name].shape_str()} dtype={tensors[lm_name].dtype} "
                  "== [vocab, H]")
        else:
            checks.append((f"{lm_name}: {tensors[lm_name].shape_str()} != [vocab, H]", False))
            print(f"[FAIL] {lm_name} {tensors[lm_name].shape_str()} (expected [vocab, H])")
    if mtp_dedicated_emb is not False:
        warnings.append("mtp_use_dedicated_embeddings is not false; MTP embedding sharing "
                        "with the target model is not confirmed")
        print(f"[WARN] mtp_use_dedicated_embeddings={fmt(mtp_dedicated_emb)}")
    print()

    if not args.no_stats:
        print("--- floating point statistics (BF16) ---")
        for name in mtp_names:
            t = tensors[name]
            if dtype_key(t.dtype) != "BF16":
                print(f"{name}: dtype={t.dtype}, skipped")
                continue
            s = bf16_stats(t.data)
            print(f"{name}: min={s['min']:.6g} max={s['max']:.6g} mean={s['mean']:.6g} "
                  f"l2={s['l2']:.6g} finite={s['finite']} nan={s['nan']} inf={s['inf']}")
        print()

    print("--- summary ---")
    failures = [c for c in checks if not c[1]]
    for c in checks:
        if c[1]:
            continue
        print(f"FAIL: {c[0]}")
    for w in warnings:
        print(f"WARN: {w}")
    if failures:
        print(f"MTP_WEIGHT_CONTRACT: FAIL ({len(failures)} failing check(s))")
        return 1
    print("MTP_WEIGHT_CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
