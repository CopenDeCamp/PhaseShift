#!/usr/bin/env python3
"""Qwen3.5 MTP Gate 1 golden fixture exporter.

実 BF16 checkpoint を読み込み、vLLM Qwen3.5 MTP semantics で MTP 1 層の
forward を CPU eager (torch) 上で実行し、中間テンソルを golden fixture
として保存する。

PhaseNonShift の実装コードを Python 側にコピーしていない。
vLLM Qwen3.5 MTP semantics (concat [embedding, hidden]、Gemma RMSNorm (1+w)、
per-head q/gate split、per-head Q/K norm、partial RoPE、GQA attention
(kv_head = h / group)、attention output gate (sigmoid) の o_proj 前適用、
tied embedding / LM head) を直接実装する。

数値 policy:
  計算はすべて FP32 (CPU、決定論)。PhaseNonShift correctness path が
  BF16 へ丸める境界で、同じ RNE 丸め (torch bfloat16 cast) を行う。

fixture:
  <out-dir>/<case>/metadata.json
  <out-dir>/<case>/fixture.safetensors
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FIXTURE_VERSION = 1

MTP_TENSOR_NAMES = [
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]

EMBED_NAME = "model.language_model.embed_tokens.weight"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        raw = f.read()
    cfg = json.loads(raw.decode("utf-8"))
    return cfg, sha256_bytes(raw)


def build_tensor_index(model_dir: str) -> dict:
    index = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fn)
        with safe_open(path, framework="pt") as st:
            for name in st.keys():
                index[name] = path
    return index


def load_tensor(index: dict, name: str, expected_shape, expected_dtype) -> torch.Tensor:
    if name not in index:
        raise KeyError(f"tensor not found in checkpoint: {name}")
    with safe_open(index[name], framework="pt") as st:
        t = st.get_tensor(name)
    if tuple(t.shape) != tuple(expected_shape):
        raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {tuple(expected_shape)}")
    if expected_dtype is not None and t.dtype != expected_dtype:
        raise ValueError(f"{name}: dtype {t.dtype} != expected {expected_dtype}")
    return t


def sha256_tensor(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16:
        t = t.contiguous().view(torch.uint16)
    return sha256_bytes(bytes(t.contiguous().numpy().tobytes()))


class MtpWeights:
    def __init__(self, index, model_dir, cfg, H):
        self.H = H
        self.I = cfg["text_config"]["intermediate_size"] if "text_config" in cfg else cfg["intermediate_size"]
        self.num_heads = cfg["text_config"]["num_attention_heads"] if "text_config" in cfg else cfg["num_attention_heads"]
        self.num_kv_heads = cfg["text_config"]["num_key_value_heads"] if "text_config" in cfg else cfg["num_key_value_heads"]
        self.head_dim = cfg["text_config"]["head_dim"] if "text_config" in cfg else cfg["head_dim"]
        self.vocab = cfg["text_config"]["vocab_size"] if "text_config" in cfg else cfg["vocab_size"]
        self.eps = float(cfg["text_config"]["rms_norm_eps"] if "text_config" in cfg else cfg["rms_norm_eps"])
        rope = (cfg.get("text_config", cfg)).get("rope_parameters", {})
        self.rope_theta = float(rope.get("rope_theta", 10000000.0))
        self.partial = float(rope.get("partial_rotary_factor", 0.25))
        self.rotary_dim = int(round(self.head_dim * self.partial))
        self.tie = bool(cfg["text_config"]["tie_word_embeddings"] if "text_config" in cfg else cfg["tie_word_embeddings"])
        self.attn_gate = bool(cfg["text_config"]["attn_output_gate"] if "text_config" in cfg else cfg["attn_output_gate"])
        self.mtp_layers = int(cfg["text_config"]["mtp_num_hidden_layers"] if "text_config" in cfg else cfg["mtp_num_hidden_layers"])
        self.mtp_dedicated = bool(cfg["text_config"].get("mtp_use_dedicated_embeddings", False)
                                  if "text_config" in cfg else cfg.get("mtp_use_dedicated_embeddings", False))

        def f32(name, shape):
            return load_tensor(index, name, shape, torch.bfloat16).float()

        H = self.H
        I = self.I
        Q = self.num_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim
        V = self.vocab

        self.embed = load_tensor(index, EMBED_NAME, (V, H), torch.bfloat16).float()
        self.fc = f32("mtp.fc.weight", (H, 2 * H))
        self.pre_fc_norm_embedding = f32("mtp.pre_fc_norm_embedding.weight", (H,))
        self.pre_fc_norm_hidden = f32("mtp.pre_fc_norm_hidden.weight", (H,))
        self.input_layernorm = f32("mtp.layers.0.input_layernorm.weight", (H,))
        self.post_attention_layernorm = f32("mtp.layers.0.post_attention_layernorm.weight", (H,))
        self.q_proj = f32("mtp.layers.0.self_attn.q_proj.weight", (2 * Q, H))
        self.k_proj = f32("mtp.layers.0.self_attn.k_proj.weight", (KV, H))
        self.v_proj = f32("mtp.layers.0.self_attn.v_proj.weight", (KV, H))
        self.o_proj = f32("mtp.layers.0.self_attn.o_proj.weight", (H, Q))
        self.q_norm = f32("mtp.layers.0.self_attn.q_norm.weight", (self.head_dim,))
        self.k_norm = f32("mtp.layers.0.self_attn.k_norm.weight", (self.head_dim,))
        self.gate_proj = f32("mtp.layers.0.mlp.gate_proj.weight", (I, H))
        self.up_proj = f32("mtp.layers.0.mlp.up_proj.weight", (I, H))
        self.down_proj = f32("mtp.layers.0.mlp.down_proj.weight", (H, I))
        self.final_norm = f32("mtp.norm.weight", (H,))

        self.sha256 = {}
        for name in MTP_TENSOR_NAMES:
            with safe_open(index[name], framework="pt") as st:
                self.sha256[name] = sha256_tensor(st.get_tensor(name))
        self.embed_sha256 = sha256_tensor(self.embed.to(torch.bfloat16))

    def self_check(self, H):
        Q = self.num_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim
        assert self.embed.shape == (self.vocab, H)
        assert self.fc.shape == (H, 2 * H)
        assert self.q_proj.shape == (2 * Q, H)
        assert self.k_proj.shape == (KV, H)
        assert self.v_proj.shape == (KV, H)
        assert self.o_proj.shape == (H, Q)
        assert self.gate_proj.shape == (self.I, H)
        assert self.up_proj.shape == (self.I, H)
        assert self.down_proj.shape == (H, self.I)
        assert self.mtp_layers == 1
        assert self.attn_gate is True
        assert self.tie is True
        assert self.mtp_dedicated is False
        assert self.rotary_dim % 2 == 0 and self.rotary_dim <= self.head_dim


def to_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, group: int) -> torch.Tensor:
    n = x.numel() // group
    xg = x.view(n, group)
    var = (xg * xg).mean(dim=1, keepdim=True)
    y = xg * torch.rsqrt(var + eps)
    if w.numel() == group:
        y = y * (1.0 + w.view(1, group))
    else:
        y = y * (1.0 + w.view(n, group))
    return y.view(-1)


def linear_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w.t()


def split_q_gate(qproj: torch.Tensor, heads: int, hd: int):
    x = qproj.view(heads, 2 * hd)
    q = x[:, :hd].reshape(-1).contiguous()
    g = x[:, hd:].reshape(-1).contiguous()
    return q, g


def rope(x: torch.Tensor, position: int, heads: int, hd: int, rotary: int, theta: float) -> torch.Tensor:
    half = rotary // 2
    i = torch.arange(half, dtype=torch.float64)
    freqs = theta ** (-i / half)
    ang = float(position) * freqs
    cs = torch.cos(ang).float()
    sn = torch.sin(ang).float()
    xh = x.view(heads, hd)
    out = xh.clone()
    x0 = xh[:, :half]
    x1 = xh[:, half:2 * half]
    out[:, :half] = x0 * cs - x1 * sn
    out[:, half:2 * half] = x1 * cs + x0 * sn
    return out.reshape(-1).contiguous()


def attention_isolated(q_rope: torch.Tensor, v: torch.Tensor, q_heads: int, kv_heads: int, hd: int) -> torch.Tensor:
    group = q_heads // kv_heads
    kv = v.view(kv_heads, hd)
    out = torch.empty(q_heads, hd)
    for h in range(q_heads):
        out[h] = kv[h // group]
    return out.reshape(-1).contiguous()


def mtp_forward(W, token: int, position: int, target_hidden: torch.Tensor):
    H = W.H
    Q = W.num_heads * W.head_dim
    KV = W.num_kv_heads * W.head_dim
    hd = W.head_dim
    eps = W.eps
    t = {}

    emb = W.embed[token]
    t["embedding_raw"] = (emb, "bf16")

    en = to_bf16(gemma_rmsnorm(emb, W.pre_fc_norm_embedding, eps, H))
    t["embedding_norm"] = (en, "bf16")
    hn = to_bf16(gemma_rmsnorm(target_hidden, W.pre_fc_norm_hidden, eps, H))
    t["target_hidden_norm"] = (hn, "bf16")

    cat = torch.cat([en, hn])
    t["concat_embedding_hidden"] = (cat, "bf16")

    fc_out = to_bf16(linear_f32(cat, W.fc))
    t["fc_out"] = (fc_out, "bf16")
    t["residual0"] = (fc_out, "bf16")

    normed = to_bf16(gemma_rmsnorm(fc_out, W.input_layernorm, eps, H))
    t["input_layernorm_out"] = (normed, "bf16")

    qproj = to_bf16(linear_f32(normed, W.q_proj))
    kproj = to_bf16(linear_f32(normed, W.k_proj))
    vproj = to_bf16(linear_f32(normed, W.v_proj))
    t["q_proj_raw"] = (qproj, "bf16")
    t["k_proj_raw"] = (kproj, "bf16")
    t["v_proj_raw"] = (vproj, "bf16")

    q, gate = split_q_gate(qproj, W.num_heads, hd)
    t["q_pre_norm"] = (q, "bf16")
    t["gate_pre_sigmoid"] = (gate, "bf16")

    qn = gemma_rmsnorm(q, W.q_norm, eps, hd)
    t["q_norm"] = (qn, "f32")
    kn = gemma_rmsnorm(kproj, W.k_norm, eps, hd)
    t["k_norm"] = (kn, "f32")

    qr = to_bf16(rope(qn, position, W.num_heads, hd, W.rotary_dim, W.rope_theta))
    kr = to_bf16(rope(kn, position, W.num_kv_heads, hd, W.rotary_dim, W.rope_theta))
    t["q_rope"] = (qr, "bf16")
    t["k_rope"] = (kr, "bf16")
    t["v"] = (vproj, "bf16")

    ac = attention_isolated(qr, vproj, W.num_heads, W.num_kv_heads, hd)
    t["attention_core_out"] = (ac, "f32")

    gs = torch.sigmoid(gate)
    t["gate_sigmoid"] = (gs, "f32")

    att = to_bf16(ac * gs)
    t["attention_gated"] = (att, "bf16")

    oout = to_bf16(linear_f32(att, W.o_proj))
    t["o_proj_out"] = (oout, "bf16")

    res1 = to_bf16(oout + fc_out)
    t["residual1"] = (res1, "bf16")

    pnorm = to_bf16(gemma_rmsnorm(res1, W.post_attention_layernorm, eps, H))
    t["post_attention_norm_out"] = (pnorm, "bf16")

    g = to_bf16(linear_f32(pnorm, W.gate_proj))
    u = to_bf16(linear_f32(pnorm, W.up_proj))
    t["mlp_gate_proj"] = (g, "bf16")
    t["mlp_up_proj"] = (u, "bf16")
    silu_g = g / (1.0 + torch.exp(-g))
    t["mlp_silu_gate"] = (silu_g, "f32")
    sw = to_bf16(u * silu_g)
    t["mlp_mul"] = (sw, "bf16")

    d = to_bf16(linear_f32(sw, W.down_proj))
    t["mlp_down_proj"] = (d, "bf16")

    fres = to_bf16(d + res1)
    t["final_residual"] = (fres, "bf16")
    t["selected_hidden"] = (fres, "bf16")

    fn = to_bf16(gemma_rmsnorm(fres, W.final_norm, eps, H))
    t["final_norm"] = (fn, "bf16")

    logits = linear_f32(fn, W.embed)
    t["logits"] = (logits, "f32")

    top1 = int(torch.argmax(logits).item())
    t["argmax"] = (torch.tensor([top1], dtype=torch.int32), "i32")
    return t, top1


def validate_tensors(t, W):
    H = W.H
    Q = W.num_heads * W.head_dim
    KV = W.num_kv_heads * W.head_dim
    I = W.I
    expected = {
        "embedding_raw": (H, "bf16"),
        "embedding_norm": (H, "bf16"),
        "target_hidden_norm": (H, "bf16"),
        "concat_embedding_hidden": (2 * H, "bf16"),
        "fc_out": (H, "bf16"),
        "residual0": (H, "bf16"),
        "input_layernorm_out": (H, "bf16"),
        "q_proj_raw": (2 * Q, "bf16"),
        "k_proj_raw": (KV, "bf16"),
        "v_proj_raw": (KV, "bf16"),
        "q_pre_norm": (Q, "bf16"),
        "gate_pre_sigmoid": (Q, "bf16"),
        "q_norm": (Q, "f32"),
        "k_norm": (KV, "f32"),
        "q_rope": (Q, "bf16"),
        "k_rope": (KV, "bf16"),
        "v": (KV, "bf16"),
        "attention_core_out": (Q, "f32"),
        "gate_sigmoid": (Q, "f32"),
        "attention_gated": (Q, "bf16"),
        "o_proj_out": (H, "bf16"),
        "residual1": (H, "bf16"),
        "post_attention_norm_out": (H, "bf16"),
        "mlp_gate_proj": (I, "bf16"),
        "mlp_up_proj": (I, "bf16"),
        "mlp_silu_gate": (I, "f32"),
        "mlp_mul": (I, "bf16"),
        "mlp_down_proj": (H, "bf16"),
        "final_residual": (H, "bf16"),
        "selected_hidden": (H, "bf16"),
        "final_norm": (H, "bf16"),
        "logits": (W.vocab, "f32"),
        "argmax": (1, "i32"),
    }
    for name, (n, dtype) in expected.items():
        if name not in t:
            raise AssertionError(f"missing tensor: {name}")
        val, dt = t[name]
        if dt != dtype:
            raise AssertionError(f"{name}: dtype {dt} != {dtype}")
        if val.numel() != n:
            raise AssertionError(f"{name}: numel {val.numel()} != {n}")
        if not torch.isfinite(val).all():
            raise AssertionError(f"{name}: non-finite value")


def make_target_hidden(seed: int, H: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(H, generator=g, dtype=torch.float32)
    return to_bf16(x)


def run_case(W, case: str, token: int, position: int, target_hidden: torch.Tensor):
    t, top1 = mtp_forward(W, token, position, target_hidden)
    validate_tensors(t, W)
    t2, top1_2 = mtp_forward(W, token, position, target_hidden)
    for name in t:
        if not torch.equal(t[name][0], t2[name][0]):
            raise AssertionError(f"reproducibility failed for {case}/{name}")
    if top1 != top1_2:
        raise AssertionError(f"reproducibility failed for {case}/argmax")
    return t, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts/mtp_gate1")
    ap.add_argument("--token", type=int, default=925)
    ap.add_argument("--token-alt", type=int, default=5678)
    ap.add_argument("--hidden-seed", type=int, default=42)
    ap.add_argument("--positions", default="0,1,127")
    args = ap.parse_args()

    torch.set_num_threads(1)

    cfg, cfg_sha = load_config(args.model_dir)
    tc = cfg.get("text_config", cfg)
    H = int(tc["hidden_size"])
    index = build_tensor_index(args.model_dir)
    W = MtpWeights(index, args.model_dir, cfg, H)
    W.self_check(H)

    exporter_sha = sha256_bytes(open(__file__, "rb").read())
    positions = [int(p) for p in args.positions.split(",")]
    target_hidden = make_target_hidden(args.hidden_seed, H)

    cases = []
    for p in positions:
        cases.append((f"position_{p}", args.token, p))
    cases.append(("token_alt", args.token_alt, positions[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    for case, token, position in cases:
        print(f"[export] case={case} token={token} position={position}")
        t, top1 = run_case(W, case, token, position, target_hidden)
        topk = torch.topk(t["logits"][0], 10, largest=True, sorted=True)
        topk_ids = [int(x) for x in topk.indices.tolist()]
        topk_logits = [float(x) for x in topk.values.tolist()]

        tensors_out = {}
        tensor_sha = {}
        tensor_shapes = {}
        for name, (val, dt) in t.items():
            if dt == "bf16":
                stored = val.to(torch.bfloat16)
            elif dt == "f32":
                stored = val.to(torch.float32)
            else:
                stored = val.to(torch.int32)
            tensors_out[name] = stored
            tensor_sha[name] = sha256_tensor(stored)
            tensor_shapes[name] = list(stored.shape)
        tensors_out["input_token_id"] = torch.tensor([token], dtype=torch.int32)
        tensor_sha["input_token_id"] = sha256_tensor(tensors_out["input_token_id"])
        tensor_shapes["input_token_id"] = [1]
        tensors_out["position"] = torch.tensor([position], dtype=torch.int32)
        tensor_sha["position"] = sha256_tensor(tensors_out["position"])
        tensor_shapes["position"] = [1]
        tensors_out["target_hidden"] = target_hidden.to(torch.bfloat16)
        tensor_sha["target_hidden"] = sha256_tensor(tensors_out["target_hidden"])
        tensor_shapes["target_hidden"] = list(tensors_out["target_hidden"].shape)

        out_case = os.path.join(args.out_dir, case)
        os.makedirs(out_case, exist_ok=True)
        save_file(tensors_out, os.path.join(out_case, "fixture.safetensors"))

        metadata = {
            "fixture_version": FIXTURE_VERSION,
            "case": case,
            "model_path": os.path.abspath(args.model_dir),
            "config_sha256": cfg_sha,
            "mtp_tensor_sha256": W.sha256,
            "embed_sha256": W.embed_sha256,
            "token_id": token,
            "position": position,
            "hidden_seed": args.hidden_seed,
            "hidden_sha256": tensor_sha["target_hidden"],
            "H": H,
            "I": W.I,
            "Q": W.num_heads * W.head_dim,
            "KV": W.num_kv_heads * W.head_dim,
            "num_heads": W.num_heads,
            "num_kv_heads": W.num_kv_heads,
            "head_dim": W.head_dim,
            "rms_norm_eps": W.eps,
            "rope_theta": W.rope_theta,
            "partial_rotary_factor": W.partial,
            "rotary_dim": W.rotary_dim,
            "attn_output_gate": W.attn_gate,
            "tie_word_embeddings": W.tie,
            "mtp_use_dedicated_embeddings": W.mtp_dedicated,
            "mtp_num_hidden_layers": W.mtp_layers,
            "dtype": "bf16",
            "attention_mode": "isolated_t1",
            "reference_revision": exporter_sha,
            "reference_impl": "vllm-qwen35-mtp-semantics/cpu-torch-eager",
            "top1_token_id": top1,
            "top1_logit": float(t["logits"][0][top1].item()),
            "top10_token_ids": topk_ids,
            "top10_logits": topk_logits,
            "tensor_sha256": tensor_sha,
            "tensor_shapes": tensor_shapes,
        }
        with open(os.path.join(out_case, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[export] case={case} top1={top1} tensors={len(tensors_out)} "
              f"-> {out_case}")

    print("MTP_GATE1_EXPORT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
