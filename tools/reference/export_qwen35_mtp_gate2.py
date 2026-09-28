#!/usr/bin/env python3
"""Qwen3.5 MTP Gate 2 golden fixture exporter.

確定済み token 列を MTP へ teacher-force したときの persistent KV / position /
token-hidden alignment を検証するための golden fixture を作る。

Gate 1 exporter の数式実装（vLLM Qwen3.5 MTP semantics）を再利用しつつ、
Gate 2 では logical KV history を Python list として保持し、attention を毎 step
履歴全体から直接再計算する（physical cache を持ち込まない）。

出力:
  <out-dir>/synthetic/metadata.json
  <out-dir>/synthetic/sequence.safetensors
  <out-dir>/synthetic/checkpoints/step_<n>.safetensors
  <out-dir>/synthetic/offset_<p>/...        （position offset 検証用・短い列）
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

FIXTURE_VERSION = 2


def load_gate1(path):
    spec = importlib.util.spec_from_file_location("gate1_exporter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_tensor(t: torch.Tensor) -> str:
    if t.dtype == torch.bfloat16:
        t = t.contiguous().view(torch.uint16)
    return hashlib.sha256(bytes(t.contiguous().numpy().tobytes())).hexdigest()


def build_tokens(W, steps, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, W.vocab, (steps,), generator=g).tolist()


def build_hidden(W, steps, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(steps, W.H, generator=g, dtype=torch.float32)
    return x.to(torch.bfloat16)


def attention_history(g1, q_rope, k_hist, v_hist, q_heads, kv_heads, hd):
    n = len(k_hist)
    K = torch.stack(k_hist).flatten().float().view(n, kv_heads, hd)
    V = torch.stack(v_hist).flatten().float().view(n, kv_heads, hd)
    group = q_heads // kv_heads
    idx = [h // group for h in range(q_heads)]
    Kq = K[:, idx, :]
    Vq = V[:, idx, :]
    q = q_rope.float().view(q_heads, hd)
    scale = 1.0 / float(hd) ** 0.5
    scores = torch.einsum("qh,nqh->nq", q, Kq) * scale
    probs = torch.softmax(scores, dim=0)
    out = torch.einsum("nq,nqh->qh", probs, Vq)
    return out.reshape(-1).contiguous(), probs


def mtp_step(g1, W, token, position, target_hidden, k_hist, v_hist):
    H = W.H
    hd = W.head_dim
    eps = W.eps
    t = {}

    emb = W.embed[token]
    t["embedding_raw"] = emb

    en = g1.to_bf16(g1.gemma_rmsnorm(emb, W.pre_fc_norm_embedding, eps, H))
    t["embedding_norm"] = en
    hn = g1.to_bf16(g1.gemma_rmsnorm(target_hidden, W.pre_fc_norm_hidden, eps, H))
    t["target_hidden_norm"] = hn

    cat = torch.cat([en, hn])
    t["concat_embedding_hidden"] = cat
    fc_out = g1.to_bf16(g1.linear_f32(cat, W.fc))
    t["fc_out"] = fc_out

    normed = g1.to_bf16(g1.gemma_rmsnorm(fc_out, W.input_layernorm, eps, H))
    t["input_layernorm_out"] = normed

    qproj = g1.to_bf16(g1.linear_f32(normed, W.q_proj))
    kproj = g1.to_bf16(g1.linear_f32(normed, W.k_proj))
    vproj = g1.to_bf16(g1.linear_f32(normed, W.v_proj))
    t["q_proj_raw"] = qproj
    t["k_proj_raw"] = kproj
    t["v_proj_raw"] = vproj

    q, gate = g1.split_q_gate(qproj, W.num_heads, hd)
    t["q_pre_norm"] = q
    t["gate_pre_sigmoid"] = gate

    qn = g1.gemma_rmsnorm(q, W.q_norm, eps, hd)
    kn = g1.gemma_rmsnorm(kproj, W.k_norm, eps, hd)
    t["q_norm"] = qn
    t["k_norm"] = kn

    qr = g1.to_bf16(g1.rope(qn, position, W.num_heads, hd, W.rotary_dim, W.rope_theta))
    kr = g1.to_bf16(g1.rope(kn, position, W.num_kv_heads, hd, W.rotary_dim, W.rope_theta))
    t["q_rope"] = qr
    t["k_rope"] = kr
    t["v"] = vproj

    ac, probs = attention_history(
        g1, qr, k_hist + [kr], v_hist + [vproj], W.num_heads, W.num_kv_heads, hd)
    t["attention_core_out"] = ac
    t["attention_probs_sum"] = probs.sum(dim=0)

    gs = torch.sigmoid(gate)
    t["gate_sigmoid"] = gs
    att = g1.to_bf16(ac * gs)
    t["attention_gated"] = att

    oout = g1.to_bf16(g1.linear_f32(att, W.o_proj))
    t["o_proj_out"] = oout
    res1 = g1.to_bf16(oout + fc_out)
    t["residual1"] = res1

    pnorm = g1.to_bf16(g1.gemma_rmsnorm(res1, W.post_attention_layernorm, eps, H))
    t["post_attention_norm_out"] = pnorm

    g = g1.to_bf16(g1.linear_f32(pnorm, W.gate_proj))
    u = g1.to_bf16(g1.linear_f32(pnorm, W.up_proj))
    silu_g = g / (1.0 + torch.exp(-g))
    sw = g1.to_bf16(u * silu_g)
    t["mlp_mul"] = sw

    d = g1.to_bf16(g1.linear_f32(sw, W.down_proj))
    t["mlp_down_proj"] = d
    fres = g1.to_bf16(d + res1)
    t["final_residual"] = fres

    fn = g1.to_bf16(g1.gemma_rmsnorm(fres, W.final_norm, eps, H))
    t["final_norm"] = fn
    logits = g1.linear_f32(fn, W.embed)
    t["logits"] = logits

    top1 = int(torch.argmax(logits).item())
    t["argmax"] = torch.tensor([top1], dtype=torch.int32)
    return t, kr, vproj, logits, top1


CHECKPOINT_STEPS = [0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 127, 255]


def write_case(g1, W, out_dir, name, tokens, hidden, positions, offset, checkpoints):
    steps = len(tokens)
    k_hist = []
    v_hist = []
    per_step = {k: [] for k in [
        "k_rope_current", "v_current", "attention_core_out", "o_proj_out",
        "final_norm", "argmax", "top10_ids", "top10_logits", "logits_l2"]}
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    step_meta = []
    for t in range(steps):
        tens, kr, v, logits, top1 = mtp_step(
            g1, W, tokens[t], positions[t], hidden[t], k_hist, v_hist)
        k_hist.append(kr)
        v_hist.append(v)
        per_step["k_rope_current"].append(kr)
        per_step["v_current"].append(v)
        per_step["attention_core_out"].append(tens["attention_core_out"])
        per_step["o_proj_out"].append(tens["o_proj_out"])
        per_step["final_norm"].append(tens["final_norm"])
        per_step["argmax"].append(tens["argmax"])
        topk = torch.topk(logits, 10, largest=True, sorted=True)
        per_step["top10_ids"].append(topk.indices.to(torch.int32))
        per_step["top10_logits"].append(topk.values.to(torch.float32))
        per_step["logits_l2"].append(torch.tensor(
            [float(torch.sqrt((logits.double() ** 2).sum()))], dtype=torch.float32))
        step_meta.append({
            "step": t,
            "input_token_id": int(tokens[t]),
            "mtp_position": int(positions[t]),
            "absolute_position": int(positions[t]),
            "kv_length_before": t,
            "kv_length_after": t + 1,
            "hidden_source_index": t,
            "committed_token_index": t,
            "argmax": top1,
            "top10_token_ids": [int(x) for x in topk.indices.tolist()],
            "logits_sha256": sha256_tensor(logits.to(torch.float32)),
            "k_check": sha256_tensor(kr),
            "v_check": sha256_tensor(v),
        })
        if t in checkpoints:
            ck = {
                "K_history": torch.stack(k_hist).to(torch.bfloat16),
                "V_history": torch.stack(v_hist).to(torch.bfloat16),
                "logits": logits.to(torch.float32),
                "attention_core_out": tens["attention_core_out"].to(torch.float32),
            }
            save_file(ck, os.path.join(ckpt_dir, f"step_{t}.safetensors"))

    os.makedirs(out_dir, exist_ok=True)
    seq = {
        "target_hidden": hidden.to(torch.bfloat16),
        "k_rope_current": torch.stack(per_step["k_rope_current"]).to(torch.bfloat16),
        "v_current": torch.stack(per_step["v_current"]).to(torch.bfloat16),
        "attention_core_out": torch.stack(per_step["attention_core_out"]).to(torch.float32),
        "o_proj_out": torch.stack(per_step["o_proj_out"]).to(torch.bfloat16),
        "final_norm": torch.stack(per_step["final_norm"]).to(torch.bfloat16),
        "argmax": torch.stack(per_step["argmax"]).to(torch.int32),
        "top10_ids": torch.stack(per_step["top10_ids"]).to(torch.int32),
        "top10_logits": torch.stack(per_step["top10_logits"]).to(torch.float32),
        "logits_l2": torch.stack(per_step["logits_l2"]).to(torch.float32),
    }
    save_file(seq, os.path.join(out_dir, "sequence.safetensors"))
    return step_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts/mtp_gate2")
    ap.add_argument("--steps", type=int, default=257)
    ap.add_argument("--position-offset", type=int, default=0)
    ap.add_argument("--token-seed", type=int, default=20260919)
    ap.add_argument("--hidden-seed", type=int, default=4242)
    ap.add_argument("--offset-steps", type=int, default=8)
    ap.add_argument("--gate1-exporter",
                    default=os.path.join(os.path.dirname(__file__), "export_qwen35_mtp_gate1.py"))
    ap.add_argument("--real-dir", default="",
                    help="PHASESHIFT_MTP_GATE2_REAL_DUMP が書いた real_hidden.bin/real_tokens.bin/"
                         "real_meta.json のあるディレクトリ。指定すると real/ case を作る。")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))

    g1 = load_gate1(args.gate1_exporter)
    cfg, cfg_sha = g1.load_config(args.model_dir)
    tc = cfg.get("text_config", cfg)
    H = int(tc["hidden_size"])
    index = g1.build_tensor_index(args.model_dir)
    W = g1.MtpWeights(index, args.model_dir, cfg, H)
    W.self_check(H)

    tokens = build_tokens(W, args.steps, args.token_seed)
    hidden = build_hidden(W, args.steps, args.hidden_seed)
    positions = [args.position_offset + t for t in range(args.steps)]

    root = os.path.join(args.out_dir, "synthetic")
    step_meta = write_case(g1, W, root, "synthetic", tokens, hidden, positions,
                           args.position_offset, set(CHECKPOINT_STEPS) | {args.steps - 1})

    meta = {
        "fixture_version": FIXTURE_VERSION,
        "case": "synthetic",
        "model_path": os.path.abspath(args.model_dir),
        "config_sha256": cfg_sha,
        "H": H,
        "steps": args.steps,
        "position_offset": args.position_offset,
        "position_offsets": [0, 17, 127],
        "token_seed": args.token_seed,
        "hidden_seed": args.hidden_seed,
        "dtype": "bf16",
        "attention_mode": "full_history",
        "checkpoint_steps": sorted(set(CHECKPOINT_STEPS) | {args.steps - 1}),
        "step_meta": step_meta,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] synthetic steps={args.steps} offset={args.position_offset} -> {root}")

    for off in (17, 127):
        sub = os.path.join(args.out_dir, f"synthetic_offset_{off}")
        n = min(args.offset_steps, args.steps)
        toks = tokens[:n]
        hid = hidden[:n]
        pos = [off + t for t in range(n)]
        sm = write_case(g1, W, sub, f"synthetic_offset_{off}", toks, hid, pos, off,
                        set(range(n)))
        m = dict(meta)
        m.update({"case": f"synthetic_offset_{off}", "steps": n, "position_offset": off,
                  "checkpoint_steps": list(range(n)), "step_meta": sm,
                  "parent_hidden_seed": args.hidden_seed, "parent_token_seed": args.token_seed})
        with open(os.path.join(sub, "metadata.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"[export] offset={off} steps={n} -> {sub}")

    if args.real_dir:
        with open(os.path.join(args.real_dir, "real_meta.json")) as f:
            rmeta = json.load(f)
        n = int(rmeta["N"])
        real_h = int(rmeta["H"])
        if real_h != H:
            print(f"FAIL: real hidden H={real_h} != checkpoint H={H}")
            return 1
        hidden = torch.from_file(
            os.path.join(args.real_dir, "real_hidden.bin"),
            size=n * H, dtype=torch.bfloat16).reshape(n, H)
        tokens = np.fromfile(
            os.path.join(args.real_dir, "real_tokens.bin"), dtype=np.int32).tolist()
        positions = list(range(n))
        sub = os.path.join(args.out_dir, "real")
        os.makedirs(sub, exist_ok=True)
        sm = write_case(g1, W, sub, "real", tokens, hidden, positions, 0,
                        set(CHECKPOINT_STEPS) | {n - 1})
        m = dict(meta)
        m.update({
            "case": "real",
            "steps": n,
            "position_offset": 0,
            "checkpoint_steps": sorted(set(CHECKPOINT_STEPS) | {n - 1}),
            "step_meta": sm,
            "source": "phaseshift target token_hidden",
        })
        with open(os.path.join(sub, "metadata.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"[export] real steps={n} -> {sub}")

    print("MTP_GATE2_EXPORT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
