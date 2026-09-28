#!/usr/bin/env python3
"""Qwen3.8-27B DFlash2 Gate 5.1 numeric contract trace.

- 公式実装の forward を BF16 / F32 で実行し、各 stage を PhaseNonShift と同じ
  [rows, features] layout で f32 dump する。BF16 run では tensor dtype と統計も記録する。
- PhaseNonShift の op 順序・buffer 境界を PyTorch primitive で再現した
  explicit BF16 replay を作り、PhaseNonShift の dump と 3-way 比較する。

使い方:
  python3 tools/reference/trace_qwen38_dflash2_gate51.py --mode all
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))


def load_exporter():
    path = os.path.join(HERE, "export_qwen38_dflash2_gate5.py")
    spec = importlib.util.spec_from_file_location("g5exporter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g5exporter"] = mod
    spec.loader.exec_module(mod)
    return mod


def tensor_stats(t):
    if t is None:
        return None
    tf = t.detach().float().reshape(-1)
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "min": float(tf.min()),
        "max": float(tf.max()),
        "absmax": float(tf.abs().max()),
        "mean": float(tf.mean()),
        "rms": float(tf.pow(2).mean().sqrt()),
        "finite": int(torch.isfinite(tf).sum()),
        "numel": int(tf.numel()),
    }


REPLAY_F32 = False


def bf16(t):
    if REPLAY_F32:
        return t.float() if t.dtype != torch.float32 else t
    return t.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# PhaseNonShift の primitive を PyTorch で再現（丸め位置も含めて）
# ---------------------------------------------------------------------------

def ps_rmsnorm(x_bf16, weight_bf16, eps, group_size=None):
    """launch_dflash2_rmsnorm_direct_bf16: f32 variance, norm を bf16 へ丸めてから weight 乗算."""
    shape = x_bf16.shape
    if group_size is None:
        group_size = shape[-1]
    flat = x_bf16.reshape(-1, group_size).float()
    variance = flat.pow(2).mean(-1, keepdim=True)
    norm = flat * torch.rsqrt(variance + eps)
    norm_rounded = bf16(norm)
    w = weight_bf16.reshape(-1)
    if group_size == shape[-1]:
        wb = w
    else:
        wb = w  # per-head weight は group 内で同一 index
    out = bf16(norm_rounded * wb)
    return out.reshape(shape)


def ps_gemm(x_bf16, weight_bf16):
    return bf16(x_bf16.float() @ weight_bf16.float().t())


def ps_conv(hidden_bf16, dyn_phase_bf16, base_phase_bf16, group_size):
    """grouped_dynamic_conv_kernel と同じ丸め列.

    hidden: [rows, hidden]
    dyn_phase: [rows, kernel_size * groups]  (offset-major, group-minor)
    base_phase: [kernel_size, hidden]
    """
    rows, hidden_size = hidden_bf16.shape
    groups = hidden_size // group_size
    h = hidden_bf16.float()
    b = base_phase_bf16.float()
    d = dyn_phase_bf16.float().view(rows, 2, groups).repeat_interleave(group_size, dim=2)
    d0 = d[:, 0]
    d1 = d[:, 1]
    v0 = h
    t0 = bf16(b[0].unsqueeze(0) * v0).float()
    o0 = bf16(t0 + d0 * v0).float()
    acc = o0
    v1 = torch.zeros_like(h)
    v1[1:] = h[:-1]
    t1 = bf16(b[1].unsqueeze(0) * v1).float()
    o1 = bf16(acc + t1).float()
    acc = bf16(o1 + d1 * v1).float()
    return bf16(acc)


def ps_rope(x_bf16, features, head_dim, position_start, theta):
    """dflash2_rope_kernel: inv_freq=powf(theta,-2i/d), angle=pos*inv_freq, cos/sin と積を bf16 へ."""
    rows = x_bf16.shape[0]
    heads = features // head_dim
    half = head_dim // 2
    x = x_bf16.float().reshape(rows, heads, head_dim)
    exponent = (-2.0 * torch.arange(half, dtype=torch.float32) / float(head_dim))
    inv_freq = torch.tensor(theta, dtype=torch.float32) ** exponent
    positions = torch.arange(position_start, position_start + rows, dtype=torch.float32)
    angle = positions[:, None] * inv_freq[None, :]
    c = bf16(angle.cos()).float()
    s = bf16(angle.sin()).float()
    x1 = x[..., :half]
    x2 = x[..., half:]
    p1 = bf16(x1 * c[:, None, :]).float()
    p2 = bf16(x2 * s[:, None, :]).float()
    p3 = bf16(x2 * c[:, None, :]).float()
    p4 = bf16(x1 * s[:, None, :]).float()
    out = torch.empty_like(x)
    out[..., :half] = bf16(p1 - p2)
    out[..., half:] = bf16(p3 + p4)
    return out.reshape(rows, features)


def ps_attention(q_bf16, k_ctx_bf16, v_ctx_bf16, k_noise_bf16, v_noise_bf16,
                 context_position_start, block_position_start, window, scale,
                 q_heads, kv_heads, head_dim):
    """f32 score / f32 softmax / f32 value accumulate / bf16 output."""
    rows = q_bf16.shape[0]
    ctx = k_ctx_bf16.shape[0]
    group = q_heads // kv_heads
    q = q_bf16.float().reshape(rows, q_heads, head_dim)
    kc = k_ctx_bf16.float().reshape(ctx, kv_heads, head_dim)
    kn = k_noise_bf16.float().reshape(rows, kv_heads, head_dim)
    vc = v_ctx_bf16.float().reshape(ctx, kv_heads, head_dim)
    vn = v_noise_bf16.float().reshape(rows, kv_heads, head_dim)
    k = torch.cat([kc, kn], dim=0)
    v = torch.cat([vc, vn], dim=0)
    keys = ctx + rows
    kq = k.repeat_interleave(group, dim=1)
    vq = v.repeat_interleave(group, dim=1)
    q_pos = block_position_start + torch.arange(rows)
    k_pos = torch.cat([context_position_start + torch.arange(ctx),
                       block_position_start + torch.arange(rows)])
    visible = (k_pos[None, :] - q_pos[:, None]).abs() < window
    scores = torch.einsum("rhd,jhd->rhj", q, kq) * scale
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    m = scores.amax(dim=2, keepdim=True)
    e = torch.exp(scores - m)
    ssum = e.sum(dim=2, keepdim=True)
    acc = torch.einsum("rhj,jhd->rhd", e, vq)
    out = bf16(acc * (1.0 / ssum))
    return out.reshape(rows, q_heads * head_dim)


def ps_swiglu(gate_bf16, up_bf16):
    g = gate_bf16.float()
    silu = bf16(g / (1.0 + torch.exp(-g))).float()
    return bf16(silu * up_bf16.float())


# ---------------------------------------------------------------------------
# 公式 forward trace
# ---------------------------------------------------------------------------

class OfficialModel:
    def __init__(self, official, cfg, weights, dtype):
        self.official = official
        model = official.DFlash2DraftModel(cfg)
        model.load_state_dict(weights, strict=False)
        model.eval()
        self.model = model.to(dtype)
        if dtype != torch.float32:
            self.model.rotary_emb = official.Qwen3RotaryEmbedding(cfg)
        self.dtype = dtype
        self.cfg = cfg



def flat_heads(t, rows, heads):
    """[b, *, *, d] を [rows, heads*head_dim] へ. heads 軸を自動判定する。"""
    x = t[0] if t.dim() == 4 else t
    if x.shape[0] == heads:
        x = x.transpose(0, 1)
    elif x.shape[1] != heads:
        raise RuntimeError(f"cannot locate head axis in {tuple(t.shape)}")
    return x.reshape(rows, -1)

def run_official_trace(capture, case, out_dir, record_stats):
    """公式 forward を 1 回だけ実行し、全 layer の stage を一度に収集する。"""
    model = capture.model
    cfg = capture.cfg
    dtype = capture.dtype
    official = capture.official
    os.makedirs(out_dir, exist_ok=True)
    from transformers.models.qwen3 import modeling_qwen3 as tq

    rows = case["block_rows"]
    ctx = case["context_rows"]
    noise = case["noise_embedding"].to(dtype).unsqueeze(0)
    concat = case["target_concat"].to(dtype).unsqueeze(0)
    position_ids = torch.arange(case["context_start"], case["context_start"] + ctx + rows,
                                dtype=torch.long)[None, :]

    nlayers = int(cfg.num_hidden_layers)
    rec = [{} for _ in range(nlayers)]
    conv_rec = [{} for _ in range(nlayers)]
    rope_rec = [{} for _ in range(nlayers)]
    sdpa_rec = [{} for _ in range(nlayers)]
    kcalls = [[] for _ in range(nlayers)]
    vcalls = [[] for _ in range(nlayers)]
    knorm = [[] for _ in range(nlayers)]
    handles = []
    final_holder = {}

    for li in range(nlayers):
        layer = model.layers[li]
        r = rec[li]

        def hook(mod, key, store):
            def f(m, i, o):
                store[key] = o.detach()
            handles.append(mod.register_forward_hook(f))

        def pre_hook(mod, key, store):
            def f(m, args, kwargs):
                v = args[0] if len(args) > 0 else (kwargs.get("hidden_states") if kwargs else None)
                store[key] = v.detach()
            handles.append(mod.register_forward_pre_hook(f, with_kwargs=True))

        hook(layer, "layer_output", r)
        pre_hook(layer, "layer_input", r)
        hook(layer.input_layernorm, "input_norm", r)
        hook(layer.self_attn.q_proj, "q_raw", r)
        hook(layer.self_attn.q_norm, "q_norm", r)
        hook(layer.self_attn.o_proj, "attention_o", r)
        pre_hook(layer.post_attention_layernorm, "attention_residual", r)
        hook(layer.post_attention_layernorm, "post_attention_norm", r)
        hook(layer.mlp.gate_proj, "mlp_gate", r)
        hook(layer.mlp.up_proj, "mlp_up", r)
        pre_hook(layer.mlp.down_proj, "mlp_swiglu", r)
        hook(layer.mlp.down_proj, "mlp_down", r)
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            lambda m, i, o, s=kcalls[li]: s.append(o.detach())))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            lambda m, i, o, s=vcalls[li]: s.append(o.detach())))
        handles.append(layer.self_attn.k_norm.register_forward_hook(
            lambda m, i, o, s=knorm[li]: s.append(o.detach())))

        cr = conv_rec[li]
        for cname, conv in (("attn", layer.attention_conv), ("mlp", layer.mlp_conv)):
            orig_prepare, orig_finish = conv.prepare, conv.finish

            def make_prepare(orig, store, key):
                def f(h, _orig=orig, _store=store, _key=key):
                    out, dyn = _orig(h)
                    _store[_key + "_prepared"] = out.detach()
                    _store[_key + "_dynamic"] = dyn.detach()
                    return out, dyn
                return f

            def make_finish(orig, store, key):
                def f(h, d, _orig=orig, _store=store, _key=key):
                    out = _orig(h, d)
                    _store[_key + "_finished"] = out.detach()
                    return out
                return f

            conv.prepare = make_prepare(orig_prepare, cr, cname)
            conv.finish = make_finish(orig_finish, cr, cname)

    # rope と sdpa は layer ごとに 1 回ずつ呼ばれる。呼び出し順 = layer 順。
    orig_rope = official.apply_rotary_pos_emb
    rope_index = [0]

    def rope_hook(q, k, cos, sin, *a, **kw):
        qo, ko = orig_rope(q, k, cos, sin, *a, **kw)
        idx = rope_index[0]
        rope_index[0] += 1
        if idx < nlayers:
            rope_rec[idx]["q_out"] = qo.detach()
            rope_rec[idx]["k_out"] = ko.detach()
            rope_rec[idx]["cos"] = cos.detach()
            rope_rec[idx]["sin"] = sin.detach()
        return qo, ko
    official.apply_rotary_pos_emb = rope_hook

    orig_sdpa = tq.ALL_ATTENTION_FUNCTIONS["sdpa"]
    sdpa_index = [0]

    def sdpa_hook(module, query, key, value, mask, **kw):
        out = orig_sdpa(module, query, key, value, mask, **kw)
        idx = sdpa_index[0]
        sdpa_index[0] += 1
        if idx < nlayers:
            sdpa_rec[idx]["out"] = (out[0] if isinstance(out, tuple) else out).detach()
            sdpa_rec[idx]["scale"] = float(kw.get("scaling", 0.0))
        return out
    tq.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_hook

    try:
        with torch.inference_mode():
            result = model(position_ids=position_ids, attention_mask=None,
                           noise_embedding=noise, target_hidden=concat,
                           past_key_values=None, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        official.apply_rotary_pos_emb = orig_rope
        tq.ALL_ATTENTION_FUNCTIONS["sdpa"] = orig_sdpa

    stat_records = {}
    dumped = {}
    for li in range(nlayers):
        r = rec[li]
        cr = conv_rec[li]
        rr = rope_rec[li]
        sr = sdpa_rec[li]
        k_ctx_raw, k_noise_raw = kcalls[li][0], kcalls[li][1]
        v_ctx, v_noise = vcalls[li][0], vcalls[li][1]
        k_norm_out = knorm[li][0]
        pref = f"layer{li}_"
        d = {}

        def put(name, t):
            if t is None:
                return
            d[pref + name] = t

        put("layer_input", r["layer_input"][0].reshape(rows, -1))
        put("input_norm", r["input_norm"][0].reshape(rows, -1))
        put("attention_prepared", cr["attn_prepared"][0].reshape(rows, -1))
        put("q_raw", r["q_raw"][0].reshape(rows, -1))
        put("q_norm", r["q_norm"][0].reshape(rows, -1))
        put("q_rope", flat_heads(rr["q_out"], rows, int(cfg.num_attention_heads)))
        put("k_ctx_raw", k_ctx_raw[0].reshape(ctx, -1))
        put("k_ctx_norm", k_norm_out[0, :ctx].reshape(ctx, -1))
        put("k_ctx_rope", flat_heads(rr["k_out"][:, :, :ctx, :], ctx,
                                     int(cfg.num_key_value_heads)))
        put("v_ctx", v_ctx[0].reshape(ctx, -1))
        put("k_noise_raw", k_noise_raw[0].reshape(rows, -1))
        put("k_noise_norm", k_norm_out[0, ctx:].reshape(rows, -1))
        put("k_noise_rope", flat_heads(rr["k_out"][:, :, ctx:, :], rows,
                                       int(cfg.num_key_value_heads)))
        put("v_noise", v_noise[0].reshape(rows, -1))
        put("attention_context", flat_heads(sr["out"], rows, int(cfg.num_attention_heads)))
        put("attention_o", r["attention_o"][0].reshape(rows, -1))
        put("attention_finished", cr["attn_finished"][0].reshape(rows, -1))
        put("attention_residual", r["attention_residual"][0].reshape(rows, -1))
        put("post_attention_norm", r["post_attention_norm"][0].reshape(rows, -1))
        put("mlp_prepared", cr["mlp_prepared"][0].reshape(rows, -1))
        put("mlp_gate", r["mlp_gate"][0].reshape(rows, -1))
        put("mlp_up", r["mlp_up"][0].reshape(rows, -1))
        put("mlp_swiglu", r["mlp_swiglu"][0].reshape(rows, -1))
        put("mlp_down", r["mlp_down"][0].reshape(rows, -1))
        put("mlp_finished", cr["mlp_finished"][0].reshape(rows, -1))
        put("layer_output", r["layer_output"][0].reshape(rows, -1))

        if record_stats:
            stat_records[f"layer{li}"] = {
                name: tensor_stats(d.get(pref + name)) for name in
                ["layer_input", "input_norm", "attention_prepared", "q_raw", "q_norm", "q_rope",
                 "k_ctx_raw", "k_ctx_norm", "k_ctx_rope", "v_ctx",
                 "k_noise_raw", "k_noise_norm", "k_noise_rope", "v_noise",
                 "attention_context", "attention_o", "attention_finished", "attention_residual",
                 "post_attention_norm", "mlp_prepared", "mlp_gate", "mlp_up", "mlp_swiglu",
                 "mlp_down", "mlp_finished", "layer_output"]
                if d.get(pref + name) is not None
            }
            stat_records[f"layer{li}"]["cos"] = tensor_stats(rr.get("cos"))
            stat_records[f"layer{li}"]["sin"] = tensor_stats(rr.get("sin"))
            stat_records[f"layer{li}"]["sdpa_scale"] = sr.get("scale")

        dumped.update(d)

    dumped["final_hidden"] = result.detach()[0].reshape(rows, -1)
    for name, t in dumped.items():
        t.detach().float().reshape(-1).numpy().astype(np.float32).tofile(
            os.path.join(out_dir, name + ".f32"))
    return stat_records


# ---------------------------------------------------------------------------
# explicit BF16 replay (PhaseNonShift の op 順序)
# ---------------------------------------------------------------------------

def run_replay(cfg, weights, case, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    W = {k: (v.float() if REPLAY_F32 else v.to(torch.bfloat16)) for k, v in weights.items()}
    rows = case["block_rows"]
    ctx = case["context_rows"]
    hidden = int(cfg.hidden_size)
    group_size = int(dict(cfg.dflash_config)["conv_group_size"])
    eps = float(cfg.rms_norm_eps)
    theta = float(dict(cfg.rope_parameters).get("rope_theta", 1e7))
    head_dim = int(cfg.head_dim)
    q_heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    q_features = q_heads * head_dim
    kv_features = kv_heads * head_dim
    window = int(cfg.sliding_window)
    scale = 1.0 / float(head_dim) ** 0.5
    ctx_start = case["context_start"]
    blk_start = case["block_start"]

    src_dtype = torch.float32 if REPLAY_F32 else torch.bfloat16
    target_feature = case["target_feature"].to(src_dtype)
    hidden_states = case["noise_embedding"].to(src_dtype)
    d = {}

    def put(name, t):
        d[name] = t

    for li in range(int(cfg.num_hidden_layers)):
        p = f"layer{li}_"
        s = f"layers.{li}."
        put(p + "layer_input", hidden_states)
        x = hidden_states
        input_norm = ps_rmsnorm(x, W[s + "input_layernorm.weight"], eps)
        put(p + "input_norm", input_norm)
        attn_dyn = ps_gemm(input_norm, W[s + "attention_conv.kernel_projection.weight"])
        attn_dyn = attn_dyn.view(rows, 2, 2, group_size and hidden // group_size)
        base_a = W[s + "attention_conv.base_kernel"]
        attn_prepared = ps_conv(input_norm, attn_dyn[:, 0].reshape(rows, -1), base_a[0], group_size)
        put(p + "attention_prepared", attn_prepared)
        q_raw = ps_gemm(attn_prepared, W[s + "self_attn.q_proj.weight"])
        k_ctx_raw = ps_gemm(target_feature, W[s + "self_attn.k_proj.weight"])
        v_ctx = ps_gemm(target_feature, W[s + "self_attn.v_proj.weight"])
        k_noise_raw = ps_gemm(attn_prepared, W[s + "self_attn.k_proj.weight"])
        v_noise = ps_gemm(attn_prepared, W[s + "self_attn.v_proj.weight"])
        put(p + "q_raw", q_raw)
        put(p + "k_ctx_raw", k_ctx_raw)
        put(p + "v_ctx", v_ctx)
        put(p + "k_noise_raw", k_noise_raw)
        put(p + "v_noise", v_noise)
        Wq = W[s + "self_attn.q_norm.weight"]
        Wk = W[s + "self_attn.k_norm.weight"]
        q_norm = ps_rmsnorm(q_raw, Wq, eps, group_size=head_dim)
        k_ctx_norm = ps_rmsnorm(k_ctx_raw, Wk, eps, group_size=head_dim)
        k_noise_norm = ps_rmsnorm(k_noise_raw, Wk, eps, group_size=head_dim)
        put(p + "q_norm", q_norm)
        put(p + "k_ctx_norm", k_ctx_norm)
        put(p + "k_noise_norm", k_noise_norm)
        q_rope = ps_rope(q_norm, q_features, head_dim, blk_start, theta)
        k_ctx_rope = ps_rope(k_ctx_norm, kv_features, head_dim, ctx_start, theta)
        k_noise_rope = ps_rope(k_noise_norm, kv_features, head_dim, blk_start, theta)
        put(p + "q_rope", q_rope)
        put(p + "k_ctx_rope", k_ctx_rope)
        put(p + "k_noise_rope", k_noise_rope)
        attn_ctx = ps_attention(q_rope, k_ctx_rope, v_ctx, k_noise_rope, v_noise, ctx_start,
                                blk_start, window, scale, q_heads, kv_heads, head_dim)
        put(p + "attention_context", attn_ctx)
        attention_o = ps_gemm(attn_ctx, W[s + "self_attn.o_proj.weight"])
        put(p + "attention_o", attention_o)
        attn_finished = ps_conv(attention_o, attn_dyn[:, 1].reshape(rows, -1), base_a[1], group_size)
        put(p + "attention_finished", attn_finished)
        attention_residual = bf16(x.float() + attn_finished.float())
        put(p + "attention_residual", attention_residual)
        post_norm = ps_rmsnorm(attention_residual, W[s + "post_attention_layernorm.weight"], eps)
        put(p + "post_attention_norm", post_norm)
        mlp_dyn = ps_gemm(post_norm, W[s + "mlp_conv.kernel_projection.weight"])
        mlp_dyn = mlp_dyn.view(rows, 2, 2, group_size and hidden // group_size)
        base_m = W[s + "mlp_conv.base_kernel"]
        mlp_prepared = ps_conv(post_norm, mlp_dyn[:, 0].reshape(rows, -1), base_m[0], group_size)
        put(p + "mlp_prepared", mlp_prepared)
        gate = ps_gemm(mlp_prepared, W[s + "mlp.gate_proj.weight"])
        up = ps_gemm(mlp_prepared, W[s + "mlp.up_proj.weight"])
        put(p + "mlp_gate", gate)
        put(p + "mlp_up", up)
        swiglu = ps_swiglu(gate, up)
        put(p + "mlp_swiglu", swiglu)
        down = ps_gemm(swiglu, W[s + "mlp.down_proj.weight"])
        put(p + "mlp_down", down)
        mlp_finished = ps_conv(down, mlp_dyn[:, 1].reshape(rows, -1), base_m[1], group_size)
        put(p + "mlp_finished", mlp_finished)
        layer_output = bf16(attention_residual.float() + mlp_finished.float())
        put(p + "layer_output", layer_output)
        hidden_states = layer_output

    final = ps_rmsnorm(hidden_states, W["norm.weight"], eps)
    put("final_hidden", final)
    for name, t in d.items():
        t.detach().float().reshape(-1).numpy().astype(np.float32).tofile(
            os.path.join(out_dir, name + ".f32"))
    print(f"[trace] {label} replay done -> {out_dir}")


# ---------------------------------------------------------------------------
# 比較
# ---------------------------------------------------------------------------

SPECS = [
    ("input_norm", "block", "hidden"),
    ("attention_prepared", "block", "hidden"),
    ("q_raw", "block", "q"),
    ("q_norm", "block", "q"),
    ("q_rope", "block", "q"),
    ("k_ctx_raw", "ctx", "kv"),
    ("k_ctx_norm", "ctx", "kv"),
    ("k_ctx_rope", "ctx", "kv"),
    ("v_ctx", "ctx", "kv"),
    ("k_noise_raw", "block", "kv"),
    ("k_noise_norm", "block", "kv"),
    ("k_noise_rope", "block", "kv"),
    ("v_noise", "block", "kv"),
    ("attention_context", "block", "q"),
    ("attention_o", "block", "hidden"),
    ("attention_finished", "block", "hidden"),
    ("attention_residual", "block", "hidden"),
    ("post_attention_norm", "block", "hidden"),
    ("mlp_prepared", "block", "hidden"),
    ("mlp_gate", "block", "inter"),
    ("mlp_up", "block", "inter"),
    ("mlp_swiglu", "block", "inter"),
    ("mlp_down", "block", "hidden"),
    ("mlp_finished", "block", "hidden"),
    ("layer_output", "block", "hidden"),
]


def tensor_from_f32(path, shape):
    if not os.path.isfile(path):
        return None
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).copy()).view(*shape)


def tensor_from_bf16_bin(path, shape):
    if not os.path.isfile(path):
        return None
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(f32.copy()).view(*shape)


def rel_l2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = float(torch.linalg.vector_norm(a))
    return float(torch.linalg.vector_norm(a - b) / n) if n > 0 else 0.0


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def compare_all(out_root, ps_dir, case, cfg):
    rows, ctx = case["block_rows"], case["context_rows"]
    hidden = int(cfg.hidden_size)
    q_f = int(cfg.num_attention_heads) * int(cfg.head_dim)
    kv_f = int(cfg.num_key_value_heads) * int(cfg.head_dim)
    inter = int(cfg.intermediate_size)
    dims = {"hidden": hidden, "q": q_f, "kv": kv_f, "inter": inter}

    def dim_rows(kind):
        return ctx if kind == "ctx" else rows

    modes = ["official_bf16", "official_f32", "replay_bf16"]
    report = {}
    for li in range(int(cfg.num_hidden_layers)):
        report[f"layer{li}"] = {}
        for name, rk, fk in SPECS:
            r, f = dim_rows(rk), dims[fk]
            ps = tensor_from_bf16_bin(os.path.join(ps_dir, f"layer{li}_{name}.bin"), (r, f))
            if ps is None:
                continue
            entry = {}
            for mode in modes:
                ref = tensor_from_f32(os.path.join(out_root, mode, f"layer{li}_{name}.f32"), (r, f))
                if ref is None:
                    continue
                entry[mode] = {"rel_l2": rel_l2(ref, ps), "cosine": cosine(ref, ps),
                               "absmax": float(ref.abs().max())}
            report[f"layer{li}"][name] = entry
    report["final"] = {}
    ps = tensor_from_bf16_bin(os.path.join(ps_dir, "final_hidden.bin"), (rows, hidden))
    for mode in modes:
        ref = tensor_from_f32(os.path.join(out_root, mode, "final_hidden.f32"), (rows, hidden))
        if ref is None:
            continue
        report["final"][mode] = {"rel_l2": rel_l2(ref, ps), "cosine": cosine(ref, ps),
                                 "absmax": float(ref.abs().max())}
    json.dump(report, open(os.path.join(out_root, "comparison.json"), "w"), indent=2)

    print("=== PhaseNonShift vs reference (rel_l2) ===")
    print(f"{'stage':22s} {'off_bf16':>11s} {'off_f32':>11s} {'replay_bf16':>12s}  {'mult':>6s}")
    for li in range(int(cfg.num_hidden_layers)):
        print(f"--- layer {li}")
        prev = None
        for name, rk, fk in SPECS:
            r, f = dim_rows(rk), dims[fk]
            ps = tensor_from_bf16_bin(os.path.join(ps_dir, f"layer{li}_{name}.bin"), (r, f))
            if ps is None:
                continue
            vals = []
            for mode in modes:
                ref = tensor_from_f32(os.path.join(out_root, mode, f"layer{li}_{name}.f32"), (r, f))
                vals.append(f"{rel_l2(ref, ps):.3e}" if ref is not None else "-")
            cur = report[f"layer{li}"][name].get("official_bf16", {}).get("rel_l2")
            mult = f"{cur/prev:.2f}" if (cur is not None and prev not in (None, 0)) else "-"
            if cur is not None:
                prev = cur
            print(f"{name:22s} {vals[0]:>11s} {vals[1]:>11s} {vals[2]:>12s}  {mult:>6s}")
    print("--- final")
    ps = tensor_from_bf16_bin(os.path.join(ps_dir, "final_hidden.bin"), (rows, hidden))
    for name in ["final_hidden"]:
        vals = []
        for mode in modes:
            ref = tensor_from_f32(os.path.join(out_root, mode, "final_hidden.f32"), (rows, hidden))
            vals.append(f"{rel_l2(ref, ps):.3e}" if ref is not None and ps is not None else "-")
        print(f"{name:22s} {vals[0]:>11s} {vals[1]:>11s} {vals[2]:>12s}")

    print("\n=== replay_bf16 vs official_bf16 (harness fidelity) ===")
    for li in range(int(cfg.num_hidden_layers)):
        for name, rk, fk in SPECS:
            r, f = dim_rows(rk), dims[fk]
            a = tensor_from_f32(os.path.join(out_root, "replay_bf16", f"layer{li}_{name}.f32"), (r, f))
            b = tensor_from_f32(os.path.join(out_root, "official_bf16", f"layer{li}_{name}.f32"), (r, f))
            c = tensor_from_f32(os.path.join(out_root, "official_f32", f"layer{li}_{name}.f32"), (r, f))
            if a is None or b is None:
                continue
            print(f"layer{li} {name:22s} replay_vs_off_bf16={rel_l2(b,a):.3e}  "
                  f"off_bf16_vs_off_f32={rel_l2(c,b):.3e}  "
                  f"replay_vs_off_f32={rel_l2(c,a):.3e}")
    a = tensor_from_f32(os.path.join(out_root, "replay_bf16", "final_hidden.f32"), (rows, hidden))
    b = tensor_from_f32(os.path.join(out_root, "official_bf16", "final_hidden.f32"), (rows, hidden))
    c = tensor_from_f32(os.path.join(out_root, "official_f32", "final_hidden.f32"), (rows, hidden))
    if a is not None and b is not None and c is not None:
        print(f"final  replay_vs_off_bf16={rel_l2(b,a):.3e}  off_bf16_vs_off_f32={rel_l2(c,b):.3e}  "
              f"replay_vs_off_f32={rel_l2(c,a):.3e}")



def logit_sensitivity(out_root, ps_dir, case, cfg, target_dir, fixture_dir, case_name):
    """同じ target lm_head を official / PhaseNonShift の final_hidden に適用し top-k を比較する."""
    from safetensors import safe_open
    idx_path = os.path.join(target_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path))["weight_map"]
    head_key = "lm_head.weight"
    with safe_open(os.path.join(target_dir, idx[head_key]), framework="pt") as f:
        head = f.get_tensor(head_key).float()
    draft = dict(cfg.dflash_config)
    mult = float(draft.get("output_multiplier", 1.0))
    softcap = draft.get("final_logit_softcapping")
    top_k = int(draft.get("selector_top_k", 16))
    rows, hidden = case["block_rows"], head.shape[1]
    print(f"lm_head {tuple(head.shape)} dtype=bf16 output_multiplier={mult} softcap={softcap} top_k={top_k}")
    ps = tensor_from_bf16_bin(os.path.join(ps_dir, "final_hidden.bin"), (rows, hidden))
    if ps is None:
        print("PS final_hidden missing")
        return
    fixture_ref = None
    fix_path = os.path.join(fixture_dir, case_name, "fixture.safetensors")
    if os.path.isfile(fix_path):
        from safetensors.torch import load_file as _load
        fixture_ref = _load(fix_path)["final_hidden"].to(torch.float32).reshape(rows, hidden)
    report = {}
    for mode in ["official_bf16", "official_f32", "replay_bf16"]:
        ref = tensor_from_f32(os.path.join(out_root, mode, "final_hidden.f32"), (rows, hidden))
        if ref is None and mode == "official_bf16":
            ref = fixture_ref
        if ref is None:
            continue
        lg_ref = (ref.float() @ head.t()) * mult
        lg_ps = (ps.float() @ head.t()) * mult
        if softcap is not None and float(softcap) > 0:
            sc = float(softcap)
            lg_ref = torch.tanh(lg_ref / sc) * sc
            lg_ps = torch.tanh(lg_ps / sc) * sc
        entries = []
        for r in range(rows):
            a = lg_ref[r]
            b = lg_ps[r]
            ta = torch.topk(a, top_k).indices
            tb = torch.topk(b, top_k).indices
            overlap = len(set(ta.tolist()) & set(tb.tolist()))
            am = int(a.argmax().item())
            bm = int(b.argmax().item())
            # top1 margin
            sa = torch.sort(a, descending=True).values
            sb = torch.sort(b, descending=True).values
            entry = {
                "row": r, "argmax_ref": am, "argmax_ps": bm, "argmax_match": am == bm,
                "topk_overlap": f"{overlap}/{top_k}",
                "margin_ref": float(sa[0] - sa[1]), "margin_ps": float(sb[0] - sb[1]),
                "l2_logits": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)),
                "top1_rank_in_ps": int((b > b[am]).sum().item()) + 1,
                "top1_value_ref": float(a[am]), "top1_value_ps": float(b[am]),
            }
            entries.append(entry)
            print(f"  {mode} row {r}: argmax {am}->{bm} {'OK' if am==bm else 'MISMATCH'} "
                  f"top{top_k} {overlap}/{top_k} logit_rel_l2={entry['l2_logits']:.3f} "
                  f"ref_top1_rank_in_ps={entry['top1_rank_in_ps']} "
                  f"margin {entry['margin_ref']:.3f}->{entry['margin_ps']:.3f}")
        report[mode] = entries
    json.dump(report, open(os.path.join(out_root, "logit_sensitivity.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B-DFlash2")
    ap.add_argument("--fixture-dir", default="build/dflash2-gate5-reference")
    ap.add_argument("--official-dir", default="build/dflash2-gate5-reference")
    ap.add_argument("--case", default="ctx8_blk8")
    ap.add_argument("--out", default="build/dflash2-gate51")
    ap.add_argument("--ps-dir", default="build/dflash2-gate51/ps")
    ap.add_argument("--target-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--mode", default="all",
                    choices=["official", "replay", "replay_f32", "compare", "all", "rope", "logits"])
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(16, (os.cpu_count() or 2) // 2)))

    exporter = load_exporter()
    official = exporter.import_official(os.path.join(args.official_dir, "_official", "model.py"))
    from transformers.models.qwen3 import modeling_qwen3 as tq
    config, _ = exporter.read_config(args.model_dir)
    cfg = exporter.build_config(tq, config)
    weights = exporter.load_full_weights(args.model_dir)
    meta = json.load(open(os.path.join(args.fixture_dir, args.case, "metadata.json")))
    fix = load_file(os.path.join(args.fixture_dir, args.case, "fixture.safetensors"))
    case = {
        "context_rows": meta["context_rows"], "block_rows": meta["block_rows"],
        "context_start": meta["context_start"], "block_start": meta["block_start"],
        "noise_embedding": fix["noise_embedding"], "target_concat": fix["target_concat"],
        "target_feature": fix["target_feature"],
        "taps": [fix[f"tap{i}"] for i in range(5)],
    }
    os.makedirs(args.out, exist_ok=True)

    if args.mode in ("official", "all"):
        cap = OfficialModel(official, cfg, weights, torch.bfloat16)
        stats = run_official_trace(cap, case, os.path.join(args.out, "official_bf16"), True)
        json.dump(stats, open(os.path.join(args.out, "official_dtype_trace.json"), "w"), indent=2)
        print("[trace] official bf16 done")
        cap32 = OfficialModel(official, cfg, weights, torch.float32)
        run_official_trace(cap32, case, os.path.join(args.out, "official_f32"), False)
        print("[trace] official f32 done")

    if args.mode in ("replay", "all"):
        run_replay(cfg, weights, case, os.path.join(args.out, "replay_bf16"), "bf16")

    if args.mode in ("replay_f32", "all"):
        global REPLAY_F32
        REPLAY_F32 = True
        run_replay(cfg, weights, case, os.path.join(args.out, "replay_f32"), "f32")
        REPLAY_F32 = False

    if args.mode in ("compare", "all"):
        compare_all(args.out, args.ps_dir, case, cfg)

    if args.mode == "rope":
        rope_diagnostic(args.ps_dir, case, cfg)

    if args.mode == "logits":
        logit_sensitivity(args.out, args.ps_dir, case, cfg, args.target_dir, args.fixture_dir, args.case)
    return 0


def rope_diagnostic(ps_dir, case, cfg):
    rows, q_f, head_dim = case["block_rows"], int(cfg.num_attention_heads) * int(cfg.head_dim), int(cfg.head_dim)
    q_norm = tensor_from_bf16_bin(os.path.join(ps_dir, "layer0_q_norm.bin"), (rows, q_f))
    q_rope = tensor_from_bf16_bin(os.path.join(ps_dir, "layer0_q_rope.bin"), (rows, q_f))
    if q_norm is None or q_rope is None:
        print("rope diagnostic: PS dump missing")
        return
    for theta in (1e7,):
        got = ps_rope(q_norm, q_f, head_dim, case["block_start"], theta)
        diff = (got.float() - q_rope.float()).abs()
        print(f"ps_rope replay vs PS kernel: mismatched={int((diff>0).sum())}/{diff.numel()} "
              f"rel_l2={rel_l2(q_rope, got):.3e}")
    half = head_dim // 2
    inv_pow = torch.tensor(1e7, dtype=torch.float32) ** (
        -2.0 * torch.arange(half, dtype=torch.float32) / float(head_dim))
    inv_torch = (1.0 / (1e7 ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))).to(torch.float32)
    print("inv_freq max rel diff:", float(((inv_pow - inv_torch).abs() / inv_torch.abs()).max()))
    pos = torch.arange(case["context_start"], case["context_start"] + case["context_rows"] + rows,
                       dtype=torch.float32)
    a_cos = (pos[:, None] * inv_pow[None, :]).cos().to(torch.bfloat16)
    b_cos = (pos[:, None] * inv_torch[None, :]).cos().to(torch.bfloat16)
    a_sin = (pos[:, None] * inv_pow[None, :]).sin().to(torch.bfloat16)
    b_sin = (pos[:, None] * inv_torch[None, :]).sin().to(torch.bfloat16)
    print("cos bf16 mismatch:", int((a_cos != b_cos).sum()), "/", a_cos.numel(),
          " sin mismatch:", int((a_sin != b_sin).sum()), "/", a_sin.numel())


if __name__ == "__main__":
    raise SystemExit(main())
