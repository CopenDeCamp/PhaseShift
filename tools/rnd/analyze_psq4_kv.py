#!/usr/bin/env python3
"""Analyze PSQ4 W32 (CB10) KV quality against FP8 on a calibration dump.

  python3 tools/rnd/analyze_psq4_kv.py \
      --k /tmp/psq4-calib/k.bf16 --v /tmp/psq4-calib/v.bf16 \
      --out-json /tmp/psq4-calib/psq4_kv_analysis.json

holdout split 上で、

  A: PSQ4 MAXABS  (4.5 bpw)
  B: PSQ4 LSQ1    (4.5 bpw)
  C: PSQ4 CANONICAL65
  D: FP8 E4M3 (per W32 block scale, 8 bpw)

を K/V 別に比較し、MSE 比 (MAXABS/C65, LSQ1/C65) を出す。
"""
import argparse
import json
import sys

import numpy as np

BLOCK = 32
MAGS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
THRESH = np.array([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0], dtype=np.float32)
E4M3_MAX = 448.0


def load_bf16(path):
    raw = np.fromfile(path, dtype="<u2")
    f = (raw.astype(np.uint32) << 16).view(np.float32)
    return f.reshape(-1, BLOCK).copy()


def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    bits = (u >> np.uint32(16)).astype(np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def mag_index(a):
    return np.searchsorted(THRESH, a, side="left").astype(np.int32)


def encode_fixed_scale(x, s):
    """x: (n,32), s: (n,) float32 scale -> recon (n,32) with CB10 nearest."""
    safe = np.where(s > 0.0, s, 1.0).astype(np.float32)
    t = x / safe[:, None]
    t = np.where(s[:, None] > 0.0, t, 0.0)
    idx = mag_index(np.abs(t))
    q = np.sign(t) * MAGS[idx]
    return s[:, None] * q


def psq4_maxabs(x):
    amax = np.abs(x).max(axis=1).astype(np.float32)
    s = np.where(amax > 0.0, to_bf16(amax * 0.1), 0.0).astype(np.float32)
    return encode_fixed_scale(x, s)


def psq4_lsq1(x):
    amax = np.abs(x).max(axis=1).astype(np.float32)
    s0 = np.where(amax > 0.0, to_bf16(amax * 0.1), 0.0).astype(np.float32)
    safe = np.where(s0 > 0.0, s0, 1.0).astype(np.float32)
    t = np.where(s0[:, None] > 0.0, x / safe[:, None], 0.0)
    idx = mag_index(np.abs(t))
    q = np.sign(t) * MAGS[idx]
    num = (x * q).sum(axis=1)
    den = (q * q).sum(axis=1)
    s1 = np.where(den > 0.0, to_bf16(num / np.maximum(den, 1e-30)), 0.0).astype(np.float32)
    return encode_fixed_scale(x, s1)


def psq4_canonical65(x):
    n = x.shape[0]
    amax = np.abs(x).max(axis=1).astype(np.float32)
    s0 = np.where(amax > 0.0, amax / 10.0, 0.0).astype(np.float32)
    best_err = np.full(n, np.inf, dtype=np.float32)
    best_recon = np.zeros_like(x)
    for j in range(-32, 33):
        s = to_bf16(s0 * np.exp2(np.float32(j) / 64.0)).astype(np.float32)
        valid = (s > 0.0) & np.isfinite(s)
        safe = np.where(valid, s, 1.0).astype(np.float32)
        t = x / safe[:, None]
        idx = mag_index(np.abs(t))
        q = np.sign(t) * MAGS[idx]
        recon = s[:, None] * q
        err = ((x - recon) ** 2).sum(axis=1)
        better = valid & (err < best_err)
        best_err = np.where(better, err, best_err)
        best_recon = np.where(better[:, None], recon, best_recon)
    return best_recon


def fp8_e4m3(x):
    try:
        import torch
    except ImportError:
        return None
    amax = np.abs(x).max(axis=1).astype(np.float32)
    s = np.where(amax > 0.0, amax / E4M3_MAX, 1.0).astype(np.float32)
    t = torch.from_numpy((x / s[:, None]).astype(np.float32))
    q = t.to(torch.float8_e4m3fn).float().numpy()
    return q * s[:, None]


def metrics(x, recon, prefix, out):
    err = recon - x
    mse = float((err ** 2).mean())
    sig = float((x ** 2).mean())
    rel_l2 = float(np.sqrt(mse / max(sig, 1e-30)))
    xf = x.reshape(-1)
    rf = recon.reshape(-1)
    denom = float(np.linalg.norm(xf) * np.linalg.norm(rf))
    cos = float(np.dot(xf, rf) / denom) if denom > 0 else 1.0
    block_mse = (err ** 2).mean(axis=1)
    out[prefix + "_mse"] = mse
    out[prefix + "_rel_l2"] = rel_l2
    out[prefix + "_cosine"] = cos
    out[prefix + "_block_mse_p50"] = float(np.percentile(block_mse, 50))
    out[prefix + "_block_mse_p95"] = float(np.percentile(block_mse, 95))
    out[prefix + "_block_mse_p99"] = float(np.percentile(block_mse, 99))
    return block_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True)
    ap.add_argument("--v", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    k_blocks = load_bf16(args.k)
    v_blocks = load_bf16(args.v)
    n = k_blocks.shape[0]
    holdout = 0.2
    seed = 20240901
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = int(round(n * holdout))
    hold_idx = perm[:n_hold]

    report = {
        "block": BLOCK,
        "bit_per_value": 4.5,
        "calibration_blocks": int(n),
        "holdout_blocks": int(n_hold),
        "holdout_fraction": holdout,
        "seed": seed,
    }

    for name, blocks in (("K", k_blocks), ("V", v_blocks)):
        ho = blocks[hold_idx]
        metrics(ho, psq4_maxabs(ho), name + "_psq4_maxabs", report)
        metrics(ho, psq4_lsq1(ho), name + "_psq4_lsq1", report)
        metrics(ho, psq4_canonical65(ho), name + "_psq4_canonical65", report)
        r_fp8 = fp8_e4m3(ho)
        if r_fp8 is not None:
            metrics(ho, r_fp8, name + "_fp8_block", report)
        report[name + "_maxabs_over_canonical65"] = (
            report[name + "_psq4_maxabs_mse"] / report[name + "_psq4_canonical65_mse"])
        report[name + "_lsq1_over_canonical65"] = (
            report[name + "_psq4_lsq1_mse"] / report[name + "_psq4_canonical65_mse"])
        report[name + "_psq4_lsq1_over_fp8"] = (
            report[name + "_psq4_lsq1_mse"] / report[name + "_fp8_block_mse"]
            if name + "_fp8_block_mse" in report else None)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for name in ("K", "V"):
        print(f"== {name} ==")
        for meth in ("psq4_maxabs", "psq4_lsq1", "psq4_canonical65", "fp8_block"):
            key = f"{name}_{meth}"
            if key + "_mse" not in report:
                continue
            print(f"  {meth:18s} mse={report[key+'_mse']:.6f} "
                  f"rel_l2={report[key+'_rel_l2']:.4f} cos={report[key+'_cosine']:.5f} "
                  f"p95={report[key+'_block_mse_p95']:.6f} p99={report[key+'_block_mse_p99']:.6f}")
        print(f"  maxabs/c65 = {report[name+'_maxabs_over_canonical65']:.4f}")
        print(f"  lsq1/c65   = {report[name+'_lsq1_over_canonical65']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
