#!/usr/bin/env python3
"""Qwen3.5 MTP Gate 2 独立クロスチェック。

C++ テスト（test_qwen35_mtp_state_real）が
PHASESHIFT_MTP_GATE2_DUMP=<dir> で書き出す実測 dump を、Gate 2 golden
（synthetic）と比較する。C++ とは別経路で

  FIRST_STATE_MISMATCH_STEP
  FIRST_MISMATCH
  stable / ambiguous argmax classification

を出す。

dump 形式（raw little-endian）:
  <tag>_argmax.bin        int32  x steps
  <tag>_k.bin             bf16   x steps x kv_f
  <tag>_v.bin             bf16   x steps x kv_f
  <tag>_top10_ids.bin     int32  x steps x 10
  <tag>_top10_logits.bin  float  x steps x 10
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

K_REL_L2_MAX = 1e-2
V_REL_L2_MAX = 1e-2


def read_raw(path, dtype, shape):
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path}: size {arr.size} != {int(np.prod(shape))}")
    return arr.reshape(shape)


def to_f64_bf16(a):
    x = a.astype(np.uint16).astype(np.uint32) << 16
    return x.view(np.float32).astype(np.float64)


def rel_l2(actual, reference):
    num = float(np.sum((actual - reference) ** 2))
    den = float(np.sum(reference ** 2))
    return 0.0 if den <= 0.0 else (num / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True,
                    help="Gate 2 fixture root (contains synthetic/sequence.safetensors)")
    ap.add_argument("--dump", required=True, help="PHASESHIFT_MTP_GATE2_DUMP dir")
    ap.add_argument("--tag", default="synthetic")
    ap.add_argument("--checkpoint", action="store_true",
                    help="step 0/1/2/7/15/... だけでなく全 step を判定する")
    args = ap.parse_args()

    golden_dir = os.path.join(args.golden, args.tag)
    seq_path = os.path.join(golden_dir, "sequence.safetensors")
    if not os.path.exists(seq_path):
        print(f"FAIL: golden not found: {seq_path}")
        return 1
    g = load_file(seq_path, device="cpu")

    g_k = g["k_rope_current"].to(torch.float32).numpy()
    g_v = g["v_current"].to(torch.float32).numpy()
    g_arg = g["argmax"].to(torch.int32).numpy().reshape(-1)
    g_t10i = g["top10_ids"].to(torch.int32).numpy()
    g_t10l = g["top10_logits"].to(torch.float32).numpy()
    steps = g_arg.shape[0]
    kv_f = g_k.shape[-1]

    base = os.path.join(args.dump, args.tag)
    a_arg = read_raw(base + "_argmax.bin", np.int32, (steps,))
    a_k = read_raw(base + "_k.bin", np.uint16, (steps, kv_f))
    a_v = read_raw(base + "_v.bin", np.uint16, (steps, kv_f))
    a_t10i = read_raw(base + "_top10_ids.bin", np.int32, (steps, 10))
    a_t10l = read_raw(base + "_top10_logits.bin", np.float32, (steps, 10))

    a_kf = to_f64_bf16(a_k)
    a_vf = to_f64_bf16(a_v)
    g_kf = g_k.astype(np.float64)
    g_vf = g_v.astype(np.float64)

    first_state_step = -1
    first_state_field = ""
    first_mismatch = ""
    stable = 0
    ambiguous = 0
    match = 0
    worst_k = 0.0
    worst_v = 0.0

    for t in range(steps):
        rk = rel_l2(a_kf[t], g_kf[t])
        rv = rel_l2(a_vf[t], g_vf[t])
        worst_k = max(worst_k, rk)
        worst_v = max(worst_v, rv)
        if rk > K_REL_L2_MAX or rv > V_REL_L2_MAX:
            if first_state_step < 0:
                first_state_step = t
                first_state_field = "k_rope" if rk > K_REL_L2_MAX else "v"
            if not first_mismatch:
                first_mismatch = f"{first_state_field}@{t}"

        if int(a_arg[t]) != int(g_arg[t]):
            ref_margin = float(g_t10l[t, 0] - g_t10l[t, 1])
            eps = 0.0
            g_ids = g_t10i[t]
            a_ids = set(int(x) for x in a_t10i[t])
            for j in range(10):
                if int(g_ids[j]) in a_ids:
                    pos = int(np.where(a_t10i[t] == g_ids[j])[0][0])
                    eps = max(eps, abs(float(g_t10l[t, j]) - float(a_t10l[t, pos])))
            act_margin = float(a_t10l[t, 0] - a_t10l[t, 1])
            amb = (ref_margin <= 2.0 * eps) or (act_margin <= 2.0 * eps)
            if amb:
                ambiguous += 1
            else:
                stable += 1
                if first_state_step < 0:
                    first_state_step = t
                    first_state_field = "argmax"
                if not first_mismatch:
                    first_mismatch = f"argmax@{t}"
        else:
            match += 1

    print(f"steps={steps} argmax_match={match}/{steps} "
          f"ambiguous={ambiguous} stable={stable}")
    print(f"worst k_rel_l2={worst_k:.4e} v_rel_l2={worst_v:.4e}")
    print(f"FIRST_STATE_MISMATCH_STEP={first_state_step}")
    print(f"FIRST_STATE_MISMATCH_FIELD={first_state_field or '<none>'}")
    print(f"FIRST_MISMATCH={first_mismatch or '<none>'}")

    ok = (first_state_step < 0) and (stable == 0)
    print(f"MTP_GATE2_COMPARE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
