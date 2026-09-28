#!/usr/bin/env python3
"""MTP Gate 1 trace comparison tool.

golden fixture (export_qwen35_mtp_gate1.py 生成) と
PhaseNonShift dump (test_qwen35_mtp_forward_real 生成) を比較し、
各テンソルの数値指標と FIRST_MISMATCH を報告する。

dump 形式:
  <dump-dir>/info.json   {"<name>": {"dtype": "bf16"|"f32"|"i32", "features": N}}
  <dump-dir>/<name>.bin  生バイト (row 0, features * elem_bytes)

比較順は Gate 1 仕様 §24 に従う。
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

COMPARE_ORDER = [
    "target_hidden",
    "embedding_raw",
    "embedding_norm",
    "target_hidden_norm",
    "concat_embedding_hidden",
    "fc_out",
    "input_layernorm_out",
    "q_proj_raw",
    "k_proj_raw",
    "v_proj_raw",
    "q_pre_norm",
    "gate_pre_sigmoid",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attention_core_out",
    "gate_sigmoid",
    "attention_gated",
    "o_proj_out",
    "residual1",
    "post_attention_norm_out",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_silu_gate",
    "mlp_mul",
    "mlp_down_proj",
    "final_residual",
    "selected_hidden",
    "final_norm",
    "logits",
]

# name -> (rel_l2_max, max_abs_diff_max, cosine_min)
# --report-only 時は無視される。
THRESHOLDS = {
    "target_hidden": (0.0, 0.0, 1.0),
    "embedding_raw": (0.0, 0.0, 1.0),
    "embedding_norm": (1e-2, None, 0.9999),
    "target_hidden_norm": (1e-2, None, 0.9999),
    "concat_embedding_hidden": (1e-2, None, 0.9999),
    "fc_out": (1e-2, None, 0.9999),
    "input_layernorm_out": (1e-2, None, 0.9999),
    "q_proj_raw": (1e-2, None, 0.9999),
    "k_proj_raw": (1e-2, None, 0.9999),
    "v_proj_raw": (1e-2, None, 0.9999),
    "q_pre_norm": (1e-2, None, 0.9999),
    "gate_pre_sigmoid": (1e-2, None, 0.9999),
    "q_norm": (1e-2, None, 0.99999),
    "k_norm": (1e-2, None, 0.99999),
    "q_rope": (1e-2, None, 0.9999),
    "k_rope": (1e-2, None, 0.9999),
    "attention_core_out": (1e-2, None, 0.99999),
    "gate_sigmoid": (1e-2, None, 0.99999),
    "attention_gated": (1e-2, None, 0.9999),
    "o_proj_out": (2e-2, None, 0.9999),
    "residual1": (2e-2, None, 0.9999),
    "post_attention_norm_out": (2e-2, None, 0.9999),
    "mlp_gate_proj": (2e-2, None, 0.9999),
    "mlp_up_proj": (2e-2, None, 0.9999),
    "mlp_silu_gate": (2e-2, None, 0.9999),
    "mlp_mul": (2e-2, None, 0.9999),
    "mlp_down_proj": (2e-2, None, 0.9999),
    "final_residual": (2e-2, None, 0.9999),
    "selected_hidden": (2e-2, None, 0.9999),
    "final_norm": (2e-2, None, 0.9999),
    "logits": (2e-2, None, 0.9999),
}

# bit exact を要求する tensor (計算を伴わない / 単一参照)。
BIT_EXACT = {"target_hidden", "embedding_raw"}

STRICT_SHAPE_EXACT = True


def _to_f64(t):
    if t.dtype == torch.bfloat16:
        u16 = t.view(torch.uint16).numpy().astype(np.uint32)
        return (u16 << 16).view(np.float32).astype(np.float64)
    return t.numpy().astype(np.float64)


def load_golden(case_dir):
    with open(os.path.join(case_dir, "metadata.json")) as f:
        meta = json.load(f)
    tensors = {}
    with safe_open(os.path.join(case_dir, "fixture.safetensors"), framework="pt") as st:
        for name in st.keys():
            if name in ("input_token_id", "position", "argmax"):
                continue
            tensors[name] = _to_f64(st.get_tensor(name))
    return meta, tensors


def load_dump(dump_dir):
    with open(os.path.join(dump_dir, "info.json")) as f:
        info = json.load(f)
    tensors = {}
    for name, spec in info.items():
        path = os.path.join(dump_dir, name + ".bin")
        with open(path, "rb") as f:
            raw = f.read()
        dt = spec["dtype"]
        features = spec["features"]
        if dt == "bf16":
            v = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = v.astype("<u4").view("<f4")
        elif dt == "f32":
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        elif dt == "i32":
            arr = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        else:
            raise ValueError(f"unknown dump dtype: {dt}")
        tensors[name] = arr[:features]
    return info, tensors


def metrics(ref, act, name):
    m = {
        "name": name,
        "shape": (list(ref.shape), list(act.shape)),
        "shape_ok": ref.shape == act.shape,
    }
    if not m["shape_ok"]:
        return m
    if not (np.isfinite(ref).all() and np.isfinite(act).all()):
        m["finite"] = False
        return m
    m["finite"] = True
    ref64 = ref.astype(np.float64)
    act64 = act.astype(np.float64)
    diff = np.abs(ref64 - act64)
    m["min"] = (float(ref64.min()), float(act64.min()))
    m["max"] = (float(ref64.max()), float(act64.max()))
    m["mean"] = (float(ref64.mean()), float(act64.mean()))
    l2 = np.sqrt((ref64 * ref64).sum())
    m["l2_ref"] = float(l2)
    m["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    m["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    num = np.sqrt(((ref64 - act64) ** 2).sum())
    den = np.sqrt((ref64 ** 2).sum())
    m["relative_l2"] = float(num / den) if den > 0 else 0.0
    dot = float((ref64 * act64).sum())
    nr = float(np.sqrt((ref64 ** 2).sum()))
    na = float(np.sqrt((act64 ** 2).sum()))
    m["cosine"] = dot / (nr * na) if nr > 0 and na > 0 else 1.0
    if diff.size:
        wi = int(diff.argmax())
        m["worst_index"] = wi
        m["worst_ref"] = float(ref64.flat[wi])
        m["worst_act"] = float(act64.flat[wi])
    return m


def evaluate(metrics_row, name):
    if not metrics_row.get("shape_ok", False):
        return False, "shape mismatch"
    if not metrics_row.get("finite", False):
        return False, "non-finite value"
    if name in BIT_EXACT:
        if metrics_row["max_abs_diff"] == 0.0:
            return True, "bit-exact"
        return False, f"bit-exact expected, max_abs_diff={metrics_row['max_abs_diff']}"
    t = THRESHOLDS.get(name)
    if t is None:
        return True, "no-threshold"
    rel_max, mad_max, cos_min = t
    if metrics_row["relative_l2"] > rel_max:
        return False, f"relative_l2={metrics_row['relative_l2']:.3e} > {rel_max:.1e}"
    if mad_max is not None and metrics_row["max_abs_diff"] > mad_max:
        return False, f"max_abs_diff={metrics_row['max_abs_diff']:.3e} > {mad_max}"
    if metrics_row["cosine"] < cos_min:
        return False, f"cosine={metrics_row['cosine']:.6f} < {cos_min}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="golden case dir (metadata.json + fixture.safetensors)")
    ap.add_argument("--dump", required=True, help="PhaseNonShift dump dir (info.json + *.bin)")
    ap.add_argument("--report-only", action="store_true",
                    help="threshold を適用せず指標だけ報告する")
    args = ap.parse_args()

    meta, golden = load_golden(args.golden)
    info, dump = load_dump(args.dump)

    print(f"=== MTP Gate 1 comparison: case={meta['case']} ===")
    print(f"golden: {args.golden}")
    print(f"dump:   {args.dump}")
    print()

    first_mismatch = None
    failures = []
    header = (f"{'tensor':<26} {'rel_l2':>12} {'max_abs':>12} {'cosine':>12} "
              f"{'worst_idx':>10} {'ref':>14} {'act':>14}  verdict")
    print(header)
    print("-" * len(header))

    for name in COMPARE_ORDER:
        if name not in golden:
            print(f"{name:<26} MISSING IN GOLDEN")
            continue
        if name == "mlp_silu_gate" and name not in dump:
            if "mlp_gate_proj" not in dump:
                print(f"{name:<26} MISSING IN DUMP (mlp_gate_proj)")
                failures.append((name, "missing in dump (mlp_gate_proj)"))
                if first_mismatch is None:
                    first_mismatch = name
                continue
            g = dump["mlp_gate_proj"]
            dump[name] = g / (1.0 + np.exp(-g))
        if name not in dump:
            print(f"{name:<26} MISSING IN DUMP")
            failures.append((name, "missing in dump"))
            if first_mismatch is None:
                first_mismatch = name
            continue
        row = metrics(golden[name], dump[name], name)
        if not row.get("shape_ok", False):
            verdict = "FAIL shape"
        elif not row.get("finite", False):
            verdict = "FAIL non-finite"
        else:
            ok, why = evaluate(row, name) if not args.report_only else (True, "report-only")
            verdict = "PASS" if ok else f"FAIL {why}"
        extra = ""
        if row.get("shape_ok") and row.get("finite"):
            extra = (f"{row['relative_l2']:>12.3e} {row['max_abs_diff']:>12.3e} "
                     f"{row['cosine']:>12.8f} {row.get('worst_index', -1):>10d} "
                     f"{row.get('worst_ref', 0.0):>14.6f} {row.get('worst_act', 0.0):>14.6f}")
        else:
            extra = f"shape={row.get('shape')}"
        print(f"{name:<26} {extra}  {verdict}")
        if "FAIL" in verdict and first_mismatch is None:
            first_mismatch = name
            failures.append((name, verdict))
        elif "FAIL" in verdict:
            failures.append((name, verdict))

    print()
    # top-k / argmax
    ref_top1 = meta["top1_token_id"]
    ref_top10 = meta["top10_token_ids"]
    dump_top10 = None
    if "logits" in dump:
        lg = np.asarray(dump["logits"], dtype=np.float64)
        idx = np.argsort(-lg, kind="stable")[:10]
        dump_top10 = [int(x) for x in idx]
        dump_top1 = int(idx[0])
    else:
        dump_top1 = None
    if "argmax" in dump:
        dump_argmax = int(np.asarray(dump["argmax"]).flat[0])
    else:
        dump_argmax = None

    if dump_top1 is not None:
        argmax_ok = (dump_top1 == ref_top1)
        print(f"top1:    reference={ref_top1} actual={dump_top1} "
              f"{'MATCH' if argmax_ok else 'MISMATCH'}")
        if dump_argmax is not None:
            print(f"argmax:  reference={ref_top1} actual(sampling)={dump_argmax} "
                  f"{'MATCH' if dump_argmax == ref_top1 else 'MISMATCH'}")
        if dump_top10 is not None:
            print(f"top10 ref: {ref_top10}")
            print(f"top10 act: {dump_top10}")
            set_ok = set(ref_top10) == set(dump_top10)
            rank_ok = ref_top10 == dump_top10
            print(f"top10 set: {'MATCH' if set_ok else 'MISMATCH'}  "
                  f"ranking: {'MATCH' if rank_ok else 'DIFF'}")
            if not argmax_ok:
                if first_mismatch is None:
                    first_mismatch = "argmax"
                failures.append(("argmax", "top1 mismatch"))
    else:
        print("top1:    (logits not in dump)")

    print()
    if first_mismatch is None:
        print("FIRST_MISMATCH=<none>")
        print("MTP_GATE1_COMPARE: PASS")
        return 0
    print(f"FIRST_MISMATCH={first_mismatch}")
    print("MTP_GATE1_COMPARE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
