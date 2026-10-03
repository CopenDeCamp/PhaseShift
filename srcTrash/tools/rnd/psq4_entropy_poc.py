#!/usr/bin/env python3
"""PSQ4 entropy PoC analyzer (Gate E0 / E1).

production runtime とは分離した RnD tool。quantized safetensors の
canonical PSQ4 codes を読み、symbol entropy を測る。

canonical layout (safetensors, row-major):
  block = 32 weights = 16 bytes
  byte j low nibble  -> k = ib*32 + j      (j in 0..15)
  byte j high nibble -> k = ib*32 + 16 + j
  metadata1 = BF16 scale 2 bytes / block
  codes row stride = k_padded / 2 bytes
  scale row stride = k_padded / 16 bytes
  (preshuffle は loader が in-memory で行うため file は canonical)
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np
from safetensors import safe_open


def shannon_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0.0]
    return float(-(p * np.log2(p)).sum())


def load_manifest(model_dir):
    path = os.path.join(model_dir, "phaseshift_quantization.json")
    with open(path) as f:
        return json.load(f)


def load_index(model_dir):
    path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(path) as f:
        return json.load(f)["weight_map"]


def extract_codes(model_dir, index, codes_tensor):
    shard = index.get(codes_tensor)
    if shard is None:
        raise KeyError(codes_tensor)
    with safe_open(os.path.join(model_dir, shard), framework="numpy") as f:
        return f.get_tensor(codes_tensor)


def canonical_rows(codes, logical_shape, k_padded):
    n, k = int(logical_shape[0]), int(logical_shape[1])
    nb = k_padded // 32
    row_bytes = k_padded // 2
    arr = np.asarray(codes, dtype=np.uint8)
    if arr.size != n * row_bytes:
        raise ValueError(
            "codes size mismatch: got %d expected %d" % (arr.size, n * row_bytes)
        )
    a = arr.reshape(n, nb, 16)
    low = a & np.uint8(0x0F)
    high = (a >> np.uint8(4)) & np.uint8(0x0F)
    row = np.concatenate([low, high], axis=-1)
    row = row.reshape(n, nb * 32)
    return row[:, :k]


def entropy_stats(row):
    flat = row.reshape(-1).astype(np.uint8)
    h0_counts = np.bincount(flat, minlength=16)
    h0 = shannon_from_counts(h0_counts)

    prev = row[:, :-1].reshape(-1).astype(np.uint16)
    cur = row[:, 1:].reshape(-1).astype(np.uint16)
    joint_k = (prev * 16 + cur).astype(np.uint32)
    ck = np.bincount(joint_k, minlength=256)
    prev_counts = np.bincount(prev, minlength=16)
    h_k = shannon_from_counts(ck) - shannon_from_counts(prev_counts)

    up = row[:-1, :].reshape(-1).astype(np.uint16)
    cur2 = row[1:, :].reshape(-1).astype(np.uint16)
    joint_r = (up * 16 + cur2).astype(np.uint32)
    cr = np.bincount(joint_r, minlength=256)
    up_counts = np.bincount(up, minlength=16)
    h_row = shannon_from_counts(cr) - shannon_from_counts(up_counts)

    pk = row[1:, :-1].reshape(-1).astype(np.uint32)
    pr = row[:-1, 1:].reshape(-1).astype(np.uint32)
    c2 = row[1:, 1:].reshape(-1).astype(np.uint32)
    idx = (pk * 256 + pr * 16 + c2).astype(np.uint32)
    c2d = np.bincount(idx, minlength=4096)
    ctx = (pk * 16 + pr).astype(np.uint32)
    cctx = np.bincount(ctx, minlength=256)
    h_2d = shannon_from_counts(c2d) - shannon_from_counts(cctx)

    # per-row and per-block H0 distribution
    row_h0 = []
    for r in range(row.shape[0]):
        row_h0.append(shannon_from_counts(np.bincount(row[r].astype(np.uint8), minlength=16)))
    block_h0 = []
    for ib in range(0, row.shape[1], 32):
        blk = row[:, ib:ib + 32].reshape(-1).astype(np.uint8)
        block_h0.append(shannon_from_counts(np.bincount(blk, minlength=16)))

    def pct(xs, q):
        return float(np.percentile(np.asarray(xs), q)) if xs else 0.0

    return {
        "raw_bpw": 4.0,
        "H0": h0,
        "Hk": h_k,
        "Hrow": h_row,
        "H2d": h_2d,
        "symbols": int(flat.size),
        "hist": [int(x) for x in h0_counts],
        "row_h0_p50": pct(row_h0, 50),
        "row_h0_p90": pct(row_h0, 90),
        "row_h0_p95": pct(row_h0, 95),
        "row_h0_p99": pct(row_h0, 99),
        "block_h0_p50": pct(block_h0, 50),
        "block_h0_p90": pct(block_h0, 90),
        "block_h0_p95": pct(block_h0, 95),
        "block_h0_p99": pct(block_h0, 99),
    }


def selected_tensors(tensors, roles, layer_stride, layers):
    out = []
    for name, meta in tensors.items():
        if meta.get("encoding") != "psq4":
            continue
        if roles and meta.get("role", "").lower() not in roles:
            continue
        layer = None
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer = int(parts[i + 1])
                except ValueError:
                    layer = None
                break
        if layers:
            if layer is None or layer not in layers:
                continue
        elif layer_stride and layer is not None and layer % layer_stride != 0:
            continue
        out.append((name, meta, layer))
    out.sort(key=lambda x: (x[1]["role"], x[2] if x[2] is not None else -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--roles", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--layer-stride", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    roles = [r.lower() for r in args.roles.split(",") if r]
    layers = [int(x) for x in args.layers.split(",") if x] if args.layers else None

    man = load_manifest(args.model_dir)
    index = load_index(args.model_dir)
    tensors = man["tensors"]
    sel = selected_tensors(tensors, roles, args.layer_stride, layers)
    if args.limit:
        sel = sel[: args.limit]
    if not sel:
        print("no tensors selected", file=sys.stderr)
        return 1

    results = []
    for i, (name, meta, layer) in enumerate(sel):
        codes_name = meta["codes"]["tensor"]
        raw = extract_codes(args.model_dir, index, codes_name)
        row = canonical_rows(raw, meta["logical_shape"], meta["k_padded"])
        st = entropy_stats(row)
        st.update({"name": name, "role": meta["role"], "layer": layer,
                   "shape": list(meta["logical_shape"]), "k_padded": meta["k_padded"]})
        results.append(st)
        print("[%d/%d] %s H0=%.4f Hk=%.4f Hrow=%.4f H2d=%.4f symbols=%d"
              % (i + 1, len(sel), name, st["H0"], st["Hk"], st["Hrow"], st["H2d"],
                 st["symbols"]), flush=True)

    with open(args.out, "w") as f:
        json.dump({"model_dir": args.model_dir, "results": results}, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
