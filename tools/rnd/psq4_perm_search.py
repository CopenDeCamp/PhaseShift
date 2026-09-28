#!/usr/bin/env python3
"""PSQ4-E Gate E1: block-preserving permutation search + entropy re-measure.

32-weight block (= one PSQ scale block) を壊さない範囲で
intermediate channel を並べ替える。Gate/Up は row 単位 (= intermediate)、
Down は K 軸 (= intermediate) の block 単位。

permutation は探索 heuristic であり production ではない。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psq4_entropy_poc import (  # noqa: E402
    canonical_rows,
    entropy_stats,
    load_index,
    load_manifest,
    extract_codes,
)

BLOCK = 32


def hist_of(vec, minlength=16):
    return np.bincount(vec.reshape(-1).astype(np.uint8), minlength=minlength).astype(np.float64)


def greedy_nn(sig):
    m = sig.shape[0]
    used = np.zeros(m, dtype=bool)
    order = [0]
    used[0] = True
    for _ in range(m - 1):
        d = np.abs(sig - sig[order[-1]]).sum(axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
    return np.asarray(order, dtype=np.int64)


def row_block_signatures(row):
    n = row.shape[0]
    nb = n // BLOCK
    blocks = row[: nb * BLOCK].reshape(nb, BLOCK, row.shape[1])
    sig = np.stack([hist_of(blocks[g]) for g in range(nb)])
    return blocks, sig


def row_permutation(row, do_blocks, do_within):
    n = row.shape[0]
    nb = n // BLOCK
    blocks, sig = row_block_signatures(row)
    block_order = greedy_nn(sig) if do_blocks else np.arange(nb)
    ordered = blocks[block_order]
    within = np.arange(BLOCK) if not do_within else None
    out = np.empty_like(ordered)
    for g in range(nb):
        if do_within:
            s = np.stack([hist_of(ordered[g, r]) for r in range(BLOCK)])
            p = greedy_nn(s)
        else:
            p = within
        out[g] = ordered[g][p]
    return out.reshape(nb * BLOCK, row.shape[1]), block_order


def col_permutation(row, do_blocks, do_within):
    n, k = row.shape
    kb = k // BLOCK
    cols = row[:, : kb * BLOCK].reshape(n, kb, BLOCK)
    sig = np.stack([hist_of(cols[:, g, :]) for g in range(kb)])
    block_order = greedy_nn(sig) if do_blocks else np.arange(kb)
    ordered = cols[:, block_order, :]
    out = np.empty_like(ordered)
    for g in range(kb):
        if do_within:
            s = np.stack([hist_of(ordered[:, g, c]) for c in range(BLOCK)])
            p = greedy_nn(s)
        else:
            p = np.arange(BLOCK)
        out[:, g, :] = ordered[:, g, :][:, p]
    return out.reshape(n, kb * BLOCK)


def pick_layer(model_dir, role, encoding, layer):
    man = load_manifest(model_dir)
    suffix = ".mlp.gate_proj.weight" if role == "ffn_gate" else (
        ".mlp.up_proj.weight" if role == "ffn_up" else ".mlp.down_proj.weight")
    name = "model.language_model.layers.%d%s" % (layer, suffix)
    meta = man["tensors"][name]
    if meta["encoding"] != encoding:
        return None, None
    return name, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = load_manifest(args.model_dir)
    index = load_index(args.model_dir)

    report = {"layer": args.layer, "cases": []}
    for role, encoding, axis in (("ffn_gate", "psq4", "row"),
                                 ("ffn_up", "psq4", "row"),
                                 ("ffn_down", "psq4", "col")):
        name, meta = pick_layer(args.model_dir, role, encoding, args.layer)
        if name is None:
            report["cases"].append({"role": role, "encoding": encoding, "skipped": True})
            continue
        raw = extract_codes(args.model_dir, index, meta["codes"]["tensor"])
        row = canonical_rows(raw, meta["logical_shape"], meta["k_padded"])
        del raw
        base = entropy_stats(row)
        rec = {"role": role, "encoding": encoding, "name": name,
               "shape": list(meta["logical_shape"]), "before": base, "variants": {}}
        if axis == "row":
            for tag, db, dw in (("block_only", True, False),
                                ("within_only", False, True),
                                ("both", True, True)):
                perm, order = row_permutation(row, db, dw)
                st = entropy_stats(perm)
                rec["variants"][tag] = {"H0": st["H0"], "Hk": st["Hk"],
                                        "Hrow": st["Hrow"], "H2d": st["H2d"]}
                del perm
            rec["block_order"] = [int(x) for x in order[:16]]
        else:
            for tag, db, dw in (("block_only", True, False),
                                ("within_only", False, True),
                                ("both", True, True)):
                perm = col_permutation(row, db, dw)
                st = entropy_stats(perm)
                rec["variants"][tag] = {"H0": st["H0"], "Hk": st["Hk"],
                                        "Hrow": st["Hrow"], "H2d": st["H2d"]}
                del perm
        report["cases"].append(rec)
        print("%s %s before Hrow=%.4f Hk=%.4f H2d=%.4f" %
              (role, encoding, base["Hrow"], base["Hk"], base["H2d"]), flush=True)
        for tag, v in rec["variants"].items():
            print("   %-12s H0=%.4f Hk=%.4f Hrow=%.4f H2d=%.4f" %
                  (tag, v["H0"], v["Hk"], v["Hrow"], v["H2d"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
