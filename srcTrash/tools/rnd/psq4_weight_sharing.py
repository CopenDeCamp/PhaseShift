#!/usr/bin/env python3
"""PSQ4 weight sharing feasibility analysis.

lossless な共有（exact duplicate）と、lossy な共有（spectral / subspace /
codebook）の両方を実データで測る。production format は変更しない。
"""

import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from sklearn.utils.extmath import randomized_svd

CB = np.array([(-1.0 if (c & 8) else 1.0) * [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0][c & 7]
               for c in range(16)], dtype=np.float32)


def load_manifest(model_dir):
    with open(os.path.join(model_dir, "phaseshift_quantization.json")) as f:
        return json.load(f)["tensors"]


def load_index(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def read_tensor(model_dir, index, name):
    with safe_open(os.path.join(model_dir, index[name]), framework="numpy") as f:
        return f.get_tensor(name)


def load_psq4(model_dir, index, meta):
    codes = read_tensor(model_dir, index, meta["codes"]["tensor"])
    scales = read_tensor(model_dir, index, meta["metadata1"]["tensor"])
    m = codes.shape[0]
    nb = codes.shape[1] // 16
    blk = codes.reshape(m, nb, 16)
    low = blk & np.uint8(0x0F)
    high = (blk >> np.uint8(4)) & np.uint8(0x0F)
    idx = np.concatenate([low, high], axis=-1).reshape(m, nb * 32)
    sc_u16 = scales.view(np.uint16)
    sc_f = (sc_u16.astype(np.uint32) << 16).view(np.float32)
    k = int(meta["logical_shape"][1])
    w = CB[idx] * np.repeat(sc_f, 32, axis=1).astype(np.float32)
    return idx[:, :k], w[:, :k], blk, sc_u16


def block_duplicates(blk, sc):
    n = blk.shape[0] * blk.shape[1]
    flat = blk.reshape(-1, 16)
    u = flat.view(np.uint64).reshape(-1, 2)
    _, counts = np.unique(u, axis=0, return_counts=True)
    code_blocks = int(flat.shape[0])
    code_unique = int(counts.shape[0])
    sc_u16 = sc.reshape(-1).view(np.uint16) if sc.dtype != np.uint16 else sc.reshape(-1)
    sc_unique = int(np.unique(np.asarray(sc_u16)).shape[0])
    return {"blocks": code_blocks, "code_unique": code_unique,
            "code_dup_ratio": 1.0 - code_unique / code_blocks,
            "scale_unique": sc_unique, "scale_values": int(sc_u16.size)}


def hamming_nn(blk, sample=8000, seed=1234):
    flat = blk.reshape(-1, 16).astype(np.uint16)
    n = flat.shape[0]
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=min(sample, n), replace=False)
    s = flat[sel].astype(np.uint16)
    a = np.zeros((s.shape[0], 16), dtype=np.uint8)
    bits = np.zeros(s.shape[0], dtype=np.int32)
    mins = np.full(s.shape[0], 10 ** 9, dtype=np.int32)
    chunk = 512
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for start in range(0, s.shape[0], chunk):
        blk_a = s[start:start + chunk]
        x = np.bitwise_xor(blk_a[:, None, :], s[None, :, :]).astype(np.uint8)
        d = pop[x].sum(axis=2)
        d[np.arange(blk_a.shape[0]), np.arange(start, start + blk_a.shape[0])] = 10 ** 9
        mins[start:start + chunk] = d.min(axis=1)
    hist = {int(v): int(c) for v, c in zip(*np.unique(mins, return_counts=True))}
    return {"sample": int(s.shape[0]), "max_possible": 128,
            "min": int(mins.min()), "median": float(np.median(mins)),
            "mean": float(mins.mean()), "hist": hist}


def spectrum(w, ranks, seed=1234):
    m, k = w.shape
    rmax = min(max(ranks) + 32, min(m, k))
    u, s, vt = randomized_svd(w, n_components=rmax, random_state=seed)
    total = float(np.sum(w.astype(np.float64) ** 2))
    cum = np.cumsum(s.astype(np.float64) ** 2)
    out = {"fro": total, "sigma": [float(x) for x in s[:16]]}
    for r in ranks:
        if r <= len(s):
            out["energy_at_%d" % r] = float(cum[r - 1] / total)
    return out, u, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--role", default="ffn_gate")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--ranks", default="16,32,64,128,256,512")
    ap.add_argument("--skip-hamming", action="store_true")
    ap.add_argument("--skip-spectrum", action="store_true")
    ap.add_argument("--cross-layers", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tensors = load_manifest(args.model_dir)
    index = load_index(args.model_dir)
    ranks = [int(x) for x in args.ranks.split(",") if x]
    report = {"role": args.role, "cases": []}

    suffix = {"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
              "ffn_down": "mlp.down_proj", "attn_q": "self_attn.q_proj",
              "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
              "attn_o": "self_attn.o_proj"}[args.role]

    for layer in [int(x) for x in args.layers.split(",") if x]:
        name = "model.language_model.layers.%d.%s.weight" % (layer, suffix)
        meta = tensors.get(name)
        if meta is None or meta["encoding"] != "psq4":
            print("skip", name, flush=True)
            continue
        idx, w, blk, sc = load_psq4(args.model_dir, index, meta)
        rec = {"name": name, "shape": list(w.shape)}
        rec["blocks"] = block_duplicates(blk, sc)
        if not args.skip_hamming:
            rec["hamming_nn"] = hamming_nn(blk)
        if not args.skip_spectrum:
            sp, u, s = spectrum(w, ranks)
            rec["spectrum"] = sp
            np.save("/tmp/opencode/g11i/basis_%s_%d.npy" % (args.role, layer),
                    u.astype(np.float32))
        report["cases"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "spectrum"},
                         ensure_ascii=False)[:400], flush=True)
        if "spectrum" in rec:
            print("   energy:", {k: round(v, 5) for k, v in rec["spectrum"].items()
                                 if k.startswith("energy")}, flush=True)
        del w, idx

    if args.cross_layers:
        layers = [int(x) for x in args.cross_layers.split(",") if x]
        ref = layers[0]
        ref_name = "model.language_model.layers.%d.%s.weight" % (ref, suffix)
        meta = tensors[ref_name]
        idx, w, blk, sc = load_psq4(args.model_dir, index, meta)
        u, s, vt = randomized_svd(w, n_components=256, random_state=1234)
        q = u.astype(np.float32)
        del w
        cross = []
        for layer in layers[1:]:
            name = "model.language_model.layers.%d.%s.weight" % (layer, suffix)
            m2 = tensors[name]
            _, w2, _, _ = load_psq4(args.model_dir, index, m2)
            proj = q.T @ w2
            captured = float(np.sum(proj.astype(np.float64) ** 2))
            total = float(np.sum(w2.astype(np.float64) ** 2))
            cross.append({"layer": layer, "captured_ratio": captured / total})
            print("cross ref=%d layer=%d captured=%.5f" % (ref, layer, captured / total),
                  flush=True)
            del w2, proj
        report["cross"] = {"ref": ref, "rank": 256, "results": cross}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
