import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ffn_k_tile_oracle"))

from oracle_down import read_corpus  # noqa: E402
from run_cancel import capture_partials  # noqa: E402
from run_oracle import load_model  # noqa: E402
from subset_select import norms2, objective  # noqa: E402
from sketch_select import SKETCH_METHODS, choose  # noqa: E402
from projection import apply_R, make_R_all  # noqa: E402


def stats(J, y2, gsum):
    q = torch.quantile(J, torch.tensor([0.5, 0.9, 0.99], device=J.device))
    return {
        "n": int(J.numel()),
        "mean_err": float(J.mean()),
        "median_err": float(q[0]),
        "p90_err": float(q[1]),
        "p99_err": float(q[2]),
        "mean_R_output": float((J / (y2 + 1e-30)).mean()),
        "mean_R_group": float((J / (gsum + 1e-30)).mean()),
    }


def mask_of(idx, ng):
    m = torch.zeros(idx.shape[0], ng, dtype=torch.bool, device=idx.device)
    m.scatter_(1, idx, True)
    return m


def overlap(idx, ref, ng):
    a = mask_of(idx, ng)
    b = mask_of(ref, ng)
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float().clamp_min(1.0)
    k = float(idx.shape[1])
    return float((inter / k).mean()), float((inter / union).mean())


def select_all(c, k):
    return {
        "small_l2": choose("small_l2", c, None, k),
        "residual": choose("exact_residual", c, None, k),
        "pair": choose("exact_pair", c, None, k),
    }


def select_sketch(z, k, seed_count):
    return {name: choose(name, None, z, k, seed_count) for name in SKETCH_METHODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--layers", default="0,16,28,32,48,63")
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--dims", default="2,4,8,16")
    ap.add_argument("--skips", default="10,14,16,17,20,24,27,28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_sketch/diagnostic.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    layers = [int(x) for x in args.layers.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    skips = [int(x) for x in args.skips.split(",")]
    dmax = max(dims)

    tokens = read_corpus(args.corpus)
    stride = max(1, len(tokens) // args.n_tokens)
    positions = list(range(0, len(tokens), stride))[:args.n_tokens]

    t0 = time.time()
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    partials = capture_partials(model, tokens, args.k_group, args.tile_width, layers,
                                positions, device)
    del model
    torch.cuda.empty_cache()
    P = len(positions)
    print(f"[diag] captured in {time.time()-t0:.1f}s", flush=True)

    store = {}
    per_layer = {}
    for L in layers:
        c = partials[L].to(device).float()
        R_rows, ng, D = c.shape
        nt = R_rows // P
        c4 = c.reshape(P, nt, ng, D)
        y = c.sum(1)
        y2 = (y * y).sum(-1)
        gsum = norms2(c).sum(-1)
        Rall = make_R_all(L, nt, dmax, args.seed).to(device)
        layer_entry = {}
        for k in skips:
            ex = select_all(c, k)
            ref_pair = ex["pair"]
            ref_res = ex["residual"]
            Jpair = objective(c, ref_pair)
            Jres = objective(c, ref_res)
            layer_entry[str(k)] = {}
            for name, idx in ex.items():
                J = objective(c, idx)
                ov, jac = overlap(idx, ref_pair, ng)
                m = stats(J, y2, gsum)
                m["ratio_vs_pair"] = float(J.mean() / Jpair.mean())
                m["ratio_vs_residual"] = float(J.mean() / Jres.mean())
                m["intersection"] = ov
                m["jaccard"] = jac
                layer_entry[str(k)][name] = m
                rec = store.setdefault(("exact", name, k), {"J": [], "ov": [], "jac": []})
                rec["J"].append(J.cpu())
                rec["ov"].append(ov)
                rec["jac"].append(jac)
            for d in dims:
                z = apply_R(c4, Rall, d).reshape(R_rows, ng, d)
                for name, idx in select_sketch(z, k, args.seed_count).items():
                    J = objective(c, idx)
                    ov, jac = overlap(idx, ref_pair, ng)
                    m = stats(J, y2, gsum)
                    m["ratio_vs_pair"] = float(J.mean() / Jpair.mean())
                    m["ratio_vs_residual"] = float(J.mean() / Jres.mean())
                    m["intersection"] = ov
                    m["jaccard"] = jac
                    layer_entry[str(k)][f"d{d}:{name}"] = m
                    rec = store.setdefault((f"d{d}", name, k), {"J": [], "ov": [], "jac": []})
                    rec["J"].append(J.cpu())
                    rec["ov"].append(ov)
                    rec["jac"].append(jac)
        per_layer[str(L)] = layer_entry
        print(f"[diag] layer {L} done", flush=True)

    aggregate = {}
    for (group, name, k), rec in store.items():
        J = torch.cat(rec["J"])
        m = stats(J, torch.ones_like(J), torch.ones_like(J))
        refp = torch.cat(store[("exact", "pair", k)]["J"])
        refr = torch.cat(store[("exact", "residual", k)]["J"])
        m["ratio_vs_pair"] = float(J.mean() / refp.mean())
        m["ratio_vs_residual"] = float(J.mean() / refr.mean())
        m["intersection"] = float(sum(rec["ov"]) / len(rec["ov"]))
        m["jaccard"] = float(sum(rec["jac"]) / len(rec["jac"]))
        aggregate.setdefault(group, {}).setdefault(name, {})[str(k)] = m

    meta = {"model_dir": args.model_dir, "corpus": args.corpus, "layers": layers,
            "n_tokens": P, "positions": positions, "dims": dims, "skips": skips,
            "seed": args.seed, "seed_count": args.seed_count, "tile_width": args.tile_width,
            "n_groups": ng}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "aggregate": aggregate, "per_layer": per_layer}, f, indent=1)
    print(f"[diag] wrote {args.out} in {time.time()-t0:.1f}s", flush=True)

    for k in skips:
        parts = [f"k={k:2d}({k/68*100:5.2f}%)"]
        for g, n in (("exact", "residual"), ("exact", "pair"), ("d2", "sk_residual"),
                     ("d4", "sk_residual"), ("d4", "sk_pair4"), ("d8", "sk_residual"),
                     ("d8", "sk_pair4"), ("d16", "sk_pair4")):
            if g in aggregate and n in aggregate[g] and str(k) in aggregate[g][n]:
                parts.append(f"{g}/{n}={aggregate[g][n][str(k)]['ratio_vs_pair']:.3f}")
        print(" ".join(parts), flush=True)


if __name__ == "__main__":
    main()
