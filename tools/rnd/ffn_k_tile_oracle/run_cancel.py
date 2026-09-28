import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_down import group_partials, read_corpus  # noqa: E402
from run_oracle import environment_info, git_info, load_model, mlp_list, perplexity  # noqa: E402
from subset_select import (apply_skip_indices, norms2, objective, pair_greedy_swap,  # noqa: E402
                           pair_seeded_greedy, partials_to_rows, residual_greedy, small_l2)

METHODS = ("small_l2", "residual_greedy", "pair_seeded_greedy", "pair_greedy_swap")
EPS = 1e-30


def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x != ""]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(name, C, k, seed_count=16, max_swap_iters=8):
    if k <= 0:
        return torch.zeros(C.shape[0], 0, dtype=torch.long, device=C.device)
    if name == "small_l2":
        return small_l2(C, k)
    if name == "residual_greedy":
        return residual_greedy(C, k)
    extra = small_l2(C, 2)
    if name == "pair_seeded_greedy":
        return pair_seeded_greedy(C, k, seed_count, extra_seed=extra)
    if name == "pair_greedy_swap":
        return pair_greedy_swap(C, k, seed_count, max_swap_iters, extra_seed=extra)
    raise ValueError(name)


class KSubsetOracleDown:
    def __init__(self, model, k_group, tile_width, method, skip_count, chunk_tokens=64,
                 seed_count=16, max_swap_iters=8):
        self.model = model
        self.k_group = k_group
        self.tile_width = tile_width
        self.method = method
        self.skip_count = skip_count
        self.chunk_tokens = chunk_tokens
        self.seed_count = seed_count
        self.max_swap_iters = max_swap_iters
        self._orig = {}
        self.skipped = 0
        self.total = 0

    def _forward_factory(self, mlp):
        w = mlp.down_proj.weight
        kg = self.k_group
        tw = self.tile_width
        skip = self.skip_count
        chunk = self.chunk_tokens
        state = self

        def forward(x):
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            h = mlp.act_fn(gate) * up
            shp = h.shape
            hf = h.reshape(-1, shp[-1])
            T = hf.shape[0]
            parts = []
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                c = group_partials(hf[start:end], w, kg)
                rows = partials_to_rows(c, tw).contiguous()
                idx = select(self.method, rows, skip, self.seed_count, self.max_swap_iters)
                parts.append(apply_skip_indices(c, idx, tw))
                state.skipped += int(idx.numel())
                state.total += int(rows.shape[0] * rows.shape[1])
            y = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            return y.to(h.dtype).reshape(*shp[:-1], w.shape[0])

        return forward

    def attach(self):
        for layer, mlp in mlp_list(self.model):
            self._orig[id(mlp)] = (mlp, mlp.forward)
            mlp.forward = self._forward_factory(mlp)
        return self

    def detach(self):
        for mlp, orig in self._orig.values():
            mlp.forward = orig
        self._orig.clear()

    def actual_skip_ratio(self):
        if self.total == 0:
            return 0.0
        return self.skipped / self.total


def capture_partials(model, tokens, k_group, tile_width, layers, positions, device):
    mlps = dict(mlp_list(model))
    out = {}
    orig = {}
    for layer in layers:
        mlp = mlps[layer]
        orig[layer] = mlp.forward

        def make(L, m):
            def fwd(x):
                gate = m.gate_proj(x)
                up = m.up_proj(x)
                h = m.act_fn(gate) * up
                hf = h.reshape(-1, h.shape[-1])
                pos = torch.tensor(positions, dtype=torch.long, device=hf.device)
                hsel = hf.index_select(0, pos).float()
                c = group_partials(hsel, m.down_proj.weight, k_group)
                out[L] = partials_to_rows(c, tile_width).contiguous().cpu()
                return m.down_proj(h)
            return fwd

        mlp.forward = make(layer, mlp)

    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for layer, fwd in orig.items():
            mlps[layer].forward = fwd
    torch.cuda.empty_cache()
    return {L: out[L] for L in layers}


def pair_statistics(C):
    B, ng, D = C.shape
    G = torch.bmm(C, C.transpose(1, 2))
    n2 = norms2(C)
    nrm = n2.sqrt().clamp_min(1e-30)
    cos = G / (nrm[:, :, None] * nrm[:, None, :])
    i, j = torch.triu_indices(ng, ng, offset=1, device=C.device)
    cv = cos[:, i, j].reshape(-1)
    pc = n2[:, i] + n2[:, j] + 2.0 * G[:, i, j]
    ratio = pc / (n2[:, i] + n2[:, j]).clamp_min(1e-30)
    q = torch.quantile(cv, torch.tensor([0.01, 0.05], device=C.device))
    per_row_best = ratio.min(-1).values
    return {
        "n_pairs": int(cv.numel()),
        "cos_mean": float(cv.mean()),
        "cos_min": float(cv.min()),
        "cos_p01": float(q[0]),
        "cos_p05": float(q[1]),
        "cos_frac_lt_m0.5": float((cv < -0.5).float().mean()),
        "cos_frac_lt_m0.7": float((cv < -0.7).float().mean()),
        "cos_frac_lt_m0.9": float((cv < -0.9).float().mean()),
        "best_pair_cancel_ratio_mean": float(per_row_best.mean()),
        "best_pair_cancel_ratio_min": float(ratio.min()),
        "pair_cancel_ratio_p01": float(
            torch.quantile(ratio.reshape(-1), torch.tensor(0.01, device=C.device))),
    }


def err_metrics(J, y2, gsum):
    q = torch.quantile(J, torch.tensor([0.90, 0.99], device=J.device))
    return {
        "mean_err": float(J.mean()),
        "median_err": float(J.median()),
        "p90_err": float(q[0]),
        "p99_err": float(q[1]),
        "mean_R_output": float((J / (y2 + EPS)).mean()),
        "mean_R_group": float((J / (gsum + EPS)).mean()),
    }


def run_diagnostic(args, model, tokens, device):
    layers = parse_list(args.layers, int)
    skip_groups = parse_list(args.skip_groups, int)
    positions = list(range(0, len(tokens),
                           max(1, len(tokens) // max(1, args.stats_positions))))
    positions = positions[:args.stats_positions]
    captured = capture_partials(model, tokens, args.k_group, args.tile_width, layers,
                                positions, device)
    pair_stats = {}
    j_by = {}
    per_layer = {}
    for L, C in captured.items():
        C = C.to(device).float()
        y2 = (C.sum(1).pow(2)).sum(-1)
        gsum = norms2(C).sum(-1)
        pair_stats[str(L)] = pair_statistics(C)
        layer_res = {}
        for k in skip_groups:
            entry = {}
            base = None
            for name in METHODS:
                idx = select(name, C, k, args.seed_count, args.max_swap_iters)
                J = objective(C, idx)
                j_by.setdefault(str(k), {}).setdefault(name, []).append(J.cpu())
                m = err_metrics(J, y2, gsum)
                if name == "small_l2":
                    base = m["mean_err"]
                entry[name] = m
            for name in METHODS:
                entry[name]["improvement_vs_small_l2"] = (
                    base / entry[name]["mean_err"] if entry[name]["mean_err"] > 0 else float("inf"))
            layer_res[str(k)] = entry
        per_layer[str(L)] = layer_res
        print(f"[diag] layer {L} done", flush=True)

    aggregate = {}
    for k in skip_groups:
        entry = {}
        base = None
        for name in METHODS:
            J = torch.cat(j_by[str(k)][name])
            q = torch.quantile(J, torch.tensor([0.90, 0.99], device=J.device))
            agg = {"mean_err": float(J.mean()),
                   "median_err": float(J.median()),
                   "p90_err": float(q[0]),
                   "p99_err": float(q[1])}
            if name == "small_l2":
                base = agg["mean_err"]
            entry[name] = agg
        for name in METHODS:
            entry[name]["improvement_vs_small_l2"] = (
                base / entry[name]["mean_err"] if entry[name]["mean_err"] > 0 else float("inf"))
        aggregate[str(k)] = entry

    out = {
        "meta": {
            "model_dir": str(args.model_dir),
            "corpus": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "corpus_tokens": len(tokens),
            "k_group": args.k_group,
            "tile_width": args.tile_width,
            "layers": layers,
            "positions": positions,
            "skip_groups": skip_groups,
            "actual_skip_fractions": {str(k): k / 68.0 for k in skip_groups},
            "n_groups": 68,
            "seed_count": args.seed_count,
            "max_swap_iters": args.max_swap_iters,
            "git": git_info(),
        },
        "aggregate": aggregate,
        "per_layer": per_layer,
    }
    return out, pair_stats


def run_ppl(args, model, tokens, device, out_dir):
    skip_counts = parse_list(args.skip_groups, int)
    methods = parse_list(args.methods, str) if args.methods else list(METHODS)
    result = {
        "meta": {
            "model_dir": str(args.model_dir),
            "corpus": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "corpus_tokens": len(tokens),
            "k_group": args.k_group,
            "tile_width": args.tile_width,
            "methods": methods,
            "skip_groups": skip_counts,
            "max_tokens": args.max_tokens,
            "chunk_tokens": args.chunk_tokens,
            "seed_count": args.seed_count,
            "max_swap_iters": args.max_swap_iters,
            "gpus": [int(g) for g in args.gpu_ids],
            "git": git_info(),
        },
        "environment": environment_info(args.gpu_ids),
        "baseline": None,
        "parity": None,
        "runs": [],
    }
    t = time.time()
    base = perplexity(model, tokens, device, args.max_tokens)
    base["secs"] = time.time() - t
    result["baseline"] = base
    print(f"[ppl] baseline ppl={base['perplexity']:.6f} positions={base['positions']}", flush=True)
    base_ppl = base["perplexity"]

    for method in methods:
        for skip in skip_counts:
            if skip == 0 and method != methods[0]:
                continue
            oracle = KSubsetOracleDown(model, args.k_group, args.tile_width, method, skip,
                                       chunk_tokens=args.chunk_tokens,
                                       seed_count=args.seed_count,
                                       max_swap_iters=args.max_swap_iters).attach()
            t = time.time()
            try:
                res = perplexity(model, tokens, device, args.max_tokens)
            finally:
                oracle.detach()
            res["secs"] = time.time() - t
            d = (res["perplexity"] / base_ppl - 1.0) * 100.0
            rec = {
                "method": method,
                "skip_group_count": skip,
                "actual_skip_fraction": oracle.actual_skip_ratio(),
                "target_skip_fraction": skip / 68.0,
                "perplexity": res["perplexity"],
                "mean_nll": res["mean_nll"],
                "delta_ppl_percent": d,
                "positions": res["positions"],
                "finite": res["finite"],
                "secs": res["secs"],
            }
            result["runs"].append(rec)
            print(f"[ppl] method={method} skip={skip}/68 actual={rec['actual_skip_fraction']:.5f} "
                  f"ppl={res['perplexity']:.6f} d={d:+.3f}% finite={res['finite']} "
                  f"({res['secs']:.1f}s)", flush=True)
            if skip == 0:
                result["parity"] = {
                    "decomposed_ppl": res["perplexity"],
                    "delta_ppl_percent": d,
                    "passed": abs(d) <= 0.1,
                    "mean_nll": res["mean_nll"],
                    "positions": res["positions"],
                }
                if abs(d) > 0.1 and not args.allow_parity_fail:
                    print(f"[ppl] PARITY FAIL {d:.4f}% > 0.1%; stopping", flush=True)
                    result["parity_failed_stop"] = True
                    break
        if result.get("parity_failed_stop"):
            break

    with open(out_dir / "oracle_n64.json", "w") as f:
        json.dump(result, f, indent=1)
    with open(out_dir / "environment.json", "w") as f:
        json.dump({"environment": environment_info(args.gpu_ids), "git": git_info()}, f, indent=1)
    with open(out_dir / "baseline.json", "w") as f:
        json.dump({"baseline": result["baseline"], "parity": result["parity"],
                   "meta": result["meta"]}, f, indent=1)
    print(f"[ppl] wrote {out_dir / 'oracle_n64.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["diagnostic", "ppl"], required=True)
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=64)
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--layers", default="0,16,28,32,48,63")
    ap.add_argument("--stats-positions", type=int, default=64)
    ap.add_argument("--skip-groups", default="7,10,14,17,20,27")
    ap.add_argument("--methods", default="")
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--max-swap-iters", type=int, default=8)
    ap.add_argument("--allow-parity-fail", action="store_true")
    ap.add_argument("--out-dir", default="artifacts/ffn_k_tile_cancel_oracle")
    args = ap.parse_args()

    if args.gpus:
        gpu_ids = tuple(int(x) for x in args.gpus.split(","))
    else:
        gpu_ids = tuple(range(min(3, torch.cuda.device_count())))
    args.gpu_ids = gpu_ids
    device = f"cuda:{gpu_ids[0]}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = read_corpus(args.corpus)[:args.max_tokens] if args.max_tokens else read_corpus(args.corpus)
    t0 = time.time()
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    print(f"[cancel] loaded in {time.time()-t0:.1f}s tokens={len(tokens)}", flush=True)

    if args.stage == "diagnostic":
        diag, pair_stats = run_diagnostic(args, model, tokens, device)
        with open(out_dir / "diagnostic.json", "w") as f:
            json.dump(diag, f, indent=1)
        with open(out_dir / "pair_stats.json", "w") as f:
            json.dump(pair_stats, f, indent=1)
        with open(out_dir / "environment.json", "w") as f:
            json.dump({"environment": environment_info(gpu_ids), "git": git_info()}, f, indent=1)
        print(f"[cancel] wrote {out_dir}/diagnostic.json", flush=True)
    else:
        run_ppl(args, model, tokens, device, out_dir)


if __name__ == "__main__":
    main()
