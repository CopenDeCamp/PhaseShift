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

from oracle_down import group_partials, read_corpus  # noqa: E402
from run_oracle import load_model, mlp_list, perplexity  # noqa: E402
from subset_select import apply_skip_indices, partials_to_rows  # noqa: E402
from prefilter import act_energy, make_pool, pool_scores, weight_norm2  # noqa: E402
from pool_sketch import pool_sketch_select  # noqa: E402
from projection import make_R_all  # noqa: E402


class PoolSketchPPL:
    def __init__(self, model, k_group, tile_width, kind, C, dim, method, skip,
                 seed=0, seed_count=4, chunk_tokens=64):
        self.model = model
        self.k_group = k_group
        self.tile_width = tile_width
        self.kind = kind
        self.C = C
        self.dim = dim
        self.method = method
        self.skip = skip
        self.seed = seed
        self.seed_count = seed_count
        self.chunk_tokens = chunk_tokens
        self._orig = {}
        self.skipped = 0
        self.total = 0

    def _forward_factory(self, layer, mlp):
        w = mlp.down_proj.weight
        kg = self.k_group
        tw = self.tile_width
        kind = self.kind
        C = self.C
        d = self.dim
        method = self.method
        skip = self.skip
        chunk = self.chunk_tokens
        state = self
        nt = w.shape[0] // tw
        nn = weight_norm2(w.detach().to("cpu"), kg, tw).to(w.device).float() \
            if kind == "act_wnorm" else None
        Rall = make_R_all(layer, nt, d, self.seed).to(w.device)

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
                hc = hf[start:end]
                c = group_partials(hc, w, kg)
                rows = partials_to_rows(c, tw).contiguous()
                a = act_energy(hc, kg)
                scores = pool_scores(kind, rows, a, nn)
                pool = make_pool(scores, C)
                idx = pool_sketch_select(rows, pool, Rall, d, method, skip,
                                         state.seed_count)
                parts.append(apply_skip_indices(c, idx, tw))
                state.skipped += int(idx.numel())
                state.total += int(idx.shape[0] * c.shape[0])
            y = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            return y.to(h.dtype).reshape(*shp[:-1], w.shape[0])

        return forward

    def attach(self):
        for layer, mlp in mlp_list(self.model):
            self._orig[id(mlp)] = (mlp, mlp.forward)
            mlp.forward = self._forward_factory(layer, mlp)
        return self

    def detach(self):
        for mlp, orig in self._orig.values():
            mlp.forward = orig
        self._orig.clear()

    def actual_skip_ratio(self):
        return self.skipped / self.total if self.total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--configs", default="")
    ap.add_argument("--skips", default="16,24,27,28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=64)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_candidate_selector/sketch_pool_ppl.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    configs = []
    for tok in args.configs.split(","):
        if not tok:
            continue
        parts = tok.split(":")
        configs.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    skips = [int(x) for x in args.skips.split(",")]

    tokens = read_corpus(args.corpus)[:args.max_tokens]
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    t = time.time()
    base = perplexity(model, tokens, device, args.max_tokens)
    print(f"[ppl] baseline ppl={base['perplexity']:.6f} positions={base['positions']} "
          f"({time.time()-t:.1f}s)", flush=True)

    result = {"baseline": base, "runs": [], "configs": args.configs, "skips": skips,
              "seed": args.seed, "max_tokens": args.max_tokens}
    for kind, C, dim, method in configs:
        for skip in skips:
            obj = PoolSketchPPL(model, 256, 64, kind, C, dim, method, skip, args.seed,
                                args.seed_count, args.chunk_tokens).attach()
            t = time.time()
            try:
                res = perplexity(model, tokens, device, args.max_tokens)
            finally:
                obj.detach()
            torch.cuda.empty_cache()
            dd = (res["perplexity"] / base["perplexity"] - 1.0) * 100.0
            rec = {"prefilter": kind, "C": C, "dim": dim, "method": method, "skip": skip,
                   "actual_skip": obj.actual_skip_ratio(), "perplexity": res["perplexity"],
                   "delta_ppl_percent": dd, "positions": res["positions"],
                   "finite": res["finite"], "secs": time.time() - t}
            result["runs"].append(rec)
            print(f"[ppl] {kind:10s} C={C:2d} d={dim:2d} {method:8s} skip={skip:2d} "
                  f"({skip/68*100:5.2f}%) ppl={res['perplexity']:.6f} d={dd:+.3f}% "
                  f"({rec['secs']:.1f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("[ppl] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
