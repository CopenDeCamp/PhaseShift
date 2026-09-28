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
from prefilter import (act_energy, make_pool, pool_scores, restricted_pair,  # noqa: E402
                       unrestricted_pair, weight_norm2)


class PoolPPL:
    def __init__(self, model, k_group, tile_width, kind, C, skip, seed_count=16,
                 swap_iters=8, chunk_tokens=64):
        self.model = model
        self.k_group = k_group
        self.tile_width = tile_width
        self.kind = kind
        self.C = C
        self.skip = skip
        self.seed_count = seed_count
        self.swap_iters = swap_iters
        self.chunk_tokens = chunk_tokens
        self._orig = {}
        self.skipped = 0
        self.total = 0

    def _forward_factory(self, mlp):
        w = mlp.down_proj.weight
        kg = self.k_group
        tw = self.tile_width
        kind = self.kind
        C = self.C
        skip = self.skip
        chunk = self.chunk_tokens
        state = self
        nn = weight_norm2(w.detach().to("cpu"), kg, tw).to(w.device).float() \
            if kind == "act_wnorm" else None

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
                if kind == "unrestricted" or C >= 68:
                    idx = unrestricted_pair(rows, skip, state.seed_count, state.swap_iters)
                else:
                    a = act_energy(hc, kg)
                    scores = pool_scores(kind, rows, a, nn)
                    pool = make_pool(scores, C)
                    idx = restricted_pair(rows, pool, skip, state.seed_count, state.swap_iters)
                parts.append(apply_skip_indices(c, idx, tw))
                state.skipped += int(idx.numel())
                state.total += int(idx.shape[0] * c.shape[0])
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
        return self.skipped / self.total if self.total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--configs", default="unrestricted:68,exact_l2:32,exact_l2:40,exact_l2:48,"
                                        "act_wnorm:32,act_wnorm:40,act_wnorm:48,"
                                        "act_energy:40,act_energy:48")
    ap.add_argument("--skips", default="16,24,27,28")
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--swap-iters", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=64)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_candidate_selector/pool_ppl.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    configs = []
    for tok in args.configs.split(","):
        if not tok:
            continue
        name, c = tok.split(":")
        configs.append((name, int(c)))
    skips = [int(x) for x in args.skips.split(",")]

    tokens = read_corpus(args.corpus)[:args.max_tokens]
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    t = time.time()
    base = perplexity(model, tokens, device, args.max_tokens)
    print(f"[ppl] baseline ppl={base['perplexity']:.6f} positions={base['positions']} "
          f"({time.time()-t:.1f}s)", flush=True)

    result = {"baseline": base, "runs": [], "configs": args.configs, "skips": skips,
              "seed_count": args.seed_count, "max_tokens": args.max_tokens}
    for name, C in configs:
        for skip in skips:
            obj = PoolPPL(model, 256, 64, name, C, skip, args.seed_count,
                          args.swap_iters, args.chunk_tokens).attach()
            t = time.time()
            try:
                res = perplexity(model, tokens, device, args.max_tokens)
            finally:
                obj.detach()
            torch.cuda.empty_cache()
            dd = (res["perplexity"] / base["perplexity"] - 1.0) * 100.0
            rec = {"prefilter": name, "C": C, "skip": skip,
                   "actual_skip": obj.actual_skip_ratio(),
                   "perplexity": res["perplexity"], "delta_ppl_percent": dd,
                   "positions": res["positions"], "finite": res["finite"],
                   "secs": time.time() - t}
            result["runs"].append(rec)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
            print(f"[ppl] {name:12s} C={C:2d} skip={skip:2d} ({skip/68*100:5.2f}%) "
                  f"ppl={res['perplexity']:.6f} d={dd:+.3f}% ({rec['secs']:.1f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("[ppl] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
