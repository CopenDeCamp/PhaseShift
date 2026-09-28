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
from sketch_select import choose  # noqa: E402
from projection import apply_R, make_R_all  # noqa: E402


class SketchPPL:
    def __init__(self, model, k_group, tile_width, dim, method, skip, seed=0,
                 seed_count=4, chunk_tokens=64):
        self.model = model
        self.k_group = k_group
        self.tile_width = tile_width
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
        dim = self.dim
        method = self.method
        skip = self.skip
        seed = self.seed
        seed_count = self.seed_count
        chunk = self.chunk_tokens
        state = self
        Rall = make_R_all(layer, w.shape[0] // tw, max(dim, 1), seed).to(w.device)

        def forward(x):
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            h = mlp.act_fn(gate) * up
            shp = h.shape
            hf = h.reshape(-1, shp[-1])
            T = hf.shape[0]
            nt = w.shape[0] // tw
            parts = []
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                c = group_partials(hf[start:end], w, kg)
                if method in ("small_l2", "exact_residual", "exact_pair"):
                    idx = choose(method, partials_to_rows(c, tw).contiguous(), None, skip,
                                 seed_count)
                else:
                    c4 = c.reshape(c.shape[0], end - start, nt, tw).permute(1, 2, 0, 3)
                    z = apply_R(c4, Rall, dim).reshape(-1, c.shape[0], dim).contiguous()
                    idx = choose(method, None, z, skip, seed_count)
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
    ap.add_argument("--configs", default="exact_pair:0,small_l2:0,sk_residual:4,sk_pair4:4,"
                                        "sk_residual:8,sk_pair4:8,sk_pair4:16")
    ap.add_argument("--skips", default="16,24,27,28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=64)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out", default="artifacts/ffn_k256_sketch/ppl_float_sketch.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    configs = []
    for tok in args.configs.split(","):
        if not tok:
            continue
        name, d = tok.split(":")
        configs.append((name, int(d)))
    skips = [int(x) for x in args.skips.split(",")]

    tokens = read_corpus(args.corpus)[:args.max_tokens]
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    t = time.time()
    base = perplexity(model, tokens, device, args.max_tokens)
    print(f"[ppl] baseline ppl={base['perplexity']:.6f} pos={base['positions']} "
          f"({time.time()-t:.1f}s)", flush=True)

    result = {"baseline": base, "runs": [], "configs": args.configs, "skips": skips,
              "seed": args.seed, "max_tokens": args.max_tokens}
    for name, d in configs:
        for skip in skips:
            if skip == 0 and (name, d) != configs[0]:
                continue
            obj = SketchPPL(model, 256, 64, d, name, skip, args.seed, args.seed_count,
                            args.chunk_tokens).attach()
            t = time.time()
            try:
                res = perplexity(model, tokens, device, args.max_tokens)
            finally:
                obj.detach()
            dd = (res["perplexity"] / base["perplexity"] - 1.0) * 100.0
            rec = {"method": name, "dim": d, "skip": skip, "actual_skip": obj.actual_skip_ratio(),
                   "perplexity": res["perplexity"], "delta_ppl_percent": dd,
                   "positions": res["positions"], "finite": res["finite"], "secs": time.time() - t}
            result["runs"].append(rec)
            print(f"[ppl] {name:16s} d={d:2d} skip={skip:2d} ({skip/68*100:5.2f}%) "
                  f"ppl={res['perplexity']:.6f} d={dd:+.3f}% ({rec['secs']:.1f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("[ppl] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
