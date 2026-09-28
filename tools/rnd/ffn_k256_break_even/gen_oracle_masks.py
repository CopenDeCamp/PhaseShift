import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ffn_k_tile_oracle"))

from oracle_down import read_corpus  # noqa: E402
from run_cancel import capture_partials  # noqa: E402
from run_oracle import load_model  # noqa: E402
from subset_select import pair_greedy_swap, small_l2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--layers", default="0,16,28,32,48,63")
    ap.add_argument("--token-index", type=int, default=0)
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--max-swap-iters", type=int, default=8)
    ap.add_argument("--skips", default="0,7,10,14,16,17,20,27,28,30")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out-dir", default="artifacts/ffn_k256_break_even/oracle_masks")
    args = ap.parse_args()

    if args.gpus:
        gpu_ids = tuple(int(x) for x in args.gpus.split(","))
    else:
        gpu_ids = tuple(range(min(3, torch.cuda.device_count())))
    device = f"cuda:{gpu_ids[0]}"
    layers = [int(x) for x in args.layers.split(",")]
    skips = [int(x) for x in args.skips.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = read_corpus(args.corpus)
    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    partials = capture_partials(model, tokens, args.k_group, args.tile_width, layers,
                                [args.token_index], device)
    meta = {"model_dir": args.model_dir, "corpus": args.corpus,
            "token_index": args.token_index, "layers": layers, "skips": skips,
            "seed_count": args.seed_count, "max_swap_iters": args.max_swap_iters}
    for L in layers:
        C = partials[L].to(device).float()
        n_rows = C.shape[0]
        layer_dir = out_dir / f"L{L}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for k in skips:
            if k == 0:
                idx = torch.zeros(n_rows, 0, dtype=torch.long, device=device)
            else:
                idx = pair_greedy_swap(C, k, args.seed_count, args.max_swap_iters,
                                       extra_seed=small_l2(C, 2))
            idx = idx.sort(dim=-1).values.cpu()
            with open(layer_dir / f"oracle_mask_k{k}.txt", "w") as f:
                for r in range(n_rows):
                    f.write(" ".join(str(int(g)) for g in idx[r].tolist()) + "\n")
        print(f"[masks] layer {L} wrote {n_rows} tiles x {len(skips)} skips", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
