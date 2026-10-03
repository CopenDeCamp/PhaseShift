import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ffn_k_tile_oracle"))

from run_oracle import load_model, mlp_list  # noqa: E402
from prefilter import weight_norm2  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--mem-gib", type=int, default=27)
    ap.add_argument("--out-dir", default="artifacts/ffn_k256_candidate_selector")
    args = ap.parse_args()

    gpu_ids = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else \
        tuple(range(min(3, torch.cuda.device_count())))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hip": getattr(torch.version, "hip", None),
        "gcn_arch": torch.cuda.get_device_properties(gpu_ids[0]).gcnArchName,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": f"physical GPU {gpu_ids} (other workload on GPU0/1)",
        "model_dir": args.model_dir,
        "corpus": args.corpus,
        "corpus_sha256": sha256_file(args.corpus),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=1))

    model, _, _ = load_model(args.model_dir, gpu_ids, mem_gib=args.mem_gib)
    layers = []
    for layer, mlp in mlp_list(model):
        n = weight_norm2(mlp.down_proj.weight, args.k_group, args.tile_width).float().cpu()
        flat = n.reshape(-1)
        layers.append({
            "layer": layer,
            "mean": float(flat.mean()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "p10": float(flat.quantile(0.10)),
            "p50": float(flat.quantile(0.5)),
            "p90": float(flat.quantile(0.90)),
        })
        print(f"[meta] layer {layer} done", flush=True)

    meta = {
        "name": "down_proj per-(N64 tile, K256 group) weight Frobenius norm squared",
        "shape": [len(layers), 80, 68],
        "dtype_recommended": "float16 or float32",
        "k_group": args.k_group,
        "tile_width": args.tile_width,
        "deterministic": True,
        "generation": "n[t,g] = sum over N64 tile rows and K256 group cols of W^2",
        "layer_stats": layers,
    }
    (out / "prefilter_metadata.json").write_text(json.dumps(meta, indent=1))
    print("[meta] wrote", out / "prefilter_metadata.json", flush=True)


if __name__ == "__main__":
    main()
