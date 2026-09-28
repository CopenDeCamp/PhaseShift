import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_down import (actual_skip_fraction, apply_mask, group_count,  # noqa: E402
                         group_partials, group_scores, keep_count_for_fraction,
                         oracle_down_chunk, read_corpus, select_keep_mask,
                         skip_count_for_fraction, tile_count)
from stream_load import stream_load  # noqa: E402


def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x != ""]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", here] + list(a),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"revision": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_short": run("status", "--short")}


def environment_info(gpu_ids):
    import transformers
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    arch = ""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        pass
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "gcn_arch": arch,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu_count_visible": n,
        "gpu_names": names,
        "gpus_used": [int(g) for g in gpu_ids],
    }


def mlp_list(model):
    out = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp"):
            out.append((int(name.split(".")[-2]), mod))
    out.sort(key=lambda x: x[0])
    return out


class KTileOracleDown:
    def __init__(self, model, k_group, tile_width, skip_frac, chunk_tokens=64):
        self.model = model
        self.k_group = k_group
        self.tile_width = tile_width
        self.skip_frac = skip_frac
        self.chunk_tokens = chunk_tokens
        self._orig = {}
        self.kept = 0
        self.total = 0

    def _make_forward(self, mlp):
        w = mlp.down_proj.weight
        kg = self.k_group
        tw = self.tile_width
        ng = group_count(w.shape[1], kg)
        keep = keep_count_for_fraction(ng, self.skip_frac)
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
                y, _, mask = oracle_down_chunk(hf[start:end], w, kg, tw, keep)
                parts.append(y)
                state.kept += int(mask.sum().item())
                state.total += int(mask.numel())
            y = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            return y.to(h.dtype).reshape(*shp[:-1], w.shape[0])

        return forward

    def attach(self):
        for layer, mlp in mlp_list(self.model):
            self._orig[id(mlp)] = (mlp, mlp.forward)
            mlp.forward = self._make_forward(mlp)
        return self

    def detach(self):
        for mlp, orig in self._orig.values():
            mlp.forward = orig
        self._orig.clear()

    def actual_keep_ratio(self):
        if self.total == 0:
            return 1.0
        return self.kept / self.total


def perplexity(model, tokens, device, max_tokens, pos_chunk=256):
    ids = torch.tensor(tokens[:max_tokens], dtype=torch.long, device=device)
    S = ids.numel()
    total_nll = 0.0
    total = 0
    finite = True
    with torch.no_grad():
        x = ids.unsqueeze(0)
        out = model(input_ids=x, use_cache=False)
        logits = out.logits[0, :-1, :]
        tgt = ids[1:]
        S1 = logits.shape[0]
        for c0 in range(0, S1, pos_chunk):
            c1 = min(c0 + pos_chunk, S1)
            lg = logits[c0:c1].float()
            if not bool(torch.isfinite(lg).all()):
                finite = False
            tt = tgt[c0:c1]
            nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tt[:, None]).squeeze(1)
            total_nll += float(nll.sum().item())
            total += int(nll.numel())
            del lg, nll
        del out, logits
    torch.cuda.empty_cache()
    mean = total_nll / max(total, 1)
    ppl = math.exp(mean)
    if not math.isfinite(ppl):
        finite = False
    return {"mean_nll": mean, "perplexity": ppl, "positions": total, "finite": finite}


def capture_tile_scores(model, tokens, k_group, tile_width, positions, device):
    mlps = dict(mlp_list(model))
    out = {}
    orig = {}
    for layer, mlp in mlps.items():
        orig[layer] = mlp.forward

        def make(m):
            def fwd(x):
                gate = m.gate_proj(x)
                up = m.up_proj(x)
                h = m.act_fn(gate) * up
                hf = h.reshape(-1, h.shape[-1])
                pos = torch.tensor(positions, dtype=torch.long, device=hf.device)
                hsel = hf.index_select(0, pos).float()
                c = group_partials(hsel, m.down_proj.weight, k_group)
                out[id(m)] = group_scores(c, tile_width).cpu()
                return m.down_proj(h)
            return fwd

        mlp.forward = make(mlp)

    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        for layer, fwd in orig.items():
            mlps[layer].forward = fwd
    scores = {}
    for layer, mlp in mlps.items():
        scores[layer] = out[id(mlp)]
    torch.cuda.empty_cache()
    return scores


def layer_stats(score, skip_profile):
    ng = score.shape[-1]
    s = score.reshape(-1, ng).float()
    n_rows = s.shape[0]
    total = s.sum(-1, keepdim=True).clamp_min(1e-30)
    frac = s / total
    sd, _ = torch.sort(frac, dim=-1, descending=True)
    top1 = float(sd[:, 0].mean())
    top4 = float(sd[:, :min(4, ng)].sum(-1).mean())
    top8 = float(sd[:, :min(8, ng)].sum(-1).mean())

    def bottom(p):
        k = int(round(p * ng))
        return float(sd[:, ng - k:].sum(-1).mean()) if k > 0 else 0.0

    flat = s.reshape(-1)
    q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99]))
    retained = {}
    for f in skip_profile:
        kc = keep_count_for_fraction(ng, f)
        retained[f"{f:.2f}"] = float(sd[:, :kc].sum(-1).mean())
    return {
        "n_rows": n_rows,
        "n_groups": ng,
        "score_mean": float(flat.mean()),
        "score_p50": float(q[0]),
        "score_p90": float(q[1]),
        "score_p99": float(q[2]),
        "top1_frac": top1,
        "top4_frac": top4,
        "top8_frac": top8,
        "bottom10_frac": bottom(0.10),
        "bottom20_frac": bottom(0.20),
        "bottom30_frac": bottom(0.30),
        "bottom40_frac": bottom(0.40),
        "retained_proxy": retained,
    }


def profile_summary(stats_by_layer, ref_skip):
    key = f"{ref_skip:.2f}"
    rows = [(L, st["retained_proxy"][key]) for L, st in stats_by_layer.items()]
    rows.sort(key=lambda x: x[1])
    worst = rows[0]
    best = rows[-1]
    med = rows[len(rows) // 2]
    return {
        "ref_skip": ref_skip,
        "best_layer": int(best[0]), "best_retained": best[1],
        "median_layer": int(med[0]), "median_retained": med[1],
        "worst_layer": int(worst[0]), "worst_retained": worst[1],
        "all": [{"layer": int(L), "retained": v} for L, v in rows],
    }


def resolve_gpus(arg):
    if arg:
        return tuple(int(x) for x in arg.split(","))
    n = torch.cuda.device_count()
    return tuple(range(min(3, n)))


def load_model(model_dir, gpu_ids, dtype=torch.bfloat16, mem_gib=22):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_dir)
    model, info = stream_load(model_dir, AutoModelForCausalLM, cfg, dtype=dtype,
                              gpu_ids=gpu_ids, mem_gib=mem_gib, verbose=True)
    return model, cfg, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--k-group", type=int, default=256)
    ap.add_argument("--tile-width", type=int, default=64)
    ap.add_argument("--skip-fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--score", default="l2")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=64)
    ap.add_argument("--stats-positions", type=int, default=64)
    ap.add_argument("--profile-skip", default="0.2,0.3,0.4")
    ap.add_argument("--allow-parity-fail", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", default="artifacts/ffn_k_tile_oracle/oracle.json")
    args = ap.parse_args()

    if args.score != "l2":
        raise SystemExit("only --score l2 is supported (primary score)")
    gpus = resolve_gpus(args.gpus)
    device = f"cuda:{gpus[0]}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    git = git_info()
    env = environment_info(gpus)
    tokens = read_corpus(args.corpus)
    corpus_sha = sha256_file(args.corpus)
    if args.max_tokens:
        tokens = tokens[:args.max_tokens]
    profile_skip = parse_list(args.profile_skip, float)
    skip_fracs = parse_list(args.skip_fracs, float)

    t0 = time.time()
    model, cfg, info = load_model(args.model_dir, gpus)
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = int(getattr(text_cfg, "num_hidden_layers"))
    intermediate = int(getattr(text_cfg, "intermediate_size"))
    out_features = int(getattr(text_cfg, "hidden_size"))
    ng = group_count(intermediate, args.k_group)
    nt = tile_count(out_features, args.tile_width)
    print(f"[oracle] loaded in {time.time()-t0:.1f}s layers={n_layers} "
          f"K={intermediate} N={out_features} ng={ng} ntiles={nt}", flush=True)

    result = {
        "meta": {
            "model_dir": str(args.model_dir),
            "corpus": str(args.corpus),
            "corpus_sha256": corpus_sha,
            "corpus_tokens": len(tokens),
            "k_group": args.k_group,
            "tile_width": args.tile_width,
            "score": args.score,
            "max_tokens": args.max_tokens,
            "chunk_tokens": args.chunk_tokens,
            "n_layers": n_layers,
            "intermediate": intermediate,
            "out_features": out_features,
            "n_groups": ng,
            "n_tiles": nt,
            "skip_fracs": skip_fracs,
            "stats_positions": args.stats_positions,
            "gpus": [int(g) for g in gpus],
            "git": git,
        },
        "environment": env,
        "loader": info,
        "baseline": None,
        "parity": None,
        "runs": [],
        "contribution": None,
        "layer_profile": None,
    }

    torch.cuda.reset_peak_memory_stats()
    if not args.skip_baseline:
        t = time.time()
        base = perplexity(model, tokens, device, args.max_tokens)
        base["secs"] = time.time() - t
        result["baseline"] = base
        print(f"[oracle] baseline ppl={base['perplexity']:.6f} nll={base['mean_nll']:.6f} "
              f"positions={base['positions']} finite={base['finite']}", flush=True)
    else:
        result["baseline"] = None

    base_ppl = result["baseline"]["perplexity"] if result["baseline"] else None

    for i, frac in enumerate(skip_fracs):
        skip_count = skip_count_for_fraction(ng, frac)
        keep_count = ng - skip_count
        oracle = KTileOracleDown(model, args.k_group, args.tile_width, frac,
                                 chunk_tokens=args.chunk_tokens).attach()
        t = time.time()
        try:
            res = perplexity(model, tokens, device, args.max_tokens)
        finally:
            oracle.detach()
        res["secs"] = time.time() - t
        d = None
        if base_ppl:
            d = (res["perplexity"] / base_ppl - 1.0) * 100.0
        rec = {
            "target_skip_fraction": frac,
            "skip_group_count": skip_count,
            "keep_group_count": keep_count,
            "actual_skip_fraction": actual_skip_fraction(ng, skip_count),
            "actual_keep_ratio": oracle.actual_keep_ratio(),
            "perplexity": res["perplexity"],
            "mean_nll": res["mean_nll"],
            "delta_ppl_percent": d,
            "positions": res["positions"],
            "finite": res["finite"],
            "secs": res["secs"],
        }
        result["runs"].append(rec)
        print(f"[oracle] target_skip={frac:.3f} groups={skip_count}/{ng} "
              f"actual={rec['actual_skip_fraction']:.5f} ppl={res['perplexity']:.6f} "
              f"d={d if d is None else round(d, 4)}% finite={res['finite']} "
              f"({res['secs']:.1f}s)", flush=True)
        if frac == 0.0:
            result["parity"] = {
                "decomposed_ppl": res["perplexity"],
                "delta_ppl_percent": d,
                "passed": bool(d is not None and abs(d) <= 0.1),
                "mean_nll": res["mean_nll"],
                "positions": res["positions"],
            }
            if d is not None and abs(d) > 0.1 and not args.allow_parity_fail:
                print(f"[oracle] PARITY FAIL delta={d:.4f}% > 0.1%; stopping sweep", flush=True)
                result["parity_failed_stop"] = True
                break

    positions = list(range(0, len(tokens),
                           max(1, len(tokens) // max(1, args.stats_positions))))
    if len(positions) > args.stats_positions:
        positions = positions[:args.stats_positions]
    t = time.time()
    score_by_layer = capture_tile_scores(model, tokens, args.k_group, args.tile_width,
                                         positions, device)
    stats_by_layer = {L: layer_stats(sc, profile_skip) for L, sc in score_by_layer.items()}
    result["contribution"] = {
        "positions": positions,
        "layers": {str(L): st for L, st in stats_by_layer.items()},
    }
    result["layer_profile"] = {
        "reference_skip": profile_skip,
        "summaries": {f"{f:.2f}": profile_summary(stats_by_layer, f) for f in profile_skip},
    }
    print(f"[oracle] contribution stats in {time.time()-t:.1f}s", flush=True)

    peaks = {}
    for g in gpus:
        try:
            peaks[str(g)] = torch.cuda.max_memory_allocated(int(g))
        except Exception:  # noqa: BLE001
            pass
    result["gpu_peak_bytes"] = peaks
    result["elapsed_secs"] = time.time() - t0

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    env_path = out_path.parent / "environment.json"
    with open(env_path, "w") as f:
        json.dump({"environment": env, "git": git}, f, indent=1)
    if result["baseline"]:
        base_path = out_path.parent / "baseline.json"
        with open(base_path, "w") as f:
            json.dump({"baseline": result["baseline"], "parity": result["parity"],
                       "meta": result["meta"]}, f, indent=1)
    print("[oracle] wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
