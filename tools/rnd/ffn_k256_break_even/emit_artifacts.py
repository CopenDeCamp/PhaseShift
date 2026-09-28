import argparse
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

import transformers


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_rev(root):
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return ""


def manifest_layers(model_dir):
    d = json.loads((Path(model_dir) / "phaseshift_quantization.json").read_text())
    out = {}
    for k, v in d["tensors"].items():
        m = re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$", k)
        if m:
            out[int(m.group(1))] = v.get("encoding")
    return out


def flat(results, field, pattern):
    for r in results:
        if r["pattern"] == pattern:
            yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k256_break_even")
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B-PSQ")
    ap.add_argument("--corpus", default="artifacts/ffn_prune/corpus.psktok")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    d = Path(args.dir)
    p4 = json.loads((d / "dense_psq4.json").read_text())
    p8 = json.loads((d / "dense_psq8.json").read_text())
    layers = manifest_layers(args.model_dir)
    counts = dict(Counter(layers.values()))

    (d / "current_manifest.json").write_text(json.dumps(
        {"counts": counts, "layers": {str(k): v for k, v in sorted(layers.items())},
         "model_dir": args.model_dir}, indent=1))

    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hip": getattr(torch.version, "hip", None),
        "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rocm_bin_rocprofv3": "1.3.5",
        "device_used": "GPU2 (logical 0); GPU0/1 reserved for other workload",
        "git_revision": git_rev(args.root),
        "corpus": args.corpus,
        "corpus_sha256": sha256(args.corpus) if Path(args.corpus).exists() else "",
        "model_dir": args.model_dir,
    }
    (d / "environment.json").write_text(json.dumps(env, indent=1))

    def collect(field, out_name):
        out = {"psq4": {}, "psq8": {}}
        for kind, blob in (("psq4", p4), ("psq8", p8)):
            for pat in ("contiguous", "random", "oracle"):
                for r in flat(blob["results"], field, pat):
                    out[kind].setdefault(pat, {})[str(r["skip"])] = r[field]
        (d / out_name).write_text(json.dumps(out, indent=1))

    collect("dense", "dense.json")
    collect("ideal", "ideal_static_skip.json")
    collect("runtime", "runtime_mask_skip.json")

    replay = {}
    for f in ["replay_psq8_L32.json", "replay_psq8_L0.json", "replay_psq4_L63.json"]:
        if (d / f).exists():
            replay[f.replace(".json", "")] = json.loads((d / f).read_text())["results"]
    (d / "oracle_replay.json").write_text(json.dumps(replay, indent=1))

    prof = {"counters": {"available": False,
                         "note": "rocprofv3 1.3.5 collected FETCH_SIZE / SQ_INSTS_VALU but all "
                                 "values were 0; hardware PMC not available in this environment."}}
    bw = {}
    for kind, blob, dense_bytes in (("psq4", p4, p4["meta"]["working_set_mb"] / p4["meta"]["copies"] * 1e6),
                                    ("psq8", p8, p8["meta"]["working_set_mb"] / p8["meta"]["copies"] * 1e6)):
        dense_us = blob["results"][0]["dense"]["p50"]
        for r in blob["results"]:
            if r["pattern"] != "random":
                continue
            keep = (68 - r["skip"]) / 68.0
            rt = r["runtime"]["p50"]
            bw.setdefault(kind, {})[str(r["skip"])] = {
                "retained_bytes_mb": dense_bytes * keep / 1e6,
                "runtime_us": rt,
                "effective_gbs": dense_bytes * keep / (rt * 1e-6) / 1e9,
            }
        bw[kind]["dense_gbs"] = dense_bytes / (dense_us * 1e-6) / 1e9
    prof["bandwidth"] = bw
    (d / "profiling.json").write_text(json.dumps(prof, indent=1))

    be = json.loads((d / "break_even.json").read_text()) if (d / "break_even.json").exists() else None
    lines = ["# ffn_k256_break_even summary", ""]
    lines.append(f"counts: {counts}; dense psq4={p4['results'][0]['dense']['p50']:.2f}us "
                 f"psq8={p8['results'][0]['dense']['p50']:.2f}us")
    lines.append("")
    lines.append("| kind | pattern | skip | dense | ideal | runtime | efficiency |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for kind, blob in (("psq4", p4), ("psq8", p8)):
        dense = blob["results"][0]["dense"]["p50"]
        for r in blob["results"]:
            frac = r["skip"] / 68.0
            lin = dense * frac
            eff = (dense - r["runtime"]["p50"]) / lin if lin > 0 else 0.0
            lines.append(f"| {kind} | {r['pattern']} | {r['skip']} | {dense:.2f} "
                         f"| {r['ideal']['p50']:.2f} | {r['runtime']['p50']:.2f} | {eff:.3f} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote artifacts to", d)


if __name__ == "__main__":
    main()
