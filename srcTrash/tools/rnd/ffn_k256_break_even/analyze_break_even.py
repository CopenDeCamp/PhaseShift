import argparse
import json
import re
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def pick(results, pattern, skip):
    for r in results:
        if r["pattern"] == pattern and r["skip"] == skip:
            return r
    for r in results:
        if r["pattern"] == "random" and r["skip"] == skip:
            return r
    return None


def layer_kind_map(model_dir):
    p = Path(model_dir) / "phaseshift_quantization.json"
    d = json.loads(p.read_text())
    out = {}
    for k, v in d["tensors"].items():
        m = re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$", k)
        if m:
            out[int(m.group(1))] = v.get("encoding")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k256_break_even")
    ap.add_argument("--model-dir", default="models/Qwen3.8-27B-PSQ")
    ap.add_argument("--pattern", default="oracle")
    ap.add_argument("--e2e-baseline", default="")
    ap.add_argument("--out", default="artifacts/ffn_k256_break_even/break_even.json")
    args = ap.parse_args()
    d = Path(args.dir)
    psq4 = load(d / "dense_psq4.json")
    psq8 = load(d / "dense_psq8.json")
    kinds = layer_kind_map(args.model_dir)
    n_psq4 = sum(1 for v in kinds.values() if v == "psq4")
    n_psq8 = sum(1 for v in kinds.values() if v == "psq8")
    n_total = n_psq4 + n_psq8
    skips = sorted({r["skip"] for r in psq4["results"]})

    if args.e2e_baseline and Path(args.e2e_baseline).exists():
        e2e = load(args.e2e_baseline)
        e2e_src = e2e.get("source", args.e2e_baseline)
    else:
        e2e = {"tok_per_s": 29.36, "source": "docs/perf/current.md (tg128 fusion+graph) reused"}
        e2e_src = e2e["source"]
    t_base_ms = 1000.0 / e2e["tok_per_s"]

    oracle_dppl = {7: None, 10: None, 14: None, 16: 1.851, 17: 2.885, 20: None,
                   27: 4.714, 28: 4.545, 30: None}

    ops = []
    for skip in skips:
        r4 = pick(psq4["results"], args.pattern, skip)
        r8 = pick(psq8["results"], args.pattern, skip)
        if r4 is None or r8 is None:
            continue
        dense4 = r4["dense"]["p50"]
        dense8 = r8["dense"]["p50"]
        rt4 = r4["runtime"]["p50"]
        rt8 = r8["runtime"]["p50"]
        ideal4 = r4["ideal"]["p50"]
        ideal8 = r8["ideal"]["p50"]
        frac = skip / 68.0
        gross4 = dense4 - rt4
        gross8 = dense8 - rt8
        gross_token = n_psq4 * gross4 + n_psq8 * gross8
        lin4 = dense4 * frac
        lin8 = dense8 * frac
        eff4 = gross4 / lin4 if lin4 > 0 else 0.0
        eff8 = gross8 / lin8 if lin8 > 0 else 0.0
        lin_total = n_psq4 * lin4 + n_psq8 * lin8
        eff = gross_token / lin_total if lin_total > 1e-9 else 0.0
        mask_over4 = rt4 - ideal4
        mask_over8 = rt8 - ideal8
        ratio4 = mask_over4 / max(dense4 - ideal4, 1e-9)
        ratio8 = mask_over8 / max(dense8 - ideal8, 1e-9)
        gross_token_ms = gross_token / 1000.0

        def budget(target_pct):
            req_ms = t_base_ms - t_base_ms / (1.0 + target_pct / 100.0)
            total_ms = gross_token_ms - req_ms
            return total_ms, total_ms * 1000.0 / 64.0

        b3, b3l = budget(3)
        b5, b5l = budget(5)
        b8, b8l = budget(8)
        ops.append({
            "skip_groups": skip,
            "skip_fraction": frac,
            "oracle_dppl": oracle_dppl.get(skip),
            "dense_us": dense4,
            "dense_us_psq4": dense4, "dense_us_psq8": dense8,
            "ideal_sparse_us": ideal4, "ideal_sparse_us_psq4": ideal4,
            "ideal_sparse_us_psq8": ideal8,
            "runtime_mask_sparse_us": rt4, "runtime_mask_sparse_us_psq4": rt4,
            "runtime_mask_sparse_us_psq8": rt8,
            "gross_save_us_psq4": gross4, "gross_save_us_psq8": gross8,
            "down_speedup_psq4": dense4 / rt4, "down_speedup_psq8": dense8 / rt8,
            "linear_ideal_save_us": gross_token / (n_psq4 + n_psq8),
            "skip_efficiency": eff, "skip_efficiency_psq4": eff4,
            "skip_efficiency_psq8": eff8,
            "mask_overhead_us_psq4": mask_over4, "mask_overhead_us_psq8": mask_over8,
            "mask_overhead_ratio_psq4": ratio4, "mask_overhead_ratio_psq8": ratio8,
            "selector_break_even_us_per_layer": gross_token / n_total,
            "selector_budget_75_us_per_layer": gross_token / n_total * 0.25,
            "selector_budget_50_us_per_layer": gross_token / n_total * 0.50,
            "projected_gross_save_token_ms": gross_token_ms,
            "selector_budget_for_e2e_3pct_us_per_layer": b3l,
            "selector_budget_for_e2e_5pct_us_per_layer": b5l,
            "selector_budget_for_e2e_8pct_us_per_layer": b8l,
            "projected_tok_s_selector_0us": 1000.0 / (t_base_ms - gross_token_ms) if t_base_ms > gross_token_ms else None,
        })
        e = ops[-1]
        for label, us in (("2us", 2.0), ("5us", 5.0), ("10us", 10.0)):
            t_new = t_base_ms - gross_token_ms + us * 64.0 / 1000.0
            e[f"projected_tok_s_selector_{label}"] = (1000.0 / t_new) if t_new > 0 else None
        for label, usl in (("3pct", b3l), ("5pct", b5l), ("8pct", b8l)):
            if usl > 0:
                t_new = t_base_ms - gross_token_ms + usl * 64.0 / 1000.0
                e[f"projected_tok_s_budget_{label}"] = (1000.0 / t_new) if t_new > 0 else None
            else:
                e[f"projected_tok_s_budget_{label}"] = None

    result = {
        "meta": {"pattern": args.pattern, "n_psq4": n_psq4, "n_psq8": n_psq8,
                 "n_layers": n_total, "dense_psq4_us": psq4["results"][0]["dense"]["p50"],
                 "dense_psq8_us": psq8["results"][0]["dense"]["p50"],
                 "e2e_baseline_tok_s": e2e["tok_per_s"], "e2e_baseline_ms": t_base_ms,
                 "e2e_baseline_source": e2e_src,
                 "psq4_working_set_mb": psq4["meta"]["working_set_mb"],
                 "psq8_working_set_mb": psq8["meta"]["working_set_mb"]},
        "operating_points": ops,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["meta"], indent=1))
    for e in ops:
        print(f"skip={e['skip_groups']:2d} ({e['skip_fraction']*100:5.2f}%) "
              f"gross={e['projected_gross_save_token_ms']*1000:7.1f}us/token "
              f"eff={e['skip_efficiency']:.3f} "
              f"BE={e['selector_break_even_us_per_layer']:5.2f}us/L "
              f"b5={e['selector_budget_for_e2e_5pct_us_per_layer']:7.2f}us/L "
              f"b3={e['selector_budget_for_e2e_3pct_us_per_layer']:7.2f}us/L")


if __name__ == "__main__":
    main()
