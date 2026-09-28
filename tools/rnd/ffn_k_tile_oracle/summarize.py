import argparse
import json
from pathlib import Path


def fmt_pct(x):
    return "n/a" if x is None else f"{x:+.3f}"


def sweep_table(result):
    runs = result["runs"]
    lines = ["| target skip | groups skipped | actual skip | PPL | dPPL % |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        lines.append(
            f"| {r['target_skip_fraction']:.4f} | {r['skip_group_count']}/{result['meta']['n_groups']} "
            f"| {r['actual_skip_fraction']:.5f} | {r['perplexity']:.6f} "
            f"| {fmt_pct(r['delta_ppl_percent'])} |")
    return "\n".join(lines)


def skip_for_budget(result, budget):
    runs = sorted((r for r in result["runs"] if r["delta_ppl_percent"] is not None),
                  key=lambda r: r["actual_skip_fraction"])
    best = 0.0
    for r in runs:
        if r["delta_ppl_percent"] <= budget:
            best = r["actual_skip_fraction"]
    return best


def contribution_summary(result):
    layers = result["contribution"]["layers"]
    keys = ["score_mean", "score_p50", "score_p90", "score_p99", "top1_frac", "top4_frac",
            "top8_frac", "bottom10_frac", "bottom20_frac", "bottom30_frac", "bottom40_frac"]
    lines = ["| stat | min | mean | max |", "| --- | ---: | ---: | ---: |"]
    for k in keys:
        vals = [layers[L][k] for L in layers]
        lines.append(f"| {k} | {min(vals):.6g} | {sum(vals)/len(vals):.6g} | {max(vals):.6g} |")
    for sk in result["layer_profile"]["reference_skip"]:
        key = f"{sk:.2f}"
        vals = [layers[L]["retained_proxy"][key] for L in layers]
        lines.append(f"| retained_proxy@skip{key} | {min(vals):.6f} | "
                     f"{sum(vals)/len(vals):.6f} | {max(vals):.6f} |")
    return "\n".join(lines)


def profile_section(result):
    lines = []
    for key, s in result["layer_profile"]["summaries"].items():
        lines.append(f"skip {key}: best L{s['best_layer']} {s['best_retained']:.6f}, "
                     f"median L{s['median_layer']} {s['median_retained']:.6f}, "
                     f"worst L{s['worst_layer']} {s['worst_retained']:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = json.loads(Path(args.inp).read_text())
    meta = result["meta"]
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"# {args.inp}")
        print(f"tile_width={meta['tile_width']} n_tiles={meta['n_tiles']} "
              f"k_group={meta['k_group']} n_groups={meta['n_groups']} "
              f"baseline={result['baseline']['perplexity']:.6f} "
              f"positions={result['baseline']['positions']}")
        parity = result.get("parity")
        if parity:
            print(f"parity skip0 ppl={parity['decomposed_ppl']:.6f} "
                  f"delta={fmt_pct(parity['delta_ppl_percent'])}% passed={parity['passed']}")
        print()
        print(sweep_table(result))
        print()
        print(f"max skip within +2%: {skip_for_budget(result, 2.0):.5f}")
        print(f"max skip within +5%: {skip_for_budget(result, 5.0):.5f}")
        print()
        print(contribution_summary(result))
        print()
        print(profile_section(result))
    text = buf.getvalue()
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
