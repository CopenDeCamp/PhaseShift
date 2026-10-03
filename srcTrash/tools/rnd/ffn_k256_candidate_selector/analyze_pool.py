import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def info_table(diag):
    a = diag["aggregate"]
    skips = diag["meta"]["skips"]
    lines = []
    for kind in ["unrestricted"] + diag["meta"]["prefilters"]:
        for C in diag["meta"]["pools"]:
            if str(C) not in a.get(kind, {}):
                continue
            for k in skips:
                m = a[kind][str(C)][str(k)]
                rec = m["recall"]["mean"] if m["recall"] else 1.0
                ov = m["overlap"]
                ovs = f"{ov:.3f}" if ov is not None else "-"
                lines.append(f"| {kind} | {C} | {k} ({k/68*100:.2f}%) | {rec:.3f} | "
                             f"{m['ratio_vs_unrestricted']:.4f} | {m['improvement_vs_small_l2']:.3f} | "
                             f"{ovs} |")
    return lines


def ppl_table(ppl):
    lines = ["| prefilter | C | skip | PPL | dPPL % |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in sorted(ppl["runs"], key=lambda r: (r["prefilter"], r["C"], r["skip"])):
        lines.append(f"| {r['prefilter']} | {r['C']} | {r['skip']} ({r['skip']/68*100:.2f}%) | "
                     f"{r['perplexity']:.6f} | {r['delta_ppl_percent']:+.3f} |")
    return lines


def headroom(ppl, name, C, budget):
    pts = [(0.0, 0.0)]
    for r in ppl["runs"]:
        if r["prefilter"] == name and r["C"] == C:
            pts.append((r["skip"] / 68.0, r["delta_ppl_percent"]))
    pts.sort()
    best = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 > y0 and y0 <= budget <= y1:
            best = x0 + (budget - y0) * (x1 - x0) / (y1 - y0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k256_candidate_selector")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    diag = load(d / "pool_diag.json")
    ppl = load(d / "pool_ppl.json")

    out = []
    out.append("## Candidate pool information (J / unrestricted exact pair)\n")
    out.append("| prefilter | C | k | recall | J/unrestricted | imp vs SMALL_L2 | overlap |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    out.extend(info_table(diag))
    out.append("")
    out.append("## Full PPL\n")
    out.append(f"baseline PPL = {ppl['baseline']['perplexity']:.6f} "
               f"(positions {ppl['baseline']['positions']})")
    out.append("")
    out.extend(ppl_table(ppl))
    out.append("")
    out.append("## Headroom within budget\n")
    out.append("| prefilter | C | +2% | +5% |")
    out.append("| --- | ---: | ---: | ---: |")
    seen = []
    for r in ppl["runs"]:
        if (r["prefilter"], r["C"]) not in seen:
            seen.append((r["prefilter"], r["C"]))
    for name, C in seen:
        out.append(f"| {name} | {C} | {headroom(ppl, name, C, 2.0)*100:.1f}% "
                   f"| {headroom(ppl, name, C, 5.0)*100:.1f}% |")
    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
