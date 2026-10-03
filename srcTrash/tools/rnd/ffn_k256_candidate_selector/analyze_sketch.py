import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def diag_table(diag):
    a = diag["aggregate"]
    skips = diag["meta"]["skips"]
    lines = []
    for kind in diag["meta"]["prefilters"]:
        for C in diag["meta"]["pools"]:
            for d in diag["meta"]["dims"]:
                for m in diag["meta"]["methods"]:
                    for k in skips:
                        r = a[kind][str(C)][str(d)][m][str(k)]
                        lines.append(f"| {kind} | {C} | {d} | {m} | {k} ({k/68*100:.2f}%) | "
                                     f"{r['ratio_vs_pool_exact']:.3f} | "
                                     f"{r['ratio_vs_unrestricted']:.3f} | "
                                     f"{r['overlap_pool_exact']:.3f} |")
    return lines


def ppl_table(ppl):
    lines = ["| prefilter | C | d | selector | skip | PPL | dPPL % |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for r in sorted(ppl["runs"], key=lambda r: (r["prefilter"], r["C"], r["dim"],
                                                r["method"], r["skip"])):
        lines.append(f"| {r['prefilter']} | {r['C']} | {r['dim']} | {r['method']} | "
                     f"{r['skip']} ({r['skip']/68*100:.2f}%) | {r['perplexity']:.6f} | "
                     f"{r['delta_ppl_percent']:+.3f} |")
    return lines


def headroom(ppl, name, C, d, m, budget):
    pts = [(0.0, 0.0)]
    for r in ppl["runs"]:
        if (r["prefilter"], r["C"], r["dim"], r["method"]) == (name, C, d, m):
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
    args = ap.parse_args()
    d = Path(args.dir)
    diag = load(d / "sketch_pool_diag.json")
    out = []
    out.append("## Pool-restricted sketch (J / pool exact pair)\n")
    out.append("| prefilter | C | d | selector | k | J/pool-exact | J/unrestricted | overlap |")
    out.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    out.extend(diag_table(diag))
    ppl_path = d / "sketch_pool_ppl.json"
    if ppl_path.is_file():
        ppl = load(ppl_path)
        out.append("")
        out.append("## Stage B full PPL\n")
        out.append(f"baseline PPL = {ppl['baseline']['perplexity']:.6f}")
        out.append("")
        out.extend(ppl_table(ppl))
        out.append("")
        out.append("## +5% headroom\n")
        out.append("| prefilter | C | d | selector | +2% | +5% |")
        out.append("| --- | ---: | ---: | --- | ---: | ---: |")
        seen = []
        for r in ppl["runs"]:
            key = (r["prefilter"], r["C"], r["dim"], r["method"])
            if key not in seen:
                seen.append(key)
        for name, C, dd, m in seen:
            out.append(f"| {name} | {C} | {dd} | {m} | "
                       f"{headroom(ppl, name, C, dd, m, 2.0)*100:.1f}% | "
                       f"{headroom(ppl, name, C, dd, m, 5.0)*100:.1f}% |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
