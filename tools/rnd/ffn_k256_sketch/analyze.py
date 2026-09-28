import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def table(diag):
    a = diag["aggregate"]
    groups = [("exact", "small_l2"), ("exact", "residual"), ("exact", "pair"),
              ("d2", "sk_residual"), ("d4", "sk_residual"), ("d4", "sk_pair4"),
              ("d8", "sk_residual"), ("d8", "sk_pair4"), ("d16", "sk_pair4")]
    lines = ["| selector | " + " | ".join(
        f"k={k} ({int(k)/68*100:.1f}%)" for k in diag["meta"]["skips"]) + " |",
        "| --- | " + " | ".join("---:" for _ in diag["meta"]["skips"]) + " |"]
    for g, n in groups:
        if g not in a or n not in a[g]:
            continue
        cells = []
        for k in diag["meta"]["skips"]:
            m = a[g][n].get(str(k))
            cells.append(f"{m['ratio_vs_pair']:.3f}" if m else "-")
        lines.append(f"| {g}/{n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ppl_table(ppl):
    runs = ppl["runs"]
    lines = ["| method | d | skip | actual | PPL | dPPL % |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    order = {}
    for r in runs:
        key = (r["method"], r["dim"])
        order.setdefault(key, 0)
    for r in sorted(runs, key=lambda r: (r["method"], r["dim"], r["skip"])):
        lines.append(f"| {r['method']} | {r['dim']} | {r['skip']} ({r['skip']/68*100:.2f}%) | "
                     f"{r['actual_skip']:.5f} | {r['perplexity']:.6f} | {r['delta_ppl_percent']:+.3f} |")
    return "\n".join(lines)


def headroom(ppl, method, dim, budget):
    pts = []
    for r in ppl["runs"]:
        if r["method"] == method and r["dim"] == dim:
            pts.append((r["skip"] / 68.0, r["delta_ppl_percent"]))
    pts.append((0.0, 0.0))
    pts.sort()
    best = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 >= budget >= y0 and y1 > y0:
            best = x0 + (budget - y0) * (x1 - x0) / (y1 - y0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k256_sketch")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    diag = load(d / "diagnostic.json")
    ppl = load(d / "ppl_float_sketch.json")

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("## Information Gate: mean J_exact / exact_pair J\n")
        print(table(diag))
        print()
        print("## Full PPL (float sketch, seed 0)\n")
        print(f"baseline PPL = {ppl['baseline']['perplexity']:.6f} "
              f"(positions {ppl['baseline']['positions']})")
        print()
        print(ppl_table(ppl))
        print()
        print("## Headroom (fraction of 68 groups) within budget\n")
        print("| method | d | +2% | +5% |")
        print("| --- | ---: | ---: | ---: |")
        for r in sorted(ppl["runs"], key=lambda r: (r["method"], r["dim"])):
            key = (r["method"], r["dim"])
        seen = []
        for r in ppl["runs"]:
            if (r["method"], r["dim"]) not in seen:
                seen.append((r["method"], r["dim"]))
        for m, dim in seen:
            print(f"| {m} | {dim} | {headroom(ppl, m, dim, 2.0):.4f} "
                  f"| {headroom(ppl, m, dim, 5.0):.4f} |")
    text = buf.getvalue()
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
