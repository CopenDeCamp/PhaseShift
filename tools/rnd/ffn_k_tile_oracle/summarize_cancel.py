import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def diag_table(diag):
    methods = ["small_l2", "residual_greedy", "pair_seeded_greedy", "pair_greedy_swap"]
    lines = ["| k | actual skip | " + " | ".join(f"{m} mean" for m in methods) +
             " | imp greed | imp pair |", "| ---: | ---: | " + " | ".join("---:" for _ in methods) +
             " | ---: | ---: |"]
    for k in map(str, diag["meta"]["skip_groups"]):
        a = diag["aggregate"][k]
        cells = " | ".join(f"{a[m]['mean_err']:.5g}" for m in methods)
        lines.append(f"| {k} | {int(k)/68:.4f} | {cells} | "
                     f"{a['residual_greedy']['improvement_vs_small_l2']:.3f} | "
                     f"{a['pair_greedy_swap']['improvement_vs_small_l2']:.3f} |")
    return "\n".join(lines)


def pair_table(ps):
    lines = ["| layer | cos mean | cos min | p01 | p05 | frac<-0.5 | frac<-0.7 | best ratio mean | best ratio min |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for L in sorted(ps, key=int):
        s = ps[L]
        lines.append(f"| {L} | {s['cos_mean']:.4f} | {s['cos_min']:.3f} | {s['cos_p01']:.3f} | "
                     f"{s['cos_p05']:.3f} | {s['cos_frac_lt_m0.5']:.2e} | {s['cos_frac_lt_m0.7']:.2e} | "
                     f"{s['best_pair_cancel_ratio_mean']:.4f} | {s['best_pair_cancel_ratio_min']:.4f} |")
    return "\n".join(lines)


def ppl_table(res):
    runs = res["runs"]
    methods = res["meta"]["methods"]
    lines = ["| skip groups | actual skip | method | PPL | dPPL % |",
             "| ---: | ---: | --- | ---: | ---: |"]
    order = {m: i for i, m in enumerate(methods)}
    for r in sorted(runs, key=lambda r: (r["skip_group_count"], order.get(r["method"], 99))):
        d = r["delta_ppl_percent"]
        lines.append(f"| {r['skip_group_count']}/68 | {r['actual_skip_fraction']:.5f} | {r['method']} | "
                     f"{r['perplexity']:.6f} | {d:+.3f} |")
    return "\n".join(lines)


def headroom(res, method, budget):
    pts = [(0.0, 0.0)]
    for r in res["runs"]:
        if r["method"] == method:
            pts.append((r["actual_skip_fraction"], r["delta_ppl_percent"]))
    pts.sort()
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 >= budget >= y0 and y1 > y0:
            return x0 + (budget - y0) * (x1 - x0) / (y1 - y0)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k_tile_cancel_oracle")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if (d / "diagnostic.json").exists():
            print("## Stage A diagnostic\n")
            print(diag_table(load(d / "diagnostic.json")))
            print()
        if (d / "pair_stats.json").exists():
            print("## Pair cancellation statistics\n")
            print(pair_table(load(d / "pair_stats.json")))
            print()
        if (d / "oracle_n64.json").exists():
            res = load(d / "oracle_n64.json")
            print("## Stage B full PPL\n")
            print(ppl_table(res))
            print()
            for m in res["meta"]["methods"]:
                print(f"{m}: +2% headroom = {headroom(res, m, 2.0):.5f}, "
                      f"+5% headroom = {headroom(res, m, 5.0):.5f}")
            print()
    text = buf.getvalue()
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
