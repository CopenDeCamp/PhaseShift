import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/ffn_k256_candidate_selector")
    ap.add_argument("--out", default="")
    ap.add_argument("--inputs", default="pool_ppl_run1.json,pool_ppl_run2.json")
    args = ap.parse_args()
    d = Path(args.dir)
    runs = []
    baseline = None
    for name in args.inputs.split(","):
        p = d / name
        if not p.is_file():
            continue
        obj = json.loads(p.read_text())
        if baseline is None:
            baseline = obj["baseline"]
        for r in obj["runs"]:
            r = dict(r)
            r["source"] = name
            runs.append(r)
    out = {"baseline": baseline, "runs": runs,
           "note": "run1 and run2 measured in separate processes; each has its own "
                   "same-run unrestricted reference"}
    path = d / (args.out or "pool_ppl.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}: {len(runs)} runs")


if __name__ == "__main__":
    main()
