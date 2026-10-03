import argparse
import json
from pathlib import Path

BITS_INT2 = 2
SCALE_BYTES = 2
TILE_ROWS = 80
K_GROUP = 256


def projected_bytes(C, d, bits=BITS_INT2, scale_bytes=SCALE_BYTES):
    code = TILE_ROWS * C * d * K_GROUP * bits / 8.0
    scale = TILE_ROWS * C * d * scale_bytes
    return code, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw-gbps", type=float, default=600.0,
                    help="sustained read bandwidth GB/s (break-even gate measured 578-636)")
    ap.add_argument("--cs", default="32,40,48,56")
    ap.add_argument("--ds", default="16,24,32,48")
    ap.add_argument("--out", default="artifacts/ffn_k256_candidate_selector/traffic_model.json")
    args = ap.parse_args()

    cs = [int(x) for x in args.cs.split(",")]
    ds = [int(x) for x in args.ds.split(",")]
    rows = []
    for C in cs:
        for d in ds:
            code, scale = projected_bytes(C, d)
            total = code + scale
            us = total / (args.bw_gbps * 1e9) * 1e6
            rows.append({"C": C, "d": d, "format": "INT2",
                         "code_mb": code / 1e6, "scale_mb": scale / 1e6,
                         "total_mb": total / 1e6, "lower_bound_us": us})
    result = {"bw_gbps": args.bw_gbps, "tile_rows": TILE_ROWS, "k_group": K_GROUP,
              "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print("| C | d | format | code MB/L | total MB/L | lower-bound us/L | +5% (<=7us) |")
    print("| ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for r in rows:
        ok = "yes" if r["lower_bound_us"] <= 7.0 else ("tight" if r["lower_bound_us"] <= 10.0 else "no")
        print(f"| {r['C']} | {r['d']} | {r['format']} | {r['code_mb']:.2f} | {r['total_mb']:.2f} "
              f"| {r['lower_bound_us']:.2f} | {ok} |")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
