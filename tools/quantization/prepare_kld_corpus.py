#!/usr/bin/env python3
"""Prepare a deterministic PSKLDTOK token-id corpus for FPX KLD evaluation.

Usage:
    python3 tools/quantization/prepare_kld_corpus.py \
        --model-dir models/Qwen3.5-4B \
        --text wiki.test.raw \
        --output wiki-test.tokens
"""
import argparse
import hashlib
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("transformers is required; not a build/test dependency") from e

    tok = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok.encode(text, add_special_tokens=False)
    if args.max_tokens > 0:
        ids = ids[: args.max_tokens]

    header = b"PSKLDTOK" + struct.pack("<I", 1) + struct.pack("<Q", len(ids))
    payload = b"".join(struct.pack("<I", int(t)) for t in ids)
    data = header + payload

    with open(args.output, "wb") as f:
        f.write(data)

    sha = hashlib.sha256(data).hexdigest()
    print(f"tokens: {len(ids)}")
    print(f"output: {args.output}")
    print(f"corpus_sha256: {sha}")


if __name__ == "__main__":
    main()
