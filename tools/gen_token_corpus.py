#!/usr/bin/env python3
"""Generate a PSKLDTOK token corpus from text using a model's tokenizer.

  python3 tools/gen_token_corpus.py \
      --model-dir models/Qwen3.5-4B-PSQ --text "..." --output corpus.psktok
"""
import argparse
import struct
import sys

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    if args.text is None and args.text_file is None:
        ap.error("either --text or --text-file is required")
    text = args.text
    if text is None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    if not ids:
        print("empty tokenization", file=sys.stderr)
        return 1

    vocab = int(tok.vocab_size)
    for t in ids:
        if t < 0 or t >= vocab:
            print(f"token {t} out of range {vocab}", file=sys.stderr)
            return 1

    with open(args.output, "wb") as f:
        f.write(b"PSKLDTOK")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", len(ids)))
        f.write(struct.pack("<%di" % len(ids), *ids))
    print(f"wrote {len(ids)} tokens to {args.output} (vocab={vocab})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
