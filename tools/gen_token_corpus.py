#!/usr/bin/env python3
"""Generate a PSKLDTOK token corpus from text using a model's tokenizer.

  python3 tools/gen_token_corpus.py \
      --model-dir models/Qwen3.5-4B-PSQ --text "..." --output corpus.psktok

Repeated --text-file arguments are concatenated in order. An existing
PSKLDTOK file can be prepended with --base-psktok, so an extended corpus
keeps the original token prefix.
"""
import argparse
import json
import os
import struct
import sys

from transformers import AutoTokenizer


def model_vocab_size(model_dir, fallback):
    try:
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return fallback
    text_config = cfg.get("text_config") or {}
    return int(text_config.get("vocab_size") or cfg.get("vocab_size") or fallback)


def read_psktok(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"PSKLDTOK":
            raise ValueError(f"{path}: not a PSKLDTOK file")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        count = struct.unpack("<Q", f.read(8))[0]
        data = f.read(count * 4)
        if len(data) != count * 4:
            raise ValueError(f"{path}: truncated payload")
        return list(struct.unpack(f"<{count}i", data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", action="append", default=None)
    ap.add_argument("--base-psktok", default=None)
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    texts = []
    if args.text is not None:
        texts.append(args.text)
    for path in args.text_file or []:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    if not texts and args.base_psktok is None:
        ap.error("at least one of --text, --text-file, --base-psktok is required")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    vocab = model_vocab_size(args.model_dir, int(tok.vocab_size))

    ids = []
    if args.base_psktok is not None:
        ids.extend(read_psktok(args.base_psktok))
    for text in texts:
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    if not ids:
        print("empty tokenization", file=sys.stderr)
        return 1

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
