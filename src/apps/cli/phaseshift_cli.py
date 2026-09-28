#!/usr/bin/env python3
"""PhaseShift chat UI.

Wraps phaseshift-compute with the HF tokenizer and chat template.
Runs single-shot (--prompt) or interactive multi-turn chat.

Usage:
  ./build/phaseshift-cli --model-dir /path/to/Qwen3.5-4B --prompt 'こんにちは'
  ./build/phaseshift-cli --model-dir /path/to/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _setup_chat_path() -> None:
    candidates = []
    env = os.environ.get("PHASESHIFT_CHAT_COMMON")
    if env:
        candidates.append(Path(env))
    here = Path(sys.argv[0]).resolve().parent
    candidates.append(here / "phaseshift-server-lib" / "common")
    for base in Path(sys.argv[0]).resolve().parents:
        candidates.append(base / "src" / "apps" / "common")
    candidates.append(Path.cwd() / "src" / "apps" / "common")
    for candidate in candidates:
        if (candidate / "phaseshift_chat" / "codec.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_setup_chat_path()

from phaseshift_chat import codec  # noqa: E402


def find_compute_binary(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(os.environ.get("PHASESHIFT_COMPUTE", "") or sys.argv[0]).resolve()
    candidate = here.parent / "phaseshift-compute"
    if candidate.is_file():
        return candidate
    repo_build = Path.cwd() / "build" / "phaseshift-compute"
    if repo_build.is_file():
        return repo_build
    return Path("phaseshift-compute")


def load_processor(model_dir: str):
    try:
        return codec.load_processor(model_dir)
    except ImportError as e:
        print(f"ERROR: transformers not available: {e}", file=sys.stderr)
        sys.exit(1)


def prompt_ids(processor, messages: list[dict]) -> list[int]:
    return codec.prompt_ids(processor, messages)


def run_compute(binary: Path, args, input_ids: list[int]) -> list[int]:
    if len(input_ids) + args.max_new_tokens > args.max_seq_len:
        print("ERROR: prompt + generation exceeds max-seq-len", file=sys.stderr)
        sys.exit(1)
    with tempfile.TemporaryDirectory() as tmp:
        ids_file = Path(tmp) / "input_ids.txt"
        ids_file.write_text("\n".join(str(t) for t in input_ids) + "\n")
        cmd = [
            str(binary),
            "--model-dir", args.model_dir,
            "--input-ids-file", str(ids_file),
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-seq-len", str(args.max_seq_len),
            "--arena-gib", str(args.arena_gib),
            "--device", str(args.device),
            "--temperature", repr(float(args.temperature)),
            "--top-p", repr(float(args.top_p)),
            "--top-k", str(args.top_k),
            "--seed", str(args.seed),
        ]
        if args.dflash2_model_dir:
            cmd += [
                "--dflash2-model-dir", args.dflash2_model_dir,
                "--dflash2-drafts", str(args.dflash2_drafts),
                "--dflash2-stats", str(args.dflash2_stats),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            print(f"ERROR: phaseshift-compute failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(1)
        for line in result.stdout.splitlines():
            if line.startswith("GENERATED_IDS="):
                payload = line[len("GENERATED_IDS="):]
                return [int(x) for x in payload.split(",") if x.strip()]
    print("ERROR: GENERATED_IDS not found in phaseshift-compute output", file=sys.stderr)
    print(result.stdout)
    sys.exit(1)


def chat_once(processor, binary: Path, args, messages: list[dict]) -> str:
    ids = prompt_ids(processor, messages)
    generated = run_compute(binary, args, ids)
    if not generated:
        print("WARNING: no tokens generated", file=sys.stderr)
        return ""
    reply = codec.decode_generated(processor, generated)
    messages.append({"role": "assistant", "content": reply})
    return reply


def main() -> int:
    parser = argparse.ArgumentParser(description="PhaseShift chat UI")
    parser.add_argument("--model-dir", required=True, help="Qwen3.5-4B model directory")
    parser.add_argument("--binary", default=None, help="phaseshift-compute path (default: next to this script)")
    parser.add_argument("--prompt", default=None, help="single-shot prompt (omit for interactive mode)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--arena-gib", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dflash2-model-dir", default=None,
                        help="DFlash2 draft model directory; enables greedy speculative decoding")
    parser.add_argument("--dflash2-drafts", type=int, default=7,
                        help="speculative draft tokens per round (default 7)")
    parser.add_argument("--dflash2-stats", type=int, default=1,
                        help="print speculative decode statistics (default 1)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        return 1
    binary = find_compute_binary(args.binary)
    if not binary.is_file():
        print(f"ERROR: phaseshift-compute not found: {binary}", file=sys.stderr)
        return 1

    processor = load_processor(str(model_dir))
    messages: list[dict] = []

    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        reply = chat_once(processor, binary, args, messages)
        print(f"user: {args.prompt}")
        print(f"assistant: {reply}")
        if not reply.strip():
            return 2
        return 0

    print("PhaseShift chat (Ctrl-D or 'exit' to quit)")
    while True:
        try:
            line = input("user> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": line})
        print(f"assistant: {chat_once(processor, binary, args, messages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
