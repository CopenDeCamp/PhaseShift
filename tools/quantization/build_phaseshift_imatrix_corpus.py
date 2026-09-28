#!/usr/bin/env python3
"""
Build a PhaseShift-oriented iMatrix calibration corpus from a source tree
or a .tar.gz project snapshot.

The corpus mixes:
  - HIP/C/C++/Python/CMake source code
  - Japanese technical specifications / architecture / quantization docs

This is a generalized version of the script used to generate:
  phaseshift_imatrix_domain_corpus.txt

No external Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

DEFAULT_CODE_TARGET = 1_380_000
DEFAULT_DOC_TARGET = 760_000
DEFAULT_MAX_BLOCK_CHARS = 28_000
DEFAULT_SEED = 20260817

_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
    "build",
    "Testing",
    ".worktrees",
    "artifacts",
    "environment",
    ".rocprofv3",
    ".vscode",
    ".superpowers",
    "fixtures",
}

_EXCLUDED_SUFFIXES = {
    ".bin",
    ".o",
    ".so",
    ".a",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".pyc",
    ".pyo",
    ".whl",
    ".egg",
    ".db",
    ".sqlite",
}


def _is_excluded(path: Path, root: Path, output: Path | None) -> bool:
    if output is not None:
        try:
            if path.resolve() == output.resolve():
                return True
        except Exception:
            pass
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:
        if part in _EXCLUDED_DIRS:
            return True
    if path.suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def jp_ratio(text: str) -> float:
    return len(JP_RE.findall(text)) / max(1, len(text))


def split_blocks(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in re.split(r"(\n\s*\n)", text):
        if current_chars + len(paragraph) > max_chars and current:
            block = "".join(current).strip()
            if len(block) >= 300:
                parts.append(block)
            current = []
            current_chars = 0
        current.append(paragraph)
        current_chars += len(paragraph)
    if current:
        block = "".join(current).strip()
        if len(block) >= 300:
            parts.append(block)
    return parts


def find_project_root(extracted_root: Path) -> Path:
    entries = [p for p in extracted_root.iterdir() if p.name not in {".", ".."}]
    dirs = [p for p in entries if p.is_dir()]
    if len(entries) == 1 and len(dirs) == 1:
        return dirs[0]
    candidate = extracted_root / "phaseshift"
    if candidate.is_dir():
        return candidate
    return extracted_root


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _validate_members(tf: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        member_path = Path(member.name)
        if member_path.is_absolute():
            raise ValueError(f"Rejected absolute tar member: {member.name}")
        if ".." in member_path.parts:
            raise ValueError(f"Rejected tar member with traversal: {member.name}")
        target = dest / member.name
        if not _is_within_directory(dest_resolved, target):
            raise ValueError(f"Rejected tar member outside destination: {member.name}")


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    _validate_members(tf, dest)
    for member in tf.getmembers():
        tf.extract(member, dest)


def prepare_source(input_path: Path) -> tuple[Path, Path | None]:
    if input_path.is_dir():
        return input_path.resolve(), None
    lower = input_path.name.lower()
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        temp_root = Path(tempfile.mkdtemp(prefix="phaseshift_imatrix_"))
        try:
            with tarfile.open(input_path, "r:*") as tf:
                _validate_members(tf, temp_root)
                try:
                    tf.extractall(temp_root, filter="data")
                except TypeError:
                    _safe_extract(tf, temp_root)
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise
        return find_project_root(temp_root), temp_root
    raise ValueError(
        f"Unsupported input: {input_path}\n"
        "Pass a source directory, .tar.gz, .tgz, or .tar archive."
    )


def collect_document_candidates(root: Path, output: Path | None = None):
    candidates = []
    for pattern in ("*.md", "*.txt"):
        for path in root.rglob(pattern):
            if _is_excluded(path, root, output):
                continue
            text = read_text(path)
            if not text:
                continue
            rel = path.relative_to(root).as_posix()
            ratio = jp_ratio(text)
            score = ratio
            if rel.startswith("docs/superpowers/specs/"):
                score += 0.18
            if rel.startswith("docs/architecture/"):
                score += 0.12
            if rel.startswith("docs/quantization/"):
                score += 0.10
            if rel.startswith("docs/decisions/"):
                score += 0.08
            if rel.startswith("docs/performance/"):
                score += 0.06
            if rel == "README.md":
                score += 0.05
            if (
                ratio >= 0.035
                or "/specs/" in rel
                or rel.startswith("docs/architecture/")
            ):
                candidates.append((score, ratio, rel, text))
    candidates.sort(reverse=True)
    return candidates


def collect_code_candidates(root: Path, seed: int, output: Path | None = None):
    ext_weight = {
        ".hip": 5.0,
        ".cpp": 4.0,
        ".cc": 4.0,
        ".cxx": 4.0,
        ".hpp": 4.0,
        ".h": 3.5,
        ".hh": 3.5,
        ".c": 3.5,
        ".py": 2.5,
        ".cmake": 2.0,
        ".sh": 1.8,
        ".bash": 1.8,
    }
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path, root, output):
            continue
        ext = path.suffix.lower()
        if path.name == "CMakeLists.txt":
            ext = ".cmake"
        if ext not in ext_weight:
            continue
        text = read_text(path)
        if len(text) < 300:
            continue
        rel = path.relative_to(root).as_posix()
        weight = ext_weight[ext]
        if rel.startswith("kernels/"):
            weight += 1.5
        if rel.startswith("benchmarks/"):
            weight += 0.6
        if "test" in rel.lower():
            weight += 0.8
        candidates.append((weight, rel, text))
    candidates.sort(key=lambda x: x[1])
    rng = random.Random(seed)
    scored = []
    for weight, rel, text in candidates:
        u = max(rng.random(), 1e-12)
        key = -math.log(u) / weight
        scored.append((key, rel, text))
    scored.sort()
    return scored


def build_corpus(
    root: Path,
    output: Path,
    *,
    code_target_chars: int,
    doc_target_chars: int,
    max_block_chars: int,
    seed: int,
    code_per_cycle: int,
    docs_per_cycle: int,
) -> dict:
    if code_per_cycle <= 0 or docs_per_cycle <= 0:
        raise ValueError("code_per_cycle and docs_per_cycle must be > 0")
    doc_candidates = collect_document_candidates(root, output)
    doc_blocks: list[str] = []
    doc_chars = 0
    for _, ratio, rel, text in doc_candidates:
        for index, block in enumerate(split_blocks(text, max_block_chars), start=1):
            if doc_chars >= doc_target_chars:
                break
            header = (
                f"### [PHASESHIFT-SPEC] {rel} "
                f"part={index} jp_ratio={ratio:.3f}\n\n"
            )
            payload = header + block + "\n"
            doc_blocks.append(payload)
            doc_chars += len(payload)
        if doc_chars >= doc_target_chars:
            break
    code_candidates = collect_code_candidates(root, seed, output)
    code_blocks: list[str] = []
    code_chars = 0
    for _, rel, text in code_candidates:
        for index, block in enumerate(split_blocks(text, max_block_chars), start=1):
            if code_chars >= code_target_chars:
                break
            header = f"### [PHASESHIFT-CODE] {rel} part={index}\n\n"
            payload = header + block + "\n"
            code_blocks.append(payload)
            code_chars += len(payload)
        if code_chars >= code_target_chars:
            break
    mixed: list[str] = []
    ci = 0
    di = 0
    while ci < len(code_blocks) or di < len(doc_blocks):
        for _ in range(code_per_cycle):
            if ci < len(code_blocks):
                mixed.append(code_blocks[ci])
                ci += 1
        for _ in range(docs_per_cycle):
            if di < len(doc_blocks):
                mixed.append(doc_blocks[di])
                di += 1
    preamble = """# PhaseShift iMatrix domain calibration corpus
# Generated from a PhaseShift project snapshot/source tree.
# Purpose: coding-oriented iMatrix calibration with Japanese technical specifications mixed in.
# The file intentionally contains raw code + technical prose.
# It is NOT an instruction-tuning dataset.

"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(preamble + "\n\n".join(mixed), encoding="utf-8")
    final_text = output.read_text(encoding="utf-8")
    return {
        "project_root": str(root),
        "output": str(output),
        "bytes": output.stat().st_size,
        "characters": len(final_text),
        "approx_tokens_at_4_chars_per_token": round(len(final_text) / 4),
        "japanese_characters": len(JP_RE.findall(final_text)),
        "japanese_character_ratio": round(jp_ratio(final_text), 4),
        "code_blocks": len(code_blocks),
        "spec_blocks": len(doc_blocks),
        "code_chars": code_chars,
        "spec_chars": doc_chars,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PhaseShift-oriented iMatrix calibration corpus."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="PhaseShift source directory or .tar.gz/.tgz/.tar snapshot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("phaseshift_imatrix_domain_corpus.txt"),
        help="Output text file.",
    )
    parser.add_argument(
        "--code-target-chars",
        type=int,
        default=DEFAULT_CODE_TARGET,
    )
    parser.add_argument(
        "--doc-target-chars",
        type=int,
        default=DEFAULT_DOC_TARGET,
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=DEFAULT_MAX_BLOCK_CHARS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--code-per-cycle",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--docs-per-cycle",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    if args.code_per_cycle <= 0:
        parser.error("--code-per-cycle must be > 0")
    if args.docs_per_cycle <= 0:
        parser.error("--docs-per-cycle must be > 0")
    return args


def main() -> int:
    args = parse_args()
    temp_root = None
    try:
        project_root, temp_root = prepare_source(args.input)
        stats = build_corpus(
            project_root,
            args.output,
            code_target_chars=args.code_target_chars,
            doc_target_chars=args.doc_target_chars,
            max_block_chars=args.max_block_chars,
            seed=args.seed,
            code_per_cycle=args.code_per_cycle,
            docs_per_cycle=args.docs_per_cycle,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
