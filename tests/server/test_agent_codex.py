#!/usr/bin/env python3
"""Gate 6B: Codex CLI agent E2E through PhaseShift Server (Responses API)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

CONFIG_TOML = """model = "phaseshift"
model_provider = "phaseshift"
model_reasoning_effort = "none"
model_context_window = 16384
model_max_output_tokens = 1024

[model_providers.phaseshift]
name = "PhaseShift"
base_url = "{base_url}"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
"""


def find_codex(bindir: Path):
    codex = shutil.which("codex")
    if codex is None:
        return None, None
    node = shutil.which("node")
    if node is None:
        bun = shutil.which("bun")
        if bun is None:
            return codex, None
        shim = bindir / "node"
        if not shim.exists():
            shim.symlink_to(bun)
        node = str(shim)
    return codex, node


def read_request_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def main() -> int:
    checker = Checker("agent-codex")
    base = Path(tempfile.mkdtemp(prefix="ps-codex-"))
    bindir = base / "bin"
    bindir.mkdir()
    codex, node = find_codex(bindir)
    if codex is None or node is None:
        print("[SKIP] agent-codex: codex or node runtime not installed")
        shutil.rmtree(base, ignore_errors=True)
        return 77
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    codex_home = base / "codex-home"
    codex_home.mkdir()
    workspace = base / "ws"
    workspace.mkdir()
    (workspace / "input.txt").write_text("100\n")
    request_log = base / "requests.txt"
    token_log = base / "tokens.txt"
    compute_log = base / "compute.log"

    try:
        with ServerHarness(
                max_seq_len=16384, arena_gib=24,
                prefix_cache_capacity_tokens=16384,
                prefix_cache_max_entries=8,
                device=int(os.environ.get("PHASESHIFT_TEST_DEVICE", "0")),
                env={"PHASESHIFT_BACKEND_REQUEST_LOG": str(request_log),
                     "PHASESHIFT_BACKEND_TOKEN_LOG": str(token_log),
                     "PHASESHIFT_PREFIX_CACHE_TRACE": "1",
                     "PHASESHIFT_COMPUTE_LOG": str(compute_log)}) as server:
            (codex_home / "config.toml").write_text(
                CONFIG_TOML.format(base_url=server.base_url))
            env = dict(server.env)
            env["CODEX_HOME"] = str(codex_home)
            env["PHASESHIFT_API_KEY"] = "dummy"
            env["PWD"] = str(workspace)
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")

            task = ("input.txt を読んでください。数値に23を足し、answer.txt に結果だけを書いてください。"
                    "作業が完了したらtoolを呼ぶのをやめ、最後に通常の文章で完了を報告してください。")

            process = None
            answer_ok = False
            for attempt in range(1, 4):
                attempt_ws = workspace / f"attempt{attempt}"
                attempt_ws.mkdir()
                (attempt_ws / "input.txt").write_text("100\n")
                process = subprocess.run(
                    [codex, "exec",
                     "-C", str(attempt_ws),
                     "--skip-git-repo-check",
                     "--dangerously-bypass-approvals-and-sandbox",
                     task],
                    env=env, capture_output=True, text=True, timeout=1800)
                answer = attempt_ws / "answer.txt"
                if (process.returncode == 0 and answer.is_file()
                        and answer.read_text().strip() == "123"):
                    answer_ok = True
                    break

            checker.check("codex-exit-and-answer", answer_ok,
                          f"rc={process.returncode if process else None} "
                          f"stderr={(process.stderr or '')[-1000:] if process else ''}")

            entries = read_request_log(request_log)
            roles = [m["role"] for e in entries for m in e.get("messages", [])]
            assistant_with_calls = any(
                any(m["role"] == "assistant" and m["has_tool_calls"]
                    for m in e.get("messages", []))
                for e in entries)
            checker.check("codex-responses-roundtrip-tool", "tool" in roles,
                          repr(roles[-10:]))
            checker.check("codex-responses-assistant-calls", assistant_with_calls, "")
            checker.check("codex-responses-multiple-requests", len(entries) >= 2,
                          f"requests={len(entries)}")
            if entries:
                checker.check("codex-wire-messages-not-prompt",
                              all(not e.get("prompt_empty") for e in entries),
                              repr([e.get("prompt_empty") for e in entries]))

            generations = 0
            if token_log.is_file():
                generations = sum(1 for line in token_log.read_text().splitlines() if line.strip())
            checker.check("codex-multiple-generations", generations >= 2,
                          f"generations={generations}")
            compute_text = (compute_log.read_text(errors="replace")
                            if compute_log.exists() else "")
            checker.check("prefix-cache-enabled",
                          "PREFIX_CACHE_ENABLED=1" in compute_text,
                          compute_text[-400:])
            checker.check("prefix-cache-hit", "PREFIX_CACHE_HIT" in compute_text,
                          compute_text[-600:])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
