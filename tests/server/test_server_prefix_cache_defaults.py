#!/usr/bin/env python3
"""Gate 10B: agent-ready server defaults and overrides."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    COMMON_DIR,
    REPO_ROOT,
    ServerHarness,
    http_json,
    model_dir,
    prompt_ids,
    test_arena_gib,
)

LONG_TEXT = ("The quick brown fox jumps over the lazy dog while the patient "
             "researcher records every token for later analysis. ") * 400
SHORT_TEXT = ("Explain the water cycle in a short paragraph, then stop. "
              "Keep the answer concise.")


def read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def compute_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()


def backend_default_output(checker) -> None:
    for extra in (str(REPO_ROOT / "src" / "apps" / "server"),
                  str(REPO_ROOT / "src" / "apps" / "server" / "backend")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))
    import phaseshift_backend as backend

    service = backend.PhaseShiftBackend()
    service._max_seq_len = 32768
    service._default_max_output_tokens = 4096

    class Request:
        def __init__(self, tokens):
            self.Tokens = tokens

    checker.check("backend-default-output-4096",
                  service._max_new_tokens(Request(0), 100) == 4096,
                  "expected default output budget 4096")
    checker.check("backend-explicit-output-wins",
                  service._max_new_tokens(Request(32), 100) == 32,
                  "explicit max_tokens must win")
    checker.check("backend-remaining-context-clip",
                  service._max_new_tokens(Request(0), 32760) == 8,
                  "default must clip to remaining context")


def default_profile(checker) -> None:
    compute_log = Path("/tmp") / "phaseshift-g10b-default-compute.log"
    if compute_log.exists():
        compute_log.unlink()
    server_log = Path("/tmp") / "phaseshift-g10b-default-server.log"
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_BATCH_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute_log)}
    with ServerHarness(startup_timeout=300.0, use_defaults=True,
                       arena_override=test_arena_gib(),
                       log_path=server_log, env=env) as server:
        long_ids = prompt_ids(LONG_TEXT)
        checker.check("long-prompt-beyond-4k", len(long_ids) > 4096, str(len(long_ids)))
        first = http_json(f"{server.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": LONG_TEXT}],
            "temperature": 0, "max_tokens": 8,
        })
        first_content = first["choices"][0]["message"].get("content")
        checker.check("long-prompt-accepted", first_content is not None,
                      repr(first)[:300])
        second = http_json(f"{server.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": LONG_TEXT},
                         {"role": "assistant", "content": first_content},
                         {"role": "user", "content": "続けて。"}],
            "temperature": 0, "max_tokens": 8,
        })
        checker.check("long-prompt-multiturn",
                      second["choices"][0]["message"].get("content") is not None,
                      repr(second)[:300])

        explicit = http_json(f"{server.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": SHORT_TEXT}],
            "temperature": 0, "max_tokens": 24,
        })
        usage = explicit.get("usage") or {}
        completion = int(usage.get("completion_tokens", -1))
        checker.check("explicit-max-tokens-wins",
                      0 <= completion <= 24, repr(usage))

        statuses = []
        lock = threading.Lock()

        def worker(index):
            try:
                reply = http_json(f"{server.base_url}/chat/completions", {
                    "model": "phaseshift",
                    "messages": [{"role": "user",
                                  "content": f"{SHORT_TEXT} (session {index})"}],
                    "temperature": 0, "max_tokens": 16,
                })
                ok = reply["choices"][0]["message"].get("content") is not None
            except Exception:  # noqa: BLE001
                ok = False
            with lock:
                statuses.append(ok)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        checker.check("default-concurrency-4", len(statuses) == 4 and all(statuses),
                      repr(statuses))

    banner = read_log(Path(server.log_path))
    checker.check("default-context", "Context:      32768" in banner, banner[-800:])
    checker.check("default-concurrency", "Concurrency:  4" in banner, banner[-800:])
    checker.check("default-kv-dtype", "KV dtype:     bf16" in banner, banner[-800:])
    checker.check("default-prefix-cache",
                  "Prefix cache: 16384 tokens / 8 entries" in banner,
                  banner[-800:])
    checker.check("default-output-budget",
                  "Default output: 4096" in banner, banner[-800:])
    default_lines = [line for line in banner.splitlines()
                     if line.strip().startswith(("Prefix cache:", "Default output:"))]
    checker.check("default-no-optin-label",
                  bool(default_lines)
                  and all("(opt-in)" not in line for line in default_lines),
                  repr(default_lines))
    checker.check("reasoning-opt-in-label",
                  "Reasoning:    yes (opt-in)" in banner, banner[-800:])

    time.sleep(0.3)
    lines = compute_lines(compute_log)
    checker.check("default-multi-turn-hit",
                  any("PREFIX_CACHE_HIT" in line for line in lines),
                  repr(lines[-12:]))
    max_requests = 0
    for line in lines:
        if line.startswith("PHASESHIFT_BATCH_STEP") and "requests=" in line:
            try:
                max_requests = max(
                    max_requests,
                    int(line.split("requests=")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    checker.check("default-concurrency-batched", max_requests >= 2,
                  repr([l for l in lines if l.startswith("PHASESHIFT_BATCH_STEP")][:8]))


def override_profile(checker) -> None:
    compute_log = Path("/tmp") / "phaseshift-g10b-override-compute.log"
    if compute_log.exists():
        compute_log.unlink()
    server_log = Path("/tmp") / "phaseshift-g10b-override-server.log"
    env = {"PHASESHIFT_PREFIX_CACHE_TRACE": "1",
           "PHASESHIFT_COMPUTE_LOG": str(compute_log)}
    with ServerHarness(max_seq_len=4096, use_defaults=False,
                       max_concurrent_requests=1,
                       prefix_cache_capacity_tokens=0,
                       log_path=server_log, env=env) as server:
        reply = http_json(f"{server.base_url}/chat/completions", {
            "model": "phaseshift",
            "messages": [{"role": "user", "content": SHORT_TEXT}],
            "temperature": 0, "max_tokens": 16,
        })
        checker.check("override-still-serves",
                      reply["choices"][0]["message"].get("content") is not None,
                      repr(reply)[:300])

    banner = read_log(Path(server.log_path))
    checker.check("override-context", "Context:      4096" in banner, banner[-800:])
    checker.check("override-concurrency", "Concurrency:  1" in banner, banner[-800:])
    checker.check("override-cache-off", "Prefix cache: off" in banner, banner[-800:])
    time.sleep(0.3)
    lines = compute_lines(compute_log)
    checker.check("override-no-hit",
                  not any("PREFIX_CACHE_HIT" in line for line in lines),
                  repr(lines[-12:]))


def main() -> int:
    checker = Checker("server-prefix-cache-defaults")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        backend_default_output(checker)
        default_profile(checker)
        override_profile(checker)
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
