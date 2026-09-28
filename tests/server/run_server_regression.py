#!/usr/bin/env python3
"""Canonical PhaseShift server regression runner.

The manifest is explicit, not a glob. `support.py`, the manual closure soak and
the benchmark script must not be pulled into the default regression run by
accident, so every participating test is named here.

Usage:
    python3 tests/server/run_server_regression.py --all
    python3 tests/server/run_server_regression.py --group reasoning --group agent
    python3 tests/server/run_server_regression.py --list

Each selected test runs as a subprocess and must exit 0. A non-zero exit is a
failure, including a missing prerequisite (for example a missing LocalAI binary);
nothing is silently treated as a skip.

Environment (inherited by every test):
    PHASESHIFT_BUILD_DIR                    default: <repo>/build
    PHASESHIFT_LOCALAI_BINARY               patched LocalAI runtime
    PHASESHIFT_LOCALAI_VANILLA_BINARY       unpatched LocalAI v4.10.0 (capability)
    PHASESHIFT_LOCALAI_CHAT_ONLY_BINARY     0001-only LocalAI (capability)
    PHASESHIFT_LOCALAI_NO_TOOL_POLICY_BINARY pre-0003 LocalAI (tool handshake)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = Path(__file__).resolve().parent

# Group order is the canonical `--all` order. A test appears in exactly one
# group. Intentionally excluded from the manifest:
#   support.py                  shared helpers, not a test
#   test_server_closure_soak.py manual long-running soak, not a regression
#   tools/bench_server_reasoning_constraints.py  benchmark, not a test
GROUPS = OrderedDict([
    ("core", [
        "test_compute_service.py",
        "test_compute_dflash2.py",
        "test_server_dflash2.py",
        "test_server_chat.py",
        "test_server_stream.py",
        "test_server_responses.py",
        "test_server_eos_contract.py",
        "test_server_unconstrained_oracle.py",
        "test_server_public_surface.py",
    ]),
    ("constraints", [
        "test_compute_constraints.py",
        "test_compute_structural_tag.py",
        "test_compute_composite_constraint.py",
        "test_constraint_concurrency.py",
        "test_constraint_dependency.py",
        "test_constraint_graph.py",
    ]),
    ("tools", [
        "test_server_tools.py",
        "test_server_stream_tools.py",
        "test_server_strict_tools.py",
        "test_server_strict_tools_stream.py",
        "test_server_responses_strict_tools.py",
        "test_tool_constraint_builder.py",
        "test_tool_policy_transport.py",
    ]),
    ("structured", [
        "test_server_structured.py",
        "test_server_structured_stream.py",
        "test_server_responses_structured.py",
        "test_server_responses_structured_stream.py",
        "test_localai_fail_closed.py",
        "test_localai_responses_fail_closed.py",
    ]),
    ("composition", [
        "test_server_structured_tools.py",
        "test_server_structured_tools_stream.py",
        "test_server_responses_structured_tools.py",
        "test_server_responses_structured_tools_stream.py",
        "test_tool_composition_builder.py",
        "test_tool_composition_transport.py",
    ]),
    ("prefix-cache", [
        "test_prefix_cache_no_host_transfer.py",
        "test_compute_prefix_cache.py",
        "test_compute_prefix_checkpoint.py",
        "test_server_prefix_cache.py",
        "test_server_agent_prefix_cache.py",
        "test_server_responses_prefix_cache.py",
        "test_server_prefix_cache_defaults.py",
    ]),
    ("reasoning", [
        "test_reasoning_transport.py",
        "test_server_reasoning.py",
        "test_server_reasoning_stream.py",
        "test_server_responses_reasoning.py",
        "test_server_reasoning_prefix_cache.py",
        "test_server_reasoning_concurrency.py",
    ]),
    ("reasoning-tools", [
        "test_reasoning_tool_parser.py",
        "test_server_reasoning_tools.py",
        "test_server_reasoning_tools_stream.py",
        "test_server_reasoning_structured.py",
        "test_server_reasoning_composition.py",
        "test_server_responses_reasoning_tools.py",
        "test_server_reasoning_tool_roundtrip.py",
        "test_compute_reasoning_constraint.py",
    ]),
    ("concurrency-cancel", [
        "test_compute_cancel.py",
        "test_compute_concurrency.py",
        "test_server_concurrency.py",
        "test_server_stream_cancel.py",
        "test_server_context_boundary.py",
    ]),
    ("capability", [
        "test_server_localai_capability.py",
    ]),
    ("agent", [
        "test_agent_opencode.py",
        "test_agent_codex.py",
    ]),
])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[],
                        help="group to run (repeatable)")
    parser.add_argument("--all", action="store_true", help="run every group")
    parser.add_argument("--list", action="store_true", help="list groups and exit")
    parser.add_argument("--tail", type=int, default=40,
                        help="output lines to show for a failing test")
    return parser.parse_args(argv)


def selected(args):
    if args.all:
        return list(GROUPS)
    if args.group:
        unknown = [g for g in args.group if g not in GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s): {', '.join(unknown)}")
        return args.group
    raise SystemExit("select at least one --group or --all")


def test_env():
    env = dict(os.environ)
    env.setdefault("PHASESHIFT_BUILD_DIR", str(REPO_ROOT / "build"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_test(name, env, tail):
    path = SERVER_DIR / name
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(SERVER_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.monotonic() - started
    output = proc.stdout or ""
    if proc.returncode != 0:
        lines = output.rstrip().splitlines()
        shown = "\n".join(lines[-tail:]) if tail > 0 else ""
        print(shown)
        print(f"[FAIL] {name} (rc={proc.returncode}, {duration:.1f}s)")
        return False
    summary = ""
    for line in reversed(output.splitlines()):
        if "passed=" in line:
            summary = line.strip()
            break
    print(f"[PASS] {name} ({duration:.1f}s) {summary}")
    return True


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        for group, tests in GROUPS.items():
            print(f"{group}: {len(tests)}")
            for test in tests:
                print(f"  {test}")
        return 0

    groups = selected(args)
    env = test_env()

    failures = []
    passed = 0
    print(f"server regression: groups={','.join(groups)}")
    for group in groups:
        print(f"===== group {group} =====")
        for test in GROUPS[group]:
            if run_test(test, env, args.tail):
                passed += 1
            else:
                failures.append(test)

    print(f"===== SUMMARY passed={passed} failed={len(failures)}")
    if failures:
        for test in failures:
            print(f"  FAIL {test}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
