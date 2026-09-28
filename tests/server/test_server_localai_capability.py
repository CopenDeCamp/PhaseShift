#!/usr/bin/env python3
"""Gate 8C: launcher LocalAI fail-closed capability probe.

Confirms that `phaseshift-server` starts only with a LocalAI runtime that is
fail-closed for both Chat structured output and Responses structured output.

Three runtime tiers are exercised:

  * patched (Chat + Responses patches) -> Ready.
  * Chat-only patch                    -> rejected before Ready.
  * vanilla v4.10.0                    -> rejected before Ready.

The negative tiers need unpatched / Chat-only LocalAI v4.10.0 binaries via
`PHASESHIFT_LOCALAI_VANILLA_BINARY` and `PHASESHIFT_LOCALAI_CHAT_ONLY_BINARY`.
Without them the test skips (exit 77).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ServerHarness,
    localai_binary,
    localai_chat_only_binary,
    localai_vanilla_binary,
    model_dir,
)


def wait_for_log(log_path: Path, needle: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        if log_path.is_file():
            text = log_path.read_text(errors="replace")
            if needle in text:
                return text
        time.sleep(0.2)
    return text


def wait_for_exit(harness: ServerHarness, timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if harness.proc is not None and harness.proc.poll() is not None:
            return harness.proc.returncode
        time.sleep(0.2)
    return harness.proc.poll() if harness.proc is not None else None


def run_negative(checker: Checker, binary: str, label: str) -> None:
    saved = os.environ.get("PHASESHIFT_LOCALAI_BINARY")
    os.environ["PHASESHIFT_LOCALAI_BINARY"] = binary
    harness = ServerHarness(max_seq_len=512, startup_timeout=300)
    try:
        start_raised = False
        try:
            harness.start()
        except Exception:  # noqa: BLE001
            start_raised = True
        # /v1/models can become ready before the probe runs, so do not rely on
        # start() alone: the launcher must exit non-zero and never print Ready.
        returncode = wait_for_exit(harness, timeout=240.0)
        log = harness.log_path.read_text(errors="replace") if harness.log_path.is_file() else ""
        checker.check(f"{label}-rejected",
                      start_raised or (returncode is not None and returncode != 0),
                      f"start_raised={start_raised} rc={returncode} log={log[-400:]}")
        checker.check(f"{label}-no-ready", "Ready." not in log, log[-400:])
        checker.check(f"{label}-fail-closed-report",
                      "fail-closed" in log and "bundled LocalAI runtime" in log,
                      log[-600:])
    finally:
        harness.close()
        if saved is None:
            os.environ.pop("PHASESHIFT_LOCALAI_BINARY", None)
        else:
            os.environ["PHASESHIFT_LOCALAI_BINARY"] = saved


def main() -> int:
    checker = Checker("server-localai-capability")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2

    patched = localai_binary()
    vanilla = localai_vanilla_binary()
    chat_only = localai_chat_only_binary()
    if patched is None:
        print("[SKIP] server-localai-capability: no patched LocalAI binary")
        return 77
    if vanilla is None:
        print("[SKIP] server-localai-capability: "
              "set PHASESHIFT_LOCALAI_VANILLA_BINARY for the negative probe")
        return 77
    if chat_only is None:
        print("[SKIP] server-localai-capability: "
              "set PHASESHIFT_LOCALAI_CHAT_ONLY_BINARY for the Responses probe")
        return 77

    # Positive: fully patched runtime passes the startup capability probe.
    try:
        with ServerHarness(max_seq_len=512) as server:
            log = wait_for_log(server.log_path, "Ready.", timeout=30.0)
            checker.check("patched-ready", "Ready." in log, log[-400:])
            checker.check("patched-no-fail-closed-error",
                          "requires a LocalAI runtime" not in log, log[-400:])
    except Exception as exc:  # noqa: BLE001
        checker.check("patched-startup", False, repr(exc))

    # Negative: vanilla and Chat-only runtimes are rejected before Ready.
    run_negative(checker, vanilla, "vanilla")
    run_negative(checker, chat_only, "chat-only")

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
