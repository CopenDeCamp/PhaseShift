#!/usr/bin/env python3
"""Gate 7A: ComputeClient cancellation and reuse acceptance."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    compute_binary,
    model_dir,
)

BACKEND_DIR = Path(__file__).resolve().parents[2] / "src" / "apps" / "server" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import compute_client  # noqa: E402

PROMPT = [248041, 77091]


def make_client() -> compute_client.ComputeClient:
    argv = [
        str(compute_binary()),
        "--model-dir", str(model_dir()),
        "--serve-stdio",
        "--max-seq-len", "512",
        "--arena-gib", "24",
        "--page-tokens", "16",
        "--device", os.environ.get("PHASESHIFT_TEST_DEVICE", "0"),
        "--kv-cache-dtype", "bf16",
    ]
    client = compute_client.ComputeClient(argv)
    client.start()
    return client


def main() -> int:
    checker = Checker("compute-cancel")
    client = make_client()
    try:
        pid = client._proc.pid

        # Stream and abandon after a few tokens: the client must cancel and
        # drain before releasing the single-flight lock.
        stream = client.generate_stream(PROMPT, 400, temperature=0.0)
        seen = 0
        for event in stream:
            if "token_id" in event:
                seen += 1
                if seen == 3:
                    stream.close()
                    break
        checker.check("stream-abandoned-after-tokens", seen == 3, f"seen={seen}")

        # Reuse immediately after the abandoned stream.
        done = client.generate(PROMPT, 8, temperature=0.0)
        checker.check("reuse-after-abandon",
                      done.get("finish_reason") == "length"
                      and len(done.get("generated_ids", [])) == 8,
                      repr(done))
        checker.check("compute-pid-stable", client._proc.pid == pid,
                      f"{pid} -> {client._proc.pid}")

        # Non-stream cancellation via the should_cancel callback.
        cancelled = False
        try:
            client.generate(PROMPT, 400, temperature=0.0,
                            should_cancel=lambda: True)
        except compute_client.ComputeCancelled:
            cancelled = True
        checker.check("generate-should-cancel", cancelled, "")

        done2 = client.generate(PROMPT, 4, temperature=0.0)
        checker.check("reuse-after-cancel",
                      done2.get("finish_reason") == "length"
                      and len(done2.get("generated_ids", [])) == 4,
                      repr(done2))
        checker.check("compute-pid-stable-2", client._proc.pid == pid,
                      f"{pid} -> {client._proc.pid}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
