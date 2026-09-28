#!/usr/bin/env python3
"""Gate 10A: production prefix cache must stay device-resident.

This is a static guard, not a substitute for the functional native test.
It fails if the production prefix cache source reintroduces host round-trips
or extra synchronization. Test-side verification copies are exempt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, REPO_ROOT  # noqa: E402

PREFIX_SOURCE = (REPO_ROOT / "src" / "phaseshift" / "models" / "qwen35"
                 / "runtime" / "prefix_cache.cpp")
STATE_DIR = REPO_ROOT / "src" / "phaseshift" / "models" / "qwen35" / "state"

FORBIDDEN_IN_PREFIX = (
    "hipMemcpyDeviceToHost",
    "hipMemcpyHostToDevice",
    "hipStreamSynchronize",
    "hipDeviceSynchronize",
)


def main() -> int:
    checker = Checker("prefix-cache-no-host-transfer")

    checker.check("prefix source exists", PREFIX_SOURCE.is_file(),
                  str(PREFIX_SOURCE))
    if PREFIX_SOURCE.is_file():
        text = PREFIX_SOURCE.read_text(encoding="utf-8")
        for token in FORBIDDEN_IN_PREFIX:
            checker.check(f"prefix cache has no {token}", token not in text,
                          f"{token} present in {PREFIX_SOURCE.name}")
        checker.check("prefix cache uses D2D copy primitives",
                      "copy_page_from" in text and "copy_slot_from" in text)
        checker.check("host payload fields removed",
                      "std::vector<uint8_t>" not in text)

    # The KV / GDN page copy primitives must not introduce host transfers.
    for name in ("paged_kv_pool.cpp", "gdn_state_pool.cpp"):
        path = STATE_DIR / name
        if not path.is_file():
            checker.check(f"{name} exists", False, str(path))
            continue
        text = path.read_text(encoding="utf-8")
        checker.check(f"{name} has no host payload transfer",
                      "hipMemcpyHostToDevice" not in text
                      and "hipMemcpyDeviceToHost" not in text)
        checker.check(f"{name} uses D2D", "hipMemcpyDeviceToDevice" in text)

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
