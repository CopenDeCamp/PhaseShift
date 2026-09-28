#!/usr/bin/env python3
"""Gate 8R: XGrammar dependency + TokenizerInfo serialization contract.

Server structured generation is a supported capability, so a missing or
mismatched Python xgrammar is a hard failure (no skip).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    REPO_ROOT,
    Checker,
    build_dir,
    constraint_tokenizer_info,
    generation_stop_probe,
    xgrammar_version,
)

EXPECTED_PYTHON_XGRAMMAR = "0.2.5.post1"


def main() -> int:
    checker = Checker("constraint-dependency")

    version = xgrammar_version()
    checker.check("python-xgrammar-present", version is not None, repr(version))
    checker.check("python-xgrammar-pinned", version == EXPECTED_PYTHON_XGRAMMAR,
                  f"got={version} expected={EXPECTED_PYTHON_XGRAMMAR}")

    version_file = REPO_ROOT / "vendor" / "xgrammar" / "VERSION"
    checker.check("native-version-file-present", version_file.is_file(),
                  str(version_file))
    native_lines = version_file.read_text().split("\n") if version_file.is_file() else []
    native_tag = native_lines[0].strip() if native_lines else ""
    native_commit = native_lines[1].strip() if len(native_lines) > 1 else ""
    checker.check("native-tag-v0.2.5", native_tag == "v0.2.5", repr(native_tag))
    checker.check("native-commit-2ea71da", native_commit == "2ea71da", repr(native_commit))

    sidecar = constraint_tokenizer_info()
    checker.check("sidecar-generation", sidecar is not None, repr(sidecar))
    if sidecar is not None:
        data = json.loads(Path(sidecar).read_text())
        checker.check("sidecar-json-object", isinstance(data, dict), type(data).__name__)
        checker.check("sidecar-nonempty", len(json.dumps(data)) > 0)

    probe = generation_stop_probe()
    stop_token = probe["model_config_eos"]

    binary = build_dir() / "tests" / "test_constraint_mask"
    checker.check("native-constraint-test-binary", binary.is_file(), str(binary))
    if sidecar is not None and binary.is_file() and stop_token is not None:
        env = dict(os.environ)
        env["PHASESHIFT_CONSTRAINT_SIDECAR"] = str(sidecar)
        env["PHASESHIFT_CONSTRAINT_STOP_TOKEN"] = str(int(stop_token))
        result = subprocess.run([str(binary)], capture_output=True, text=True,
                                timeout=120, env=env)
        checker.check("native-deserialize-python-sidecar", result.returncode == 0,
                      f"rc={result.returncode} out={result.stdout[-400:]} err={result.stderr[-400:]}")

    status_doc = REPO_ROOT / "docs" / "developer" / "server_status.md"
    checker.check("server-status-doc", status_doc.is_file(), str(status_doc))
    if status_doc.is_file():
        text = status_doc.read_text()
        for needle in ("BLOCKED_EXTERNAL", "Full JSON Schema semantics",
                       "Chat structured fail-closed", "Grammar-Constrained Decode"):
            checker.check(f"status-doc-{needle}", needle in text, needle)

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
