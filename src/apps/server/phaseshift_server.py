#!/usr/bin/env python3
"""PhaseShift Server launcher.

User entrypoint:

    ./build/phaseshift-server --model-dir /path/to/Qwen3.5-model --port 8000

The launcher owns the LocalAI runtime, hands it a generated model config that
routes the ``phaseshift`` model to this repository's LocalAI backend, and waits
until the OpenAI-compatible endpoint is actually serving the model.

LocalAI is an internal implementation detail. Users never start it themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_BACKEND_REL = Path("phaseshift-server-lib") / "backend" / "phaseshift_backend.py"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def find_server_lib(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.is_file():
            return candidate
        raise SystemExit(f"backend executable not found: {candidate}")
    env = os.environ.get("PHASESHIFT_SERVER_BACKEND")
    if env:
        candidate = Path(env).resolve()
        if candidate.is_file():
            return candidate
    here = Path(__file__).resolve().parent
    candidate = here / REPO_BACKEND_REL
    if candidate.is_file():
        return candidate
    repo = Path(__file__).resolve()
    for parent in repo.parents:
        candidate = parent / "src" / "apps" / "server" / "backend" / "phaseshift_backend.py"
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "PhaseShift Server support files are missing.\n"
        "Expected the staged backend at build/phaseshift-server-lib/backend/."
    )


def find_localai(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).is_file():
            return str(Path(explicit).resolve())
        raise SystemExit(f"LocalAI binary not found: {explicit}")
    env = os.environ.get("PHASESHIFT_LOCALAI_BINARY")
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    here = Path(__file__).resolve().parent
    for name in ("local-ai", "local-ai.exe"):
        for base in (here, here / "runtime", here / "localai"):
            candidate = base / name
            if candidate.is_file():
                return str(candidate)
    which = shutil.which("local-ai")
    if which:
        return which
    raise SystemExit(
        "PhaseShift Server requires the bundled LocalAI runtime.\n"
        "Development override: --localai-binary PATH"
    )


def write_model_config(path: Path, opts) -> None:
    def yaml_str(value: str) -> str:
        return json.dumps(value)

    options = [
        f"phaseshift_model_dir:{opts.model_dir}",
        f"phaseshift_compute:{opts.compute}",
        f"phaseshift_arena_gib:{opts.arena_gib}",
        f"phaseshift_device:{opts.device}",
        f"phaseshift_kv_cache_dtype:{opts.kv_cache_dtype}",
        f"phaseshift_max_concurrent_requests:{opts.max_concurrent_requests}",
        f"phaseshift_kv_cache_capacity_tokens:{opts.kv_cache_capacity_tokens}",
        f"phaseshift_prefix_cache_capacity_tokens:{opts.prefix_cache_capacity_tokens}",
        f"phaseshift_prefix_cache_max_entries:{opts.prefix_cache_max_entries}",
        f"phaseshift_default_max_output_tokens:{opts.default_max_output_tokens}",
    ]
    if opts.dflash2_model_dir:
        options.append(f"phaseshift_dflash2_model_dir:{opts.dflash2_model_dir}")
        options.append(f"phaseshift_dflash2_drafts:{opts.dflash2_drafts}")
    if opts.page_tokens:
        options.append(f"phaseshift_page_tokens:{opts.page_tokens}")
    if opts.verify_weights:
        options.append("phaseshift_verify_weights:1")

    lines = [
        f"name: {yaml_str(opts.model_name)}",
        "backend: \"phaseshift\"",
        f"model: {yaml_str(opts.model_name)}",
        f"context_size: {opts.max_seq_len}",
        "template:",
        "  use_tokenizer_template: true",
        "options:",
    ]
    for option in options:
        lines.append(f"  - {yaml_str(option)}")
    path.write_text("\n".join(lines) + "\n")


def poll_models(url: str, model_name: str, proc: subprocess.Popen, timeout: float,
                should_stop=None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for entry in payload.get("data", []):
                if entry.get("id") == model_name:
                    return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(0.5)
    return False


PROBE_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "__phaseshift_invalid_type__"}},
    "required": ["x"],
    "additionalProperties": False,
}


def _probe_http(address: str, path: str, payload: dict, timeout: float,
                needles: tuple[str, ...]) -> tuple[bool, str]:
    request = urllib.request.Request(
        f"http://{address}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        return False, f"probe request failed: {exc}"
    if status != 400:
        return False, f"expected HTTP 400, got HTTP {status}: {body[:400]}"
    lowered = body.lower()
    if not any(needle in lowered for needle in needles):
        return False, (
            "probe error body did not reference "
            f"{'/'.join(needles)}: {body[:400]}")
    return True, ""


def probe_structured_fail_closed(address: str, model_name: str,
                                 timeout: float = 60.0) -> tuple[bool, str]:
    chat_ok, chat_reason = _probe_http(
        address,
        "/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "probe"}],
            "temperature": 0,
            "max_tokens": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "phaseshift_probe",
                    "strict": True,
                    "schema": PROBE_SCHEMA,
                },
            },
        },
        timeout,
        ("response_format", "json_schema"),
    )
    if not chat_ok:
        return False, f"chat: {chat_reason}"

    responses_ok, responses_reason = _probe_http(
        address,
        "/v1/responses",
        {
            "model": model_name,
            "input": "probe",
            "temperature": 0,
            "max_output_tokens": 1,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "phaseshift_probe",
                    "schema": PROBE_SCHEMA,
                    "strict": True,
                },
            },
        },
        timeout,
        ("text.format", "json_schema"),
    )
    if not responses_ok:
        return False, f"responses: {responses_reason}"

    return True, ""


def tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description="PhaseShift Server")
    parser.add_argument("--model-dir", required=True, help="Qwen3.5 model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--arena-gib", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", default="bf16",
                        choices=("bf16", "fp8_e4m3", "psq4", "psq8"))
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--max-concurrent-requests", type=int, default=4)
    parser.add_argument("--kv-cache-capacity-tokens", type=int, default=0)
    parser.add_argument("--prefix-cache-capacity-tokens", type=int, default=16384)
    parser.add_argument("--prefix-cache-max-entries", type=int, default=8)
    parser.add_argument("--default-max-output-tokens", type=int, default=4096)
    parser.add_argument("--dflash2-model-dir", default=None,
                        help="DFlash2 draft model directory (enables speculative decoding)")
    parser.add_argument("--dflash2-drafts", type=int, default=7)
    parser.add_argument("--compute", default=None, help="phaseshift-compute path")
    parser.add_argument("--model-name", default="phaseshift")
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--localai-binary", default=None, help="developer override")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    opts = parser.parse_args()
    if opts.max_concurrent_requests < 1:
        _log("--max-concurrent-requests must be >= 1")
        return 2
    if opts.default_max_output_tokens < 1:
        _log("--default-max-output-tokens must be >= 1")
        return 2
    if opts.prefix_cache_capacity_tokens > 0 and opts.prefix_cache_max_entries < 1:
        _log("--prefix-cache-max-entries must be >= 1 when the prefix cache is enabled")
        return 2

    model_dir = Path(opts.model_dir).resolve()
    if not model_dir.is_dir():
        _log(f"model directory not found: {model_dir}")
        return 2
    opts.model_dir = str(model_dir)

    if opts.dflash2_model_dir:
        dflash2_dir = Path(opts.dflash2_model_dir).resolve()
        if not dflash2_dir.is_dir():
            _log(f"dflash2 model directory not found: {dflash2_dir}")
            return 2
        opts.dflash2_model_dir = str(dflash2_dir)
        if opts.prefix_cache_capacity_tokens > 0:
            _log("--dflash2-model-dir disables the prefix cache "
                 "(target prefix cache is unsupported with speculative decoding)")
            opts.prefix_cache_capacity_tokens = 0
        if opts.max_concurrent_requests != 1:
            _log(f"--dflash2-model-dir forces --max-concurrent-requests 1 "
                 f"(requested {opts.max_concurrent_requests})")
            opts.max_concurrent_requests = 1
        if opts.dflash2_drafts < 1:
            _log("--dflash2-drafts must be >= 1")
            return 2

    here = Path(__file__).resolve().parent
    if opts.compute:
        compute = str(Path(opts.compute).resolve())
    elif os.environ.get("PHASESHIFT_COMPUTE"):
        compute = os.environ["PHASESHIFT_COMPUTE"]
    else:
        candidate = here / "phaseshift-compute"
        compute = str(candidate) if candidate.is_file() else "phaseshift-compute"
    if not Path(compute).is_file():
        _log(f"phaseshift-compute not found: {compute}")
        return 2

    backend_exec = find_server_lib(None)
    localai = find_localai(opts.localai_binary)

    workdir = Path(tempfile.mkdtemp(prefix="phaseshift-server-"))
    log_path = workdir / "localai.log"
    (workdir / "models").mkdir()
    (workdir / "backends").mkdir()
    (workdir / "data").mkdir()
    (workdir / "configuration").mkdir()
    model_config = workdir / "models" / f"{opts.model_name}.yaml"
    write_model_config(model_config, argparse.Namespace(**{**vars(opts), "compute": compute}))

    address = f"{opts.host}:{opts.port}"
    command = [
        localai, "run",
        "--address", address,
        "--models-path", str(workdir / "models"),
        "--backends-path", str(workdir / "backends"),
        "--data-path", str(workdir / "data"),
        "--localai-config-dir", str(workdir / "configuration"),
        "--models-config-file", str(model_config),
        "--external-grpc-backends", f"phaseshift:{backend_exec}",
        "--max-concurrent-backend-requests", str(opts.max_concurrent_requests),
    ]

    worker_env = os.environ.copy()
    worker_env["PHASESHIFT_MAX_CONCURRENT_REQUESTS"] = str(opts.max_concurrent_requests)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                            env=worker_env)

    stop_requested = {"flag": False}

    def request_stop(signum=None, frame=None):
        # Signal handlers must not touch subprocess.wait(): the main thread may
        # already hold subprocess's internal wait lock, and re-entering it
        # deadlocks. Only flip a flag; cleanup happens on the main path.
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def stop_process(timeout: float = 30.0) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    models_url = f"http://{address}/v1/models"
    ready = poll_models(models_url, opts.model_name, proc, opts.startup_timeout,
                        should_stop=lambda: stop_requested["flag"])
    if stop_requested["flag"]:
        stop_process()
        shutil.rmtree(workdir, ignore_errors=True)
        return 0
    if not ready:
        _log("PhaseShift Server failed to become ready.")
        _log(f"LocalAI exit status: {proc.poll()}")
        _log(f"LocalAI log: {log_path}")
        tail_text = tail(log_path)
        if tail_text:
            _log("--- LocalAI log tail ---")
            _log(tail_text)
        stop_process(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    probe_ok, probe_reason = probe_structured_fail_closed(address, opts.model_name)
    if not probe_ok:
        _log("")
        _log("PhaseShift Server requires a LocalAI runtime with")
        _log("fail-closed Structured Output support (Chat and Responses).")
        _log("")
        _log("The selected LocalAI binary accepted an invalid")
        _log("json_schema without rejecting the request.")
        _log("")
        _log("Use the PhaseShift bundled LocalAI runtime.")
        _log(f"Probe failure: {probe_reason}")
        stop_process(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    _log("")
    _log("PhaseShift Server")
    _log("")
    _log(f"Model:        {opts.model_name} ({model_dir.name})")
    _log(f"Endpoint:     http://{address}/v1")
    _log(f"Context:      {opts.max_seq_len}")
    _log(f"Concurrency:  {opts.max_concurrent_requests}")
    _log(f"KV dtype:     {opts.kv_cache_dtype}")
    _log(f"KV capacity:  {'auto' if opts.kv_cache_capacity_tokens == 0 else opts.kv_cache_capacity_tokens}")
    if opts.prefix_cache_capacity_tokens > 0:
        _log(f"Prefix cache: {opts.prefix_cache_capacity_tokens} tokens / "
             f"{opts.prefix_cache_max_entries} entries")
    else:
        _log("Prefix cache: off")
    if opts.dflash2_model_dir:
        _log(f"Speculative:  DFlash2 ({opts.dflash2_drafts} drafts)")
        _log(f"Draft model:  {Path(opts.dflash2_model_dir).name}")
    _log(f"Default output: {opts.default_max_output_tokens}")
    _log("")
    _log("Capabilities:")
    _log("  Chat:         yes")
    _log("  Streaming:    yes")
    if opts.dflash2_model_dir:
        _log("  Tool calling: no (DFlash2 speculative decoding)")
        _log("  Structured:   no (DFlash2 speculative decoding)")
        _log("  Greedy only:  yes (temperature must be 0)")
    else:
        _log("  Tool calling: yes")
        _log("  Structured:   fail-closed (Chat + Responses)")
    _log("  Responses API:yes")
    _log("  Reasoning:    yes (opt-in)")
    _log("")
    _log("Compatible:")
    _log("  OpenAI SDK")
    _log("  OpenCode")
    _log("  Codex CLI")
    _log("")
    _log("Ready.")

    while proc.poll() is None and not stop_requested["flag"]:
        time.sleep(0.2)

    if stop_requested["flag"]:
        _log("PhaseShift Server shutting down...")
    stop_process()
    log_file.close()
    returncode = proc.returncode if proc.returncode is not None else 0
    shutil.rmtree(workdir, ignore_errors=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
