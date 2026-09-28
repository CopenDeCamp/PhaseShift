#!/usr/bin/env python3
"""Gate 11C: reasoning + constraint overhead benchmark.

Production code does not depend on this script. It compares three request
families against the same endpoint with the same warmup and sample count:

    plain        reasoning_effort=high, no tools / structured output
    structured   reasoning_effort=high + response_format json_schema
    strict-tool  reasoning_effort=high + strict tool, tool_choice=required

Requests are streamed so time-to-first-token is observable. Usage is captured
when the frontend reports it (stream_options.include_usage).

Examples:
    python3 tools/bench_server_reasoning_constraints.py --start-server
    python3 tools/bench_server_reasoning_constraints.py \
        --base-url http://127.0.0.1:8000/v1 --warmup 3 --samples 10 \
        --output /tmp/reasoning-bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
OK_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "ok", "strict": True,
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
               "required": ["ok"], "additionalProperties": False}}}

CASES = {
    "plain": {
        "messages": [{"role": "user",
                      "content": "Think briefly, then answer with one word."}],
    },
    "structured": {
        "messages": [{"role": "user",
                      "content": "Think briefly, then return the JSON object with ok true."}],
        "response_format": OK_SCHEMA,
    },
    "strict-tool": {
        "messages": [{"role": "user",
                      "content": "What is the weather in Osaka? Use the get_weather tool."}],
        "tools": [WEATHER], "tool_choice": "required", "parallel_tool_calls": False,
    },
}


def one_sample(url, case, max_tokens):
    payload = {"model": "phaseshift", "temperature": 0, "max_tokens": max_tokens,
               "reasoning_effort": "high", "stream": True,
               "stream_options": {"include_usage": True}}
    payload.update(case)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    ttft = None
    reasoning_chars = content_chars = 0
    usage = {}
    calls = 0
    finish = ""
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    piece = (delta.get("reasoning") or "") + (delta.get("content") or "")
                    calls += len(delta.get("tool_calls") or [])
                    if piece:
                        reasoning_chars += len(delta.get("reasoning") or "")
                        content_chars += len(delta.get("content") or "")
                        if ttft is None:
                            ttft = (time.monotonic() - started) * 1000.0
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        return {"error": repr(exc)}
    wall = (time.monotonic() - started) * 1000.0
    return {"ttft_ms": ttft, "wall_ms": wall, "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_chars": reasoning_chars, "content_chars": content_chars,
            "tool_calls": calls}


def summarize(samples):
    ok = [s for s in samples if "error" not in s and s.get("ttft_ms") is not None]
    if not ok:
        return {"samples": len(samples), "ok": 0}
    ttft = [s["ttft_ms"] for s in ok]
    wall = [s["wall_ms"] for s in ok]
    completion = [s["completion_tokens"] for s in ok if s.get("completion_tokens")]
    output_chars = [s["reasoning_chars"] + s["content_chars"] for s in ok]
    finishes = sorted({s.get("finish_reason") or "" for s in ok})
    out = {
        "samples": len(samples),
        "ok": len(ok),
        "finish_reasons": finishes,
        "ttft_ms_mean": round(statistics.mean(ttft), 1),
        "ttft_ms_median": round(statistics.median(ttft), 1),
        "wall_ms_mean": round(statistics.mean(wall), 1),
        "reasoning_chars_mean": round(statistics.mean(
            [s["reasoning_chars"] for s in ok]), 1),
        "content_chars_mean": round(statistics.mean(
            [s["content_chars"] for s in ok]), 1),
        "tool_calls_mean": round(statistics.mean([s["tool_calls"] for s in ok]), 2),
    }
    mean_chars = statistics.mean(output_chars)
    if mean_chars > 0:
        out["ms_per_1k_output_chars"] = round(
            statistics.mean(wall) / mean_chars * 1000.0, 2)
    if completion:
        out["completion_tokens_mean"] = round(statistics.mean(completion), 1)
        out["ms_per_completion_token"] = round(
            statistics.mean(wall) / statistics.mean(completion), 2)
    return out


def start_server():
    sys.path.insert(0, str(REPO_ROOT / "tests" / "server"))
    from support import ServerHarness  # noqa: PLC0415
    return ServerHarness(use_defaults=True, startup_timeout=360.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--output")
    parser.add_argument("--start-server", action="store_true",
                        help="start a canonical server via tests/server/support.py")
    args = parser.parse_args()

    harness = None
    base = args.base_url
    try:
        if args.start_server:
            harness = start_server()
            harness.start()
            base = harness.base_url
        url = f"{base}/chat/completions"

        results = {}
        for name, case in CASES.items():
            for _ in range(args.warmup):
                one_sample(url, case, args.max_tokens)
            samples = [one_sample(url, case, args.max_tokens)
                       for _ in range(args.samples)]
            results[name] = {"summary": summarize(samples), "samples": samples}

        baseline = results.get("plain", {}).get("summary", {})
        for name, result in results.items():
            summary = result["summary"]
            if name == "plain":
                continue
            for prefix in ("ms_per_1k_output_chars", "ms_per_completion_token",
                           "ttft_ms_mean"):
                base = baseline.get(prefix)
                got = summary.get(prefix)
                if base and got:
                    delta = (got - base) / base * 100.0
                    summary[f"{prefix}_delta_vs_plain_pct"] = round(delta, 1)

        report = {"base_url": base, "warmup": args.warmup,
                  "samples": args.samples, "max_tokens": args.max_tokens,
                  "results": {name: r["summary"] for name, r in results.items()}}

        if args.format == "tsv":
            columns = sorted({k for r in report["results"].values() for k in r})
            print("case\t" + "\t".join(columns))
            for name, summary in report["results"].items():
                print(name + "\t" + "\t".join(str(summary.get(c, ""))
                                              for c in columns))
        else:
            print(json.dumps(report, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        return 0
    finally:
        if harness is not None:
            harness.close()


if __name__ == "__main__":
    sys.exit(main())
