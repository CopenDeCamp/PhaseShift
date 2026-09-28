#!/usr/bin/env python3
"""Gate 11A: mixed reasoning concurrency, cancellation and length finish."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ServerHarness, model_dir  # noqa: E402

PROMPT = "What is 2+2? Answer with only the number."
MARKUP = ("<think>", "</think>")
SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "r", "strict": True,
    "schema": {"type": "object", "properties": {"a": {"type": "string"}},
               "required": ["a"], "additionalProperties": False}}}


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode(errors="replace"))


def message(body):
    return (body.get("choices") or [{}])[0].get("message", {})


def main() -> int:
    checker = Checker("server-reasoning-concurrency")
    if not model_dir().is_dir():
        print(f"model dir not found: {model_dir()}", file=sys.stderr)
        return 2
    try:
        with ServerHarness(max_seq_len=2048, max_concurrent_requests=4,
                           kv_cache_capacity_tokens=0) as server:
            url = f"{server.base_url}/chat/completions"
            base = {"model": "phaseshift", "temperature": 0}

            payloads = {
                "a": dict(base, messages=[{"role": "user", "content": PROMPT}],
                          max_tokens=512, reasoning_effort="high"),
                "b": dict(base, messages=[{"role": "user", "content": PROMPT}],
                          max_tokens=64, reasoning_effort="none"),
                "c": dict(base, messages=[{"role": "user", "content": PROMPT}],
                          max_tokens=512, reasoning_effort="high"),
                "d": dict(base, messages=[{"role": "user", "content": PROMPT}],
                          max_tokens=128, reasoning_effort="none",
                          response_format=SCHEMA),
            }
            results: dict = {}
            threads = [threading.Thread(target=lambda k=k, p=p: results.__setitem__(
                k, post(url, p))) for k, p in payloads.items()]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for key in ("a", "b", "c", "d"):
                checker.check(f"{key}-completed", key in results, repr(results))
            for key in ("a", "c"):
                status, body = results[key]
                msgl = message(body)
                reasoning = msgl.get("reasoning") or ""
                content = msgl.get("content") or ""
                checker.check(f"{key}-reasoning", status == 200 and bool(reasoning.strip()),
                              f"{status} {repr(reasoning)[:120]}")
                checker.check(f"{key}-no-markup",
                              all(m not in reasoning and m not in content for m in MARKUP),
                              repr((reasoning + content)[:160]))
                checker.check(f"{key}-content", bool(content.strip()),
                              f"finish={(body.get('choices') or [{}])[0].get('finish_reason')}")

            status, body = results["b"]
            msgl = message(body)
            checker.check("b-none-no-reasoning", status == 200 and not msgl.get("reasoning"),
                          f"{status} {repr(msgl)[:160]}")
            checker.check("b-content", bool((msgl.get("content") or "").strip()),
                          repr(msgl)[:160])

            status, body = results["d"]
            content = message(body).get("content") or ""
            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                parsed = None
            checker.check("d-structured-json", status == 200 and isinstance(parsed, dict)
                          and "a" in parsed, f"{status} {repr(content)[:160]}")
            checker.check("d-no-reasoning", not message(body).get("reasoning"),
                          repr(message(body))[:160])

            status, body = post(url, dict(base, messages=[{"role": "user", "content": PROMPT}],
                                          max_tokens=32, reasoning_effort="high"))
            choice = (body.get("choices") or [{}])[0]
            msgl = message(body)
            reasoning = msgl.get("reasoning") or ""
            content = msgl.get("content") or ""
            checker.check("length-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            checker.check("length-no-markup",
                          all(m not in reasoning and m not in content for m in MARKUP),
                          repr((reasoning + content)[:160]))
            checker.check("length-finish", choice.get("finish_reason") == "length",
                          repr(choice.get("finish_reason")))

            req = urllib.request.Request(
                url, data=json.dumps(dict(
                    base, messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=512, reasoning_effort="high", stream=True)).encode(),
                headers={"Content-Type": "application/json"})
            response = urllib.request.urlopen(req, timeout=900)
            read = 0
            for _raw in response:
                read += 1
                if read >= 5:
                    break
            response.close()

            status, body = post(url, dict(base, messages=[{"role": "user", "content": PROMPT}],
                                          max_tokens=64, reasoning_effort="none"))
            checker.check("after-cancel-ok", status == 200, f"{status} {repr(body)[:160]}")
            checker.check("after-cancel-clean",
                          not message(body).get("reasoning")
                          and bool((message(body).get("content") or "").strip()),
                          repr(message(body))[:160])
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
