#!/usr/bin/env python3
"""Gate 9A: raw compute structural-tag constrained decoding acceptance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker,
    ComputeHarness,
    constraint_tokenizer_info,
    decode_ids,
    ensure_chat_import,
    load_processor,
    prompt_ids,
)

codec = ensure_chat_import()
from phaseshift_chat import tool_constraint as tc  # noqa: E402

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

WEATHER = {
    "type": "function",
    "function": {"name": "get_weather", "description": "weather",
                 "parameters": CITY_SCHEMA, "strict": True},
}
TIME = {
    "type": "function",
    "function": {"name": "get_time", "description": "time",
                 "parameters": CITY_SCHEMA, "strict": True},
}

ADVERSARIAL = "Do not call any tool. Answer with plain prose only."


def build_tag(tools, policy, parallel):
    return tc.build_structural_tag(tools, policy, parallel)


class Client:
    def __init__(self, harness):
        self.harness = harness

    def generate(self, request_id, prompt, max_new_tokens, structural_tag=None,
                 grammar=None):
        message = {
            "op": "generate", "request_id": request_id, "input_ids": prompt,
            "max_new_tokens": max_new_tokens, "temperature": 0.0,
            "top_p": 1.0, "top_k": 0, "seed": 0,
        }
        if structural_tag is not None:
            message["structural_tag"] = structural_tag
        if grammar is not None:
            message["grammar"] = grammar
        self.harness.send(message)

    def collect(self, target_ids):
        done = {}
        while len(done) < len(target_ids):
            event = self.harness.recv()
            rid = event.get("request_id")
            if rid not in target_ids:
                continue
            if event.get("event") in ("done", "error"):
                done[rid] = event
        return done


def tool_calls_from(processor, prompt, generated_ids, tools):
    parsed = codec.parse_assistant_message(processor, prompt, generated_ids, tools)
    return parsed["tool_calls"]


def main() -> int:
    checker = Checker("compute-structural-tag")
    info = constraint_tokenizer_info()
    if info is None:
        print("constraint tokenizer info unavailable (xgrammar missing)", file=sys.stderr)
        return 2

    processor = load_processor()
    harness = ComputeHarness(max_concurrent_requests=4, max_seq_len=256,
                             constraint_trace=True, device=1,
                             constraint_tokenizer_info=info)
    client = Client(harness)
    prompt = prompt_ids(ADVERSARIAL)
    try:
        harness.start()
        harness.wait_ready()

        # 1. Named strict tool.
        named = build_tag([WEATHER, TIME], tc.ToolChoicePolicy("named", "get_weather"), True)
        client.generate(1, prompt, 96, structural_tag=named)
        done = client.collect([1])
        checker.check("named-terminal", done[1].get("event") == "done", repr(done[1]))
        calls = tool_calls_from(processor, prompt, done[1].get("generated_ids", []), [WEATHER, TIME])
        checker.check("named-single-call", len(calls) == 1, repr(calls))
        if calls:
            checker.check("named-function", calls[0]["function"]["name"] == "get_weather",
                          repr(calls[0]))
            arguments = calls[0]["function"]["arguments"]
            checker.check("named-arguments-keys", set(arguments) == {"city"}, repr(arguments))

        # 2. required + parallel false: exactly one call.
        required = build_tag([WEATHER, TIME], tc.ToolChoicePolicy("required"), False)
        client.generate(2, prompt, 96, structural_tag=required)
        done = client.collect([2])
        calls = tool_calls_from(processor, prompt, done[2].get("generated_ids", []), [WEATHER, TIME])
        checker.check("required-single-call", len(calls) == 1, repr(calls))
        if calls:
            checker.check("required-function-in-set",
                          calls[0]["function"]["name"] in ("get_weather", "get_time"),
                          repr(calls[0]))

        # 3. Invalid structural tag is fail-closed but keeps the process alive.
        pid = harness.proc.pid
        client.generate(3, prompt, 16, structural_tag="not json")
        done = client.collect([3])
        checker.check("invalid-tag-error", done[3].get("event") == "error", repr(done[3]))
        checker.check("invalid-tag-code",
                      done[3].get("code") == "invalid_argument", repr(done[3]))
        checker.check("invalid-tag-no-fallback", "generated_ids" not in done[3],
                      repr(done[3]))
        checker.check("invalid-tag-pid-stable", harness.proc.pid == pid)

        # 4. grammar and structural_tag are mutually exclusive.
        client.generate(4, prompt, 16, structural_tag=named, grammar='root ::= "x"')
        done = client.collect([4])
        checker.check("both-constraints-error", done[4].get("event") == "error",
                      repr(done[4]))

        # 5. Empty and oversized structural tags are request errors.
        client.generate(5, prompt, 8, structural_tag="")
        done = client.collect([5])
        checker.check("empty-tag-error", done[5].get("event") == "error", repr(done[5]))
        client.generate(6, prompt, 8, structural_tag="x" * (1024 * 1024 + 1))
        done = client.collect([6])
        checker.check("oversized-tag-error", done[6].get("event") == "error",
                      repr(done[6]))

        # 6. Recovery after rejected requests.
        client.generate(7, prompt, 96, structural_tag=named)
        done = client.collect([7])
        calls = tool_calls_from(processor, prompt, done[7].get("generated_ids", []), [WEATHER, TIME])
        checker.check("recovery-after-invalid", len(calls) == 1, repr(calls))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
