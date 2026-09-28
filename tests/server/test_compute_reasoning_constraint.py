#!/usr/bin/env python3
"""Gate 11B: raw compute reasoning envelope + constrained final phase."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import (  # noqa: E402
    Checker, ComputeHarness, constraint_tokenizer_info,
    ensure_chat_import, load_processor,
)

codec = ensure_chat_import()
from phaseshift_chat import tool_constraint as tc  # noqa: E402

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
OK_GRAMMAR = 'root ::= "{" ws "\\"ok\\"" ws ":" ws ("true" | "false") ws "}"\nws ::= [ \\t\\n]*'
TOOL_PROMPT = "What is the weather in Osaka? Use the get_weather tool."
MARKUP = ("<think>", "</think>", "<tool_call>", "<function=", "</function>")


class Client:
    def __init__(self, harness):
        self.harness = harness

    def generate(self, rid, prompt, max_new_tokens, structural_tag=None, grammar=None):
        message = {"op": "generate", "request_id": rid, "input_ids": prompt,
                   "max_new_tokens": max_new_tokens, "temperature": 0.0,
                   "top_p": 1.0, "top_k": 0, "seed": 0}
        if structural_tag is not None:
            message["structural_tag"] = structural_tag
        if grammar is not None:
            message["grammar"] = grammar
        self.harness.send(message)

    def collect(self, ids):
        done = {}
        while len(done) < len(ids):
            event = self.harness.recv()
            rid = event.get("request_id")
            if rid in ids and event.get("event") in ("done", "error"):
                done[rid] = event
        return done


def parse_final(processor, generated_ids):
    text = codec.decode_generated(processor, generated_ids)
    reasoning, final_text = codec.split_reasoning_final(text)
    return text, reasoning, final_text


def main() -> int:
    checker = Checker("compute-reasoning-constraint")
    info = constraint_tokenizer_info()
    if info is None:
        print("constraint tokenizer info unavailable (xgrammar missing)", file=sys.stderr)
        return 2
    processor = load_processor()
    harness = ComputeHarness(max_concurrent_requests=1, max_seq_len=4096,
                             constraint_trace=True, device=0,
                             constraint_tokenizer_info=info)
    client = Client(harness)
    try:
        harness.start()
        harness.wait_ready()
        # Case A: reasoning envelope + strict required tool.
        tool_tag = tc.build_reasoning_structural_tag(
            [WEATHER], tc.ToolChoicePolicy("required"), False, "", False)
        prompt = codec.prompt_ids(
            processor, [{"role": "user", "content": TOOL_PROMPT}],
            tools=[WEATHER], enable_thinking=True)
        client.generate(1, prompt, 256, structural_tag=tool_tag)
        done = client.collect([1])
        checker.check("A-done", done[1].get("event") == "done", repr(done[1])[:200])
        if done[1].get("event") == "done":
            text, reasoning, final_text = parse_final(
                processor, done[1].get("generated_ids", []))
            checker.check("A-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            checker.check("A-delimiter", "</think>" in text, repr(text)[:160])
            parsed = processor.parse_response(
                final_text, codec.TOOL_RESPONSE_TEMPLATE, prefix="")
            calls = parsed.get("tool_calls") or []
            checker.check("A-call", len(calls) == 1
                          and calls[0]["function"]["name"] == "get_weather",
                          repr(calls))
            if calls:
                args = calls[0]["function"]["arguments"]
                checker.check("A-args", isinstance(args, dict) and set(args) <= {"city"},
                              repr(args))
            visible = reasoning + (parsed.get("content") or "")
            checker.check("A-no-markup",
                          all(m not in visible for m in MARKUP),
                          repr(visible[-80:]))

        # Case B: reasoning envelope + structured grammar.
        grammar_tag = tc.build_reasoning_structural_tag(
            None, tc.ToolChoicePolicy("auto"), True, OK_GRAMMAR, False)
        prompt = codec.prompt_ids(
            processor, [{"role": "user",
                         "content": 'Produce the JSON object {"ok":true}.'}],
            enable_thinking=True)
        client.generate(2, prompt, 1024, structural_tag=grammar_tag)
        done = client.collect([2])
        checker.check("B-done", done[2].get("event") == "done", repr(done[2])[:200])
        if done[2].get("event") == "done":
            text, reasoning, final_text = parse_final(
                processor, done[2].get("generated_ids", []))
            checker.check("B-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            stripped = final_text.strip()
            parsed = None
            try:
                parsed = json.loads(stripped)
            except Exception:  # noqa: BLE001
                parsed = None
            checker.check("B-grammar", isinstance(parsed, dict) and "ok" in parsed,
                          repr(final_text)[:160])

        # Case C: composite reasoning + structured + tools (auto) - tool branch.
        composite = tc.build_reasoning_structural_tag(
            [WEATHER], tc.ToolChoicePolicy("auto"), True, OK_GRAMMAR, True)
        prompt = codec.prompt_ids(
            processor, [{"role": "user", "content": TOOL_PROMPT}],
            tools=[WEATHER], enable_thinking=True)
        client.generate(3, prompt, 512, structural_tag=composite)
        done = client.collect([3])
        checker.check("C-done", done[3].get("event") == "done", repr(done[3])[:200])
        if done[3].get("event") == "done":
            text, reasoning, final_text = parse_final(
                processor, done[3].get("generated_ids", []))
            checker.check("C-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            parsed = processor.parse_response(
                final_text, codec.TOOL_RESPONSE_TEMPLATE, prefix="")
            calls = parsed.get("tool_calls") or []
            text_out = final_text.strip()
            is_json = False
            try:
                is_json = isinstance(json.loads(text_out), dict)
            except Exception:  # noqa: BLE001
                is_json = False
            checker.check("C-outcome",
                          (bool(calls)
                           and all(c["function"]["name"] == "get_weather" for c in calls))
                          or is_json,
                          f"calls={[c['function']['name'] for c in calls]} "
                          f"final={repr(final_text)[:80]}")

        # Case D: composite reasoning + structured + tools (auto) - text branch.
        prompt = codec.prompt_ids(
            processor, [{"role": "user",
                         "content": "Do not call any tool. Return the JSON object with ok true."}],
            tools=[WEATHER], enable_thinking=True)
        client.generate(4, prompt, 512, structural_tag=composite)
        done = client.collect([4])
        checker.check("D-done", done[4].get("event") == "done", repr(done[4])[:200])
        if done[4].get("event") == "done":
            text, reasoning, final_text = parse_final(
                processor, done[4].get("generated_ids", []))
            checker.check("D-reasoning", bool(reasoning.strip()), repr(reasoning)[:120])
            parsed = processor.parse_response(
                final_text, codec.TOOL_RESPONSE_TEMPLATE, prefix="")
            calls = parsed.get("tool_calls") or []
            stripped = final_text.strip()
            is_json = False
            try:
                is_json = isinstance(json.loads(stripped), dict)
            except Exception:  # noqa: BLE001
                is_json = False
            checker.check("D-outcome", is_json
                          or (bool(calls)
                              and all(c["function"]["name"] == "get_weather"
                                      for c in calls)),
                          f"calls={[c['function']['name'] for c in calls]} "
                          f"final={repr(final_text)[:80]}")
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    finally:
        harness.close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
