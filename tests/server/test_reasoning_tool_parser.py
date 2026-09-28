#!/usr/bin/env python3
"""Gate 11B: reasoning + tool/content parser and constraint builder units."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ensure_chat_import, load_processor  # noqa: E402

codec = ensure_chat_import()
from phaseshift_chat import tool_constraint as tc  # noqa: E402

CITY = {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": False}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "weather", "parameters": CITY, "strict": True}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "time", "parameters": CITY, "strict": True}}

def call(name, city):
    return (f"<tool_call>\n<function={name}>\n<parameter=city>\n{city}\n"
            "</parameter>\n</function>\n</tool_call>")


def suffix(previous, current):
    if current.startswith(previous):
        return current[len(previous):]
    limit = min(len(previous), len(current))
    index = 0
    while index < limit and previous[index] == current[index]:
        index += 1
    return current[index:]


def stream_split(processor, raw):
    factory = codec._response_parser_factory(processor)
    parser = factory.get_response_parser(
        response_template=codec.TOOL_RESPONSE_TEMPLATE, prefix="")
    last_reasoning = last_final = ""
    events = []
    for index in range(1, len(raw) + 1):
        reasoning, final_text = codec.split_reasoning_partial(raw[:index])
        final_delta = suffix(last_final, final_text)
        last_reasoning, last_final = reasoning, final_text
        if final_delta:
            events.extend(parser.feed(final_delta))
    reasoning, final_text = codec.split_reasoning_final(raw)
    final_delta = suffix(last_final, final_text)
    last_reasoning, last_final = reasoning, final_text
    if final_delta:
        events.extend(parser.feed(final_delta))
    message, final_events = parser.finalize()
    events.extend(final_events)
    return last_reasoning, last_final, message, events


def parser_cases(checker, processor):
    # reasoning + final content
    raw = "Some reasoning here.\n</think>\n\nFinal answer."
    reasoning, final_text, message, _ = stream_split(processor, raw)
    checker.check("content-reasoning", reasoning.strip() == "Some reasoning here.",
                  repr(reasoning))
    checker.check("content-final", "Final answer." in final_text
                  and (message.get("content") or "").strip() == "Final answer.",
                  repr((message.get("content"), final_text)))
    checker.check("content-no-calls", not message.get("tool_calls"), repr(message))

    # reasoning + single tool
    raw = "I will call the weather tool.\n</think>\n\n" + call("get_weather", "Osaka")
    reasoning, final_text, message, _ = stream_split(processor, raw)
    calls = message.get("tool_calls") or []
    checker.check("single-call", len(calls) == 1
                  and calls[0]["function"]["name"] == "get_weather"
                  and calls[0]["function"]["arguments"] == {"city": "Osaka"},
                  repr(calls))
    checker.check("single-reasoning", "weather tool" in reasoning, repr(reasoning))
    checker.check("single-content-clean",
                  "<tool_call>" not in (message.get("content") or ""),
                  repr(message.get("content")))

    # reasoning + parallel tools
    raw = ("reasoning\n</think>\n\n" + call("get_weather", "Osaka") + "\n"
           + call("get_time", "Tokyo"))
    _, _, message, _ = stream_split(processor, raw)
    calls = message.get("tool_calls") or []
    checker.check("parallel-calls", len(calls) == 2
                  and {c["function"]["name"] for c in calls} == {"get_weather", "get_time"},
                  repr(calls))
    for c in calls:
        checker.check(f"parallel-args-{c['function']['name']}",
                      isinstance(c["function"]["arguments"], dict),
                      repr(c["function"]["arguments"]))

    # no delimiter: everything is reasoning, nothing is a tool call
    raw = "reasoning only, no closing delimiter"
    reasoning, final_text, message, _ = stream_split(processor, raw)
    checker.check("no-delim-reasoning", reasoning.strip() == raw, repr(reasoning))
    checker.check("no-delim-final-empty", final_text == "", repr(final_text))
    checker.check("no-delim-no-calls", not message.get("tool_calls"), repr(message))

    # delimiter split at every position still yields the same result
    raw = "reasoning text\n</think>\n\n" + call("get_weather", "Osaka")
    for cut in range(len("</think>") + 1):
        text = "reasoning text\n</think>\n\n" + call("get_weather", "Osaka")
        split_at = text.index("</think>") + cut
        chunks = (text[:split_at], text[split_at:])
        reasoning_acc = ""
        final_acc = ""
        calls = []
        factory = codec._response_parser_factory(processor)
        parser = factory.get_response_parser(
            response_template=codec.TOOL_RESPONSE_TEMPLATE, prefix="")
        full = ""
        for chunk in chunks:
            full += chunk
            r, f = codec.split_reasoning_partial(full)
            fd = suffix(final_acc, f)
            reasoning_acc, final_acc = r, f
            if fd:
                calls += [e for e in parser.feed(fd)
                          if e.get("type") == "region_close"]
        msg, fin = parser.finalize()
        msg_calls = msg.get("tool_calls") or []
        checker.check(f"delim-split-{cut}", len(msg_calls) == 1
                      and msg_calls[0]["function"]["arguments"] == {"city": "Osaka"},
                      f"cut={cut} calls={msg_calls}")

    # markup leakage: no raw markup in reasoning/content
    raw = "reasoning\n</think>\n\n" + call("get_weather", "Osaka")
    reasoning, final_text, message, _ = stream_split(processor, raw)
    visible = reasoning + (message.get("content") or "")
    checker.check("no-markup-leak",
                  all(m not in visible for m in ("<think>", "</think>", "<tool_call>",
                                                 "<function=")),
                  repr(visible))

    # non-stream: tool-open markup after reasoning with no parseable call fails closed
    malformed = "reasoning\n</think>\n\n<tool_call>\n"
    try:
        codec.parse_assistant_message(
            processor, [], processor.tokenizer.encode(malformed), [WEATHER], reasoning=True)
        checker.check("malformed-fail-closed", False, "no error raised")
    except codec.ToolParseError:
        checker.check("malformed-fail-closed", True, "")

    # non-stream: reasoning + tool via parse_assistant_message
    for label, text, tools in (
            ("ns-tool", "reasoning\n</think>\n\n" + call("get_weather", "Osaka"), [WEATHER]),
            ("ns-content", "reasoning\n</think>\n\nDone.", None),
            ("ns-none", "only reasoning", [WEATHER])):
        generated = processor.tokenizer.encode(text)
        parsed = codec.parse_assistant_message(
            processor, [], generated, tools, reasoning=True)
        checker.check(f"{label}-reasoning", parsed["reasoning_content"].strip() != "",
                      repr(parsed["reasoning_content"]))
        if label == "ns-tool":
            checker.check("ns-tool-calls",
                          len(parsed["tool_calls"]) == 1
                          and parsed["tool_calls"][0]["function"]["arguments"]
                          == {"city": "Osaka"}, repr(parsed["tool_calls"]))
            checker.check("ns-tool-content-empty", parsed["content"] == "",
                          repr(parsed["content"]))
        if label == "ns-content":
            checker.check("ns-content-value", parsed["content"].strip() == "Done.",
                          repr(parsed["content"]))
        if label == "ns-none":
            checker.check("ns-none-no-calls", not parsed["tool_calls"], repr(parsed))


def builder_cases(checker):
    def fmt(tools, policy, parallel, grammar, composition):
        raw = tc.build_reasoning_structural_tag(
            tools, policy, parallel, grammar, composition)
        return json.loads(raw) if raw else None

    def envelope_of(document):
        return document["format"]["elements"][0] if document else None

    def final_of(document):
        return document["format"]["elements"][1] if document else None

    for label, tools, policy, parallel, grammar, composition, final_type in (
            ("grammar", None, tc.ToolChoicePolicy("auto"), True,
             'root ::= "YES"', False, "grammar"),
            ("required", [WEATHER], tc.ToolChoicePolicy("required"), False,
             "", False, "tags_with_separator"),
            ("named", [WEATHER], tc.ToolChoicePolicy("named", "get_weather"), True,
             "", False, "tag"),
            ("auto", [WEATHER], tc.ToolChoicePolicy("auto"), True, "", False,
             "triggered_tags"),
            ("compose-auto", [WEATHER], tc.ToolChoicePolicy("auto"), True,
             'root ::= "YES"', True, "or"),
            ("compose-required", [WEATHER], tc.ToolChoicePolicy("required"), True,
             'root ::= "YES"', True, "tags_with_separator"),
            ("compose-named", [WEATHER], tc.ToolChoicePolicy("named", "get_weather"),
             True, 'root ::= "YES"', True, "tag"),
    ):
        document = fmt(tools, policy, parallel, grammar, composition)
        checker.check(f"build-{label}-present", document is not None, repr(document))
        if document is None:
            continue
        tag = envelope_of(document)
        checker.check(f"build-{label}-envelope",
                      tag == {"type": "tag", "begin": "",
                              "content": {"type": "any_text"}, "end": "</think>\n\n"},
                      repr(tag))
        final = final_of(document)
        checker.check(f"build-{label}-final", (final or {}).get("type") == final_type,
                      repr(final))

    plain = fmt(None, tc.ToolChoicePolicy("auto"), True, "", False)
    checker.check("build-plain-none", plain is None, repr(plain))
    loose = fmt([{"type": "function", "function": {
        "name": "f", "description": "", "parameters": {}}}],
        tc.ToolChoicePolicy("auto"), True, "", False)
    checker.check("build-loose-none", loose is None, repr(loose))

    first = tc.build_reasoning_structural_tag(
        [WEATHER], tc.ToolChoicePolicy("auto"), True, 'root ::= "YES"', True)
    second = tc.build_reasoning_structural_tag(
        [WEATHER], tc.ToolChoicePolicy("auto"), True, 'root ::= "YES"', True)
    checker.check("build-deterministic", first == second, "bytes differ")


def main() -> int:
    checker = Checker("reasoning-tool-parser")
    try:
        builder_cases(checker)
        parser_cases(checker, load_processor())
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
