#!/usr/bin/env python3
"""Gate 9A: tool constraint builder and tool policy parsing (CPU only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, ensure_chat_import  # noqa: E402

codec = ensure_chat_import()
from phaseshift_chat import tool_constraint as tc  # noqa: E402

CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def chat_tool(name="get_weather", strict=True, description="weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": json.loads(json.dumps(CITY_SCHEMA)),
            "strict": strict,
        },
    }


def expect_error(checker, label, fn):
    try:
        fn()
    except tc.ToolConstraintError:
        checker.check(label, True)
        return
    checker.check(label, False, "expected ToolConstraintError")


def main() -> int:
    checker = Checker("tool-constraint-builder")

    # tool_choice parsing.
    checker.check("choice-none", tc.resolve_tool_choice(None).mode == "auto")
    checker.check("choice-empty", tc.resolve_tool_choice("").mode == "auto")
    for mode in ("auto", "none", "required"):
        checker.check(f"choice-{mode}",
                      tc.resolve_tool_choice(json.dumps(mode)).mode == mode)
    chat_named = tc.resolve_tool_choice(json.dumps(
        {"type": "function", "function": {"name": "get_weather"}}))
    checker.check("choice-chat-named",
                  chat_named.mode == "named" and chat_named.name == "get_weather")
    resp_named = tc.resolve_tool_choice(json.dumps(
        {"type": "function", "name": "get_weather"}))
    checker.check("choice-responses-named",
                  resp_named.mode == "named" and resp_named.name == "get_weather")
    expect_error(checker, "choice-bare-name-rejected",
                 lambda: tc.resolve_tool_choice("get_weather"))
    expect_error(checker, "choice-unknown-type-rejected",
                 lambda: tc.resolve_tool_choice(json.dumps({"type": "allowed_tools"})))
    expect_error(checker, "choice-malformed-rejected",
                 lambda: tc.resolve_tool_choice("["))
    expect_error(checker, "choice-missing-name-rejected",
                 lambda: tc.resolve_tool_choice(json.dumps({"type": "function"})))

    # parallel flag parsing.
    checker.check("parallel-default", tc.parse_parallel_tool_calls(None) is True)
    checker.check("parallel-true", tc.parse_parallel_tool_calls("true") is True)
    checker.check("parallel-false", tc.parse_parallel_tool_calls("false") is False)
    expect_error(checker, "parallel-invalid",
                 lambda: tc.parse_parallel_tool_calls("yes"))

    # strict schema validation.
    valid = tc.validate_strict_parameters(CITY_SCHEMA, "get_weather")
    checker.check("strict-valid", valid == CITY_SCHEMA)
    missing = dict(CITY_SCHEMA)
    missing.pop("additionalProperties")
    expect_error(checker, "strict-missing-additional",
                 lambda: tc.validate_strict_parameters(missing, "t"))
    expect_error(checker, "strict-additional-true",
                 lambda: tc.validate_strict_parameters(
                     {**CITY_SCHEMA, "additionalProperties": True}, "t"))
    expect_error(checker, "strict-property-not-required",
                 lambda: tc.validate_strict_parameters(
                     {"type": "object",
                      "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                      "required": ["a"], "additionalProperties": False}, "t"))
    expect_error(checker, "strict-unknown-required",
                 lambda: tc.validate_strict_parameters(
                     {"type": "object", "properties": {"a": {"type": "string"}},
                      "required": ["a", "b"], "additionalProperties": False}, "t"))
    expect_error(checker, "strict-boolean-schema",
                 lambda: tc.validate_strict_parameters(True, "t"))
    nullable = {"type": "object",
                "properties": {"a": {"type": ["string", "null"]}},
                "required": ["a"], "additionalProperties": False}
    checker.check("strict-nullable", tc.validate_strict_parameters(nullable, "t") == nullable)
    nested = {"type": "object",
              "properties": {"inner": {"type": "object",
                                       "properties": {"x": {"type": "integer"}},
                                       "required": ["x"], "additionalProperties": False}},
              "required": ["inner"], "additionalProperties": False}
    checker.check("strict-nested", tc.validate_strict_parameters(nested, "t") == nested)
    empty = tc.validate_strict_parameters(None, "t")
    checker.check("strict-empty-parameters",
                  empty == {"type": "object", "properties": {}, "required": [],
                            "additionalProperties": False})

    # tool collection.
    expect_error(checker, "tools-duplicate",
                 lambda: tc.collect_tools([chat_tool(), chat_tool()]))
    expect_error(checker, "tools-bad-name",
                 lambda: tc.collect_tools([chat_tool(name="bad.name")]))
    expect_error(checker, "tools-empty-name",
                 lambda: tc.collect_tools([chat_tool(name="")]))

    # builder.
    loose_auto = tc.build_structural_tag(
        [chat_tool(strict=False)], tc.ToolChoicePolicy("auto"), True)
    checker.check("builder-loose-auto-none", loose_auto is None)

    strict_auto = json.loads(tc.build_structural_tag(
        [chat_tool()], tc.ToolChoicePolicy("auto"), True))
    fmt = strict_auto["format"]
    checker.check("builder-auto-triggered", fmt["type"] == "triggered_tags")
    checker.check("builder-auto-at-least-one", fmt["at_least_one"] is False)
    checker.check("builder-auto-stop", fmt["stop_after_first"] is False)
    checker.check("builder-auto-trigger", fmt["triggers"] == ["<tool_call>\n<function=get_weather>\n"])
    checker.check("builder-strict-content",
                  fmt["tags"][0]["content"]["type"] == "qwen_xml_parameter")
    checker.check("builder-strict-any-order",
                  fmt["tags"][0]["content"]["any_order"] is True)

    required_serial = json.loads(tc.build_structural_tag(
        [chat_tool(), chat_tool(name="get_time")],
        tc.ToolChoicePolicy("required"), False))
    rfmt = required_serial["format"]
    checker.check("builder-required-type", rfmt["type"] == "tags_with_separator")
    checker.check("builder-required-at-least-one", rfmt["at_least_one"] is True)
    checker.check("builder-required-stop", rfmt["stop_after_first"] is True)
    checker.check("builder-required-separator", rfmt["separator"] == "\n")

    named = json.loads(tc.build_structural_tag(
        [chat_tool(), chat_tool(name="get_time")],
        tc.ToolChoicePolicy("named", "get_time"), False))
    checker.check("builder-named-single", named["format"]["type"] == "tag")
    checker.check("builder-named-begin",
                  "function=get_time" in named["format"]["begin"])
    expect_error(checker, "builder-named-unknown",
                 lambda: tc.build_structural_tag(
                     [chat_tool()], tc.ToolChoicePolicy("named", "nope"), True))

    loose_content = json.loads(tc.build_structural_tag(
        [chat_tool(strict=False)], tc.ToolChoicePolicy("required"), True))
    checker.check("builder-loose-content",
                  loose_content["format"]["tags"][0]["content"] == {"type": "any_text"})

    # deterministic serialization and description independence.
    first = tc.build_structural_tag([chat_tool()], tc.ToolChoicePolicy("auto"), True)
    second = tc.build_structural_tag([chat_tool(description="changed")],
                                     tc.ToolChoicePolicy("auto"), True)
    checker.check("builder-deterministic", first == second)

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
