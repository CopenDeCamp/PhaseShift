#!/usr/bin/env python3
"""Gate 9B: composite structural tag builder (CPU only)."""

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
GRAMMAR = 'root ::= "{\\"answer\\":\\"yes\\"}"'


def tool(name="get_weather", strict=True):
    return {"type": "function", "function": {
        "name": name, "description": "d", "parameters": json.loads(json.dumps(CITY_SCHEMA)),
        "strict": strict}}


def main() -> int:
    checker = Checker("tool-composition-builder")

    # auto: or(response grammar, pure tool branch).
    auto = json.loads(tc.build_composite_structural_tag(
        [tool()], tc.ToolChoicePolicy("auto"), True, GRAMMAR))
    fmt = auto["format"]
    checker.check("auto-or", fmt["type"] == "or", repr(fmt.get("type")))
    checker.check("auto-elements", len(fmt["elements"]) == 2, repr(fmt["elements"]))
    checker.check("auto-grammar-verbatim",
                  fmt["elements"][0] == {"type": "grammar", "grammar": GRAMMAR},
                  repr(fmt["elements"][0]))
    tool_branch = fmt["elements"][1]
    checker.check("auto-tool-type", tool_branch["type"] == "tags_with_separator",
                  repr(tool_branch.get("type")))
    checker.check("auto-tool-at-least-one", tool_branch["at_least_one"] is True)
    checker.check("auto-tool-stop", tool_branch["stop_after_first"] is False)
    checker.check("auto-strict-content",
                  tool_branch["tags"][0]["content"]["type"] == "qwen_xml_parameter")

    auto_single = json.loads(tc.build_composite_structural_tag(
        [tool()], tc.ToolChoicePolicy("auto"), False, GRAMMAR))
    checker.check("auto-parallel-false-stop",
                  auto_single["format"]["elements"][1]["stop_after_first"] is True)

    # required: tool-only, no structured branch.
    required = json.loads(tc.build_composite_structural_tag(
        [tool(), tool("get_time")], tc.ToolChoicePolicy("required"), False, GRAMMAR))
    checker.check("required-tool-only", required["format"]["type"] == "tags_with_separator")
    checker.check("required-at-least-one", required["format"]["at_least_one"] is True)
    checker.check("required-stop", required["format"]["stop_after_first"] is True)
    checker.check("required-no-grammar-branch",
                  "grammar" not in json.dumps(required), json.dumps(required))

    # named: single tag.
    named = json.loads(tc.build_composite_structural_tag(
        [tool(), tool("get_time")], tc.ToolChoicePolicy("named", "get_time"), True, GRAMMAR))
    checker.check("named-single", named["format"]["type"] == "tag")
    checker.check("named-begin", "function=get_time" in named["format"]["begin"])

    # none: structured text only (no structural tag).
    checker.check("none-no-tag",
                  tc.build_composite_structural_tag(
                      [tool()], tc.ToolChoicePolicy("none"), True, GRAMMAR) is None)

    # loose tool keeps any_text.
    loose = json.loads(tc.build_composite_structural_tag(
        [tool(strict=False)], tc.ToolChoicePolicy("auto"), True, GRAMMAR))
    checker.check("loose-any-text",
                  loose["format"]["elements"][1]["tags"][0]["content"] == {"type": "any_text"},
                  repr(loose["format"]["elements"][1]["tags"][0]["content"]))

    # deterministic and description independent.
    first = tc.build_composite_structural_tag(
        [tool()], tc.ToolChoicePolicy("auto"), True, GRAMMAR)
    second = tc.build_composite_structural_tag(
        [tool()], tc.ToolChoicePolicy("auto"), True, GRAMMAR)
    checker.check("deterministic", first == second)

    # errors.
    try:
        tc.build_composite_structural_tag([tool()], tc.ToolChoicePolicy("auto"), True, "")
        checker.check("requires-grammar", False, "expected error")
    except tc.ToolConstraintError:
        checker.check("requires-grammar", True)
    try:
        tc.build_composite_structural_tag([], tc.ToolChoicePolicy("auto"), True, GRAMMAR)
        checker.check("requires-tools", False, "expected error")
    except tc.ToolConstraintError:
        checker.check("requires-tools", True)

    # Gate 9A regular builder is unchanged (auto -> triggered_tags).
    regular = json.loads(tc.build_structural_tag(
        [tool()], tc.ToolChoicePolicy("auto"), True))
    checker.check("regular-auto-triggered",
                  regular["format"]["type"] == "triggered_tags",
                  repr(regular["format"]["type"]))
    checker.check("regular-loose-returns-none",
                  tc.build_structural_tag(
                      [tool(strict=False)], tc.ToolChoicePolicy("auto"), True) is None)

    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
