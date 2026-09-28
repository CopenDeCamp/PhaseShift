"""Qwen3.5 function-tool constraint builder.

This module turns OpenAI tool definitions, ``tool_choice`` and
``parallel_tool_calls`` into an XGrammar Structural Tag JSON document for the
local backend. It owns tool policy parsing and strict schema validation; it
never touches HTTP, the compute process or response parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_BEGIN = "<tool_call>\n<function={name}>\n"
_TOOL_CALL_END = "\n</function>\n</tool_call>"
_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_STRICT_ARRAY_KEYS = ("anyOf", "oneOf", "allOf")


class ToolConstraintError(ValueError):
    """Raised when a tool request cannot be turned into a constraint."""


@dataclass(frozen=True)
class ToolChoicePolicy:
    mode: str
    name: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    strict: bool
    parameters: dict[str, Any]


def resolve_tool_choice(raw: Any) -> ToolChoicePolicy:
    """Parse the frontend ``tool_choice`` value fail closed.

    Accepts the raw LocalAI payload (a JSON string), a plain policy string, or
    the Chat / Responses named-function object shape. Anything unrecognised is a
    request error; there is no silent ``auto`` fallback.
    """
    if raw is None:
        return ToolChoicePolicy("auto")
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return ToolChoicePolicy("auto")
        if text in ("auto", "none", "required"):
            return ToolChoicePolicy(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolConstraintError(f"invalid tool_choice: {text}") from exc
        return _resolve_parsed(parsed, text)
    if isinstance(raw, dict):
        return _resolve_object(raw)
    raise ToolConstraintError(f"malformed tool_choice: {raw!r}")


def _resolve_parsed(parsed: Any, original: str) -> ToolChoicePolicy:
    if isinstance(parsed, str):
        if parsed in ("auto", "none", "required"):
            return ToolChoicePolicy(parsed)
        raise ToolConstraintError(f"unknown tool_choice: {parsed}")
    if isinstance(parsed, dict):
        return _resolve_object(parsed)
    raise ToolConstraintError(f"malformed tool_choice: {original}")


def _resolve_object(obj: dict[str, Any]) -> ToolChoicePolicy:
    choice_type = obj.get("type")
    if choice_type not in (None, "function"):
        raise ToolConstraintError(f"unknown tool_choice type: {choice_type!r}")
    name = None
    function = obj.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    elif isinstance(function, str):
        name = function
    if name is None:
        name = obj.get("name")
    if not isinstance(name, str) or not name:
        raise ToolConstraintError("tool_choice is missing a function name")
    return ToolChoicePolicy("named", name)


def parse_parallel_tool_calls(raw: Any) -> bool:
    """Interpret the frontend parallel flag; default is true."""
    if raw is None:
        return True
    if raw is True or raw is False:
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text == "":
            return True
        if text == "true":
            return True
        if text == "false":
            return False
    raise ToolConstraintError(f"invalid parallel_tool_calls: {raw!r}")


def collect_tools(tools: list[dict] | None) -> list[ToolDefinition]:
    """Validate canonical tool definitions and return the builder view."""
    if not tools:
        return []
    parsed: list[ToolDefinition] = []
    seen: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            raise ToolConstraintError("tool is missing a function definition")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ToolConstraintError("tool function is missing a name")
        if not _FUNCTION_NAME.match(name):
            raise ToolConstraintError(f"invalid function name: {name!r}")
        if name in seen:
            raise ToolConstraintError(f"duplicate function name: {name}")
        seen.add(name)
        strict = function.get("strict") is True
        parameters = function.get("parameters")
        if strict:
            parameters = validate_strict_parameters(parameters, name)
        parsed.append(ToolDefinition(name=name, strict=strict, parameters=parameters))
    return parsed


def validate_strict_parameters(parameters: Any, name: str) -> dict[str, Any]:
    """Validate an OpenAI strict function parameter schema.

    Returns a canonical schema. Raises ``ToolConstraintError`` when the schema
    cannot satisfy OpenAI strict semantics.
    """
    if parameters is None:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    if not isinstance(parameters, dict):
        raise ToolConstraintError(f"tool {name!r} parameters must be an object")
    _validate_schema(parameters, name, path="parameters")
    return parameters


def _validate_schema(schema: Any, name: str, path: str) -> None:
    if isinstance(schema, bool):
        raise ToolConstraintError(
            f"tool {name!r} {path} must not be a boolean schema in strict mode")
    if not isinstance(schema, dict):
        raise ToolConstraintError(f"tool {name!r} {path} must be an object")

    if schema.get("type") == "object" or "properties" in schema or "required" in schema:
        _validate_object_schema(schema, name, path)

    for key in _STRICT_ARRAY_KEYS:
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                _validate_schema(variant, name, f"{path}.{key}[{index}]")

    items = schema.get("items")
    if isinstance(items, dict):
        _validate_schema(items, name, f"{path}.items")
    elif isinstance(items, list):
        for index, item in enumerate(items):
            _validate_schema(item, name, f"{path}.items[{index}]")

    for defs_key in ("$defs", "definitions"):
        definitions = schema.get(defs_key)
        if isinstance(definitions, dict):
            for def_name, definition in definitions.items():
                _validate_schema(definition, name, f"{path}.{defs_key}.{def_name}")


def _validate_object_schema(schema: dict[str, Any], name: str, path: str) -> None:
    if schema.get("type") not in (None, "object"):
        raise ToolConstraintError(f"tool {name!r} {path} type must be \"object\"")
    additional = schema.get("additionalProperties")
    if additional is not False:
        raise ToolConstraintError(
            f"tool {name!r} {path} must set additionalProperties to false")
    properties = schema.get("properties")
    required = schema.get("required")
    prop_keys = set(properties.keys()) if isinstance(properties, dict) else set()
    if properties is not None and not isinstance(properties, dict):
        raise ToolConstraintError(f"tool {name!r} {path} properties must be an object")
    if required is not None and not isinstance(required, list):
        raise ToolConstraintError(f"tool {name!r} {path} required must be an array")
    required_keys = set()
    if isinstance(required, list):
        for key in required:
            if not isinstance(key, str):
                raise ToolConstraintError(
                    f"tool {name!r} {path} required keys must be strings")
            required_keys.add(key)
    missing = prop_keys - required_keys
    if missing:
        raise ToolConstraintError(
            f"tool {name!r} {path} has properties missing from required: "
            f"{sorted(missing)}")
    unknown = required_keys - prop_keys
    if unknown:
        raise ToolConstraintError(
            f"tool {name!r} {path} requires unknown properties: {sorted(unknown)}")
    if isinstance(properties, dict):
        for key, value in properties.items():
            _validate_schema(value, name, f"{path}.properties.{key}")


def needs_structural_constraint(tools: list[ToolDefinition], policy: ToolChoicePolicy,
                                parallel: bool) -> bool:
    if not tools:
        return False
    if policy.mode in ("required", "named"):
        return True
    if not parallel:
        return True
    return any(tool.strict for tool in tools)


def _dump(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _select(definitions: list[ToolDefinition],
            policy: ToolChoicePolicy) -> list[ToolDefinition]:
    if policy.mode == "named":
        selected = [tool for tool in definitions if tool.name == policy.name]
        if not selected:
            raise ToolConstraintError(
                f"tool_choice references an unknown function: {policy.name!r}")
        return selected
    return definitions


def _tool_tags(definitions: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [_tool_tag(tool) for tool in definitions]


def _tags_with_separator(tags: list[dict[str, Any]], parallel: bool) -> dict[str, Any]:
    return {
        "type": "tags_with_separator",
        "tags": tags,
        "separator": "\n",
        "at_least_one": True,
        "stop_after_first": not parallel,
    }


def _triggered_tags(tags: list[dict[str, Any]], parallel: bool) -> dict[str, Any]:
    return {
        "type": "triggered_tags",
        "triggers": [tag["begin"] for tag in tags],
        "tags": tags,
        "excludes": [],
        "at_least_one": False,
        "stop_after_first": not parallel,
    }


def _build_regular_tool_format(definitions: list[ToolDefinition],
                               policy: ToolChoicePolicy,
                               parallel: bool) -> dict[str, Any] | None:
    if not definitions or policy.mode == "none":
        return None
    if not needs_structural_constraint(definitions, policy, parallel):
        return None
    if policy.mode == "named":
        return _tool_tag(_select(definitions, policy)[0])
    if policy.mode == "required":
        return _tags_with_separator(_tool_tags(definitions), parallel)
    return _triggered_tags(_tool_tags(definitions), parallel)


def _build_tool_only_format(definitions: list[ToolDefinition],
                            policy: ToolChoicePolicy,
                            parallel: bool) -> dict[str, Any] | None:
    """Tool branch used by composite constraints: pure tool calls only."""
    if not definitions or policy.mode == "none":
        return None
    if policy.mode == "named":
        return _tool_tag(_select(definitions, policy)[0])
    return _tags_with_separator(_tool_tags(definitions), parallel)


def _tool_tag(tool: ToolDefinition) -> dict[str, Any]:
    if tool.strict:
        content: dict[str, Any] = {
            "type": "qwen_xml_parameter",
            "json_schema": tool.parameters,
            "any_order": True,
        }
    else:
        content = {"type": "any_text"}
    return {
        "type": "tag",
        "begin": _TOOL_CALL_BEGIN.format(name=tool.name),
        "content": content,
        "end": _TOOL_CALL_END,
    }


def build_structural_tag(tools: list[dict] | None, policy: ToolChoicePolicy,
                         parallel: bool) -> str | None:
    """Build a deterministic Structural Tag JSON document, or ``None``."""
    definitions = collect_tools(tools)
    root = _build_regular_tool_format(definitions, policy, parallel)
    if root is None:
        return None
    return _dump({"type": "structural_tag", "format": root})


def build_composite_structural_tag(tools: list[dict] | None,
                                   policy: ToolChoicePolicy, parallel: bool,
                                   response_grammar: str) -> str | None:
    """Compose a structured-text branch and a pure-tool branch.

    Only ``auto`` produces an ``or`` of the response grammar and the tool
    branch. ``required`` / ``named`` already exclude the structured branch and
    return the tool-only format; ``none`` returns ``None`` (structured text
    only, carried as a plain grammar).
    """
    if policy.mode == "none":
        return None
    if not response_grammar:
        raise ToolConstraintError("composite constraint requires a response grammar")
    definitions = collect_tools(tools)
    if not definitions:
        raise ToolConstraintError("composite constraint requires at least one tool")
    tool_format = _build_tool_only_format(definitions, policy, parallel)
    if tool_format is None:
        return None
    if policy.mode == "auto":
        root: dict[str, Any] = {
            "type": "or",
            "elements": [
                {"type": "grammar", "grammar": response_grammar},
                tool_format,
            ],
        }
    else:
        root = tool_format
    return _dump({"type": "structural_tag", "format": root})


REASONING_END = "</think>\n\n"


def _reasoning_tag() -> dict[str, Any]:
    return {
        "type": "tag",
        "begin": "",
        "content": {"type": "any_text"},
        "end": REASONING_END,
    }


def build_reasoning_envelope(final_format: dict[str, Any] | None) -> str | None:
    """Wrap a final format in the reasoning envelope as one Structural Tag.

    The generation prompt pre-fills ``<think>\\n``, so generation starts inside
    the reasoning block. An empty-begin tag with ``any_text`` content accepts
    free-form reasoning and closes on ``</think>\\n\\n``; the existing final
    format then applies unchanged. Only existing XGrammar primitives are used.
    """
    if final_format is None:
        return None
    root: dict[str, Any] = {
        "type": "sequence",
        "elements": [_reasoning_tag(), final_format],
    }
    return _dump({"type": "structural_tag", "format": root})


def _grammar_format(grammar: str) -> dict[str, Any]:
    return {"type": "grammar", "grammar": grammar}


def build_reasoning_final_format(tools: list[dict] | None, policy: ToolChoicePolicy,
                                 parallel: bool, response_grammar: str,
                                 composition: bool) -> dict[str, Any] | None:
    """The Gate 9A / 9B final format that follows the reasoning block."""
    definitions = collect_tools(tools)
    if policy.mode == "none" or not definitions:
        return _grammar_format(response_grammar) if response_grammar else None
    if composition:
        if policy.mode == "auto":
            if not response_grammar:
                raise ToolConstraintError(
                    "composite constraint requires a response grammar")
            tool_format = _build_tool_only_format(definitions, policy, parallel)
            if tool_format is None:
                return _grammar_format(response_grammar)
            return {
                "type": "or",
                "elements": [_grammar_format(response_grammar), tool_format],
            }
        return _build_tool_only_format(definitions, policy, parallel)
    return _build_regular_tool_format(definitions, policy, parallel)


def build_reasoning_structural_tag(tools: list[dict] | None,
                                   policy: ToolChoicePolicy, parallel: bool,
                                   response_grammar: str,
                                   composition: bool) -> str | None:
    """One Structural Tag: reasoning envelope around the final format.

    Returns ``None`` when no constraint is needed: plain reasoning, or a loose
    best-effort tool request that the existing policy does not constrain.
    """
    final = build_reasoning_final_format(
        tools, policy, parallel, response_grammar, composition)
    return build_reasoning_envelope(final)
