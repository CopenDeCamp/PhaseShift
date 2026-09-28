"""LocalAI gRPC messages <-> Hugging Face chat messages.

This module owns every conversion between the LocalAI wire schema and the
message dictionaries the Qwen3.5 chat template consumes. It deliberately does
not build prompts, inject tool instructions, or reinterpret roles beyond what
the Hugging Face chat template requires.
"""

from __future__ import annotations

import json
from typing import Any


class MessageError(ValueError):
    """Raised for a request that cannot be converted to a chat message."""


_KNOWN_ROLES = {"system", "developer", "user", "assistant", "tool"}


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content: keep only text parts; tool templates never need
        # image/audio payloads for Gate 1-3.
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _parse_arguments(arguments: Any) -> dict:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise MessageError("tool call arguments must be a JSON string or object")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise MessageError(f"tool call arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MessageError("tool call arguments must decode to a JSON object")
    return parsed


def _convert_assistant_tool_calls(raw: str) -> list[dict]:
    try:
        calls = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MessageError(f"assistant tool_calls is not valid JSON: {exc}") from exc
    if not isinstance(calls, list):
        raise MessageError("assistant tool_calls must be a JSON array")

    converted = []
    for call in calls:
        if not isinstance(call, dict):
            raise MessageError("assistant tool_calls entries must be objects")
        function = call.get("function")
        if not isinstance(function, dict):
            raise MessageError("assistant tool_calls entry is missing 'function'")
        name = function.get("name")
        if not name:
            raise MessageError("assistant tool_calls entry is missing a function name")
        converted.append(
            {
                "type": call.get("type", "function"),
                "function": {
                    "name": name,
                    "arguments": _parse_arguments(function.get("arguments")),
                },
            }
        )
    return converted


def proto_messages_to_hf(proto_messages) -> list[dict]:
    """Convert LocalAI proto Messages into Hugging Face chat message dicts.

    Qwen3.5's chat template only accepts a system message at the very start.
    System and Responses-API ``developer`` instructions are therefore merged
    into a single leading system message instead of being emitted in place
    (which the template would reject).
    """
    messages: list[dict] = []
    system_parts: list[str] = []
    for message in proto_messages:
        role = (message.role or "").strip()
        if role not in _KNOWN_ROLES:
            raise MessageError(f"unsupported message role: {role!r}")

        content = _content_to_str(message.content)

        if role in ("system", "developer"):
            if content:
                system_parts.append(content)
            continue

        if role == "assistant":
            hf: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning = getattr(message, "reasoning_content", "") or ""
            if reasoning:
                hf["reasoning_content"] = reasoning
            if message.tool_calls:
                hf["tool_calls"] = _convert_assistant_tool_calls(message.tool_calls)
            messages.append(hf)
        elif role == "tool":
            hf = {"role": "tool", "content": content}
            if message.name:
                hf["name"] = message.name
            messages.append(hf)
        else:
            messages.append({"role": role, "content": content})

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return messages


def parse_tools(tools_json: str) -> list[dict] | None:
    """Parse the LocalAI PredictOptions.Tools JSON array.

    Returns ``None`` when no tools were supplied, otherwise the list of tool
    definitions unchanged (they are forwarded to the chat template verbatim).
    """
    if not tools_json:
        return None
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as exc:
        raise MessageError(f"tools is not valid JSON: {exc}") from exc
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MessageError("tools must be a JSON array")
    return tools


def normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    """Normalize tool definitions to the canonical Chat Completions shape.

    LocalAI v4.10.0 already forwards the canonical
    ``{"type":"function","function":{...}}`` shape for both Chat Completions and
    Responses, but this accepts the Responses top-level shape
    (``{"type":"function","name":...,"parameters":...}``) as well. Unknown tool
    types are rejected rather than silently treated as functions.
    """
    if tools is None:
        return None
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise MessageError("each tool must be an object")
        if tool.get("type") != "function":
            raise MessageError(f"unsupported tool type: {tool.get('type')!r}")
        function = tool.get("function")
        if isinstance(function, dict):
            if not function.get("name"):
                raise MessageError("tool function is missing a name")
            normalized.append(tool)
            continue
        name = tool.get("name")
        if not name:
            raise MessageError("tool function is missing a name")
        canonical_function = {"name": name}
        if tool.get("description") is not None:
            canonical_function["description"] = tool["description"]
        if tool.get("parameters") is not None:
            canonical_function["parameters"] = tool["parameters"]
        if tool.get("strict") is not None:
            canonical_function["strict"] = tool["strict"]
        normalized.append({"type": "function", "function": canonical_function})
    return normalized


def resolve_tool_choice(tool_choice: str) -> str:
    """Normalize LocalAI's ToolChoice into one of auto / none / required / name.

    Anything more specific than ``required`` is reported as ``name`` so the
    backend can reject it explicitly rather than silently treating it as auto.
    """
    if tool_choice is None:
        return "auto"
    raw = tool_choice.strip()
    if raw == "":
        return "auto"
    if raw in ("auto", "none", "required"):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "auto"
    if isinstance(parsed, str) and parsed in ("auto", "none", "required"):
        return parsed
    if isinstance(parsed, dict):
        function = parsed.get("function")
        if isinstance(function, dict) and function.get("name"):
            return "name"
        if parsed.get("type") == "function":
            return "name"
    return "name"
