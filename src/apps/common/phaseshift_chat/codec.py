"""PhaseShift shared chat codec.

Bridges the Hugging Face Qwen processor (tokenizer + chat template) and the
PhaseShift compute engine's raw token-ID contract.

This module is the single source of truth for:

* processor loading
* chat-template prompt construction (system / user / assistant / tool)
* the turn-stable prefix boundary used by the prefix cache protocol contract
* generation decoding
* tool-call response parsing

It is imported by both ``phaseshift-cli`` and ``phaseshift-server`` so that the
token sequence sent to ``phaseshift-compute`` never diverges between the two
entry points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = [
    "ToolParseError",
    "ToolCallingUnavailable",
    "PromptRender",
    "TOOL_RESPONSE_TEMPLATE",
    "load_processor",
    "prompt_ids",
    "prompt_ids_with_cache_boundary",
    "decode_generated",
    "tool_calling_available",
    "streaming_tool_calling_available",
    "probe_tool_calling",
    "new_response_parser",
    "parse_assistant_message",
    "strip_tool_markup",
]


class ToolParseError(RuntimeError):
    """Raised when a tool-calling generation cannot be parsed."""


class ToolCallingUnavailable(RuntimeError):
    """Raised when the active Transformers install cannot parse tool calls."""


# Qwen3.5 emits tool calls as::
#
#     <tool_call>
#     <function=NAME>
#     <parameter=KEY>
#     VALUE
#     </parameter>
#     </function>
#     </tool_call>
#
# This declarative response template is consumed by Transformers'
# ``parse_response``. It is intentionally a template (not a hand-written
# regex loop): the parser in transformers.utils.chat_parsing owns the scanning,
# overlap handling, and structured-value coercion.
# The chat template (enable_thinking=False) pre-fills the assistant turn with
# the open/close thinking tokens; generation starts right after the closing
# one. Anchoring on it lets the parser discard the prompt tail (including the
# tool-usage example in the system block) when a real prompt token sequence is
# supplied as ``prefix``.
_THINK_CLOSE = "".join(
    chr(code) for code in (0x3C, 0x2F, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x3E))

TOOL_RESPONSE_TEMPLATE: dict[str, Any] = {
    "version": 1,
    "start_anchor": _THINK_CLOSE,
    "defaults": {"role": "assistant", "content": ""},
    "fields": {
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\s]+)>",
            "close": "</tool_call>",
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
            },
            "repeats": True,
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def load_processor(model_dir: str, trust_remote_code: bool = True):
    """Load the Hugging Face processor for ``model_dir``."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)


def _flatten_ids(ids: Any) -> list[int]:
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return list(ids[0])
    return list(ids)


def prompt_ids(
    processor,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = False,
) -> list[int]:
    """Render ``messages`` through the model chat template and return token IDs.

    ``tools`` is passed through to the chat template unchanged; no Prompt-magic
    or tool-description synthesis is performed here.
    """
    return _render(processor, messages, tools, enable_thinking, True)


@dataclass(frozen=True)
class PromptRender:
    """A rendered prompt plus the token position that is stable across turns."""

    ids: list[int]
    cache_boundary: int


def _render(processor, messages, tools, enable_thinking, add_generation_prompt):
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    ids = processor.apply_chat_template(messages, **kwargs)
    return _flatten_ids(ids)


def prompt_ids_with_cache_boundary(
    processor,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = False,
) -> PromptRender:
    """Render the generation prompt and locate the turn-stable boundary.

    The ``add_generation_prompt=False`` rendering is the conversation history.
    The next turn re-renders the same history, so that prefix is stable across
    turns while the ``add_generation_prompt=True`` tail (thinking prelude and
    assistant opener) is not. The boundary is only used when it is an exact
    token prefix of the full prompt; otherwise the request falls back to a
    boundary of 0 and behaves exactly as before.
    """
    full = _render(processor, messages, tools, enable_thinking, True)
    try:
        stable = _render(processor, messages, tools, enable_thinking, False)
    except Exception:  # noqa: BLE001 - boundary is a protocol hint only
        return PromptRender(full, 0)
    if not stable or len(stable) > len(full) or full[:len(stable)] != stable:
        return PromptRender(full, 0)
    return PromptRender(full, len(stable))


def decode_generated(processor, generated_ids: Iterable[int]) -> str:
    """Decode generated token IDs exactly like the PhaseShift CLI does."""
    return processor.decode(list(generated_ids), skip_special_tokens=True)


def strip_tool_markup(text: str) -> str:
    """Remove Qwen tool-call markup that leaked into plain content."""
    if not text or _TOOL_OPEN not in text:
        return text
    out = []
    i = 0
    while True:
        start = text.find(_TOOL_OPEN, i)
        if start == -1:
            out.append(text[i:])
            break
        end = text.find(_TOOL_CLOSE, start)
        if end == -1:
            out.append(text[i:start])
            break
        out.append(text[i:start])
        i = end + len(_TOOL_CLOSE)
    return "".join(out).strip()


def _parse_response(processor, text: str, prompt: Optional[Iterable[int]]) -> dict[str, Any]:
    return processor.parse_response(
        text, TOOL_RESPONSE_TEMPLATE, prefix=list(prompt) if prompt is not None else "")


def _response_parser_factory(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "get_response_parser"):
        return tokenizer
    if hasattr(processor, "get_response_parser"):
        return processor
    raise AttributeError("processor has no get_response_parser")


def new_response_parser(processor, prompt: Iterable[int], prompt_prefix: bool = True):
    """Create one stateful streaming parser for a single generation.

    A parser must never be shared across requests: it carries the region state
    (and captured function names) of exactly one assistant turn.

    ``prompt_prefix`` passes the chat prompt as the parser's ``prefix`` so the
    pre-filled assistant turn is truncated past its ``</think>`` anchor.
    """
    factory = _response_parser_factory(processor)
    prefix = list(prompt) if prompt_prefix else ""
    return factory.get_response_parser(
        response_template=TOOL_RESPONSE_TEMPLATE, prefix=prefix)


def probe_tool_calling(processor) -> dict[str, Any]:
    """Probe the installed Transformers for Qwen3.5 tool support.

    Returns a capability dict. Never raises; callers decide how to report a
    missing capability.
    """
    capability = {
        "chat_template_tools": False,
        "response_parser": False,
        "streaming_response_parser": False,
        "detail": "",
    }

    def note(message: str) -> None:
        capability["detail"] = (capability["detail"] + "; " if capability["detail"] else "") + message

    probe_tools = [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "description": "noop",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
        }
    ]
    probe_prompt: list[int] = []
    try:
        probe_prompt = prompt_ids(
            processor, [{"role": "user", "content": "ping"}], tools=probe_tools)
        capability["chat_template_tools"] = True
    except Exception as exc:  # noqa: BLE001 - capability probe reports, never raises
        note(f"chat template tools: {exc}")

    probe_text = (
        "<tool_call>\n<function=probe>\n<parameter=x>\n1\n</parameter>\n"
        "</function>\n</tool_call>")

    try:
        parsed = _parse_response(processor, probe_text, probe_prompt)
        capability["response_parser"] = bool(parsed.get("tool_calls"))
    except Exception as exc:  # noqa: BLE001
        note(f"response parser: {exc}")

    try:
        parser = new_response_parser(processor, probe_prompt)
        events = parser.feed(probe_text)
        message, final_events = parser.finalize()
        events = list(events) + list(final_events)
        names = []
        args_ok = True
        for event in events:
            if event.get("type") == "region_close" and event.get("field") == "tool_calls":
                function = (event.get("value") or {}).get("function") or {}
                if function.get("name"):
                    names.append(function["name"])
                if not isinstance(function.get("arguments"), dict):
                    args_ok = False
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name"):
                names.append(function["name"])
        capability["streaming_response_parser"] = bool(names) and args_ok
    except Exception as exc:  # noqa: BLE001
        note(f"streaming response parser: {exc}")

    return capability


def _capability(processor) -> dict[str, Any]:
    cached = getattr(processor, "_phaseshift_tool_capability", None)
    if cached is None:
        cached = probe_tool_calling(processor)
        try:
            processor._phaseshift_tool_capability = cached
        except Exception:  # noqa: BLE001 - best-effort cache only
            pass
    return cached


def tool_calling_available(processor) -> bool:
    cached = _capability(processor)
    return bool(cached["chat_template_tools"] and cached["response_parser"])


def streaming_tool_calling_available(processor) -> bool:
    cached = _capability(processor)
    return bool(
        cached["chat_template_tools"] and cached["streaming_response_parser"])


def _parse_tool_calls(processor, final_text: str, prefix):
    """Parse tool calls from the final (post-anchor) assistant text.

    ``prefix`` is the response parser's chat-prompt token IDs; the parser
    truncates the pre-filled assistant turn past its ``</think>`` anchor.

    Returns ``(tool_calls, content)``. Raises ``ToolParseError`` when tool-call
    markup is present but could not be parsed into a call.
    """
    if not tool_calling_available(processor):
        raise ToolCallingUnavailable(
            "installed Transformers cannot parse Qwen3.5 tool calls")

    try:
        parsed = _parse_response(processor, final_text, prefix)
    except Exception as exc:  # noqa: BLE001
        if _TOOL_OPEN in final_text:
            raise ToolParseError(f"failed to parse tool call: {exc}") from exc
        return [], strip_tool_markup(final_text)

    tool_calls = []
    for call in parsed.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        arguments = fn.get("arguments")
        if not name:
            if _TOOL_OPEN in final_text:
                raise ToolParseError("parsed tool call is missing a function name")
            continue
        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else {"value": arguments}
        tool_calls.append(
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        )

    if not tool_calls and _TOOL_OPEN in final_text:
        raise ToolParseError("tool-call markup present but no tool call parsed")

    return tool_calls, strip_tool_markup(parsed.get("content", ""))


def parse_assistant_message(
    processor,
    prompt_ids_seq: Iterable[int],
    generated_ids: Iterable[int],
    tools: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Decode and parse one assistant generation.

    ``prompt_ids_seq`` is the prompt token sequence that produced
    ``generated_ids``; it is passed to the response parser as ``prefix`` so the
    parser can discard the pre-filled assistant turn.

    Returns ``{"content": str, "tool_calls": list}`` where each tool call uses
    the OpenAI shape ``{"type": "function", "function": {"name", "arguments"}}``
    and ``arguments`` is a parsed mapping (the wire serialization to a JSON
    string is the server's responsibility).

    Raises ``ToolParseError`` when a tool request produced tool-call-like output
    that the response template could not parse. Raw markup is never silently
    promoted to ``content``.
    """
    text = decode_generated(processor, generated_ids)
    if not tools:
        return {"content": strip_tool_markup(text), "tool_calls": []}
    tool_calls, content = _parse_tool_calls(processor, text, prompt_ids_seq)
    return {"content": content, "tool_calls": tool_calls}
