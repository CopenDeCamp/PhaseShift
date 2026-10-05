"""OpenAI ChatRequest to PhaseShift compute and back.

ChatService owns the transformation chain and nothing else:

    OpenAI ChatRequest
          -> HF messages
          -> HF chat template
          -> token IDs
          -> AsyncComputeClient
          -> generated token IDs
          -> Qwen response parser
          -> internal assistant response

HTTP concerns live in ``phaseshift_server.py``; wire encoding lives in
``openai_protocol.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from phaseshift_chat import codec

from .compute_client import ComputeCancelled, ComputeError, ComputeInvalidArgument
from .openai_protocol import (
    ChatRequest,
    ChatServiceError,
    OpenAIProtocolError,
    openai_messages_to_hf,
)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
}


@dataclass(frozen=True)
class PreparedChat:
    prompt_ids: list[int]
    tools: Optional[list[dict]]
    max_new_tokens: int
    prefix_cache_checkpoint_position: int


@dataclass(frozen=True)
class AssistantResult:
    content: str
    tool_calls: list[dict]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    call_id: str
    name: Optional[str] = None
    arguments: str = ""


@dataclass(frozen=True)
class StreamDone:
    finish_reason: str


def map_finish_reason(raw: str, *, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    mapped = _FINISH_REASON_MAP.get(raw)
    if mapped is None:
        raise ChatServiceError(
            f"unsupported compute finish reason: {raw!r}",
            code="compute_error",
        )
    return mapped


def decode_stable(processor, generated_ids: list[int]) -> Optional[str]:
    """Decode generated IDs, or return ``None`` if the text ends mid-rune."""
    text = codec.decode_generated(processor, generated_ids)
    if text.endswith("�"):
        return None
    return text


def monotonic_delta(previous: str, current: str) -> Optional[str]:
    """Return the newly appended text, or ``None`` if decoding regressed."""
    if current.startswith(previous):
        return current[len(previous):]
    return None


def log_tokens(generated_ids) -> None:
    path = os.environ.get("PHASESHIFT_SERVER_TOKEN_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(",".join(str(int(t)) for t in generated_ids) + "\n")
    except OSError:
        pass


class ChatService:
    def __init__(self, processor, compute, model_name: str, max_seq_len: int,
                 default_max_output_tokens: int = 4096):
        self._processor = processor
        self._compute = compute
        self._model_name = model_name
        self._max_seq_len = max_seq_len
        self._default_max_output_tokens = default_max_output_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def default_max_output_tokens(self) -> int:
        return self._default_max_output_tokens

    @property
    def compute(self):
        return self._compute

    # ------------------------------------------------------------------ prepare

    def prepare(self, request: ChatRequest) -> PreparedChat:
        messages = openai_messages_to_hf(request.messages)

        tools = request.tools if request.tool_choice != "none" else None

        try:
            rendered = codec.prompt_ids_with_cache_boundary(
                self._processor,
                messages,
                tools=tools,
                enable_thinking=False,
            )
        except OpenAIProtocolError:
            raise
        except Exception as exc:
            raise OpenAIProtocolError(
                f"chat template failed: {exc}",
                param="messages",
            ) from exc

        if len(rendered.ids) >= self._max_seq_len:
            raise OpenAIProtocolError(
                "context length exceeded",
                param="messages",
            )

        remaining = self._max_seq_len - len(rendered.ids)

        if request.max_tokens_explicit and request.max_new_tokens > remaining:
            raise OpenAIProtocolError(
                "max_tokens exceeds remaining context",
                param="max_tokens",
            )

        max_new_tokens = min(request.max_new_tokens, remaining)

        checkpoint = 0
        if self._compute.capabilities.prefix_cache:
            checkpoint = rendered.cache_boundary

        return PreparedChat(
            prompt_ids=rendered.ids,
            tools=tools,
            max_new_tokens=max_new_tokens,
            prefix_cache_checkpoint_position=checkpoint,
        )

    # ----------------------------------------------------------------- non-stream

    async def complete(self, request: ChatRequest) -> AssistantResult:
        prepared = self.prepare(request)

        done = await self._generate(prepared, request)

        generated_ids = [int(x) for x in done.get("generated_ids", [])]
        log_tokens(generated_ids)

        try:
            parsed = codec.parse_assistant_message(
                self._processor,
                prepared.prompt_ids,
                generated_ids,
                tools=prepared.tools,
            )
        except codec.ToolParseError as exc:
            raise ChatServiceError(
                f"tool call parse failure: {exc}",
                code="tool_parse_failure",
            ) from exc
        except codec.ToolCallingUnavailable as exc:
            raise ChatServiceError(
                str(exc),
                code="tool_calling_unavailable",
                status=400,
            ) from exc

        tool_calls = parsed["tool_calls"]

        return AssistantResult(
            content=parsed["content"],
            tool_calls=tool_calls,
            finish_reason=map_finish_reason(
                str(done.get("finish_reason", "")), has_tool_calls=bool(tool_calls)),
            prompt_tokens=len(prepared.prompt_ids),
            completion_tokens=len(generated_ids),
        )

    async def _generate(self, prepared: PreparedChat, request: ChatRequest) -> dict:
        try:
            return await self._compute.generate(
                prepared.prompt_ids,
                prepared.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                seed=request.seed,
                prefix_cache_checkpoint_position=(
                    prepared.prefix_cache_checkpoint_position),
            )
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc), param=None) from exc
        except ComputeCancelled as exc:
            raise ChatServiceError(str(exc), code="cancelled") from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc

    # ------------------------------------------------------------------- streaming

    async def stream(self, request: ChatRequest,
                     prepared: Optional[PreparedChat] = None) -> AsyncIterator[Any]:
        if prepared is None:
            prepared = self.prepare(request)
        inner = self._stream_impl(prepared, request)
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()

    async def _stream_impl(self, prepared: PreparedChat,
                           request: ChatRequest) -> AsyncIterator[Any]:
        if prepared.tools:
            async for event in self._stream_tools(prepared, request):
                yield event
        else:
            async for event in self._stream_text(prepared, request):
                yield event

    def _stream_open(self, prepared: PreparedChat, request: ChatRequest):
        return self._compute.generate_stream(
            prepared.prompt_ids,
            prepared.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            prefix_cache_checkpoint_position=(
                prepared.prefix_cache_checkpoint_position),
        )

    async def _stream_text(self, prepared: PreparedChat,
                           request: ChatRequest) -> AsyncIterator[Any]:
        generated: list[int] = []
        emitted = ""

        stream = self._stream_open(prepared, request)
        try:
            async for event in stream:
                if "token_id" in event:
                    generated.append(event["token_id"])
                    full = decode_stable(self._processor, generated)
                    if full is None:
                        continue
                    delta = monotonic_delta(emitted, full)
                    emitted = full
                    if delta:
                        yield ContentDelta(delta)
                elif "done" in event:
                    done = event["done"]
                    final_ids = [int(x) for x in done.get("generated_ids", generated)]
                    log_tokens(final_ids)
                    final_text = codec.decode_generated(self._processor, final_ids)
                    delta = monotonic_delta(emitted, final_text)
                    if delta:
                        yield ContentDelta(delta)
                    yield StreamDone(map_finish_reason(
                        str(done.get("finish_reason", "")), has_tool_calls=False))
                    return
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc)) from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc
        finally:
            await stream.aclose()

    async def _stream_tools(self, prepared: PreparedChat,
                            request: ChatRequest) -> AsyncIterator[Any]:
        parser = codec.new_response_parser(
            self._processor, prepared.prompt_ids, prompt_prefix=True)
        state: dict[str, Any] = {
            "emitted": "", "content": "", "leading": "", "calls": 0}
        generated: list[int] = []
        incremental_ok = True
        done: dict | None = None

        stream = self._stream_open(prepared, request)
        try:
            async for event in stream:
                if "token_id" in event:
                    generated.append(event["token_id"])
                    if not incremental_ok:
                        continue
                    full = decode_stable(self._processor, generated)
                    if full is None:
                        continue
                    delta = monotonic_delta(state["emitted"], full)
                    if delta is None:
                        incremental_ok = False
                        continue
                    state["emitted"] = full
                    if delta:
                        for item in self._feed_parser(parser, delta, state):
                            yield item
                elif "done" in event:
                    done = event["done"]
                    final_ids = [int(x) for x in done.get("generated_ids", generated)]
                    log_tokens(final_ids)
                    if not incremental_ok:
                        for item in self._parse_final(prepared, final_ids, state):
                            yield item
                        break
                    final_text = codec.decode_generated(self._processor, final_ids)
                    if not final_text.startswith(state["emitted"]):
                        for item in self._parse_final(prepared, final_ids, state):
                            yield item
                        break
                    remainder = final_text[len(state["emitted"]):]
                    state["emitted"] = final_text
                    if remainder:
                        for item in self._feed_parser(parser, remainder, state):
                            yield item

                    try:
                        message, final_events = parser.finalize()
                    except Exception as exc:
                        raise ChatServiceError(
                            f"streaming tool parse failed: {exc}",
                            code="tool_parse_failure",
                        ) from exc

                    if final_events:
                        for item in self._feed_parser(parser, "", state,
                                                      events=final_events):
                            yield item
                    for item in self._finalize_tool_calls(message, state):
                        yield item
                    break
        except ComputeInvalidArgument as exc:
            raise OpenAIProtocolError(str(exc)) from exc
        except ComputeError as exc:
            raise ChatServiceError(str(exc), code="compute_error") from exc
        finally:
            await stream.aclose()

        if done is None:
            raise ChatServiceError("compute ended without a done event",
                                   code="compute_error")
        yield StreamDone(map_finish_reason(
            str(done.get("finish_reason", "")), has_tool_calls=state["calls"] > 0))

    def _feed_parser(self, parser, text: str, state: dict):
        events = parser.feed(text) if text else []
        for event in events:
            kind = event.get("type")
            if kind == "region_chunk" and event.get("field") == "content" \
                    and not event.get("dirty"):
                chunk = event.get("text", "")
                if not chunk:
                    continue
                delta = self._content_delta(state, chunk)
                if delta:
                    yield ContentDelta(delta)
            elif kind == "region_close" and event.get("field") == "tool_calls":
                state["leading"] = ""
                call = _tool_call_delta(event.get("value"), state["calls"])
                if call is not None:
                    state["calls"] += 1
                    yield call

    def _content_delta(self, state: dict, chunk: str) -> str:
        if state["content"] == "":
            state["leading"] += chunk
            stripped = state["leading"].lstrip()
            if stripped == "":
                return ""
            state["leading"] = ""
            state["content"] = stripped
            return stripped
        state["content"] += chunk
        return chunk

    def _finalize_tool_calls(self, message, state: dict):
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls[state["calls"]:]:
            delta = _tool_call_delta(call, state["calls"])
            if delta is not None:
                state["calls"] += 1
                yield delta
        content = message.get("content") or ""
        if content.startswith(state["content"]):
            tail = content[len(state["content"]):]
            if tail:
                state["content"] += tail
                yield ContentDelta(tail)

    def _parse_final(self, prepared: PreparedChat, final_ids: list[int], state: dict):
        try:
            parsed = codec.parse_assistant_message(
                self._processor, prepared.prompt_ids, final_ids, tools=prepared.tools)
        except codec.ToolParseError as exc:
            raise ChatServiceError(
                f"tool call parse failure: {exc}",
                code="tool_parse_failure",
            ) from exc
        except codec.ToolCallingUnavailable as exc:
            raise ChatServiceError(
                str(exc), code="tool_calling_unavailable", status=400) from exc

        for call in parsed["tool_calls"][state["calls"]:]:
            delta = _tool_call_delta(call, state["calls"])
            if delta is not None:
                state["calls"] += 1
                yield delta
        content = parsed["content"] or ""
        if content.startswith(state["content"]):
            tail = content[len(state["content"]):]
            if tail:
                state["content"] += tail
                yield ContentDelta(tail)


def _tool_call_delta(value, index: int) -> Optional[ToolCallDelta]:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        payload = arguments
    else:
        payload = json.dumps(
            {} if arguments is None else arguments,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return ToolCallDelta(
        index=index,
        call_id=f"call_{index}",
        name=str(name),
        arguments=payload,
    )
