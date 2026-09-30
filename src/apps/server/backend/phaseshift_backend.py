#!/usr/bin/env python3
"""PhaseShift LocalAI external gRPC backend.

LocalAI spawns this process with ``--addr HOST:PORT`` and drives the standard
``backend.Backend`` gRPC service. The backend owns a persistent
``phaseshift-compute --serve-stdio`` child and keeps the boundary strict:

    LocalAI messages -> HF chat template -> token IDs -> compute
    compute generated IDs -> HF response parser -> LocalAI ChatDelta

No HTTP, OpenAI schema, JSON Schema, tool execution, or chat-template logic
lives in ``phaseshift-compute``; no tokenizer or chat template lives in LocalAI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent
for _extra in (_HERE, _LIB, _LIB / "common"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import grpc  # noqa: E402

from localai_proto import backend_pb2 as pb  # noqa: E402
from localai_proto import backend_pb2_grpc as pb_grpc  # noqa: E402
from phaseshift_chat import codec  # noqa: E402
from phaseshift_chat import tool_constraint  # noqa: E402

import compute_client  # noqa: E402
import message_codec  # noqa: E402

DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_ARENA_GIB = 16
DEFAULT_PAGE_TOKENS = 16
DEFAULT_KV_CACHE_DTYPE = "bf16"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_DEFAULT_MAX_OUTPUT_TOKENS = 4096

TOOL_POLICY_TRANSPORT_KEY = "phaseshift.tool_policy_transport"
PARALLEL_TOOL_CALLS_KEY = "phaseshift.parallel_tool_calls"
COMPOSITION_TRANSPORT_KEY = "phaseshift.constraint_composition_transport"


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedRequest:
    prompt: list[int]
    tools: list[dict] | None
    tools_requested: bool
    tool_policy: tool_constraint.ToolChoicePolicy
    parallel_tool_calls: bool
    structural_tag: str | None
    grammar: str
    cache_checkpoint_position: int = 0
    reasoning_enabled: bool = False
    reasoning_effort: str = ""


def _parse_options(options) -> dict:
    parsed: dict[str, str] = {}
    for raw in options or []:
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_compute_binary(options: dict) -> str:
    explicit = options.get("phaseshift_compute") or os.environ.get("PHASESHIFT_COMPUTE")
    if explicit:
        return explicit
    sibling = _HERE.parent / "phaseshift-compute"
    if sibling.is_file():
        return str(sibling)
    return "phaseshift-compute"


class PhaseShiftBackend(pb_grpc.BackendServicer):
    def __init__(self) -> None:
        self._client: compute_client.ComputeClient | None = None
        self._processor = None
        self._model_dir: str | None = None
        self._max_seq_len = DEFAULT_MAX_SEQ_LEN
        self._default_max_output_tokens = DEFAULT_DEFAULT_MAX_OUTPUT_TOKENS
        self._loaded_signature: tuple | None = None
        self._constraint_dir: str | None = None
        self._prefix_cache_enabled = False

    # ------------------------------------------------------------------ health

    def Health(self, request, context):
        return pb.Reply(message=b"OK")

    def Free(self, request, context):
        self._close_client()
        return pb.Result(success=True)

    # ------------------------------------------------------------------ loading

    def LoadModel(self, request, context):
        options = _parse_options(request.Options)
        model_dir = options.get("phaseshift_model_dir")
        if not model_dir:
            return pb.Result(
                success=False,
                message="phaseshift_model_dir option is required",
            )
        if not Path(model_dir).is_dir():
            return pb.Result(
                success=False,
                message=f"phaseshift_model_dir does not exist: {model_dir}",
            )

        max_seq_len = int(request.ContextSize) if request.ContextSize > 0 else DEFAULT_MAX_SEQ_LEN
        arena_gib = int(options.get("phaseshift_arena_gib", DEFAULT_ARENA_GIB))
        page_tokens = int(options.get("phaseshift_page_tokens", DEFAULT_PAGE_TOKENS))
        device = int(options.get("phaseshift_device", "0"))
        kv_cache_dtype = options.get("phaseshift_kv_cache_dtype", DEFAULT_KV_CACHE_DTYPE)
        dflash2_model_dir = options.get("phaseshift_dflash2_model_dir") or None
        dflash2_drafts = int(options.get("phaseshift_dflash2_drafts", "7"))
        if dflash2_model_dir is not None and not Path(dflash2_model_dir).is_dir():
            return pb.Result(
                success=False,
                message=f"phaseshift_dflash2_model_dir does not exist: {dflash2_model_dir}",
            )
        verify_weights = options.get("phaseshift_verify_weights", "0") == "1"
        max_concurrent_requests = int(options.get("phaseshift_max_concurrent_requests", "1"))
        if max_concurrent_requests < 1:
            return pb.Result(
                success=False,
                message="phaseshift_max_concurrent_requests must be >= 1",
            )
        kv_cache_capacity_tokens = int(
            options.get("phaseshift_kv_cache_capacity_tokens", "0"))
        prefix_cache_capacity_tokens = int(
            options.get("phaseshift_prefix_cache_capacity_tokens", "0"))
        prefix_cache_max_entries = int(
            options.get("phaseshift_prefix_cache_max_entries", "16"))
        if prefix_cache_capacity_tokens > 0 and prefix_cache_max_entries < 1:
            return pb.Result(
                success=False,
                message="phaseshift_prefix_cache_max_entries must be >= 1 "
                        "when the prefix cache is enabled",
            )
        default_max_output_tokens = int(
            options.get("phaseshift_default_max_output_tokens",
                        DEFAULT_DEFAULT_MAX_OUTPUT_TOKENS))
        if default_max_output_tokens < 1:
            return pb.Result(
                success=False,
                message="phaseshift_default_max_output_tokens must be >= 1",
            )
        compute_binary = _resolve_compute_binary(options)

        signature = (model_dir, max_seq_len, arena_gib, page_tokens, device,
                     kv_cache_dtype, verify_weights, max_concurrent_requests,
                     kv_cache_capacity_tokens, prefix_cache_capacity_tokens,
                     prefix_cache_max_entries, default_max_output_tokens,
                     compute_binary, dflash2_model_dir, dflash2_drafts)
        if self._loaded_signature == signature and self._client is not None and self._client.is_alive():
            return pb.Result(success=True, message="already loaded")
        if self._loaded_signature is not None and self._loaded_signature != signature:
            return pb.Result(
                success=False,
                message="PhaseshiftGate1 supports a single model; refusing to reload a different model",
            )

        try:
            if self._processor is None:
                self._processor = codec.load_processor(model_dir)
                caps = codec.probe_tool_calling(self._processor)
                if caps["chat_template_tools"] and caps["response_parser"]:
                    print("Tool calling: available", file=sys.stderr, flush=True)
                else:
                    print(f"Tool calling: unavailable ({caps['detail']})",
                          file=sys.stderr, flush=True)
                if not codec.thinking_available(self._processor):
                    self._processor = None
                    self._close_client()
                    return pb.Result(
                        success=False,
                        message="model chat template has no thinking mode; "
                                "PhaseShift supports Qwen3.5 only")
                print("Reasoning: available (opt-in)", file=sys.stderr, flush=True)

            constraint_dir = None
            constraint_path = None
            try:
                written = _write_constraint_tokenizer_info(self._processor, model_dir)
                if written is not None:
                    constraint_dir, constraint_path = written
                    self._constraint_dir = constraint_dir
            except BackendError as exc:
                self._close_client()
                return pb.Result(success=False, message=str(exc))

            argv = [
                compute_binary,
                "--model-dir", model_dir,
                "--serve-stdio",
                "--max-seq-len", str(max_seq_len),
                "--arena-gib", str(arena_gib),
                "--page-tokens", str(page_tokens),
                "--device", str(device),
                "--kv-cache-dtype", kv_cache_dtype,
                "--max-concurrent-requests", str(max_concurrent_requests),
                "--kv-cache-capacity-tokens", str(kv_cache_capacity_tokens),
                "--prefix-cache-capacity-tokens", str(prefix_cache_capacity_tokens),
                "--prefix-cache-max-entries", str(prefix_cache_max_entries),
            ]
            if constraint_path is not None:
                argv += ["--constraint-tokenizer-info", constraint_path]
            if dflash2_model_dir is not None:
                argv += ["--dflash2-model-dir", dflash2_model_dir,
                         "--dflash2-drafts", str(dflash2_drafts)]
            if verify_weights:
                argv += ["--verify-weights", "1"]

            client = compute_client.ComputeClient(
                argv, log=sys.stderr, max_inflight=max_concurrent_requests)
            client.start()
        except Exception as exc:  # noqa: BLE001 - report load failure to LocalAI
            self._close_client()
            return pb.Result(success=False, message=f"compute startup failed: {exc}")

        self._client = client
        self._model_dir = model_dir
        self._max_seq_len = max_seq_len
        self._default_max_output_tokens = default_max_output_tokens
        self._loaded_signature = signature
        self._prefix_cache_enabled = prefix_cache_capacity_tokens > 0
        self._constraint_dir = constraint_dir
        return pb.Result(success=True, message="loaded")

    def ModelMetadata(self, request, context):
        processor = self._processor
        if processor is None:
            return pb.ModelMetadataResponse(supports_thinking=False, rendered_template="")
        supported = codec.thinking_available(processor)
        template = codec.render_thinking_probe(processor) if supported else ""
        return pb.ModelMetadataResponse(
            supports_thinking=supported, rendered_template=template)

    # ------------------------------------------------------------------ predict

    def _require_loaded(self):
        if self._client is None or self._processor is None:
            raise BackendError("model is not loaded")
        if not self._client.is_alive():
            raise BackendError("compute process is not running")

    def _prepare(self, request) -> PreparedRequest:
        self._require_loaded()
        _log_request(request)
        try:
            reasoning_enabled, reasoning_effort = codec.resolve_reasoning_policy(
                request.Metadata)
        except codec.ReasoningEffortError as exc:
            raise BackendError(str(exc)) from exc
        messages = message_codec.proto_messages_to_hf(request.Messages)
        tools = message_codec.normalize_tools(message_codec.parse_tools(request.Tools))
        policy = tool_constraint.resolve_tool_choice(request.ToolChoice)

        tools_present = bool(tools)
        if reasoning_enabled and not codec.thinking_available(self._processor):
            raise BackendError(
                "reasoning was requested but this model chat template has no "
                "thinking mode")
        if tools_present:
            if request.Metadata.get(TOOL_POLICY_TRANSPORT_KEY) != "1":
                raise BackendError(
                    "tool calling requires the PhaseShift bundled LocalAI runtime "
                    "(missing tool policy transport marker)")
        parallel = tool_constraint.parse_parallel_tool_calls(
            request.Metadata.get(PARALLEL_TOOL_CALLS_KEY))

        composition = request.Metadata.get(COMPOSITION_TRANSPORT_KEY) == "1"
        response_grammar = request.Grammar or ""
        if composition and (not tools_present or not response_grammar):
            raise BackendError(
                "constraint composition marker present without tools or a response grammar")

        tools_for_template = tools
        tools_requested = tools_present
        if policy.mode == "none":
            tools_for_template = None
            tools_requested = False

        if tools_requested and not codec.tool_calling_available(self._processor):
            raise BackendError("tool calling requested but unavailable in this environment")

        structural_tag = None
        if reasoning_enabled:
            grammar = ""
            structural_tag = tool_constraint.build_reasoning_structural_tag(
                tools, policy, parallel, response_grammar, composition)
        elif composition:
            if policy.mode == "none":
                grammar = response_grammar
            elif policy.mode == "auto":
                grammar = ""
                structural_tag = tool_constraint.build_composite_structural_tag(
                    tools, policy, parallel, response_grammar)
            else:
                grammar = ""
                structural_tag = tool_constraint.build_structural_tag(
                    tools, policy, parallel)
        elif tools_requested:
            grammar = ""
            structural_tag = tool_constraint.build_structural_tag(tools, policy, parallel)
        else:
            grammar = request.Grammar or ""

        render = _render_prompt(
            self._processor, messages, tools_for_template,
            enable_thinking=reasoning_enabled)
        prompt_log = os.environ.get("PHASESHIFT_BACKEND_PROMPT_LOG")
        if prompt_log:
            try:
                import hashlib
                digest = hashlib.sha256(
                    ",".join(str(int(t)) for t in render.ids).encode("utf-8")
                ).hexdigest()[:16]
                with open(prompt_log, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"{digest} len={len(render.ids)} "
                        f"tools={tools_requested} reasoning={reasoning_enabled}\n")
            except OSError:
                pass
        return PreparedRequest(
            prompt=render.ids,
            tools=tools,
            tools_requested=tools_requested,
            tool_policy=policy,
            parallel_tool_calls=parallel,
            structural_tag=structural_tag,
            grammar=grammar,
            cache_checkpoint_position=(
                render.cache_boundary if self._prefix_cache_enabled else 0),
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
        )

    def _max_new_tokens(self, request, prompt_len: int) -> int:
        remaining = self._max_seq_len - prompt_len
        if remaining <= 0:
            raise BackendError("prompt exceeds the configured context window")
        if request.Tokens > 0:
            requested = int(request.Tokens)
            if requested > remaining:
                raise BackendError(
                    "requested max_tokens exceeds the remaining context window "
                    f"({requested} > {remaining})")
            return requested
        return min(self._default_max_output_tokens, remaining)

    def Predict(self, request, context):
        try:
            prepared = self._prepare(request)
        except (BackendError, message_codec.MessageError,
                tool_constraint.ToolConstraintError) as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return pb.Reply()

        try:
            max_new_tokens = self._max_new_tokens(request, len(prepared.prompt))
        except BackendError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return pb.Reply()
        try:
            done = self._client.generate(
                prepared.prompt,
                max_new_tokens,
                temperature=float(request.Temperature),
                top_p=float(request.TopP) if request.TopP > 0 else 1.0,
                top_k=int(request.TopK),
                seed=int(request.Seed),
                should_cancel=lambda: not context.is_active(),
                grammar=prepared.grammar,
                structural_tag=prepared.structural_tag,
                prefix_cache_checkpoint_position=prepared.cache_checkpoint_position,
            )
        except compute_client.ComputeCancelled as exc:
            if context.is_active():
                context.set_code(grpc.StatusCode.CANCELLED)
                context.set_details(str(exc))
            return pb.Reply()
        except compute_client.ComputeInvalidArgument as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"compute error: {exc}")
            return pb.Reply()
        except compute_client.ComputeError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"compute error: {exc}")
            return pb.Reply()

        generated = [int(t) for t in done.get("generated_ids", [])]
        _log_tokens(generated)
        try:
            parsed = codec.parse_assistant_message(
                self._processor, prepared.prompt, generated,
                prepared.tools if prepared.tools_requested else None,
                reasoning=prepared.reasoning_enabled)
        except codec.ToolParseError as exc:
            print(f"tool parse failure: {exc}", file=sys.stderr, flush=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb.Reply()
        except codec.ToolCallingUnavailable as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return pb.Reply()

        content = parsed["content"]
        reasoning_content = parsed.get("reasoning_content", "")
        tool_calls = parsed["tool_calls"]

        reply = pb.Reply(tokens=len(generated), prompt_tokens=len(prepared.prompt))
        if tool_calls:
            if content:
                reply.message = content.encode("utf-8")
            delta = pb.ChatDelta(reasoning_content=reasoning_content, content=content)
            for index, call in enumerate(tool_calls):
                function = call["function"]
                delta.tool_calls.append(pb.ToolCallDelta(
                    index=index,
                    id=_new_tool_call_id(),
                    name=function["name"],
                    arguments=_arguments_to_json(function["arguments"]),
                ))
            reply.chat_deltas.append(delta)
        elif reasoning_content:
            reply.message = content.encode("utf-8")
            reply.chat_deltas.append(pb.ChatDelta(
                reasoning_content=reasoning_content, content=content))
        else:
            reply.message = content.encode("utf-8")

        return reply

    def PredictStream(self, request, context):
        try:
            prepared = self._prepare(request)
        except (BackendError, message_codec.MessageError,
                tool_constraint.ToolConstraintError) as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return

        try:
            max_new_tokens = self._max_new_tokens(request, len(prepared.prompt))
        except BackendError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return
        try:
            if not prepared.tools_requested:
                yield from self._stream_text(prepared.prompt, request, max_new_tokens,
                                             context, prepared.grammar,
                                             prepared.cache_checkpoint_position,
                                             prepared.reasoning_enabled)
            else:
                if not codec.streaming_tool_calling_available(self._processor):
                    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                    context.set_details(
                        "streaming tool calls unavailable in this environment")
                    return
                yield from self._stream_tools(
                    prepared.prompt, prepared.tools, request, max_new_tokens, context,
                    prepared.structural_tag, prepared.cache_checkpoint_position,
                    prepared.reasoning_enabled)
        except compute_client.ComputeCancelled as exc:
            if context.is_active():
                context.set_code(grpc.StatusCode.CANCELLED)
                context.set_details(str(exc))
            return
        except compute_client.ComputeInvalidArgument as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"compute error: {exc}")
            return
        except compute_client.ComputeError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"compute error: {exc}")
            return

    def _stream_text(self, prompt, request, max_new_tokens, context, grammar="",
                     checkpoint_position=0, reasoning_enabled=False):
        generated: list[int] = []
        emitted = ""
        last_reasoning = ""
        last_content = ""
        stream = self._client.generate_stream(
            prompt,
            max_new_tokens,
            temperature=float(request.Temperature),
            top_p=float(request.TopP) if request.TopP > 0 else 1.0,
            top_k=int(request.TopK),
            seed=int(request.Seed),
            grammar=grammar,
            prefix_cache_checkpoint_position=checkpoint_position,
        )
        try:
            for event in stream:
                if not context.is_active():
                    return
                if "token_id" in event:
                    generated.append(event["token_id"])
                    full = _decode_stable(self._processor, generated)
                    if full is None:
                        continue
                    if reasoning_enabled:
                        reasoning, content = codec.split_reasoning_partial(full)
                        reasoning_delta = _suffix_delta(last_reasoning, reasoning)
                        content_delta = _suffix_delta(last_content, content)
                        last_reasoning, last_content = reasoning, content
                        if reasoning_delta or content_delta:
                            yield pb.Reply(
                                message=content_delta.encode("utf-8"),
                                tokens=len(generated),
                                prompt_tokens=len(prompt),
                                chat_deltas=[pb.ChatDelta(
                                    reasoning_content=reasoning_delta,
                                    content=content_delta)],
                            )
                    else:
                        delta = _monotonic_delta(emitted, full)
                        emitted = full
                        if delta:
                            yield pb.Reply(
                                message=delta.encode("utf-8"),
                                tokens=len(generated),
                                prompt_tokens=len(prompt),
                            )
                elif "done" in event:
                    final_ids = [int(t) for t in event["done"].get("generated_ids", generated)]
                    _log_tokens(final_ids)
                    final_text = codec.decode_generated(self._processor, final_ids)
                    if reasoning_enabled:
                        reasoning, content = codec.split_reasoning_final(final_text)
                        reasoning_delta = _suffix_delta(last_reasoning, reasoning)
                        content_delta = _suffix_delta(last_content, content)
                        if reasoning_delta or content_delta:
                            yield pb.Reply(
                                message=content_delta.encode("utf-8"),
                                tokens=len(final_ids),
                                prompt_tokens=len(prompt),
                                chat_deltas=[pb.ChatDelta(
                                    reasoning_content=reasoning_delta,
                                    content=content_delta)],
                            )
                    else:
                        delta = _monotonic_delta(emitted, final_text)
                        if delta:
                            yield pb.Reply(
                                message=delta.encode("utf-8"),
                                tokens=len(final_ids),
                                prompt_tokens=len(prompt),
                            )
                    break
        finally:
            stream.close()

    def _stream_tools(self, prompt, tools, request, max_new_tokens, context,
                      structural_tag=None, checkpoint_position=0,
                      reasoning_enabled=False):
        parser = codec.new_response_parser(
            self._processor, prompt, prompt_prefix=not reasoning_enabled)
        state = {"emitted": "", "content": "", "leading": "", "calls": 0,
                 "reasoning": "", "final": ""}
        generated: list[int] = []
        incremental_ok = True

        stream = self._client.generate_stream(
            prompt,
            max_new_tokens,
            temperature=float(request.Temperature),
            top_p=float(request.TopP) if request.TopP > 0 else 1.0,
            top_k=int(request.TopK),
            seed=int(request.Seed),
            structural_tag=structural_tag,
            prefix_cache_checkpoint_position=checkpoint_position,
        )
        try:
            for event in stream:
                if not context.is_active():
                    return
                if "token_id" in event:
                    generated.append(event["token_id"])
                    if not incremental_ok:
                        continue
                    full = _decode_stable(self._processor, generated)
                    if full is None:
                        continue
                    if reasoning_enabled:
                        yield from self._feed_reasoning_tools(
                            parser, state, full, final=False)
                    else:
                        delta = _monotonic_delta(state["emitted"], full)
                        if delta is None:
                            incremental_ok = False
                            continue
                        state["emitted"] = full
                        if delta:
                            yield from self._feed_parser(parser, delta, state)
                elif "done" in event:
                    final_ids = [int(t) for t in event["done"].get("generated_ids", generated)]
                    _log_tokens(final_ids)

                    if not incremental_ok:
                        yield from self._stream_tools_fallback(
                            prompt, tools, final_ids, state, reasoning_enabled)
                        break

                    final_text = codec.decode_generated(self._processor, final_ids)
                    if not final_text.startswith(state["emitted"]):
                        yield from self._stream_tools_fallback(
                            prompt, tools, final_ids, state, reasoning_enabled)
                        break

                    if reasoning_enabled:
                        yield from self._feed_reasoning_tools(
                            parser, state, final_text, final=True)
                    else:
                        remainder = final_text[len(state["emitted"]):]
                        state["emitted"] = final_text
                        if remainder:
                            yield from self._feed_parser(parser, remainder, state)

                    try:
                        message, final_events = parser.finalize()
                    except Exception as exc:  # noqa: BLE001
                        print(f"streaming tool parse finalize failure: {exc}",
                              file=sys.stderr, flush=True)
                        raise compute_client.ComputeError(
                            f"streaming tool parse failed: {exc}") from exc

                    if final_events:
                        yield from self._feed_parser(parser, "", state, events=final_events)
                    yield from self._finalize_tool_calls(message, state)
                    break
        finally:
            stream.close()

    def _feed_reasoning_tools(self, parser, state, full, final):
        """Split accumulated text into reasoning and post-delimiter final text.

        Reasoning deltas are emitted first; only the post-delimiter text is fed
        to the tool parser, so the parser never sees reasoning and the
        transition is one-way (reasoning -> tool/content, never back).
        """
        if final:
            reasoning, final_text = codec.split_reasoning_final(full)
        else:
            reasoning, final_text = codec.split_reasoning_partial(full)
        state["emitted"] = full
        reasoning_delta = _suffix_delta(state["reasoning"], reasoning)
        final_delta = _suffix_delta(state["final"], final_text)
        state["reasoning"], state["final"] = reasoning, final_text
        if reasoning_delta:
            yield pb.Reply(chat_deltas=[pb.ChatDelta(reasoning_content=reasoning_delta)])
        if final_delta:
            yield from self._feed_parser(parser, final_delta, state)

    def _feed_parser(self, parser, text, state, events=None):
        if events is None:
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
                    yield pb.Reply(chat_deltas=[pb.ChatDelta(content=delta)])
            elif kind == "region_close" and event.get("field") == "tool_calls":
                state["leading"] = ""
                call = _tool_call_delta(event.get("value"), state["calls"])
                if call is not None:
                    state["calls"] += 1
                    yield pb.Reply(chat_deltas=[pb.ChatDelta(tool_calls=[call])])

    def _content_delta(self, state, chunk):
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

    def _finalize_tool_calls(self, message, state):
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls[state["calls"]:]:
            delta = _tool_call_delta(call, state["calls"])
            if delta is not None:
                state["calls"] += 1
                yield pb.Reply(chat_deltas=[pb.ChatDelta(tool_calls=[delta])])
        content = message.get("content") or ""
        if content.startswith(state["content"]):
            tail = content[len(state["content"]):]
            if tail:
                state["content"] += tail
                yield pb.Reply(chat_deltas=[pb.ChatDelta(content=tail)])

    def _stream_tools_fallback(self, prompt, tools, final_ids, state,
                               reasoning_enabled=False):
        try:
            parsed = codec.parse_assistant_message(
                self._processor, prompt, final_ids, tools,
                reasoning=reasoning_enabled)
        except codec.ToolParseError as exc:
            raise compute_client.ComputeError(f"streaming tool parse failed: {exc}") from exc
        if reasoning_enabled:
            reasoning = parsed["reasoning_content"] or ""
            delta = _suffix_delta(state["reasoning"], reasoning)
            if delta:
                state["reasoning"] += delta
                yield pb.Reply(chat_deltas=[pb.ChatDelta(reasoning_content=delta)])
        for call in parsed["tool_calls"][state["calls"]:]:
            delta = _tool_call_delta(call, state["calls"])
            if delta is not None:
                state["calls"] += 1
                yield pb.Reply(chat_deltas=[pb.ChatDelta(tool_calls=[delta])])
        content = parsed["content"] or ""
        if content.startswith(state["content"]):
            tail = content[len(state["content"]):]
            if tail:
                state["content"] += tail
                yield pb.Reply(chat_deltas=[pb.ChatDelta(content=tail)])

    # ------------------------------------------------------------------ teardown

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            self._client = None
        if self._constraint_dir is not None:
            try:
                shutil.rmtree(self._constraint_dir, ignore_errors=True)
            finally:
                self._constraint_dir = None


def _render_prompt(processor, messages, tools, enable_thinking=False):
    try:
        return codec.prompt_ids_with_cache_boundary(
            processor, messages, tools=tools, enable_thinking=enable_thinking)
    except Exception as exc:  # noqa: BLE001 - chat-template failures are request errors
        raise BackendError(f"chat template failed: {exc}") from exc


def _model_text_config(model_dir):
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("text_config", config)


def _generation_stop_tokens(model_dir):
    """Generation stop token ids from generation_config.json.

    Qwen3.5/3.8 list both <|im_end|> (turn end) and <|endoftext|>. Falls back to
    the single text_config.eos_token_id when no generation config is present.
    """
    path = Path(model_dir) / "generation_config.json"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            config = {}
        eos = config.get("eos_token_id") if isinstance(config, dict) else None
        if isinstance(eos, bool):
            eos = None
        if isinstance(eos, int):
            return [int(eos)]
        if isinstance(eos, list):
            tokens = [int(t) for t in eos if isinstance(t, int) and not isinstance(t, bool)]
            if tokens:
                return tokens
    eos = _model_text_config(model_dir).get("eos_token_id")
    return [int(eos)] if eos is not None else []


EXPECTED_XGRAMMAR_VERSION = "0.2.5.post1"


def _write_constraint_tokenizer_info(processor, model_dir):
    try:
        import importlib.metadata

        package_version = importlib.metadata.version("xgrammar")
    except Exception as exc:  # noqa: BLE001 - xgrammar is a required server dependency
        raise BackendError(
            "xgrammar is required for phaseshift-server structured generation "
            f"(pip install xgrammar=={EXPECTED_XGRAMMAR_VERSION}): {exc}") from exc
    if package_version != EXPECTED_XGRAMMAR_VERSION:
        raise BackendError(
            f"xgrammar {package_version} is not the supported version "
            f"({EXPECTED_XGRAMMAR_VERSION})")
    import xgrammar
    text_config = _model_text_config(model_dir)
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab_size = int(text_config.get("vocab_size", 0)) or None
    stop_tokens = _generation_stop_tokens(model_dir)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if stop_tokens and tokenizer_eos is not None and int(tokenizer_eos) not in stop_tokens:
        print(
            f"Generation stop tokens: {stop_tokens} (tokenizer EOS metadata: "
            f"{tokenizer_eos}); using the model generation stop tokens for XGrammar",
            file=sys.stderr, flush=True)
    try:
        info = xgrammar.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size,
            stop_token_ids=stop_tokens or None,
        )
        serialized = info.serialize_json()
    except Exception as exc:  # noqa: BLE001 - report a clear load failure
        raise BackendError(f"xgrammar tokenizer info failed: {exc}") from exc
    directory = tempfile.mkdtemp(prefix="phaseshift-constraint-")
    path = Path(directory) / "xgrammar-tokenizer-info.json"
    path.write_text(serialized, encoding="utf-8")
    print("Grammar constraints: enabled", file=sys.stderr, flush=True)
    return directory, str(path)


def _log_tokens(generated_ids) -> None:
    path = os.environ.get("PHASESHIFT_BACKEND_TOKEN_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(",".join(str(int(t)) for t in generated_ids) + "\n")
    except OSError:
        pass


def _log_request(request) -> None:
    path = os.environ.get("PHASESHIFT_BACKEND_REQUEST_LOG")
    if not path:
        return
    try:
        summary = {
            "prompt_empty": not request.Prompt,
            "messages": [
                {
                    "role": m.role,
                    "content_len": len(m.content or ""),
                    "reasoning_len": len(m.reasoning_content or ""),
                    "has_tool_calls": bool(m.tool_calls),
                    "name": m.name,
                    "tool_call_id": m.tool_call_id,
                }
                for m in request.Messages
            ],
            "tools": request.Tools,
            "tool_choice": request.ToolChoice,
            "grammar_len": len(request.Grammar or ""),
            "tokens": request.Tokens,
            "temperature": request.Temperature,
            "top_p": request.TopP,
            "top_k": request.TopK,
            "seed": request.Seed,
            "metadata": dict(request.Metadata),
        }
        if os.environ.get("PHASESHIFT_BACKEND_REQUEST_LOG_FULL") == "1":
            summary["grammar"] = request.Grammar or ""
            summary["proto"] = str(request)[:6000]
            summary["messages_full"] = [
                {"role": m.role, "content": m.content,
                 "reasoning_content": m.reasoning_content,
                 "tool_calls": m.tool_calls, "name": m.name,
                 "tool_call_id": m.tool_call_id}
                for m in request.Messages
            ]
        import json
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _decode_stable(processor, generated_ids):
    """Decode generated IDs, or return ``None`` if the text ends mid-rune.

    ``decode`` of a byte-level BPE prefix can end in an incomplete UTF-8
    sequence, which the tokenizer renders as U+FFFD. Holding that token back
    until the next token completes the character keeps the emitted-text
    stream a monotonic prefix of the final text.
    """
    text = codec.decode_generated(processor, generated_ids)
    if text.endswith("\ufffd"):
        return None
    return text


def _monotonic_delta(previous: str, current: str):
    """Return the newly appended text, or ``None`` if decoding regressed."""
    if current.startswith(previous):
        return current[len(previous):]
    return None


def _suffix_delta(previous: str, current: str) -> str:
    """Return the part of ``current`` not already covered by ``previous``.

    The reasoning splitter can retract a buffered delimiter tail, so unlike
    ``_monotonic_delta`` this tolerates a shorter ``current`` and only re-emits
    the genuinely new suffix.
    """
    if current.startswith(previous):
        return current[len(previous):]
    limit = min(len(previous), len(current))
    index = 0
    while index < limit and previous[index] == current[index]:
        index += 1
    return current[index:]


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


def _tool_call_delta(value, index: int):
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"value": arguments}
    return pb.ToolCallDelta(
        index=index,
        id=_new_tool_call_id(),
        name=name,
        arguments=_arguments_to_json(arguments),
    )


def _arguments_to_json(arguments) -> str:
    import json

    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def main() -> int:
    import signal

    parser = argparse.ArgumentParser(description="PhaseShift LocalAI backend")
    parser.add_argument("--addr", required=True, help="gRPC listen address (LocalAI supplies this)")
    args = parser.parse_args()

    backend = PhaseShiftBackend()
    try:
        gpu_concurrency = int(os.environ.get("PHASESHIFT_MAX_CONCURRENT_REQUESTS", "1"))
    except ValueError:
        gpu_concurrency = 1
    gpu_concurrency = max(1, gpu_concurrency)
    max_workers = max(4, gpu_concurrency + 2)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_BackendServicer_to_server(backend, server)
    server.add_insecure_port(args.addr)
    server.start()

    def _stop(signum, frame):
        backend._close_client()
        server.stop(grace=0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    server.wait_for_termination()
    backend._close_client()
    return 0


if __name__ == "__main__":
    sys.exit(main())
