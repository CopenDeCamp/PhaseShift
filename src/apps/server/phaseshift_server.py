#!/usr/bin/env python3
"""PhaseShift Server: OpenAI Chat Completions adapter over aiohttp.

    ./build/phaseshift-server --model-dir /path/to/model --port 8000

The server speaks an explicit subset of the OpenAI Chat Completions API and
drives ``phaseshift-compute --serve-stdio`` directly over JSONL. HTTP itself is
served here; there is no external frontend and no constrained decoding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from aiohttp import web

_HERE = Path(__file__).resolve().parent


def _bootstrap_paths() -> None:
    staged = _HERE / "phaseshift-server-lib"
    candidates = [
        _HERE.parent / "common",
        _HERE,
        staged / "common",
        staged,
    ]
    override = os.environ.get("PHASESHIFT_SERVER_LIB")
    if override:
        candidates.append(Path(override).resolve())
    for root in candidates:
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


_bootstrap_paths()

from phaseshift_chat import codec  # noqa: E402

from serverlib import openai_protocol  # noqa: E402
from serverlib.chat_service import ChatService, StreamDone  # noqa: E402
from serverlib.compute_client import AsyncComputeClient, ComputeError  # noqa: E402
from serverlib.openai_protocol import (  # noqa: E402
    ChatServiceError,
    OpenAIProtocolError,
    parse_chat_request,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _resolve_compute_binary(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit).resolve()
        if not candidate.is_file():
            raise SystemExit(f"phaseshift-compute not found: {candidate}")
        return str(candidate)
    env = os.environ.get("PHASESHIFT_COMPUTE")
    if env:
        return env
    candidate = _HERE / "phaseshift-compute"
    if candidate.is_file():
        return str(candidate)
    return "phaseshift-compute"


def build_compute_argv(opts) -> list[str]:
    argv = [
        opts.compute,
        "--model-dir", opts.model_dir,
        "--serve-stdio",
        "--max-seq-len", str(opts.max_seq_len),
        "--arena-gib", str(opts.arena_gib),
        "--page-tokens", str(opts.page_tokens),
        "--device", str(opts.device),
        "--kv-cache-dtype", opts.kv_cache_dtype,
        "--max-concurrent-requests", str(opts.max_concurrent_requests),
        "--kv-cache-capacity-tokens", str(opts.kv_cache_capacity_tokens),
    ]
    if opts.dflash2_model_dir:
        argv += [
            "--dflash2-model-dir", opts.dflash2_model_dir,
            "--dflash2-drafts", str(opts.dflash2_drafts),
        ]
    if opts.verify_weights:
        argv += ["--verify-weights", "1"]
    return argv


async def handle_models(request: web.Request) -> web.Response:
    service: ChatService = request.app["chat_service"]

    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": service.model_name,
                "object": "model",
                "owned_by": "phaseshift",
            }
        ],
    })


def _error_response(exc) -> web.Response:
    return web.json_response(openai_protocol.error_body(
        exc.message, param=exc.param, code=exc.code), status=exc.status)


def _log_request(chat, result) -> None:
    path = os.environ.get("PHASESHIFT_SERVER_REQUEST_LOG")
    if not path:
        return
    summary = {
        "stream": chat.stream,
        "tool_choice": chat.tool_choice,
        "messages": [
            {
                "role": m.get("role"),
                "has_tool_calls": bool(m.get("tool_calls")),
            }
            for m in chat.messages
        ],
        "finish_reason": result,
    }
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except OSError:
        pass


async def handle_chat_completions(request: web.Request) -> web.Response:
    service: ChatService = request.app["chat_service"]

    try:
        body = await _read_json(request)
        chat = parse_chat_request(
            body, default_max_new_tokens=service.default_max_output_tokens)

        if chat.model != service.model_name:
            raise OpenAIProtocolError(
                f"model not found: {chat.model}",
                param="model",
                code="model_not_found",
                status=404,
            )

        if chat.stream:
            prepared = service.prepare(chat)
            return await handle_streaming(request, service, chat, prepared)

        result = await service.complete(chat)
        _log_request(chat, result.finish_reason)
        return web.json_response(openai_protocol.encode_chat_completion(chat, result))

    except OpenAIProtocolError as exc:
        return _error_response(exc)
    except ChatServiceError as exc:
        return web.json_response(
            openai_protocol.error_body(exc.message, code=exc.code),
            status=exc.status)
    except ComputeError as exc:
        return web.json_response(
            openai_protocol.error_body(str(exc), code="compute_error"),
            status=500)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        _log("unhandled chat completion failure:")
        _log(traceback.format_exc())
        return web.json_response(
            openai_protocol.error_body(f"internal error: {exc}",
                                       code="internal_error"),
            status=500)


async def _read_json(request: web.Request):
    try:
        raw = await request.text()
    except (UnicodeDecodeError, web.HTTPRequestEntityTooLarge) as exc:
        raise OpenAIProtocolError(f"invalid request body: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenAIProtocolError(
            f"request body is not valid JSON: {exc}") from exc


async def _finish_stream(response: web.StreamResponse, body: dict) -> None:
    """SSE error を送って stream を閉じる。transport が既に壊れている場合は黙って諦める。"""
    try:
        await response.write(openai_protocol.sse_data(body))
        await response.write(openai_protocol.DONE)
        await response.write_eof()
    except (ConnectionResetError, BrokenPipeError):
        pass


async def handle_streaming(request: web.Request, service: ChatService,
                           chat, prepared) -> web.StreamResponse:
    completion_id = openai_protocol.new_completion_id()
    created = int(time.time())

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

    await response.prepare(request)

    events = service.stream(chat, prepared)
    finish_reason = None
    try:
        try:
            async for event in events:
                chunk = openai_protocol.encode_stream_event(
                    chat, event, completion_id=completion_id, created=created)

                if isinstance(event, StreamDone):
                    finish_reason = event.finish_reason

                if chunk is not None:
                    await response.write(openai_protocol.sse_data(chunk))

            await response.write(openai_protocol.DONE)
            await response.write_eof()
        except (ConnectionResetError, BrokenPipeError):
            pass
        except OpenAIProtocolError as exc:
            await _finish_stream(
                response, openai_protocol.error_body(
                    exc.message, param=exc.param, code=exc.code))
        except ChatServiceError as exc:
            await _finish_stream(
                response, openai_protocol.error_body(
                    exc.message, code=exc.code))
        except ComputeError as exc:
            await _finish_stream(
                response, openai_protocol.error_body(
                    str(exc), code="compute_error"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - headers are already sent
            _log("unhandled streaming failure:")
            _log(traceback.format_exc())
            await _finish_stream(
                response, openai_protocol.error_body(
                    f"internal error: {exc}", code="internal_error"))
    finally:
        await events.aclose()

    _log_request(chat, finish_reason)
    return response


async def handle_not_found(request: web.Request) -> web.Response:
    if request.path == "/v1/responses":
        return web.json_response(
            openai_protocol.error_body(
                "/v1/responses is not supported",
                code="unsupported_endpoint"),
            status=404)
    return web.json_response(
        openai_protocol.error_body(
            f"{request.path} not found",
            code="not_found"),
        status=404)


def create_app(service: ChatService, compute: AsyncComputeClient) -> web.Application:
    app = web.Application(
        client_max_size=4 * 1024 * 1024
    )

    app["chat_service"] = service
    app["compute"] = compute

    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/responses", handle_not_found)
    app.router.add_post("/v1/responses", handle_not_found)
    app.router.add_route("*", "/{tail:.*}", handle_not_found)

    app.cleanup_ctx.append(compute_cleanup)
    return app


async def compute_cleanup(app: web.Application):
    compute: AsyncComputeClient = app["compute"]
    try:
        yield
    finally:
        try:
            await compute.shutdown()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            _log(f"compute shutdown failed: {exc}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PhaseShift Server")
    parser.add_argument("--model-dir", required=True, help="model directory")
    parser.add_argument("--model", default="phaseshift",
                        help="model name reported by the API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--compute", default=None, help="phaseshift-compute path")

    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-concurrent-requests", type=int, default=4)

    parser.add_argument("--arena-gib", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", default="bf16",
                        choices=("bf16", "fp8_e4m3", "psq4", "psq8"))
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--kv-cache-capacity-tokens", type=int, default=0)

    parser.add_argument("--default-max-output-tokens", type=int, default=4096)

    parser.add_argument("--dflash2-model-dir", default=None,
                        help="DFlash2 draft model directory (enables speculative decoding)")
    parser.add_argument("--dflash2-drafts", type=int, default=7)

    parser.add_argument("--verify-weights", action="store_true")
    opts = parser.parse_args(argv)

    if opts.max_seq_len < 1:
        parser.error("--max-seq-len must be >= 1")
    if opts.max_concurrent_requests < 1:
        parser.error("--max-concurrent-requests must be >= 1")
    if opts.default_max_output_tokens < 1:
        parser.error("--default-max-output-tokens must be >= 1")
    if opts.page_tokens < 1:
        parser.error("--page-tokens must be >= 1")
    if opts.dflash2_drafts < 1:
        parser.error("--dflash2-drafts must be >= 1")
    return opts


def _validate_model_dirs(opts) -> None:
    model_dir = Path(opts.model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")
    opts.model_dir = str(model_dir)

    if opts.dflash2_model_dir:
        dflash2_dir = Path(opts.dflash2_model_dir).resolve()
        if not dflash2_dir.is_dir():
            raise SystemExit(f"dflash2 model directory not found: {dflash2_dir}")
        opts.dflash2_model_dir = str(dflash2_dir)
        if opts.max_concurrent_requests > 1 and opts.kv_cache_capacity_tokens == 0:
            opts.kv_cache_capacity_tokens = (
                opts.max_concurrent_requests + 1) * (opts.max_seq_len + 1)
            _log(f"--dflash2-model-dir raises --kv-cache-capacity-tokens to "
                 f"{opts.kv_cache_capacity_tokens} for "
                 f"{opts.max_concurrent_requests} concurrent requests")


async def _serve(opts) -> int:
    processor = await asyncio.to_thread(codec.load_processor, opts.model_dir)

    compute = AsyncComputeClient(
        build_compute_argv(opts),
        max_inflight=opts.max_concurrent_requests,
    )
    await compute.start()

    service = ChatService(
        processor=processor,
        compute=compute,
        model_name=opts.model,
        max_seq_len=opts.max_seq_len,
        default_max_output_tokens=opts.default_max_output_tokens,
    )

    app = create_app(service, compute)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, opts.host, opts.port)
    await site.start()

    address = f"http://{opts.host}:{opts.port}"
    _log("")
    _log("PhaseShift Server")
    _log("")
    _log(f"Model:        {opts.model} ({Path(opts.model_dir).name})")
    _log(f"Endpoint:     {address}/v1")
    _log(f"Context:      {opts.max_seq_len}")
    _log(f"Concurrency:  {opts.max_concurrent_requests}")
    _log(f"KV dtype:     {opts.kv_cache_dtype}")
    _log(f"KV capacity:  {'auto' if opts.kv_cache_capacity_tokens == 0 else opts.kv_cache_capacity_tokens}")
    _log(f"Prefix cache: protocol only (capabilities.prefix_cache="
         f"{str(compute.capabilities.prefix_cache).lower()})")
    if opts.dflash2_model_dir:
        _log(f"Speculative:  DFlash2 ({opts.dflash2_drafts} drafts)")
        _log(f"Draft model:  {Path(opts.dflash2_model_dir).name}")
    _log(f"Default output: {opts.default_max_output_tokens}")
    _log("")
    _log("Supported:   Chat Completions, SSE, tools, sampling, concurrency, cancel")
    _log("Unsupported: Responses, structured output, strict tools, reasoning")
    _log("")
    _log("Ready.")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass

    await stop.wait()
    _log("PhaseShift Server shutting down...")
    await runner.cleanup()
    return 0


def main(argv=None) -> int:
    opts = parse_args(argv)
    _validate_model_dirs(opts)
    opts.compute = _resolve_compute_binary(opts.compute)
    if not Path(opts.compute).is_file():
        _log(f"phaseshift-compute not found: {opts.compute}")
        return 2
    try:
        return asyncio.run(_serve(opts))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
