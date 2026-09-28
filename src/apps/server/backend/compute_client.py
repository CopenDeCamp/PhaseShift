"""Persistent JSONL client for ``phaseshift-compute --serve-stdio``.

One compute process is started per backend instance and held for its lifetime.
The protocol is one JSON object per line. A single stdout reader thread routes
events to per-request mailboxes by ``request_id``; events without a request ID
(pong / shutdown) go to a control queue. This allows several generation
lifecycles to run concurrently without a global lock while keeping exactly one
stdout consumer and one stdin writer per line.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from typing import Callable, Optional


class ComputeError(RuntimeError):
    """Raised when the compute process fails or reports an error event."""


class ComputeInvalidArgument(ComputeError):
    """Raised when compute rejects the request before generation."""


class ComputeProtocolError(ComputeError):
    """Raised when the compute process violates the JSONL protocol."""


class ComputeCancelled(ComputeError):
    """Raised when a generation ended because this client cancelled it."""


class ComputeClient:
    def __init__(self, argv: list[str], log=None, max_inflight: int = 1):
        self._argv = list(argv)
        self._log = log if log is not None else sys.stderr
        self._proc: Optional[subprocess.Popen] = None
        self._mailboxes: dict[int, "queue.Queue"] = {}
        self._control_queue: "queue.Queue" = queue.Queue()
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._next_request_id = 1
        self._max_inflight = max(1, int(max_inflight))
        self._semaphore = threading.BoundedSemaphore(self._max_inflight)
        self._closing = False
        self._shutdown_started = False
        self._trace_path = os.environ.get("PHASESHIFT_COMPUTE_LOG")

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._proc is not None:
            raise ComputeError("compute process already started")
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.ping()

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                self._fail_all(ComputeProtocolError(
                    f"compute wrote non-JSON to stdout: {stripped!r} ({exc})"))
                continue
            request_id = event.get("request_id")
            if request_id is None:
                self._control_queue.put(event)
                continue
            with self._state_lock:
                mailbox = self._mailboxes.get(int(request_id))
            if mailbox is None:
                self._fail_all(ComputeProtocolError(
                    f"event for unknown request {request_id}: {event!r}"))
                continue
            mailbox.put(event)
        self._fail_all(ComputeError(
            f"compute process exited with code {self.returncode()}"))

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._log.write("[compute] " + line)
            self._log.flush()
            if self._trace_path:
                try:
                    with open(self._trace_path, "a", encoding="utf-8") as handle:
                        handle.write(line)
                except OSError:
                    pass

    def _fail_all(self, exc: Exception) -> None:
        with self._state_lock:
            mailboxes = list(self._mailboxes.values())
        for mailbox in mailboxes:
            mailbox.put(exc)
        self._control_queue.put(exc)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> Optional[int]:
        return None if self._proc is None else self._proc.returncode

    # ------------------------------------------------------------------ protocol

    def _send(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ComputeError("compute process is not running")
        line = json.dumps(payload, separators=(",", ":"))
        with self._write_lock:
            if self._proc is None or self._proc.stdin is None:
                raise ComputeError("compute process is not running")
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                raise ComputeError("compute process closed its stdin") from exc

    @staticmethod
    def _poll_box(mailbox: "queue.Queue", timeout: float):
        try:
            event = mailbox.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(event, Exception):
            raise event
        return event

    @staticmethod
    def _wait_box(mailbox: "queue.Queue"):
        event = mailbox.get()
        if isinstance(event, Exception):
            raise event
        return event

    def ping(self) -> None:
        with self._control_lock:
            self._send({"op": "ping"})
            while True:
                event = self._control_queue.get(timeout=30.0)
                if isinstance(event, Exception):
                    raise event
                if event.get("event") == "pong":
                    return

    # ------------------------------------------------------------------ requests

    def _begin_request(self) -> tuple[int, "queue.Queue"]:
        self._semaphore.acquire()
        with self._state_lock:
            if self._closing:
                self._semaphore.release()
                raise ComputeError("compute client is shutting down")
            request_id = self._next_request_id
            self._next_request_id += 1
            mailbox: "queue.Queue" = queue.Queue()
            self._mailboxes[request_id] = mailbox
        return request_id, mailbox

    def _end_request(self, request_id: int) -> None:
        with self._state_lock:
            self._mailboxes.pop(request_id, None)
        self._semaphore.release()

    def _send_generate(
        self,
        request_id: int,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        grammar: str | None = None,
        structural_tag: str | None = None,
        prefix_cache_checkpoint_position: int = 0,
    ) -> None:
        payload = {
            "op": "generate",
            "request_id": request_id,
            "input_ids": list(input_ids),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "seed": int(seed),
        }
        if grammar:
            payload["grammar"] = grammar
        if structural_tag:
            payload["structural_tag"] = structural_tag
        if prefix_cache_checkpoint_position:
            payload["prefix_cache_checkpoint_position"] = int(
                prefix_cache_checkpoint_position)
        self._send(payload)

    @staticmethod
    def _error_from_event(event: dict) -> ComputeError:
        message = event.get("message", "compute error")
        if event.get("code") == "invalid_argument":
            return ComputeInvalidArgument(message)
        return ComputeError(message)

    def _cancel(self, request_id: int) -> None:
        try:
            self._send({"op": "cancel", "request_id": request_id})
        except ComputeError:
            pass

    def _drain(self, mailbox: "queue.Queue") -> None:
        while True:
            try:
                event = self._wait_box(mailbox)
            except ComputeError:
                return
            kind = event.get("event")
            if kind in ("done", "error"):
                return

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        should_cancel: Optional[Callable[[], bool]] = None,
        grammar: str | None = None,
        structural_tag: str | None = None,
        prefix_cache_checkpoint_position: int = 0,
    ) -> dict:
        request_id, mailbox = self._begin_request()
        cancel_sent = False
        try:
            self._send_generate(request_id, input_ids, max_new_tokens,
                                temperature, top_p, top_k, seed, grammar,
                                structural_tag, prefix_cache_checkpoint_position)
            while True:
                if (should_cancel is not None and not cancel_sent
                        and should_cancel()):
                    self._cancel(request_id)
                    cancel_sent = True
                event = self._poll_box(mailbox, 0.05)
                if event is None:
                    continue
                kind = event.get("event")
                if kind == "error":
                    raise self._error_from_event(event)
                if kind == "done":
                    if cancel_sent or event.get("finish_reason") == "cancelled":
                        raise ComputeCancelled(
                            f"request {request_id} was cancelled")
                    return event
        finally:
            self._end_request(request_id)

    def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        grammar: str | None = None,
        structural_tag: str | None = None,
        prefix_cache_checkpoint_position: int = 0,
    ):
        """Yield ``{"token_id": N}`` events, then one ``{"done": event}``.

        Each stream owns its request mailbox. If the consumer abandons the
        stream, only this request is cancelled and its terminal event is drained
        before the in-flight slot is released. Other requests are unaffected.
        """
        request_id, mailbox = self._begin_request()
        sent = False
        completed = False
        try:
            self._send_generate(request_id, input_ids, max_new_tokens,
                                temperature, top_p, top_k, seed, grammar,
                                structural_tag, prefix_cache_checkpoint_position)
            sent = True
            while True:
                event = self._wait_box(mailbox)
                kind = event.get("event")
                if kind == "token":
                    yield {"token_id": int(event["token_id"])}
                elif kind == "error":
                    completed = True
                    raise self._error_from_event(event)
                elif kind == "done":
                    completed = True
                    yield {"done": event}
                    return
        finally:
            if sent and not completed:
                self._cancel(request_id)
                self._drain(mailbox)
            self._end_request(request_id)

    # ------------------------------------------------------------------ shutdown

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._state_lock:
            self._closing = True
            if self._shutdown_started:
                return
            self._shutdown_started = True
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send({"op": "shutdown"})
            while True:
                event = self._control_queue.get(timeout=timeout)
                if isinstance(event, Exception):
                    break
                if event.get("event") == "shutdown":
                    break
        except (ComputeError, queue.Empty):
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=timeout)

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self.shutdown()
        self._proc = None
