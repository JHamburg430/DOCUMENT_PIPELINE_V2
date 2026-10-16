"""Bounded NDJSON transport with disconnect cancellation and no retry."""

from __future__ import annotations

import json
import select
import socket
import threading
import time
from contextlib import contextmanager
from urllib.request import Request, urlopen


def _interrupt(response) -> None:
    # Interrupt a blocked readline before closing its buffered file.
    raw_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if raw_socket is not None:
        try:
            raw_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class NDJSONResponse:
    def __init__(self, url: str, body: dict, *, timeout: float):
        self.deadline = time.monotonic() + timeout
        request = Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
        self.response = urlopen(request, timeout=timeout)
        if self.response.headers.get_content_type() != "application/x-ndjson":
            self.response.close()
            raise RuntimeError("upstream does not support NDJSON streaming")
        self.cancelled = threading.Event()
        self.timer = threading.Timer(max(0.001, self.deadline - time.monotonic()), self.cancel)
        self.timer.daemon = True
        self.timer.start()
        self.done = False

    def cancel(self):
        self.cancelled.set()
        _interrupt(self.response)

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError("upstream stream cancelled or timed out")
        line = self.response.readline(2_000_001)
        if not line:
            if self.done:
                raise StopIteration
            raise RuntimeError("upstream stream ended without completion")
        if len(line) > 2_000_000:
            raise ValueError("upstream stream event too large")
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("error"):
            raise RuntimeError("upstream stream failed")
        data = event.get("response", event)
        if not isinstance(data, dict):
            raise ValueError("invalid upstream stream event")
        if data.get("done") is True:
            self.done = True
        return event

    def close(self):
        self.timer.cancel()
        self.cancel()
        self.response.close()


@contextmanager
def watch_disconnect(connection, cancel):
    stopped = threading.Event()

    def watch():
        while not stopped.wait(0.1):
            try:
                readable, _, _ = select.select([connection], [], [], 0)
                if readable and connection.recv(1, socket.MSG_PEEK) == b"":
                    cancel()
                    return
            except OSError:
                cancel()
                return

    worker = threading.Thread(target=watch, daemon=True, name="inference-disconnect")
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=0.3)


def relay_ndjson(handler, stream, *, wrap=lambda event: event):
    """Validate the first event before headers; send later errors only once."""
    first = next(stream)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Connection", "close")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.close_connection = True
    try:
        handler.wfile.write(json.dumps(wrap(first)).encode() + b"\n")
        handler.wfile.flush()
        for event in stream:
            handler.wfile.write(json.dumps(wrap(event)).encode() + b"\n")
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        stream.cancel()
    except Exception:
        # No raw upstream error text, response, prompt, or credential is exposed.
        try:
            handler.wfile.write(b'{"error":{"code":"upstream_stream_failed"},"done":true}\n')
            handler.wfile.flush()
        except OSError:
            pass
