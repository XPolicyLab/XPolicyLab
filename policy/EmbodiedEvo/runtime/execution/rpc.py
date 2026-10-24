"""JSON framing for the host-to-workspace session socket."""

from __future__ import annotations

import json

MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete session response; execution may have happened. Do not retry automatically."
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def send(stream, value):
    data = json.dumps(value, ensure_ascii=True).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Session message exceeds 16 MiB")
    stream.write(data)
    stream.flush()
