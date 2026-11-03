"""The offline programs of this policy, served to PyRUA-Lean's openai runtime in place of a language model.

Each RoboDojo task has one program in ``programs/<task>.py`` (written ahead of time, no model in the loop).
A small chat-completions endpoint on 127.0.0.1 answers the first request of every episode with the task's
whole program as one ``python`` tool call and every later request with ``finish``.  The runtime, the
``robo`` tools and the episode driver are the same as in the PyRUA_Lean policy; only the source of the
code differs.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROGRAMS = Path(__file__).resolve().parent / "programs"
_LOCK = threading.Lock()
_STATE = {"task": None, "conversation": -1, "replies": 0}
_SERVER: ThreadingHTTPServer | None = None


def program_for(task: str) -> str:
    path = PROGRAMS / f"{task}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no offline program for task {task!r} (expected {path})")
    return path.read_text()


def select(task: str) -> None:
    """The task of the episode about to start (one episode at a time per policy server)."""

    program_for(task)  # fail early when the program is missing
    with _LOCK:
        _STATE["task"] = task


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        messages = body.get("messages") or []
        with _LOCK:
            if not any(m.get("role") == "assistant" for m in messages):
                _STATE["conversation"] += 1
                _STATE["replies"] = 0
            conversation, reply, task = _STATE["conversation"], _STATE["replies"], _STATE["task"]
            _STATE["replies"] += 1
        if reply == 0 and task:
            name, arguments = "python", {"code": program_for(task)}
        else:
            name, arguments = "finish", {"status": "success", "summary": "offline program finished"}
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{conversation}_{reply}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}]}
        data = json.dumps({"id": f"offline-{conversation}-{reply}", "object": "chat.completion",
                           "created": int(time.time()), "model": body.get("model"),
                           "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                           "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start() -> str:
    """Start the endpoint once per process; returns its base URL."""

    global _SERVER
    with _LOCK:
        if _SERVER is None:
            _SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            threading.Thread(target=_SERVER.serve_forever, name="offline-programs", daemon=True).start()
        return f"http://127.0.0.1:{_SERVER.server_address[1]}/v1"
