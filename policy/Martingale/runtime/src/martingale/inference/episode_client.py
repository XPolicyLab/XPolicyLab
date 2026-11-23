# Copyright © 2026 Martingale. All Rights Reserved.
"""Remote Codex turns with the robot's live Python worker on the evaluator."""
import json
import secrets
import threading

from websockets.sync.client import connect

from .episode_protocol import ENDPOINT, MAX_MESSAGE, parse_frame, tool_response


class RemoteEpisode:
    def __init__(self, token, model, effort, url=ENDPOINT):
        self.token = token
        self.url = url
        self.model = model
        self.effort = effort
        self.episode_key = secrets.token_hex(32)
        self.thread_id = None

    def connect(self):
        return connect(self.url, additional_headers={"Authorization": "Bearer " + self.token},
                       max_size=MAX_MESSAGE, compression=None, open_timeout=15, close_timeout=2)

    def turn(self, prompt, toolkit, report, cancelled):
        with self.connect() as ws:
            finished = threading.Event()

            def watch_episode():
                while not finished.wait(0.2):
                    if cancelled():
                        ws.close()
                        break

            watcher = threading.Thread(target=watch_episode, daemon=True)
            watcher.start()
            try:
                ws.send(json.dumps({"type": "turn", "episode": self.episode_key,
                                    "thread": self.thread_id, "model": self.model,
                                    "effort": self.effort, "prompt": prompt}))
                thread_seen = False
                while not cancelled():
                    message = parse_frame(ws.recv())
                    kind = message.get("type")
                    if kind == "error":
                        raise RuntimeError(message["message"])
                    if kind == "thread":
                        ident = message.get("thread")
                        if thread_seen or not isinstance(ident, str) or not ident or (
                                self.thread_id is not None and ident != self.thread_id):
                            raise RuntimeError("thread_identity_mismatch")
                        self.thread_id = ident
                        report({"type": "thread.started", "thread_id": ident})
                        thread_seen = True
                        ws.send(json.dumps({"type": "ack", "thread": ident}))
                    elif not thread_seen:
                        raise RuntimeError("thread_identity_mismatch: event before thread")
                    elif kind == "tool" and message.get("name") == "python":
                        result = toolkit.execute_tool("python", message["arguments"])
                        if not cancelled():
                            ws.send(json.dumps({"type": "result", "call": message["call"],
                                                **tool_response(result)}))
                    elif kind == "message":
                        report({"type": "item.completed", "item":
                                {"type": "agent_message", "text": message["text"]}})
                    elif kind == "complete":
                        report({"type": "turn.completed"})
                        return
                    else:
                        raise RuntimeError("Unexpected episode message")
            finally:
                finished.set()
                watcher.join(timeout=3)

    def release(self):
        if self.thread_id is None:
            return
        with self.connect() as ws:
            ws.send(json.dumps({"type": "release", "episode": self.episode_key,
                                "thread": self.thread_id}))
            reply = parse_frame(ws.recv(timeout=5))
            if reply.get("type") != "released":
                raise RuntimeError("Remote episode cleanup was deferred")
