# Copyright © 2026 Martingale. All Rights Reserved.
"""Evaluation-only WebSocket bridge. The Codex control socket stays on loopback."""
import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
import hmac
from http import HTTPStatus
import json
import logging
from pathlib import Path
import re
import tempfile
import time

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .codex_cli import DISABLED_FEATURES
from .config import InferenceConfig
from .episode_protocol import MAX_MESSAGE, parse_frame, python_tool

LOG = logging.getLogger("martingale.episode")

class CodexError(RuntimeError):
    """A JSON-RPC request was explicitly rejected by the Codex server."""


async def send(ws, message):
    await ws.send(json.dumps(message))


class CodexRPC:
    def __init__(self, ws):
        self.ws = ws
        self.sequence = 0
        self.events = deque()

    async def request(self, method, params):
        self.sequence += 1
        await send(self.ws, {"id": self.sequence, "method": method, "params": params})
        async with asyncio.timeout(60):
            while True:
                msg = parse_frame(await self.ws.recv())
                if "method" in msg:
                    self.events.append(msg)
                elif msg.get("id") == self.sequence:
                    if "error" in msg:
                        raise CodexError(str(msg["error"].get("message", "Codex request failed")))
                    return msg["result"]

    async def event(self):
        return self.events.popleft() if self.events else parse_frame(await self.ws.recv())


@dataclass
class Episode:
    workspace: tempfile.TemporaryDirectory
    thread: str | None = None
    busy: bool = False
    uncertain: bool = False
    used: float = 0.0


class EpisodeService:
    def __init__(self, evaluator_token, upstream_token, config, workspace_root,
                 upstream_url="ws://127.0.0.1:4500"):
        if not evaluator_token or not upstream_token or evaluator_token == upstream_token:
            raise ValueError("Two distinct endpoint credentials are required")
        self.auth = ("Bearer " + evaluator_token).encode()
        self.upstream_token = upstream_token
        self.upstream_url = upstream_url
        self.config = config
        self.workspace_root = workspace_root
        self.episodes = {}

    def authorize(self, connection, request):
        if request.path != "/episode":
            return connection.respond(HTTPStatus.NOT_FOUND, "Unknown endpoint\n")
        headers = request.headers.get_all("Authorization")
        if len(headers) != 1 or not hmac.compare_digest(headers[0].encode(), self.auth):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Invalid evaluator credential\n")

    def forget(self, key):
        self.episodes.pop(key).workspace.cleanup()

    async def reap(self):
        while True:
            await asyncio.sleep(60)
            for key, state in list(self.episodes.items()):
                if not state.busy and time.monotonic() - state.used > 7200:
                    self.forget(key)

    async def handle(self, runner):
        try:
            async with asyncio.timeout(30):
                message = parse_frame(await runner.recv())
            key = message.get("episode")
            if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
                raise ValueError("Invalid episode identifier")
            expected = message.get("thread")
            state = self.episodes.get(key)
            if state and state.busy:
                raise RuntimeError("Connection already serving this episode; retry later")
            if expected is not None and (not state or expected != state.thread):
                raise RuntimeError("thread_identity_mismatch: original session unavailable")
            if message.get("type") == "release":
                if state:
                    self.forget(key)
                await send(runner, {"type": "released"})
                return
            if (message.get("type") != "turn" or not isinstance(message.get("prompt"), str)
                    or not message["prompt"] or message.get("model") != self.config.model
                    or message.get("effort") != self.config.effort):
                raise ValueError("Invalid turn or inference configuration")
            if state is None:
                state = Episode(tempfile.TemporaryDirectory(dir=self.workspace_root, prefix="episode-"))
                self.episodes[key] = state
            if state.uncertain:
                raise RuntimeError("thread_identity_mismatch: previous turn termination unconfirmed")
            state.busy = True
            try:
                async with connect(self.upstream_url, proxy=None, compression=None,
                                   additional_headers={"Authorization": "Bearer " + self.upstream_token},
                                   max_size=MAX_MESSAGE, open_timeout=15, close_timeout=2) as ws:
                    await self.run_turn(runner, CodexRPC(ws), state, message["prompt"])
            finally:
                state.busy = False
                state.used = time.monotonic()
        except ConnectionClosed:
            return
        except Exception as exc:
            LOG.warning("Episode interrupted (%s)", type(exc).__name__)
            try:
                await send(runner, {"type": "error", "message": str(exc)})
            except ConnectionClosed:
                pass

    async def run_turn(self, runner, rpc, state, prompt):
        await rpc.request("initialize", {"clientInfo": {"name": "martingale", "version": "1.0"},
                                         "capabilities": {"experimentalApi": True}})
        await send(rpc.ws, {"method": "initialized"})
        features = {feature: False for feature in DISABLED_FEATURES}
        features.update(code_mode_host=True, skip_host_skill_discovery=True)
        params = {"model": self.config.model, "serviceTier": self.config.serving,
                  "cwd": state.workspace.name, "approvalPolicy": "never", "sandbox": "read-only",
                  "config": {"web_search": "disabled", "features": features,
                             "project_doc_max_bytes": 0, "model_reasoning_effort": self.config.effort}}
        if state.thread:
            reply = await rpc.request("thread/resume", {**params, "threadId": state.thread,
                                                       "excludeTurns": True})
        else:
            spec = python_tool()
            tool = {"type": "function", "name": spec["name"], "description": spec["description"],
                    "inputSchema": spec["input_schema"], "deferLoading": False}
            reply = await rpc.request("thread/start", {**params, "dynamicTools": [tool],
                                                       "allowProviderModelFallback": False})
        ident = reply["thread"]["id"]
        if not isinstance(ident, str) or not ident or (state.thread and state.thread != ident):
            raise RuntimeError("thread_identity_mismatch")
        state.thread = ident
        await send(runner, {"type": "thread", "thread": ident})
        async with asyncio.timeout(30):
            if parse_frame(await runner.recv()) != {"type": "ack", "thread": ident}:
                raise ValueError("Expected runner thread acknowledgement")

        state.uncertain = True
        try:
            reply = await rpc.request("turn/start", {"threadId": ident, "effort": self.config.effort,
                                                    "input": [{"type": "text", "text": prompt}]})
        except CodexError:
            state.uncertain = False
            raise
        turn = reply["turn"]["id"]
        relay = asyncio.create_task(self.relay(runner, rpc, state, turn))
        closed = asyncio.create_task(runner.wait_closed())
        try:
            done, _ = await asyncio.wait({relay, closed}, return_when=asyncio.FIRST_COMPLETED)
            if relay in done:
                await relay
        finally:
            for task in (relay, closed):
                task.cancel()
            await asyncio.gather(relay, closed, return_exceptions=True)
            if state.uncertain:
                try:
                    async with asyncio.timeout(12):
                        await rpc.request("turn/interrupt", {"threadId": ident, "turnId": turn})
                        while True:
                            event = await rpc.event()
                            data = event.get("params", {})
                            if (event.get("method") == "turn/completed" and data.get("threadId") == ident
                                    and data.get("turn", {}).get("id") == turn):
                                state.uncertain = False
                                break
                except (TimeoutError, ConnectionClosed, RuntimeError):
                    LOG.warning("Could not confirm interruption of a Codex turn")

    async def relay(self, runner, rpc, state, turn):
        while True:
            event = await rpc.event()
            method = event.get("method")
            data = event.get("params", {})
            owned = data.get("threadId") == state.thread
            if "id" in event and method:
                if (method == "item/tool/call" and owned and data.get("turnId") == turn
                        and data.get("tool") == "python" and not data.get("namespace")):
                    await self.call_tool(runner, rpc, event)
                else:
                    await send(rpc.ws, {"id": event["id"], "error": {"code": -32601,
                                                                    "message": "Tool unavailable"}})
            elif owned and method == "item/completed" and data.get("turnId") == turn:
                item = data.get("item", {})
                if item.get("type") == "agentMessage":
                    await send(runner, {"type": "message", "text": item.get("text", "")})
            elif owned and method == "turn/completed" and data.get("turn", {}).get("id") == turn:
                outcome = data["turn"]
                state.uncertain = False
                if outcome["status"] != "completed":
                    error = outcome.get("error") or {}
                    raise RuntimeError(str(error.get("message", outcome["status"])))
                await send(runner, {"type": "complete"})
                return

    async def call_tool(self, runner, rpc, event):
        data = event["params"]
        await send(runner, {"type": "tool", "call": data["callId"],
                            "name": "python", "arguments": data["arguments"]})
        async with asyncio.timeout(3300):
            reply = parse_frame(await runner.recv())
        if (reply.get("type") != "result" or reply.get("call") != data["callId"]
                or type(reply.get("success")) is not bool or not isinstance(reply.get("contentItems"), list)):
            raise ValueError("Mismatched tool response")
        for item in reply["contentItems"]:
            if item.get("type") == "inputText" and isinstance(item.get("text"), str):
                continue
            if item.get("type") == "inputImage" and str(item.get("imageUrl", "")).startswith("data:image/png;base64,"):
                continue
            raise ValueError("Unsupported tool response content")
        await send(rpc.ws, {"id": event["id"], "result": {
            "success": reply["success"], "contentItems": reply["contentItems"]}})


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/martingale"))
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    state = args.state_dir
    root = args.workspace_root or state / "episodes"
    root.mkdir(mode=0o700, exist_ok=True)
    service = EpisodeService((state / "robodojo.token").read_text().strip(),
                             (state / "endpoint.token").read_text().strip(),
                             InferenceConfig.load(args.config), root)
    async with serve(service.handle, "127.0.0.1", 4510, process_request=service.authorize,
                     max_size=MAX_MESSAGE, compression=None, close_timeout=2, server_header=None):
        await service.reap()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
