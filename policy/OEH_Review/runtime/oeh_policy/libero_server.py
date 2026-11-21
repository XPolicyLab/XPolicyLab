"""One-episode OEH WebSocket endpoint for pinned LIBERO-family OpenWAM clients.

    python -m oeh_policy.libero_server --config <frozen oeh.policy.libero.v1> --host H --port P --attempt-id A

The official OpenWAM server is built by the official `openwam.deploy.server.main`
from the same flags the direct condition passes (`oeh_eval.libero_openwam`), and
reports the same effective runtime config; only its WebSocket loop is replaced by
this service, which routes every prediction through the OEH runtime. Every reply
carries the runtime events emitted since the previous one (`oeh_events`), so the
simulator-side client writes the campaign sidecar live. One episode per process:
the server exits after the episode connection closes. A review condition whose
teacher runs on the Codex app-server starts it with the process, while OpenWAM
loads, as RoboDojo's server does (oeh_policy.model): the first reset then takes a
started teacher instead of paying the app-server's start inside its call.

With `--openwam-address ws://H:P` the official server is a session of the GPU's
shared OpenWAM service (oeh_eval.shared_openwam), leased by the launcher, which
also reports that service's runtime config: this process loads no weights and
reaches the session through `ProtocolPolicyClient` (reset, predict, ping).
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
from pathlib import Path
import re
import signal

from .contracts import digest
from .libero_boundary import BENCHMARKS, load_official_conversion, validate_native_action
from .libero_config import check as check_frozen, resolve_config_path
from .libero_runtime import LiberoBackend, LiberoContract, serve_step

MAX_MESSAGE_BYTES = 32 * 1024 * 1024  # the official server's cap
EVENTS = "oeh_events"


class LiberoService:
    """Single benchmark attempt; native judgment stays in the official client."""

    def __init__(self, frozen: dict, official_server, attempt_id: str | None = None, *, checked: bool = False,
                 teacher_start=None):
        if not checked:
            check_frozen(frozen)
        self.cfg = frozen
        self.teacher_start = teacher_start  # review: the prestarted teacher (teacher_prestart) the episode takes
        self.contract = LiberoContract(frozen["benchmark"])
        self.contract.validate_config(frozen)
        self.official = official_server
        folder = BENCHMARKS[frozen["benchmark"]]
        self.convert = load_official_conversion(Path(frozen["source_root"]), frozen["benchmark"],
            frozen["source_hashes"][f"benchmarks/{folder}/openwam2libero_interface.py"])
        self.backend = LiberoBackend(self.official, self.convert)
        self.attempt_id = attempt_id
        self.runtime = None
        self.episode_used = False
        self.closed = False
        self.finished = None

    def ping(self):
        contract = self.official._ckpt_contract()
        if contract.get("representation") != "eef":
            raise ValueError("official checkpoint representation mismatch")
        return {"type": "pong", **contract}

    async def reset(self):
        from .runtime import Runtime

        if self.episode_used:
            raise RuntimeError("one LIBERO episode per policy server")
        self.episode_used = True
        start, self.teacher_start = self.teacher_start, None
        try:
            self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id, contract=self.contract,
                                   teacher_start=start)
        except BaseException:
            discard_prestart(start)
            raise
        try:
            await self.runtime.start()  # includes the actual official server reset
        except BaseException:
            await self.runtime.close("preflight_failed")
            raise
        return {"type": "reset_ack"}

    async def predict(self, payload):
        if self.runtime is None or self.runtime.closed:
            raise RuntimeError("policy server must receive native reset first")
        return await serve_step(self.runtime, self.backend, payload)

    async def receipt(self, payload):
        if self.runtime is None or self.runtime.closed:
            raise RuntimeError("policy server has no active episode")
        if set(payload) != {"type", "ticket", "action", "terminal", "state", "state_sha256"}:
            raise ValueError("native receipt shape mismatch")
        action = validate_native_action(payload["action"])
        state = self.contract.project_state(payload["state"])
        if digest(state) != payload["state_sha256"]:
            raise ValueError("post-step state receipt hash mismatch")
        await self.runtime.after_action(payload["ticket"], action, state, payload["terminal"])
        return {"type": "oeh_receipt_ack"}

    async def end_episode(self, reason):
        if self.runtime is not None and not self.runtime.closed:
            await self.runtime.close(reason)
        return {"type": "oeh_close_ack"}

    async def drain(self):
        return await self.runtime.drain_events() if self.runtime is not None else []

    async def close(self, reason="client_closed"):
        await self.end_episode(reason)
        if not self.closed:
            self.closed = True
            start, self.teacher_start = self.teacher_start, None
            discard_prestart(start)  # a server that served no episode
            if hasattr(self.official, "shutdown"):
                self.official.shutdown()

    async def handle(self, websocket):
        close_reason = "client_closed"
        episode = False
        try:
            async for raw in websocket:
                kind = None
                try:
                    message = json.loads(raw)
                    kind = message.get("type", "obs")
                    if kind == "ping":
                        reply = self.ping()
                    elif kind == "reset":
                        episode = True
                        reply = await self.reset()
                    elif kind == "obs":
                        reply = await self.predict(message)
                    elif kind == "oeh_receipt":
                        reply = await self.receipt(message)
                    elif kind == "oeh_close":
                        reason = message.get("reason")
                        reply = await self.end_episode(reason if reason in {"client_closed", "native_terminal"}
                                                       else "client_closed")
                    else:
                        raise ValueError("unknown policy wire message")
                    reply[EVENTS] = await self.drain()
                    await websocket.send(json.dumps(reply))
                except Exception as exc:
                    close_reason = type(exc).__name__
                    if self.runtime is not None and not self.runtime.closed:
                        self.runtime.emit("endpoint_failure", {"stage": kind,
                                                              "reason": close_reason})
                    await websocket.send(json.dumps({"type": "error", "code": "oeh_policy_error",
                                                     "message": f"{type(exc).__name__}: {exc}",
                                                     EVENTS: await self.drain()}))
                    if kind != "ping":
                        break  # never retry ambiguous admission or native receipt
        finally:
            # A readiness ping on its own connection is not an episode.
            if episode:
                await self.close(close_reason)
                if self.finished is not None:
                    self.finished.set()

    async def serve(self, host, port):
        import websockets

        self.finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.finished.set)
        async with websockets.serve(self.handle, host, port, max_size=MAX_MESSAGE_BYTES, ping_interval=None):
            await self.finished.wait()
        await self.close("server_stopped")


def teacher_prestart(cfg: dict):
    """The Codex app-server teacher of a review condition, started with its preflight in a thread of its own
    (``codex_teacher.prestart``), or None. The prompt is the one the runtime's client would get (``runtime_prompt``)."""
    teacher = (cfg.get("review") or {}).get("teacher") or {}
    if cfg.get("mode") != "review" or teacher.get("api") != "codex_app_server":
        return None
    from .codex_teacher import prestart
    from .teacher import prompt_text
    return prestart(teacher, prompt_text(cfg["review"].get("schedule"), cfg["benchmark"]))


def discard_prestart(start) -> None:
    if start is not None:
        from .codex_teacher import discard
        discard(start)


def main(argv=None):
    from oeh_eval.libero_openwam import import_official_server, install_report, server_argv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--attempt-id")
    parser.add_argument("--openwam-address", help="a shared OpenWAM session (ws://host:port) instead of loading "
                                                   "the official server in this process")
    args = parser.parse_args(argv)
    if args.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", args.attempt_id)
                            or args.attempt_id in {".", ".."}):
        raise ValueError("invalid attempt identity")
    cfg = json.loads(resolve_config_path(args.config).read_text(encoding="utf-8"))
    check_frozen(cfg)
    if args.openwam_address:
        serve_shared(cfg, args)
        return
    module = import_official_server(Path(cfg["source_root"]), cfg["source_hashes"]["openwam/deploy/server.py"])
    official_argv = server_argv(deploy_config=cfg["official_deploy_config"], ckpt_dir=cfg["checkpoint_dir"],
                                ckpt_name=cfg["checkpoint_name"], device=cfg["compute"]["device"],
                                flags=cfg["official_server"], host=args.host, port=args.port)
    pending = {"start": teacher_prestart(cfg)}  # overlaps the OpenWAM load below

    def serve_with_oeh(server):
        def run(host="0.0.0.0", port=8848):
            if server._ckpt_contract().get("representation") != "eef":
                raise ValueError("checkpoint does not advertise LIBERO EEF10 representation")
            if callable(getattr(server, "_init_policy", None)):
                server._init_policy()  # as the official run() does before it listens
            start, pending["start"] = pending["start"], None
            if start is not None:
                # The start finishes before the server listens (and so before the harness sees it ready); every
                # step of it is bounded, and a failure stays in the future for the runtime's start to report.
                concurrent.futures.wait([start])
            service = LiberoService(cfg, official_server=server, attempt_id=args.attempt_id, checked=True,
                                    teacher_start=start)
            asyncio.run(service.serve(host, port))
        server.run = run

    install_report(module, official_argv, on_built=serve_with_oeh)
    try:
        module.main(official_argv)
    finally:
        discard_prestart(pending["start"])  # the server failed before it could serve


def serve_shared(cfg: dict, args) -> None:
    """The OEH service over a shared OpenWAM session: the same LiberoService, the official server's methods over
    the official protocol. The teacher starts while the session answers its first ping."""
    from .openwam_protocol import ProtocolPolicyClient

    start = teacher_prestart(cfg)
    try:
        official = ProtocolPolicyClient(args.openwam_address)
        if official._ckpt_contract().get("representation") != "eef":
            raise ValueError("checkpoint does not advertise LIBERO EEF10 representation")
        if start is not None:
            concurrent.futures.wait([start])
        service = LiberoService(cfg, official_server=official, attempt_id=args.attempt_id, checked=True,
                                teacher_start=start)
        start = None
    finally:
        discard_prestart(start)  # the session failed before the service could take the teacher
    asyncio.run(service.serve(args.host, args.port))


if __name__ == "__main__":
    main()
