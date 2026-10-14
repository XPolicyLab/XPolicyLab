"""Many independent task/layout episodes behind one WebSocket endpoint."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
import logging
import math
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from client_server.ws.model_server import PolicyServer
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType
from XPolicyLab.policy.embodiedevo.model import Model
from XPolicyLab.policy.embodiedevo.runtime.config import setting
from XPolicyLab.policy.embodiedevo.runtime.files import append_jsonl, write_json
from XPolicyLab.policy.embodiedevo.runtime.service_lifecycle import AgentCapacity, SessionModel, StartupScheduler
from XPolicyLab.policy.embodiedevo.runtime.workspace import POLICY_DIR, resolve_workspace
from XPolicyLab.utils.process_data import decode_obs_images

logger = logging.getLogger(__name__)
CONTROL_MESSAGES = {MessageType.HELLO, MessageType.HEARTBEAT, MessageType.CLOSE}


@dataclass(eq=False)
class Peer:
    generation: int
    cases: set = field(default_factory=set)
    closed_at: float | None = None


class CaseSlot(PolicyServer):
    """A client case has its own model; each env_idx gets a fresh episode worker."""

    def __init__(self, model, service, identity):
        super().__init__(model, service.config)
        self.service, self.identity = service, identity
        self.dispatch_lock = asyncio.Lock()
        self.task_name = model.cfg["task_name"]
        self.last_activity = time.monotonic()
        self.prepared_at = None
        self.trial_id = None
        self.layouts = {}
        self.retiring = False

    async def _invoke_method(self, method, *args):
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        runner = self.service.run_cleanup if getattr(method, "__name__", "") == "on_trial_end" else self.service.run_model
        result = await runner(method, *args)
        return await result if inspect.isawaitable(result) else result

    def _targets(self, batch, indices=None):
        indices = (list(self.model.contexts) if indices is None else indices) if batch else [0]
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int for i in indices) or len(indices) != len(set(indices))
                or not set(indices) <= self.model.contexts.keys()):
            raise WsError(ErrorCode.INVALID_FRAME, "Expected distinct, prepared environment indices")
        return [self.model.contexts[i] for i in indices]

    async def _handle_call(self, frame):
        name = frame.payload.get("func_name")
        if name in {"get_action", "get_action_batch"}:
            batch = name == "get_action_batch"
            contexts = self._targets(batch, frame.payload.get("obs"))
            if not self.service.scheduler.ready(contexts):
                await asyncio.sleep(self.service.poll_interval_s)
                return self._reply(frame, MessageType.CALL_RESULT, {
                    "ok": True, "result": [[] for _ in contexts] if batch else [],
                    "latency_ms": self.service.poll_interval_s * 1000,
                })
        result = await super()._handle_call(frame)
        if name in {"get_action", "get_action_batch"} and not any(result.payload.get("result", [])):
            # Waiting for a model API must not occupy a thread per client.
            await asyncio.sleep(self.service.poll_interval_s)
        return result

    async def _handle_infer(self, frame):
        obs = frame.payload.get("observation")
        if obs is None:
            raise WsError(ErrorCode.INVALID_FRAME, "infer payload missing observation")
        obs = await asyncio.to_thread(decode_obs_images, obs)
        await self._invoke_method(self.model.update_obs, obs)
        if self.service.scheduler.ready(self._targets(False)):
            result = await self._invoke_method(self.model.get_action)
        else:
            result = []
        if not result:
            await asyncio.sleep(self.service.poll_interval_s)
        return self._reply(frame, MessageType.INFER_RESULT, {"actions": result, "latency_ms": 0})

    async def _dispatch_frame(self, frame):
        if frame.message_type == MessageType.CALL and frame.payload.get("func_name") not in {
            "update_obs", "update_obs_batch", "get_action", "get_action_batch",
        }:
            raise WsError(ErrorCode.INVALID_FRAME, "Unsupported agent call")
        async with self.dispatch_lock:
            if self.retiring:
                raise WsError(ErrorCode.INVALID_FRAME, "Session is retiring; use a new evaluation_id")
            self.last_activity = time.monotonic()
            kind = frame.message_type
            if kind == MessageType.RESET:
                await self.service.scheduler.detach(self.model.contexts.values())
                await self.service.run_cleanup(self.model.reset)
                self.service.capacity.release(self.identity)
                self.layouts, self.prepared_at, self.trial_id = {}, None, frame.trial_id
                return self._reply(frame, MessageType.RESET_RESULT, {"ok": True})
            if frame.trial_id is not None and self.trial_id is not None and frame.trial_id != self.trial_id:
                raise WsError(ErrorCode.INVALID_FRAME, "trial_id conflicts with the current reset")
            if kind == MessageType.PREPARE_CASE:
                if self.model.contexts:
                    raise WsError(ErrorCode.INVALID_FRAME, "Reset before preparing another episode")
                environments = frame.payload.get("environments", [{"env_idx": 0, "layout_id": frame.payload.get("layout_id")}])
                if not isinstance(environments, list) or not environments or len(environments) > 16:
                    raise WsError(ErrorCode.INVALID_FRAME, "prepare_case requires 1..16 environments")
                layouts = {}
                for item in environments:
                    if not isinstance(item, dict) or type(item.get("env_idx")) is not int or item["env_idx"] < 0:
                        raise WsError(ErrorCode.INVALID_FRAME, "Invalid env_idx")
                    index, layout = item["env_idx"], item.get("layout_id")
                    if index in layouts:
                        raise WsError(ErrorCode.INVALID_FRAME, "Duplicate env_idx")
                    if layout is not None and (type(layout) not in (int, str) or len(str(layout)) > 256):
                        raise WsError(ErrorCode.INVALID_FRAME, "Invalid layout_id")
                    layouts[index] = layout
                self.service.capacity.reserve(self.identity, len(environments))
                # Normalize optional top-level layout metadata for old clients.
                frame.payload["environments"] = environments
                try:
                    result = await super()._dispatch_frame(frame)
                except Exception:
                    await self.service.run_cleanup(self.model.reset)
                    self.service.capacity.release(self.identity)
                    raise
                self.layouts, self.prepared_at = layouts, time.monotonic()
                await self.service.run_model(write_json, self.model.run / "session.json", {
                    "evaluation_id": self.identity[0], "action_case_id": self.identity[1],
                    "task_name": self.task_name, "trial_id": self.trial_id,
                    "layouts": layouts, "template_dir": str(self.service.models[self.task_name].run),
                })
                return result
            if not self.model.contexts:
                raise WsError(ErrorCode.INVALID_FRAME, "prepare_case must precede observations and inference")
            if kind == MessageType.TRIAL_END:
                result = frame.payload.get("result", frame.payload)
                items = result.get("results", [{"env_idx": 0}])
                indices = [item.get("env_idx") for item in items]
                contexts = self._targets(True, indices)
                await self.service.scheduler.detach(contexts)
                response = await super()._dispatch_frame(frame)
                self.service.capacity.reserve(self.identity, len(self.model.contexts))
                for index in indices:
                    self.layouts.pop(index, None)
                if not self.model.contexts:
                    self.prepared_at = None
                return response
            return await super()._dispatch_frame(frame)


class TaskRoutingServer(PolicyServer):
    """Routing catalog is per task; mutable ownership is per client case/episode."""

    def __init__(self, models, config, *, run_dir, max_active_agents=256,
                 disconnect_grace_s=900, max_sessions=1024, max_connections=1024,
                 max_inflight_requests=4096, startup_parallelism=16, startup_timeout_s=120,
                 idle_session_s=1800, episode_timeout_s=7200, model_threads=64,
                 cleanup_threads=16, poll_interval_s=0.05):
        super().__init__(None, config)
        self.models = models
        self.capacity = AgentCapacity(max_active_agents)
        self.disconnect_grace_s, self.idle_session_s = disconnect_grace_s, idle_session_s
        self.episode_timeout_s, self.poll_interval_s = episode_timeout_s, poll_interval_s
        self.max_sessions, self.max_connections = max_sessions, max_connections
        self.max_inflight_requests, self.startup_timeout_s = max_inflight_requests, startup_timeout_s
        self.model_pool = ThreadPoolExecutor(max_workers=model_threads, thread_name_prefix="policy-call")
        self.cleanup_pool = ThreadPoolExecutor(max_workers=cleanup_threads, thread_name_prefix="policy-cleanup")
        self.scheduler = StartupScheduler(startup_parallelism, startup_timeout_s, self.run_model)
        self.sessions, self.creating, self.peers = {}, set(), set()
        self.bindings, self.expired, self.replies = OrderedDict(), OrderedDict(), OrderedDict()
        self.case_peers, self.case_inflight = {}, {}
        self.run_dir = Path(run_dir)
        self.peer_context = ContextVar("embodiedevo_peer", default=None)
        self.next_generation = 0
        self.reaper = None
        self.stopping = asyncio.Event()
        self.stopped = False

    async def run_model(self, method, *args):
        return await asyncio.get_running_loop().run_in_executor(self.model_pool, method, *args)

    async def run_cleanup(self, method, *args):
        return await asyncio.get_running_loop().run_in_executor(self.cleanup_pool, method, *args)

    def status(self):
        return {"connections": len(self.peers), "sessions": len(self.sessions),
                "reserved_agents": self.capacity.total, "agent_limit": self.capacity.limit,
                "starting_agents": self.scheduler.starting, "queued_agents": self.scheduler.queued,
                "peak_starting_agents": self.scheduler.peak_starting,
                "inflight_requests": len(self._inflight), "server_instance_id": self._instance_id}

    async def start(self):
        if self.stopped:
            raise RuntimeError("A stopped service cannot be restarted; create a new instance")
        await super().start()
        if self.reaper is None:
            self.reaper = asyncio.create_task(self._reap())
        await asyncio.to_thread(write_json, self.run_dir / "service-status.json", dict(self.status(), ready=True, url=self.url))

    async def stop(self):
        if self.stopped:
            return
        self.stopping.set()
        self.scheduler.closed = True
        # Signal every worker before waiting for any joins. Long inference in
        # another case must not serialize shutdown of hundreds of containers.
        for slot in list(self.sessions.values()):
            for context in list(slot.model.contexts.values()):
                if context.worker is not None and context.worker.is_alive():
                    context.worker.terminate()
        results = await asyncio.gather(super().stop(),
            *(self._retire(identity, reason="server shutdown") for identity in list(self.sessions)),
            return_exceptions=True)
        if self.reaper is not None:
            await self.reaper
            self.reaper = None
        # Creation/startup requests admitted before stop may have just drained.
        results += await asyncio.gather(*(self._retire(identity, reason="server shutdown")
                                          for identity in list(self.sessions)), return_exceptions=True)
        self.stopped = True
        await asyncio.to_thread(self.model_pool.shutdown, wait=True)
        await asyncio.to_thread(self.cleanup_pool.shutdown, wait=True)
        errors = [str(value) for value in results if isinstance(value, BaseException)]
        await asyncio.to_thread(write_json, self.run_dir / "service-status.json", dict(self.status(), ready=False, cleanup_errors=errors))
        if errors:
            raise RuntimeError("Service cleanup failed: " + "; ".join(errors))

    async def _handle_connection(self, websocket):
        if self.stopping.is_set() or len(self.peers) >= self.max_connections:
            await websocket.close(code=1013, reason="Service connection capacity exhausted; retry later")
            return
        self.next_generation += 1
        peer = Peer(self.next_generation)
        self.peers.add(peer)
        token = self.peer_context.set(peer)

        async def observe_disconnect():
            await websocket.wait_closed()
            peer.closed_at = peer.closed_at or time.monotonic()

        watcher = asyncio.create_task(observe_disconnect())
        try:
            await super()._handle_connection(websocket)
        finally:
            peer.closed_at = peer.closed_at or time.monotonic()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            self.peers.discard(peer)
            self.peer_context.reset(token)

    async def _attach_peer(self, identity):
        peer = self.peer_context.get()
        previous = self.case_peers.get(identity)
        if peer is None:
            return
        if previous is not None and previous.generation > peer.generation:
            raise WsError(ErrorCode.INVALID_FRAME, "Stale connection: this case resumed on a newer connection")
        self.case_peers[identity] = peer
        peer.cases.add(identity)
        if previous is not None and previous is not peer:
            previous.cases.discard(identity)
            await self.run_model(append_jsonl, self.run_dir / "connections.jsonl", {
                "evaluation_id": identity[0], "action_case_id": identity[1],
                "reason": "connection resumed", "generation": peer.generation, "timestamp": time.time(),
            })

    async def _expire(self, identity, reason):
        if identity not in self.expired:
            await self.run_model(append_jsonl, self.run_dir / "connections.jsonl", {
                "evaluation_id": identity[0], "action_case_id": identity[1],
                "reason": reason, "timestamp": time.time(),
            })
        self.expired[identity] = reason
        self.expired.move_to_end(identity)
        while len(self.expired) > 4096:
            self.expired.popitem(last=False)
        self.replies.pop(identity, None)

    async def _retire(self, identity, *, reason, peer=None):
        slot = self.sessions.get(identity)
        if slot is None:
            # CLOSE may race a first RESET which is still creating its model.
            key = self.case_inflight.get(identity)
            entry = self._inflight.get(key)
            if entry is not None and entry[1] is not asyncio.current_task():
                await asyncio.shield(entry[1])
            slot = self.sessions.get(identity)
        if slot is None:
            return
        async with slot.dispatch_lock:
            if self.sessions.get(identity) is not slot:
                return
            if peer is not None and self.case_peers.get(identity) is not peer:
                return
            slot.retiring = True
            await self._expire(identity, reason)
            await self.scheduler.detach(slot.model.contexts.values())
            # Failed cleanup keeps the reservation and slot quarantined; the
            # reaper retries, rather than admitting an unbounded orphan fleet.
            await self.run_cleanup(slot.model.close)
            self.capacity.release(identity)
            self.sessions.pop(identity, None)
            owner = self.case_peers.pop(identity, None)
            if owner is not None:
                owner.cases.discard(identity)

    async def _reap(self):
        interval = min(1.0, max(0.02, min(self.disconnect_grace_s, self.idle_session_s, self.episode_timeout_s) / 4))
        last_status = 0
        while not self.stopping.is_set():
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass
            now, expired = time.monotonic(), []
            for identity, slot in list(self.sessions.items()):
                peer = self.case_peers.get(identity)
                reason = None
                if slot.retiring:
                    reason = "retry cleanup"
                elif peer is not None and peer.closed_at is not None and now - peer.closed_at >= self.disconnect_grace_s:
                    reason = "disconnect grace expired"
                elif now - slot.last_activity >= self.idle_session_s:
                    reason = "session idle timeout"
                elif slot.prepared_at is not None and now - slot.prepared_at >= self.episode_timeout_s:
                    reason = "episode wall timeout"
                elif any(self.scheduler.entries.get(context) and self.scheduler.entries[context].error
                         for context in slot.model.contexts.values()):
                    reason = "agent startup failed"
                else:
                    for context in list(slot.model.contexts.values()):
                        entry = self.scheduler.entries.get(context)
                        if (entry and entry.ready and context.worker is not None
                                and not context.finished and not context.worker.is_alive()):
                            marker = context.run / "episodes" / f"episode_{context.episode:04d}" / "runtime.json"
                            state = json.loads(marker.read_text()) if marker.is_file() else {}
                            if state.get("status") != "finished":
                                reason = "agent worker failed"
                                break
                if reason:
                    expired.append(self._retire(identity, reason=reason,
                                               peer=peer if reason == "disconnect grace expired" else None))
            results = await asyncio.gather(*expired, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.error("Agent cleanup failed; reservation retained: %s", result)
            if now - last_status >= 5:
                await asyncio.to_thread(write_json, self.run_dir / "service-status.json", dict(self.status(), ready=True, url=self.url))
                last_status = now

    def _cache_response(self, request_id, msg_type, response):
        identity = (response.evaluation_id, response.action_case_id)
        if identity in self.expired:
            return
        cache = self.replies.setdefault(identity, OrderedDict())
        cache[request_id] = (msg_type, response)
        cache.move_to_end(request_id)
        while len(cache) > 64:
            cache.popitem(last=False)
        self.replies.move_to_end(identity)
        active = self.sessions.keys() | self.creating | self.case_inflight.keys()
        for old in list(self.replies):
            if len(self.replies) <= self.max_sessions + 1024:
                break
            if old not in active:
                self.replies.pop(old)
        for old in list(self.bindings):
            if len(self.bindings) <= self.max_sessions + 1024:
                break
            if old not in active:
                self.bindings.pop(old)
                owner = self.case_peers.pop(old, None)
                if owner is not None:
                    owner.cases.discard(old)

    async def process_frame(self, frame):
        identity = (frame.evaluation_id, frame.action_case_id)
        key = (*identity, frame.request_id)
        is_task = frame.message_type not in CONTROL_MESSAGES
        try:
            if is_task:
                if identity in self.expired:
                    raise WsError(ErrorCode.INVALID_FRAME,
                                  f"Session expired ({self.expired[identity]}); start a new evaluation_id")
                self.resolve_task(frame)
                await self._attach_peer(identity)
            cached = self.replies.get(identity, {}).get(frame.request_id)
            if cached is not None:
                original_type, response = cached
                return response if original_type == frame.message_type else self._reused_id_error(frame, original_type)
            entry = self._inflight.get(key)
            if entry is not None:
                original_type, task = entry
                if original_type != frame.message_type:
                    return self._reused_id_error(frame, original_type)
            else:
                if self.stopping.is_set():
                    raise WsError(ErrorCode.CALL_FAILED, "Server is shutting down")
                if is_task and identity in self.case_inflight:
                    raise WsError(ErrorCode.CALL_FAILED, "Case already has a request in flight; serialize its calls")
                if len(self._inflight) >= self.max_inflight_requests:
                    raise WsError(ErrorCode.CALL_FAILED, "Server request capacity exhausted",
                                  details={"retryable": True, "retry_after_sec": 1})
                task = asyncio.create_task(self._execute_frame(frame))
                self._inflight[key] = (frame.message_type, task)
                if is_task:
                    self.case_inflight[identity] = key

                def done(_):
                    self._inflight.pop(key, None)
                    if self.case_inflight.get(identity) == key:
                        self.case_inflight.pop(identity, None)

                task.add_done_callback(done)
            return await asyncio.shield(task)
        except WsError as exc:
            return self._error_reply(frame, exc.code, exc.message, exc.details)

    def resolve_task(self, frame):
        if any(not isinstance(value, str) or not value or len(value) > 256
               for value in (frame.evaluation_id, frame.action_case_id)):
            raise WsError(ErrorCode.INVALID_FRAME, "Task calls require evaluation_id and action_case_id (1..256 chars)")
        identity = (frame.evaluation_id, frame.action_case_id)
        explicit = frame.payload.get("task_name")
        obs = frame.payload.get("obs")
        nested = obs.get("task_name") if isinstance(obs, dict) else None
        if explicit is not None and nested is not None and explicit != nested:
            raise WsError(ErrorCode.INVALID_FRAME, "Conflicting task_name fields")
        explicit = explicit if explicit is not None else nested
        inferred = frame.action_case_id.removesuffix("_case") if frame.action_case_id.endswith("_case") else None
        previous = self.bindings.get(identity)
        if (inferred is not None and inferred.removesuffix("_random") not in self.models
                and (explicit is not None or previous is not None)):
            inferred = None
        normalized = []
        for value in (v for v in (explicit, inferred) if v is not None):
            if not isinstance(value, str) or not value or Path(value).name != value:
                raise WsError(ErrorCode.INVALID_FRAME, "Invalid task_name")
            task = value.removesuffix("_random")
            if task not in self.models:
                raise WsError(ErrorCode.INVALID_FRAME, f"Task is not served: {value}")
            normalized.append(task)
        if previous is not None:
            normalized.append(previous)
        if not normalized:
            raise WsError(ErrorCode.INVALID_FRAME, "Supply task_name or action_case_id=<task>_case")
        if len(set(normalized)) != 1:
            raise WsError(ErrorCode.INVALID_FRAME, "Task metadata conflicts with the bound case")
        return identity, normalized[0]

    async def _dispatch_frame(self, frame):
        if frame.message_type == MessageType.CLOSE:
            peer = self.peer_context.get()
            if peer is not None:
                peer.closed_at = time.monotonic()
                await asyncio.gather(*(self._retire(identity, peer=peer, reason="client closed")
                                      for identity in list(peer.cases)))
            return None
        if frame.message_type in {MessageType.HELLO, MessageType.HEARTBEAT}:
            reply = await super()._dispatch_frame(frame)
            if frame.message_type == MessageType.HELLO:
                reply.payload["service"] = self.status()
            return reply
        identity, task = self.resolve_task(frame)
        slot = self.sessions.get(identity)
        if slot is None:
            if frame.message_type not in {MessageType.RESET, MessageType.PREPARE_CASE}:
                raise WsError(ErrorCode.INVALID_FRAME, "Begin a case with reset and prepare_case")
            if len(self.sessions) + len(self.creating) >= self.max_sessions:
                raise WsError(ErrorCode.CALL_FAILED, "Session capacity exhausted",
                              details={"retryable": True, "retry_after_sec": 1})
            self.creating.add(identity)
            self.bindings[identity] = task
            try:
                directory = self.models[task].run / "sessions" / uuid4().hex
                model = await self.run_model(lambda: SessionModel(self.models[task], directory,
                                                                  startup_timeout=self.startup_timeout_s))
                slot = self.sessions[identity] = CaseSlot(model, self, identity)
                await self.run_model(append_jsonl, self.run_dir / "routes.jsonl", {
                    "evaluation_id": identity[0], "action_case_id": identity[1],
                    "task_name": task, "session_dir": str(directory), "timestamp": time.time(),
                })
            finally:
                self.creating.discard(identity)
        try:
            return await slot._dispatch_frame(frame)
        except Exception as exc:
            # Input/prepare errors are retryable without disturbing a prepared
            # case. A worker/model failure is terminal and must release resources.
            invalid = isinstance(exc, WsError) and exc.code == ErrorCode.INVALID_FRAME
            if frame.message_type != MessageType.PREPARE_CASE and not invalid:
                await self._retire(identity, reason="agent call failed")
            raise


def selected_tasks(cfg):
    root = resolve_workspace(cfg)
    manifest = json.loads((root / "recipes/manifest.json").read_text())
    available = {Path(name).stem for name in manifest["files"] if name.endswith(".md")}
    requested = cfg.get("server_tasks")
    if requested is None:
        return sorted(available)
    if not isinstance(requested, list) or not requested:
        raise ValueError("server_tasks must be null (all tasks) or a nonempty list")
    tasks = []
    for task in requested:
        if not isinstance(task, str) or task != Path(task).name:
            raise ValueError("server_tasks must contain task names")
        task = task.removesuffix("_random")
        if task not in available or task in tasks:
            raise ValueError(f"Unknown or duplicate server task: {task}")
        tasks.append(task)
    return tasks


def close_models(models):
    for model in models.values():
        model.close()


def server_settings(cfg):
    fields = {
        "max_active_agents": ("server_max_active_agents", 256),
        "max_sessions": ("server_max_sessions", 1024),
        "max_connections": ("server_max_connections", 1024),
        "max_inflight_requests": ("server_max_inflight_requests", 4096),
        "startup_parallelism": ("server_startup_parallelism", 16),
        "model_threads": ("server_model_threads", 64),
        "cleanup_threads": ("server_cleanup_threads", 16),
    }
    result = {}
    for name, (key, default) in fields.items():
        value = cfg.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        result[name] = value
    for name, key, default in (
        ("disconnect_grace_s", "server_disconnect_grace_sec", 900),
        ("startup_timeout_s", "server_startup_timeout_sec", 120),
        ("idle_session_s", "server_idle_session_sec", 1800),
        ("episode_timeout_s", "server_episode_timeout_sec", 7200),
        ("poll_interval_s", "server_poll_interval_sec", 0.05),
    ):
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        result[name] = float(value)
    return result


def prepare_models(cfg):
    cfg = dict(cfg)
    settings = server_settings(cfg)
    tasks = selected_tasks(cfg)
    if not tasks:
        raise ValueError("No tasks found in the workspace manifest")
    value = setting("RUN_DIR") or cfg.get("run_dir")
    if value:
        run = Path(value).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
    else:
        parent = POLICY_DIR.parent.parent / "runs/embodiedevo"
        parent.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="server-", dir=parent))
    models = {}
    try:
        for task in tasks:
            models[task] = Model({**cfg, "task_name": task}, run_dir=run / "tasks" / task)
        write_json(run / "server.json", {
            "diagnostic": cfg["diagnostic"], "tasks": tasks, "task_snapshots": len(tasks),
            "agent_start": "first_inference_after_observation", "run_dir": str(run),
            "isolation": "evaluation_id/action_case_id/episode/env_idx", **settings,
        })
    except BaseException:
        close_models(models)
        raise
    return models, run
