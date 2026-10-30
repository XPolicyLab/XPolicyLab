"""Official WS protocol with sticky GPU sessions and independent execution."""

import asyncio
import json
import logging
import math
import time
import inspect
from uuid import uuid4

from .replay import ReplayError, dropped_log_count
from .replay_server import ReplayServerMixin

from client_server.ws.model_server import PolicyServer
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType

LOG = logging.getLogger(__name__)
ALLOWED_CALLS = {"reset", "update_obs", "update_obs_batch", "get_action", "get_action_batch",
                 "prepare_case", "on_trial_end"}
READ_ONLY_CALLS = {"online_info", "get_runtime_info", "session_status"}


def replace_frame(frame, **values):
    return frame.model_copy(update=values) if hasattr(frame, "model_copy") else frame.copy(update=values)


class TimedProtocol(PolicyServer):
    """Measure model RPC wall time inside its thread, excluding observation decode."""
    last_model_seconds = None

    async def _invoke_method(self, method, *args):
        if inspect.iscoroutinefunction(method):
            start = time.monotonic()
            try:
                return await method(*args)
            finally:
                self.last_model_seconds = time.monotonic() - start
        def invoke():
            start = time.monotonic()
            try:
                return method(*args)
            finally:
                self.last_model_seconds = time.monotonic() - start
        result = await asyncio.to_thread(invoke)
        return await result if inspect.isawaitable(result) else result


class SessionPolicyServer(ReplayServerMixin, PolicyServer):
    def __init__(self, manager, config, *, max_sessions=8, idle_timeout=1800,
                 ledger=None, disconnect_grace=180, replay_token=None):
        super().__init__(manager, config)
        if max_sessions < 1 or not math.isfinite(idle_timeout) or idle_timeout <= 0:
            raise ValueError("Positive session capacity and idle timeout are required")
        if not math.isfinite(disconnect_grace) or disconnect_grace <= 0:
            raise ValueError("Disconnect grace must be finite and positive")
        self.manager = manager
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.sessions = {}
        self._gpu_locks = {}
        self._janitor = None
        self._arrivals = {}
        self._stopping = False
        self.init_replay(ledger, disconnect_grace, replay_token)

    async def start(self):
        await super().start()
        self._stopping = False
        if self._janitor is None:
            self._janitor = asyncio.create_task(self._reap_loop())

    async def stop(self):
        self._stopping = True
        if self._janitor is not None:
            self._janitor.cancel()
            await asyncio.gather(self._janitor, return_exceptions=True)
            self._janitor = None
        await super().stop()
        # Direct protocol callers also have shielded execution tasks to drain.
        tasks = [task for _, task in self._inflight.values()]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=35)
            if pending:
                await asyncio.to_thread(self.manager.close)
                await asyncio.wait(pending, timeout=35)
        for key, entry in list(self.sessions.items()):
            await self._release(key, entry, "server_stop")

    def _expired(self, key, entry):
        trial = self.trials.get(key, {})
        if trial.get('state') == 'detached':
            # Disconnection has its own grace, independent of the idle timeout.
            return time.monotonic() >= trial['detached_at'] + self.disconnect_grace
        if trial.get('state') in ('closing', 'lost'):
            return True
        deadline = entry["last_used"] + self.idle_timeout
        queued = any(tuple(json.loads(request)[:2]) == key and arrived < deadline
                     for request, arrived in self._arrivals.items())
        return (time.monotonic() >= deadline and not queued
                and not entry["lock"].locked() and not entry["waiting"])

    async def _release_locked(self, key, entry, reason):
        # Caller owns both this session and its GPU until pipe cleanup completes.
        if self.sessions.get(key) is not entry:
            return
        trial = self.trials[key]
        trial['state'] = 'closing' if trial['state'] != 'lost' else 'lost'
        try:
            await asyncio.to_thread(self.manager.release, key)
        except Exception:
            trial['state'] = 'lost'
            entry['closing'] = True
            entry['release_failed'] = True
            LOG.exception('session_release_failed evaluation=%s trial=%s', *key)
            return  # Quarantine: no successful release means no free slot.
        self.sessions.pop(key)
        trial['state'] = 'ended' if trial['state'] != 'lost' else 'lost'
        LOG.info("session_released evaluation=%s trial=%s worker=%s reason=%s",
                 *key, entry["worker"], reason)

    async def _release(self, key, entry, reason):
        async with entry["lock"], self._gpu_locks[entry["worker"]]:
            await self._release_locked(key, entry, reason)

    async def _reap(self):
        self.ledger.expire()
        for key in list(self.sessions):
            try:
                self._check_worker(key)
            except ReplayError:
                pass
        expired = [(key, entry) for key, entry in list(self.sessions.items())
                   if not entry.get('release_failed') and self._expired(key, entry)]
        # Mark synchronously so a new request cannot revive a state being freed.
        for key, entry in expired:
            entry["closing"] = True
            if self.trials[key]["state"] != "lost":
                self.trials[key]["state"] = "closing"
        await asyncio.gather(*(self._release(key, entry, "idle_timeout") for key, entry in expired))

    async def _reap_loop(self):
        while True:
            await asyncio.sleep(min(30, self.idle_timeout / 2, self.disconnect_grace / 2))
            task = asyncio.create_task(self._reap())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    def status(self, exclude_request=None):
        now = time.monotonic()
        active = [list(key) for key, value in self.sessions.items() if value["executing"]]
        return {"connections": len(self.connections), "max_connections": 120,
                "dropped_log_records": dropped_log_count(),
                "active_environment_count": sum(len(getattr(v['model'], '_batch_active', None) or [])
                                                for v in self.sessions.values()),
                "executing_inference_count": sum(v['executing'] and v.get('inference', False)
                                                 for v in self.sessions.values()),
                "replay_records": len(self.ledger.records), "replay_bytes": self.ledger.bytes,
                "replay_metrics": dict(self.metrics), "max_pending": self.ledger.max_pending,
                "per_session_pending": self.ledger.per_session_pending,
                "session_isolation": True, "scheduler": "sticky_per_gpu",
                "active_sessions": len(self.sessions), "max_sessions": self.max_sessions,
                "max_sessions_per_gpu": getattr(self.manager, "max_sessions_per_gpu", 1),
                "max_envs_per_session": getattr(self.manager, "max_envs_per_session", 1),
                "max_environments": self.max_sessions * getattr(self.manager, "max_envs_per_session", 1),
                "idle_timeout_s": self.idle_timeout,
                "inflight_requests": len(self._inflight),
                "executing_sessions": active,
                "sessions": [{"evaluation_id": key[0], "trial_id": key[1],
                              "session_state": self.trials[key]["state"],
                              "worker_index": value["worker"], "waiting_requests": value["waiting"],
                              "active_env_ids": getattr(value["model"], "_batch_active", None),
                              "executing": value["executing"], "closing": value["closing"],
                              "idle_seconds": max(0, now - value["last_used"]),
                              "failed": bool(getattr(value["model"], "_official_failed", False))}
                             for key, value in self.sessions.items()]}

    async def _dispatch_frame(self, frame):
        kind = frame.message_type
        name = frame.payload.get("func_name") if kind == MessageType.CALL else None
        if kind in {MessageType.HELLO, MessageType.HEARTBEAT, MessageType.CLOSE}:
            return await super()._dispatch_frame(frame)
        if name in READ_ONLY_CALLS:
            result = (self.status(frame.request_id) if name == "session_status"
                      else await asyncio.to_thread(getattr(self.manager, name)))
            return self._reply(frame, MessageType.CALL_RESULT, {"ok": True, "result": result})
        if self._stopping:
            raise WsError(ErrorCode.CALL_FAILED, "Policy server is stopping")
        if kind == MessageType.CALL and name not in ALLOWED_CALLS:
            raise WsError(ErrorCode.INVALID_FRAME, "RPC is not allowed in session mode")
        # R2SI may prepare a task before assigning a trial. Never occupy a GPU.
        if kind == MessageType.PREPARE_CASE and not frame.trial_id:
            case_key = (frame.evaluation_id, frame.action_case_id)
            if case_key not in self.prepared_cases and len(self.prepared_cases) >= self.ledger.max_records:
                raise ReplayError('NOT_ADMITTED')
            identity = self.prepared_cases.setdefault(case_key, {})
            self._validate_task(frame, {'task_identity': identity})
            return self._reply(frame, MessageType.PREPARE_CASE_ACK, {"ok": True})
        key = (frame.evaluation_id, frame.trial_id)
        if not all(isinstance(v, str) and v.strip() and len(v) <= 256 for v in key):
            raise WsError(ErrorCode.INVALID_FRAME, "Non-empty evaluation_id and trial_id are required")
        ending = kind == MessageType.TRIAL_END or name == "on_trial_end"
        resetting = kind == MessageType.RESET or name == "reset"
        entry = self.sessions.get(key)
        if entry is not None and self._expired(key, entry):
            entry["closing"] = True
            await self._release(key, entry, "idle_timeout")
            entry = self.sessions.get(key)
        trial = self.trials[key]
        if ending and trial['state'] == 'ended':
            return self._reply(frame, MessageType.TRIAL_END_ACK if kind == MessageType.TRIAL_END
                               else MessageType.CALL_RESULT, {'ok': True})
        if trial["state"] in ("closing", "ended", "lost"):
            raise ReplayError("SESSION_LOST" if trial["state"] == "lost" else "SESSION_ENDED")
        if entry is None:
            if ending:
                trial["state"] = "ended"
                return self._reply(frame, MessageType.TRIAL_END_ACK if kind == MessageType.TRIAL_END
                                   else MessageType.CALL_RESULT, {"ok": True})
            if not resetting and kind != MessageType.PREPARE_CASE and name != "prepare_case":
                raise WsError(ErrorCode.CALL_FAILED, "Session is absent or expired; reset before continuing")
            if len(self.sessions) >= self.max_sessions:
                raise ReplayError("CAPACITY_EXHAUSTED")
            # Allocation only reserves a GPU and creates a Python coordinator.
            # No await here: concurrent creations cannot oversubscribe capacity.
            allocate = getattr(self.manager, "allocate", self.manager.create)
            try:
                model = allocate(key)
            except Exception as exc:
                raise ReplayError("CAPACITY_EXHAUSTED") from exc
            worker = getattr(model, "worker_index", 0)
            self._gpu_locks.setdefault(worker, asyncio.Lock())
            entry = {"model": model, "worker": worker, "lock": asyncio.Lock(),
                     "protocol": TimedProtocol(model, self.config), "last_used": time.monotonic(),
                     "waiting": 0, "executing": False, "closing": False,
                     "initialized": not hasattr(self.manager, "allocate")}
            self.sessions[key] = entry
            prepared = self.prepared_cases.get((frame.evaluation_id, frame.action_case_id), {})
            trial['task_identity'].update(prepared)
            trial['session_generation'] = uuid4().hex
            trial['worker_generation'], _ = self._worker_stamp(worker)
            if trial['state'] != 'detached':
                trial['state'] = 'active'
            LOG.info("session_created evaluation=%s trial=%s worker=%s", *key, worker)
        entry["waiting"] += 1
        queued_at = time.monotonic()
        try:
            async with entry["lock"], self._gpu_locks[entry["worker"]]:
                if self.sessions.get(key) is not entry or entry["closing"]:
                    raise WsError(ErrorCode.CALL_FAILED, "Session ended or expired; reset before continuing")
                self._check_worker(key)
                self._validate_task(frame, trial)
                entry["executing"] = True
                entry['inference'] = name in ('get_action', 'get_action_batch') or kind == MessageType.INFER
                start = time.monotonic()
                entry['protocol'].last_model_seconds = None
                LOG.info('request_admitted evaluation=%s trial=%s request=%s session_gpu_queue_s=%.6f',
                         *key, frame.request_id, start - queued_at)
                LOG.info("session_request_begin evaluation=%s trial=%s worker=%s request=%s method=%s",
                         *key, entry["worker"], frame.request_id, name or kind.value)
                try:
                    if ending:
                        await self._release_locked(key, entry, "trial_end")
                        if trial['state'] == 'lost':
                            raise ReplayError('SESSION_LOST', admitted=True)
                        return self._reply(frame, MessageType.TRIAL_END_ACK if kind == MessageType.TRIAL_END
                                           else MessageType.CALL_RESULT, {"ok": True})
                    if not entry["initialized"] and not resetting:
                        await asyncio.to_thread(entry["model"].reset)
                        entry["initialized"] = True
                    # A private protocol dispatcher avoids mutating self.model
                    # across concurrent calls and preserves official decoding.
                    # Binding belongs to transport and must never reach models.
                    clean = replace_frame(frame, payload={k: v for k, v in frame.payload.items() if k != '_replay'})
                    result = await entry["protocol"]._dispatch_frame(clean)
                    entry["initialized"] = True
                    return result
                finally:
                    entry["executing"] = False
                    entry["last_used"] = time.monotonic()
                    LOG.info("session_request_end evaluation=%s trial=%s worker=%s request=%s method=%s elapsed=%.3f",
                             *key, entry["worker"], frame.request_id, name or kind.value, time.monotonic() - start)
                    LOG.info('model_rpc_timing evaluation=%s trial=%s request=%s model_rpc_s=%s',
                             *key, frame.request_id, entry['protocol'].last_model_seconds)
                    LOG.info('batch_mapping evaluation=%s trial=%s request=%s env_ids=%s',
                             *key, frame.request_id, getattr(entry['model'], '_batch_active', None))
        finally:
            entry["waiting"] -= 1

    def _validate_task(self, frame, trial):
        from hashlib import sha256
        from .replay import canonical
        identity = trial['task_identity']
        candidates = []
        if frame.action_case_id is not None:
            candidates.append({'action_case_id': frame.action_case_id})
        payload = frame.payload
        name = payload.get('func_name')
        if name == 'update_obs_batch':
            rows = payload.get('obs')
            if isinstance(rows, (list, tuple)) and not 1 <= len(rows) <= getattr(self.manager, 'max_envs_per_session', 1):
                raise ReplayError('BATCH_CAPACITY_EXHAUSTED')
        if frame.message_type in (MessageType.PREPARE_CASE, MessageType.RESET):
            candidates.append(payload)
        elif name == 'prepare_case':
            candidates.append(payload.get('obs') or {})
        elif name in ('update_obs', 'update_obs_batch'):
            obs = payload.get('obs')
            candidates.extend(obs if isinstance(obs, list) else [obs])
        elif frame.message_type == MessageType.INFER:
            candidates.append(payload.get('observation'))
        changes = dict(identity)
        for row in candidates:
            if not isinstance(row, dict):
                continue
            for field in ('task_name', 'task_config', 'task_id', 'action_case_id'):
                if field in row:
                    value = sha256(canonical(row[field])).digest()
                    if field in changes and changes[field] != value:
                        raise ReplayError('TASK_IDENTITY_CONFLICT')
                    changes[field] = value
        identity.update(changes)
