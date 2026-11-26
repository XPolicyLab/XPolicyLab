"""Session-local flash/memory state over shared GPU-local WSP weights.

RPCs are serialized per GPU; learned modules are never copied per session.
"""

import copy
from functools import partial
from threading import RLock

from .official_client import OfficialClientMixin
from .fixed_batch import FixedBatchMixin
from .worker_isolation import IsolatedFanoutMixin
from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.multi_gpu import (
    MultiGpuModel, ProcessWorker,
)


class SessionCallError(RuntimeError):
    pass


def fork_adapter(base):
    """Share learned modules; replace every episode-owned adapter field via reset.

Wan22 rolling KVs and event/visual memory are explicit inputs/outputs. The
single-rank plugin/kernel shells get private coordination counters; learned
modules and per-call scratch computation remain shared under serialization.
"""
    adapter = copy.copy(base)
    policy = copy.copy(base.policy)
    policy._modules = dict(base.policy._modules)
    wam = copy.copy(base.policy.wam)
    wam._modules = dict(base.policy.wam._modules)
    context = base.policy.wam._distributed
    if context.world_size != 1 or context.collective is not None:
        raise ValueError("Session sharing requires independent single-rank workers")
    from worldscape_policy.wam.wan22.distributed import Wan22DistributedContext
    wam._distributed = Wan22DistributedContext.single_rank()
    wam._numerical_kernel = copy.copy(base.policy.wam._numerical_kernel)
    wam._numerical_kernel._distributed = wam._distributed
    policy.wam = wam
    adapter.policy = policy
    adapter.reset()
    return adapter


class WorkerSessions:
    """One worker process, one loaded model, independent adapters per session."""

    def __init__(self, config):
        import os
        import tempfile
        if os.environ.get("WSP_SERVER_TRITON_ROOT"):
            os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(
                prefix=f"worker-{os.getpid()}-", dir=os.environ["WSP_SERVER_TRITON_ROOT"])
        import torch
        from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.robodojo_model import RoboDojoModel
        torch.cuda.set_device(config["device"])
        self.base = RoboDojoModel(config)
        self.sessions = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def session_call(self, key, method, args):
        if method == "release":
            self.sessions.pop(key, None)
            return {"ok": True, "result": None}
        try:
            if method == "reset":
                # Drop old state before allocating the new generator/memory.
                self.sessions.pop(key, None)
                self.sessions[key] = fork_adapter(self.base)
                return {"ok": True, "result": None}
            if key not in self.sessions:
                raise SessionCallError("Session state is absent; reset before continuing")
            result = getattr(self.sessions[key], method)(*args)
            return {"ok": True, "result": result}
        except Exception as exc:
            # Device/transport failures retain the original fail-closed pool
            # behavior. Ordinary observation/sequence errors poison only a trial.
            import torch
            import traceback
            traceback.print_exc()
            if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                marker in str(exc).lower() for marker in
                ("cuda error", "device-side assert", "illegal memory access", "cublas", "cudnn")
            ):
                raise
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class SessionCoordinator(MultiGpuModel):
    """Original routing/commit algorithms with a private transaction per trial."""

    def __init__(self, manager, key, worker_index):
        pool = manager.pool
        self.__dict__.update(pool.__dict__)
        self._shared_pool = pool
        self.worker_index = worker_index
        # A virtual one-worker coordinator preserves native commit semantics.
        self._workers = [pool._workers[worker_index]]
        self.max_envs_per_session = getattr(manager, "max_envs_per_session", 1)
        self.capacity = self.max_envs_per_session
        self._session_key = key
        self._owners = {}
        self._latest = []
        self._active = None
        self._pending = None
        self._closed = False
        self._failed = False

    def _fanout(self, calls):
        self._check()
        result = self._shared_pool._fanout({
            self.worker_index: ("session_call", (self._session_key, method, args))
            for index, (method, args) in calls.items()
        })
        errors = [value["error"] for value in result.values() if not value["ok"]]
        if errors:
            raise SessionCallError("; ".join(errors))
        return {0: result[self.worker_index]["result"]} if calls else {}

    def close(self):
        # Parent error handling must never terminate another session's workers.
        self._failed = True

    def get_runtime_info(self):
        return {"runtime_profile": "robodojo", "supports_batch": True,
                "requires_explicit_commit": True, "multi_gpu": True,
                "session_isolation": True}


class OfficialSession(FixedBatchMixin, OfficialClientMixin, SessionCoordinator):
    pass


class ConcurrentWeightPool(IsolatedFanoutMixin, MultiGpuModel):
    """Disjoint worker pipes may run concurrently; fatal close happens once."""

    def __init__(self, *args, **kwargs):
        self._close_lock = RLock()
        super().__init__(*args, **kwargs)

    def close(self):
        with self._close_lock:
            super().close()


class SessionPool:
    def __init__(self, config, *, max_sessions_per_gpu=1, max_envs_per_session=1):
        if max_envs_per_session not in range(1, 11):
            raise ValueError("Supported batch capacities are integers 1-10")
        self.max_envs_per_session = max_envs_per_session
        if max_sessions_per_gpu != 1:
            raise ValueError("This deployment requires one session per GPU")
        if config.get("wam_history_mode") != "rolling" or config.get("mode") != "interactive":
            raise ValueError("Session mode requires rolling history and interactive mode")
        self.pool = ConcurrentWeightPool(config, worker_factory=partial(ProcessWorker, model_factory=WorkerSessions))
        self.model_cfg = self.pool.model_cfg
        self.max_sessions_per_gpu = max_sessions_per_gpu
        self.assignments = {}
        self._allocation_lock = RLock()

    def allocate(self, key):
        # Reservation is short and never performs GPU or pipe I/O.
        with self._allocation_lock:
            self.pool._check()
            if key in self.assignments:
                raise ValueError("Session already allocated")
            counts = [list(self.assignments.values()).count(i) for i in range(len(self.pool._workers))]
            available = [i for i in range(len(counts))
                         if not hasattr(self.pool, 'worker_available') or self.pool.worker_available(i)]
            if not available:
                raise ValueError("GPU session capacity exhausted; no healthy worker")
            index = min(available, key=lambda i: (counts[i], i))
            if counts[index] >= self.max_sessions_per_gpu:
                raise ValueError("GPU session capacity reached; end a trial before starting another")
            session = OfficialSession(self, key, index)
            self.assignments[key] = index
            return session

    def create(self, key):
        session = self.allocate(key)
        try:
            session.reset()
        except BaseException:
            self.release(key)
            raise
        return session

    def release(self, key):
        with self._allocation_lock:
            index = self.assignments.get(key)
        if index is None:
            return
        if not self.pool._closed:
            result = self.pool._fanout({index: ("session_call", (key, "release", ()))})
            if not result[index]["ok"]:
                raise SessionCallError(result[index]["error"])
        with self._allocation_lock:
            self.assignments.pop(key, None)

    def get_runtime_info(self):
        return {"runtime_profile": "robodojo", "supports_batch": self.max_envs_per_session > 1,
                "max_envs_per_session": self.max_envs_per_session,
                "max_environments": len(self.pool._workers) * self.max_envs_per_session,
                "batch_lifecycle": "fixed_shrinking_until_reset",
                "supports_single_env_batch": True,
                "requires_explicit_commit": False, "multi_gpu": True,
                "client_mode": "official", "session_isolation": True,
                "scheduler": "sticky_per_gpu", "max_sessions_per_gpu": self.max_sessions_per_gpu,
                "worker_count": len(self.pool._workers),
                "observation_contract": "one_update_per_executed_action"}

    def close(self):
        self.pool.close()
