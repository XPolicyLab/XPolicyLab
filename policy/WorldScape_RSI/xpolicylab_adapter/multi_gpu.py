"""One stateful batch interface backed by independent, GPU-local WSP processes.

No torch import in the coordinator: CUDA is initialized only in spawned workers.
The existing WebSocket server serializes calls to this object, including retries.
"""

import atexit
from collections.abc import Mapping
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import math
import multiprocessing as mp
from numbers import Integral
import os
import signal
from threading import current_thread, main_thread
import traceback
from types import SimpleNamespace


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _env_ids(values):
    ids = [_integer(env, "env_idx") for env in values]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Environment indices must be non-empty and unique")
    return ids


def _build_model(config):
    import torch

    from .robodojo_model import RoboDojoModel as Model

    torch.cuda.set_device(config["device"])
    return Model(config)


def _worker_main(connection, config, model_factory):
    try:
        model = model_factory(config)
        metadata = {
            name: getattr(model, name, None) for name in (
                "model_cfg", "action_horizon", "action_steps", "upstream_root", "checkpoint",
                "task_window_chunks", "joint_action_representation", "constrain_action_padding",
            )
        }
        metadata["mode"] = getattr(getattr(model, "mode", None), "value", None)
        connection.send((True, metadata))
        while True:
            method, args = connection.recv()
            if method == "__close__":
                break
            # Only the local coordinator writes to this pipe.
            try:
                result = getattr(model, method)(*args)
            except Exception:
                connection.send((False, traceback.format_exc()))
            else:
                connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class ProcessWorker:
    """A private pipe, never an additional externally reachable policy endpoint."""

    def __init__(self, config, *, model_factory=_build_model):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker_main, args=(child, config, model_factory), daemon=True,
        )
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def _receive(self):
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(f"Policy worker failed:\n{result}")
        return result

    def ready(self):
        return self._receive()

    def call(self, method, *args):
        self.connection.send((method, args))
        return self._receive()

    def close(self):
        # Also interrupts a blocked pipe read/write or a hung CUDA call.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()


class MultiGpuModel:
    """Sticky environment routing; predictions/commits form one global transaction.

    A worker failure poisons the coordinator and stops all workers. Continuing
    after a partially applied commit would silently corrupt causal histories.
    Recovery requires restarting this service (and restarting the client trial).
    """

    supports_batch = True
    requires_explicit_commit = True

    def __init__(self, config, gpu_ids=None, *, worker_factory=ProcessWorker):
        from .runtime_config import validate_config
        config = validate_config(config)
        if config.get("runtime_profile") != "robodojo":
            raise ValueError("Multi-GPU replicas require runtime_profile=robodojo")
        gpu_ids = gpu_ids if gpu_ids is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if any(not str(gpu).isdecimal() for gpu in gpu_ids):
            raise ValueError("Multi-GPU IDs must be non-negative integers")
        self.gpu_ids = [str(int(gpu)) for gpu in gpu_ids]
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("The worker pool requires at least one distinct GPU ID")
        settings = config.get("multi_gpu") or {}
        capacity = settings.get("max_envs_per_gpu")
        self.capacity = None if capacity is None else _integer(capacity, "max_envs_per_gpu", 1)
        self.timeout = float(settings.get("worker_timeout_s", 1200))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("worker_timeout_s must be finite and positive")
        self._workers = []
        self._pool = ThreadPoolExecutor(max_workers=len(self.gpu_ids))
        self._closed = False
        self._failed = False
        self._owners = {}
        self._latest = []
        self._active = None
        self._pending = None
        self._signal_handler = None
        self._previous_sigterm = None
        atexit.register(self.close)
        if current_thread() is main_thread():
            self._previous_sigterm = signal.getsignal(signal.SIGTERM)

            def stop(signum, frame):
                self.close()
                raise SystemExit(128 + signum)

            self._signal_handler = stop
            signal.signal(signal.SIGTERM, stop)
        try:
            for index, gpu in enumerate(self.gpu_ids):
                # CUDA_VISIBLE_DEVICES is set once by the launcher. Use its local
                # ordinals, never physical IDs as torch device indices.
                worker_config = {**config, "device": f"cuda:{index}", "gpu_id": index,
                                 "multi_gpu": {"enabled": False}}
                self._workers.append(worker_factory(worker_config))
                print(f"[WorldScape_RSI multi-GPU] worker={index} gpu={gpu} device=cuda:{index}", flush=True)
            metadata = self._fanout({index: ("__ready__", ()) for index in range(len(self._workers))})
            first = metadata[0]
            for value in metadata.values():
                for key in ("action_horizon", "action_steps", "mode", "checkpoint"):
                    if value[key] != first[key]:
                        raise ValueError(f"GPU workers disagree on {key}")
            for name, value in first.items():
                setattr(self, name, SimpleNamespace(value=value) if name == "mode" else value)
            self.model_cfg = {**self.model_cfg, "device": config.get("device", "cuda"),
                              "multi_gpu": config.get("multi_gpu", {})}
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self._failed or self._closed:
            raise RuntimeError("Multi-GPU policy is unavailable; restart the server and client trial")

    def close(self):
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.close()
        self._pool.shutdown(wait=True, cancel_futures=True)
        atexit.unregister(self.close)
        if (self._signal_handler is not None and current_thread() is main_thread()
                and signal.getsignal(signal.SIGTERM) is self._signal_handler):
            signal.signal(signal.SIGTERM, self._previous_sigterm)

    def _fanout(self, calls):
        self._check()
        futures = {}
        try:
            for index, (method, args) in calls.items():
                worker = self._workers[index]
                futures[index] = (
                    self._pool.submit(worker.ready) if method == "__ready__"
                    else self._pool.submit(worker.call, method, *args)
                )
            done, pending = wait(futures.values(), timeout=self.timeout, return_when=FIRST_EXCEPTION)
            for future in done:
                if future.exception() is not None:
                    future.result()
            if pending:
                raise TimeoutError(f"Policy workers exceeded {self.timeout}s; restart the server")
            return {index: future.result() for index, future in futures.items()}
        except BaseException:
            self._failed = True
            self.close()
            raise

    def _groups(self, ids):
        groups = {}
        for env in ids:
            if env not in self._owners:
                raise ValueError(f"Unknown environment {env}; call update_obs_batch after reset")
            if self._active is not None and env not in self._active:
                raise ValueError(f"Environment {env} has retired; reset before reusing it")
            groups.setdefault(self._owners[env], []).append(env)
        return groups

    def reset(self):
        self._fanout({index: ("reset", ()) for index in range(len(self._workers))})
        self._owners = {}
        self._latest = []
        self._active = None
        self._pending = None

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, observations):
        self._check()
        if not all(isinstance(obs, Mapping) for obs in observations):
            raise TypeError("update_obs_batch expects observation mappings")
        ids = _env_ids([obs.get("env_idx", index) for index, obs in enumerate(observations)])
        if not self._owners:
            if self.capacity is not None and len(ids) > len(self._workers) * self.capacity:
                raise ValueError(
                    f"Batch size {len(ids)} exceeds {len(self._workers)} GPUs x {self.capacity} environments"
                )
            # Balanced contiguous groups on the initial batch; never redistribute
            # survivors when other environments finish. Keep global env IDs.
            self._owners = {
                env: index * len(self._workers) // len(ids) for index, env in enumerate(sorted(ids))
            }
            print(f"[WorldScape_RSI multi-GPU] environment_workers={self._owners}", flush=True)
        by_env = {env: {**obs, "env_idx": env} for env, obs in zip(ids, observations, strict=True)}
        self._fanout({
            index: ("update_obs_batch", ([by_env[env] for env in envs],))
            for index, envs in self._groups(ids).items()
        })
        self._latest = ids

    def get_action(self):
        return self.get_action_batch(self._latest[:1])[0]

    def get_action_batch(self, env_idx_list=None):
        self._check()
        if self._pending is not None:
            raise RuntimeError("Commit or discard the pending prediction before predicting again")
        ids = _env_ids(self._latest if env_idx_list is None else env_idx_list)
        groups = self._groups(ids)
        results = self._fanout({
            index: ("get_action_batch", (envs,)) for index, envs in groups.items()
        })
        try:
            actions = {}
            for index, envs in groups.items():
                actions.update(zip(envs, results[index], strict=True))
            horizons = {env: len(actions[env]) for env in ids}
            if len(set(horizons.values())) != 1 or next(iter(horizons.values())) == 0:
                raise ValueError("Workers must return equal, non-empty action horizons")
        except Exception:
            self._failed = True
            self.close()
            raise
        self._pending = horizons
        return [actions[env] for env in ids]

    def commit_action(self, execution=None):
        self._check()
        if self._pending is None:
            raise RuntimeError("There is no pending prediction to commit")
        if execution is None:
            executed, finished = self._pending, set()
        else:
            if not isinstance(execution, Mapping) or not isinstance(execution.get("executed_steps"), Mapping):
                raise ValueError("commit_action requires an executed_steps mapping")
            # Wire formats may stringify dict keys. Validate the whole report
            # before any worker mutates its committed state.
            executed = {int(env): _integer(count, "executed_steps")
                        for env, count in execution["executed_steps"].items()}
            if len(executed) != len(execution["executed_steps"]):
                raise ValueError("Duplicate execution environment indices")
            finished = {_integer(env, "finished env_idx") for env in execution.get("finished_env_idx_list", [])}
        if set(executed) != set(self._pending) or not finished.issubset(self._pending):
            raise ValueError("Execution report must match the pending environment indices")
        for env, horizon in self._pending.items():
            if executed[env] > horizon or (env not in finished and executed[env] != horizon):
                raise ValueError(f"Invalid executed steps for environment {env}; discard partial execution")
        self._fanout({
            index: ("commit_action", ({
                "executed_steps": {env: executed[env] for env in envs},
                "finished_env_idx_list": [env for env in envs if env in finished],
            },)) for index, envs in self._groups(list(self._pending)).items()
        })
        self._active = set(self._pending) - finished
        self._pending = None

    def discard_action(self, execution=None):
        self._check()
        # Broadcast, including workers whose prediction may have failed or whose
        # environments just ended. Native discard is a no-op with no pending chunk.
        args = () if execution is None else (execution,)
        self._fanout({index: ("discard_action", args) for index in range(len(self._workers))})
        self._pending = None

    def prepare_case(self, case_meta=None):
        self._fanout({index: ("prepare_case", (case_meta,)) for index in range(len(self._workers))})

    def on_trial_end(self, result=None):
        self._fanout({index: ("on_trial_end", (result,)) for index in range(len(self._workers))})
