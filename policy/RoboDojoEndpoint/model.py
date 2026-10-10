"""Transport-only adapter with one remote session per batched environment."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
import os
import threading
from uuid import uuid4

import numpy as np

from XPolicyLab.client_server.ws import WsModelClient
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    """Forward observations and actions; inference remains on the remote service."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if model_cfg.get("action_type") != "joint":
            raise ValueError("RoboDojoEndpoint supports action_type=joint")
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("RoboDojoEndpoint supports env_cfg_type=arx_x5")
        self.url = os.environ.get("POLICY_ENDPOINT_URL", "")
        if not self.url.startswith(("ws://", "wss://")):
            raise ValueError("Set POLICY_ENDPOINT_URL to the supplied endpoint")
        self.batch_size = int(model_cfg.get("max_batch_size", 8))
        if not 1 <= self.batch_size <= 8:
            raise ValueError("max_batch_size must be between 1 and 8")
        dimensions = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(dimensions["arm_dim"]) != 2 or len(dimensions["ee_dim"]) != 2:
            raise ValueError("Expected a dual-arm robot configuration")
        self.action_dimensions = {}
        for index, prefix in enumerate(("left_", "right_")):
            self.action_dimensions[prefix + "arm_joint_state"] = dimensions["arm_dim"][index]
            self.action_dimensions[prefix + "ee_joint_state"] = dimensions["ee_dim"][index]
        self._clients, self._pending = {}, {}
        self._single = None
        self._active = []
        self._case_meta = None
        self._mode = None
        self._closed = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.batch_size)

    def _parallel(self, function, items):
        futures = [self._executor.submit(function, item) for item in items]
        results, error = [], None
        # Drain all calls before propagating an error; reset must not race an RPC.
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(None)
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return results

    def _new_client(self):
        session = uuid4().hex
        task = self.model_cfg.get("task_name")
        case_id = (f"{task.strip()}_case"
                   if isinstance(task, str) and task.strip() and len(task) <= 128
                   else f"case-{session}")
        if isinstance(self._case_meta, dict):
            supplied_case = self._case_meta.get("action_case_id")
            if isinstance(supplied_case, str) and supplied_case and len(supplied_case) <= 256:
                case_id = supplied_case
        client = WsModelClient(url=self.url, evaluation_id=f"endpoint-model-{session}",
                               trial_id=f"trial-{session}", action_case_id=case_id)
        try:
            if self._case_meta is not None:
                client.call(func_name="prepare_case", obs=self._session_payload(self._case_meta))
            client.call(func_name="reset")
        except Exception:
            client.close()
            raise
        return client

    @staticmethod
    def _session_payload(payload):
        # The outer evaluator and inner per-environment clients have different
        # protocol identities. Do not forward an outer identity as an inner one.
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("Case/trial metadata must be a dictionary")
        return {key: value for key, value in payload.items()
                if key not in {"evaluation_id", "trial_id", "action_case_id", "repeat_index"}}

    def _select_mode(self, mode):
        if self._closed:
            raise RuntimeError("Endpoint adapter is closed")
        if self._mode is not None and self._mode != mode:
            raise ValueError("Call reset() before switching single/batch APIs")
        self._mode = mode

    @property
    def client(self):
        with self._lock:
            self._select_mode("single")
            if self._single is None:
                self._single = self._new_client()
            return self._single

    def _validate_actions(self, actions):
        if not isinstance(actions, list) or not actions:
            raise ValueError("Endpoint must return a nonempty action chunk")
        for action in actions:
            if not isinstance(action, dict) or set(action) != set(self.action_dimensions):
                raise ValueError("Endpoint returned unexpected action keys")
            for key, dimension in self.action_dimensions.items():
                value = np.asarray(action[key])
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise ValueError("Endpoint returned invalid joint action dimensions or values")
                action[key] = value
        return actions

    def update_obs(self, obs):
        with self._lock:
            # Official policy server supplies decoded RGB. Do not decode here.
            return self.client.call(func_name="update_obs", obs=obs)

    def get_action(self):
        with self._lock:
            return self._validate_actions(self.client.call(func_name="get_action"))

    def _env_indices(self, indices):
        result = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise ValueError("env_idx must be a nonnegative integer")
            result.append(int(index))
        if len(result) != len(set(result)):
            raise ValueError("Duplicate env_idx in batch")
        if len(result) > self.batch_size:
            raise ValueError("Batch exceeds max_batch_size")
        return result

    def _drop_clients(self, indices):
        clients = [self._clients.pop(index) for index in indices]
        for index in indices:
            self._pending.pop(index, None)
        self._parallel(lambda client: client.close(), clients)

    def update_obs_batch(self, obs_list):
        with self._lock:
            self._select_mode("batch")
            if not isinstance(obs_list, list) or any(not isinstance(obs, dict) for obs in obs_list):
                raise ValueError("Expected a list of observation dictionaries")
            indices = self._env_indices([obs.get("env_idx") for obs in obs_list])
            self._drop_clients([index for index in self._clients if index not in indices])

            def update(item):
                index, obs = item
                client = self._clients.get(index)
                if client is None:
                    client = self._new_client()
                    self._clients[index] = client
                    self._pending[index] = deque()
                return client.call(func_name="update_obs", obs=obs)

            self._parallel(update, list(zip(indices, obs_list)))
            self._active = indices

    def get_action_batch(self, env_idx_list=None):
        with self._lock:
            self._select_mode("batch")
            indices = self._env_indices(self._active if env_idx_list is None else env_idx_list)
            if any(index not in self._active for index in indices):
                raise ValueError("Each active environment needs update_obs_batch before inference")
            if not indices:
                return []

            def fetch(index):
                actions = self._clients[index].call(func_name="get_action")
                self._pending[index].extend(self._validate_actions(actions))

            self._parallel(fetch, [index for index in indices if not self._pending[index]])
            # Equal chunks are required by the official loop. Preserve unused
            # tails rather than pad, drop, or repeat actions.
            size = min(len(self._pending[index]) for index in indices)
            return [[self._pending[index].popleft() for _ in range(size)] for index in indices]

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Endpoint adapter is closed")
            self._drop_clients(list(self._clients))
            self._active, self._mode = [], None
            if self._single is not None:
                self._single.close()
                self._single = None

    def prepare_case(self, case_meta=None):
        with self._lock:
            payload = self._session_payload(case_meta)
            self._case_meta = case_meta
            clients = list(self._clients.values()) + ([self._single] if self._single else [])
            supplied_case = case_meta.get("action_case_id") if case_meta else None
            if isinstance(supplied_case, str) and supplied_case and len(supplied_case) <= 256:
                for client in clients:
                    client.action_case_id = supplied_case
            self._parallel(lambda client: client.call(func_name="prepare_case", obs=payload), clients)

    def on_trial_end(self, result=None):
        with self._lock:
            payload = self._session_payload(result)
            clients = list(self._clients.values()) + ([self._single] if self._single else [])
            try:
                self._parallel(lambda client: client.call(func_name="trial_end", obs=payload), clients)
            finally:
                self.reset()

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self.reset()
            finally:
                self._closed = True
                self._executor.shutdown(wait=True)
