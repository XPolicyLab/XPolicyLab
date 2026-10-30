"""One-way task-request switching; current observations stay with each episode."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
import os

import numpy as np

from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .gm.runtime.session import SwitchRequest


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, default=_json, sort_keys=True).encode()).hexdigest()


@dataclass
class Episode:
    identity: str
    env_idx: int
    branch: str = "gm"
    step: int = 0
    observed: dict = field(default_factory=dict)
    cache: deque = field(default_factory=deque)
    counts: Counter = field(default_factory=Counter)
    last_sequence: int = 0
    status: dict = field(default_factory=dict)
    switch_step: int | None = None
    reason: str | None = None
    wam_calls: int = 0
    wam_latencies_ms: list = field(default_factory=list)
    get_action_latencies_ms: list = field(default_factory=list)
    first_wam_action_recorded: bool = False
    started: float = field(default_factory=time.monotonic)
    stopped: bool = False
    observation_sequence: int = 0
    last_switch_sequence: int = 0


class SkillSwitchController:
    def __init__(self, model):
        self.model = model
        self.cfg = model.model_cfg
        seed = int(self.cfg.get("seed", -1))
        transport = float(os.environ.get("XPL_REQUEST_TIMEOUT_S", "120"))
        self.request_budget_s = max(0.01, min(float(self.cfg.get("geact_request_timeout_s", 600)), transport - 10.0))
        if self.cfg.get("memory_enabled", False) or self.cfg.get("bbox_enabled", False):
            raise ValueError("skill switching requires the existing memory-free, bbox-free WAM interface")
        if self.cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("skill switching requires the verified arx_x5 robot layout")
        dims = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        self.action_shapes = {}
        for arm, arm_dim, ee_dim in zip(("left", "right"), dims["arm_dim"], dims["ee_dim"]):
            self.action_shapes[f"{arm}_arm_joint_state"] = (int(arm_dim),)
            self.action_shapes[f"{arm}_ee_joint_state"] = (int(ee_dim),)
        # Authored arx_x5 hard joint limits.
        self.arm_joint_limits = np.array([[-9.999999450937775, 9.999999450937775]] * 5 + [[-3.139999816941817, 3.139999816941817]])
        self.joint_limit_source = {"kind": "authored_x5_hard_limits"}
        if list(self.action_shapes.values()) != [(6,), (1,), (6,), (1,)]:
            raise ValueError("skill switching requires two 6-joint arms and two grippers")
        self.episodes = {}
        self.generations = Counter()
        self.latest = []
        self.rpc_id = 0
        out = (self.cfg.get("code") or {}).get("out_dir")
        self.out = Path(out) if out else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self.base = {"task_config": self.cfg.get("task_name"), "seed": seed,
                     "config_id": _digest(self.cfg)}

    def _write(self, name, record):
        if self.out:
            with (self.out / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=_json, ensure_ascii=False) + "\n")

    def _event(self, ep, kind, *, before=None, reason=None, features=None, elapsed_ms=None,
               observation_ref=None, action_ref=None, remaining_ms=None, thresholds=None):
        budget = self.cfg.get("control_step_budget")
        self._write("events.jsonl", {
            **self.base, "episode_id": ep.identity, "env_idx": ep.env_idx,
            "control_step": ep.step, "rpc_id": self.rpc_id, "elapsed_ms": elapsed_ms,
            "stage": ep.status.get("stage"), "event_type": kind,
            "branch_before": before or ep.branch, "branch_after": ep.branch,
            "reason": reason, "features": features or {}, "thresholds": thresholds or {},
            "observation_ref": observation_ref, "action_ref": action_ref,
            "remaining_control_steps": None if budget is None else max(0, int(budget) - ep.step),
            "remaining_request_ms": remaining_ms,
        })

    def _end(self, key, reason):
        ep = self.episodes.pop(key, None)
        if ep is None:
            return
        self._event(ep, "episode_end", reason=reason)
        self._write("episodes.jsonl", {
            **self.base, "episode_id": ep.identity, "env_idx": key,
            "status": "result_pending", "success": None, "score_0_1": None,
            "termination_reason": reason, "control_steps": ep.step,
            "switch_count": int(ep.switch_step is not None),
            "switch_step": ep.switch_step, "switch_reason": ep.reason,
            "wam_calls": ep.wam_calls,
            "wam_inference_latencies_ms": ep.wam_latencies_ms,
            "policy_get_action_latencies_ms": ep.get_action_latencies_ms,
            "rpc_latencies_ms": ep.wam_latencies_ms,
            "rpc_latency_source": "legacy WAM chunk inference; whole RPC is timed on the evaluation client side",
            "total_wall_s": time.monotonic() - ep.started,
            "trace_ref": ep.status.get("trace_ref"), "video_ref": None,
            "wam_error": ep.stopped,
        })
        for branch in (self.model.code, self.model.wam):
            drop = getattr(branch, "drop_env", None)
            if drop:
                drop(key)

    def update_obs_batch(self, observations):
        indices = self.model._env_indices(observations)
        # A shrinking batch finishes dropped envs; a new generation starts only
        # after reset or an explicitly dropped index appears in a later batch.
        for key in set(self.latest) - set(indices):
            self._end(key, "left_active_batch")
        self.latest = indices
        code_obs = []
        for key, raw_obs in zip(indices, observations):
            # Preserve resolved batch identity when forwarding a code subset or
            # one WAM observation whose original input omitted env_idx.
            obs = {**raw_obs, "env_idx": key}
            ep = self.episodes.get(key)
            if ep is None:
                generation = self.generations[key]
                self.generations[key] += 1
                ep = self.episodes[key] = Episode(f"batch-{generation:06d}/env-{key:03d}", key)
                self._event(ep, "episode_start", features={"identity_source": "adapter_reset_generation_and_env_idx"})
            ep.observed = obs
            ep.observation_sequence += 1
            if ep.branch == "gm":
                code_obs.append(obs)
        if code_obs:
            self.model.code.update_obs_batch(code_obs)

    def _check(self, ep):
        fresh = [ev for ev in ep.status.get("events", []) if ev.get("sequence", 0) > ep.last_sequence]
        for ev in fresh:
            ep.last_sequence = max(ep.last_sequence, ev.get("sequence", 0))
            ep.counts[ev["kind"]] += 1
            self._event(ep, "code_event", features=ev)
        return None

    def _snapshot(self, ep, label):
        if not self.out:
            return None
        destination = self.out / "observations" / ep.identity
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"step-{ep.step:06d}-{label}.npz"
        arrays = {f"state__{k}": np.asarray(v) for k, v in ep.observed["state"].items()}
        arrays.update({f"rgb__{cam}": np.asarray(value["color"])
                       for cam, value in ep.observed["vision"].items() if "color" in value})
        arrays["instruction"] = np.asarray(ep.observed.get("instruction") or ep.observed.get("instructions") or "")
        arrays["env_idx"] = np.asarray(ep.env_idx)
        np.savez_compressed(path, **arrays)
        return str(path)

    def _switch(self, ep, rule, remaining_ms):
        self.model.code.drop_env(ep.env_idx)
        ep.cache.clear()
        ep.branch = "wam"
        ep.switch_step = ep.step
        ep.reason = rule["name"]
        self._event(ep, "skill_switch", before="gm", reason=ep.reason,
                    features={"failure": ep.status.get("failure"), "code_step": ep.status.get("step")},
                    thresholds=rule, observation_ref=self._snapshot(ep, "skill_switch"), remaining_ms=remaining_ms)

    def _hold(self, ep):
        return {key: np.asarray(ep.observed["state"][key], dtype=np.float32).copy()
                for key in self.action_shapes}

    def _validate_wam(self, chunk):
        if not isinstance(chunk, list) or not chunk:
            raise ValueError("WAM returned an empty/non-list action chunk")
        for action in chunk:
            if set(action) != set(self.action_shapes):
                raise ValueError("WAM action keys do not match robot layout")
            for key, shape in self.action_shapes.items():
                value = np.asarray(action[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid WAM action {key}: {value.shape}")
                lower, upper = (0, 1) if "ee_joint" in key else (self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1])
                if np.any(value < lower - 1e-5) or np.any(value > upper + 1e-5):
                    raise ValueError(f"WAM action outside authored robot joint limits: {key}")

    def _wam_action(self, ep, deadline):
        if ep.stopped:
            return self._hold(ep)
        if not ep.cache:
            started = time.monotonic()
            try:
                remaining = deadline - started
                if remaining <= 0:
                    raise TimeoutError("composite request budget exhausted before WAM inference")
                wam = self.model._wam()
                # Bound metadata/connect/inference together using the transport's
                # absolute deadline; finite timeout closes a stale connection.
                client = getattr(wam, "client", None)
                if client is not None:
                    client.deadline_monotonic = deadline
                wam.update_obs(ep.observed)
                ep.wam_calls += 1
                chunk = wam.get_action_batch([ep.env_idx])[0]
                self._validate_wam(chunk)
                elapsed_ms = (time.monotonic() - started) * 1000
                ep.wam_latencies_ms.append(elapsed_ms)
                ep.cache.extend(chunk)
                delta = max(float(np.max(np.abs(np.asarray(chunk[0][key]) - ep.observed["state"][key])))
                            for key in self.action_shapes if "arm_joint" in key)
                self._event(ep, "wam_chunk", elapsed_ms=elapsed_ms,
                            features={"length": len(chunk), "first_action_delta_rad": delta,
                                      "joint_limit_source": self.joint_limit_source},
                            action_ref=_digest(chunk), remaining_ms=(deadline - time.monotonic()) * 1000)
            except Exception as exc:
                ep.stopped = True
                ep.cache.clear()
                self._event(ep, "wam_error", reason=repr(exc),
                            elapsed_ms=(time.monotonic() - started) * 1000,
                            features={"controlled_action": "hold_current_joint_and_gripper_state", "retry": False})
                return self._hold(ep)
            finally:
                if "client" in locals() and client is not None:
                    client.deadline_monotonic = None
        action = ep.cache.popleft()
        if not ep.first_wam_action_recorded:
            ep.first_wam_action_recorded = True
            self._event(ep, "first_wam_action", action_ref=_digest(action),
                        observation_ref=self._snapshot(ep, "first-wam-action"),
                        features={"source": "current_episode_wam_chunk", "cache_remaining": len(ep.cache)},
                        remaining_ms=max(0, (deadline - time.monotonic()) * 1000))
        return action

    def get_action_batch(self, indices=None):
        indices = self.latest if indices is None else [int(i) for i in indices]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate requested environment indices")
        if not indices or any(key not in self.episodes for key in indices):
            raise RuntimeError("get_action called without current observations")
        self.rpc_id += 1
        started = time.monotonic()
        deadline = started + self.request_budget_s
        code_keys = [key for key in indices if self.episodes[key].branch == "gm"]
        actions = {}
        if code_keys:
            gm = self.model.code
            chunks = gm.get_action_batch(code_keys)
            for key, chunk in zip(code_keys, chunks):
                ep = self.episodes[key]
                ep.status = gm.status(key)
                self._check(ep)
                self._event(ep, "code_tick", elapsed_ms=ep.status.get("compute_ms"),
                            features={"done": ep.status.get("done"), "failure": ep.status.get("failure"),
                                      "tracking_error_rad": ep.status.get("tracking_error_rad"),
                                      "compute_wait_ms": ep.status.get("compute_wait_ms"),
                                      "compute_measurement": ep.status.get("compute_measurement"),
                                      "startup_elapsed_ms": ep.status.get("startup_elapsed_ms"),
                                      "state": ep.observed["state"]}, action_ref=_digest(chunk))
                request_data = ep.status.get("skill_switch_request")
                if request_data:
                    request = SwitchRequest.parse(request_data)
                    if request.sequence > ep.last_switch_sequence:
                        ep.last_switch_sequence = request.sequence
                        self._event(ep, "switch_boundary", reason=request.source,
                            features={"request": request.record(), "source_semantics":"explicit_code_abandonment"})
                    if not ep.status.get("switch_paused"):
                        raise RuntimeError("skill switch requires a pre-cleanup yield checkpoint")
                    self._switch(ep, {"name":request.source}, (deadline-time.monotonic())*1000)
                if ep.branch == "gm":
                    if not chunk or len(chunk) != 1:
                        raise ValueError("code policy must return exactly one action")
                    actions[key] = chunk[0]
        for key in indices:
            ep = self.episodes[key]
            if ep.branch == "wam":
                actions[key] = self._wam_action(ep, deadline)
            ep.step += 1
        elapsed_ms = (time.monotonic() - started) * 1000
        for key in indices:
            ep = self.episodes[key]
            ep.get_action_latencies_ms.append(elapsed_ms)
            self._event(ep, "policy_get_action", elapsed_ms=elapsed_ms,
                        features={"requested_env_idx": indices, "shared_batch_measurement": True},
                        remaining_ms=max(0, (deadline - time.monotonic()) * 1000))
        return [[actions[key]] for key in indices]

    def reset(self, reason="reset"):
        for key in list(self.episodes):
            self._end(key, reason)
        self.latest.clear()
