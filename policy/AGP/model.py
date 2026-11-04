"""AGP stub policy-server model. The agent runs in child processes of the RoboDojo eval client (deploy.py), so this
model only answers the protocol and never decodes images; outside the debug closed loop, asking it for an action is an
error. In XPolicyLab's debug closed loop (EVAL_ENV_TYPE=debug) update_obs checks each observation and get_action
answers with DEBUG_CHUNK joint actions that hold the observed joint state, keyed and sized by
get_robot_action_dim_info(env_cfg_type). With AGP_STUB_TRACE=1 (default) the server's log gets one line per request
frame it receives.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from XPolicyLab.model_template import ModelTemplate

DEBUG_CHUNK = 2     # actions per get_action in the debug closed loop (as policy/demo_policy)


class _FrameTrace(logging.Filter):
    """Turn websockets' per-frame debug records into one decoded line per received request; drop the rest."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.msg == "< %s" and record.args:
                frame = record.args[0] if isinstance(record.args, tuple) else record.args
                data = getattr(frame, "data", None)
                if data is not None and getattr(getattr(frame, "opcode", None), "name", "") == "BINARY":
                    from client_server.ws.protocol.codec import decode_envelope

                    envelope = decode_envelope(bytes(data))
                    print(f"[AGP stub] <- {envelope.message_type.value} request_id={envelope.request_id} "
                          f"trial_id={envelope.trial_id}", flush=True)
        except Exception as exc:  # tracing must never disturb the server
            print(f"[AGP stub] frame trace unavailable: {type(exc).__name__}: {exc}", flush=True)
        return False


def _install_frame_trace() -> None:
    logger = logging.getLogger("websockets.server")
    if any(isinstance(f, _FrameTrace) for f in logger.filters):
        return
    logger.addFilter(_FrameTrace())
    logger.setLevel(logging.DEBUG)


class _DebugActions:
    """EVAL_ENV_TYPE=debug only: checks the observations and answers with hold-position joint actions."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        if model_cfg.get("action_type") != "joint":
            raise ValueError(f"AGP sends joint targets; action_type must be joint, got {model_cfg.get('action_type')!r}")
        info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
        if len(arm_dims) != len(ee_dims) or len(arm_dims) not in (1, 2):
            raise ValueError(f"unsupported robot action dimensions {info}")
        prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
        self.dims: dict[str, int] = {}
        for prefix, arm_dim, ee_dim in zip(prefixes, arm_dims, ee_dims):
            self.dims[f"{prefix}arm_joint_state"] = int(arm_dim)
            self.dims[f"{prefix}ee_joint_state"] = int(ee_dim)
        self.obs: dict[str, Any] | None = None
        self.observations = 0
        self.chunks = 0

    def update(self, obs: Any) -> None:
        import numpy as np

        if not isinstance(obs, dict):
            raise TypeError(f"observation must be a dict, got {type(obs).__name__}")
        cameras = {}
        for name, view in (obs.get("vision") or {}).items():
            color = view.get("color") if isinstance(view, dict) else None
            if not (isinstance(color, np.ndarray) and color.dtype == np.uint8 and color.ndim == 3
                    and color.shape[-1] == 3):
                raise ValueError(f"camera {name}: expected a decoded HxWx3 uint8 RGB array, got "
                                 f"{type(color).__name__} {getattr(color, 'shape', None)}")
            cameras[name] = tuple(color.shape)
        self.obs = obs
        self.observations += 1
        if self.observations == 1:
            print(f"[AGP stub] debug: first observation, cameras {cameras}, state keys "
                  f"{sorted((obs.get('state') or {}).keys())}", flush=True)

    def actions(self) -> list[dict[str, Any]]:
        import numpy as np

        if self.obs is None:
            raise RuntimeError("get_action before update_obs")
        state = self.obs.get("state") or {}
        hold = {}
        for key, dim in self.dims.items():
            value = np.asarray(state[key], dtype=np.float32).reshape(-1) if key in state else None
            if value is None or value.shape != (dim,):
                raise ValueError(f"observation state {key!r}: expected {dim} values, got "
                                 f"{None if value is None else value.shape}")
            hold[key] = value
        self.chunks += 1
        if self.chunks == 1:
            print(f"[AGP stub] debug: answering get_action with {DEBUG_CHUNK} joint actions holding the observed "
                  f"state, keys {dict(self.dims)}", flush=True)
        return [{key: value.copy() for key, value in hold.items()} for _ in range(DEBUG_CHUNK)]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.resets = 0
        self.debug = _DebugActions(model_cfg) if os.environ.get("EVAL_ENV_TYPE") == "debug" else None
        if os.environ.get("AGP_STUB_TRACE", "1") != "0":
            _install_frame_trace()
        print(f"[AGP stub] ready: policy={model_cfg.get('policy_name')} task={model_cfg.get('task_name')} "
              f"port={model_cfg.get('port')} (holds no model; the agent runs in the eval client's children)",
              flush=True)
        if self.debug is not None:
            print("[AGP stub] EVAL_ENV_TYPE=debug: XPolicyLab debug closed loop, get_action answers with joint "
                  f"actions {dict(self.debug.dims)}", flush=True)

    def reset(self) -> None:
        self.resets += 1
        print(f"[AGP stub] reset #{self.resets}", flush=True)

    def update_obs(self, obs: Any) -> None:
        if self.debug is not None:
            self.debug.update(obs)
        del obs

    def update_obs_batch(self, obs_list: Any) -> None:
        del obs_list

    def get_action(self) -> Any:
        if self.debug is not None:
            return self.debug.actions()
        raise RuntimeError("AGP serves no actions: the agent acts inside the RoboDojo eval client (deploy.py)")

    def get_action_batch(self, env_idx_list: Any = None) -> Any:
        del env_idx_list
        raise RuntimeError("AGP serves no actions and supports no batch evaluation (eval_batch: false)")
