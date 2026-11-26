import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from XPolicyLab.model_template import ModelTemplate

POLICY_DIR = Path(__file__).resolve().parent


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray, *, resize: bool = True) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if resize and image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if resize and image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized RGB shape (240, 320, 3), got {image.shape}")
    return image


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        if (
            self.action_type != "ee"
            or self.model_cfg.get("deployment_adapter", "robodojo_ee16") != "robodojo_ee16"
        ):
            raise ValueError("This adapter only supports RoboDojo EE16")
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 24)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.deployment_adapter = (
            str(self.model_cfg.get("deployment_adapter") or "robodojo_ee16").strip().lower()
        )
        if (
            "robodojo_ee16" in {"robodojo_ee16", "robodojo_delta_eef14"}
            and self.action_type != "ee"
        ):
            raise ValueError(f"{'robodojo_ee16'} requires action_type='ee'.")
        if (
            "robodojo_ee16" in {"robodojo_joint14", "robodojo_delta_joint14"}
            and self.action_type != "joint"
        ):
            raise ValueError(f"{'robodojo_ee16'} requires action_type='joint'.")
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.model = None
        self._request_context = {}
        self._virtual_env_ids = {}
        self._next_virtual_env_id = 0
        self._client_last_use = {}
        self._client_use_counter = 0
        self._max_retained_clients = int(self.model_cfg.get("max_retained_clients") or 64)
        self._last_infer_step = {}
        self.history_guard_duplicates = 0
        self.history_guard_mismatches = 0
        self._active_client = None
        self._seen_client_ids = False
        self._last_obs_by_client = {}
        self._last_instruction_by_client = {}
        self._last_virtual_by_client = {}
        self._infer_count_by_virtual = {}
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get(
            "ckpt_setting"
        )
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")
        if _is_none_like(checkpoint_path):
            raise FileNotFoundError(
                "RecoverWAM requires checkpoint_path/ckpt_setting for real deployment."
            )
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("RecoverWAM requires dataset_stats_path for real deployment.")
        from recoverwam.policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(checkpoint_path).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(dataset_stats_path).expanduser().resolve())
        self.model = get_model(upstream_cfg)
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)

    @staticmethod
    def _pack_robodojo_ee16_state(obs: dict) -> np.ndarray:
        state = obs["state"]
        parts = []
        for side in ("left", "right"):
            pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float32).reshape(7)
            gripper = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(1)
            parts.extend((pose, gripper))
        packed = np.concatenate(parts, axis=0)
        if packed.shape != (16,):
            raise ValueError(f"Expected RoboDojo EE state [16], got {packed.shape}")
        return packed

    @staticmethod
    def _unpack_robodojo_ee16_actions(actions: np.ndarray):
        packed = np.asarray(actions, dtype=np.float32)
        if packed.ndim == 1:
            packed = packed[None, :]
        if packed.ndim != 2 or packed.shape[1] != 16:
            raise ValueError(f"Expected RoboDojo EE actions [T,16], got {packed.shape}")
        result = []
        for action in packed:
            result.append(
                {
                    "left_ee_pose": action[0:7],
                    "left_ee_joint_state": action[7:8],
                    "right_ee_pose": action[8:15],
                    "right_ee_joint_state": action[15:16],
                }
            )
        return result

    def _encode_observation(self, obs: dict) -> dict:
        vision = obs["vision"]
        preserve_native_views = "robodojo_ee16" in {
            "robodojo_ee16",
            "robodojo_delta_eef14",
            "robodojo_joint14",
            "robodojo_delta_joint14",
        }
        adapted = {
            "env_idx": self._virtual_env_idx(int(obs.get("env_idx", 0))),
            "observation": {
                "head_camera": {
                    "rgb": _standardize_rgb(vision["cam_head"]["color"], resize=not True)
                },
                "left_camera": {
                    "rgb": _standardize_rgb(vision["cam_left_wrist"]["color"], resize=not True)
                },
                "right_camera": {
                    "rgb": _standardize_rgb(vision["cam_right_wrist"]["color"], resize=not True)
                },
            },
            "joint_action": {"vector": self._pack_robodojo_ee16_state(obs)},
        }
        return adapted

    CLIENT_ID_FIELD = "_recoverwam_client_id"

    def _bind_client(self, obs) -> None:
        client_id = obs.get(self.CLIENT_ID_FIELD) if isinstance(obs, dict) else None
        if client_id is None:
            self._active_client = None
            return
        self._seen_client_ids = True
        self._active_client = ("client", str(client_id))

    def _client_key(self):
        if getattr(self, "_active_client", None) is not None:
            return self._active_client
        context = getattr(self, "_request_context", None) or {}
        return (context.get("evaluation_id"), context.get("trial_id"))

    def _forget_virtual(self, virtual: int) -> None:
        self._last_infer_step.pop(virtual, None)
        self._infer_count_by_virtual.pop(virtual, None)
        if self.model is not None and hasattr(self.model, "reset_env"):
            self.model.reset_env(virtual)

    def _virtual_env_idx(self, env_idx: int) -> int:
        client = self._client_key()
        self._client_use_counter += 1
        self._client_last_use[client] = self._client_use_counter
        key = (client, int(env_idx))
        virtual = self._virtual_env_ids.get(key)
        if virtual is None:
            virtual = self._next_virtual_env_id
            self._next_virtual_env_id += 1
            self._virtual_env_ids[key] = virtual
            self._evict_stale_clients()
        return virtual

    def _evict_stale_clients(self) -> None:
        clients = {key[0] for key in self._virtual_env_ids}
        excess = len(clients) - self._max_retained_clients
        if excess <= 0:
            return
        current = self._client_key()
        stale = sorted(
            (c for c in clients if c != current), key=lambda c: self._client_last_use.get(c, 0)
        )[:excess]
        for client in stale:
            for key in [k for k in self._virtual_env_ids if k[0] == client]:
                self._forget_virtual(self._virtual_env_ids.pop(key))
            self._client_last_use.pop(client, None)
            self._last_obs_by_client.pop(client, None)
            self._last_instruction_by_client.pop(client, None)
            self._last_virtual_by_client.pop(client, None)

    def set_request_context(self, context):
        self._request_context = dict(context or {})
        if self.model is not None and hasattr(self.model, "set_request_context"):
            self.model.set_request_context(self._request_context)

    def _remember(self, encoded, raw_obs) -> None:
        client = self._client_key()
        self.last_obs = encoded
        self.last_instruction = _get_instruction(raw_obs, self.default_instruction)
        self._last_obs_by_client[client] = self.last_obs
        self._last_instruction_by_client[client] = self.last_instruction
        self._last_virtual_by_client[client] = encoded["env_idx"]

    def update_obs(self, obs):
        self._bind_client(obs)
        observation_history = obs.get("_recoverwam_observation_history")
        if observation_history:
            step = (self._request_context or {}).get("step")
            virtual = self._virtual_env_idx(int(obs.get("env_idx", 0)))
            if step is not None:
                last = self._last_infer_step.get(virtual)
                if last is not None and int(step) <= last:
                    self.history_guard_duplicates += 1
                    print(
                        f"[RecoverWAM][history-guard] duplicate infer step={step} last={last} client={self._client_key()} total_duplicates={self.history_guard_duplicates}",
                        flush=True,
                    )
                    current = observation_history[-1]
                    self._remember(self._encode_observation(current), current)
                    self._infer_count_by_virtual[virtual] = max(
                        0, self._infer_count_by_virtual.get(virtual, 0) - 1
                    )
                    return
                self._last_infer_step[virtual] = int(step)
            for history_obs in observation_history:
                if not isinstance(history_obs, dict):
                    continue
                encoded = self._encode_observation(history_obs)
                self._remember(encoded, history_obs)
                if self.model is not None and hasattr(self.model, "observe"):
                    self.model.observe(encoded)
            return
        encoded = self._encode_observation(obs)
        self._remember(encoded, obs)
        if self.model is not None and hasattr(self.model, "observe"):
            self.model.observe(encoded)

    def _check_history_alignment(self, virtual: int) -> None:
        """Before the k-th inference of an episode exactly replan_steps*k
        observations must have been recorded after the anchor (step 0)."""
        count = self._infer_count_by_virtual.get(virtual, 0)
        self._infer_count_by_virtual[virtual] = count + 1
        model = self.model
        if model is None or not getattr(model, "history_condition_frames", 0):
            return
        recorded = getattr(model, "_history_step_by_env", {}).get(virtual)
        expected = int(self.replan_steps) * count
        if recorded != expected:
            self.history_guard_mismatches += 1
            print(
                f"[RecoverWAM][history-guard] MISMATCH inference={count} recorded_obs_step={recorded} expected={expected} client={self._client_key()} total_mismatches={self.history_guard_mismatches}",
                flush=True,
            )

    def _infer_actions(self, obs, instruction):
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        action_chunk = action_chunk[:n_exec]
        return self._unpack_robodojo_ee16_actions(action_chunk)

    def get_action(self, obs=None):
        self._bind_client(obs)
        client = self._client_key()
        last_obs = self._last_obs_by_client.get(client, self.last_obs)
        instruction = self._last_instruction_by_client.get(client, self.last_instruction)
        virtual = self._last_virtual_by_client.get(client)
        if virtual is not None:
            self._check_history_alignment(virtual)
        return self._infer_actions(last_obs, instruction)

    def _reset_client_state(self, client) -> None:
        for key in [k for k in self._virtual_env_ids if k[0] == client]:
            self._forget_virtual(self._virtual_env_ids.pop(key))
        self._last_obs_by_client.pop(client, None)
        self._last_instruction_by_client.pop(client, None)
        self._last_virtual_by_client.pop(client, None)

    def reset_client(self, obs=None):
        """Episode reset for one identified client (upstream per-call protocol)."""
        self._bind_client(obs)
        client = self._client_key()
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self.model is not None and hasattr(self.model, "reset_env"):
            self._reset_client_state(client)
        elif self.model is not None:
            self.model.reset()

    def reset(self):
        self._active_client = None
        self.last_obs = None
        self.last_instruction = self.default_instruction
        client = self._client_key()
        if self.model is not None and client != (None, None) and hasattr(self.model, "reset_env"):
            self._reset_client_state(client)
            return
        if self._seen_client_ids:
            return
        self._virtual_env_ids.clear()
        self._last_infer_step.clear()
        self._infer_count_by_virtual.clear()
        self._last_obs_by_client.clear()
        self._last_instruction_by_client.clear()
        self._last_virtual_by_client.clear()
        if self.model is not None:
            self.model.reset()
