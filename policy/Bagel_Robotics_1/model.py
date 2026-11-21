import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .temporal_ensemble import TemporalEnsemble
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .assets import prepare_model_config
from .backend import configure_backend

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
        self.model_cfg = prepare_model_config(model_cfg, POLICY_DIR)
        self.backend_settings = configure_backend(self.model_cfg["seed"])
        info = get_robot_action_dim_info(self.model_cfg["env_cfg_type"])
        if len(info["arm_dim"]) != 2 or list(info["ee_dim"]) != [1, 1]:
            raise ValueError("The checkpoint requires two arms and one gripper value per arm")
        self.robot_action_dim_info = info
        self.action_dim = sum(7 + int(d) for d in info["ee_dim"])
        if self.action_dim != 16:
            raise ValueError("Checkpoint action width differs from the registered robot")
        self.action_type = self.model_cfg["action_type"]
        if (
            self.action_type != "ee"
            or self.model_cfg.get("deployment_adapter", "robodojo_ee16") != "robodojo_ee16"
        ):
            raise ValueError("This adapter only supports RoboDojo EE16")
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 16)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.deployment_adapter = (
            str(self.model_cfg.get("deployment_adapter") or "robodojo_ee16").strip().lower()
        )
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
        self._ensemble_by_virtual = {}
        self._last_ensemble = None
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get(
            "ckpt_setting"
        )
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")
        if _is_none_like(checkpoint_path):
            raise FileNotFoundError(
                "VPP2 requires checkpoint_path/ckpt_setting for real deployment."
            )
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("VPP2 requires dataset_stats_path for real deployment.")
        from bagel_cp1000.policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(checkpoint_path).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(dataset_stats_path).expanduser().resolve())
        self.model = get_model(upstream_cfg)
        video = self.model.model.video_expert
        if sum(p.numel() for p in video.parameters()) != 12468409664:
            raise ValueError("Loaded video expert is not the pinned compact CP1000 model")
        if any("mask_logits" in name for name, _ in video.named_parameters()):
            raise ValueError("Training mask parameters must not be present at inference")
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)
        if self.model_cfg.get("temporal_ensemble") != {"enabled": True, "new_weight": 0.9, "old_weight": 0.1}:
            raise ValueError("Expected the authorized 90/10 temporal ensemble")
        if (self.action_horizon, self.replan_steps) != (32, 16):
            raise ValueError("Expected horizon 32, execute 16")

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
        adapted = {
            "env_idx": self._virtual_env_idx(int(obs.get("env_idx", 0))),
            "observation": {
                "head_camera": {
                    "rgb": _standardize_rgb(vision["cam_head"]["color"], resize=False)
                },
                "left_camera": {
                    "rgb": _standardize_rgb(vision["cam_left_wrist"]["color"], resize=False)
                },
                "right_camera": {
                    "rgb": _standardize_rgb(vision["cam_right_wrist"]["color"], resize=False)
                },
            },
            "joint_action": {"vector": self._pack_robodojo_ee16_state(obs)},
        }
        return adapted

    CLIENT_ID_FIELD = "_vpp2_client_id"

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
        self._ensemble_by_virtual.pop(virtual, None)
        self._last_ensemble = None
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
        observation_history = obs.get("_vpp2_observation_history")
        if observation_history:
            step = (self._request_context or {}).get("step")
            virtual = self._virtual_env_idx(int(obs.get("env_idx", 0)))
            if step is not None:
                last = self._last_infer_step.get(virtual)
                if last is not None and int(step) <= last:
                    self.history_guard_duplicates += 1
                    print(
                        f"[VPP2][history-guard] duplicate infer step={step} last={last} client={self._client_key()} total_duplicates={self.history_guard_duplicates}",
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
                f"[VPP2][history-guard] MISMATCH inference={count} recorded_obs_step={recorded} expected={expected} client={self._client_key()} total_mismatches={self.history_guard_mismatches}",
                flush=True,
            )

    def _infer_actions(self, obs, instruction):
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        virtual = int(obs["env_idx"])
        ensemble = self._ensemble_by_virtual.setdefault(virtual, TemporalEnsemble())
        action_chunk = ensemble.apply(action_chunk)
        self._last_ensemble = ensemble.last_info
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
        self._request_context = {}
        self._seen_client_ids = False
        self._next_virtual_env_id = 0
        self._client_use_counter = 0
        self.last_obs = None
        self.last_instruction = self.default_instruction
        for state in (self._virtual_env_ids, self._client_last_use,
                      self._last_infer_step, self._infer_count_by_virtual,
                      self._ensemble_by_virtual, self._last_obs_by_client,
                      self._last_instruction_by_client, self._last_virtual_by_client):
            state.clear()
        self._last_ensemble = None
        self.history_guard_duplicates = 0
        self.history_guard_mismatches = 0
        if self.model is not None:
            self.model.reset()

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise NotImplementedError("CP1000 is qualified for one environment per instance")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise NotImplementedError("CP1000 is qualified for one environment per instance")
        return [self.get_action()]
