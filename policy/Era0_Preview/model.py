"""Era0-Preview RoboDojo inference adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runtime.hy_vla.modeling_xvla_adapter import HyXVLA
from .runtime.robodojo_eval.io import encode_obs
from .runtime.robodojo_eval.gripper_control import GripperCloseHold
from .runtime.robodojo_eval.io import _convert_pose_robo_dojo, action_chunk_to_robodojo_dicts
from .runtime.hy_vla.utils.transform_utils import convert_frame_robo_to_umi, convert_frame_umi_to_robo
from .runtime.robotwin_eval.transforms import (
    get_norm_data,
    pos_rotation_matrix_to_pos_quat,
    relative_to_dual_arm_poses,
)


def _bool_config(value: Any, name: str) -> bool:
    """Parse server config booleans without Python's truthy-string trap."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _pad_network_state(state: np.ndarray, state_dim: int = 32) -> np.ndarray:
    """Pad normalized 20-d PosRotMat proprioception as in training."""
    if state.ndim != 2 or state.shape[-1] > state_dim:
        raise ValueError(f"Expected [B,<={state_dim}] network state, got {state.shape}")
    if state.shape[-1] == state_dim:
        return state
    padded = np.zeros((state.shape[0], state_dim), dtype=state.dtype)
    padded[:, :state.shape[-1]] = state
    return padded


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_action_representation(value: Any) -> str | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    alias = normalized.lower()
    if alias in ("rel", "relative", "relative_chunk_ee_rt"):
        return "relative_chunk_ee_RT"
    if alias in ("abs", "absolute", "absolute_chunk_ee_rt"):
        return "absolute_chunk_ee_RT"
    raise ValueError(
        "action_representation must be relative_chunk_ee_RT or "
        f"absolute_chunk_ee_RT, got {value!r}"
    )


def _normalize_action_normalization(value: Any) -> str | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    alias = normalized.lower()
    if alias in ("zscore", "normalized"):
        return "zscore"
    if alias in ("none", "raw"):
        return "none"
    raise ValueError(f"action_normalization must be zscore or none, got {value!r}")


def _normalize_action_loss_mode(value: Any) -> str | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    alias = normalized.lower()
    if alias in ("flow_matching", "flow"):
        return "flow_matching"
    if alias == "regression":
        return "regression"
    raise ValueError(f"action_loss_mode must be flow_matching or regression, got {value!r}")


def _parse_history_per_view(value: Any, *, history_size: int, num_views: int = 3) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in ("none", "null"):
            return None
        stripped = stripped.strip("[]()")
        raw_values = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        raw_values = list(value)
    try:
        result = tuple(int(item) for item in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "img_history_size_per_view must be a 3-value integer list, "
            f"got {value!r}"
        ) from exc
    if len(result) != num_views:
        raise ValueError(
            f"img_history_size_per_view must contain {num_views} values, got {result}"
        )
    if any(item < 1 for item in result):
        raise ValueError(f"img_history_size_per_view entries must be >= 1, got {result}")
    if max(result) != int(history_size):
        raise ValueError(
            "img_history_size must equal max(img_history_size_per_view): "
            f"{history_size} != max({result})"
        )
    return result


from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .checkpoint import resolve_assets
from .obs_lag import ImageObservationLag, observation_lag, legacy_render_setting


class Model(ModelTemplate):
    """Model-server interface matching RoboDojo's websocket protocol."""

    def __init__(self, cfg: dict[str, Any]):
        cfg = dict(cfg)
        self.obs_lag = observation_lag()
        legacy_render_setting(self.obs_lag)
        self._image_lag = ImageObservationLag(self.obs_lag)
        print(f"[obs_delay] image lag={self.obs_lag} observation updates; state/language=current", flush=True)
        if cfg.get("action_type", "ee") != "ee" or cfg.get("env_cfg_type", "arx_x5") != "arx_x5":
            raise ValueError("This checkpoint supports RoboDojo arx_x5 with action_type=ee only")
        dims = get_robot_action_dim_info(cfg.get("env_cfg_type", "arx_x5"))
        if len(dims["arm_dim"]) != 2 or dims["ee_dim"] != [1, 1]:
            raise ValueError(f"Unsupported robot dimensions: {dims}")
        assets = resolve_assets(cfg)
        cfg.update(ckpt_path=str(assets["model"]), norm_path=str(assets["norm"]))
        self.ckpt_path = str(cfg["ckpt_path"])
        if not Path(self.ckpt_path).is_dir():
            raise FileNotFoundError(f"X-VLA checkpoint directory not found: {self.ckpt_path}")
        norm_path = cfg.get("norm_path") or os.environ.get("HY_VLA_NORM_PATH")
        if not norm_path or not Path(norm_path).is_file():
            raise FileNotFoundError(f"Normalization file not found: {norm_path!r}")
        expected_action_representation = _normalize_action_representation(
            cfg.get("action_representation")
        )
        expected_action_normalization = _normalize_action_normalization(
            cfg.get("action_normalization")
        )
        expected_action_loss_mode = _normalize_action_loss_mode(cfg.get("action_loss_mode"))
        if _bool_config(cfg.get("with_absolute", False), "with_absolute"):
            raise ValueError(
                "Hy_XVLA_RoboDojo supports exclusive relative or absolute "
                "25-token checkpoints; legacy rel+abs with_absolute=true is not supported."
            )
        self.blend_mode = str(cfg.get("blend_mode", "rel_only"))
        self.exc_action_size = int(cfg.get("exc_action_size", 10))
        self.exc_action_interval = int(cfg.get("exc_action_interval", 1))
        self.history_cadence = str(cfg.get("history_cadence", "chunk")).lower()
        self.img_history_size = int(cfg.get("img_history_size", 9))
        self.img_history_size_per_view = _parse_history_per_view(
            cfg.get("img_history_size_per_view"),
            history_size=self.img_history_size,
        )
        self.img_history_interval = int(cfg.get("img_history_interval", 40))
        self.inference_steps = int(cfg.get("inference_steps", 10))
        self.gripper_close_hold = GripperCloseHold(
            offset=cfg.get("gripper_close_offset", 0.0),
            motion_tolerance=cfg.get("gripper_motion_tolerance", 0.01),
        )
        if self.gripper_close_hold.offset:
            print(
                "[evaluation gripper postprocess] closing/holding raw target minus "
                f"{self.gripper_close_hold.offset:g}, floor=0; "
                f"opening restores raw target; motion tolerance={self.gripper_close_hold.motion_tolerance:g}; "
                "full-horizon preclose, executable opening releases; "
                "independent flags per environment/hand; observed state unchanged",
                flush=True,
            )
        self.preserve_padding_noise_inference = _bool_config(
            cfg.get("preserve_padding_noise_inference", False),
            "preserve_padding_noise_inference",
        )
        self.umi_coord_frame = _bool_config(cfg.get("umi_coord_frame", True), "umi_coord_frame")
        self.umi_gripper_space = _bool_config(cfg.get("umi_gripper_space", False), "umi_gripper_space")
        if self.umi_gripper_space and not self.umi_coord_frame:
            raise ValueError("umi_gripper_space=true requires umi_coord_frame=true")
        if self.exc_action_size < 1 or self.exc_action_interval < 1:
            raise ValueError("exc_action_size and exc_action_interval must both be positive")
        if self.history_cadence not in ("step", "chunk"):
            raise ValueError(f"history_cadence must be 'step' or 'chunk', got {self.history_cadence!r}")
        history_step_span = self.exc_action_interval
        if self.history_cadence == "chunk":
            history_step_span *= self.exc_action_size
        if self.img_history_interval % history_step_span != 0:
            raise ValueError(
                "img_history_interval must be divisible by the active history cadence span: "
                f"{self.img_history_interval} % {history_step_span} != 0"
            )
        self.norm_data = get_norm_data(str(norm_path))
        self.policy = HyXVLA.from_pretrained(
            self.ckpt_path,
            local_files_only=True,
            inference_steps=self.inference_steps,
            domain_id=int(cfg.get("domain_id", 0)),
        )
        self.action_representation = _normalize_action_representation(
            self.policy.action_representation
        )
        self.action_normalization = _normalize_action_normalization(
            self.policy.action_normalization
        )
        self.action_loss_mode = _normalize_action_loss_mode(
            getattr(self.policy.xvla, "action_loss_mode", getattr(self.policy.config, "action_loss_mode", ""))
        )
        if (
            expected_action_representation is not None
            and self.action_representation != expected_action_representation
        ):
            raise ValueError(
                "Evaluation action_representation conflicts with checkpoint: "
                f"{expected_action_representation!r} != {self.action_representation!r}"
            )
        if (
            expected_action_normalization is not None
            and self.action_normalization != expected_action_normalization
        ):
            raise ValueError(
                "Evaluation action_normalization conflicts with checkpoint: "
                f"{expected_action_normalization!r} != {self.action_normalization!r}"
            )
        if (
            expected_action_loss_mode is not None
            and self.action_loss_mode != expected_action_loss_mode
        ):
            raise ValueError(
                "Evaluation action_loss_mode conflicts with checkpoint: "
                f"{expected_action_loss_mode!r} != {self.action_loss_mode!r}"
            )
        if self.action_representation not in ("relative_chunk_ee_RT", "absolute_chunk_ee_RT"):
            raise ValueError(
                "Hy_XVLA_RoboDojo evaluation supports relative_chunk_ee_RT "
                "or absolute_chunk_ee_RT checkpoints; this checkpoint uses "
                f"{self.action_representation!r}."
            )
        if self.action_normalization not in ("zscore", "none"):
            raise ValueError(
                "Hy_XVLA_RoboDojo evaluation supports zscore or none action "
                f"normalization; this checkpoint uses {self.action_normalization!r}."
            )
        if self.action_normalization == "none" and self.action_representation != "absolute_chunk_ee_RT":
            raise ValueError("Raw action_normalization=none is supported only for absolute_chunk_ee_RT.")
        if self.action_normalization == "none" and self.action_loss_mode != "regression":
            raise ValueError("Raw absolute evaluation requires a regression checkpoint.")
        if self.action_representation == "relative_chunk_ee_RT" and self.blend_mode != "rel_only":
            raise ValueError("relative_chunk_ee_RT evaluation requires blend_mode=rel_only.")
        if self.action_representation == "absolute_chunk_ee_RT" and self.blend_mode != "abs_only":
            raise ValueError("absolute_chunk_ee_RT evaluation requires blend_mode=abs_only.")
        if self.action_representation == "absolute_chunk_ee_RT" and self.action_normalization == "zscore":
            if self.norm_data.get("act_mean_abs") is None or self.norm_data.get("act_std_abs") is None:
                raise ValueError(
                    f"absolute_chunk_ee_RT z-score evaluation requires action_mean_abs/std_abs in {norm_path!r}"
                )
        self.policy.preserve_padding_noise_inference = self.preserve_padding_noise_inference
        self.policy.to(torch.bfloat16).cuda().eval()
        # The model adapter owns the network itself; observation encoding,
        # coordinate conversion, and optional per-view history mirror the
        # Hy-VLA RoboDojo training contract.
        self.use_video_encoder = True
        self._obs_by_env: dict[int, dict[str, Any]] = {}
        self._top_by_env: dict[int, list[np.ndarray]] = {}
        self._left_by_env: dict[int, list[np.ndarray]] = {}
        self._right_by_env: dict[int, list[np.ndarray]] = {}
        self._latest_env_idx_list: list[int] = [0]

    def reset(self):
        self.policy.reset()
        self.gripper_close_hold.reset()
        self._image_lag.reset()
        self._obs_by_env.clear()
        self._top_by_env.clear()
        self._left_by_env.clear()
        self._right_by_env.clear()
        self._latest_env_idx_list = [0]
        return "X-VLA RoboDojo adapter reset"

    def update_obs(self, obs: dict[str, Any]):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]] | None = None, **kwargs):
        # Accept both the positional RPC value and the keyword RPC form.
        if obs_list is None:
            obs_list = kwargs.get("obs")
        if obs_list is None:
            raise ValueError("update_obs_batch requires an observation list")
        self._latest_env_idx_list = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        for env_idx, obs in zip(self._latest_env_idx_list, obs_list):
            batch = self._image_lag.apply(env_idx, encode_obs(obs))
            self._obs_by_env[env_idx] = batch
            if self.use_video_encoder and self.history_cadence == "step":
                self._append_history_frames(env_idx, batch)

    def _append_history_frames(self, env_idx: int, batch: dict[str, Any]) -> None:
        self._top_by_env.setdefault(env_idx, []).append(batch["raw_images.top_head"])
        self._left_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_left"])
        self._right_by_env.setdefault(env_idx, []).append(batch["raw_images.hand_right"])

    def _history(
        self,
        frames: list[np.ndarray],
        *,
        history_size: int | None = None,
    ) -> tuple[torch.Tensor, list[bool]]:
        if not frames:
            raise AssertionError("update_obs must be called before get_action")
        latest = len(frames) - 1
        history_step_span = self.exc_action_interval
        if self.history_cadence == "chunk":
            history_step_span *= self.exc_action_size
        history_interval = self.img_history_interval // history_step_span
        history_size = self.img_history_size if history_size is None else int(history_size)
        indices = [max(latest - (history_size - 1 - i) * history_interval, 0)
                   for i in range(history_size)]
        valid = [(latest - (history_size - 1 - i) * history_interval) >= 0
                 for i in range(history_size)]
        result = torch.from_numpy(np.stack([frames[idx] for idx in indices], axis=0))
        result = result.permute(0, 3, 1, 2).float().div(255.0)
        for i, is_valid in enumerate(valid):
            if not is_valid:
                result[i].zero_()
        if history_size < self.img_history_size:
            pad_size = self.img_history_size - history_size
            pad = torch.zeros((pad_size, *result.shape[1:]), dtype=result.dtype)
            result = torch.cat([pad, result], dim=0)
            valid = [False] * pad_size + valid
        return result.unsqueeze(0), valid

    @torch.no_grad()
    def get_action(self, **kwargs):
        if not self._obs_by_env:
            raise AssertionError("update_obs must be called before get_action")
        return self._infer_chunk(self._latest_env_idx_list[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_by_env:
            raise AssertionError("update_obs_batch must be called before get_action_batch")
        if env_idx_list is None:
            env_idx_list = kwargs.get("obs")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        return [self._infer_chunk(int(env_idx)) for env_idx in env_idx_list]

    @torch.no_grad()
    def _infer_chunk(self, env_idx: int):
        batch = self._obs_by_env[env_idx]
        if self.use_video_encoder and self.history_cadence == "chunk":
            self._append_history_frames(env_idx, batch)

        state16 = batch["observation.state"][0, :16]
        initial = state16.copy()
        initial[3:7] = state16[[4, 5, 6, 3]]
        initial[11:15] = state16[[12, 13, 14, 11]]
        if self.umi_coord_frame:
            initial = convert_frame_robo_to_umi(
                initial[None, :], convert_gripper=self.umi_gripper_space,
            )[0]
        net_state = _convert_pose_robo_dojo(
            state16, self.norm_data["qpos_mean"], self.norm_data["qpos_std"],
            umi_coord_frame=self.umi_coord_frame,
            umi_gripper_space=self.umi_gripper_space,
        )
        # RoboDojo training normalizes the 20-d PosRotMat state then pads it
        # to config.state_dim=32. HyXVLA consumes this padded representation.
        net_state = _pad_network_state(net_state, state_dim=32)
        net_batch = dict(batch)
        net_batch["observation.state"] = net_state
        if self.use_video_encoder:
            if self.img_history_size_per_view is None:
                net_batch["observation.images.top_head"] = self._history(
                    self._top_by_env[env_idx]
                )[0]
                net_batch["observation.images.hand_left"] = self._history(
                    self._left_by_env[env_idx]
                )[0]
                net_batch["observation.images.hand_right"] = self._history(
                    self._right_by_env[env_idx]
                )[0]
            else:
                head_k, left_k, right_k = self.img_history_size_per_view
                top_history, top_mask = self._history(
                    self._top_by_env[env_idx], history_size=head_k
                )
                left_history, left_mask = self._history(
                    self._left_by_env[env_idx], history_size=left_k
                )
                right_history, right_mask = self._history(
                    self._right_by_env[env_idx], history_size=right_k
                )
                net_batch["observation.images.top_head"] = top_history
                net_batch["observation.images.hand_left"] = left_history
                net_batch["observation.images.hand_right"] = right_history
                net_batch["observation.images.history_mask"] = torch.tensor(
                    [[top_mask, left_mask, right_mask]],
                    dtype=torch.bool,
                )
                net_batch["observation.images.history_size"] = torch.tensor(
                    [self.img_history_size_per_view],
                    dtype=torch.int32,
                )

        feed = {}
        for key, value in net_batch.items():
            if key.startswith("raw_images.") or key == "task":
                continue
            if isinstance(value, np.ndarray):
                feed[key] = torch.from_numpy(value).to(torch.bfloat16).cuda()
            elif isinstance(value, torch.Tensor):
                if key in (
                    "observation.images.history_mask",
                    "observation.images.history_size",
                ):
                    feed[key] = value.cuda()
                else:
                    feed[key] = value.to(torch.bfloat16).cuda()
            else:
                feed[key] = value
        feed["task"] = net_batch["task"]
        self.policy.reset()
        actions = [self.policy.select_action(feed)]
        while self.policy._action_queue:
            actions.append(self.policy._action_queue.popleft())
        action_array = torch.cat(actions, dim=0).float().cpu().numpy()
        required_actions = 1 + self.exc_action_size * self.exc_action_interval
        if action_array.shape[0] < required_actions:
            raise RuntimeError(
                f"Predicted action chunk has {action_array.shape[0]} slots, "
                f"but execution requires {required_actions} (including identity slot)."
            )
        decoded = self._decode_actions(action_array[:, :20], initial)
        if self.umi_coord_frame:
            decoded = convert_frame_umi_to_robo(
                decoded, convert_gripper=self.umi_gripper_space,
            )
        decoded[:, 3:7] = decoded[:, [6, 3, 4, 5]]
        decoded[:, 11:15] = decoded[:, [14, 11, 12, 13]]
        execution_indices = np.arange(1, required_actions, self.exc_action_interval)
        # Inspect every predicted non-anchor slot, including the unexecuted
        # tail, while persisting state only through selected execution slots.
        decoded[1:] = self.gripper_close_hold.apply(
            decoded[1:], state16[[7, 15]], env_idx,
            executable_indices=execution_indices - 1,
        )
        if self.exc_action_interval > 1:
            decoded = decoded[1:required_actions:self.exc_action_interval]
        else:
            decoded = decoded[1:required_actions]
        return action_chunk_to_robodojo_dicts(decoded)

    def _decode_actions(self, actions: np.ndarray, initial_xyzw: np.ndarray) -> np.ndarray:
        if self.action_representation == "relative_chunk_ee_RT":
            rel = actions * self.norm_data["act_std"] + self.norm_data["act_mean"]
            return relative_to_dual_arm_poses(rel, initial_xyzw)

        if self.action_normalization == "zscore":
            abs_actions = (
                actions * self.norm_data["act_std_abs"]
                + self.norm_data["act_mean_abs"]
            )
        else:
            abs_actions = actions
        decoded = np.zeros((abs_actions.shape[0], 16), dtype=abs_actions.dtype)
        for i in range(abs_actions.shape[0]):
            left = pos_rotation_matrix_to_pos_quat(abs_actions[i, :10])
            right = pos_rotation_matrix_to_pos_quat(abs_actions[i, 10:20])
            decoded[i] = np.concatenate([left, right])
        return decoded
