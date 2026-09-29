"""Checkpoint-aware tri-view-composite adapter between RoboDojo and CogWAM.

This module deliberately lives in the CogWAM repository. The RoboDojo checkout
supplies the simulator and the XPolicyLab transport only; its own
``policy/<name>`` directory is never imported.

The adapter is the stateful half of the evaluation stack: the policy server is
stateless, so every per-environment fact (the cached action chunk, the semantic
memory, the cached current subtask, the step counter) lives here and is cleared
by :meth:`Model.reset` at each episode boundary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

# cogwam.data.composite is deliberately dependency-light (numpy/torch/PIL/
# torchvision only) and cogwam.data.__init__ defers the dataset stack, so the
# simulator can import the exact compositor that produced the training pixels
# without dragging in PyAV/parquet/Accelerate.
from cogwam.data.composite import (
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
    build_per_view_images,
    build_tri_view_composite,
)

# Train/deploy ABI from cogwam/data/robodojo.py.
_EXPECTED_STATE_KEYS = [
    "state.left_joints",
    "state.left_gripper",
    "state.right_joints",
    "state.right_gripper",
]
_EXPECTED_ACTION_KEYS = [
    "action.left_joints",
    "action.left_gripper",
    "action.right_joints",
    "action.right_gripper",
]
# Private carrier for the raw per-camera frames between _convert_obs and
# update_obs_batch. Popped before the observation is cached, so it can never be
# serialized into an inference payload. Both the stitched composite and the
# per-camera planner stream are rebuilt from it on replan steps only.
_VLM_SOURCE_KEY = "__vlm_source_frames"

_REQUIRED_EVENT_FIELDS = {
    "memory",
    "cached_subtask",
    "decision",
    "memory_add",
    "cache_valid",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _semantic_memory_items(value: Any, empty_value: str = "None.") -> list[str]:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip(" \t.;")
    empty = re.sub(r"\s+", " ", str(empty_value)).strip(" \t.;").lower()
    if not normalized or normalized.lower() in {empty, "none", "nothing", "n/a"}:
        return []
    pieces = re.split(r"(?:\r?\n|\s*[;|]\s*|(?<=[.!?])\s+)", str(value).strip())
    return [
        re.sub(r"\s+", " ", piece).strip(" \t.;")
        for piece in pieces
        if re.sub(r"\s+", " ", piece).strip(" \t.;")
    ]


def _append_semantic_memory(previous: Any, delta: Any, empty_value: str) -> str:
    merged: list[str] = []
    seen = set()
    for value in (previous, delta):
        for item in _semantic_memory_items(value, empty_value):
            key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return empty_value if not merged else ". ".join(merged) + "."


def _decode_image(image: Any) -> np.ndarray:
    """Normalize an already-decoded RoboDojo observation to contiguous HWC uint8.

    The XPolicyLab policy server decodes every observation before forwarding it,
    so this only reshapes and casts — it never decodes. Encoded buffers are
    rejected rather than decoded here: stored image bits come in two byte
    formats, and only ``XPolicyLab.utils.process_data.decode_image_bit`` reads
    the marker that tells them apart, so a hand-rolled OpenCV decode plus
    channel swap is correct on at most one of them and silently reverses red and
    blue on the other. See XPolicyLab's AGENTS.md, "Decoding goes through the
    shared helpers".
    """

    if isinstance(image, (bytes, bytearray, memoryview)) or (
        isinstance(image, np.ndarray) and image.ndim == 1
    ):
        raise TypeError(
            "Received encoded image bits, but this adapter never decodes. The "
            "XPolicyLab policy server decodes observations before forwarding "
            "them; if you are calling this outside the server, decode with "
            "XPolicyLab.utils.process_data.decode_image_bit first."
        )
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def _extract_camera(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        camera = vision[name]
        if isinstance(camera, dict):
            for key in ("color", "rgb", "colors"):
                if key in camera:
                    return _decode_image(camera[key])
        else:
            return _decode_image(camera)
    raise KeyError(f"Missing RoboDojo camera; tried {list(names)}; available={list(vision)}")


def _instruction(observation: dict[str, Any], fallback: str) -> str:
    value = observation.get("task_instruction")
    if value is None:
        value = observation.get("instruction", observation.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is not None and hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _validate_checkpoint_selection(
    expected_checkpoint: Any,
    metadata: dict[str, Any],
) -> None:
    if not expected_checkpoint:
        return
    server_checkpoint = metadata.get("ckpt_path")
    if not server_checkpoint:
        raise RuntimeError(
            "CogWAM server metadata omitted ckpt_path; cannot verify that client and server selected the "
            "same checkpoint."
        )
    expected_path = Path(str(expected_checkpoint)).expanduser().resolve()
    server_path = Path(str(server_checkpoint)).expanduser().resolve()
    if server_path != expected_path:
        raise RuntimeError(f"RoboDojo client/server checkpoint mismatch: client={expected_path}, server={server_path}.")


def _validate_mot_runtime_contract(metadata: dict[str, Any]) -> None:
    """Validate the recipe actually instantiated by the causal-MoT server."""

    if metadata.get("framework_name") != "CogWAM":
        raise RuntimeError(
            "RoboDojo evaluation requires a CogWAM policy server, got "
            f"framework_name={metadata.get('framework_name')!r}."
        )

    planner_mask_contract = metadata.get("planner_query_mask_contract")
    if planner_mask_contract != "positional_suffix_v2":
        raise RuntimeError(
            "Loaded CogWAM server is running stale planner-mask code: "
            "expected planner_query_mask_contract='positional_suffix_v2', "
            f"got {planner_mask_contract!r}. Update and restart every CogWAM policy server before evaluation."
        )

    version = metadata.get("mot_contract_version")
    interaction_mode = str(metadata.get("mot_interaction_mode", "")).lower()
    if interaction_mode not in {"base", "joint"}:
        raise RuntimeError(f"Loaded causal-MoT server has an invalid interaction mode: {interaction_mode!r}.")

    actual = {
        "layerwise": _as_bool(metadata.get("mot_layerwise_planner_coupling", False)),
        "prediction_type": str(metadata.get("mot_action_prediction_type", "")).lower(),
        "velocity_target": str(metadata.get("mot_action_velocity_target", "")).lower(),
        "multires_world": _as_bool(metadata.get("mot_multires_world_input", False)),
    }
    if actual["velocity_target"] not in {"clean_minus_noise", "noise_minus_clean"}:
        raise RuntimeError(
            f"Loaded causal-MoT server has an invalid action velocity target: {actual['velocity_target']!r}."
        )
    saved_steps = int(metadata.get("mot_num_inference_timesteps", 0) or 0)

    if version == "legacy_shared_context_state_world_v1":
        if actual["prediction_type"] == "jit_x":
            # jit_x converts its clean-action prediction to scheduler velocity.
            expected = {
                "layerwise": False,
                "prediction_type": "jit_x",
                "multires_world": False,
            }
        else:
            expected = {
                "layerwise": False,
                "prediction_type": "velocity",
                "velocity_target": "noise_minus_clean",
                "multires_world": False,
            }
        if saved_steps not in {10, 20}:
            raise RuntimeError(f"Released causal-MoT checkpoints must save 10 or 20 inference steps, got {saved_steps}.")
    elif version == "layerwise_query_only_world_v2":
        expected = {
            "layerwise": True,
            "prediction_type": "velocity",
            "multires_world": False,
        }
        if saved_steps != 10:
            raise RuntimeError(
                f"Layer-wise causal-MoT checkpoints must train/save with exactly 10 inference steps, got {saved_steps}."
            )
    elif version == "layerwise_query_only_multires_world_v3":
        expected = {
            "layerwise": True,
            "prediction_type": "velocity",
            "multires_world": True,
        }
        if saved_steps != 10:
            raise RuntimeError(
                "Multi-resolution causal-MoT checkpoints must train/save with exactly 10 inference steps, "
                f"got {saved_steps}."
            )
    elif version == "legacy_planner_multires_world_v3":
        expected = {
            "layerwise": False,
            "prediction_type": "velocity",
            "velocity_target": "noise_minus_clean",
            "multires_world": True,
        }
        if saved_steps not in {10, 20}:
            raise RuntimeError(
                f"Legacy-planner multi-resolution causal-MoT checkpoints must save 10 or 20 inference steps, "
                f"got {saved_steps}."
            )
    else:
        raise RuntimeError(f"Loaded CogWAM server did not publish a supported checkpoint contract version: {version!r}.")

    mismatches = {
        key: {"loaded": actual[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Loaded causal-MoT train/inference recipe mismatch: {mismatches}.")
    if actual["multires_world"]:
        current_tokens = int(metadata.get("mot_current_dino_tokens", 0) or 0)
        future_tokens = int(metadata.get("mot_future_dino_tokens", 0) or 0)
        if min(current_tokens, future_tokens) <= 0 or current_tokens <= future_tokens:
            raise RuntimeError(
                "Multi-resolution causal-MoT requires the clean current token count to exceed the future "
                f"denoising token count, got current={current_tokens}, future={future_tokens}."
            )


class Model(ModelTemplate):
    """XPolicyLab policy facade backed by the active CogWAM websocket server."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type", "joint"))
        if self.action_type != "joint":
            raise ValueError("The RoboDojo CogWAM adapter requires action_type='joint'.")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("The RoboDojo CogWAM adapter requires env_cfg_type.")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        expected_action_dim = int(self.model_cfg.get("expected_action_dim", 14))
        # ARX X5 joint observations have a fixed 14-D proprioceptive ABI.
        self.state_dim = 14
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"RoboDojo {self.env_cfg_type} has action_dim={self.action_dim}; "
                f"checkpoint adapter expects {expected_action_dim}."
            )

        from cogwam.serve.protocol import WebsocketClientPolicy

        self._build_composite = build_tri_view_composite
        self._build_per_view_images = build_per_view_images

        self.client = WebsocketClientPolicy(
            str(self.model_cfg.get("policy_server_host", "127.0.0.1")),
            int(self.model_cfg.get("policy_server_port", 7777)),
        )
        metadata = self.client.get_server_metadata()
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid CogWAM server metadata: {metadata!r}")
        _validate_checkpoint_selection(self.model_cfg.get("expected_checkpoint_path"), metadata)
        _validate_mot_runtime_contract(metadata)

        # The composite geometry is a hard ABI, not an image-processing
        # suggestion: the checkpoint's DINO grid is derived from it.
        checkpoint_image_size = tuple(int(value) for value in metadata.get("obs_image_size") or ())
        self.image_size = tuple(int(value) for value in self.model_cfg.get("image_size", checkpoint_image_size))
        if len(checkpoint_image_size) != 2 or self.image_size != checkpoint_image_size:
            raise RuntimeError(
                f"Checkpoint composite size (width,height)={checkpoint_image_size!r}; deployment expects "
                f"{self.image_size!r}."
            )
        expected_layout = str(self.model_cfg.get("expected_image_layout", TRI_VIEW_COMPOSITE_LAYOUT))
        expected_key = str(self.model_cfg.get("expected_composite_view_key", TRI_VIEW_COMPOSITE_VIEW_KEY))
        checkpoint_composite_contract = (metadata.get("image_layout"), metadata.get("composite_view_key"))
        if checkpoint_composite_contract != (expected_layout, expected_key):
            raise RuntimeError(
                f"Checkpoint composite layout/key={checkpoint_composite_contract!r}; "
                f"expected {(expected_layout, expected_key)!r}."
            )
        # (width, height) per camera when the checkpoint's planner reads the
        # three cameras separately instead of the stitched composite. Empty
        # for every composite checkpoint, in which case nothing extra is sent.
        vlm_view_size = list(metadata.get("vlm_view_size") or [])
        if vlm_view_size and len(vlm_view_size) != 2:
            raise RuntimeError(f"Checkpoint vlm_view_size must be (width,height): {vlm_view_size!r}.")
        self.vlm_view_size = (int(vlm_view_size[0]), int(vlm_view_size[1])) if vlm_view_size else None
        if list(metadata.get("composite_source_view_keys") or []) != TRI_VIEW_COMPOSITE_SOURCE_VIEW_KEYS:
            raise RuntimeError(
                "Checkpoint composite camera order is not [head,left_wrist,right_wrist]: "
                f"{metadata.get('composite_source_view_keys')!r}."
            )

        self.action_chunk_size = int(metadata["action_chunk_size"])
        expected_chunk = int(self.model_cfg.get("expected_action_chunk_size", 25))
        if self.action_chunk_size != expected_chunk:
            raise RuntimeError(
                f"Checkpoint chunk={self.action_chunk_size}; RoboDojo experiment requires {expected_chunk}."
            )
        self.replan_interval = int(self.model_cfg.get("replan_interval", 10))
        if not 1 <= self.replan_interval <= self.action_chunk_size:
            raise ValueError(
                f"RoboDojo replan_interval must be in [1,{self.action_chunk_size}], got {self.replan_interval}."
            )
        self.rtc_enabled = _as_bool(self.model_cfg.get("rtc_enabled", False))
        self.rtc_overlap = self.action_chunk_size - self.replan_interval
        self.rtc_execution_horizon = int(self.model_cfg.get("rtc_execution_horizon", max(self.rtc_overlap, 1)))
        self.rtc_inference_delay = int(self.model_cfg.get("rtc_inference_delay", 1))
        self.rtc_max_guidance_weight = float(self.model_cfg.get("rtc_max_guidance_weight", 20.0))
        self.rtc_prefix_attention_schedule = (
            str(self.model_cfg.get("rtc_prefix_attention_schedule", "linear")).strip().lower()
        )
        self.rtc_debug_max_replans = int(self.model_cfg.get("rtc_debug_max_replans", 8))
        if self.rtc_debug_max_replans < 0:
            raise ValueError("rtc_debug_max_replans must be non-negative")
        if self.rtc_enabled:
            if not _as_bool(metadata.get("mot_rtc_supported", False)):
                raise RuntimeError("Selected checkpoint/server does not support causal-MoT RTC guidance")
            if self.rtc_overlap <= 0:
                raise ValueError(
                    "RTC requires replan_interval < action_chunk_size so the previous chunk has an unexecuted tail"
                )
            if not 1 <= self.rtc_execution_horizon <= self.rtc_overlap:
                raise ValueError(
                    f"rtc_execution_horizon must be in [1,{self.rtc_overlap}] for chunk={self.action_chunk_size}, "
                    f"replan={self.replan_interval}; got {self.rtc_execution_horizon}"
                )
            if not 0 <= self.rtc_inference_delay <= self.rtc_execution_horizon:
                raise ValueError(
                    f"rtc_inference_delay must be in [0,{self.rtc_execution_horizon}], got {self.rtc_inference_delay}"
                )
            if self.rtc_max_guidance_weight <= 0:
                raise ValueError("rtc_max_guidance_weight must be positive")
            if self.rtc_prefix_attention_schedule not in {"exp", "linear", "ones", "zeros"}:
                raise ValueError(
                    "rtc_prefix_attention_schedule must be exp, linear, ones, or zeros, got "
                    f"{self.rtc_prefix_attention_schedule!r}"
                )
        if "expects_state" not in metadata:
            raise RuntimeError("Selected checkpoint server did not declare its state-input contract.")
        self.expects_state = _as_bool(metadata["expects_state"])
        actual_action_keys = metadata.get("action_keys")
        if actual_action_keys is None or list(actual_action_keys) != _EXPECTED_ACTION_KEYS:
            raise RuntimeError(f"Checkpoint action_keys={actual_action_keys!r}; expected {_EXPECTED_ACTION_KEYS!r}.")
        if self.expects_state:
            actual_state_keys = metadata.get("state_keys")
            if actual_state_keys is None or list(actual_state_keys) != _EXPECTED_STATE_KEYS:
                raise RuntimeError(f"Checkpoint state_keys={actual_state_keys!r}; expected {_EXPECTED_STATE_KEYS!r}.")

        self.unnorm_key = str(self.model_cfg.get("unnorm_key", "new_embodiment"))
        available = list(metadata.get("available_unnorm_keys") or [])
        if self.unnorm_key not in available:
            raise RuntimeError(f"unnorm_key={self.unnorm_key!r} not available; checkpoint has {available}.")
        self.use_ddim = _as_bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        if not self.use_ddim or self.num_ddim_steps != 10:
            raise ValueError(
                "RoboDojo causal-MoT evaluation requires exactly 10 flow steps, got "
                f"use_ddim={self.use_ddim}, num_ddim_steps={self.num_ddim_steps}."
            )
        self.text_planning_enabled = _as_bool(metadata.get("text_planning_enabled", False))
        self.event_memory_enabled = _as_bool(metadata.get("event_memory_enabled", False))
        if self.event_memory_enabled and not self.text_planning_enabled:
            raise RuntimeError("Event-memory checkpoint must also declare text_planning_enabled")
        self.event_semantic_fields = dict(metadata.get("event_semantic_fields") or {})
        if self.event_memory_enabled and set(self.event_semantic_fields) != _REQUIRED_EVENT_FIELDS:
            raise RuntimeError(
                f"Event-memory server published an invalid semantic-field ABI: {self.event_semantic_fields!r}"
            )
        self.event_empty_memory = str(metadata.get("event_empty_memory", "None.")).strip()
        self.event_empty_cached_subtask = str(metadata.get("event_empty_cached_subtask", "None.")).strip()
        self.event_semantic_offset = int(metadata.get("event_semantic_offset", 0))
        self.event_replan_interval = int(metadata.get("event_replan_interval", 0))
        if self.event_memory_enabled and (not self.event_empty_memory or not self.event_empty_cached_subtask):
            raise RuntimeError("Event-memory empty sentinels must be non-empty")
        if self.event_memory_enabled and (
            self.event_replan_interval != self.replan_interval
            or self.event_semantic_offset != -self.replan_interval
        ):
            raise RuntimeError(
                "Event-memory train/eval cadence mismatch: checkpoint uses "
                f"semantic_offset={self.event_semantic_offset}, "
                f"replan_interval={self.event_replan_interval}, while the "
                f"client uses replan_interval={self.replan_interval}."
            )
        self.log_planner_text = _as_bool(self.model_cfg.get("log_planner_text", False))
        self.default_instruction = str(self.model_cfg.get("task_name") or "follow the instruction")
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.vlm_source_by_env: dict[int, list[np.ndarray]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.normalized_action_chunks_by_env: dict[int, np.ndarray] = {}
        self.rtc_debug_replans_by_env: dict[int, int] = {}
        self.planner_text_by_env: dict[int, str] = {}
        self.semantic_memory_by_env: dict[int, str] = {}
        self.cached_current_subtask_by_env: dict[int, str] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]
        checkpoint_path = Path(str(metadata.get("ckpt_path", "")))
        checkpoint_run = checkpoint_path.parents[1].name if len(checkpoint_path.parents) >= 2 else checkpoint_path.name
        rtc_summary = "disabled"
        if self.rtc_enabled:
            rtc_summary = (
                f"overlap{self.rtc_overlap}/h{self.rtc_execution_horizon}"
                f"/delay{self.rtc_inference_delay}/"
                f"{self.rtc_prefix_attention_schedule}"
                f"/w{self.rtc_max_guidance_weight:g}"
                f"/debug{self.rtc_debug_max_replans}"
            )
        print(
            "[CogWAM][RoboDojo] contract OK: "
            f"chunk={self.action_chunk_size}, replan={self.replan_interval}, "
            f"rtc={rtc_summary}, "
            f"text_planning={self.text_planning_enabled}, "
            f"event_memory={self.event_memory_enabled}, "
            f"eval_flow_steps={self.num_ddim_steps}, "
            f"saved_flow_steps={metadata.get('mot_num_inference_timesteps')}, "
            f"mot_contract={metadata.get('mot_contract_version')}, "
            f"learned_query_mask={metadata.get('planner_query_mask_contract')}, "
            f"qwen35_conv={metadata.get('qwen35_causal_conv1d_backend')}, "
            f"qwen35_fla={metadata.get('qwen35_fla_backend')}, "
            "qwen35_attn="
            f"{metadata.get('qwen_attn_implementation')}/"
            f"{metadata.get('qwen35_attn_implementation_source')}, "
            "dino_tokens="
            f"current{metadata.get('mot_current_dino_tokens')}/"
            f"future{metadata.get('mot_future_dino_tokens')}/"
            f"action{self.action_chunk_size}, "
            f"checkpoint={checkpoint_run}, "
            f"state={'14D/zscore' if self.expects_state else 'disabled'}, "
            f"image={metadata.get('image_layout')}-{self.image_size[0]}x{self.image_size[1]}"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        source_images = [
            _extract_camera(observation, ("cam_head", "cam_high", "head_camera")),
            _extract_camera(observation, ("cam_left_wrist", "left_camera")),
            _extract_camera(observation, ("cam_right_wrist", "right_camera")),
        ]
        converted = {
            "lang": _instruction(observation, self.default_instruction),
            # "image" is filled in _infer_chunks. This method runs on every
            # control step while the composite is consumed once per
            # replan_interval, so building it here threw ~90% of the work away.
            _VLM_SOURCE_KEY: source_images,
        }
        if self.expects_state:
            state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
            if state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            if state.ndim != 1 or state.shape[0] != self.state_dim:
                raise ValueError(
                    f"Expected {self.state_dim}-D [left_arm,left_gripper,right_arm,right_gripper] state, "
                    f"got {state.shape}."
                )
            # Raw state is intentional: PolicyServerWrapper applies the exact
            # training-time z-score transform exactly once.
            converted["state"] = state
        return converted

    def _composite_frame(self, source_images: list) -> np.ndarray:
        """Stitch the tri-view composite the DINO/world branch consumes.

        Called once per replan instead of once per control step: the simulator
        calls update_obs before every action, but the composite is only read
        when _infer_chunks assembles a request. Same builder, same frames, so
        the pixels are unchanged; only the timing moves.
        """

        composite = self._build_composite(source_images)
        if composite.size != self.image_size:
            raise ValueError(f"Composite size {composite.size} != expected {self.image_size}.")
        return np.asarray(composite, dtype=np.uint8).copy()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            converted = self._convert_obs(obs)
            source = converted.pop(_VLM_SOURCE_KEY)
            self.vlm_source_by_env[env_idx] = source
            self.obs_by_env[env_idx] = converted

    def _infer_chunks(self, env_idx_list: list[int]) -> dict[int, np.ndarray]:
        if not env_idx_list:
            return {}
        missing = [env_idx for env_idx in env_idx_list if env_idx not in self.obs_by_env]
        if missing:
            raise RuntimeError(f"update_obs must be called before get_action; missing envs={missing}.")
        examples = []
        for env_idx in env_idx_list:
            example = dict(self.obs_by_env[env_idx])
            source = self.vlm_source_by_env.get(env_idx)
            if source is None:
                raise RuntimeError(
                    f"missing per-camera frames for env {env_idx}; update_obs must run before get_action"
                )
            example["image"] = [self._composite_frame(source)]
            if self.vlm_view_size is not None:
                # Built here rather than in _convert_obs: only replan steps
                # reach this path, so the resize runs once per replan interval
                # instead of once per control step. Same function, same frames,
                # so the pixels the planner sees are unchanged.
                example["image_vlm"] = [
                    np.asarray(view, dtype=np.uint8).copy()
                    for view in self._build_per_view_images(source, self.vlm_view_size)
                ]
            if self.event_memory_enabled:
                fields = self.event_semantic_fields
                cache_valid = env_idx in self.cached_current_subtask_by_env
                example[fields["memory"]] = self.semantic_memory_by_env.get(env_idx, self.event_empty_memory)
                example[fields["cached_subtask"]] = self.cached_current_subtask_by_env.get(
                    env_idx, self.event_empty_cached_subtask
                )
                example[fields["cache_valid"]] = bool(cache_valid)
            examples.append(example)
        payload = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "unnorm_key": self.unnorm_key,
        }
        if self.rtc_enabled:
            previous = np.zeros(
                (len(env_idx_list), self.action_chunk_size, self.action_dim),
                dtype=np.float32,
            )
            prefix_lengths = np.zeros(len(env_idx_list), dtype=np.int64)
            normalized_cache = self.normalized_action_chunks_by_env
            for position, env_idx in enumerate(env_idx_list):
                old_chunk = normalized_cache.get(env_idx)
                if old_chunk is None:
                    continue
                old_chunk = np.asarray(old_chunk, dtype=np.float32)
                expected_old = (self.action_chunk_size, self.action_dim)
                if old_chunk.shape != expected_old:
                    raise ValueError(
                        f"Cached normalized RTC chunk for env {env_idx} has shape {old_chunk.shape}, "
                        f"expected {expected_old}"
                    )
                tail = old_chunk[self.replan_interval :]
                length = min(len(tail), self.rtc_execution_horizon)
                previous[position, :length] = tail[:length]
                prefix_lengths[position] = length
            payload.update(
                prev_action_chunk_normalized=previous,
                rtc_prefix_lengths=prefix_lengths,
                inference_delay=self.rtc_inference_delay,
                execution_horizon=self.rtc_execution_horizon,
                prefix_attention_schedule=self.rtc_prefix_attention_schedule,
                max_guidance_weight=self.rtc_max_guidance_weight,
            )
        response = self.client.predict_action(payload)
        if not response.get("ok", False):
            raise RuntimeError(f"CogWAM inference failed: {response.get('error', response)}")
        response_data = response["data"]
        chunks = np.asarray(response_data["actions"], dtype=np.float32)
        expected = (len(env_idx_list), self.action_chunk_size, self.action_dim)
        if chunks.shape != expected:
            raise ValueError(f"Expected unnormalized action chunks {expected}, got {chunks.shape}.")
        if self.rtc_enabled:
            raw_normalized = response_data.get("normalized_actions")
            if raw_normalized is None:
                raise ValueError("RTC requires normalized_actions in the policy-server response")
            normalized_chunks = np.asarray(raw_normalized, dtype=np.float32)
            if normalized_chunks.shape != expected:
                raise ValueError(
                    f"RTC requires normalized_actions aligned with actions; expected {expected}, "
                    f"got {normalized_chunks.shape}"
                )
            normalized_cache = self.normalized_action_chunks_by_env
            debug_counts = self.rtc_debug_replans_by_env
            debug_limit = int(self.rtc_debug_max_replans)
            for position, env_idx in enumerate(env_idx_list):
                old_chunk = normalized_cache.get(env_idx)
                prefix_length = int(prefix_lengths[position])
                debug_count = debug_counts.get(env_idx, 0)
                if old_chunk is not None and prefix_length > 0 and debug_count < debug_limit:
                    old_chunk = np.asarray(old_chunk, dtype=np.float32)
                    new_chunk = normalized_chunks[position]
                    old_tail = old_chunk[self.replan_interval : self.replan_interval + prefix_length]
                    previous_action = old_chunk[self.replan_interval - 1]

                    def _rmse(delta: np.ndarray) -> float:
                        return float(np.sqrt(np.mean(np.square(delta))))

                    planned_step_rmse = _rmse(old_tail[0] - previous_action)
                    new_switch_rmse = _rmse(new_chunk[0] - previous_action)
                    rtc_target_rmse = _rmse(new_chunk[0] - old_tail[0])
                    prefix_rmse = _rmse(new_chunk[:prefix_length] - old_tail)
                    print(
                        "[CogWAM][RoboDojo][RTC] "
                        f"env={env_idx} replan={debug_count + 1} "
                        f"planned_step_rmse={planned_step_rmse:.6f} "
                        f"new_switch_rmse={new_switch_rmse:.6f} "
                        f"rtc_target_rmse={rtc_target_rmse:.6f} "
                        f"prefix_rmse={prefix_rmse:.6f}",
                        flush=True,
                    )
                    debug_counts[env_idx] = debug_count + 1
                normalized_cache[env_idx] = normalized_chunks[position].copy()

        planner_texts = response_data.get("planner_text")
        if self.event_memory_enabled:
            decisions = response_data.get("semantic_decision")
            memory_adds = response_data.get("semantic_memory_add")
            current_subtasks = response_data.get("semantic_current_subtask")
            if not all(
                isinstance(values, (list, tuple)) and len(values) == len(env_idx_list)
                for values in (decisions, memory_adds, current_subtasks)
            ):
                raise RuntimeError(
                    "Event-memory response fields must be aligned lists: "
                    f"decisions={decisions!r}, memory_adds={memory_adds!r}, current_subtasks={current_subtasks!r}"
                )
            if planner_texts is not None and isinstance(planner_texts, str):
                planner_texts = [planner_texts]
            for position, env_idx in enumerate(env_idx_list):
                decision = str(decisions[position]).strip().upper()
                if decision == "KEEP":
                    if memory_adds[position] is not None or current_subtasks[position] is not None:
                        raise RuntimeError("KEEP must not carry a memory delta or replacement subtask")
                elif decision == "UPDATE":
                    memory_add = str(memory_adds[position] or "").strip()
                    current_subtask = str(current_subtasks[position] or "").strip()
                    if not memory_add or not current_subtask:
                        raise RuntimeError("UPDATE must carry non-empty Memory Add and Current Subtask")
                    previous = self.semantic_memory_by_env.get(env_idx, self.event_empty_memory)
                    self.semantic_memory_by_env[env_idx] = _append_semantic_memory(
                        previous, memory_add, self.event_empty_memory
                    )
                    self.cached_current_subtask_by_env[env_idx] = current_subtask
                else:
                    raise RuntimeError(f"Unknown semantic decision for env {env_idx}: {decision!r}")
                if planner_texts is not None:
                    if len(planner_texts) != len(env_idx_list):
                        raise RuntimeError("Event planner_text must align with the request batch")
                    self.planner_text_by_env[env_idx] = str(planner_texts[position]).strip()
                if self.log_planner_text:
                    print(
                        "[CogWAM][RoboDojo] "
                        f"env={env_idx} semantic_decision={decision} "
                        f"memory={self.semantic_memory_by_env.get(env_idx, self.event_empty_memory)!r} "
                        f"subtask={self.cached_current_subtask_by_env.get(env_idx, self.event_empty_cached_subtask)!r}",
                        flush=True,
                    )
        return {env_idx: np.asarray(chunks[position], dtype=np.float32) for position, env_idx in enumerate(env_idx_list)}

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = [int(env_idx) for env_idx in (env_idx_list or self._latest_env_idx_list)]
        replan_envs = []
        for env_idx in env_idx_list:
            step = self.step_by_env.get(env_idx, 0)
            if env_idx not in self.action_chunks_by_env or step % self.replan_interval == 0:
                replan_envs.append(env_idx)
        if replan_envs:
            self.action_chunks_by_env.update(self._infer_chunks(replan_envs))

        actions = []
        for env_idx in env_idx_list:
            step = self.step_by_env.get(env_idx, 0)
            # Only the first ``replan_interval`` actions of each freshly
            # predicted chunk are ever executed; the remaining tail is the RTC
            # consistency overlap and is discarded when RTC is off.
            chunk_offset = step % self.replan_interval
            action = np.asarray(self.action_chunks_by_env[env_idx][chunk_offset], dtype=np.float32)
            self.step_by_env[env_idx] = step + 1
            actions.append(
                [
                    unpack_robot_state(
                        action,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                ]
            )
        return actions

    def reset(self):
        self.obs_by_env.clear()
        self.vlm_source_by_env.clear()
        self.action_chunks_by_env.clear()
        self.normalized_action_chunks_by_env.clear()
        self.rtc_debug_replans_by_env.clear()
        self.planner_text_by_env.clear()
        self.semantic_memory_by_env.clear()
        self.cached_current_subtask_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]


__all__ = ["Model"]
