from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.WorldScape_Policy_2.xpolicylab_adapter.history_window import match_memory_task
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_ENV_CFG = "arx_x5"
SUPPORTED_ACTION_TYPE = "joint"
CHECKPOINT_EMBODIMENT = "arx_x5_joint"
CAMERA_ORDER = ("cam_head", "cam_right_wrist", "cam_left_wrist")
SUPPORTED_INPUT_COLOR_ORDERS = {"rgb", "bgr"}


def _boolean_config(value: Any, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _wam_history_mode_config(config: Mapping[str, Any]) -> str:
    mode = config.get("wam_history_mode")
    if mode is None:
        mode = "periodic"
    if mode not in ("periodic", "rolling"):
        raise ValueError("wam_history_mode must be periodic or rolling")
    return mode


def _integer_config(value: Any, *, name: str, default: int, minimum: int) -> int:
    resolved = default if value is None else value
    if isinstance(resolved, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    try:
        integer = int(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}") from exc
    if integer != resolved or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return integer


def _cat_optional_tensors(values: list[torch.Tensor | None], *, name: str) -> torch.Tensor | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Cannot batch mixed presence for {name}")
    return torch.cat(values, dim=0)  # type: ignore[arg-type]


def _merge_optional_lists(values: list[list[Any] | None], *, name: str) -> list[Any] | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Cannot batch mixed presence for {name}")
    return [item for value in values for item in value]  # type: ignore[union-attr]


def _stack_observation_batches(observations: list[Any]) -> Any:
    if not observations:
        raise ValueError("Cannot stack an empty observation batch")
    cls = type(observations[0])
    if any(type(observation) is not cls for observation in observations):
        raise TypeError("All policy observations must have the same type")
    return cls(
        images=torch.cat([value.images for value in observations], dim=0),
        head_view=torch.cat([value.head_view for value in observations], dim=0),
        proprioception=torch.cat([value.proprioception for value in observations], dim=0),
        embodiment_id=torch.cat([value.embodiment_id for value in observations], dim=0),
        vlm_history_images=_cat_optional_tensors(
            [value.vlm_history_images for value in observations],
            name="vlm_history_images",
        ),
        vlm_history_mask=_cat_optional_tensors(
            [value.vlm_history_mask for value in observations],
            name="vlm_history_mask",
        ),
    )


def _stack_prompt_batches(prompts: list[Any]) -> Any:
    if not prompts:
        raise ValueError("Cannot stack an empty prompt batch")
    cls = type(prompts[0])
    if any(type(prompt) is not cls for prompt in prompts):
        raise TypeError("All policy prompts must have the same type")
    return cls(
        vlm_planning_text=_merge_optional_lists(
            [value.vlm_planning_text for value in prompts], name="vlm_planning_text"
        ),
        language_instruction=_merge_optional_lists(
            [value.language_instruction for value in prompts], name="language_instruction"
        ),
        negative_vlm_text=_merge_optional_lists(
            [value.negative_vlm_text for value in prompts], name="negative_vlm_text"
        ),
        negative_language_instruction=_merge_optional_lists(
            [value.negative_language_instruction for value in prompts],
            name="negative_language_instruction",
        ),
        planning_labels_text=_merge_optional_lists(
            [value.planning_labels_text for value in prompts], name="planning_labels_text"
        ),
        visual_prompt="none",
    )


def _select_signature(
    signature: tuple[str, ...] | None,
    positions: list[int],
    previous_batch_size: int,
) -> tuple[str, ...] | None:
    if signature is None:
        return None
    if len(signature) % previous_batch_size:
        raise ValueError(
            f"Prompt signature length is not divisible by its batch size: {len(signature)} vs {previous_batch_size}"
        )
    selected: list[str] = []
    for offset in range(0, len(signature), previous_batch_size):
        block = signature[offset : offset + previous_batch_size]
        selected.extend(block[position] for position in positions)
    return tuple(selected)


def _select_tensor_batch(
    value: torch.Tensor,
    positions: list[int],
    previous_batch_size: int,
    *,
    batch_dim: int,
) -> torch.Tensor:
    if value.ndim <= batch_dim or value.shape[batch_dim] != previous_batch_size:
        raise ValueError(
            "Causal-state tensor has an unexpected batch dimension: "
            f"shape={tuple(value.shape)}, dim={batch_dim}, expected={previous_batch_size}"
        )
    index = torch.tensor(positions, device=value.device, dtype=torch.long)
    return value.index_select(batch_dim, index)


def _select_runtime_cache(value: Any, positions: list[int], previous_batch_size: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _select_tensor_batch(value, positions, previous_batch_size, batch_dim=1)
    if isinstance(value, list):
        return [_select_runtime_cache(item, positions, previous_batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_runtime_cache(item, positions, previous_batch_size) for item in value)
    if isinstance(value, dict):
        return {key: _select_runtime_cache(item, positions, previous_batch_size) for key, item in value.items()}
    raise TypeError(f"Unsupported WAM runtime-cache value: {type(value).__name__}")


def _select_causal_state(value: Any, positions: list[int], previous_batch_size: int) -> Any:
    """Select active environments from WorldScape's explicit causal state."""

    if value is None:
        return None
    if not is_dataclass(value):
        raise TypeError(f"Unsupported causal-state value: {type(value).__name__}")
    updates = {}
    for state_field in fields(value):
        item = getattr(value, state_field.name)
        if state_field.name == "prompt_signature":
            updates[state_field.name] = _select_signature(item, positions, previous_batch_size)
        elif state_field.name in {
            "positive_kv_cache",
            "negative_kv_cache",
            "positive_cross_attention_cache",
            "negative_cross_attention_cache",
        }:
            updates[state_field.name] = _select_runtime_cache(item, positions, previous_batch_size)
        elif torch.is_tensor(item):
            updates[state_field.name] = _select_tensor_batch(item, positions, previous_batch_size, batch_dim=0)
        elif is_dataclass(item):
            updates[state_field.name] = _select_causal_state(item, positions, previous_batch_size)
        else:
            updates[state_field.name] = item
    return replace(value, **updates)


def _instruction(obs: Mapping[str, Any], fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions", fallback))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _input_color_order_config(model_cfg: Mapping[str, Any]) -> str:
    resolved = str(model_cfg.get("input_color_order") or "rgb").strip().lower()
    if resolved not in SUPPORTED_INPUT_COLOR_ORDERS:
        raise ValueError(
            f"input_color_order must be one of {sorted(SUPPORTED_INPUT_COLOR_ORDERS)}, "
            f"got {resolved!r}"
        )
    return resolved


def _t5_prompt_template_config(model_cfg: Mapping[str, Any]) -> str | None:
    """Use raw instructions unless a checkpoint-specific template is explicit."""

    template = model_cfg.get("t5_prompt_template")
    if template is None or template == "":
        return None
    if not isinstance(template, str):
        raise TypeError("t5_prompt_template must be a string or null")
    if "{instruction}" not in template:
        raise ValueError("t5_prompt_template must contain {instruction}")
    return template


def _format_policy_instruction(
    instruction: str, *, mode: Any, t5_prompt_template: str | None = None
) -> str:
    """Preserve raw Interactive text; apply only explicitly requested templates."""

    mode_value = getattr(mode, "value", mode)
    if str(mode_value).strip().lower() != "interactive" or not t5_prompt_template:
        return instruction
    return t5_prompt_template.format(instruction=instruction.lower())


def _rgb_camera(obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
    vision = obs.get("vision")
    if not isinstance(vision, Mapping) or camera_name not in vision:
        available = sorted(vision) if isinstance(vision, Mapping) else []
        raise KeyError(f"Missing camera {camera_name!r}; available cameras: {available}")
    camera = vision[camera_name]
    if not isinstance(camera, Mapping) or "color" not in camera:
        raise KeyError(f"Camera {camera_name!r} is missing decoded 'color'")
    image = np.asarray(camera["color"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera {camera_name!r} must be HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"Camera {camera_name!r} must be uint8 RGB, got {image.dtype}")
    return np.ascontiguousarray(image)


def _resize_policy_image(
    image: np.ndarray, *, width: int, height: int
) -> np.ndarray:
    """Match the post-training cache downsampling interpolation exactly."""

    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _resize_policy_camera(
    obs: Mapping[str, Any],
    camera_name: str,
    *,
    input_color_order: str,
    resize_image: Any,
) -> np.ndarray:
    """Apply the checkpoint color order, then the native evaluation resize."""

    image = _rgb_camera(obs, camera_name)
    if input_color_order == "bgr":
        image = np.ascontiguousarray(image[..., ::-1])
    resized = np.asarray(resize_image(image))
    if resized.ndim != 3 or resized.shape[-1] != 3:
        raise ValueError(f"Resized camera {camera_name!r} must be HWC RGB/BGR, got {resized.shape}")
    if resized.dtype != np.uint8:
        raise TypeError(f"Resized camera {camera_name!r} must be uint8, got {resized.dtype}")
    return np.ascontiguousarray(resized)


def _compose_diffusion_history(
    observations: list[Mapping[str, Any]], *, input_color_order: str,
    view_order: tuple[int, ...], output_size: tuple[int, int],
) -> np.ndarray:
    """Match training's CPU resize, uint8 rounding and camera placement.

    Compose at most eight native frames at once, as the training transform does.
    Keep the small canvas in rolling history instead of retaining native views.
    """
    from worldscape_policy.visual_mosaic import compose_three_view_robotwin

    parts = []
    for start in range(0, len(observations), 8):
        frames = np.stack([
            np.stack([_rgb_camera(obs, CAMERA_ORDER[index]) for index in view_order])
            for obs in observations[start:start + 8]
        ])
        if input_color_order == "bgr":
            frames = np.ascontiguousarray(frames[..., ::-1])
        views = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).unsqueeze(0)
        canvas = compose_three_view_robotwin(views, output_size=output_size)
        parts.append(canvas.clamp_(0, 255).round().to(torch.uint8))
    return torch.cat(parts, dim=1).squeeze(0).permute(0, 2, 3, 1).contiguous().numpy()


def _sample_observation_history(
    observations: list[Mapping[str, Any]], *, segments: int = 8, enabled: bool = True
) -> list[Mapping[str, Any]]:
    """Retain the starting observation and segment endpoints at native cadence."""

    if not observations:
        raise ValueError("Cannot sample an empty observation history")
    if not enabled:
        return [observations[-1]]
    if segments < 1:
        raise ValueError(f"observation history segments must be positive, got {segments}")
    horizon = len(observations) - 1
    indices = np.linspace(0, horizon, min(horizon, segments) + 1, dtype=np.int64)
    return [observations[index] for index in indices]


def _state_fields(
    obs: Mapping[str, Any],
    *,
    robot_action_dim_info: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    state = obs.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("Observation is missing state")
    arm_dims = list(robot_action_dim_info["arm_dim"])
    gripper_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != 2 or len(gripper_dims) != 2:
        raise ValueError("WorldScape_Policy_2 joint mode requires two arms and two grippers")
    result: dict[str, np.ndarray] = {}
    for side, arm_dim, gripper_dim in zip(
        ("left", "right"), arm_dims, gripper_dims, strict=True
    ):
        joint_key = f"{side}_arm_joint_state"
        gripper_key = f"{side}_ee_joint_state"
        if joint_key not in state or gripper_key not in state:
            raise KeyError(
                f"Joint state must contain {joint_key!r} and {gripper_key!r}"
            )
        joint = np.asarray(state[joint_key], dtype=np.float32).reshape(-1)
        gripper = np.asarray(state[gripper_key], dtype=np.float32).reshape(-1)
        if joint.shape != (int(arm_dim),) or not np.isfinite(joint).all():
            raise ValueError(
                f"{joint_key} must contain {arm_dim} finite values, got {joint}"
            )
        if gripper.shape != (int(gripper_dim),) or not np.isfinite(gripper).all():
            raise ValueError(
                f"{gripper_key} must contain {gripper_dim} finite values, got {gripper}"
            )
        result[f"state.{side}_joint"] = joint.reshape(1, int(arm_dim))
        result[f"state.{side}_gripper"] = gripper.reshape(1, int(gripper_dim))
    return result


def _relative_action_keys(transform: Any) -> set[str]:
    keys = {str(key) for key in getattr(transform, "relative_action_keys", ())}
    metadata = getattr(transform, "modality_metadata", None)
    if isinstance(metadata, Mapping):
        keys.update(
            str(key)
            for key, value in metadata.items()
            if str(key).startswith("action.") and getattr(value, "absolute", True) is False
        )
    for child in getattr(transform, "transforms", ()):
        keys.update(_relative_action_keys(child))
    return keys


def _action_fields(
    output: Any,
    bundle: Any,
    current_state: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    action = output.require_action().detach().float().cpu()
    if action.ndim == 3:
        if action.shape[0] != 1:
            raise ValueError(f"WorldScape_Policy_2 requires batch size 1, got {action.shape}")
        action = action[0]
    if action.ndim != 2:
        raise ValueError(f"Expected action [H,D], got {tuple(action.shape)}")
    transformed = bundle.transform.unapply({"action": action})
    fields = {
        str(key): np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
        for key, value in transformed.items()
        if str(key).startswith("action.")
    }
    required = {
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise KeyError(f"Checkpoint action transform is missing fields: {missing}")
    for key in _relative_action_keys(bundle.transform):
        if key not in fields:
            continue
        state_key = f"state.{key.removeprefix('action.')}"
        if state_key not in current_state:
            raise KeyError(f"Relative action {key!r} requires {state_key!r}")
        reference = np.asarray(current_state[state_key], dtype=np.float32).reshape(-1)
        value = np.asarray(fields[key], dtype=np.float32)
        if value.shape[-1] != reference.shape[-1]:
            raise ValueError(
                f"Relative action/state dimensions differ for {key}: {value.shape[-1]} != {reference.shape[-1]}"
            )
        fields[key] = value + reference
    return fields


def _fields_to_robodojo_actions(
    fields: Mapping[str, np.ndarray],
    *,
    action_steps: int,
    robot_action_dim_info: Mapping[str, Any],
) -> list[dict[str, np.ndarray]]:
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
    arm_dims = list(robot_action_dim_info["arm_dim"])
    gripper_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != 2 or len(gripper_dims) != 2:
        raise ValueError("WorldScape_Policy_2 joint mode requires two arms and two grippers")
    keys = (
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    )
    missing = sorted(set(keys).difference(arrays))
    if missing:
        raise KeyError(f"Checkpoint action transform is missing fields: {missing}")
    horizon = min(action_steps, *(arrays[key].shape[0] for key in keys))
    if horizon < 1:
        raise ValueError("WorldScape_Policy_2 returned an empty action chunk")
    for key, width in zip(
        keys,
        (arm_dims[0], gripper_dims[0], arm_dims[1], gripper_dims[1]),
        strict=True,
    ):
        if arrays[key].ndim != 2 or arrays[key].shape[1] != int(width):
            raise ValueError(
                f"{key} must have shape [H,{width}], got {arrays[key].shape}"
            )
    return [
        {
            "left_arm_joint_state": arrays["action.left_joint"][step].copy(),
            "left_ee_joint_state": np.clip(
                arrays["action.left_gripper"][step], 0.0, 1.0
            ).astype(np.float32, copy=False),
            "right_arm_joint_state": arrays["action.right_joint"][step].copy(),
            "right_ee_joint_state": np.clip(
                arrays["action.right_gripper"][step], 0.0, 1.0
            ).astype(np.float32, copy=False),
        }
        for step in range(horizon)
    ]


def _validate_joint_transform_bundle(
    bundle: Any, robot_action_dim_info: Mapping[str, Any]
) -> None:
    """Require the 14-D position/gripper training layout."""

    arm_dims = [int(value) for value in robot_action_dim_info["arm_dim"]]
    gripper_dims = [int(value) for value in robot_action_dim_info["ee_dim"]]
    if arm_dims != [6, 6] or gripper_dims != [1, 1]:
        raise ValueError(
            "WorldScape_Policy_2 arx_x5 joint checkpoints require "
            f"arm_dim=[6, 6] and ee_dim=[1, 1], got {robot_action_dim_info}"
        )
    embodiment = getattr(bundle.transform, "embodiment", None)
    state_fields = tuple(getattr(embodiment, "state_fields", ()))
    action_fields = tuple(getattr(embodiment, "action_fields", ()))
    expected_state = (
        ("state.left_joint", 6, True),
        ("state.left_gripper", 1, True),
        ("state.right_joint", 6, True),
        ("state.right_gripper", 1, True),
    )
    # Native RoboDojo exports use one relative_joint_action flag for both arms.
    # State and grippers remain absolute in either representation.
    expected_action_layouts = tuple(
        (
            ("action.left_joint", 6, absolute),
            ("action.left_gripper", 1, True),
            ("action.right_joint", 6, absolute),
            ("action.right_gripper", 1, True),
        )
        for absolute in (False, True)
    )

    def signature(fields: tuple[Any, ...]) -> tuple[tuple[str, int, bool], ...]:
        return tuple(
            (str(field.key), int(field.size), bool(field.absolute)) for field in fields
        )

    if signature(state_fields) != expected_state:
        raise ValueError(
            "Joint checkpoint state layout does not match training: "
            f"expected {expected_state}, got {signature(state_fields)}"
        )
    if signature(action_fields) not in expected_action_layouts:
        raise ValueError(
            "Joint checkpoint action layout does not match training: "
            "expected both arms relative or both absolute, with absolute grippers "
            "and widths [6, 1, 6, 1]; "
            f"got {signature(action_fields)}"
        )
    non_q99 = [
        field.key
        for field in (*state_fields, *action_fields)
        if getattr(field, "normalization", None) != "q99"
    ]
    if non_q99:
        raise ValueError(
            "Joint checkpoint must use q99 normalization for every state/action "
            f"field; mismatched fields: {non_q99}"
        )
    if int(bundle.max_state_dim) < 14 or int(bundle.max_action_dim) < 14:
        raise ValueError(
            "Joint checkpoint padded widths must be at least state=14, action=14, got "
            f"state={bundle.max_state_dim}, action={bundle.max_action_dim}"
        )


def _validate_joint_checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    provenance_path = checkpoint / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Joint checkpoint is missing provenance.json: {checkpoint}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("action_space") != "robodojo_joint_14d":
        raise ValueError(
            "Joint checkpoint provenance must declare "
            "action_space='robodojo_joint_14d', got "
            f"{provenance.get('action_space')!r}"
        )
    return provenance


@dataclass
class _PendingPrediction:
    env_idx_list: list[int]
    horizons: dict[int, int]
    event_memory: Any
    visual_memory: Any
    history_start: dict[int, int]
    rolling_cameras: dict[int, list[np.ndarray]]


class RoboDojoModel(ModelTemplate):
    """Multi-environment RoboDojo joint adapter for a native WorldScape checkpoint."""

    supports_batch = True
    requires_explicit_commit = True

    def __init__(self, model_cfg: Mapping[str, Any]):
        from .runtime_config import validate_config
        self.model_cfg = validate_config({"runtime_profile": "robodojo", **model_cfg})
        self.constrain_action_padding = self.model_cfg["constrain_action_padding"]
        self.wam_history_mode = _wam_history_mode_config(self.model_cfg)
        self.short_window_for_non_memory_tasks = self.model_cfg["short_window_for_non_memory_tasks"]
        if self.short_window_for_non_memory_tasks and self.wam_history_mode != "rolling":
            raise ValueError("short_window_for_non_memory_tasks requires wam_history_mode=rolling")
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type"))
        self.action_type = str(self.model_cfg.get("action_type") or SUPPORTED_ACTION_TYPE)
        if self.env_cfg_type != SUPPORTED_ENV_CFG:
            raise ValueError(
                f"WorldScape_Policy_2 supports env_cfg_type={SUPPORTED_ENV_CFG!r}, got {self.env_cfg_type!r}"
            )
        if self.action_type != SUPPORTED_ACTION_TYPE:
            raise ValueError(
                f"WorldScape_Policy_2 supports only action_type={SUPPORTED_ACTION_TYPE!r}, "
                f"got {self.action_type!r}"
            )
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if len(self.robot_action_dim_info["arm_dim"]) != 2:
            raise ValueError("WorldScape_Policy_2 RoboDojo checkpoints require a dual-arm robot")

        from .runtime_config import configure_policy_inputs, prepare_source
        self.upstream_root = prepare_source(self.model_cfg)

        self.checkpoint = resolve_checkpoint_root(
            self.model_cfg,
            POLICY_DIR / "checkpoints",
            policy_dir=POLICY_DIR,
        )
        for artifact in ("transform_bundle.json", "checkpoint_manifest.json"):
            if not (self.checkpoint / artifact).is_file():
                raise FileNotFoundError(f"Checkpoint is missing {artifact}: {self.checkpoint}")

        from evals.agilex.observation_adapter import (
            POLICY_VIDEO_NATIVE_H,
            POLICY_VIDEO_NATIVE_W,
            AgileXObservationAdapter,
        )
        from evals.common.checkpoint_runtime import load_checkpoint_transform_bundle
        from worldscape_policy.native_builder import (
            _load_native_model_config,
            build_wan22_policy_from_checkpoint,
        )

        self.device = torch.device(str(self.model_cfg.get("device") or "cuda"))
        self.seed = int(self.model_cfg.get("seed") or 0)
        self.input_color_order = _input_color_order_config(self.model_cfg)
        action_steps = self.model_cfg.get("replan_steps")
        self.observation_history_segments = self.model_cfg["observation_history_segments"]
        self.use_observation_history = self.model_cfg["use_observation_history"]
        self.use_causal_memory = self.model_cfg["use_causal_memory"]
        self.reset_independent_blocks = self.model_cfg["reset_independent_blocks"]
        if self.reset_independent_blocks and self.use_causal_memory:
            raise ValueError(
                "reset_independent_blocks requires use_causal_memory=false"
            )
        if self.wam_history_mode == "rolling" and (
            not self.use_observation_history or not self.use_causal_memory
            or self.reset_independent_blocks
            or self.observation_history_segments != 8
        ):
            raise ValueError(
                "wam_history_mode=rolling requires use_observation_history=true, "
                "use_causal_memory=true, reset_independent_blocks=false, "
                "observation_history_segments=8"
            )
        self.policy_image_size = (int(POLICY_VIDEO_NATIVE_W), int(POLICY_VIDEO_NATIVE_H))
        self._resize_image = partial(
            _resize_policy_image,
            width=self.policy_image_size[0],
            height=self.policy_image_size[1],
        )
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        from worldscape_policy.types import InteractionMode
        checkpoint_config = _load_native_model_config(self.checkpoint).model
        self.mode = InteractionMode.parse(checkpoint_config.mode)
        requested_mode = self.model_cfg.get("mode")
        if requested_mode is not None:
            from worldscape_policy.native_builder import checkpoint_supports_mode
            requested_mode = InteractionMode.parse(requested_mode)
            if not checkpoint_supports_mode(self.mode, requested_mode):
                raise ValueError("Requested mode is incompatible with the checkpoint")
            self.mode = requested_mode
        from .runtime_config import resolve_number_prompt_config, resolve_rollout_config
        provenance = _validate_joint_checkpoint_provenance(self.checkpoint)
        self.model_cfg = resolve_rollout_config(self.model_cfg, checkpoint_config, provenance)
        self.model_cfg = resolve_number_prompt_config(self.model_cfg, provenance)
        self.t5_prompt_template = _t5_prompt_template_config(self.model_cfg)
        qwen_config = self.model_cfg.get("interactive_qwen") or {}
        number_config = self.model_cfg.get("press_by_number_prompt") or {}
        number_enabled = number_config.get("enabled", False)
        qwen_enabled = number_enabled and qwen_config.get("enabled", False)
        if qwen_enabled and self.mode.value != "interactive":
            raise ValueError("interactive_qwen requires mode=interactive")
        # Validate deployment geometry from metadata before allocating model weights.
        self.action_horizon = self.model_cfg["action_horizon"]
        self.action_steps = self.action_horizon if action_steps is None else int(action_steps)
        if self.action_steps > self.action_horizon:
            raise ValueError("replan_steps exceeds checkpoint action_horizon")
        interval = self.model_cfg.get("observation_interval")
        if interval is not None:
            if self.action_steps % interval or self.action_steps // interval != self.observation_history_segments:
                raise ValueError("observation_interval must agree with replan_steps and observation_history_segments")

        if self.wam_history_mode == "rolling" and self.action_steps != self.action_horizon:
            raise ValueError(
                "wam_history_mode=rolling requires action_steps to match checkpoint "
                f"action_horizon={self.action_horizon}; got {self.action_steps}"
            )
        self.wam_max_chunk_size: int | None = None
        self.task_window_chunks: tuple[int, int] | None = None
        if self.wam_history_mode == "rolling":
            self.wam_max_chunk_size = int(checkpoint_config.wam.core.parameters.get("max_chunk_size", -1))
            if self.wam_max_chunk_size < 1:
                raise ValueError("rolling history requires a finite checkpoint max_chunk_size")
            self.task_window_chunks = (self.wam_max_chunk_size, self.wam_max_chunk_size)
        if self.short_window_for_non_memory_tasks:
            memory_chunks = self.model_cfg["memory_window_chunks"]
            ordinary_chunks = self.model_cfg["ordinary_window_chunks"]
            for name, count in (("memory_window_chunks", memory_chunks), ("ordinary_window_chunks", ordinary_chunks)):
                if count > self.wam_max_chunk_size:
                    raise ValueError(f"{name} exceeds checkpoint max_chunk_size={self.wam_max_chunk_size}")
            self.task_window_chunks = (memory_chunks, ordinary_chunks)
        self.bundle = load_checkpoint_transform_bundle(self.checkpoint, CHECKPOINT_EMBODIMENT)
        _validate_joint_transform_bundle(self.bundle, self.robot_action_dim_info)
        self.joint_action_representation = (
            "relative" if _relative_action_keys(self.bundle.transform) else "absolute"
        )
        self.visual_input_range = self.bundle.transform.image_input_range
        self.policy = build_wan22_policy_from_checkpoint(
            self.checkpoint,
            visual_input_range=self.visual_input_range,
            device=self.device,
            expected_mode=self.mode,
            expected_action_horizon=self.action_horizon,
            training=False,
            diffusion_view_layout=self.model_cfg.get("diffusion_view_layout"),
            validate_checkpoint_artifacts=self.model_cfg["validate_checkpoint_artifacts"],
            constrain_action_padding=self.constrain_action_padding,
        )
        configure_policy_inputs(self.policy, self.model_cfg)
        self._precompose_diffusion = (
            getattr(self.policy.visual_memory.codec, "diffusion_view_layout", None)
            == "robotwin_concat"
        )
        if self.wam_history_mode == "rolling":
            configure = getattr(self.policy.wam, "set_observation_history_mode", None)
            if not callable(configure):
                raise RuntimeError("Selected WorldScape source does not support wam_history_mode=rolling")
            if configure("rolling") != self.wam_max_chunk_size:
                raise ValueError("Loaded history capacity does not match checkpoint max_chunk_size")
        self._number_prompts = None
        if qwen_enabled:
            from .runtime_qwen import get_runtime_qwen
            shared_qwen = get_runtime_qwen(
                self.policy, checkpoint=self.checkpoint, device=self.device,
                verify_hashes=self.model_cfg["validate_checkpoint_artifacts"],
            )
            print(f"[WorldScape_Policy_2] interactive_qwen source={shared_qwen.source}")
        if number_enabled:
            from .press_by_number import (
                EpisodeNumberPrompts,
                QwenNumberReader,
            )
            self._number_prompts = EpisodeNumberPrompts(QwenNumberReader(
                model=shared_qwen.model, processor=shared_qwen.processor,
                device=str(self.device), batch_size=number_config.get("batch_size", 4),
            ))
        self._observation_adapter_class = AgileXObservationAdapter
        self._generator: torch.Generator | None = None
        self._event_memory: Any | None = None
        self._visual_memory: Any | None = None
        self._memory_env_idx_list: list[int] = []
        self._prediction_index = 0
        self._latest_obs_by_env: dict[int, Mapping[str, Any]] = {}
        self._obs_history_by_env: dict[int, list[Mapping[str, Any]]] = {}
        self._observation_index_by_env: dict[int, int] = {}
        self._latest_env_idx_list: list[int] = []
        self.reset()

    def update_obs(self, obs: Mapping[str, Any]):
        self.update_obs_batch([obs])

    def _uses_number_prompt(self, obs: Mapping[str, Any]) -> bool:
        # A shared server's startup task_name does not identify the live task.
        # Match the full raw instruction, never the fallback or a task keyword.
        return (
            getattr(self, "_number_prompts", None) is not None
            and match_memory_task(_instruction(obs, "")) == "press_by_number"
        )

    def update_obs_batch(self, obs_list: list[Mapping[str, Any]]):
        if not obs_list:
            raise ValueError("update_obs_batch requires at least one observation")
        env_idx_list: list[int] = []
        for fallback_idx, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise TypeError(f"update_obs_batch expects mappings, got {type(obs).__name__} at index {fallback_idx}")
            env_idx_list.append(int(obs.get("env_idx", fallback_idx)))
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(f"update_obs_batch received duplicate env indices: {env_idx_list}")

        for env_idx, obs in zip(env_idx_list, obs_list):
            self._ensure_env(env_idx)
            if self._uses_number_prompt(obs):
                # Snapshot raw RGB before policy resize/BGR conversion.
                self._number_prompts.observe(env_idx, _rgb_camera(obs, "cam_head"))
            elif getattr(self, "_number_prompts", None) is not None:
                # Do not retain an old task's count prompt when an env ID is reused.
                self._number_prompts.first_frames.pop(env_idx, None)
                self._number_prompts.records.pop(env_idx, None)
            self._latest_obs_by_env[env_idx] = obs
            self._obs_history_by_env[env_idx].append(obs)
            self._observation_index_by_env[env_idx] += 1
        self._latest_env_idx_list = env_idx_list

    def _ensure_env(self, env_idx: int) -> None:
        if env_idx in self._observation_index_by_env:
            return
        self._observation_index_by_env[env_idx] = 0
        self._obs_history_by_env[env_idx] = []

    def _prepare_observation_cameras(self, env_idx: int) -> list[np.ndarray]:
        rolling = getattr(self, "wam_history_mode", "periodic") == "rolling"
        previous = self._rolling_history_by_env.get(env_idx) if rolling else None
        if rolling:
            expected = self.action_steps + 1 if previous is not None else 1
            if len(self._obs_history_by_env[env_idx]) != expected:
                raise ValueError(
                    f"rolling history expected {expected} observations for env {env_idx}; "
                    "report every executed action and the final observation before prediction"
                )
        history = _sample_observation_history(
            self._obs_history_by_env[env_idx],
            segments=self.observation_history_segments,
            enabled=self.use_observation_history,
        )
        cameras = [
            np.stack(
                [
                    _resize_policy_camera(
                        history_obs,
                        name,
                        input_color_order=self.input_color_order,
                        resize_image=self._resize_image,
                    )
                    for history_obs in history
                ]
            )
            for name in CAMERA_ORDER
        ]
        if getattr(self, "_precompose_diffusion", False):
            # The first three streams remain thumbnails for the observation
            # adapter/head view. The fourth is the training-resolution canvas;
            # all four share exactly the same temporal sampling and eviction.
            cameras.append(_compose_diffusion_history(
                history,
                input_color_order=self.input_color_order,
                view_order=tuple(self.model_cfg.get("diffusion_concat_view_order", (0, 2, 1))),
                output_size=self.policy.visual_memory.codec.diffusion_concat_size,
            ))
        if rolling:
            window = self._history_window_by_env.get(env_idx, self.wam_max_chunk_size)
            max_frames = 1 + 8 * (window - 1)
            if previous is not None:
                cameras = [
                    np.concatenate([old, current[1:]], axis=0)
                    for old, current in zip(previous, cameras, strict=True)
                ]
            # Own bounded uint8 arrays on CPU; views would retain evicted frames.
            cameras = [camera[-max_frames:].copy() for camera in cameras]
        return cameras

    def _resolve_history_windows(self, env_idx_list: list[int]) -> None:
        """Choose once per episode from raw prompts, before any model sampling."""
        if not getattr(self, "short_window_for_non_memory_tasks", False):
            return
        memory_chunks, default_chunks = self.task_window_chunks
        selected: dict[int, tuple[int, str | None]] = {}
        for env in env_idx_list:
            if env not in self._history_window_by_env:
                # Missing/empty text must not match an unrelated configured
                # default_instruction. This decision never formats the T5 input.
                task = match_memory_task(_instruction(self._latest_obs_by_env[env], ""))
                selected[env] = (memory_chunks if task is not None else default_chunks, task)
        windows = {
            env: selected[env][0] if env in selected else self._history_window_by_env[env]
            for env in env_idx_list
        }
        if len(set(windows.values())) != 1:
            raise ValueError(
                f"Cannot mix {memory_chunks}- and {default_chunks}-chunk history windows in one inference batch; "
                f"evaluate memory and non-memory tasks separately. Windows: {windows}"
            )
        for env, (window, task) in selected.items():
            self._history_window_by_env[env] = window
            print(
                f"[WorldScape_Policy_2 history] env_idx={env} "
                f"short_window_for_non_memory_tasks=True "
                f"matched_memory_task={task!r} checkpoint_max_chunk_size={self.wam_max_chunk_size} "
                f"effective_chunk_size={window}",
                flush=True,
            )

    def _adapt_observation(self, env_idx: int, *, cameras: list[np.ndarray] | None = None) -> Any:
        obs = self._latest_obs_by_env.get(env_idx)
        if obs is None:
            raise RuntimeError(f"Call update_obs_batch() for env {env_idx} before get_action_batch()")
        if cameras is None:
            cameras = self._prepare_observation_cameras(env_idx)
        raw_state = _state_fields(
            obs,
            robot_action_dim_info=self.robot_action_dim_info,
        )
        instruction = _format_policy_instruction(
            _instruction(obs, self.default_instruction),
            mode=self.mode,
            t5_prompt_template=self.t5_prompt_template,
        )
        if self._uses_number_prompt(obs):
            instruction = self._number_prompts.records[env_idx]["instruction"]
        adapter = self._observation_adapter_class(
            self.bundle,
            mode=self.mode,
            instruction=instruction,
            visual_input_range=self.visual_input_range,
            device=self.device,
            session_id=f"robodojo-{env_idx}",
        )
        adapted = adapter((*cameras[:3], raw_state, self._observation_index_by_env[env_idx]))
        if getattr(self, "_precompose_diffusion", False):
            canvas = torch.from_numpy(cameras[3]).permute(0, 3, 1, 2)
            adapted.observation.images = (
                adapter._convert_images(canvas).unsqueeze(0).unsqueeze(2).to(self.device)
            )
        adapted.prompts.visual_prompt = "none"
        return adapted

    def get_action(self) -> list[dict[str, np.ndarray]]:
        env_idx = self._latest_env_idx_list[0] if self._latest_env_idx_list else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if self._pending_prediction is not None:
            raise RuntimeError("Commit or discard the pending prediction before predicting again")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch requires at least one environment")
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(f"get_action_batch received duplicate env indices: {env_idx_list}")

        missing_envs = [env_idx for env_idx in env_idx_list if env_idx not in self._latest_obs_by_env]
        if missing_envs:
            raise KeyError(f"Missing observations for env indices: {missing_envs}")

        if self._generator is None:
            raise RuntimeError("WorldScape batch generator is not initialized")
        if self._memory_env_idx_list:
            unknown = sorted(set(env_idx_list).difference(self._memory_env_idx_list))
            if unknown:
                raise ValueError(f"A causal batch may only shrink until reset(); new environment indices: {unknown}")
            positions = [self._memory_env_idx_list.index(env_idx) for env_idx in env_idx_list]
            event_memory = _select_causal_state(
                self._event_memory,
                positions,
                len(self._memory_env_idx_list),
            )
            visual_memory = _select_causal_state(
                self._visual_memory,
                positions,
                len(self._memory_env_idx_list),
            )
        else:
            event_memory = None
            visual_memory = None

        self._resolve_history_windows(env_idx_list)
        number_envs = [
            env for env in env_idx_list if self._uses_number_prompt(self._latest_obs_by_env[env])
        ]
        if number_envs:
            self._number_prompts.prepare(number_envs)
        generator_state = self._generator.get_state()
        try:
            if self.reset_independent_blocks:
                self.policy.reset_episode()
                self._generator.manual_seed(self.seed + self._prediction_index)
            rolling_cameras = {}
            if getattr(self, "wam_history_mode", "periodic") == "rolling":
                rolling_cameras = {env: self._prepare_observation_cameras(env) for env in env_idx_list}
                adapted_batch = [self._adapt_observation(env, cameras=rolling_cameras[env]) for env in env_idx_list]
            else:
                adapted_batch = [self._adapt_observation(env_idx) for env_idx in env_idx_list]
            observation = _stack_observation_batches([adapted.observation for adapted in adapted_batch])
            prompts = _stack_prompt_batches([adapted.prompts for adapted in adapted_batch])
            output = self.policy.sample(
                mode=self.mode,
                observation=observation,
                prompts=prompts,
                generator=self._generator,
                event_memory=event_memory,
                visual_memory=visual_memory,
            )
            action = output.require_action()
            if action.ndim != 3 or action.shape[0] != len(env_idx_list):
                raise ValueError(
                    "WorldScape_Policy_2 must return action [B,H,D] matching the request; "
                    f"got {tuple(action.shape)} for B={len(env_idx_list)}"
                )
            if rolling_cameras and action.shape[1] != self.action_steps:
                raise ValueError(f"rolling history requires a complete {self.action_steps}-action prediction")

            action_batch = []
            for batch_index, (adapted, env_idx) in enumerate(zip(adapted_batch, env_idx_list)):
                env_output = replace(output, action=action[batch_index : batch_index + 1])
                fields = _action_fields(
                    env_output,
                    self.bundle,
                    adapted.raw_state,
                )
                env_actions = _fields_to_robodojo_actions(
                    fields,
                    action_steps=self.action_steps,
                    robot_action_dim_info=self.robot_action_dim_info,
                )
                if rolling_cameras and len(env_actions) != self.action_steps:
                    raise ValueError(f"rolling history requires {self.action_steps} executable actions after decoding")
                action_batch.append(env_actions)

        except Exception:
            self._generator.set_state(generator_state)
            raise

        self._pending_prediction = _PendingPrediction(
            env_idx_list=list(env_idx_list),
            horizons={env: len(actions) for env, actions in zip(env_idx_list, action_batch, strict=True)},
            event_memory=output.next_memory if self.use_causal_memory else None,
            visual_memory=output.next_visual_memory if self.use_causal_memory else None,
            history_start={env: len(self._obs_history_by_env[env]) - 1 for env in env_idx_list},
            rolling_cameras=rolling_cameras,
        )
        return action_batch

    def commit_action(self, execution: Mapping[str, Any] | None = None) -> None:
        """Commit a completed chunk; retire environments whose episodes ended.

        The deployment loop reports executed_steps and finished_env_idx_list.
        Direct Python callers may omit the report to confirm full execution of
        every returned chunk, matching native PolicyRuntime.commit().
        """

        pending = self._pending_prediction
        if pending is None:
            raise RuntimeError("There is no pending prediction to commit")
        if execution is None:
            executed = pending.horizons
            finished = set()
        else:
            if not isinstance(execution, Mapping) or not isinstance(execution.get("executed_steps"), Mapping):
                raise ValueError("commit_action requires an executed_steps mapping")
            executed = {
                int(env): _integer_config(count, name="executed_steps", default=0, minimum=0)
                for env, count in execution["executed_steps"].items()
            }
            if len(executed) != len(execution["executed_steps"]):
                raise ValueError("Duplicate environment indices in executed_steps")
            finished = {int(env) for env in execution.get("finished_env_idx_list", [])}
        expected = set(pending.env_idx_list)
        if set(executed) != expected or not finished.issubset(expected):
            raise ValueError("Execution report must match the pending environment indices")
        for env, horizon in pending.horizons.items():
            count = executed[env]
            if count > horizon or (env not in finished and count != horizon):
                raise ValueError(
                    f"Environment {env} executed {count}/{horizon} actions; "
                    "discard a partial chunk before continuing inference"
                )
        active = [env for env in pending.env_idx_list if env not in finished]
        event_memory = visual_memory = None
        if self.use_causal_memory and active:
            if active == pending.env_idx_list:
                event_memory, visual_memory = pending.event_memory, pending.visual_memory
            else:
                positions = [pending.env_idx_list.index(env) for env in active]
                event_memory = _select_causal_state(pending.event_memory, positions, len(pending.env_idx_list))
                visual_memory = _select_causal_state(pending.visual_memory, positions, len(pending.env_idx_list))
        self._event_memory, self._visual_memory = event_memory, visual_memory
        self._memory_env_idx_list = active if self.use_causal_memory else []
        self._rolling_history_by_env = {
            env: pending.rolling_cameras[env] for env in active if env in pending.rolling_cameras
        }
        for env in pending.env_idx_list:
            # Keep the prediction-start frame plus observations received during
            # execution. The next loop update supplies the final executed frame.
            self._obs_history_by_env[env] = (
                self._obs_history_by_env[env][pending.history_start[env]:] if env in active else []
            )
        self._prediction_index += 1
        self._pending_prediction = None

    def discard_action(self, execution: Mapping[str, Any] | None = None) -> None:
        """Discard an unexecuted candidate, or rebase after partial/uncertain execution."""

        execution_started = False
        if execution is not None:
            if not isinstance(execution, Mapping):
                raise TypeError("discard_action expects an execution mapping")
            execution_started = _boolean_config(
                execution.get("execution_started"), name="execution_started", default=False
            )
        pending = self._pending_prediction
        if pending is None:
            return
        # Like native PolicyRuntime.discard(), a successful but discarded
        # prediction has already consumed its random samples.
        self._pending_prediction = None
        if execution_started:
            # Execution cannot be rolled back; restart context from a live observation.
            self._event_memory = None
            self._visual_memory = None
            self._memory_env_idx_list = []
            self._latest_obs_by_env = {}
            self._latest_env_idx_list = []
            self._obs_history_by_env = {env: [] for env in self._observation_index_by_env}
            self._rolling_history_by_env = {}
            self.policy.reset_episode()

    def reset(self):
        if getattr(self, "_number_prompts", None) is not None:
            self._number_prompts.reset()
        self.policy.reset_episode()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)
        self._event_memory = None
        self._visual_memory = None
        self._memory_env_idx_list = []
        self._prediction_index = 0
        self._pending_prediction: _PendingPrediction | None = None
        self._latest_obs_by_env = {}
        self._obs_history_by_env = {}
        self._rolling_history_by_env: dict[int, list[np.ndarray]] = {}
        self._history_window_by_env: dict[int, int] = {}
        self._observation_index_by_env = {}
        self._latest_env_idx_list = []
