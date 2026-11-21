from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch import Tensor

from worldscape_policy.types import InteractionMode


class ConditionMode(str, Enum):
    """Per-sample persistent-conditioning mode for the shared native WAM."""

    T2VA = "t2va"
    GOAL_IMAGE_TO_VA = "goal_image_to_va"
    VIDEO_TO_VA = "video_to_va"

    @classmethod
    def parse(cls, value: ConditionMode | str) -> ConditionMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"unknown condition mode {value!r}; expected one of: {choices}"
            ) from exc

    @property
    def id(self) -> int:
        return {
            ConditionMode.T2VA: 0,
            ConditionMode.GOAL_IMAGE_TO_VA: 1,
            ConditionMode.VIDEO_TO_VA: 2,
        }[self]


@dataclass(frozen=True)
class VisualPromptMetadata:
    """Identity and provenance attached to one visual prompt."""

    task_id: str
    embodiment: str
    source_episode_id: str
    source_session_id: str
    trusted_same_sample: bool = False
    override_audit_reason: str | None = None

    def validate(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("embodiment", self.embodiment),
            ("source_episode_id", self.source_episode_id),
            ("source_session_id", self.source_session_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"visual prompt {name} must be a non-empty string")
        if not isinstance(self.trusted_same_sample, bool):
            raise TypeError("visual prompt trusted_same_sample must be bool")
        if self.override_audit_reason is not None and (
            not isinstance(self.override_audit_reason, str)
            or not self.override_audit_reason.strip()
        ):
            raise ValueError(
                "visual prompt override_audit_reason must be a non-empty string"
            )


@dataclass(frozen=True)
class EventSample:
    """One explicit event chunk before model-specific transformation."""

    episode_id: str
    event_id: str
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    robot_state: np.ndarray
    high_level_instruction: str | None
    event_instruction: str | None
    goal_image: np.ndarray | None
    demo_video: np.ndarray | None
    history_head_frames: np.ndarray | None
    embodiment: str
    task_id: str
    session_id: str
    condition_mode: ConditionMode | str | None = None
    goal_prompt_metadata: VisualPromptMetadata | None = None
    demo_prompt_metadata: VisualPromptMetadata | None = None
    planning_labels_text: str | None = None
    planning_labels: np.ndarray | None = None
    semantic_target: np.ndarray | None = None
    semantic_mask: np.ndarray | None = None
    action_dim_mask: np.ndarray | None = None
    has_real_action: bool = True
    observation_valid_masks: dict[str, np.ndarray] | None = None
    action_valid_mask: np.ndarray | None = None
    robot_state_valid_mask: np.ndarray | None = None
    source_indices: dict[str, np.ndarray] | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.embodiment:
            raise ValueError("embodiment must be non-empty")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.observations:
            raise ValueError("observations must contain at least one modality")
        for name, value in self.observations.items():
            _require_array(f"observations[{name!r}]", value, min_ndim=1)
        _require_array("actions", self.actions, min_ndim=2)
        _require_array("robot_state", self.robot_state, min_ndim=2)
        for name, value in (
            ("high_level_instruction", self.high_level_instruction),
            ("event_instruction", self.event_instruction),
            ("planning_labels_text", self.planning_labels_text),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        for name, value in (
            ("goal_image", self.goal_image),
            ("demo_video", self.demo_video),
            ("history_head_frames", self.history_head_frames),
        ):
            if value is not None:
                _require_array(name, value, min_ndim=3)
        if self.goal_image is not None and self.goal_image.ndim not in (3, 4):
            raise ValueError("goal_image must have shape [H,W,C] or [V,H,W,C]")
        if self.demo_video is not None and self.demo_video.ndim < 4:
            raise ValueError("demo_video must include time and image dimensions")
        for prompt_name, prompt, metadata in (
            ("goal", self.goal_image, self.goal_prompt_metadata),
            ("demo", self.demo_video, self.demo_prompt_metadata),
        ):
            if (prompt is None) != (metadata is None):
                raise ValueError(
                    f"{prompt_name} prompt and {prompt_name}_prompt_metadata "
                    "must be provided together"
                )
            if metadata is not None:
                metadata.validate()
                if metadata.trusted_same_sample and (
                    metadata.task_id != self.task_id
                    or metadata.embodiment != self.embodiment
                    or metadata.source_episode_id != self.episode_id
                    or metadata.source_session_id != self.session_id
                ):
                    raise ValueError(
                        f"trusted same-sample {prompt_name} metadata does not "
                        "match the sample"
                    )
        if self.condition_mode is not None:
            condition_mode = ConditionMode.parse(self.condition_mode)
            has_goal = self.goal_image is not None
            has_demo = self.demo_video is not None
            if condition_mode is ConditionMode.T2VA and (has_goal or has_demo):
                raise ValueError("T2VA samples cannot carry a persistent visual prompt")
            if condition_mode is ConditionMode.GOAL_IMAGE_TO_VA and (
                not has_goal or has_demo
            ):
                raise ValueError("goal-image-to-VA samples require only a goal image")
            if condition_mode is ConditionMode.VIDEO_TO_VA and (
                has_goal or not has_demo
            ):
                raise ValueError("video-to-VA samples require only a demo video")
        if self.history_head_frames is not None and self.history_head_frames.ndim != 4:
            raise ValueError("history_head_frames must have shape [T,H,W,C]")
        if self.planning_labels is not None:
            _require_array("planning_labels", self.planning_labels, min_ndim=1)
            if self.planning_labels.dtype.kind not in {"i", "u"}:
                raise TypeError("planning_labels must contain integer token ids")
        if self.semantic_target is not None:
            _require_array("semantic_target", self.semantic_target, min_ndim=2)
        if self.semantic_mask is not None:
            _require_array("semantic_mask", self.semantic_mask, min_ndim=1)
            if self.semantic_target is None:
                raise ValueError("semantic_mask requires semantic_target")
        if self.action_dim_mask is not None:
            _require_array("action_dim_mask", self.action_dim_mask, min_ndim=1)
            if self.action_dim_mask.ndim != 1:
                raise ValueError("action_dim_mask must have shape [D]")
            if self.action_dim_mask.shape[0] != self.actions.shape[-1]:
                raise ValueError("action_dim_mask must match the action dimension")
        if not isinstance(self.has_real_action, (bool, np.bool_)):
            raise TypeError("has_real_action must be bool")
        if self.observation_valid_masks is not None:
            if set(self.observation_valid_masks) != set(self.observations):
                raise ValueError("observation_valid_masks must match observation keys")
            for name, mask in self.observation_valid_masks.items():
                _validate_numpy_mask(
                    f"observation_valid_masks[{name!r}]",
                    mask,
                    len(self.observations[name]),
                )
        if self.action_valid_mask is not None:
            _validate_numpy_mask("action_valid_mask", self.action_valid_mask, len(self.actions))
        if self.robot_state_valid_mask is not None:
            _validate_numpy_mask(
                "robot_state_valid_mask", self.robot_state_valid_mask, len(self.robot_state)
            )
        if self.source_indices is not None:
            for name, indices in self.source_indices.items():
                _require_array(f"source_indices[{name!r}]", indices, min_ndim=1)
                if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
                    raise TypeError("source indices must be one-dimensional integers")


@dataclass(frozen=True)
class TransformedEventSample:
    episode_id: str
    event_id: str
    observations: dict[str, Tensor]
    actions: Tensor
    robot_state: Tensor
    high_level_instruction: str | None
    event_instruction: str | None
    goal_image: Tensor | None
    demo_video: Tensor | None
    history_head_frames: Tensor | None
    embodiment: str
    task_id: str
    session_id: str
    goal_prompt_metadata: VisualPromptMetadata | None
    demo_prompt_metadata: VisualPromptMetadata | None
    mode: InteractionMode
    condition_mode: ConditionMode
    planning_labels_text: str | None = None
    planning_labels: Tensor | None = None
    semantic_target: Tensor | None = None
    semantic_mask: Tensor | None = None
    action_dim_mask: Tensor | None = None
    has_real_action: bool = True
    observation_valid_masks: dict[str, Tensor] | None = None
    action_valid_mask: Tensor | None = None
    robot_state_valid_mask: Tensor | None = None
    source_indices: dict[str, Tensor] | None = None
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    """Padded, batch-first native training input."""

    episode_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    observations: dict[str, Tensor]
    observation_masks: dict[str, Tensor]
    actions: Tensor
    action_mask: Tensor
    robot_state: Tensor
    robot_state_mask: Tensor
    high_level_instructions: tuple[str | None, ...]
    event_instructions: tuple[str | None, ...]
    embodiments: tuple[str, ...]
    modes: tuple[InteractionMode, ...]
    mode_mask: Tensor
    condition_modes: tuple[ConditionMode, ...] | None = None
    condition_ids: Tensor | None = None
    task_ids: tuple[str, ...] | None = None
    session_ids: tuple[str, ...] | None = None
    goal_prompt_metadata: tuple[VisualPromptMetadata | None, ...] | None = None
    demo_prompt_metadata: tuple[VisualPromptMetadata | None, ...] | None = None
    goal_images: Tensor | None = None
    goal_image_mask: Tensor | None = None
    demo_videos: Tensor | None = None
    demo_video_mask: Tensor | None = None
    history_head_frames: Tensor | None = None
    history_mask: Tensor | None = None
    planning_labels_text: tuple[str | None, ...] | None = None
    planning_labels: Tensor | None = None
    semantic_target: Tensor | None = None
    semantic_mask: Tensor | None = None
    action_dim_mask: Tensor | None = None
    has_real_action: Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.episode_ids)

    def validate(self) -> None:
        batch_size = self.batch_size
        if batch_size == 0:
            raise ValueError("TrainingBatch cannot be empty")
        for name, values in (
            ("event_ids", self.event_ids),
            ("high_level_instructions", self.high_level_instructions),
            ("event_instructions", self.event_instructions),
            ("embodiments", self.embodiments),
            ("modes", self.modes),
        ):
            if len(values) != batch_size:
                raise ValueError(f"{name} must contain {batch_size} values")
        for name, values in (
            ("task_ids", self.task_ids),
            ("session_ids", self.session_ids),
            ("goal_prompt_metadata", self.goal_prompt_metadata),
            ("demo_prompt_metadata", self.demo_prompt_metadata),
        ):
            if values is not None and len(values) != batch_size:
                raise ValueError(f"{name} must contain {batch_size} values")
        if not self.observations:
            raise ValueError("observations must contain at least one modality")
        if set(self.observations) != set(self.observation_masks):
            raise ValueError(
                "observations and observation_masks must have identical keys"
            )
        for name, tensor in self.observations.items():
            _require_batch_tensor(f"observations[{name!r}]", tensor, batch_size)
            _validate_temporal_mask(
                f"observation_masks[{name!r}]",
                self.observation_masks[name],
                tensor,
                batch_size,
            )
        for name, tensor, mask in (
            ("actions", self.actions, self.action_mask),
            ("robot_state", self.robot_state, self.robot_state_mask),
        ):
            _require_batch_tensor(name, tensor, batch_size)
            _validate_temporal_mask(f"{name}_mask", mask, tensor, batch_size)
        if self.mode_mask.dtype is not torch.bool or self.mode_mask.shape != (
            batch_size,
        ):
            raise ValueError(f"mode_mask must be bool with shape [{batch_size}]")
        expected_mode_mask = torch.tensor(
            [mode is InteractionMode.AUTO for mode in self.modes],
            dtype=torch.bool,
            device=self.mode_mask.device,
        )
        if not torch.equal(self.mode_mask, expected_mode_mask):
            raise ValueError("mode_mask must be true exactly for Auto samples")
        if (self.condition_modes is None) != (self.condition_ids is None):
            raise ValueError(
                "condition_modes and condition_ids must be provided together"
            )
        if self.condition_modes is not None:
            if len(self.condition_modes) != batch_size:
                raise ValueError(
                    f"condition_modes must contain {batch_size} values"
                )
            assert self.condition_ids is not None
            if self.condition_ids.dtype not in (torch.int32, torch.int64) or (
                self.condition_ids.shape != (batch_size,)
            ):
                raise ValueError(
                    f"condition_ids must be integer with shape [{batch_size}]"
                )
            expected_condition_ids = torch.tensor(
                [ConditionMode.parse(mode).id for mode in self.condition_modes],
                dtype=self.condition_ids.dtype,
                device=self.condition_ids.device,
            )
            if not torch.equal(self.condition_ids, expected_condition_ids):
                raise ValueError("condition_ids do not match condition_modes")
        if (
            self.planning_labels_text is not None
            and len(self.planning_labels_text) != batch_size
        ):
            raise ValueError(f"planning_labels_text must contain {batch_size} values")
        if self.planning_labels is not None:
            if (
                self.planning_labels.ndim != 2
                or self.planning_labels.shape[0] != batch_size
            ):
                raise ValueError("planning_labels must have shape [B,T]")
            if self.planning_labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("planning_labels must contain integer token ids")
        if self.semantic_target is not None:
            _require_batch_tensor("semantic_target", self.semantic_target, batch_size)
        if self.semantic_mask is not None:
            if self.semantic_target is None:
                raise ValueError("semantic_mask requires semantic_target")
            if self.semantic_mask.dtype is not torch.bool:
                raise ValueError("semantic_mask must have bool dtype")
            if self.semantic_mask.shape != self.semantic_target.shape[:2]:
                raise ValueError("semantic_mask must match semantic target tokens")
        if self.action_dim_mask is not None and self.action_dim_mask.shape != (
            batch_size,
            self.actions.shape[-1],
        ):
            raise ValueError("action_dim_mask must have shape [B,D]")
        if self.has_real_action is not None and (
            self.has_real_action.dtype is not torch.bool
            or self.has_real_action.shape != (batch_size,)
        ):
            raise ValueError(
                f"has_real_action must be bool with shape [{batch_size}]"
            )
        for value_name, mask_name in (
            ("goal_images", "goal_image_mask"),
            ("demo_videos", "demo_video_mask"),
            ("history_head_frames", "history_mask"),
        ):
            value = getattr(self, value_name)
            mask = getattr(self, mask_name)
            if (value is None) != (mask is None):
                raise ValueError(
                    f"{value_name} and {mask_name} must be provided together"
                )
            if value is not None:
                _require_batch_tensor(value_name, value, batch_size)
                if mask.dtype is not torch.bool or mask.shape[0] != batch_size:
                    raise ValueError(f"{mask_name} must be bool and batch-first")
        for prompt_name, metadata_values, mask in (
            ("goal", self.goal_prompt_metadata, self.goal_image_mask),
            ("demo", self.demo_prompt_metadata, self.demo_video_mask),
        ):
            if metadata_values is None:
                if mask is not None and bool(mask.any().item()):
                    raise ValueError(
                        f"{prompt_name} prompt requires provenance metadata"
                    )
                continue
            for index, metadata in enumerate(metadata_values):
                if metadata is not None:
                    metadata.validate()
                present = False if mask is None else bool(mask[index].any().item())
                if present != (metadata is not None):
                    raise ValueError(
                        f"{prompt_name} prompt metadata presence must match its mask"
                    )
                if metadata is None:
                    continue
                if self.task_ids is None:
                    raise ValueError(
                        f"{prompt_name} prompt provenance requires batch task_ids"
                    )
                mismatches = []
                if metadata.task_id != self.task_ids[index]:
                    mismatches.append(
                        f"task {metadata.task_id!r} != {self.task_ids[index]!r}"
                    )
                if metadata.embodiment != self.embodiments[index]:
                    mismatches.append(
                        "embodiment "
                        f"{metadata.embodiment!r} != {self.embodiments[index]!r}"
                    )
                if mismatches and metadata.override_audit_reason is None:
                    raise ValueError(
                        f"incompatible {prompt_name} visual prompt at batch index "
                        f"{index}: {', '.join(mismatches)}; an audited override is "
                        "required"
                    )
                if metadata.trusted_same_sample:
                    if self.session_ids is None:
                        raise ValueError(
                            f"trusted legacy {prompt_name} prompt requires session_ids"
                        )
                    if (
                        metadata.task_id != self.task_ids[index]
                        or metadata.embodiment != self.embodiments[index]
                        or metadata.source_episode_id != self.episode_ids[index]
                        or metadata.source_session_id != self.session_ids[index]
                    ):
                        raise ValueError(
                            f"trusted legacy {prompt_name} prompt marker does not "
                            f"match batch index {index}"
                        )
        if self.condition_modes is not None:
            for index, mode in enumerate(self.condition_modes):
                parsed = ConditionMode.parse(mode)
                has_goal = (
                    self.goal_image_mask is not None
                    and bool(self.goal_image_mask[index].any().item())
                )
                has_demo = (
                    self.demo_video_mask is not None
                    and bool(self.demo_video_mask[index].any().item())
                )
                expected = (
                    ConditionMode.GOAL_IMAGE_TO_VA
                    if has_goal
                    else (
                        ConditionMode.VIDEO_TO_VA
                        if has_demo
                        else ConditionMode.T2VA
                    )
                )
                if has_goal and has_demo:
                    raise ValueError(
                        f"sample {index} cannot carry both goal and demo prompts"
                    )
                if parsed is not expected:
                    raise ValueError(
                        f"condition mode {parsed.value!r} does not match prompts "
                        f"for sample {index}"
                    )


def _require_array(name: str, value: object, *, min_ndim: int) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if value.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")


def _validate_numpy_mask(name: str, mask: np.ndarray, length: int) -> None:
    _require_array(name, mask, min_ndim=1)
    if mask.dtype != np.bool_ or mask.shape != (length,):
        raise ValueError(f"{name} must be bool with shape [{length}]")


def _require_batch_tensor(name: str, tensor: Tensor, batch_size: int) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim < 2 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must be batch-first with batch size {batch_size}")


def _validate_temporal_mask(
    name: str, mask: Tensor, value: Tensor, batch_size: int
) -> None:
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must have bool dtype")
    if mask.shape != (batch_size, value.shape[1]):
        raise ValueError(
            f"{name} must have shape [{batch_size}, {value.shape[1]}], "
            f"got {tuple(mask.shape)}"
        )
