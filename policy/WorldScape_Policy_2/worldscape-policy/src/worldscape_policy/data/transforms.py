from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from worldscape_policy.action_space import (
    convert_eef_actions_to_relative,
    parse_action_mode,
)
from worldscape_policy.data.augmentation import NativeVideoAugmentation
from worldscape_policy.data.collate import (
    NativeTrainingCollator,
    TrainingCollator,
)
from worldscape_policy.data.normalization import GlobalZScoreNormalizer
from worldscape_policy.data.sampling import (
    ModeSampler,
    PromptModality,
    PromptModalitySampler,
    VisualPromptSampler,
)
from worldscape_policy.data.schema import (
    ConditionMode,
    EventSample,
    TransformedEventSample,
)
from worldscape_policy.types import InteractionMode


class NativeEventTransform:
    """Convert schema arrays to tensors without legacy key interpretation."""

    def __init__(
        self,
        mode_sampler: ModeSampler | None = None,
        *,
        fixed_mode: InteractionMode | str | None = None,
        prompt_modality_sampler: PromptModalitySampler | None = None,
        video_augmentation: NativeVideoAugmentation | None = None,
        seed: int = 0,
        wo_norm: bool = True,
        action_mode: str = "eef",
        relative_action: bool = False,
        action_dim_mask: Sequence[float | bool] | None = None,
        normalization_statistics_path: str | None = None,
        normalization_clip_range: Sequence[float] = (-5.0, 5.0),
        raw_video_observation_key: str | None = None,
    ) -> None:
        if mode_sampler is not None and fixed_mode is not None:
            raise ValueError("configure either mode_sampler or fixed_mode, not both")
        self.mode_sampler = mode_sampler
        self.fixed_mode = (
            InteractionMode.parse(fixed_mode) if fixed_mode is not None else None
        )
        self.prompt_modality_sampler = prompt_modality_sampler
        self.video_augmentation = video_augmentation
        self.seed = int(seed)
        self.wo_norm = bool(wo_norm)
        self.action_mode = parse_action_mode(action_mode)
        self.relative_action = bool(relative_action)
        self.raw_video_observation_key = (
            None
            if raw_video_observation_key is None
            else str(raw_video_observation_key).strip()
        )
        if self.raw_video_observation_key == "":
            raise ValueError("raw_video_observation_key must be non-empty")
        if self.raw_video_observation_key == "video":
            raise ValueError("raw_video_observation_key must differ from 'video'")
        if self.action_mode == "joint" and self.relative_action:
            raise ValueError("joint action_mode currently supports absolute actions only")
        if not self.wo_norm and normalization_statistics_path is None:
            raise ValueError(
                "normalization_statistics_path is required when wo_norm is false"
            )
        if len(normalization_clip_range) != 2:
            raise ValueError("normalization_clip_range must contain [min, max]")
        self.normalizer = (
            None
            if self.wo_norm
            else GlobalZScoreNormalizer(
                normalization_statistics_path,
                clip_range=(
                    float(normalization_clip_range[0]),
                    float(normalization_clip_range[1]),
                ),
            )
        )
        self.action_dim_mask = (
            None
            if action_dim_mask is None
            else np.asarray(action_dim_mask, dtype=np.float32).reshape(-1)
        )
        if self.action_dim_mask is not None and (
            not np.all(np.isfinite(self.action_dim_mask))
            or np.any(self.action_dim_mask < 0)
        ):
            raise ValueError("action_dim_mask weights must be finite and non-negative")

    def __call__(
        self,
        sample: EventSample,
        *,
        mode: InteractionMode | str | None = None,
        rng: np.random.Generator | None = None,
    ) -> TransformedEventSample:
        sample.validate()
        if (
            self.action_dim_mask is not None
            and self.action_dim_mask.shape != (sample.actions.shape[-1],)
        ):
            raise ValueError(
                "configured action_dim_mask must match the sample action dimension"
            )
        if mode is not None:
            parsed_mode = InteractionMode.parse(mode)
        elif self.fixed_mode is not None:
            parsed_mode = self.fixed_mode
        elif self.mode_sampler is not None:
            parsed_mode = self.mode_sampler.sample(rng=rng)
        else:
            raise ValueError(
                "NativeEventTransform requires an explicit mode or fixed_mode; "
                "per-item random mode sampling is not a safe training default"
            )
        generator = rng or self._sample_rng(sample)
        actions = (
            convert_eef_actions_to_relative(sample.actions, sample.robot_state)
            if self.relative_action
            else sample.actions
        )
        robot_state = sample.robot_state
        if self.normalizer is not None:
            robot_state = self.normalizer.normalize_state(robot_state)
            actions = self.normalizer.normalize_action(actions)
        observations = sample.observations
        raw_video = (
            observations.get("video")
            if self.raw_video_observation_key is not None
            else None
        )
        if self.raw_video_observation_key is not None and raw_video is None:
            raise KeyError(
                "raw_video_observation_key requires observations['video']"
            )
        goal_image = sample.goal_image
        demo_video = sample.demo_video
        goal_metadata = sample.goal_prompt_metadata
        demo_metadata = sample.demo_prompt_metadata
        if self.prompt_modality_sampler is not None:
            modality = self.prompt_modality_sampler.sample(rng=generator)
            if modality is PromptModality.TEXT:
                goal_image = None
                demo_video = None
                goal_metadata = None
                demo_metadata = None
                condition_mode = ConditionMode.T2VA
            elif modality in {PromptModality.GOAL, PromptModality.TEXT_GOAL}:
                prompt = VisualPromptSampler("goal").sample(sample, rng=generator)
                goal_image = prompt.goal_image
                demo_video = None
                goal_metadata = prompt.goal_metadata
                demo_metadata = None
                condition_mode = ConditionMode.GOAL_IMAGE_TO_VA
            elif modality in {PromptModality.DEMO, PromptModality.TEXT_DEMO}:
                prompt = VisualPromptSampler("demo").sample(sample, rng=generator)
                goal_image = None
                demo_video = prompt.demo_video
                goal_metadata = None
                demo_metadata = prompt.demo_metadata
                condition_mode = ConditionMode.VIDEO_TO_VA
            else:  # pragma: no cover - exhaustive enum guard
                raise AssertionError(f"unsupported prompt modality: {modality}")
        else:
            condition_mode = self._resolve_condition_mode(sample)
        history = sample.history_head_frames
        if self.video_augmentation is not None:
            observations = {
                name: (
                    self.video_augmentation(value, rng=generator)
                    if value.dtype == np.uint8 and value.ndim in {4, 5}
                    else value
                )
                for name, value in observations.items()
            }
            if goal_image is not None:
                goal_image = self.video_augmentation(goal_image[None], rng=generator)[0]
            if demo_video is not None:
                demo_video = self.video_augmentation(demo_video, rng=generator)
            if history is not None:
                history = self.video_augmentation(history, rng=generator)
        if self.raw_video_observation_key is not None:
            assert raw_video is not None
            observations = dict(observations)
            observations[self.raw_video_observation_key] = raw_video
        observation_valid_masks = (
            None
            if sample.observation_valid_masks is None
            else dict(sample.observation_valid_masks)
        )
        if (
            self.raw_video_observation_key is not None
            and observation_valid_masks is not None
        ):
            observation_valid_masks[self.raw_video_observation_key] = (
                observation_valid_masks["video"]
            )
        return TransformedEventSample(
            episode_id=sample.episode_id,
            event_id=sample.event_id,
            observations={
                name: _as_tensor(value) for name, value in observations.items()
            },
            actions=_as_tensor(actions),
            robot_state=_as_tensor(robot_state),
            high_level_instruction=sample.high_level_instruction,
            event_instruction=sample.event_instruction,
            goal_image=_optional_tensor(goal_image),
            demo_video=_optional_tensor(demo_video),
            history_head_frames=_optional_tensor(history),
            embodiment=sample.embodiment,
            task_id=sample.task_id,
            session_id=sample.session_id,
            goal_prompt_metadata=goal_metadata,
            demo_prompt_metadata=demo_metadata,
            mode=parsed_mode,
            condition_mode=condition_mode,
            planning_labels_text=sample.planning_labels_text,
            planning_labels=_optional_tensor(sample.planning_labels),
            semantic_target=_optional_tensor(sample.semantic_target),
            semantic_mask=_optional_tensor(sample.semantic_mask),
            action_dim_mask=_optional_tensor(
                self.action_dim_mask
                if self.action_dim_mask is not None
                else sample.action_dim_mask
            ),
            has_real_action=bool(sample.has_real_action),
            observation_valid_masks=(
                None
                if observation_valid_masks is None
                else {
                    name: _as_tensor(value)
                    for name, value in observation_valid_masks.items()
                }
            ),
            action_valid_mask=_optional_tensor(sample.action_valid_mask),
            robot_state_valid_mask=_optional_tensor(sample.robot_state_valid_mask),
            source_indices=(
                None
                if sample.source_indices is None
                else {
                    name: _as_tensor(value)
                    for name, value in sample.source_indices.items()
                }
            ),
            provenance=dict(sample.provenance),
        )

    def _sample_rng(self, sample: EventSample) -> np.random.Generator:
        identity = f"{sample.episode_id}\0{sample.event_id}".encode()
        identity_seed = zlib.crc32(identity)
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, identity_seed])
        )

    @staticmethod
    def _resolve_condition_mode(sample: EventSample) -> ConditionMode:
        if sample.condition_mode is not None:
            return ConditionMode.parse(sample.condition_mode)
        has_goal = sample.goal_image is not None
        has_demo = sample.demo_video is not None
        if has_goal and has_demo:
            raise ValueError(
                "samples with both goal and demo prompts require a "
                "PromptModalitySampler"
            )
        if has_goal:
            return ConditionMode.GOAL_IMAGE_TO_VA
        if has_demo:
            return ConditionMode.VIDEO_TO_VA
        return ConditionMode.T2VA


def _as_tensor(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(value))


def _optional_tensor(value: np.ndarray | None) -> Tensor | None:
    return None if value is None else _as_tensor(value)


# Backward-compatible import aliases for historical transform module paths.
EventTransform = NativeEventTransform

__all__ = [
    "EventTransform",
    "NativeEventTransform",
    "NativeTrainingCollator",
    "TrainingCollator",
]
