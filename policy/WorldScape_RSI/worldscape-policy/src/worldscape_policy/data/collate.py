"""Batch collation for transformed native event samples."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from worldscape_policy.data.schema import TrainingBatch, TransformedEventSample
from worldscape_policy.types import InteractionMode


class NativeTrainingCollator:
    """Pad temporal fields and retain explicit masks for every padded value."""

    def __call__(self, samples: Sequence[TransformedEventSample]) -> TrainingBatch:
        if not samples:
            raise ValueError("cannot collate an empty sample sequence")
        observation_keys = set(samples[0].observations)
        if any(set(sample.observations) != observation_keys for sample in samples[1:]):
            raise ValueError(
                "all samples must contain identical observation modalities"
            )

        observations: dict[str, Tensor] = {}
        observation_masks: dict[str, Tensor] = {}
        for name in sorted(observation_keys):
            observations[name], observation_masks[name] = _pad_temporal(
                [sample.observations[name] for sample in samples], name
            )
            explicit = [
                None
                if sample.observation_valid_masks is None
                else sample.observation_valid_masks[name]
                for sample in samples
            ]
            observation_masks[name] &= _pad_explicit_masks(
                explicit, [sample.observations[name] for sample in samples]
            )
        actions, action_mask = _pad_temporal(
            [sample.actions for sample in samples], "actions"
        )
        robot_state, robot_state_mask = _pad_temporal(
            [sample.robot_state for sample in samples], "robot_state"
        )
        action_mask &= _pad_explicit_masks(
            [sample.action_valid_mask for sample in samples],
            [sample.actions for sample in samples],
        )
        robot_state_mask &= _pad_explicit_masks(
            [sample.robot_state_valid_mask for sample in samples],
            [sample.robot_state for sample in samples],
        )
        goal_images, goal_image_mask = _stack_optional(
            [sample.goal_image for sample in samples], "goal_image"
        )
        demo_videos, demo_video_mask = _pad_optional_temporal(
            [sample.demo_video for sample in samples], "demo_video"
        )
        history, history_mask = _pad_optional_temporal(
            [sample.history_head_frames for sample in samples],
            "history_head_frames",
        )
        planning_labels = _pad_optional_labels(
            [sample.planning_labels for sample in samples]
        )
        semantic_target, inferred_semantic_mask = _pad_optional_temporal(
            [sample.semantic_target for sample in samples], "semantic_target"
        )
        explicit_semantic_masks = [sample.semantic_mask for sample in samples]
        if any(value is not None for value in explicit_semantic_masks):
            if semantic_target is None:
                raise ValueError("semantic masks require semantic targets")
            filled_masks = [
                value.to(dtype=torch.bool)
                if value is not None
                else (
                    torch.ones(sample.semantic_target.shape[0], dtype=torch.bool)
                    if sample.semantic_target is not None
                    else torch.empty(0, dtype=torch.bool)
                )
                for sample, value in zip(samples, explicit_semantic_masks, strict=True)
            ]
            semantic_mask, _ = _pad_temporal(filled_masks, "semantic_mask")
            semantic_mask &= inferred_semantic_mask
        else:
            semantic_mask = inferred_semantic_mask
        action_dim_mask = torch.stack(
            [
                (
                    sample.action_dim_mask.to(dtype=torch.float32)
                    if sample.action_dim_mask is not None
                    else torch.ones(
                        sample.actions.shape[-1],
                        dtype=torch.float32,
                        device=sample.actions.device,
                    )
                )
                for sample in samples
            ]
        )
        has_real_action = torch.tensor(
            [bool(sample.has_real_action) for sample in samples],
            dtype=torch.bool,
            device=actions.device,
        )
        modes = tuple(sample.mode for sample in samples)
        condition_modes = tuple(sample.condition_mode for sample in samples)
        batch = TrainingBatch(
            episode_ids=tuple(sample.episode_id for sample in samples),
            event_ids=tuple(sample.event_id for sample in samples),
            observations=observations,
            observation_masks=observation_masks,
            actions=actions,
            action_mask=action_mask,
            robot_state=robot_state,
            robot_state_mask=robot_state_mask,
            high_level_instructions=tuple(
                sample.high_level_instruction for sample in samples
            ),
            event_instructions=tuple(sample.event_instruction for sample in samples),
            embodiments=tuple(sample.embodiment for sample in samples),
            task_ids=tuple(sample.task_id for sample in samples),
            session_ids=tuple(sample.session_id for sample in samples),
            goal_prompt_metadata=tuple(
                sample.goal_prompt_metadata for sample in samples
            ),
            demo_prompt_metadata=tuple(
                sample.demo_prompt_metadata for sample in samples
            ),
            modes=modes,
            mode_mask=torch.tensor(
                [mode is InteractionMode.AUTO for mode in modes], dtype=torch.bool
            ),
            condition_modes=condition_modes,
            condition_ids=torch.tensor(
                [mode.id for mode in condition_modes], dtype=torch.long
            ),
            goal_images=goal_images,
            goal_image_mask=goal_image_mask,
            demo_videos=demo_videos,
            demo_video_mask=demo_video_mask,
            history_head_frames=history,
            history_mask=history_mask,
            planning_labels_text=(
                tuple(sample.planning_labels_text for sample in samples)
                if any(sample.planning_labels_text is not None for sample in samples)
                else None
            ),
            planning_labels=planning_labels,
            semantic_target=semantic_target,
            semantic_mask=semantic_mask,
            action_dim_mask=action_dim_mask,
            has_real_action=has_real_action,
            metadata={
                "source_indices": tuple(sample.source_indices for sample in samples),
                "provenance": tuple(dict(sample.provenance) for sample in samples),
            },
        )
        batch.validate()
        return batch


def _pad_temporal(values: Sequence[Tensor], name: str) -> tuple[Tensor, Tensor]:
    if any(value.ndim == 0 for value in values):
        raise ValueError(f"{name} values must have a temporal dimension")
    trailing_shape = values[0].shape[1:]
    dtype = values[0].dtype
    device = values[0].device
    if any(
        value.shape[1:] != trailing_shape
        or value.dtype != dtype
        or value.device != device
        for value in values[1:]
    ):
        raise ValueError(f"{name} values must have matching trailing shape and dtype")
    max_length = max(value.shape[0] for value in values)
    output = torch.zeros(
        (len(values), max_length, *trailing_shape), dtype=dtype, device=device
    )
    mask = torch.zeros((len(values), max_length), dtype=torch.bool, device=device)
    for index, value in enumerate(values):
        output[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return output, mask


def _stack_optional(
    values: Sequence[Tensor | None], name: str
) -> tuple[Tensor | None, Tensor | None]:
    present = [value for value in values if value is not None]
    if not present:
        return None, None
    reference = present[0]
    if any(
        value.shape != reference.shape or value.dtype != reference.dtype
        for value in present
    ):
        raise ValueError(f"{name} values must have matching shape and dtype")
    output = torch.zeros(
        (len(values), *reference.shape),
        dtype=reference.dtype,
        device=reference.device,
    )
    mask = torch.zeros(len(values), dtype=torch.bool, device=reference.device)
    for index, value in enumerate(values):
        if value is not None:
            output[index] = value
            mask[index] = True
    return output, mask


def _pad_optional_temporal(
    values: Sequence[Tensor | None], name: str
) -> tuple[Tensor | None, Tensor | None]:
    present = [value for value in values if value is not None]
    if not present:
        return None, None
    reference = present[0]
    filled = [
        value
        if value is not None
        else torch.empty(
            (0, *reference.shape[1:]),
            dtype=reference.dtype,
            device=reference.device,
        )
        for value in values
    ]
    return _pad_temporal(filled, name)


def _pad_optional_labels(values: Sequence[Tensor | None]) -> Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if any(value.ndim != 1 for value in present):
        raise ValueError("planning_labels values must be one-dimensional")
    max_length = max(value.shape[0] for value in present)
    output = torch.full(
        (len(values), max_length),
        -100,
        dtype=torch.long,
        device=present[0].device,
    )
    for index, value in enumerate(values):
        if value is not None:
            output[index, : value.shape[0]] = value.to(dtype=torch.long)
    return output


def _pad_explicit_masks(
    masks: Sequence[Tensor | None], values: Sequence[Tensor]
) -> Tensor:
    filled = [
        (
            torch.ones(len(value), dtype=torch.bool, device=value.device)
            if mask is None
            else mask.to(dtype=torch.bool, device=value.device)
        )
        for mask, value in zip(masks, values, strict=True)
    ]
    output, _ = _pad_temporal(filled, "explicit_valid_mask")
    return output


# Short alias retained for configuration files.
TrainingCollator = NativeTrainingCollator


__all__ = ["NativeTrainingCollator", "TrainingCollator"]
