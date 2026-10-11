"""Dense configurable-width collation for planned world-unified samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from leap.data.world_unified.metadata import Metadata
from leap.data.world_unified.plan import WorldUnifiedPlan
from leap.data.world_unified.types import (
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    VAE_TEMPORAL_DOWNSAMPLE,
    WorldUnifiedSample,
)


@dataclass(frozen=True)
class PlannedWorldUnifiedSample:
    sample: WorldUnifiedSample
    plan: WorldUnifiedPlan


@dataclass
class WorldUnifiedBatch:
    """Canonical batch plus all masks needed by the model-side token packer."""

    video: Union[Tensor, Tuple[Tensor, ...]]
    action: Tensor
    state: Tensor
    domain_ids: Tensor
    task_ids: Tensor
    camera_counts: Tensor
    video_time_valid_mask: Tensor
    latent_valid_mask: Tensor
    action_time_valid_mask: Tensor
    action_dim_valid_mask: Tensor
    action_valid_mask: Tensor
    state_dim_valid_mask: Tensor
    video_condition_mask: Tensor
    video_loss_mask: Tensor
    latent_condition_mask: Tensor
    latent_loss_mask: Tensor
    action_condition_mask: Tensor
    action_loss_mask: Tensor
    action_transition_ids: Tensor
    action_substep_ids: Tensor
    instructions: List[str]
    text_metadata: List[Optional[Metadata]]
    main_images: Tuple[Tensor, ...]
    plans: Tuple[WorldUnifiedPlan, ...]

    def as_dict(self) -> Dict[str, object]:
        return dict(vars(self))


def _indexes_mask(indexes: Tuple[int, ...], length: int) -> Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if indexes:
        mask[list(indexes)] = True
    return mask


def _rgb_to_latent_valid(video_valid: Tensor) -> Tensor:
    """Map RGB validity flags to Wan's causal latent timesteps."""

    first = video_valid[:1]
    future = video_valid[1:]
    if future.numel() % VAE_TEMPORAL_DOWNSAMPLE:
        raise ValueError(
            "video validity is incompatible with Wan's temporal stride"
        )
    future = future.view(-1, VAE_TEMPORAL_DOWNSAMPLE)
    return torch.cat((first, future.any(dim=1)))


class WorldUnifiedCollator:
    """Pad action/current-state dimensions and materialize task masks.

    The legacy default remains 26 so existing data configs keep their model
    boundary. Unified-action-space configs opt into the schema maximum of 48.
    """

    def __init__(
        self,
        *,
        return_dict: bool = True,
        max_action_dim: int = MAX_ACTION_DIM,
        max_state_dim: int = MAX_STATE_DIM,
    ) -> None:
        self.return_dict = bool(return_dict)
        self.max_action_dim = int(max_action_dim)
        self.max_state_dim = int(max_state_dim)
        if self.max_action_dim < 1:
            raise ValueError(
                "max_action_dim must be positive, "
                f"got {self.max_action_dim}"
            )
        if self.max_state_dim < 1:
            raise ValueError(
                "max_state_dim must be positive, "
                f"got {self.max_state_dim}"
            )

    def __call__(
        self, items: Sequence[PlannedWorldUnifiedSample]
    ) -> Union[WorldUnifiedBatch, Dict[str, object]]:
        if not items:
            raise ValueError("Cannot collate an empty world-unified batch")
        if not all(isinstance(item, PlannedWorldUnifiedSample) for item in items):
            raise TypeError("collator items must be PlannedWorldUnifiedSample objects")

        batch_size = len(items)
        rgb_frames = items[0].sample.rgb_frames
        action_steps = items[0].sample.action_steps
        actions = torch.zeros(batch_size, action_steps, self.max_action_dim)
        states = torch.zeros(batch_size, self.max_state_dim)
        action_time_valid = torch.zeros(
            batch_size, action_steps, dtype=torch.bool
        )
        action_dim_valid = torch.zeros(
            batch_size, self.max_action_dim, dtype=torch.bool
        )
        state_dim_valid = torch.zeros(
            batch_size, self.max_state_dim, dtype=torch.bool
        )

        videos: List[Tensor] = []
        video_valid_masks: List[Tensor] = []
        latent_valid_masks: List[Tensor] = []
        video_condition_masks: List[Tensor] = []
        video_loss_masks: List[Tensor] = []
        latent_condition_masks: List[Tensor] = []
        latent_loss_masks: List[Tensor] = []
        action_condition_masks: List[Tensor] = []
        action_loss_masks: List[Tensor] = []
        transition_ids: List[Tensor] = []
        substep_ids: List[Tensor] = []
        domain_ids: List[int] = []
        task_ids: List[int] = []
        camera_counts: List[int] = []
        instructions: List[str] = []
        text_metadata: List[Optional[Metadata]] = []
        main_images: List[Tensor] = []
        for batch_index, item in enumerate(items):
            sample, plan = item.sample, item.plan
            if (
                sample.rgb_frames != rgb_frames
                or plan.rgb_frames != rgb_frames
                or sample.action_steps != action_steps
                or plan.action_steps != action_steps
            ):
                raise ValueError("sample temporal geometry conflicts with collator")
            videos.append(sample.video)
            action_dim = sample.action_dim
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"sample action_dim {action_dim} exceeds collator "
                    f"max_action_dim {self.max_action_dim}"
                )
            state_dim = sample.state_dim
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"sample state_dim {state_dim} exceeds collator "
                    f"max_state_dim {self.max_state_dim}"
                )
            actions[batch_index, :, :action_dim] = sample.action
            states[batch_index, :state_dim] = sample.state
            action_time_valid[batch_index] = sample.action_time_valid_mask
            action_dim_valid[batch_index, :action_dim] = sample.action_dim_valid_mask
            state_dim_valid[batch_index, :state_dim] = sample.state_dim_valid_mask

            video_valid = sample.video_time_valid_mask
            latent_valid = _rgb_to_latent_valid(video_valid)
            video_condition = (
                _indexes_mask(plan.clean_video_frames, rgb_frames) & video_valid
            )
            video_loss = _indexes_mask(plan.video_loss_frames, rgb_frames) & video_valid

            # Plan masks are constant over each causal VAE group.  Intersecting
            # with latent validity handles partially padded final trajectories
            # in the same way as Lance's temporal token-valid helper.
            latent_condition = _rgb_to_latent_valid(video_condition) & latent_valid
            latent_loss = _rgb_to_latent_valid(video_loss) & latent_valid

            action_step_condition = _indexes_mask(
                plan.clean_action_steps, action_steps
            )
            action_step_loss = _indexes_mask(
                plan.action_loss_steps, action_steps
            )
            sample_action_valid = (
                sample.action_time_valid_mask[:, None]
                & sample.action_dim_valid_mask[None, :]
            )
            padded_action_valid = torch.zeros(
                action_steps, self.max_action_dim, dtype=torch.bool
            )
            padded_action_valid[:, :action_dim] = sample_action_valid
            action_condition = action_step_condition[:, None] & padded_action_valid
            action_loss = action_step_loss[:, None] & padded_action_valid

            sample_transition = torch.zeros(action_steps, dtype=torch.long)
            sample_substep = torch.zeros(action_steps, dtype=torch.long)
            for step in range(action_steps):
                if sample.action_time_valid_mask[step]:
                    transition, substep = plan.action_temporal_position(step)
                    sample_transition[step] = transition
                    sample_substep[step] = substep

            video_valid_masks.append(video_valid)
            latent_valid_masks.append(latent_valid)
            video_condition_masks.append(video_condition)
            video_loss_masks.append(video_loss)
            latent_condition_masks.append(latent_condition)
            latent_loss_masks.append(latent_loss)
            action_condition_masks.append(action_condition)
            action_loss_masks.append(action_loss)
            transition_ids.append(sample_transition)
            substep_ids.append(sample_substep)
            domain_ids.append(sample.domain_id)
            task_ids.append(plan.task_id)
            camera_counts.append(sample.num_cameras)
            instructions.append(sample.instruction)
            text_metadata.append(sample.text_metadata)
            if sample.main_image is None:
                raise ValueError("ME_U0 requires a ViT main_image for every sample")
            main_images.append(sample.main_image)
        # Preserve heterogeneous camera mosaics without spatially padding one
        # embodiment into another.  Same-width source-homogeneous batches still
        # use a single dense tensor for efficient host-to-device transfer.
        if all(tuple(video.shape) == tuple(videos[0].shape) for video in videos):
            batched_video: Union[Tensor, Tuple[Tensor, ...]] = torch.stack(videos)
        else:
            batched_video = tuple(videos)

        action_valid = action_time_valid[:, :, None] & action_dim_valid[:, None, :]
        # Defense in depth: even hand-built canonical objects cannot leave data
        # in the collated invalid region or in the configured padding suffix.
        actions.masked_fill_(~action_valid, 0)
        states.masked_fill_(~state_dim_valid, 0)

        batch = WorldUnifiedBatch(
            video=batched_video,
            action=actions,
            state=states,
            domain_ids=torch.tensor(domain_ids, dtype=torch.long),
            task_ids=torch.tensor(task_ids, dtype=torch.long),
            camera_counts=torch.tensor(camera_counts, dtype=torch.long),
            video_time_valid_mask=torch.stack(video_valid_masks),
            latent_valid_mask=torch.stack(latent_valid_masks),
            action_time_valid_mask=action_time_valid,
            action_dim_valid_mask=action_dim_valid,
            action_valid_mask=action_valid,
            state_dim_valid_mask=state_dim_valid,
            video_condition_mask=torch.stack(video_condition_masks),
            video_loss_mask=torch.stack(video_loss_masks),
            latent_condition_mask=torch.stack(latent_condition_masks),
            latent_loss_mask=torch.stack(latent_loss_masks),
            action_condition_mask=torch.stack(action_condition_masks),
            action_loss_mask=torch.stack(action_loss_masks),
            action_transition_ids=torch.stack(transition_ids),
            action_substep_ids=torch.stack(substep_ids),
            instructions=instructions,
            text_metadata=text_metadata,
            main_images=tuple(main_images),
            plans=tuple(item.plan for item in items),
        )
        return batch.as_dict() if self.return_dict else batch
