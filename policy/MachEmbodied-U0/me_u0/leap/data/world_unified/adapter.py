"""Pure-torch adapter from source dictionaries to the canonical contract."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from leap.data.world_unified.types import (
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    RGB_CHANNELS,
    WorldUnifiedSample,
)


def _tensor(sample: Mapping[str, Any], *keys: str) -> Tensor:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, Tensor):
            return value
    raise KeyError(f"sample must provide a tensor under one of {keys}")


def _valid_from_pad(
    value: Any, *, length: int, name: str
) -> Tensor:
    if isinstance(value, Mapping):
        masks = [torch.as_tensor(mask, dtype=torch.bool) for mask in value.values()]
        if not masks:
            raise ValueError(f"`{name}` camera-mask mapping cannot be empty")
        reference = masks[0]
        for mask in masks[1:]:
            if not torch.equal(reference, mask):
                raise ValueError(
                    f"all camera streams in `{name}` must share temporal padding"
                )
        value = reference
    mask = torch.as_tensor(value, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] < length:
        raise ValueError(
            f"`{name}` must contain at least {length} entries, got "
            f"{tuple(mask.shape)}"
        )
    return ~mask[:length]


def _explicit_valid_mask(
    sample: Mapping[str, Any],
    *,
    valid_key: str,
    pad_key: str,
    length: int,
) -> Optional[Tensor]:
    if valid_key in sample:
        mask = torch.as_tensor(sample[valid_key], dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != length:
            raise ValueError(
                f"`{valid_key}` must have shape [{length}], got {tuple(mask.shape)}"
            )
        return mask
    if pad_key in sample:
        return _valid_from_pad(sample[pad_key], length=length, name=pad_key)
    return None


def _prefix_valid(length: int, valid_prefix: Optional[int], name: str) -> Tensor:
    if valid_prefix is None:
        return torch.ones(length, dtype=torch.bool)
    if not 1 <= valid_prefix <= length:
        raise ValueError(f"`{name}` must be in [1,{length}], got {valid_prefix}")
    mask = torch.zeros(length, dtype=torch.bool)
    mask[:valid_prefix] = True
    return mask


def canonical_sample_from_mapping(
    sample: Mapping[str, Any],
    *,
    domain_id: int,
    num_cameras: int,
    valid_action_dim: Optional[int] = None,
    valid_state_dim: Optional[int] = None,
    max_action_dim: int = MAX_ACTION_DIM,
    max_state_dim: int = MAX_STATE_DIM,
) -> WorldUnifiedSample:
    """Convert one decoded sample without tokenization or external packages.

    Accepted video layouts are ``[3,T,H,W]`` and the legacy Lance
    ``[T,3,H,W]``. A legacy ``proprio`` trajectory is reduced to its
    first/current state; the canonical object never retains a state sequence.
    Invalid time/dimension entries are zeroed before strict validation.
    """

    video = _tensor(sample, "video")
    if video.ndim != 4:
        raise ValueError(f"`video` must be rank 4, got {tuple(video.shape)}")
    if video.shape[1] == RGB_CHANNELS and video.shape[0] != RGB_CHANNELS:
        video = video.permute(1, 0, 2, 3)
    video = video.float()
    if video.shape[0] != RGB_CHANNELS or video.shape[1] < 2:
        raise ValueError(
            "source video must already be decoded/resized to [3,T,H,W], "
            f"got {tuple(video.shape)}"
        )
    rgb_frames = int(video.shape[1])

    action = _tensor(sample, "action").float()
    if action.ndim != 2 or action.shape[0] < 1:
        raise ValueError(
            "`action` must have shape [T,D] with T >= 1, got "
            f"{tuple(action.shape)}"
        )
    action_steps = int(action.shape[0])
    if isinstance(max_action_dim, bool) or not isinstance(max_action_dim, int):
        raise TypeError("`max_action_dim` must be an integer")
    if not 1 <= action.shape[1] <= max_action_dim:
        raise ValueError(
            f"source action dimension must be in [1,{max_action_dim}], got "
            f"{action.shape[1]}"
        )

    if isinstance(sample.get("state"), Tensor):
        state = sample["state"]
        if state.ndim != 1:
            raise ValueError(
                "`state` must be the current rank-1 vector; do not pass a state "
                f"trajectory (got {tuple(state.shape)})"
            )
    else:
        proprio = _tensor(sample, "proprio")
        if proprio.ndim == 1:
            state = proprio
        elif proprio.ndim == 2 and proprio.shape[0] > 0:
            state = proprio[0]
        else:
            raise ValueError(
                "`proprio` must be [D] or a non-empty legacy [T,D] trajectory"
            )
    state = state.float()
    if isinstance(max_state_dim, bool) or not isinstance(max_state_dim, int):
        raise TypeError("`max_state_dim` must be an integer")
    if not 1 <= state.shape[0] <= max_state_dim:
        raise ValueError(
            f"source state dimension must be in [1,{max_state_dim}], got "
            f"{state.shape[0]}"
        )

    if "instruction" in sample:
        instruction = sample["instruction"]
    elif "task" in sample:
        instruction = sample["task"]
    else:
        raise KeyError(
            "sample must provide raw `instruction` (or raw `task`); templated "
            "`prompt` is deliberately not accepted"
        )
    if not isinstance(instruction, str):
        raise TypeError("raw instruction must be a string")

    video_valid = _explicit_valid_mask(
        sample,
        valid_key="video_time_valid_mask",
        pad_key="video_is_pad",
        length=rgb_frames,
    )
    if video_valid is None and "image_is_pad" in sample:
        video_valid = _valid_from_pad(
            sample["image_is_pad"], length=rgb_frames, name="image_is_pad"
        )
    if video_valid is None:
        video_valid = torch.ones(rgb_frames, dtype=torch.bool)

    action_time_valid = _explicit_valid_mask(
        sample,
        valid_key="action_time_valid_mask",
        pad_key="action_is_pad",
        length=action_steps,
    )
    if action_time_valid is None:
        action_time_valid = torch.ones(action_steps, dtype=torch.bool)

    action_dim_valid = _explicit_valid_mask(
        sample,
        valid_key="action_dim_valid_mask",
        pad_key="action_dim_is_pad",
        length=action.shape[1],
    )
    if action_dim_valid is None:
        action_mask = sample.get("action_mask")
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            if action_mask.shape != action.shape:
                raise ValueError(
                    "`action_mask` must have the same [T,D] shape as action"
                )
            action_dim_valid = action_mask.any(dim=0)
            mask_time_valid = action_mask.any(dim=1)
            expected = mask_time_valid[:, None] & action_dim_valid[None, :]
            if not torch.equal(action_mask, expected):
                raise ValueError(
                    "`action_mask` must factor into independent time and dimension masks"
                )
            action_time_valid &= mask_time_valid
        else:
            action_dim_valid = _prefix_valid(
                action.shape[1], valid_action_dim, "valid_action_dim"
            )

    state_dim_valid = _explicit_valid_mask(
        sample,
        valid_key="state_dim_valid_mask",
        pad_key="state_dim_is_pad",
        length=state.shape[0],
    )
    if state_dim_valid is None:
        state_dim_valid = _prefix_valid(
            state.shape[0], valid_state_dim, "valid_state_dim"
        )

    # Own the tensors and make invalid padding observably inert at the data
    # boundary.  The collator repeats this masking after pad-to-26 as defense in
    # depth, but canonical samples already satisfy the invariant themselves.
    video = video.clone()
    action = action.clone()
    state = state.clone()
    video[:, ~video_valid] = 0
    action[~action_time_valid] = 0
    action[:, ~action_dim_valid] = 0
    state[~state_dim_valid] = 0

    metadata = dict(sample.get("metadata") or {})
    if "_debug_meta" in sample:
        metadata["debug"] = sample["_debug_meta"]
    return WorldUnifiedSample(
        video=video,
        action=action,
        state=state,
        instruction=instruction,
        domain_id=domain_id,
        num_cameras=num_cameras,
        text_metadata=sample.get("text_metadata"),
        video_time_valid_mask=video_valid,
        action_time_valid_mask=action_time_valid,
        action_dim_valid_mask=action_dim_valid,
        state_dim_valid_mask=state_dim_valid,
        metadata=metadata,
        main_image=(
            torch.as_tensor(sample["main_image"]).float()
            if sample.get("main_image") is not None
            else None
        ),
        max_action_dim=max_action_dim,
        max_state_dim=max_state_dim,
    )
