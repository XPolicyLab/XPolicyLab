from __future__ import annotations

import pytest
import torch
from torch import nn

from worldscape_policy.memory.visual import VisualPrefillManager
from worldscape_policy.types import PromptBatch, VisualMemoryState, WAMInferenceState


class _Codec(nn.Module):
    def encode_visual(self, video):
        return video.float().mean(dim=tuple(range(1, video.ndim))).reshape(-1, 1, 1)


class _MosaicCodec(_Codec):
    def __init__(self) -> None:
        super().__init__()
        self.prepared_observations = False

    def prepare_diffusion_video(self, video):
        self.prepared_observations = True
        return video.float().sum(dim=2)

    def encode_normalized(self, video):
        return video


def test_visual_prefill_uses_diffusion_mosaic_path_for_three_view_observations() -> None:
    codec = _MosaicCodec()
    manager = VisualPrefillManager(codec)
    images = torch.ones(1, 2, 3, 3, 4, 5)

    encoded = manager.encode_recent_observations(images)

    assert codec.prepared_observations
    assert encoded.shape == (1, 2, 3, 4, 5)
    torch.testing.assert_close(encoded, torch.full_like(encoded, 3.0))


def test_visual_prefill_keeps_prompt_and_refreshes_recent_observations() -> None:
    manager = VisualPrefillManager(_Codec())
    prompt = torch.full((1, 1, 3, 2, 2), 3.0)
    first = manager.prepare(
        images=torch.ones(1, 1, 3, 2, 2),
        prompts=PromptBatch(vlm_planning_text=["plan"], goal_images=prompt),
    )
    second = manager.prepare(
        images=torch.full((1, 1, 3, 2, 2), 5.0),
        prompts=PromptBatch(vlm_planning_text=["plan"]),
        previous_state=first,
    )
    assert second.persistent_prompt_latents is first.persistent_prompt_latents
    assert second.persistent_prompt_version == first.persistent_prompt_version
    torch.testing.assert_close(
        second.recent_observation_latents,
        torch.full((1, 1, 1), 5.0),
    )


def test_visual_prefill_prompt_change_resets_wam_state() -> None:
    manager = VisualPrefillManager(_Codec())
    previous = VisualMemoryState(
        persistent_prompt_latents=torch.ones(1, 1, 1),
        persistent_prompt_version=4,
        wam_state=WAMInferenceState(current_start_frame=3),
    )
    changed = manager.prepare(
        images=torch.ones(1, 1, 3, 2, 2),
        prompts=PromptBatch(
            vlm_planning_text=["plan"],
            goal_images=torch.full((1, 1, 3, 2, 2), 2.0),
        ),
        previous_state=previous,
    )
    assert changed.persistent_prompt_version == 5
    assert changed.wam_state == WAMInferenceState()


def test_visual_prefill_rejects_two_persistent_prompt_modalities() -> None:
    manager = VisualPrefillManager(_Codec())
    visual = torch.ones(1, 1, 3, 2, 2)
    with pytest.raises(ValueError, match="Only one"):
        manager.encode_persistent_prompt(visual, visual)
