"""Online VLM understanding refresh and cache policy."""

from typing import Any, Optional

import torch

from wam.model.modules.memory.proprio_encoder import append_proprio_to_context
from wam.model.modules.understanding.sequence_conditioning import (
    build_vlm_condition,
)

def _get_online_vlm_pack(
    model,
    *,
    vlm_current_images: torch.Tensor,
    proprio: torch.Tensor,
    prompt: str,
    vlm_view_names: Any = None,
    vlm_current_view_is_pad: Optional[torch.Tensor] = None,
) -> Optional[dict[str, torch.Tensor]]:
    if not model.understanding_enabled:
        return None
    vlm_pack = build_vlm_condition(
        model,
        frames=vlm_current_images.unsqueeze(0).unsqueeze(1),
        prompts=[str(prompt)],
        view_names=[str(vlm_view_names or "")],
        frame_labels=["current"],
        view_valid_mask=(
            None
            if vlm_current_view_is_pad is None
            or not bool(
                model.understanding_cfg.get("use_view_valid_mask", True)
            )
            else ~torch.as_tensor(
                vlm_current_view_is_pad,
                device="cpu",
                dtype=torch.bool,
            ).reshape(1, 1, -1)
        ),
    )
    if vlm_pack is not None:
        vlm_context, vlm_mask = append_proprio_to_context(
            model,
            context=vlm_pack["vlm_context"],
            context_mask=vlm_pack["vlm_mask"],
            proprio=proprio,
            proprio_encoder=model.action_proprio_encoder,
        )
        vlm_pack["vlm_context"] = vlm_context
        vlm_pack["vlm_mask"] = vlm_mask
        vlm_pack["vlm_mask_all_valid"] = bool(vlm_mask.all().item())
    return vlm_pack
