from __future__ import annotations

from typing import Dict, Mapping

import torch

from psi_policy.model.observation.fusions.base import BaseFusion


class IdentityFusion(BaseFusion):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(
        self,
        *,
        token_map: Mapping[str, torch.Tensor],
        obs_dict: Mapping[str, torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del obs_dict, training
        return {k: v for k, v in token_map.items()}


__all__ = ["IdentityFusion"]
