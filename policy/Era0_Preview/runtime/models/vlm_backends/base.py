"""Backend contract for semantic VLM plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Mapping

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..configuration_xvla import XVLAConfig
    from ..modeling_xvla import XVLA


class VLMBackend(ABC):
    """A stateless adapter around the registered ``XVLA.vlm`` module."""

    name: str
    handles_expert_detach: bool = False
    action_context_mode: str = "legacy_joint_v1"
    policy_contract: str = "legacy_v1"

    @abstractmethod
    def build_model(self, config: "XVLAConfig") -> nn.Module:
        """Construct the module registered under the legacy ``vlm.*`` prefix."""

    @abstractmethod
    def native_hidden_size(self, model: nn.Module) -> int:
        """Return the backbone hidden size before the XVLA output adapter."""

    def output_hidden_size(self, config: "XVLAConfig", model: nn.Module) -> int:
        configured = int(getattr(config, "vlm_projection_dim", 0) or 0)
        return configured or self.native_hidden_size(model)

    def build_output_adapter(
        self,
        config: "XVLAConfig",
        model: nn.Module,
    ) -> nn.Module:
        native_dim = self.native_hidden_size(model)
        output_dim = self.output_hidden_size(config, model)
        if native_dim == output_dim:
            return nn.Identity()
        adapter = nn.Linear(native_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(adapter.weight)
        return adapter

    def build_policy_modules(
        self,
        config: "XVLAConfig",
        *,
        model: nn.Module,
        output_dim: int,
    ) -> nn.ModuleDict:
        """Construct optional backend-owned modules outside ``vlm.*``."""
        return nn.ModuleDict()

    def build_geometry_adapter(
        self,
        config: "XVLAConfig",
        model: nn.Module,
    ) -> nn.Module:
        """Project backend-native action geometry into the policy width."""
        return nn.Identity()

    def precompute_policy_visual(
        self,
        owner: "XVLA",
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None,
    ) -> Any:
        """Optionally cache backend-native visual features within one forward."""
        return None

    @abstractmethod
    def has_lm_head(self, model: nn.Module) -> bool:
        """Whether caption/FAST causal or seq2seq CE is available."""

    @abstractmethod
    def encode_image(
        self,
        owner: "XVLA",
        pixel_values: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode canonical ImageNet-normalized RGB into policy visual tokens."""

    @abstractmethod
    def forward_policy(
        self,
        owner: "XVLA",
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        precomputed_features: Any,
    ) -> Dict[str, torch.Tensor]:
        """Return the semantic and auxiliary token streams consumed by XVLA."""




    def validate_runtime(self, config: "XVLAConfig") -> None:
        """Validate backend-specific config before heavy model use."""

    def describe(self, config: "XVLAConfig", model: nn.Module) -> Dict[str, Any]:
        return {
            "name": self.name,
            "native_hidden_size": self.native_hidden_size(model),
            "output_hidden_size": self.output_hidden_size(config, model),
            "has_lm_head": self.has_lm_head(model),
            "action_context_mode": getattr(
                config,
                "action_context_mode",
                self.action_context_mode,
            ),
            "policy_contract": getattr(
                config,
                "vlm_policy_contract",
                self.policy_contract,
            ),
            "policy_modules": [],
        }
