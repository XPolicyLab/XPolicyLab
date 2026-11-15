"""Embodied policy inference framework.

Usage:
    from soma.inference.config import EmbodiedConfig
    from soma.inference import build_policy

    cfg = EmbodiedConfig(checkpoint_path="/path/to/ckpt")
    cfg.load_model_config()  # auto-discovers config.yaml
    policy = build_policy(cfg, device="cuda")
    result = policy.infer(video_frame, instruction, seed=42)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from soma.inference.config import EmbodiedConfig
    from soma.inference.policies.base import BasePolicy


def build_policy(
    cfg: EmbodiedConfig,
    device: str | torch.device = "cuda",
) -> BasePolicy:
    """Build policy from EmbodiedConfig.

    Args:
        cfg: EmbodiedConfig dataclass. Must have load_model_config() called first.
        device: torch device string.
    """
    # Control-plane imports must bind the HTTP endpoint before loading torch.
    from soma.inference.policies.base import BasePolicy

    if not cfg.model_kwargs:
        cfg.load_model_config()

    policy_type = cfg.policy.type
    if isinstance(policy_type, str):
        from falcon.utils.locate import locate

        policy_type = locate(policy_type)
    policy_factory = cast(Callable[..., BasePolicy], policy_type)
    return policy_factory(cfg, device=device)
