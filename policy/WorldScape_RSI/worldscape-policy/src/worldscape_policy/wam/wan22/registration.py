from __future__ import annotations

from typing import Any

from torch import nn

from worldscape_policy.wam.protocol import VisualCodecProvider
from worldscape_policy.wam.registry import WAMRegistry
from worldscape_policy.wam.wan22.distributed import Wan22DistributedContext
from worldscape_policy.wam.wan22.image_conditioning import Wan22ImageConditioner
from worldscape_policy.wam.wan22.plugin import (
    Wan22NumericalKernel,
    Wan22WAMConfig,
    Wan22WAMPlugin,
)


def _build_wan22(
    *,
    config: Wan22WAMConfig,
    core: nn.Module,
    image_encoder: nn.Module,
    visual_codec_provider: VisualCodecProvider,
    numerical_kernel: Wan22NumericalKernel,
    distributed_context: Wan22DistributedContext | None = None,
    **unexpected: Any,
) -> Wan22WAMPlugin:
    if unexpected:
        raise TypeError(
            f"Unexpected Wan2.2 dependencies: {', '.join(sorted(unexpected))}"
        )
    if config.distributed.enabled and distributed_context is None:
        raise RuntimeError(
            "configured multi-rank WAM requires a distributed context and "
            "image-parallel backend"
        )
    if (
        distributed_context is not None
        and distributed_context.config != config.distributed
    ):
        raise ValueError("Wan22 WAM config and distributed context do not match")
    return Wan22WAMPlugin(
        core=core,
        image_encoder=image_encoder,
        visual_codec_provider=visual_codec_provider,
        numerical_kernel=numerical_kernel,
        distributed_context=distributed_context,
        image_conditioner=Wan22ImageConditioner(
            num_frames=config.num_frames,
            tiled=config.tiled,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
            visual_input_range=config.visual_input_range,
        ),
    )


def register_wan22(registry: WAMRegistry) -> None:
    registry.register(
        name="wan22",
        version="2.2",
        capabilities={
            "training",
            "sampling",
            "causal_cache",
            "image_conditioning",
            "distributed_inference",
        },
        config_type=Wan22WAMConfig,
        factory=_build_wan22,
    )


__all__ = ["register_wan22"]
