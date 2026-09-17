from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torch import nn

from worldscape_policy.conditioning import (
    AutoConditioner,
    ConditionRouter,
    InteractiveConditioner,
)
from worldscape_policy.memory.visual import VisualPrefillManager, WanVisualCodec
from worldscape_policy.memory.visual.normalization import VisualInputRange
from worldscape_policy.policy import WorldScapePolicy
from worldscape_policy.types import InteractionMode
from worldscape_policy.wam.registry import DEFAULT_WAM_REGISTRY, WAMRegistry
from worldscape_policy.wam.wan22 import (
    Wan22DistributedConfig,
    Wan22DistributedContext,
    Wan22KernelConfig,
    Wan22LegacyExactKernel,
    Wan22WAMConfig,
)
from worldscape_policy.wam.wan22.plugin import Wan22NumericalKernel


@dataclass(frozen=True)
class Wan22PolicyBuildConfig:
    num_frames: int
    persistent_prompt: str = "goal_or_demo"
    semantic_gate_only: bool = False
    semantic_grad_clip_norm: float = 0.5
    max_history_steps: int = 16
    view_index: int = 0
    tiled: bool = False
    tile_size: tuple[int, int] = (34, 34)
    tile_stride: tuple[int, int] = (18, 16)
    visual_input_range: VisualInputRange = "zero_one"
    diffusion_view_layout: Literal[
        "mosaic_2x2", "robotwin_concat"
    ] = "mosaic_2x2"
    distributed: Wan22DistributedConfig = field(
        default_factory=Wan22DistributedConfig
    )


def build_wan22_policy(
    *,
    config: Wan22PolicyBuildConfig,
    vlm: nn.Module,
    token_pooler: nn.Module,
    projector: nn.Module,
    event_memory: nn.Module,
    t5: nn.Module,
    vae: nn.Module,
    core: nn.Module,
    image_encoder: nn.Module,
    numerical_kernel: Wan22NumericalKernel | None = None,
    kernel_config: Wan22KernelConfig | None = None,
    output_norm: nn.Module | None = None,
    configured_mode: InteractionMode | str | None = None,
    wam_registry: WAMRegistry = DEFAULT_WAM_REGISTRY,
    distributed_context: Wan22DistributedContext | None = None,
) -> WorldScapePolicy:
    """Build the one canonical native module ownership tree."""

    ownership_modules = {
        "vlm": vlm,
        "token_pooler": token_pooler,
        "projector": projector,
        "event_memory": event_memory,
        "t5": t5,
        "vae": vae,
        "core": core,
        "image_encoder": image_encoder,
    }
    if output_norm is not None:
        ownership_modules["output_norm"] = output_norm
    _assert_disjoint_registered_modules(**ownership_modules)
    if numerical_kernel is None:
        if kernel_config is None:
            raise ValueError("kernel_config is required when numerical_kernel is omitted")
        numerical_kernel = Wan22LegacyExactKernel(
            kernel_config,
            distributed_context=distributed_context,
        )

    auto = AutoConditioner(
        vlm=vlm,
        token_pooler=token_pooler,
        projector=projector,
        event_memory=event_memory,
        output_norm=output_norm,
        max_history_steps=config.max_history_steps,
        visual_input_range=config.visual_input_range,
        semantic_gate_only=config.semantic_gate_only,
        semantic_grad_clip_norm=config.semantic_grad_clip_norm,
    )
    router = ConditionRouter(
        auto_conditioner=auto,
        interactive_conditioner=InteractiveConditioner(t5=t5),
    )
    codec = WanVisualCodec(
        vae,
        view_index=config.view_index,
        tiled=config.tiled,
        tile_size=config.tile_size,
        tile_stride=config.tile_stride,
        visual_input_range=config.visual_input_range,
        diffusion_view_layout=config.diffusion_view_layout,
    )
    visual_memory = VisualPrefillManager(
        codec,
        persistent_prompt=config.persistent_prompt,
    )
    wam = wam_registry.construct(
        Wan22WAMConfig(
            num_frames=config.num_frames,
            tiled=config.tiled,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
            visual_input_range=config.visual_input_range,
            distributed=(
                distributed_context.config
                if distributed_context is not None
                else config.distributed
            ),
        ),
        required_capabilities={"training", "sampling"},
        core=core,
        image_encoder=image_encoder,
        visual_codec_provider=lambda: visual_memory.codec,
        numerical_kernel=numerical_kernel,
        distributed_context=distributed_context,
    )
    return WorldScapePolicy(
        condition_router=router,
        visual_memory=visual_memory,
        wam=wam,
        configured_mode=configured_mode,
    )


def _assert_disjoint_registered_modules(**owners: nn.Module) -> None:
    seen: dict[int, str] = {}
    for owner_name, owner in owners.items():
        for module in owner.modules():
            previous = seen.get(id(module))
            if previous is not None:
                raise ValueError(
                    f"Registered module is shared by {previous!r} and "
                    f"{owner_name!r}; native checkpoint ownership must be unique"
                )
            seen[id(module)] = owner_name
