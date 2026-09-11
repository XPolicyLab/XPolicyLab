from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import torch
from torch import Tensor, nn

from worldscape_policy.memory.visual.normalization import VisualInputRange
from worldscape_policy.types import (
    VisualMemoryState,
    WAMInferenceState,
    WanI2VCondition,
    WorldActionOutput,
)
from worldscape_policy.wam.protocol import VisualCodec, VisualCodecProvider
from worldscape_policy.wam.wan22.distributed import (
    Wan22DistributedConfig,
    Wan22DistributedContext,
)
from worldscape_policy.wam.wan22.image_conditioning import Wan22ImageConditioner


@dataclass(frozen=True)
class Wan22WAMConfig:
    num_frames: int
    tiled: bool = False
    tile_size: tuple[int, int] = (34, 34)
    tile_stride: tuple[int, int] = (18, 16)
    visual_input_range: VisualInputRange = "zero_one"
    distributed: Wan22DistributedConfig = field(
        default_factory=Wan22DistributedConfig
    )


class Wan22NumericalKernel(Protocol):
    """Unchanged Wan2.2 flow/caching implementation behind the native boundary."""

    def training_forward(
        self,
        *,
        core: nn.Module,
        i2v_condition: WanI2VCondition,
        clean_video_latents: Tensor,
        clean_action: Tensor,
        noisy_video: Tensor,
        noisy_action: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        recent_visual_prefill: Tensor | None,
    ) -> WorldActionOutput: ...

    def sample(
        self,
        *,
        core: nn.Module,
        i2v_condition: WanI2VCondition,
        anchor_latent: Tensor | None,
        chunk_latents: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        inference_state: WAMInferenceState,
        generator: torch.Generator,
    ) -> tuple[WorldActionOutput, WAMInferenceState]: ...


class Wan22WAMPlugin(nn.Module):
    """Own native Wan modules while delegating unchanged numerical operations.

    The visual codec is obtained through a non-owning provider so its VAE
    remains registered exactly once under ``visual_memory.codec.vae``.
    """

    def __init__(
        self,
        *,
        core: nn.Module,
        image_encoder: nn.Module,
        visual_codec_provider: VisualCodecProvider,
        numerical_kernel: Wan22NumericalKernel,
        image_conditioner: Wan22ImageConditioner,
        distributed_context: Wan22DistributedContext | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(core, nn.Module):
            raise TypeError("core must be an nn.Module")
        if not isinstance(image_encoder, nn.Module):
            raise TypeError("image_encoder must be an nn.Module")
        if not callable(visual_codec_provider):
            raise TypeError("visual_codec_provider must be callable")
        if not hasattr(numerical_kernel, "training_forward") or not hasattr(
            numerical_kernel, "sample"
        ):
            raise TypeError("numerical_kernel must implement training_forward and sample")
        self.core = core
        self.image_encoder = image_encoder
        self._visual_codec_provider = visual_codec_provider
        self._numerical_kernel = numerical_kernel
        self._image_conditioner = image_conditioner
        kernel_context = getattr(numerical_kernel, "_distributed", None)
        self._distributed = (
            distributed_context
            or (kernel_context if isinstance(kernel_context, Wan22DistributedContext) else None)
            or Wan22DistributedContext.single_rank()
        )
        if (
            distributed_context is not None
            and isinstance(kernel_context, Wan22DistributedContext)
            and kernel_context is not distributed_context
        ):
            raise ValueError("plugin and kernel must share one distributed context")
        set_context = getattr(core, "set_image_parallel_context", None)
        if callable(set_context):
            set_context(self._distributed)
        elif self._distributed.world_size > 1:
            raise RuntimeError(
                "multi-rank Wan22 core must implement set_image_parallel_context"
            )

    def _codec(self) -> VisualCodec:
        codec = self._visual_codec_provider()
        if not isinstance(codec, nn.Module) or not isinstance(codec, VisualCodec):
            raise TypeError("visual_codec_provider must return a visual codec module")
        return codec

    def _vae(self) -> nn.Module:
        vae = getattr(self._codec(), "vae", None)
        if not isinstance(vae, nn.Module):
            raise TypeError("visual codec must expose its VAE as .vae")
        return vae

    def _core_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.core.parameters(), None)
        if parameter is None:
            return torch.device("cpu"), torch.float32
        return parameter.device, parameter.dtype

    def _move_core_inputs(
        self,
        *,
        i2v_condition: WanI2VCondition,
        video: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        positive: Tensor,
        negative: Tensor | None,
    ) -> tuple[WanI2VCondition, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        device, dtype = self._core_device_dtype()
        return (
            WanI2VCondition(
                clip_features=i2v_condition.clip_features.to(device=device, dtype=dtype),
                masked_latent_y=i2v_condition.masked_latent_y.to(
                    device=device, dtype=dtype
                ),
            ),
            video.to(device=device, dtype=dtype),
            state.to(device=device, dtype=dtype),
            embodiment_id.to(device=device),
            positive.to(device=device, dtype=dtype),
            negative.to(device=device, dtype=dtype) if negative is not None else None,
        )

    @staticmethod
    def _reference_from_video(clean_video: Tensor) -> Tensor:
        if clean_video.ndim != 5:
            raise ValueError("clean_video must have shape [B,T,C,H,W]")
        if clean_video.shape[2] not in (1, 3, 4):
            raise ValueError("clean_video must use the [B,T,C,H,W] public layout")
        return clean_video[:, :1]

    def starts_new_window(
        self,
        visual_memory: VisualMemoryState | None,
        *,
        observation_num_frames: int,
    ) -> bool:
        if visual_memory is None or visual_memory.wam_state.current_start_frame == 0:
            return False
        prepare_state = getattr(self._numerical_kernel, "prepare_inference_state", None)
        if not callable(prepare_state):
            return False
        prepared = prepare_state(
            core=self.core,
            inference_state=visual_memory.wam_state,
            observation_num_frames=observation_num_frames,
        )
        return (
            prepared.current_start_frame == 0
            or prepared.rebase_observation_window
        )

    def training_forward(
        self,
        *,
        clean_video: Tensor,
        clean_action: Tensor,
        noisy_video: Tensor,
        noisy_action: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        recent_visual_prefill: Tensor | None,
        clean_video_latents: Tensor | None = None,
        clean_video_normalized: bool = False,
    ) -> WorldActionOutput:
        if self._distributed.world_size > 1:
            raise RuntimeError(
                "Wan22 multi-rank training is not implemented; distributed WAM "
                "support is inference-only"
            )
        reference_frame = self._reference_from_video(clean_video)
        i2v_condition, _ = self._image_conditioner.encode(
            reference_frame=reference_frame,
            image_encoder=self.image_encoder,
            vae=self._vae(),
            normalized=clean_video_normalized,
        )
        if clean_video_latents is None:
            clean_video_latents = self._codec().encode_visual(clean_video)
        (
            i2v_condition,
            clean_video_latents,
            state,
            embodiment_id,
            cross_attention_tokens,
            negative_cross_attention_tokens,
        ) = self._move_core_inputs(
            i2v_condition=i2v_condition,
            video=clean_video_latents,
            state=state,
            embodiment_id=embodiment_id,
            positive=cross_attention_tokens,
            negative=negative_cross_attention_tokens,
        )
        device, dtype = self._core_device_dtype()
        clean_action = clean_action.to(device=device, dtype=dtype)
        noisy_video = noisy_video.to(device=device, dtype=dtype)
        noisy_action = noisy_action.to(device=device, dtype=dtype)
        video_timestep = video_timestep.to(device=device)
        action_timestep = action_timestep.to(device=device)
        persistent_prefill = (
            persistent_prefill.to(device=device, dtype=dtype)
            if persistent_prefill is not None
            else None
        )
        recent_visual_prefill = (
            recent_visual_prefill.to(device=device, dtype=dtype)
            if recent_visual_prefill is not None
            else None
        )
        output = self._numerical_kernel.training_forward(
            core=self.core,
            i2v_condition=i2v_condition,
            clean_video_latents=clean_video_latents,
            clean_action=clean_action,
            noisy_video=noisy_video,
            noisy_action=noisy_action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            state=state,
            embodiment_id=embodiment_id,
            cross_attention_tokens=cross_attention_tokens,
            negative_cross_attention_tokens=negative_cross_attention_tokens,
            persistent_prefill=persistent_prefill,
            recent_visual_prefill=recent_visual_prefill,
        )
        if not isinstance(output, WorldActionOutput):
            raise TypeError("Wan22 numerical kernel must return WorldActionOutput")
        return output

    @torch.no_grad()
    def sample(
        self,
        *,
        reference_frame: Tensor,
        reference_frame_normalized: bool = False,
        chunk_latents: Tensor,
        observation_num_frames: int,
        prompt_signature: tuple[str, ...],
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        visual_memory: VisualMemoryState,
        generator: torch.Generator,
    ) -> WorldActionOutput:
        self._distributed.coordinate("plugin.sample.enter")
        inference_state = visual_memory.wam_state.fork()
        self._distributed.validate_state_owner(
            inference_state.cache_owner_rank,
            inference_state.cache_world_size,
        )
        prepare_state = getattr(self._numerical_kernel, "prepare_inference_state", None)
        if callable(prepare_state):
            inference_state = prepare_state(
                core=self.core,
                inference_state=inference_state,
                observation_num_frames=observation_num_frames,
            )
        if self._condition_changed(
            inference_state,
            cross_attention_tokens,
            negative_cross_attention_tokens,
        ) or (
            inference_state.current_start_frame != 0
            and inference_state.prompt_signature is not None
            and inference_state.prompt_signature != prompt_signature
        ):
            inference_state = WAMInferenceState()
        anchor_latent = None
        i2v_condition = inference_state.i2v_condition
        if i2v_condition is None or inference_state.current_start_frame == 0:
            i2v_condition, anchor_latent = self._image_conditioner.encode(
                reference_frame=reference_frame,
                image_encoder=self.image_encoder,
                vae=self._vae(),
                normalized=reference_frame_normalized,
            )
            inference_state = replace(
                inference_state,
                i2v_condition=i2v_condition,
            )
        (
            i2v_condition,
            chunk_latents,
            state,
            embodiment_id,
            cross_attention_tokens,
            negative_cross_attention_tokens,
        ) = self._move_core_inputs(
            i2v_condition=i2v_condition,
            video=chunk_latents,
            state=state,
            embodiment_id=embodiment_id,
            positive=cross_attention_tokens,
            negative=negative_cross_attention_tokens,
        )
        inference_state = replace(
            inference_state,
            i2v_condition=i2v_condition,
            condition_tokens=cross_attention_tokens.detach().clone(),
            negative_condition_tokens=(
                negative_cross_attention_tokens.detach().clone()
                if negative_cross_attention_tokens is not None
                else None
            ),
            prompt_signature=prompt_signature,
        )
        device, dtype = self._core_device_dtype()
        persistent_prefill = visual_memory.persistent_prompt_latents
        persistent_prefill = (
            persistent_prefill.to(device=device, dtype=dtype)
            if persistent_prefill is not None
            else None
        )

        output, next_inference_state = self._numerical_kernel.sample(
            core=self.core,
            i2v_condition=i2v_condition,
            anchor_latent=anchor_latent,
            chunk_latents=chunk_latents,
            state=state,
            embodiment_id=embodiment_id,
            cross_attention_tokens=cross_attention_tokens,
            negative_cross_attention_tokens=negative_cross_attention_tokens,
            persistent_prefill=persistent_prefill,
            inference_state=inference_state,
            generator=generator,
        )
        if not isinstance(output, WorldActionOutput):
            raise TypeError("Wan22 numerical kernel must return WorldActionOutput")
        if not isinstance(next_inference_state, WAMInferenceState):
            raise TypeError("Wan22 numerical kernel must return WAMInferenceState")
        next_inference_state = replace(
            next_inference_state,
            cache_owner_rank=self._distributed.rank,
            cache_world_size=self._distributed.world_size,
        )
        output.next_visual_memory = VisualMemoryState(
            persistent_prompt_latents=visual_memory.persistent_prompt_latents,
            persistent_prompt_version=visual_memory.persistent_prompt_version,
            recent_observation_latents=visual_memory.recent_observation_latents,
            wam_state=next_inference_state,
        )
        self._distributed.coordinate("plugin.sample.exit")
        return output

    def reset_episode(self) -> None:
        self._distributed.reset_episode()
        reset = getattr(self._numerical_kernel, "reset_episode", None)
        if callable(reset):
            reset()

    @staticmethod
    def _condition_changed(
        inference_state: WAMInferenceState,
        positive: Tensor,
        negative: Tensor | None,
    ) -> bool:
        if inference_state.current_start_frame == 0:
            return False
        previous_positive = inference_state.condition_tokens
        previous_negative = inference_state.negative_condition_tokens
        if previous_positive is None:
            return False
        comparable_positive = positive.to(
            device=previous_positive.device,
            dtype=previous_positive.dtype,
        )
        if (
            previous_positive.shape != comparable_positive.shape
            or not torch.equal(previous_positive, comparable_positive)
        ):
            return True
        if previous_negative is None:
            return negative is not None
        if negative is None:
            return True
        comparable_negative = negative.to(
            device=previous_negative.device,
            dtype=previous_negative.dtype,
        )
        return (
            previous_negative.shape != comparable_negative.shape
            or not torch.equal(previous_negative, comparable_negative)
        )
