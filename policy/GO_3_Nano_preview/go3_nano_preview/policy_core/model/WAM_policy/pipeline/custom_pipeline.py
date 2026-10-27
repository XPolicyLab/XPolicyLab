# Copyright 2024 Lightricks and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring

from policy_core.model.WAM_policy.pipeline.scheduling_flow_match_euler_discrete_jit import FlowMatchEulerDiscreteSchedulerJiT
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


from einops import rearrange
from policy_core.utils.WAM_policy.data_utils import gen_noise_from_condition_frame_latent
from policy_core.utils.WAM_policy.video_sparse_utils import (
    bilinear_downsample_video,
    build_frame_latent_spec,
    build_mixed_conditioning_tokens,
    build_mixed_video_layout,
    expand_frame_indices_for_views,
    normalize_sparse_downsample_factor,
    normalize_sparse_spatial_rope_mode,
    unpack_frame_token_group,
    upsample_video_bilinear,
    validate_mixed_video_sparse_mode,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


@dataclass
class CustomPipelineOutput(BaseOutput):
    frames: torch.Tensor


def cross_attn_judgement(timesteps):
    return timesteps>=0

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def get_end_id(latents, eps=0.001):
    b, c, t, h, w = latents.shape
    end_frame = torch.ones((b, c, 1, h, w)).to(latents.device, dtype=latents.dtype) * -1
    assert b == 1, "do not support multi-video AR generation"
    diff = torch.abs(latents - end_frame)
    mask = torch.mean(diff, dim=(0, 1, 3, 4)) < eps
    indices = torch.where(mask)[0]
    if indices.numel() == 0:
        return -1
    else:
        return indices[0].item()


def update_init_latents(init_latents, n_prev, new_latents):
    init_latents = torch.cat((init_latents, new_latents), dim=2)
    return init_latents[:, :, -n_prev:]


class CustomPipeline(DiffusionPipeline, FromSingleFileMixin):

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler_action=FlowMatchEulerDiscreteScheduler(),
        dtype=torch.bfloat16,
        diffusion_mode='FM',
        timestep_scale: float = 1.0,
        vlm_text_only: bool = False,
        imf_v_use_u_head: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            scheduler_action=scheduler_action,
        )

        self.diffusion_mode = diffusion_mode

        self.timestep_scale = timestep_scale

        self.vlm_text_only = vlm_text_only

        self.imf_v_use_u_head = imf_v_use_u_head

        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio if hasattr(self, "vae") else 64
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio if hasattr(self, "vae") else 8

        self.transformer_spatial_patch_size = None
        self.transformer_temporal_patch_size = None
        if hasattr(self, "transformer"):
            if getattr(self.transformer, "unpack_in_forward", False):
                self.transformer_spatial_patch_size = 1
                self.transformer_temporal_patch_size = 1
            else:
                if hasattr(self.transformer.config, "patch_size"):
                    if isinstance(self.transformer.config.patch_size, (list, tuple)):
                        if len(self.transformer.config.patch_size) == 3:
                            assert(self.transformer.config.patch_size[1] ==  self.transformer.config.patch_size[2])
                            self.transformer_spatial_patch_size = self.transformer.config.patch_size[1]
                            self.transformer_temporal_patch_size = self.transformer.config.patch_size[0]
                        elif len(self.transformer.config.patch_size) == 2:
                            assert(self.transformer.config.patch_size[0] ==  self.transformer.config.patch_size[1])
                            self.transformer_spatial_patch_size = self.transformer.config.patch_size[0]
                    else:
                        self.transformer_spatial_patch_size = self.transformer.config.patch_size
                if self.transformer_temporal_patch_size is None:
                    if hasattr(self.transformer.config, "patch_size_t"):
                        self.transformer_temporal_patch_size = self.transformer.config.patch_size_t
        if self.transformer_spatial_patch_size is None:
            self.transformer_spatial_patch_size = 1
        if self.transformer_temporal_patch_size is None:
            self.transformer_temporal_patch_size = 1


        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt with 256->128
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1 if latents is None else latents.size(2)

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            conditioning_mask = latents.new_zeros(shape)
            conditioning_mask[:, :, 0] = 1.0
            conditioning_mask = self._pack_latents(conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
            return latents.to(device=device, dtype=dtype), conditioning_mask

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [retrieve_latents(self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)), generator[i]) for i in range(batch_size)]
        else:
            init_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), generator) for img in image]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)
        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = self._pack_latents(conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size).squeeze(
            -1
        )
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        return latents, conditioning_mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt


    def decode_to_frames(self, latents, dtype, generator, device, batch_size, decode_timestep, decode_noise_scale, mode="3dvae"):
        if mode == "3dvae":

            latents = self._denormalize_latents(latents, self.vae.latents_mean, self.vae.latents_std, getattr(self.vae.config, "scaling_factor", 1.0))
            latents = latents.to(dtype)

            if not getattr(self.vae.config, "timestep_conditioning", False):
                timestep = None
            else:
                noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[:, None, None, None, None]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, temb=timestep, return_dict=False)[0]

        else:
            latents = latents.to(dtype)
            batch_size = latents.shape[0]
            _x = rearrange(latents, "b c t h w -> (b t) c h w")
            _st = getattr(self, "stats_tracker", None)
            _x = _st.denormalize(_x) if _st is not None else _x / self.vae.config.scaling_factor
            video = self.vae.decode(_x, return_dict=False)[0]
            video = rearrange(video, "(b t) c h w -> b c t h w", b=batch_size)

        return video.clamp(-1, 1)


    @torch.no_grad()
    def infer(
        self,
        image: PipelineImageInput = None,
        concat_cond_future_image: Optional[torch.Tensor] = None,
        goal_image: PipelineImageInput = None,
        goal_cond_image: Optional[torch.Tensor] = None,
        n_prev: int = 4,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 192,
        width: int = 256,
        chunk: int = 8,
        frame_rate: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        precomputed_vlm_hidden_states: Optional[List[torch.Tensor]] = None,
        vlm_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 128,
        return_init=True,
        n_view: int = 3,
        return_action: bool = False,
        action_chunk: int = 57,
        action_dim: int = 14,
        return_video: bool = True,
        noise_seed: int = None,
        history_action_state: torch.Tensor = None,
        pixel_wise_timestep: bool = True,
        n_chunk: int = 1,
        view_ids: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        dense_frame_count: Optional[int] = None,
        sparse_downsample_factor: int = 1,
        force_mixed_path: bool = False,
        sparse_spatial_rope_mode: str = "continuous",
        **kwargs,
    ):
        if return_action:
            assert n_chunk==1, "Action-inference pipeline only support single chunk prediction now"

        self.transformer.eval()
        latent_num_frames = n_prev + chunk
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        sparse_downsample_factor = normalize_sparse_downsample_factor(sparse_downsample_factor)
        sparse_spatial_rope_mode = normalize_sparse_spatial_rope_mode(sparse_spatial_rope_mode)
        sparse_mode_active = sparse_downsample_factor > 1
        mixed_mode_active = bool(force_mixed_path or sparse_mode_active)
        t2v_mode = bool(
            getattr(getattr(self.transformer, "config", None), "t2v_mode", False)
        )
        if t2v_mode:
            if n_prev != 1:
                raise NotImplementedError(
                    f"t2v_mode currently requires n_prev=1, got {n_prev}."
                )
            if n_chunk != 1:
                raise NotImplementedError(
                    f"t2v_mode currently requires n_chunk=1, got {n_chunk}."
                )
            if mixed_mode_active:
                raise NotImplementedError(
                    "t2v_mode currently supports dense video tokens only."
                )
        if mixed_mode_active and return_action:
            raise ValueError("Mixed sparse video mode is video-gen only and does not support return_action.")
        if mixed_mode_active:
            validate_mixed_video_sparse_mode(
                sparse_downsample_factor=sparse_downsample_factor,
                sparse_spatial_rope_mode=sparse_spatial_rope_mode,
                force_mixed_path=force_mixed_path,
                vae_encoding_mode=getattr(self.vae, "vae_encoding_mode", "3dvae"),
                patch_size=self.transformer_spatial_patch_size,
                patch_size_t=self.transformer_temporal_patch_size,
                use_joint_ae_handoff=False,
                frame_indices=frame_indices,
                dense_frame_count=dense_frame_count,
                require_infer_schedule=True,
            )
            if n_chunk != 1:
                raise ValueError("Mixed sparse video mode currently supports n_chunk == 1 in inference.")
            dense_frame_count = min(int(dense_frame_count), int(chunk))
            if latent_height % sparse_downsample_factor != 0 or latent_width % sparse_downsample_factor != 0:
                raise ValueError(
                    "Mixed sparse video mode requires latent height/width divisible by sparse_downsample_factor. "
                    f"Got latent size {(latent_height, latent_width)} and factor {sparse_downsample_factor}."
                )
            if torch.as_tensor(frame_indices).shape[-1] != latent_num_frames:
                raise ValueError(
                    "Mixed sparse video mode expects frame_indices to cover memory + future frames exactly. "
                    f"Got {torch.as_tensor(frame_indices).shape[-1]} entries for {latent_num_frames} frames."
                )
            sparse_latent_h = latent_height // sparse_downsample_factor
            sparse_latent_w = latent_width // sparse_downsample_factor
            video_sequence_length = (
                (n_prev + dense_frame_count) * latent_height * latent_width
                + (chunk - dense_frame_count) * sparse_latent_h * sparse_latent_w
            )
        else:
            dense_frame_count = chunk if dense_frame_count is None else min(int(dense_frame_count), int(chunk))
            scheduler_num_frames = latent_num_frames - n_prev if t2v_mode else latent_num_frames
            video_sequence_length = (
                scheduler_num_frames * latent_height * latent_width
            )

        if len(image.shape) == 4:
            image = image.unsqueeze(2)

        preds = {}

        vlm_inputs = None

        if self.tokenizer is not None:
            if prompt_embeds is None:
                (
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                ) = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    num_videos_per_prompt=num_videos_per_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )
                preds["text_embedding"] = {}
                preds["text_embedding"]["prompt_embeds"] = prompt_embeds
                preds["text_embedding"]["prompt_attention_mask"] = prompt_attention_mask
                preds["text_embedding"]["negative_prompt_embeds"] = negative_prompt_embeds
                preds["text_embedding"]["negative_prompt_attention_mask"] = negative_prompt_attention_mask

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        else:
            if (
                precomputed_vlm_hidden_states is None
                and getattr(self.transformer, "vlm_interface", None) is not None
            ):
                if getattr(self.transformer, "use_bbox_condition", False):
                    raise ValueError("BBox evaluation must use TaskFlow.infer_deploy_step with bbox/reference images")
                if getattr(self, "vlm_text_only", False):
                    vlm_inputs = self.transformer.vlm_interface.build_qwenvl_inputs(
                        images=None, instructions=prompt, drop_images=True,)
                else:
                    n_view = image.shape[0] // len(prompt)
                    vlm_images = rearrange(image[:,:,n_prev-1,:,:], "(b v) c h w -> b v h w c", v=n_view)
                    if bool(getattr(self.transformer, "vlm_head_view_only", False)):
                        _head_idx = int(getattr(self.transformer, "vlm_head_view_index", 0))
                        vlm_images = vlm_images[:, _head_idx:_head_idx + 1]
                    vlm_images = (vlm_images.data.cpu()+1)/2*255.0
                    vlm_inputs = self.transformer.vlm_interface.build_qwenvl_inputs(images=vlm_images, instructions=prompt,)

        image = image.to(device=device, dtype=self.dtype)

        video_list = None
        if return_init:
            video_list = image.clone()

        mem_size = image.shape[2]
        sample_mode = "argmax" if noise_seed else "sample"

        mode = getattr(self.vae, "vae_encoding_mode", "3dvae")
        conditioning_video = image

        if mode == "3dvae":
            image = rearrange(image, "b c t h w -> (b t) c h w").unsqueeze(2)
            init_latents = retrieve_latents(self.vae.encode(image), generator, sample_mode=sample_mode)
            init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        else:
            image = rearrange(image, "b c t h w -> (b t) c h w")
            _st = getattr(self, "stats_tracker", None)
            init_latents = self.vae.encode(image).latent
            init_latents = _st.normalize(init_latents) if _st is not None else init_latents * self.vae.config.scaling_factor
            init_latents = init_latents.unsqueeze(2)

        if return_action and goal_image is not None:
            if len(goal_image.shape) == 4:
                goal_image = goal_image.unsqueeze(2)
            goal_image = goal_image.to(device=device, dtype=self.dtype)
            goal_latents = retrieve_latents(self.vae.encode(goal_image), generator, sample_mode=sample_mode)
            goal_latents = self._normalize_latents(goal_latents, self.vae.latents_mean, self.vae.latents_std)

        if mem_size == 1 and n_prev > 1:
            init_latents = init_latents.repeat(1, 1, n_prev, 1, 1)
        else:
            init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)

        _concat_future_latents = None
        if concat_cond_future_image is not None:
            if mode == "3dvae":
                raise NotImplementedError("teacher-forced concat cond supports 2dvae only")
            _cf = concat_cond_future_image.to(device=device, dtype=self.dtype)
            _cf_b = _cf.shape[0]
            _cf = rearrange(_cf, "b c t h w -> (b t) c h w")
            _st_cf = getattr(self, "stats_tracker", None)
            _cf_lat = self.vae.encode(_cf).latent
            _cf_lat = _st_cf.normalize(_cf_lat) if _st_cf is not None else _cf_lat * self.vae.config.scaling_factor
            _concat_future_latents = rearrange(_cf_lat, "(b t) c h w -> b c t h w", b=_cf_b)

        goal_cond_latents = None
        if goal_cond_image is not None:
            if mode == "3dvae":
                raise NotImplementedError("goal_cond_image supports 2dvae only")
            _g = goal_cond_image.to(device=device, dtype=self.dtype)
            if _g.dim() == 5:
                _g = _g.squeeze(2)
            _st_g = getattr(self, "stats_tracker", None)
            _g_lat = self.vae.encode(_g).latent
            _g_lat = _st_g.normalize(_g_lat) if _st_g is not None else _g_lat * self.vae.config.scaling_factor
            goal_cond_latents = _g_lat.unsqueeze(2)

        video_layout = None
        sparse_init_latents = None
        if mixed_mode_active:
            sparse_mem_video = bilinear_downsample_video(conditioning_video[:, :, :1], sparse_downsample_factor)
            sparse_mem_video = rearrange(sparse_mem_video, "b c f h w -> (b f) c h w")
            _st = getattr(self, "stats_tracker", None)
            sparse_init_latents = self.vae.encode(sparse_mem_video).latent
            sparse_init_latents = _st.normalize(sparse_init_latents) if _st is not None else sparse_init_latents * self.vae.config.scaling_factor
            sparse_init_latents = rearrange(sparse_init_latents, "(b f) c h w -> b c f h w", b=init_latents.shape[0], f=1)

            frame_indices_bv = expand_frame_indices_for_views(
                frame_indices,
                batch_size=batch_size,
                n_view=n_view,
                device=device,
            )
            frame_latent_hw, frame_is_sparse = build_frame_latent_spec(
                mem_frame_count=n_prev,
                dense_frame_count=dense_frame_count,
                sparse_frame_count=chunk - dense_frame_count,
                full_latent_hw=init_latents.shape[-2:],
                sparse_latent_hw=sparse_init_latents.shape[-2:],
                device=device,
                sparse_downsample_factor=sparse_downsample_factor,
            )
            video_layout = build_mixed_video_layout(
                frame_indices=frame_indices_bv,
                frame_latent_hw=frame_latent_hw,
                frame_is_sparse=frame_is_sparse,
                sparse_downsample_factor=sparse_downsample_factor,
                sparse_spatial_rope_mode=sparse_spatial_rope_mode,
            )
        mixed_mem_token_count = None if video_layout is None else int(video_layout.frame_offsets[n_prev].item())

        if return_action:
            if noise_seed:
                action_generator = torch.Generator(device=device).manual_seed(noise_seed)
            else:
                action_generator = generator
            actions = randn_tensor((batch_size, action_chunk, action_dim), device=device, dtype=self.dtype, generator=action_generator)
        else:
            actions = None


        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )

        _, _ = retrieve_timesteps(
            self.scheduler_action,
            num_inference_steps,
            device,
            None,
            sigmas=sigmas,
            mu=mu,
        )

        diffusion_mode = getattr(self, 'diffusion_mode', 'FM')
        if diffusion_mode == 'JiT':
            self.scheduler = FlowMatchEulerDiscreteSchedulerJiT.from_scheduler(self.scheduler)
            self.scheduler_action = FlowMatchEulerDiscreteSchedulerJiT.from_scheduler(self.scheduler_action)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            _, _ = retrieve_timesteps(
                self.scheduler_action,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )

        if getattr(self.scheduler.config, "final_sigmas_type", None) == "sigma_min":
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        self._num_timesteps = len(timesteps)


        rope_interpolation_scale = [1.0,1.0,1.0]


        video_attention_mask = None
        video_states_buffer = None
        if noise_seed:
            video_generator = torch.Generator(device=device).manual_seed(noise_seed)
        else:
            video_generator = generator

        for i_chunk in range(n_chunk):
            if mixed_mode_active:
                latents, conditioning_mask, cond_indicator = build_mixed_conditioning_tokens(
                    mem_latents=init_latents,
                    dense_frame_count=dense_frame_count,
                    sparse_frame_count=chunk - dense_frame_count,
                    sparse_init_latents=sparse_init_latents,
                    generator=video_generator,
                    noise_to_condition_frames=0,
                )
            else:
                latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                    init_latents, latent_num_frames, latent_height, latent_width, video_generator, noise_to_condition_frames=0
                )

            cond_latents = None
            cond_mode_ids = None
            _model_num_frames = latent_num_frames

            if return_action and goal_image is not None:
                goal_latents = rearrange(goal_latents, "b c f h w -> b (f h w) c")
                latents = torch.cat((latents[:, :-(latent_height*latent_width)], goal_latents), dim=1)

            goal_cond_active = goal_cond_latents is not None
            if goal_cond_active:
                if mixed_mode_active:
                    raise NotImplementedError(
                        "goal_cond_image is only wired for the plain (non-mixed, "
                        "non-concat) path."
                    )
                if n_chunk != 1:
                    raise NotImplementedError(
                        "goal_cond_image requires n_chunk == 1 (one goal per chunk)."
                    )
                _goal_tok_start = (latent_num_frames - 1) * latent_height * latent_width
                latents[:, _goal_tok_start:] = rearrange(
                    goal_cond_latents.to(latents.dtype), "b c f h w -> b (f h w) c"
                )
                conditioning_mask[:, _goal_tok_start:] = 1.0
                cond_indicator[:, -1:] = 1.0


            if self.do_classifier_free_guidance:
                conditioning_mask = torch.cat([conditioning_mask, conditioning_mask], dim=0)
                if mixed_mode_active:
                    cond_indicator = torch.cat([cond_indicator, cond_indicator], dim=0)
                if cond_latents is not None:
                    cond_latents = torch.cat([cond_latents, cond_latents], dim=0)
                    cond_mode_ids = torch.cat([cond_mode_ids, cond_mode_ids], dim=0)

            _v_head_kwargs = (
                {"compute_v_head": False}
                if (diffusion_mode == 'iMF-v' and self.imf_v_use_u_head)
                else {}
            )

            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents.clone()
                latent_model_input = latent_model_input.to(self.dtype)

                compute_video = i == 0 or return_video
                store_buffer = i == 0 and not return_video

                if return_action:
                    action_timesteps = t.clone()
                    action_timesteps = action_timesteps.unsqueeze(-1).repeat(actions.shape[0], actions.shape[1])
                    history_action_state_in = None
                    if self.do_classifier_free_guidance:
                        actions_in = torch.cat([actions, actions])
                        action_timesteps = torch.cat([action_timesteps, action_timesteps])
                        if history_action_state is not None:
                            history_action_state_in = torch.cat((history_action_state, history_action_state), dim=0)

                    else:
                        actions_in = actions.clone()
                        if history_action_state is not None:
                            history_action_state_in = history_action_state.clone()

                    actions_in = actions_in.to(self.dtype)
                    if history_action_state is not None:
                        history_action_state_in = history_action_state_in.to(device=device, dtype=self.dtype)

                else:
                    actions_in = None
                    action_timesteps = None
                    history_action_state_in = None

                video_rope_coords_in = None
                if mixed_mode_active:
                    video_rope_coords_in = video_layout.video_rope_coords
                    if self.do_classifier_free_guidance:
                        video_rope_coords_in = torch.cat([video_rope_coords_in, video_rope_coords_in], dim=0)

                use_tr = bool(getattr(getattr(self.transformer, "config", None), "use_t_and_r_embedding", False))
                r_for_model = None
                if diffusion_mode == 'iMF':
                    if i < len(timesteps) - 1:
                        r = timesteps[i + 1]
                    else:
                        r = torch.zeros_like(t)
                    h = t - r
                    if use_tr:
                        timestep_for_model = t * (self.timestep_scale / 1000.0)
                        r_for_model = r * (self.timestep_scale / 1000.0)
                    else:
                        timestep_for_model = h * (self.timestep_scale / 1000.0)
                elif diffusion_mode == 'iMF-v':
                    if use_tr:
                        timestep_for_model = t * (self.timestep_scale / 1000.0)
                        r_for_model = timestep_for_model.clone()
                    else:
                        timestep_for_model = torch.zeros_like(t)
                else:
                    timestep_for_model = t
                    if use_tr:
                        r_for_model = t

                timestep = timestep_for_model.expand(latent_model_input.shape[0])
                if pixel_wise_timestep:
                    timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
                else:
                    timestep = timestep.unsqueeze(-1) * (1 - cond_indicator)

                timestep_r_arg = None
                if r_for_model is not None:
                    timestep_r_arg = r_for_model.expand(latent_model_input.shape[0])
                    if pixel_wise_timestep:
                        timestep_r_arg = timestep_r_arg.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        timestep_r_arg = timestep_r_arg.unsqueeze(-1) * (1 - cond_indicator)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    timestep_r=timestep_r_arg,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=_model_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    action_states=actions_in,
                    action_timestep=action_timesteps,
                    return_video=compute_video,
                    return_action=return_action,
                    n_view=n_view,
                    view_ids=view_ids,
                    video_states_buffer=video_states_buffer,
                    store_buffer=store_buffer,
                    video_attention_mask=video_attention_mask,
                    history_action_state=history_action_state_in,
                    condition_mask=conditioning_mask,
                    cond_latents=cond_latents,
                    cond_mode_ids=cond_mode_ids,
                    vlm_inputs=vlm_inputs,
                    precomputed_vlm_hidden_states=precomputed_vlm_hidden_states,
                    vlm_attention_mask=vlm_attention_mask,
                    video_rope_coords=video_rope_coords_in,
                    **_v_head_kwargs,
                )[0]

                if store_buffer:
                    video_states_buffer = noise_pred.get("video_states_buffer")

                if return_action:
                    if self.do_classifier_free_guidance:
                        action_noise_pred_uncond, action_noise_pred = noise_pred["action"].float().chunk(2)
                        action_noise_pred = action_noise_pred_uncond + self.guidance_scale * (action_noise_pred - action_noise_pred_uncond)
                    else:
                        action_noise_pred = noise_pred["action"].float()

                    actions = self.scheduler_action.step(action_noise_pred, t, actions, return_dict=False)[0]

                if return_video:
                    video_key = "v_video" if (diffusion_mode == 'iMF-v' and "v_video" in noise_pred) else "video"
                    video_noise_pred = noise_pred[video_key].float()

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = video_noise_pred.chunk(2)
                        video_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        timestep, _ = timestep.chunk(2)

                    if mixed_mode_active:
                        future_noise_pred = video_noise_pred[:, mixed_mem_token_count:]
                        future_noise_latents = latents[:, mixed_mem_token_count:]
                        pred_future_tokens = self.scheduler.step(
                            future_noise_pred, t, future_noise_latents, return_dict=False
                        )[0]
                        latents = torch.cat([latents[:, :mixed_mem_token_count], pred_future_tokens], dim=1)
                    else:
                        video_noise_pred = self._unpack_latents(
                            video_noise_pred,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        latents = self._unpack_latents(
                            latents,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )

                        video_noise_pred = video_noise_pred[:, :, n_prev:]
                        noise_latents = latents[:, :, n_prev:]
                        pred_latents = self.scheduler.step(
                            video_noise_pred, t, noise_latents, return_dict=False
                        )[0]
                        if goal_cond_active:
                            pred_latents[:, :, -1:] = goal_cond_latents.to(pred_latents.dtype)

                        latents = torch.cat([latents[:, :, :n_prev], pred_latents], dim=2)
                        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

                if XLA_AVAILABLE:
                    xm.mark_step()

            if return_video:
                if mixed_mode_active:
                    dense_future_latents = unpack_frame_token_group(
                        latents,
                        frame_offsets=video_layout.frame_offsets,
                        frame_latent_hw=video_layout.frame_latent_hw,
                        frame_start=n_prev,
                        frame_count=dense_frame_count,
                    )
                    sparse_future_latents = unpack_frame_token_group(
                        latents,
                        frame_offsets=video_layout.frame_offsets,
                        frame_latent_hw=video_layout.frame_latent_hw,
                        frame_start=n_prev + dense_frame_count,
                        frame_count=chunk - dense_frame_count,
                    )

                    dense_frames = None
                    if dense_future_latents is not None:
                        dense_frames = self.decode_to_frames(
                            dense_future_latents,
                            dtype=self.dtype,
                            generator=generator,
                            device=device,
                            batch_size=batch_size * n_view,
                            decode_timestep=decode_timestep,
                            decode_noise_scale=decode_noise_scale,
                            mode=mode,
                        )

                    sparse_frames = None
                    if sparse_future_latents is not None:
                        sparse_frames = self.decode_to_frames(
                            sparse_future_latents,
                            dtype=self.dtype,
                            generator=generator,
                            device=device,
                            batch_size=batch_size * n_view,
                            decode_timestep=decode_timestep,
                            decode_noise_scale=decode_noise_scale,
                            mode=mode,
                        )
                        if sparse_mode_active:
                            sparse_frames = upsample_video_bilinear(
                                sparse_frames,
                                target_height=height,
                                target_width=width,
                            )

                    if dense_frames is None:
                        clean_frames = sparse_frames
                    elif sparse_frames is None:
                        clean_frames = dense_frames
                    else:
                        clean_frames = torch.cat((dense_frames, sparse_frames), dim=2)
                else:
                    clean_frames = self.decode_to_frames(
                        pred_latents,
                        dtype=self.dtype,
                        generator=generator,
                        device=device,
                        batch_size=batch_size * n_view,
                        decode_timestep=decode_timestep,
                        decode_noise_scale=decode_noise_scale,
                        mode=mode,
                    )
                if video_list is None:
                    video_list = clean_frames
                else:
                    video_list = torch.cat((video_list, clean_frames), dim=2)

                if i_chunk < n_chunk-1:
                    self.scheduler._step_index = None

                    new_mem_idxs = torch.linspace(0, video_list.shape[2]-1, n_prev).round().long()
                    new_mems = video_list[:, :, new_mem_idxs, :, :].clone()
                    new_mems = rearrange(new_mems, "bv c t h w -> (bv t) c h w").unsqueeze(2)
                    init_latents = retrieve_latents(self.vae.encode(new_mems), generator, sample_mode=sample_mode)
                    init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
                    init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)


        self.maybe_free_model_hooks()

        if return_video:
            preds["video"] = video_list

        if return_action:
            preds["action"] = actions

        self.transformer.train()

        if not return_dict:
            return (preds,)

        return CustomPipelineOutput(frames=preds)
