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
from contextlib import nullcontext
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Union
import yaml
import traceback
import copy
import json
import datetime
import numpy as np
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

from diffusers.training_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
original_sys_path = sys.path.copy()


import pdb
import time

import torchvision
from einops import rearrange


from diffusers import AutoencoderDC
from policy_core.model.WAM_policy.ltx_models.transformer_ltx_multiview_action_expert import ActionExpert
from policy_core.utils.WAM_policy.profile_utils import (
    StageProfiler,
    inference_profile_scope,
    stage_profile_enabled,
)


def get_action_chunk(mask: torch.Tensor, threshold: float = 0.5) -> int:
    mask = mask.squeeze()
    above_thresh = mask >= threshold
    if above_thresh.any():
        return above_thresh.nonzero(as_tuple=False)[-1].item() + 1
    else:
        return 1


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


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


# Reference: https://github.com/Lightricks/LTX-Video
class VABindingPipe(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderDC,
        transformer: ActionExpert,
        scheduler_action=FlowMatchEulerDiscreteScheduler(),
        dtype = torch.bfloat16,
        args = None,
        ON_LINE_TRAINING = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler_action=scheduler_action,
        )

        self.args = args


        self.training_scheduler = FlowMatchEulerDiscreteScheduler()
        self.scheduler_sigmas = self.training_scheduler.sigmas.clone()

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


    @torch.no_grad()
    def infer(
        self,
        vision_feature=None,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        action_chunk: int = 52,
        action_dim: int = 16,
        action_mask: Optional[torch.Tensor] = None,
        state_cond: torch.Tensor = None,
        perlin_noise: bool = False,
        guidance_scale=1.0,
        timesteps=None,
        noise_seed=None,
        noise: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        body_emb_id: Optional[torch.Tensor] = None,
    ):
        assert (guidance_scale - 1.0) < 1e-4, 'Do not support cfg guidance'
        batch_size = vision_feature.shape[0]
        device = self._execution_device

        pipeline_profiler = StageProfiler("VA_PROFILE", device)
        step_profiles = []
        was_training = self.transformer.training
        self.transformer.eval()

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False


        device = self._execution_device

        preds = {}

        dtype = self.transformer.dtype

        if noise_seed:
            action_generator = torch.Generator(device=device).manual_seed(noise_seed)

        else:
            action_generator = generator

        pipeline_profiler.mark("setup")

        if noise is not None:
            actions = noise.to(device=device, dtype=dtype)
        else:
            actions = randn_tensor((batch_size, action_chunk, action_dim), device=device, dtype=dtype, generator=action_generator)

        if action_mask is not None:
            action_mask = action_mask.to(device=device, dtype=dtype)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(1)
            actions = actions * action_mask

        pipeline_profiler.mark("noise_preparation")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler_action,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas
        )

        self._num_timesteps = len(timesteps)

        pipeline_profiler.mark("timestep_preparation")



        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                step_profiler = StageProfiler("VA_PROFILE_STEP", device)

                action_timesteps = t.clone()

                action_timesteps = action_timesteps.unsqueeze(-1).repeat(actions.shape[0], actions.shape[1])

                actions_in = actions.clone().to(dtype)

                step_profiler.mark("preparation")

                profile_ctx = (
                    inference_profile_scope()
                    if stage_profile_enabled()
                    else nullcontext()
                )
                with profile_ctx:
                    noise_pred = self.transformer(
                        action_states=actions_in,
                        action_timestep=action_timesteps,
                        vision_velocity=vision_feature,
                        state_cond=state_cond,
                        view_ids=view_ids,
                        body_emb_id=body_emb_id,
                    )

                step_profiler.mark("transformer")

                action_noise_pred = noise_pred["action"].float()

                if action_mask is not None:
                    action_noise_pred = action_noise_pred * action_mask.float()

                step_profiler.mark("prediction_postprocess")

                actions = self.scheduler_action.step(action_noise_pred, t, actions, return_dict=False)[0]

                step_profiler.mark("scheduler")

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                step_profiler.mark("loop_overhead")
                step_profiler.finish()
                step_profiles.append(step_profiler)

        pipeline_profiler.mark("denoise_loop")

        self.maybe_free_model_hooks()

        preds["action"] = actions

        self.transformer.train(was_training)

        pipeline_profiler.mark("finalize")
        pipeline_profiler.finish()

        step_order = (
            "preparation",
            "transformer",
            "prediction_postprocess",
            "scheduler",
            "loop_overhead",
        )
        for step_index, step_profiler in enumerate(step_profiles):
            step_profiler.report(
                metadata={"step": f"{step_index + 1}/{num_inference_steps}"},
                order=step_order,
            )

        aggregate_ms = {
            "step_preparation_total": sum(
                item.get_ms("preparation") for item in step_profiles
            ),
            "transformer_total": sum(
                item.get_ms("transformer") for item in step_profiles
            ),
            "prediction_postprocess_total": sum(
                item.get_ms("prediction_postprocess") for item in step_profiles
            ),
            "scheduler_total": sum(
                item.get_ms("scheduler") for item in step_profiles
            ),
        }
        pipeline_profiler.report(
            order=(
                "setup",
                "noise_preparation",
                "timestep_preparation",
                "step_preparation_total",
                "transformer_total",
                "prediction_postprocess_total",
                "scheduler_total",
                "denoise_loop",
                "finalize",
            ),
            extra_ms=aggregate_ms,
        )

        return (preds,)
