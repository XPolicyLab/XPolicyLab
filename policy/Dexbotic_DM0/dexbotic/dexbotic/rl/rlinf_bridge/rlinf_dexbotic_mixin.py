"""Shared RLinf adapter mixin for Dexbotic policies.

Both ``DexboticDM0ForRLActionPrediction`` and ``DexboticPi0ForRLActionPrediction``
used to re-implement the same RLinf ``BasePolicy`` glue: tokenization pipeline
driver, observation preprocessing, image batching, PPO-style ``default_forward``
plumbing, Gaussian log-prob / entropy helpers, and the ``predict_action_batch``
wrapper. That boilerplate lives here now, leaving each concrete policy to
focus on the backbone-specific bits (``_no_split_modules``, dtype probing,
prefix/suffix MoT routing, and ``sample_actions`` / ``get_log_prob_value``).

Usage
-----
Each concrete policy inherits ``RLinfDexboticMixin`` *before* its backbone::

    class DexboticPi0ForRLActionPrediction(RLinfDexboticMixin, Pi0ForCausalLM):
        def __init__(self, config):
            Pi0ForCausalLM.__init__(self, config)
            # backbone-specific wrapping (FSDP modules, dtype, ...)
            ...
            self._init_rlinf_common_state(config)
            self.pi0_tokenization = None

        def _tokenize_prompt(self, prompt: str) -> np.ndarray:
            return self.pi0_tokenization([{"value": prompt}])["input_ids"]

The MRO becomes ``<Policy> -> RLinfDexboticMixin -> BasePolicy -> <Backbone>``,
so ``self.process_images`` / ``self.model`` / ``self.tokenizer`` still resolve
to the backbone, and RLinf sees a proper ``BasePolicy`` subclass.

``apply_common_rl_config_overrides`` and ``build_rl_transform_lists`` are
module-level helpers for the two ``get_model`` factories, which share the same
Hydra ``cfg`` → ``config`` field plumbing and the same
``ActionNorm``/``ActionDenorm`` pipelines.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger

_REQUIRED_NUM_IMAGES = 3


class RLinfDexboticMixin(BasePolicy):
    """RLinf ``BasePolicy`` glue shared by all Dexbotic flow-matching policies.

    Subclasses are expected to:
      * Call their backbone's ``__init__`` first (e.g. ``Pi0ForCausalLM.__init__``)
        to build ``self.model`` / ``self.tokenizer`` etc., do any
        backbone-specific FSDP / dtype wiring, then call
        ``self._init_rlinf_common_state(config)``.
      * Implement :meth:`_tokenize_prompt` to turn a raw prompt string into a
        backbone-specific ``input_ids`` ndarray.
      * Implement ``sample_actions``, ``get_log_prob_value``, and
        ``sample_mean_var_val`` (their prefix/suffix MoT wiring differs per
        backbone).
    """

    # ------------------------------------------------------------------
    # Lifecycle / hooks
    # ------------------------------------------------------------------
    def _init_rlinf_common_state(self, config) -> None:
        """Initialize RL-specific attributes shared by every Dexbotic policy.

        Must be called *after* the backbone ``__init__`` has produced
        ``self.model`` with its final dtype, because ``value_head`` is cast to
        match ``self.model.action_out_proj.weight.dtype``.
        """
        self.logger = get_logger()
        self.config = config
        self.num_steps = config.num_steps
        self.action_horizon = config.chunk_size
        self.num_action_chunks = getattr(
            config, "output_action_chunks", config.chunk_size
        )
        self.action_dim = config.action_dim
        # Indices of absolute (non-delta) action dims; defaults to [6] (gripper).
        self.non_delta_mask = getattr(config, "non_delta_mask", [6])
        self.global_step = 0
        self.use_vlm_value = False
        self.value_head = nn.Linear(config.action_config.hidden_size, 1)
        self.value_head = self.value_head.to(
            dtype=self.model.action_out_proj.weight.dtype
        )
        self._input_transform = None
        self._output_transform = None
        self.norm_stats = None

    def _tokenize_prompt(self, prompt: str) -> np.ndarray:
        """Turn a prompt string into the backbone's ``input_ids`` ndarray.

        DM0 expects ``{"from": "human", "value": prompt}``; Pi0 expects
        ``{"value": prompt}``. Subclass responsibility because each backbone
        ships its own tokenization wrapper.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Freezing / normalization helpers
    # ------------------------------------------------------------------
    def freeze_vlm(self) -> None:
        """Freeze the VLM stack (vision tower + LLM + projector).

        Only active when ``config.train_expert_only`` is set; otherwise logs a
        warning and returns without touching anything.
        """
        if not getattr(self.config, "train_expert_only", False):
            self.logger.warning(
                "freeze_vlm() called but train_expert_only is False"
            )
            return
        for component in ("mm_vision_tower", "llm", "mm_projector"):
            mod = getattr(self.model, component, None)
            if mod is not None:
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

    def _read_normalization_stats(self, norm_stats_file):
        if not os.path.exists(norm_stats_file):
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_file}. "
                "Make sure the checkpoint directory contains norm_stats.json"
            )
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def setup_wrappers(self, transforms=(), output_transforms=()) -> None:
        self._input_transform = Pipeline(transforms) if transforms else None
        self._output_transform = (
            Pipeline(output_transforms) if output_transforms else None
        )

    # ------------------------------------------------------------------
    # Observation / action pipelines
    # ------------------------------------------------------------------
    def input_transform(self, obs: dict, transpose: bool = True) -> dict:
        if "prompt" in obs:
            prompts = obs["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, torch.Tensor):
                prompts = [str(p) for p in prompts]
            batch_input_ids = [self._tokenize_prompt(p) for p in prompts]
            batch_input_ids = torch.from_numpy(np.array(batch_input_ids))
            batch_attention_mask = (
                batch_input_ids != self.tokenizer.pad_token_id
            )
            obs["tokenized_prompt"] = batch_input_ids
            obs["tokenized_prompt_mask"] = batch_attention_mask

        if self._input_transform is not None and "observation/state" in obs:
            state_tensor = obs["observation/state"]
            if isinstance(state_tensor, torch.Tensor):
                state_value = state_tensor.cpu().float().numpy()
            else:
                state_value = state_tensor
            state_dict = self._input_transform({"state": state_value})
            obs["observation/state"] = state_dict["state"]
            obs["states"] = state_dict["state"]
        return obs

    def output_transform(self, outputs: dict) -> dict:
        if self._output_transform is None:
            self.logger.warning(
                "[output_transform] WARNING: _output_transform is None! "
                "Actions will NOT be denormalized!"
            )
            return outputs

        state_batch = outputs.get("state", None)
        meta_data = outputs.get("meta_data", {})
        batch_size = outputs["actions"].shape[0]
        transformed_actions = []
        for i in range(batch_size):
            sample = {"action": outputs["actions"][i].cpu().numpy()}
            if state_batch is not None:
                sample["state"] = (
                    state_batch[i].cpu().numpy()
                    if isinstance(state_batch, torch.Tensor)
                    else state_batch[i]
                )
            if meta_data:
                sample["meta_data"] = meta_data
            sample = self._output_transform(sample)
            transformed_actions.append(torch.from_numpy(sample["action"]))

        outputs["actions"] = torch.stack(transformed_actions, dim=0).to(
            outputs["actions"].device
        )
        outputs["actions"] = outputs["actions"][:, : self.num_action_chunks]
        return outputs

    def precision_processor(self, processed_obs: dict) -> dict:
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def obs_processor(self, env_obs: dict) -> dict:
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        state = env_obs["states"]
        if torch.is_tensor(state):
            state = state.to(dtype=torch.float32)
        processed_obs["observation/state"] = state
        if "wrist_images" in env_obs:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        return processed_obs

    # ------------------------------------------------------------------
    # Forward dispatcher + shared default_forward
    # ------------------------------------------------------------------
    def forward(self, forward_type: str = "default_forward", **kwargs):
        if "forward_inputs" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("forward_inputs")
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"Forward type {forward_type} not implemented"
        )

    def default_forward(self, data, **kwargs):
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        if "tokenized_prompt" in data:
            observation = data
        else:
            observation = self.input_transform(data, transpose=False)

        device = chains.device
        raw_main_images = observation["observation/image"]
        raw_wrist_images = observation.get("observation/wrist_image", None)
        images, img_masks = self._process_images_for_training(
            raw_main_images, raw_wrist_images, device
        )

        target_dtype = next(self.parameters()).dtype
        lang_tokens = observation["tokenized_prompt"].to(device)
        lang_masks = observation["tokenized_prompt_mask"].to(device)
        state = observation["observation/state"].to(device=device)
        chains = data["chains"].to(device=device, dtype=target_dtype)

        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.num_action_chunks, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.num_action_chunks, : self.config.action_env_dim
        ]
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _process_images_for_training(
        self, raw_main_images, raw_wrist_images, device
    ):
        if torch.is_tensor(raw_main_images):
            raw_main_images = raw_main_images.cpu().numpy()
        if raw_wrist_images is not None and torch.is_tensor(raw_wrist_images):
            raw_wrist_images = raw_wrist_images.cpu().numpy()

        batch_size = raw_main_images.shape[0]
        base_pil_images = []
        for i in range(batch_size):
            img_np = raw_main_images[i]
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            base_pil_images.append(Image.fromarray(img_np))

        wrist_pil_images = []
        if raw_wrist_images is not None:
            for i in range(batch_size):
                wrist_np = raw_wrist_images[i].astype(np.uint8)
                wrist_pil_images.append(Image.fromarray(wrist_np))

        images_list = []
        for i in range(batch_size):
            pil_list = [base_pil_images[i]]
            if wrist_pil_images:
                pil_list.append(wrist_pil_images[i])
            images_list.append(self.process_images(pil_list))

        images = torch.stack(images_list, dim=0).to(
            device=device, dtype=next(self.parameters()).dtype
        )

        num_views = images.shape[1]
        if num_views < _REQUIRED_NUM_IMAGES:
            pad_size = _REQUIRED_NUM_IMAGES - num_views
            padding = torch.zeros(
                batch_size,
                pad_size,
                *images.shape[2:],
                dtype=images.dtype,
                device=device,
            )
            images = torch.cat([images, padding], dim=1)
        image_masks = torch.zeros(
            batch_size, _REQUIRED_NUM_IMAGES, dtype=torch.bool, device=device
        )
        image_masks[:, :num_views] = True
        return images, image_masks

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def get_logprob_norm(self, sample, mu, sigma):
        if self.config.safe_get_logprob:
            return -torch.pow((sample - mu), 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    # ------------------------------------------------------------------
    # predict_action_batch orchestration
    # ------------------------------------------------------------------
    def predict_action_batch(self, env_obs, **kwargs):
        mode = kwargs.get("mode", "train")
        compute_values = kwargs.get("compute_values", True)
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)

        outputs = self.sample_actions(
            processed_obs=processed_obs,
            mode=mode,
            compute_values=compute_values,
        )
        if self._output_transform is not None:
            state_for_transform = processed_obs.get("observation/state")
            if state_for_transform is not None:
                outputs["state"] = (
                    state_for_transform.cpu().numpy()
                    if isinstance(state_for_transform, torch.Tensor)
                    else state_for_transform
                )
                outputs["meta_data"] = {
                    "non_delta_mask": np.array(self.non_delta_mask)
                }
            outputs = self.output_transform(outputs)

        actions = outputs["actions"][:, :, : self.config.action_env_dim]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }
        if "tokenized_prompt" in processed_obs:
            forward_inputs["tokenized_prompt"] = processed_obs[
                "tokenized_prompt"
            ]
        if "tokenized_prompt_mask" in processed_obs:
            forward_inputs["tokenized_prompt_mask"] = processed_obs[
                "tokenized_prompt_mask"
            ]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)

        return actions, {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }


# ----------------------------------------------------------------------
# get_model helpers (module-level, so they can be reused without inheriting)
# ----------------------------------------------------------------------
def apply_common_rl_config_overrides(config, cfg) -> None:
    """Apply the RL-specific fields that both DM0 and Pi0 ``get_model`` set.

    The block below is literally identical between the two factories; keeping
    it in one place avoids the inevitable drift when one side adds a new key.
    ``config`` is the backbone config (``DM0Config`` / ``Pi0Config``); ``cfg``
    is the Hydra ``DictConfig`` passed into ``get_model``.
    """
    config.num_steps = cfg.get("num_steps", 10)
    config.action_env_dim = cfg.action_dim
    config.add_value_head = cfg.get("add_value_head", True)
    config.noise_level = cfg.get("dexbotic", {}).get("noise_level", 0.5)
    config.noise_method = cfg.get("dexbotic", {}).get("noise_method", "flow_sde")
    config.detach_critic_input = cfg.get("dexbotic", {}).get(
        "detach_critic_input", True
    )
    config.train_expert_only = cfg.get("dexbotic", {}).get(
        "train_expert_only", False
    )
    config.action_horizon = config.chunk_size
    config.output_action_chunks = cfg.num_action_chunks
    config.safe_get_logprob = cfg.get("safe_get_logprob", False)
    config.chunk_critic_input = cfg.get("chunk_critic_input", True)
    config.noise_anneal = cfg.get("noise_anneal", False)
    config.joint_logprob = cfg.get("joint_logprob", False)
    config.value_after_vlm = cfg.get("value_after_vlm", False)
    config.processor_config = cfg.model_path


def build_rl_transform_lists(norm_stats, state_pad_ndim: int):
    """Build the ``(input_transforms, output_transforms)`` lists for RL rollout.

    ``state_pad_ndim`` is the target last-axis size for
    :class:`~dexbotic.data.dataset.transform.action.PadState`:

    * DM0 passes ``config.action_dim`` (env state happens to match).
    * Pi0 passes ``proprio_pad_ndim_from_norm_stats(norm_stats)`` because its
      checkpoint-side ``norm_stats['state']`` is wider than ``action_dim``.
    """
    from dexbotic.data.dataset.transform.action import ActionNorm, PadState
    from dexbotic.data.dataset.transform.common import ToTensor
    from dexbotic.data.dataset.transform.output import (
        AbsoluteAction,
        ActionDenorm,
    )

    if norm_stats is None:
        return [], []
    input_transforms = [
        PadState(ndim=state_pad_ndim, axis=-1),
        ActionNorm(statistic_mapping=norm_stats, strict=False),
        ToTensor(),
    ]
    output_transforms = [
        ToNumpy(),
        ActionDenorm(statistic_mapping=norm_stats, strict=False),
        AbsoluteAction(),
    ]
    return input_transforms, output_transforms
