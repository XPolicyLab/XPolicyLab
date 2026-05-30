"""RLinf adapter for Dexbotic-Pi0 (``Pi0ForCausalLM``) policies.

Shares :class:`~dexbotic.rl.rlinf_bridge.rlinf_dexbotic_mixin.RLinfDexboticMixin`
with the DM0 variant; only the Pi0-specific bits live here:

* ``_no_split_modules`` (Gemma family; branches on ``train_expert_only``),
  dtype probing (LLM *or* action-expert layers), plus dropping
  ``action_expert.embed_tokens`` and tagging ``_fsdp_wrap_name``;
* Pi0 tokenization format (``{"value": prompt}``);
* MoT prefix/suffix plumbing built on ``embed_prefix`` / ``embed_suffix`` +
  ``make_attn_mask`` / ``make_attn_mask_4d`` with per-layer LLM and
  ``action_expert`` forwards;
* ``sample_mean_var_val`` / ``sample_actions`` / ``get_log_prob_value``
  (the Pi0 flow-matching loop, with optional ``use_vlm_value`` via
  ``get_value_from_vlm``);
* a Pi0-flavored ``get_model`` factory (uses ``Pi0Config`` / ``Pi0Tokenization``
  and aligns ``PadState`` ndim to the checkpoint's ``norm_stats['state']``).
"""

import glob
import os
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoTokenizer, DynamicCache

from dexbotic.model.pi0.pi0_arch import (
    Pi0Config,
    Pi0ForCausalLM,
    make_attn_mask,
    make_attn_mask_4d,
)
from dexbotic.rl.rlinf_bridge.norm_stats_utils import proprio_pad_ndim_from_norm_stats
from dexbotic.rl.rlinf_bridge.rlinf_dexbotic_mixin import (
    RLinfDexboticMixin,
    apply_common_rl_config_overrides,
    build_rl_transform_lists,
)
from dexbotic.tokenization.process import Pi0Tokenization
from rlinf.utils.logging import get_logger


class DexboticPi0ForRLActionPrediction(RLinfDexboticMixin, Pi0ForCausalLM):
    _no_split_names = [
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    def __init__(self, config):
        Pi0ForCausalLM.__init__(self, config)
        if getattr(config, "train_expert_only", False):
            self._no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            self._no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]

        model_dtype = None
        if (
            hasattr(self.model, "llm")
            and hasattr(self.model.llm, "layers")
            and len(self.model.llm.layers) > 0
        ):
            for param in self.model.llm.layers[0].parameters():
                model_dtype = param.dtype
                break
        elif (
            hasattr(self.model, "action_expert")
            and hasattr(self.model.action_expert, "layers")
            and len(self.model.action_expert.layers) > 0
        ):
            for param in self.model.action_expert.layers[0].parameters():
                model_dtype = param.dtype
                break
        if model_dtype is None:
            all_params = list(self.model.parameters())
            model_dtype = all_params[0].dtype if all_params else torch.float32
        self.model = self.model.to(dtype=model_dtype)

        if hasattr(self.model, "action_expert") and hasattr(
            self.model.action_expert, "embed_tokens"
        ):
            self.model.action_expert.embed_tokens = None
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self._init_rlinf_common_state(config)
        self.pi0_tokenization = None

    def _tokenize_prompt(self, prompt: str) -> np.ndarray:
        return self.pi0_tokenization([{"value": prompt}])["input_ids"]

    def _normalize_state(self, state):
        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            return state
        if "state" not in self.norm_stats:
            return state
        stats = self.norm_stats["state"]
        mean = torch.tensor(stats["mean"], device=state.device, dtype=state.dtype)
        std = torch.tensor(stats["std"], device=state.device, dtype=state.dtype)
        return (state - mean) / (std + 1e-6)

    # ------------------------------------------------------------------
    # MoT prefix/suffix plumbing (Pi0-specific)
    # ------------------------------------------------------------------
    def _forward_mot_prefix(
        self,
        prefix_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        """Forward the LLM prefix only, building the KV cache.

        Uses each layer's forward() method so FSDP can properly gather/release
        parameters per layer.
        """
        hidden_states = prefix_embeds
        past_key_values = DynamicCache()
        # Cast mask to match model dtype (make_attn_mask_4d produces float32,
        # but SDPA requires mask dtype == query dtype).
        if mask is not None:
            mask = mask.to(dtype=hidden_states.dtype)

        for layer in self.model.llm.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=cache_position,
                past_key_value=past_key_values,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        prefix_output = self.model.llm.norm(hidden_states)
        return prefix_output, past_key_values

    def _forward_mot_suffix(
        self,
        suffix_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple] = None,
        past_key_values: Optional[DynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward the action expert suffix only, attending to the LLM KV cache.

        Uses each layer's forward() method so FSDP can properly gather/release
        parameters per layer. Clones the KV cache to avoid mutating the
        original prefix cache (needed for multiple suffix calls).
        """
        hidden_states = suffix_embeds

        if mask is not None:
            mask = mask.to(dtype=hidden_states.dtype)

        cloned_cache = DynamicCache()
        if past_key_values is not None:
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                cloned_cache.key_cache.append(k)
                cloned_cache.value_cache.append(v)

        for layer in self.model.action_expert.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=cloned_cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        suffix_output = self.model.action_expert.norm(hidden_states)
        return suffix_output

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        batch_size = state.shape[0]
        device = state.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            states=state,
            noisy_actions=x_t,
            time=timestep,
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = prefix_pad_masks.unsqueeze(1).repeat(
            1, suffix_tokens.shape[1], 1
        )
        full_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
        full_attn_mask = make_attn_mask_4d(full_attn_mask)
        full_positions = (
            prefix_pad_masks.sum(axis=-1).unsqueeze(-1)
            + torch.cumsum(suffix_mask, dim=-1)
            - 1
        )
        full_position_embeddings = self.model.llm.rotary_emb(
            suffix_tokens, full_positions
        )
        suffix_out = self._forward_mot_suffix(
            suffix_embeds=suffix_tokens,
            mask=full_attn_mask,
            position_embeddings=full_position_embeddings,
            past_key_values=past_key_values,
            position_ids=full_positions,
        )
        return suffix_out.clone()[:, -self.config.chunk_size :]

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)
        # Timesteps: [1, 9/10, 8/10, ..., 1/10, 0] for 10 steps
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.model.action_out_proj(suffix_out)
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.chunk_size], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    @torch.no_grad()
    def sample_actions(
        self, processed_obs, noise=None, mode="train", compute_values=True
    ):
        original_training_mode = self.training
        self.eval()
        try:
            input_ids = processed_obs.get("tokenized_prompt")
            attention_mask = processed_obs.get("tokenized_prompt_mask")
            states = processed_obs["observation/state"].to(
                device=next(self.parameters()).device
            )

            raw_images = processed_obs["observation/image"]
            batch_size = raw_images.shape[0]
            device = states.device

            base_pil_images = []
            for batch_idx in range(batch_size):
                img_np = raw_images[batch_idx].cpu().numpy()
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                base_pil_images.append(Image.fromarray(img_np))

            wrist_pil_images = []
            if "observation/wrist_image" in processed_obs:
                wrist_raw = processed_obs["observation/wrist_image"]
                for batch_idx in range(batch_size):
                    wrist_np = wrist_raw[batch_idx].cpu().numpy().astype(np.uint8)
                    wrist_pil_images.append(Image.fromarray(wrist_np))
            images_list = []
            for batch_idx in range(batch_size):
                if wrist_pil_images:
                    pil_pair = [base_pil_images[batch_idx], wrist_pil_images[batch_idx]]
                    processed = self.process_images(pil_pair)
                else:
                    processed = self.process_images([base_pil_images[batch_idx]])
                images_list.append(processed)
            images = torch.stack(images_list, dim=0).to(
                device=device, dtype=next(self.parameters()).dtype
            )
            num_views = images.shape[1]
            required_num_images = 3

            if num_views < required_num_images:
                pad_size = required_num_images - num_views
                padding = torch.zeros(
                    batch_size,
                    pad_size,
                    images.shape[2],
                    images.shape[3],
                    images.shape[4],
                    dtype=images.dtype,
                    device=device,
                )
                images = torch.cat([images, padding], dim=1)
            image_masks = torch.zeros(
                batch_size, required_num_images, dtype=torch.bool, device=device
            )
            image_masks[:, :num_views] = True
            target_dtype = next(self.parameters()).dtype
            num_steps = self.num_steps

            # Prefix forward: embed and cache LLM prefix
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_masks=image_masks,
            )
            prefix_att_2d_masks = make_attn_mask(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = make_attn_mask_4d(prefix_att_2d_masks)

            prefix_output, past_key_values = self._forward_mot_prefix(
                prefix_embeds=prefix_embs,
                mask=prefix_att_2d_masks_4d,
                position_embeddings=self.model.llm.rotary_emb(
                    prefix_embs, prefix_position_ids
                ),
                cache_position=prefix_position_ids,
            )

            # Init noise
            x_t = torch.randn(
                batch_size, self.config.chunk_size, self.config.action_dim,
                device=device, dtype=target_dtype,
            )

            chains = []
            log_probs = []
            values = []
            chains.append(x_t)

            if self.use_vlm_value:
                values_vlm = self.get_value_from_vlm(prefix_output)
            if self.config.joint_logprob:
                initial_log_prob = self.get_logprob_norm(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
                log_probs.append(initial_log_prob)

            # Build denoise_inds
            if mode == "train":
                if self.config.joint_logprob:
                    denoise_inds = torch.arange(num_steps)
                else:
                    if getattr(self.config, "ignore_last", False):
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
            else:
                denoise_inds = torch.tensor([-1] * num_steps)
            denoise_inds = denoise_inds[None].repeat(batch_size, 1)

            # Diffusion loop
            for idx in range(num_steps):
                sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    x_t,
                    idx,
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    sample_mode,
                    num_steps,
                    compute_values,
                )
                x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
                log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
                values.append(value_t)
                chains.append(x_t)
                log_probs.append(log_prob)

            x_0 = x_t
            chains = torch.stack(chains, dim=1)

            log_probs = torch.stack(log_probs, dim=1)[
                :, :, : self.num_action_chunks, : self.config.action_env_dim
            ]
            if self.config.joint_logprob:
                log_probs = log_probs.mean(dim=1)
            else:
                log_probs = log_probs[
                    torch.arange(log_probs.shape[0]),
                    denoise_inds[:, 0],
                ]

            if self.use_vlm_value:
                values = values_vlm[:, None]
            else:
                values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

            return {
                "actions": x_0,
                "chains": chains,
                "prev_logprobs": log_probs,
                "prev_values": values,
                "denoise_inds": denoise_inds,
            }
        finally:
            if original_training_mode:
                self.train()

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]

        # When train_expert_only, the vision tower, LLM, and projector are all
        # frozen.  Wrapping the prefix computation in no_grad avoids storing
        # intermediate activations for these frozen modules, which is the main
        # source of memory savings.
        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                images=images,
                image_masks=img_masks,
            )
            prefix_att_2d_masks = make_attn_mask(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = make_attn_mask_4d(prefix_att_2d_masks)

            prefix_output, past_key_values = self._forward_mot_prefix(
                prefix_embeds=prefix_embs,
                mask=prefix_att_2d_masks_4d,
                position_embeddings=self.model.llm.rotary_emb(
                    prefix_embs, prefix_position_ids
                ),
                cache_position=prefix_position_ids,
            )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)

            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)

            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy


def get_model(cfg: DictConfig, torch_dtype: Optional[Any] = None):
    import safetensors.torch

    logger = get_logger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    try:
        config = Pi0Config.from_pretrained(cfg.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=False, local_files_only=True
        )
        apply_common_rl_config_overrides(config, cfg)

        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            model = DexboticPi0ForRLActionPrediction(config)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline
        model.tokenizer = tokenizer

        model.pi0_tokenization = Pi0Tokenization(tokenizer)
        weight_paths = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
        weight_paths = [p for p in weight_paths if not p.endswith(".index.json")]
        if not weight_paths:
            weight_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weights found in {cfg.model_path}")
            weight_paths = [weight_path]
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path)
            model_keys = {n for n, _ in model.named_parameters()}
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(state_dict, strict=False)
        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.norm_stats = model._read_normalization_stats(norm_stats_file)
        else:
            model.norm_stats = None

        model._train_expert_only = getattr(config, "train_expert_only", False)

    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        raise

    state_pad_ndim = (
        proprio_pad_ndim_from_norm_stats(model.norm_stats)
        if model.norm_stats is not None
        else 0
    )
    input_transforms, output_transforms = build_rl_transform_lists(
        model.norm_stats, state_pad_ndim=state_pad_ndim
    )
    model.setup_wrappers(
        transforms=input_transforms, output_transforms=output_transforms
    )
    return model
