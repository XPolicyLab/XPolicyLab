"""RLinf adapter for Dexbotic-DM0 (``DM0ForCausalLM``) policies.

Shares :class:`~dexbotic.rl.rlinf_bridge.rlinf_dexbotic_mixin.RLinfDexboticMixin`
with the Pi0 variant; only the DM0-specific bits live here:

* ``_no_split_modules`` / dtype probing for ``DM0ForCausalLM``;
* DM0 tokenization format (``{"from": "human", "value": ...}``);
* MoT prefix/suffix plumbing built on
  ``get_prefix_hidden_states`` / ``get_suffix_hidden_states`` +
  ``make_attn_mask_2d`` / ``make_attn_mask_4d`` / ``make_suffix_attn_mask_2d``;
* ``sample_mean_var_val`` / ``sample_actions`` / ``get_log_prob_value`` because
  the flow-matching inner loop references those DM0-specific helpers;
* a DM0-flavored ``get_model`` factory (forces SDPA, patches
  ``PEVisionTower`` device/dtype after FSDP flattens the parameters, etc.).
"""

import glob
import os
import random
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoTokenizer, DynamicCache

from dexbotic.model.dm0.dm0_arch import DM0Config, DM0ForCausalLM
from dexbotic.model.dm0.dm0_utils import (
    make_attn_mask_2d,
    make_attn_mask_4d,
    make_suffix_attn_mask_2d,
)
from dexbotic.rl.rlinf_bridge.rlinf_dexbotic_mixin import (
    RLinfDexboticMixin,
    apply_common_rl_config_overrides,
    build_rl_transform_lists,
)
from dexbotic.tokenization.process import DM0Tokenization
from rlinf.utils.logging import get_logger


class DexboticDM0ForRLActionPrediction(RLinfDexboticMixin, DM0ForCausalLM):
    _no_split_names = [
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    def __init__(self, config):
        DM0ForCausalLM.__init__(self, config)
        # Fine-grained FSDP wrapping (Qwen3MLP level); avoids per-decoder-layer
        # wrapping so _merged_attention_forward can reach sub-module params.
        self._no_split_modules = ["Qwen3MLP"]

        # Force uniform dtype so FSDP can flatten parameters without error.
        model_dtype = None
        if (
            hasattr(self.model, "llm")
            and hasattr(self.model.llm, "layers")
            and len(self.model.llm.layers) > 0
        ):
            for param in self.model.llm.layers[0].parameters():
                model_dtype = param.dtype
                break
        if model_dtype is None:
            all_params = list(self.model.parameters())
            model_dtype = all_params[0].dtype if all_params else torch.float32
        self.model = self.model.to(dtype=model_dtype)

        self._init_rlinf_common_state(config)
        self.dm0_tokenization = None

    def _tokenize_prompt(self, prompt: str) -> np.ndarray:
        return self.dm0_tokenization([{"from": "human", "value": prompt}])[
            "input_ids"
        ]

    # ------------------------------------------------------------------
    # MoT prefix/suffix plumbing (DM0-specific)
    # ------------------------------------------------------------------
    def _build_prefix_kv_cache(self, input_ids, attention_mask, images, image_masks):
        """Build KV cache from prefix (images + language) using per-layer LLM forward."""
        prefix_hidden_states, prefix_padding_mask, prefix_attn_mask = (
            self.get_prefix_hidden_states(input_ids, attention_mask, images, image_masks)
        )
        prefix_attn_mask_2d = make_attn_mask_2d(
            padding_mask=prefix_padding_mask, attn_mask=prefix_attn_mask
        )
        prefix_attn_mask_4d = make_attn_mask_4d(
            prefix_attn_mask_2d, dtype=prefix_hidden_states.dtype
        )
        positions = torch.cumsum(prefix_padding_mask, dim=1) - 1

        hidden_states = prefix_hidden_states
        past_key_values = DynamicCache()
        mask = prefix_attn_mask_4d.to(dtype=hidden_states.dtype)
        position_embeddings = self.model.llm.rotary_emb(hidden_states, positions)

        for layer in self.model.llm.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=positions,
                past_key_value=past_key_values,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        del hidden_states, mask, prefix_attn_mask_4d, prefix_attn_mask_2d, position_embeddings
        torch.cuda.empty_cache()
        return prefix_padding_mask, prefix_attn_mask, past_key_values

    def get_suffix_out(
        self,
        prefix_padding_mask,
        prefix_attn_mask,
        kv_cache,
        x_t,
        timestep,
    ):
        """Run suffix (action expert) using cached prefix KV, per-layer forward."""
        batch_size = x_t.shape[0]
        device = x_t.device

        model_dtype = self.model.action_in_proj.weight.dtype
        x_t = x_t.to(dtype=model_dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.broadcast_to(batch_size)
        timestep = timestep.to(dtype=model_dtype)

        suffix_hidden_states, suffix_padding_mask, suffix_attn_mask = (
            self.get_suffix_hidden_states(x_t, timestep)
        )
        suffix_attn_mask_2d = make_suffix_attn_mask_2d(
            suffix_padding_mask=suffix_padding_mask,
            suffix_attn_mask=suffix_attn_mask,
            prefix_padding_mask=prefix_padding_mask,
            prefix_attn_mask=prefix_attn_mask,
        )
        full_attn_mask_4d = make_attn_mask_4d(
            suffix_attn_mask_2d, dtype=suffix_hidden_states.dtype
        )
        prefix_offsets = torch.sum(prefix_padding_mask, dim=-1)[:, None]
        full_positions = prefix_offsets + torch.cumsum(suffix_padding_mask, dim=1) - 1

        # Shallow-clone the KV cache so suffix forward doesn't corrupt the prefix cache
        cloned_cache = DynamicCache()
        for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
            cloned_cache.key_cache.append(k)
            cloned_cache.value_cache.append(v)

        hidden_states = suffix_hidden_states
        mask = full_attn_mask_4d.to(dtype=hidden_states.dtype)
        position_embeddings = self.model.llm.rotary_emb(hidden_states, full_positions)

        del full_attn_mask_4d, suffix_attn_mask_2d

        for layer in self.model.action_expert.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=full_positions,
                past_key_value=cloned_cache,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        del cloned_cache, mask, position_embeddings
        hidden_states = self.model.action_expert.model.norm(hidden_states)
        suffix_out = hidden_states[:, -self.config.chunk_size :].clone()
        return suffix_out

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        prefix_padding_mask,
        prefix_attn_mask,
        kv_cache,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)

        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = torch.tensor(
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps,
                device=device,
            )
        else:
            noise_level = torch.tensor(self.config.noise_level, device=device)

        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            prefix_padding_mask, prefix_attn_mask, kv_cache, x_t, t_input
        )
        v_t = self.model.action_out_proj(
            suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        )

        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.chunk_size]
                if self.config.chunk_critic_input
                else suffix_out,
                dim=1,
                keepdim=False,
            )
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros(bsize, device=device)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x_t_mean = (1 - (t_input - delta)) * x0_pred + (t_input - delta) * x1_pred
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
                x_t_mean = (
                    (1 - (t_input - delta)) * x0_pred
                    + (t_input - delta - sigma_i**2 * delta / (2 * t_input)) * x1_pred
                )
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x_t_mean = (
                    (1 - (t_input - delta)) * x0_pred
                    + (t_input - delta) * cos_term * x1_pred
                )
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (t_input - delta) * x1_pred
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
            for i in range(batch_size):
                img_np = raw_images[i].cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
                base_pil_images.append(Image.fromarray(img_np))

            wrist_pil_images = []
            if "observation/wrist_image" in processed_obs:
                for i in range(batch_size):
                    wrist_np = processed_obs["observation/wrist_image"][i].cpu().numpy().astype(np.uint8)
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
            required_num_images = 3
            if num_views < required_num_images:
                pad_size = required_num_images - num_views
                padding = torch.zeros(
                    batch_size, pad_size, *images.shape[2:],
                    dtype=images.dtype, device=device,
                )
                images = torch.cat([images, padding], dim=1)
            image_masks = torch.zeros(
                batch_size, required_num_images, dtype=torch.bool, device=device
            )
            image_masks[:, :num_views] = True

            target_dtype = next(self.parameters()).dtype
            num_steps = self.num_steps

            # Build prefix KV cache
            prefix_padding_mask, prefix_attn_mask, kv_cache = (
                self._build_prefix_kv_cache(input_ids, attention_mask, images, image_masks)
            )

            # Init noise
            x_t = torch.randn(
                batch_size, self.config.chunk_size, self.config.action_dim,
                device=device, dtype=target_dtype,
            )

            chains = [x_t]
            log_probs = []
            values = []

            if self.config.joint_logprob:
                log_probs.append(
                    self.get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
                )

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
                    prefix_padding_mask,
                    prefix_attn_mask,
                    kv_cache,
                    sample_mode,
                    num_steps,
                    compute_values,
                )
                x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
                log_probs.append(self.get_logprob_norm(x_t, x_t_mean, x_t_std))
                values.append(value_t)
                chains.append(x_t)

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
                raise NotImplementedError("use_vlm_value is not supported for DM0")
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

        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            prefix_padding_mask, prefix_attn_mask, kv_cache = (
                self._build_prefix_kv_cache(lang_tokens, lang_masks, images, img_masks)
            )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            chains_log_probs.append(
                self.get_logprob_norm(
                    chains[:, 0],
                    torch.zeros_like(chains[:, 0]),
                    torch.ones_like(chains[:, 0]),
                )
            )
            chains_entropy.append(self.gaussian_entropy(torch.ones_like(chains[:, 0])))
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                prefix_padding_mask,
                prefix_attn_mask,
                kv_cache,
                "train",
                self.config.num_steps,
                compute_values,
            )
            chains_log_probs.append(self.get_logprob_norm(chains_next, x_t_mean, x_t_std))
            chains_entropy.append(self.gaussian_entropy(x_t_std))
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
        config = DM0Config.from_pretrained(cfg.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=False, local_files_only=True
        )
        apply_common_rl_config_overrides(config, cfg)

        # Force SDPA attention to avoid eager attention's O(S²) memory usage
        if hasattr(config, "llm_config") and config.llm_config is not None:
            config.llm_config._attn_implementation = "sdpa"
        if hasattr(config, "action_config") and config.action_config is not None:
            config.action_config._attn_implementation = "sdpa"

        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = DexboticDM0ForRLActionPrediction(config)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline

        model.tokenizer = tokenizer
        model.dm0_tokenization = DM0Tokenization(tokenizer)

        weight_paths = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
        weight_paths = [p for p in weight_paths if not p.endswith(".index.json")]
        if not weight_paths:
            weight_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weights found in {cfg.model_path}")
            weight_paths = [weight_path]
        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)

        # Weights loaded from checkpoint may restore float32 params that
        # DM0's to_bfloat16_for_selected_params() intentionally kept in fp32
        # (layernorms, etc.).  FSDP requires all params within a wrapped unit
        # to share the same dtype, so we enforce uniform dtype across the
        # entire model (including lm_head, value_head, etc.) after load.
        target_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
        model = model.to(dtype=target_dtype)

        # PEVisionTower.device and .dtype use list(self.vision_tower.parameters())[-1]
        # which returns empty when FSDP (use_orig_params=False) has consumed the
        # parameters into a flat tensor.  Patch them with a dynamic fallback:
        # - Try the original parameters()-based lookup first (works for plain models
        #   and FSDP with use_orig_params=True).
        # - Fall back to a cached value only when the parameter list is empty (FSDP
        #   with use_orig_params=False).  The cached dtype is fixed at load time;
        #   the cached device is read from the model's mm_projector (which is always
        #   accessible, even under FSDP) so it follows model.to(device) calls.
        _pe_dtype_fallback = target_dtype
        _pe_model_ref = model  # weak-ish ref; the closure keeps the model alive anyway

        from dexbotic.model.modules.mm_vision.pe.pe_encoder import PEVisionTower

        def _pe_device_property(self):
            params = list(self.vision_tower.parameters())
            if params:
                return params[-1].device
            # FSDP flattened the params; ask the projector instead.
            try:
                return next(_pe_model_ref.model.mm_projector.parameters()).device
            except StopIteration:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _pe_dtype_property(self):
            params = list(self.vision_tower.parameters())
            if params:
                return params[-1].dtype
            return _pe_dtype_fallback

        PEVisionTower.device = property(_pe_device_property)
        PEVisionTower.dtype = property(_pe_dtype_property)

        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.norm_stats = model._read_normalization_stats(norm_stats_file)
        else:
            model.norm_stats = None

        model._train_expert_only = getattr(config, "train_expert_only", False)

    except Exception as e:
        logger.error(f"Failed to load pretrained DM0 model: {e}")
        raise

    input_transforms, output_transforms = build_rl_transform_lists(
        model.norm_stats, state_pad_ndim=config.action_dim
    )
    model.setup_wrappers(
        transforms=input_transforms, output_transforms=output_transforms
    )
    return model
