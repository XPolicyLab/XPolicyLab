from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from psi_policy.model.wan.runtime import WanModel
from psi_policy.policy.wan_video_policy import WanVideoPolicy

from wan.modules.attention import attention as wan_attention
from wan.modules.model import WanLayerNorm
from wan.modules.model import rope_apply
from wan.modules.model import sinusoidal_embedding_1d


class WanSequenceEncoder(nn.Module):
    """Project a low-dimensional sequence into Wan token space."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        max_tokens: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(output_dim)
        self.projector = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(output_dim)),
        )
        self.norm = nn.LayerNorm(int(output_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(max_tokens), int(output_dim)))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.projector(tokens)
        projected = projected + self.pos_embedding[:, : projected.shape[1]]
        return self.norm(projected)


class WanSequenceHead(nn.Module):
    """AdaLN-style decoder head for action velocity prediction."""

    def __init__(self, dim: int, out_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = WanLayerNorm(int(dim), eps)
        self.head = nn.Linear(int(dim), int(out_dim))
        self.modulation = nn.Parameter(torch.randn(1, 2, int(dim)) / float(dim) ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.device.type == "cuda"):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanJointActionPolicy(WanVideoPolicy):
    """
    DreamZero-style approximation:
    - video latent tokens and action tokens share Wan DiT blocks
    - optional proprio/state tokens participate as conditioning tokens
    - video and action are both trained with flow-matching denoising losses
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        state_key: str = "robot_state",
        use_state_condition: bool = True,
        state_hidden_dim: int | None = None,
        action_hidden_dim: int | None = None,
        video_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_key = str(state_key).strip()
        self.use_state_condition = bool(use_state_condition)
        self.video_loss_weight = float(video_loss_weight)
        self.action_loss_weight = float(action_loss_weight)

        if self.action_dim <= 0:
            raise ValueError("WanJointActionPolicy requires action_dim > 0.")
        if self.action_horizon <= 0:
            raise ValueError("WanJointActionPolicy requires action_horizon > 0.")
        if self.video_loss_weight < 0.0 or self.action_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")

        self.state_dim = 0
        if self.use_state_condition:
            state_shape = self.obs_encoder.obs_shapes.get(self.state_key)
            if state_shape is None:
                raise KeyError(
                    f"state_key={self.state_key!r} is not present in obs_encoder.obs_shapes. "
                    "Please add it in the task dataset config."
                )
            if len(state_shape) != 1:
                raise ValueError(f"state_key={self.state_key!r} expects 1D shape, got {state_shape}")
            self.state_dim = int(state_shape[0])

        model_dim = int(self.model.dim)
        model_dtype = next(self.model.parameters()).dtype

        self.action_token_encoder = WanSequenceEncoder(
            input_dim=self.action_dim,
            output_dim=model_dim,
            max_tokens=self.action_horizon,
            hidden_dim=action_hidden_dim,
        ).to(dtype=model_dtype)
        self.action_head = WanSequenceHead(
            dim=model_dim,
            out_dim=self.action_dim,
        ).to(dtype=model_dtype)

        self.state_token_encoder = None
        if self.use_state_condition:
            self.state_token_encoder = WanSequenceEncoder(
                input_dim=self.state_dim,
                output_dim=model_dim,
                max_tokens=max(1, int(self.obs_encoder.obs_horizon)),
                hidden_dim=state_hidden_dim,
            ).to(dtype=model_dtype)

        self._last_loss_components: dict[str, float] = {}

    def _normalize_non_image_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_keys = {
            str(self.obs_encoder.image_key),
            str(self.obs_encoder.target_key),
            str(self.obs_encoder.fallback_target_key),
        }
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in obs_dict.items():
            if key in image_keys:
                normalized[key] = value
                continue
            if key in self.normalizer.params_dict:
                normalized[key] = self.normalizer[key].normalize(value)
            else:
                normalized[key] = value
        return normalized

    def _encode_state_tokens(
        self,
        state_obs: torch.Tensor | None,
        *,
        model_device: torch.device,
        model_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_state_condition:
            return None
        if self.state_token_encoder is None:
            raise RuntimeError("state_token_encoder is not initialized.")
        if state_obs is None:
            raise KeyError(f"Missing state key: {self.state_key}")

        state = state_obs
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"Expected state tensor to be [B,T,D] or [B,D], got {tuple(state.shape)} for key {self.state_key}"
            )
        state = state.to(device=model_device, dtype=next(self.state_token_encoder.parameters()).dtype)
        state_tokens = self.state_token_encoder(state)
        return state_tokens.to(device=model_device, dtype=model_dtype)

    def _build_text_context(self, context_list: List[torch.Tensor], *, model: WanModel) -> torch.Tensor:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        padded_context = []
        for context in context_list:
            local_context = context.to(device=model_device, dtype=model_dtype)
            if local_context.shape[0] > self.text_len:
                local_context = local_context[: self.text_len]
            if local_context.shape[0] < self.text_len:
                local_context = torch.cat(
                    [
                        local_context,
                        local_context.new_zeros(self.text_len - local_context.shape[0], local_context.shape[1]),
                    ],
                    dim=0,
                )
            padded_context.append(local_context)
        return model.text_embedding(torch.stack(padded_context, dim=0))

    def _compute_time_condition(
        self,
        timesteps: torch.Tensor,
        *,
        model: WanModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_device = next(model.parameters()).device
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        elif timesteps.ndim != 2:
            raise ValueError(f"Expected timesteps with shape [B] or [B, L], got {tuple(timesteps.shape)}")

        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=model_device.type == "cuda"):
            bt = timesteps.size(0)
            tl = timesteps.size(1)
            flat_timesteps = timesteps.to(device=model_device, dtype=torch.float32).flatten()
            time_embed = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, flat_timesteps).unflatten(0, (bt, tl)).float()
            )
            modulation = model.time_projection(time_embed).unflatten(2, (6, model.dim))
        return time_embed, modulation

    def _joint_self_attention(
        self,
        *,
        attn_module,
        x: torch.Tensor,
        video_token_count: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        b, l, n, d = x.size(0), x.size(1), attn_module.num_heads, attn_module.head_dim
        q = attn_module.norm_q(attn_module.q(x)).view(b, l, n, d)
        k = attn_module.norm_k(attn_module.k(x)).view(b, l, n, d)
        v = attn_module.v(x).view(b, l, n, d)

        if video_token_count > 0:
            q_video = rope_apply(q[:, :video_token_count], grid_sizes, freqs)
            k_video = rope_apply(k[:, :video_token_count], grid_sizes, freqs)
            if video_token_count < l:
                q = torch.cat([q_video, q[:, video_token_count:]], dim=1)
                k = torch.cat([k_video, k[:, video_token_count:]], dim=1)
            else:
                q = q_video
                k = k_video

        attn_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        out = wan_attention(
            q=q,
            k=k,
            v=v,
            window_size=attn_module.window_size,
            dtype=attn_dtype,
        )
        out = out.flatten(2)
        out = attn_module.o(out)
        return out

    def _joint_block_forward(
        self,
        *,
        block,
        tokens: torch.Tensor,
        modulation: torch.Tensor,
        video_token_count: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=tokens.device.type == "cuda"):
            e = (block.modulation.unsqueeze(0) + modulation).chunk(6, dim=2)

        sa_in = block.norm1(tokens).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        sa_out = self._joint_self_attention(
            attn_module=block.self_attn,
            x=sa_in,
            video_token_count=video_token_count,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=tokens.device.type == "cuda"):
            tokens = tokens + sa_out * e[2].squeeze(2)

        tokens = tokens + block.cross_attn(block.norm3(tokens), context, None)
        ffn_out = block.ffn(
            block.norm2(tokens).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        )
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=tokens.device.type == "cuda"):
            tokens = tokens + ffn_out * e[5].squeeze(2)
        return tokens

    def _run_joint_model(
        self,
        *,
        noisy_latents: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        condition_latents: torch.Tensor,
        latent_condition_mask: torch.Tensor,
        state_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        autocast_context = contextlib.nullcontext()
        if model_device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.autocast(device_type="cuda", dtype=model_dtype)

        noisy_latents = noisy_latents.to(device=model_device, dtype=model_dtype)
        condition_latents = condition_latents.to(device=model_device, dtype=model_dtype)
        latent_condition_mask = latent_condition_mask.to(device=model_device, dtype=model_dtype)
        noisy_actions = noisy_actions.to(
            device=model_device,
            dtype=next(self.action_token_encoder.parameters()).dtype,
        )

        with autocast_context:
            model_inputs = latent_condition_mask * noisy_latents + (1.0 - latent_condition_mask) * condition_latents
            video_embed = model.patch_embedding(model_inputs)
            grid_sizes = torch.tensor(
                [list(video_embed.shape[2:]) for _ in range(video_embed.shape[0])],
                dtype=torch.long,
                device=model_device,
            )
            video_tokens = video_embed.flatten(2).transpose(1, 2).contiguous()
            video_token_count = int(video_tokens.shape[1])

            action_tokens = self.action_token_encoder(noisy_actions)
            action_tokens = action_tokens.to(device=model_device, dtype=model_dtype)
            action_token_count = int(action_tokens.shape[1])

            state_tokens = None
            if self.use_state_condition:
                state_tokens = self._encode_state_tokens(
                    state_obs,
                    model_device=model_device,
                    model_dtype=model_dtype,
                )
            state_token_count = int(state_tokens.shape[1]) if state_tokens is not None else 0

            total_tokens = torch.cat(
                [part for part in (video_tokens, action_tokens, state_tokens) if part is not None],
                dim=1,
            )

            video_timesteps = self._format_masked_timesteps_for_wan(
                timesteps,
                latent_condition_mask=latent_condition_mask,
                seq_len=video_token_count,
                device=model_device,
            )
            action_timesteps = self._format_timesteps_for_wan(
                timesteps,
                seq_len=action_token_count,
                device=model_device,
            )

            video_time_embed, video_mod = self._compute_time_condition(video_timesteps, model=model)
            action_time_embed, action_mod = self._compute_time_condition(action_timesteps, model=model)
            if action_mod.shape[1] == 1:
                action_mod = action_mod.expand(-1, action_token_count, -1, -1)

            modulation_parts = [video_mod, action_mod]
            if state_token_count > 0:
                state_mod = torch.zeros(
                    total_tokens.shape[0],
                    state_token_count,
                    6,
                    model.dim,
                    device=model_device,
                    dtype=torch.float32,
                )
                modulation_parts.append(state_mod)
            total_mod = torch.cat(modulation_parts, dim=1)

            if model.freqs.device != model_device:
                model.freqs = model.freqs.to(model_device)
            context = self._build_text_context(context_list, model=model)

            for block in model.blocks:
                total_tokens = self._joint_block_forward(
                    block=block,
                    tokens=total_tokens,
                    modulation=total_mod,
                    video_token_count=video_token_count,
                    grid_sizes=grid_sizes,
                    freqs=model.freqs,
                    context=context,
                )

            video_tokens = total_tokens[:, :video_token_count]
            action_tokens = total_tokens[:, video_token_count : video_token_count + action_token_count]

            video_pred_tokens = model.head(video_tokens, video_time_embed)
            video_pred = model.unpatchify(video_pred_tokens, grid_sizes)
            video_pred = torch.stack([sample.float() for sample in video_pred], dim=0)
            action_pred = self.action_head(action_tokens, action_time_embed).float()
        return video_pred, action_pred

    def compute_loss(self, batch: dict, training: bool = True) -> torch.Tensor:
        if "obs" not in batch:
            raise KeyError("batch missing obs")
        if "action" not in batch:
            raise KeyError("WanJointActionPolicy requires batch['action'].")

        nobs = self._normalize_non_image_obs(batch["obs"])
        batch_inputs = self.obs_encoder.prepare_batch(nobs, training=training)
        clean_latents = batch_inputs["clean_latents"]
        condition_latents = batch_inputs["condition_latents"]
        latent_condition_mask = batch_inputs["latent_condition_mask"]
        num_condition_latent_steps = int(batch_inputs["num_condition_latent_steps"])

        if "action" in self.normalizer.params_dict:
            clean_actions = self.normalizer["action"].normalize(batch["action"].float())
        else:
            clean_actions = batch["action"].float()

        batch_size = clean_latents.shape[0]
        timestep_idx = torch.randint(
            0,
            self.train_timesteps.shape[0],
            (batch_size,),
            device=self.device,
        )
        timesteps = self.train_timesteps[timestep_idx]

        video_sigmas = self.train_sigmas[timestep_idx].view(batch_size, 1, 1, 1, 1).to(clean_latents.dtype)
        video_noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - video_sigmas) + video_noise * video_sigmas
        target_video_velocity = video_noise - clean_latents

        action_sigmas = self.train_sigmas[timestep_idx].view(batch_size, 1, 1).to(clean_actions.dtype)
        action_noise = torch.randn_like(clean_actions)
        noisy_actions = clean_actions * (1.0 - action_sigmas) + action_noise * action_sigmas
        target_action_velocity = action_noise - clean_actions

        pred_video_velocity, pred_action_velocity = self._run_joint_model(
            noisy_latents=noisy_latents,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            context_list=self._build_context_list(batch_size=batch_size),
            condition_latents=condition_latents,
            latent_condition_mask=latent_condition_mask,
            state_obs=nobs.get(self.state_key),
        )

        if self.use_future_only_loss:
            pred_video_velocity = pred_video_velocity[:, :, num_condition_latent_steps:, :, :]
            target_video_velocity = target_video_velocity[:, :, num_condition_latent_steps:, :, :]

        video_loss = self._compute_loss_tensor(pred_video_velocity, target_video_velocity)
        action_loss = self._compute_loss_tensor(pred_action_velocity, target_action_velocity)
        total_loss = self.video_loss_weight * video_loss + self.action_loss_weight * action_loss
        self._last_loss_components = {
            "video_loss": float(video_loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss

    def _sample_joint_predictions(
        self,
        *,
        obs_dict: Dict[str, torch.Tensor],
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nobs = self._normalize_non_image_obs(obs_dict)
        condition_inputs = self.obs_encoder.prepare_condition_only(nobs)
        condition_latents = condition_inputs["condition_latents"]
        latent_condition_mask = condition_inputs["latent_condition_mask"]
        batch_size, latent_channels, latent_t, latent_h, latent_w = condition_inputs["latent_shape"]

        model_dtype = next(self.model.parameters()).dtype
        latent = torch.randn(
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
            device=self.device,
            dtype=model_dtype,
        )
        action = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=self.device,
            dtype=next(self.action_token_encoder.parameters()).dtype,
        )
        condition_latents = condition_latents.to(device=self.device, dtype=model_dtype)
        latent_condition_mask = latent_condition_mask.to(device=self.device, dtype=model_dtype)
        latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents

        sampling_sigmas = self._build_sigmas(self.num_inference_steps, shift=self.sampling_shift).to(
            device=self.device,
            dtype=torch.float32,
        )
        sampling_timesteps = sampling_sigmas * self.num_train_timesteps
        context_list = self._build_context_list(batch_size=batch_size)

        step_iterator = range(self.num_inference_steps)
        if show_progress:
            from tqdm.auto import tqdm

            step_iterator = tqdm(
                step_iterator,
                total=self.num_inference_steps,
                desc=progress_desc or "Wan joint sampling",
                leave=False,
                dynamic_ncols=True,
            )

        for step_idx in step_iterator:
            timestep = sampling_timesteps[step_idx].expand(batch_size)
            pred_video_velocity, pred_action_velocity = self._run_joint_model(
                noisy_latents=latent,
                noisy_actions=action,
                timesteps=timestep,
                context_list=context_list,
                condition_latents=condition_latents,
                latent_condition_mask=latent_condition_mask,
                state_obs=nobs.get(self.state_key),
            )
            pred_video_velocity = pred_video_velocity.to(latent.dtype)
            pred_action_velocity = pred_action_velocity.to(action.dtype)

            sigma = sampling_sigmas[step_idx]
            sigma_next = (
                sampling_sigmas[step_idx + 1]
                if step_idx + 1 < self.num_inference_steps
                else torch.tensor(0.0, device=self.device, dtype=sampling_sigmas.dtype)
            )
            step_delta = sigma_next - sigma
            latent = latent + pred_video_velocity * step_delta
            latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents
            action = action + pred_action_velocity * step_delta

        if "action" in self.normalizer.params_dict:
            action = self.normalizer["action"].unnormalize(action.float())
        else:
            action = action.float()
        return latent.float(), action

    @torch.no_grad()
    def predict_visual(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del output_path
        visual_pred, action_pred = self._sample_joint_predictions(
            obs_dict=obs_dict,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        return {
            "visual_pred": visual_pred,
            "action_pred": action_pred,
        }

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del output_path
        _, action_pred = self._sample_joint_predictions(obs_dict=obs_dict)
        return {
            "action": action_pred,
            "action_pred": action_pred,
        }

    def get_last_loss_components(self) -> dict[str, float]:
        return dict(self._last_loss_components)


__all__ = ["WanJointActionPolicy"]
