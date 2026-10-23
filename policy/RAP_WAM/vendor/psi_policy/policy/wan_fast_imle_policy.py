from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from psi_policy.model.wan.runtime import WanModel
from psi_policy.policy.wan_joint_action_policy import WanSequenceEncoder
from psi_policy.policy.wan_video_policy import WanVideoPolicy

from wan.modules.attention import attention as wan_attention
from wan.modules.model import WanLayerNorm
from wan.modules.model import rope_apply
from wan.modules.model import sinusoidal_embedding_1d


class WanActionDirectHead(nn.Module):
    """Direct action decoder head for IMLE trajectories."""

    def __init__(self, dim: int, out_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = WanLayerNorm(int(dim), eps)
        self.head = nn.Linear(int(dim), int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.device.type == "cuda"):
            x = self.head(self.norm(x))
        return x


class WanFastImlePolicy(WanVideoPolicy):
    """
    Strongly coupled Wan policy:
    - training: full joint graph over video/action/state tokens
    - video branch: flow matching on future video tokens
    - action branch: direct IMLE trajectory prediction
    - inference: action uses the closed prefix/action/state subgraph only
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
        n_action_samples_per_condition: int = 8,
        rs_imle_epsilon: float = 0.03,
        inference_num_action_samples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_key = str(state_key).strip()
        self.use_state_condition = bool(use_state_condition)
        self.video_loss_weight = float(video_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self.n_action_samples_per_condition = int(n_action_samples_per_condition)
        self.rs_imle_epsilon = float(rs_imle_epsilon)
        self.inference_num_action_samples = int(inference_num_action_samples)

        if self.action_dim <= 0:
            raise ValueError("WanFastImlePolicy requires action_dim > 0.")
        if self.action_horizon <= 0:
            raise ValueError("WanFastImlePolicy requires action_horizon > 0.")
        if self.n_action_samples_per_condition <= 0:
            raise ValueError("n_action_samples_per_condition must be >= 1.")
        if self.inference_num_action_samples <= 0:
            raise ValueError("inference_num_action_samples must be >= 1.")
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

        self.action_query_encoder = WanSequenceEncoder(
            input_dim=self.action_dim,
            output_dim=model_dim,
            max_tokens=self.action_horizon,
            hidden_dim=action_hidden_dim,
        ).to(dtype=model_dtype)
        self.action_output_head = WanActionDirectHead(
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
        timesteps = timesteps.to(device=next(model.parameters()).device, dtype=torch.float32)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        elif timesteps.ndim != 2:
            raise ValueError(f"Expected timesteps with shape [B] or [B, L], got {tuple(timesteps.shape)}")

        batch_size, seq_len = timesteps.shape
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=timesteps.device.type == "cuda"):
            time_embed = model.time_embedding(
                sinusoidal_embedding_1d(
                    model.freq_dim,
                    timesteps.reshape(-1),
                ).unflatten(0, (batch_size, seq_len)).float()
            )
            time_mod = model.time_projection(time_embed).unflatten(2, (6, model.dim))
        return time_embed, time_mod

    def _build_prefix_video_token_mask(
        self,
        latent_condition_mask: torch.Tensor,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        patch_t, patch_h, patch_w = self.patch_size
        token_mask = latent_condition_mask[:, 0, ::patch_t, ::patch_h, ::patch_w].contiguous().view(
            latent_condition_mask.shape[0], -1
        )
        if token_mask.shape[1] != seq_len:
            raise ValueError(
                f"Tokenized condition mask mismatch: expected seq_len={seq_len}, got {token_mask.shape[1]}"
            )
        return token_mask.to(device=device) < 0.5

    def _build_joint_attention_mask(
        self,
        prefix_video_token_mask: torch.Tensor,
        *,
        action_token_count: int,
        state_token_count: int,
    ) -> torch.Tensor | None:
        future_video_token_mask = ~prefix_video_token_mask
        if not bool(future_video_token_mask.any().item()):
            return None

        device = prefix_video_token_mask.device
        batch_size, video_token_count = prefix_video_token_mask.shape
        closed_query_mask = torch.cat(
            [
                prefix_video_token_mask,
                torch.ones(
                    batch_size,
                    action_token_count + state_token_count,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        future_key_mask = torch.cat(
            [
                future_video_token_mask,
                torch.zeros(
                    batch_size,
                    action_token_count + state_token_count,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        total_len = video_token_count + action_token_count + state_token_count
        allow_mask = torch.ones(batch_size, total_len, total_len, device=device, dtype=torch.bool)
        block_mask = closed_query_mask.unsqueeze(2) & future_key_mask.unsqueeze(1)
        allow_mask = allow_mask & (~block_mask)
        return allow_mask

    def _joint_self_attention(
        self,
        *,
        attn_module,
        x: torch.Tensor,
        video_token_count: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: torch.Tensor | None,
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
        if attn_mask is None:
            out = wan_attention(
                q=q,
                k=k,
                v=v,
                window_size=attn_module.window_size,
                dtype=attn_dtype,
            )
        else:
            q = q.transpose(1, 2).to(attn_dtype)
            k = k.transpose(1, 2).to(attn_dtype)
            v = v.transpose(1, 2).to(attn_dtype)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.unsqueeze(1),
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous()

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
        attn_mask: torch.Tensor | None,
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
            attn_mask=attn_mask,
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

    def _sample_action_queries(self, *, batch_size: int, num_samples: int) -> torch.Tensor:
        return torch.randn(
            batch_size * num_samples,
            self.action_horizon,
            self.action_dim,
            device=self.device,
            dtype=next(self.action_query_encoder.parameters()).dtype,
        )

    def _repeat_batch(self, tensor: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
        if tensor is None or repeats == 1:
            return tensor
        return tensor.repeat_interleave(int(repeats), dim=0)

    def _run_joint_model(
        self,
        *,
        video_latents: torch.Tensor,
        action_queries: torch.Tensor,
        context_list: List[torch.Tensor],
        state_obs: torch.Tensor | None,
        timesteps: torch.Tensor | None,
        latent_condition_mask: torch.Tensor | None,
        return_video: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        model = self.model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        autocast_context = contextlib.nullcontext()
        if model_device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.autocast(device_type="cuda", dtype=model_dtype)

        video_latents = video_latents.to(device=model_device, dtype=model_dtype)
        action_queries = action_queries.to(
            device=model_device,
            dtype=next(self.action_query_encoder.parameters()).dtype,
        )
        if latent_condition_mask is not None:
            latent_condition_mask = latent_condition_mask.to(device=model_device, dtype=model_dtype)

        with autocast_context:
            video_embed = model.patch_embedding(video_latents)
            grid_sizes = torch.tensor(
                [list(video_embed.shape[2:]) for _ in range(video_embed.shape[0])],
                dtype=torch.long,
                device=model_device,
            )
            video_tokens = video_embed.flatten(2).transpose(1, 2).contiguous()
            batch_size = int(video_tokens.shape[0])
            video_token_count = int(video_tokens.shape[1])

            if latent_condition_mask is None:
                prefix_video_token_mask = torch.ones(
                    batch_size,
                    video_token_count,
                    device=model_device,
                    dtype=torch.bool,
                )
            else:
                prefix_video_token_mask = self._build_prefix_video_token_mask(
                    latent_condition_mask,
                    seq_len=video_token_count,
                    device=model_device,
                )

            action_tokens = self.action_query_encoder(action_queries)
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

            if timesteps is None:
                video_time_embed = torch.zeros(
                    batch_size,
                    video_token_count,
                    model.dim,
                    device=model_device,
                    dtype=torch.float32,
                )
                video_mod = torch.zeros(
                    batch_size,
                    video_token_count,
                    6,
                    model.dim,
                    device=model_device,
                    dtype=torch.float32,
                )
            else:
                if latent_condition_mask is None:
                    video_timesteps = self._format_timesteps_for_wan(
                        timesteps,
                        seq_len=video_token_count,
                        device=model_device,
                    )
                    if video_timesteps.shape[1] == 1:
                        video_timesteps = video_timesteps.expand(-1, video_token_count)
                else:
                    video_timesteps = self._format_masked_timesteps_for_wan(
                        timesteps,
                        latent_condition_mask=latent_condition_mask,
                        seq_len=video_token_count,
                        device=model_device,
                    )
                video_time_embed, video_mod = self._compute_time_condition(video_timesteps, model=model)

            action_mod = torch.zeros(
                batch_size,
                action_token_count,
                6,
                model.dim,
                device=model_device,
                dtype=torch.float32,
            )

            modulation_parts = [video_mod, action_mod]
            if state_token_count > 0:
                state_mod = torch.zeros(
                    batch_size,
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
            attn_mask = self._build_joint_attention_mask(
                prefix_video_token_mask,
                action_token_count=action_token_count,
                state_token_count=state_token_count,
            )

            for block in model.blocks:
                total_tokens = self._joint_block_forward(
                    block=block,
                    tokens=total_tokens,
                    modulation=total_mod,
                    video_token_count=video_token_count,
                    grid_sizes=grid_sizes,
                    freqs=model.freqs,
                    context=context,
                    attn_mask=attn_mask,
                )

            video_tokens = total_tokens[:, :video_token_count]
            action_tokens = total_tokens[:, video_token_count : video_token_count + action_token_count]
            action_pred = self.action_output_head(action_tokens).float()

            video_pred = None
            if return_video:
                video_pred_tokens = model.head(video_tokens, video_time_embed)
                video_pred = model.unpatchify(video_pred_tokens, grid_sizes)
                video_pred = torch.stack([sample.float() for sample in video_pred], dim=0)

        return video_pred, action_pred

    def _run_action_model(
        self,
        *,
        prefix_latents: torch.Tensor,
        action_queries: torch.Tensor,
        context_list: List[torch.Tensor],
        state_obs: torch.Tensor | None,
    ) -> torch.Tensor:
        _, action_pred = self._run_joint_model(
            video_latents=prefix_latents,
            action_queries=action_queries,
            context_list=context_list,
            state_obs=state_obs,
            timesteps=None,
            latent_condition_mask=None,
            return_video=False,
        )
        return action_pred

    def _rs_imle_assign(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        epsilon: float = 0.03,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = real_samples.shape[0]
        real_flat = real_samples.reshape(batch_size, 1, -1)
        fake_flat = fake_samples.reshape(batch_size, fake_samples.shape[1], -1)

        distances = torch.cdist(real_flat, fake_flat).squeeze(1)
        valid_samples = distances > float(epsilon)
        valid_real = valid_samples.any(dim=1)

        max_dist = distances.max().detach()
        masked_distances = distances.masked_fill(~valid_samples, max_dist)
        matched_idx = masked_distances.argmin(dim=1)
        fallback_idx = distances.argmin(dim=1)
        matched_idx = torch.where(valid_real, matched_idx, fallback_idx)
        matched_distances = distances.gather(1, matched_idx.unsqueeze(1)).squeeze(1)
        return matched_idx, valid_real, matched_distances

    def _reduce_imle_distances(
        self,
        matched_distances: torch.Tensor,
        valid_matches: torch.Tensor,
    ) -> torch.Tensor:
        valid_float = valid_matches.float()
        loss = (matched_distances * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        return torch.where(valid_matches.any(), loss, torch.zeros_like(loss))

    @torch.no_grad()
    def _select_action_queries(
        self,
        *,
        prefix_latents: torch.Tensor,
        clean_actions: torch.Tensor,
        state_obs: torch.Tensor | None,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(prefix_latents.shape[0])
        candidate_queries = self._sample_action_queries(
            batch_size=batch_size,
            num_samples=num_samples,
        ).reshape(batch_size, num_samples, self.action_horizon, self.action_dim)
        candidate_actions = self._run_action_model(
            prefix_latents=prefix_latents.repeat_interleave(num_samples, dim=0),
            action_queries=candidate_queries.reshape(batch_size * num_samples, self.action_horizon, self.action_dim),
            context_list=self._build_context_list(batch_size=batch_size * num_samples),
            state_obs=self._repeat_batch(state_obs, num_samples),
        ).reshape(batch_size, num_samples, self.action_horizon, self.action_dim)
        matched_idx, valid_matches, _ = self._rs_imle_assign(
            clean_actions,
            candidate_actions,
            epsilon=self.rs_imle_epsilon,
        )
        batch_idx = torch.arange(batch_size, device=candidate_queries.device)
        selected_queries = candidate_queries[batch_idx, matched_idx]
        return selected_queries, valid_matches

    def compute_loss(self, batch: dict, training: bool = True) -> torch.Tensor:
        if "obs" not in batch:
            raise KeyError("batch missing obs")
        if "action" not in batch:
            raise KeyError("WanFastImlePolicy requires batch['action'].")

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

        prefix_latents = condition_latents[:, :, :num_condition_latent_steps].contiguous()
        if self.action_loss_weight > 0.0:
            selected_action_queries, valid_matches = self._select_action_queries(
                prefix_latents=prefix_latents,
                clean_actions=clean_actions,
                state_obs=nobs.get(self.state_key),
                num_samples=self.n_action_samples_per_condition,
            )
        else:
            selected_action_queries = self._sample_action_queries(
                batch_size=clean_latents.shape[0],
                num_samples=1,
            )
            valid_matches = torch.zeros(clean_latents.shape[0], device=self.device, dtype=torch.bool)

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
        model_inputs = latent_condition_mask * noisy_latents + (1.0 - latent_condition_mask) * condition_latents

        pred_video_velocity, pred_action = self._run_joint_model(
            video_latents=model_inputs,
            action_queries=selected_action_queries,
            context_list=self._build_context_list(batch_size=batch_size),
            state_obs=nobs.get(self.state_key),
            timesteps=timesteps,
            latent_condition_mask=latent_condition_mask,
            return_video=True,
        )
        if pred_video_velocity is None:
            raise RuntimeError("Joint training path expects video predictions.")

        if self.use_future_only_loss:
            pred_video_velocity = pred_video_velocity[:, :, num_condition_latent_steps:, :, :]
            target_video_velocity = target_video_velocity[:, :, num_condition_latent_steps:, :, :]

        if self.video_loss_weight > 0.0:
            video_loss = self._compute_loss_tensor(pred_video_velocity, target_video_velocity)
        else:
            video_loss = pred_action.new_zeros(())

        if self.action_loss_weight > 0.0:
            matched_distances = torch.linalg.vector_norm(
                (pred_action - clean_actions).reshape(batch_size, -1),
                dim=1,
            )
            action_loss = self._reduce_imle_distances(matched_distances, valid_matches)
        else:
            action_loss = pred_action.new_zeros(())

        total_loss = self.video_loss_weight * video_loss + self.action_loss_weight * action_loss
        self._last_loss_components = {
            "video_loss": float(video_loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss

    def _predict_action_from_normalized_obs(
        self,
        nobs: Dict[str, torch.Tensor],
        *,
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        condition_inputs = self.obs_encoder.prepare_condition_only(nobs)
        condition_latents = condition_inputs["condition_latents"]
        num_condition_latent_steps = int(condition_inputs["num_condition_latent_steps"])
        batch_size = int(condition_latents.shape[0])
        prefix_latents = condition_latents[:, :, :num_condition_latent_steps].contiguous()

        action_queries = self._sample_action_queries(batch_size=batch_size, num_samples=num_samples)
        pred_actions = self._run_action_model(
            prefix_latents=prefix_latents.repeat_interleave(num_samples, dim=0),
            action_queries=action_queries,
            context_list=self._build_context_list(batch_size=batch_size * num_samples),
            state_obs=self._repeat_batch(nobs.get(self.state_key), num_samples),
        )
        pred_actions = pred_actions.reshape(batch_size, num_samples, self.action_horizon, self.action_dim)

        if "action" in self.normalizer.params_dict:
            pred_actions = self.normalizer["action"].unnormalize(pred_actions.float())
        else:
            pred_actions = pred_actions.float()

        action_pred = pred_actions[:, 0]
        result = {
            "action": action_pred,
            "action_pred": action_pred,
        }
        if num_samples > 1:
            result["action_samples"] = pred_actions
        return result

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
        condition_latents = condition_latents.to(device=self.device, dtype=model_dtype)
        latent_condition_mask = latent_condition_mask.to(device=self.device, dtype=model_dtype)
        latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents

        action_queries = self._sample_action_queries(batch_size=batch_size, num_samples=1)
        sampling_sigmas = self._build_sigmas(self.num_inference_steps, shift=self.sampling_shift).to(
            device=self.device,
            dtype=torch.float32,
        )
        sampling_timesteps = sampling_sigmas * self.num_train_timesteps
        context_list = self._build_context_list(batch_size=batch_size)

        pred_action = None
        step_iterator = range(self.num_inference_steps)
        if show_progress:
            from tqdm.auto import tqdm

            step_iterator = tqdm(
                step_iterator,
                total=self.num_inference_steps,
                desc=progress_desc or "Wan fast-IMLE joint sampling",
                leave=False,
                dynamic_ncols=True,
            )

        for step_idx in step_iterator:
            timestep = sampling_timesteps[step_idx].expand(batch_size)
            pred_video_velocity, pred_action = self._run_joint_model(
                video_latents=latent,
                action_queries=action_queries,
                context_list=context_list,
                state_obs=nobs.get(self.state_key),
                timesteps=timestep,
                latent_condition_mask=latent_condition_mask,
                return_video=True,
            )
            if pred_video_velocity is None:
                raise RuntimeError("Joint sampling expects video predictions.")

            pred_video_velocity = pred_video_velocity.to(latent.dtype)
            sigma = sampling_sigmas[step_idx]
            sigma_next = (
                sampling_sigmas[step_idx + 1]
                if step_idx + 1 < self.num_inference_steps
                else torch.tensor(0.0, device=self.device, dtype=sampling_sigmas.dtype)
            )
            latent = latent + pred_video_velocity * (sigma_next - sigma)
            latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents

        if pred_action is None:
            raise RuntimeError("Joint sampling produced no action prediction.")
        if "action" in self.normalizer.params_dict:
            pred_action = self.normalizer["action"].unnormalize(pred_action.float())
        else:
            pred_action = pred_action.float()
        return latent.float(), pred_action

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del output_path
        nobs = self._normalize_non_image_obs(obs_dict)
        return self._predict_action_from_normalized_obs(
            nobs,
            num_samples=self.inference_num_action_samples,
        )

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
            "action": action_pred,
            "action_pred": action_pred,
        }

    def get_last_loss_components(self) -> dict[str, float]:
        return dict(self._last_loss_components)


__all__ = ["WanFastImlePolicy"]
