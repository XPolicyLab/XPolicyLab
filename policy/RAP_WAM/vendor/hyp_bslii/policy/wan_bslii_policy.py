"""Bottleneck Scaling Law for Implicit Imagination (BSLII) policy.

Extends WanJointActionPolicy with a compact set of K learnable latent
"imagination" tokens that participate in the joint Wan transformer alongside
video and action tokens. An auxiliary alignment loss pulls the post-transformer
hidden state at these query positions toward a deterministic compressed
representation of the future video latent (Wan VAE encode), producing a
controllable bottleneck whose capacity (K * align_dim) we can sweep.

Design choices vs WanJointActionPolicy:
- Latent queries are appended after state tokens so they participate in joint
  self-attention through all DiT blocks without changing the existing token
  ordering for video/action/state.
- Time modulation for the queries uses the action timestep schedule so the
  AdaLN-Zero modulations remain on-distribution (queries co-evolve with action
  denoising rather than video denoising).
- Alignment target is computed inside ``compute_loss`` from ``clean_latents``
  (already produced by the obs_encoder) via deterministic spatial+temporal
  pooling into K positions, projected to ``align_dim``. No new trainable params
  on the target side except a single Linear projection (channels 48 -> align_dim).
- K=0 falls back to pure WanJointActionPolicy behaviour.

Authors: hyp pipeline auto-generated (BSLII plan).
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from psi_policy.policy.wan_joint_action_policy import WanJointActionPolicy
from psi_policy.model.wan.runtime import WanModel


class WanBSLIIPolicy(WanJointActionPolicy):
    """Wan joint policy with K learnable latent imagination tokens.

    Adds:
    - ``self.latent_queries``: nn.Parameter of shape (1, K, model.dim).
    - ``self.query_proj``: project transformer hidden at query positions to
      align_dim.
    - ``self.target_proj``: project pooled future-latent channels to align_dim.

    Hyperparameters:
        num_latent_queries: K, the bottleneck capacity. K in {0, 1, 4, 16, 64,
            256} for the scaling law sweep. K=0 disables latent queries entirely
            and produces the parent's behaviour.
        align_loss_weight: lambda_align. Default 0.1.
        align_dim: dimensionality of the alignment target / probe. Default 128.
        align_loss_type: "cosine" (1 - cosine_similarity) or "mse".
        target_temporal_pool: temporal pooling size for future latent (1 means
            average across all future timesteps to a single slot).
    """

    def __init__(
        self,
        *,
        num_latent_queries: int = 16,
        align_loss_weight: float = 0.1,
        align_dim: int = 128,
        align_loss_type: str = "cosine",
        align_query_init_std: float = 0.02,
        target_temporal_pool: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.K = int(num_latent_queries)
        if self.K < 0:
            raise ValueError("num_latent_queries must be >= 0")

        self.align_loss_weight = float(align_loss_weight)
        self.align_loss_type = str(align_loss_type).lower()
        if self.align_loss_type not in ("cosine", "mse"):
            raise ValueError(f"Unknown align_loss_type: {align_loss_type}")
        self.align_dim = int(align_dim)
        self.target_temporal_pool = max(1, int(target_temporal_pool))

        if self.K > 0:
            model_dim = int(self.model.dim)
            model_dtype = next(self.model.parameters()).dtype
            param_dtype = torch.float32  # parameters live in fp32; cast at forward time

            self.latent_queries = nn.Parameter(
                torch.randn(1, self.K, model_dim, dtype=param_dtype) * float(align_query_init_std)
            )
            # Project Wan VAE channels (typically 48 for Wan2.2-TI2V-5B) -> align_dim.
            self.target_proj = nn.Linear(48, self.align_dim).to(dtype=param_dtype)
            # Project transformer hidden at query positions -> align_dim.
            self.query_proj = nn.Linear(model_dim, self.align_dim).to(dtype=param_dtype)
            self._model_dtype = model_dtype
        else:
            self.latent_queries = None
            self.target_proj = None
            self.query_proj = None
            self._model_dtype = next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # Joint forward with latent queries appended
    # ------------------------------------------------------------------
    def _run_joint_model_with_queries(
        self,
        *,
        noisy_latents: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        condition_latents: torch.Tensor,
        latent_condition_mask: torch.Tensor,
        state_obs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward through the joint Wan transformer with K latent queries.

        Returns (pred_video_velocity, pred_action_velocity, query_hidden).
        ``query_hidden`` is (B, K, model.dim) or None when K=0.
        """
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
            model_inputs = (
                latent_condition_mask * noisy_latents
                + (1.0 - latent_condition_mask) * condition_latents
            )
            video_embed = model.patch_embedding(model_inputs)
            grid_sizes = torch.tensor(
                [list(video_embed.shape[2:]) for _ in range(video_embed.shape[0])],
                dtype=torch.long,
                device=model_device,
            )
            video_tokens = video_embed.flatten(2).transpose(1, 2).contiguous()
            B = video_tokens.shape[0]
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

            query_tokens = None
            query_token_count = 0
            if self.K > 0:
                # Expand learnable queries to batch.
                query_tokens = self.latent_queries.to(device=model_device, dtype=model_dtype)
                query_tokens = query_tokens.expand(B, -1, -1).contiguous()
                query_token_count = self.K

            parts = [video_tokens, action_tokens]
            if state_tokens is not None:
                parts.append(state_tokens)
            if query_tokens is not None:
                parts.append(query_tokens)
            total_tokens = torch.cat(parts, dim=1)

            # Time conditioning.
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

            video_time_embed, video_mod = self._compute_time_condition(
                video_timesteps, model=model
            )
            action_time_embed, action_mod = self._compute_time_condition(
                action_timesteps, model=model
            )
            if action_mod.shape[1] == 1:
                action_mod = action_mod.expand(-1, action_token_count, -1, -1)

            modulation_parts = [video_mod, action_mod]
            if state_token_count > 0:
                modulation_parts.append(
                    torch.zeros(
                        B,
                        state_token_count,
                        6,
                        model.dim,
                        device=model_device,
                        dtype=torch.float32,
                    )
                )
            if query_token_count > 0:
                # Use action-side time embed as the query modulation so AdaLN-Zero
                # stays on-distribution for the new positions.
                query_mod = action_mod[:, :1].expand(-1, query_token_count, -1, -1).contiguous()
                modulation_parts.append(query_mod)
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
            action_tokens = total_tokens[
                :, video_token_count : video_token_count + action_token_count
            ]
            query_hidden = None
            if query_token_count > 0:
                query_start = video_token_count + action_token_count + state_token_count
                query_end = query_start + query_token_count
                query_hidden = total_tokens[:, query_start:query_end]

            video_pred_tokens = model.head(video_tokens, video_time_embed)
            video_pred = model.unpatchify(video_pred_tokens, grid_sizes)
            video_pred = torch.stack([sample.float() for sample in video_pred], dim=0)
            action_pred = self.action_head(action_tokens, action_time_embed).float()
        return video_pred, action_pred, query_hidden

    # ------------------------------------------------------------------
    # Alignment target: pool future VAE latent to (B, K, align_dim)
    # ------------------------------------------------------------------
    def _build_align_target(
        self,
        clean_latents: torch.Tensor,
        num_condition_latent_steps: int,
    ) -> torch.Tensor:
        """Build a (B, K, align_dim) alignment target from the future portion
        of the clean Wan VAE latent.

        clean_latents shape: (B, C=48, T, H', W'). The future slice starts at
        ``num_condition_latent_steps`` along T.
        """
        if self.K <= 0:
            raise RuntimeError("_build_align_target called with K=0")

        device = clean_latents.device
        future = clean_latents[:, :, num_condition_latent_steps:]
        if future.shape[2] == 0:
            future = clean_latents  # fallback if no future portion
        B, C, T, H, W = future.shape
        # Choose a roughly square spatial grid of K = Kh * Kw tokens.
        Kh, Kw = self._pick_spatial_grid(self.K)
        Kt = self.target_temporal_pool

        future = future.float()
        pooled = F.adaptive_avg_pool3d(future, output_size=(Kt, Kh, Kw))
        # If Kt > 1, we collapse it together with spatial: K' = Kt * Kh * Kw.
        if Kt * Kh * Kw != self.K:
            # Fallback: reshape after pooling to exactly K tokens by truncation.
            pooled = pooled.reshape(B, C, -1)
            if pooled.shape[-1] >= self.K:
                pooled = pooled[..., : self.K]
            else:
                pad = self.K - pooled.shape[-1]
                pooled = F.pad(pooled, (0, pad))
        else:
            pooled = pooled.reshape(B, C, self.K)
        pooled = pooled.transpose(1, 2)  # (B, K, C)

        target = self.target_proj(pooled.to(dtype=self.target_proj.weight.dtype))
        return target

    @staticmethod
    def _pick_spatial_grid(K: int) -> Tuple[int, int]:
        """Pick (Kh, Kw) such that Kh*Kw == K, with Kh closest to Kw."""
        if K <= 0:
            return (0, 0)
        best = (1, K)
        for h in range(1, K + 1):
            if K % h == 0:
                w = K // h
                if abs(h - w) < abs(best[0] - best[1]):
                    best = (h, w)
        return best

    def _compute_align_loss(
        self,
        query_hidden: torch.Tensor,
        align_target: torch.Tensor,
    ) -> torch.Tensor:
        """Alignment loss between projected query hidden states and target.

        Cosine: 1 - mean(cos_sim) per token, averaged.
        MSE: torch.nn.functional.mse_loss.
        """
        q_proj = self.query_proj(query_hidden.to(dtype=self.query_proj.weight.dtype))
        target = align_target.to(dtype=q_proj.dtype)
        if self.align_loss_type == "cosine":
            q_norm = F.normalize(q_proj, dim=-1)
            t_norm = F.normalize(target, dim=-1)
            return (1.0 - (q_norm * t_norm).sum(dim=-1)).mean()
        return F.mse_loss(q_proj, target)

    # ------------------------------------------------------------------
    # compute_loss override
    # ------------------------------------------------------------------
    def compute_loss(self, batch: dict, training: bool = True) -> torch.Tensor:
        if "obs" not in batch:
            raise KeyError("batch missing obs")
        if "action" not in batch:
            raise KeyError("WanBSLIIPolicy requires batch['action'].")

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

        video_sigmas = (
            self.train_sigmas[timestep_idx]
            .view(batch_size, 1, 1, 1, 1)
            .to(clean_latents.dtype)
        )
        video_noise = torch.randn_like(clean_latents)
        noisy_latents = (
            clean_latents * (1.0 - video_sigmas) + video_noise * video_sigmas
        )
        target_video_velocity = video_noise - clean_latents

        action_sigmas = (
            self.train_sigmas[timestep_idx]
            .view(batch_size, 1, 1)
            .to(clean_actions.dtype)
        )
        action_noise = torch.randn_like(clean_actions)
        noisy_actions = (
            clean_actions * (1.0 - action_sigmas) + action_noise * action_sigmas
        )
        target_action_velocity = action_noise - clean_actions

        pred_video_velocity, pred_action_velocity, query_hidden = (
            self._run_joint_model_with_queries(
                noisy_latents=noisy_latents,
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                context_list=self._build_context_list(batch_size=batch_size),
                condition_latents=condition_latents,
                latent_condition_mask=latent_condition_mask,
                state_obs=nobs.get(self.state_key),
            )
        )

        if self.use_future_only_loss:
            pred_video_velocity = pred_video_velocity[:, :, num_condition_latent_steps:, :, :]
            target_video_velocity = target_video_velocity[
                :, :, num_condition_latent_steps:, :, :
            ]

        video_loss = self._compute_loss_tensor(pred_video_velocity, target_video_velocity)
        action_loss = self._compute_loss_tensor(pred_action_velocity, target_action_velocity)

        align_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        if self.K > 0 and self.align_loss_weight > 0.0 and query_hidden is not None:
            with torch.no_grad():
                align_target = self._build_align_target(
                    clean_latents.detach(), num_condition_latent_steps
                )
            align_loss = self._compute_align_loss(query_hidden, align_target.detach())

        total_loss = (
            self.video_loss_weight * video_loss
            + self.action_loss_weight * action_loss
            + self.align_loss_weight * align_loss
        )

        self._last_loss_components = {
            "video_loss": float(video_loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "align_loss": float(align_loss.detach().item()) if self.K > 0 else 0.0,
            "total_loss": float(total_loss.detach().item()),
            "K": float(self.K),
        }
        return total_loss


__all__ = ["WanBSLIIPolicy"]
