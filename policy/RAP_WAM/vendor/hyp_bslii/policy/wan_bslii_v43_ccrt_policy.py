"""BSLII-v43: CCRT -- Cross-Chunk Rollout Training (the GENUINE rollout direction).

The first version that trains on a REAL second chunk with its OWN GT supervision,
conditioning on the model's OWN drifted prediction of chunk-0 -- i.e. it reproduces,
inside compute_loss, the actual eval-time autoregressive feed (decode chunk-0
prediction -> last frame becomes chunk-1's condition -> supervise chunk-1 on GT).

Why this is different from v40/v41/v42
--------------------------------------
v40 (MSDC, K-step self-denoised consistency), v41 (PRTF, scheduled-sampling), v42
(RWA, real-error warp) ALL operated within a SINGLE 16-frame window: they perturbed
the history synthetically and either enforced action consistency or re-supervised
the SAME future. There was no genuine next chunk, and no decode->encode pixel
round-trip. All three landed in v15's noise band [0.00678, 0.00821] (v40 0.00792 /
v41 0.00774 / v42 0.00805). Their own docstrings predicted this: "cross-chunk
rollout TRAINING (a train-loop change) is required to reach 0.00454."

v43 supplies the missing pieces via a 2-chunk dataset (RynnVLAZarr2ChunkDataset):
  * chunk-0: super().compute_loss (GT condition) -- the normal v15 objective, untouched.
  * chunk-0 prediction: a FEW-step (ccrt_sample_steps) no_grad sampling of chunk-0's
    future, decoded to pixels -> its LAST frame is the model's own (drifted) view of
    "where it ended up" -- exactly what infer.run_autoregressive feeds forward
    (infer.py:674-676: cur_cond = pred_future[-1].repeat(n_obs)).
  * chunk-1: condition on that drifted frame, but SUPERVISE on the GENUINE next-chunk
    GT future (batch["obs2"]) and GT action (batch["action_2"]). This is the real
    DAgger / scheduled-sampling-across-chunks objective: "given your own drifted
    history, predict the true next chunk."
  * total = chunk0_loss + lambda * chunk1_loss.

Safety (lessons v9 SCRG=0.01347, v22-v42, v31 variance band)
------------------------------------------------------------
* ccrt_lambda_max=0 OR no batch["obs2"] OR training=False  -> byte-identical to v15.
  Main chunk-0 loss is NEVER touched. annealed lambda & p; prob-gated.
* chunk-0 prediction is fully no_grad (few-step sampling + VAE decode -> NO graph).
  chunk-1 clean/condition latents are DETACHED (VAE encode -> NO graph). So the ONLY
  extra GRADIENT forward is chunk-1's single _run_joint_model_with_queries -- exactly
  ONE extra grad forward, same order as v40 (confirmed OOM-safe on a single 4090).
  >>> WATCH first ~100 steps for CUDA OOM. <<<
* num_inference_steps is temporarily lowered to ccrt_sample_steps (default 4) for the
  chunk-0 sampling and ALWAYS restored in a finally block (never corrupts eval).
* Any exception in the rollout branch -> caught -> degrade to chunk-0-only loss
  (ccrt_applied=0). Never crashes training.
* Gates on the explicit ``training`` arg (NOT self.training). val_loader -> chunk-0 only.
* No new nn.Parameters -> state_dict identical to v15 -> eval loads v15-style, the
  inference path is UNCHANGED (v43 differs only in the training loss).

Diagnostics: ccrt_lambda_current, ccrt_chunk1_loss, ccrt_applied, ccrt_step.

Author: hyp pipeline (BSLII-v43 plan, user-approved direction A: true cross-chunk rollout).
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

import torch

from hyp_bslii.policy.wan_bslii_v15_residual_action_policy import (
    WanBSLIIv15ResidualActionPolicy,
)


class WanBSLIIv43CCRTPolicy(WanBSLIIv15ResidualActionPolicy):
    """v15 + genuine cross-chunk rollout training (drifted chunk-0 cond -> GT chunk-1)."""

    def __init__(
        self,
        *args,
        ccrt_lambda_max: float = 1.0,
        ccrt_warmup_steps: int = 12000,
        ccrt_p_max: float = 0.25,
        ccrt_sample_steps: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ccrt_lambda_max = float(ccrt_lambda_max)
        self.ccrt_warmup_steps = int(ccrt_warmup_steps)
        self.ccrt_p_max = float(ccrt_p_max)
        self.ccrt_sample_steps = int(ccrt_sample_steps)
        self._ccrt_step = 0
        # cond / future obs keys (used to build the chunk-1 obs dict)
        self._cond_key = (
            getattr(self.obs_encoder, "image_key", None)
            or getattr(self.obs_encoder, "condition_key", None)
            or "rgb_head"
        )
        self._future_key = getattr(self.obs_encoder, "target_key", None) or "rgb_head_future"
        self._dbg_lam = 0.0
        self._dbg_c1 = 0.0
        self._dbg_ap = 0.0

        if not hasattr(self, "_last_loss_components") or self._last_loss_components is None:
            self._last_loss_components = {}  # type: ignore[assignment]
        self._last_loss_components["ccrt_lambda_max"] = float(self.ccrt_lambda_max)
        self._last_loss_components["ccrt_lambda_current"] = 0.0
        self._last_loss_components["ccrt_chunk1_loss"] = 0.0
        self._last_loss_components["ccrt_applied"] = 0.0

    def _anneal(self, mx: float) -> float:
        if self.ccrt_warmup_steps <= 0:
            return mx
        return mx * min(1.0, self._ccrt_step / float(self.ccrt_warmup_steps))

    # ------------------------------------------------------------------
    # chunk-1 supervised loss: drifted chunk-0 prediction as condition,
    # genuine next-chunk GT future/action as supervision.
    # ------------------------------------------------------------------
    def _ccrt_chunk1_loss(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        obs = batch.get("obs", None)
        obs2 = batch.get("obs2", None)
        if obs is None or obs2 is None:
            return None
        cond = obs.get(self._cond_key, None)
        gt_future2 = obs2.get(self._future_key, None)
        if cond is None or gt_future2 is None:
            return None

        # ---- 1) chunk-0 prediction (NO grad, few-step) -> drifted last frame ----
        chunk0_obs: Dict[str, torch.Tensor] = {self._cond_key: cond}
        if self.state_key in obs:
            chunk0_obs[self.state_key] = obs[self.state_key]
        saved_steps = self.num_inference_steps
        try:
            self.num_inference_steps = int(self.ccrt_sample_steps)
            with torch.no_grad():
                pred = self.predict_visual(chunk0_obs, show_progress=False)
                pred_future_pix = self.decode_visual_tokens(pred["visual_pred"])  # [B,T,C,H,W] in [0,1]
        finally:
            self.num_inference_steps = saved_steps
        if pred_future_pix is None or pred_future_pix.shape[1] < 1:
            return None
        drift_last = pred_future_pix[:, -1:].detach().to(cond.dtype)  # [B,1,C,H,W]

        # ---- 2) chunk-1 obs = drifted cond + GENUINE GT future / state ----
        chunk1_obs: Dict[str, torch.Tensor] = {
            self._cond_key: drift_last,
            self._future_key: gt_future2,
        }
        if self.state_key in obs2:
            chunk1_obs[self.state_key] = obs2[self.state_key]

        # ---- 3) chunk-1 supervised flow-matching loss (full gradient on model) ----
        nobs = self._normalize_non_image_obs(chunk1_obs)
        batch_inputs = self.obs_encoder.prepare_batch(nobs, training=True)
        # targets are constants: detach so the ONLY grad path is the model forward
        clean_latents = batch_inputs["clean_latents"].detach()
        condition_latents = batch_inputs["condition_latents"].detach()
        latent_condition_mask = batch_inputs["latent_condition_mask"].detach()
        num_condition_latent_steps = int(batch_inputs["num_condition_latent_steps"])

        batch_size = clean_latents.shape[0]
        timestep_idx = torch.randint(
            0, self.train_timesteps.shape[0], (batch_size,), device=self.device
        )
        timesteps = self.train_timesteps[timestep_idx]

        video_sigmas = (
            self.train_sigmas[timestep_idx].view(batch_size, 1, 1, 1, 1).to(clean_latents.dtype)
        )
        video_noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - video_sigmas) + video_noise * video_sigmas
        target_video_velocity = video_noise - clean_latents

        action2 = batch.get("action_2", None)
        have_action = action2 is not None
        if have_action:
            if "action" in self.normalizer.params_dict:
                clean_actions = self.normalizer["action"].normalize(action2.float())
            else:
                clean_actions = action2.float()
            action_sigmas = (
                self.train_sigmas[timestep_idx].view(batch_size, 1, 1).to(clean_actions.dtype)
            )
            action_noise = torch.randn_like(clean_actions)
            noisy_actions = clean_actions * (1.0 - action_sigmas) + action_noise * action_sigmas
            target_action_velocity = action_noise - clean_actions
        else:
            noisy_actions = torch.randn(
                batch_size, self.action_horizon, self.action_dim,
                device=self.device, dtype=clean_latents.dtype,
            )

        pred_video_velocity, pred_action_velocity, _ = self._run_joint_model_with_queries(
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
        if have_action:
            action_loss = self._compute_loss_tensor(pred_action_velocity, target_action_velocity)
        else:
            action_loss = torch.zeros((), device=self.device, dtype=video_loss.dtype)

        return self.video_loss_weight * video_loss + self.action_loss_weight * action_loss

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, Any], training: bool = True):
        loss = super().compute_loss(batch, training=training)  # chunk-0 main loss (v15, GT cond)
        self._dbg_lam = 0.0
        self._dbg_c1 = 0.0
        self._dbg_ap = 0.0

        if training and self.ccrt_lambda_max > 0.0 and ("obs2" in batch):
            self._ccrt_step += 1
            p = self._anneal(self.ccrt_p_max)
            if random.random() < p:
                try:
                    c1 = self._ccrt_chunk1_loss(batch)
                    if c1 is not None:
                        lam = self._anneal(self.ccrt_lambda_max)
                        if lam > 0.0:
                            loss = loss + lam * c1.to(loss.dtype)
                            self._dbg_lam = float(lam)
                            self._dbg_c1 = float(c1.detach().item())
                            self._dbg_ap = 1.0
                except Exception:
                    self._dbg_ap = 0.0

        try:
            if isinstance(self._last_loss_components, dict):
                self._last_loss_components["ccrt_lambda_current"] = float(self._dbg_lam)
                self._last_loss_components["ccrt_chunk1_loss"] = float(self._dbg_c1)
                self._last_loss_components["ccrt_applied"] = float(self._dbg_ap)
                self._last_loss_components["ccrt_step"] = float(self._ccrt_step)
                self._last_loss_components["total_loss"] = float(loss.detach().item())
        except Exception:
            pass

        return loss


__all__ = ["WanBSLIIv43CCRTPolicy"]
