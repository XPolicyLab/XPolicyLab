"""BSLII-v50: OPSR -- On-Policy State Rollout (closed-loop state scheduled sampling).

Closes the gap that v43-v49 ALL left open. The real-robot closed-loop eval shows a
~13x optimism gap: with GT-refreshed state each chunk, action_mse ~0.007; with the TRUE
closed loop (next-chunk input state <- the policy's OWN last predicted action, a 28-D
absolute joint-angle command) action_mse ~0.09. v43-v49 (CCRT/CCRTL/RDM/FPR/RA/FDM/CKW)
all plateaued at 0.089-0.098 because their rollout supervision only simulated *video*
drift -- the state/action CONDITION was ALWAYS GT. The state-closed-loop distribution
shift the eval exposes was never present at training time.

v50 fix -- on-policy state rollout via scheduled sampling
---------------------------------------------------------
Reuse v43's CCRT chunk-1 machinery (chunk-0 no-grad few-step prediction -> drifted video
condition -> chunk-1 supervised on GT action_2). v43's chunk-0 predict_visual() ALREADY
returns action_pred (raw 28-D) for free; v43 just discards it. v50 keeps it and, with an
annealed probability p, feeds the LAST predicted action step as the chunk-1 INPUT STATE
(instead of the GT state). This pushes the training condition from "GT-state" toward
"predicted-state" -- exactly the closed-loop eval's distribution (state <- pred_action[-1],
normalized via normalizer["robot_state"]; state ~= action physically). Scheduled-sampling
anneal (p: 0 -> opsr_p_max over opsr_warmup_steps) keeps early training grounded on GT so
the policy first learns the base skill, THEN learns to RECOVER from its own drifted state.

Academic basis: DAgger (Ross & Bagnell, AISTATS 2011) + Scheduled Sampling (Bengio et al.,
NeurIPS 2015), realized on a video-action flow-matching policy. ORTHOGONAL to v43-v49's
loss-reweighting line (this acts on the CONDITIONING distribution, not the loss term).

Implementation / safety
-----------------------
* Inherits v43 CCRT. Overrides ONLY __init__ (+opsr_p_max/opsr_warmup_steps) and
  _ccrt_chunk1_loss (v43 line-for-line, except the ">>> v50" step-2 state swap).
* REUSES v43's single chunk-0 no-grad predict_visual call -> the predicted action is
  FREE (zero extra forward, zero extra memory vs v43; same OOM profile).
* opsr_p_max=0 -> state always GT -> byte-identical to v43. Single controlled variable
  vs v43 = opsr_p_max. compute_loss / anneal / prob-gate / try-except all inherited.
* No new nn.Parameter -> state_dict == v15 -> eval path unchanged.
* pred-action -> state slot is raw 28-D absolute joints; _normalize_non_image_obs
  normalizes it via normalizer["robot_state"] (state~=action; same assumption as the
  closed-loop eval's `closed` mode). Shape strictly matched to obs2[state_key].

Diagnostics: inherits ccrt_* + adds opsr_p_max / opsr_used (1.0 if this step used
predicted state, else 0.0).

Author: hyp pipeline (BSLII-v50 plan; on-policy state rollout, scheduled sampling).
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

import torch

from hyp_bslii.policy.wan_bslii_v43_ccrt_policy import WanBSLIIv43CCRTPolicy


class WanBSLIIv50OPSRPolicy(WanBSLIIv43CCRTPolicy):
    """v43 CCRT + closed-loop state scheduled sampling on the chunk-1 INPUT state.

    With annealed probability p, the chunk-1 conditioning state is the chunk-0 PREDICTED
    action's last step (closed-loop, the real-robot eval distribution) instead of the GT
    state -- so the policy learns to recover from its own drifted proprioception.
    """

    def __init__(self, *args, opsr_p_max: float = 0.5, opsr_warmup_steps: int = 3000, **kwargs):
        super().__init__(*args, **kwargs)
        self.opsr_p_max = float(opsr_p_max)
        self.opsr_warmup_steps = int(opsr_warmup_steps)
        self._dbg_opsr_used = 0.0
        try:
            if isinstance(getattr(self, "_last_loss_components", None), dict):
                self._last_loss_components["opsr_p_max"] = float(self.opsr_p_max)
                self._last_loss_components["opsr_used"] = 0.0
        except Exception:
            pass

    def _opsr_p(self) -> float:
        """Annealed scheduled-sampling probability: 0 -> opsr_p_max over opsr_warmup_steps."""
        if self.opsr_p_max <= 0.0:
            return 0.0
        step = float(getattr(self, "_ccrt_step", 0))
        frac = min(1.0, step / max(1.0, float(self.opsr_warmup_steps)))
        return self.opsr_p_max * frac

    # ------------------------------------------------------------------
    # v43 CCRT chunk-1 loss, with the chunk-1 INPUT STATE scheduled-sampled to the
    # chunk-0 PREDICTED action (closed-loop) instead of GT state. Identical to v43
    # except the ">>> v50" state-swap block. Reuses v43's single chunk-0 no-grad
    # predict_visual call -> the predicted action is FREE (no extra forward/memory).
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

        # ---- 1) chunk-0 prediction (NO grad, few-step) -> drifted frame + PRED action ----
        chunk0_obs: Dict[str, torch.Tensor] = {self._cond_key: cond}
        if self.state_key in obs:
            chunk0_obs[self.state_key] = obs[self.state_key]
        saved_steps = self.num_inference_steps
        try:
            self.num_inference_steps = int(self.ccrt_sample_steps)
            with torch.no_grad():
                pred = self.predict_visual(chunk0_obs, show_progress=False)
                pred_future_pix = self.decode_visual_tokens(pred["visual_pred"])
                pred_action0 = pred.get("action_pred")  # [B, A, 28] RAW (unnormalized) or None
        finally:
            self.num_inference_steps = saved_steps
        if pred_future_pix is None or pred_future_pix.shape[1] < 1:
            return None
        drift_last = pred_future_pix[:, -1:].detach().to(cond.dtype)

        # ---- 2) chunk-1 obs = drifted cond + GT future; STATE scheduled-sampled ----
        chunk1_obs: Dict[str, torch.Tensor] = {
            self._cond_key: drift_last,
            self._future_key: gt_future2,
        }
        # >>> v50 OPSR: with annealed prob p, feed the chunk-0 PREDICTED action's last
        #     step as the chunk-1 INPUT STATE (closed-loop = the real-robot eval
        #     distribution), else GT state. opsr_p_max=0 -> always GT -> v43. <<<
        gt_state = obs2.get(self.state_key, None)
        use_pred_state = (
            self.opsr_p_max > 0.0
            and pred_action0 is not None
            and random.random() < self._opsr_p()
        )
        self._dbg_opsr_used = 1.0 if use_pred_state else 0.0
        if use_pred_state:
            last = pred_action0[:, -1:, :].detach()  # [B,1,28] raw absolute joints
            if gt_state is not None:
                if gt_state.dim() == 3 and gt_state.shape[1] > 1:
                    last = last.repeat(1, gt_state.shape[1], 1)  # match n_obs rank
                pred_state = last.reshape(gt_state.shape).to(gt_state.dtype)
            else:
                pred_state = last.to(cond.dtype)
            chunk1_obs[self.state_key] = pred_state
        elif gt_state is not None:
            chunk1_obs[self.state_key] = gt_state

        try:
            if isinstance(self._last_loss_components, dict):
                self._last_loss_components["opsr_used"] = float(self._dbg_opsr_used)
        except Exception:
            pass

        # ---- 3) chunk-1 supervised flow-matching loss (v43 IDENTICAL from here) ----
        nobs = self._normalize_non_image_obs(chunk1_obs)
        batch_inputs = self.obs_encoder.prepare_batch(nobs, training=True)
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


__all__ = ["WanBSLIIv50OPSRPolicy"]
