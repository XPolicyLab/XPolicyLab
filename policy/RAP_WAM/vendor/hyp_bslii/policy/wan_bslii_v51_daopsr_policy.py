"""BSLII-v51: DA-OPSR -- Difficulty-Adaptive On-Policy State Rollout.

Builds directly on v50 OPSR (the new champion: real closed-loop action_mse 0.0437 vs
v43's 0.0892, -51%). v50's ONLY weakness is a large std (0.0347 vs v43's 0.0081): it
crushes most episodes (ep1/ep2~=0.04, ep7=0.0015) but a few HARD episodes still blow up
(ep6=0.096). Root cause: v50 exposes the predicted-state with the SAME annealed
probability p for every sample, so the HARD samples (where the policy's own prediction is
inaccurate and drift is large -- exactly the ones that blow up at test time) don't get
enough closed-loop recovery training.

v51 fix -- difficulty-adaptive scheduled sampling
-------------------------------------------------
Use the chunk-0 PREDICTION ERROR as a per-sample difficulty signal (free at train time:
pred_action0 is already produced by v43's chunk-0 predict_visual, and the chunk-0 GT
action is batch["action"]). HARD samples (large pred-vs-GT error) get forced to a HIGH
probability of closed-loop predicted-state (p -> 1, heavy exposure); EASY samples keep
the low annealed p. This concentrates capacity on the states the policy is most likely to
blow up on -> tightens the std and rescues the hard episodes.

    d_i      = || pred_action0[i,-1] - GT_action0[i,-1] ||           (per-sample difficulty)
    d_norm_i = (d_i - min_b d) / (max_b d - min_b d)                  (batch min-max -> [0,1])
    p_i      = p_base(t) + opsr_difficulty_gamma * (1 - p_base(t)) * d_norm_i
    use_pred_i ~ Bernoulli(p_i)                                       (PER-SAMPLE)
    state_i  = pred_state_i  if use_pred_i  else  GT_state_i          (per-sample mix)

where p_base(t) is v50's annealed probability (_opsr_p()). gamma=0 -> p_i = p_base for all
-> byte-identical to v50. gamma>0 -> hard samples approach p=1.

Academic basis: difficulty-adaptive / hard-example-aware Scheduled Sampling (Bengio 2015)
+ DAgger (Ross & Bagnell 2011), with a curriculum/hard-mining axis. ORTHOGONAL to v50
(v50 = uniform p; v51 = difficulty-modulated p) and to v43-v49's loss-reweighting.

Implementation / safety
-----------------------
* Inherits v50 (=> v43 CCRT + v50 OPSR machinery). Overrides __init__ (+opsr_difficulty_gamma)
  and _ccrt_chunk1_loss (self-contained rewrite: v43 structure + v50 closed-loop state, but
  the state choice is PER-SAMPLE difficulty-adaptive instead of v50's per-call).
* REUSES v43's single chunk-0 no-grad predict_visual call -> pred_action0 + difficulty are
  FREE (zero extra forward, zero extra memory vs v50).
* opsr_difficulty_gamma=0 -> p_i = p_base (uniform) -> byte-identical to v50. Single
  controlled variable vs v50 = opsr_difficulty_gamma.
* Robust degrade: if batch has no chunk-0 GT action ("action") or no pred_action0, falls
  back to v50's uniform per-call behavior (never crashes).
* No new nn.Parameter -> state_dict == v15 -> eval path unchanged.
* pred-action -> state slot is raw 28-D abs joints, normalized via normalizer["robot_state"]
  (state~=action; same assumption as the closed-loop eval's `closed` mode). torch.where mix
  is shape-matched to obs2[state_key].

Diagnostics: inherits opsr_* + adds opsr_difficulty_gamma / opsr_used (batch mean of the
per-sample use_pred fraction this step).

Author: hyp pipeline (BSLII-v51 plan; difficulty-adaptive on-policy state rollout).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from hyp_bslii.policy.wan_bslii_v50_opsr_policy import WanBSLIIv50OPSRPolicy


class WanBSLIIv51DAOPSRPolicy(WanBSLIIv50OPSRPolicy):
    """v50 OPSR + per-sample difficulty-adaptive closed-loop state exposure.

    Hard samples (large chunk-0 prediction error) are forced to a high probability of
    feeding the predicted state into chunk-1, so the policy learns to recover precisely
    on the states it is most likely to blow up on -- tightening v50's large std.
    """

    def __init__(self, *args, opsr_difficulty_gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.opsr_difficulty_gamma = float(opsr_difficulty_gamma)
        try:
            if isinstance(getattr(self, "_last_loss_components", None), dict):
                self._last_loss_components["opsr_difficulty_gamma"] = float(self.opsr_difficulty_gamma)
        except Exception:
            pass

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

        # ---- 2) chunk-1 obs = drifted cond + GT future; STATE = per-sample difficulty-adaptive ----
        chunk1_obs: Dict[str, torch.Tensor] = {
            self._cond_key: drift_last,
            self._future_key: gt_future2,
        }
        gt_state = obs2.get(self.state_key, None)
        self._dbg_opsr_used = 0.0
        if self.opsr_p_max > 0.0 and pred_action0 is not None and gt_state is not None:
            with torch.no_grad():
                # pred state slot [B,1,28] (last predicted action step), shape-matched to GT
                last = pred_action0[:, -1:, :].detach()  # [B,1,28] raw abs joints
                if gt_state.dim() == 3 and gt_state.shape[1] > 1:
                    last = last.repeat(1, gt_state.shape[1], 1)
                pred_state = last.reshape(gt_state.shape).to(gt_state.dtype)

                B = pred_state.shape[0]
                p_base = float(self._opsr_p())  # v50 annealed base prob
                gt_action0 = batch.get("action", None)  # [B, A, 28] raw chunk-0 GT
                if (
                    self.opsr_difficulty_gamma > 0.0
                    and gt_action0 is not None
                    and gt_action0.shape[0] == B
                ):
                    # per-sample difficulty = chunk-0 prediction error (last step), batch min-max
                    pa = pred_action0[:, -1, :].float()
                    ga = gt_action0[:, -1, :].float().to(pa.device)
                    d = (pa - ga).norm(dim=-1)  # [B]
                    dmin = d.min()
                    d_norm = (d - dmin) / (d.max() - dmin + 1e-6)  # [0,1], hardest=1
                    p_i = p_base + self.opsr_difficulty_gamma * (1.0 - p_base) * d_norm
                    p_i = p_i.clamp(0.0, 1.0)  # [B]
                else:
                    p_i = torch.full((B,), p_base, device=pred_state.device)

                use_pred = torch.rand(B, device=pred_state.device) < p_i  # [B] per-sample
                self._dbg_opsr_used = float(use_pred.float().mean().item())
                m = use_pred.view(B, *([1] * (gt_state.dim() - 1)))  # broadcast mask
                state_mix = torch.where(m, pred_state, gt_state.to(pred_state.dtype))
            chunk1_obs[self.state_key] = state_mix
        elif gt_state is not None:
            chunk1_obs[self.state_key] = gt_state

        try:
            if isinstance(self._last_loss_components, dict):
                self._last_loss_components["opsr_used"] = float(self._dbg_opsr_used)
        except Exception:
            pass

        # ---- 3) chunk-1 supervised flow-matching loss (v43 IDENTICAL) ----
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


__all__ = ["WanBSLIIv51DAOPSRPolicy"]
