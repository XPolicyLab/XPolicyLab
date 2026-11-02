"""v58_light  —  DAOPSR + PUGA (the two closed-loop winners), nothing else.

Composition change (2026-06-27, driven by the same-data closed-loop ablation):
  Re-based from the full v58 PUGA stack (base+CCRT+OPSR+RDASC+DHR+PUGA) onto
  **v51 DAOPSR** (base+CCRT+OPSR+difficulty-weighted scheduled-sampling), then the
  PUGA value-of-information layer is grafted ON TOP.  RDASC (v52) and DHR (v56) are
  DROPPED — the closed-loop eval showed them to be dead weight (RDASC/CCS/DTG/RAC all
  tied OPSR on phonebox; PUGA already dominates DHR on justpickphone), so removing
  them cuts the deepest rollout forwards with no measured closed-loop loss.

  Closed-loop evidence (action_mse, lower=better; same data per cluster):
    foldingbox : CCRT 0.092 -> OPSR 0.055 -> **DAOPSR 0.037** (DAOPSR best, -60% vs CCRT)
    justpickphone : DHR 0.056 -> SAGA 0.052 -> **PUGA 0.048** (PUGA best)
  -> v58_light = DAOPSR (best state-closure) + PUGA (best VoI reweight).
  NOTE: the two winners were never tested TOGETHER / on one dataset — this combo is a
  hypothesis to validate with a same-data ablation on the (pre-encoded) v59 abs_test11.

PUGA's epistemic-variance term U is still replaced by the FREE deployment-aligned U'
( = CCRT/DAOPSR chunk-1 per-patch error, EMA over the (H,W) latent grid ), and the
two "light" systems wins are kept:
  * B1  : the gradient forward is reused as the PUGA anchor (PUGA 3 forwards -> 2).
  * C   : read PRE-ENCODED latents by default (offline pre-encode job warms the cache;
          measured real-train -13.5%); cold-cache guard makes it a no-op if unwarmed.
(torch.compile / FlashAttention-2 were tried and dropped — ~0% on this matmul-bound 5B
 DiT; fused AdamW is already the psi_policy default.)

Why U' still works on the DAOPSR base: v51 DAOPSR provides its OWN _ccrt_chunk1_loss
(v43 structure + v50 closed-loop state), so wrapping super()._ccrt_chunk1_loss still
captures the chunk-1 per-patch deployment error that feeds U'.

Design contract: pure subclass of v51 DAOPSR; the grafted PUGA helpers are copied
VERBATIM from v58 PUGA so PUGA behaviour is identical.  No existing file is modified.
"""
import math
import random
from typing import Any, Dict, Optional

import torch

from hyp_bslii.policy.wan_bslii_v51_daopsr_policy import WanBSLIIv51DAOPSRPolicy


class WanBSLIIv58LightPolicy(WanBSLIIv51DAOPSRPolicy):
    def __init__(
        self,
        *args,
        # ---- PUGA knobs (grafted; copied from v58 PUGA defaults) ----
        puga_weight: float = 0.0,
        puga_act_weight: float = 0.0,
        puga_w_unc: float = 1.0,
        puga_w_lev: float = 1.0,
        puga_w_diff: float = 0.5,
        puga_act_delta: float = 0.1,
        puga_warmup_steps: int = 6000,
        puga_p: float = 0.5,
        puga_eps: float = 1.0e-6,
        puga_n_samples: int = 1,          # kept for config-compat; U' path ignores it (always 1)
        # ---- v58_light U' knobs ----
        uprime_ema_decay: float = 0.9,
        uprime_use_gap: bool = False,
        uprime_grid: int = 14,
        # ---- v58_light C (latent cache) knobs ----
        latent_cache: bool = True,
        latent_cache_dir: str = "/share_data/heyuanpeng/v58_light/lat_cache",
        **kwargs,
    ):
        # DAOPSR base gets ONLY its own kwargs (puga_*/uprime_*/cache_* are NOT passed up —
        # v51 DAOPSR does not accept them and would raise).
        super().__init__(*args, **kwargs)

        # PUGA attrs (verbatim from v58 PUGA __init__) ----------------------------------
        self.puga_weight = float(puga_weight)
        self.puga_act_weight = float(puga_act_weight)
        self.puga_n_samples = 1                  # U' replaces the stochastic-variance forward
        self.puga_w_unc = float(puga_w_unc)
        self.puga_w_lev = float(puga_w_lev)
        self.puga_w_diff = float(puga_w_diff)
        self.puga_act_delta = float(puga_act_delta)
        self.puga_warmup_steps = int(puga_warmup_steps)
        self.puga_p = float(puga_p)
        self.puga_eps = float(puga_eps)
        self._puga_step = 0
        self._puga_cond = None
        self._puga_mask = None

        # U' deployment-difficulty prior (EMA over the (H,W) latent grid) ---------------
        self.uprime_ema_decay = float(uprime_ema_decay)
        self.uprime_use_gap = bool(uprime_use_gap)
        g = int(uprime_grid)
        self.register_buffer("_uprime_ema", torch.zeros(g, g), persistent=True)
        self._uprime_ready = False

        # C: latent cache (default ON; "light" = read pre-encoded latents) -------------
        self._latent_cache_dir = latent_cache_dir
        self._latent_cache_default = bool(latent_cache)
        self._latcache_init = False

    # ============================ grafted PUGA helpers (verbatim from v58 PUGA) =========
    def _puga_on(self) -> bool:
        return (self.puga_weight > 0.0) or (self.puga_act_weight > 0.0)

    def _puga_anneal(self) -> float:
        if self.puga_warmup_steps <= 0:
            return 1.0
        return min(1.0, self._puga_step / float(self.puga_warmup_steps))

    def _puga_norm01(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalisation over all non-batch dims -> [0, 1]."""
        b = x.shape[0]
        flat = x.reshape(b, -1).float()
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        out = (flat - mn) / (mx - mn).clamp_min(self.puga_eps)
        return out.reshape_as(x)

    def _puga_noise(self, clean_latents, clean_actions, tix):
        """Mirror the base flow-matching noising (returns noisy inputs + targets)."""
        b = clean_latents.shape[0]
        v_sig = self.train_sigmas[tix].view(b, 1, 1, 1, 1).to(clean_latents.dtype)
        v_noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - v_sig) + v_noise * v_sig
        tgt_video = v_noise - clean_latents
        a_sig = self.train_sigmas[tix].view(b, 1, 1).to(clean_actions.dtype)
        a_noise = torch.randn_like(clean_actions)
        noisy_actions = clean_actions * (1.0 - a_sig) + a_noise * a_sig
        tgt_action = a_noise - clean_actions
        return noisy_latents, noisy_actions, tgt_video, tgt_action

    def _puga_forward_video(self, noisy_latents, noisy_actions, timesteps, nobs, b):
        out = self._run_joint_model_with_queries(
            noisy_latents=noisy_latents,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            context_list=self._build_context_list(batch_size=b),
            condition_latents=self._puga_cond,
            latent_condition_mask=self._puga_mask,
            state_obs=nobs.get(self.state_key),
        )
        return out  # (pred_video_velocity, pred_action_velocity, query_hidden)

    # ====================================== compute_loss ===============================
    def compute_loss(self, batch: dict, training: bool = True):
        # C: lazy-enable the pre-encoded latent cache on first training step.
        if (not self._latcache_init) and self._latent_cache_default and training:
            self._latcache_init = True
            try:
                self.enable_latent_cache(self._latent_cache_dir)
            except Exception as e:
                print("[v58_light] latent cache not enabled:", repr(e)[:120])

        # DAOPSR pipeline: base chunk-0 + CCRT chunk-1 (difficulty-weighted scheduled
        # sampling); also drives the U' capture via the _ccrt_chunk1_loss override below.
        loss = super().compute_loss(batch, training=training)

        # Graft the PUGA value-of-information loss on top (logic verbatim from v58 PUGA).
        if not (training and self._puga_on()):
            return loss
        self._puga_step += 1
        if self.puga_p < 1.0 and random.random() >= self.puga_p:
            return loss
        try:
            extra = self._puga_loss(batch)
        except Exception:
            extra = None
        if extra is not None:
            loss = loss + extra.to(loss.dtype)
            comps = getattr(self, "_last_loss_components", None)
            if isinstance(comps, dict):
                comps["puga_loss"] = float(extra.detach().item())
        return loss

    # ----------------------------------------- (C) VAE latent cache (default ON, read-only)
    def enable_latent_cache(self, cache_dir=None, write=False):
        """Memoise the obs-encoder VAE video-encode to a content-keyed disk cache so training
        reads PRE-ENCODED latents (the offline pre-encode job warms it) and skips the ~74ms
        encode (measured CFS read ~1ms).  Key = full-content blake2b over the input video
        (safe — no collision-corruption).  Latents stored fp32 (exact, ~0.2MB each).

        write=False (TRAINING default): hits read, misses encode but DO NOT write -> the
        model-drift condition videos (which differ every step and always miss) never
        accumulate cache files.  write=True is ONLY for the offline pre-encode job."""
        import os, hashlib
        cache_dir = cache_dir or self._latent_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        enc = self.obs_encoder
        if getattr(enc, "_latcache_on", False):
            return
        # cold-cache guard: read-only on an un-warmed cache only adds hashing overhead with
        # no hits -> skip (so v58_light is never slower than baseline if no pre-encode was run).
        if (not write) and len([f for f in os.listdir(cache_dir) if f.endswith(".pt")]) == 0:
            print("[v58_light] latent cache cold (no pre-encode) -> skipped, encoding normally")
            return
        orig = enc._encode_video_batch

        def cached(video_bcthw):
            try:
                raw = video_bcthw.detach().to("cpu", torch.float16).contiguous().numpy().tobytes()
                key = hashlib.blake2b(raw, digest_size=16).hexdigest() + "_" + "x".join(map(str, video_bcthw.shape))
                p = os.path.join(cache_dir, key + ".pt")
                if os.path.exists(p):
                    return torch.load(p, map_location=video_bcthw.device).float()
                out = orig(video_bcthw)
                if getattr(enc, "_latcache_write", False):           # only the pre-encode job writes
                    try:
                        torch.save(out.detach().float().cpu(), p + ".tmp"); os.replace(p + ".tmp", p)
                    except Exception:
                        pass
                return out
            except Exception:
                return orig(video_bcthw)  # never let the cache break training

        enc._encode_video_batch = cached
        enc._latcache_on = True
        enc._latcache_write = bool(write)
        self._latent_cache_dir = cache_dir
        print("[v58_light] VAE latent cache ON ->", cache_dir)

    # ------------------------------------------ CCRT/DAOPSR override: reuse parent, stash U'
    def _ccrt_chunk1_loss(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        cap: Dict[str, torch.Tensor] = {}
        orig = self._compute_loss_tensor  # bound method of the psi_policy base

        def wrapped(pred, target, *a, **k):
            # the first 5-D (B,C,T,H,W) call inside the parent body is the chunk-1
            # video-velocity loss -> that is the per-patch deployment error we want
            if "permap" not in cap and torch.is_tensor(pred) and pred.dim() == 5:
                cap["permap"] = (pred.detach().float() - target.detach().float()).pow(2).mean(dim=1)
            return orig(pred, target, *a, **k)

        self._compute_loss_tensor = wrapped  # instance attr shadows the bound method
        try:
            scalar = super()._ccrt_chunk1_loss(batch)  # v51 DAOPSR body runs verbatim
        finally:
            del self._compute_loss_tensor               # drop shadow -> class lookup, parent untouched
        _ = orig
        if scalar is not None and "permap" in cap:
            self._update_uprime(cap["permap"])
        return scalar

    @torch.no_grad()
    def _update_uprime(self, permap: torch.Tensor):
        """permap (B,T,H,W) chunk-1 deployment error -> EMA over the (H,W) grid."""
        spatial = permap.mean(dim=(0, 1)).float()
        if spatial.shape != self._uprime_ema.shape:
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                spatial[None, None], tuple(self._uprime_ema.shape))[0, 0]
        spatial = spatial.to(self._uprime_ema.device)
        d = self.uprime_ema_decay
        if not self._uprime_ready:
            self._uprime_ema.copy_(spatial)
            self._uprime_ready = True
        else:
            self._uprime_ema.mul_(d).add_(spatial, alpha=1.0 - d)

    # ------------------------------------------ PUGA loss: U <- U' (no variance fwd) + B1
    def _puga_loss(self, batch: dict) -> Optional[torch.Tensor]:
        nobs = self._normalize_non_image_obs(batch["obs"])
        bi = self.obs_encoder.prepare_batch(nobs, training=True)
        clean_latents = bi["clean_latents"]
        self._puga_cond = bi["condition_latents"]
        self._puga_mask = bi["latent_condition_mask"]
        ncond = int(bi["num_condition_latent_steps"])

        if "action" in self.normalizer.params_dict:
            clean_actions = self.normalizer["action"].normalize(batch["action"].float())
        else:
            clean_actions = batch["action"].float()

        b = clean_latents.shape[0]
        tix = torch.randint(0, self.train_timesteps.shape[0], (b,), device=self.device)
        timesteps = self.train_timesteps[tix]
        noisy_latents, noisy_actions, tgt_video, tgt_action = self._puga_noise(
            clean_latents, clean_actions, tix
        )

        def fut(x):
            return x[:, :, ncond:, :, :] if self.use_future_only_loss else x

        # ---------- B1: the ONE gradient-carrying forward, REUSED as the anchor ----------
        pv, pa, _q = self._puga_forward_video(noisy_latents, noisy_actions, timesteps, nobs, b)
        pv = fut(pv)

        # ---------- dynamic VoI map under no_grad (V is a detached weight) ----------
        with torch.no_grad():
            anchor = pv.detach()                                              # reuse the grad forward
            delta = self.puga_act_delta * torch.randn_like(noisy_actions)
            pert = fut(self._puga_forward_video(noisy_latents, noisy_actions + delta, timesteps, nobs, b)[0])
            L = (pert.float() - anchor.float()).pow(2).mean(dim=1)             # (B,T,H,W) action leverage
            D = (anchor.float() - fut(tgt_video).float()).pow(2).mean(dim=1)   # (B,T,H,W) difficulty

            # U' = EMA deployment-error prior, broadcast over (B,T) onto the (H,W) grid
            T = L.shape[1]
            up = self._uprime_ema.to(device=L.device, dtype=L.dtype)           # (H,W)
            if up.shape != L.shape[-2:]:                                       # grid guard
                up = torch.nn.functional.interpolate(
                    up[None, None], size=tuple(L.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
            U = up[None, None].expand(b, T, -1, -1).contiguous()              # (B,T,H,W)
            if self.uprime_use_gap:
                U = (U - D).clamp_min(0.0)                                     # exposure-bias gap

            Un, Ln, Dn = self._puga_norm01(U), self._puga_norm01(L), self._puga_norm01(D)
            V = (1.0 + self.puga_w_unc * Un + self.puga_w_lev * Ln + self.puga_w_diff * Dn)
            Ut = self._puga_norm01(U.mean(dim=(2, 3)))                         # (B,T)
            Lt = self._puga_norm01(L.mean(dim=(2, 3)))                         # (B,T)
            ch = 1.0 + self.puga_w_unc * Ut + self.puga_w_lev * Lt            # (B,T) temporal credit

        out = pv.new_zeros((), dtype=torch.float32)   # pv/pa already computed above (B1 reuse)

        if self.puga_weight > 0.0:
            sq = (pv.float() - fut(tgt_video).float()).pow(2)                  # (B,C,T,H,W)
            w5 = V.unsqueeze(1)                                               # (B,1,T,H,W)
            num = (sq * w5).sum()
            den = w5.expand_as(sq).sum().clamp_min(self.puga_eps)
            out = out + self.puga_weight * (num / den)

        if self.puga_act_weight > 0.0:
            n_act = pa.shape[1]
            Tt = ch.shape[1]
            reps = int(math.ceil(n_act / max(Tt, 1)))
            w_act = ch.repeat_interleave(reps, dim=1)[:, :n_act]              # (B,n_act)
            per_step = (pa.float() - tgt_action.float()).pow(2).mean(dim=2)   # (B,n_act)
            num = (per_step * w_act).sum()
            den = w_act.sum().clamp_min(self.puga_eps)
            out = out + self.puga_act_weight * (num / den)

        return self._puga_anneal() * out
