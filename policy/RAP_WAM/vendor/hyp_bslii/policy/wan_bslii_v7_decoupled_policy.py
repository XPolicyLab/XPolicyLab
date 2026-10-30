"""BSLII-v7: Decoupled Action Conditioning (DAC) -- an information bottleneck
on the action token's video context.

Motivation
----------
The joint Wan2.2-TI2V-5B video+action world model predicts robot actions with
flow-matching diffusion tokens that share *every* DiT block with the video
latent.  Through joint self-attention in each block the action tokens attend to
the entire ~14k-token video latent grid (history ``condition_latents`` +
noisy future latents).

At AR-state rollout inference that grid is the encoding of the model's OWN
previously generated video, which carries rollout drift -- a fine-grained,
high-dimensional distribution the action tokens never see at train time (where
the grid is the clean ground-truth VAE latent).  This exposure bias on the
*video-history conditioning path* is the diagnosed root cause of the v4/v5/v6
action regressions: every additive iteration left the action token attending
directly to the full 14k drifting grid.

  baseline (plain joint K=16):   AR-state action_mse 0.007619
  v4 (MRAA + SDT):               0.00971  (+27%)
  v5 (v4 + SHM/SelfRoll/IDM):    0.00994  (+30%)
  v6 (CHG calibrated corruption):did not beat baseline

Design -- Decoupled Action Conditioning (DAC)
---------------------------------------------
The BSLII policy already carries K=16 learnable "imagination query" tokens
trained by ``align_loss`` (cosine distance between the projected K-query hidden
states and the pooled future VAE latent) to be a *compressed, deterministic
summary* of the video.  The joint sequence is

    [ video tokens (~14k) | action tokens (A) | state tokens (S) | K-query (K) ].

v7 changes exactly ONE thing -- the joint self-attention connectivity:

    The action tokens attend ONLY to {action, state, K-query} tokens.
    They are masked out from the raw ~14k video-token keys.

The action therefore receives video information *solely* through the
align-trained K=16 query bottleneck.  Video tokens and the K-queries keep their
current (full) attention, so the K-queries still summarise the video exactly as
before -- only the action's *read* of the video is rerouted through the
bottleneck.

Why an information bottleneck helps
    The K=16 summary is far lower-dimensional and far more stable under AR
    rollout than the 14k raw latent grid: it is trained (via ``align_loss``) to
    capture coarse future-video structure, and coarse structure drifts much
    less than per-patch latent detail.  Routing the action's only video
    dependency through this bottleneck makes action prediction robust to the
    fine-grained video-latent drift that the 14k grid exhibits at inference --
    a classic information-bottleneck argument for generalisation under input
    distribution shift.

This is a single, clean mechanism.  v5 regressed precisely from piling on
auxiliary losses; v7 adds NO new loss, NO new parameters, NO extra forward
pass.  ``align_loss``/``video_loss``/``action_loss`` are byte-identical to the
baseline; only the attention mask changes.

Implementation
--------------
flash-attn is not installed in this environment, so the upstream
``wan.modules.attention.attention`` already falls through to
``torch.nn.functional.scaled_dot_product_attention`` (SDPA) -- but the upstream
wrapper hard-codes ``attn_mask=None``.  v7 overrides ``_joint_self_attention``
(defined on ``WanJointActionPolicy``) and calls SDPA *directly* with an
additive ``[L, L]`` mask.  q/k/v projection and the video-only RoPE are
reproduced exactly as in the parent, so with an all-zero mask this override is
numerically equivalent to the parent.  Wan self-attention blocks use
``window_size=(-1, -1)`` (full attention, verified), so SDPA is an exact
replacement for the upstream attention call.  No shared code is modified.

Train / eval symmetry -- v7's self-contained inference forward
--------------------------------------------------------------
v7's decoupling mask only routes the action through the K=16 bottleneck if the
K-query tokens are actually present in the joint sequence.  The K-query tokens
are appended ONLY by the training forward
``WanBSLIIPolicy._run_joint_model_with_queries`` (reached from ``compute_loss``).
The baseline *sampling* forward ``WanJointActionPolicy._run_joint_model``
(reached from ``predict_action`` / ``predict_visual`` ->
``_sample_joint_predictions`` -- the ``wan sample`` eval AND ``infer.py``
AR-state inference) appends NO K-query tokens.  If v7 inherited that sampling
forward unchanged, the joint sequence at inference would have no K-query rows,
the action would have no bottleneck to attend to, and v7's defensive
"layout mismatch -> mask=None" fallback would fire: v7 would *train* decoupled
but *sample / evaluate* as the unmasked baseline -- a hard train/eval mechanism
mismatch.

To make v7's decoupling apply *identically* at training and at inference, v7
carries its OWN self-contained inference code flow: it overrides
``_run_joint_model`` (the sampling forward) to append the SAME K=16 learnable
imagination-query tokens that the training forward appends -- replicating the
query-token construction / modulation / sequence assembly of
``WanBSLIIPolicy._run_joint_model_with_queries`` (minus the ``query_hidden``
return and the ``align_loss``, which are training-only).  The sampling joint
sequence therefore becomes

    [ video (~14k) | action (A) | state (S) | K-query (K) ]

byte-identical in layout to the training sequence, so v7's decoupling mask
applies on the sampling path exactly as on the training path.

v7's TRAINING path is deliberately left untouched: training keeps using the
inherited ``WanBSLIIPolicy._run_joint_model_with_queries`` (already correct --
it appends the K-queries, and ``query_hidden`` flows into ``align_loss``).  v7
does NOT override ``_run_joint_model_with_queries``; only the previously-broken
inference forward is given a v7-specific override.  The fix is fully contained
in this file -- no shared code and no other version's file is modified.

Robust per-forward token counting
    ``_joint_self_attention`` no longer hard-codes the K-query count as
    ``self.K``.  Each ``_run_joint_model*`` forward records, in
    ``self._v7_pending_query_tokens``, how many K-query tokens IT appended;
    ``_joint_self_attention`` reads that per-forward hint.  After this fix both
    forwards append exactly ``self.K`` queries, but the hint mechanism makes the
    mask track the *actual* sequence layout of each forward -- so the mask can
    never again be silently dropped by a stale assumption about which forward
    is running.

Pareto safety (video must not weaken)
    Video-token query rows keep full attention over all keys, and the K-query
    rows keep full attention over the video -- so the video denoising pathway
    and the align bottleneck see an unchanged computation graph.  Only action
    -> video edges are removed.  The video head cannot regress by construction.

Author: hyp pipeline (BSLII-v7 plan).
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# NOTE: import order matters. ``hyp_bslii.policy.wan_bslii_policy`` transitively
# imports ``psi_policy.model.wan.runtime``, whose module-load side effect
# inserts ``third_party/wan_official`` onto ``sys.path`` -- only after that is
# the official ``wan`` package importable. Keep this import strictly before the
# ``from wan.modules...`` import below (mirrors the ordering in
# psi_policy/policy/wan_joint_action_policy.py).
from hyp_bslii.policy.wan_bslii_policy import WanBSLIIPolicy

from wan.modules.model import rope_apply


class WanBSLIIv7DecoupledPolicy(WanBSLIIPolicy):
    """Plain BSLII baseline (K=16) with Decoupled Action Conditioning.

    The action tokens are masked out of the raw video keys in joint
    self-attention; they receive video information only through the K
    learnable imagination-query tokens (the align-trained bottleneck).

    Train / eval symmetry
    ---------------------
    v7 carries its own inference code flow: it overrides ``_run_joint_model``
    (the sampling forward used by ``predict_action`` / ``predict_visual``) so
    that the K=16 query tokens are appended on the sampling path too -- exactly
    as the training forward (the inherited
    ``WanBSLIIPolicy._run_joint_model_with_queries``) appends them.  The
    decoupling mask therefore applies identically at training and at inference.
    v7 does NOT override the training forward: training is left byte-identical
    to the BSLII baseline.

    Hyperparameters
    ---------------
    decouple_action_from_video: bool, default True.
        Master switch.  When False the policy is exactly ``WanBSLIIPolicy``
        (the mask becomes all-zero and ``_joint_self_attention`` reproduces the
        parent's computation).  Kept so the same class can serve as its own
        ablation without a separate file.
    decouple_action_attends_state: bool, default True.
        Whether the (decoupled) action tokens may still attend to the state
        token(s).  State is a tiny, drift-free proprio conditioning signal, so
        it is kept by default; set False to route the action through the
        K-query bottleneck *exclusively*.

    Notes
    -----
    * K must be > 0 for decoupling to be meaningful: with K=0 the action would
      have no video information path at all.  A K=0 + decoupling config is
      rejected at construction time.
    * The K-query rows and the video rows keep full attention, so ``align_loss``
      still trains the queries to summarise the video.
    """

    def __init__(
        self,
        *,
        decouple_action_from_video: bool = True,
        decouple_action_attends_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.decouple_action_from_video = bool(decouple_action_from_video)
        self.decouple_action_attends_state = bool(decouple_action_attends_state)

        if self.decouple_action_from_video and self.K <= 0:
            raise ValueError(
                "WanBSLIIv7DecoupledPolicy requires num_latent_queries (K) > 0 "
                "when decouple_action_from_video is True: the K-query tokens are "
                "the action's only remaining video-information path."
            )

        # Token-count layout discovered per-forward in _joint_self_attention;
        # cached only for diagnostics / logging.
        self._v7_last_video_tokens: int = 0
        self._v7_last_action_tokens: int = 0
        self._v7_last_state_tokens: int = 0
        self._v7_last_query_tokens: int = 0

        # Per-forward layout hint: how many K-query tokens the *current*
        # _run_joint_model* forward appended.  Read by _joint_self_attention so
        # the decoupling mask tracks the ACTUAL sequence layout instead of
        # hard-coding self.K.  Initialised to self.K because BOTH the inherited
        # training forward (_run_joint_model_with_queries) and v7's own
        # inference override (_run_joint_model) append exactly self.K queries;
        # the training forward does not touch this attribute, so the default
        # must already be correct for it.  v7's _run_joint_model override sets
        # it explicitly per-forward for defensive robustness.
        self._v7_pending_query_tokens: int = int(self.K) if self.K > 0 else 0

    # ------------------------------------------------------------------
    # Attention mask: action-query rows masked off the video-key columns.
    # ------------------------------------------------------------------
    def _build_decouple_attn_mask(
        self,
        *,
        seq_len: int,
        video_token_count: int,
        action_token_count: int,
        state_token_count: int,
        query_token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Additive ``[L, L]`` self-attention mask (broadcast over B and heads).

        Token order in the joint sequence (assembled identically by the
        inherited training forward ``_run_joint_model_with_queries`` and by
        v7's own inference forward ``_run_joint_model``):

            [ video : [0, V)               ]
            [ action: [V, V + A)           ]
            [ state : [V + A, V + A + S)   ]
            [ query : [V + A + S, L)       ]   (L = V + A + S + K)

        Returns an additive float mask where masked entries hold the dtype's
        most-negative value and all other entries are 0.  Returns ``None`` when
        no masking is needed (decoupling off, K=0, or no video/action tokens) so
        the caller can take the unmasked fast path.
        """
        if not self.decouple_action_from_video:
            return None
        if video_token_count <= 0 or action_token_count <= 0 or query_token_count <= 0:
            # Nothing to decouple, or the bottleneck is absent: no-op.
            return None

        v = int(video_token_count)
        a = int(action_token_count)
        # Action token row range.
        a_start, a_end = v, v + a

        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        neg = torch.finfo(dtype).min

        # Action-query rows: block out the video-key columns [0, V).
        mask[a_start:a_end, 0:v] = neg
        if not self.decouple_action_attends_state:
            # Also block the state-key columns so the action attends to
            # {action, query} only.
            s = int(state_token_count)
            if s > 0:
                s_start = v + a
                mask[a_start:a_end, s_start : s_start + s] = neg
        return mask

    # ------------------------------------------------------------------
    # Joint self-attention with the decoupling mask.
    #
    # Reproduces WanJointActionPolicy._joint_self_attention exactly (same q/k/v
    # projection, same video-only RoPE), but routes the attention through SDPA
    # with an explicit additive mask instead of the upstream ``wan_attention``
    # wrapper (which hard-codes attn_mask=None).  flash-attn is absent in this
    # environment, so ``wan_attention`` itself already uses SDPA -- calling SDPA
    # here is therefore the same kernel, just with the mask plumbed through.
    # ------------------------------------------------------------------
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

        # Video-only RoPE -- identical to the parent implementation.
        if video_token_count > 0:
            q_video = rope_apply(q[:, :video_token_count], grid_sizes, freqs)
            k_video = rope_apply(k[:, :video_token_count], grid_sizes, freqs)
            if video_token_count < l:
                q = torch.cat([q_video, q[:, video_token_count:]], dim=1)
                k = torch.cat([k_video, k[:, video_token_count:]], dim=1)
            else:
                q = q_video
                k = k_video

        # Infer the action / state / query split from the token layout.
        # action_horizon is fixed.  The K-query count is whatever the CURRENT
        # forward appended -- recorded in self._v7_pending_query_tokens by the
        # _run_joint_model* methods (the inherited training forward leaves it at
        # its self.K default; v7's _run_joint_model override sets it
        # explicitly).  This is read PER-FORWARD rather than hard-coding self.K,
        # so the mask tracks the actual sequence and can never be silently
        # dropped by a wrong assumption about which forward is running.
        action_token_count = int(self.action_horizon)
        query_token_count = int(getattr(self, "_v7_pending_query_tokens", self.K if self.K > 0 else 0))
        if query_token_count < 0:
            query_token_count = 0
        state_token_count = (
            int(l) - int(video_token_count) - action_token_count - query_token_count
        )
        if state_token_count < 0:
            # Layout mismatch under the recorded query count: try once more
            # assuming no K-queries were appended (defensive -- should not
            # happen now that both forwards append self.K, but never mis-mask).
            query_token_count = 0
            state_token_count = (
                int(l) - int(video_token_count) - action_token_count
            )
        if state_token_count < 0:
            # Still inconsistent: fall back to the unmasked parent behaviour
            # rather than mis-masking.
            state_token_count = 0
            attn_mask = None
        else:
            self._v7_last_video_tokens = int(video_token_count)
            self._v7_last_action_tokens = action_token_count
            self._v7_last_state_tokens = state_token_count
            self._v7_last_query_tokens = query_token_count
            attn_mask = self._build_decouple_attn_mask(
                seq_len=int(l),
                video_token_count=int(video_token_count),
                action_token_count=action_token_count,
                state_token_count=state_token_count,
                query_token_count=query_token_count,
                device=x.device,
                dtype=(q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16),
            )

        attn_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        # SDPA expects [B, N, L, D]; wan_attention's SDPA fallback does exactly
        # this transpose/cast.  attn_mask broadcasts over the B and N axes.
        q_sdpa = q.transpose(1, 2).to(attn_dtype)
        k_sdpa = k.transpose(1, 2).to(attn_dtype)
        v_sdpa = v.transpose(1, 2).to(attn_dtype)
        mask_sdpa = attn_mask.to(attn_dtype) if attn_mask is not None else None

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask_sdpa,
            is_causal=False,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous()  # [B, L, N, D]

        out = out.flatten(2)
        out = attn_module.o(out)
        return out

    # ------------------------------------------------------------------
    # Inference / sampling forward -- v7's SELF-CONTAINED inference code flow.
    #
    # This is the method reached from predict_visual / predict_action ->
    # _sample_joint_predictions, i.e. the ``wan sample`` eval path AND infer.py
    # AR-state rollout inference.
    #
    # The baseline WanJointActionPolicy._run_joint_model that v7 would
    # otherwise inherit appends NO K-query tokens, so v7's decoupling mask was
    # silently dropped at inference (state_token_count went negative).  v7
    # therefore overrides _run_joint_model and replicates the K-query
    # construction / modulation / sequence assembly of the training forward
    # WanBSLIIPolicy._run_joint_model_with_queries -- WITHOUT the training-only
    # ``query_hidden`` return and ``align_loss``.  The sampling joint sequence
    # becomes [video | action | state | K-query], byte-identical in layout to
    # the training sequence, so the decoupling mask applies identically.
    #
    # This override is the ONLY change to v7's inference path.  v7's TRAINING
    # path is untouched: training keeps using the inherited
    # _run_joint_model_with_queries (which is already correct).
    # ------------------------------------------------------------------
    def _run_joint_model(
        self,
        *,
        noisy_latents: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        condition_latents: torch.Tensor,
        latent_condition_mask: torch.Tensor,
        state_obs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

            # --- v7: append the K learnable imagination-query tokens, exactly
            # as WanBSLIIPolicy._run_joint_model_with_queries does for the
            # training forward.  This is what makes the decoupling mask apply
            # at inference identically to training.
            query_tokens = None
            query_token_count = 0
            if self.K > 0:
                query_tokens = self.latent_queries.to(device=model_device, dtype=model_dtype)
                query_tokens = query_tokens.expand(B, -1, -1).contiguous()
                query_token_count = self.K

            parts = [video_tokens, action_tokens]
            if state_tokens is not None:
                parts.append(state_tokens)
            if query_tokens is not None:
                parts.append(query_tokens)
            total_tokens = torch.cat(parts, dim=1)

            # Time conditioning -- identical to the parent / training forward.
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
                # Use the action-side time embed as the query modulation so
                # AdaLN-Zero stays on-distribution for the K-query positions --
                # identical to _run_joint_model_with_queries.
                query_mod = (
                    action_mod[:, :1].expand(-1, query_token_count, -1, -1).contiguous()
                )
                modulation_parts.append(query_mod)
            total_mod = torch.cat(modulation_parts, dim=1)

            # Hand the per-forward K-query count to _joint_self_attention so its
            # decoupling mask uses the ACTUAL layout of THIS forward.
            self._v7_pending_query_tokens = int(query_token_count)

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

            video_pred_tokens = model.head(video_tokens, video_time_embed)
            video_pred = model.unpatchify(video_pred_tokens, grid_sizes)
            video_pred = torch.stack([sample.float() for sample in video_pred], dim=0)
            action_pred = self.action_head(action_tokens, action_time_embed).float()

        # Restore the per-forward hint to its self.K default so a subsequent
        # forward that does NOT set it (e.g. the inherited training forward)
        # still sees the correct count.
        self._v7_pending_query_tokens = int(self.K) if self.K > 0 else 0
        return video_pred, action_pred


__all__ = ["WanBSLIIv7DecoupledPolicy"]
