"""AE-judge primitives shared by DSRL and 2-step cotrain.

Owns the AE-side "judge" that converts a pair of video latents (gen, ref) into
per-member rewards via a paired AE forward at a fixed action-timestep. Both
callers — `runner/dsrl_step.py` (policy gradient) and `runner/two_step_cotrain.py`
(rejection SFT) — share this code path; their only structural difference is
whether gradient flows back through `v_gen` (DSRL: yes; 2-step: no).

What lives here:
  - `_apply_rotary_emb`              shared RoPE helper
  - `_build_ae_vision_cond_from_pair` shared dense/sparse vision-cond builder
  - `_split_ae_latents`              shared (B, ...) -> (dense_5d, sparse_5d)
  - `make_noise_capture_hook`        capture hook for vgen_gt-style refs
  - `make_noise_replay_hook`         replay hook for vgen_gt-style refs
  - `AEJudgeConfig`                  judge-only config (subset of TwoStepConfig)
  - `compute_reference_latents`      gt | vgen_gt reference selection
  - `ae_judge_fixed_t`               paired AE forward + per-member dense MSE

Import direction: this module imports only from utils/. Both `dsrl_step.py`
and `two_step_cotrain.py` may import from here without creating cycles.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Union

import torch
from einops import rearrange

from go3_nano_preview.sft.task_model import SFTTaskModel as WAM_policyTaskModel
from policy_core.utils.WAM_policy.data_utils import randn_tensor
from policy_core.utils.common_utils import get_device


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _apply_rotary_emb(x, freqs):
    """Mirrors the apply_rotary_emb defined at top of wm_trainer_mf_v2_model.py."""
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def _build_ae_vision_cond_from_pair(
    trainer,
    ae_dense_5d: torch.Tensor,
    ae_sparse_5d: torch.Tensor,
    n_view: int,
) -> torch.Tensor:
    """Run RoPE positional encoding on (dense, sparse) latent pair.

    Mirrors the positional-encoding section of ae_train_step (lines ~1530-1597)
    but skips augmentation, token dropout, drop_cur_obs — DSRL's judge wants
    identical conditioning structure on the gen and GT forwards.

    Keeps the forward graph through ae_dense_5d so gradients reach VGen.
    """
    n_frame_dense = ae_dense_5d.shape[2]
    sparse_length = ae_sparse_5d.shape[2]
    _, _, _, l_h_dense, l_w_dense = ae_dense_5d.shape
    _, _, _, l_h_sparse, l_w_sparse = ae_sparse_5d.shape

    v_dense = rearrange(ae_dense_5d, "bv c t h w -> bv (t h w) c")
    v_sparse = rearrange(ae_sparse_5d, "bv c t h w -> bv (t h w) c")

    pe_dense = trainer.action_diffusion_model.get_vision_rope(
        v_dense, n_frame_dense, l_h_dense, l_w_dense, is_sparse=False, factor=1,
    )
    v_dense_pos = _apply_rotary_emb(v_dense, pe_dense)
    v_dense_pos = rearrange(v_dense_pos, "(b v) l c -> b v l c", v=n_view)

    dummy_offset = torch.zeros(
        (v_sparse.shape[0], n_frame_dense * l_h_sparse * l_w_sparse, v_sparse.shape[-1]),
        device=v_sparse.device, dtype=v_sparse.dtype,
    )
    v_sparse_full = torch.cat([dummy_offset, v_sparse], dim=1)
    # Same rope tagging as every AE-training site, derived from one place —
    # the judge must stay in lockstep with training (a hardcoded True/2
    # leftover of the abandoned resize experiment diverged here once).
    pe_sparse_full = trainer.action_diffusion_model.get_vision_rope(
        v_sparse_full, n_frame_dense + sparse_length, l_h_sparse, l_w_sparse,
        **WAM_policyTaskModel.ae_sparse_rope_args(trainer.cfg),
    )
    v_sparse_pos_full = _apply_rotary_emb(v_sparse_full, pe_sparse_full)
    v_sparse_pos = v_sparse_pos_full[:, n_frame_dense * l_h_sparse * l_w_sparse:]
    v_sparse_pos = rearrange(v_sparse_pos, "(b v) l c -> b v l c", v=n_view)

    return torch.cat([v_dense_pos, v_sparse_pos], dim=2)  # (b, v, L, c)


def _split_ae_latents(
    ae_latents: Union[torch.Tensor, tuple],
    sparse_length: int,
) -> tuple:
    """Normalize ae_latents to (dense_5d, sparse_5d) regardless of mode."""
    if isinstance(ae_latents, tuple):
        return ae_latents  # already (dense_with_mem_5d, sparse_5d) from mixed mode
    # Single 5D tensor: split by frame count along T axis
    dense_5d = ae_latents[:, :, :-sparse_length]
    sparse_5d = ae_latents[:, :, -sparse_length:]
    return dense_5d, sparse_5d


# ---------------------------------------------------------------------------
# Noise capture / replay hooks (for vgen_gt-style references)
# ---------------------------------------------------------------------------


def make_noise_capture_hook():
    """Return (hook, holder) pair. The hook is a noise_hook compatible with
    TaskFlow VGen that records `noise` into `holder["noise"]` and returns
    it unchanged. Use this on the candidate forward, then read holder["noise"]
    to feed make_noise_replay_hook for the reference forward.
    """
    holder: dict = {}

    def _capture(noise: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        holder["noise"] = noise.detach().clone()
        return noise

    return _capture, holder


def make_noise_replay_hook(captured_noise: torch.Tensor):
    """Return a noise_hook that replaces the freshly-sampled noise with
    `captured_noise`. Shape must match what vgen samples internally for the
    given (batch_size, n_view, data_config) — guaranteed when the candidate
    and reference forwards use the same expanded batch + sigma_override.
    """
    def _replay(noise: torch.Tensor, conditioning_mask: torch.Tensor) -> torch.Tensor:
        if captured_noise.shape != noise.shape:
            raise RuntimeError(
                f"noise replay shape mismatch: captured {tuple(captured_noise.shape)} "
                f"vs vgen-sampled {tuple(noise.shape)}"
            )
        return captured_noise.to(device=noise.device, dtype=noise.dtype)

    return _replay


# ---------------------------------------------------------------------------
# IDM freeze helper for sep-loss judge forward
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _frozen_idm(trainer):
    """Temporarily freeze IDM params + disable IDM gradient checkpointing.

    Use around a ``ae_judge_fixed_t`` forward that needs gradients to flow
    back through IDM to its *input* (e.g., the sep-loss path), without
    (a) accumulating grads on IDM params or (b) silently re-enabling them
    via checkpoint recompute at backward time.

    Checkpoint subtlety: ``torch.utils.checkpoint`` re-runs the forward at
    backward time using the *current* model state, not the state captured
    during the original forward. If we only flipped ``requires_grad`` and
    left checkpointing on, the recompute would observe the restored
    ``requires_grad=True`` and accumulate grads on IDM params after all.
    Disabling IDM checkpointing for this forward avoids the recompute.
    """
    idm = trainer.action_diffusion_model
    params = list(idm.parameters())
    saved_grad = [p.requires_grad for p in params]
    saved_ckpt = bool(getattr(idm, "gradient_checkpointing", False))
    has_setter = hasattr(idm, "_set_gradient_checkpointing")

    for p in params:
        p.requires_grad_(False)
    if saved_ckpt and has_setter:
        idm._set_gradient_checkpointing(enable=False)

    try:
        yield
    finally:
        for p, was in zip(params, saved_grad):
            p.requires_grad_(was)
        if saved_ckpt and has_setter:
            idm._set_gradient_checkpointing(enable=True)


# ---------------------------------------------------------------------------
# Shared judge config + functions
# ---------------------------------------------------------------------------


@dataclass
class AEJudgeConfig:
    """Subset of TwoStepConfig that the AE judge actually consumes.

    Lives here (not in two_step_cotrain) so DSRL can construct its own
    instance without importing the 2-step module. TwoStepConfig is a strict
    superset of this — fields here have the same names/defaults.
    """
    group_size: int = 4
    judge_sigma: float = 0.5
    ref_mode: str = "gt"           # "gt" | "vgen_gt"
    ref_gt_weight: float = 0.1     # only for "vgen_gt"; start_sigma = 1 - this


def compute_reference_latents(
    trainer,
    *,
    candidate_outputs,             # merged TaskFlow VGen outputs from the candidate forward
    candidate_noise: Optional[torch.Tensor],   # captured (bv, tokens, c_lat); required for vgen_gt
    vg_batch,                      # group-expanded batch (B*N)
    vg_video,                      # group-expanded video (B*N)
    n_view: int,
    vg_text_conditions,
    vg_precomputed_vlm,
    vlm_inputs,                    # B-scale; precomputed_vlm path takes precedence
    vg_batch_size: int,            # B * N
    weight_dtype,
    generator,
    data_config: dict,
    config: AEJudgeConfig,
    precomputed_vae_latents=None,
    vg_vlm_attention_mask=None,    # VLM cross-attn pad mask (B*N, L); threaded to the ref forward
):
    """Return reference latents at B*N scale for the AE judge.

    "gt":       reuse VAE-encoded GT from the candidate forward (already at
                B*N with N redundant copies per group).
    "vgen_gt":  fresh no_grad VGen forward at B*N with start_sigma = 1 - ref_gt_weight,
                replaying the captured per-member noise so each ref_i shares
                noise with candidate_i (common-random-numbers).
    """
    if config.ref_mode == "gt":
        return candidate_outputs["gt_video_latents_for_ae"]

    if config.ref_mode == "vgen_gt":
        if candidate_noise is None:
            raise RuntimeError(
                "ref_mode='vgen_gt' requires candidate_noise from a capture hook "
                "on the candidate VGen forward."
            )
        sigma = 1.0 - float(config.ref_gt_weight)
        replay_hook = make_noise_replay_hook(candidate_noise)
        with torch.no_grad():
            ref_model_outputs, ref_loss_dict = trainer.task_flow(
                trainer.build_vgen_task_inputs(
                    batch=vg_batch,
                    video=vg_video,
                    batch_size=vg_batch_size,
                    n_view=n_view,
                    weight_dtype=weight_dtype,
                    generator=generator,
                    text_conditions=vg_text_conditions,
                    vlm_inputs=vlm_inputs,
                    precomputed_vlm_hidden_states=vg_precomputed_vlm,
                    vlm_attention_mask=vg_vlm_attention_mask,
                    gradient_vgen2ae=False,
                    data_config=data_config,
                    one_step_gen_only=True,
                    start_sigma_override=sigma,
                    noise_hook=replay_hook,
                    precomputed_vae_latents=precomputed_vae_latents,
                )
            )
            ref_outputs = {**ref_model_outputs, **ref_loss_dict}
        return ref_outputs["video_latents_for_ae"]

    raise ValueError(f"Unknown ref_mode: {config.ref_mode!r}")


def ae_judge_fixed_t(
    trainer,
    *,
    gen_latents,                   # tuple or tensor, (B*N, ...)
    ref_latents,                   # tuple or tensor, (B*N, ...) — per-member reference
    batch,                         # original B-scale batch (actions, state, action_mask, view_ids)
    n_view: int,
    sparse_length: int,
    data_config: dict,
    config: AEJudgeConfig,
    weight_dtype,
    grad_on_gen: bool = False,
    freeze_idm: bool = False,
    return_ref_energy: bool = False,
    generator=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fixed-t AE judge. Returns (B, N) rewards = -MSE over dense timesteps.

    With ``return_ref_energy=True`` also returns the per-group GT velocity
    energy ``E_ref`` (B,) — the scale anchor used by GT-anchored confidence
    weighting (see ``two_step_cotrain.confidence_weights``). Default False keeps
    the single-tensor return, so existing callers are unaffected.

    Candidates use B*N rows. Repeated GT references use B rows and share
    their predictions across N members; member-specific references retain
    B*N rows. All members use the same per-scene action noise and state.

    `grad_on_gen=False` (default, 2-step rejection): both AE forwards run
    under no_grad; gen_latents is detached. Returned `rewards` has no graph.

    `grad_on_gen=True` (DSRL policy gradient or sep-loss): the v_gen forward
    runs with autograd enabled so the gradient flows back through gen_latents.
    The v_ref forward stays under no_grad — refs are targets, never params.
    Returned `rewards` carries the v_gen graph.

    `freeze_idm=True` (sep-loss path): IDM params are temporarily set to
    ``requires_grad=False`` and IDM gradient checkpointing is disabled for
    the v_gen forward, both restored on exit. Use together with
    ``grad_on_gen=True`` to get rewards that backprop to VGen *without*
    accumulating grads on IDM params (see ``_frozen_idm`` for the
    checkpoint-recompute caveat).
    """
    N = max(1, int(config.group_size))
    device = get_device()

    actions = batch["actions"].to(device, dtype=weight_dtype).contiguous()[:, :-1]
    state = batch["state"].to(device, dtype=weight_dtype).contiguous()
    action_mask = batch["action_mask"].to(device, dtype=weight_dtype).unsqueeze(1)  # (B, 1, A)
    # AE forward asserts view_ids is a (B, n_view) tensor (see action_expert).
    view_ids = batch["view_ids"]
    if not isinstance(view_ids, torch.Tensor):
        raise TypeError(f"batch['view_ids'] must be a torch.Tensor, got {type(view_ids).__name__}")
    body_emb_id = batch.get("body_emb_id")
    if body_emb_id is not None:
        body_emb_id = body_emb_id.to(device=device, dtype=torch.long)
        if body_emb_id.ndim == 2 and body_emb_id.shape[-1] == 1:
            body_emb_id = body_emb_id[:, 0]
        if body_emb_id.ndim != 1 or body_emb_id.shape[0] != actions.shape[0]:
            raise ValueError(
                "batch['body_emb_id'] must have shape [B] (or [B, 1]) for the AE judge, "
                f"got {tuple(body_emb_id.shape)}"
            )
    B = actions.shape[0]

    with torch.no_grad():
        # Fixed sigma for judge — variance reduction across batches
        sigma = float(config.judge_sigma)
        action_sigmas = torch.full((B,), sigma, device=device, dtype=weight_dtype)
        action_timesteps = (action_sigmas * 1000.0).long().unsqueeze(-1).repeat(1, actions.shape[1])
        # Judge action-noise. generator=None -> global RNG (legacy behavior, kept
        # for callers that don't thread a generator). The cotrain call sites pass
        # the seeded rank-offset generator so candidate SELECTION is reproducible:
        # the reward = -MSE(v_gen, v_ref) at judge_sigma is sensitive to this noise,
        # and unseeded draws made cotrain non-reproducible run-to-run (and its
        # divergence over-sensitive to any global-RNG draw-count shift).
        noise_actions = randn_tensor(
            actions.shape, device=device, dtype=weight_dtype, generator=generator
        )
        a_ss = action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])
        noisy_actions = ((1.0 - a_ss) * actions + a_ss * noise_actions) * action_mask  # (B, T, A)

    # High/low state selection: action_state_adaptor expects state_dim=30
    # but batch["state"] is 32-dim. Mirror ae_train_step (wm_trainer ~3132): drop
    # dims (7,15) for high-level members and (6,14) for low-level, picked by the
    # action_mask length. Must match the trainer or the judge conditions on a
    # misaligned state.
    idx_high = torch.tensor([k for k in range(32) if k not in [7, 15]], device=state.device)
    idx_low = torch.tensor([k for k in range(32) if k not in [6, 13]], device=state.device)
    _mask_sums = action_mask.sum(dim=-1)
    is_high = (_mask_sums >= 16) | (_mask_sums == 8)  # keep in lockstep with ae_train_step / validate
    new_state = torch.where(is_high.view(-1, 1, 1), state[..., idx_high], state[..., idx_low])

    # Candidates and member-specific references use B*N rows.
    if N > 1:
        noisy_actions_exp = noisy_actions.repeat_interleave(N, dim=0)
        action_timesteps_exp = action_timesteps.repeat_interleave(N, dim=0)
        state_exp = new_state.repeat_interleave(N, dim=0)
        view_ids_exp = view_ids.repeat_interleave(N, dim=0)
        body_emb_id_exp = (
            body_emb_id.repeat_interleave(N, dim=0) if body_emb_id is not None else None
        )
        action_mask_exp = action_mask.repeat_interleave(N, dim=0)                   # (B*N, 1, A)
    else:
        noisy_actions_exp = noisy_actions
        action_timesteps_exp = action_timesteps
        state_exp = new_state
        view_ids_exp = view_ids
        body_emb_id_exp = body_emb_id
        action_mask_exp = action_mask

    # Split + AE-scale rescale at the judge boundary. Detach gen iff caller does
    # not need gradient through v_gen (2-step). DSRL keeps the graph.
    gen_dense, gen_sparse = _split_ae_latents(gen_latents, sparse_length)
    ref_dense, ref_sparse = _split_ae_latents(ref_latents, sparse_length)
    shared_gt_reference = config.ref_mode == "gt" and N > 1
    if shared_gt_reference:
        # Reference order is (sample, member, view), matching group expansion.
        ref_dense = ref_dense.unflatten(0, (B, N, n_view))[:, 0].flatten(0, 1)
        ref_sparse = ref_sparse.unflatten(0, (B, N, n_view))[:, 0].flatten(0, 1)
    if not grad_on_gen:
        gen_dense = gen_dense.detach()
        gen_sparse = gen_sparse.detach()
    rescale_kwargs = {
        "cfg": trainer.cfg,
        "stats_tracker": getattr(trainer, "stats_tracker", None),
        "vae": trainer.vae,
        "ae_video_scaling_factor": trainer.ae_video_scaling_factor,
    }
    gen_dense = WAM_policyTaskModel.rescale_for_ae(gen_dense, **rescale_kwargs)
    gen_sparse = WAM_policyTaskModel.rescale_for_ae(gen_sparse, **rescale_kwargs)
    ref_dense = WAM_policyTaskModel.rescale_for_ae(ref_dense, **rescale_kwargs)
    ref_sparse = WAM_policyTaskModel.rescale_for_ae(ref_sparse, **rescale_kwargs)

    vision_gen = _build_ae_vision_cond_from_pair(trainer, gen_dense, gen_sparse, n_view)  # (B*N, v, L, c)
    vision_ref = _build_ae_vision_cond_from_pair(trainer, ref_dense, ref_sparse, n_view)

    # v_ref forward is always no_grad — the reference is a target, never a param.
    with torch.no_grad():
        if shared_gt_reference:
            v_ref = trainer.action_diffusion_model(
                vision_velocity=vision_ref,
                action_states=noisy_actions,
                action_timestep=action_timesteps,
                state_cond=new_state,
                view_ids=view_ids,
                body_emb_id=body_emb_id,
            )["action"].float()
            v_ref = v_ref.repeat_interleave(N, dim=0)
        else:
            v_ref = trainer.action_diffusion_model(
                vision_velocity=vision_ref,
                action_states=noisy_actions_exp,
                action_timestep=action_timesteps_exp,
                state_cond=state_exp,
                view_ids=view_ids_exp,
                body_emb_id=body_emb_id_exp,
            )["action"].float()

    # v_gen forward: under no_grad for 2-step (default), with grad for DSRL
    # or sep-loss. When freeze_idm=True, IDM params are temporarily frozen
    # AND IDM checkpointing is disabled (checkpoint recompute would otherwise
    # restore requires_grad=True at backward time and break the freeze).
    gen_grad_ctx = (
        torch.enable_grad() if grad_on_gen else torch.no_grad()
    )
    freeze_ctx = _frozen_idm(trainer) if freeze_idm else contextlib.nullcontext()
    with freeze_ctx, gen_grad_ctx:
        v_gen = trainer.action_diffusion_model(
            vision_velocity=vision_gen,
            action_states=noisy_actions_exp,
            action_timestep=action_timesteps_exp,
            state_cond=state_exp,
            view_ids=view_ids_exp,
            body_emb_id=body_emb_id_exp,
        )["action"].float()  # (B*N, T, A)

    # Per-member dense MSE -> (B, N) reward
    T_full = v_gen.shape[1]
    T_dense = min(int(data_config.get("continious_action_length", T_full)), T_full)
    diff_sq = (v_gen - v_ref).pow(2)[:, :T_dense] * action_mask_exp                  # (B*N, T_dense, A)
    valid_A_per_member = action_mask_exp.squeeze(1).sum(dim=-1)                      # (B*N,)
    denom = valid_A_per_member * T_dense + 1e-6                                      # (B*N,)
    mse_per_member_flat = diff_sq.sum(dim=(-2, -1)) / denom                          # (B*N,)
    rewards = -mse_per_member_flat.view(B, N)
    if return_ref_energy:
        # GT velocity energy = MSE(0, v_ref): per-scene scale anchor for
        # confidence weighting. Same mask/denominator as the reward; refs are
        # shared across a group's members, so average over N to get (B,).
        ref_sq = v_ref.pow(2)[:, :T_dense] * action_mask_exp                         # (B*N, T_dense, A)
        e_ref = (ref_sq.sum(dim=(-2, -1)) / denom).view(B, N).mean(dim=1)            # (B,)
        return rewards, e_ref
    return rewards
