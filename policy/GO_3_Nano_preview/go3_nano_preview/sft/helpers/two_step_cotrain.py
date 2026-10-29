"""2-step filtered cotrain.

Pipeline (RSFT-flavored, with shared per-member noise for variance reduction):
  1. Group-expand inputs (B -> B*N) with independent VGen noise per member.
  2. Candidate VGen forward at B*N (grad on, start_sigma=1.0). When ref_mode
     is "vgen_gt", a noise_hook captures the per-member noise tensor for
     replay in the reference forward.
  3. Reference latents (B*N, ...), one per member, by mode:
       - "gt":      reuse candidate_outputs["gt_video_latents_for_ae"] as-is.
                    All N members of a group share the same VAE-encoded GT.
       - "vgen_gt": fresh no_grad VGen forward at B*N with start_sigma =
                    1 - ref_gt_weight, REPLAYING the captured per-member noise.
                    Each ref_i sees the same noise as candidate_i, so the
                    reward MSE isolates the GT signal from the noise draw
                    (common-random-numbers variance reduction).
  4. AE judge at fixed sigma (no_grad). Both v_gen and v_ref are at B*N.
     Per-member reward = -MSE(v_gen_i, v_ref_i) over the dense action portion.
  5. std_z filter: drop groups whose reward std < std_threshold; within
     surviving groups, keep members with z-score > z_threshold. Returns a
     (B*N,) sample_weights tensor of 0s and 1s.
  6. AE train step at random sigma (grad on) with sample_weights. Filtered
     samples contribute neither to numerator nor denominator of loss_action,
     so gradients flow only from survivors with correct per-sample scale.

This is NOT a separate phase. It's a toggle inside the existing "cotrain"
phase. When two_step_cotrain.enabled is False, the cotrain branch runs
identically to today.

Reuses helpers from dsrl_step.py to avoid duplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from go3_nano_preview.sft.helpers.ae_judge import make_noise_capture_hook
from go3_nano_preview.sft.helpers.batch import _expand_batch_dict


@dataclass
class TwoStepConfig:
    enabled: bool = False
    group_size: int = 4
    judge_sigma: float = 0.5           # placeholder; sweep
    # ---- VRAM-saving two-phase backward ----
    # 0 disables (single-phase: generate all N with grad, mask the loss).
    # >0 enables two-phase: Phase-1 generates all N under no_grad to score +
    # filter, then Phase-2 *regenerates only the top-`backward_topk` members
    # per group* with grad (replaying their captured noise) and backprops only
    # those. Peak grad activations scale with backward_topk, not group_size.
    # Incompatible with sep_loss_enabled (which needs all-N reward grads).
    backward_topk: int = 0
    # Debug saver: members decoded per VAE forward, chunked to cap the rank-0
    # decode VRAM spike (which otherwise scales with the full group size).
    debug_decode_chunk_members: int = 2
    # ---- Filter selection ----
    # "std_z":         group_std >= std_threshold  AND  z > z_threshold (legacy).
    # "separability":  K=2 K-means split + cliff_ratio gate (see runner/group_filter.py).
    #                  Set sep_threshold <= 0 for log-only mode.
    # "none":          keep all members (mask is all True).
    filter_type: str = "std_z"
    # std_z params
    std_threshold: float = 0.0     # group dropped if std < this
    z_threshold: float = 0.0       # member kept if (r - mean)/std > this
    # separability params
    sep_threshold: float = 3.0     # min cliff_ratio to mark a group separable
    sep_sigma_floor: float = 0.3   # gap-floor as fraction of sigma_group / (N-1)
    # Threshold for the "no-cliff but rewards differ" debug bucket: only
    # save batches where reward range exceeds this (absolute units).
    debug_notsep_wide_spread: float = 0.1
    # ---- Reference latents for the AE judge ----
    ref_mode: str = "gt"           # "gt" | "vgen_gt"
    ref_gt_weight: float = 0.1     # only for "vgen_gt"; start_sigma = 1 - this
    # ---- Reward-separation auxiliary loss (entropy of softmax over z-scored rewards) ----
    # Encourages per-group reward distribution to become sharper (one clear
    # winner + tight loser cluster) so rejection sampling has a stronger signal.
    # IDM is frozen for this branch (sep_freeze_idm always True in current impl);
    # gradient flows back through generated latents to VGen.
    sep_loss_enabled: bool = False
    sep_lambda: float = 0.0
    sep_eps: float = 1e-6
    # ---- GT-anchored confidence weighting (between-group gradient weight) ----
    # weight_mode="mask":       sample_weights are the 0/1 filter gate (legacy).
    # weight_mode="gt_anchored": weight each kept winner by tanh(margin / E_ref),
    #   margin = median(MSE) - MSE_winner (the winner's advantage over a typical
    #   candidate; robust, so a good winner tied with an equally-good runner-up
    #   still scores high), E_ref = the GT velocity energy (per-scene scale anchor
    #   from the AE judge). Renormalized downstream, so it redistributes — never
    #   rescales — the gradient budget. See confidence_weights(). Default "mask"
    #   reproduces current behavior.
    weight_mode: str = "mask"      # "mask" | "gt_anchored"
    weight_temp: float = 3.0       # tanh sharpness, in NMSE (margin / E_ref) units
    eref_floor: float = 0.1        # floor E_ref at this * median(E_ref); near-static-GT guard


@dataclass
class TwoStepContext:
    """All the per-step state the cotrain branch needs after setup.

    When `enabled=False`, vg_* fields shadow the original B-scale inputs and
    capture_hook/noise_holder are None — the cotrain branch behaves identically
    to today.
    """
    enabled: bool
    config: Optional[TwoStepConfig]
    vg_batch: dict
    vg_video: torch.Tensor
    vg_text_conditions: dict
    vg_precomputed_vlm: Optional[list]
    vg_vlm_attention_mask: Optional[torch.Tensor]  # VLM cross-attn pad mask, member-leading (B*N, L)
    vg_batch_size: int
    capture_hook: Optional[Callable[..., Any]]
    noise_holder: Optional[dict]


def setup_two_step_inputs(
    task_flow,
    args,
    batch,
    video,
    captions,
    precomputed_vlm,
    batch_size,
    precomputed_vlm_mask=None,
) -> TwoStepContext:
    """Read args.two_step_cotrain config and prepare expanded inputs.

    Returns a TwoStepContext. When the toggle is off, returns a passthrough
    context (vg_* = original B-scale inputs, hooks None).
    """
    ts_cfg = getattr(args.cotrain, "two_step_cotrain", {}) or {}
    text_conditions = task_flow.prepare_vgen_text_conditioning(captions)
    if not bool(ts_cfg.get("enabled", False)):
        return TwoStepContext(
            enabled=False, config=None,
            vg_batch=batch, vg_video=video,
            vg_text_conditions=text_conditions,
            vg_precomputed_vlm=precomputed_vlm,
            vg_vlm_attention_mask=precomputed_vlm_mask,
            vg_batch_size=batch_size,
            capture_hook=None, noise_holder=None,
        )

    config = TwoStepConfig(
        enabled=True,
        group_size=int(ts_cfg.get("group_size", 4)),
        judge_sigma=float(ts_cfg.get("judge_sigma", 0.5)),
        backward_topk=int(ts_cfg.get("backward_topk", 0)),
        debug_decode_chunk_members=int(ts_cfg.get("debug_decode_chunk_members", 2)),
        filter_type=str(ts_cfg.get("filter_type", "std_z")),
        std_threshold=float(ts_cfg.get("std_threshold", 0.0)),
        z_threshold=float(ts_cfg.get("z_threshold", 0.0)),
        sep_threshold=float(ts_cfg.get("sep_threshold", 3.0)),
        sep_sigma_floor=float(ts_cfg.get("sep_sigma_floor", 0.3)),
        debug_notsep_wide_spread=float(ts_cfg.get("debug_notsep_wide_spread", 0.1)),
        ref_mode=str(ts_cfg.get("ref_mode", "gt")),
        ref_gt_weight=float(ts_cfg.get("ref_gt_weight", 0.1)),
        sep_loss_enabled=bool(ts_cfg.get("sep_loss_enabled", False)),
        sep_lambda=float(ts_cfg.get("sep_lambda", 0.0)),
        sep_eps=float(ts_cfg.get("sep_eps", 1e-6)),
        weight_mode=str(ts_cfg.get("weight_mode", "mask")),
        weight_temp=float(ts_cfg.get("weight_temp", 3.0)),
        eref_floor=float(ts_cfg.get("eref_floor", 0.1)),
    )
    vg_batch, vg_video, vg_text_conditions, vg_vlm, vg_vlm_mask = group_expand_batch(
        batch, video, text_conditions, precomputed_vlm,
        config.group_size, precomputed_vlm_mask,
    )
    # Capture per-member noise when vgen_gt refs need replay, OR when two-phase
    # backward is on (Phase-2 replays the top-k winners' noise to regenerate
    # them deterministically under grad).
    if config.ref_mode == "vgen_gt" or config.backward_topk > 0:
        capture_hook, noise_holder = make_noise_capture_hook()
    else:
        capture_hook, noise_holder = None, None

    return TwoStepContext(
        enabled=True, config=config,
        vg_batch=vg_batch, vg_video=vg_video,
        vg_text_conditions=vg_text_conditions,
        vg_precomputed_vlm=vg_vlm,
        vg_vlm_attention_mask=vg_vlm_mask,
        vg_batch_size=batch_size * config.group_size,
        capture_hook=capture_hook, noise_holder=noise_holder,
    )


def build_two_step_diagnostics(rewards: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    """Pack reward / filter stats for logging. All values are detached scalar tensors
    (or floats) ready to be reduced across ranks by the caller.

    rewards:    (B, N)  per-member rewards from the AE judge.
    valid_mask: (B, N)  bool mask from std_z_mask.
    """
    with torch.no_grad():
        B, N = rewards.shape
        per_group_std = rewards.std(dim=1, unbiased=False)                  # (B,)
        per_group_spread = rewards.max(dim=1).values - rewards.min(dim=1).values  # (B,)
        valid_per_group = valid_mask.sum(dim=1)                             # (B,)
        return {
            "ts_reward_mean": rewards.mean().detach(),
            "ts_reward_std_within_group": per_group_std.mean().detach(),
            "ts_reward_spread_within_group": per_group_spread.mean().detach(),
            "ts_valid_fraction": valid_mask.float().mean().detach(),
            "ts_dropped_groups_frac": (valid_per_group == 0).float().mean().detach(),
            "ts_valid_per_group_mean": valid_per_group.float().mean().detach(),
            "ts_group_size": float(N),
        }


def build_goal_mode_diagnostics(
    batch: dict,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> dict:
    if "cotrain_goal_swap_mask" not in batch:
        return {}
    goal_only = batch["goal_cond"].to(valid_mask.device).bool()
    use_goal = batch["cotrain_goal_swap_mask"].to(valid_mask.device).bool()
    group_valid_fraction = valid_mask.float().mean(dim=1)
    group_weights = sample_weights.detach().reshape(valid_mask.shape[0], -1).mean(dim=1)
    diagnostics = {}
    for mode_name, mode_mask in (
        ("goal_only", goal_only),
        ("language_goal", ~goal_only & use_goal),
        ("language_only", ~goal_only & ~use_goal),
    ):
        diagnostics[f"ts_{mode_name}_fraction"] = mode_mask.float().mean()
        diagnostics[f"ts_{mode_name}_kept_fraction"] = (
            group_valid_fraction * mode_mask
        ).mean()
        diagnostics[f"ts_{mode_name}_weight_mass"] = (
            group_weights * mode_mask
        ).mean()
    return diagnostics


def group_expand_batch(
    batch,
    video,
    text_conditions,
    precomputed_vlm_hidden,
    N,
    precomputed_vlm_mask=None,
):
    """Expand (B, ...) -> (B*N, ...) via repeat_interleave for the inputs
    the VGen TaskFlow call consumes. VLM inputs are not expanded here — the VLM
    forward should happen at B scale and its precomputed hidden states (and
    cross-attn mask) are what get duplicated.

    Returns expanded batch metadata, video, text conditions, VLM states and VLM mask.
    Raw batch video is omitted; training consumes the separate selected video.

    Ordering: [a, b] with N=2 -> [a, a, b, b]. This matches the (B, N)
    reshape used in the reward aggregator (flat index b*N + n -> row b, col n).
    """
    # Preserve the original raw batch for optional debug export.
    batch = {key: value for key, value in batch.items() if key != "video"}
    exp_batch = _expand_batch_dict(batch, N)
    exp_video = video.repeat_interleave(N, dim=0)
    exp_text_conditions = {
        key: value.repeat_interleave(N, dim=0) if value is not None else None
        for key, value in text_conditions.items()
    }
    exp_vlm = (
        [h.repeat_interleave(N, dim=0) for h in precomputed_vlm_hidden]
        if precomputed_vlm_hidden is not None else None
    )
    exp_vlm_mask = (
        precomputed_vlm_mask.repeat_interleave(N, dim=0)
        if precomputed_vlm_mask is not None else None
    )
    return exp_batch, exp_video, exp_text_conditions, exp_vlm, exp_vlm_mask


def restore_raw_video_for_debug(batch, source_batch, group_size, selected_idx=None):
    """Materialize omitted raw pixels only when periodic debug export needs them."""
    raw = source_batch["video"]
    if selected_idx is None:
        indices = torch.arange(raw.shape[0], device=raw.device).repeat_interleave(group_size)
    else:
        indices = selected_idx.to(device=raw.device) // group_size
    return {**batch, "video": raw.index_select(0, indices)}


# ---------------------------------------------------------------------------
# Two-phase backward: select top-k winners and slice all inputs down to them.
# These are the inverse of the group_expand_* helpers above.
# ---------------------------------------------------------------------------


def _slice_member_dim(
    t: torch.Tensor, member_idx: torch.Tensor, n_members: int, n_view: int
) -> torch.Tensor:
    """Index a tensor whose leading dim is either the member count (B*N) or the
    member-view count (B*N*n_view, ``(b v)``-major) down to the selected members.

    Used to slice vg_video / prompts / VLM hidden states (member-leading) and the
    captured generation noise (member-view-leading) to the same set of members.
    """
    L = t.shape[0]
    idx = member_idx.to(t.device)
    if L == n_members:
        return t.index_select(0, idx)
    if L == n_members * n_view:
        view = torch.arange(n_view, device=t.device)
        bv_idx = (idx.unsqueeze(1) * n_view + view).reshape(-1)        # (S*n_view,)
        return t.index_select(0, bv_idx)
    raise ValueError(
        f"_slice_member_dim: leading dim {L} is neither n_members={n_members} "
        f"nor n_members*n_view={n_members * n_view}"
    )


def select_topk_members(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Return flat member indices of the top-k rewarded members per group.

    rewards: (B, N). Returns a LongTensor of shape (B * min(k, N),) with indices
    into the flattened [0, B*N) member layout (row b, col n -> b*N + n), matching
    the repeat_interleave ordering used everywhere else. Deterministic and of
    fixed size B*k, so every DDP rank backprops the same graph shape. Non-finite
    rewards sort last (their loss is still masked by the caller's keep-mask).
    """
    B, N = rewards.shape
    k = max(1, min(int(k), N))
    safe = torch.where(
        torch.isfinite(rewards), rewards, torch.full_like(rewards, -float("inf"))
    )
    top_cols = safe.topk(k, dim=1).indices                              # (B, k) local cols
    offset = (torch.arange(B, device=rewards.device) * N).unsqueeze(1)  # (B, 1)
    return (top_cols + offset).reshape(-1).long()                       # (B*k,)


def confidence_weights(
    rewards: torch.Tensor,
    e_ref: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_idx: torch.Tensor,
    config: TwoStepConfig,
) -> torch.Tensor:
    """Per-winner gradient weights for the AE loss, shape ``(B*k,)`` aligned to
    ``selected_idx`` (the flat member indices ``select_topk_members`` returns).

    ``config.weight_mode``:
      - ``"mask"`` (default): reproduce the legacy 0/1 filter gate exactly.
      - ``"gt_anchored"``: scale each group's kept winner by its *advantage over a
        typical candidate* — how much lower the kept winner's MSE-to-GT is than
        the group's median MSE, in units of the GT velocity energy ``e_ref``.
        ``e_ref`` is a per-scene anchor, so heterogeneous scenes stay comparable;
        the median baseline is robust, so a collapsed member can neither inflate
        the weight (it is not the winner) nor deflate it (it barely moves the
        median). A winner that merely ties an equally-good runner-up still scores
        high here (it still beats the typical candidate) — unlike a top-2 margin.
        The result is renormalized by the element-pooled AE loss denominator
        downstream, so it *redistributes* — never rescales — the gradient budget.

    Between-group factor only; the within-group softmax (for ``backward_topk>1``)
    is a no-op at ``backward_topk=1`` and is omitted until needed.

    rewards:      (B, N) detached per-member rewards (= -MSE to the GT ref).
    e_ref:        (B,)   per-group GT velocity energy from the AE judge.
    valid_mask:   (B, N) bool keep-mask from apply_filter (the hard gate).
    selected_idx: (B*k,) flat member indices of the kept top-k per group.
    """
    keep = valid_mask.view(-1)[selected_idx].float()                 # (B*k,) hard gate
    if config.weight_mode != "gt_anchored":
        return keep
    B, N = rewards.shape
    k = max(1, selected_idx.numel() // B)
    mse = (-rewards).float()                                         # (B, N) MSE-to-GT (fp32 for quantile)
    kept_mse = mse.view(-1)[selected_idx].view(B, k).mean(dim=1)     # (B,) mean MSE of the kept top-k
    # Interpolated median (torch.median picks the lower-middle for even N, which
    # at N=4 collapses back to the runner-up — defeating the point; quantile
    # averages the two middles, the robust group center we want).
    baseline = torch.quantile(mse, 0.5, dim=1)                       # (B,) robust typical candidate
    margin = (baseline - kept_mse).clamp_min(0.0)                    # (B,) advantage over typical (>= 0)
    floor = config.eref_floor * e_ref.float().median()              # near-static-GT guard
    rho = margin / (e_ref.float().clamp_min(floor) + 1e-6)          # (B,) per-scene NMSE units
    conf = torch.tanh(rho / max(config.weight_temp, 1e-6))          # (B,) in [0, 1)
    return keep * conf.repeat_interleave(k)                          # (B*k,)


def gather_batch_dict(
    batch: dict, member_idx: torch.Tensor, n_members: int
) -> dict:
    """Inverse of _expand_batch_dict: select ``member_idx`` from every
    member-indexed field. Tensors whose leading dim == n_members are
    index_selected; lists/tuples of length n_members are picked element-wise;
    everything else passes through unchanged.
    """
    idx_list = member_idx.tolist()
    out = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == n_members:
            out[key] = val.index_select(0, member_idx.to(val.device))
        elif isinstance(val, (list, tuple)) and len(val) == n_members:
            out[key] = type(val)(val[i] for i in idx_list)
        else:
            out[key] = val
    return out


def gather_member_noise(
    noise: torch.Tensor, member_idx: torch.Tensor, n_members: int, n_view: int
) -> torch.Tensor:
    """Slice captured generation noise ((B*N*n_view, ...), ``(b v)``-major) down
    to the selected members, preserving member_idx order so it lines up with a
    batch sliced by the same indices. Feed the result to make_noise_replay_hook.
    """
    return _slice_member_dim(noise, member_idx, n_members, n_view)


def repeat_members(t: torch.Tensor, n_repeat: int, n_view: int) -> torch.Tensor:
    """Expand a ``(b v)``-major per-member tensor (B*n_view, ...) to
    (B*n_repeat*n_view, ...) by replicating each base member ``n_repeat`` times,
    matching ``video.repeat_interleave(n_repeat, dim=0)`` on the unexpanded
    video. Used to broadcast VAE latents encoded once at B across a group's N
    members (the inverse direction of _slice_member_dim).
    """
    lead = t.shape[0]
    b = lead // n_view
    rest = t.shape[1:]
    return (
        t.reshape(b, n_view, *rest)
        .repeat_interleave(n_repeat, dim=0)
        .reshape(b * n_repeat * n_view, *rest)
    )


def slice_two_step_ctx(
    ts_ctx: "TwoStepContext", member_idx: torch.Tensor, n_view: int
) -> "TwoStepContext":
    """Return a TwoStepContext with every member-indexed input sliced down to
    ``member_idx`` (Phase-2 of two-phase backward). Hooks are dropped — Phase-2
    builds a replay hook separately.
    """
    n_members = ts_ctx.vg_batch_size

    def _sl(t):
        return _slice_member_dim(t, member_idx, n_members, n_view) if t is not None else None

    return TwoStepContext(
        enabled=ts_ctx.enabled,
        config=ts_ctx.config,
        vg_batch=gather_batch_dict(ts_ctx.vg_batch, member_idx, n_members),
        vg_video=_sl(ts_ctx.vg_video),
        vg_text_conditions={
            key: _sl(value) for key, value in ts_ctx.vg_text_conditions.items()
        },
        vg_precomputed_vlm=(
            [_slice_member_dim(h, member_idx, n_members, n_view) for h in ts_ctx.vg_precomputed_vlm]
            if ts_ctx.vg_precomputed_vlm is not None else None
        ),
        vg_vlm_attention_mask=_sl(ts_ctx.vg_vlm_attention_mask),
        vg_batch_size=int(member_idx.numel()),
        capture_hook=None,
        noise_holder=None,
    )


def std_z_mask(rewards: torch.Tensor, std_threshold: float, z_threshold: float) -> torch.Tensor:
    """Per-group filter: drop groups with std < std_threshold; within survivors,
    keep members with z-score > z_threshold.

    rewards: (B, N) -> bool mask (B, N).
    Survivor count per group is variable (could be 0 when a group is dropped
    or all members fall below z_threshold). Downstream loss is normalized by
    valid count, so 0-survivor groups vanish gracefully.
    """
    group_mean = rewards.mean(dim=1, keepdim=True)                                  # (B, 1)
    group_std = rewards.std(dim=1, keepdim=True, unbiased=False)                    # (B, 1)
    z = (rewards - group_mean) / (group_std + 1e-8)                                 # (B, N)
    group_alive = group_std >= float(std_threshold)                                 # (B, 1)
    keep = (z > float(z_threshold)) & group_alive                                   # (B, N)
    return keep


def apply_filter(rewards: torch.Tensor, config: TwoStepConfig) -> tuple:
    """Dispatch on ``config.filter_type``; return ``(keep_mask, extra_diagnostics, filter_info)``.

    keep_mask: (B, N) bool — True = keep this member's gradient contribution.
    extra_diagnostics: dict of detached scalars to merge into the standard
        two-step log dict (empty for none).
    filter_info: dict carrying per-group / per-member tensors that the debug
        saver consumes for filenames + JSON. ``filter_type`` key tags it.
    """
    B, N = rewards.shape
    if config.filter_type == "none" or N < 2:
        return (
            torch.ones_like(rewards, dtype=torch.bool),
            {},
            {"filter_type": "none"},
        )

    if config.filter_type == "std_z":
        keep = std_z_mask(rewards, config.std_threshold, config.z_threshold)
        gm = rewards.mean(dim=1, keepdim=True)
        gs = rewards.std(dim=1, keepdim=True, unbiased=False)
        z = (rewards - gm) / (gs + 1e-8)
        info = {
            "filter_type": "std_z",
            "z": z.detach(),                        # (B, N)
            "group_mean": gm.squeeze(-1).detach(),   # (B,)
            "group_std": gs.squeeze(-1).detach(),    # (B,)
        }
        return keep, {}, info

    if config.filter_type == "separability":
        from go3_nano_preview.sft.helpers.group_filter import separability_filter
        result = separability_filter(
            rewards, threshold=config.sep_threshold, sigma_floor_factor=config.sep_sigma_floor,
        )
        # Kept-vs-dropped mean reward across the batch (only meaningful when
        # at least one group was separable; falls back to 0 otherwise).
        keep_f = result.keep_mask.float()
        drop_f = 1.0 - keep_f
        keep_n = keep_f.sum().clamp(min=1.0)
        drop_n = drop_f.sum().clamp(min=1.0)
        kept_mean = (rewards * keep_f).sum() / keep_n
        dropped_mean = (rewards * drop_f).sum() / drop_n
        kept_minus_dropped = (
            (kept_mean - dropped_mean) if drop_f.sum() > 0 else torch.zeros((), device=rewards.device)
        )
        cliff_q = torch.quantile(
            result.cliff_ratio.float(),
            torch.tensor([0.25, 0.5, 0.75], device=rewards.device),
        )
        diag = {
            "ts_sep_cliff_ratio_mean": result.cliff_ratio.mean().detach(),
            "ts_sep_cliff_ratio_p25": cliff_q[0].detach(),
            "ts_sep_cliff_ratio_p50": cliff_q[1].detach(),
            "ts_sep_cliff_ratio_p75": cliff_q[2].detach(),
            "ts_sep_cliff_ratio_max": result.cliff_ratio.max().detach(),
            "ts_sep_distance_mean": result.distance.mean().detach(),
            "ts_sep_separable_frac": result.is_separable.float().mean().detach(),
            "ts_sep_n_high_mean": result.n_high.float().mean().detach(),
            "ts_sep_n_high_min": result.n_high.float().min().detach(),
            "ts_sep_n_high_max": result.n_high.float().max().detach(),
            "ts_sep_kept_minus_dropped": kept_minus_dropped.detach(),
        }
        info = {
            "filter_type": "separability",
            "cliff_ratio": result.cliff_ratio.detach(),
            "is_separable": result.is_separable.detach(),
            "split_idx": result.split_idx.detach(),
            "n_high": result.n_high.detach(),
            "mu_low": result.mu_low.detach(),
            "mu_high": result.mu_high.detach(),
            "sigma_pooled": result.sigma_pooled.detach(),
            "sigma_group": result.sigma_group.detach(),
            "keep_mask": result.keep_mask.detach(),
        }
        return result.keep_mask, diag, info

    raise ValueError(f"unknown filter_type: {config.filter_type!r}")


# `compute_reference_latents` and `ae_judge_fixed_t` now live in runner/ae_judge.py.


def compute_sep_entropy_loss(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Entropy of softmax over z-scored rewards, averaged across the batch.

    Minimizing this sharpens the per-group reward distribution into one clear
    winner + tight loser cluster, giving rejection sampling a stronger
    signal. The z-score (mean=0, std=1 per group) bounds the entropy minimum
    and decouples loss scale from per-prompt reward magnitude.

    Args:
        rewards: (B, K) per-member rewards. Caller must keep the graph
            attached from VGen → x_k → IDM → r_k for the loss to update VGen.
        eps: numerical floor for the group std.

    Returns:
        (loss, diag) — scalar entropy loss and a dict of detached scalars
        for logging.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (B, K); got shape {tuple(rewards.shape)}")
    B, K = rewards.shape
    if K < 2:
        zero = rewards.new_zeros(())
        return zero, {"sep_loss": zero.detach(), "sep_skipped_K_lt_2": 1.0}

    group_mean = rewards.mean(dim=1, keepdim=True)                # (B, 1)
    group_std = rewards.std(dim=1, keepdim=True, unbiased=False)  # (B, 1)
    z = (rewards - group_mean) / (group_std + eps)                # (B, K)

    # log_softmax + (p * log_p) is the numerically-stable entropy form.
    log_p = torch.log_softmax(z, dim=1)                           # (B, K)
    p = log_p.exp()                                               # (B, K)
    entropy_per_group = -(p * log_p).sum(dim=1)                   # (B,)
    loss = entropy_per_group.mean()                               # scalar

    with torch.no_grad():
        diag = {
            "sep_loss": loss.detach(),
            "sep_entropy_mean": entropy_per_group.mean().detach(),
            "sep_entropy_min": entropy_per_group.min().detach(),
            "sep_entropy_max": entropy_per_group.max().detach(),
            "sep_group_mean": group_mean.mean().detach(),
            "sep_group_std": group_std.mean().detach(),
            "sep_z_abs_max": z.abs().max().detach(),
            "sep_z_abs_mean": z.abs().mean().detach(),
        }
    return loss, diag
