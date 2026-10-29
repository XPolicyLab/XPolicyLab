"""Group separability filter via 1D K-means (K=2) / Otsu's method.

Given a (B, N) tensor of rewards (B groups of size N), find for each group
whether the rewards split cleanly into a "good" cluster and a "bad" cluster.
If so, keep only the good cluster's members; otherwise keep all (assumes no
group is entirely bad).

Math
----
**Step 1 — locate the split.** For each row, find the K=2 partition that
minimises within-cluster sum of squares (equivalently, maximises Fisher F).
In 1D the optimum lies at one of the N-1 adjacent-pair positions, so we
evaluate them all in O(N) using prefix sums over the sorted row.

**Step 2 — judge the split.** A K=2 partition always exists (K-means is
greedy); the question is whether it's a *real cliff*. We use the
gap-at-split ratio:

    cliff_ratio = split_gap / max(mean(other_gaps), gap_floor)

where ``split_gap = r_sorted[k*] - r_sorted[k*-1]`` is the gap between the
two clusters and ``other_gaps`` are the remaining N-2 adjacent gaps. The
floor ``gap_floor = sigma_floor_factor * sigma_group / (N-1)`` prevents the
denominator from collapsing on tight clusters (e.g. all-equal halves).

For a uniform-null group every gap is identical so ``cliff_ratio = 1.0``,
regardless of N — uniform spreads are correctly treated as "not separable".
A real cliff sends ``cliff_ratio`` to many multiples of 1.

Diagnostic: we also report the normalised centroid distance
``d = (mu_high - mu_low) / sigma_pooled``. It captures something different
(how *distinct* the two halves look in K-means terms) and is informative,
but is not the gating metric since uniform-null already gives d ≈ 3.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GroupFilterResult:
    """Per-group result of the K=2 separability filter.

    Shapes: (B,) for scalars, (B, N) for ``keep_mask``.
    """
    keep_mask: torch.Tensor       # (B, N) bool — True = keep this member
    split_idx: torch.Tensor       # (B,) long, in [1, N-1] — size of low cluster
    n_high: torch.Tensor          # (B,) long — size of high cluster (= N - split_idx)
    cliff_ratio: torch.Tensor     # (B,) float — split_gap / mean(other_gaps); GATING metric
    is_separable: torch.Tensor    # (B,) bool — cliff_ratio >= threshold
    distance: torch.Tensor        # (B,) float — (mu_high - mu_low) / sigma_pooled; diagnostic only
    mu_high: torch.Tensor         # (B,) float
    mu_low: torch.Tensor          # (B,) float
    sigma_pooled: torch.Tensor    # (B,) float — within-cluster pooled std (post-floor)
    sigma_group: torch.Tensor     # (B,) float — full-group std (pre-split)


def _kmeans2_1d_split(rewards: torch.Tensor):
    """Optimal 1D K=2 partition per row via prefix sums over the sorted row.

    Returns the chosen split position ``k*`` (= size of low cluster), the
    cluster means / population variances at that split, the sorted rewards,
    and the ascending sort index.
    """
    B, N = rewards.shape
    if N < 2:
        raise ValueError(f"separability_filter requires N >= 2 (got {N})")

    r_sorted, sort_idx = rewards.sort(dim=-1)               # ascending
    cs = r_sorted.cumsum(dim=-1)                            # (B, N)
    css = (r_sorted * r_sorted).cumsum(dim=-1)              # (B, N)

    # Splits k = 1..N-1 (low cluster has k elements).
    sum_low = cs[:, : N - 1]                                # (B, N-1)
    sum_high = cs[:, -1:] - sum_low
    sumsq_low = css[:, : N - 1]
    sumsq_high = css[:, -1:] - sumsq_low

    k = torch.arange(1, N, device=rewards.device, dtype=rewards.dtype).unsqueeze(0)
    n_low_all = k                                           # (1, N-1)
    n_high_all = N - k

    mu_low_all = sum_low / n_low_all
    mu_high_all = sum_high / n_high_all
    var_low_all = (sumsq_low / n_low_all - mu_low_all.square()).clamp_min(0.0)
    var_high_all = (sumsq_high / n_high_all - mu_high_all.square()).clamp_min(0.0)

    # Minimise within-cluster SS — equivalent to maximising between-cluster.
    within_ss = n_low_all * var_low_all + n_high_all * var_high_all   # (B, N-1)
    k_minus_1 = within_ss.argmin(dim=-1)                              # (B,) in [0, N-2]
    split_idx = k_minus_1 + 1                                         # in [1, N-1]

    g = k_minus_1.unsqueeze(-1)
    mu_low = mu_low_all.gather(-1, g).squeeze(-1)
    mu_high = mu_high_all.gather(-1, g).squeeze(-1)
    var_low = var_low_all.gather(-1, g).squeeze(-1)
    var_high = var_high_all.gather(-1, g).squeeze(-1)
    return split_idx, mu_low, mu_high, var_low, var_high, r_sorted, sort_idx


def separability_filter(
    rewards: torch.Tensor,
    threshold: float = 3.0,
    sigma_floor_factor: float = 0.3,
) -> GroupFilterResult:
    """K=2 separability filter; keeps only the high cluster when a real cliff exists.

    Args:
        rewards: ``(B, N)`` per-group rewards.
        threshold: minimum ``cliff_ratio`` to count as separable. ``cliff_ratio``
            equals 1.0 for uniform-null groups; pick the threshold above 1 to
            require a real cliff. Set ``<= 0`` to never mark separable
            (``keep_mask`` becomes all-True; useful for log-only mode).
        sigma_floor_factor: floor on the gap denominator (and pooled std) as a
            fraction of ``sigma_group``. Prevents the cliff ratio from blowing
            up when the within-cluster gaps are degenerate (tight clusters).

    Returns:
        ``GroupFilterResult`` with per-group diagnostics and the keep mask.
    """
    B, N = rewards.shape

    split_idx, mu_low, mu_high, var_low, var_high, r_sorted, sort_idx = _kmeans2_1d_split(rewards)

    # --- Primary metric: cliff ratio at the chosen split ---
    sorted_gaps = r_sorted[:, 1:] - r_sorted[:, :-1]                 # (B, N-1)
    gather_idx = (split_idx - 1).unsqueeze(-1)
    split_gap = sorted_gaps.gather(-1, gather_idx).squeeze(-1)       # (B,)
    other_gaps_mean = (sorted_gaps.sum(dim=-1) - split_gap) / max(N - 2, 1)

    sigma_group = rewards.std(dim=-1, unbiased=False)
    # Floor as a fraction of typical uniform-null gap scale: sigma_group / (N-1).
    gap_floor = sigma_floor_factor * sigma_group / max(N - 1, 1)
    cliff_denom = torch.maximum(other_gaps_mean, gap_floor)
    cliff_ratio = split_gap / (cliff_denom + 1e-8)

    # --- Secondary diagnostic: centroid distance ---
    n_low = split_idx.to(rewards.dtype)
    n_high = (N - split_idx).to(rewards.dtype)
    pooled_var = (n_low * var_low + n_high * var_high) / N
    sigma_pooled_raw = pooled_var.clamp_min(0.0).sqrt()
    sigma_pooled = torch.maximum(sigma_pooled_raw, sigma_floor_factor * sigma_group)
    distance = (mu_high - mu_low) / (sigma_pooled + 1e-8)

    if threshold <= 0:
        is_separable = torch.zeros(B, dtype=torch.bool, device=rewards.device)
    else:
        is_separable = cliff_ratio >= threshold

    # Map sorted-order top-(N-k*) cluster back to original positions.
    ranks = sort_idx.argsort(dim=-1)                                  # (B, N)
    high_mask = ranks >= split_idx.unsqueeze(-1)                      # (B, N)
    keep_mask = torch.where(
        is_separable.unsqueeze(-1),
        high_mask,
        torch.ones_like(high_mask),
    )

    return GroupFilterResult(
        keep_mask=keep_mask,
        split_idx=split_idx,
        n_high=N - split_idx,
        cliff_ratio=cliff_ratio,
        is_separable=is_separable,
        distance=distance,
        mu_high=mu_high,
        mu_low=mu_low,
        sigma_pooled=sigma_pooled,
        sigma_group=sigma_group,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is True.

    Falls back to a plain mean when ``mask`` is empty (all False) so the loss
    stays finite. Designed for use as the loss reducer when applying a
    separability filter to per-member advantages.
    """
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom
