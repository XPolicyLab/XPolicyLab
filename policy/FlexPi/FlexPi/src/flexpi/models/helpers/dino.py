"""Stateless DINO helpers: RoPE position encoding and x0-parameterization.

Spatial grid IDs and RoPE frequencies for DINO patch tokens in a composed
multi-camera grid, plus the x̂0 → v̂ conversion and head zero-init used when
``dino_pred_x0=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_dino_mesh_id(
    n_frames: int,
    cam_regions: list[tuple[int, int, int, int]],
    cam_patches: list[tuple[int, int]],
    frame_indices: list[int] | None = None,
) -> torch.Tensor:
    """Build (f, h, w) grid for DINO patches in a composed spatial grid.

    Each camera occupies a non-overlapping region in the composed grid so that
    every DINO patch gets a unique (h, w) position — no RoPE collisions.

    RoboTwin 3-cam composed grid (21 rows × 14 cols):
      cam_high:        rows 7-20, cols 0-13   (14×14 = 196 patches)
      cam_left_wrist:  rows 0-6,  cols 0-6    (7×7 = 49 patches)
      cam_right_wrist: rows 0-6,  cols 7-13   (7×7 = 49 patches)

    Args:
        n_frames:      number of temporal frames (F). Ignored if `frame_indices` is given.
        cam_regions:   list of (h_start, h_end, w_start, w_end) per camera,
                       in the order cameras are concatenated in the feature tensor
        cam_patches:   list of (n_h, n_w) patch counts per camera
        frame_indices: optional explicit temporal positions for each frame block,
                       e.g. [0, 2] for an anchor + last-frame goal. When None,
                       defaults to range(n_frames).

    Returns:
        Tensor [3, F_used * sum(n_h_i * n_w_i)] with rows (f, h, w)
    """
    cam_h_list = []
    cam_w_list = []
    for i, (h_start, h_end, w_start, w_end) in enumerate(cam_regions):
        nh, nw = cam_patches[i]
        i_idx = torch.arange(nh, dtype=torch.float32)
        j_idx = torch.arange(nw, dtype=torch.float32)
        h_pos = (h_start + i_idx * (h_end - h_start) / nh).round().long()
        w_pos = (w_start + j_idx * (w_end - w_start) / nw).round().long()
        h_pos = h_pos.clamp(h_start, h_end - 1)
        w_pos = w_pos.clamp(w_start, w_end - 1)
        hh, ww = torch.meshgrid(h_pos, w_pos, indexing="ij")
        cam_h_list.append(hh.flatten())
        cam_w_list.append(ww.flatten())

    cam_h = torch.cat(cam_h_list)  # [N_total]
    cam_w = torch.cat(cam_w_list)
    n_total = cam_h.shape[0]

    if frame_indices is None:
        f_idx = torch.arange(n_frames)
    else:
        f_idx = torch.tensor(frame_indices, dtype=torch.long)
    F_used = f_idx.shape[0]
    ff = f_idx[:, None].expand(F_used, n_total).reshape(-1)
    hh = cam_h[None].expand(F_used, n_total).reshape(-1)
    ww = cam_w[None].expand(F_used, n_total).reshape(-1)

    return torch.stack([ff, hh, ww], dim=0)  # [3, F_used * N_total]


def compute_dino_freqs(
    freqs_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_frames: int,
    cam_regions: list[tuple[int, int, int, int]],
    cam_patches: list[tuple[int, int]],
    device: torch.device,
    frame_indices: list[int] | None = None,
) -> torch.Tensor:
    """Build RoPE frequencies for DINO tokens using the video expert's freqs.

    Indexes into the same precomputed ``freqs_cis_3d`` at DINO-specific
    spatial positions from the composed multi-camera grid.

    Args:
        freqs_3d:      tuple of (f_freqs, h_freqs, w_freqs) from video expert
        n_frames:      number of temporal frames (ignored if `frame_indices` is given)
        cam_regions:   camera spatial regions in composed grid
        cam_patches:   patch grid sizes per camera
        device:        target device
        frame_indices: optional explicit temporal positions for each frame block;
                       see `get_dino_mesh_id`.

    Returns:
        Tensor [F_used * N_total, 1, rope_dim] — same format as video freqs
    """
    grid = get_dino_mesh_id(
        n_frames=n_frames,
        cam_regions=cam_regions,
        cam_patches=cam_patches,
        frame_indices=frame_indices,
    )  # [3, F_used*N]
    f_idx = grid[0].long()
    h_idx = grid[1].long()
    w_idx = grid[2].long()

    freqs_f, freqs_h, freqs_w = freqs_3d

    freqs = torch.cat([
        freqs_f[f_idx],
        freqs_h[h_idx],
        freqs_w[w_idx],
    ], dim=-1)  # [F*N, rope_dim_complex]

    return freqs.unsqueeze(1).to(device)  # [F*N, 1, rope_dim_complex]


def select_aux_frame_slots(
    num_latent: int,
    stride: int,
    keep_far: bool = False,
) -> list[int]:
    """Latent-slot indices for the DINO/aux temporal stride.

    Frame 0 (the clean anchor) is always included. ``keep_far=False`` is the
    legacy selection (stride forward from slot 1, bit-identical to the
    original inline loops in ``DinoEncoder.encode_video`` and
    ``FlexPi._aux_per_frame_is_pad``); ``keep_far=True`` strides BACKWARD
    from the last slot so the farthest future frame is always supervised
    (e.g. num_latent=3, stride=2: [0, 2] instead of [0, 1]). Both selections
    have identical length ``1 + ceil((num_latent-1)/stride)``, so frame-count
    bookkeeping is unaffected.

    Args:
        num_latent: number of VAE-aligned latent slots (frame 0 included).
        stride: temporal stride (1 = every slot).
        keep_far: stride from the end instead of the start.

    Returns:
        Ascending list of latent-slot indices, starting with 0.
    """
    if keep_far:
        return [0] + sorted(range(num_latent - 1, 0, -stride))
    return [0] + list(range(1, num_latent, stride))


# ── x0-parameterization (``dino_pred_x0``) ───────────────────────────────────
# x0-parameterization (dino_pred_x0): sigma floor for the analytic x̂0 → v̂
# conversion. With the shift-6 training sampler P(sigma < 0.05) ≈ 0.9%, so the
# clamp touches almost no samples while capping the 1/sigma error amplification
# at 20×; inference schedules never reach sigma < 0.05 below ~115 steps.
_DINO_X0_SIGMA_MIN = 0.05


def _dino_x0_to_velocity(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    timestep: torch.Tensor,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Convert a clean-feature prediction x̂0 into a velocity v̂ analytically.

    Flow-matching algebra: x_t = (1-σ)x0 + σε and v = ε − x0 give
    v = (x_t − x0)/σ, so v̂ = (x_t − x̂0)/σ. Training the v-space MSE on this
    v̂ keeps ``loss_dino`` numerically comparable with v-parameterized runs; at
    inference the converted v̂ feeds ``scheduler.step`` unchanged.

    Args:
        x_t: noisy DINO features, any layout matching ``x0_pred``.
        x0_pred: the head's clean-feature prediction (same layout as ``x_t``).
        timestep: per-sample timestep, pre-viewed for broadcasting against
            ``x_t`` (e.g. ``t.view(B, 1, 1)`` for [B, S, D]).
        num_train_timesteps: scheduler's timestep scale (σ = t / this).

    Returns:
        v̂ in float32 (callers cast back as needed).
    """
    sigma = (timestep.float() / float(num_train_timesteps)).clamp(min=_DINO_X0_SIGMA_MIN)
    return (x_t.float() - x0_pred.float()) / sigma


def _zero_init_dino_x0_head_(head: nn.Module) -> None:
    """Zero-init the FINAL layer of a DINO output head, in place.

    Used when ``dino_pred_x0=True`` for every head variant: the head output is
    read as x̂0, and zero-init gives x̂0 ≈ 0 at step 0. ``head`` is the proj
    module: a Linear, or a Sequential ending in one."""
    last = head[-1] if isinstance(head, nn.Sequential) else head
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
