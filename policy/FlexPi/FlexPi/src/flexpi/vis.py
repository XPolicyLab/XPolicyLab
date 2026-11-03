"""Validation-visualization row builders for every stream.

One module because the trainer assembles a single val-vis grid out of all of
them, and they share the PCA / colormap / tile-arrangement helpers:

  video / DINO   ``build_vae_row``, ``build_dino_row``, ``visualize_vae_latent``
  pointmap       ``build_pointmap_row_vae``, ``build_pointmap_depth_row_vae``,
                 ``build_inverse_depth_row``

Spatial arrangement is layout-driven (``LayoutSpec`` from
``flexpi.composite_layouts``), so any registered layout —
tshape_robotwin_384x320_uniform, tshape_libero_2cam_448x512, future entries —
produces visualizations that mirror the model's composite shape.

Nothing here is on an inference path: the trainer's val step and the YAM
prediction recorder are the only callers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from flexpi.composite_layouts import LayoutSpec, get_layout


def scalar_to_heatmap_u8(norm: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] scalar map to jet-like RGB heatmap.

    Args:
        norm: [...] float in [0, 1]

    Returns:
        [..., 3] uint8 RGB
    """
    x = np.clip(norm.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def visualize_vae_latent(
    vae_latent: torch.Tensor,
    target_h: int = 384,
    target_w: int = 320,
    pca_basis: tuple[np.ndarray, np.ndarray] | None = None,
    value_range: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """PCA→RGB and norm-heatmap for VAE latents.

    The VAE latent is already in the concatenated camera layout (the VAE
    encodes the full concatenated frame), so no per-camera rearrangement
    is needed — just PCA over channels and upscale.

    Args:
        vae_latent: [C, F, H, W] (e.g., [48, 9, 24, 20])
        target_h: output height (384 = RoboTwin frame height)
        target_w: output width (320 = RoboTwin frame width)
        pca_basis: optional pre-fitted (mean, vt[:3]) — projects onto a
            shared color space so identical features visualize identically.
        value_range: optional ``(pca_min[3], pca_max[3], norm_min, norm_max)``
            to use instead of the per-frame min/max normalization. When
            given, every frame is normalized with these *fixed* bounds so
            colors are consistent across calls (used by the deploy recorder
            to keep colors stable across chunks). ``None`` = per-frame
            (default; matches training-eval behavior).

    Returns:
        pca_u8:  [F, target_h, target_w, 3] uint8
        norm_u8: [F, target_h, target_w, 3] uint8
        basis: (mean, vt[:3]) — fitted-or-passed-through; reusable.
    """
    C, nF, H, W = vae_latent.shape
    feat = vae_latent.permute(1, 2, 3, 0).contiguous().to(torch.float32).cpu().numpy()  # [F, H, W, C]

    # PCA: either fit fresh on this input or reuse the supplied basis.
    x_flat = feat.reshape(nF * H * W, C)
    if pca_basis is None:
        mean = x_flat.mean(axis=0, keepdims=True)
        _, _, vt_full = np.linalg.svd(x_flat - mean, full_matrices=False)
        vt = vt_full[:3]
    else:
        mean, vt = pca_basis
    proj = ((x_flat - mean) @ vt.T).reshape(nF, H, W, 3)
    if value_range is not None:
        pca_min, pca_max, _, _ = value_range
        mn = np.asarray(pca_min, dtype=np.float32).reshape(1, 1, 1, 3)
        mx = np.asarray(pca_max, dtype=np.float32).reshape(1, 1, 1, 3)
    else:
        mn = proj.reshape(nF, -1, 3).min(axis=1)[:, None, None, :]
        mx = proj.reshape(nF, -1, 3).max(axis=1)[:, None, None, :]
    proj = (proj - mn) / (mx - mn + 1e-8)

    # Mean-norm (RMS over channels)
    norm = np.sqrt(np.mean(feat * feat, axis=-1))  # [F, H, W]
    if value_range is not None:
        _, _, norm_min, norm_max = value_range
        norm_mn = float(norm_min)
        norm_mx = float(norm_max)
    else:
        norm_mn = norm.reshape(nF, -1).min(axis=1)[:, None, None]
        norm_mx = norm.reshape(nF, -1).max(axis=1)[:, None, None]
    norm = (norm - norm_mn) / (norm_mx - norm_mn + 1e-8)

    if value_range is not None:
        # Fixed bounds (from a different tensor) can push values outside
        # [0, 1]; clip so the uint8 conversion doesn't wrap. No-op for the
        # per-frame path (exact min/max already yields [0, 1]).
        proj = np.clip(proj, 0.0, 1.0)
        norm = np.clip(norm, 0.0, 1.0)

    # Upscale to target size
    pca_u8 = (_resize_np(proj, target_h, target_w, "bilinear") * 255.0).astype(np.uint8)
    norm_u8 = scalar_to_heatmap_u8(_resize_np(norm, target_h, target_w, "bilinear"))

    return pca_u8, norm_u8, (mean, vt)


def build_vae_row(
    gt_latent: torch.Tensor,
    pred_latent: torch.Tensor | None = None,
    share_pca_basis: bool = True,
    target_hw: tuple[int, int] = (384, 320),
) -> np.ndarray:
    """Build VAE latent visualization row.

    Layout: ``[Pred_VAE_PCA | GT_VAE_PCA]`` — each tile sized by ``target_hw``,
    concatenated horizontally. Matches the DINO and RGB row widths when
    ``target_hw`` equals the layout's ``composite_hw``.

    If ``pred_latent`` is None (e.g., no video generation at eval), the
    first column shows GT PCA instead.

    Args:
        gt_latent:   [C, F, H, W] (e.g., [48, 9, 24, 20])
        pred_latent: [C, F, H, W] or None
        share_pca_basis: when True (default), pred and gt are projected onto
            the same GT-fitted PCA basis so identical features yield identical
            colors. Set False to fit independent bases (legacy behavior).
        target_hw: (H, W) per-tile output size. Default RoboTwin (384, 320);
            pass ``model._layout.composite_hw`` for non-RoboTwin layouts.

    Returns:
        [F, target_h, 2*target_w, 3] uint8 numpy array
    """
    th, tw = target_hw
    gt_pca, _gt_norm, gt_basis = visualize_vae_latent(gt_latent, target_h=th, target_w=tw)

    if pred_latent is not None:
        basis = gt_basis if share_pca_basis else None
        pred_pca, _, _ = visualize_vae_latent(
            pred_latent, target_h=th, target_w=tw, pca_basis=basis,
        )
    else:
        pred_pca = gt_pca  # no predicted latent available, show GT PCA in both columns

    # [Pred_PCA | GT_PCA]
    return np.concatenate([pred_pca, gt_pca], axis=2)  # [F, 384, 640, 3]


def _compute_dino_pca_and_norm(
    dino_features: torch.Tensor,
    cam_patch_sizes: list[tuple[int, int]],
    pca_basis: tuple[np.ndarray, np.ndarray] | None = None,
    value_range: tuple | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Compute per-camera PCA and norm arrays from DINO features.

    Args:
        dino_features: [C, F, N_total] or [C, F, N_total, 1]
        cam_patch_sizes: list of (n_h, n_w) per camera
        pca_basis: optional pre-fitted (mean, vt[:3]) to project onto. When
            given, the same basis is used (no SVD on this input) so that two
            different feature tensors plotted side-by-side share a color
            space — identical features give identical colors.
        value_range: optional ``(pca_min[3], pca_max[3], norm_min, norm_max)``
            fixed normalization bounds (instead of per-frame min/max). Used
            by the deploy recorder for color stability across chunks.
            ``None`` = per-frame (default; matches training-eval behavior).

    Returns:
        pca_per_cam:  list of [F, n_h, n_w, 3] float in [0, 1] per camera
        norm_per_cam: list of [F, n_h, n_w] float in [0, 1] per camera
        basis: (mean, vt[:3]) — fitted-or-passed-through; reusable for
            projecting another tensor onto the same color space.
    """
    if dino_features.ndim == 4:
        dino_features = dino_features[:, :, :, 0]
    C, nF, N = dino_features.shape
    feat = dino_features.permute(1, 2, 0).contiguous().to(torch.float32).cpu().numpy()  # [nF, N, C]

    # PCA: either fit fresh on this input or reuse the supplied basis.
    x_flat = feat.reshape(nF * N, C)
    if pca_basis is None:
        mean = x_flat.mean(axis=0, keepdims=True)
        _, _, vt_full = np.linalg.svd(x_flat - mean, full_matrices=False)
        vt = vt_full[:3]
    else:
        mean, vt = pca_basis
    proj = ((x_flat - mean) @ vt.T).reshape(nF, N, 3)
    if value_range is not None:
        pca_min, pca_max, _, _ = value_range
        mn = np.asarray(pca_min, dtype=np.float32).reshape(1, 1, 3)
        mx = np.asarray(pca_max, dtype=np.float32).reshape(1, 1, 3)
    else:
        mn = proj.min(axis=1, keepdims=True)
        mx = proj.max(axis=1, keepdims=True)
    proj = (proj - mn) / (mx - mn + 1e-8)

    # Mean-norm (RMS over channels)
    norm = np.sqrt(np.mean(feat * feat, axis=-1))  # [nF, N]
    if value_range is not None:
        _, _, norm_min, norm_max = value_range
        norm_mn = float(norm_min)
        norm_mx = float(norm_max)
    else:
        norm_mn = norm.min(axis=1, keepdims=True)
        norm_mx = norm.max(axis=1, keepdims=True)
    norm = (norm - norm_mn) / (norm_mx - norm_mn + 1e-8)

    if value_range is not None:
        proj = np.clip(proj, 0.0, 1.0)
        norm = np.clip(norm, 0.0, 1.0)

    # Split per camera
    pca_per_cam, norm_per_cam = [], []
    offset = 0
    for ph, pw in cam_patch_sizes:
        n = ph * pw
        pca_per_cam.append(proj[:, offset:offset + n, :].reshape(nF, ph, pw, 3))
        norm_per_cam.append(norm[:, offset:offset + n].reshape(nF, ph, pw))
        offset += n

    return pca_per_cam, norm_per_cam, (mean, vt)


def _resize_np(arr: np.ndarray, target_h: int, target_w: int, mode: str = "bilinear") -> np.ndarray:
    """Resize [F, H, W, C] or [F, H, W] numpy array using torch interpolation."""
    if arr.ndim == 3:
        t = torch.from_numpy(arr).unsqueeze(1).float()  # [F, 1, H, W]
        t = F.interpolate(t, size=(target_h, target_w), mode=mode,
                          **({"align_corners": False} if mode == "bilinear" else {}))
        return t.squeeze(1).clamp(0, 1).numpy()
    else:
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [F, C, H, W]
        t = F.interpolate(t, size=(target_h, target_w), mode=mode,
                          **({"align_corners": False} if mode == "bilinear" else {}))
        return t.permute(0, 2, 3, 1).clamp(0, 1).numpy()


def _arrange_layout(
    per_cam: list[np.ndarray],
    is_rgb: bool = True,
    layout: Optional[LayoutSpec] = None,
) -> np.ndarray:
    """Arrange per-camera arrays into the composite spatial layout.

    Walks ``layout.cam_slots()`` and pastes each per-cam array (resized to its
    slot HW) at the slot's ``(top, left)`` position. Black slots stay zero.

    Args:
        per_cam: list of per-camera arrays, in ``layout.cam_slots()`` order.
            Each ``[F, h, w, 3]`` (RGB) or ``[F, h, w]`` (scalar).
        is_rgb: True if arrays have channel dim (3), False for scalar maps.
        layout: target ``LayoutSpec``. Defaults to RoboTwin (back-compat).

    Returns:
        ``[F, H_total, W_total, 3]`` uint8.
    """
    if layout is None:
        layout = get_layout("tshape_robotwin_384x320_uniform")
    cam_slots = layout.cam_slots()
    if len(per_cam) != len(cam_slots):
        raise ValueError(
            f"Layout {layout.name!r} expects {len(cam_slots)} cams, got {len(per_cam)}"
        )

    nF = per_cam[0].shape[0]
    H_total, W_total = layout.composite_hw

    if is_rgb:
        canvas = np.zeros((nF, H_total, W_total, 3), dtype=np.float32)
    else:
        canvas = np.zeros((nF, H_total, W_total), dtype=np.float32)

    for cam, slot in zip(per_cam, cam_slots):
        resized = _resize_np(cam, slot.h, slot.w, "bilinear")
        canvas[:, slot.top:slot.top + slot.h, slot.left:slot.left + slot.w] = resized

    if is_rgb:
        return (canvas * 255.0).astype(np.uint8)
    return scalar_to_heatmap_u8(canvas)


# Back-compat alias for legacy callers.
def _arrange_robotwin_layout(per_cam, is_rgb: bool = True) -> np.ndarray:
    return _arrange_layout(per_cam, is_rgb=is_rgb, layout=get_layout("tshape_robotwin_384x320_uniform"))


def visualize_dino_layout(
    dino_features: torch.Tensor,
    cam_patch_sizes: list[tuple[int, int]] | None = None,
    pca_basis: tuple[np.ndarray, np.ndarray] | None = None,
    layout: Optional[LayoutSpec] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Visualize DINO features in the composite layout.

    Args:
        dino_features: ``[C, F, N_total, 1]`` or ``[C, F, N_total]``.
        cam_patch_sizes: per-camera patch grid sizes. Defaults to the layout's
            ``dino_cam_patches``.
        pca_basis: optional pre-fitted (mean, vt[:3]) — projects onto a shared
            color space so identical features visualize as identical colors.
        layout: target ``LayoutSpec``. Defaults to RoboTwin (back-compat).

    Returns:
        pca_u8:  ``[F, H_total, W_total, 3]`` uint8 — PCA→RGB in composite layout.
        norm_u8: ``[F, H_total, W_total, 3]`` uint8 — norm heatmap in composite layout.
    """
    if layout is None:
        layout = get_layout("tshape_robotwin_384x320_uniform")
    if cam_patch_sizes is None:
        cam_patch_sizes = [tuple(p) for p in layout.dino_cam_patches]

    pca_per_cam, norm_per_cam, _basis = _compute_dino_pca_and_norm(
        dino_features, cam_patch_sizes, pca_basis=pca_basis,
    )
    pca_u8 = _arrange_layout(pca_per_cam, is_rgb=True, layout=layout)
    norm_u8 = _arrange_layout(norm_per_cam, is_rgb=False, layout=layout)
    return pca_u8, norm_u8


# Back-compat alias.
def visualize_dino_robotwin(
    dino_features: torch.Tensor,
    cam_patch_sizes: list[tuple[int, int]] | None = None,
    pca_basis: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return visualize_dino_layout(
        dino_features, cam_patch_sizes=cam_patch_sizes,
        pca_basis=pca_basis, layout=get_layout("tshape_robotwin_384x320"),
    )


def _overlay_tile_label(tile: np.ndarray, label: str) -> np.ndarray:
    """Draw a small text label on the top-left of an HxWx3 uint8 tile."""
    img = Image.fromarray(tile)
    draw = ImageDraw.Draw(img)
    # ImageFont.load_default() works without installing TTF fonts — small
    # but always available, sufficient for short tags.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 3
    # Black filled background for legibility against any color content.
    text_w = max(8 * len(label), 20)
    text_h = 14
    draw.rectangle((0, 0, text_w + 2 * pad, text_h + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def _unfold_pixel_unshuffle_dino(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
    cam_patch_sizes: list[tuple[int, int]],
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Un-fold pixel-unshuffle DINO back to native resolution for visualization.

    Pixel-unshuffle folds each view's native ``h×w`` grid into ``(h/f)×(w/f)``
    tokens of ``f²·768`` channels (lossless). For viz we want the NATIVE grid,
    so we ``pixel_shuffle`` (the exact inverse) each cam back to ``h×w × 768``,
    re-laid on the token axis. gt is ``[C=f²·768, F, N, 1]`` and pred is
    ``[F*N, C]``; returns gt ``[768, F, N_native, 1]``, pred ``[F*N_native, 768]``
    and the native per-cam grids ``[(h·f, w·f), ...]``. Inverse of
    ``DinoEncoder._pixel_unshuffle_patches``; loss/training are unaffected (this
    is render-only).
    """
    f = factor
    g = gt_features[0] if gt_features.ndim == 5 else gt_features      # [C, F, N, 1]
    g = g.squeeze(-1)                                                 # [C, F, N]
    C, nF, _ = g.shape
    D = C // (f * f)
    p = pred_features[0] if pred_features.ndim == 3 else pred_features  # [F*N, C]
    Ntot = p.shape[0] // nF
    p = p.reshape(nF, Ntot, C)                                       # [F, N, C]

    g_cams, p_cams, native = [], [], []
    off = 0
    for (ph, pw) in cam_patch_sizes:
        n = ph * pw
        # gt: [C, F, n] -> [F, C, ph, pw] -> shuffle -> [F, D, ph*f, pw*f]
        gs = g[:, :, off:off + n].reshape(C, nF, ph, pw).permute(1, 0, 2, 3)
        gs = F.pixel_shuffle(gs, f)                                  # [F, D, ph*f, pw*f]
        H, W = ph * f, pw * f
        g_cams.append(gs.permute(1, 0, 2, 3).reshape(D, nF, H * W))  # [D, F, n*f²]
        # pred: [F, n, C] -> [F, C, ph, pw] -> shuffle -> [F, D, H, W]
        ps = p[:, off:off + n, :].reshape(nF, ph, pw, C).permute(0, 3, 1, 2)
        ps = F.pixel_shuffle(ps, f)                                  # [F, D, H, W]
        p_cams.append(ps.permute(0, 2, 3, 1).reshape(nF, H * W, D))  # [F, n*f², D]
        native.append((H, W))
        off += n
    g_out = torch.cat(g_cams, dim=2).unsqueeze(-1)                   # [D, F, N_native, 1]
    p_out = torch.cat(p_cams, dim=1).reshape(nF * sum(h * w for h, w in native), D)
    return g_out, p_out, native


def build_dino_row(
    gt_features: torch.Tensor,
    pred_features: torch.Tensor,
    cam_patch_sizes: list[tuple[int, int]] | None = None,
    num_video_latent_frames: int | None = None,
    anchor_count: int = 0,
    share_pca_basis: bool = True,
    layout: Optional[LayoutSpec] = None,
    pixel_unshuffle: int = 0,
) -> np.ndarray:
    """Build the DINO visualization row aligned to video latent frame count.

    Layout: ``[Pred_DINO_PCA | GT_DINO_PCA]`` — each
    ``H_total × W_total`` in the active composite layout, concatenated to
    ``H_total × 2*W_total``.

    When DINO has fewer frames than video (``dino_temporal_stride > 1``),
    DINO frames are repeated (hold-last) to fill all video latent frames
    so the result is temporally aligned with the video row.

    Args:
        gt_features:    [C, F_dino, N, 1] or [B, C, F_dino, N, 1]
        pred_features:  [B, F_dino*N, C] (model output)
        cam_patch_sizes: per-camera patch grid sizes. Defaults to
            ``layout.dino_cam_patches``.
        num_video_latent_frames: total VAE latent frames (F_video).
            If provided and > F_dino, DINO frames are repeat-expanded to match.
        anchor_count: number of leading tiles in pred that are *clean anchors*
            (e.g. obs DINO carried into val-vis), not denoised predictions.
            Those tiles get an "ANCHOR" text overlay; the rest get "PRED".
        share_pca_basis: when True (default), pred is projected onto the same
            GT-fitted PCA basis so identical features (e.g. an anchor tile
            equal to gt frame 0) produce identical colors. Set False to fit
            independent bases (legacy behavior).
        layout: target ``LayoutSpec``. Defaults to RoboTwin (back-compat).

    Returns:
        ``[F_video, H_total, 2*W_total, 3]`` uint8 numpy array.
    """
    if layout is None:
        layout = get_layout("tshape_robotwin_384x320_uniform")
    if cam_patch_sizes is None:
        cam_patch_sizes = [tuple(p) for p in layout.dino_cam_patches]

    if pixel_unshuffle:
        # Pixel-unshuffle folds f² spatial sub-patches into channels; un-fold
        # back to the NATIVE grid (h·f × w·f, 768-d) so the PCA renders the full
        # spatial detail the lossless fold preserves (not the coarse h×w tokens).
        gt_features, pred_features, cam_patch_sizes = _unfold_pixel_unshuffle_dino(
            gt_features, pred_features, cam_patch_sizes, pixel_unshuffle,
        )

    if gt_features.ndim == 5:
        gt_features = gt_features[0]

    nC, nF_dino, nN = gt_features.shape[0], gt_features.shape[1], gt_features.shape[2]

    if pred_features.ndim == 3:
        pred_features = pred_features[0]
    pred_reshaped = pred_features.reshape(nF_dino, nN, nC).permute(2, 0, 1).unsqueeze(-1)

    # When share_pca_basis=True (default), fit PCA on GT and reuse the basis
    # for pred so identical features (e.g. an anchor tile equal to gt frame 0)
    # produce identical colors. When False, both fit their own bases (legacy).
    gt_per_cam_pca, _gt_per_cam_norm, gt_basis = _compute_dino_pca_and_norm(
        gt_features, cam_patch_sizes,
    )
    pred_basis = gt_basis if share_pca_basis else None
    pred_per_cam_pca, _pred_norm, _ = _compute_dino_pca_and_norm(
        pred_reshaped, cam_patch_sizes, pca_basis=pred_basis,
    )
    gt_pca = _arrange_layout(gt_per_cam_pca, is_rgb=True, layout=layout)
    pred_pca = _arrange_layout(pred_per_cam_pca, is_rgb=True, layout=layout)
    # each [nF_dino, H_total, W_total, 3] uint8

    # Per-tile labels on the Pred column: clarify which tiles are anchors (the
    # clean obs DINO carried through unchanged) vs genuine model predictions.
    if anchor_count > 0:
        labeled = []
        for i in range(nF_dino):
            tag = "ANCHOR" if i < anchor_count else "PRED"
            labeled.append(_overlay_tile_label(pred_pca[i], tag))
        pred_pca = np.stack(labeled, axis=0)

    # Stitch: [Pred_PCA | GT_PCA]
    dino_row = np.concatenate([pred_pca, gt_pca], axis=2)  # [nF_dino, 384, 640, 3]

    # Temporal alignment: repeat-expand to match video latent frame count
    if num_video_latent_frames is not None and num_video_latent_frames > nF_dino:
        # Map each video frame to the nearest DINO frame (hold-last)
        indices = np.clip(
            np.arange(num_video_latent_frames) * nF_dino // num_video_latent_frames,
            0, nF_dino - 1,
        )
        dino_row = dino_row[indices]  # [F_video, 384, 640, 3]

    return dino_row


# ── pointmap rows ────────────────────────────────────────────────────────────
def build_pointmap_row_vae(
    pred: torch.Tensor,
    vae_recon: torch.Tensor,
) -> np.ndarray:
    """Build the ``vae_parallel`` pointmap visualization row.

    Both tiles are XYZ composites in ``[0, 1]`` rendered directly as RGB
    (the ``(x+1)/2`` remap from ``[-1, 1]`` XYZ is the caller's responsibility,
    mirroring how the video row already passes tensors in ``[0, 1]``).

    Args:
        pred: ``[3, F, H, W]`` — decoded pred pointmap latents.
        vae_recon: ``[3, F, H, W]`` — GT composite → VAE encode → decode.

    Returns:
        uint8 array ``[F, H, 2*W, 3]``.
    """
    if pred.shape != vae_recon.shape:
        raise ValueError(
            f"pred/vae_recon shape mismatch: "
            f"{tuple(pred.shape)} vs {tuple(vae_recon.shape)}"
        )
    stitched = torch.cat([pred, vae_recon], dim=-1).clamp(0.0, 1.0)  # [3, F, H, 2W]
    arr = (stitched * 255.0).to(torch.uint8).cpu().numpy()
    return np.transpose(arr, (1, 2, 3, 0))  # [F, H, 2W, 3]


def _turbo_colormap(t: torch.Tensor) -> torch.Tensor:
    """Turbo colormap (Mikhailov 2019) via degree-5 polynomial approximation.

    Near (t≈0) → dark purple/blue, mid → green/yellow, far (t≈1) → red/orange.
    Coefficients from Anton Mikhailov's reference shader (public domain).
    """
    t = t.clamp(0.0, 1.0).float()

    def _poly(c0, c1, c2, c3, c4, c5):
        return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))

    r = _poly(0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
    g = _poly(0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604)
    b = _poly(0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973)
    return torch.stack([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)], dim=0)


def _gray_colormap(t: torch.Tensor) -> torch.Tensor:
    """Grayscale: t in [0, 1] → [3, ...] with all channels equal to t."""
    t = t.clamp(0.0, 1.0).float()
    return t.unsqueeze(0).expand(3, *t.shape).contiguous()


_DEPTH_COLORMAPS = {"turbo": _turbo_colormap, "gray": _gray_colormap}


def _depth_colormap(t: torch.Tensor, mode: str = "turbo") -> torch.Tensor:
    """Dispatch to a depth colormap by name. Supported: ``turbo`` (default), ``gray``."""
    try:
        fn = _DEPTH_COLORMAPS[mode]
    except KeyError:
        raise ValueError(
            f"Unknown depth vis_mode {mode!r}; choose from {sorted(_DEPTH_COLORMAPS.keys())}"
        )
    return fn(t)


def build_pointmap_depth_row_vae(
    pred: torch.Tensor,
    vae_recon: torch.Tensor,
    z_min: float,
    z_max: float,
    vis_mode: str = "turbo",
) -> np.ndarray:
    """Project the Z channel of the XYZ composite back to depth and render a
    side-by-side row ``[pred_depth | vae_recon_depth]`` as grayscale.

    The caller already mapped XYZ from ``[-1, 1]`` to ``[0, 1]``; channel 2 of
    each tile is the normalized depth, linearly spanning ``[z_min, z_max]``
    meters. We display it directly as grayscale (brightness = depth), so the
    same metric depth is the same shade across pred / recon and across
    frames. ``z_min`` / ``z_max`` are accepted for API symmetry but are not
    used for the on-screen mapping — re-normalizing would break the cross-tile
    comparison.

    Invalid pixels (``PointmapEncoder._unproject`` zeros them via its validity
    gate → ``z01 ≈ 0``) show up as pure black, same as the nearest valid depth.

    Args:
        pred / vae_recon: ``[3, F, H, W]`` in ``[0, 1]`` (same inputs as
            ``build_pointmap_row_vae``).
        z_min, z_max: unused; kept for call-site symmetry with the metric
            bounds stored on ``PointmapEncoder.pt_min/pt_max``.

    Returns:
        uint8 array ``[F, H, 2*W, 3]``.
    """
    del z_min, z_max  # unused — depth shown directly from the [0, 1] Z channel
    if pred.shape != vae_recon.shape:
        raise ValueError(
            f"pred/vae_recon shape mismatch: "
            f"{tuple(pred.shape)} vs {tuple(vae_recon.shape)}"
        )

    def _depth_rgb(xyz01: torch.Tensor) -> torch.Tensor:
        # xyz01: [3, F, H, W] — Z is channel 2, already in [0, 1]
        return _depth_colormap(xyz01[2].detach().cpu(), mode=vis_mode)  # [3, F, H, W]

    pred_d = _depth_rgb(pred)
    recon_d = _depth_rgb(vae_recon)

    stitched = torch.cat([pred_d, recon_d], dim=-1).clamp(0.0, 1.0)  # [3, F, H, 2W]
    arr = (stitched * 255.0).to(torch.uint8).numpy()
    return np.transpose(arr, (1, 2, 3, 0))  # [F, H, 2W, 3]


def build_inverse_depth_row(
    pred_inv_per_cam: dict,
    gt_inv_per_cam: dict,
    vis_mode: str = "turbo",
) -> np.ndarray:
    """Side-by-side row ``[pred_inv_depth | gt_inv_depth]`` for FlexPiXWAM.

    Both inputs are inverse depth in ``[0, 1]`` (1.0 = nearest, 0.0 = far/invalid).
    Per-cam tiles are bilinearly resized to the standard RoboTwin slot layout
    (head 256×320, wrists 128×160) and tiled into a 384×320 frame. Pred and GT
    side-by-side gives a 384×640 frame — the same width as
    ``build_pointmap_depth_row_vae``.

    Args:
        pred_inv_per_cam / gt_inv_per_cam: dict[cam → tensor]. Each tensor is
            ``[F, H, W]`` or ``[F, H, W, 1]`` (the trailing 1 is squeezed) in
            ``[0, 1]``. Keys ``cam_high``, ``cam_left_wrist``, ``cam_right_wrist``.
        vis_mode: ``"turbo"`` (default) or ``"gray"``.

    Returns:
        uint8 array ``[F, 384, 640, 3]``.
    """
    cam_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    slot_sizes = ((256, 320), (128, 160), (128, 160))

    def _to_3d(t: torch.Tensor) -> torch.Tensor:
        x = t.detach().to(torch.float32).cpu()
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 3:
            raise ValueError(f"Per-cam inverse depth must be [F,H,W] or [F,H,W,1], got {tuple(t.shape)}")
        return x.clamp(0.0, 1.0)

    def _build_panel(per_cam: dict) -> torch.Tensor:
        tiles = []
        for key, (h, w) in zip(cam_keys, slot_sizes):
            inv = _to_3d(per_cam[key])  # [F, H_cam, W_cam]
            inv = F.interpolate(
                inv.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze(1).clamp(0.0, 1.0)
            tiles.append(inv)
        head, left, right = tiles
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([head, bottom], dim=-2)  # [F, 384, 320]

    pred_panel = _build_panel(pred_inv_per_cam)
    gt_panel = _build_panel(gt_inv_per_cam)

    pred_rgb = _depth_colormap(pred_panel, mode=vis_mode)  # [3, F, 384, 320]
    gt_rgb = _depth_colormap(gt_panel, mode=vis_mode)
    # Match the 384×640 layout used by other depth rows.
    stitched = torch.cat([pred_rgb, gt_rgb], dim=-1).clamp(0.0, 1.0)
    arr = (stitched * 255.0).to(torch.uint8).numpy()
    return np.transpose(arr, (1, 2, 3, 0))  # [F, 384, 640, 3]
