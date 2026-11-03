"""Frozen 3D pointmap tokenizer: GT depth → unproject → normalize → composite.

Layout-aware: any registered ``LayoutSpec`` can drive the per-cam iteration
and slot recompose.
Defaults to RoboTwin for back-compat.

The encoder consumes depth from the dataset (``sample['per_cam_depth']``),
unprojects with the sample's per-camera K, min-max normalizes XYZ into
``[-1, 1]``, and pastes the resulting "XYZ-as-RGB" tensors into the composite
layout. The composite then goes through the same WAN VAE that encodes the RGB
composite, so the pointmap latents land 1:1 with the video latents.

Depth *source* is external to this module — it can come from the RoboTwin
simulator's GT, from a hardware depth sensor, or from any offline estimator
written into the lerobot dataset as ``observation.depth_ffv1.*``. The loader
doesn't care, so neither does the encoder.

Intrinsics are NOT stored on the encoder. The caller supplies per-camera K at
the depth grid via ``encode_composite(..., camera_intrinsics=...)``. The
typical source is ``sample["camera_intrinsics"]`` attached by
``RobotVideoDataset``.

This module holds no parameters — unproject/normalize/paste is pure tensor
math — so it is always frozen.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexpi.composite_layouts import LayoutSpec, get_layout
from flexpi.per_cam_compose import compose_from_per_cam
from flexpi.utils.logging_config import get_logger

logger = get_logger(__name__)


class PointmapEncoder(nn.Module):
    """GT-depth → normalized XYZ composite, in the RGB composite's layout.

    Depth is consumed from the sample dict; source (sim / sensor / offline
    estimate) is transparent to the encoder.
    """

    def __init__(
        self,
        norm_bounds: Optional[dict] = None,
        max_depth_m: float = 2.0,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.max_depth_m = float(max_depth_m)
        self._torch_dtype = torch_dtype

        # --- Normalization bounds (fixed, per maniflow defaults) ---
        nb = norm_bounds or {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 1.5]}
        pt_min = torch.tensor([nb["x"][0], nb["y"][0], nb["z"][0]], dtype=torch.float32).view(1, 3, 1, 1)
        pt_max = torch.tensor([nb["x"][1], nb["y"][1], nb["z"][1]], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pt_min", pt_min, persistent=False)
        self.register_buffer("pt_max", pt_max, persistent=False)

    # ------------------------------------------------------------------
    # Unprojection + normalization
    # ------------------------------------------------------------------

    def _unproject(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Depth [B,1,H,W] + K [B,3,3] → pointmap [B,3,H,W] in camera space (meters)."""
        B, _, H, W = depth.shape
        device, dtype = depth.device, depth.dtype
        v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        z = depth[:, 0]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        valid = (z > 0.01) & (z < self.max_depth_m)
        zero = torch.zeros_like(z)
        x = torch.where(valid, x, zero)
        y = torch.where(valid, y, zero)
        z = torch.where(valid, z, zero)
        return torch.stack([x, y, z], dim=1)

    def _normalize_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        pt_min = self.pt_min.to(device=pointmap.device, dtype=pointmap.dtype)
        pt_max = self.pt_max.to(device=pointmap.device, dtype=pointmap.dtype)
        clipped = torch.clamp(pointmap, pt_min, pt_max)
        return 2.0 * (clipped - pt_min) / (pt_max - pt_min) - 1.0

    # ------------------------------------------------------------------
    # Depth → per-cam XYZ
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _per_cam_xyz(
        self,
        per_cam_depth: Dict[str, torch.Tensor],
        camera_intrinsics: torch.Tensor,
        frame_indices: list,
        layout: Optional[LayoutSpec] = None,
        slot_key_map: Optional[Mapping[str, str]] = None,
    ) -> list:
        """Unproject + normalize per-cam GT depth at the depth-native resolution.

        Args:
            per_cam_depth: dict ``{dataset_cam_key: [B, T, H, W] uint16 mm}``.
            camera_intrinsics: ``[num_cams, 3, 3]`` at the depth grid (row order
                matches ``layout.cam_slots()``).
            frame_indices: T-indices to keep (list of ints).
            layout: target ``LayoutSpec``. Defaults to RoboTwin.
            slot_key_map: optional override mapping
                ``slot_placeholder -> dataset_cam_key``.

        Returns:
            list of tensors ``[B*F_dim, 3, H, W]`` in [-1, 1], in
            ``layout.cam_slots()`` order.
        """
        if layout is None:
            layout = get_layout("tshape_robotwin_384x320")
        kmap = layout.resolve_slot_key_map(slot_key_map)

        cam_slots = layout.cam_slots()
        cam_names = [kmap[s.key] for s in cam_slots]
        missing = [n for n in cam_names if n not in per_cam_depth]
        if missing:
            raise KeyError(
                f"per_cam_depth missing cams {missing}; got {list(per_cam_depth)}"
            )
        results = []
        for cam_idx, name in enumerate(cam_names):
            depth = per_cam_depth[name]
            if depth.ndim != 4:
                raise ValueError(
                    f"per_cam_depth['{name}'] must be [B, T, H, W]; got {tuple(depth.shape)}"
                )
            B, T, H, W = depth.shape
            # Cast to float BEFORE indexing — CUDA doesn't support index on
            # uint16 (spotty uint16 support in torch 2.x).
            depth_m_full = depth.float() / 1000.0
            if list(frame_indices) == list(range(len(frame_indices))):
                # Contiguous prefix (deploy first frame = [0]): slice instead
                # of list-indexing — identical values, no CPU index-tensor
                # upload (a stream sync; CUDA-graph-illegal).
                depth_m = depth_m_full[:, :len(frame_indices)]
            else:
                depth_m = depth_m_full[:, frame_indices]
            F_dim = depth_m.shape[1]
            depth_m = depth_m.reshape(B * F_dim, 1, H, W)
            K_cam = camera_intrinsics[cam_idx].to(
                device=depth_m.device, dtype=torch.float32,
            )
            K_batch = K_cam.unsqueeze(0).expand(B * F_dim, 3, 3)
            xyz = self._unproject(depth_m, K_batch)
            xyz = self._normalize_pointmap(xyz)
            results.append(xyz)
        return results

    @torch.no_grad()
    def encode_composite(
        self,
        per_cam_depth: Dict[str, torch.Tensor],
        camera_intrinsics: torch.Tensor,
        concat_mode: str = "tshape_robotwin_384x320",
        first_frame_only: bool = False,
        layout: Optional[LayoutSpec] = None,
        slot_key_map: Optional[Mapping[str, str]] = None,
    ) -> torch.Tensor:
        """Per-cam depth → composite pointmap tensor (the RGB-composite shape).

        Feeds the normalized pointmap composite through the same WAN VAE that
        encodes the RGB composite, so the VAE produces pointmap latents aligned
        with the video latents.

        Order: unproject at depth-native K, then NEAREST-resize per-cam XYZ
        into each slot's size (head 256×320, wrists 128×160). K is never
        rescaled here — the slot sizes are reached by geometric resampling of
        already-metric XYZ, which preserves depth discontinuities.

        Args:
            per_cam_depth: dict[cam -> [B, T, H, W]] uint16 mm. T comes from
                the dataset's video-subsample count (9 with the default
                ``num_frames=33``, ``action_video_freq_ratio=4``); the WAN VAE
                then compresses 9 → 3 latent frames, matching the 3 RGB video
                latents ``infer_joint`` allocates. When
                ``first_frame_only=True`` only T=1 is used.
            camera_intrinsics: ``[num_cams, 3, 3]`` at the depth grid.
            first_frame_only: if True, only encode frame 0.

        Returns:
            composite ``[B, 3, F_pt, 384, 320]`` in [-1, 1] (float32-cast to
            the encoder's torch_dtype), F_pt = 1 or T.
        """
        if camera_intrinsics is None:
            raise ValueError("`camera_intrinsics` is required for encode_composite.")
        if layout is None:
            layout = get_layout(concat_mode)
        kmap = layout.resolve_slot_key_map(slot_key_map)
        cam_slots = layout.cam_slots()

        if camera_intrinsics.ndim == 4:
            camera_intrinsics = camera_intrinsics[0]
        if (
            camera_intrinsics.shape[-2:] != (3, 3)
            or camera_intrinsics.shape[0] != len(cam_slots)
        ):
            raise ValueError(
                f"Expected camera_intrinsics of shape [{len(cam_slots)}, 3, 3], "
                f"got {tuple(camera_intrinsics.shape)}"
            )

        cam_names = [kmap[s.key] for s in cam_slots]
        missing = [n for n in cam_names if n not in per_cam_depth]
        if missing:
            raise KeyError(
                f"per_cam_depth missing cams {missing}; got {list(per_cam_depth)}"
            )
        first = per_cam_depth[cam_names[0]]
        if first.ndim != 4:
            raise ValueError(
                f"per_cam_depth['{cam_names[0]}'] must be [B, T, H, W]; got {tuple(first.shape)}"
            )
        B, T_src = first.shape[:2]
        frame_indices = [0] if first_frame_only else list(range(T_src))
        F_dim = len(frame_indices)

        # Per-cam unproject + normalize at depth-native resolution.
        xyz_per_cam = self._per_cam_xyz(
            per_cam_depth=per_cam_depth,
            camera_intrinsics=camera_intrinsics,
            frame_indices=frame_indices,
            layout=layout, slot_key_map=slot_key_map,
        )

        # NEAREST-resize each cam's XYZ to its composite slot size, then build
        # a [B*F, 3, T=1, H, W] per-cam dict keyed by dataset cam name and call
        # the generic `compose_from_per_cam`. Reusing the compose helper keeps
        # slot-pasting + black-slot fill in one place.
        per_cam_5d: Dict[str, torch.Tensor] = {}
        slot_hw_map = layout.slot_hw()
        for cam_idx, slot in enumerate(cam_slots):
            xyz = xyz_per_cam[cam_idx]
            H_dst, W_dst = slot_hw_map[slot.key]
            BF, C, H_src, W_src = xyz.shape
            if (H_src, W_src) != (H_dst, W_dst):
                xyz = F.interpolate(xyz, size=(H_dst, W_dst), mode="nearest")
            # [B*F, 3, H, W] → [B*F, 3, 1, H, W] for the 5D compose API.
            per_cam_5d[cam_names[cam_idx]] = xyz.unsqueeze(2)

        # Compose into [B*F, 3, 1, H_total, W_total] and squeeze the time dim.
        composite_flat = compose_from_per_cam(
            per_cam_5d, layout, slot_key_map=kmap,
        ).squeeze(2)

        H_total, W_total = layout.composite_hw
        composite = composite_flat.view(B, F_dim, 3, H_total, W_total).permute(0, 2, 1, 3, 4).contiguous()
        return composite.to(dtype=self._torch_dtype)
