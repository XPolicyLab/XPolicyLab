"""Frozen DINOv3 ViT-B/16 encoder for on-the-fly feature extraction.

Extracts patch tokens from raw video frames during training, matching the
temporal sampling of the VAE latent frames (stride-4 boundary frames).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from flexpi.composite_layouts import (
    LayoutSpec, get_layout,
)
from flexpi.utils.logging_config import get_logger

from .helpers.dino import select_aux_frame_slots

logger = get_logger(__name__)

# ImageNet normalization (DINO models expect this)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """Frozen DINOv3 ViT-B/16 for on-the-fly DINO feature extraction.

    Loads a timm ViT model, freezes all parameters, and provides an
    ``encode_video`` method that:
      1. Temporally subsamples raw frames at stride-4 boundaries
      2. Splits the concatenated multi-camera frame into individual views
      3. Resizes each view to 224×224
      4. Runs DINOv3 forward_features() to get patch tokens
      5. Pools wrist camera patches (2×2 avg pool: 14×14 → 7×7)
      6. Concatenates all cameras' patches per frame

    The model is always in eval mode with no gradients.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        dino_size: int = 224,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for DinoEncoder. Install with: pip install timm")

        self.model_name = model_name
        self.dino_size = dino_size

        logger.info(f"Loading DINO model: {model_name}")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(device=device, dtype=torch_dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)
        self.embed_dim = self.model.embed_dim
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.patches_per_side = dino_size // patch_size
        self.patches_per_frame = self.patches_per_side ** 2

        logger.info(
            f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"patches_per_frame={self.patches_per_frame}, "
            f"num_prefix_tokens={self.num_prefix_tokens}"
        )

        # Register ImageNet normalization constants (on same device as model)
        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN, device=device, dtype=torch_dtype).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD, device=device, dtype=torch_dtype).view(1, 3, 1, 1), persistent=False
        )

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] range to ImageNet mean/std."""
        # [-1, 1] → [0, 1]
        x = (x + 1.0) * 0.5
        return (x - self._mean) / self._std

    @staticmethod
    def _split_composite_frame(
        frames: torch.Tensor,
        layout: LayoutSpec,
    ) -> Dict[str, torch.Tensor]:
        """Split a composite ``[N, 3, H, W]`` tensor into per-slot crops.

        Returns a dict keyed by the layout's slot placeholder (e.g.
        ``"slot_top"``), in cam_slots() order. Black slots are NOT included
        (DINO doesn't run on them).
        """
        H, W = layout.composite_hw
        if frames.shape[-2:] != (H, W):
            raise ValueError(
                f"Layout {layout.name!r} expects composite {(H, W)}, "
                f"got {tuple(frames.shape[-2:])}"
            )
        out: Dict[str, torch.Tensor] = {}
        for s in layout.cam_slots():
            out[s.key] = frames[:, :, s.top:s.top + s.h, s.left:s.left + s.w]
        return out

    # Back-compat alias: returns a 3-tuple matching today's RoboTwin slot order
    # (cam_high, cam_left_wrist, cam_right_wrist). Used by `encode_frames` and
    # any external caller still on the legacy API.
    @staticmethod
    def _split_robotwin_frame(
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = get_layout("tshape_robotwin_384x320_uniform")
        d = DinoEncoder._split_composite_frame(frames, layout)
        return d["slot_top"], d["slot_bl"], d["slot_br"]

    @staticmethod
    def _pool_patches(
        features: torch.Tensor,
        patch_grid: int = 14,
        target_grid: Optional[tuple[int, int]] = None,
        mode: str = "avg",
    ) -> torch.Tensor:
        """Pool DINO patch tokens from ``patch_grid×patch_grid`` to
        ``target_grid``.

        When ``target_grid`` evenly divides ``patch_grid`` (e.g. 14→7, 14→14,
        7×14), uses a strided non-overlapping ``avg_pool2d`` — bit-equal to the
        legacy behavior. For non-divisor targets (e.g. 14→10, 14→12) it falls
        back to ``adaptive_avg_pool2d``, whose windows are uneven and may
        overlap but still average local patches. This makes intermediate wrist
        grids between the 7×7 (lossy) and 14×14 (full) reachable via
        ``dino_cam_patches``.

        Args:
            features: [N, P, D] tokens (P = patch_grid²).
            patch_grid: side length of the input square patch grid (default 14).
            target_grid: target ``(rows, cols)``. Defaults to ``(7, 7)`` (the
                legacy 2×2 pool used for RoboTwin wrists). Each dim must be in
                ``[1, patch_grid]``.
            mode: ``"avg"`` (default; the divisor/adaptive average below) or
                ``"bilinear"`` (anti-aliased bilinear resample — a better
                low-pass that retains more localized detail at non-divisor
                grids; measured to beat ``"avg"`` on DINO feature retention).

        Returns:
            [N, target_h * target_w, D] pooled tokens.
        """
        if target_grid is None:
            target_grid = (7, 7)
        target_h, target_w = target_grid
        if not (1 <= target_h <= patch_grid and 1 <= target_w <= patch_grid):
            raise ValueError(
                f"target_grid={target_grid} must have each dim in "
                f"[1, patch_grid={patch_grid}]"
            )
        if mode not in ("avg", "bilinear"):
            raise ValueError(f"pool mode must be 'avg' or 'bilinear', got {mode!r}")
        N, P, D = features.shape
        if P != patch_grid * patch_grid:
            raise ValueError(
                f"Expected P={patch_grid * patch_grid} tokens, got P={P}"
            )
        feat_2d = features.float().view(N, patch_grid, patch_grid, D).permute(0, 3, 1, 2)
        if target_h == patch_grid and target_w == patch_grid:
            pooled = feat_2d  # identity (both modes)
        elif mode == "bilinear":
            # Anti-aliased bilinear resample: a better low-pass than the box
            # average for non-divisor downsamples → retains more localized
            # detail (measured) with no ringing. Clean GxG grid (RoPE-safe).
            pooled = F.interpolate(
                feat_2d, size=(target_h, target_w),
                mode="bilinear", align_corners=False, antialias=True,
            )
        elif patch_grid % target_h == 0 and patch_grid % target_w == 0:
            # Exact divisor → strided non-overlapping avg pool (bit-equal legacy).
            kh, kw = patch_grid // target_h, patch_grid // target_w
            pooled = F.avg_pool2d(feat_2d, kernel_size=(kh, kw), stride=(kh, kw))
        else:
            # Non-divisor avg (e.g. 14→10, 14→12) → uneven/overlapping local avg.
            pooled = F.adaptive_avg_pool2d(feat_2d, output_size=(target_h, target_w))
        return pooled.permute(0, 2, 3, 1).reshape(N, -1, D).to(features.dtype)

    @staticmethod
    def _pixel_unshuffle_patches(
        features: torch.Tensor,
        patch_grid: Tuple[int, int],
        factor: int,
    ) -> torch.Tensor:
        """Lossless space-to-channel fold (the LOSSLESS alternative to pooling).

        ``[N, h*w, D]`` -> ``[N, (h/f)*(w/f), D*f*f]``. A pure reshape (inverse of
        ``pixel_shuffle``): each ``f×f`` spatial block's sub-patches move into the
        channel axis instead of being averaged, so ALL spatial detail survives
        while the token count drops ``f²×``. Unlike ``_pool_patches`` this adds no
        low-pass blur. Channel layout is ``F.pixel_unshuffle``'s ``(D, f, f)``
        interleave — order is irrelevant downstream (the embedder/proj_out are
        full Linears and the per-element DINO MSE is permutation-invariant).
        """
        N, P, D = features.shape
        h, w = patch_grid
        if P != h * w:
            raise ValueError(f"Expected P={h * w} tokens, got P={P}")
        if h % factor or w % factor:
            raise ValueError(
                f"pixel_unshuffle factor {factor} must divide grid {patch_grid}"
            )
        x = features.view(N, h, w, D).permute(0, 3, 1, 2)          # [N, D, h, w]
        x = F.pixel_unshuffle(x, factor)                            # [N, D*f*f, h/f, w/f]
        return x.permute(0, 2, 3, 1).reshape(
            N, (h // factor) * (w // factor), D * factor * factor
        )

    # No `@torch.no_grad()` so the ViT can be fine-tuned when
    # `FlexPiLatent(freeze_dino_encoder=False)`. When frozen (default), all
    # params have requires_grad=False and no autograd graph is built.
    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Run DINOv3 and extract patch tokens.

        Args:
            images: [N, 3, H, W] in [-1, 1] range

        Returns:
            [N, patches_per_frame, embed_dim]
        """
        # Resize to dino_size × dino_size. antialias=True matches torchvision's
        # `resize(antialias=True)` used by the dataset workers; without it,
        # downsamples (e.g. 256×320 head → 224×224) alias high-frequency
        # texture. No-op when upsampling (legacy 128×160 wrist → 224×224).
        if images.shape[-2] != self.dino_size or images.shape[-1] != self.dino_size:
            images = F.interpolate(
                images, size=(self.dino_size, self.dino_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        # Normalize to ImageNet stats
        images = self._normalize_imagenet(images)
        images = images.to(dtype=next(self.model.parameters()).dtype)
        # Forward
        out = self.model.forward_features(images)  # [N, prefix + patches, D]
        return out[:, self.num_prefix_tokens:]  # [N, patches, D]

    # No outer `@torch.no_grad()` — see `_extract_patches` for rationale.
    def encode_video(
        self,
        video: Optional[torch.Tensor] = None,
        concat_mode: str = "tshape_robotwin_384x320_uniform",
        inference_batch_size: int = 27,
        temporal_stride: int = 1,
        first_frame_only: bool = False,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        layout: Optional[LayoutSpec] = None,
        slot_key_map: Optional[Mapping[str, str]] = None,
        cam_patches: Optional[Sequence[Tuple[int, int]]] = None,
        pool_mode: str = "avg",
        pixel_unshuffle: int = 0,
        stride_keep_far: bool = False,
    ) -> torch.Tensor:
        """Encode raw video frames to DINO patch features.

        Temporal alignment: selects stride-4 boundary frames [0, 4, 8, ..., T-1]
        to match VAE latent frame count, then applies ``temporal_stride`` to
        control DINO density relative to VAE latent frames.

        Args:
            video: [B, 3, T, H, W] raw video in [-1, 1]
            concat_mode: camera concatenation mode ("tshape_robotwin_384x320_uniform")
            inference_batch_size: micro-batch size for DINO forward passes
            temporal_stride: DINO frame stride relative to VAE latent frames.
                1 = every latent frame (default), 2 = every other, etc.
                First frame (observation) is always included.
            first_frame_only: if True, only encode the first frame.
            stride_keep_far: with ``temporal_stride>1``, stride from the END
                so the farthest future frame is always encoded (same frame
                count). No effect at stride 1 or under ``first_frame_only``.
            per_cam: alternative to ``video``. Dict with keys ``cam_high``
                [B, 3, T, 256, 320], ``cam_left_wrist`` / ``cam_right_wrist``
                each [B, 3, T, 224, 224] in ``[-1, 1]``. Used when the sample
                dict comes from RobotVideoDataset; ``video`` is then ignored.
            cam_patches: per-cam target patch grid (one ``(H, W)`` per cam slot,
                in ``layout.cam_slots()`` order). Defaults to
                ``layout.dino_cam_patches`` when None. Pass the model's pooled
                ``self.dino_cam_patches`` (e.g. when ``dino_pool_factor>1``)
                so the encoder's output token count matches the model's RoPE
                freqs computed from the same pooled grid. The ViT runs at
                ``patches_per_side`` (14×14) per cam regardless; pooling reduces
                that to ``cam_patches[i]`` post-hoc.

        Returns:
            [B, embed_dim, F_dino, N_patches, 1]
            F_dino depends on temporal_stride and first_frame_only.
        """
        if per_cam is None and video is None:
            raise ValueError("encode_video requires either `video` or `per_cam`.")

        # Resolve layout + key map. ``layout`` kwarg takes precedence over
        # ``concat_mode``; back-compat default is the RoboTwin layout.
        if layout is None:
            layout = get_layout(concat_mode)
        kmap = layout.resolve_slot_key_map(slot_key_map)

        cam_slots = layout.cam_slots()
        if per_cam is not None:
            proto_key = kmap[cam_slots[0].key]
            if proto_key not in per_cam:
                raise ValueError(
                    f"per_cam missing {proto_key!r} (slot {cam_slots[0].key!r}); "
                    f"got {list(per_cam)!r}"
                )
            head = per_cam[proto_key]
            if head.ndim != 5 or head.shape[1] != 3:
                raise ValueError(
                    f"per_cam[{proto_key!r}] must be [B, 3, T, H, W]; got {tuple(head.shape)}"
                )
            B, _, T, _, _ = head.shape
        else:
            B, C, T, H, W = video.shape

        # 1. Compute VAE-aligned frame indices (stride-4 boundary frames)
        num_latent_frames = (T - 1) // 4 + 1  # e.g., T=33 → 9
        all_frame_indices = [min(4 * i, T - 1) for i in range(num_latent_frames)]

        # 2. Apply DINO temporal stride
        if first_frame_only:
            frame_indices = [all_frame_indices[0]]
        else:
            # Always include first frame, then stride through the rest
            # (keep_far strides from the END so the farthest frame survives;
            # must stay in lockstep with FlexPi._aux_per_frame_is_pad).
            slot_ids = select_aux_frame_slots(
                num_latent_frames, temporal_stride, keep_far=stride_keep_far,
            )
            frame_indices = [all_frame_indices[i] for i in slot_ids]
        F_dim = len(frame_indices)

        if per_cam is not None:
            per_cam_flat: Dict[str, torch.Tensor] = {}
            for slot in cam_slots:
                cam_key = kmap[slot.key]
                if cam_key not in per_cam:
                    raise ValueError(
                        f"per_cam missing {cam_key!r} (slot {slot.key!r}); got {list(per_cam)}"
                    )
                v = per_cam[cam_key]
                if v.ndim != 5 or v.shape[1] != 3:
                    raise ValueError(
                        f"per_cam[{cam_key!r}] must be [B, 3, T, H, W]; got {tuple(v.shape)}"
                    )
                if frame_indices == list(range(len(frame_indices))):
                    # Contiguous prefix (deploy first_frame_only = [0]): slice
                    # instead of list-indexing — identical values, but no CPU
                    # index-tensor upload (a stream sync; CUDA-graph-illegal).
                    v = v[:, :, :len(frame_indices)]
                else:
                    v = v[:, :, frame_indices]
                Hw, Ww = v.shape[-2:]
                # Key the flat dict by SLOT key so the encode loop is layout-agnostic.
                per_cam_flat[slot.key] = v.permute(0, 2, 1, 3, 4).reshape(B * F_dim, 3, Hw, Ww)
            return self._encode_layout(
                frames=None, B=B, F_dim=F_dim,
                inference_batch_size=inference_batch_size,
                layout=layout,
                per_cam_flat=per_cam_flat,
                cam_patches=cam_patches,
                pool_mode=pool_mode,
                pixel_unshuffle=pixel_unshuffle,
            )

        frames = video[:, :, frame_indices]  # [B, 3, F_dino, H, W]

        return self._encode_layout(
            frames, B, F_dim, inference_batch_size,
            layout=layout,
            cam_patches=cam_patches,
            pool_mode=pool_mode,
            pixel_unshuffle=pixel_unshuffle,
        )

    # No outer `@torch.no_grad()` — see `_extract_patches` for rationale.
    def encode_frames(
        self,
        frames: torch.Tensor,
        inference_batch_size: int = 32,
        layout: Optional[LayoutSpec] = None,
        cam_patches: Optional[Sequence[Tuple[int, int]]] = None,
        pool_mode: str = "avg",
    ) -> torch.Tensor:
        """Encode composite frames to DINO patch features.

        Processes individual frames without temporal/video wrapping logic.

        Args:
            frames: ``[N, 3, H, W]`` in [-1, 1] range, where (H, W) matches
                ``layout.composite_hw``.
            inference_batch_size: micro-batch size for DINO forward passes.
            layout: target ``LayoutSpec``. Defaults to RoboTwin (back-compat).

        Returns:
            ``[N, sum(h*w for h,w in layout.dino_cam_patches), 768]``.
            For RoboTwin: 196 + 49 + 49 = 294 patches per frame.
        """
        if layout is None:
            layout = get_layout("tshape_robotwin_384x320")
        cam_dict = self._split_composite_frame(frames, layout)
        cam_slots = layout.cam_slots()
        if cam_patches is None:
            cam_patches = layout.dino_cam_patches
        elif len(cam_patches) != len(cam_slots):
            raise ValueError(
                f"cam_patches length {len(cam_patches)} != cam slots "
                f"{len(cam_slots)} for layout {layout.name!r}"
            )

        all_features = []
        for slot, target_grid in zip(cam_slots, cam_patches):
            cam_images = cam_dict[slot.key]
            patches_list = []
            for i in range(0, cam_images.shape[0], inference_batch_size):
                batch = cam_images[i : i + inference_batch_size]
                patches = self._extract_patches(batch)  # [mb, patch_grid², D]
                if target_grid != (self.patches_per_side, self.patches_per_side):
                    patches = self._pool_patches(
                        patches, patch_grid=self.patches_per_side, target_grid=target_grid,
                        mode=pool_mode,
                    )
                patches_list.append(patches)
            cam_features = torch.cat(patches_list, dim=0)
            all_features.append(cam_features)
        return torch.cat(all_features, dim=1)

    def _encode_layout(
        self,
        frames: Optional[torch.Tensor],
        B: int,
        F_dim: int,
        inference_batch_size: int,
        layout: LayoutSpec,
        per_cam_flat: Optional[Dict[str, torch.Tensor]] = None,
        cam_patches: Optional[Sequence[Tuple[int, int]]] = None,
        pool_mode: str = "avg",
        pixel_unshuffle: int = 0,
    ) -> torch.Tensor:
        """Encode multi-cam frames using ``layout``.

        Args:
            frames: ``[B, 3, F, H, W]`` selected frames (composite). Must be
                provided unless ``per_cam_flat`` supplies all cams directly.
            B: batch size.
            F_dim: number of latent frames.
            inference_batch_size: micro-batch for DINO.
            layout: target ``LayoutSpec``. Determines slot geometry, per-slot
                pool target grids, and total token count.
            per_cam_flat: optional pre-flattened per-cam dict keyed by SLOT
                placeholder (e.g. ``"slot_top"`` / ``"slot_bl"``), each
                ``[B*F, 3, H, W]``. When provided, the composite-slice path
                is skipped entirely. Used by per-cam datasets.

        Returns:
            ``[B, embed_dim, F_dim, n_patches_total, 1]`` with
            ``n_patches_total = sum(h*w for h,w in layout.dino_cam_patches)``.
        """
        cam_slots = layout.cam_slots()
        if cam_patches is None:
            cam_patches = layout.dino_cam_patches
        elif len(cam_patches) != len(cam_slots):
            raise ValueError(
                f"cam_patches length {len(cam_patches)} != cam slots "
                f"{len(cam_slots)} for layout {layout.name!r}"
            )
        if per_cam_flat is not None:
            cam_inputs = [per_cam_flat[s.key] for s in cam_slots]
        else:
            H, W = layout.composite_hw
            frames_flat = frames.permute(0, 2, 1, 3, 4).reshape(B * F_dim, 3, H, W)
            split = self._split_composite_frame(frames_flat, layout)
            cam_inputs = [split[s.key] for s in cam_slots]

        all_features = []
        for cam_images, target_grid in zip(cam_inputs, cam_patches):
            patches_list = []
            for i in range(0, cam_images.shape[0], inference_batch_size):
                batch = cam_images[i : i + inference_batch_size]
                patches = self._extract_patches(batch)  # [mb, patch_grid², D]
                if target_grid != (self.patches_per_side, self.patches_per_side):
                    patches = self._pool_patches(
                        patches, patch_grid=self.patches_per_side, target_grid=target_grid,
                        mode=pool_mode,
                    )
                patches_list.append(patches)
            cam_features = torch.cat(patches_list, dim=0)
            if pixel_unshuffle:
                # Lossless f×f space-to-channel fold of THIS view's grid (no pool;
                # cam_patches here is the native pre-fold grid). Views stay on the
                # token axis below, each folded view -> (h/f)*(w/f) tokens of D*f².
                cam_features = self._pixel_unshuffle_patches(
                    cam_features, patch_grid=target_grid, factor=pixel_unshuffle
                )
            all_features.append(cam_features)

        features = torch.cat(all_features, dim=1)  # [B*F, N_total, D]
        n_patches_total = features.shape[1]
        # ``features.shape[2]`` (not ``self.embed_dim``) so pixel_unshuffle's
        # widened D*f² flows through; bit-equal otherwise (== embed_dim).
        feat_dim = features.shape[2]
        features = features.view(B, F_dim, n_patches_total, feat_dim)
        features = features.permute(0, 3, 1, 2).unsqueeze(-1)  # [B, D(or V*D), F, N, 1]
        return features

    # Back-compat alias for legacy callers (kept thin so anyone importing
    # ``_encode_robotwin`` still works during the migration).
    def _encode_robotwin(
        self,
        frames: Optional[torch.Tensor],
        B: int,
        F_dim: int,
        inference_batch_size: int,
        per_cam_flat: Optional[Dict[str, torch.Tensor]] = None,
        cam_patches: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        layout = get_layout("tshape_robotwin_384x320_uniform")
        # The legacy per_cam_flat keyed by dataset cam keys. Translate to the
        # new slot-keyed format using the layout's default key map.
        if per_cam_flat is not None:
            kmap = layout.default_slot_key_map
            per_cam_flat = {s.key: per_cam_flat[kmap[s.key]] for s in layout.cam_slots()}
        return self._encode_layout(
            frames=frames, B=B, F_dim=F_dim,
            inference_batch_size=inference_batch_size, layout=layout,
            per_cam_flat=per_cam_flat,
            cam_patches=cam_patches,
        )
