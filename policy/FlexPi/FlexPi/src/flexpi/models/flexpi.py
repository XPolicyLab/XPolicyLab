"""FlexPi — the unified 4-stream flex world-action model.

One model, all eight inference regimes. Integrates what were previously
``FlexPi`` (video + DINO + pointmap + action co-training) and
``FlexPi`` (per-stream joint flags + flex-joint per-sample
randomization) into a single :class:`FlexPi` class over the flexpi-owned
:class:`~flexpi.models.backbone.FlexPiBackbone`.

The seven ``FlexPi`` methods that the joint layer overrode *and*
delegated back into are preserved as ``_base_*`` so the two-layer behavior
is bit-identical after the flatten.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from flexpi.utils.logging_config import get_logger

from .dino_encoder import DinoEncoder
from .helpers.dino import (  # noqa: F401
    _DINO_X0_SIGMA_MIN,
    _dino_x0_to_velocity,
    _zero_init_dino_x0_head_,
    compute_dino_freqs,
)
from .mot import MoT
from .pointmap_encoder import PointmapEncoder
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)


from typing import Dict, Optional, Sequence, Tuple

from flexpi.utils.pytorch_utils import _staged_randn

from .backbone import FlexPiBackbone
from .helpers.flex_joint import FlexJointConfig, sample_flex_batch_flags
from .inference_opt.step_skip import StepSkipController, resolve_sim_sources


class FlexPi(FlexPiBackbone):
    """FlexPi co-training video + DINO + pointmap + action.

    Action attends only to first-frame anchors of video / DINO / pointmap.
    Future latents of all three stream types are co-denoised with their
    standard flow-matching losses. For configurable future-latent visibility
    at the action head, use :class:`FlexPi`.

    Joint flags
    -----------

    FlexPi with configurable action→future-latent attention.
    """

    FROZEN_MODULES: set[str] = {
        "vae", "text_encoder", "dino_encoder", "pointmap_encoder",
    }

    def _base_init(
        self,
        video_expert,
        action_expert,
        mot: MoT,
        vae,
        dino_encoder: DinoEncoder,
        pointmap_encoder: PointmapEncoder,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        # DINO-specific
        dino_dim: int = 768,
        dino_train_shift: float = 5.0,
        dino_infer_shift: float = 5.0,
        dino_num_train_timesteps: int = 1000,
        loss_lambda_dino: float = 1.0,
        dino_cam_regions: list | None = None,
        dino_cam_patches: list | None = None,
        dino_temporal_stride: int = 1,
        dino_stride_keep_far: bool = False,
        dino_pred_x0: bool = False,
        dino_pool_mode: str = "avg",
        freeze_dino_encoder: bool = True,
        dino_pixel_unshuffle: int = 0,
        # Pointmap-specific
        pointmap_train_shift: float = 5.0,
        pointmap_infer_shift: float = 5.0,
        pointmap_num_train_timesteps: int = 1000,
        loss_lambda_pointmap: float = 1.0,
        pointmap_depth_vis_mode: str = "turbo",
        # Composite layout (forwarded to FlexPi). Defaults to RoboTwin so
        # existing checkpoints / configs keep working unchanged.
        composite_layout=None,
        composite_layout_slot_key_map=None,
        dino_pool_factor: int = 1,
    ):
        # Resolve composite layout default early so we can derive default
        # dino_cam_regions from it. Falls back to
        # RoboTwin (same as today's hardcoded defaults).
        from flexpi.composite_layouts import get_layout
        _layout_resolved = get_layout(composite_layout)

        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            composite_layout=composite_layout,
            composite_layout_slot_key_map=composite_layout_slot_key_map,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )

        # --- DINO state ---
        self.dino_encoder = dino_encoder
        self.dino_dim = int(dino_dim)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.dino_temporal_stride = int(dino_temporal_stride)
        self.dino_stride_keep_far = bool(dino_stride_keep_far)
        self.dino_pred_x0 = bool(dino_pred_x0)
        if self.dino_pred_x0:
            logger.info(
                "FlexPi: dino_pred_x0=True — DINO head output is x̂0; "
                "v̂ = (x_t − x̂0)/σ is derived analytically for loss and inference."
            )
        self.dino_pool_mode = str(dino_pool_mode)
        # Defaults come from the resolved layout, optionally pooled. Asymmetric
        # layouts cannot be uniformly pooled — with_dino_pool raises.
        _grid_factor = dino_pool_factor if dino_pool_factor != 1 else max(int(dino_pixel_unshuffle), 1)
        if dino_cam_patches is None and dino_cam_regions is None and _grid_factor != 1:
            patches, regions, _ = _layout_resolved.with_dino_pool(_grid_factor)
            self.dino_cam_patches = [tuple(p) for p in patches]
            self.dino_cam_regions = [tuple(r) for r in regions]
        else:
            self.dino_cam_regions = dino_cam_regions or [tuple(r) for r in _layout_resolved.dino_cam_regions]
            self.dino_cam_patches = dino_cam_patches or [tuple(p) for p in _layout_resolved.dino_cam_patches]

        # Default DINO layout scaffolding.
        self._dino_per_view_dim = self.dino_dim
        self._dino_encoder_cam_patches = None

        # ``self.dino_cam_patches`` is the post-fold grid; the encoder runs at
        # post-fold × f and folds back to it.
        self._dino_pixel_unshuffle = int(dino_pixel_unshuffle)
        if self._dino_pixel_unshuffle:
            f = self._dino_pixel_unshuffle
            self._dino_encoder_cam_patches = [
                (h * f, w * f) for (h, w) in self.dino_cam_patches
            ]
            # LOSSLESS only when the fold runs on the NATIVE ViT grid (no pool
            # first): post-fold × f must equal patches_per_side. Otherwise the
            # encoder would avg-pool 14→native then fold → silently lossy,
            # defeating the whole point. Direct the user to the right grid.
            ps = int(self.dino_encoder.patches_per_side)
            if any(g != (ps, ps) for g in self._dino_encoder_cam_patches):
                raise ValueError(
                    f"dino_pixel_unshuffle={f} is lossless only when each view's "
                    f"post-fold grid × {f} equals the ViT native grid {ps}×{ps} "
                    f"(no pooling before the fold); got encoder grids "
                    f"{self._dino_encoder_cam_patches}. Use a composite_layout "
                    f"whose per-cam DINO grid is a uniform {ps // f}×{ps // f} "
                    f"(e.g. tshape_robotwin_384x320_uniform) so each view folds {ps}×{ps} → "
                    f"{ps // f}×{ps // f}, or set dino_pixel_unshuffle=0."
                )
            self.dino_dim = (f * f) * int(dino_dim)
            self._dino_per_view_dim = self.dino_dim  # joint LayerNorm over f²*768

        # --- Pointmap state ---
        # The pointmap has no patch-grid config of its own: it goes through the
        # WAN VAE on a composite built exactly like the RGB one, so its RoPE
        # positions come from the encoded latent shape (see
        # ``_compute_pointmap_freqs``).
        self.pointmap_encoder = pointmap_encoder
        self.loss_lambda_pointmap = float(loss_lambda_pointmap)
        self.pointmap_depth_vis_mode = str(pointmap_depth_vis_mode)
        self._camera_intrinsics: Optional[torch.Tensor] = None

        # Per-batch flex-joint state. Set by ``FlexPi.training_loss``
        # when ``flex_joint.enabled=True``; consumed by the token-zeroing /
        # loss-masking hooks in this class and the per-sample mask builder in
        # the joint subclass. ``None`` => bit-identical legacy behavior.
        self._batch_flex = None

        # Instance-level FROZEN_MODULES. The DINO encoder can be made trainable
        # via freeze_dino_encoder; the pointmap encoder is pure
        # unproject/normalize arithmetic (no parameters) and is always frozen.
        frozen = {"vae", "text_encoder", "pointmap_encoder"}
        if freeze_dino_encoder:
            frozen.add("dino_encoder")
        self.FROZEN_MODULES = frozen

        video_hidden_dim = self.video_expert.hidden_dim

        # --- DINO embedding layers ---
        # LayerNorm on the raw DINO features at the Linear-embed input.
        self.dino_feature_norm = nn.LayerNorm(self._dino_per_view_dim)
        self.dino_embedder = nn.Linear(self.dino_dim, video_hidden_dim)
        self.dino_proj_out = nn.Linear(video_hidden_dim, self.dino_dim)
        self._init_dino_layers()

        # --- Pointmap embedding layers ---
        self._build_pointmap_layers()

        # --- Schedulers ---
        self.train_dino_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=dino_num_train_timesteps, shift=dino_train_shift,
        )
        self.infer_dino_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=dino_num_train_timesteps, shift=dino_infer_shift,
        )
        self.train_pointmap_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=pointmap_num_train_timesteps, shift=pointmap_train_shift,
        )
        self.infer_pointmap_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=pointmap_num_train_timesteps, shift=pointmap_infer_shift,
        )

        # Parent's self.to() ran before these layers existed; re-apply.
        self.to(device=self.device, dtype=self.torch_dtype)

        logger.info(
            "FlexPi: dino_temporal_stride=%d dino_stride_keep_far=%s",
            self.dino_temporal_stride, self.dino_stride_keep_far,
        )

    # ------------------------------------------------------------------
    # Layer init helpers (mirror FlexPiLatent / FlexPi3D)
    # ------------------------------------------------------------------

    def _init_dino_layers(self):
        def _xavier(module):  # Linear, or every Linear inside an MLP Sequential
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        _xavier(self.dino_embedder)
        _xavier(self.dino_proj_out)
        if self.dino_pred_x0:
            # Zero-init the head's final layer so x̂0 ≈ 0 at step 0.
            _zero_init_dino_x0_head_(self.dino_proj_out)
        nn.init.ones_(self.dino_feature_norm.weight)
        nn.init.zeros_(self.dino_feature_norm.bias)

    def _build_pointmap_layers(self):
        self.pt_patch_embedding = copy.deepcopy(self.video_expert.patch_embedding)
        self.pt_head = copy.deepcopy(self.video_expert.head)

    @property
    def _pointmap_globally_off(self) -> bool:
        """True when the pointmap stream can never contribute and is safe to
        skip end-to-end (depth dataload + VAE encode + tokens + MoT + loss).

        Two independent ways in:

        1. ``enable_pointmap=False`` — the run declares no pointmap stream,
           either because the dataset has no depth or as a deliberate ablation.
        2. Every flex path that could activate pointmap is off:
           ``p_present_pointmap == 0`` (never present), ``p_jp == 0`` (never
           joined), and ``cross_modal_predict_pointmap == False`` (absent never
           denoised from other streams). Kept because it is still *true* — and
           because checkpoints predating ``enable_pointmap`` express the intent
           this way, so dropping it would change how they load.

        Under either, pointmap tokens would otherwise be built, zeroed, and
        forwarded as pure dead weight every step.
        """
        if not bool(getattr(self, "enable_pointmap", True)):
            return True
        fj = getattr(self, "flex_joint", None)
        if fj is None or not getattr(fj, "enabled", False):
            return False
        return (
            float(fj.p_present_pointmap) == 0.0
            and float(fj.p_jp) == 0.0
            and not bool(fj.cross_modal_predict_pointmap)
        )

    def _configure_pointmap_off(self) -> None:
        """Apply the pointmap-globally-off setup so train and deploy match.

        Call after ``flex_joint`` and the ``_infer_present_*`` defaults are
        finalized. No-op when pointmap is active. Four effects:

        1. Freeze the pointmap head modules (via FROZEN_MODULES) — with the
           stream skipped end-to-end they receive no gradient, so keep them out
           of the trainable set rather than handing the optimizer params that
           never update.
        2. Default the inference presence flag to absent (``_infer_present_p =
           False``). Training drops pointmap entirely, so deployment must too —
           otherwise ``infer_action`` would default to present_p=True and
           re-enable the ff_v↔ff_p / ff_d↔ff_p anchor attention the model never
           saw in training. An explicit ``present_pointmap=`` at call time still
           overrides this default.
        3. Clear the trained joint default (``joint_pointmap = False``). It
           gates the dispatch — ``any_joint = joint_video or joint_dino or
           joint_pointmap`` — so leaving it True would send a caller who asked
           for action-only (video+dino off, pointmap unspecified) down the
           heavy joint path to denoise a stream with no tokens.
        4. Clear ``cross_modal_predict_pointmap``. ``sample_flex_batch_flags``
           already forces ``cm_p`` False under pointmap-off, but that forcing
           is training-side only — the inference mask builder reads the config
           field raw. Left True it would keep rem_p rows alive on the joint
           action path, which still carries Sp>0 placeholder tokens, so the
           action would attend a stream training never gave it.
        """
        if not self._pointmap_globally_off:
            return
        self.FROZEN_MODULES = set(self.FROZEN_MODULES) | {
            "pt_patch_embedding", "pt_head",
        }
        if hasattr(self, "_infer_present_p"):
            self._infer_present_p = False
        self.joint_pointmap = False
        self.flex_joint.cross_modal_predict_pointmap = False
        reason = (
            "enable_pointmap=False" if not bool(getattr(self, "enable_pointmap", True))
            else "p_present_pointmap=0, p_jp=0, cross_modal_predict_pointmap=False"
        )
        logger.info(
            "Pointmap stream GLOBALLY OFF (%s): skipping depth encode + "
            "pointmap tokens + loss in training; inference defaults to "
            "present_pointmap=False. Pointmap head modules frozen.",
            reason,
        )

    # ------------------------------------------------------------------
    # DINO helpers
    # ------------------------------------------------------------------

    def _embed_dino(self, dino_features: torch.Tensor) -> torch.Tensor:
        x = dino_features.squeeze(-1).permute(0, 2, 3, 1).reshape(
            dino_features.shape[0], -1, self.dino_dim
        )
        x = self.dino_feature_norm(x)
        return self.dino_embedder(x)

    def _project_dino_out(self, dino_tokens_out: torch.Tensor) -> torch.Tensor:
        """Project trunk DINO tokens [B, Sd, H] -> DINO feature space [B, Sd, dino_dim]."""
        return self.dino_proj_out(dino_tokens_out)

    def _dino_encode_kwargs(self) -> dict:
        """DINO encoder kwargs. Under the pixel-unshuffle fold the encoder runs
        at the native pre-fold grid (post-fold × f) and folds back to the model
        RoPE grid (``self.dino_cam_patches``)."""
        out = super()._dino_encode_kwargs()
        if self._dino_pixel_unshuffle:
            # Encoder runs at the native pre-fold grid (post-fold × f) and folds.
            out["cam_patches"] = self._dino_encoder_cam_patches
            out["pixel_unshuffle"] = self._dino_pixel_unshuffle
        if self.dino_stride_keep_far:
            out["stride_keep_far"] = True
        return out

    def _glue_memo(self, key, build):
        """Memoize per-call-constant inference glue (see ``glue_cache`` in
        ``prepare_for_inference``). Bypassed unless the knob is on AND the
        model is in eval mode; hits are bit-identical (pure shape functions)."""
        if not (getattr(self, "_glue_cache_enabled", False) and not self.training):
            return build()
        hit = self._glue_cache.get(key)
        if hit is None:
            hit = self._glue_cache[key] = build()
        return hit

    def _compute_dino_freqs(self, num_frames: int, device: torch.device) -> torch.Tensor:
        return self._glue_memo(
            ("dino_freqs", int(num_frames), str(device)),
            lambda: compute_dino_freqs(
                freqs_3d=self.video_expert.freqs,
                n_frames=num_frames,
                cam_regions=self.dino_cam_regions,
                cam_patches=self.dino_cam_patches,
                device=device,
            ),
        )

    @torch.no_grad()
    def _encode_first_frame_dino(
        self,
        input_image: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """First-frame DINO features, embedded into video hidden dim."""
        if per_cam is None and input_image is None:
            raise ValueError(
                "_encode_first_frame_dino requires either `input_image` or `per_cam`."
            )
        if per_cam is not None:
            per_cam_5d: Dict[str, torch.Tensor] = {}
            for k, v in per_cam.items():
                v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                if v.ndim == 4:
                    v = v.unsqueeze(2)
                per_cam_5d[k] = v
            dino_feat = self.dino_encoder.encode_video(
                video=None, per_cam=per_cam_5d, concat_mode="robotwin", **self._dino_encode_kwargs(),
                first_frame_only=True,
            )
        else:
            video_single = input_image.unsqueeze(2)
            dino_feat = self.dino_encoder.encode_video(
                video_single, concat_mode="robotwin", **self._dino_encode_kwargs(), first_frame_only=True,
            )
        return self._embed_dino(dino_feat)

    # ------------------------------------------------------------------
    # Pointmap helpers (mirror FlexPi3D)
    # ------------------------------------------------------------------

    def _embed_pointmap(self, pointmap_raw: torch.Tensor):
        x = self.pt_patch_embedding(pointmap_raw)
        f, h, w = x.shape[2:]
        tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        ptpf = h * w
        pt_meta = {"grid_size": (int(f), int(h), int(w))}
        return tokens, ptpf, pt_meta

    def _project_pointmap_out(
        self, pt_tokens_out: torch.Tensor, pt_time: torch.Tensor, pt_meta: dict,
    ) -> torch.Tensor:
        f, h, w = pt_meta["grid_size"]
        x = self.pt_head(pt_tokens_out, pt_time)
        ps = self.video_expert.patch_size
        return rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w, x=ps[0], y=ps[1], z=ps[2],
        )

    def _compute_pointmap_freqs(
        self,
        num_frames: int,
        device: torch.device,
        pt_meta: Optional[dict] = None,
        start_frame_idx: int = 0,
    ) -> torch.Tensor:
        grid = tuple(pt_meta["grid_size"]) if (pt_meta is not None and "grid_size" in pt_meta) else None
        return self._glue_memo(
            ("pt_freqs", int(num_frames), str(device), grid, int(start_frame_idx)),
            lambda: self._compute_pointmap_freqs_impl(
                num_frames, device, pt_meta=pt_meta, start_frame_idx=start_frame_idx,
            ),
        )

    def _compute_pointmap_freqs_impl(
        self,
        num_frames: int,
        device: torch.device,
        pt_meta: Optional[dict] = None,
        start_frame_idx: int = 0,
    ) -> torch.Tensor:
        assert pt_meta is not None, "pointmap freqs require pt_meta with grid_size"
        f, h, w = pt_meta["grid_size"]
        freqs_3d = self.video_expert.freqs
        t_lo, t_hi = start_frame_idx, start_frame_idx + f
        return torch.cat([
            freqs_3d[0][t_lo:t_hi].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)

    def set_camera_intrinsics(self, camera_intrinsics: torch.Tensor) -> None:
        if camera_intrinsics.ndim == 4:
            camera_intrinsics = camera_intrinsics[0]
        self._camera_intrinsics = camera_intrinsics.to(device=self.device)

    def _resolve_camera_intrinsics(
        self, explicit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = explicit if explicit is not None else self._camera_intrinsics
        if K is None:
            raise RuntimeError(
                "No camera intrinsics available. Attach them per-sample via "
                "`sample['camera_intrinsics']` or call "
                "`model.set_camera_intrinsics(K)` before inference."
            )
        return K

    @torch.no_grad()
    def _encode_first_frame_pointmap_raw(
        self,
        camera_intrinsics: Optional[torch.Tensor] = None,
        tiled: bool = False,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if per_cam_depth is None:
            raise ValueError(
                "_encode_first_frame_pointmap_raw requires `per_cam_depth`."
            )
        K = self._resolve_camera_intrinsics(camera_intrinsics)
        per_cam_depth_4d: Dict[str, torch.Tensor] = {}
        for k, v in per_cam_depth.items():
            v = v.to(device=self.device, non_blocking=True)
            if v.ndim == 2:
                v = v.unsqueeze(0).unsqueeze(0)
            elif v.ndim == 3:
                v = v.unsqueeze(1)
            elif v.ndim != 4:
                raise ValueError(
                    f"per_cam_depth['{k}'] must be 2/3/4-d; got {tuple(v.shape)}"
                )
            per_cam_depth_4d[k] = v

        comp = self.pointmap_encoder.encode_composite(
            per_cam_depth=per_cam_depth_4d,
            camera_intrinsics=K,
            concat_mode="robotwin", **self._layout_kwargs(),
            first_frame_only=True,
        )
        return self._encode_video_latents(comp, tiled=tiled)

    # ------------------------------------------------------------------
    # Attention mask — 4-stream layout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_hbridge_self_masks_unified(
        self,
        video_seq_len: int,
        action_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        """Per-sub-stream self-masks for the joint [V||D||P||A] sequence.

        Empty sub-streams (length 0) are omitted — happens in UCC modes where
        pointmap collapses into V or D.
        """
        if not self.mot.hbridge_enabled:
            return None, None
        Sv, Sa, Sd, Sp = video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len
        tpf, dtpf, ptpf = video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame

        def _build():
            lens: list[int] = []
            masks: list[torch.Tensor] = []
            if Sv > 0:
                lens.append(Sv)
                masks.append(self.video_expert.build_video_to_video_mask(Sv, tpf, device))
            if Sd > 0:
                d_self = torch.ones((Sd, Sd), dtype=torch.bool, device=device)
                if dtpf < Sd:
                    d_self[:dtpf, dtpf:] = False
                lens.append(Sd)
                masks.append(d_self)
            if Sp > 0:
                p_self = torch.ones((Sp, Sp), dtype=torch.bool, device=device)
                if ptpf < Sp:
                    p_self[:ptpf, ptpf:] = False
                lens.append(Sp)
                masks.append(p_self)
            if Sa > 0:
                lens.append(Sa)
                masks.append(torch.ones((Sa, Sa), dtype=torch.bool, device=device))
            return lens, masks

        return self._glue_memo(
            ("hbridge", Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, str(device)), _build,
        )

    @torch.no_grad()
    def _build_video_hbridge_self_masks_unified(
        self,
        video_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        """Per-sub-stream self-masks for the video-only prefill sequence."""
        if not self.mot.hbridge_enabled:
            return None, None
        Sv, Sd, Sp = video_seq_len, dino_seq_len, pointmap_seq_len
        tpf, dtpf, ptpf = video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame
        lens: list[int] = []
        masks: list[torch.Tensor] = []
        if Sv > 0:
            lens.append(Sv)
            masks.append(self.video_expert.build_video_to_video_mask(Sv, tpf, device))
        if Sd > 0:
            d_self = torch.ones((Sd, Sd), dtype=torch.bool, device=device)
            if dtpf < Sd:
                d_self[:dtpf, dtpf:] = False
            lens.append(Sd)
            masks.append(d_self)
        if Sp > 0:
            p_self = torch.ones((Sp, Sp), dtype=torch.bool, device=device)
            if ptpf < Sp:
                p_self[:ptpf, ptpf:] = False
            lens.append(Sp)
            masks.append(p_self)
        return lens, masks

    @torch.no_grad()
    def _base_build_mot_attention_mask_unified(
        self,
        video_seq_len: int,
        action_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the 2D boolean attention mask for the 4-stream MoT layout.

        Sequence layout::

            [ ff_v(tpf) | rem_v(Sv-tpf) |
              ff_d(dtpf) | rem_d(Sd-dtpf) |
              ff_p(ptpf) | rem_p(Sp-ptpf) |
              action(Sa) ]

        Rules (direct superset of FlexPiLatent + FlexPi3D):

          - Video↔video: parent ``video_expert.build_video_to_video_mask``.
          - DINO↔DINO, pointmap↔pointmap: full block; ff rows cannot see rem.
          - First-frame anchors (ff_v, ff_d, ff_p) are pairwise bidirectional.
          - rem_v / rem_d / rem_p cross-attend to every other modality's
            entire stream (both ff and rem). This is the baseline mask; the
            Joint subclass overrides the pairwise rem↔rem cells.
          - action block self-attends and attends to all three ff_* anchors.
        """
        Sv, Sa, Sd, Sp = video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len
        tpf, dtpf, ptpf = (
            video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
        )
        total = Sv + Sd + Sp + Sa
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # Offsets
        v0, v1 = 0, Sv
        d0, d1 = Sv, Sv + Sd
        p0, p1 = Sv + Sd, Sv + Sd + Sp
        a0, a1 = Sv + Sd + Sp, total

        # --- Self blocks ---
        mask[v0:v1, v0:v1] = self.video_expert.build_video_to_video_mask(Sv, tpf, device)

        mask[d0:d1, d0:d1] = True
        if dtpf < Sd:
            mask[d0:d0 + dtpf, d0 + dtpf:d1] = False  # ff_d can't see rem_d

        mask[p0:p1, p0:p1] = True
        if ptpf < Sp:
            mask[p0:p0 + ptpf, p0 + ptpf:p1] = False  # ff_p can't see rem_p

        # --- First-frame anchors: pairwise bidirectional ---
        mask[v0:v0 + tpf, d0:d0 + dtpf] = True
        mask[d0:d0 + dtpf, v0:v0 + tpf] = True
        mask[v0:v0 + tpf, p0:p0 + ptpf] = True
        mask[p0:p0 + ptpf, v0:v0 + tpf] = True
        mask[d0:d0 + dtpf, p0:p0 + ptpf] = True
        mask[p0:p0 + ptpf, d0:d0 + dtpf] = True

        # --- Remaining tokens cross-attend (baseline mask) ---
        if tpf < Sv:
            mask[tpf:Sv, d0:d1] = True                 # rem_v -> all DINO
            mask[tpf:Sv, p0:p1] = True                 # rem_v -> all pointmap
        if dtpf < Sd:
            mask[d0 + dtpf:d1, v0:v1] = True           # rem_d -> all video
            mask[d0 + dtpf:d1, p0:p1] = True           # rem_d -> all pointmap
        if ptpf < Sp:
            mask[p0 + ptpf:p1, v0:v1] = True           # rem_p -> all video
            mask[p0 + ptpf:p1, d0:d1] = True           # rem_p -> all DINO

        # --- Action block ---
        mask[a0:a1, a0:a1] = True
        mask[a0:a1, v0:v0 + tpf] = True                # action -> ff_v
        mask[a0:a1, d0:d0 + dtpf] = True               # action -> ff_d
        mask[a0:a1, p0:p0 + ptpf] = True               # action -> ff_p

        return mask

    @torch.no_grad()
    def _build_video_hbridge_self_masks_dino_pointmap_only(
        self,
        dino_seq_len: int,
        pointmap_seq_len: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        """Per-sub-stream self-masks for the prefill ``[D||P]`` sequence."""
        if not self.mot.hbridge_enabled:
            return None, None
        Sd, Sp = dino_seq_len, pointmap_seq_len
        dtpf, ptpf = dino_tokens_per_frame, pointmap_tokens_per_frame
        lens: list[int] = []
        masks: list[torch.Tensor] = []
        if Sd > 0:
            d_self = torch.ones((Sd, Sd), dtype=torch.bool, device=device)
            if dtpf < Sd:
                d_self[:dtpf, dtpf:] = False
            lens.append(Sd)
            masks.append(d_self)
        if Sp > 0:
            p_self = torch.ones((Sp, Sp), dtype=torch.bool, device=device)
            if ptpf < Sp:
                p_self[:ptpf, ptpf:] = False
            lens.append(Sp)
            masks.append(p_self)
        return lens, masks

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def _base_from_wan22_pretrained(
        cls,
        # DINO params
        dino_dim: int = 768,
        dino_model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        dino_train_shift: float = 5.0,
        dino_infer_shift: float = 5.0,
        dino_num_train_timesteps: int = 1000,
        loss_lambda_dino: float = 1.0,
        dino_cam_regions: list | None = None,
        dino_cam_patches: list | None = None,
        dino_temporal_stride: int = 1,
        dino_stride_keep_far: bool = False,
        dino_pred_x0: bool = False,
        dino_pool_mode: str = "avg",
        freeze_dino_encoder: bool = True,
        dino_pixel_unshuffle: int = 0,
        # Pointmap params
        pointmap_norm_bounds: dict | None = None,
        pointmap_max_depth_m: float = 2.0,
        pointmap_train_shift: float = 5.0,
        pointmap_infer_shift: float = 5.0,
        pointmap_num_train_timesteps: int = 1000,
        loss_lambda_pointmap: float = 1.0,
        pointmap_depth_vis_mode: str = "turbo",
        **kwargs,
    ):
        from .action_dit import ActionDiT
        from .helpers.loader import load_wan22_ti2v_5b_components

        video_dit_config = kwargs.get("video_dit_config")
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")

        components = load_wan22_ti2v_5b_components(
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            model_id=kwargs.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
            tokenizer_model_id=kwargs.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
            tokenizer_max_len=kwargs.get("tokenizer_max_len", 512),
            redirect_common_files=kwargs.get("redirect_common_files", True),
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=kwargs.get("skip_dit_load_from_pretrain", False),
            load_text_encoder=kwargs.get("load_text_encoder", True),
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=kwargs.get("action_dit_config"),
            action_dit_pretrained_path=kwargs.get("action_dit_pretrained_path"),
            skip_dit_load_from_pretrain=kwargs.get("skip_dit_load_from_pretrain", False),
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
        )

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=kwargs.get("mot_checkpoint_mixed_attn", True),
            hbridge_enabled=kwargs.get("hbridge_enabled", False),
            hbridge_bottom_ratio=kwargs.get("hbridge_bottom_ratio", 0.25),
            hbridge_top_ratio=kwargs.get("hbridge_top_ratio", 0.25),
        )

        dino_encoder = DinoEncoder(
            model_name=dino_model_name,
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
        )
        pointmap_encoder = PointmapEncoder(
            norm_bounds=pointmap_norm_bounds,
            max_depth_m=pointmap_max_depth_m,
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            dino_encoder=dino_encoder,
            pointmap_encoder=pointmap_encoder,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=kwargs.get("proprio_dim"),
            device=kwargs.get("device", "cuda"),
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            video_train_shift=kwargs.get("video_train_shift", 5.0),
            video_infer_shift=kwargs.get("video_infer_shift", 5.0),
            video_num_train_timesteps=kwargs.get("video_num_train_timesteps", 1000),
            action_train_shift=kwargs.get("action_train_shift", 5.0),
            action_infer_shift=kwargs.get("action_infer_shift", 5.0),
            action_num_train_timesteps=kwargs.get("action_num_train_timesteps", 1000),
            loss_lambda_video=kwargs.get("loss_lambda_video", 1.0),
            loss_lambda_action=kwargs.get("loss_lambda_action", 1.0),
            dino_dim=dino_dim,
            dino_train_shift=dino_train_shift,
            dino_infer_shift=dino_infer_shift,
            dino_num_train_timesteps=dino_num_train_timesteps,
            loss_lambda_dino=loss_lambda_dino,
            dino_cam_regions=dino_cam_regions,
            dino_cam_patches=dino_cam_patches,
            dino_temporal_stride=dino_temporal_stride,
            dino_stride_keep_far=dino_stride_keep_far,
            dino_pred_x0=dino_pred_x0,
            dino_pool_mode=dino_pool_mode,
            freeze_dino_encoder=freeze_dino_encoder,
            dino_pixel_unshuffle=dino_pixel_unshuffle,
            pointmap_train_shift=pointmap_train_shift,
            pointmap_infer_shift=pointmap_infer_shift,
            pointmap_num_train_timesteps=pointmap_num_train_timesteps,
            loss_lambda_pointmap=loss_lambda_pointmap,
            pointmap_depth_vis_mode=pointmap_depth_vis_mode,
            composite_layout=kwargs.get("composite_layout"),
            composite_layout_slot_key_map=kwargs.get("composite_layout_slot_key_map"),
            dino_pool_factor=int(kwargs.get("dino_pool_factor", 1)),
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if kwargs.get("skip_dit_load_from_pretrain", False)
                else kwargs.get("action_dit_pretrained_path")
            ),
            "dino_encoder": dino_model_name,
        }
        return model

    # ------------------------------------------------------------------
    # Build inputs — extend parent with DINO + pointmap
    # ------------------------------------------------------------------

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)

        # --- DINO features (encoded on the fly from per_cam / video) ---
        per_cam = None
        if "per_cam" in sample:
            per_cam = {
                k: v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                for k, v in sample["per_cam"].items()
            }
        video = None
        if per_cam is None:
            video = sample["video"].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True,
            )
        with torch.no_grad():
            dino_features = self.dino_encoder.encode_video(
                video=video, concat_mode="robotwin", **self._dino_encode_kwargs(),
                temporal_stride=self.dino_temporal_stride,
                per_cam=per_cam,
            )
        inputs["dino_features"] = dino_features

        # --- Pointmap (requires camera_intrinsics + per_cam_depth in sample) ---
        # When pointmap is globally off, skip the depth dataload + VAE encode
        # entirely — no stream consumer exists this run.
        if self._pointmap_globally_off:
            inputs["pointmap_raw"] = None
            return inputs
        if "camera_intrinsics" not in sample:
            raise KeyError(
                "sample['camera_intrinsics'] is missing. Use the depth dataset "
                "(RobotVideoDataset) with meta/camera_intrinsics.json."
            )
        if "per_cam_depth" not in sample:
            raise KeyError(
                "sample['per_cam_depth'] is missing. FlexPi requires the "
                "depth dataset: a data config declaring shape_meta.depth, e.g. "
                "configs/data/robotwin.yaml."
            )
        K = sample["camera_intrinsics"].to(device=self.device)
        self.set_camera_intrinsics(K)
        per_cam_depth = {
            k: v.to(device=self.device, non_blocking=True)
            for k, v in sample["per_cam_depth"].items()
        }
        with torch.no_grad():
            composite = self.pointmap_encoder.encode_composite(
                per_cam_depth=per_cam_depth,
                camera_intrinsics=K,
                concat_mode="robotwin", **self._layout_kwargs(),
            )
            pointmap_raw = self._encode_video_latents(composite, tiled=tiled)
        inputs["pointmap_raw"] = pointmap_raw
        return inputs

    # ------------------------------------------------------------------
    # Merge auxiliaries into video stream
    # ------------------------------------------------------------------

    def _merge_aux_into_video_stream(
        self,
        video_pre: dict,
        dino_tokens: torch.Tensor,
        dino_t_mod: torch.Tensor,
        dino_freqs: torch.Tensor,
        pointmap_tokens: torch.Tensor,
        pointmap_t_mod: torch.Tensor,
        pointmap_freqs: torch.Tensor,
    ):
        """Concatenate aux streams into the video stream.

        Concats DINO + pointmap tokens after the video tokens.

        Returns ``(merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask)``.
        """
        # --- Concat present streams ---
        # ``dino_tokens=None`` or ``pointmap_tokens=None`` mean that stream is
        # absent (runtime encoder-skip via ``present_dino=False`` / trained-time
        # Order is always video → dino
        # → pointmap among the streams that ARE present.
        parts_tokens = [video_pre["tokens"]]
        parts_freqs  = [video_pre["freqs"]]
        parts_t_mod  = [video_pre["t_mod"]]
        aux_extra = 0
        if dino_tokens is not None:
            parts_tokens.append(dino_tokens)
            parts_freqs.append(dino_freqs)
            parts_t_mod.append(dino_t_mod)
            aux_extra += dino_tokens.shape[1]
        if pointmap_tokens is not None:
            parts_tokens.append(pointmap_tokens)
            parts_freqs.append(pointmap_freqs)
            parts_t_mod.append(pointmap_t_mod)
            aux_extra += pointmap_tokens.shape[1]
        merged_tokens = torch.cat(parts_tokens, dim=1)
        merged_freqs  = torch.cat(parts_freqs, dim=0)
        merged_t_mod  = torch.cat(parts_t_mod, dim=1)

        video_ctx_mask = video_pre["context_mask"]
        if video_ctx_mask.ndim == 3 and aux_extra > 0:
            aux_ctx_mask = video_ctx_mask[:, :1, :].expand(-1, aux_extra, -1)
            merged_ctx_mask = torch.cat([video_ctx_mask, aux_ctx_mask], dim=1)
        else:
            merged_ctx_mask = video_ctx_mask

        return merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask

    def _build_stream_t_mod(
        self, timestep_per_token: torch.Tensor, seq_len: int, B: int,
    ) -> torch.Tensor:
        """Shared time-embedding path for DINO / pointmap tokens.

        ``timestep_per_token`` is [B, seq_len] with 0 for first-frame anchors.
        """
        t_emb = sinusoidal_embedding_1d(
            self.video_expert.freq_dim, timestep_per_token.reshape(-1),
        )
        t = self.video_expert.time_embedding(t_emb).reshape(
            B, seq_len, self.video_expert.hidden_dim,
        )
        return self.video_expert.time_projection(t).unflatten(
            2, (6, self.video_expert.hidden_dim),
        )

    # ------------------------------------------------------------------
    # Flex-joint hooks (no-op when ``self._batch_flex is None``).
    # ------------------------------------------------------------------

    def _flex_zero_absent_video_tokens(
        self, video_tokens: torch.Tensor, tpf: int,
    ) -> torch.Tensor:
        """Per-sample zero ff_v (and rem_v if not cross-modal) for absent video.

        ``video_tokens``: [B, Sv, D]. Returns same shape with rows zeroed for
        samples where ``present_v=False``. When ``cm_v=True``, only the first
        ``tpf`` rows (ff_v) are zeroed; rem_v survives for cross-modal denoise.
        """
        bf = self._batch_flex
        if bf is None:
            return video_tokens
        # absent = ~present_v ; mask is True where we should zero a token.
        absent_v = (~bf.present_v).to(device=video_tokens.device)  # [B]
        if not absent_v.any():
            return video_tokens
        B, Sv, _ = video_tokens.shape
        # Build [B, Sv] zero-mask: True at rows to keep, False at rows to zero.
        keep = torch.ones((B, Sv), dtype=torch.bool, device=video_tokens.device)
        if bf.cm_v:
            # Only zero ff_v rows for absent samples.
            keep[absent_v, :tpf] = False
        else:
            keep[absent_v, :] = False
        return video_tokens * keep.unsqueeze(-1).to(video_tokens.dtype)

    def _flex_zero_absent_dino_tokens(
        self, dino_tokens: torch.Tensor, dtpf: int,
    ) -> torch.Tensor:
        """Per-sample zero ff_d (and rem_d if not cross-modal) for absent dino.

        ``dino_tokens``: [B, Sd, D]. ``dtpf`` is the ff_d token count.
        """
        bf = self._batch_flex
        if bf is None:
            return dino_tokens
        absent_d = (~bf.present_d).to(device=dino_tokens.device)  # [B]
        if not absent_d.any():
            return dino_tokens
        B, Sd, _ = dino_tokens.shape
        keep = torch.ones((B, Sd), dtype=torch.bool, device=dino_tokens.device)
        if bf.cm_d:
            keep[absent_d, :dtpf] = False
        else:
            keep[absent_d, :] = False
        return dino_tokens * keep.unsqueeze(-1).to(dino_tokens.dtype)

    def _flex_zero_absent_pointmap_tokens(
        self, pointmap_tokens: torch.Tensor, ptpf: int,
    ) -> torch.Tensor:
        """Per-sample zero ff_p (and rem_p if not cross-modal) for absent pointmap.

        ``pointmap_tokens``: [B, Sp, D]. ``ptpf`` is the ff_p token count.
        """
        bf = self._batch_flex
        if bf is None:
            return pointmap_tokens
        absent_p = (~bf.present_p).to(device=pointmap_tokens.device)  # [B]
        if not absent_p.any():
            return pointmap_tokens
        B, Sp, _ = pointmap_tokens.shape
        keep = torch.ones((B, Sp), dtype=torch.bool, device=pointmap_tokens.device)
        if bf.cm_p:
            keep[absent_p, :ptpf] = False
        else:
            keep[absent_p, :] = False
        return pointmap_tokens * keep.unsqueeze(-1).to(pointmap_tokens.dtype)

    def _flex_reduce_per_sample_loss(
        self,
        per_sample_loss: torch.Tensor,
        weights: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None,
        cross_modal_active: bool = False,
    ) -> torch.Tensor:
        """Mean over the present samples in the batch (flex-aware).

        Default (``self._batch_flex is None``): standard ``.mean()``.
        Flex: loss is included for samples that contribute training signal —
        i.e. ``present`` OR ``cross_modal_active``. Absent-non-cm samples
        contribute zero.
        """
        bf = self._batch_flex
        weighted = per_sample_loss * weights
        if bf is None or present_mask is None:
            return weighted.mean()
        mask = present_mask.to(device=weighted.device, dtype=weighted.dtype)
        if cross_modal_active:
            # Every sample contributes (cross-modal denoise everywhere).
            return weighted.mean()
        denom = mask.sum().clamp(min=1.0)
        return (weighted * mask).sum() / denom

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _base_training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        B = input_latents.shape[0]
        context, context_mask = inputs["context"], inputs["context_mask"]
        action = inputs["action"]
        action_is_pad, image_is_pad = inputs["action_is_pad"], inputs["image_is_pad"]
        dino_features = inputs["dino_features"]     # [B, 768, F_d, 294, 1]
        pointmap_raw = inputs["pointmap_raw"]       # mode-specific shape

        # --- Video noise ---
        noise_video = torch.randn_like(input_latents)
        t_video = self.train_video_scheduler.sample_training_t(B, self.device, input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, t_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, t_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # --- Action noise ---
        noise_action = torch.randn_like(action)
        t_action = self.train_action_scheduler.sample_training_t(B, self.device, action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, t_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, t_action)

        # --- DINO noise ---
        noise_dino = torch.randn_like(dino_features)
        t_dino = self.train_dino_scheduler.sample_training_t(B, self.device, dino_features.dtype)
        noisy_dino = self.train_dino_scheduler.add_noise(dino_features, noise_dino, t_dino)
        target_dino = self.train_dino_scheduler.training_target(dino_features, noise_dino, t_dino)
        noisy_dino[:, :, 0:1] = dino_features[:, :, 0:1]

        # --- Pointmap noise ---
        if self._pointmap_globally_off:
            # Pointmap globally off (see _pointmap_globally_off): skip noise +
            # target so the stream produces no loss; tokens are skipped below.
            noisy_pointmap = t_pointmap = target_pointmap = None
        else:
            noise_pt = torch.randn_like(pointmap_raw)
            t_pointmap = self.train_pointmap_scheduler.sample_training_t(
                B, self.device, pointmap_raw.dtype,
            )
            noisy_pointmap = self.train_pointmap_scheduler.add_noise(
                pointmap_raw, noise_pt, t_pointmap,
            )
            target_pointmap = self.train_pointmap_scheduler.training_target(
                pointmap_raw, noise_pt, t_pointmap,
            )
            noisy_pointmap[:, :, 0:1] = pointmap_raw[:, :, 0:1]

        # --- Pre-dit ---
        video_pre = self.video_expert.pre_dit(
            x=latents, timestep=t_video, context=context, context_mask=context_mask,
            action=action, fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=t_action,
            context=context, context_mask=context_mask,
        )

        Sv = video_pre["tokens"].shape[1]
        Sa = action_pre["tokens"].shape[1]
        tpf = int(video_pre["meta"]["tokens_per_frame"])

        # Flex: per-sample zero ff_v (and rem_v if not cross-modal) for samples
        # where video is presence-dropped this step. No-op when _batch_flex is None.
        video_pre["tokens"] = self._flex_zero_absent_video_tokens(
            video_pre["tokens"], tpf,
        )

        # --- DINO embed + time + freqs ---
        dino_tokens = self._embed_dino(noisy_dino)
        Sd = dino_tokens.shape[1]
        num_dino_frames = int(dino_features.shape[2])
        dtpf = int(dino_features.shape[3])
        # Flex: per-sample zero ff_d (and rem_d if not cross-modal) for samples
        # where dino is presence-dropped this step. No-op when _batch_flex is None.
        dino_tokens = self._flex_zero_absent_dino_tokens(dino_tokens, dtpf)

        dino_token_timesteps = t_dino.view(B, 1).expand(B, Sd).clone()
        dino_token_timesteps[:, :dtpf] = 0
        dino_t_mod = self._build_stream_t_mod(dino_token_timesteps, Sd, B)
        dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre["tokens"].device)

        # --- Pointmap embed + time + freqs ---
        if self._pointmap_globally_off:
            # Skip pointmap tokenization. None tokens drop the stream from the
            # merged sequence (_merge_aux_into_video_stream omits None aux) and
            # Sp=0 collapses all pointmap mask/loss edits into no-ops.
            pointmap_tokens = pt_t_mod = pt_freqs = pt_t = pt_meta = None
            Sp = ptpf = 0
        else:
            pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(noisy_pointmap)
            # Flex: per-sample zero ff_p (and rem_p if not cross-modal) for samples
            # where pointmap is presence-dropped this step. No-op when _batch_flex is None.
            pointmap_tokens = self._flex_zero_absent_pointmap_tokens(pointmap_tokens, ptpf)
            Sp = pointmap_tokens.shape[1]
            num_pt_frames = int(noisy_pointmap.shape[2])

            pt_token_timesteps = t_pointmap.view(B, 1).expand(B, Sp).clone()
            # Frame 0 is a clean anchor → force its timesteps to 0.
            pt_token_timesteps[:, :ptpf] = 0
            pt_t_mod = self._build_stream_t_mod(pt_token_timesteps, Sp, B)
            pt_freqs = self._compute_pointmap_freqs(
                num_pt_frames, video_pre["tokens"].device, pt_meta=pt_meta,
            )
            # Also build a per-stream time [B, Sp, hidden] for _project_pointmap_out head.
            pt_t_emb = sinusoidal_embedding_1d(
                self.video_expert.freq_dim, pt_token_timesteps.reshape(-1),
            )
            pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(
                B, Sp, self.video_expert.hidden_dim,
            )

        # --- Merge & mask ---
        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = (
            self._merge_aux_into_video_stream(
                video_pre, dino_tokens, dino_t_mod, dino_freqs,
                pointmap_tokens, pt_t_mod, pt_freqs,
            )
        )
        effective_dtpf = dtpf
        effective_ptpf = ptpf
        Sp_for_mask = Sp
        ptpf_for_mask = effective_ptpf
        attention_mask = self._build_mot_attention_mask_unified(
            Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device,
        )
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks_unified(
            Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device,
        )

        # --- MoT forward ---
        tokens_out = self.mot(
            embeds_all={"video": merged_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": merged_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": merged_ctx_mask},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": merged_t_mod, "action": action_pre["t_mod"]},
            sub_stream_lens=sub_stream_lens,
            sub_stream_self_masks=sub_stream_self_masks,
        )

        # --- Split ---
        video_tokens_out = tokens_out["video"][:, :Sv, :]
        dino_tokens_out = tokens_out["video"][:, Sv:Sv + Sd, :]
        pt_tokens_out = tokens_out["video"][:, Sv + Sd:, :]

        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_dino = self._project_dino_out(dino_tokens_out)  # [B, Sd, dino_dim]
        # Pointmap absent from the sequence when globally off → no head to run.
        pred_pointmap = (
            None if self._pointmap_globally_off
            else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        )

        # --- Video loss ---
        include_init = inputs["first_frame_latents"] is None
        if not include_init:
            pred_video, target_video = pred_video[:, :, 1:], target_video[:, :, 1:]
        lv = self._compute_video_loss_per_sample(pred_video, target_video, image_is_pad, include_init)
        wv = self.train_video_scheduler.training_weight(t_video).to(lv.device, dtype=lv.dtype)
        # Flex: mean over present_v samples (or all if cm_v active). No-op when flex off.
        loss_video = self._flex_reduce_per_sample_loss(
            lv, wv,
            present_mask=(self._batch_flex.present_v if self._batch_flex is not None else None),
            cross_modal_active=(self._batch_flex.cm_v if self._batch_flex is not None else False),
        )

        # --- Action loss ---
        la = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(2)
        if action_is_pad is not None:
            v = (~action_is_pad).to(la.device, dtype=la.dtype)
            la = (la * v).sum(1) / v.sum(1).clamp(min=1.0)
        else:
            la = la.mean(1)
        wa = self.train_action_scheduler.training_weight(t_action).to(la.device, dtype=la.dtype)
        loss_action = (la * wa).mean()

        # --- DINO loss ---
        if target_dino is None:
            loss_dino = torch.tensor(0.0, device=self.device)
        else:
            target_dino_flat = target_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(
                B, -1, self.dino_dim,
            )
            if self.dino_pred_x0:
                # Head output is x̂0 → derive v̂ = (x_t − x̂0)/σ for the MSE
                # below. Frame-0's v̂ is meaningless (x_t there is the clean
                # anchor) but ld[:, dtpf:] already excludes it from the loss.
                noisy_dino_flat = noisy_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(
                    B, -1, self.dino_dim,
                )
                pred_dino_for_loss = _dino_x0_to_velocity(
                    noisy_dino_flat, pred_dino, t_dino.view(B, 1, 1),
                    self.train_dino_scheduler.num_train_timesteps,
                )
            else:
                pred_dino_for_loss = pred_dino
            ld = F.mse_loss(pred_dino_for_loss.float(), target_dino_flat.float(), reduction="none").mean(2)
            n_rem = num_dino_frames - 1
            if image_is_pad is None or n_rem <= 0:
                # Legacy path — bit-exact when no padding info is available.
                ld = ld[:, dtpf:].mean(1)
            else:
                # Per-frame reduce so we can mask out frames whose source RGB
                # was padded.
                ld_rem = ld[:, dtpf:]
                ld_per_frame = ld_rem.view(B, n_rem, dtpf).mean(2)
                dino_is_pad_rem = self._aux_per_frame_is_pad(
                    image_is_pad, self.dino_temporal_stride,
                    keep_far=self.dino_stride_keep_far,
                )[:, 1:]
                if dino_is_pad_rem.shape[1] != n_rem:
                    raise RuntimeError(
                        f"DINO mask shape mismatch: pad={dino_is_pad_rem.shape[1]} "
                        f"vs num_dino_frames-1={n_rem}"
                    )
                ld = self._masked_loss_reduction(ld_per_frame, dino_is_pad_rem)
            wd = self.train_dino_scheduler.training_weight(t_dino).to(ld.device, dtype=ld.dtype)
            # Flex: mean over present_d samples (or all if cm_d active). No-op when flex off.
            loss_dino = self._flex_reduce_per_sample_loss(
                ld, wd,
                present_mask=(self._batch_flex.present_d if self._batch_flex is not None else None),
                cross_modal_active=(self._batch_flex.cm_d if self._batch_flex is not None else False),
            )

        # --- Pointmap loss ---
        if target_pointmap is None:
            loss_pointmap = torch.tensor(0.0, device=self.device)
        else:
            # Skip the ff_p anchor at index 0.
            pred_pt_for_loss = pred_pointmap[:, :, 1:].float()
            tgt_pt_for_loss = target_pointmap[:, :, 1:].float()
            # Pointmap latents share the video VAE's temporal layout, so
            # reuse the video-loss helper. `include_initial=False` because
            # the anchor is already excluded from pred_pt_for_loss.
            if image_is_pad is None:
                # Strict bit-exactness with the legacy pointmap reduction
                # when no padding info is available. (The video-loss helper
                # would use `mean(dim=(1,3,4)).mean(1)` here — same answer,
                # ~1e-7 fp32 reorder. We keep the legacy `flatten(1).mean(1)`
                # to avoid any numerical drift in backwards-compat paths.)
                err = F.mse_loss(pred_pt_for_loss, tgt_pt_for_loss, reduction="none")
                lp = err.flatten(1).mean(1)
            else:
                lp = self._compute_video_loss_per_sample(
                    pred_pt_for_loss, tgt_pt_for_loss, image_is_pad,
                    include_initial_video_step=False,
                )
            wp = self.train_pointmap_scheduler.training_weight(t_pointmap).to(lp.device, dtype=lp.dtype)
            # Flex: mean over present_p samples (or all if cm_p active). No-op when flex off.
            loss_pointmap = self._flex_reduce_per_sample_loss(
                lp, wp,
                present_mask=(self._batch_flex.present_p if self._batch_flex is not None else None),
                cross_modal_active=(self._batch_flex.cm_p if self._batch_flex is not None else False),
            )

        total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_dino * loss_dino
            + self.loss_lambda_pointmap * loss_pointmap
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_dino": self.loss_lambda_dino * float(loss_dino.detach().item()),
            "loss_pointmap": self.loss_lambda_pointmap * float(loss_pointmap.detach().item()),
        }
        return total, loss_dict

    # ------------------------------------------------------------------
    # Joint noise prediction — used by inference at eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_joint_noise_unified(
        self,
        latents_video: torch.Tensor,
        latents_dino: torch.Tensor,
        latents_pointmap: Optional[torch.Tensor],
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_dino: torch.Tensor,
        timestep_pointmap: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action=None,
        dino_freqs: torch.Tensor | None = None,
        pt_freqs: torch.Tensor | None = None,
        flex_block_attention: bool = False,
    ):
        """Single MoT forward that predicts noise for all four streams.

        ``dino_freqs`` and ``pt_freqs`` may
        be precomputed by the caller. They depend only on cam geometry +
        frame counts, static across denoise steps. Passing them in avoids a
        CPU tensor leak inside the compile target.
        """
        # trt_joint_engine_path: the engine call inside the impl can't live in
        # an inductor CUDA graph — run the impl uncompiled (eager pre/post
        # around the engine); other compile targets are unaffected.
        if (
            getattr(self, "_compile_inference", False)
            and self.device.type == "cuda"
            and getattr(self, "_trt_joint_runner", None) is None
        ):
            if not getattr(self, "_joint_unified_step_is_compiled", False):
                self._joint_unified_step_compiled = self._compile_for_inference(
                    self._predict_joint_noise_unified_impl,
                )
                self._joint_unified_step_is_compiled = True
            # attn_backend="auto": cuDNN-first priority for the masked joint
            # attention (no-op ctx otherwise). Wraps the call so the choice is
            # made inside, at trace/capture time.
            with self._sdpa_priority_ctx():
                return self._joint_unified_step_compiled(
                    latents_video=latents_video,
                    latents_dino=latents_dino,
                    latents_pointmap=latents_pointmap,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_dino=timestep_dino,
                    timestep_pointmap=timestep_pointmap,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    gt_action=gt_action,
                    dino_freqs=dino_freqs,
                    pt_freqs=pt_freqs,
                    flex_block_attention=flex_block_attention,
                )
        with self._sdpa_priority_ctx():
            return self._predict_joint_noise_unified_impl(
                latents_video=latents_video,
                latents_dino=latents_dino,
                latents_pointmap=latents_pointmap,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_dino=timestep_dino,
                timestep_pointmap=timestep_pointmap,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                gt_action=gt_action,
                dino_freqs=dino_freqs,
                pt_freqs=pt_freqs,
                flex_block_attention=flex_block_attention,
            )

    def _predict_joint_noise_unified_impl(
        self,
        latents_video: torch.Tensor,
        latents_dino: torch.Tensor,
        latents_pointmap: Optional[torch.Tensor],
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_dino: torch.Tensor,
        timestep_pointmap: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action=None,
        dino_freqs: torch.Tensor | None = None,
        pt_freqs: torch.Tensor | None = None,
        flex_block_attention: bool = False,
    ):
        B = latents_video.shape[0]

        video_pre = self.video_expert.pre_dit(
            x=latents_video, timestep=timestep_video,
            context=context, context_mask=context_mask,
            action=gt_action, fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action, timestep=timestep_action,
            context=context, context_mask=context_mask,
        )

        Sv = video_pre["tokens"].shape[1]
        Sa = action_pre["tokens"].shape[1]
        tpf = int(video_pre["meta"]["tokens_per_frame"])

        # DINO
        dino_tokens = self._embed_dino(latents_dino)
        Sd = dino_tokens.shape[1]
        num_dino_frames = int(latents_dino.shape[2])
        dtpf = int(latents_dino.shape[3])
        dino_token_timesteps = timestep_dino.view(B, 1).expand(B, Sd).clone()
        dino_token_timesteps[:, :dtpf] = 0
        dino_t_mod = self._build_stream_t_mod(dino_token_timesteps, Sd, B)
        # DINO RoPE — use precomputed when caller hoisted it (compile mode);
        # otherwise compute on demand (training / eager-eval).
        if dino_freqs is None:
            dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre["tokens"].device)

        # Pointmap. ``latents_pointmap=None`` means the run carries no pointmap
        # stream at all — the same Sp=0 shape training_loss builds under
        # ``_pointmap_globally_off``. None tokens drop the stream from the merged
        # sequence and Sp=0 collapses every pointmap mask edit into a no-op, so
        # val-vis matches the layout the model was trained on.
        if latents_pointmap is None:
            pointmap_tokens = pt_t_mod = pt_freqs = pt_meta = pt_t = None
            Sp = ptpf = 0
        else:
            pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(latents_pointmap)
            Sp = pointmap_tokens.shape[1]
            num_pt_frames = int(latents_pointmap.shape[2])
            pt_token_timesteps = timestep_pointmap.view(B, 1).expand(B, Sp).clone()
            pt_token_timesteps[:, :ptpf] = 0
            pt_t_mod = self._build_stream_t_mod(pt_token_timesteps, Sp, B)
            if pt_freqs is None:
                pt_freqs = self._compute_pointmap_freqs(
                    num_pt_frames, video_pre["tokens"].device, pt_meta=pt_meta,
                )
            pt_t_emb = sinusoidal_embedding_1d(
                self.video_expert.freq_dim, pt_token_timesteps.reshape(-1),
            )
            pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(
                B, Sp, self.video_expert.hidden_dim,
            )

        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = (
            self._merge_aux_into_video_stream(
                video_pre, dino_tokens, dino_t_mod, dino_freqs,
                pointmap_tokens, pt_t_mod, pt_freqs,
            )
        )
        effective_dtpf = dtpf
        effective_ptpf = ptpf
        Sp_for_mask = Sp
        ptpf_for_mask = effective_ptpf
        if flex_block_attention:
            # attn_backend="flex": the joint full-attention mask is served as a
            # BlockMask via the class-level ``MoT.attention_mask`` (prepared by
            # ``_prepare_joint_flex_block_mask`` before the denoise loop);
            # passing None routes ``_mixed_attention`` to its FlexAttention
            # branch. HBridge outer-layer sub-masks stay explicit (SDPA).
            attention_mask = None
        else:
            attention_mask = self._build_mot_attention_mask_unified(
                Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device,
            )
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks_unified(
            Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device,
        )

        merged_out, action_out = self.mot._forward_joint_inner(
            video_tokens=merged_tokens,
            action_tokens=action_pre["tokens"],
            video_freqs=merged_freqs,
            action_freqs=action_pre["freqs"],
            video_t_mod=merged_t_mod,
            action_t_mod=action_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": merged_ctx_mask},
            action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            attention_mask=attention_mask,
            sub_stream_lens=sub_stream_lens,
            sub_stream_self_masks=sub_stream_self_masks,
        )

        video_tokens_out = merged_out[:, :Sv, :]
        dino_tokens_out = merged_out[:, Sv:Sv + Sd, :]
        pt_tokens_out = merged_out[:, Sv + Sd:, :]

        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(action_out, action_pre)
        pred_dino = self._project_dino_out(dino_tokens_out)
        # Reshape pred_dino back to raw [B, D, F_d, dtpf, 1] for the scheduler step.
        pred_dino = pred_dino.reshape(B, num_dino_frames, dtpf, self.dino_dim)
        pred_dino = pred_dino.permute(0, 3, 1, 2).unsqueeze(-1)
        if self.dino_pred_x0:
            # Head output is x̂0 → convert to v̂ in the packed
            # layout so everything downstream (scheduler.step, frame-0
            # re-clamp, step-skip caches) stays velocity-based.
            pred_dino = _dino_x0_to_velocity(
                latents_dino, pred_dino, timestep_dino.view(-1, 1, 1, 1, 1),
                self.infer_dino_scheduler.num_train_timesteps,
            ).to(pred_dino.dtype)
        pred_pointmap = (
            None if pointmap_tokens is None
            else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        )

        return pred_video, pred_dino, pred_pointmap, pred_action

    # ------------------------------------------------------------------
    # Inference — action only (first-frame anchors in KV cache)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _base_infer_action(
        self,
        prompt=None,
        input_image=None,
        action_horizon=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        camera_intrinsics: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
        present_video: Optional[bool] = None,
        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
    ):
        """Baseline action-only inference.

        Prefills the video KV cache with first-frame video + DINO + pointmap
        anchors, then denoises the action only (no future-latent generation).

        ``present_video=False`` / ``present_dino=False`` / ``present_pointmap=False``
        skip the corresponding encoder and drop the stream from the prefill
        layout for this call. ``None`` = honor the trained config only.
        ``present_video=False`` routes through the no-video path (DINO and/or
        pointmap anchor only). Combining all three as False is invalid (no
        conditioning anchors left).
        """
        _skip_pointmap = (present_pointmap is False)
        _skip_dino = (present_dino is False)
        _skip_video = (present_video is False)
        if _skip_video:
            return self._infer_action_no_video(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                camera_intrinsics=camera_intrinsics,
                per_cam=per_cam,
                per_cam_depth=per_cam_depth,
                present_dino=present_dino,
                present_pointmap=present_pointmap,
            )
        self.eval()
        if camera_intrinsics is not None:
            self.set_camera_intrinsics(camera_intrinsics)
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        if per_cam_depth is None and not _skip_pointmap:
            raise ValueError(
                "`per_cam_depth` is required at inference for FlexPi when the "
                "pointmap stream is present. Pass present_pointmap=False (or use "
                "a pointmap-disabled model) to run without depth."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._zero_padded_action(latents_action)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # Text context
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is None and context is None:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # Video pre_dit (first frame)
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype, device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents, timestep=timestep_video,
            context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )

        # First-frame DINO + pointmap tokens
        # ``_skip_dino`` / ``_skip_pointmap`` short-circuit the corresponding
        # encoder, per-call (used by the flex joint subclass to honor presence
        # overrides without forcing the slow joint denoise path).
        if _skip_dino:
            dino_tokens, dtpf = None, 0
            dino_t_mod = None
            dino_freqs = None
        else:
            dino_tokens = self._encode_first_frame_dino(
                input_image=input_image, per_cam=per_cam,
            )
            dtpf = dino_tokens.shape[1]
            t_dino_zero = torch.zeros(1, dtpf, device=self.device, dtype=self.torch_dtype)
            dino_t_mod = self._build_stream_t_mod(t_dino_zero, dtpf, 1)
            dino_freqs = self._compute_dino_freqs(1, video_pre["tokens"].device)

        if _skip_pointmap:
            pointmap_tokens, ptpf, pt_meta = None, 0, None
            pt_t_mod = None
            pt_freqs = None
        else:
            pt_raw_ff = self._encode_first_frame_pointmap_raw(
                tiled=tiled, per_cam_depth=per_cam_depth,
            )
            pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(pt_raw_ff)
            t_pt_zero = torch.zeros(1, ptpf, device=self.device, dtype=self.torch_dtype)
            pt_t_mod = self._build_stream_t_mod(t_pt_zero, ptpf, 1)
            pt_freqs = self._compute_pointmap_freqs(
                1, video_pre["tokens"].device, pt_meta=pt_meta,
            )

        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = (
            self._merge_aux_into_video_stream(
                video_pre, dino_tokens, dino_t_mod, dino_freqs,
                pointmap_tokens, pt_t_mod, pt_freqs,
            )
        )

        Sv = video_pre["tokens"].shape[1]
        # Skipped streams contribute 0 to the layout.
        video_seq_len = Sv + dtpf + ptpf
        Sa = latents_action.shape[1]
        total_seq = video_seq_len + Sa

        # Action attends to all first-frame anchors; all ff tokens self-attend.
        mask = torch.zeros((total_seq, total_seq), dtype=torch.bool, device=self.device)
        mask[:video_seq_len, :video_seq_len] = True
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True

        # attn_backend="auto": every block above is set True, so this mask is
        # all-True by construction — equivalent to no mask. Dropping it (and
        # any all-visible context masks) lets SDPA dispatch flash instead of
        # the masked memory-efficient kernel (~4x faster at [Sa=32, kv≈566]).
        # Only on the compiled fast paths — the uncompiled fallback goes
        # through `forward_action_with_video_cache`, which requires a mask.
        if (
            getattr(self, "_infer_attn_backend", "sdpa") == "auto"
            and getattr(self, "_compile_inference", False)
        ):
            mask = None
            context_mask = self._drop_trivial_mask(context_mask)
            merged_ctx_mask = self._drop_trivial_mask(merged_ctx_mask)

        # HBridge: per-sub-stream prefill (V vs DINO vs P).
        # Skipped sub-streams contribute 0 length; the helper drops them via
        # its `if S > 0` guards.
        Sp_for_split = ptpf
        ptpf_for_split = ptpf
        v_sub_lens, v_sub_masks = self._build_video_hbridge_self_masks_unified(
            video_seq_len=Sv,
            dino_seq_len=dtpf,
            pointmap_seq_len=Sp_for_split,
            video_tokens_per_frame=Sv,
            dino_tokens_per_frame=dtpf,
            pointmap_tokens_per_frame=ptpf_for_split,
            device=self.device,
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        if self._use_loop_compile():
            # torch_compile_scope="loop": KV-cache prefill + entire denoise
            # loop fused into one compiled call (single CUDA Graph region per
            # call — no Python between prefill and steps). Identical math to
            # the eager prefill + per-step loop below.
            latents_action = self._run_action_prefill_denoise_loop(
                merged_tokens=merged_tokens,
                merged_freqs=merged_freqs,
                merged_t_mod=merged_t_mod,
                video_context=video_pre["context"],
                merged_ctx_mask=merged_ctx_mask,
                prefill_attention_mask=(
                    mask[:video_seq_len, :video_seq_len] if mask is not None else None
                ),
                v_sub_lens=v_sub_lens,
                v_sub_masks=v_sub_masks,
                latents_action=latents_action,
                infer_timesteps=infer_timesteps,
                infer_deltas=infer_deltas,
                context=context,
                context_mask=context_mask,
                attention_mask=mask,
                video_seq_len=video_seq_len,
            )
        else:
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=merged_tokens,
                video_freqs=merged_freqs,
                video_t_mod=merged_t_mod,
                video_context_payload={"context": video_pre["context"], "mask": merged_ctx_mask},
                video_attention_mask=(
                    mask[:video_seq_len, :video_seq_len] if mask is not None else None
                ),
                video_sub_stream_lens=v_sub_lens,
                video_sub_stream_self_masks=v_sub_masks,
            )
            for step_t, step_delta in zip(infer_timesteps, infer_deltas):
                timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                pred_action = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context, context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=video_seq_len,
                )
                latents_action = self.infer_action_scheduler.step(
                    pred_action, step_delta, latents_action,
                )
                latents_action = self._zero_padded_action(latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def _action_prefill_denoise_loop_body(
        self,
        merged_tokens: torch.Tensor,
        merged_freqs: torch.Tensor,
        merged_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        merged_ctx_mask: torch.Tensor,
        prefill_attention_mask: torch.Tensor,
        v_sub_lens,
        v_sub_masks,
        latents_action: torch.Tensor,
        infer_timesteps: torch.Tensor,
        infer_deltas: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_attention_mask: torch.Tensor,
        action_freqs: torch.Tensor,
        action_only_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """torch_compile_scope="loop" target for the action-only fast path:
        MoT prefill + all denoise steps + Euler updates in one traced graph.
        Reuses ``FlexPi._action_denoise_loop_body`` (inlined by dynamo) after
        flattening the prefill's list[dict] cache."""
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=merged_tokens,
            video_freqs=merged_freqs,
            video_t_mod=merged_t_mod,
            video_context_payload={"context": video_context, "mask": merged_ctx_mask},
            video_attention_mask=prefill_attention_mask,
            video_sub_stream_lens=v_sub_lens,
            video_sub_stream_self_masks=v_sub_masks,
        )
        cache_k = [c["k"] for c in video_kv_cache]
        cache_v = [c["v"] for c in video_kv_cache]
        return self._action_denoise_loop_body(
            latents_action=latents_action,
            infer_timesteps=infer_timesteps,
            infer_deltas=infer_deltas,
            context=context,
            context_mask=context_mask,
            video_cache_k=cache_k,
            video_cache_v=cache_v,
            action_attention_mask=action_attention_mask,
            action_freqs=action_freqs,
            action_only_attention_mask=action_only_attention_mask,
        )

    @torch.no_grad()
    def _run_action_prefill_denoise_loop(
        self,
        merged_tokens: torch.Tensor,
        merged_freqs: torch.Tensor,
        merged_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        merged_ctx_mask: torch.Tensor,
        prefill_attention_mask: torch.Tensor,
        v_sub_lens,
        v_sub_masks,
        latents_action: torch.Tensor,
        infer_timesteps: torch.Tensor,
        infer_deltas: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Loop-scope dispatcher for the action-only fast path: hoists the
        per-call slicing (same as the Tier-1 per-step dispatch), then one
        compiled call covering prefill + the entire denoise loop."""
        if not getattr(self, "_action_prefill_loop_is_compiled", False):
            self._action_prefill_denoise_loop_compiled = self._compile_for_inference(
                self._action_prefill_denoise_loop_body,
            )
            self._action_prefill_loop_is_compiled = True
        action_seq_len = latents_action.shape[1]
        total_seq_len = video_seq_len + action_seq_len
        # attn_backend="auto" callers pass attention_mask=None — keep None-safe.
        action_attention_mask = (
            attention_mask[video_seq_len:total_seq_len, :total_seq_len]
            if attention_mask is not None else None
        )
        action_freqs = self.action_expert.freqs[:action_seq_len].view(action_seq_len, 1, -1).to(latents_action.device)
        # NOTE: never mask-drop action_only_attention_mask — its presence is
        # the HBridge outer-layer routing flag, not just a visibility mask.
        action_only_attention_mask = self._build_action_only_attention_mask(
            action_seq_len=action_seq_len,
            device=latents_action.device,
        )
        return self._action_prefill_denoise_loop_compiled(
            merged_tokens=merged_tokens,
            merged_freqs=merged_freqs,
            merged_t_mod=merged_t_mod,
            video_context=video_context,
            merged_ctx_mask=merged_ctx_mask,
            prefill_attention_mask=prefill_attention_mask,
            v_sub_lens=v_sub_lens,
            v_sub_masks=v_sub_masks,
            latents_action=latents_action,
            infer_timesteps=infer_timesteps,
            infer_deltas=infer_deltas,
            context=context,
            context_mask=context_mask,
            action_attention_mask=action_attention_mask,
            action_freqs=action_freqs,
            action_only_attention_mask=action_only_attention_mask,
        )

    @torch.no_grad()
    def _infer_action_no_video(
        self,
        prompt,
        input_image,
        action_horizon,
        proprio,
        context,
        context_mask,
        num_inference_steps,
        sigma_shift,
        seed,
        rand_device,
        tiled,
        camera_intrinsics,
        per_cam,
        per_cam_depth,
        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
    ):
        """infer_action path when video is absent (no VAE / no video DiT prefill).

        Triggered at runtime by passing ``present_video=False`` into
        ``infer_action``.
        Cache prefilled with first-frame DINO + first-frame pointmap (any of
        which can be additionally skipped via ``present_dino=False`` /
        ``present_pointmap=False``).
        """
        _skip_dino = (present_dino is False)
        _skip_pointmap = (present_pointmap is False)
        self.eval()
        if camera_intrinsics is not None:
            self.set_camera_intrinsics(camera_intrinsics)
        if per_cam_depth is None and not _skip_pointmap:
            raise ValueError(
                "`per_cam_depth` is required at inference for FlexPi when the "
                "pointmap stream is present. Pass present_pointmap=False (or use "
                "a pointmap-disabled model) to run without depth."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._zero_padded_action(latents_action)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)

        # Text context
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is None and context is None:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # First-frame DINO + pointmap tokens (each gated by its skip flag).
        if _skip_dino:
            dino_tokens = None
            dtpf = 0
            dino_t_mod = None
            dino_freqs = None
        else:
            dino_tokens = self._encode_first_frame_dino(
                input_image=input_image, per_cam=per_cam,
            )
            dtpf = dino_tokens.shape[1]
            t_dino_zero = torch.zeros(1, dtpf, device=self.device, dtype=self.torch_dtype)
            dino_t_mod = self._build_stream_t_mod(t_dino_zero, dtpf, 1)
            dino_freqs = self._compute_dino_freqs(1, dino_tokens.device)

        if _skip_pointmap:
            pointmap_tokens = None
            ptpf = 0
            pt_meta = None
            pt_t_mod = None
            pt_freqs = None
        else:
            pt_raw_ff = self._encode_first_frame_pointmap_raw(
                tiled=tiled, per_cam_depth=per_cam_depth,
            )
            pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(pt_raw_ff)
            t_pt_zero = torch.zeros(1, ptpf, device=self.device, dtype=self.torch_dtype)
            pt_t_mod = self._build_stream_t_mod(t_pt_zero, ptpf, 1)
            pt_freqs = self._compute_pointmap_freqs(
                1, pointmap_tokens.device, pt_meta=pt_meta,
            )

        if _skip_dino and _skip_pointmap:
            raise ValueError(
                "No-video infer_action requires at least one of DINO or pointmap "
                "to be present. Got present_video=False, present_dino=False, "
                "present_pointmap=False — there are no conditioning anchors to "
                "prefill the action cache against."
            )

        embedded_context = self.video_expert.text_embedding(context)

        parts_tok = [t for t in (dino_tokens, pointmap_tokens) if t is not None]
        parts_freq = [f for f in (dino_freqs, pt_freqs) if f is not None]
        parts_tmod = [m for m in (dino_t_mod, pt_t_mod) if m is not None]
        merged_tokens = torch.cat(parts_tok, dim=1)
        merged_freqs  = torch.cat(parts_freq, dim=0)
        merged_t_mod  = torch.cat(parts_tmod, dim=1)
        merged_video_seq_len = dtpf + ptpf
        Sp_for_split = ptpf
        ptpf_for_split = ptpf

        merged_ctx_mask = context_mask.unsqueeze(1).expand(-1, merged_video_seq_len, -1)

        Sa = latents_action.shape[1]
        total_seq = merged_video_seq_len + Sa
        mask = torch.zeros((total_seq, total_seq), dtype=torch.bool, device=self.device)
        mask[:merged_video_seq_len, :merged_video_seq_len] = True
        mask[merged_video_seq_len:, merged_video_seq_len:] = True
        mask[merged_video_seq_len:, :merged_video_seq_len] = True

        v_sub_lens, v_sub_masks = self._build_video_hbridge_self_masks_dino_pointmap_only(
            dino_seq_len=dtpf,
            pointmap_seq_len=Sp_for_split,
            dino_tokens_per_frame=dtpf,
            pointmap_tokens_per_frame=ptpf_for_split,
            device=self.device,
        )

        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=merged_tokens,
            video_freqs=merged_freqs,
            video_t_mod=merged_t_mod,
            video_context_payload={"context": embedded_context, "mask": merged_ctx_mask},
            video_attention_mask=mask[:merged_video_seq_len, :merged_video_seq_len],
            video_sub_stream_lens=v_sub_lens,
            video_sub_stream_self_masks=v_sub_masks,
        )

        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context, context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=mask,
                video_seq_len=merged_video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action,
            )
            latents_action = self._zero_padded_action(latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def _base_infer_joint(
        self,
        prompt=None,
        input_image=None,
        num_video_frames=None,
        action_horizon=None,
        action=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        camera_intrinsics: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Joint inference for video + DINO + pointmap + action.

        Mirrors FlexPiLatent.infer_joint and extends it with a pointmap
        stream.
        """
        self.eval()
        if camera_intrinsics is not None:
            self.set_camera_intrinsics(camera_intrinsics)
        if per_cam_depth is None and not self._pointmap_globally_off:
            raise ValueError(
                "`per_cam_depth` is required at inference for FlexPi when the "
                "pointmap stream is on. Train with enable_pointmap=false to run "
                "without depth."
            )

        # --- Action-only consistency check ---
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone() if input_image is not None else None,
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                per_cam={k: v.clone() for k, v in per_cam.items()} if per_cam is not None else None,
                per_cam_depth=(
                    None if per_cam_depth is None
                    else {k: v.clone() for k, v in per_cam_depth.items()}
                ),
            )["action"]

        # --- Input validation ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        _, _, height, width = input_image.shape
        self._check_resize_height_width(height, width, num_video_frames)
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Init video / action latents ---
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        dino_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        pt_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_gen, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_gen, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._zero_padded_action(latents_action)

        # --- Encode first-frame VAE ---
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled,
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # --- DINO latents init + first-frame anchor ---
        n_patches = sum(h * w for h, w in self.dino_cam_patches)
        if per_cam is not None:
            per_cam_5d_for_dino: Dict[str, torch.Tensor] = {}
            for k, v in per_cam.items():
                v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                if v.ndim == 4:
                    v = v.unsqueeze(2)
                per_cam_5d_for_dino[k] = v
            first_frame_dino = self.dino_encoder.encode_video(
                video=None, per_cam=per_cam_5d_for_dino,
                concat_mode="robotwin", **self._dino_encode_kwargs(), first_frame_only=True,
            )
        else:
            first_frame_dino = self.dino_encoder.encode_video(
                input_image.unsqueeze(2), concat_mode="robotwin", **self._dino_encode_kwargs(), first_frame_only=True,
            )

        num_dino_frames = 1 + len(range(1, latent_t, self.dino_temporal_stride))
        latents_dino = torch.randn(
            (1, self.dino_dim, num_dino_frames, n_patches, 1),
            generator=dino_gen, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_dino[:, :, 0:1] = first_frame_dino.clone()

        # --- Pointmap latents init + first-frame anchor ---
        # Pointmap tracks the full VAE temporal axis (4× compression), same as
        # the video stream. Both stay None when the run carries no pointmap
        # stream: the unified forward then runs at Sp=0, the shape training used.
        if self._pointmap_globally_off:
            first_frame_pointmap_raw = None
            latents_pointmap = None
        else:
            first_frame_pointmap_raw = self._encode_first_frame_pointmap_raw(
                tiled=tiled, per_cam_depth=per_cam_depth,
            )
            num_pt_frames = latent_t
            pt_full_shape = list(first_frame_pointmap_raw.shape)
            pt_full_shape[2] = num_pt_frames
            latents_pointmap = torch.randn(
                tuple(pt_full_shape),
                generator=pt_gen, device=rand_device, dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            latents_pointmap[:, :, 0:1] = first_frame_pointmap_raw.clone()

        # Shape contract with training_loss / _predict_joint_noise_unified.
        # If these drift, val-vis silently desyncs from the training MoT path —
        # crash early with the offending counts instead.
        expected_dino_t = 1 + len(range(1, latent_t, self.dino_temporal_stride))
        if latents_dino.shape[2] != expected_dino_t:
            raise RuntimeError(
                f"latents_dino temporal axis = {latents_dino.shape[2]} but expected "
                f"{expected_dino_t} "
                f"(dino_temporal_stride={self.dino_temporal_stride}, latent_t={latent_t})"
            )
        if latents_pointmap is not None and latents_pointmap.shape[2] != latent_t:
            raise RuntimeError(
                f"latents_pointmap temporal axis = {latents_pointmap.shape[2]} but "
                f"expected {latent_t}"
            )

        # --- Text context ---
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is None and context is None:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # --- Denoising loop ---
        ts_video, deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_video.dtype, shift_override=sigma_shift,
        )
        ts_action, deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        ts_dino, deltas_dino = self.infer_dino_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_dino.dtype, shift_override=sigma_shift,
        )
        ts_pt, deltas_pt = self.infer_pointmap_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype if latents_pointmap is None else latents_pointmap.dtype,
            shift_override=sigma_shift,
        )

        for step_tv, dv, step_ta, da, step_td, dd, step_tp, dp in zip(
            ts_video, deltas_video, ts_action, deltas_action,
            ts_dino, deltas_dino, ts_pt, deltas_pt,
        ):
            t_video = step_tv.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            t_action = step_ta.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            t_dino = step_td.unsqueeze(0).to(dtype=latents_dino.dtype, device=self.device)
            t_pointmap = (
                None if latents_pointmap is None
                else step_tp.unsqueeze(0).to(dtype=latents_pointmap.dtype, device=self.device)
            )

            pred_video, pred_dino, pred_pointmap, pred_action = self._predict_joint_noise_unified(
                latents_video=latents_video,
                latents_dino=latents_dino,
                latents_pointmap=latents_pointmap,
                latents_action=latents_action,
                timestep_video=t_video,
                timestep_dino=t_dino,
                timestep_pointmap=t_pointmap,
                timestep_action=t_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )

            latents_video = self.infer_video_scheduler.step(pred_video, dv, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, da, latents_action)
            latents_action = self._zero_padded_action(latents_action)
            latents_dino = self.infer_dino_scheduler.step(pred_dino, dd, latents_dino)
            if latents_pointmap is not None:
                latents_pointmap = self.infer_pointmap_scheduler.step(
                    pred_pointmap, dp, latents_pointmap,
                )

            # Keep first frames clean
            latents_video[:, :, 0:1] = first_frame_latents.clone()
            latents_dino[:, :, 0:1] = first_frame_dino.clone()
            if latents_pointmap is not None:
                latents_pointmap[:, :, 0:1] = first_frame_pointmap_raw.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(f"Action from infer_joint and infer_action differ: max_diff={max_diff:.6f}")

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "video_latents": latents_video.detach().cpu(),
            "dino_latents": latents_dino.detach().cpu(),
            "pointmap_latents": (
                None if latents_pointmap is None else latents_pointmap.detach().cpu()
            ),
            "action": action_out,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt,
        input_image,
        num_frames,
        action=None,
        action_horizon=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        camera_intrinsics: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """FlexPi-specific infer dispatch.

        Forwards per-cam inputs to ``infer_joint`` (which is overridden on
        this class to consume them — unlike the inherited
        ``FlexPi.infer_joint``).
        """
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            camera_intrinsics=camera_intrinsics,
            per_cam=per_cam,
            per_cam_depth=per_cam_depth,
        )

    # ------------------------------------------------------------------
    # Checkpoint save/load — include DINO + pointmap layers
    # ------------------------------------------------------------------

    _POINTMAP_CKPT_KEYS: tuple = ("pt_patch_embedding", "pt_head")
    _DINO_CKPT_KEYS: tuple = ("dino_embedder", "dino_proj_out", "dino_feature_norm")

    def _mode_ckpt_keys(self) -> tuple:
        return self._POINTMAP_CKPT_KEYS

    def _dino_ckpt_keys(self) -> tuple:
        return self._DINO_CKPT_KEYS

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        for key in self._dino_ckpt_keys() + self._mode_ckpt_keys():
            payload[key] = getattr(self, key).state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, strict_shape: bool = True):
        payload = super().load_checkpoint(path, optimizer, strict_shape=strict_shape)
        # ``pointmap_mode='encoder_cond'`` (pointmap tokenized by its own frozen
        # ViT instead of the WAN VAE) was removed — the pointmap head modules
        # differ, so such a checkpoint cannot map onto this model.
        saved_mode = payload.get("pointmap_mode")
        if saved_mode is not None and saved_mode != "vae_parallel":
            raise ValueError(
                f"Checkpoint was saved with pointmap_mode={saved_mode!r}. That "
                f"mode no longer exists; refusing to load."
            )
        # ``disable_video_stream=True`` (the CLWM-style 3-stream mode) was
        # removed — FlexPi always carries the video stream. Such a checkpoint
        # has a different token layout and would load silently wrong.
        if payload.get("disable_video_stream"):
            raise ValueError(
                "Checkpoint was saved with disable_video_stream=True (no video "
                "stream). That mode no longer exists; refusing to load."
            )
        for key in self._dino_ckpt_keys() + self._mode_ckpt_keys():
            if key not in payload:
                logger.warning("Checkpoint missing '%s'; keeping current params.", key)
                continue
            try:
                getattr(self, key).load_state_dict(payload[key])
            except RuntimeError as e:
                # Honor strict_shape for the DINO/mode heads too (the base only
                # shape-filters 'mot'/proprio). On warm-init across DINO layouts
                # the I/O heads change shape and should RE-INIT, not crash. Resume
                # (strict_shape=True default) still hard-fails on any mismatch.
                if strict_shape:
                    raise
                logger.warning(
                    "Shape mismatch loading '%s' (%s); keeping fresh init "
                    "(expected on warm-init across DINO/pointmap layouts with "
                    "pretrained_ckpt_strict_shape=false).", key, e,
                )
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    def __init__(
        self,
        *args,
        joint_video: bool = False,
        joint_dino: bool = False,
        joint_pointmap: bool = False,
        flex_joint: Optional[FlexJointConfig] = None,
        enable_pointmap: bool = True,
        **kwargs,
    ):
        self._base_init(*args, **kwargs)
        self.joint_video = bool(joint_video)
        self.joint_dino = bool(joint_dino)
        self.joint_pointmap = bool(joint_pointmap)
        self.flex_joint = flex_joint if flex_joint is not None else FlexJointConfig()
        # Whether this run carries a pointmap stream at all. False needs a
        # depth-free data config too — the model skips the stream either way,
        # but only the data config stops the loader decoding depth.
        self.enable_pointmap = bool(enable_pointmap)
        # Inference-time presence overrides (Python scalar bools). Default True
        # = stream present (matches legacy behavior). Mutated by ``infer_action``
        # with try/finally restore; the mask builder reads them via scalar
        # attribute access so torch.compile / Dynamo can specialize per regime
        # without graph breaks or CPU syncs.
        self._infer_present_v = True
        self._infer_present_d = True
        self._infer_present_p = True
        # One-shot guard so ``infer_action`` only warns about each
        # train-unreachable (stream, combo) once per process — eval loops
        # call infer_action thousands of times.
        self._unreachable_warned: set = set()
        # Dedup set for the effective-regime log line (keyed on the resolved
        # (joint_v, joint_d, joint_p, present_v, present_d, present_p) tuple)
        # so eval loops emit one line per distinct regime.
        self._regime_logged: set = set()
        self._configure_pointmap_off()

    # ------------------------------------------------------------------
    # Factory — parent classmethod builds cls(...) without joint flags,
    # so set them on the instance after construction.
    # ------------------------------------------------------------------

    @classmethod
    def from_wan22_pretrained(
        cls,
        joint_video: bool = False,
        joint_dino: bool = False,
        joint_pointmap: bool = False,
        flex_joint: Optional[FlexJointConfig] = None,
        enable_pointmap: bool = True,
        **kwargs,
    ):
        model = cls._base_from_wan22_pretrained(**kwargs)
        model.joint_video = bool(joint_video)
        model.joint_dino = bool(joint_dino)
        model.joint_pointmap = bool(joint_pointmap)
        model.flex_joint = flex_joint if flex_joint is not None else FlexJointConfig()
        model.enable_pointmap = bool(enable_pointmap)
        model._infer_present_v = True
        model._infer_present_d = True
        model._infer_present_p = True
        model._unreachable_warned = set()
        model._regime_logged = set()
        model._configure_pointmap_off()
        logger.info(
            "FlexPi: joint_video=%s joint_dino=%s joint_pointmap=%s "
            "flex_joint.enabled=%s",
            model.joint_video, model.joint_dino, model.joint_pointmap,
            model.flex_joint.enabled,
        )
        if model.flex_joint.enabled:
            logger.info(
                "FlexPi flex_joint: p_present_video=%.2f p_present_dino=%.2f "
                "p_present_pointmap=%.2f p_jv=%.2f p_jd=%.2f p_jp=%.2f "
                "cross_modal_predict_video=%s cross_modal_predict_dino=%s "
                "cross_modal_predict_pointmap=%s",
                model.flex_joint.p_present_video, model.flex_joint.p_present_dino,
                model.flex_joint.p_present_pointmap,
                model.flex_joint.p_jv, model.flex_joint.p_jd, model.flex_joint.p_jp,
                model.flex_joint.cross_modal_predict_video,
                model.flex_joint.cross_modal_predict_dino,
                model.flex_joint.cross_modal_predict_pointmap,
            )
        return model

    # ------------------------------------------------------------------
    # Attention mask — widen action rows + pairwise rem↔rem alignment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_joint_flag_deltas(
        self,
        mask: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        joint_video: bool,
        joint_dino: bool,
        joint_pointmap: bool,
    ) -> None:
        """In-place: apply XOR drops + action widening for the given joint flags.

        Shared between the legacy single-mask path and the per-sample flex path.
        """
        Sv, Sd, Sp = video_seq_len, dino_seq_len, pointmap_seq_len
        tpf, dtpf, ptpf = (
            video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
        )
        v_rem_start = tpf
        d_start, d_end = Sv, Sv + Sd
        d_rem_start = d_start + dtpf
        p_start, p_end = Sv + Sd, Sv + Sd + Sp
        p_rem_start = p_start + ptpf
        a_start = Sv + Sd + Sp

        # --- Pairwise rem↔rem alignment (XOR drop) ---
        if (joint_video != joint_dino) and tpf < Sv and dtpf < Sd:
            mask[..., v_rem_start:Sv, d_rem_start:d_end] = False
            mask[..., d_rem_start:d_end, v_rem_start:Sv] = False
        if (joint_video != joint_pointmap) and tpf < Sv and ptpf < Sp:
            mask[..., v_rem_start:Sv, p_rem_start:p_end] = False
            mask[..., p_rem_start:p_end, v_rem_start:Sv] = False
        if (joint_dino != joint_pointmap) and dtpf < Sd and ptpf < Sp:
            mask[..., d_rem_start:d_end, p_rem_start:p_end] = False
            mask[..., p_rem_start:p_end, d_rem_start:d_end] = False

        # --- Action-row widening ---
        if joint_video:
            mask[..., a_start:, :Sv] = True
        if joint_dino:
            mask[..., a_start:, d_start:d_end] = True
        if joint_pointmap:
            mask[..., a_start:, p_start:p_end] = True

    @torch.no_grad()
    def _apply_presence_absent_edits(
        self,
        mask: torch.Tensor,
        video_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        present_v: bool,
        present_d: bool,
        present_p: bool,
        cross_modal_v: bool,
        cross_modal_d: bool,
        cross_modal_p: bool,
    ) -> None:
        """In-place: kill rows/cols for absent streams (flex only).

        When a stream is absent:
        - ``cross_modal=False``: kill all of its rows/cols (no info flows).
        - ``cross_modal=True``: kill only its ff_X rows/cols (no past anchor),
          rem_X stays in the layout to be denoised from other streams.
        """
        Sv, Sd, Sp = video_seq_len, dino_seq_len, pointmap_seq_len
        tpf, dtpf, ptpf = (
            video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
        )
        if not present_v:
            if cross_modal_v:
                # ff_v is absent (zero); rem_v stays.
                mask[..., :tpf, :] = False
                mask[..., :, :tpf] = False
            else:
                # Whole video stream absent.
                mask[..., :Sv, :] = False
                mask[..., :, :Sv] = False
        if not present_d:
            d_start = Sv
            d_end = d_start + Sd
            if cross_modal_d:
                mask[..., d_start:d_start + dtpf, :] = False
                mask[..., :, d_start:d_start + dtpf] = False
            else:
                mask[..., d_start:d_end, :] = False
                mask[..., :, d_start:d_end] = False
        if not present_p:
            p_start = Sv + Sd
            p_end = p_start + Sp
            if cross_modal_p:
                mask[..., p_start:p_start + ptpf, :] = False
                mask[..., :, p_start:p_start + ptpf] = False
            else:
                mask[..., p_start:p_end, :] = False
                mask[..., :, p_start:p_end] = False

    @torch.no_grad()
    def _build_mot_attention_mask_unified(
        self,
        video_seq_len: int,
        action_seq_len: int,
        dino_seq_len: int,
        pointmap_seq_len: int,
        video_tokens_per_frame: int,
        dino_tokens_per_frame: int,
        pointmap_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        bf = getattr(self, "_batch_flex", None)
        cache_key = None
        if bf is None and getattr(self, "_glue_cache_enabled", False) and not self.training:
            # Inference mask is a pure function of shapes + device + the
            # runtime regime bits below (cm_* are per-instance config, fixed
            # for the cache's lifetime — prepare_for_inference resets it).
            cache_key = (
                "joint_mask", video_seq_len, action_seq_len, dino_seq_len,
                pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame,
                pointmap_tokens_per_frame, str(device),
                bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap),
                bool(self._infer_present_v), bool(self._infer_present_d),
                bool(self._infer_present_p),
            )
            hit = self._glue_cache.get(cache_key)
            if hit is not None:
                return hit
        base = self._base_build_mot_attention_mask_unified(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            dino_seq_len=dino_seq_len,
            pointmap_seq_len=pointmap_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            dino_tokens_per_frame=dino_tokens_per_frame,
            pointmap_tokens_per_frame=pointmap_tokens_per_frame,
            device=device,
        )
        if bf is None:
            # Inference / non-flex training: one [S, S] mask built from
            # Python scalar attrs. Dynamo specializes on ``self.joint_*`` +
            # ``self._infer_present_*`` (always bool, default True) so each
            # unique regime triggers its own torch.compile guard — no
            # .tolist() / CPU sync, CUDA-Graph capture friendly.
            self._apply_joint_flag_deltas(
                base, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len,
                video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
                bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap),
            )
            if (
                (not self._infer_present_v)
                or (not self._infer_present_d)
                or (not self._infer_present_p)
            ):
                self._apply_presence_absent_edits(
                    base, video_seq_len, dino_seq_len, pointmap_seq_len,
                    video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
                    present_v=bool(self._infer_present_v),
                    present_d=bool(self._infer_present_d),
                    present_p=bool(self._infer_present_p),
                    cross_modal_v=bool(self.flex_joint.cross_modal_predict_video),
                    cross_modal_d=bool(self.flex_joint.cross_modal_predict_dino),
                    cross_modal_p=bool(self.flex_joint.cross_modal_predict_pointmap),
                )
            if cache_key is not None:
                self._glue_cache[cache_key] = base
            return base

        # Flex path: build a per-sample [B, S, S] mask. Pull all per-sample
        # booleans to host in one shot (6 syncs total) to avoid 6*B per-sample
        # ``.item()`` syncs inside the loop.
        B = bf.B
        j_v_list = bf.j_v.tolist()
        j_d_list = bf.j_d.tolist()
        j_p_list = bf.j_p.tolist()
        present_v_list = bf.present_v.tolist()
        present_d_list = bf.present_d.tolist()
        present_p_list = bf.present_p.tolist()
        per_sample: list[torch.Tensor] = []
        for b in range(B):
            m_b = base.clone()
            self._apply_joint_flag_deltas(
                m_b, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len,
                video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
                bool(j_v_list[b]), bool(j_d_list[b]), bool(j_p_list[b]),
            )
            self._apply_presence_absent_edits(
                m_b, video_seq_len, dino_seq_len, pointmap_seq_len,
                video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame,
                present_v=bool(present_v_list[b]),
                present_d=bool(present_d_list[b]),
                present_p=bool(present_p_list[b]),
                cross_modal_v=bf.cm_v, cross_modal_d=bf.cm_d, cross_modal_p=bf.cm_p,
            )
            per_sample.append(m_b)
        return torch.stack(per_sample, dim=0)

    # ------------------------------------------------------------------
    # Train-unreachable combo warning (used by infer_action)
    # ------------------------------------------------------------------

    def _maybe_warn_unreachable_combo(self) -> None:
        """Warn once per stream if the effective deploy regime is one that
        training cannot reach.

        ``sample_flex_batch_flags`` enforces ``j_X &= (present_X | cm_X)``
        per-sample ([helpers/flex_joint.py:110-113](helpers/flex_joint.py:110)),
        so the combo ``(joint_X=True, present_X=False, cm_X=False)`` is
        IMPOSSIBLE at training: the model never sees a sample where rem_X is
        a denoise target while ff_X is absent. At deploy the mask still
        collapses correctly (presence-absent edits kill the entire X stream),
        but the user may have expected "imagine future X from non-X" semantics
        — which only the cross-modal regime (``cm_X=True``) provides.

        Reads the EFFECTIVE state (post joint/present override), so it fires
        whether the unreachable combo came from explicit user args or from
        the trained-default ``self.joint_*`` interacting with a present
        override.
        """
        if (self.joint_video
                and not self._infer_present_v
                and not self.flex_joint.cross_modal_predict_video
                and "video" not in self._unreachable_warned):
            logger.warning(
                "FlexPi deploy combo (joint_video=True, "
                "present_video=False, cross_modal_predict_video=False) is "
                "UNREACHABLE at training (sampler couples j_v &= present_v "
                "unless cm_v=True). Mask collapses to joint_video=False — "
                "rem_v stays in the layout but is fully isolated from "
                "attention, so no future-video imagination happens. To get "
                "cross-modal video imagination, retrain with "
                "flex_joint.cross_modal_predict_video=true. (warned once)"
            )
            self._unreachable_warned.add("video")
        if (self.joint_dino
                and not self._infer_present_d
                and not self.flex_joint.cross_modal_predict_dino
                and "dino" not in self._unreachable_warned):
            logger.warning(
                "FlexPi deploy combo (joint_dino=True, "
                "present_dino=False, cross_modal_predict_dino=False) is "
                "UNREACHABLE at training. Mask collapses to "
                "joint_dino=False. To get cross-modal dino imagination, "
                "retrain with flex_joint.cross_modal_predict_dino=true. "
                "(warned once)"
            )
            self._unreachable_warned.add("dino")
        if (self.joint_pointmap
                and not self._infer_present_p
                and not self.flex_joint.cross_modal_predict_pointmap
                and "pointmap" not in self._unreachable_warned):
            logger.warning(
                "FlexPi deploy combo (joint_pointmap=True, "
                "present_pointmap=False, cross_modal_predict_pointmap=False) "
                "is UNREACHABLE at training. Mask collapses to "
                "joint_pointmap=False. To get cross-modal pointmap "
                "imagination, retrain with "
                "flex_joint.cross_modal_predict_pointmap=true. (warned once)"
            )
            self._unreachable_warned.add("pointmap")

    def _warn_once_pointmap_override(self) -> None:
        """One-time note that a pointmap override was dropped on a 2D run."""
        if "override" in self._unreachable_warned:
            return
        logger.warning(
            "present_pointmap/joint_pointmap were requested but this run has no "
            "pointmap stream (enable_pointmap=False, or all three flex pointmap "
            "knobs off). Both are clamped to False — the pointmap head is frozen "
            "and training ran at Sp=0. (warned once)"
        )
        self._unreachable_warned.add("override")

    def _maybe_log_effective_regime(self) -> None:
        """Log the resolved deploy regime once per unique combination.

        Reads the effective state (post joint/present override) and emits
        a single line summarizing which streams are present in the input
        layout, which streams the model is denoising (future generation),
        and which absent-but-denoised streams rely on cross-modal imagination.
        Deduped on (joint_v, joint_d, joint_p, present_v, present_d, present_p)
        so eval loops only print once per distinct regime.
        """
        jv = bool(self.joint_video)
        jd = bool(self.joint_dino)
        jp = bool(self.joint_pointmap)
        pv = bool(self._infer_present_v)
        pd = bool(self._infer_present_d)
        pp = bool(self._infer_present_p)
        key = (jv, jd, jp, pv, pd, pp)
        if key in self._regime_logged:
            return
        self._regime_logged.add(key)
        cm_imagine_v = jv and (not pv) and bool(self.flex_joint.cross_modal_predict_video)
        cm_imagine_d = jd and (not pd) and bool(self.flex_joint.cross_modal_predict_dino)
        cm_imagine_p = jp and (not pp) and bool(self.flex_joint.cross_modal_predict_pointmap)
        logger.info(
            "FlexPi inference regime: "
            "present=(video=%s, dino=%s, pointmap=%s) | "
            "denoise=(action=True, video=%s, dino=%s, pointmap=%s) | "
            "cross_modal_imagine=(video=%s, dino=%s, pointmap=%s)",
            pv, pd, pp, jv, jd, jp, cm_imagine_v, cm_imagine_d, cm_imagine_p,
        )

    # ------------------------------------------------------------------
    # Training — flex_joint regime sampling
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        """Override to inject per-sample flex flags when enabled.

        When ``flex_joint.enabled=False`` this is a thin pass-through to
        ``FlexPi._base_training_loss`` (bit-identical legacy behavior).
        When enabled, samples per-sample presence + joint flags, stashes them
        on ``self._batch_flex``, then delegates. The parent's training_loss +
        the overridden mask builder consume the stashed flags.
        """
        if not self.flex_joint.enabled:
            return self._base_training_loss(sample, tiled=tiled)

        # Batch size is read from ``action`` (always required by the unified
        # input contract — see build_inputs).
        action = sample.get("action")
        if action is None:
            raise RuntimeError(
                "flex_joint requires `sample['action']` for batch size inference."
            )
        B = int(action.shape[0])
        bf = sample_flex_batch_flags(
            cfg=self.flex_joint, batch_size=B, device=self.device,
            pointmap_off=self._pointmap_globally_off,
        )
        self._batch_flex = bf
        try:
            return self._base_training_loss(sample, tiled=tiled)
        finally:
            self._batch_flex = None

    # ------------------------------------------------------------------
    # Inference dispatcher — route on the three flags
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image=None,
        action_horizon=None,
        num_video_frames: Optional[int] = None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        camera_intrinsics: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
        # Step-skipping (joint-path only; (F, F, F) silently ignores these).
        dynamic_step_skip: bool = False,
        step_skip_thresholds: Sequence[Tuple[float, int]] = ((0.95, 4), (0.93, 2)),
        step_skip_sim_sources: Optional[Sequence[str]] = None,
        step_skip_sim_aggregation: str = "mean",
        # When False, skip the D2H copies of the denoised video/dino/pointmap
        # latents in the returned dict (each is a blocking pageable transfer).
        # Deploy and eval read only ``out["action"]``, so this defaults False.
        return_stream_latents: bool = False,
        # Flex runtime overrides — only effective for models trained with
        # ``flex_joint.enabled=True``. ``None`` => keep the trained default.
        # Combined with the regime selected here, the model can be dispatched
        # into any of 8 joint regimes at deploy without re-instantiation.
        joint_video: Optional[bool] = None,
        joint_dino: Optional[bool] = None,
        joint_pointmap: Optional[bool] = None,
        # Flex runtime presence overrides — drop a stream from the input layout
        # at deploy (mirrors training-time ``p_present_*`` dropout). When False,
        # the stream's mask rows/cols are killed (and ``rem_X`` stays in the
        # layout to be denoised when ``cross_modal_predict_X=True``). Passing
        # any flag forces routing through the joint denoise path so the
        # per-sample mask builder fires — the fast KV-cache path does not
        # honor presence. ``None`` => no override (treated as present=True).
        present_video: Optional[bool] = None,
        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
    ):
        # Temporarily override self.joint_* when runtime flags are supplied.
        # Restore on exit so concurrent calls / future calls see the trained
        # defaults again.
        # A pointmap-off run has no pointmap stream to ask for: the head is
        # frozen and untrained, and training built the sequence at Sp=0. Honor
        # the caller's other overrides but clamp the two pointmap ones, so the
        # stock eval launchers (which pass INFER_{PRESENT,JOINT}_POINTMAP=true
        # for every model) cannot hand a 2D checkpoint a live 3D stream.
        if self._pointmap_globally_off:
            if present_pointmap or joint_pointmap:
                self._warn_once_pointmap_override()
            present_pointmap = False
            joint_pointmap = False
        _saved_joint = (self.joint_video, self.joint_dino, self.joint_pointmap)
        if joint_video is not None:
            self.joint_video = bool(joint_video)
        if joint_dino is not None:
            self.joint_dino = bool(joint_dino)
        if joint_pointmap is not None:
            self.joint_pointmap = bool(joint_pointmap)

        # Stash presence overrides as **Python scalar bools** on ``self``
        # (not as a per-sample tensor in ``_batch_flex``). The mask builder
        # reads these directly and applies presence-absent edits with scalar
        # bools — no ``.tolist()`` / CPU sync — so torch.compile +
        # CUDA-Graph capture works and Dynamo specializes on each unique
        # (joint_*, present_*) tuple (one fresh compile per regime, ≤32 max
        # entries across 8 joint × 4 presence combos). ``_batch_flex`` is
        # untouched at inference; it remains the training-only per-sample path.
        _saved_infer_present_v = self._infer_present_v
        _saved_infer_present_d = self._infer_present_d
        _saved_infer_present_p = self._infer_present_p
        if present_video is not None:
            self._infer_present_v = bool(present_video)
        if present_dino is not None:
            self._infer_present_d = bool(present_dino)
        if present_pointmap is not None:
            self._infer_present_p = bool(present_pointmap)
        # Force the joint denoise path only when the user has actually
        # requested joint denoising. The fast KV-cache path now honors all
        # three presence overrides (video drop → no-video route; dino drop →
        # skip dino encoder; pointmap drop → skip pointmap encoder), so a
        # pure presence drop with all joint_*=False can stay on the fast path.
        _force_joint_path = False
        # Warn once per stream if the effective regime is one that training
        # cannot reach (j_X=True + present_X=False + cm_X=False). Fires AFTER
        # overrides are applied so the check sees what the user actually
        # asked for (or what the trained-default + presence override resolve to).
        self._maybe_warn_unreachable_combo()
        self._maybe_log_effective_regime()
        try:
            return self._infer_action_dispatch(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                camera_intrinsics=camera_intrinsics,
                per_cam=per_cam,
                per_cam_depth=per_cam_depth,
                dynamic_step_skip=dynamic_step_skip,
                step_skip_thresholds=step_skip_thresholds,
                step_skip_sim_sources=step_skip_sim_sources,
                step_skip_sim_aggregation=step_skip_sim_aggregation,
                return_stream_latents=return_stream_latents,
                force_joint_path=_force_joint_path,
            )
        finally:
            self.joint_video, self.joint_dino, self.joint_pointmap = _saved_joint
            self._infer_present_v = _saved_infer_present_v
            self._infer_present_d = _saved_infer_present_d
            self._infer_present_p = _saved_infer_present_p

    def _infer_action_dispatch(
        self,
        prompt,
        input_image,
        action_horizon,
        num_video_frames,
        proprio,
        context,
        context_mask,
        negative_prompt,
        text_cfg_scale,
        num_inference_steps,
        sigma_shift,
        seed,
        rand_device,
        tiled,
        camera_intrinsics,
        per_cam,
        per_cam_depth,
        dynamic_step_skip,
        step_skip_thresholds,
        step_skip_sim_sources,
        step_skip_sim_aggregation,
        return_stream_latents: bool = False,
        force_joint_path: bool = False,
    ):
        any_joint = self.joint_video or self.joint_dino or self.joint_pointmap
        # Action-only uses the fast KV-cache path. Regime-FiLM is applied there
        # too (video offset at prefill, action offset in the cache denoiser), so
        # action-only keeps its low latency AND stays train/deploy matched —
        # no need to force the heavier joint forward.
        if not any_joint and not force_joint_path:
            return self._base_infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                camera_intrinsics=camera_intrinsics,
                per_cam=per_cam,
                per_cam_depth=per_cam_depth,
                present_video=self._infer_present_v,
                present_dino=self._infer_present_d,
                present_pointmap=self._infer_present_p,
            )

        if num_video_frames is None:
            raise ValueError(
                "`num_video_frames` is required when any joint_* flag is True "
                "or a presence override (present_video/present_dino/present_pointmap) is supplied."
            )
        return self._infer_action_joint(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            camera_intrinsics=camera_intrinsics,
            per_cam=per_cam,
            per_cam_depth=per_cam_depth,
            dynamic_step_skip=dynamic_step_skip,
            step_skip_thresholds=step_skip_thresholds,
            step_skip_sim_sources=step_skip_sim_sources,
            step_skip_sim_aggregation=step_skip_sim_aggregation,
            return_stream_latents=return_stream_latents,
        )

    # ------------------------------------------------------------------
    # Val-vis dispatcher — disable the action-only consistency check when
    # any joint flag is on. In joint regimes the action attends to noisy
    # future tokens, so its prediction is not expected to match the
    # action-only KV-cache path; the check would also pass an incompatible
    # signature to ``infer_action`` (no ``num_video_frames``). Mirrors the
    # pattern used by FlexPiJoint / FlexPiLatentJoint / FlexPi3DJoint.
    #
    # Delegates to ``FlexPi.infer_joint`` (the full 4-stream
    # rollout: video + DINO + pointmap + action).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_joint(
        self,
        prompt=None,
        input_image=None,
        num_video_frames=None,
        action_horizon=None,
        action=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        camera_intrinsics: Optional[torch.Tensor] = None,
        per_cam: Optional[Dict[str, torch.Tensor]] = None,
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None,
    ):
        any_joint = self.joint_video or self.joint_dino or self.joint_pointmap
        if any_joint and test_action_with_infer_action:
            logger.warning(
                "FlexPi.infer_joint: forcing test_action_with_infer_action=False "
                "(joint_video=%s joint_dino=%s joint_pointmap=%s) — joint regimes diverge from "
                "action-only by design.",
                self.joint_video, self.joint_dino, self.joint_pointmap,
            )
            test_action_with_infer_action = False
        return self._base_infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=test_action_with_infer_action,
            camera_intrinsics=camera_intrinsics,
            per_cam=per_cam,
            per_cam_depth=per_cam_depth,
        )

    # ------------------------------------------------------------------
    # Joint-regime action inference
    #
    # Runs the full MoT forward at every denoising step. Frozen streams
    # collapse to their single clean first-frame anchor — their scheduler
    # step is skipped; active streams are denoised jointly with action.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_action_joint(
        self,
        prompt,
        input_image,
        action_horizon,
        num_video_frames,
        proprio,
        context,
        context_mask,
        negative_prompt,
        text_cfg_scale,
        num_inference_steps,
        sigma_shift,
        seed,
        rand_device,
        tiled,
        camera_intrinsics,
        per_cam,
        per_cam_depth,
        dynamic_step_skip: bool = False,
        step_skip_thresholds: Sequence[Tuple[float, int]] = ((0.95, 4), (0.93, 2)),
        step_skip_sim_sources: Optional[Sequence[str]] = None,
        step_skip_sim_aggregation: str = "mean",
        return_stream_latents: bool = False,
    ):
        self.eval()
        if camera_intrinsics is not None:
            self.set_camera_intrinsics(camera_intrinsics)
        if per_cam_depth is None and bool(self._infer_present_p):
            raise ValueError(
                "`per_cam_depth` is required at inference for FlexPi "
                "when the pointmap stream is present. Pass present_pointmap=False "
                "(or use a pointmap-disabled model) to run without depth."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames,
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Latent shapes ---
        full_latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_latent_t = full_latent_t if self.joint_video else 1

        dino_denoised = bool(self.joint_dino)
        if dino_denoised:
            num_dino_frames = 1 + len(range(1, full_latent_t, self.dino_temporal_stride))
        else:
            num_dino_frames = 1

        denoise_pointmap = bool(self.joint_pointmap)
        if denoise_pointmap:
            num_pt_frames = full_latent_t
        else:
            # jp=False: pointmap collapses to a single length-1 stream. With the
            # no-anchor flag the slot contains random noise instead of the clean
            # anchor; action ignores it via the mask (ptpf_for_mask=0).
            num_pt_frames = 1

        # --- Random latents ---
        video_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        dino_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        pt_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        latents_video = _staged_randn(
            (1, self.vae.model.z_dim, video_latent_t, latent_h, latent_w),
            video_gen, rand_device, self.device, self.torch_dtype,
        )
        latents_action = _staged_randn(
            (1, action_horizon, self.action_expert.action_dim),
            action_gen, rand_device, self.device, self.torch_dtype,
        )
        latents_action = self._zero_padded_action(latents_action)

        n_patches_dino = sum(h * w for h, w in self.dino_cam_patches)
        latents_dino = _staged_randn(
            (1, self.dino_dim, num_dino_frames, n_patches_dino, 1),
            dino_gen, rand_device, self.device, self.torch_dtype,
        )

        # --- Encode first-frame anchors ---
        # Encoder-skip plumbing: when ``present_X=False`` at deploy, skip the
        # corresponding encoder forward. The attention mask (built downstream)
        # kills ff_X rows/cols (and rem_X rows/cols too when cm_X=False), so
        # the content of ff_X / rem_X tokens is invariant to the action output.
        # Skipping saves ~10–30 ms of one-time encoder compute per call without
        # changing the layout shape — torch.compile guards remain consistent.
        _skip_v = not bool(self._infer_present_v)
        _skip_d = not bool(self._infer_present_d)
        _skip_p = not bool(self._infer_present_p)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if _skip_v:
            # Skip VAE encode; leave latents_video[:, :, 0:1] as random init.
            # Mask kills ff_v rows/cols, so the content doesn't reach action.
            first_frame_latents = None
        else:
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled,
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # First-frame DINO (raw features for the latent tensor anchor).
        if _skip_d:
            # Skip DINO ViT forward; leave latents_dino[:, :, 0:1] as random
            # init. Mask kills ff_d rows/cols (and rem_d when cm_d=False).
            first_frame_dino = None
        else:
            if per_cam is not None:
                per_cam_5d: Dict[str, torch.Tensor] = {}
                for k, v in per_cam.items():
                    v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    if v.ndim == 4:
                        v = v.unsqueeze(2)
                    per_cam_5d[k] = v
                first_frame_dino = self.dino_encoder.encode_video(
                    video=None, per_cam=per_cam_5d,
                    concat_mode="robotwin", **self._dino_encode_kwargs(), first_frame_only=True,
                )
            else:
                first_frame_dino = self.dino_encoder.encode_video(
                    input_image.unsqueeze(2), concat_mode="robotwin", **self._dino_encode_kwargs(), first_frame_only=True,
                )
            latents_dino[:, :, 0:1] = first_frame_dino

        # First-frame pointmap (raw; shape depends on mode). Skipping the
        # encoder requires reconstructing the latents_pointmap shape from the
        # model attrs since we no longer have the encoded ff_p tensor to read it from.
        if _skip_p:
            first_frame_pointmap = None
            # Latent shape mirrors the video VAE latents.
            latents_pointmap = _staged_randn(
                (1, self.vae.model.z_dim, num_pt_frames, latent_h, latent_w),
                pt_gen, rand_device, self.device, self.torch_dtype,
            )
        else:
            first_frame_pointmap = self._encode_first_frame_pointmap_raw(
                tiled=tiled, per_cam_depth=per_cam_depth,
            )
            _, pt_C, _, pt_H, pt_W = first_frame_pointmap.shape
            latents_pointmap = _staged_randn(
                (1, pt_C, num_pt_frames, pt_H, pt_W),
                pt_gen, rand_device, self.device, self.torch_dtype,
            )
            latents_pointmap[:, :, 0:1] = first_frame_pointmap

        # --- Text context ---
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is None and context is None:
            raise ValueError("Either `prompt` or `context/context_mask` must be provided.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # --- Schedulers ---
        # When the video stream is disabled the video scheduler is unused;
        # build it against latents_action.dtype to avoid `None.dtype`.
        video_dtype = (
            latents_video.dtype if latents_video is not None else latents_action.dtype
        )
        ts_video, deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=video_dtype, shift_override=sigma_shift,
        )
        ts_action, deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        ts_dino, deltas_dino = self.infer_dino_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_dino.dtype, shift_override=sigma_shift,
        )
        ts_pt, deltas_pt = self.infer_pointmap_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_pointmap.dtype, shift_override=sigma_shift,
        )

        # --- Step-skip controller ---
        # Auto-pick (explicit=None) returns ALL denoised streams; combined with
        # the default ``sim_aggregation="mean"`` the controller averages
        # per-stream cos-sims. Under flex this gives a modality-balanced skip
        # signal that adapts to whichever regime the runtime override selected.
        sim_sources = resolve_sim_sources(
            explicit=step_skip_sim_sources,
            denoised_flags={
                "video": bool(self.joint_video),
                "dino": dino_denoised,
                "pointmap": denoise_pointmap,
            },
        )
        skip_ctrl = StepSkipController(
            enabled=dynamic_step_skip and bool(sim_sources),
            thresholds=step_skip_thresholds,
            sim_aggregation=step_skip_sim_aggregation,
        )

        # --- DINO RoPE freqs (static across denoise steps; hoisted out of the
        # compiled per-step body so torch.compile + reduce-overhead can capture
        # a CUDA Graph without a CPU tensor leak from `compute_dino_freqs`).
        dino_freqs_precomputed = self._compute_dino_freqs(
            latents_dino.shape[2], latents_dino.device,
        )
        # Pointmap RoPE freqs are NOT hoisted: they need ``pt_meta`` from
        # ``_embed_pointmap`` (the VAE-encoded latent grid), so they stay inside.
        pt_freqs_precomputed = None

        # --- FlexAttention backend (attn_backend="flex") ---
        # Replace the dense-mask SDPA joint attention (memory-efficient
        # backend) with FlexAttention over a per-regime BlockMask. The
        # BlockMask is built once per (regime, seq-len) and served through the
        # class-level ``MoT.attention_mask`` exactly like the training flex
        # path; HBridge outer-layer sub-masks stay on SDPA.
        flex_attn_active = bool(
            getattr(self, "_infer_attn_backend", "sdpa") == "flex"
        )
        if flex_attn_active:
            self._prepare_joint_flex_block_mask(
                latents_video, latents_dino, latents_pointmap, action_horizon,
            )

        # --- Denoising loop ---
        # torch_compile_scope="loop": the entire loop (all steps + scheduler
        # updates + frame-0 re-clamps) is one compiled graph — no Python
        # between steps. Requires step-skip off (its cos-sim decisions are
        # data-dependent host control flow). Identical math to the loop below.
        _loop_kwargs = dict(
            latents_video=latents_video,
            latents_dino=latents_dino,
            latents_pointmap=latents_pointmap,
            latents_action=latents_action,
            ts_video=ts_video, deltas_video=deltas_video,
            ts_action=ts_action, deltas_action=deltas_action,
            ts_dino=ts_dino, deltas_dino=deltas_dino,
            ts_pt=ts_pt, deltas_pt=deltas_pt,
            context=context, context_mask=context_mask,
            fuse_flag=fuse_flag,
            dino_denoised=dino_denoised,
            denoise_pointmap=denoise_pointmap,
            first_frame_latents=first_frame_latents,
            first_frame_dino=first_frame_dino,
            first_frame_pointmap=first_frame_pointmap,
            dino_freqs=dino_freqs_precomputed,
            pt_freqs=pt_freqs_precomputed,
            flex_block_attention=flex_attn_active,
        )
        # joint_loop_cuda_graph: manual whole-loop CUDA graph (Euler-only, the
        # body's fixed contract). Engages only when every active scheduler is
        # Euler; a non-Euler solver needs the per-step timestep= path below.
        _loop_graph = (
            getattr(self, "_joint_loop_cuda_graph_enabled", False)
            and not skip_ctrl.enabled
            and all(
                getattr(sch, "solver", "euler") == "euler"
                for sch in (
                    self.infer_action_scheduler, self.infer_video_scheduler,
                    self.infer_dino_scheduler, self.infer_pointmap_scheduler,
                )
                if sch is not None
            )
        )
        if _loop_graph:
            latents_video, latents_dino, latents_pointmap, latents_action = (
                self._joint_loop_graph_runner(**_loop_kwargs)
            )
            ts_video = ts_video[:0]  # loop below is skipped (empty schedule)
            ts_action = ts_action[:0]
            ts_dino = ts_dino[:0]
            ts_pt = ts_pt[:0]
        elif self._use_loop_compile() and not skip_ctrl.enabled:
            latents_video, latents_dino, latents_pointmap, latents_action = (
                self._run_joint_denoise_loop(**_loop_kwargs)
            )
            ts_video = ts_video[:0]  # loop below is skipped (empty schedule)
            ts_action = ts_action[:0]
            ts_dino = ts_dino[:0]
            ts_pt = ts_pt[:0]
        for step_tv, delta_v, step_ta, delta_a, step_td, delta_d, step_tp, delta_p in zip(
            ts_video, deltas_video, ts_action, deltas_action,
            ts_dino, deltas_dino, ts_pt, deltas_pt,
        ):
            if skip_ctrl.should_run():
                # Frozen streams → pre_dit with t=0 so tokens are clean anchors.
                _video_dtype = latents_video.dtype
                t_video = (
                    step_tv.unsqueeze(0).to(dtype=_video_dtype, device=self.device)
                    if self.joint_video
                    else torch.zeros((1,), dtype=_video_dtype, device=self.device)
                )
                t_dino = (
                    step_td.unsqueeze(0).to(dtype=latents_dino.dtype, device=self.device)
                    if dino_denoised
                    else torch.zeros((1,), dtype=latents_dino.dtype, device=self.device)
                )
                t_pt = (
                    step_tp.unsqueeze(0).to(dtype=latents_pointmap.dtype, device=self.device)
                    if denoise_pointmap
                    else torch.zeros((1,), dtype=latents_pointmap.dtype, device=self.device)
                )
                t_action = step_ta.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                pred_video, pred_dino, pred_pointmap, pred_action = (
                    self._predict_joint_noise_unified(
                        latents_video=latents_video,
                        latents_dino=latents_dino,
                        latents_pointmap=latents_pointmap,
                        latents_action=latents_action,
                        timestep_video=t_video,
                        timestep_dino=t_dino,
                        timestep_pointmap=t_pt,
                        timestep_action=t_action,
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                        gt_action=None,
                        dino_freqs=dino_freqs_precomputed,
                        pt_freqs=pt_freqs_precomputed,
                        flex_block_attention=flex_attn_active,
                    )
                )

                _src_tensors = {
                    "video": pred_video, "dino": pred_dino, "pointmap": pred_pointmap,
                }
                sim_refs = [_src_tensors[s] for s in sim_sources]
                skip_ctrl.record(
                    sim_refs=sim_refs,
                    preds=(pred_video, pred_dino, pred_pointmap, pred_action),
                )
            else:
                pred_video, pred_dino, pred_pointmap, pred_action = skip_ctrl.cached()

            # timestep=… engages the DPM-Solver++(2M) multistep when the
            # scheduler's solver knob is on (default euler → timestep ignored,
            # exact prior update). This eager/per-step joint loop (also the TRT-
            # engine path, which runs torch_compile=false) is where the solver
            # lives; the loop-scope-compiled `_run_joint_denoise_loop` stays
            # Euler (a per-step host sync would break its CUDA-graph capture).
            latents_action = self.infer_action_scheduler.step(
                pred_action, delta_a, latents_action, timestep=t_action,
            )
            latents_action = self._zero_padded_action(latents_action)
            if self.joint_video:
                latents_video = self.infer_video_scheduler.step(
                    pred_video, delta_v, latents_video, timestep=t_video,
                )
                # Re-clamp ff_v to the clean anchor only when we have one;
                # under present_v=False we skipped the encoder, so there's
                # no anchor to clamp to. Mask kills ff_v rows/cols anyway.
                if first_frame_latents is not None:
                    latents_video[:, :, 0:1] = first_frame_latents.clone()
            if dino_denoised:
                latents_dino = self.infer_dino_scheduler.step(
                    pred_dino, delta_d, latents_dino, timestep=t_dino,
                )
                if first_frame_dino is not None:
                    latents_dino[:, :, 0:1] = first_frame_dino.clone()
            if denoise_pointmap:
                latents_pointmap = self.infer_pointmap_scheduler.step(
                    pred_pointmap, delta_p, latents_pointmap, timestep=t_pt,
                )
                if first_frame_pointmap is not None:
                    latents_pointmap[:, :, 0:1] = first_frame_pointmap.clone()

        if skip_ctrl.enabled:
            sims_fmt = "[" + ", ".join(
                "{" + ", ".join(f"{name}: {v:.4f}" for name, v in zip(sim_sources, step_sims))
                + "}→" + decision
                for step_sims, decision in zip(
                    skip_ctrl.per_stream_history, skip_ctrl.decision_history,
                )
            ) + "]"
            print(
                f"[step-skip] {skip_ctrl.steps_run}/{num_inference_steps} forwards run "
                f"({num_inference_steps - skip_ctrl.steps_run} skipped, "
                f"sources={sim_sources}, agg={step_skip_sim_aggregation}, sims={sims_fmt})"
            )

        out = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        # Expose denoised stream latents for val-time visualization. None when
        # the stream is frozen (joint_* False) — caller treats None as "no
        # prediction, use first-frame anchor as placeholder". Deploy passes
        # return_stream_latents=False: it reads only ``action``, and each of
        # these D2H copies is a blocking pageable transfer.
        if return_stream_latents:
            if self.joint_video and latents_video is not None:
                out["video_latents"] = latents_video.detach().to(device="cpu", dtype=torch.float32)
            if dino_denoised:
                out["dino_latents"] = latents_dino.detach().to(device="cpu", dtype=torch.float32)
            if denoise_pointmap:
                out["pointmap_latents"] = latents_pointmap.detach().to(device="cpu", dtype=torch.float32)
        return out

    # ------------------------------------------------------------------
    # torch_compile_scope="loop" — whole-denoise-loop compile for the joint
    # path. The body unrolls all steps (impl forward + scheduler updates +
    # frame-0 re-clamps) so reduce-overhead captures ONE CUDA Graph per call.
    # Bypasses the per-step ``_predict_joint_noise_unified`` dispatcher and
    # calls the raw ``_impl`` directly; math is identical to the eager loop.
    # ------------------------------------------------------------------

    def _joint_denoise_loop_body(
        self,
        latents_video: Optional[torch.Tensor],
        latents_dino: torch.Tensor,
        latents_pointmap: torch.Tensor,
        latents_action: torch.Tensor,
        ts_video: torch.Tensor, deltas_video: torch.Tensor,
        ts_action: torch.Tensor, deltas_action: torch.Tensor,
        ts_dino: torch.Tensor, deltas_dino: torch.Tensor,
        ts_pt: torch.Tensor, deltas_pt: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
        dino_denoised: bool,
        denoise_pointmap: bool,
        first_frame_latents: Optional[torch.Tensor],
        first_frame_dino: Optional[torch.Tensor],
        first_frame_pointmap: Optional[torch.Tensor],
        dino_freqs: Optional[torch.Tensor],
        pt_freqs: Optional[torch.Tensor],
        flex_block_attention: bool = False,
    ):
        device = latents_action.device
        for i in range(ts_action.shape[0]):
            _video_dtype = (
                latents_video.dtype if latents_video is not None else latents_action.dtype
            )
            t_video = (
                ts_video[i].reshape(1).to(dtype=_video_dtype)
                if self.joint_video
                else torch.zeros((1,), dtype=_video_dtype, device=device)
            )
            t_dino = (
                ts_dino[i].reshape(1).to(dtype=latents_dino.dtype)
                if dino_denoised
                else torch.zeros((1,), dtype=latents_dino.dtype, device=device)
            )
            t_pt = (
                ts_pt[i].reshape(1).to(dtype=latents_pointmap.dtype)
                if denoise_pointmap
                else torch.zeros((1,), dtype=latents_pointmap.dtype, device=device)
            )
            t_action = ts_action[i].reshape(1).to(dtype=latents_action.dtype)

            pred_video, pred_dino, pred_pointmap, pred_action = (
                self._predict_joint_noise_unified_impl(
                    latents_video=latents_video,
                    latents_dino=latents_dino,
                    latents_pointmap=latents_pointmap,
                    latents_action=latents_action,
                    timestep_video=t_video,
                    timestep_dino=t_dino,
                    timestep_pointmap=t_pt,
                    timestep_action=t_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    gt_action=None,
                    dino_freqs=dino_freqs,
                    pt_freqs=pt_freqs,
                    flex_block_attention=flex_block_attention,
                )
            )

            latents_action = self.infer_action_scheduler.step(
                pred_action, deltas_action[i], latents_action,
            )
            latents_action = self._zero_padded_action(latents_action)
            if self.joint_video:
                latents_video = self.infer_video_scheduler.step(
                    pred_video, deltas_video[i], latents_video,
                )
                if first_frame_latents is not None:
                    latents_video[:, :, 0:1] = first_frame_latents.clone()
            if dino_denoised:
                latents_dino = self.infer_dino_scheduler.step(
                    pred_dino, deltas_dino[i], latents_dino,
                )
                if first_frame_dino is not None:
                    latents_dino[:, :, 0:1] = first_frame_dino.clone()
            if denoise_pointmap:
                latents_pointmap = self.infer_pointmap_scheduler.step(
                    pred_pointmap, deltas_pt[i], latents_pointmap,
                )
                if first_frame_pointmap is not None:
                    latents_pointmap[:, :, 0:1] = first_frame_pointmap.clone()
        return latents_video, latents_dino, latents_pointmap, latents_action

    @torch.no_grad()
    def _run_joint_denoise_loop(self, **kwargs):
        if not getattr(self, "_joint_loop_is_compiled", False):
            self._joint_denoise_loop_compiled = self._compile_for_inference(
                self._joint_denoise_loop_body,
            )
            self._joint_loop_is_compiled = True
        # attn_backend="auto": cuDNN-first priority for the masked joint
        # attention (no-op ctx otherwise); choice is baked at trace/capture.
        with self._sdpa_priority_ctx():
            return self._joint_denoise_loop_compiled(**kwargs)

    @torch.no_grad()
    def _prepare_joint_flex_block_mask(
        self,
        latents_video: Optional[torch.Tensor],
        latents_dino: torch.Tensor,
        latents_pointmap: torch.Tensor,
        action_horizon: int,
    ) -> None:
        """attn_backend="flex": build (or fetch from the per-regime cache) the
        BlockMask for the joint full-attention mask and install it as the
        class-level ``MoT.attention_mask``; switch the MoT to attn_mode="flex"
        so ``_mixed_attention`` routes mask-less calls to FlexAttention.

        Seq-len derivation mirrors ``_predict_joint_noise_unified_impl``
        (token counts are pure functions of latent shapes + patch size).
        """
        import torch.nn.functional as _F

        p = getattr(self.video_expert, "patch_size", (1, 2, 2))
        tpf = (latents_video.shape[-2] // p[1]) * (latents_video.shape[-1] // p[2])
        Sv = int(latents_video.shape[2]) * tpf
        dtpf = int(latents_dino.shape[3])
        Sd = int(latents_dino.shape[2]) * dtpf
        ptpf = (latents_pointmap.shape[-2] // p[1]) * (latents_pointmap.shape[-1] // p[2])
        Sp = int(latents_pointmap.shape[2]) * ptpf
        Sa = int(action_horizon)

        effective_dtpf = dtpf
        effective_ptpf = ptpf
        Sp_for_mask = Sp
        ptpf_for_mask = effective_ptpf
        total = Sv + Sd + Sp_for_mask + Sa

        key = (
            total,
            bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap),
            bool(self._infer_present_v), bool(self._infer_present_d), bool(self._infer_present_p),
        )
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = {}
            self._flex_block_mask_cache = cache
        bm = cache.get(key)
        if bm is None:
            from torch.nn.attention.flex_attention import create_block_mask

            dense = self._build_mot_attention_mask_unified(
                Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, self.device,
            )
            pad_total = (total + 127) // 128 * 128
            dense_padded = _F.pad(dense, (0, pad_total - total, 0, pad_total - total))

            def _mask_mod(b, h, q_idx, kv_idx):
                return dense_padded[q_idx, kv_idx]

            bm = create_block_mask(
                _mask_mod, B=None, H=None, Q_LEN=pad_total, KV_LEN=pad_total,
                device=str(self.device),
            )
            cache[key] = bm
            logger.info(
                "[flex-attn] built BlockMask for regime key=%s (S=%d, padded=%d)",
                key, total, pad_total,
            )
        type(self.mot).attention_mask = bm
        if self.mot.attn_mode != "flex":
            self.mot.set_attn_mode("flex")
