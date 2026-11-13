from __future__ import annotations
from .inference_opt.step_skip import StepSkipController, resolve_sim_sources
import copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from cloudrobo_wm.utils.logging_config import get_logger
from .dino_encoder import DinoEncoder
from .helpers.dino import _DINO_X0_SIGMA_MIN, _dino_x0_to_velocity, _zero_init_dino_x0_head_, compute_dino_freqs
from .mot import MoT
from .pointmap_encoder import PointmapEncoder
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)
from typing import Dict, Optional, Sequence, Tuple
from cloudrobo_wm.utils.pytorch_utils import _staged_randn
from .backbone import CloudRoboWMBackbone
from .helpers.flex_joint import FlexJointConfig

class CloudRoboWM(CloudRoboWMBackbone):
    FROZEN_MODULES: set[str] = {'vae', 'text_encoder', 'dino_encoder', 'pointmap_encoder'}

    def _base_init(self, video_expert, action_expert, mot: MoT, vae, dino_encoder: DinoEncoder, pointmap_encoder: PointmapEncoder, text_encoder=None, tokenizer=None, text_dim: Optional[int]=None, proprio_dim: Optional[int]=None, device: str='cpu', torch_dtype: torch.dtype=torch.float32, video_train_shift: float=5.0, video_infer_shift: float=5.0, video_num_train_timesteps: int=1000, action_train_shift: float=5.0, action_infer_shift: float=5.0, action_num_train_timesteps: int=1000, loss_lambda_video: float=1.0, loss_lambda_action: float=1.0, dino_dim: int=768, dino_train_shift: float=5.0, dino_infer_shift: float=5.0, dino_num_train_timesteps: int=1000, loss_lambda_dino: float=1.0, dino_cam_regions: list | None=None, dino_cam_patches: list | None=None, dino_temporal_stride: int=1, dino_stride_keep_far: bool=False, dino_pred_x0: bool=False, dino_pool_mode: str='avg', freeze_dino_encoder: bool=True, dino_pixel_unshuffle: int=0, pointmap_train_shift: float=5.0, pointmap_infer_shift: float=5.0, pointmap_num_train_timesteps: int=1000, loss_lambda_pointmap: float=1.0, pointmap_depth_vis_mode: str='turbo', composite_layout=None, composite_layout_slot_key_map=None, dino_pool_factor: int=1):
        from cloudrobo_wm.composite_layouts import get_layout
        _layout_resolved = get_layout(composite_layout)
        super().__init__(video_expert=video_expert, action_expert=action_expert, mot=mot, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, text_dim=text_dim, proprio_dim=proprio_dim, composite_layout=composite_layout, composite_layout_slot_key_map=composite_layout_slot_key_map, device=device, torch_dtype=torch_dtype, video_train_shift=video_train_shift, video_infer_shift=video_infer_shift, video_num_train_timesteps=video_num_train_timesteps, action_train_shift=action_train_shift, action_infer_shift=action_infer_shift, action_num_train_timesteps=action_num_train_timesteps, loss_lambda_video=loss_lambda_video, loss_lambda_action=loss_lambda_action)
        self.dino_encoder = dino_encoder
        self.dino_dim = int(dino_dim)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.dino_temporal_stride = int(dino_temporal_stride)
        self.dino_stride_keep_far = bool(dino_stride_keep_far)
        self.dino_pred_x0 = bool(dino_pred_x0)
        if self.dino_pred_x0:
            logger.info('CloudRoboWM: dino_pred_x0=True — DINO head output is x̂0; v̂ = (x_t − x̂0)/σ is derived analytically for loss and inference.')
        self.dino_pool_mode = str(dino_pool_mode)
        _grid_factor = dino_pool_factor if dino_pool_factor != 1 else max(int(dino_pixel_unshuffle), 1)
        if dino_cam_patches is None and dino_cam_regions is None and (_grid_factor != 1):
            patches, regions, _ = _layout_resolved.with_dino_pool(_grid_factor)
            self.dino_cam_patches = [tuple(p) for p in patches]
            self.dino_cam_regions = [tuple(r) for r in regions]
        else:
            self.dino_cam_regions = dino_cam_regions or [tuple(r) for r in _layout_resolved.dino_cam_regions]
            self.dino_cam_patches = dino_cam_patches or [tuple(p) for p in _layout_resolved.dino_cam_patches]
        self._dino_per_view_dim = self.dino_dim
        self._dino_encoder_cam_patches = None
        self._dino_pixel_unshuffle = int(dino_pixel_unshuffle)
        if self._dino_pixel_unshuffle:
            f = self._dino_pixel_unshuffle
            self._dino_encoder_cam_patches = [(h * f, w * f) for h, w in self.dino_cam_patches]
            ps = int(self.dino_encoder.patches_per_side)
            if any((g != (ps, ps) for g in self._dino_encoder_cam_patches)):
                raise ValueError(f"dino_pixel_unshuffle={f} is lossless only when each view's post-fold grid × {f} equals the ViT native grid {ps}×{ps} (no pooling before the fold); got encoder grids {self._dino_encoder_cam_patches}. Use a composite_layout whose per-cam DINO grid is a uniform {ps // f}×{ps // f} (e.g. tshape_robotwin_384x320_uniform) so each view folds {ps}×{ps} → {ps // f}×{ps // f}, or set dino_pixel_unshuffle=0.")
            self.dino_dim = f * f * int(dino_dim)
            self._dino_per_view_dim = self.dino_dim
        self.pointmap_encoder = pointmap_encoder
        self.loss_lambda_pointmap = float(loss_lambda_pointmap)
        self.pointmap_depth_vis_mode = str(pointmap_depth_vis_mode)
        self._camera_intrinsics: Optional[torch.Tensor] = None
        self._batch_flex = None
        frozen = {'vae', 'text_encoder', 'pointmap_encoder'}
        if freeze_dino_encoder:
            frozen.add('dino_encoder')
        self.FROZEN_MODULES = frozen
        video_hidden_dim = self.video_expert.hidden_dim
        self.dino_feature_norm = nn.LayerNorm(self._dino_per_view_dim)
        self.dino_embedder = nn.Linear(self.dino_dim, video_hidden_dim)
        self.dino_proj_out = nn.Linear(video_hidden_dim, self.dino_dim)
        self._init_dino_layers()
        self._build_pointmap_layers()
        self.train_dino_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=dino_num_train_timesteps, shift=dino_train_shift)
        self.infer_dino_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=dino_num_train_timesteps, shift=dino_infer_shift)
        self.train_pointmap_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=pointmap_num_train_timesteps, shift=pointmap_train_shift)
        self.infer_pointmap_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=pointmap_num_train_timesteps, shift=pointmap_infer_shift)
        self.to(device=self.device, dtype=self.torch_dtype)
        logger.info('CloudRoboWM: dino_temporal_stride=%d dino_stride_keep_far=%s', self.dino_temporal_stride, self.dino_stride_keep_far)

    def _init_dino_layers(self):

        def _xavier(module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        _xavier(self.dino_embedder)
        _xavier(self.dino_proj_out)
        if self.dino_pred_x0:
            _zero_init_dino_x0_head_(self.dino_proj_out)
        nn.init.ones_(self.dino_feature_norm.weight)
        nn.init.zeros_(self.dino_feature_norm.bias)

    def _build_pointmap_layers(self):
        self.pt_patch_embedding = copy.deepcopy(self.video_expert.patch_embedding)
        self.pt_head = copy.deepcopy(self.video_expert.head)

    @property
    def _pointmap_globally_off(self) -> bool:
        if not bool(getattr(self, 'enable_pointmap', True)):
            return True
        fj = getattr(self, 'flex_joint', None)
        if fj is None or not getattr(fj, 'enabled', False):
            return False
        return float(fj.p_present_pointmap) == 0.0 and float(fj.p_jp) == 0.0 and (not bool(fj.cross_modal_predict_pointmap))

    def _configure_pointmap_off(self) -> None:
        if not self._pointmap_globally_off:
            return
        self.FROZEN_MODULES = set(self.FROZEN_MODULES) | {'pt_patch_embedding', 'pt_head'}
        if hasattr(self, '_infer_present_p'):
            self._infer_present_p = False
        self.joint_pointmap = False
        self.flex_joint.cross_modal_predict_pointmap = False
        reason = 'enable_pointmap=False' if not bool(getattr(self, 'enable_pointmap', True)) else 'p_present_pointmap=0, p_jp=0, cross_modal_predict_pointmap=False'
        logger.info('Pointmap stream GLOBALLY OFF (%s): skipping depth encode + pointmap tokens + loss in training; inference defaults to present_pointmap=False. Pointmap head modules frozen.', reason)

    def _embed_dino(self, dino_features: torch.Tensor) -> torch.Tensor:
        x = dino_features.squeeze(-1).permute(0, 2, 3, 1).reshape(dino_features.shape[0], -1, self.dino_dim)
        x = self.dino_feature_norm(x)
        return self.dino_embedder(x)

    def _project_dino_out(self, dino_tokens_out: torch.Tensor) -> torch.Tensor:
        return self.dino_proj_out(dino_tokens_out)

    def _dino_encode_kwargs(self) -> dict:
        out = super()._dino_encode_kwargs()
        if self._dino_pixel_unshuffle:
            out['cam_patches'] = self._dino_encoder_cam_patches
            out['pixel_unshuffle'] = self._dino_pixel_unshuffle
        if self.dino_stride_keep_far:
            out['stride_keep_far'] = True
        return out

    def _glue_memo(self, key, build):
        if not (getattr(self, '_glue_cache_enabled', False) and (not self.training)):
            return build()
        hit = self._glue_cache.get(key)
        if hit is None:
            hit = self._glue_cache[key] = build()
        return hit

    def _compute_dino_freqs(self, num_frames: int, device: torch.device) -> torch.Tensor:
        return self._glue_memo(('dino_freqs', int(num_frames), str(device)), lambda: compute_dino_freqs(freqs_3d=self.video_expert.freqs, n_frames=num_frames, cam_regions=self.dino_cam_regions, cam_patches=self.dino_cam_patches, device=device))

    @torch.no_grad()
    def _encode_first_frame_dino(self, input_image: Optional[torch.Tensor]=None, per_cam: Optional[Dict[str, torch.Tensor]]=None) -> torch.Tensor:
        if per_cam is None and input_image is None:
            raise ValueError('_encode_first_frame_dino requires either `input_image` or `per_cam`.')
        if per_cam is not None:
            per_cam_5d: Dict[str, torch.Tensor] = {}
            for k, v in per_cam.items():
                v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                if v.ndim == 4:
                    v = v.unsqueeze(2)
                per_cam_5d[k] = v
            dino_feat = self.dino_encoder.encode_video(video=None, per_cam=per_cam_5d, concat_mode='robotwin', **self._dino_encode_kwargs(), first_frame_only=True)
        else:
            video_single = input_image.unsqueeze(2)
            dino_feat = self.dino_encoder.encode_video(video_single, concat_mode='robotwin', **self._dino_encode_kwargs(), first_frame_only=True)
        return self._embed_dino(dino_feat)

    def _embed_pointmap(self, pointmap_raw: torch.Tensor):
        x = self.pt_patch_embedding(pointmap_raw)
        f, h, w = x.shape[2:]
        tokens = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        ptpf = h * w
        pt_meta = {'grid_size': (int(f), int(h), int(w))}
        return (tokens, ptpf, pt_meta)

    def _project_pointmap_out(self, pt_tokens_out: torch.Tensor, pt_time: torch.Tensor, pt_meta: dict) -> torch.Tensor:
        f, h, w = pt_meta['grid_size']
        x = self.pt_head(pt_tokens_out, pt_time)
        ps = self.video_expert.patch_size
        return rearrange(x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)', f=f, h=h, w=w, x=ps[0], y=ps[1], z=ps[2])

    def _compute_pointmap_freqs(self, num_frames: int, device: torch.device, pt_meta: Optional[dict]=None, start_frame_idx: int=0) -> torch.Tensor:
        grid = tuple(pt_meta['grid_size']) if pt_meta is not None and 'grid_size' in pt_meta else None
        return self._glue_memo(('pt_freqs', int(num_frames), str(device), grid, int(start_frame_idx)), lambda: self._compute_pointmap_freqs_impl(num_frames, device, pt_meta=pt_meta, start_frame_idx=start_frame_idx))

    def _compute_pointmap_freqs_impl(self, num_frames: int, device: torch.device, pt_meta: Optional[dict]=None, start_frame_idx: int=0) -> torch.Tensor:
        assert pt_meta is not None, 'pointmap freqs require pt_meta with grid_size'
        f, h, w = pt_meta['grid_size']
        freqs_3d = self.video_expert.freqs
        t_lo, t_hi = (start_frame_idx, start_frame_idx + f)
        return torch.cat([freqs_3d[0][t_lo:t_hi].view(f, 1, 1, -1).expand(f, h, w, -1), freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(device)

    def set_camera_intrinsics(self, camera_intrinsics: torch.Tensor) -> None:
        if camera_intrinsics.ndim == 4 and (not getattr(self.pointmap_encoder, 'supports_batched_intrinsics', False)):
            camera_intrinsics = camera_intrinsics[0]
        self._camera_intrinsics = camera_intrinsics.to(device=self.device)

    def _resolve_camera_intrinsics(self, explicit: Optional[torch.Tensor]=None) -> torch.Tensor:
        K = explicit if explicit is not None else self._camera_intrinsics
        if K is None:
            raise RuntimeError("No camera intrinsics available. Attach them per-sample via `sample['camera_intrinsics']` or call `model.set_camera_intrinsics(K)` before inference.")
        return K

    @torch.no_grad()
    def _encode_first_frame_pointmap_raw(self, camera_intrinsics: Optional[torch.Tensor]=None, tiled: bool=False, per_cam_depth: Optional[Dict[str, torch.Tensor]]=None) -> torch.Tensor:
        if per_cam_depth is None:
            raise ValueError('_encode_first_frame_pointmap_raw requires `per_cam_depth`.')
        K = self._resolve_camera_intrinsics(camera_intrinsics)
        per_cam_depth_4d: Dict[str, torch.Tensor] = {}
        for k, v in per_cam_depth.items():
            v = v.to(device=self.device, non_blocking=True)
            if v.ndim == 2:
                v = v.unsqueeze(0).unsqueeze(0)
            elif v.ndim == 3:
                v = v.unsqueeze(1)
            elif v.ndim != 4:
                raise ValueError(f"per_cam_depth['{k}'] must be 2/3/4-d; got {tuple(v.shape)}")
            per_cam_depth_4d[k] = v
        comp = self.pointmap_encoder.encode_composite(per_cam_depth=per_cam_depth_4d, camera_intrinsics=K, concat_mode='robotwin', **self._layout_kwargs(), first_frame_only=True)
        return self._encode_video_latents(comp, tiled=tiled)

    @torch.no_grad()
    def _build_hbridge_self_masks_unified(self, video_seq_len: int, action_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        if not self.mot.hbridge_enabled:
            return (None, None)
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)

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
            return (lens, masks)
        return self._glue_memo(('hbridge', Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, str(device)), _build)

    @torch.no_grad()
    def _build_video_hbridge_self_masks_unified(self, video_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        if not self.mot.hbridge_enabled:
            return (None, None)
        Sv, Sd, Sp = (video_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
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
        return (lens, masks)

    @torch.no_grad()
    def _base_build_mot_attention_mask_unified(self, video_seq_len: int, action_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        total = Sv + Sd + Sp + Sa
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        v0, v1 = (0, Sv)
        d0, d1 = (Sv, Sv + Sd)
        p0, p1 = (Sv + Sd, Sv + Sd + Sp)
        a0, a1 = (Sv + Sd + Sp, total)
        mask[v0:v1, v0:v1] = self.video_expert.build_video_to_video_mask(Sv, tpf, device)
        mask[d0:d1, d0:d1] = True
        if dtpf < Sd:
            mask[d0:d0 + dtpf, d0 + dtpf:d1] = False
        mask[p0:p1, p0:p1] = True
        if ptpf < Sp:
            mask[p0:p0 + ptpf, p0 + ptpf:p1] = False
        mask[v0:v0 + tpf, d0:d0 + dtpf] = True
        mask[d0:d0 + dtpf, v0:v0 + tpf] = True
        mask[v0:v0 + tpf, p0:p0 + ptpf] = True
        mask[p0:p0 + ptpf, v0:v0 + tpf] = True
        mask[d0:d0 + dtpf, p0:p0 + ptpf] = True
        mask[p0:p0 + ptpf, d0:d0 + dtpf] = True
        if tpf < Sv:
            mask[tpf:Sv, d0:d1] = True
            mask[tpf:Sv, p0:p1] = True
        if dtpf < Sd:
            mask[d0 + dtpf:d1, v0:v1] = True
            mask[d0 + dtpf:d1, p0:p1] = True
        if ptpf < Sp:
            mask[p0 + ptpf:p1, v0:v1] = True
            mask[p0 + ptpf:p1, d0:d1] = True
        mask[a0:a1, a0:a1] = True
        mask[a0:a1, v0:v0 + tpf] = True
        mask[a0:a1, d0:d0 + dtpf] = True
        mask[a0:a1, p0:p0 + ptpf] = True
        return mask

    @torch.no_grad()
    def _build_video_hbridge_self_masks_dino_pointmap_only(self, dino_seq_len: int, pointmap_seq_len: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        if not self.mot.hbridge_enabled:
            return (None, None)
        Sd, Sp = (dino_seq_len, pointmap_seq_len)
        dtpf, ptpf = (dino_tokens_per_frame, pointmap_tokens_per_frame)
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
        return (lens, masks)

    def _merge_aux_into_video_stream(self, video_pre: dict, dino_tokens: torch.Tensor, dino_t_mod: torch.Tensor, dino_freqs: torch.Tensor, pointmap_tokens: torch.Tensor, pointmap_t_mod: torch.Tensor, pointmap_freqs: torch.Tensor):
        parts_tokens = [video_pre['tokens']]
        parts_freqs = [video_pre['freqs']]
        parts_t_mod = [video_pre['t_mod']]
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
        merged_freqs = torch.cat(parts_freqs, dim=0)
        merged_t_mod = torch.cat(parts_t_mod, dim=1)
        video_ctx_mask = video_pre['context_mask']
        if video_ctx_mask.ndim == 3 and aux_extra > 0:
            aux_ctx_mask = video_ctx_mask[:, :1, :].expand(-1, aux_extra, -1)
            merged_ctx_mask = torch.cat([video_ctx_mask, aux_ctx_mask], dim=1)
        else:
            merged_ctx_mask = video_ctx_mask
        return (merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask)

    def _build_stream_t_mod(self, timestep_per_token: torch.Tensor, seq_len: int, B: int) -> torch.Tensor:
        t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, timestep_per_token.reshape(-1))
        t = self.video_expert.time_embedding(t_emb).reshape(B, seq_len, self.video_expert.hidden_dim)
        return self.video_expert.time_projection(t).unflatten(2, (6, self.video_expert.hidden_dim))

    def _flex_zero_absent_video_tokens(self, video_tokens: torch.Tensor, tpf: int) -> torch.Tensor:
        bf = self._batch_flex
        if bf is None:
            return video_tokens
        absent_v = (~bf.present_v).to(device=video_tokens.device)
        if not absent_v.any():
            return video_tokens
        B, Sv, _ = video_tokens.shape
        keep = torch.ones((B, Sv), dtype=torch.bool, device=video_tokens.device)
        if bf.cm_v:
            keep[absent_v, :tpf] = False
        else:
            keep[absent_v, :] = False
        return video_tokens * keep.unsqueeze(-1).to(video_tokens.dtype)

    def _flex_zero_absent_dino_tokens(self, dino_tokens: torch.Tensor, dtpf: int) -> torch.Tensor:
        bf = self._batch_flex
        if bf is None:
            return dino_tokens
        absent_d = (~bf.present_d).to(device=dino_tokens.device)
        if not absent_d.any():
            return dino_tokens
        B, Sd, _ = dino_tokens.shape
        keep = torch.ones((B, Sd), dtype=torch.bool, device=dino_tokens.device)
        if bf.cm_d:
            keep[absent_d, :dtpf] = False
        else:
            keep[absent_d, :] = False
        return dino_tokens * keep.unsqueeze(-1).to(dino_tokens.dtype)

    def _flex_zero_absent_pointmap_tokens(self, pointmap_tokens: torch.Tensor, ptpf: int) -> torch.Tensor:
        bf = self._batch_flex
        if bf is None:
            return pointmap_tokens
        absent_p = (~bf.present_p).to(device=pointmap_tokens.device)
        if not absent_p.any():
            return pointmap_tokens
        B, Sp, _ = pointmap_tokens.shape
        keep = torch.ones((B, Sp), dtype=torch.bool, device=pointmap_tokens.device)
        if bf.cm_p:
            keep[absent_p, :ptpf] = False
        else:
            keep[absent_p, :] = False
        return pointmap_tokens * keep.unsqueeze(-1).to(pointmap_tokens.dtype)

    @torch.no_grad()
    def _predict_joint_noise_unified(self, latents_video: torch.Tensor, latents_dino: torch.Tensor, latents_pointmap: Optional[torch.Tensor], latents_action: torch.Tensor, timestep_video: torch.Tensor, timestep_dino: torch.Tensor, timestep_pointmap: Optional[torch.Tensor], timestep_action: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, fuse_vae_embedding_in_latents: bool, gt_action=None, dino_freqs: torch.Tensor | None=None, pt_freqs: torch.Tensor | None=None, flex_block_attention: bool=False):
        if getattr(self, '_compile_inference', False) and self.device.type == 'cuda' and (getattr(self, '_trt_joint_runner', None) is None):
            if not getattr(self, '_joint_unified_step_is_compiled', False):
                self._joint_unified_step_compiled = self._compile_for_inference(self._predict_joint_noise_unified_impl)
                self._joint_unified_step_is_compiled = True
            with self._sdpa_priority_ctx():
                return self._joint_unified_step_compiled(latents_video=latents_video, latents_dino=latents_dino, latents_pointmap=latents_pointmap, latents_action=latents_action, timestep_video=timestep_video, timestep_dino=timestep_dino, timestep_pointmap=timestep_pointmap, timestep_action=timestep_action, context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents, gt_action=gt_action, dino_freqs=dino_freqs, pt_freqs=pt_freqs, flex_block_attention=flex_block_attention)
        with self._sdpa_priority_ctx():
            return self._predict_joint_noise_unified_impl(latents_video=latents_video, latents_dino=latents_dino, latents_pointmap=latents_pointmap, latents_action=latents_action, timestep_video=timestep_video, timestep_dino=timestep_dino, timestep_pointmap=timestep_pointmap, timestep_action=timestep_action, context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents, gt_action=gt_action, dino_freqs=dino_freqs, pt_freqs=pt_freqs, flex_block_attention=flex_block_attention)

    def _predict_joint_noise_unified_impl(self, latents_video: torch.Tensor, latents_dino: torch.Tensor, latents_pointmap: Optional[torch.Tensor], latents_action: torch.Tensor, timestep_video: torch.Tensor, timestep_dino: torch.Tensor, timestep_pointmap: Optional[torch.Tensor], timestep_action: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, fuse_vae_embedding_in_latents: bool, gt_action=None, dino_freqs: torch.Tensor | None=None, pt_freqs: torch.Tensor | None=None, flex_block_attention: bool=False):
        B = latents_video.shape[0]
        video_pre = self.video_expert.pre_dit(x=latents_video, timestep=timestep_video, context=context, context_mask=context_mask, action=gt_action, fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        action_pre = self.action_expert.pre_dit(action_tokens=latents_action, timestep=timestep_action, context=context, context_mask=context_mask)
        Sv = video_pre['tokens'].shape[1]
        Sa = action_pre['tokens'].shape[1]
        tpf = int(video_pre['meta']['tokens_per_frame'])
        dino_tokens = self._embed_dino(latents_dino)
        Sd = dino_tokens.shape[1]
        num_dino_frames = int(latents_dino.shape[2])
        dtpf = int(latents_dino.shape[3])
        dino_token_timesteps = timestep_dino.view(B, 1).expand(B, Sd).clone()
        dino_token_timesteps[:, :dtpf] = 0
        dino_t_mod = self._build_stream_t_mod(dino_token_timesteps, Sd, B)
        if dino_freqs is None:
            dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre['tokens'].device)
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
                pt_freqs = self._compute_pointmap_freqs(num_pt_frames, video_pre['tokens'].device, pt_meta=pt_meta)
            pt_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, pt_token_timesteps.reshape(-1))
            pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(B, Sp, self.video_expert.hidden_dim)
        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = self._merge_aux_into_video_stream(video_pre, dino_tokens, dino_t_mod, dino_freqs, pointmap_tokens, pt_t_mod, pt_freqs)
        effective_dtpf = dtpf
        effective_ptpf = ptpf
        Sp_for_mask = Sp
        ptpf_for_mask = effective_ptpf
        if flex_block_attention:
            attention_mask = None
        else:
            attention_mask = self._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        if hasattr(self, '_refresh_history'):
            self._refresh_history([('video', Sv, tpf), ('dino', Sd, effective_dtpf),
                                   ('pointmap', Sp_for_mask, ptpf_for_mask), ('action', Sa, 1)])
        merged_out, action_out = self.mot._forward_joint_inner(video_tokens=merged_tokens, action_tokens=action_pre['tokens'], video_freqs=merged_freqs, action_freqs=action_pre['freqs'], video_t_mod=merged_t_mod, action_t_mod=action_pre['t_mod'], video_context_payload={'context': video_pre['context'], 'mask': merged_ctx_mask}, action_context_payload={'context': action_pre['context'], 'mask': action_pre['context_mask']}, attention_mask=attention_mask, sub_stream_lens=sub_stream_lens, sub_stream_self_masks=sub_stream_self_masks)
        video_tokens_out = merged_out[:, :Sv, :]
        dino_tokens_out = merged_out[:, Sv:Sv + Sd, :]
        pt_tokens_out = merged_out[:, Sv + Sd:, :]
        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(action_out, action_pre)
        pred_dino = self._project_dino_out(dino_tokens_out)
        pred_dino = pred_dino.reshape(B, num_dino_frames, dtpf, self.dino_dim)
        pred_dino = pred_dino.permute(0, 3, 1, 2).unsqueeze(-1)
        if self.dino_pred_x0:
            pred_dino = _dino_x0_to_velocity(latents_dino, pred_dino, timestep_dino.view(-1, 1, 1, 1, 1), self.infer_dino_scheduler.num_train_timesteps).to(pred_dino.dtype)
        pred_pointmap = None if pointmap_tokens is None else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        return (pred_video, pred_dino, pred_pointmap, pred_action)
    _POINTMAP_CKPT_KEYS: tuple = ('pt_patch_embedding', 'pt_head')
    _DINO_CKPT_KEYS: tuple = ('dino_embedder', 'dino_proj_out', 'dino_feature_norm')

    def _mode_ckpt_keys(self) -> tuple:
        return self._POINTMAP_CKPT_KEYS

    def _dino_ckpt_keys(self) -> tuple:
        return self._DINO_CKPT_KEYS

    def __init__(self, *args, joint_video: bool=False, joint_dino: bool=False, joint_pointmap: bool=False, flex_joint: Optional[FlexJointConfig]=None, enable_pointmap: bool=True, **kwargs):
        self._base_init(*args, **kwargs)
        self.joint_video = bool(joint_video)
        self.joint_dino = bool(joint_dino)
        self.joint_pointmap = bool(joint_pointmap)
        self.flex_joint = flex_joint if flex_joint is not None else FlexJointConfig()
        self.enable_pointmap = bool(enable_pointmap)
        self._infer_present_v = True
        self._infer_present_d = True
        self._infer_present_p = True
        self._unreachable_warned: set = set()
        self._regime_logged: set = set()
        self._configure_pointmap_off()

    @torch.no_grad()
    def _apply_joint_flag_deltas(self, mask: torch.Tensor, video_seq_len: int, action_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, joint_video: bool, joint_dino: bool, joint_pointmap: bool) -> None:
        Sv, Sd, Sp = (video_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        v_rem_start = tpf
        d_start, d_end = (Sv, Sv + Sd)
        d_rem_start = d_start + dtpf
        p_start, p_end = (Sv + Sd, Sv + Sd + Sp)
        p_rem_start = p_start + ptpf
        a_start = Sv + Sd + Sp
        if joint_video != joint_dino and tpf < Sv and (dtpf < Sd):
            mask[..., v_rem_start:Sv, d_rem_start:d_end] = False
            mask[..., d_rem_start:d_end, v_rem_start:Sv] = False
        if joint_video != joint_pointmap and tpf < Sv and (ptpf < Sp):
            mask[..., v_rem_start:Sv, p_rem_start:p_end] = False
            mask[..., p_rem_start:p_end, v_rem_start:Sv] = False
        if joint_dino != joint_pointmap and dtpf < Sd and (ptpf < Sp):
            mask[..., d_rem_start:d_end, p_rem_start:p_end] = False
            mask[..., p_rem_start:p_end, d_rem_start:d_end] = False
        if joint_video:
            mask[..., a_start:, :Sv] = True
        if joint_dino:
            mask[..., a_start:, d_start:d_end] = True
        if joint_pointmap:
            mask[..., a_start:, p_start:p_end] = True

    @torch.no_grad()
    def _apply_presence_absent_edits(self, mask: torch.Tensor, video_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, present_v: bool, present_d: bool, present_p: bool, cross_modal_v: bool, cross_modal_d: bool, cross_modal_p: bool) -> None:
        Sv, Sd, Sp = (video_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        if not present_v:
            if cross_modal_v:
                mask[..., :tpf, :] = False
                mask[..., :, :tpf] = False
            else:
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
    def _build_mot_attention_mask_unified(self, video_seq_len: int, action_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        bf = getattr(self, '_batch_flex', None)
        cache_key = None
        if bf is None and getattr(self, '_glue_cache_enabled', False) and (not self.training):
            cache_key = ('joint_mask', video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, str(device), bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap), bool(self._infer_present_v), bool(self._infer_present_d), bool(self._infer_present_p))
            hit = self._glue_cache.get(cache_key)
            if hit is not None:
                return hit
        base = self._base_build_mot_attention_mask_unified(video_seq_len=video_seq_len, action_seq_len=action_seq_len, dino_seq_len=dino_seq_len, pointmap_seq_len=pointmap_seq_len, video_tokens_per_frame=video_tokens_per_frame, dino_tokens_per_frame=dino_tokens_per_frame, pointmap_tokens_per_frame=pointmap_tokens_per_frame, device=device)
        if bf is None:
            self._apply_joint_flag_deltas(base, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap))
            if not self._infer_present_v or not self._infer_present_d or (not self._infer_present_p):
                self._apply_presence_absent_edits(base, video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, present_v=bool(self._infer_present_v), present_d=bool(self._infer_present_d), present_p=bool(self._infer_present_p), cross_modal_v=bool(self.flex_joint.cross_modal_predict_video), cross_modal_d=bool(self.flex_joint.cross_modal_predict_dino), cross_modal_p=bool(self.flex_joint.cross_modal_predict_pointmap))
            if cache_key is not None:
                self._glue_cache[cache_key] = base
            return base
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
            self._apply_joint_flag_deltas(m_b, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, bool(j_v_list[b]), bool(j_d_list[b]), bool(j_p_list[b]))
            self._apply_presence_absent_edits(m_b, video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, present_v=bool(present_v_list[b]), present_d=bool(present_d_list[b]), present_p=bool(present_p_list[b]), cross_modal_v=bf.cm_v, cross_modal_d=bf.cm_d, cross_modal_p=bf.cm_p)
            per_sample.append(m_b)
        return torch.stack(per_sample, dim=0)

    def _maybe_warn_unreachable_combo(self) -> None:
        if self.joint_video and (not self._infer_present_v) and (not self.flex_joint.cross_modal_predict_video) and ('video' not in self._unreachable_warned):
            logger.warning('CloudRoboWM deploy combo (joint_video=True, present_video=False, cross_modal_predict_video=False) is UNREACHABLE at training (sampler couples j_v &= present_v unless cm_v=True). Mask collapses to joint_video=False — rem_v stays in the layout but is fully isolated from attention, so no future-video imagination happens. To get cross-modal video imagination, retrain with flex_joint.cross_modal_predict_video=true. (warned once)')
            self._unreachable_warned.add('video')
        if self.joint_dino and (not self._infer_present_d) and (not self.flex_joint.cross_modal_predict_dino) and ('dino' not in self._unreachable_warned):
            logger.warning('CloudRoboWM deploy combo (joint_dino=True, present_dino=False, cross_modal_predict_dino=False) is UNREACHABLE at training. Mask collapses to joint_dino=False. To get cross-modal dino imagination, retrain with flex_joint.cross_modal_predict_dino=true. (warned once)')
            self._unreachable_warned.add('dino')
        if self.joint_pointmap and (not self._infer_present_p) and (not self.flex_joint.cross_modal_predict_pointmap) and ('pointmap' not in self._unreachable_warned):
            logger.warning('CloudRoboWM deploy combo (joint_pointmap=True, present_pointmap=False, cross_modal_predict_pointmap=False) is UNREACHABLE at training. Mask collapses to joint_pointmap=False. To get cross-modal pointmap imagination, retrain with flex_joint.cross_modal_predict_pointmap=true. (warned once)')
            self._unreachable_warned.add('pointmap')

    def _warn_once_pointmap_override(self) -> None:
        if 'override' in self._unreachable_warned:
            return
        logger.warning('present_pointmap/joint_pointmap were requested but this run has no pointmap stream (enable_pointmap=False, or all three flex pointmap knobs off). Both are clamped to False — the pointmap head is frozen and training ran at Sp=0. (warned once)')
        self._unreachable_warned.add('override')

    def _maybe_log_effective_regime(self) -> None:
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
        logger.info('CloudRoboWM inference regime: present=(video=%s, dino=%s, pointmap=%s) | denoise=(action=True, video=%s, dino=%s, pointmap=%s) | cross_modal_imagine=(video=%s, dino=%s, pointmap=%s)', pv, pd, pp, jv, jd, jp, cm_imagine_v, cm_imagine_d, cm_imagine_p)

    @torch.no_grad()
    def infer_action(self, prompt=None, input_image=None, action_horizon=None, num_video_frames: Optional[int]=None, proprio=None, context=None, context_mask=None, negative_prompt=None, text_cfg_scale: float=1.0, num_inference_steps: int=20, sigma_shift=None, seed=None, rand_device: str='cpu', tiled: bool=False, camera_intrinsics: Optional[torch.Tensor]=None, per_cam: Optional[Dict[str, torch.Tensor]]=None, per_cam_depth: Optional[Dict[str, torch.Tensor]]=None, dynamic_step_skip: bool=False, step_skip_thresholds: Sequence[Tuple[float, int]]=((0.95, 4), (0.93, 2)), step_skip_sim_sources: Optional[Sequence[str]]=None, step_skip_sim_aggregation: str='mean', return_stream_latents: bool=False, joint_video: Optional[bool]=None, joint_dino: Optional[bool]=None, joint_pointmap: Optional[bool]=None, present_video: Optional[bool]=None, present_dino: Optional[bool]=None, present_pointmap: Optional[bool]=None):
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
        _saved_infer_present_v = self._infer_present_v
        _saved_infer_present_d = self._infer_present_d
        _saved_infer_present_p = self._infer_present_p
        if present_video is not None:
            self._infer_present_v = bool(present_video)
        if present_dino is not None:
            self._infer_present_d = bool(present_dino)
        if present_pointmap is not None:
            self._infer_present_p = bool(present_pointmap)
        _force_joint_path = False
        self._maybe_warn_unreachable_combo()
        self._maybe_log_effective_regime()
        try:
            return self._infer_action_dispatch(prompt=prompt, input_image=input_image, action_horizon=action_horizon, num_video_frames=num_video_frames, proprio=proprio, context=context, context_mask=context_mask, negative_prompt=negative_prompt, text_cfg_scale=text_cfg_scale, num_inference_steps=num_inference_steps, sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, tiled=tiled, camera_intrinsics=camera_intrinsics, per_cam=per_cam, per_cam_depth=per_cam_depth, dynamic_step_skip=dynamic_step_skip, step_skip_thresholds=step_skip_thresholds, step_skip_sim_sources=step_skip_sim_sources, step_skip_sim_aggregation=step_skip_sim_aggregation, return_stream_latents=return_stream_latents, force_joint_path=_force_joint_path)
        finally:
            self.joint_video, self.joint_dino, self.joint_pointmap = _saved_joint
            self._infer_present_v = _saved_infer_present_v
            self._infer_present_d = _saved_infer_present_d
            self._infer_present_p = _saved_infer_present_p

    def _infer_action_dispatch(self, prompt, input_image, action_horizon, num_video_frames, proprio, context, context_mask, negative_prompt, text_cfg_scale, num_inference_steps, sigma_shift, seed, rand_device, tiled, camera_intrinsics, per_cam, per_cam_depth, dynamic_step_skip, step_skip_thresholds, step_skip_sim_sources, step_skip_sim_aggregation, return_stream_latents: bool=False, force_joint_path: bool=False):
        any_joint = self.joint_video or self.joint_dino or self.joint_pointmap
        if not any_joint and (not force_joint_path):
            return self._base_infer_action(prompt=prompt, input_image=input_image, action_horizon=action_horizon, proprio=proprio, context=context, context_mask=context_mask, negative_prompt=negative_prompt, text_cfg_scale=text_cfg_scale, num_inference_steps=num_inference_steps, sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, tiled=tiled, camera_intrinsics=camera_intrinsics, per_cam=per_cam, per_cam_depth=per_cam_depth, present_video=self._infer_present_v, present_dino=self._infer_present_d, present_pointmap=self._infer_present_p)
        if num_video_frames is None:
            raise ValueError('`num_video_frames` is required when any joint_* flag is True or a presence override (present_video/present_dino/present_pointmap) is supplied.')
        return self._infer_action_joint(prompt=prompt, input_image=input_image, action_horizon=action_horizon, num_video_frames=num_video_frames, proprio=proprio, context=context, context_mask=context_mask, negative_prompt=negative_prompt, text_cfg_scale=text_cfg_scale, num_inference_steps=num_inference_steps, sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, tiled=tiled, camera_intrinsics=camera_intrinsics, per_cam=per_cam, per_cam_depth=per_cam_depth, dynamic_step_skip=dynamic_step_skip, step_skip_thresholds=step_skip_thresholds, step_skip_sim_sources=step_skip_sim_sources, step_skip_sim_aggregation=step_skip_sim_aggregation, return_stream_latents=return_stream_latents)

    @torch.no_grad()
    def _infer_action_joint(self, prompt, input_image, action_horizon, num_video_frames, proprio, context, context_mask, negative_prompt, text_cfg_scale, num_inference_steps, sigma_shift, seed, rand_device, tiled, camera_intrinsics, per_cam, per_cam_depth, dynamic_step_skip: bool=False, step_skip_thresholds: Sequence[Tuple[float, int]]=((0.95, 4), (0.93, 2)), step_skip_sim_sources: Optional[Sequence[str]]=None, step_skip_sim_aggregation: str='mean', return_stream_latents: bool=False):
        self.eval()
        if camera_intrinsics is not None:
            self.set_camera_intrinsics(camera_intrinsics)
        if per_cam_depth is None and bool(self._infer_present_p):
            raise ValueError('`per_cam_depth` is required at inference for CloudRoboWM when the pointmap stream is present. Pass present_pointmap=False (or use a pointmap-disabled model) to run without depth.')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f'`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}')
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f'`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})')
        if checked_t != num_video_frames:
            raise ValueError(f'`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}')
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError('`proprio` was provided but `proprio_dim=None`.')
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
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
            num_pt_frames = 1
        video_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        dino_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        pt_gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = _staged_randn((1, self.vae.model.z_dim, video_latent_t, latent_h, latent_w), video_gen, rand_device, self.device, self.torch_dtype)
        latents_action = _staged_randn((1, action_horizon, self.action_expert.action_dim), action_gen, rand_device, self.device, self.torch_dtype)
        n_patches_dino = sum((h * w for h, w in self.dino_cam_patches))
        latents_dino = _staged_randn((1, self.dino_dim, num_dino_frames, n_patches_dino, 1), dino_gen, rand_device, self.device, self.torch_dtype)
        _skip_v = not bool(self._infer_present_v)
        _skip_d = not bool(self._infer_present_d)
        _skip_p = not bool(self._infer_present_p)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if _skip_v:
            first_frame_latents = None
        else:
            first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))
        if _skip_d:
            first_frame_dino = None
        else:
            if per_cam is not None:
                per_cam_5d: Dict[str, torch.Tensor] = {}
                for k, v in per_cam.items():
                    v = v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    if v.ndim == 4:
                        v = v.unsqueeze(2)
                    per_cam_5d[k] = v
                first_frame_dino = self.dino_encoder.encode_video(video=None, per_cam=per_cam_5d, concat_mode='robotwin', **self._dino_encode_kwargs(), first_frame_only=True)
            else:
                first_frame_dino = self.dino_encoder.encode_video(input_image.unsqueeze(2), concat_mode='robotwin', **self._dino_encode_kwargs(), first_frame_only=True)
            latents_dino[:, :, 0:1] = first_frame_dino
        if _skip_p:
            first_frame_pointmap = None
            latents_pointmap = _staged_randn((1, self.vae.model.z_dim, num_pt_frames, latent_h, latent_w), pt_gen, rand_device, self.device, self.torch_dtype)
        else:
            first_frame_pointmap = self._encode_first_frame_pointmap_raw(tiled=tiled, per_cam_depth=per_cam_depth)
            _, pt_C, _, pt_H, pt_W = first_frame_pointmap.shape
            latents_pointmap = _staged_randn((1, pt_C, num_pt_frames, pt_H, pt_W), pt_gen, rand_device, self.device, self.torch_dtype)
            latents_pointmap[:, :, 0:1] = first_frame_pointmap
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError('`prompt` and `context/context_mask` are mutually exclusive.')
        if prompt is None and context is None:
            raise ValueError('Either `prompt` or `context/context_mask` must be provided.')
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
            context, context_mask = self._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)
        video_dtype = latents_video.dtype if latents_video is not None else latents_action.dtype
        ts_video, deltas_video = self.infer_video_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=video_dtype, shift_override=sigma_shift)
        ts_action, deltas_action = self.infer_action_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift)
        ts_dino, deltas_dino = self.infer_dino_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=latents_dino.dtype, shift_override=sigma_shift)
        ts_pt, deltas_pt = self.infer_pointmap_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=latents_pointmap.dtype, shift_override=sigma_shift)
        sim_sources = resolve_sim_sources(explicit=step_skip_sim_sources, denoised_flags={'video': bool(self.joint_video), 'dino': dino_denoised, 'pointmap': denoise_pointmap})
        skip_ctrl = StepSkipController(enabled=dynamic_step_skip and bool(sim_sources), thresholds=step_skip_thresholds, sim_aggregation=step_skip_sim_aggregation)
        dino_freqs_precomputed = self._compute_dino_freqs(latents_dino.shape[2], latents_dino.device)
        pt_freqs_precomputed = None
        flex_attn_active = bool(getattr(self, '_infer_attn_backend', 'sdpa') == 'flex')
        if flex_attn_active:
            self._prepare_joint_flex_block_mask(latents_video, latents_dino, latents_pointmap, action_horizon)
        _loop_kwargs = dict(latents_video=latents_video, latents_dino=latents_dino, latents_pointmap=latents_pointmap, latents_action=latents_action, ts_video=ts_video, deltas_video=deltas_video, ts_action=ts_action, deltas_action=deltas_action, ts_dino=ts_dino, deltas_dino=deltas_dino, ts_pt=ts_pt, deltas_pt=deltas_pt, context=context, context_mask=context_mask, fuse_flag=fuse_flag, dino_denoised=dino_denoised, denoise_pointmap=denoise_pointmap, first_frame_latents=first_frame_latents, first_frame_dino=first_frame_dino, first_frame_pointmap=first_frame_pointmap, dino_freqs=dino_freqs_precomputed, pt_freqs=pt_freqs_precomputed, flex_block_attention=flex_attn_active)
        _loop_graph = getattr(self, '_joint_loop_cuda_graph_enabled', False) and (not skip_ctrl.enabled) and all((getattr(sch, 'solver', 'euler') == 'euler' for sch in (self.infer_action_scheduler, self.infer_video_scheduler, self.infer_dino_scheduler, self.infer_pointmap_scheduler) if sch is not None))
        if _loop_graph:
            latents_video, latents_dino, latents_pointmap, latents_action = self._joint_loop_graph_runner(**_loop_kwargs)
            ts_video = ts_video[:0]
            ts_action = ts_action[:0]
            ts_dino = ts_dino[:0]
            ts_pt = ts_pt[:0]
        elif self._use_loop_compile() and (not skip_ctrl.enabled):
            latents_video, latents_dino, latents_pointmap, latents_action = self._run_joint_denoise_loop(**_loop_kwargs)
            ts_video = ts_video[:0]
            ts_action = ts_action[:0]
            ts_dino = ts_dino[:0]
            ts_pt = ts_pt[:0]
        for step_tv, delta_v, step_ta, delta_a, step_td, delta_d, step_tp, delta_p in zip(ts_video, deltas_video, ts_action, deltas_action, ts_dino, deltas_dino, ts_pt, deltas_pt):
            if skip_ctrl.should_run():
                _video_dtype = latents_video.dtype
                t_video = step_tv.unsqueeze(0).to(dtype=_video_dtype, device=self.device) if self.joint_video else torch.zeros((1,), dtype=_video_dtype, device=self.device)
                t_dino = step_td.unsqueeze(0).to(dtype=latents_dino.dtype, device=self.device) if dino_denoised else torch.zeros((1,), dtype=latents_dino.dtype, device=self.device)
                t_pt = step_tp.unsqueeze(0).to(dtype=latents_pointmap.dtype, device=self.device) if denoise_pointmap else torch.zeros((1,), dtype=latents_pointmap.dtype, device=self.device)
                t_action = step_ta.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                pred_video, pred_dino, pred_pointmap, pred_action = self._predict_joint_noise_unified(latents_video=latents_video, latents_dino=latents_dino, latents_pointmap=latents_pointmap, latents_action=latents_action, timestep_video=t_video, timestep_dino=t_dino, timestep_pointmap=t_pt, timestep_action=t_action, context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=fuse_flag, gt_action=None, dino_freqs=dino_freqs_precomputed, pt_freqs=pt_freqs_precomputed, flex_block_attention=flex_attn_active)
                _src_tensors = {'video': pred_video, 'dino': pred_dino, 'pointmap': pred_pointmap}
                sim_refs = [_src_tensors[s] for s in sim_sources]
                skip_ctrl.record(sim_refs=sim_refs, preds=(pred_video, pred_dino, pred_pointmap, pred_action))
            else:
                pred_video, pred_dino, pred_pointmap, pred_action = skip_ctrl.cached()
            latents_action = self.infer_action_scheduler.step(pred_action, delta_a, latents_action, timestep=t_action)
            if self.joint_video:
                latents_video = self.infer_video_scheduler.step(pred_video, delta_v, latents_video, timestep=t_video)
                if first_frame_latents is not None:
                    latents_video[:, :, 0:1] = first_frame_latents.clone()
            if dino_denoised:
                latents_dino = self.infer_dino_scheduler.step(pred_dino, delta_d, latents_dino, timestep=t_dino)
                if first_frame_dino is not None:
                    latents_dino[:, :, 0:1] = first_frame_dino.clone()
            if denoise_pointmap:
                latents_pointmap = self.infer_pointmap_scheduler.step(pred_pointmap, delta_p, latents_pointmap, timestep=t_pt)
                if first_frame_pointmap is not None:
                    latents_pointmap[:, :, 0:1] = first_frame_pointmap.clone()
        if skip_ctrl.enabled:
            sims_fmt = '[' + ', '.join(('{' + ', '.join((f'{name}: {v:.4f}' for name, v in zip(sim_sources, step_sims))) + '}→' + decision for step_sims, decision in zip(skip_ctrl.per_stream_history, skip_ctrl.decision_history))) + ']'
            print(f'[step-skip] {skip_ctrl.steps_run}/{num_inference_steps} forwards run ({num_inference_steps - skip_ctrl.steps_run} skipped, sources={sim_sources}, agg={step_skip_sim_aggregation}, sims={sims_fmt})')
        out = {'action': latents_action[0].detach().to(device='cpu', dtype=torch.float32)}
        if return_stream_latents:
            if self.joint_video and latents_video is not None:
                out['video_latents'] = latents_video.detach().to(device='cpu', dtype=torch.float32)
            if dino_denoised:
                out['dino_latents'] = latents_dino.detach().to(device='cpu', dtype=torch.float32)
            if denoise_pointmap:
                out['pointmap_latents'] = latents_pointmap.detach().to(device='cpu', dtype=torch.float32)
        return out
