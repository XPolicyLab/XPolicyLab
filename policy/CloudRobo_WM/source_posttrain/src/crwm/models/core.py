from __future__ import annotations
import copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from crwm.utils.logging_config import get_logger
from .dino_encoder import DinoEncoder
from .helpers.dino import _DINO_X0_SIGMA_MIN, _dino_x0_to_velocity, _zero_init_dino_x0_head_, compute_dino_freqs
from .mot import MoT
from .pointmap_encoder import PointmapEncoder
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)
from typing import Dict, Optional, Sequence, Tuple
from crwm.utils.pytorch_utils import _staged_randn
from .backbone import TrainingBackbone
from .helpers.modality_dropout import ModalityDropoutConfig, sample_modality_flags

class CRWMCore(TrainingBackbone):
    FROZEN_MODULES: set[str] = {'vae', 'text_encoder', 'dino_encoder', 'pointmap_encoder'}

    def _base_init(self, video_expert, action_expert, mot: MoT, vae, dino_encoder: DinoEncoder, pointmap_encoder: PointmapEncoder, text_encoder=None, tokenizer=None, text_dim: Optional[int]=None, proprio_dim: Optional[int]=None, device: str='cpu', torch_dtype: torch.dtype=torch.float32, video_train_shift: float=5.0, video_infer_shift: float=5.0, video_num_train_timesteps: int=1000, action_train_shift: float=5.0, action_infer_shift: float=5.0, action_num_train_timesteps: int=1000, loss_lambda_video: float=1.0, loss_lambda_action: float=1.0, dino_dim: int=768, dino_train_shift: float=5.0, dino_infer_shift: float=5.0, dino_num_train_timesteps: int=1000, loss_lambda_dino: float=1.0, dino_cam_regions: list | None=None, dino_cam_patches: list | None=None, dino_temporal_stride: int=1, dino_stride_keep_far: bool=False, dino_pred_x0: bool=False, dino_pool_mode: str='avg', freeze_dino_encoder: bool=True, dino_pixel_unshuffle: int=0, pointmap_train_shift: float=5.0, pointmap_infer_shift: float=5.0, pointmap_num_train_timesteps: int=1000, loss_lambda_pointmap: float=1.0, pointmap_depth_vis_mode: str='turbo', composite_layout=None, composite_layout_slot_key_map=None, dino_pool_factor: int=1):
        from crwm.composite_layouts import get_layout
        _layout_resolved = get_layout(composite_layout)
        super().__init__(video_expert=video_expert, action_expert=action_expert, mot=mot, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, text_dim=text_dim, proprio_dim=proprio_dim, composite_layout=composite_layout, composite_layout_slot_key_map=composite_layout_slot_key_map, device=device, torch_dtype=torch_dtype, video_train_shift=video_train_shift, video_infer_shift=video_infer_shift, video_num_train_timesteps=video_num_train_timesteps, action_train_shift=action_train_shift, action_infer_shift=action_infer_shift, action_num_train_timesteps=action_num_train_timesteps, loss_lambda_video=loss_lambda_video, loss_lambda_action=loss_lambda_action)
        self.dino_encoder = dino_encoder
        self.dino_dim = int(dino_dim)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.dino_temporal_stride = int(dino_temporal_stride)
        self.dino_stride_keep_far = bool(dino_stride_keep_far)
        self.dino_pred_x0 = bool(dino_pred_x0)
        if self.dino_pred_x0:
            logger.info('CRWMCore: dino_pred_x0=True — DINO head output is x̂0; v̂ = (x_t − x̂0)/σ is derived analytically for loss and inference.')
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
        self._batch_dropout = None
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
        logger.info('CRWMCore: dino_temporal_stride=%d dino_stride_keep_far=%s', self.dino_temporal_stride, self.dino_stride_keep_far)

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
        fj = getattr(self, 'modality_dropout', None)
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
        self.modality_dropout.cross_modal_predict_pointmap = False
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

    def _dropout_zero_absent_video_tokens(self, video_tokens: torch.Tensor, tpf: int) -> torch.Tensor:
        bf = self._batch_dropout
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

    def _dropout_zero_absent_dino_tokens(self, dino_tokens: torch.Tensor, dtpf: int) -> torch.Tensor:
        bf = self._batch_dropout
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

    def _dropout_zero_absent_pointmap_tokens(self, pointmap_tokens: torch.Tensor, ptpf: int) -> torch.Tensor:
        bf = self._batch_dropout
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

    def _dropout_reduce_per_sample_loss(self, per_sample_loss: torch.Tensor, weights: torch.Tensor, present_mask: Optional[torch.Tensor]=None, cross_modal_active: bool=False) -> torch.Tensor:
        bf = self._batch_dropout
        weighted = per_sample_loss * weights
        if bf is None or present_mask is None:
            return weighted.mean()
        mask = present_mask.to(device=weighted.device, dtype=weighted.dtype)
        if cross_modal_active:
            return weighted.mean()
        denom = mask.sum().clamp(min=1.0)
        return (weighted * mask).sum() / denom
    _POINTMAP_CKPT_KEYS: tuple = ('pt_patch_embedding', 'pt_head')
    _DINO_CKPT_KEYS: tuple = ('dino_embedder', 'dino_proj_out', 'dino_feature_norm')

    def _pointmap_metadata(self) -> dict:
        encoder = getattr(self, 'pointmap_encoder', None)
        metadata_fn = getattr(encoder, 'encoding_metadata', None)
        if not callable(metadata_fn):
            return {}
        return {'pointmap_encoding': metadata_fn()}

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    def __init__(self, *args, joint_video: bool=False, joint_dino: bool=False, joint_pointmap: bool=False, modality_dropout: Optional[ModalityDropoutConfig]=None, enable_pointmap: bool=True, **kwargs):
        self._base_init(*args, **kwargs)
        self.joint_video = bool(joint_video)
        self.joint_dino = bool(joint_dino)
        self.joint_pointmap = bool(joint_pointmap)
        self.modality_dropout = modality_dropout if modality_dropout is not None else ModalityDropoutConfig()
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
        bf = getattr(self, '_batch_dropout', None)
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
                self._apply_presence_absent_edits(base, video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, present_v=bool(self._infer_present_v), present_d=bool(self._infer_present_d), present_p=bool(self._infer_present_p), cross_modal_v=bool(self.modality_dropout.cross_modal_predict_video), cross_modal_d=bool(self.modality_dropout.cross_modal_predict_dino), cross_modal_p=bool(self.modality_dropout.cross_modal_predict_pointmap))
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

    def training_loss(self, sample, tiled: bool=False):
        if not self.modality_dropout.enabled:
            return self._base_training_loss(sample, tiled=tiled)
        action = sample.get('action')
        if action is None:
            raise RuntimeError("modality_dropout requires `sample['action']` for batch size inference.")
        B = int(action.shape[0])
        bf = sample_modality_flags(cfg=self.modality_dropout, batch_size=B, device=self.device, pointmap_off=self._pointmap_globally_off)
        self._batch_dropout = bf
        try:
            return self._base_training_loss(sample, tiled=tiled)
        finally:
            self._batch_dropout = None
