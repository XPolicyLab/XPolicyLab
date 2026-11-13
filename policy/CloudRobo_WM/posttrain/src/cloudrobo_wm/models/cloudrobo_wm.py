import copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from cloudrobo_wm.utils.logging_config import get_logger
from .dino_encoder import DinoEncoder
from .helpers.dino import _dino_x0_to_velocity, _zero_init_dino_x0_head_, compute_dino_freqs
from .mot import MoT
from .pointmap_encoder import PointmapEncoder
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)
from typing import Dict, Optional
from .backbone import CloudRoboBackbone
from .helpers.joint_modalities import JointModalityConfig, sample_joint_batch_flags

class CloudRoboWM(CloudRoboBackbone):
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
        self._batch_modalities = None
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
        fj = getattr(self, 'joint_modalities', None)
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
        self.joint_modalities.cross_modal_predict_pointmap = False
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

    def build_inputs(self, sample, tiled: bool=False):
        inputs = super().build_inputs(sample, tiled=tiled)
        per_cam = None
        if 'per_cam' in sample:
            per_cam = {k: v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True) for k, v in sample['per_cam'].items()}
        video = None
        if per_cam is None:
            video = sample['video'].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        with torch.no_grad():
            dino_features = self.dino_encoder.encode_video(video=video, concat_mode='robotwin', **self._dino_encode_kwargs(), temporal_stride=self.dino_temporal_stride, per_cam=per_cam)
        inputs['dino_features'] = dino_features
        if self._pointmap_globally_off:
            inputs['pointmap_raw'] = None
            return inputs
        if 'camera_intrinsics' not in sample:
            raise KeyError("sample['camera_intrinsics'] is missing. Use the depth dataset (RobotVideoDataset) with meta/camera_intrinsics.json.")
        if 'per_cam_depth' not in sample:
            raise KeyError("sample['per_cam_depth'] is missing. CloudRoboWM requires the depth dataset: a data config declaring shape_meta.depth, e.g. configs/data/robotwin.yaml.")
        K = sample['camera_intrinsics'].to(device=self.device)
        self.set_camera_intrinsics(K)
        per_cam_depth = {k: v.to(device=self.device, non_blocking=True) for k, v in sample['per_cam_depth'].items()}
        with torch.no_grad():
            composite = self.pointmap_encoder.encode_composite(per_cam_depth=per_cam_depth, camera_intrinsics=K, concat_mode='robotwin', **self._layout_kwargs())
            pointmap_raw = self._encode_video_latents(composite, tiled=tiled)
        inputs['pointmap_raw'] = pointmap_raw
        return inputs

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
        bf = self._batch_modalities
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
        bf = self._batch_modalities
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
        bf = self._batch_modalities
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

    def _flex_reduce_per_sample_loss(self, per_sample_loss: torch.Tensor, weights: torch.Tensor, present_mask: Optional[torch.Tensor]=None, cross_modal_active: bool=False) -> torch.Tensor:
        bf = self._batch_modalities
        weighted = per_sample_loss * weights
        if bf is None or present_mask is None:
            return weighted.mean()
        mask = present_mask.to(device=weighted.device, dtype=weighted.dtype)
        if cross_modal_active:
            return weighted.mean()
        denom = mask.sum().clamp(min=1.0)
        return (weighted * mask).sum() / denom

    def _base_training_loss(self, sample, tiled: bool=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs['input_latents']
        B = input_latents.shape[0]
        context, context_mask = (inputs['context'], inputs['context_mask'])
        action = inputs['action']
        action_is_pad, image_is_pad = (inputs['action_is_pad'], inputs['image_is_pad'])
        dino_features = inputs['dino_features']
        pointmap_raw = inputs['pointmap_raw']
        noise_video = torch.randn_like(input_latents)
        t_video = self.train_video_scheduler.sample_training_t(B, self.device, input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, t_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, t_video)
        if inputs['first_frame_latents'] is not None:
            latents[:, :, 0:1] = inputs['first_frame_latents']
        noise_action = torch.randn_like(action)
        t_action = self.train_action_scheduler.sample_training_t(B, self.device, action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, t_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, t_action)
        noise_dino = torch.randn_like(dino_features)
        t_dino = self.train_dino_scheduler.sample_training_t(B, self.device, dino_features.dtype)
        noisy_dino = self.train_dino_scheduler.add_noise(dino_features, noise_dino, t_dino)
        target_dino = self.train_dino_scheduler.training_target(dino_features, noise_dino, t_dino)
        noisy_dino[:, :, 0:1] = dino_features[:, :, 0:1]
        if self._pointmap_globally_off:
            noisy_pointmap = t_pointmap = target_pointmap = None
        else:
            noise_pt = torch.randn_like(pointmap_raw)
            t_pointmap = self.train_pointmap_scheduler.sample_training_t(B, self.device, pointmap_raw.dtype)
            noisy_pointmap = self.train_pointmap_scheduler.add_noise(pointmap_raw, noise_pt, t_pointmap)
            target_pointmap = self.train_pointmap_scheduler.training_target(pointmap_raw, noise_pt, t_pointmap)
            noisy_pointmap[:, :, 0:1] = pointmap_raw[:, :, 0:1]
        video_pre = self.video_expert.pre_dit(x=latents, timestep=t_video, context=context, context_mask=context_mask, action=action, fuse_vae_embedding_in_latents=inputs['fuse_vae_embedding_in_latents'])
        action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=t_action, context=context, context_mask=context_mask)
        Sv = video_pre['tokens'].shape[1]
        Sa = action_pre['tokens'].shape[1]
        tpf = int(video_pre['meta']['tokens_per_frame'])
        video_pre['tokens'] = self._flex_zero_absent_video_tokens(video_pre['tokens'], tpf)
        dino_tokens = self._embed_dino(noisy_dino)
        Sd = dino_tokens.shape[1]
        num_dino_frames = int(dino_features.shape[2])
        dtpf = int(dino_features.shape[3])
        dino_tokens = self._flex_zero_absent_dino_tokens(dino_tokens, dtpf)
        dino_token_timesteps = t_dino.view(B, 1).expand(B, Sd).clone()
        dino_token_timesteps[:, :dtpf] = 0
        dino_t_mod = self._build_stream_t_mod(dino_token_timesteps, Sd, B)
        dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre['tokens'].device)
        if self._pointmap_globally_off:
            pointmap_tokens = pt_t_mod = pt_freqs = pt_t = pt_meta = None
            Sp = ptpf = 0
        else:
            pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(noisy_pointmap)
            pointmap_tokens = self._flex_zero_absent_pointmap_tokens(pointmap_tokens, ptpf)
            Sp = pointmap_tokens.shape[1]
            num_pt_frames = int(noisy_pointmap.shape[2])
            pt_token_timesteps = t_pointmap.view(B, 1).expand(B, Sp).clone()
            pt_token_timesteps[:, :ptpf] = 0
            pt_t_mod = self._build_stream_t_mod(pt_token_timesteps, Sp, B)
            pt_freqs = self._compute_pointmap_freqs(num_pt_frames, video_pre['tokens'].device, pt_meta=pt_meta)
            pt_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, pt_token_timesteps.reshape(-1))
            pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(B, Sp, self.video_expert.hidden_dim)
        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = self._merge_aux_into_video_stream(video_pre, dino_tokens, dino_t_mod, dino_freqs, pointmap_tokens, pt_t_mod, pt_freqs)
        effective_dtpf = dtpf
        effective_ptpf = ptpf
        Sp_for_mask = Sp
        ptpf_for_mask = effective_ptpf
        attention_mask = self._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        tokens_out = self.mot(embeds_all={'video': merged_tokens, 'action': action_pre['tokens']}, attention_mask=attention_mask, freqs_all={'video': merged_freqs, 'action': action_pre['freqs']}, context_all={'video': {'context': video_pre['context'], 'mask': merged_ctx_mask}, 'action': {'context': action_pre['context'], 'mask': action_pre['context_mask']}}, t_mod_all={'video': merged_t_mod, 'action': action_pre['t_mod']}, sub_stream_lens=sub_stream_lens, sub_stream_self_masks=sub_stream_self_masks)
        video_tokens_out = tokens_out['video'][:, :Sv, :]
        dino_tokens_out = tokens_out['video'][:, Sv:Sv + Sd, :]
        pt_tokens_out = tokens_out['video'][:, Sv + Sd:, :]
        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'], action_pre)
        pred_dino = self._project_dino_out(dino_tokens_out)
        pred_pointmap = None if self._pointmap_globally_off else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        include_init = inputs['first_frame_latents'] is None
        if not include_init:
            pred_video, target_video = (pred_video[:, :, 1:], target_video[:, :, 1:])
        lv = self._compute_video_loss_per_sample(pred_video, target_video, image_is_pad, include_init)
        wv = self.train_video_scheduler.training_weight(t_video).to(lv.device, dtype=lv.dtype)
        loss_video = self._flex_reduce_per_sample_loss(lv, wv, present_mask=self._batch_modalities.present_v if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_v if self._batch_modalities is not None else False)
        la = F.mse_loss(pred_action.float(), target_action.float(), reduction='none').mean(2)
        if action_is_pad is not None:
            v = (~action_is_pad).to(la.device, dtype=la.dtype)
            la = (la * v).sum(1) / v.sum(1).clamp(min=1.0)
        else:
            la = la.mean(1)
        wa = self.train_action_scheduler.training_weight(t_action).to(la.device, dtype=la.dtype)
        loss_action = (la * wa).mean()
        if target_dino is None:
            loss_dino = torch.tensor(0.0, device=self.device)
        else:
            target_dino_flat = target_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(B, -1, self.dino_dim)
            if self.dino_pred_x0:
                noisy_dino_flat = noisy_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(B, -1, self.dino_dim)
                pred_dino_for_loss = _dino_x0_to_velocity(noisy_dino_flat, pred_dino, t_dino.view(B, 1, 1), self.train_dino_scheduler.num_train_timesteps)
            else:
                pred_dino_for_loss = pred_dino
            ld = F.mse_loss(pred_dino_for_loss.float(), target_dino_flat.float(), reduction='none').mean(2)
            n_rem = num_dino_frames - 1
            if image_is_pad is None or n_rem <= 0:
                ld = ld[:, dtpf:].mean(1)
            else:
                ld_rem = ld[:, dtpf:]
                ld_per_frame = ld_rem.view(B, n_rem, dtpf).mean(2)
                dino_is_pad_rem = self._aux_per_frame_is_pad(image_is_pad, self.dino_temporal_stride, keep_far=self.dino_stride_keep_far)[:, 1:]
                if dino_is_pad_rem.shape[1] != n_rem:
                    raise RuntimeError(f'DINO mask shape mismatch: pad={dino_is_pad_rem.shape[1]} vs num_dino_frames-1={n_rem}')
                ld = self._masked_loss_reduction(ld_per_frame, dino_is_pad_rem)
            wd = self.train_dino_scheduler.training_weight(t_dino).to(ld.device, dtype=ld.dtype)
            loss_dino = self._flex_reduce_per_sample_loss(ld, wd, present_mask=self._batch_modalities.present_d if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_d if self._batch_modalities is not None else False)
        if target_pointmap is None:
            loss_pointmap = torch.tensor(0.0, device=self.device)
        else:
            pred_pt_for_loss = pred_pointmap[:, :, 1:].float()
            tgt_pt_for_loss = target_pointmap[:, :, 1:].float()
            if image_is_pad is None:
                err = F.mse_loss(pred_pt_for_loss, tgt_pt_for_loss, reduction='none')
                lp = err.flatten(1).mean(1)
            else:
                lp = self._compute_video_loss_per_sample(pred_pt_for_loss, tgt_pt_for_loss, image_is_pad, include_initial_video_step=False)
            wp = self.train_pointmap_scheduler.training_weight(t_pointmap).to(lp.device, dtype=lp.dtype)
            loss_pointmap = self._flex_reduce_per_sample_loss(lp, wp, present_mask=self._batch_modalities.present_p if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_p if self._batch_modalities is not None else False)
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + self.loss_lambda_dino * loss_dino + self.loss_lambda_pointmap * loss_pointmap
        loss_dict = {'loss_video': self.loss_lambda_video * float(loss_video.detach().item()), 'loss_action': self.loss_lambda_action * float(loss_action.detach().item()), 'loss_dino': self.loss_lambda_dino * float(loss_dino.detach().item()), 'loss_pointmap': self.loss_lambda_pointmap * float(loss_pointmap.detach().item())}
        return (total, loss_dict)

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
            self._refresh_history([('video', Sv, tpf), ('dino', Sd, effective_dtpf), ('pointmap', Sp_for_mask, ptpf_for_mask), ('action', Sa, 1)])
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

    def _pointmap_metadata(self) -> dict:
        encoder = getattr(self, 'pointmap_encoder', None)
        metadata_fn = getattr(encoder, 'encoding_metadata', None)
        if not callable(metadata_fn):
            return {}
        return {'pointmap_encoding': metadata_fn()}

    def _validate_checkpoint_payload(self, payload, *, strict_shape: bool):
        from cloudrobo_wm.utils.pointmap_checkpoint import validate_pointmap_encoding
        validate_pointmap_encoding(payload, self._pointmap_metadata(), strict_shape=strict_shape)

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    def __init__(self, *args, joint_video: bool=False, joint_dino: bool=False, joint_pointmap: bool=False, joint_modalities: Optional[JointModalityConfig]=None, enable_pointmap: bool=True, **kwargs):
        self._base_init(*args, **kwargs)
        self.joint_video = bool(joint_video)
        self.joint_dino = bool(joint_dino)
        self.joint_pointmap = bool(joint_pointmap)
        self.joint_modalities = joint_modalities if joint_modalities is not None else JointModalityConfig()
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
        bf = getattr(self, '_batch_modalities', None)
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
                self._apply_presence_absent_edits(base, video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, present_v=bool(self._infer_present_v), present_d=bool(self._infer_present_d), present_p=bool(self._infer_present_p), cross_modal_v=bool(self.joint_modalities.cross_modal_predict_video), cross_modal_d=bool(self.joint_modalities.cross_modal_predict_dino), cross_modal_p=bool(self.joint_modalities.cross_modal_predict_pointmap))
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
        if not self.joint_modalities.enabled:
            return self._base_training_loss(sample, tiled=tiled)
        action = sample.get('action')
        if action is None:
            raise RuntimeError("joint_modalities requires `sample['action']` for batch size inference.")
        B = int(action.shape[0])
        bf = sample_joint_batch_flags(cfg=self.joint_modalities, batch_size=B, device=self.device, pointmap_off=self._pointmap_globally_off)
        self._batch_modalities = bf
        try:
            return self._base_training_loss(sample, tiled=tiled)
        finally:
            self._batch_modalities = None

    def _joint_denoise_loop_body(self, latents_video: Optional[torch.Tensor], latents_dino: torch.Tensor, latents_pointmap: torch.Tensor, latents_action: torch.Tensor, ts_video: torch.Tensor, deltas_video: torch.Tensor, ts_action: torch.Tensor, deltas_action: torch.Tensor, ts_dino: torch.Tensor, deltas_dino: torch.Tensor, ts_pt: torch.Tensor, deltas_pt: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, fuse_flag: bool, dino_denoised: bool, denoise_pointmap: bool, first_frame_latents: Optional[torch.Tensor], first_frame_dino: Optional[torch.Tensor], first_frame_pointmap: Optional[torch.Tensor], dino_freqs: Optional[torch.Tensor], pt_freqs: Optional[torch.Tensor], flex_block_attention: bool=False):
        device = latents_action.device
        for i in range(ts_action.shape[0]):
            _video_dtype = latents_video.dtype if latents_video is not None else latents_action.dtype
            t_video = ts_video[i].reshape(1).to(dtype=_video_dtype) if self.joint_video else torch.zeros((1,), dtype=_video_dtype, device=device)
            t_dino = ts_dino[i].reshape(1).to(dtype=latents_dino.dtype) if dino_denoised else torch.zeros((1,), dtype=latents_dino.dtype, device=device)
            t_pt = ts_pt[i].reshape(1).to(dtype=latents_pointmap.dtype) if denoise_pointmap else torch.zeros((1,), dtype=latents_pointmap.dtype, device=device)
            t_action = ts_action[i].reshape(1).to(dtype=latents_action.dtype)
            pred_video, pred_dino, pred_pointmap, pred_action = self._predict_joint_noise_unified_impl(latents_video=latents_video, latents_dino=latents_dino, latents_pointmap=latents_pointmap, latents_action=latents_action, timestep_video=t_video, timestep_dino=t_dino, timestep_pointmap=t_pt, timestep_action=t_action, context=context, context_mask=context_mask, fuse_vae_embedding_in_latents=fuse_flag, gt_action=None, dino_freqs=dino_freqs, pt_freqs=pt_freqs, flex_block_attention=flex_block_attention)
            latents_action = self.infer_action_scheduler.step(pred_action, deltas_action[i], latents_action)
            if self.joint_video:
                latents_video = self.infer_video_scheduler.step(pred_video, deltas_video[i], latents_video)
                if first_frame_latents is not None:
                    latents_video[:, :, 0:1] = first_frame_latents.clone()
            if dino_denoised:
                latents_dino = self.infer_dino_scheduler.step(pred_dino, deltas_dino[i], latents_dino)
                if first_frame_dino is not None:
                    latents_dino[:, :, 0:1] = first_frame_dino.clone()
            if denoise_pointmap:
                latents_pointmap = self.infer_pointmap_scheduler.step(pred_pointmap, deltas_pt[i], latents_pointmap)
                if first_frame_pointmap is not None:
                    latents_pointmap[:, :, 0:1] = first_frame_pointmap.clone()
        return (latents_video, latents_dino, latents_pointmap, latents_action)
