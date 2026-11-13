from __future__ import annotations
from cloudrobo_wm.contracts import data_protocol
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from cloudrobo_wm.utils.logging_config import get_logger
from .cloudrobo_wm import CloudRoboWM
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)
FLOW_LAYOUT_VERSION = 1
FLOW_ANCHOR_MODE = 'omitted'
FLOW_CACHE_SCHEMA = data_protocol('protocol_54d6a2eea6b0')

def _bool_switch(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'expected a boolean flow-mode switch, got {value!r}')

def normalize_flow_latents_clip(raw_latents: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    if raw_latents.ndim != 5:
        raise ValueError(f'flow VAE latents must be [B,C,T,H,W], got {tuple(raw_latents.shape)}')
    if raw_latents.shape[2] != 3:
        raise ValueError(f'S0 requires exactly three causal VAE latent times from nine flow frames, got T={raw_latents.shape[2]}')
    work = raw_latents.float()
    dims = (1, 2, 3, 4)
    mean = work.mean(dim=dims, keepdim=True)
    std = work.std(dim=dims, keepdim=True).clamp_min(float(eps))
    return ((work - mean) / std).to(dtype=raw_latents.dtype)

def flow_future_group_valid(transition_valid: torch.Tensor) -> torch.Tensor:
    if transition_valid.ndim != 2 or transition_valid.shape[1] != 8:
        raise ValueError(f'flow mode expects flow_transition_valid [B,8], got {tuple(transition_valid.shape)}')
    return torch.stack((transition_valid[:, 0:4].any(dim=1), transition_valid[:, 4:8].any(dim=1)), dim=1)

class FlowCloudRoboWM(CloudRoboWM):
    _FLOW_CKPT_KEYS: tuple[str, ...] = ('flow_patch_embedding', 'flow_head')

    def __init__(self, *args, flow_loss_lambda: float=0.5, p_joint_flow: float=0.5, cross_modal_predict_flow: bool=True, flow_layout_version: int=FLOW_LAYOUT_VERSION, flow_anchor_mode: str=FLOW_ANCHOR_MODE, flow_cache_schema: str=FLOW_CACHE_SCHEMA, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_flow_branch(flow_loss_lambda=flow_loss_lambda, p_joint_flow=p_joint_flow, cross_modal_predict_flow=cross_modal_predict_flow, flow_layout_version=flow_layout_version, flow_anchor_mode=flow_anchor_mode, flow_cache_schema=flow_cache_schema)

    def _init_flow_branch(self, *, flow_loss_lambda: float, p_joint_flow: float, cross_modal_predict_flow: bool, flow_layout_version: int, flow_anchor_mode: str, flow_cache_schema: str) -> None:
        if hasattr(self, 'flow_patch_embedding'):
            raise RuntimeError('FlowCloudRoboWM flow branch was initialized twice.')
        if not 0.0 <= float(p_joint_flow) <= 1.0:
            raise ValueError(f'p_joint_flow must be in [0,1], got {p_joint_flow}')
        if not bool(cross_modal_predict_flow):
            raise ValueError('S0 has no observed O_f range, so U_f must remain an active cross-modal prediction target (cross_modal_predict_flow=true).')
        if int(flow_layout_version) != FLOW_LAYOUT_VERSION:
            raise ValueError(f'unsupported flow_layout_version={flow_layout_version}; expected {FLOW_LAYOUT_VERSION}')
        if str(flow_anchor_mode) != FLOW_ANCHOR_MODE:
            raise ValueError("S0 supports only flow_anchor_mode='omitted'; present/history flow tokens are intentionally not part of this layout.")
        if str(flow_cache_schema) != FLOW_CACHE_SCHEMA:
            raise ValueError(f'unsupported flow_cache_schema={flow_cache_schema!r}; expected {FLOW_CACHE_SCHEMA!r}')
        self.flow_patch_embedding = copy.deepcopy(self.video_expert.patch_embedding)
        self.flow_head = copy.deepcopy(self.video_expert.head)
        self.loss_lambda_flow = float(flow_loss_lambda)
        self.p_joint_flow = float(p_joint_flow)
        self.cross_modal_predict_flow = True
        self.flow_layout_version = FLOW_LAYOUT_VERSION
        self.flow_anchor_mode = FLOW_ANCHOR_MODE
        self.flow_cache_schema = str(flow_cache_schema)
        self.train_flow_scheduler = copy.deepcopy(self.train_video_scheduler)
        self.infer_flow_scheduler = copy.deepcopy(self.infer_video_scheduler)
        self._batch_joint_flow: Optional[torch.Tensor] = None
        self._infer_joint_flow = False
        self.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_flow_target(self, flow_video: torch.Tensor, tiled: bool=False) -> torch.Tensor:
        if flow_video.ndim != 5 or flow_video.shape[1] != 3:
            raise ValueError(f'flow_video must be [B,3,9,H,W], got {tuple(flow_video.shape)}')
        if flow_video.shape[2] != 9:
            raise ValueError(f'S0 requires [history, eight future] = nine flow frames, got T={flow_video.shape[2]}')
        flow_video = flow_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        raw = self._encode_video_latents(flow_video, tiled=tiled)
        normalized = normalize_flow_latents_clip(raw)
        return normalized[:, :, 1:3]

    def build_inputs(self, sample, tiled: bool=False):
        inputs = super().build_inputs(sample, tiled=tiled)
        if 'flow_video' not in sample:
            raise KeyError("FlowCloudRoboWM requires sample['flow_video']; use the isolated flow-aware dataset preset.")
        if 'flow_history_valid' not in sample:
            raise KeyError("FlowCloudRoboWM requires sample['flow_history_valid'].")
        if 'flow_transition_valid' not in sample:
            raise KeyError("FlowCloudRoboWM requires sample['flow_transition_valid'] [B,8].")
        with torch.no_grad():
            inputs['flow_target'] = self._encode_flow_target(sample['flow_video'], tiled=tiled)
        history_valid = sample['flow_history_valid']
        if not torch.is_tensor(history_valid):
            history_valid = torch.as_tensor(history_valid)
        history_valid = history_valid.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if history_valid.ndim == 0:
            history_valid = history_valid.unsqueeze(0)
        if history_valid.ndim != 1 or history_valid.shape[0] != inputs['flow_target'].shape[0]:
            raise ValueError(f'flow_history_valid must be bool [B], got {tuple(history_valid.shape)}')
        inputs['flow_history_valid'] = history_valid
        transition_valid = sample['flow_transition_valid']
        if not torch.is_tensor(transition_valid):
            transition_valid = torch.as_tensor(transition_valid)
        transition_valid = transition_valid.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if transition_valid.ndim == 1 and history_valid.shape[0] == 1:
            transition_valid = transition_valid.unsqueeze(0)
        if transition_valid.shape != (history_valid.shape[0], 8):
            raise ValueError(f'flow_transition_valid must be bool [B,8], got {tuple(transition_valid.shape)}')
        inputs['flow_transition_valid'] = transition_valid
        return inputs

    def _flow_freqs(self, device: torch.device) -> torch.Tensor:
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        h, w = (24 // patch_h, 20 // patch_w)
        temporal = self.video_expert.freqs[0][1:3]
        spatial_h = self.video_expert.freqs[1][:h]
        spatial_w = self.video_expert.freqs[2][:w]
        return torch.cat([temporal.view(2, 1, 1, -1).expand(2, h, w, -1), spatial_h.view(1, h, 1, -1).expand(2, h, w, -1), spatial_w.view(1, 1, w, -1).expand(2, h, w, -1)], dim=-1).reshape(2 * h * w, 1, -1).to(device)

    def _flow_pre_dit(self, latents: torch.Tensor, timestep_video: torch.Tensor) -> dict:
        if latents.ndim != 5 or latents.shape[1:] != (self.vae.model.z_dim, 2, 24, 20):
            raise ValueError(f'future flow latents must be [B,z_dim,2,24,20], got {tuple(latents.shape)}')
        x = self.flow_patch_embedding(latents)
        B, hidden, ft, fh, fw = x.shape
        if ft != 2:
            raise RuntimeError(f'flow patch projection must preserve the two latent times; got projected T={ft}')
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, -1, hidden).contiguous()
        if tokens.shape[1] != 240:
            raise RuntimeError(f'S0 U_f must contain 240 tokens, got {tokens.shape[1]}')
        token_t = timestep_video.view(B, 1).expand(B, tokens.shape[1])
        t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, token_t.reshape(-1))
        t = self.video_expert.time_embedding(t_emb).reshape(B, tokens.shape[1], hidden)
        t_mod = self.video_expert.time_projection(t).unflatten(2, (6, hidden))
        return {'tokens': tokens, 'freqs': self._flow_freqs(tokens.device), 't': t, 't_mod': t_mod, 'meta': {'grid_size': (2, fh, fw), 'tokens_per_frame': fh * fw}}

    def _flow_post_dit(self, tokens: torch.Tensor, pre: dict) -> torch.Tensor:
        patches = self.flow_head(tokens, pre['t'])
        return self.video_expert.unpatchify(patches, pre['meta']['grid_size'])

    @staticmethod
    def _merge_flow_tokens(merged_tokens: torch.Tensor, merged_freqs: torch.Tensor, merged_t_mod: torch.Tensor, merged_context_mask: torch.Tensor, flow_pre: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((merged_tokens, flow_pre['tokens']), dim=1)
        freqs = torch.cat((merged_freqs, flow_pre['freqs']), dim=0)
        t_mod = torch.cat((merged_t_mod, flow_pre['t_mod']), dim=1)
        context_mask = merged_context_mask
        if context_mask.ndim == 3:
            flow_ctx = context_mask[:, :1, :].expand(-1, flow_pre['tokens'].shape[1], -1)
            context_mask = torch.cat((context_mask, flow_ctx), dim=1)
        return (tokens, freqs, t_mod, context_mask)

    @staticmethod
    def _insert_flow_range(base: torch.Tensor, visual_len: int, flow_len: int, action_len: int) -> torch.Tensor:
        shape_prefix = base.shape[:-2]
        total = visual_len + flow_len + action_len
        out = torch.zeros((*shape_prefix, total, total), dtype=torch.bool, device=base.device)
        old_a = visual_len
        new_a = visual_len + flow_len
        out[..., :visual_len, :visual_len] = base[..., :visual_len, :visual_len]
        out[..., :visual_len, new_a:] = base[..., :visual_len, old_a:]
        out[..., new_a:, :visual_len] = base[..., old_a:, :visual_len]
        out[..., new_a:, new_a:] = base[..., old_a:, old_a:]
        return out

    @torch.no_grad()
    def _build_flow_attention_mask(self, Sv: int, Sa: int, Sd: int, Sp: int, Sf: int, tpf: int, dtpf: int, ptpf: int, device: torch.device) -> torch.Tensor:
        if Sf != 240:
            raise ValueError(f'S0 U_f length must be 240, got {Sf}')
        base = super()._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        visual_len = Sv + Sd + Sp
        out = self._insert_flow_range(base, visual_len, Sf, Sa)
        bf = getattr(self, '_batch_modalities', None)
        batch_jf = getattr(self, '_batch_joint_flow', None)
        if bf is None and batch_jf is None:
            regimes = [(bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap), bool(getattr(self, '_infer_joint_flow', False)), bool(self._infer_present_v), bool(self._infer_present_d), bool(self._infer_present_p), bool(self.joint_modalities.cross_modal_predict_video), bool(self.joint_modalities.cross_modal_predict_dino), bool(self.joint_modalities.cross_modal_predict_pointmap))]
        elif bf is None:
            out = out.unsqueeze(0).expand(batch_jf.shape[0], -1, -1).clone()
            regimes = [(bool(self.joint_video), bool(self.joint_dino), bool(self.joint_pointmap), bool(jf), True, True, not self._pointmap_globally_off, bool(self.joint_modalities.cross_modal_predict_video), bool(self.joint_modalities.cross_modal_predict_dino), bool(self.joint_modalities.cross_modal_predict_pointmap)) for jf in batch_jf.tolist()]
        else:
            jf = self._batch_joint_flow
            if jf is None or jf.shape != (bf.B,):
                raise RuntimeError('training flow joint flags were not initialized')
            regimes = list(zip(bf.j_v.tolist(), bf.j_d.tolist(), bf.j_p.tolist(), jf.tolist(), bf.present_v.tolist(), bf.present_d.tolist(), bf.present_p.tolist(), [bf.cm_v] * bf.B, [bf.cm_d] * bf.B, [bf.cm_p] * bf.B))
        masks = [out] if out.ndim == 2 else [out[b] for b in range(out.shape[0])]
        if len(masks) != len(regimes):
            raise RuntimeError('flow mask regime/batch mismatch')
        d0, p0, f0, a0 = (Sv, Sv + Sd, visual_len, visual_len + Sf)
        stream_ranges = ((0, tpf, Sv, 'v'), (d0, d0 + dtpf, d0 + Sd, 'd'), (p0, p0 + ptpf, p0 + Sp, 'p'))
        for mask, regime in zip(masks, regimes):
            jv, jd, jp, jf, pv, pd, pp, cmv, cmd, cmp = map(bool, regime)
            js = {'v': jv, 'd': jd, 'p': jp}
            present = {'v': pv, 'd': pd, 'p': pp}
            cm = {'v': cmv, 'd': cmd, 'p': cmp}
            mask[f0:a0, f0:a0] = True
            for start, rem_start, end, name in stream_ranges:
                if present[name] and rem_start > start:
                    mask[f0:a0, start:rem_start] = True
                u_active = present[name] or cm[name]
                if rem_start < end and u_active and (js[name] == jf):
                    mask[f0:a0, rem_start:end] = True
                    mask[rem_start:end, f0:a0] = True
            if jf:
                mask[a0:, f0:a0] = True
            mask[:a0, a0:] = False
        return out

    @torch.no_grad()
    def _build_flow_hbridge_masks(self, Sv: int, Sa: int, Sd: int, Sp: int, Sf: int, tpf: int, dtpf: int, ptpf: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        if not self.mot.hbridge_enabled:
            return (None, None)
        lens, masks = super()._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        lens, masks = (list(lens), list(masks))
        insert_at = len(lens) - (1 if Sa > 0 else 0)
        lens.insert(insert_at, Sf)
        masks.insert(insert_at, torch.ones((Sf, Sf), dtype=torch.bool, device=device))
        return (lens, masks)

    def _training_mot(self, **kwargs):
        return self.mot(**kwargs)

    def _sample_flow_joint_flags(self, batch_size: int) -> torch.Tensor:
        return torch.rand((batch_size,), device=self.device) < self.p_joint_flow

    def _corrupt_flow_target(self, flow_target: torch.Tensor, t_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_flow = torch.randn_like(flow_target)
        noisy_flow = self.train_flow_scheduler.add_noise(flow_target, noise_flow, t_video)
        target_flow = self.train_flow_scheduler.training_target(flow_target, noise_flow, t_video)
        return (noisy_flow, target_flow, noise_flow)

    def _flow_loss(self, pred_flow: torch.Tensor, target_flow: torch.Tensor, t_video: torch.Tensor, history_valid: torch.Tensor, transition_valid: torch.Tensor) -> torch.Tensor:
        per_time = F.mse_loss(pred_flow.float(), target_flow.float(), reduction='none').mean(dim=(1, 3, 4))
        group_valid = flow_future_group_valid(transition_valid).to(device=per_time.device)
        valid_time_bool = group_valid & history_valid.to(device=per_time.device, dtype=torch.bool).unsqueeze(1)
        valid_time = valid_time_bool.to(dtype=per_time.dtype)
        per_sample = (per_time * valid_time).sum(dim=1) / valid_time.sum(dim=1).clamp_min(1.0)
        valid_sample = valid_time_bool.any(dim=1).to(dtype=per_sample.dtype)
        weight = self.train_flow_scheduler.training_weight(t_video).to(device=per_sample.device, dtype=per_sample.dtype)
        denom = valid_sample.sum().clamp_min(1.0)
        return (per_sample * weight * valid_sample).sum() / denom

    def _base_training_loss(self, sample, tiled: bool=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs['input_latents']
        B = input_latents.shape[0]
        context, context_mask = (inputs['context'], inputs['context_mask'])
        action = inputs['action']
        action_is_pad, image_is_pad = (inputs['action_is_pad'], inputs['image_is_pad'])
        dino_features = inputs['dino_features']
        pointmap_raw = inputs['pointmap_raw']
        flow_target = inputs['flow_target']
        sampled_joint_flow = self._sample_flow_joint_flags(B)
        noise_video = torch.randn_like(input_latents)
        t_video = self.train_video_scheduler.sample_training_t(B, self.device, input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, t_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, t_video)
        if inputs['first_frame_latents'] is not None:
            latents[:, :, 0:1] = inputs['first_frame_latents']
        noisy_flow, target_flow, _noise_flow = self._corrupt_flow_target(flow_target, t_video)
        noise_action = torch.randn_like(action)
        if 'action_valid' in inputs:
            noise_action = torch.where(inputs['action_valid'], noise_action, 0)
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
        self._batch_joint_flow = sampled_joint_flow
        try:
            video_pre = self.video_expert.pre_dit(x=latents, timestep=t_video, context=context, context_mask=context_mask, action=action, fuse_vae_embedding_in_latents=inputs['fuse_vae_embedding_in_latents'])
            action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=t_action, context=context, context_mask=context_mask)
            flow_pre = self._flow_pre_dit(noisy_flow, t_video)
            Sv = video_pre['tokens'].shape[1]
            Sa = action_pre['tokens'].shape[1]
            Sf = flow_pre['tokens'].shape[1]
            tpf = int(video_pre['meta']['tokens_per_frame'])
            video_pre['tokens'] = self._flex_zero_absent_video_tokens(video_pre['tokens'], tpf)
            dino_tokens = self._embed_dino(noisy_dino)
            Sd = dino_tokens.shape[1]
            num_dino_frames = int(dino_features.shape[2])
            dtpf = int(dino_features.shape[3])
            dino_tokens = self._flex_zero_absent_dino_tokens(dino_tokens, dtpf)
            dino_token_t = t_dino.view(B, 1).expand(B, Sd).clone()
            dino_token_t[:, :dtpf] = 0
            dino_t_mod = self._build_stream_t_mod(dino_token_t, Sd, B)
            dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre['tokens'].device)
            if self._pointmap_globally_off:
                pointmap_tokens = pt_t_mod = pt_freqs = pt_t = pt_meta = None
                Sp = ptpf = 0
            else:
                pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(noisy_pointmap)
                pointmap_tokens = self._flex_zero_absent_pointmap_tokens(pointmap_tokens, ptpf)
                Sp = pointmap_tokens.shape[1]
                pt_token_t = t_pointmap.view(B, 1).expand(B, Sp).clone()
                pt_token_t[:, :ptpf] = 0
                pt_t_mod = self._build_stream_t_mod(pt_token_t, Sp, B)
                pt_freqs = self._compute_pointmap_freqs(int(noisy_pointmap.shape[2]), video_pre['tokens'].device, pt_meta=pt_meta)
                pt_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, pt_token_t.reshape(-1))
                pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(B, Sp, self.video_expert.hidden_dim)
            merged = self._merge_aux_into_video_stream(video_pre, dino_tokens, dino_t_mod, dino_freqs, pointmap_tokens, pt_t_mod, pt_freqs)
            merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = self._merge_flow_tokens(*merged, flow_pre)
            attention_mask = self._build_flow_attention_mask(Sv, Sa, Sd, Sp, Sf, tpf, dtpf, ptpf, merged_tokens.device)
            sub_lens, sub_masks = self._build_flow_hbridge_masks(Sv, Sa, Sd, Sp, Sf, tpf, dtpf, ptpf, merged_tokens.device)
            if getattr(self, '_ms_active', False):
                attention_mask, sub_masks = self._multisource_attention(attention_mask, sub_lens, sub_masks)
            tokens_out = self._training_mot(embeds_all={'video': merged_tokens, 'action': action_pre['tokens']}, attention_mask=attention_mask, freqs_all={'video': merged_freqs, 'action': action_pre['freqs']}, context_all={'video': {'context': video_pre['context'], 'mask': merged_ctx_mask}, 'action': {'context': action_pre['context'], 'mask': action_pre['context_mask']}}, t_mod_all={'video': merged_t_mod, 'action': action_pre['t_mod']}, sub_stream_lens=sub_lens, sub_stream_self_masks=sub_masks)
            visual_out = tokens_out['video']
            video_tokens_out = visual_out[:, :Sv]
            dino_tokens_out = visual_out[:, Sv:Sv + Sd]
            pt_tokens_out = visual_out[:, Sv + Sd:Sv + Sd + Sp]
            flow_tokens_out = visual_out[:, Sv + Sd + Sp:]
            pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
            pred_action = self.action_expert.post_dit(tokens_out['action'], action_pre)
            pred_dino = self._project_dino_out(dino_tokens_out)
            pred_pointmap = None if self._pointmap_globally_off else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
            pred_flow = self._flow_post_dit(flow_tokens_out, flow_pre)
        finally:
            self._batch_joint_flow = None
        include_init = inputs['first_frame_latents'] is None
        if not include_init:
            pred_video, target_video = (pred_video[:, :, 1:], target_video[:, :, 1:])
        lv = self._compute_video_loss_per_sample(pred_video, target_video, image_is_pad, include_init)
        wv = self.train_video_scheduler.training_weight(t_video).to(lv.device, dtype=lv.dtype)
        loss_video = self._flex_reduce_per_sample_loss(lv, wv, present_mask=self._batch_modalities.present_v if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_v if self._batch_modalities is not None else False)
        la = F.mse_loss(pred_action.float(), target_action.float(), reduction='none').mean(2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(la.device, dtype=la.dtype)
            la = (la * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        else:
            la = la.mean(1)
        wa = self.train_action_scheduler.training_weight(t_action).to(la.device, dtype=la.dtype)
        loss_action = (la * wa).mean()
        target_dino_flat = target_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(B, -1, self.dino_dim)
        if self.dino_pred_x0:
            from .helpers.dino import _dino_x0_to_velocity
            noisy_flat = noisy_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(B, -1, self.dino_dim)
            pred_dino = _dino_x0_to_velocity(noisy_flat, pred_dino, t_dino.view(B, 1, 1), self.train_dino_scheduler.num_train_timesteps)
        ld = F.mse_loss(pred_dino.float(), target_dino_flat.float(), reduction='none').mean(2)
        n_rem = num_dino_frames - 1
        if image_is_pad is None or n_rem <= 0:
            ld = ld[:, dtpf:].mean(1)
        else:
            ld = ld[:, dtpf:].view(B, n_rem, dtpf).mean(2)
            dino_pad = self._aux_per_frame_is_pad(image_is_pad, self.dino_temporal_stride, keep_far=self.dino_stride_keep_far)[:, 1:]
            ld = self._masked_loss_reduction(ld, dino_pad)
        wd = self.train_dino_scheduler.training_weight(t_dino).to(ld.device, dtype=ld.dtype)
        loss_dino = self._flex_reduce_per_sample_loss(ld, wd, present_mask=self._batch_modalities.present_d if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_d if self._batch_modalities is not None else False)
        if target_pointmap is None:
            loss_pointmap = torch.zeros((), device=self.device)
        else:
            pred_pt = pred_pointmap[:, :, 1:].float()
            target_pt = target_pointmap[:, :, 1:].float()
            if image_is_pad is None:
                lp = F.mse_loss(pred_pt, target_pt, reduction='none').flatten(1).mean(1)
            else:
                lp = self._compute_video_loss_per_sample(pred_pt, target_pt, image_is_pad, False)
            wp = self.train_pointmap_scheduler.training_weight(t_pointmap).to(lp.device, dtype=lp.dtype)
            loss_pointmap = self._flex_reduce_per_sample_loss(lp, wp, present_mask=self._batch_modalities.present_p if self._batch_modalities is not None else None, cross_modal_active=self._batch_modalities.cm_p if self._batch_modalities is not None else False)
        loss_flow = self._flow_loss(pred_flow, target_flow, t_video, inputs['flow_history_valid'], inputs['flow_transition_valid'])
        if getattr(self, '_ms_active', False):
            loss_video, loss_action, loss_dino, loss_pointmap, loss_flow = self._multisource_losses(locals())
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + self.loss_lambda_dino * loss_dino + self.loss_lambda_pointmap * loss_pointmap + self.loss_lambda_flow * loss_flow
        metric = (lambda value: value.detach()) if getattr(self, '_ms_active', False) else float
        losses = {'loss_video': self.loss_lambda_video * metric(loss_video.detach()), 'loss_action': self.loss_lambda_action * metric(loss_action.detach()), 'loss_dino': self.loss_lambda_dino * metric(loss_dino.detach()), 'loss_pointmap': self.loss_lambda_pointmap * metric(loss_pointmap.detach()), 'loss_flow': self.loss_lambda_flow * metric(loss_flow.detach())}
        if getattr(self, '_ms_active', False):
            losses.update(self._multisource_loss_metrics)
        return (total, losses)

    def _mode_ckpt_keys(self) -> tuple:
        return super()._mode_ckpt_keys() + self._FLOW_CKPT_KEYS

    def _flow_metadata(self) -> dict:
        return {'dino_objective_contract': {'version': 1, 'objective': 'legacy_x0_via_noisy_to_velocity_loss', 'dino_pred_x0': True, 'training_noisy_x0_conversion': True, 'training_sigma_division': True, 'sigma_floor_in_legacy_conversion': 0.05, 'clean_x0_loss_repair_enabled': False, 'anchor_supervised': False, 'pretrained_initialization': 'same_raw170000_retained_tensors_no_extra_head_reset', 'inference': 'legacy_x0_to_velocity_then_original_scheduler'}, 'flow_layout_version': self.flow_layout_version, 'flow_anchor_mode': self.flow_anchor_mode, 'flow_cache_schema': self.flow_cache_schema}

    def _validate_checkpoint_payload(self, payload, *, strict_shape: bool):
        super()._validate_checkpoint_payload(payload, strict_shape=strict_shape)
        if strict_shape:
            missing = [key for key in self._FLOW_CKPT_KEYS if key not in payload]
            if missing:
                raise ValueError(f'Strict flow resume requires all flow interfaces; missing {missing}. Use strict_shape=False only for a legacy warm-start.')
            expected = self._flow_metadata()
            keys = set(expected)
            if 'flow_input_normalization' in payload:
                keys.add('flow_input_normalization')
            mismatch = {key: (payload.get(key), expected.get(key)) for key in keys if payload.get(key) != expected.get(key)}
            if mismatch:
                raise ValueError(f'Strict flow resume metadata mismatch (saved, expected): {mismatch}')
        else:
            expected_representation = getattr(self, 'flow_input_normalization', None)
            saved_representation = payload.get('flow_input_normalization')
            if saved_representation != expected_representation:
                logger.warning('Non-strict flow warm-start changes the input representation (saved=%s, destination=%s). Loading checkpoint weights; keeping the destination representation. This is not a strict resume.', saved_representation, expected_representation)

def _validate_matching_flow_scheduler(flow_scheduler, video_scheduler) -> None:
    if flow_scheduler is None:
        return
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(flow_scheduler, DictConfig):
            flow_scheduler = OmegaConf.to_container(flow_scheduler, resolve=True)
        if isinstance(video_scheduler, DictConfig):
            video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    except ImportError:
        pass
    flow_scheduler = dict(flow_scheduler)
    video_scheduler = dict(video_scheduler or {})
    for key in ('train_shift', 'infer_shift', 'num_train_timesteps'):
        if key in flow_scheduler and key in video_scheduler:
            if flow_scheduler[key] != video_scheduler[key]:
                raise ValueError(f'S0 requires the flow schedule to equal the video schedule; {key}: flow={flow_scheduler[key]!r}, video={video_scheduler[key]!r}')
