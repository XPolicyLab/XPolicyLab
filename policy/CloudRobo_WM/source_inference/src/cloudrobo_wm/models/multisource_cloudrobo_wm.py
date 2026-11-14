from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from .flow_cloudrobo_wm import FlowCloudRoboWM
from .backbone import CloudRoboWMBackbone
from .multisource_conditioning import MultiSourceConditioning, label_masks, restrict_attention
logger = logging.getLogger(__name__)

class MultiSourceCloudRoboWM(FlowCloudRoboWM):
    _FLOW_CKPT_KEYS = FlowCloudRoboWM._FLOW_CKPT_KEYS + ('multisource_conditioning',)

    def enable_dual_cooperative_action(self):
        from .dual_multisource import enable_dual_cooperative_action
        return enable_dual_cooperative_action(self)

    def init_multisource(self, condition_count=256, field_weights=None, horizon_weights=None, skip_unlabeled_flow_tokens=False, text_config=None, perception=None, dino_inference_batch_size=27, perception_checkpoint_layers=True, perception_checkpoint_every=1):
        from cloudrobo_wm.text_conditioning import text_contract, validate_text_contract
        self.text_contract = validate_text_contract(text_config or text_contract())
        import math
        for values, size, name in ((field_weights, 90, 'field_weights'), (horizon_weights, 32, 'horizon_weights')):
            if values is not None and (len(values) != size or any((not math.isfinite(x) or x < 0 for x in values))):
                raise ValueError(f'{name} requires {size} finite nonnegative weights')
        if self.proprio_dim != 96 or self.action_expert.action_dim != 90:
            raise ValueError('MultiSourceCloudRoboWM requires proprio_dim=96, action_dim=90')
        if self.dino_temporal_stride != 1:
            raise ValueError('Initial multisource timing contract requires dino_temporal_stride=1')
        dims = {k: self.video_expert.hidden_dim for k in ('video', 'dino', 'pointmap', 'flow')}
        dims['action'] = self.action_expert.action_encoder.out_features
        self.multisource_conditioning = MultiSourceConditioning(self.proprio_encoder.out_features, dims, condition_count).to(device=self.device, dtype=self.torch_dtype)
        self.multisource_contract = None
        self.multisource_field_weights = field_weights
        self.multisource_horizon_weights = horizon_weights
        self.skip_unlabeled_flow_tokens = bool(skip_unlabeled_flow_tokens)
        if isinstance(dino_inference_batch_size, bool) or not isinstance(dino_inference_batch_size, int) or dino_inference_batch_size <= 0:
            raise ValueError('dino_inference_batch_size must be a positive integer')
        if not isinstance(perception_checkpoint_layers, bool):
            raise ValueError('perception_checkpoint_layers must be boolean')
        if isinstance(perception_checkpoint_every, bool) or not isinstance(perception_checkpoint_every, int) or perception_checkpoint_every <= 0:
            raise ValueError('perception_checkpoint_every must be a positive integer')
        self.dino_inference_batch_size = dino_inference_batch_size
        self.perception_checkpoint_layers = perception_checkpoint_layers
        self.perception_checkpoint_every = perception_checkpoint_every
        self._ms_active = False
        self._ms_sample = {}
        self._ms_token_valid = {}
        self._ms_handles = []
        for branch, module in (('video', self.video_expert.patch_embedding), ('flow', self.flow_patch_embedding), ('pointmap', self.pt_patch_embedding), ('dino', self.dino_embedder), ('action', self.action_expert.action_encoder)):
            self._ms_handles.append(module.register_forward_hook(self._embedding_hook(branch)))
        self._ms_handles.append(self.action_expert.action_encoder.register_forward_pre_hook(self._action_input_hook))
        self.init_perception(perception)

    def _mode_ckpt_keys(self):
        return super()._mode_ckpt_keys() + (('perception',) if hasattr(self, 'perception') else ())

    def bind_multisource_contract(self, contract):
        from cloudrobo_wm.text_conditioning import checkpoint_contract
        contract = checkpoint_contract(contract, warn_legacy=True)
        if contract['text'] != self.text_contract:
            raise ValueError('Dataset/checkpoint and model text conditioning contracts differ')
        count = len(contract['conditions'])
        if count > self.multisource_conditioning.conditions.num_embeddings:
            raise ValueError('Condition vocabulary exceeds model capacity')
        self.multisource_contract = contract

    def encode_prompt(self, prompt):
        if hasattr(self, 'text_contract') and self.tokenizer is not None:
            from cloudrobo_wm.text_conditioning import check_prompt_lengths
            check_prompt_lengths(self.tokenizer, prompt, self.text_contract)
        return super().encode_prompt(prompt, padding_mode='masked')

    def prepare_for_inference(self, *args, **kwargs):
        if kwargs.get('attn_backend') == 'flex' and kwargs.get('joint_loop_cuda_graph', False):
            raise ValueError('Changing label masks requires SDPA/auto for the manual whole-loop CUDA graph')
        self.video_expert.freqs = tuple((x.to(self.device) for x in self.video_expert.freqs))
        self.action_expert.freqs = self.action_expert.freqs.to(self.device)
        return super().prepare_for_inference(*args, **kwargs)

    def _set_multisource_context(self, sample):

        def copy_tree(old, new):
            if isinstance(new, dict):
                return {k: copy_tree(old.get(k) if isinstance(old, dict) else None, v) for k, v in new.items()}
            if torch.is_tensor(new):
                value = new.to(self.device)
                if not self.training and torch.is_tensor(old) and (old.shape == value.shape) and (old.dtype == value.dtype):
                    old.copy_(value)
                    return old
                return value
            return new
        names = ('state_valid', 'action_valid', 'condition_id', 'camera_valid', 'modality_valid', 'image_is_pad', 'flow_history_valid', 'flow_transition_valid', 'time_grid', 'time_valid', 'pointmap_pixel_valid', 'flow_pixel_valid')
        names = names + ('normalization_counts',)
        self._ms_sample = copy_tree(self._ms_sample, {k: sample[k] for k in names if k in sample})
        self._ms_masks = copy_tree(getattr(self, '_ms_masks', {}), label_masks(self._ms_sample, self.device, tuple((h * w for h, w in self.dino_cam_patches))))
        self._ms_token_valid = {}
        self._ms_active = True

    def _append_proprio_to_context(self, context, context_mask, proprio):
        if not self._ms_active:
            return super()._append_proprio_to_context(context, context_mask, proprio)
        valid = self._ms_sample['state_valid'].bool()
        if valid.ndim == 3:
            valid = valid[:, 0]
        p = torch.where(valid, proprio, 0).to(context.dtype)
        token = self.proprio_encoder(p) + self.multisource_conditioning.state_valid(valid.to(context.dtype))
        mask = valid.any(-1, keepdim=True)
        token = token[:, None] * mask[..., None]
        condition = self.multisource_conditioning.conditions(self._ms_sample['condition_id'].long())[:, None]
        return (torch.cat([context, token, condition.to(context.dtype)], 1), torch.cat([context_mask, mask, torch.ones_like(mask)], 1))

    def _action_input_hook(self, module, args):
        if not self._ms_active:
            return args
        valid = self._ms_masks['action']
        return (torch.where(valid, args[0], 0), *args[1:])

    def _embedding_hook(self, branch):

        def hook(module, args, output):
            if not self._ms_active:
                return output
            original = output.shape
            spatial = output.ndim == 5
            if spatial:
                B, D, T, H, W = original
                tokens = output.flatten(2).transpose(1, 2)
                raw = self._ms_masks[branch][:, :T]
                m = F.adaptive_avg_pool2d(raw.float().flatten(0, 1)[:, None], (H, W))[:, 0] >= 1 - 1e-07
                valid = m.reshape(B, T * H * W)
                tpf = H * W
            else:
                tokens = output
                if branch == 'action':
                    valid = self._ms_masks[branch].any(-1)
                    tokens = tokens + self.multisource_conditioning.action_valid(self._ms_masks[branch].to(tokens.dtype))
                    T, tpf = (valid.shape[1], 1)
                else:
                    raw = self._ms_masks[branch]
                    tpf = raw.shape[-1]
                    T = tokens.shape[1] // tpf
                    valid = raw[:, :T].flatten(1)
            grid = self._ms_sample['time_grid'][branch][:, :T]
            tm = self._ms_sample['time_valid'][branch][:, :T]
            time = self.multisource_conditioning.time(branch, grid, tm, tokens.dtype)
            tokens = (tokens + time.repeat_interleave(tpf, 1)) * valid[..., None]
            self._ms_token_valid[branch] = valid
            return tokens.transpose(1, 2).reshape(original) if spatial else tokens
        return hook

    def _multisource_attention(self, mask, sub_lens=None, sub_masks=None):
        order = ('video', 'dino', 'pointmap', 'flow', 'action')
        values = [self._ms_token_valid[k] for k in order if k in self._ms_token_valid]
        valid = torch.cat(values, 1)
        result = restrict_attention(mask, valid)
        if sub_lens is not None and sub_masks is not None:
            start = 0
            updated = []
            for length, sm in zip(sub_lens, sub_masks):
                updated.append(restrict_attention(sm, valid[:, start:start + length]))
                start += length
            return (result, updated)
        return (result, sub_masks)

    def _sequence_valid(self, branch, length, tpf):
        raw = self._ms_masks[branch]
        if branch == 'action':
            return raw.any(-1)[:, :length]
        if not length:
            return raw.new_zeros((raw.shape[0], 0))
        frames = length // tpf
        if raw.ndim == 4:
            h, w = self.video_expert.patch_size[1:]
            raw = F.avg_pool2d(raw[:, :frames].float().flatten(0, 1)[:, None], (h, w))[:, 0] >= 1 - 1e-07
            return raw.reshape(-1, length)
        return raw[:, :frames].flatten(1)

    def _restricted_layout(self, mask, layout):
        valid = torch.cat([self._sequence_valid(k, n, p) for k, n, p in layout], 1)
        result = restrict_attention(mask, valid)
        return result[0] if mask.ndim == 2 and valid.shape[0] == 1 else result

    def _build_video_hbridge_self_masks_unified(self, video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        lens, masks = super()._build_video_hbridge_self_masks_unified(video_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device)
        if not self._ms_active or lens is None:
            return (lens, masks)
        layout = [(k, n, p) for k, n, p in (('video', video_seq_len, video_tokens_per_frame), ('dino', dino_seq_len, dino_tokens_per_frame), ('pointmap', pointmap_seq_len, pointmap_tokens_per_frame)) if n]
        return (lens, [self._restricted_layout(m, [part]) for m, part in zip(masks, layout)])

    def _build_video_hbridge_self_masks_dino_pointmap_only(self, dino_seq_len, pointmap_seq_len, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        return self._build_video_hbridge_self_masks_unified(0, dino_seq_len, pointmap_seq_len, 0, dino_tokens_per_frame, pointmap_tokens_per_frame, device)

    def _build_mot_attention_mask_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        mask = super()._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        if not self._ms_active:
            return mask
        return self._restricted_layout(mask, [('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)])

    def _build_hbridge_self_masks_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        lens, masks = super()._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        if not self._ms_active or lens is None:
            return (lens, masks)
        layout = [(k, n, p) for k, n, p in (('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)) if n > 0]
        return (lens, [self._restricted_layout(m, [part]) for m, part in zip(masks, layout)])

    @torch.no_grad()
    def infer_action(self, *args, state_valid=None, action_valid=None, condition_id=None, time_grid=None, time_valid=None, camera_valid=None, pointmap_pixel_valid=None, **kwargs):
        if state_valid is None or condition_id is None:
            raise ValueError('S96/A90 inference requires state_valid and deployment condition_id')
        B = state_valid.shape[0]
        horizon = kwargs.get('action_horizon', 32)
        if horizon != 32:
            raise ValueError('S96/A90 inference uses 32-action chunks')
        if action_valid is None:
            raise ValueError('S96/A90 inference requires the deployment action field mask')
        device = self.device
        zeros = lambda n: torch.zeros(B, n, device=device, dtype=torch.bool)
        shapes = dict(action=32, video=3, pointmap=3, dino=3, flow=2)
        sample = dict(state_valid=state_valid, action_valid=action_valid, condition_id=condition_id, camera_valid=camera_valid if camera_valid is not None else torch.ones(B, 3, device=device, dtype=torch.bool), modality_valid=torch.ones(B, 5, device=device, dtype=torch.bool), image_is_pad=zeros(9), flow_history_valid=torch.ones(B, device=device, dtype=torch.bool), flow_transition_valid=torch.ones(B, 8, device=device, dtype=torch.bool), time_grid=time_grid or {k: torch.zeros(B, n, 8, device=device) for k, n in shapes.items()}, time_valid=time_valid or {k: zeros(n) for k, n in shapes.items()})
        if pointmap_pixel_valid is None and kwargs.get('per_cam_depth') is not None:
            from .helpers.observed_pointmap_mask import observed_pointmap_mask
            pointmap_pixel_valid = observed_pointmap_mask(kwargs['per_cam_depth'], max_depth_m=self.pointmap_encoder.max_depth_m, depth_validity=getattr(self.pointmap_encoder, 'depth_validity', 'legacy_clip'))
        if pointmap_pixel_valid is not None:
            if pointmap_pixel_valid.shape != (B, 9, 384, 320):
                raise ValueError('pointmap_pixel_valid must have shape [B,9,384,320]')
            sample['pointmap_pixel_valid'] = pointmap_pixel_valid
        self._set_multisource_context(sample)
        try:
            return super().infer_action(*args, **kwargs)
        finally:
            self._ms_active = False

    def init_perception(self, config=None):
        from types import SimpleNamespace
        if config and config.get('enabled'):
            raise ValueError('Auxiliary perception is unavailable in this inference model')
        self.perception_config = SimpleNamespace(enabled=False)
        self._perception_inputs = None
        self._perception_step = 0
