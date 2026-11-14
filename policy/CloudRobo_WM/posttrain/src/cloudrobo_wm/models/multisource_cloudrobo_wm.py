from __future__ import annotations
from cloudrobo_wm.platform.resize import interpolate as platform_interpolate
from cloudrobo_wm.contracts import data_protocol
import logging
import torch
import torch.nn.functional as F
from .flow_cloudrobo_wm import FlowCloudRoboWM
from .backbone import CloudRoboBackbone
from .multisource_conditioning import MultiSourceConditioning, label_masks, restrict_attention, masked_modality_loss
from .multisource_loss import masked_action_loss, reduce_loss_statistics, action_group_loss_statistics, ACTION_LOSS_GROUPS
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

    def init_perception(self, config=None):
        from .perception import PerceptionConfig, PerceptionHeads
        self.perception_config = PerceptionConfig(**dict(config or {}))
        self._perception_inputs = None
        self._perception_step = 0
        if not self.perception_config.enabled:
            return
        if self._pointmap_globally_off or 'dino_encoder' not in self.FROZEN_MODULES:
            raise ValueError('Perception requires pointmap enabled and a frozen DINO teacher')
        if not getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False):
            raise ValueError('Perception requires a clean observed first frame')
        with torch.random.fork_rng(devices=[]):
            self.perception = PerceptionHeads(self.video_expert, self._layout, self.dino_encoder.patches_per_side, self.dino_encoder.embed_dim).to(device=self.device, dtype=self.torch_dtype)

    def _capture_perception_inputs(self, inputs, sample):
        first = inputs['first_frame_latents']
        if first is None or inputs['pointmap_raw'] is None:
            raise ValueError('Perception requires RGB and pointmap first-frame latents')
        cam = sample['camera_valid'].to(self.device).bool()
        current = (~sample['image_is_pad'][:, 0]).to(self.device).bool()
        modality = sample['modality_valid'].to(self.device).bool()
        dino_valid = cam & (current & modality[:, 0] & modality[:, 2])[:, None]
        dino_valid = dino_valid[:, :, None, None, None]
        if 'per_cam_rgb_valid' in self._ms_sample:
            from cloudrobo_wm.datasets.multisource.schema import CAMERAS
            side = self.perception.native_side
            masks = []
            for name in CAMERAS:
                raw = sample['per_cam_rgb_valid'][name].to(self.device).float()[:, None]
                teacher = platform_interpolate(raw, size=(224, 224), mode='bilinear', align_corners=False, antialias=True) >= 1 - 1e-06
                patches = F.adaptive_avg_pool2d(teacher.float(), (side, side)) >= 1 - 1e-07
                masks.append(patches[:, 0])
            dino_valid = dino_valid & torch.stack(masks, dim=1)[..., None]
        self._perception_inputs = dict(latents=first.detach(), context=sample['context'].to(device=self.device, dtype=self.torch_dtype).detach(), context_mask=sample['context_mask'].to(device=self.device, dtype=torch.bool), rgb_valid=self._ms_masks['video'][:, 0], pointmap_target=inputs['pointmap_raw'][:, :, :1].detach(), pointmap_valid=self._ms_masks['pointmap'][:, None, :1], dino_target=inputs['perception_dino_target'].detach(), dino_valid=dino_valid)

    def _perception_predictions(self, inputs):
        active = self._ms_active
        self._ms_active = False
        try:
            return self.perception(self.video_expert, inputs['latents'], inputs['context'], inputs['context_mask'], inputs['rgb_valid'], checkpoint_layers=getattr(self, 'perception_checkpoint_layers', True) and (self.video_expert.use_gradient_checkpointing or self.mot.checkpoint_full_layers), checkpoint_every=getattr(self, 'perception_checkpoint_every', 1))
        finally:
            self._ms_active = active

    def _perception_loss(self):
        inputs = self._perception_inputs
        if inputs is None:
            raise RuntimeError('Missing perception targets from build_inputs')
        predictions = self._perception_predictions(inputs)
        weight = torch.ones(inputs['latents'].shape[0], device=self.device)
        parts = [masked_modality_loss(predictions[name], inputs[name + '_target'], inputs[name + '_valid'], weight, return_stats=True) for name in ('pointmap', 'dino')]
        losses, counts = reduce_loss_statistics(parts, torch.stack([count for _, count in parts]))
        ramp = self.perception_config.weight(self._perception_step)
        scales = (self.perception_config.lambda_pointmap, self.perception_config.lambda_dino)
        total = sum((ramp * scale * loss for scale, loss in zip(scales, losses)))
        metrics = {'perception_warmup': total.detach().new_tensor(ramp)}
        for name, loss, count, scale in zip(('pointmap', 'dino'), losses, counts, scales):
            metrics['perception_' + name + '_mse'] = loss.detach()
            metrics['loss_perception_' + name] = (ramp * scale * loss).detach()
            metrics['perception_' + name + '_valid_samples'] = count.detach()
        return (total, metrics)

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

    def _flow_metadata(self):
        metadata = {**super()._flow_metadata(), 'multisource_contract': self.multisource_contract, 'multisource_schema': data_protocol('protocol_76834ffeb390')}
        if hasattr(self, 'perception'):
            metadata['perception_contract'] = self.perception.contract(self.perception_config)
        return metadata

    def _validate_checkpoint_payload(self, payload, *, strict_shape):
        if getattr(self, 'action_topology', 'single') == 'single' and payload.get('action_topology', 'single') != 'single':
            raise ValueError('A dual checkpoint cannot be loaded into the single-action topology')
        if strict_shape:
            expected = self._flow_metadata().get('perception_contract')
            if payload.get('perception_contract') != expected:
                raise ValueError('Strict resume rejected: perception contract changed; use init_from for a warm start')
            if expected is not None:
                saved = payload.get('perception', {})
                current = self.perception.state_dict()
                if set(saved) != set(current) or any((saved[k].shape != v.shape for k, v in current.items())):
                    raise ValueError('Strict resume requires all perception parameters with matching shapes')
        from cloudrobo_wm.text_conditioning import checkpoint_contract
        saved = checkpoint_contract(payload.get('multisource_contract'))
        if saved is not None and saved['text'] != self.text_contract:
            raise ValueError('Checkpoint text contract differs from runtime; do not silently change padding/tokenizer/length')
        if strict_shape and saved != self.multisource_contract:
            raise ValueError('Strict resume rejected: multisource schema/statistics/conditions/data changed')
        checked = {**payload, 'multisource_contract': saved} if saved is not None else payload
        super()._validate_checkpoint_payload(checked, strict_shape=strict_shape)
        self._ms_rebuild_after_load = {}
        if not strict_shape and getattr(self, 'action_topology', 'single') == 'single':
            for key, module, weights in (('mot.mixtures.action.action_encoder', self.action_expert.action_encoder, payload.get('mot', {}).get('mixtures.action.action_encoder.weight')), ('mot.mixtures.action.head', self.action_expert.head, payload.get('mot', {}).get('mixtures.action.head.weight')), ('proprio_encoder', self.proprio_encoder, payload.get('proprio_encoder', {}).get('weight'))):
                if weights is not None and weights.shape != module.weight.shape:
                    self._ms_rebuild_after_load[key] = (module, {k: v.detach().clone() for k, v in module.state_dict().items()})

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

    def _build_mot_attention_mask_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        mask = super()._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        if not self._ms_active:
            return mask
        return self._restricted_layout(mask, [('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)])

    def _build_flow_attention_mask(self, Sv, Sa, Sd, Sp, Sf, tpf, dtpf, ptpf, device):
        mask = super()._build_flow_attention_mask(Sv, Sa, Sd, Sp, Sf, tpf, dtpf, ptpf, device)
        if not self._ms_active:
            return mask
        return self._restricted_layout(mask, [('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('flow', Sf, Sf // 2), ('action', Sa, 1)])

    def _build_hbridge_self_masks_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        lens, masks = super()._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        if not self._ms_active or lens is None:
            return (lens, masks)
        layout = [(k, n, p) for k, n, p in (('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)) if n > 0]
        return (lens, [self._restricted_layout(m, [part]) for m, part in zip(masks, layout)])

    def build_inputs(self, sample, tiled=False):
        self._set_multisource_context(sample)
        sample = dict(sample)
        sample['action'] = torch.where(sample['action_valid'], sample['action'], 0)
        sample['proprio'] = torch.where(sample['state_valid'][:, None], sample['proprio'], 0)
        inputs = CloudRoboBackbone.build_inputs(self, sample, tiled=tiled)
        per_cam = {k: v.to(device=self.device, dtype=self.torch_dtype) for k, v in sample['per_cam'].items()}
        B = sample['action'].shape[0]
        with torch.no_grad():
            selected = sample['modality_valid'][:, 2].bool().to(self.device)
            dino = torch.zeros(B, self.dino_dim, 3, sum((h * w for h, w in self.dino_cam_patches)), 1, device=self.device, dtype=self.torch_dtype)
            perception_enabled = getattr(self, 'perception_config', None) is not None and self.perception_config.enabled
            if perception_enabled:
                side = self.perception.native_side
                native = torch.zeros(B, len(self._layout.cam_slots()), side, side, self.perception.native_dim, device=self.device, dtype=torch.float32)
            if selected.any():
                extra = {'return_native_first_frame': True} if perception_enabled else {}
                encoded = self.dino_encoder.encode_video(per_cam={k: v[selected] for k, v in per_cam.items()}, concat_mode='robotwin', temporal_stride=1, inference_batch_size=getattr(self, 'dino_inference_batch_size', 27), **self._dino_encode_kwargs(), **extra)
                if perception_enabled:
                    encoded, first_native = encoded
                    native[selected] = first_native.to(native)
                dino[selected] = encoded.to(dino)
            inputs['dino_features'] = dino
            if perception_enabled:
                inputs['perception_dino_target'] = native.detach()
            if self._pointmap_globally_off:
                inputs['pointmap_raw'] = None
            else:
                selected = sample['modality_valid'][:, 3].bool().to(self.device)
                point = torch.zeros_like(inputs['input_latents'])
                half_pixel = sample.get('pointmap_half_pixel', torch.zeros(B, dtype=torch.bool)).to(self.device).bool()
                for use_half_pixel in (False, True):
                    group = selected & (half_pixel == use_half_pixel)
                    if not group.any():
                        continue
                    K = sample['camera_intrinsics'].to(self.device)[group]
                    depths = {k: v.to(self.device)[group] for k, v in sample['per_cam_depth'].items()}
                    composite = self.pointmap_encoder.encode_composite(per_cam_depth=depths, camera_intrinsics=K, concat_mode='robotwin', half_pixel=use_half_pixel, **self._layout_kwargs())
                    point[group] = self._encode_video_latents(composite, tiled=tiled).to(point)
                inputs['pointmap_raw'] = point
            selected = sample['modality_valid'][:, 4].bool().to(self.device)
            flow = torch.zeros_like(inputs['input_latents'][:, :, 1:])
            if selected.any():
                flow[selected] = self._encode_flow_target(sample['flow_video'].to(self.device)[selected], tiled=tiled).to(flow)
            inputs['flow_target'] = flow
        inputs['flow_history_valid'] = sample['flow_history_valid'].to(self.device)
        inputs['flow_transition_valid'] = sample['flow_transition_valid'].to(self.device)
        inputs['action_valid'] = self._ms_masks['action']
        if perception_enabled:
            self._capture_perception_inputs(inputs, sample)
        return inputs

    def training_loss(self, sample, tiled=False):
        if self.skip_unlabeled_flow_tokens and sample['modality_valid'][:, 4].any():
            raise ValueError('skip_unlabeled_flow_tokens requires a batch with no flow labels')
        try:
            total, metrics = super().training_loss(sample, tiled=tiled)
            if self.perception_config.enabled:
                auxiliary, auxiliary_metrics = self._perception_loss()
                metrics = dict(metrics, loss_main=total.detach(), **auxiliary_metrics)
                total = total + auxiliary
            return (total, metrics)
        finally:
            self._ms_active = False
            self._perception_inputs = None

    def _training_mot(self, **kwargs):
        if not self.skip_unlabeled_flow_tokens:
            return super()._training_mot(**kwargs)
        flow_size = self._ms_token_valid['flow'].shape[1]
        visual = kwargs['embeds_all']['video']
        visual_size = visual.shape[1] - flow_size
        action_size = kwargs['embeds_all']['action'].shape[1]
        keep = torch.cat((torch.arange(visual_size, device=visual.device), torch.arange(visual.shape[1], visual.shape[1] + action_size, device=visual.device)))
        kwargs['embeds_all'] = dict(kwargs['embeds_all'], video=visual[:, :visual_size])
        kwargs['freqs_all'] = dict(kwargs['freqs_all'], video=kwargs['freqs_all']['video'][:visual_size])
        kwargs['t_mod_all'] = dict(kwargs['t_mod_all'], video=kwargs['t_mod_all']['video'][:, :visual_size])
        context = dict(kwargs['context_all']['video'])
        if context['mask'] is not None and context['mask'].ndim == 3:
            context['mask'] = context['mask'][:, :visual_size]
        kwargs['context_all'] = dict(kwargs['context_all'], video=context)
        mask = kwargs['attention_mask']
        kwargs['attention_mask'] = mask.index_select(-2, keep).index_select(-1, keep)
        if kwargs['sub_stream_lens'] is not None:
            lens, masks = (list(kwargs['sub_stream_lens']), list(kwargs['sub_stream_self_masks']))
            flow_index = len(lens) - 2
            if lens[flow_index] != flow_size or sum(lens[:flow_index]) != visual_size:
                raise ValueError('Unexpected HBridge flow range')
            del lens[flow_index], masks[flow_index]
            kwargs['sub_stream_lens'], kwargs['sub_stream_self_masks'] = (lens, masks)
        output = super()._training_mot(**kwargs)
        output['video'] = torch.cat((output['video'], visual[:, visual_size:] * 0), dim=1)
        return output

    def _multisource_losses(self, values):
        v = values

        def regime(branch):
            bf = self._batch_modalities
            if bf is None:
                return None
            code = {'video': 'v', 'dino': 'd', 'pointmap': 'p'}.get(branch)
            return None if code is None or getattr(bf, 'cm_' + code) else getattr(bf, 'present_' + code)

        def loss(branch, pred, target, time, mask):
            scheduler = getattr(self, 'train_' + branch + '_scheduler')
            return masked_modality_loss(pred, target, mask, scheduler.training_weight(time), regime(branch), return_stats=True)
        offset = 0 if v['include_init'] else 1
        lv = loss('video', v['pred_video'], v['target_video'], v['t_video'], self._ms_masks['video'][:, None, offset:])
        action_weight = self.train_action_scheduler.training_weight(v['t_action'])
        la = masked_action_loss(v['pred_action'], v['target_action'], self._ms_masks['action'], action_weight, self.multisource_field_weights, self.multisource_horizon_weights, return_stats=True)
        dtpf = v['dtpf']
        ld = loss('dino', v['pred_dino'][:, dtpf:], v['target_dino_flat'][:, dtpf:], v['t_dino'], self._ms_masks['dino'][:, 1:].flatten(1)[..., None])
        lp = (torch.zeros((), device=self.device), torch.zeros((), device=self.device)) if v['target_pointmap'] is None else loss('pointmap', v['pred_pointmap'][:, :, 1:], v['target_pointmap'][:, :, 1:], v['t_pointmap'], self._ms_masks['pointmap'][:, None, 1:])
        lf = loss('flow', v['pred_flow'], v['target_flow'], v['t_video'], self._ms_masks['flow'][:, None])
        from cloudrobo_wm.datasets.multisource.schema import ACTION
        with torch.no_grad():
            logs = []
            names = ('joints', 'gripper', 'position', 'rotation', 'hand', 'base')
            for name in names:
                slices = [ACTION['base']] if name == 'base' else [ACTION[s + '.' + name] for s in ('left', 'right')]
                indices = [i for sl in slices for i in range(sl.start, sl.stop)]
                mask = self._ms_masks['action'][..., indices]
                error = torch.where(mask, (v['pred_action'][..., indices].float() - v['target_action'][..., indices].float()).square(), 0)
                logs.append(torch.stack([error.sum(), mask.sum().float()]))
            counts = self._ms_sample.get('normalization_counts')
            group_stats = action_group_loss_statistics(v['pred_action'], v['target_action'], self._ms_masks['action'], action_weight, self.multisource_field_weights, self.multisource_horizon_weights)
            diagnostics = torch.cat((torch.stack(logs).flatten(), group_stats.flatten(), counts.sum(0).float().flatten() if counts is not None else torch.zeros(4, device=self.device)))
        losses, diagnostics = reduce_loss_statistics((lv, la, ld, lp, lf), diagnostics)
        self._multisource_loss_metrics = {}
        raw_end = 2 * len(names)
        group_end = raw_end + 2 * len(ACTION_LOSS_GROUPS)
        for name, (error, count) in zip(names, diagnostics[:raw_end].reshape(-1, 2)):
            self._multisource_loss_metrics['action_' + name + '_mse'] = error / count.clamp_min(1)
            self._multisource_loss_metrics['action_' + name + '_valid_elements'] = count
        for name, (error, count) in zip(ACTION_LOSS_GROUPS, diagnostics[raw_end:group_end].reshape(-1, 2)):
            self._multisource_loss_metrics['loss_' + name] = self.loss_lambda_action * error / count.clamp_min(1)
            self._multisource_loss_metrics['action_' + name + '_valid_samples'] = count
        if counts is not None:
            for kind, (clipped, count) in zip(('state', 'action'), diagnostics[group_end:].reshape(2, 2)):
                self._multisource_loss_metrics[kind + '_clipped_fraction'] = clipped / count.clamp_min(1)
        return losses
