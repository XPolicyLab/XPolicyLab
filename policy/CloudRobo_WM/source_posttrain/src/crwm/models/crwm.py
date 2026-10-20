from __future__ import annotations
import torch
import torch.nn.functional as F
from .core import CRWMCore
from .backbone import TrainingBackbone
from .wan_video_dit import sinusoidal_embedding_1d
from .multisource_conditioning import MultiSourceConditioning, label_masks, restrict_attention, masked_modality_loss
from .multisource_loss import masked_action_loss, reduce_loss_statistics, action_group_loss_statistics, ACTION_LOSS_GROUPS
from crwm.text_conditioning import text_contract, validate_text_contract, checkpoint_contract
import logging
logger = logging.getLogger(__name__)

class CRWM(CRWMCore):

    def init_multisource(self, condition_count=256, field_weights=None, horizon_weights=None, text_config=None):
        from crwm.text_conditioning import text_contract, validate_text_contract
        self.text_contract = validate_text_contract(text_config or text_contract())
        import math
        for values, size, name in ((field_weights, 90, 'field_weights'), (horizon_weights, 32, 'horizon_weights')):
            if values is not None and (len(values) != size or any((not math.isfinite(x) or x < 0 for x in values))):
                raise ValueError(f'{name} requires {size} finite nonnegative weights')
        if self.proprio_dim != 96 or self.action_expert.action_dim != 90:
            raise ValueError('CRWM requires proprio_dim=96, action_dim=90')
        if self.dino_temporal_stride != 1:
            raise ValueError('Initial multisource timing contract requires dino_temporal_stride=1')
        dims = {k: self.video_expert.hidden_dim for k in ('video', 'dino', 'pointmap')}
        dims['action'] = self.action_expert.action_encoder.out_features
        self.multisource_conditioning = MultiSourceConditioning(self.proprio_encoder.out_features, dims, condition_count).to(device=self.device, dtype=self.torch_dtype)
        self.multisource_contract = None
        self.multisource_field_weights = field_weights
        self.multisource_horizon_weights = horizon_weights
        self._ms_active = False
        self._ms_sample = {}
        self._ms_token_valid = {}
        self._ms_handles = []
        for branch, module in (('video', self.video_expert.patch_embedding), ('pointmap', self.pt_patch_embedding), ('dino', self.dino_embedder), ('action', self.action_expert.action_encoder)):
            self._ms_handles.append(module.register_forward_hook(self._embedding_hook(branch)))
        self._ms_handles.append(self.action_expert.action_encoder.register_forward_pre_hook(self._action_input_hook))

    def bind_multisource_contract(self, contract):
        from crwm.text_conditioning import checkpoint_contract
        contract = checkpoint_contract(contract, warn_legacy=True)
        if contract['text'] != self.text_contract:
            raise ValueError('Dataset/checkpoint and model text conditioning contracts differ')
        count = len(contract['conditions'])
        if count > self.multisource_conditioning.conditions.num_embeddings:
            raise ValueError('Condition vocabulary exceeds model capacity')
        self.multisource_contract = contract

    def encode_prompt(self, prompt):
        if hasattr(self, 'text_contract') and self.tokenizer is not None:
            from crwm.text_conditioning import check_prompt_lengths
            check_prompt_lengths(self.tokenizer, prompt, self.text_contract)
        return super().encode_prompt(prompt, padding_mode='masked')

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
        names = ('state_valid', 'action_valid', 'condition_id', 'camera_valid', 'modality_valid', 'image_is_pad', 'time_grid', 'time_valid', 'pointmap_pixel_valid')
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
        order = ('video', 'dino', 'pointmap', 'action')
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

    def build_inputs(self, sample, tiled=False):
        self._set_multisource_context(sample)
        sample = dict(sample)
        sample['action'] = torch.where(sample['action_valid'], sample['action'], 0)
        sample['proprio'] = torch.where(sample['state_valid'][:, None], sample['proprio'], 0)
        inputs = TrainingBackbone.build_inputs(self, sample, tiled=tiled)
        per_cam = {k: v.to(device=self.device, dtype=self.torch_dtype) for k, v in sample['per_cam'].items()}
        B = sample['action'].shape[0]
        with torch.no_grad():
            selected = sample['modality_valid'][:, 2].bool().to(self.device)
            dino = torch.zeros(B, self.dino_dim, 3, sum((h * w for h, w in self.dino_cam_patches)), 1, device=self.device, dtype=self.torch_dtype)
            if selected.any():
                dino[selected] = self.dino_encoder.encode_video(per_cam={k: v[selected] for k, v in per_cam.items()}, concat_mode='robotwin', temporal_stride=1, **self._dino_encode_kwargs()).to(dino)
            inputs['dino_features'] = dino
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
        inputs['action_valid'] = self._ms_masks['action']
        return inputs

    def training_loss(self, sample, tiled=False):
        try:
            return super().training_loss(sample, tiled=tiled)
        finally:
            self._ms_active = False

    def _multisource_losses(self, values):
        v = values

        def regime(branch):
            bf = self._batch_dropout
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
        lf = (torch.zeros((), device=self.device), torch.zeros((), device=self.device))
        from crwm.data.schema import ACTION
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

    def _base_training_loss(self, sample, tiled: bool=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs['input_latents']
        B = input_latents.shape[0]
        context, context_mask = (inputs['context'], inputs['context_mask'])
        action = inputs['action']
        action_is_pad, image_is_pad = (inputs['action_is_pad'], inputs['image_is_pad'])
        dino_features = inputs['dino_features']
        pointmap_raw = inputs['pointmap_raw']
        # Preserve the reference discarded regime draw; there is no flow network.
        torch.rand((B,), device=self.device)
        noise_video = torch.randn_like(input_latents)
        t_video = self.train_video_scheduler.sample_training_t(B, self.device, input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, t_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, t_video)
        if inputs['first_frame_latents'] is not None:
            latents[:, :, 0:1] = inputs['first_frame_latents']
        # Preserve the reference discarded corruption draw at this exact RNG position.
        torch.randn_like(torch.zeros_like(input_latents[:, :, 1:]))
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
        self._batch_joint_flow = None
        try:
            video_pre = self.video_expert.pre_dit(x=latents, timestep=t_video, context=context, context_mask=context_mask, action=action, fuse_vae_embedding_in_latents=inputs['fuse_vae_embedding_in_latents'])
            action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=t_action, context=context, context_mask=context_mask)
            Sv = video_pre['tokens'].shape[1]
            Sa = action_pre['tokens'].shape[1]
            tpf = int(video_pre['meta']['tokens_per_frame'])
            video_pre['tokens'] = self._dropout_zero_absent_video_tokens(video_pre['tokens'], tpf)
            dino_tokens = self._embed_dino(noisy_dino)
            Sd = dino_tokens.shape[1]
            num_dino_frames = int(dino_features.shape[2])
            dtpf = int(dino_features.shape[3])
            dino_tokens = self._dropout_zero_absent_dino_tokens(dino_tokens, dtpf)
            dino_token_t = t_dino.view(B, 1).expand(B, Sd).clone()
            dino_token_t[:, :dtpf] = 0
            dino_t_mod = self._build_stream_t_mod(dino_token_t, Sd, B)
            dino_freqs = self._compute_dino_freqs(num_dino_frames, video_pre['tokens'].device)
            if self._pointmap_globally_off:
                pointmap_tokens = pt_t_mod = pt_freqs = pt_t = pt_meta = None
                Sp = ptpf = 0
            else:
                pointmap_tokens, ptpf, pt_meta = self._embed_pointmap(noisy_pointmap)
                pointmap_tokens = self._dropout_zero_absent_pointmap_tokens(pointmap_tokens, ptpf)
                Sp = pointmap_tokens.shape[1]
                pt_token_t = t_pointmap.view(B, 1).expand(B, Sp).clone()
                pt_token_t[:, :ptpf] = 0
                pt_t_mod = self._build_stream_t_mod(pt_token_t, Sp, B)
                pt_freqs = self._compute_pointmap_freqs(int(noisy_pointmap.shape[2]), video_pre['tokens'].device, pt_meta=pt_meta)
                pt_t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, pt_token_t.reshape(-1))
                pt_t = self.video_expert.time_embedding(pt_t_emb).reshape(B, Sp, self.video_expert.hidden_dim)
            merged = self._merge_aux_into_video_stream(video_pre, dino_tokens, dino_t_mod, dino_freqs, pointmap_tokens, pt_t_mod, pt_freqs)
            merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = merged
            attention_mask = self._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, merged_tokens.device)
            sub_lens, sub_masks = self._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, merged_tokens.device)
            if getattr(self, '_ms_active', False):
                attention_mask, sub_masks = self._multisource_attention(attention_mask, sub_lens, sub_masks)
            tokens_out = self.mot(embeds_all={'video': merged_tokens, 'action': action_pre['tokens']}, attention_mask=attention_mask, freqs_all={'video': merged_freqs, 'action': action_pre['freqs']}, context_all={'video': {'context': video_pre['context'], 'mask': merged_ctx_mask}, 'action': {'context': action_pre['context'], 'mask': action_pre['context_mask']}}, t_mod_all={'video': merged_t_mod, 'action': action_pre['t_mod']}, sub_stream_lens=sub_lens, sub_stream_self_masks=sub_masks)
            visual_out = tokens_out['video']
            video_tokens_out = visual_out[:, :Sv]
            dino_tokens_out = visual_out[:, Sv:Sv + Sd]
            pt_tokens_out = visual_out[:, Sv + Sd:Sv + Sd + Sp]
            pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
            pred_action = self.action_expert.post_dit(tokens_out['action'], action_pre)
            pred_dino = self._project_dino_out(dino_tokens_out)
            pred_pointmap = None if self._pointmap_globally_off else self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        finally:
            self._batch_joint_flow = None
        include_init = inputs['first_frame_latents'] is None
        if not include_init:
            pred_video, target_video = (pred_video[:, :, 1:], target_video[:, :, 1:])
        lv = self._compute_video_loss_per_sample(pred_video, target_video, image_is_pad, include_init)
        wv = self.train_video_scheduler.training_weight(t_video).to(lv.device, dtype=lv.dtype)
        loss_video = self._dropout_reduce_per_sample_loss(lv, wv, present_mask=self._batch_dropout.present_v if self._batch_dropout is not None else None, cross_modal_active=self._batch_dropout.cm_v if self._batch_dropout is not None else False)
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
        loss_dino = self._dropout_reduce_per_sample_loss(ld, wd, present_mask=self._batch_dropout.present_d if self._batch_dropout is not None else None, cross_modal_active=self._batch_dropout.cm_d if self._batch_dropout is not None else False)
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
            loss_pointmap = self._dropout_reduce_per_sample_loss(lp, wp, present_mask=self._batch_dropout.present_p if self._batch_dropout is not None else None, cross_modal_active=self._batch_dropout.cm_p if self._batch_dropout is not None else False)
        loss_flow = torch.zeros((), device=self.device)
        if getattr(self, '_ms_active', False):
            loss_video, loss_action, loss_dino, loss_pointmap, loss_flow = self._multisource_losses(locals())
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + self.loss_lambda_dino * loss_dino + self.loss_lambda_pointmap * loss_pointmap + 0 * loss_flow
        metric = (lambda value: value.detach()) if getattr(self, '_ms_active', False) else float
        losses = {'loss_video': self.loss_lambda_video * metric(loss_video.detach()), 'loss_action': self.loss_lambda_action * metric(loss_action.detach()), 'loss_dino': self.loss_lambda_dino * metric(loss_dino.detach()), 'loss_pointmap': self.loss_lambda_pointmap * metric(loss_pointmap.detach()), 'loss_flow': metric(loss_flow.detach())}
        if getattr(self, '_ms_active', False):
            losses.update(self._multisource_loss_metrics)
        return (total, losses)
    from crwm.checkpoint import CHECKPOINT_MODULES

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError('Use the trainer full-state save for optimizer state')
        from crwm.checkpoint import save_model_tensors
        # Step numbers belong to the filename and full-state directory.
        save_model_tensors(self, path)

    def load_checkpoint(self, path, optimizer=None, strict_shape=True):
        if optimizer is not None:
            raise ValueError('Initialization must use a fresh optimizer')
        if not strict_shape:
            raise ValueError('Tensor-only weights require complete keys and exact shapes')
        from crwm.checkpoint import load_model_tensors
        return load_model_tensors(self, path)
