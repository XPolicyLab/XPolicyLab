from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from einops import rearrange
from crwm.composite_layouts import get_layout
from crwm.text_conditioning import text_contract, validate_text_contract, checkpoint_contract, check_prompt_lengths
from crwm.utils.logging_config import get_logger
from crwm.utils.pytorch_utils import _staged_randn
from .helpers.dino import compute_dino_freqs, _dino_x0_to_velocity
from crwm.observation_ablation import branch_frame_valid
from .multisource_conditioning import MultiSourceConditioning, label_masks, restrict_attention
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)

class CRWM(nn.Module):

    def __init__(self, video_expert, action_expert, mot, vae, dino_encoder, pointmap_encoding, text_encoder=None, tokenizer=None, device='cuda', torch_dtype=torch.bfloat16, proprio_dim=96, dino_dim=768, dino_pixel_unshuffle=2, dino_pred_x0=True, dino_cam_patches=None, dino_cam_regions=None, dino_temporal_stride=1, dino_pool_mode='avg', composite_layout='tshape_robotwin_384x320_uniform', video_infer_shift=6.0, action_infer_shift=1.0, dino_infer_shift=6.0, pointmap_infer_shift=6.0, timestep_scale=1000):
        super().__init__()
        if proprio_dim != 96 or action_expert.action_dim != 90:
            raise ValueError('CR-WM requires S96/A90 architecture')
        if dino_temporal_stride != 1 or dino_pixel_unshuffle != 2 or (not dino_pred_x0) or (dino_pool_mode != 'avg'):
            raise ValueError('This checkpoint requires stride-1 DINO, 2x pixel unshuffle, x0 output and average pooling')
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self._layout = get_layout(composite_layout)
        self._slot_key_map = self._layout.resolve_slot_key_map()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vae = vae
        self.dino_encoder = dino_encoder
        self.pointmap_encoding = pointmap_encoding
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.proprio_dim = proprio_dim
        self.proprio_encoder = nn.Linear(proprio_dim, video_expert.text_embedding[0].in_features)
        self.dino_cam_regions = [tuple(x) for x in dino_cam_regions] if dino_cam_regions is not None else [tuple((v // 2 for v in x)) for x in self._layout.dino_cam_regions]
        self.dino_cam_patches = [tuple(x) for x in dino_cam_patches] if dino_cam_patches is not None else [(h // 2, w // 2) for h, w in self._layout.dino_cam_patches]
        self._dino_encoder_cam_patches = [(h * 2, w * 2) for h, w in self.dino_cam_patches]
        if any((x != (dino_encoder.patches_per_side, dino_encoder.patches_per_side) for x in self._dino_encoder_cam_patches)):
            raise ValueError('DINO pixel-unshuffle must operate on the native ViT grid')
        self.dino_dim = dino_dim * 4
        self.dino_feature_norm = nn.LayerNorm(self.dino_dim)
        self.dino_embedder = nn.Linear(self.dino_dim, video_expert.hidden_dim)
        self.dino_proj_out = nn.Linear(video_expert.hidden_dim, self.dino_dim)
        self.pt_patch_embedding = copy.deepcopy(video_expert.patch_embedding)
        self.pt_head = copy.deepcopy(video_expert.head)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(timestep_scale, video_infer_shift)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(timestep_scale, action_infer_shift)
        self.infer_dino_scheduler = WanContinuousFlowMatchScheduler(timestep_scale, dino_infer_shift)
        self.infer_pointmap_scheduler = WanContinuousFlowMatchScheduler(timestep_scale, pointmap_infer_shift)
        self._ms_active = False
        self._glue_cache = {}
        self.to(device=self.device, dtype=self.torch_dtype)
        self.eval().requires_grad_(False)

    @classmethod
    def from_wan22_pretrained(cls, video_dit_config, action_dit_config, device='cuda', torch_dtype=torch.bfloat16, tokenizer_max_len=128, proprio_dim=96, dino_dim=768, dino_model_name='vit_base_patch16_dinov3.lvd1689m', dino_pixel_unshuffle=2, dino_pred_x0=True, dino_cam_patches=None, dino_cam_regions=None, dino_temporal_stride=1, dino_pool_mode='avg', composite_layout='tshape_robotwin_384x320_uniform', pointmap_encoding=None, video_infer_shift=6.0, action_infer_shift=1.0, dino_infer_shift=6.0, pointmap_infer_shift=6.0, timestep_scale=1000, hbridge_enabled=True, hbridge_bottom_ratio=0.25, hbridge_top_ratio=0.25):
        from .action_dit import ActionDiT
        from .dino_encoder import DinoEncoder
        from .mot import MoT
        from .helpers.loader import load_wan22_ti2v_5b_components
        components = load_wan22_ti2v_5b_components(device=device, torch_dtype=torch_dtype, tokenizer_max_len=tokenizer_max_len, dit_config=video_dit_config)
        action = ActionDiT(**action_dit_config)
        mot = MoT({'video': components.dit, 'action': action}, hbridge_enabled=hbridge_enabled, hbridge_bottom_ratio=hbridge_bottom_ratio, hbridge_top_ratio=hbridge_top_ratio)
        dino = DinoEncoder(model_name=dino_model_name, device=device, torch_dtype=torch_dtype)
        model = cls(video_expert=components.dit, action_expert=action, mot=mot, vae=components.vae, dino_encoder=dino, pointmap_encoding=pointmap_encoding, text_encoder=components.text_encoder, tokenizer=components.tokenizer, device=device, torch_dtype=torch_dtype, proprio_dim=proprio_dim, dino_dim=dino_dim, dino_pixel_unshuffle=dino_pixel_unshuffle, dino_pred_x0=dino_pred_x0, dino_cam_patches=dino_cam_patches, dino_cam_regions=dino_cam_regions, dino_temporal_stride=dino_temporal_stride, dino_pool_mode=dino_pool_mode, composite_layout=composite_layout, video_infer_shift=video_infer_shift, action_infer_shift=action_infer_shift, dino_infer_shift=dino_infer_shift, pointmap_infer_shift=pointmap_infer_shift, timestep_scale=timestep_scale)
        model.text_contract = text_contract(tokenizer_max_len)
        return model

    def init_multisource(self, condition_count=256, text_config=None):
        self.text_contract = validate_text_contract(text_config or getattr(self, 'text_contract', text_contract()))
        dims = {k: self.video_expert.hidden_dim for k in ('video', 'dino', 'pointmap')}
        dims['action'] = self.action_expert.action_encoder.out_features
        self.multisource_conditioning = MultiSourceConditioning(self.proprio_encoder.out_features, dims, condition_count).to(device=self.device, dtype=self.torch_dtype)
        self.multisource_contract = None
        self._ms_sample = {}
        self._ms_token_valid = {}
        self._ms_handles = []
        for branch, module in (('video', self.video_expert.patch_embedding), ('pointmap', self.pt_patch_embedding), ('dino', self.dino_embedder), ('action', self.action_expert.action_encoder)):
            self._ms_handles.append(module.register_forward_hook(self._embedding_hook(branch)))
        self._ms_handles.append(self.action_expert.action_encoder.register_forward_pre_hook(self._action_input_hook))
        self.eval().requires_grad_(False)

    def prepare_for_inference(self):
        self.eval().requires_grad_(False)
        self.video_expert.freqs = tuple((x.to(self.device) for x in self.video_expert.freqs))
        self.action_expert.freqs = self.action_expert.freqs.to(self.device)
        self._glue_cache = {}

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        saved_contract = checkpoint_contract(payload.get('multisource_contract'))
        if saved_contract is None:
            raise ValueError('Checkpoint is missing the S96/A90 conditioning contract')
        if saved_contract['text'] != self.text_contract:
            raise ValueError('Checkpoint text contract differs from the runtime')
        if self.multisource_contract is None:
            self.bind_multisource_contract(saved_contract)
        if saved_contract != self.multisource_contract:
            raise ValueError('Checkpoint conditioning and normalization contract differs')
        metadata = self.pointmap_encoding
        if payload.get('pointmap_encoding') != metadata:
            raise ValueError('Checkpoint pointmap representation differs from the runtime')
        keys = ('mot', 'proprio_encoder', 'dino_feature_norm', 'dino_embedder', 'dino_proj_out', 'pt_patch_embedding', 'pt_head', 'multisource_conditioning')
        for key in keys:
            if key not in payload:
                raise ValueError(f'Checkpoint is missing required weights: {key}')
            state = payload[key]
            if key == 'multisource_conditioning':
                state = {k: v for k, v in state.items() if k != 'time_projection.flow.weight'}
            getattr(self, key).load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)
        logger.info('Loaded CR-WM joint inference weights from %s', path)
        return payload

    def _layout_kwargs(self):
        return {'layout': self._layout, 'slot_key_map': self._slot_key_map}

    def _dino_encode_kwargs(self):
        return {**self._layout_kwargs(), 'cam_patches': self._dino_encoder_cam_patches, 'pool_mode': 'avg', 'pixel_unshuffle': 2}

    @torch.no_grad()
    def encode_prompt(self, prompt):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError('Provide cached context/context_mask or load the text encoder')
        check_prompt_lengths(self.tokenizer, prompt, self.text_contract)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        target = next(self.text_encoder.parameters()).device
        mask = mask.to(device=target, dtype=torch.bool)
        context = self.text_encoder(ids.to(target), mask)
        context = torch.where(mask[..., None], context, 0)
        return (context.to(device=self.device, dtype=self.torch_dtype), mask.to(self.device))

    def _build_mot_attention_mask_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        mask = self._base_build_mot_attention_mask_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        mask[Sv + Sd + Sp:, :Sv + Sd + Sp] = True
        return self._restricted_layout(mask, [('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)])

    def _build_hbridge_self_masks_unified(self, video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len, video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame, device):
        Sv, Sa, Sd, Sp = (video_seq_len, action_seq_len, dino_seq_len, pointmap_seq_len)
        tpf, dtpf, ptpf = (video_tokens_per_frame, dino_tokens_per_frame, pointmap_tokens_per_frame)
        lens, masks = self._base_build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp, tpf, dtpf, ptpf, device)
        if lens is None:
            return (lens, masks)
        layout = [(k, n, p) for k, n, p in (('video', Sv, tpf), ('dino', Sd, dtpf), ('pointmap', Sp, ptpf), ('action', Sa, 1)) if n > 0]
        return (lens, [self._restricted_layout(m, [part]) for m, part in zip(masks, layout)])

    @torch.no_grad()
    def infer_action(self, *, prompt=None, input_image, action_horizon=32, num_video_frames=9, proprio, context=None, context_mask=None, num_inference_steps=4, sigma_shift=None, seed=None, rand_device='cpu', per_cam=None, state_valid, action_valid, condition_id, time_grid=None, time_valid=None, camera_valid=None):
        if action_horizon != 32 or num_video_frames != 9:
            raise ValueError('CR-WM expects 32 actions and 9 video frames')
        if input_image.ndim == 3:
            input_image = input_image[None]
        if tuple(input_image.shape) != (1, 3, 384, 320):
            raise ValueError('input_image must be [1,3,384,320]')
        if state_valid.ndim != 2 or tuple(state_valid.shape) != (1, 96):
            raise ValueError('state_valid must be [1,96]')
        if tuple(action_valid.shape) != (1, 32, 90):
            raise ValueError('action_valid must be [1,32,90]')
        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError('Provide prompt or cached context/context_mask')
        if prompt is None and (context is None or context_mask is None):
            raise ValueError('Provide prompt or cached context/context_mask')
        shapes = {'action': 32, 'video': 3, 'pointmap': 3, 'dino': 3}
        sample = dict(state_valid=state_valid, action_valid=action_valid, condition_id=condition_id, camera_valid=camera_valid if camera_valid is not None else torch.ones(1, 3, device=self.device, dtype=torch.bool), modality_valid=torch.ones(1, 5, device=self.device, dtype=torch.bool), image_is_pad=torch.zeros(1, 9, device=self.device, dtype=torch.bool), time_grid=time_grid or {k: torch.zeros(1, n, 8, device=self.device) for k, n in shapes.items()}, time_valid=time_valid or {k: torch.zeros(1, n, device=self.device, dtype=torch.bool) for k, n in shapes.items()}, branch_frame_valid=branch_frame_valid(self.device))
        self._set_multisource_context(sample)
        try:
            return self._infer_joint(input_image, proprio, prompt, context, context_mask, per_cam, num_inference_steps, sigma_shift, seed, rand_device)
        finally:
            self._ms_active = False

    @torch.no_grad()
    def _infer_joint(self, input_image, proprio, prompt, context, context_mask, per_cam, num_inference_steps, sigma_shift, seed, rand_device):
        image = input_image.to(device=self.device, dtype=self.torch_dtype)
        generator = lambda: None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        video = _staged_randn((1, self.vae.model.z_dim, 3, 24, 20), generator(), rand_device, self.device, self.torch_dtype)
        action = _staged_randn((1, 32, self.action_expert.action_dim), generator(), rand_device, self.device, self.torch_dtype)
        dino = _staged_randn((1, self.dino_dim, 3, sum((h * w for h, w in self.dino_cam_patches)), 1), generator(), rand_device, self.device, self.torch_dtype)
        video_anchor = self._encode_input_image_latents_tensor(image)
        video[:, :, :1] = video_anchor.clone()
        dino_anchor = self._current_dino_anchor(image, per_cam, dino)
        dino[:, :, :1] = dino_anchor
        pointmap_anchor = self._current_pointmap_anchor(video_anchor)
        _, C, _, H, W = pointmap_anchor.shape
        pointmap = _staged_randn((1, C, 3, H, W), generator(), rand_device, self.device, self.torch_dtype)
        pointmap[:, :, :1] = pointmap_anchor
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context[None]
            if context_mask.ndim == 1:
                context_mask = context_mask[None]
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio.ndim == 1:
            proprio = proprio[None]
        proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        schedules = [s.build_inference_schedule(num_inference_steps, self.device, self.torch_dtype, sigma_shift) for s in (self.infer_video_scheduler, self.infer_action_scheduler, self.infer_dino_scheduler, self.infer_pointmap_scheduler)]
        dino_freqs = self._compute_dino_freqs(3, self.device)
        for i in range(num_inference_steps):
            timesteps = [t[i].reshape(1) for t, d in schedules]
            pred_video, pred_dino, pred_pointmap, pred_action = self._predict_joint_noise_unified_impl(latents_video=video, latents_dino=dino, latents_pointmap=pointmap, latents_action=action, timestep_video=timesteps[0], timestep_action=timesteps[1], timestep_dino=timesteps[2], timestep_pointmap=timesteps[3], context=context, context_mask=context_mask, dino_freqs=dino_freqs)
            video = self.infer_video_scheduler.step(pred_video, schedules[0][1][i], video)
            action = self.infer_action_scheduler.step(pred_action, schedules[1][1][i], action)
            dino = self.infer_dino_scheduler.step(pred_dino, schedules[2][1][i], dino)
            pointmap = self.infer_pointmap_scheduler.step(pred_pointmap, schedules[3][1][i], pointmap)
            video[:, :, :1] = video_anchor.clone()
            dino[:, :, :1] = dino_anchor.clone()
            pointmap[:, :, :1] = pointmap_anchor.clone()
        return {'action': action[0].detach().to(device='cpu', dtype=torch.float32)}

    def _current_pointmap_anchor(self, video_anchor):
        # Shared VAE representation: derive the one-frame pointmap shape from
        # RGB's encoded anchor, without running pointmap geometry or VAE.
        return torch.zeros_like(video_anchor)

    def _current_dino_anchor(self, image, per_cam, dino):
        if per_cam is not None:
            cams = {k: v.to(device=self.device, dtype=self.torch_dtype) for k, v in per_cam.items()}
            return self.dino_encoder.encode_video(per_cam=cams, **self._dino_encode_kwargs())
        return self.dino_encoder.encode_video(image[:, :, None], **self._dino_encode_kwargs())

    def bind_multisource_contract(self, contract):
        from crwm.text_conditioning import checkpoint_contract
        contract = checkpoint_contract(contract)
        if contract['text'] != self.text_contract:
            raise ValueError('Dataset/checkpoint and model text conditioning contracts differ')
        count = len(contract['conditions'])
        if count > self.multisource_conditioning.conditions.num_embeddings:
            raise ValueError('Condition vocabulary exceeds model capacity')
        self.multisource_contract = contract

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

    def _set_multisource_context(self, sample):

        def copy_tree(old, new):
            if isinstance(new, dict):
                return {k: copy_tree(old.get(k) if isinstance(old, dict) else None, v) for k, v in new.items()}
            if torch.is_tensor(new):
                value = new.to(self.device)
                if torch.is_tensor(old) and old.shape == value.shape and (old.dtype == value.dtype):
                    old.copy_(value)
                    return old
                return value
            return new
        names = ('state_valid', 'action_valid', 'condition_id', 'camera_valid', 'modality_valid', 'image_is_pad', 'time_grid', 'time_valid', 'branch_frame_valid')
        self._ms_sample = copy_tree(self._ms_sample, {k: sample[k] for k in names if k in sample})
        self._ms_masks = copy_tree(getattr(self, '_ms_masks', {}), label_masks(self._ms_sample, self.device, tuple((h * w for h, w in self.dino_cam_patches))))
        self._ms_token_valid = {}
        self._ms_active = True

    def _append_proprio_to_context(self, context, context_mask, proprio):
        valid = self._ms_sample['state_valid'].bool()
        if valid.ndim == 3:
            valid = valid[:, 0]
        p = torch.where(valid, proprio, 0).to(context.dtype)
        token = self.proprio_encoder(p) + self.multisource_conditioning.state_valid(valid.to(context.dtype))
        mask = valid.any(-1, keepdim=True)
        token = token[:, None] * mask[..., None]
        condition = self.multisource_conditioning.conditions(self._ms_sample['condition_id'].long())[:, None]
        return (torch.cat([context, token, condition.to(context.dtype)], 1), torch.cat([context_mask, mask, torch.ones_like(mask)], 1))

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor):
        z = self.vae.encode(video_tensor, device=self.device)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f'`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}')
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _embed_dino(self, dino_features: torch.Tensor) -> torch.Tensor:
        x = dino_features.squeeze(-1).permute(0, 2, 3, 1).reshape(dino_features.shape[0], -1, self.dino_dim)
        x = self.dino_feature_norm(x)
        return self.dino_embedder(x)

    def _project_dino_out(self, dino_tokens_out: torch.Tensor) -> torch.Tensor:
        return self.dino_proj_out(dino_tokens_out)

    def _glue_memo(self, key, build):
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

    def _merge_aux_into_video_stream(self, video_pre: dict, dino_tokens: torch.Tensor, dino_t_mod: torch.Tensor, dino_freqs: torch.Tensor, pointmap_tokens: torch.Tensor, pointmap_t_mod: torch.Tensor, pointmap_freqs: torch.Tensor):
        merged_tokens = torch.cat([video_pre['tokens'], dino_tokens, pointmap_tokens], dim=1)
        merged_freqs = torch.cat([video_pre['freqs'], dino_freqs, pointmap_freqs], dim=0)
        merged_t_mod = torch.cat([video_pre['t_mod'], dino_t_mod, pointmap_t_mod], dim=1)
        video_ctx_mask = video_pre['context_mask']
        aux_extra = dino_tokens.shape[1] + pointmap_tokens.shape[1]
        aux_mask = video_ctx_mask[:, :1, :].expand(-1, aux_extra, -1)
        return (merged_tokens, merged_freqs, merged_t_mod, torch.cat([video_ctx_mask, aux_mask], dim=1))

    def _build_stream_t_mod(self, timestep_per_token: torch.Tensor, seq_len: int, B: int) -> torch.Tensor:
        t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, timestep_per_token.reshape(-1))
        t = self.video_expert.time_embedding(t_emb).reshape(B, seq_len, self.video_expert.hidden_dim)
        return self.video_expert.time_projection(t).unflatten(2, (6, self.video_expert.hidden_dim))

    @torch.no_grad()
    def _base_build_hbridge_self_masks_unified(self, video_seq_len: int, action_seq_len: int, dino_seq_len: int, pointmap_seq_len: int, video_tokens_per_frame: int, dino_tokens_per_frame: int, pointmap_tokens_per_frame: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
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

    def _joint_noise_core(self, latents_video: torch.Tensor, latents_dino: torch.Tensor, latents_pointmap: Optional[torch.Tensor], latents_action: torch.Tensor, timestep_video: torch.Tensor, timestep_dino: torch.Tensor, timestep_pointmap: Optional[torch.Tensor], timestep_action: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, dino_freqs: torch.Tensor | None=None, pt_freqs: torch.Tensor | None=None):
        B = latents_video.shape[0]
        video_pre = self.video_expert.pre_dit(x=latents_video, timestep=timestep_video, context=context, context_mask=context_mask)
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
        attention_mask = self._build_mot_attention_mask_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks_unified(Sv, Sa, Sd, Sp_for_mask, tpf, effective_dtpf, ptpf_for_mask, merged_tokens.device)
        merged_out, action_out = self.mot._forward_joint_inner(video_tokens=merged_tokens, action_tokens=action_pre['tokens'], video_freqs=merged_freqs, action_freqs=action_pre['freqs'], video_t_mod=merged_t_mod, action_t_mod=action_pre['t_mod'], video_context_payload={'context': video_pre['context'], 'mask': merged_ctx_mask}, action_context_payload={'context': action_pre['context'], 'mask': action_pre['context_mask']}, attention_mask=attention_mask, sub_stream_lens=sub_stream_lens, sub_stream_self_masks=sub_stream_self_masks)
        video_tokens_out = merged_out[:, :Sv, :]
        dino_tokens_out = merged_out[:, Sv:Sv + Sd, :]
        pt_tokens_out = merged_out[:, Sv + Sd:, :]
        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(action_out, action_pre)
        pred_dino = self._project_dino_out(dino_tokens_out)
        pred_dino = pred_dino.reshape(B, num_dino_frames, dtpf, self.dino_dim)
        pred_dino = pred_dino.permute(0, 3, 1, 2).unsqueeze(-1)
        pred_dino = _dino_x0_to_velocity(latents_dino, pred_dino, timestep_dino.view(-1, 1, 1, 1, 1), self.infer_dino_scheduler.timestep_scale).to(pred_dino.dtype)
        pred_pointmap = self._project_pointmap_out(pt_tokens_out, pt_t, pt_meta)
        return (pred_video, pred_dino, pred_pointmap, pred_action)

    def _predict_joint_noise_unified_impl(self, **kwargs):
        return self._joint_noise_core(**kwargs)
