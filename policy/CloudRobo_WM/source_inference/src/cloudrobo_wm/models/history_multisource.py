"""Inference boundaries for independently encoded RGB history."""
import torch
import torch.nn.functional as F
from .dual_multisource import DualCooperativeMultiSourceCloudRoboWM
from .history_attention import HistoryConfig, HistoryConditioning

class HistoryMultiSourceCloudRoboWM(DualCooperativeMultiSourceCloudRoboWM):

    def _mode_ckpt_keys(self):
        return super()._mode_ckpt_keys() + ('history_conditioning',)

    @torch.no_grad()
    def _encode_history(self, video, offsets, pixel_valid=None, tiled=False):
        if video is None or offsets is None:
            raise ValueError('History mode requires history_video and history_frame_offsets')
        split = self.history_config.sampling == 'episode_anchor_recent_split_v3'
        frames = 6 if split else 5
        if video.ndim != 5 or video.shape[1:3] != (3, frames) or video.shape[-2:] != (384, 320):
            raise ValueError(f'history_video must be [B,3,{frames},384,320], in [-1,1]')
        B = video.shape[0]
        offsets = offsets.to(device=self.device, dtype=torch.float32)
        if offsets.shape != (B, frames) or not torch.isfinite(offsets).all() or (offsets > 0).any() or (offsets[:, 1:] < offsets[:, :-1]).any():
            raise ValueError('History offsets must be actual ordered nonpositive native steps')
        if split:
            first = offsets[:, :1]
            nominal = offsets.new_tensor([-20, -16, -12, -8, -4])[None]
            expected = torch.cat((first, torch.maximum(nominal, first)), dim=1)
        elif self.history_config.sampling == 'episode_first_multiscale_v2':
            first = offsets[:, :1]
            nominal = offsets.new_tensor([-32, -16, -8, -4])[None]
            expected = torch.cat((first, torch.maximum(nominal, first)), dim=1)
        else:
            nominal = offsets.new_tensor([-16, -12, -8, -4, 0])[None]
            expected = torch.maximum(nominal, offsets[:, :1])
            if (offsets < -16).any() or (offsets[:, -1] != 0).any():
                raise ValueError('Uniform history must end at the current observation')
        if not torch.equal(offsets, offsets.round()) or not torch.equal(offsets, expected):
            raise ValueError('History offsets violate the saved sampling and first-frame padding contract')
        if pixel_valid is None:
            pixel_valid = torch.ones(B, frames, 384, 320, dtype=torch.bool, device=self.device)
        else:
            if pixel_valid.shape != (B, frames, 384, 320):
                raise ValueError(f'history_pixel_valid must be [B,{frames},384,320]')
            pixel_valid = pixel_valid.to(device=self.device, dtype=torch.bool)
        video = torch.where(pixel_valid[:, None], video.to(device=self.device, dtype=self.torch_dtype), 0)
        if not torch.isfinite(video).all() or (video.abs() > 1.00001).any():
            raise ValueError('Valid history pixels must be finite in [-1,1]')
        if split:
            anchor = self._encode_video_latents(video[:, :, :1], tiled=tiled)
            recent = self._encode_video_latents(video[:, :, 1:], tiled=tiled)
            if anchor.shape[2] != 1 or recent.shape[2] != 2:
                raise ValueError('Expected independent anchor 1 -> 1 and recent 5 -> 2')
            latents = torch.cat((anchor, recent[:, :, 1:]), dim=2)
            grouped = torch.stack((pixel_valid[:, 0], pixel_valid[:, 1:].all(1)), 1)
        else:
            latents = self._encode_video_latents(video, tiled=tiled)
            if latents.shape[2] != 2:
                raise ValueError('Expected causal VAE temporal compression 5 -> 2')
            grouped = torch.stack((pixel_valid[:, 0], pixel_valid.all(1)), 1)
        h, w = latents.shape[-2:]
        p, q = self.video_expert.patch_size[1:]
        valid = F.adaptive_avg_pool2d(grouped.float().flatten(0, 1)[:, None], (h, w)) >= 1 - 1e-07
        valid = (F.avg_pool2d(valid.float(), (p, q)) >= 1 - 1e-07).reshape(B, -1)
        self._history_encoded = dict(latents=latents, offsets=offsets, valid=valid)

    def _refresh_history(self, layout=None):
        encoded = self._history_encoded
        B = encoded['latents'].shape[0]
        bf = getattr(self, '_batch_flex', None)
        present = bf.present_v if bf is not None else torch.full((B,), bool(self._infer_present_v), device=self.device, dtype=torch.bool)
        p, q = self.video_expert.patch_size[1:]
        current = self._ms_masks['video'][:, 0].float()[:, None]
        current = (F.avg_pool2d(current, (p, q)) >= 1 - 1e-07).flatten(1).repeat(1, 2)
        payload = self.history_conditioning.prepare(self.video_expert, encoded['latents'], encoded['offsets'], encoded['valid'] & current, present)
        if layout is None:
            keys = ['video', 'dino']
            if not self._pointmap_globally_off:
                keys.append('pointmap')
            if not self.skip_unlabeled_flow_tokens:
                keys.append('flow')
            keys.append('action')
            valid = [self._ms_token_valid[k] for k in keys]
        else:
            valid = [self._sequence_valid(k, n, tpf) for k, n, tpf in layout if n]
        payload['query_valid'] = torch.cat(valid, 1)
        slots = torch.zeros_like(payload['query_valid'])
        current_count = current.shape[1] // 2
        slots[:, :current_count] = True
        payload['current_slots'] = slots
        payload['current_valid'] = slots & payload['query_valid']
        self.mot.history_payload = payload

    @torch.no_grad()
    def infer_action(self, *args, history_video=None, history_frame_offsets=None, history_pixel_valid=None, **kwargs):
        try:
            self._encode_history(history_video, history_frame_offsets, history_pixel_valid, tiled=kwargs.get('tiled', False))
            return super().infer_action(*args, **kwargs)
        finally:
            self._history_encoded = None
            self.mot.history_payload = None

def enable_history(model, value):
    config = HistoryConfig.parse(value)
    if config.mode == 'none':
        return model
    if not isinstance(model, DualCooperativeMultiSourceCloudRoboWM) or not model.mot.hbridge_enabled:
        raise ValueError('History v1 requires the existing dual-action HBridge model')
    if model.video_expert.action_conditioned:
        raise ValueError('History context must not contain future/noisy action conditioning')
    model.__class__ = HistoryMultiSourceCloudRoboWM
    model.history_config = config
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        module = HistoryConditioning(model.video_expert.hidden_dim, model.mot.num_layers, config)
    model.history_conditioning = module.to(device=model.device, dtype=model.torch_dtype)
    object.__setattr__(model.mot, 'history_conditioning', model.history_conditioning)
    model.mot.history_payload = None
    model._history_encoded = None
    return model
