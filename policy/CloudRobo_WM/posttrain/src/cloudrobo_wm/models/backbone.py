import contextlib
from typing import Mapping, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from cloudrobo_wm.composite_layouts import LayoutSpec, get_layout
from cloudrobo_wm.utils.logging_config import get_logger
from .action_dit import ActionDiT
from .helpers.dino import select_aux_frame_slots
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
logger = get_logger(__name__)

class CloudRoboBackbone(torch.nn.Module):

    def __init__(self, video_expert, action_expert: ActionDiT, mot: MoT, vae, text_encoder=None, tokenizer=None, text_dim: Optional[int]=None, proprio_dim: Optional[int]=None, device: str='cpu', torch_dtype: torch.dtype=torch.float32, video_train_shift: float=5.0, video_infer_shift: float=5.0, video_num_train_timesteps: int=1000, action_train_shift: float=5.0, action_infer_shift: float=5.0, action_num_train_timesteps: int=1000, loss_lambda_video: float=1.0, loss_lambda_action: float=1.0, composite_layout: Union[str, LayoutSpec, None]='tshape_robotwin_384x320_uniform', composite_layout_slot_key_map: Optional[Mapping[str, str]]=None):
        super().__init__()
        self._layout: LayoutSpec = get_layout(composite_layout)
        self._slot_key_map = self._layout.resolve_slot_key_map(composite_layout_slot_key_map)
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError('`text_dim` is required when `text_encoder` is not loaded.')
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=video_num_train_timesteps, shift=video_train_shift)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=action_num_train_timesteps, shift=action_train_shift)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift)
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.offload_text_encoder = False
        self.to(self.device)
    FROZEN_MODULES: set[str] = {'vae', 'text_encoder'}

    def _layout_kwargs(self) -> dict:
        return {'layout': self._layout, 'slot_key_map': self._slot_key_map}

    def _dino_encode_kwargs(self) -> dict:
        out = self._layout_kwargs()
        cam_patches = getattr(self, 'dino_cam_patches', None)
        if cam_patches is not None:
            out['cam_patches'] = cam_patches
        out['pool_mode'] = getattr(self, 'dino_pool_mode', 'avg')
        return out

    def to(self, *args, **kwargs):
        te = None
        if self.offload_text_encoder and self.text_encoder is not None:
            te = self.text_encoder
            del self._modules['text_encoder']
        super().to(*args, **kwargs)
        if te is not None:
            self._modules['text_encoder'] = te
        elif self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]], *, padding_mode='zero_visible'):
        if padding_mode not in ('zero_visible', 'masked'):
            raise ValueError(f'Unknown text padding mode: {padding_mode}')
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError('Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`.')
        if self.offload_text_encoder:
            self.text_encoder.cpu()
            ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to('cpu')
            mask = mask.to('cpu', dtype=torch.bool)
            prompt_emb = self.text_encoder(ids, mask)
        else:
            ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to(self.device)
            mask = mask.to(self.device, dtype=torch.bool)
            prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        if padding_mode == 'zero_visible':
            mask = torch.ones_like(mask)
        return (prompt_emb.to(device=self.device), mask)

    def _append_proprio_to_context(self, context: torch.Tensor, context_mask: torch.Tensor, proprio: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(f'`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}')
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f'`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}')
        proprio_token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1))

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(video_tensor, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f'`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}')
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def build_inputs(self, sample, tiled: bool=False):
        if 'per_cam' in sample and 'video' not in sample:
            from cloudrobo_wm.per_cam_compose import compose_from_per_cam
            per_cam_device = {k: v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True) for k, v in sample['per_cam'].items()}
            sample['video'] = compose_from_per_cam(per_cam_device, self._layout, slot_key_map=self._slot_key_map)
        video = sample['video']
        if 'context' not in sample or 'context_mask' not in sample:
            raise ValueError("CloudRoboWM training requires `sample['context']` and `sample['context_mask']`.")
        context = sample['context']
        context_mask = sample['context_mask']
        proprio = sample.get('proprio', None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f'Video spatial dims must be multiples of 16, got H={height}, W={width}')
        if num_frames % 4 != 1:
            raise ValueError(f'Video T must satisfy T % 4 == 1, got T={num_frames}')
        if num_frames <= 1:
            raise ValueError(f'Video T must be > 1 for action-conditioned training, got T={num_frames}')
        if 'action' not in sample:
            raise ValueError("`sample['action']` is required for CloudRoboWM training.")
        action = sample['action']
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}")
        action_is_pad = sample.get('action_is_pad', None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}")
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})")
        action_dim = int(action.shape[-1])
        action_dim_is_pad = sample.get('action_dim_is_pad', None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(f"`sample['action_dim_is_pad']` must be 2D [B, A], got shape {tuple(action_dim_is_pad.shape)}")
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action_dim:
                raise ValueError(f"`sample['action_dim_is_pad']` shape mismatch: got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action_dim})")
        image_is_pad = sample.get('image_is_pad', None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}")
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(f"`sample['image_is_pad']` shape mismatch: got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})")
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(f'`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}')
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio.to(device=self.device, dtype=self.torch_dtype))
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {'context': context, 'context_mask': context_mask, 'input_latents': input_latents, 'first_frame_latents': first_frame_latents, 'fuse_vae_embedding_in_latents': fuse_flag, 'action': action, 'action_is_pad': action_is_pad, 'action_dim_is_pad': action_dim_is_pad, 'image_is_pad': image_is_pad}

    @staticmethod
    def _aux_per_frame_is_pad(image_is_pad: torch.Tensor, aux_temporal_stride: int, keep_far: bool=False) -> torch.Tensor:
        if image_is_pad.ndim != 2:
            raise ValueError(f'image_is_pad must be [B, T_video]; got shape {tuple(image_is_pad.shape)}')
        T = image_is_pad.shape[1]
        num_latent = (T - 1) // 4 + 1
        if num_latent < 1:
            raise ValueError(f'T_video={T} produces no latent frames.')
        vae_indices = [min(4 * i, T - 1) for i in range(num_latent)]
        aux_indices = [vae_indices[i] for i in select_aux_frame_slots(num_latent, aux_temporal_stride, keep_far)]
        return image_is_pad[:, aux_indices]

    @staticmethod
    def _masked_loss_reduction(loss_per_frame: torch.Tensor, is_pad_per_frame: Optional[torch.Tensor]) -> torch.Tensor:
        if is_pad_per_frame is None:
            return loss_per_frame.mean(dim=1)
        if is_pad_per_frame.shape != loss_per_frame.shape:
            raise ValueError(f'Loss/mask shape mismatch: loss={tuple(loss_per_frame.shape)}, mask={tuple(is_pad_per_frame.shape)}')
        valid = (~is_pad_per_frame).to(device=loss_per_frame.device, dtype=loss_per_frame.dtype)
        return (loss_per_frame * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _aggregate_image_is_pad_to_video_latents(image_is_pad: torch.Tensor, temporal_factor: int, include_initial: bool) -> torch.Tensor:
        if temporal_factor <= 0:
            raise ValueError(f'temporal_factor must be positive, got {temporal_factor}.')
        if image_is_pad.shape[1] < 1:
            raise ValueError('`image_is_pad` must contain at least one frame.')
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(f'Cannot align `image_is_pad` with video latent steps: num_frames={image_is_pad.shape[1]}, temporal_factor={temporal_factor}.')
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial:
            return torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        return latent_tail_is_pad

    def _compute_video_loss_per_sample(self, pred_video: torch.Tensor, target_video: torch.Tensor, image_is_pad: Optional[torch.Tensor], include_initial_video_step: bool) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction='none').mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        video_is_pad = self._aggregate_image_is_pad_to_video_latents(image_is_pad, temporal_factor=int(self.vae.temporal_downsample_factor), include_initial=include_initial_video_step)
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(f'Video-loss mask shape mismatch: mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}.')
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _sdpa_priority_ctx(self):
        return contextlib.nullcontext()
