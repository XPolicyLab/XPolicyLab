import contextlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from cloudrobo_wm.composite_layouts import LayoutSpec, get_layout
from cloudrobo_wm.utils.logging_config import get_logger
from .action_dit import ActionDiT
from .helpers.dino import select_aux_frame_slots
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)

class CloudRoboWMBackbone(torch.nn.Module):

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
        super().to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return (height, width, num_frames)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]], *, padding_mode='zero_visible'):
        if padding_mode not in ('zero_visible', 'masked'):
            raise ValueError(f'Unknown text padding mode: {padding_mode}')
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError('Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`.')
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

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    @torch.no_grad()
    def _build_mot_attention_mask(self, video_seq_len: int, action_seq_len: int, video_tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(video_seq_len=video_seq_len, video_tokens_per_frame=video_tokens_per_frame, device=device)
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_hbridge_self_masks(self, video_seq_len: int, action_seq_len: int, video_tokens_per_frame: int, device: torch.device) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        if not self.mot.hbridge_enabled:
            return (None, None)
        video_self_mask = self.video_expert.build_video_to_video_mask(video_seq_len=video_seq_len, video_tokens_per_frame=video_tokens_per_frame, device=device)
        action_self_mask = torch.ones((action_seq_len, action_seq_len), dtype=torch.bool, device=device)
        return ([video_seq_len, action_seq_len], [video_self_mask, action_self_mask])

    def _use_loop_compile(self) -> bool:
        return getattr(self, '_compile_inference', False) and getattr(self, '_compile_scope', 'step') == 'loop' and (self.device.type == 'cuda')

    def _drop_trivial_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if getattr(self, '_infer_attn_backend', 'sdpa') == 'auto' and mask is not None and bool(mask.all()):
            return None
        return mask

    def _sdpa_priority_ctx(self):
        if getattr(self, '_infer_attn_backend', 'sdpa') == 'auto':
            torch_minor = tuple((int(x) for x in torch.__version__.split('.')[:2]))
            if torch_minor < (2, 9):
                if not getattr(self, '_warned_sdpa_priority_gate', False):
                    logger.warning("attn_backend='auto': skipping cuDNN-first joint SDPA priority on torch %s (<2.9 inductor cudnn-sdpa stride bug); action-fast-path mask drops remain active.", torch.__version__)
                    self._warned_sdpa_priority_gate = True
                return contextlib.nullcontext()
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True)
        return contextlib.nullcontext()

    def prepare_for_inference(self):
        self.eval().requires_grad_(False)
        self._glue_cache_enabled = False
        self._glue_cache = {}
        self._joint_loop_cuda_graph_enabled = False
        self._compile_inference = False
        self._compile_mode = 'default'
        self._compile_scope = 'step'
        self._infer_attn_backend = 'sdpa'
        self._compile_backend = 'inductor'
        self.mot.set_attn_mode('sdpa')
