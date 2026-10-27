"""Base class of the PrismWAM models.

It holds what Stage 1 (:class:`~.prismwam_sv.PrismWAMSV`) and Stage 2
(:class:`~.rgb_depth_action.PrismWAMRGBDepthAction`) share: flow-matching
schedulers, text and proprio context, VAE encoding, the video loss and
checkpoint I/O. The subclasses implement training and inference.
"""

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class PrismWAM(torch.nn.Module):
    """MoT world model with video (depth) and action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_use_text_embedding: bool = True,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(
                torch_dtype
            )
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.video_use_text_embedding = bool(video_use_text_embedding)
        self.to(self.device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
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
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        text_parameter = next(self.text_encoder.parameters(), None)
        text_device = self.device if text_parameter is None else text_parameter.device
        ids = ids.to(text_device)
        mask = mask.to(text_device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (prompt_emb.to(device=self.device), mask.to(device=self.device))

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(
                f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _resolve_training_language_context(
        self, sample: dict[str, Any], *, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "PrismWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        if not isinstance(context, torch.Tensor) or not isinstance(
            context_mask, torch.Tensor
        ):
            raise TypeError("`context` and `context_mask` must be tensors.")
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != int(batch_size) or tuple(context_mask.shape) != tuple(
            context.shape[:2]
        ):
            raise ValueError(
                f"Training language context batch/sequence mismatch: context={tuple(context.shape)}, mask={tuple(context_mask.shape)}, batch_size={batch_size}."
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def _resolve_inference_language_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            return self.encode_prompt(prompt)
        if context is None or context_mask is None:
            raise ValueError(
                "`context` and `context_mask` must be both provided together."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(
                f"Inference context mask must match context batch/sequence shape: {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def _video_text_embedding_enabled(self) -> bool:
        return bool(getattr(self, "video_use_text_embedding", True))

    @staticmethod
    def _text_context_payload(
        context: torch.Tensor, context_mask: torch.Tensor, *, scale: float = 1.0
    ) -> dict[str, Any]:
        return {
            "cross_attention": {
                "text": {
                    "context": context,
                    "mask": context_mask,
                    "scale": float(scale),
                }
            }
        }

    def _video_context_payload(self, video_pre: dict[str, Any]) -> dict[str, Any]:
        context = video_pre["context"]
        context_mask = video_pre["context_mask"]
        branches: dict[str, dict[str, Any]] = {}
        if self._video_text_embedding_enabled():
            branches["text"] = {
                "context": context,
                "mask": context_mask,
                "scale": float(getattr(self, "video_text_embedding_scale", 1.0)),
            }
        return {"cross_attention": branches}

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        expected_channels = int(getattr(self.vae, "input_channels", 3))
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != expected_channels
        ):
            raise ValueError(
                f"`input_image` must have shape [1,{expected_channels},H,W] or [{expected_channels},H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def build_inputs(self, sample, tiled: bool = False):
        video = sample.get("video")
        cached_video_latents = sample.get("video_latents")
        if (video is None) == (cached_video_latents is None):
            raise ValueError(
                "Provide exactly one of `sample['video']` or `sample['video_latents']`."
            )
        expected_channels = int(getattr(self.vae, "input_channels", 3))
        proprio = sample.get("proprio", None)
        image_is_pad = sample.get("image_is_pad", None)
        if video is not None:
            if video.ndim != 5:
                raise ValueError(
                    f"`sample['video']` must be 5D [B, C, T, H, W], got shape {tuple(video.shape)}"
                )
            if video.shape[1] != expected_channels:
                raise ValueError(
                    f"`sample['video']` channel dimension must be {expected_channels}, got shape {tuple(video.shape)}"
                )
            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
                )
        else:
            if cached_video_latents.ndim != 5:
                raise ValueError(
                    f"`sample['video_latents']` must be 5D [B,C,T,H,W], got shape {tuple(cached_video_latents.shape)}."
                )
            if image_is_pad is None or image_is_pad.ndim != 2:
                raise ValueError(
                    "Cached video latents require `sample['image_is_pad']` with shape [B,T_video]."
                )
            batch_size = int(cached_video_latents.shape[0])
            num_frames = int(image_is_pad.shape[1])
            latent_channels = int(
                getattr(
                    getattr(self.vae, "model", None),
                    "z_dim",
                    getattr(self.vae, "z_dim", -1),
                )
            )
            if latent_channels > 0 and cached_video_latents.shape[1] != latent_channels:
                raise ValueError(
                    f"Cached video latent channel mismatch: expected {latent_channels}, got {cached_video_latents.shape[1]}."
                )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for training, got T={num_frames}")
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for PrismWAM training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )
        action_dim_mask = sample.get("action_dim_mask", None)
        if action_dim_mask is not None:
            expected_action_dim_mask = (batch_size, int(action.shape[2]))
            if (
                action_dim_mask.dtype != torch.bool
                or tuple(action_dim_mask.shape) != expected_action_dim_mask
            ):
                raise ValueError(
                    f"`sample['action_dim_mask']` must be bool [B,D], got shape={tuple(action_dim_mask.shape)}, dtype={action_dim_mask.dtype}, expected={expected_action_dim_mask}."
                )
            if bool((~action_dim_mask.any(dim=1)).any().item()):
                raise ValueError("Every action_dim_mask row must enable a channel.")
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    f"`sample['image_is_pad']` shape mismatch: got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        if video is not None:
            input_video = video.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
        else:
            input_latents = cached_video_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        context, context_mask = self._resolve_training_language_context(
            sample, batch_size=batch_size
        )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio_dim_mask = sample.get("proprio_dim_mask", None)
            if proprio_dim_mask is not None:
                expected_proprio_dim_mask = (batch_size, self.proprio_dim)
                if (
                    proprio_dim_mask.dtype != torch.bool
                    or tuple(proprio_dim_mask.shape) != expected_proprio_dim_mask
                ):
                    raise ValueError(
                        f"`sample['proprio_dim_mask']` must be bool [B,D], got shape={tuple(proprio_dim_mask.shape)}, dtype={proprio_dim_mask.dtype}, expected={expected_proprio_dim_mask}."
                    )
                if bool((~proprio_dim_mask.any(dim=1)).any().item()):
                    raise ValueError(
                        "Every proprio_dim_mask row must enable a channel."
                    )
                proprio_dim_mask = proprio_dim_mask.to(
                    device=proprio.device, dtype=torch.bool, non_blocking=True
                )
            proprio = proprio[:, 0, :]
            if proprio_dim_mask is not None:
                proprio = proprio.masked_fill(~proprio_dim_mask, 0.0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if action_dim_mask is not None:
            action_dim_mask = action_dim_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
            action = action.masked_fill(~action_dim_mask.unsqueeze(1), 0.0)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "action": action,
            "action_dim_mask": action_dim_mask,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                f"Cannot align `image_is_pad` with video latent steps: num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video-loss mask shape mismatch: mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _extra_checkpoint_payload(self) -> dict[str, Any]:
        """Extension seam for model-family provenance and auxiliary state."""
        return {}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"step": step, "torch_dtype": str(self.torch_dtype)}
        extra_payload = self._extra_checkpoint_payload()
        if not isinstance(extra_payload, dict):
            raise TypeError("Extra checkpoint payload must be a dictionary.")
        overlap = set(payload).intersection(extra_payload)
        if overlap:
            raise ValueError(
                f"Extra checkpoint payload cannot override reserved keys: {sorted(overlap)}."
            )
        payload.update(extra_payload)
        payload["mot"] = self.mot.state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            raise ValueError(f"Checkpoint has no `mot` weights: {path}")
        missing, unexpected = self.mot.load_state_dict(payload["mot"], strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint does not match the model: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}."
            )
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(
                    payload["proprio_encoder"], strict=True
                )
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
