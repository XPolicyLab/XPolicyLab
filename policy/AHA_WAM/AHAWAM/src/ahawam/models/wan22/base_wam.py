from abc import ABC, abstractmethod
import os
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ahawam.utils.logging_config import get_logger
from ahawam.utils.video_io import save_mp4
from ahawam.utils.video_metrics import (
    pil_frames_to_video_tensor,
    video_psnr,
    video_ssim,
)

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)


class BaseWAM(torch.nn.Module, ABC):
    """Shared Wan world-model base with video/context/checkpoint utilities."""

    # === Required subclass interface ===
    @abstractmethod
    def build_inputs(self, sample, tiled: bool = False):
        """
        Build model-specific training inputs from a raw training sample.
        Most of the time you just need to call `_build_standard_inputs'
        """

        raise NotImplementedError

    @abstractmethod
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the MoT attention mask for the concrete action policy."""
        raise NotImplementedError

    @abstractmethod
    def training_loss(self, sample, tiled: bool = False):
        """Compute the model-specific training loss."""
        raise NotImplementedError

    @abstractmethod
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Run model-specific action inference."""
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        """Run model-specific top-level inference."""
        raise NotImplementedError

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
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self._cached_action_attn_mask_key: Optional[tuple[int, int, int]] = None
        self._cached_action_attn_mask: Optional[torch.Tensor] = None
        self._denoise_step_is_compiled = False

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError(
                f"`video_dit_config` is required for {cls.__name__}.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
            )
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def get_additional_trainable_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        if self.proprio_encoder is not None:
            modules["proprio_encoder"] = self.proprio_encoder
        return modules

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(
                f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        encoder_device = next(self.proprio_encoder.parameters()).device
        proprio_token = self.proprio_encoder(
            proprio.to(device=encoder_device, dtype=self.torch_dtype).unsqueeze(1)
        ).to(device=context.device, dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @staticmethod
    def _normalize_video_latent_cache_paths(
        cache_paths, batch_size: int
    ) -> list[Optional[str]] | None:
        if cache_paths is None:
            return None
        if isinstance(cache_paths, str):
            paths = [cache_paths]
        elif isinstance(cache_paths, (list, tuple)):
            paths = list(cache_paths)
        else:
            raise TypeError(
                "`sample['video_latent_cache_path']` must be str/list[str]/tuple[str], "
                f"got {type(cache_paths)}"
            )
        if len(paths) != batch_size:
            raise ValueError(
                "`sample['video_latent_cache_path']` batch mismatch: "
                f"got {len(paths)} entries vs batch_size={batch_size}"
            )
        return [None if path in (None, "") else str(path) for path in paths]

    @staticmethod
    def _save_video_latent_cache(
        cache_path: str, payload: dict[str, torch.Tensor]
    ) -> None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    @staticmethod
    def _load_video_latent_cache(cache_path: str) -> Optional[dict[str, torch.Tensor]]:
        if not os.path.exists(cache_path):
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        input_latents = payload.get("input_latents")
        first_frame_latents = payload.get("first_frame_latents")
        if not isinstance(input_latents, torch.Tensor):
            return None
        input_latents = cast(torch.Tensor, input_latents)
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, 0:1]
        if not isinstance(first_frame_latents, torch.Tensor):
            return None
        first_frame_latents = cast(torch.Tensor, first_frame_latents)
        if input_latents.ndim != 4 or first_frame_latents.ndim != 4:
            return None
        if first_frame_latents.shape[1] != 1:
            return None
        return {
            "input_latents": input_latents.contiguous(),
            "first_frame_latents": first_frame_latents.contiguous(),
        }

    def _load_or_compute_video_latents(
        self,
        video: torch.Tensor,
        cache_paths: list[Optional[str]] | None,
        *,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = int(video.shape[0])
        if cache_paths is None:
            input_video = video.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
            first_frame_latents = input_latents[:, :, 0:1]
            return input_latents, first_frame_latents

        latent_batches: list[Optional[torch.Tensor]] = [None] * batch_size
        first_frame_batches: list[Optional[torch.Tensor]] = [None] * batch_size
        missing_indices: list[int] = []

        for idx, cache_path in enumerate(cache_paths):
            if cache_path is None:
                missing_indices.append(idx)
                continue
            payload = self._load_video_latent_cache(cache_path)
            if payload is None:
                missing_indices.append(idx)
                continue
            latent_batches[idx] = payload["input_latents"].to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            first_frame_batches[idx] = payload["first_frame_latents"].to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )

        if missing_indices:
            missing_video = video[missing_indices].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            missing_latents = self._encode_video_latents(missing_video, tiled=tiled)
            missing_first_frames = missing_latents[:, :, 0:1]
            for offset, sample_idx in enumerate(missing_indices):
                latent = missing_latents[offset]
                first_frame = missing_first_frames[offset]
                latent_batches[sample_idx] = latent
                first_frame_batches[sample_idx] = first_frame
                cache_path = cache_paths[sample_idx]
                if cache_path is None:
                    continue
                self._save_video_latent_cache(
                    cache_path,
                    payload={
                        "input_latents": latent.detach().to(device="cpu").contiguous(),
                        "first_frame_latents": first_frame.detach()
                        .to(device="cpu")
                        .contiguous(),
                    },
                )

        if any(latent is None for latent in latent_batches):
            raise RuntimeError("Failed to populate all video latent cache entries.")
        if any(first_frame is None for first_frame in first_frame_batches):
            raise RuntimeError(
                "Failed to populate all first-frame latent cache entries."
            )
        input_latents = torch.stack(
            [cast(torch.Tensor, latent_batches[idx]) for idx in range(batch_size)],
            dim=0,
        )
        first_frame_latents = torch.stack(
            [cast(torch.Tensor, first_frame_batches[idx]) for idx in range(batch_size)],
            dim=0,
        )
        return input_latents, first_frame_latents

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        image_list = [image.unsqueeze(1) for image in input_image]
        z = self.vae.encode(
            image_list,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = torch.stack(z, dim=0)
        return z

    def _decode_latents(
        self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _build_standard_inputs(self, sample, tiled: bool = False) -> dict[str, Any]:
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                f"{self.__class__.__name__} training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )

        if "action" not in sample:
            raise ValueError(
                f"`sample['action']` is required for {self.__class__.__name__} training."
            )

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
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
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
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        video_latent_cache_paths = self._normalize_video_latent_cache_paths(
            sample.get("video_latent_cache_path", None),
            batch_size=batch_size,
        )
        input_latents, cached_first_frame_latents = self._load_or_compute_video_latents(
            video,
            video_latent_cache_paths,
            tiled=tiled,
        )

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = cached_first_frame_latents
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
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
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

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
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

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
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
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
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _predict_action_noise_with_cache_trainable(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        return self._predict_action_noise_with_cache_trainable(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )

    def _denoise_step_compiled(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache_k: list[torch.Tensor],
        video_kv_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        action_freqs: torch.Tensor,
    ) -> torch.Tensor:
        ae = self.action_expert
        seq_len = latents_action.shape[1]

        t = ae.time_embedding(
            sinusoidal_embedding_1d(ae.freq_dim, timestep_action)
        )
        t_mod = ae.time_projection(t).unflatten(1, (6, ae.hidden_dim))
        tokens = ae.action_encoder(latents_action)
        context_emb = ae.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        action_tokens = self.mot._forward_action_with_video_cache_inner(
            action_tokens=tokens,
            action_freqs=action_freqs,
            action_t_mod=t_mod,
            action_context=context_emb,
            action_context_mask=context_attn_mask,
            video_kv_cache_k=video_kv_cache_k,
            video_kv_cache_v=video_kv_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return ae.head(action_tokens)

    @torch.no_grad()
    def _prepare_action_start_latents(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        start_latents: Optional[torch.Tensor],
        seed: Optional[int],
        rand_device: str,
    ) -> tuple[torch.Tensor, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        batch_size = int(input_image.shape[0])
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        if start_latents is None:
            generator = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(seed)
            )
            latents_action = torch.randn(
                (batch_size, action_horizon, self.action_expert.action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if start_latents.ndim != 3:
                raise ValueError(
                    "`start_latents` must have shape [B, T, action_dim], "
                    f"got {tuple(start_latents.shape)}"
                )
            if start_latents.shape[0] != batch_size:
                raise ValueError(
                    f"`start_latents.shape[0]` must equal input_image batch={batch_size}, "
                    f"got {start_latents.shape[0]}"
                )
            if start_latents.shape[1] != action_horizon:
                raise ValueError(
                    f"`start_latents.shape[1]` must equal action_horizon={action_horizon}, "
                    f"got {start_latents.shape[1]}"
                )
            if start_latents.shape[2] != self.action_expert.action_dim:
                raise ValueError(
                    f"`start_latents.shape[2]` must equal action_dim={self.action_expert.action_dim}, "
                    f"got {start_latents.shape[2]}"
                )
            latents_action = start_latents.to(
                device=self.device, dtype=self.torch_dtype
            )
        return latents_action, batch_size

    @torch.no_grad()
    def _prepare_action_context(
        self,
        *,
        prompt: Optional[str],
        batch_size: int,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2:
                raise ValueError(
                    f"`proprio` must be [D] or [B,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            if proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` batch dim must match input_image batch={batch_size}, got {proprio.shape[0]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_context = context is not None or context_mask is not None
        if use_context:
            prompt = None
        use_prompt = prompt is not None
        if not use_prompt and not use_context:
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )

        if use_prompt:
            if isinstance(prompt, (list, tuple)) and len(prompt) != batch_size:
                raise ValueError(
                    f"`prompt` batch size must match input_image batch={batch_size}, got {len(prompt)}"
                )
            context, context_mask = self.encode_prompt(prompt)
        else:
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
            if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
                raise ValueError(
                    "`context/context_mask` batch dim must match input_image batch: "
                    f"{context.shape[0]} / {context_mask.shape[0]} vs {batch_size}"
                )
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    @torch.no_grad()
    def _prepare_action_video_pre(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        tiled: bool,
    ) -> tuple[dict[str, Any], int, int]:
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        return video_pre, video_seq_len, video_tokens_per_frame

    @torch.no_grad()
    def _prefill_action_video_cache(
        self,
        *,
        video_pre: dict[str, Any],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> list[dict[str, torch.Tensor]]:
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        return self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )

    @torch.no_grad()
    def _prepare_action_inference(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        start_latents: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "`action-only inference` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        latents_action, batch_size = self._prepare_action_start_latents(
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=start_latents,
            seed=seed,
            rand_device=rand_device,
        )
        context, context_mask = self._prepare_action_context(
            prompt=prompt,
            batch_size=batch_size,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
        )
        video_pre, video_seq_len, video_tokens_per_frame = (
            self._prepare_action_video_pre(
                input_image=input_image,
                context=context,
                context_mask=context_mask,
                tiled=tiled,
            )
        )
        attn_mask_key = (
            int(video_seq_len),
            int(latents_action.shape[1]),
            int(video_tokens_per_frame),
        )
        if self._cached_action_attn_mask_key != attn_mask_key:
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=video_tokens_per_frame,
                device=video_pre["tokens"].device,
            )
            self._cached_action_attn_mask = attention_mask
            self._cached_action_attn_mask_key = attn_mask_key
        else:
            if self._cached_action_attn_mask is None:
                raise RuntimeError("Cached action attention mask key exists without a tensor.")
            attention_mask = self._cached_action_attn_mask
        video_kv_cache = self._prefill_action_video_cache(
            video_pre=video_pre,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
        )
        return {
            "start_latents": latents_action,
            "context": context,
            "context_mask": context_mask,
            "batch_size": batch_size,
            "video_pre": video_pre,
            "video_seq_len": video_seq_len,
            "video_tokens_per_frame": video_tokens_per_frame,
            "attention_mask": attention_mask,
            "video_kv_cache": video_kv_cache,
        }

    @torch.no_grad()
    def _rollout_action_latents_with_cache(
        self,
        *,
        start_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        num_inference_steps: int,
        sigma_shift: Optional[float] = None,
        capture_step_indices: Optional[list[int] | tuple[int, ...]] = None,
    ) -> dict[str, Any]:
        latents_action = start_latents
        capture_steps = None
        captured_states: dict[int, torch.Tensor] | None = None
        if capture_step_indices is not None:
            capture_steps = tuple(sorted({int(x) for x in capture_step_indices}))
            if any(step < 0 for step in capture_steps):
                raise ValueError(
                    f"`capture_step_indices` must be non-negative, got {capture_steps}"
                )
            if capture_steps and capture_steps[-1] > int(num_inference_steps):
                raise ValueError(
                    "`capture_step_indices` cannot exceed `num_inference_steps`: "
                    f"{capture_steps} vs {num_inference_steps}"
                )
            if not capture_steps or capture_steps[0] != 0:
                raise NotImplementedError(
                    "`capture_step_indices[0]` must be 0. "
                    "Non-noise rollout starts are not supported yet."
                )
            if capture_steps[-1] != int(num_inference_steps):
                raise NotImplementedError(
                    "`capture_step_indices[-1]` must equal `num_inference_steps`. "
                    "Distillation currently requires the final rollout state."
                )
            captured_states = {0: latents_action.detach().clone()}

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        if not self._denoise_step_is_compiled:
            if self.device.type == "cuda":
                self._denoise_step_compiled = torch.compile(
                    self._denoise_step_compiled,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            self._denoise_step_is_compiled = True

        total_seq_len = int(video_seq_len) + int(latents_action.shape[1])
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        video_kv_cache_k = [layer_cache["k"] for layer_cache in video_kv_cache]
        video_kv_cache_v = [layer_cache["v"] for layer_cache in video_kv_cache]
        action_freqs = (
            self.action_expert.freqs[: latents_action.shape[1]]
            .view(latents_action.shape[1], 1, -1)
            .to(device=latents_action.device)
        )
        for step_index, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action),
            start=1,
        ):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action = self._denoise_step_compiled(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache_k=video_kv_cache_k,
                video_kv_cache_v=video_kv_cache_v,
                action_attention_mask=action_attention_mask,
                action_freqs=action_freqs,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            if captured_states is not None and step_index in capture_steps:
                captured_states[step_index] = latents_action.detach().clone()

        if captured_states is not None:
            if 0 not in captured_states:
                raise RuntimeError("Missing captured start state at step 0.")
            if int(num_inference_steps) not in captured_states:
                raise RuntimeError(
                    f"Missing captured final state at step {int(num_inference_steps)}."
                )

        return {
            "final_latents": latents_action,
            "timesteps": infer_timesteps_action.detach().clone(),
            "deltas": infer_deltas_action.detach().clone(),
            "captured_states": captured_states,
            "capture_step_indices": capture_steps,
        }

    @torch.no_grad()
    def _evaluate_action_metrics(
        self,
        *,
        sample: dict[str, Any],
        pred_action: Optional[torch.Tensor],
    ) -> dict[str, float]:
        action = sample.get("action", None)
        if action is None or pred_action is None:
            return {}
        proprio = sample.get("proprio", None)
        if proprio is None:
            raise ValueError(
                "Eval sample must contain `proprio` for action denormalization."
            )

        proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
        processor = sample["_eval_dataset"].lerobot_dataset.processor
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        denorm_actions = {}
        for action_name, raw_action in (("pred", pred_action), ("gt", action)):
            if not isinstance(raw_action, torch.Tensor):
                raise TypeError(
                    f"{action_name} action must be a torch.Tensor, got {type(raw_action)}"
                )
            if raw_action.ndim == 2:
                action_btd = raw_action.unsqueeze(0)
            elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                action_btd = raw_action
            else:
                raise ValueError(
                    f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                )
            action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
            batch = {"action": action_btd, "state": proprio}
            batch = processor.action_state_merger.backward(batch)
            batch = processor.normalizer.backward(batch)
            merged_batch = {
                "action": {
                    meta["key"]: batch["action"][meta["key"]].squeeze(0)
                    for meta in action_meta
                },
                "state": {
                    meta["key"]: batch["state"][meta["key"]].squeeze(0)
                    for meta in state_meta
                },
            }
            merged_batch = processor.action_state_merger.forward(merged_batch)
            denorm_action = merged_batch["action"].unsqueeze(0)
            if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                raise ValueError(
                    f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                )
            denorm_actions[action_name] = denorm_action

        pred_action_denorm = denorm_actions["pred"]
        gt_action_denorm = denorm_actions["gt"]
        if pred_action_denorm.shape != gt_action_denorm.shape:
            raise ValueError(
                "Predicted action/GT action shape mismatch after denormalization: "
                f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
            )
        action_diff = pred_action_denorm - gt_action_denorm
        return {
            "action_l1": float(action_diff.abs().mean().item()),
            "action_l2": float(action_diff.pow(2).mean().item()),
        }

    @torch.no_grad()
    def _build_eval_infer_kwargs(
        self,
        *,
        sample: dict[str, Any],
        eval_num_inference_steps: int,
    ) -> dict[str, Any]:
        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = (
            sample["action"][0]
            if "action" in sample and sample["action"] is not None
            else None
        )
        proprio = (
            sample["proprio"][0, 0]
            if "proprio" in sample and sample["proprio"] is not None
            else None
        )
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": int(eval_num_inference_steps),
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        return infer_kwargs

    @torch.no_grad()
    def evaluate_validation(
        self,
        sample: dict[str, Any],
        *,
        eval_num_inference_steps: int,
        eval_dir: Optional[str] = None,
        global_step: int = 0,
        process_index: int = 0,
    ) -> dict[str, float | str]:
        val_loss, val_loss_dict = self.training_loss(sample)
        pred = self.infer(
            **self._build_eval_infer_kwargs(
                sample=sample,
                eval_num_inference_steps=eval_num_inference_steps,
            )
        )

        pred_video = pred.get("video", None)
        pred_action = pred.get("action", None)
        result: dict[str, float | str] = {
            "val_loss": float(val_loss.float().item()),
        }
        for key, value in val_loss_dict.items():
            if str(key).startswith("val_"):
                result[str(key)] = float(value)

        result.update(
            self._evaluate_action_metrics(sample=sample, pred_action=pred_action)
        )
        if pred_video is None:
            return result

        video0 = sample["video"][0]
        gt_video_tensor = (
            (video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        ).contiguous()
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        if pred_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError(
                "Eval infer prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

        result["psnr_rg"] = float(
            video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        )
        result["ssim_rg"] = float(
            video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
        )

        gt_video_batch = video0.unsqueeze(0).to(
            device=self.device, dtype=self.torch_dtype
        )
        vae_latents = self._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = self._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)
        if vae_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError(
                "Eval VAE reconstruction/GT shape mismatch: "
                f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

        result["psnr_dg"] = float(
            video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        )
        result["ssim_dg"] = float(
            video_ssim(pred=vae_video_tensor, target=gt_video_tensor)
        )
        result["psnr_rd"] = float(
            video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        )
        result["ssim_rd"] = float(
            video_ssim(pred=pred_video_tensor, target=vae_video_tensor)
        )

        if eval_dir is not None:
            stitched_video_tensor = torch.cat(
                [pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=2
            ).contiguous()
            stitched_frames = []
            for t in range(stitched_video_tensor.shape[1]):
                frame = (
                    stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                    * 255.0
                ).astype(np.uint8)
                stitched_frames.append(Image.fromarray(frame))
            video_path = os.path.join(
                eval_dir,
                f"step_{int(global_step):06d}_rank_{int(process_index):03d}.mp4",
            )
            save_mp4(stitched_frames, video_path, fps=8)
            result["video_path"] = video_path
        return result

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
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
