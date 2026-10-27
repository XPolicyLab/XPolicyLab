import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn

from awomo.awomo.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class ImageWAM(torch.nn.Module):
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
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        stack: str = "wan22",
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
        concept_k: int = 16,
        lambda_concept: float = 0.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.lambda_concept = float(lambda_concept)
        self.concept_bottleneck = None
        if int(concept_k) != 0:
            raise ValueError("This inference package requires concept_k=0.")

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.stack = str(stack)
        self.omnigen2_online_text_cache_compatible = bool(omnigen2_online_text_cache_compatible)
        self.qwen_context_len = int(qwen_context_len)
        self.pack_proprio_after_text = bool(pack_proprio_after_text)

        self.to(self.device)


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            if hasattr(self.text_encoder, "to"):
                self.text_encoder.to(*args, **kwargs)
            elif hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "to"):
                self.text_encoder.model.to(*args, **kwargs)
        if hasattr(self, "dim_projector"):
            self.dim_projector.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self


    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        num_train_timesteps = float(getattr(scheduler, "num_train_timesteps", 1000))
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}.")
        return timestep / num_train_timesteps

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )

        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask,
            valid_rank,
            valid_counts[:, None] + 1 + invalid_rank,
        )
        new_context.scatter_(
            dim=1,
            index=target_indices[:, :, None].expand(-1, -1, context.shape[2]),
            src=context,
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]

        return new_context, new_context_mask

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size "
                f"({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )


    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids


    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
        cond_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        x0 = txt_len + cond_len
        a0 = txt_len + cond_len + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        mask[:, t0:r0, t0:x0] = True
        mask[:, r0:x0, t0:x0] = True
        mask[:, x0:a0, t0:a0] = True
        mask[:, a0:total, t0:x0] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        if cond_attention_mask is not None:
            if cond_attention_mask.ndim != 2 or tuple(cond_attention_mask.shape) != (batch_size, cond_len):
                raise ValueError(
                    "`cond_attention_mask` must be [B,cond_len], "
                    f"got {tuple(cond_attention_mask.shape)} for B={batch_size}, cond_len={cond_len}"
                )
            cond_valid = cond_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, r0:x0] &= cond_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}


    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        logger.info("Loading ImageWAM checkpoint from %s with payload keys=%s step=%s", path, sorted(payload.keys()), payload.get("step"))
        load_extra = getattr(self, "load_extra_checkpoint_payload", None)
        if callable(load_extra):
            load_extra(payload)
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.stack == "flux2":
                from .lora import merge_lora_state_dict_to_plain, remap_plain_linear_keys_to_lora_base

                mot_state = merge_lora_state_dict_to_plain(mot_state)
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            logger.info(
                "Loaded MoT weights from checkpoint: missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning("First missing MoT keys: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("First unexpected MoT keys: %s", unexpected_keys[:20])
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            load_result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            logger.info(
                "Loaded legacy video expert weights: missing_keys=%d unexpected_keys=%d",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Loaded proprio_encoder weights from checkpoint.")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if getattr(self, "concept_bottleneck", None) is not None and "concept_bottleneck" in payload:
            self.concept_bottleneck.load_state_dict(payload["concept_bottleneck"], strict=True)
            logger.info("Loaded concept_bottleneck weights from checkpoint.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
