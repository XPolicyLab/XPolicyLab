"""PrismWAM Stage 1: a depth video model without an action expert.

From the current depth frame and the instruction it predicts the future depth
frames. Its transformer later initializes the shared RGB/depth blocks of
Stage 2.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .prismwam import PrismWAM
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


class PrismWAMSV(PrismWAM):
    """Stage-1 depth video model: one video expert, no action expert."""

    def __init__(
        self,
        *,
        video_expert: nn.Module,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        video_use_text_embedding: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if "action" in mot.mixtures:
            raise ValueError("The Stage-1 model must not contain an Action expert.")
        if mot.mixtures["video"] is not video_expert:
            raise ValueError("`video_expert` must be the MoT video expert.")
        if text_dim is None:
            if text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when the T5 encoder is not loaded."
                )
            text_dim = int(text_encoder.dim)
        self.video_expert = video_expert
        self.action_expert = None
        self.mot = mot
        self.dit = mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_dim = int(text_dim)
        self.proprio_dim = None
        self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(video_num_train_timesteps),
            shift=float(video_train_shift),
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(video_num_train_timesteps),
            shift=float(video_infer_shift),
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        if self.loss_lambda_video <= 0:
            raise ValueError("`loss_lambda_video` must be positive.")
        self.loss_lambda_action = 0.0
        self.video_use_text_embedding = bool(video_use_text_embedding)
        self.to(self.device)

    def configure_full_dit_trainability(self) -> None:
        """Exclude the text cross-attention from the optimizer when it is disabled."""
        if self.video_use_text_embedding:
            return
        self.video_expert.text_embedding.requires_grad_(False)
        for block in self.video_expert.blocks:
            block.cross_attn.requires_grad_(False)
            block.norm3.requires_grad_(False)

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 128,
        load_text_encoder: bool = False,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any],
        skip_dit_load_from_pretrain: bool = False,
        vae_checkpoint_path: Optional[str] = None,
        vae_variant: str = "depth",
        depth_vae_require_latent_stats: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        video_use_text_embedding: bool = True,
    ) -> "PrismWAMSV":
        if not isinstance(video_dit_config, dict):
            raise TypeError("`video_dit_config` must be a dict.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=int(tokenizer_max_len),
            redirect_common_files=bool(redirect_common_files),
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
            load_text_encoder=bool(load_text_encoder),
            vae_checkpoint_path=vae_checkpoint_path,
            vae_variant=str(vae_variant),
            depth_vae_require_latent_stats=bool(depth_vae_require_latent_stats),
        )
        video_expert = components.dit
        mot = MoT(
            mixtures={"video": video_expert},
            mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        )
        model = cls(
            video_expert=video_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=float(video_train_shift),
            video_infer_shift=float(video_infer_shift),
            video_num_train_timesteps=int(video_num_train_timesteps),
            loss_lambda_video=float(loss_lambda_video),
            video_use_text_embedding=bool(video_use_text_embedding),
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": "NOT_INITIALIZED",
        }
        return model

    def build_inputs(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> dict[str, Any]:
        video = sample.get("video")
        cached_video_latents = sample.get("video_latents")
        if (video is None) == (cached_video_latents is None):
            raise ValueError(
                "Provide exactly one of `sample['video']` or `sample['video_latents']`."
            )
        image_is_pad = sample.get("image_is_pad")
        if video is not None:
            if video.ndim != 5:
                raise ValueError(
                    f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}."
                )
            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video H/W must be divisible by 16, got {(height, width)}."
                )
            input_latents = self._encode_video_latents(
                video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                tiled=tiled,
            )
        else:
            if cached_video_latents.ndim != 5:
                raise ValueError(
                    f"`sample['video_latents']` must be [B,C,T,H,W], got {tuple(cached_video_latents.shape)}."
                )
            if image_is_pad is None or image_is_pad.ndim != 2:
                raise ValueError(
                    "Cached video latents require `image_is_pad` with shape [B,T]."
                )
            batch_size = int(cached_video_latents.shape[0])
            num_frames = int(image_is_pad.shape[1])
            input_latents = cached_video_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(
                f"Video frame count must satisfy T > 1 and T % 4 == 1, got {num_frames}."
            )
        if image_is_pad is not None:
            if tuple(image_is_pad.shape) != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` must be {(batch_size, num_frames)}, got {tuple(image_is_pad.shape)}."
                )
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        context, context_mask = self._resolve_training_language_context(
            sample, batch_size=batch_size
        )
        first_frame_latents = None
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        if fuse_flag:
            first_frame_latents = input_latents[:, :, 0:1]
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "image_is_pad": image_is_pad,
        }

    def training_loss(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> tuple[torch.Tensor, dict[str, float]]:
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        noise = torch.randn_like(input_latents)
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise, timestep)
        target = self.train_video_scheduler.training_target(
            input_latents, noise, timestep
        )
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=0,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": self._video_context_payload(video_pre)},
            t_mod_all={"video": video_pre["t_mod"]},
        )
        pred = self.video_expert.post_dit(tokens_out["video"], video_pre)
        include_initial_video_step = inputs["first_frame_latents"] is None
        if not include_initial_video_step:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred,
            target_video=target,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=include_initial_video_step,
        )
        sample_weight = self.train_video_scheduler.training_weight(timestep).to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss_video = (loss_per_sample * sample_weight).mean()
        loss_total = self.loss_lambda_video * loss_video
        return (loss_total, {"loss_video": float(loss_total.detach().item())})


def create_prismwam_sv(
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    skip_dit_load_from_pretrain: bool = False,
    vae_checkpoint_path: Optional[str] = None,
    vae_variant: str = "depth",
    depth_vae_require_latent_stats: bool = False,
    video_scheduler=None,
    loss=None,
    video_use_text_embedding: bool = True,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> PrismWAMSV:
    """Hydra factory for the Stage-1 depth video model."""
    from omegaconf import DictConfig, OmegaConf

    def _plain_dict(value, *, name: str, default: Optional[dict] = None) -> dict:
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            value = {} if default is None else dict(default)
        if not isinstance(value, dict):
            raise TypeError(f"`{name}` must resolve to a dict, got {type(value)}.")
        return value

    video_dit_config = _plain_dict(video_dit_config, name="video_dit_config")
    video_scheduler = _plain_dict(video_scheduler, name="video_scheduler")
    loss = _plain_dict(loss, name="loss")
    return PrismWAMSV.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        vae_checkpoint_path=vae_checkpoint_path,
        vae_variant=str(vae_variant),
        depth_vae_require_latent_stats=bool(depth_vae_require_latent_stats),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        video_use_text_embedding=bool(video_use_text_embedding),
    )
