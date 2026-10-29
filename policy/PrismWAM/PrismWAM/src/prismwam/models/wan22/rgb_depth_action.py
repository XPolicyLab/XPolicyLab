"""PrismWAM Stage 2: RGB, depth and action experts with directed attention.

RGB and depth share one Wan transformer block stack (separate input/output
layers); actions use their own transformer. Attention masks use a
``[query, key]`` layout while names follow the information flow: ``RGB ->
Depth`` means depth queries read RGB keys/values, and ``Depth -> Action`` means
action queries read depth keys/values. There is no reverse edge and no direct
``RGB -> Action`` edge.

RGB conditions on its observed history and predicts one goal frame 32 steps
ahead (``f8``). Depth reads only the clean RGB history, so at inference the RGB
history is encoded once, its per-layer keys/values are cached, and only depth
and action are denoised.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .prismwam import PrismWAM
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)

RGB_TARGET_F8 = "f8"
# One RGB goal frame at the end of the clip: 4 or 8 frames (16 or 32 steps) ahead.
RGB_ENDPOINT_TARGET_MODES = ("f4", RGB_TARGET_F8)
RGB_DEPTH_ACTION_CHAIN = "rgb_to_depth_to_action"
ATTENTION_EDGES = ("Depth_query<-RGB_KV", "Action_query<-Depth_KV")
# RGB VAE latent channels and spatial stride (Wan2.2 VAE).
RGB_VAE_LATENT_CHANNELS = 48
RGB_VAE_SPATIAL_STRIDE = 16


def normalize_rgb_target_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in RGB_ENDPOINT_TARGET_MODES:
        raise ValueError(
            f"`rgb_target_mode` must be one of {RGB_ENDPOINT_TARGET_MODES}, got {value!r}."
        )
    return normalized


def normalize_rgb_depth_attention_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != RGB_DEPTH_ACTION_CHAIN:
        raise ValueError(
            f"`rgb_depth_attention_mode` must be {RGB_DEPTH_ACTION_CHAIN!r}, got {value!r}."
        )
    return normalized


@torch.no_grad()
def build_rgb_depth_action_attention_mask(
    *,
    rgb_self_mask: torch.Tensor,
    depth_self_mask: torch.Tensor,
    action_seq_len: int,
    rgb_condition_seq_len: int,
) -> torch.Tensor:
    """Build the directed three-stream mask in ``[query, key]`` layout.

    Depth queries read only the first ``rgb_condition_seq_len`` RGB tokens (the
    clean observed history), never the predicted RGB goal frame. Action
    queries read depth and action tokens.
    """
    for name, value in (
        ("rgb_self_mask", rgb_self_mask),
        ("depth_self_mask", depth_self_mask),
    ):
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(
                f"`{name}` must be square [S,S], got {tuple(value.shape)}."
            )
        if value.dtype != torch.bool:
            raise TypeError(f"`{name}` must be bool, got {value.dtype}.")
    if rgb_self_mask.device != depth_self_mask.device:
        raise ValueError("RGB and Depth self masks must be on the same device.")
    action_seq_len = int(action_seq_len)
    if action_seq_len <= 0:
        raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}.")
    rgb_len = int(rgb_self_mask.shape[0])
    depth_len = int(depth_self_mask.shape[0])
    rgb_condition_seq_len = int(rgb_condition_seq_len)
    if not 0 < rgb_condition_seq_len <= rgb_len:
        raise ValueError(
            f"`rgb_condition_seq_len` must be in [1,rgb_seq_len], got {rgb_condition_seq_len} for rgb_seq_len={rgb_len}."
        )
    rgb_end = rgb_len
    depth_end = rgb_len + depth_len
    total = depth_end + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=rgb_self_mask.device)
    mask[:rgb_end, :rgb_end] = rgb_self_mask
    mask[rgb_end:depth_end, rgb_end:depth_end] = depth_self_mask
    mask[depth_end:, depth_end:] = True
    mask[rgb_end:depth_end, :rgb_condition_seq_len] = True
    mask[depth_end:, rgb_end:depth_end] = True
    return mask


class PrismWAMRGBDepthAction(PrismWAM):
    """Shared RGB/Depth streams and an action expert with directed attention."""

    CHECKPOINT_FORMAT = "prismwam_rgb_depth_action_mot_v2_visual_to_action"

    def __init__(
        self,
        *,
        rgb_expert: nn.Module,
        rgb_vae: nn.Module,
        rgb_target_mode: str = RGB_TARGET_F8,
        rgb_depth_attention_mode: str = RGB_DEPTH_ACTION_CHAIN,
        share_rgb_depth_blocks: bool = True,
        rgb_fast: bool = True,
        rgb_history_num_frames: int = 1,
        rgb_history_frame_stride: int = 4,
        rgb_input_size: Optional[Sequence[int]] = None,
        architecture_name: str = "PrismWAM",
        rgb_train_shift: float = 5.0,
        rgb_num_train_timesteps: int = 1000,
        loss_lambda_rgb: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        expected_order = ["rgb", "video", "action"]
        if list(self.mot.expert_order) != expected_order:
            raise ValueError(
                f"Three-tower expert order must be {expected_order}, got {list(self.mot.expert_order)}."
            )
        if self.mot.mixtures["rgb"] is not rgb_expert:
            raise ValueError("`rgb_expert` must be the MoT `rgb` expert.")
        self.rgb_expert = rgb_expert
        self.rgb_vae = rgb_vae
        self.architecture_name = str(architecture_name)
        self.rgb_target_mode = normalize_rgb_target_mode(rgb_target_mode)
        self.rgb_depth_attention_mode = normalize_rgb_depth_attention_mode(
            rgb_depth_attention_mode
        )
        if not rgb_fast:
            raise ValueError(
                "`rgb_fast` must be true: depth reads only the clean RGB history."
            )
        if (
            str(getattr(self.rgb_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "The RGB expert requires `video_attention_mask_mode=first_frame_causal`."
            )
        self.rgb_history_num_frames = int(rgb_history_num_frames)
        self.rgb_history_frame_stride = int(rgb_history_frame_stride)
        rgb_temporal_factor = int(
            getattr(self.rgb_vae, "temporal_downsample_factor", 4)
        )
        if (
            self.rgb_history_num_frames <= 0
            or (self.rgb_history_num_frames - 1) % rgb_temporal_factor != 0
        ):
            raise ValueError(
                f"`rgb_history_num_frames` must satisfy T = 4n + 1 for the causal RGB VAE, got T={self.rgb_history_num_frames}, factor={rgb_temporal_factor}."
            )
        if self.rgb_history_frame_stride <= 0:
            raise ValueError("`rgb_history_frame_stride` must be positive.")
        self.rgb_condition_latent_frames = (
            self.rgb_history_num_frames - 1
        ) // rgb_temporal_factor + 1
        # Depth conditions on the current frame only.
        self.depth_condition_latent_frames = 1
        if (
            not share_rgb_depth_blocks
            or self.mot.mixtures["rgb"].blocks is not self.mot.mixtures["video"].blocks
        ):
            raise ValueError(
                "PrismWAM requires RGB and Depth to share one transformer block stack."
            )
        self.share_rgb_depth_blocks = True
        self.rgb_input_size = None
        if rgb_input_size is not None:
            if len(rgb_input_size) != 2:
                raise ValueError(
                    f"`rgb_input_size` must contain [height,width], got {rgb_input_size}."
                )
            self.rgb_input_size = tuple((int(value) for value in rgb_input_size))
            if any((value <= 0 for value in self.rgb_input_size)):
                raise ValueError("`rgb_input_size` values must be positive.")
            spatial_factor = int(getattr(self.rgb_vae, "upsampling_factor", 1))
            (patch_h, patch_w) = (
                int(value) for value in self.rgb_expert.patch_size[-2:]
            )
            required = (spatial_factor * patch_h, spatial_factor * patch_w)
            if any(
                (
                    size % factor != 0
                    for (size, factor) in zip(self.rgb_input_size, required)
                )
            ):
                raise ValueError(
                    f"`rgb_input_size` must be divisible by RGB VAE stride times DiT patch size {required}, got {self.rgb_input_size}."
                )
        self.loss_lambda_rgb = float(loss_lambda_rgb)
        if self.loss_lambda_rgb <= 0.0:
            raise ValueError("`loss_lambda_rgb` must be positive.")
        self.train_rgb_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(rgb_num_train_timesteps),
            shift=float(rgb_train_shift),
        )
        self.rgb_vae.eval().requires_grad_(False)
        self.rgb_vae.to(device=self.device, dtype=self.torch_dtype)

    @staticmethod
    def _validate_expert_compatibility(
        rgb_expert: nn.Module, depth_expert: nn.Module, action_expert: nn.Module
    ) -> None:
        reference = (len(depth_expert.blocks), int(depth_expert.attn_head_dim))
        for name, expert in (("rgb", rgb_expert), ("action", action_expert)):
            actual = (len(expert.blocks), int(expert.attn_head_dim))
            if actual != reference:
                raise ValueError(
                    f"{name} expert layer/head contract {actual} does not match Depth expert {reference}."
                )
        for name, expert in (("RGB", rgb_expert), ("Action", action_expert)):
            if int(expert.num_heads) != int(depth_expert.num_heads):
                raise ValueError(
                    f"{name} and Depth experts must share the attention head count, got {expert.num_heads} and {depth_expert.num_heads}."
                )

    @staticmethod
    def _validate_rgb_vae_contract(
        *, rgb_vae: nn.Module, rgb_expert: nn.Module
    ) -> None:
        expected = (RGB_VAE_LATENT_CHANNELS, RGB_VAE_SPATIAL_STRIDE)
        actual = (
            int(getattr(rgb_vae, "z_dim", -1)),
            int(getattr(rgb_vae, "upsampling_factor", -1)),
        )
        if actual != expected:
            raise ValueError(
                f"The RGB VAE must be the Wan2.2 VAE with latent channels/stride {expected}, got {actual}."
            )
        patch_volume = math.prod((int(value) for value in rgb_expert.patch_size))
        output_projection = getattr(getattr(rgb_expert, "head", None), "head", None)
        output_rows = int(getattr(output_projection, "out_features", -1))
        output_channels = (
            output_rows // patch_volume
            if patch_volume > 0 and output_rows % patch_volume == 0
            else -1
        )
        expert_io = (int(getattr(rgb_expert, "in_dim", -1)), output_channels)
        if expert_io != (RGB_VAE_LATENT_CHANNELS, RGB_VAE_LATENT_CHANNELS):
            raise ValueError(
                f"RGB expert I/O {expert_io} does not match VAE latent channels {RGB_VAE_LATENT_CHANNELS}."
            )

    @staticmethod
    def _share_visual_blocks(*, rgb_expert: nn.Module, depth_expert: nn.Module) -> None:
        """Make RGB and Depth use one depth-initialized WAN block stack.

        The modality-specific patch/text/time projections and output heads stay
        independent. Only the transformer blocks are shared, so RGB and Depth
        keep separate inputs, timesteps, and predictions while updating one set
        of visual backbone parameters.
        """
        if rgb_expert.blocks is depth_expert.blocks:
            return
        rgb_expert.blocks = depth_expert.blocks

    @staticmethod
    def _load_depth_parent(depth_expert: nn.Module, checkpoint_path: str | Path) -> int:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")
        payload = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=False
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
            raise ValueError(
                f"Stage-1 checkpoint must contain a `mot` state dictionary: {checkpoint_path}."
            )
        prefix = "mixtures.video."
        source = {
            key[len(prefix) :]: value
            for (key, value) in payload["mot"].items()
            if key.startswith(prefix)
        }
        expected = set(depth_expert.state_dict())
        provided = set(source)
        if expected != provided:
            raise ValueError(
                f"Stage-1 checkpoint does not exactly match the active Depth expert: missing={sorted(expected - provided)[:20]}, unexpected={sorted(provided - expected)[:20]}."
            )
        depth_expert.load_state_dict(source, strict=True)
        step = int(payload.get("step", -1))
        if step < 0:
            raise ValueError(
                f"Stage-1 checkpoint has no valid step: {checkpoint_path}."
            )
        logger.info(
            "Initialized Depth expert from %s at step %d.", checkpoint_path, step
        )
        return step

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
        depth_dit_config: dict[str, Any],
        rgb_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None,
        depth_vae_checkpoint_path: str,
        rgb_vae_checkpoint_path: str,
        depth_vae_require_latent_stats: bool = True,
        depth_dit_checkpoint_path: str | None = None,
        skip_depth_dit_load_from_pretrain: bool = False,
        skip_rgb_dit_load_from_pretrain: bool = False,
        skip_action_dit_load_from_pretrain: bool = False,
        checkpoint_init_only: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        proprio_dim: int | None = None,
        rgb_target_mode: str = RGB_TARGET_F8,
        rgb_depth_attention_mode: str = RGB_DEPTH_ACTION_CHAIN,
        share_rgb_depth_blocks: bool = True,
        rgb_fast: bool = True,
        rgb_history_num_frames: int = 1,
        rgb_history_frame_stride: int = 4,
        rgb_input_size: Optional[Sequence[int]] = None,
        architecture_name: str = "PrismWAM",
        video_use_text_embedding: bool = True,
        rgb_scheduler: Optional[dict[str, Any]] = None,
        depth_scheduler: Optional[dict[str, Any]] = None,
        action_scheduler: Optional[dict[str, Any]] = None,
        loss: Optional[dict[str, Any]] = None,
    ) -> "PrismWAMRGBDepthAction":
        for name, config in (
            ("depth_dit_config", depth_dit_config),
            ("rgb_dit_config", rgb_dit_config),
            ("action_dit_config", action_dit_config),
        ):
            if not isinstance(config, dict):
                raise TypeError(f"`{name}` must be a dict.")
        if checkpoint_init_only:
            if depth_dit_checkpoint_path is not None:
                raise ValueError(
                    "`checkpoint_init_only=true` requires `depth_dit_checkpoint_path=null`; the final policy checkpoint owns every RGB/Depth/Action expert weight."
                )
            if not (
                skip_depth_dit_load_from_pretrain
                and skip_rgb_dit_load_from_pretrain
                and skip_action_dit_load_from_pretrain
            ):
                raise ValueError(
                    "`checkpoint_init_only=true` requires all three pretrained expert loads to be skipped."
                )
        elif depth_dit_checkpoint_path is not None and (
            not skip_depth_dit_load_from_pretrain
        ):
            raise ValueError(
                "A Stage-1 checkpoint requires `skip_depth_dit_load_from_pretrain=true` to avoid a redundant upstream-WAN load."
            )
        if (
            not checkpoint_init_only
            and depth_dit_checkpoint_path is None
            and skip_depth_dit_load_from_pretrain
        ):
            raise ValueError(
                "Random Depth initialization is not supported here; provide `depth_dit_checkpoint_path` or enable upstream-WAN loading."
            )
        if not share_rgb_depth_blocks:
            raise ValueError("Independent visual towers are not supported.")
        depth_components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=int(tokenizer_max_len),
            redirect_common_files=bool(redirect_common_files),
            dit_config=depth_dit_config,
            skip_dit_load_from_pretrain=bool(skip_depth_dit_load_from_pretrain),
            load_text_encoder=bool(load_text_encoder),
            vae_checkpoint_path=depth_vae_checkpoint_path,
            vae_variant="depth",
            depth_vae_require_latent_stats=bool(depth_vae_require_latent_stats),
        )
        depth_parent_step = None
        if depth_dit_checkpoint_path is not None:
            depth_parent_step = cls._load_depth_parent(
                depth_components.dit, depth_dit_checkpoint_path
            )
        rgb_components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=int(tokenizer_max_len),
            redirect_common_files=bool(redirect_common_files),
            dit_config=rgb_dit_config,
            skip_dit_load_from_pretrain=bool(skip_rgb_dit_load_from_pretrain),
            load_text_encoder=False,
            vae_checkpoint_path=rgb_vae_checkpoint_path,
            vae_variant="rgb",
            depth_vae_require_latent_stats=False,
        )
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
            device=device,
            torch_dtype=torch_dtype,
        )
        cls._validate_expert_compatibility(
            rgb_components.dit, depth_components.dit, action_expert
        )
        cls._share_visual_blocks(
            rgb_expert=rgb_components.dit, depth_expert=depth_components.dit
        )
        logger.info(
            "RGB and Depth share one depth-initialized transformer block stack."
        )
        cls._validate_rgb_vae_contract(
            rgb_vae=rgb_components.vae, rgb_expert=rgb_components.dit
        )
        mot = MoT(
            mixtures={
                "rgb": rgb_components.dit,
                "video": depth_components.dit,
                "action": action_expert,
            },
            mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        )
        rgb_scheduler = dict(rgb_scheduler or {})
        depth_scheduler = dict(depth_scheduler or {})
        action_scheduler = dict(action_scheduler or {})
        loss = dict(loss or {})
        model = cls(
            rgb_expert=rgb_components.dit,
            rgb_vae=rgb_components.vae,
            rgb_target_mode=rgb_target_mode,
            rgb_depth_attention_mode=rgb_depth_attention_mode,
            share_rgb_depth_blocks=share_rgb_depth_blocks,
            rgb_fast=rgb_fast,
            rgb_history_num_frames=rgb_history_num_frames,
            rgb_history_frame_stride=rgb_history_frame_stride,
            rgb_input_size=rgb_input_size,
            architecture_name=architecture_name,
            rgb_train_shift=float(rgb_scheduler.get("train_shift", 5.0)),
            rgb_num_train_timesteps=int(rgb_scheduler.get("num_train_timesteps", 1000)),
            loss_lambda_rgb=float(loss.get("lambda_rgb", 1.0)),
            video_expert=depth_components.dit,
            action_expert=action_expert,
            mot=mot,
            vae=depth_components.vae,
            text_encoder=depth_components.text_encoder,
            tokenizer=depth_components.tokenizer,
            text_dim=int(depth_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=float(depth_scheduler.get("train_shift", 5.0)),
            video_infer_shift=float(depth_scheduler.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(
                depth_scheduler.get("num_train_timesteps", 1000)
            ),
            action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
            action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
            action_num_train_timesteps=int(
                action_scheduler.get("num_train_timesteps", 1000)
            ),
            loss_lambda_video=float(loss.get("lambda_depth", 1.0)),
            loss_lambda_action=float(loss.get("lambda_action", 1.0)),
            video_use_text_embedding=bool(video_use_text_embedding),
        )
        model.depth_parent_step = depth_parent_step
        model.model_paths = {
            "rgb_dit": rgb_components.dit_path,
            "depth_dit": depth_components.dit_path
            if depth_dit_checkpoint_path is None
            else str(Path(depth_dit_checkpoint_path).expanduser().resolve()),
            "rgb_vae": rgb_components.vae_path,
            "depth_vae": depth_components.vae_path,
            "text_encoder": depth_components.text_encoder_path,
            "tokenizer": depth_components.tokenizer_path,
            "action_dit_backbone": action_dit_pretrained_path,
        }
        return model

    def _resize_rgb_spatial(self, value: torch.Tensor) -> torch.Tensor:
        if (
            self.rgb_input_size is None
            or tuple(value.shape[-2:]) == self.rgb_input_size
        ):
            return value
        if value.ndim == 4:
            return F.interpolate(
                value, size=self.rgb_input_size, mode="bilinear", align_corners=False
            )
        if value.ndim != 5:
            raise ValueError(
                f"RGB resize expects [B,C,H,W] or [B,C,T,H,W], got {tuple(value.shape)}."
            )
        (batch, channels, frames, height, width) = value.shape
        flattened = value.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        resized = F.interpolate(
            flattened, size=self.rgb_input_size, mode="bilinear", align_corners=False
        )
        return (
            resized.reshape(batch, frames, channels, *self.rgb_input_size)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @torch.no_grad()
    def _encode_rgb_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``[history..., goal]`` RGB frames into condition + goal latents."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"`rgb_video` must be [B,3,T,H,W], got {tuple(video.shape)}."
            )

        def encode(value: torch.Tensor) -> torch.Tensor:
            latents = self.rgb_vae.encode(value, device=self.device, tiled=False)
            if isinstance(latents, list):
                latents = torch.stack(latents, dim=0)
            if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
                raise ValueError("RGB VAE must return [B,C,T,H,W] latents.")
            return latents

        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        video = self._resize_rgb_spatial(video)
        if int(video.shape[2]) < self.rgb_history_num_frames + 1:
            raise ValueError(
                f"RGB input must contain the configured history followed by the goal frame; expected at least {self.rgb_history_num_frames + 1} raw frames, got {video.shape[2]}."
            )
        first = encode(video[:, :, : self.rgb_history_num_frames])
        final = encode(video[:, :, -1:])
        if int(first.shape[2]) != self.rgb_condition_latent_frames:
            raise ValueError(
                f"RGB history encoded to the wrong number of condition latents: expected {self.rgb_condition_latent_frames}, got {first.shape[2]}."
            )
        if int(final.shape[2]) != 1:
            raise ValueError(
                f"The isolated RGB goal frame must encode to one latent frame, got {final.shape[2]}."
            )
        return torch.cat((first, final), dim=2)

    def build_inputs(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> dict[str, Any]:
        inputs = super().build_inputs(sample, tiled=tiled)
        rgb_video = sample.get("rgb_video")
        rgb_video_latents = sample.get("rgb_video_latents")
        if (rgb_video is None) == (rgb_video_latents is None):
            raise ValueError(
                "Provide exactly one of `sample['rgb_video']` or `sample['rgb_video_latents']`."
            )
        if rgb_video is not None:
            rgb_latents = self._encode_rgb_video_latents(rgb_video)
        else:
            if (
                not isinstance(rgb_video_latents, torch.Tensor)
                or rgb_video_latents.ndim != 5
            ):
                raise ValueError("`rgb_video_latents` must be [B,C,T,H,W].")
            rgb_latents = rgb_video_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        expected_t = self.rgb_condition_latent_frames + 1
        if rgb_latents.shape[2] != expected_t:
            raise ValueError(
                f"RGB latents must have T={expected_t} (history + goal), got {rgb_latents.shape[2]}."
            )
        expected_rgb_channels = int(getattr(self.rgb_expert, "in_dim", -1))
        if int(rgb_latents.shape[1]) != expected_rgb_channels:
            raise ValueError(
                f"RGB latent channel count does not match the RGB expert: {rgb_latents.shape[1]} vs {expected_rgb_channels}."
            )
        target_valid = sample.get("rgb_target_is_valid")
        if target_valid is None:
            raise ValueError("Training requires `rgb_target_is_valid`.")
        if not isinstance(target_valid, torch.Tensor):
            raise TypeError("`rgb_target_is_valid` must be a tensor.")
        target_valid = target_valid.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        ).reshape(-1)
        if target_valid.shape[0] != rgb_latents.shape[0]:
            raise ValueError("RGB goal validity batch dimension mismatch.")
        if any(
            sample.get(key) is not None
            for key in (
                "depth_history_latents",
                "depth_history_video",
                "depth_history_is_valid",
            )
        ):
            raise ValueError(
                "Depth conditions on the current frame only; depth history is not supported."
            )
        inputs.update(rgb_latents=rgb_latents, rgb_target_is_valid=target_valid)
        return inputs

    def _build_stream_attention_mask(
        self,
        *,
        rgb_pre: dict[str, Any],
        depth_pre: dict[str, Any],
        action_seq_len: int,
    ) -> torch.Tensor:
        rgb_tokens = rgb_pre["tokens"]
        depth_tokens = depth_pre["tokens"]
        rgb_tokens_per_frame = int(rgb_pre["meta"]["tokens_per_frame"])
        rgb_self = self.rgb_expert.build_video_to_video_mask(
            video_seq_len=int(rgb_tokens.shape[1]),
            video_tokens_per_frame=rgb_tokens_per_frame,
            device=rgb_tokens.device,
            num_condition_frames=self.rgb_condition_latent_frames,
        )
        depth_self = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(depth_tokens.shape[1]),
            video_tokens_per_frame=int(depth_pre["meta"]["tokens_per_frame"]),
            device=depth_tokens.device,
            num_condition_frames=self.depth_condition_latent_frames,
        )
        return build_rgb_depth_action_attention_mask(
            rgb_self_mask=rgb_self,
            depth_self_mask=depth_self,
            action_seq_len=action_seq_len,
            rgb_condition_seq_len=self.rgb_condition_latent_frames
            * rgb_tokens_per_frame,
        )

    def _predict_three_noise(
        self,
        *,
        rgb_latents: torch.Tensor,
        depth_latents: torch.Tensor,
        action_latents: torch.Tensor,
        timestep_rgb: torch.Tensor,
        timestep_depth: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_pre = self.rgb_expert.pre_dit(
            x=rgb_latents,
            timestep=timestep_rgb,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            num_condition_frames=self.rgb_condition_latent_frames,
        )
        depth_pre = self.video_expert.pre_dit(
            x=depth_latents,
            timestep=timestep_depth,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            num_condition_frames=self.depth_condition_latent_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_stream_attention_mask(
            rgb_pre=rgb_pre,
            depth_pre=depth_pre,
            action_seq_len=int(action_pre["tokens"].shape[1]),
        )
        outputs = self.mot(
            embeds_all={
                "rgb": rgb_pre["tokens"],
                "video": depth_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "rgb": rgb_pre["freqs"],
                "video": depth_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "rgb": self._video_context_payload(rgb_pre),
                "video": self._video_context_payload(depth_pre),
                "action": self._text_context_payload(
                    action_pre["context"], action_pre["context_mask"]
                ),
            },
            t_mod_all={
                "rgb": rgb_pre["t_mod"],
                "video": depth_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return (
            self.rgb_expert.post_dit(outputs["rgb"], rgb_pre),
            self.video_expert.post_dit(outputs["video"], depth_pre),
            self.action_expert.post_dit(outputs["action"], action_pre),
        )

    @torch.no_grad()
    def _prefill_clean_rgb_cache(
        self,
        *,
        condition_rgb: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Forward clean RGB-history latents once and cache their layer K/V."""
        timestep_rgb = torch.zeros(
            (int(condition_rgb.shape[0]),),
            device=self.device,
            dtype=condition_rgb.dtype,
        )
        rgb_pre = self.rgb_expert.pre_dit(
            x=condition_rgb,
            timestep=timestep_rgb,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            num_condition_frames=self.rgb_condition_latent_frames,
        )
        rgb_seq_len = int(rgb_pre["tokens"].shape[1])
        rgb_tokens_per_frame = int(rgb_pre["meta"]["tokens_per_frame"])
        expected_seq_len = self.rgb_condition_latent_frames * rgb_tokens_per_frame
        if rgb_seq_len != expected_seq_len:
            raise RuntimeError(
                f"Inference must prefill exactly the configured RGB condition prefix: tokens={rgb_seq_len}, expected={expected_seq_len}."
            )
        rgb_self_mask = self.rgb_expert.build_video_to_video_mask(
            video_seq_len=rgb_seq_len,
            video_tokens_per_frame=rgb_tokens_per_frame,
            device=rgb_pre["tokens"].device,
            num_condition_frames=self.rgb_condition_latent_frames,
        )
        rgb_kv_cache = self.mot.prefill_prefix_expert_cache(
            expert_name="rgb",
            tokens=rgb_pre["tokens"],
            freqs=rgb_pre["freqs"],
            t_mod=rgb_pre["t_mod"],
            context_payload=self._video_context_payload(rgb_pre),
            attention_mask=rgb_self_mask,
        )
        return (rgb_pre, rgb_kv_cache)

    @torch.no_grad()
    def _predict_depth_action_noise_with_rgb_cache(
        self,
        *,
        rgb_pre: dict[str, Any],
        rgb_kv_cache: list[dict[str, Any]],
        depth_latents: torch.Tensor,
        action_latents: torch.Tensor,
        timestep_depth: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise Depth+Action while reading cached clean-RGB layer K/V."""
        depth_pre = self.video_expert.pre_dit(
            x=depth_latents,
            timestep=timestep_depth,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            num_condition_frames=self.depth_condition_latent_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_stream_attention_mask(
            rgb_pre=rgb_pre,
            depth_pre=depth_pre,
            action_seq_len=int(action_pre["tokens"].shape[1]),
        )
        outputs = self.mot.forward_with_prefix_expert_cache(
            prefix_expert_name="rgb",
            prefix_kv_cache=rgb_kv_cache,
            embeds_all={"video": depth_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": depth_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": self._video_context_payload(depth_pre),
                "action": self._text_context_payload(
                    action_pre["context"], action_pre["context_mask"]
                ),
            },
            t_mod_all={"video": depth_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.video_expert.post_dit(outputs["video"], depth_pre),
            self.action_expert.post_dit(outputs["action"], action_pre),
        )

    @staticmethod
    def _action_loss_per_sample(
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_dim_mask: Optional[torch.Tensor],
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_channel = F.mse_loss(prediction.float(), target.float(), reduction="none")
        if action_dim_mask is not None:
            valid_channel = action_dim_mask.to(
                device=loss_channel.device, dtype=loss_channel.dtype
            ).unsqueeze(1)
            loss_token = (loss_channel * valid_channel).sum(dim=2) / valid_channel.sum(
                dim=2
            ).clamp(min=1.0)
        else:
            loss_token = loss_channel.mean(dim=2)
        if action_is_pad is None:
            return loss_token.mean(dim=1)
        valid = (~action_is_pad).to(device=loss_token.device, dtype=loss_token.dtype)
        return (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _sample_training_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
        rgb_dtype: torch.dtype,
        depth_dtype: torch.dtype,
        action_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an independent diffusion time for each stream."""
        timestep_rgb = self.train_rgb_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=rgb_dtype
        )
        timestep_depth = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=depth_dtype
        )
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=action_dtype
        )
        return (timestep_rgb, timestep_depth, timestep_action)

    def training_loss_with_components(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        """Compute the joint objective and keep each weighted stream loss.

        Returns ``(total, metrics, components)``: ``metrics`` holds Python
        floats for logging and ``components`` the weighted scalar tensors.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        clean_rgb = inputs["rgb_latents"]
        clean_depth = inputs["input_latents"]
        clean_action = inputs["action"]
        batch_size = int(clean_depth.shape[0])
        noise_rgb = torch.randn_like(clean_rgb)
        noise_depth = torch.randn_like(clean_depth)
        noise_action = torch.randn_like(clean_action)
        if inputs["action_dim_mask"] is not None:
            noise_action = noise_action.masked_fill(
                ~inputs["action_dim_mask"].unsqueeze(1), 0.0
            )
        (timestep_rgb, timestep_depth, timestep_action) = (
            self._sample_training_timesteps(
                batch_size=batch_size,
                device=self.device,
                rgb_dtype=clean_rgb.dtype,
                depth_dtype=clean_depth.dtype,
                action_dtype=clean_action.dtype,
            )
        )
        noisy_rgb = self.train_rgb_scheduler.add_noise(
            clean_rgb, noise_rgb, timestep_rgb
        )
        noisy_depth = self.train_video_scheduler.add_noise(
            clean_depth, noise_depth, timestep_depth
        )
        noisy_action = self.train_action_scheduler.add_noise(
            clean_action, noise_action, timestep_action
        )
        target_rgb = self.train_rgb_scheduler.training_target(
            clean_rgb, noise_rgb, timestep_rgb
        )
        target_depth = self.train_video_scheduler.training_target(
            clean_depth, noise_depth, timestep_depth
        )
        target_action = self.train_action_scheduler.training_target(
            clean_action, noise_action, timestep_action
        )
        rgb_condition_frames = self.rgb_condition_latent_frames
        depth_condition_frames = self.depth_condition_latent_frames
        (rgb_condition, rgb_condition_noise) = self._mix_clean_conditioning_latents(
            clean_rgb[:, :, :rgb_condition_frames], stream="rgb"
        )
        (depth_condition, depth_condition_noise) = self._mix_clean_conditioning_latents(
            clean_depth[:, :, :depth_condition_frames], stream="depth"
        )
        noisy_rgb[:, :, :rgb_condition_frames] = rgb_condition
        noisy_depth[:, :, :depth_condition_frames] = depth_condition
        (pred_rgb, pred_depth, pred_action) = self._predict_three_noise(
            rgb_latents=noisy_rgb,
            depth_latents=noisy_depth,
            action_latents=noisy_action,
            timestep_rgb=timestep_rgb,
            timestep_depth=timestep_depth,
            timestep_action=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        rgb_loss_steps = F.mse_loss(
            pred_rgb[:, :, rgb_condition_frames:].float(),
            target_rgb[:, :, rgb_condition_frames:].float(),
            reduction="none",
        ).mean(dim=(1, 3, 4))
        valid_rgb = inputs["rgb_target_is_valid"].to(
            device=rgb_loss_steps.device, dtype=rgb_loss_steps.dtype
        )
        rgb_loss_per_sample = rgb_loss_steps.mean(dim=1)
        weighted_rgb = rgb_loss_per_sample * self.train_rgb_scheduler.training_weight(
            timestep_rgb
        ).to(device=rgb_loss_per_sample.device, dtype=rgb_loss_per_sample.dtype)
        loss_rgb = (weighted_rgb * valid_rgb).sum() / valid_rgb.sum().clamp(min=1.0)
        depth_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_depth[:, :, depth_condition_frames:],
            target_video=target_depth[:, :, depth_condition_frames:],
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        loss_depth = (
            depth_loss_per_sample
            * self.train_video_scheduler.training_weight(timestep_depth).to(
                device=depth_loss_per_sample.device, dtype=depth_loss_per_sample.dtype
            )
        ).mean()
        action_loss_per_sample = self._action_loss_per_sample(
            prediction=pred_action,
            target=target_action,
            action_dim_mask=inputs["action_dim_mask"],
            action_is_pad=inputs["action_is_pad"],
        )
        loss_action = (
            action_loss_per_sample
            * self.train_action_scheduler.training_weight(timestep_action).to(
                device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
        ).mean()
        components = {
            "rgb": self.loss_lambda_rgb * loss_rgb,
            "depth": self.loss_lambda_video * loss_depth,
            "action": self.loss_lambda_action * loss_action,
        }
        total = sum(components.values())
        metrics = {
            "loss_rgb": float(loss_rgb.detach()) * self.loss_lambda_rgb,
            "loss_video": float(components["depth"].detach()),
            "loss_action": float(components["action"].detach()),
            "conditioning/rgb_latent_noise_fraction": rgb_condition_noise["fraction"],
            "conditioning/rgb_latent_noise_ratio": rgb_condition_noise["mean_ratio"],
            "conditioning/depth_latent_noise_fraction": depth_condition_noise[
                "fraction"
            ],
            "conditioning/depth_latent_noise_ratio": depth_condition_noise[
                "mean_ratio"
            ],
        }
        return (total, metrics, components)

    def _mix_clean_conditioning_latents(
        self, latents: torch.Tensor, *, stream: str
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the clean conditioning latents unchanged.

        The sparse RGB memory model overrides this hook to add training-time
        noise to its observed latents.
        """
        del stream
        return (latents, {"fraction": 0.0, "mean_ratio": 0.0})

    def training_loss(
        self, sample: dict[str, Any], tiled: bool = False
    ) -> tuple[torch.Tensor, dict[str, float]]:
        (total, metrics, _) = self.training_loss_with_components(sample, tiled=tiled)
        return (total, metrics)

    @staticmethod
    def _validate_single_inference_image(
        image: torch.Tensor, *, expected_channels: int, name: str
    ) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if (
            image.ndim != 4
            or image.shape[0] != 1
            or image.shape[1] != int(expected_channels)
        ):
            raise ValueError(
                f"`{name}` must have shape [1,{expected_channels},H,W] or [{expected_channels},H,W], got {tuple(image.shape)}."
            )
        return image

    @torch.no_grad()
    def _encode_rgb_history_images_latents_tensor(
        self,
        rgb_history_images: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        if rgb_history_images.ndim == 4:
            rgb_history_images = rgb_history_images.unsqueeze(0)
        expected_raw_t = self.rgb_history_num_frames
        if (
            rgb_history_images.ndim != 5
            or int(rgb_history_images.shape[0]) != 1
            or int(rgb_history_images.shape[1])
            != int(getattr(self.rgb_vae, "input_channels", 3))
            or (int(rgb_history_images.shape[2]) != expected_raw_t)
        ):
            raise ValueError(
                f"`rgb_history_images` must be [1,3,{expected_raw_t},H,W] (or without batch), got {tuple(rgb_history_images.shape)}."
            )
        rgb_history_images = self._resize_rgb_spatial(
            rgb_history_images.to(device=self.device, dtype=self.torch_dtype)
        )
        history = rgb_history_images[0]
        encoded = self.rgb_vae.encode(
            [history],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(encoded, list):
            encoded = encoded[0].unsqueeze(0)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
            raise ValueError(
                "RGB VAE must return one [B,C,T,H,W] latent tensor for the RGB history."
            )
        expected_latent_t = (expected_raw_t - 1) // int(
            getattr(self.rgb_vae, "temporal_downsample_factor", 4)
        ) + 1
        if encoded.shape[0] != 1 or encoded.shape[2] != expected_latent_t:
            raise ValueError(
                f"RGB history encoded to the wrong temporal length: expected {expected_latent_t}, got {tuple(encoded.shape)}."
            )
        return encoded.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _infer_action_latents(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        rgb_input_image: torch.Tensor,
        rgb_history_images: Optional[torch.Tensor] = None,
        num_video_frames: int,
        action_horizon: int,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> torch.Tensor:
        """Cache the clean RGB history once, then jointly denoise depth and action."""
        self.eval()
        for name, expert in (("RGB", self.rgb_expert), ("Depth", self.video_expert)):
            if (
                str(getattr(expert, "video_attention_mask_mode", ""))
                != "first_frame_causal"
            ):
                raise ValueError(
                    f"{name} online inference requires `video_attention_mask_mode='first_frame_causal'`."
                )
        input_image = self._validate_single_inference_image(
            input_image,
            expected_channels=int(getattr(self.vae, "input_channels", 1)),
            name="input_image",
        )
        rgb_input_image = self._validate_single_inference_image(
            rgb_input_image,
            expected_channels=int(getattr(self.rgb_vae, "input_channels", 3)),
            name="rgb_input_image",
        )
        (height, width) = (int(value) for value in input_image.shape[-2:])
        (checked_h, checked_w, checked_t) = self._check_resize_height_width(
            height, width, int(num_video_frames)
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"Online observations must be resized to multiples of 16, got HxW=({height},{width})."
            )
        if checked_t != int(num_video_frames):
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}."
            )
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided while the proprio encoder is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or tuple(proprio.shape) != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.proprio_dim}] or [{self.proprio_dim}], got {tuple(proprio.shape)}."
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        rgb_input_image = rgb_input_image.to(device=self.device, dtype=self.torch_dtype)
        first_depth = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        ).to(dtype=self.torch_dtype)
        if rgb_history_images is None:
            if self.rgb_history_num_frames != 1:
                raise ValueError(
                    f"This checkpoint requires `rgb_history_images` with {self.rgb_history_num_frames} raw frames."
                )
            rgb_history_images = rgb_input_image.unsqueeze(2)
        condition_rgb = self._encode_rgb_history_images_latents_tensor(
            rgb_history_images, tiled=tiled
        )
        decode_depth_latent_t = (int(num_video_frames) - 1) // int(
            getattr(self.vae, "temporal_downsample_factor", 4)
        ) + 1
        depth_latent_t = self.depth_condition_latent_frames + decode_depth_latent_t - 1

        def generator(offset: int):
            if seed is None:
                return None
            return torch.Generator(device=rand_device).manual_seed(int(seed) + offset)

        latents_depth = torch.randn(
            (1, first_depth.shape[1], depth_latent_t, *first_depth.shape[-2:]),
            generator=generator(1),
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator(2),
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_depth[:, :, : self.depth_condition_latent_frames] = first_depth
        (context, context_mask) = self._resolve_inference_language_context(
            prompt=prompt, context=context, context_mask=context_mask
        )
        if proprio is not None:
            (context, context_mask) = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        (rgb_pre, rgb_kv_cache) = self._prefill_clean_rgb_cache(
            condition_rgb=condition_rgb, context=context, context_mask=context_mask
        )
        depth_schedule = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_depth.dtype,
            shift_override=sigma_shift,
        )
        action_schedule = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        if any(
            (
                len(times) != int(num_inference_steps)
                for schedule in (depth_schedule, action_schedule)
                for times in schedule
            )
        ):
            raise RuntimeError("Depth/Action inference schedules are misaligned.")
        for index in range(int(num_inference_steps)):
            timestep_depth = (
                depth_schedule[0][index]
                .reshape(1)
                .to(device=self.device, dtype=latents_depth.dtype)
            )
            timestep_action = (
                action_schedule[0][index]
                .reshape(1)
                .to(device=self.device, dtype=latents_action.dtype)
            )
            (pred_depth, pred_action) = self._predict_depth_action_noise_with_rgb_cache(
                rgb_pre=rgb_pre,
                rgb_kv_cache=rgb_kv_cache,
                depth_latents=latents_depth,
                action_latents=latents_action,
                timestep_depth=timestep_depth,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            latents_depth = self.infer_video_scheduler.step(
                pred_depth, depth_schedule[1][index], latents_depth
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, action_schedule[1][index], latents_action
            )
            latents_depth[:, :, : self.depth_condition_latent_frames] = first_depth
        return latents_action

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        rgb_input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        rgb_history_images: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        latents_action = self._infer_action_latents(
            prompt=prompt,
            input_image=input_image,
            rgb_input_image=rgb_input_image,
            rgb_history_images=rgb_history_images,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        }

    def _extra_checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "rgb_depth_action_config": {
                "architecture_name": self.architecture_name,
                "rgb_input_size": None
                if self.rgb_input_size is None
                else list(self.rgb_input_size),
                "action_dim": int(self.action_expert.action_dim),
                "proprio_dim": self.proprio_dim,
                "rgb_target_mode": self.rgb_target_mode,
                "rgb_depth_attention_mode": self.rgb_depth_attention_mode,
                "rgb_fast": True,
                "rgb_history_num_frames": self.rgb_history_num_frames,
                "rgb_history_frame_stride": self.rgb_history_frame_stride,
                "rgb_condition_latent_frames": self.rgb_condition_latent_frames,
                "depth_condition_latent_frames": self.depth_condition_latent_frames,
                "share_rgb_depth_blocks": self.share_rgb_depth_blocks,
                "shared_visual_backbone_initialization": "depth_parent",
                "loss_weights": {
                    "rgb": self.loss_lambda_rgb,
                    "depth": self.loss_lambda_video,
                    "action": self.loss_lambda_action,
                },
                "mask_layout": "query_key",
                "attention_semantics": "information_flow_visual_to_action_v2",
                "attention_edges": list(ATTENTION_EDGES),
                "expert_order": list(self.mot.expert_order),
                "depth_parent_step": getattr(self, "depth_parent_step", None),
            },
        }

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        if payload.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported RGB/Depth/Action checkpoint format: {payload.get('format')!r}."
            )
        config = dict(payload.get("rgb_depth_action_config", {}))
        expected = {
            "architecture_name": self.architecture_name,
            "rgb_input_size": None
            if self.rgb_input_size is None
            else list(self.rgb_input_size),
            "rgb_target_mode": self.rgb_target_mode,
            "rgb_depth_attention_mode": self.rgb_depth_attention_mode,
            "share_rgb_depth_blocks": self.share_rgb_depth_blocks,
            "loss_weights": {
                "rgb": self.loss_lambda_rgb,
                "depth": self.loss_lambda_video,
                "action": self.loss_lambda_action,
            },
            "mask_layout": "query_key",
            "attention_semantics": "information_flow_visual_to_action_v2",
            "expert_order": list(self.mot.expert_order),
            "rgb_fast": True,
            "rgb_history_num_frames": self.rgb_history_num_frames,
            "rgb_history_frame_stride": self.rgb_history_frame_stride,
        }
        mismatches = {
            key: {"expected": value, "actual": config.get(key)}
            for (key, value) in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"RGB/Depth/Action checkpoint contract mismatch: {mismatches}."
            )
        return payload
