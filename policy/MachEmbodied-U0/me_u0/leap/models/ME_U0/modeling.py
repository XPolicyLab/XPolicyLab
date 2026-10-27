"""Single ME_U0 model for every retained video/action training route.

Construction/loading, MoT experts, FlexAttention, RoPE, absolute latent
positions, and the Wan VAE stay on the official audited Lance path. Continuous
tokens are always routed through the generation expert, while observations are
always encoded by the ViT and inserted into the understanding stream.
"""

from __future__ import annotations

import os.path as osp
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from leap.data.world_unified.temporal_masks import masked_token_mse
from leap.data.world_unified.instruction_template import build_instruction_text_ids
from leap.data.world_unified.metadata import Metadata
from leap.models.base_model import BatchSpec, LeapVLABase, register_model
from leap.models.ME_U0.configuration import MachEmbodiedUnifiedConfig
from leap.models.ME_U0.lance_src.data.data_utils import patchify_video_with_merge
from leap.models.ME_U0.lance_src.data.data_utils import add_special_tokens
from leap.models.ME_U0.lance_src.modeling.lance.lance import Lance, LanceConfig
from leap.models.ME_U0.lance_src.modeling.lance.qwen2_navit import Qwen2ForCausalLM
from leap.models.ME_U0.lance_src.modeling.qwen2 import Qwen2Tokenizer
from leap.models.ME_U0.lance_src.modeling.qwen2.configuration_qwen2 import Qwen2Config


def _make_llm_config(
    lance_dir: str, cfg: MachEmbodiedUnifiedConfig
) -> Qwen2Config:
    llm_config = Qwen2Config.from_json_file(osp.join(lance_dir, "llm_config.json"))
    # transformers>=5.0 dropped pad_token_id from the base PretrainedConfig
    # attribute set; the bundled qwen2_navit implementation still reads it.
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = None
    llm_config.layer_module = cfg.layer_module
    llm_config.qk_norm = cfg.llm_qk_norm
    llm_config.qk_norm_und = cfg.llm_qk_norm_und
    llm_config.qk_norm_gen = cfg.llm_qk_norm_gen
    llm_config.tie_word_embeddings = cfg.tie_word_embeddings
    llm_config.freeze_und = cfg.freeze_understanding
    llm_config.apply_qwen_2_5_vl_pos_emb = cfg.apply_qwen_2_5_vl_pos_emb
    return llm_config


def _build_mach_embodied_unified_backbone(
    cfg: MachEmbodiedUnifiedConfig,
) -> Tuple[Lance, Any, Any, Any]:
    """Build and load the unchanged Lance backbone, Wan VAE, and tokenizer."""

    lance_dir = osp.join(cfg.checkpoint_root, cfg.lance_subdir)
    vit_dir = osp.join(cfg.checkpoint_root, cfg.vit_subdir)
    vae_pth = osp.join(cfg.checkpoint_root, cfg.vae_filename)

    llm_config = _make_llm_config(lance_dir, cfg)
    language_model = Qwen2ForCausalLM(llm_config)

    vit_model = None
    vit_config = None
    if cfg.visual_und:
        from leap.models.ME_U0.lance_src.modeling.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLVisionConfig,
        )
        from leap.models.ME_U0.lance_src.modeling.vit.qwen2_5_vl_vit import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(vit_dir)
        # The bundled ViT calls flash/flex attention directly; eager is used
        # only to pass transformers>=5.0 construction-time dispatch validation.
        vit_config._attn_implementation = "eager"
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)

    vae_model = None
    vae_config = None
    if cfg.visual_gen:
        from leap.models.ME_U0.lance_src.modeling.vae.wan.model import WanVideoVAE

        vae_model = WanVideoVAE(vae_pth=vae_pth)
        vae_config = deepcopy(vae_model.vae_config)

    lance_config = LanceConfig(
        visual_gen=cfg.visual_gen,
        visual_und=cfg.visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=cfg.latent_patch_size,
        max_num_frames=cfg.max_num_frames,
        max_latent_size=cfg.max_latent_size,
        vit_max_num_patch_per_side=cfg.vit_max_num_patch_per_side,
        connector_act=cfg.connector_act,
        interpolate_pos=cfg.interpolate_pos,
        timestep_shift=cfg.timestep_shift,
    )

    from leap.models.ME_U0.lance_src.config.config_factory import TrainingArguments

    training_args = TrainingArguments()
    training_args.apply_qwen_2_5_vl_pos_emb = cfg.apply_qwen_2_5_vl_pos_emb
    training_args.freeze_und = cfg.freeze_understanding

    model = Lance(
        language_model=language_model,
        vit_model=vit_model,
        vit_type=cfg.vit_type,
        config=lance_config,
        training_args=training_args,
    )

    tokenizer = Qwen2Tokenizer.from_pretrained(lance_dir)
    tokenizer, _new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    model.update_tokenizer(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))

    load_msg = None
    if cfg.load_backbone_weights:
        state_dict = load_file(osp.join(lance_dir, "model.safetensors"), device="cpu")
        state_dict.pop("latent_pos_embed.pos_embed", None)
        load_msg = model.load_state_dict(state_dict, strict=False)
    return model, vae_model, tokenizer, load_msg


# Exactly the audited Lance generation path plus the three shared continuous
# projections and one shared domain embedding.  There are deliberately no
# per-domain projections/heads.
TRAINABLE_PARAMETER_MARKERS = (
    "_moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action_in",
    "action_out",
    "action_modality_embed",
    "state_in",
    "domain_embed",
)

# ``build_lance_base`` deliberately removes this fixed sinusoidal table before
# loading because its size is reconstructed from the local frame/grid ceiling.
# Every learned official-backbone tensor must otherwise load exactly.
EXPECTED_OFFICIAL_BASE_MISSING_KEYS = frozenset({"latent_pos_embed.pos_embed"})


@dataclass
class _PreparedSample:
    text_ids: Tensor
    latent: Tensor
    latent_valid: Tensor
    state: Tensor
    state_valid: Tensor
    action: Tensor
    action_valid: Tensor
    domain_id: int
    main_image: Tensor

def _require_tensor(name: str, value: Any, shape: Tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value


def _wan_latent_frames(rgb_frames: int) -> int:
    """Derive the causal Wan VAE length from the actual training video."""
    if rgb_frames < 5 or (rgb_frames - 1) % 4:
        raise ValueError(
            f"training video must have 1 + 4k frames with k >= 1, got {rgb_frames}"
        )
    return 1 + (rgb_frames - 1) // 4


def _validate_action_geometry(action_horizon: int, latent_frames: int) -> None:
    future_latents = latent_frames - 1
    if future_latents <= 0 or action_horizon <= 0 or action_horizon % future_latents:
        raise ValueError(
            "action horizon must be positive and divisible by future latent "
            f"frames, got action_horizon={action_horizon}, latent_frames={latent_frames}"
        )


def _collated_videos(value: Any, batch_size: int) -> List[Tensor]:
    """Accept one temporal geometry per batch, with optional spatial variation."""
    if batch_size <= 0:
        raise ValueError("collated batch must not be empty")
    if isinstance(value, Tensor):
        if value.ndim != 5 or value.shape[0] != batch_size:
            raise ValueError("dense video batch must be [B,3,T,H,W]")
        videos = list(value.unbind(0))
    elif isinstance(value, (tuple, list)) and len(value) == batch_size:
        videos = list(value)
    else:
        raise TypeError("video must be a dense tensor or one tensor per sample")
    if any(not isinstance(v, Tensor) or v.ndim != 4 or v.shape[0] != 3 for v in videos):
        raise ValueError("each video must be [3,T,H,W]")
    if any(v.shape[1] != videos[0].shape[1] for v in videos):
        raise ValueError("collated videos must have the same temporal length within a batch")
    _wan_latent_frames(int(videos[0].shape[1]))
    return videos


class MachEmbodiedUnifiedModel(LeapVLABase):
    """Joint video/action flow objectives over one unchanged Lance backbone."""

    config_class = MachEmbodiedUnifiedConfig
    base_model_prefix = "ME_U0"

    def __init__(self, config: MachEmbodiedUnifiedConfig) -> None:
        super().__init__(config)
        lance, vae, tokenizer, load_msg = _build_mach_embodied_unified_backbone(
            config
        )
        self.lance = lance
        self.vae = vae
        self.tokenizer = tokenizer
        self._load_msg = load_msg

        hid = self.lance.hidden_size
        self.hidden_size = hid
        self.vision_start_id = int(
            tokenizer.convert_tokens_to_ids("<|vision_start|>")
        )
        self.mrope = bool(getattr(config, "apply_qwen_2_5_vl_pos_emb", False))
        vision_config = getattr(self.lance.language_model.config, "vision_config", None)
        try:
            self._tps = (
                int(vision_config["tokens_per_second"])
                if vision_config is not None
                else 1
            )
        except Exception:  # noqa: BLE001
            self._tps = (
                int(getattr(vision_config, "tokens_per_second", 1))
                if vision_config is not None
                else 1
            )

        assert tuple(self.lance.latent_patch_size) == (1, 1, 1), (
            f"[lance] latent_patch_size={tuple(self.lance.latent_patch_size)} "
            "!= (1,1,1); predict_actions hardcodes a unit patch and would "
            "diverge from training. Make predict_actions patch-aware before "
            "using a non-unit latent_patch_size."
        )

        self.action_in = nn.Linear(config.action_dim, hid)
        self.action_out = nn.Linear(hid, config.action_dim)
        self.action_modality_embed = nn.Parameter(torch.zeros(hid))
        self.state_in = nn.Linear(config.state_dim, hid)
        # Preserve the original construction order before replacing this
        # zero-initialized table with the retained 17-domain table below.
        self.domain_embed = nn.Embedding(16, hid)
        nn.init.zeros_(self.domain_embed.weight)
        self._apply_freeze()

        if getattr(config, "gradient_checkpointing", False):
            self.lance.language_model.model.gradient_checkpointing = True

        if self._load_msg is not None:
            missing = frozenset(self._load_msg.missing_keys)
            unexpected = frozenset(self._load_msg.unexpected_keys)
            if missing != EXPECTED_OFFICIAL_BASE_MISSING_KEYS or unexpected:
                raise RuntimeError(
                    "Official Lance backbone checkpoint did not match the audited "
                    "construction: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        # Replace only the original zero-initialized 16-domain embedding.
        self.domain_embed = nn.Embedding(config.num_logical_domains, hid)
        nn.init.zeros_(self.domain_embed.weight)
        self._apply_freeze()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return the same ordered trainable-parameter list used by the trainer."""

        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _apply_freeze(self) -> None:
        for name, parameter in self.named_parameters():
            trainable = any(marker in name for marker in TRAINABLE_PARAMETER_MARKERS)
            if not self.config.freeze_understanding:
                # In native Lance the UND tensors keep the original Qwen names;
                # only the generation copy is suffixed `_moe_gen`.
                is_decoder_und = (
                    "lance.language_model.model.layers." in name
                    and "_moe_gen" not in name
                )
                is_final_und_norm = "lance.language_model.model.norm." in name
                trainable = trainable or is_decoder_und or is_final_und_norm
                # Match the original unified SFT policy: token embeddings and
                # lm_head are fixed even when the UND decoder is trainable. CE
                # still back-propagates through the fixed lm_head into UND.
                if any(
                    marker in name
                    for marker in (
                        "language_model.model.embed_tokens",
                        "language_model.lm_head",
                    )
                ):
                    trainable = False
            # ViT ownership is controlled only by freeze_vit.  The generation
            # allowlist above intentionally does not contain vit_model, so an
            # explicit positive branch is required for freeze_vit=False to do
            # more than merely enable autograd around the ViT forward.
            if "vit_model" in name:
                trainable = not self.config.freeze_vit
            parameter.requires_grad_(trainable)

    def _shift_t(self, value: Tensor) -> Tensor:
        shift = self.config.timestep_shift
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _sample_time(self, distribution: str, device: torch.device) -> Tensor:
        if distribution == "logitnormal":
            return self._shift_t(torch.sigmoid(torch.randn(1, device=device)))
        if distribution == "beta_1.5_1":
            first = torch._standard_gamma(torch.full((1,), 1.5, device=device))
            second = torch._standard_gamma(torch.full((1,), 1.0, device=device))
            return first / (first + second)
        return self._shift_t(torch.rand(1, device=device))

    def _latent_pos_ids(
        self,
        frames: int,
        height: int,
        width: int,
        frame_start: int,
        device: torch.device,
    ) -> Tensor:
        max_size = self.config.max_latent_size
        frame_ids = torch.arange(frame_start, frame_start + frames, device=device)
        row_ids = torch.arange(height, device=device)
        column_ids = torch.arange(width, device=device)
        frame_grid = frame_ids.view(frames, 1, 1).expand(frames, height, width)
        row_grid = row_ids.view(1, height, 1).expand(frames, height, width)
        column_grid = column_ids.view(1, 1, width).expand(frames, height, width)
        return (
            (frame_grid * max_size + row_grid) * max_size + column_grid
        ).reshape(-1).long()

    def _mp_point(self, start: int, count: int, device: torch.device) -> Tensor:
        positions = torch.arange(count, device=device, dtype=torch.long) + start
        return torch.stack([positions, positions, positions], 0)

    def _mp_grid(
        self,
        base: int,
        frames: Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        frame_count = frames.shape[0]
        frame_grid = frames.view(frame_count, 1, 1).expand(
            frame_count, height, width
        )
        row_grid = torch.arange(height, device=device).view(1, height, 1).expand(
            frame_count, height, width
        )
        column_grid = torch.arange(width, device=device).view(1, 1, width).expand(
            frame_count, height, width
        )
        temporal = (base + frame_grid * self._tps).reshape(-1)
        rows = (base + row_grid).reshape(-1)
        columns = (base + column_grid).reshape(-1)
        return torch.stack([temporal, rows, columns], 0).long()

    def _mp_action(
        self,
        base: int,
        horizon: int,
        future_frames: int,
        spatial_anchor: int,
        device: torch.device,
    ) -> Tensor:
        if self.config.action_position_mode == "group_substep_mrope":
            if horizon <= 0:
                raise ValueError("horizon must be positive for group_substep_mrope")
            if future_frames <= 0:
                raise ValueError(
                    "future_frames must be positive for group_substep_mrope"
                )
            if horizon % future_frames:
                raise ValueError(
                    "horizon must be divisible by future_frames for "
                    "group_substep_mrope"
                )
            ratio = horizon // future_frames
            indexes = torch.arange(horizon, device=device)
            temporal = base + self._tps * (1 + (indexes // ratio))
            rows = int(spatial_anchor) + (indexes % ratio)
            columns = torch.full(
                (horizon,), int(spatial_anchor), device=device, dtype=torch.long
            )
            return torch.stack([temporal.long(), rows.long(), columns], 0)

        ratio = max(1, horizon // max(1, future_frames))
        indexes = torch.arange(horizon, device=device)
        temporal = base + self._tps * (1 + (indexes // ratio))
        anchor = torch.full(
            (horizon,), int(spatial_anchor), device=device, dtype=torch.long
        )
        return torch.stack([temporal.long(), anchor, anchor], 0)

    def _embed_xt(
        self, x_t_patches: Tensor, timestep: Tensor, position_ids: Tensor
    ) -> Tensor:
        vae_dtype = self.lance.vae2llm.weight.dtype
        timestep_vector = timestep.expand(x_t_patches.shape[0])
        return (
            self.lance.vae2llm(x_t_patches.to(vae_dtype))
            + self.lance.time_embedder(timestep_vector)
            + self.lance.latent_pos_embed(position_ids)
        )

    @property
    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            required={
                "video",
                "action",
                "state",
                "domain_ids",
                "task_ids",
                "latent_valid_mask",
                "action_valid_mask",
                "state_dim_valid_mask",
                "instructions",
            },
            optional={
                "video_time_valid_mask",
                "latent_condition_mask",
                "latent_loss_mask",
                "action_condition_mask",
                "action_loss_mask",
                "main_images",
            },
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def _encode_video_batch(
        self, videos: Tensor, device: torch.device
    ) -> Tensor:
        """Encode a dense [B,3,T,H,W] mosaic batch to [B,T',h,w,C]."""

        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                "videos must be [B,3,T,H,W], got "
                f"{tuple(videos.shape)}"
            )
        rgb_frames = int(videos.shape[2])
        expected = _wan_latent_frames(rgb_frames)
        if videos.shape[-2] % 16 or videos.shape[-1] % 16:
            raise ValueError(
                "video height and width must be divisible by 16, got "
                f"{tuple(videos.shape[-2:])}"
            )
        if torch.device(self.vae.device) != device:
            self.vae.to(device)

        # WanVideoVAE.vae_encode loops over its input list and launches B
        # independent encoder forwards.  Call the same frozen posterior in one
        # dense batch so convolutions can use the GPU batch dimension.
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=self.vae.dtype,
            enabled=device.type == "cuda",
        ):
            mean, log_var = self.vae.vae.encode(
                videos.to(device=device, dtype=self.vae.dtype).contiguous()
            )
            latent = mean
            if self.vae.use_sample:
                latent = torch.randn_like(mean) * torch.exp(0.5 * log_var) + mean
        latent = latent.permute(0, 2, 3, 4, 1).contiguous().float()
        if latent.ndim != 5 or latent.shape[1] != expected:
            raise ValueError(
                f"{rgb_frames} RGB frames must encode to "
                f"exactly {expected} Wan latent frames, got {tuple(latent.shape)}"
            )
        if latent.shape[-1] != int(self.lance.latent_channel):
            raise ValueError("Wan latent channel mismatch")
        return latent

    def _encode_video(self, video: Tensor, device: torch.device) -> Tensor:
        """Encode one [3,T,H,W] mosaic to [T_latent,h,w,C]."""

        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(
                "video must be [3,T,H,W], got "
                f"{tuple(video.shape)}"
            )
        rgb_frames = int(video.shape[1])
        expected = _wan_latent_frames(rgb_frames)
        if video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError(
                "video height and width must be divisible by 16, got "
                f"{tuple(video.shape[-2:])}"
            )

        # WanVideoVAE is intentionally a plain object in the audited wrapper,
        # hence nn.Module.to() cannot move it with the rest of the model.
        if torch.device(self.vae.device) != device:
            self.vae.to(device)
        with torch.no_grad():
            latent = self.vae.vae_encode(
                [video.to(device=device).contiguous()]
            )[0].float()

        channels = int(self.lance.latent_channel)
        if latent.ndim != 4:
            raise ValueError(f"Wan VAE returned non-rank-4 latent {latent.shape}")
        if (
            latent.shape[0] == channels
            and latent.shape[1] == expected
        ):
            latent = latent.permute(1, 2, 3, 0).contiguous()
        if latent.shape[0] != expected:
            raise ValueError(
                f"{rgb_frames} RGB frames must encode to exactly "
                f"{expected} Wan latent frames, got "
                f"{latent.shape[0]}"
            )
        if latent.shape[-1] != channels:
            raise ValueError(
                f"Wan latent channel mismatch: {latent.shape[-1]} != {channels}"
            )
        return latent

    def _encode_main_image_batch(
        self, images: Tensor, device: torch.device
    ) -> Tensor:
        """Encode [B,3,224,224] images in one ViT forward."""

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(
                "main_images must be [B,3,224,224], got "
                f"{tuple(images.shape)}"
            )
        images = images.to(device=device, dtype=torch.float32)
        mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        normalized = (images.clamp(0, 1) - mean) / std
        patches = torch.cat([
            patchify_video_with_merge(
                image[:, None].expand(3, 2, 224, 224).contiguous(),
                spatial_patch_size=int(self.config.vit_patch_size),
                temporal_patch_size=int(self.config.vit_patch_size_temporal),
                merge_size=2,
            )
            for image in normalized
        ])
        grid = torch.tensor(
            [[1, 16, 16]], dtype=torch.long, device=device
        ).expand(images.shape[0], -1).contiguous()
        context = torch.no_grad() if self.config.freeze_vit else torch.enable_grad()
        with context:
            embedding = self.lance.vit_model(
                hidden_states=patches.to(
                    dtype=next(self.lance.vit_model.parameters()).dtype
                ),
                grid_thw=grid,
            )
        expected = (images.shape[0], 64, self.hidden_size)
        embedding = embedding.reshape(expected)
        return embedding

    def _encode_collated_visuals(
        self, batch: Mapping[str, Any], device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Batch homogeneous visual inputs on the current CUDA stream."""

        action = batch.get("action")
        if not isinstance(action, Tensor) or action.ndim != 3:
            raise ValueError("collated action must be rank 3")
        batch_size = int(action.shape[0])
        video_items = _collated_videos(batch.get("video"), batch_size)
        rgb_frames = int(video_items[0].shape[1])
        _validate_action_geometry(int(action.shape[1]), _wan_latent_frames(rgb_frames))

        video_valid = batch.get("video_time_valid_mask")
        if video_valid is None:
            video_valid = torch.ones(batch_size, rgb_frames, dtype=torch.bool)
        video_valid = _require_tensor(
            "video_time_valid_mask", video_valid, (batch_size, rgb_frames)
        ).to(device=device).bool()
        shapes = {tuple(item.shape) for item in video_items}
        if len(shapes) != 1:
            raise ValueError(
                "batched visual encoding requires one source/video_size per "
                f"batch; iterable shard alignment was violated: {sorted(shapes)}"
            )
        videos = torch.stack(video_items).to(device=device).float()
        videos.masked_fill_(~video_valid[:, None, :, None, None], 0)

        main_images = batch.get("main_images", ())
        if isinstance(main_images, Tensor):
            if tuple(main_images.shape) != (batch_size, 3, 224, 224):
                raise ValueError("ME_U0 requires one ViT main_image per sample")
            images = main_images.to(device)
        elif isinstance(main_images, Sequence) and len(main_images) == batch_size:
            images = torch.stack([
                torch.as_tensor(image) for image in main_images
            ]).to(device)
        else:
            raise ValueError("ME_U0 requires one ViT main_image per sample")

        return (
            self._encode_video_batch(videos, device),
            self._encode_main_image_batch(images, device),
        )

    def _latent_valid_from_video(self, video_valid: Tensor) -> Tensor:
        if video_valid.ndim != 1:
            raise ValueError("video_time_valid_mask must be rank 1")
        rgb_frames = int(video_valid.shape[0])
        latent_frames = _wan_latent_frames(rgb_frames)
        future_latents = latent_frames - 1
        future_rgb = rgb_frames - 1
        if future_latents <= 0 or future_rgb % future_latents:
            raise ValueError("RGB/latent temporal geometry is inconsistent")
        group = future_rgb // future_latents
        return torch.cat(
            (video_valid[:1], video_valid[1:].view(future_latents, group).any(dim=1))
        )

    @staticmethod
    def _task_id(value: Any) -> int:
        if isinstance(value, str):
            value = {"policy": 2, "video_action_pred": 2}.get(value, -1)
        if isinstance(value, Tensor):
            value = int(value.item())
        if isinstance(value, bool) or not isinstance(value, int) or value != 2:
            raise ValueError(f"post-training task id must be 2 (video_action_pred), got {value!r}")
        return value

    def _validate_domain_id(self, value: Any) -> int:
        if isinstance(value, Tensor):
            value = int(value.item())
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("domain_id must be an integer")
        if not 0 <= value < self.config.num_logical_domains:
            raise ValueError(
                f"domain_id must be in [0,{self.config.num_logical_domains}), got {value}"
            )
        return value

    def _task_geometry(
        self, action_horizon: int, latent_frames: int
    ) -> Tuple[Tuple[int, ...], ...]:
        _validate_action_geometry(action_horizon, latent_frames)
        current_video = (0,)
        future_video = tuple(range(1, latent_frames))
        all_actions = tuple(range(action_horizon))
        return current_video, future_video, (), all_actions

    def _validate_task_masks(
        self,
        batch: Mapping[str, Any],
        index: int,
        latent_valid: Tensor,
        action_valid: Tensor,
    ) -> None:
        clean_latent, noisy_latent, clean_action, noisy_action = (
            self._task_geometry(int(action_valid.shape[0]), int(latent_valid.shape[0]))
        )

        expected_lc = torch.zeros_like(latent_valid, dtype=torch.bool)
        expected_ll = torch.zeros_like(expected_lc)
        expected_ac = torch.zeros_like(action_valid)
        expected_al = torch.zeros_like(action_valid)
        expected_lc[list(clean_latent)] = latent_valid[list(clean_latent)]
        expected_ll[list(noisy_latent)] = latent_valid[list(noisy_latent)]
        if clean_action:
            expected_ac[list(clean_action)] = action_valid[list(clean_action)]
        if noisy_action:
            expected_al[list(noisy_action)] = action_valid[list(noisy_action)]

        for key, expected in (
            ("latent_condition_mask", expected_lc),
            ("latent_loss_mask", expected_ll),
            ("action_condition_mask", expected_ac),
            ("action_loss_mask", expected_al),
        ):
            if key not in batch:
                continue
            supplied = torch.as_tensor(batch[key][index], device=expected.device).bool()
            if key.startswith("action_"):
                supplied = supplied[: expected.shape[0]]
            if supplied.shape != expected.shape or not torch.equal(supplied, expected):
                raise ValueError(f"{key}[{index}] contradicts the policy truth table")

    @staticmethod
    def _mean_owned_losses(
        batch_size: int,
        owners: Sequence[int],
        owned_losses: Sequence[Tensor],
        zero: Tensor,
    ) -> Tensor:
        """Average one optional modality loss over raw samples, including zeros."""

        if batch_size <= 0 or zero.ndim != 0:
            raise AssertionError("per-sample loss reduction received an invalid batch")
        if len(owners) != len(owned_losses):
            raise AssertionError("target owners and losses are not aligned")
        per_sample = [zero for _ in range(batch_size)]
        seen = set()
        for owner, sample_loss in zip(owners, owned_losses):
            if owner in seen or not 0 <= owner < batch_size:
                raise AssertionError("a modality target is not owned exactly once")
            if sample_loss.ndim != 0:
                raise AssertionError("each raw-sample loss must be scalar")
            seen.add(owner)
            per_sample[owner] = sample_loss
        return torch.stack(per_sample).mean()


    def _validate_target_ownership(
        self,
        prepared: Sequence[_PreparedSample],
        sample_ranges: Sequence[Tuple[int, int]],
        video_spans: Sequence[Tuple[int, int, int]],
        video_targets: Sequence[Tensor],
        video_masks: Sequence[Tensor],
        action_spans: Sequence[Tuple[int, int, int]],
        action_targets: Sequence[Tensor],
        action_masks: Sequence[Tensor],
    ) -> None:
        """Fail before the backbone if a target escapes its raw-sample document."""

        if len(sample_ranges) != len(prepared):
            raise AssertionError("packed sample ranges are incomplete")
        expected_video = [
            index
            for index, sample in enumerate(prepared)
            if self._task_geometry(
                sample.action.shape[0], sample.latent.shape[0]
            )[1]
        ]
        expected_action = [
            index
            for index, sample in enumerate(prepared)
            if self._task_geometry(
                sample.action.shape[0], sample.latent.shape[0]
            )[3]
        ]

        def validate(
            name: str,
            spans: Sequence[Tuple[int, int, int]],
            targets: Sequence[Tensor],
            masks: Sequence[Tensor],
            expected_owners: Sequence[int],
        ) -> None:
            if not (len(spans) == len(targets) == len(masks)):
                raise AssertionError(f"{name} targets, masks, and spans drifted")
            owners = [owner for owner, _start, _stop in spans]
            if owners != list(expected_owners):
                raise AssertionError(
                    f"{name} target ownership contradicts the task truth table"
                )
            for (owner, start, stop), target, mask in zip(spans, targets, masks):
                sample_start, sample_stop = sample_ranges[owner]
                if not sample_start <= start < stop <= sample_stop:
                    raise AssertionError(f"{name} target escaped its sample document")
                if target.ndim != 2 or mask.ndim != 2:
                    raise AssertionError(f"{name} target/mask must both be rank two")
                if stop - start != target.shape[0] or mask.shape[0] != target.shape[0]:
                    raise AssertionError(f"{name} target span length is inconsistent")
                if name == "video" and mask.shape[1] != 1:
                    raise AssertionError("video target mask must be token-wise")
                if name == "action" and mask.shape != target.shape:
                    raise AssertionError("action target mask must be element-wise")

        validate(
            "video", video_spans, video_targets, video_masks, expected_video
        )
        validate(
            "action", action_spans, action_targets, action_masks, expected_action
        )

    def _validate_packed_layout(
        self,
        *,
        packed_len: int,
        sample_lens: Sequence[int],
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
        mask_sample_lens: Sequence[int],
        mask_split_lens: Sequence[int],
        mask_attn_modes: Sequence[str],
        token_valid: Tensor,
        packed_position_ids: Tensor,
    ) -> None:
        """Validate real documents and the mask-only 128-token pseudo-document."""

        block = int(self.config.flex_block_size)
        if packed_len <= 0 or block <= 0:
            raise AssertionError("packed sequence and Flex block must be non-empty")
        if not sample_lens or any(int(length) <= 0 for length in sample_lens):
            raise AssertionError("every packed raw sample must have positive length")
        if sum(sample_lens) != packed_len:
            raise AssertionError("sample_lens do not cover the real packed sequence")
        if len(split_lens) != len(attn_modes) or sum(split_lens) != packed_len:
            raise AssertionError("attention splits do not cover the packed sequence")

        pad = (-packed_len) % block
        expected_sample_lens = list(sample_lens) + ([pad] if pad else [])
        expected_split_lens = list(split_lens) + ([pad] if pad else [])
        expected_attn_modes = list(attn_modes) + (["causal"] if pad else [])
        if list(mask_sample_lens) != expected_sample_lens:
            raise AssertionError("mask sample_lens contain an invalid pseudo-document")
        if list(mask_split_lens) != expected_split_lens:
            raise AssertionError("mask split_lens contain an invalid pseudo-document")
        if list(mask_attn_modes) != expected_attn_modes:
            raise AssertionError("mask attention modes contain an invalid pseudo-document")
        mask_len = sum(mask_sample_lens)
        if mask_len != packed_len + pad or mask_len % block:
            raise AssertionError("Flex mask length is not the next 128-token boundary")
        if token_valid.dtype != torch.bool or tuple(token_valid.shape) != (mask_len,):
            raise AssertionError("token validity does not cover the Flex mask")
        if pad and bool(token_valid[-pad:].any()):
            raise AssertionError("the trailing pseudo-document must be entirely invalid")

        expected_position_shape = (
            (3, 1, packed_len) if self.mrope else (packed_len,)
        )
        if tuple(packed_position_ids.shape) != expected_position_shape:
            raise AssertionError("position IDs do not cover exactly the real tokens")

    def _prepare_collated(
        self,
        batch: Mapping[str, Any],
        precomputed_latents: Optional[Tensor] = None,
    ) -> List[_PreparedSample]:
        device = next(self.parameters()).device
        action = batch.get("action")
        state = batch.get("state")
        action_dim = int(self.config.action_dim)
        state_dim = int(self.config.state_dim)
        main_image_size = int(self.config.main_image_size)
        if not isinstance(action, Tensor) or action.ndim != 3:
            raise ValueError(
                f"collated action must be [B,H,{action_dim}]"
            )
        batch_size = action.shape[0]
        action_horizon = int(action.shape[1])
        videos = _collated_videos(batch.get("video"), batch_size)
        rgb_frames = int(videos[0].shape[1])
        latent_frames = _wan_latent_frames(rgb_frames)
        _validate_action_geometry(action_horizon, latent_frames)
        _require_tensor(
            "action", action, (batch_size, action_horizon, action_dim)
        )
        _require_tensor("state", state, (batch_size, state_dim))
        domain_ids = _require_tensor(
            "domain_ids", batch.get("domain_ids"), (batch_size,)
        )
        task_ids = _require_tensor("task_ids", batch.get("task_ids"), (batch_size,))
        latent_valid_all = _require_tensor(
            "latent_valid_mask",
            batch.get("latent_valid_mask"),
            (batch_size, latent_frames),
        ).bool()
        action_valid_all = _require_tensor(
            "action_valid_mask",
            batch.get("action_valid_mask"),
            (batch_size, action_horizon, action_dim),
        ).bool()
        state_valid_all = _require_tensor(
            "state_dim_valid_mask",
            batch.get("state_dim_valid_mask"),
            (batch_size, state_dim),
        ).bool()

        instructions = batch.get("instructions")
        if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
            raise TypeError("instructions must be a sequence of raw strings")
        if len(instructions) != batch_size or not all(isinstance(x, str) for x in instructions):
            raise ValueError("instructions must contain one raw string per sample")
        text_metadata = batch.get("text_metadata")
        if text_metadata is None:
            text_metadata = [None] * batch_size
        if (
            not isinstance(text_metadata, Sequence)
            or isinstance(text_metadata, (str, bytes))
            or len(text_metadata) != batch_size
            or not all(item is None or isinstance(item, Metadata) for item in text_metadata)
        ):
            raise ValueError(
                "text_metadata must contain one Metadata-or-None value per sample"
            )

        main_images = batch.get("main_images", ())
        if not isinstance(main_images, Sequence) or len(main_images) != batch_size:
            raise ValueError("ME_U0 requires one ViT main_image per sample")

        if "video_time_valid_mask" in batch:
            video_valid_all = _require_tensor(
                "video_time_valid_mask",
                batch["video_time_valid_mask"],
                (batch_size, rgb_frames),
            ).bool()
        else:
            video_valid_all = torch.ones(
                batch_size, rgb_frames, dtype=torch.bool
            )

        prepared: List[_PreparedSample] = []
        for i in range(batch_size):
            self._task_id(task_ids[i])
            domain_id = self._validate_domain_id(domain_ids[i])
            video_valid = video_valid_all[i].to(device=device)
            derived_latent_valid = self._latent_valid_from_video(video_valid)
            latent_valid = latent_valid_all[i].to(device=device)
            if not torch.equal(latent_valid, derived_latent_valid):
                raise ValueError(
                    f"latent_valid_mask[{i}] is inconsistent with "
                    f"{rgb_frames} RGB frames"
                )

            video = videos[i]
            if not isinstance(video, Tensor):
                raise TypeError("each video must be a tensor")
            if precomputed_latents is None:
                video = video.to(device=device).float().clone()
                video.masked_fill_(~video_valid.view(1, rgb_frames, 1, 1), 0)
                latent = self._encode_video(video, device)
            else:
                if tuple(precomputed_latents.shape[:2]) != (
                    batch_size,
                    latent_frames,
                ):
                    raise ValueError(
                        "precomputed latents must be "
                        f"[B,{latent_frames},h,w,C]"
                    )
                latent = precomputed_latents[i]

            action_valid = action_valid_all[i].to(device=device)
            state_valid = state_valid_all[i].to(device=device)
            if not bool(latent_valid[0]):
                raise ValueError(f"sample {i} has no valid current visual condition")
            if not bool(latent_valid[1:].any()):
                raise ValueError(
                    f"sample {i} has no valid future visual transition for policy"
                )
            if not bool(action_valid.any()):
                raise ValueError(f"sample {i} has no valid action condition/target")
            if not bool(state_valid.any()):
                raise ValueError(f"sample {i} has no valid current-state dimension")
            clean_action = action[i].to(device=device).float().masked_fill(~action_valid, 0)
            clean_state = state[i].to(device=device).float().masked_fill(~state_valid, 0)
            self._validate_task_masks(
                batch, i, latent_valid, action_valid
            )

            text_ids = build_instruction_text_ids(
                self.tokenizer,
                instructions[i],
                max_text_len=self.config.max_text_len,
                metadata=text_metadata[i],
            ).to(device=device)
            prepared.append(
                _PreparedSample(
                    text_ids=text_ids,
                    latent=latent,
                    latent_valid=latent_valid,
                    state=clean_state,
                    state_valid=state_valid,
                    action=clean_action,
                    action_valid=action_valid,
                    domain_id=domain_id,
                    main_image=torch.as_tensor(
                        main_images[i], device=device
                    ).float(),
                )
            )
            if tuple(prepared[-1].main_image.shape) != (
                3, main_image_size, main_image_size
            ):
                raise ValueError(
                    "main_images entries must be "
                    f"[3,{main_image_size},{main_image_size}]"
                )
        return prepared


    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------
    def _latent_tokens(
        self,
        sample: _PreparedSample,
        frames: Tuple[int, ...],
        timestep: Tensor,
        *,
        noised: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        if not frames:
            raise ValueError("latent token block cannot be empty")
        if tuple(frames) != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("latent frames must form one contiguous block")
        latent = sample.latent[list(frames)]
        _, h, w, channels = latent.shape
        clean = latent.reshape(-1, channels)
        valid = sample.latent_valid[list(frames)].repeat_interleave(h * w)
        clean = clean.masked_fill(~valid[:, None], 0)
        pos = self._latent_pos_ids(
            len(frames), h, w, int(frames[0]), latent.device
        )
        if noised:
            noise = torch.randn_like(clean).masked_fill(~valid[:, None], 0)
            x_t = ((1.0 - timestep) * clean + timestep * noise).masked_fill(
                ~valid[:, None], 0
            )
            target: Optional[Tensor] = noise - clean
        else:
            x_t = clean
            target = None
        embed = self._embed_xt(x_t, timestep, pos)
        return embed, target, valid, pos

    def _action_tokens(
        self,
        sample: _PreparedSample,
        timestep: Tensor,
        *,
        noised: bool,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        clean = sample.action.masked_fill(~sample.action_valid, 0)
        if noised:
            noise = torch.randn_like(clean).masked_fill(~sample.action_valid, 0)
            x_t = ((1.0 - timestep) * clean + timestep * noise).masked_fill(
                ~sample.action_valid, 0
            )
            target: Optional[Tensor] = noise - clean
        else:
            x_t = clean
            target = None
        domain = self.domain_embed(
            torch.tensor(sample.domain_id, device=clean.device)
        )
        embed = (
            self.action_in(x_t.to(dtype))
            + self.action_modality_embed
            + self.lance.time_embedder(timestep.expand(clean.shape[0]))
            + domain
        )
        token_valid = sample.action_valid.any(dim=-1)
        return embed, target, token_valid

    def _encode_main_image(self, image: Tensor, device: torch.device) -> Tensor:
        """Run the original Lance/Qwen2.5-VL ViT on one 224x224 main view."""

        if tuple(image.shape) != (3, 224, 224):
            raise ValueError(f"main_image must be [3,224,224], got {tuple(image.shape)}")
        image = image.to(device=device, dtype=torch.float32)
        mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        std = image.new_tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        normalized = (image.clamp(0, 1) - mean) / std
        video = normalized[:, None].expand(3, 2, 224, 224).contiguous()
        patches = patchify_video_with_merge(
            video,
            spatial_patch_size=int(self.config.vit_patch_size),
            temporal_patch_size=int(self.config.vit_patch_size_temporal),
            merge_size=2,
        )
        grid = torch.tensor([[1, 16, 16]], dtype=torch.long, device=device)
        context = torch.no_grad() if self.config.freeze_vit else torch.enable_grad()
        with context:
            embedding = self.lance.vit_model(
                hidden_states=patches.to(
                    dtype=next(self.lance.vit_model.parameters()).dtype
                ),
                grid_thw=grid,
            )
        if tuple(embedding.shape) != (64, self.hidden_size):
            raise ValueError(
                "official ViT must merge 224x224 into 64 semantic tokens, got "
                f"{tuple(embedding.shape)}"
            )
        return embedding


    def _text_image_positions(
        self,
        coarse_prefix_ids: Tensor,
        image_ids: Tensor,
        close_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extend legacy text positions with one native Qwen image grid.

        Existing instruction tokens keep the original Lance text positions.
        Image tokens use Qwen's 1x8x8 merged grid; their temporal axis is moved
        to the official Lance ViT range locally for this sample. The final
        user-turn close token keeps its original position, so the image does
        not renumber the VAE/state/action geometry.
        """

        device = coarse_prefix_ids.device
        n_prefix = int(coarse_prefix_ids.numel())
        if tuple(close_ids.shape) != (1,) or int(image_ids.numel()) != 66:
            raise ValueError("text/image skeleton has invalid geometry")
        skeleton = torch.cat((coarse_prefix_ids, image_ids, close_ids))
        local, _ = self.lance.language_model.get_rope_index(
            input_ids=skeleton.unsqueeze(0),
            image_grid_thw=torch.tensor([[1, 16, 16]], device=device),
            attention_mask=torch.ones(
                1, skeleton.numel(), device=device, dtype=torch.long
            ),
        )
        prefix_pos = self._mp_point(0, n_prefix, device)
        image_start = n_prefix
        image_stop = image_start + int(image_ids.numel())
        image_pos = local[:, 0, image_start:image_stop].clone()
        image_pos[0] += 1000 - image_pos[0, 0]
        close_pos = self._mp_point(n_prefix, 1, device)
        # The fixed 224x224 image produces vision-start + 8x8 merged patches +
        # vision-end. Keep the generation coordinates independent of this grid.
        return prefix_pos, image_pos, close_pos

    def _forward_video_action(
        self,
        batch: Mapping[str, Any],
    ) -> Dict[str, Tensor]:
        """Pack ViT/UND and generation tokens using each sample's actual lengths."""

        batched_vit: Optional[Tensor] = None
        if self.config.batch_vae_vit_encoders:
            device = next(self.parameters()).device
            batched_latents, batched_vit = self._encode_collated_visuals(batch, device)
            prepared = self._prepare_collated(batch, precomputed_latents=batched_latents)
        else:
            prepared = self._prepare_collated(batch)
        if not prepared:
            raise ValueError("ME_U0 post-training forward requires at least one sample")
        device = next(self.parameters()).device
        dtype = self.lance.language_model.model.embed_tokens.weight.dtype
        hid = self.hidden_size
        language_model = self.lance.language_model
        lm_config = language_model.config
        image_pad_id = int(lm_config.image_token_id)
        vision_end_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))

        rows: List[Tensor] = []
        position_rows: List[Tensor] = []
        valid_rows: List[Tensor] = []
        split_lens: List[int] = []
        attn_modes: List[str] = []
        sample_lens: List[int] = []
        sample_ranges: List[Tuple[int, int]] = []
        und_idx: List[int] = []
        gen_idx: List[int] = []
        video_spans: List[Tuple[int, int, int]] = []
        video_targets: List[Tensor] = []
        video_masks: List[Tensor] = []
        action_spans: List[Tuple[int, int, int]] = []
        action_targets: List[Tensor] = []
        action_masks: List[Tensor] = []
        cursor = 0

        def add(
            embedding: Tensor,
            valid: Tensor,
            mode: str,
            expert: str,
            position: Tensor,
        ) -> int:
            nonlocal cursor
            count = int(embedding.shape[0])
            if tuple(valid.shape) != (count,) or tuple(position.shape) != (3, count):
                raise ValueError("packed segment tensors are not aligned")
            start = cursor
            rows.append(embedding)
            position_rows.append(position.long())
            valid_rows.append(valid.bool())
            split_lens.append(count)
            attn_modes.append(mode)
            indexes = range(start, start + count)
            (und_idx if expert == "und" else gen_idx).extend(indexes)
            cursor += count
            return start

        for owner, sample in enumerate(prepared):
            action_horizon = int(sample.action.shape[0])
            latent_frames = int(sample.latent.shape[0])
            future_latents = latent_frames - 1
            sample_start = cursor
            clean_frames, noisy_frames, clean_actions, noisy_actions = (
                self._task_geometry(action_horizon, latent_frames)
            )
            flow_time = self._sample_time("logitnormal", device)
            clean_time = torch.zeros_like(flow_time)
            _, h, w, _ = sample.latent.shape
            legacy_text_len = int(sample.text_ids.numel())
            video_base = legacy_text_len + 1
            video_max = video_base + max(
                self._tps * future_latents, h - 1, w - 1
            )
            point_cursor = video_max + 1
            action_anchor = point_cursor + 1

            # Place the image after the instruction, inside the same user turn.
            coarse_prefix_ids = sample.text_ids[:-1].to(device=device)
            close_ids = sample.text_ids[-1:].to(device=device)
            vit = (
                batched_vit[owner]
                if batched_vit is not None
                else self._encode_main_image(sample.main_image, device)
            ).to(dtype)
            image_ids = torch.cat((
                torch.tensor([self.vision_start_id], device=device),
                torch.full((vit.shape[0],), image_pad_id, device=device, dtype=torch.long),
                torch.tensor([vision_end_id], device=device),
            ))
            image_embedding = torch.cat((
                language_model.model.embed_tokens(image_ids[:1]),
                vit,
                language_model.model.embed_tokens(image_ids[-1:]),
            ))
            prefix_pos, image_pos, close_pos = (
                self._text_image_positions(
                    coarse_prefix_ids, image_ids, close_ids
                )
            )
            add(
                language_model.model.embed_tokens(coarse_prefix_ids),
                torch.ones_like(coarse_prefix_ids, dtype=torch.bool),
                "causal", "und", prefix_pos,
            )
            add(
                image_embedding,
                torch.ones(image_ids.numel(), dtype=torch.bool, device=device),
                "full", "und", image_pos,
            )
            add(
                language_model.model.embed_tokens(close_ids),
                torch.ones_like(close_ids, dtype=torch.bool),
                "causal", "und", close_pos,
            )

            # From this marker onward, generation tokens keep the original
            # Lance positions. ViT tokens cannot renumber the
            # VAE/state/action geometry.
            marker_id = torch.tensor([self.vision_start_id], device=device)
            add(
                language_model.model.embed_tokens(marker_id).view(1, hid),
                torch.ones(1, dtype=torch.bool, device=device),
                "full", "gen", self._mp_point(legacy_text_len, 1, device),
            )

            clean_video, _, clean_video_valid, _ = self._latent_tokens(
                sample, clean_frames, clean_time, noised=False
            )
            add(
                clean_video, clean_video_valid, "full", "gen",
                self._mp_grid(
                    video_base,
                    torch.tensor(clean_frames, device=device, dtype=torch.long),
                    h, w, device,
                ),
            )

            domain = self.domain_embed(torch.tensor(sample.domain_id, device=device))
            state_embedding = (self.state_in(sample.state.to(dtype)) + domain).view(1, hid)
            add(
                state_embedding, sample.state_valid.any().view(1),
                "full", "gen", self._mp_point(point_cursor, 1, device),
            )

            noise_embeddings: List[Tensor] = []
            noise_valid: List[Tensor] = []
            noise_positions: List[Tensor] = []
            local_video_offset: Optional[int] = None
            local_action_offset: Optional[int] = None
            if noisy_frames:
                noisy_video, target, valid, _ = self._latent_tokens(
                    sample, noisy_frames, flow_time, noised=True
                )
                assert target is not None
                local_video_offset = sum(x.shape[0] for x in noise_embeddings)
                noise_embeddings.append(noisy_video)
                noise_valid.append(valid)
                noise_positions.append(self._mp_grid(
                    video_base,
                    torch.tensor(noisy_frames, device=device, dtype=torch.long),
                    h, w, device,
                ))
                video_targets.append(target)
                video_masks.append(valid[:, None])
            if noisy_actions:
                noisy_action, target, valid = self._action_tokens(
                    sample, flow_time, noised=True, dtype=dtype
                )
                assert target is not None
                local_action_offset = sum(x.shape[0] for x in noise_embeddings)
                noise_embeddings.append(noisy_action)
                noise_valid.append(valid)
                noise_positions.append(
                    self._mp_action(
                        video_base, action_horizon, future_latents, action_anchor, device
                    )
                )
                action_targets.append(target)
                action_masks.append(sample.action_valid)
            if noise_embeddings:
                noise_start = add(
                    torch.cat(noise_embeddings), torch.cat(noise_valid),
                    "noise", "gen", torch.cat(noise_positions, dim=1),
                )
                if local_video_offset is not None:
                    count = video_targets[-1].shape[0]
                    start = noise_start + local_video_offset
                    video_spans.append((owner, start, start + count))
                if local_action_offset is not None:
                    start = noise_start + local_action_offset
                    action_spans.append((owner, start, start + action_horizon))

            sample_lens.append(cursor - sample_start)
            sample_ranges.append((sample_start, cursor))

        packed_sequence = torch.cat(rows).to(dtype)
        position_ids = torch.cat(position_rows, dim=1).unsqueeze(1)
        token_valid = torch.cat(valid_rows)
        self._validate_target_ownership(
            prepared,
            sample_ranges,
            video_spans,
            video_targets,
            video_masks,
            action_spans,
            action_targets,
            action_masks,
        )
        if len(und_idx) + len(gen_idx) != packed_sequence.shape[0]:
            raise AssertionError("UND/GEN ownership does not cover the packed sequence")
        # Action substeps receive no learned embedding.  _mp_action either keeps
        # the legacy transition-shared coordinates or assigns distinct M-RoPE row
        # coordinates within each future-latent transition.

        block = int(self.config.flex_block_size)
        pad = (-packed_sequence.shape[0]) % block
        mask_sample_lens = sample_lens + ([pad] if pad else [])
        mask_split_lens = split_lens + ([pad] if pad else [])
        mask_modes = attn_modes + (["causal"] if pad else [])
        if pad:
            token_valid = torch.cat((token_valid, torch.zeros(pad, device=device, dtype=torch.bool)))
        self._validate_packed_layout(
            packed_len=int(packed_sequence.shape[0]),
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            mask_sample_lens=mask_sample_lens,
            mask_split_lens=mask_split_lens,
            mask_attn_modes=mask_modes,
            token_valid=token_valid,
            packed_position_ids=position_ids,
        )
        attention_mask = self.lance.process_attention_mask(
            mask_modes, mask_split_lens, mask_sample_lens, device,
            BLOCK_SIZE=block, token_valid=token_valid,
        )
        hidden = language_model(
            packed_sequence=packed_sequence,
            sample_lens=mask_sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=position_ids,
            packed_und_token_indexes=torch.tensor(und_idx, device=device, dtype=torch.long),
            packed_gen_token_indexes=torch.tensor(gen_idx, device=device, dtype=torch.long),
        )

        batch_size = len(prepared)
        video_sample_losses = []
        if video_spans:
            indexes = torch.tensor([i for _o, s, e in video_spans for i in range(s, e)], device=device)
            prediction = self.lance.llm2vae(hidden[indexes]).float()
            chunks = prediction.split([target.shape[0] for target in video_targets])
            video_sample_losses = [masked_token_mse(p, t.float(), m) for p, t, m in zip(chunks, video_targets, video_masks)]
            loss_video = self._mean_owned_losses(batch_size, [x[0] for x in video_spans], video_sample_losses, prediction.sum() * 0)
        else:
            zero = self.lance.llm2vae(hidden[:1]).sum() * 0
            loss_video = self._mean_owned_losses(batch_size, (), (), zero)
        if action_spans:
            indexes = torch.tensor([i for _o, s, e in action_spans for i in range(s, e)], device=device)
            prediction = self.action_out(hidden[indexes]).float()
            chunks = prediction.split([target.shape[0] for target in action_targets])
            losses = []
            for p, target, mask in zip(chunks, action_targets, action_masks):
                mask = mask.to(p.dtype)
                losses.append((((p - target.float()) ** 2) * mask).sum() / mask.sum().clamp(min=1))
            loss_action = self._mean_owned_losses(batch_size, [x[0] for x in action_spans], losses, prediction.sum() * 0)
        else:
            zero = self.action_out(hidden[:1]).sum() * 0
            loss_action = self._mean_owned_losses(batch_size, (), (), zero)
        loss = (
            self.config.video_loss_weight * loss_video
            + self.config.action_loss_weight * loss_action
        )
        return {
            "loss": loss,
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        samples: Optional[Any] = None,
        **batch: Any,
    ) -> Dict[str, Tensor]:
        if isinstance(samples, Mapping):
            if batch:
                raise ValueError("pass either one collator dict or keyword fields, not both")
            batch = dict(samples)
            samples = None
        if samples is not None:
            raise TypeError("post-training forward requires a collated batch")
        return self._forward_video_action(batch)

    @torch.no_grad()
    def predict_actions(self, *, mode: str = "auto", **batch: Any) -> Tensor:
        """Run the retained ViT-observation policy sampler."""

        if mode not in ("auto", "policy"):
            raise ValueError("ME_U0 predict_actions supports policy mode only")
        return self._predict_actions_vit_policy(**batch)

    @torch.no_grad()
    def _encode_current_observation(
        self, video: Tensor, device: torch.device
    ) -> Tensor:
        """Encode one serving observation for policy inference."""

        if video.ndim != 4 or tuple(video.shape[:2]) != (1, 3):
            raise ValueError("LIBERO policy video must be [1,3,H,W]")
        current = self.vae.vae_encode(
            [video.to(device).permute(1, 0, 2, 3).contiguous()]
        )[0].float()[:1]
        if current.ndim != 4 or current.shape[0] != 1:
            raise ValueError("current Wan latent must be [1,h,w,C]")
        return current

    @torch.no_grad()
    def _predict_actions_vit_policy(
        self,
        *,
        text_ids: Tensor,
        state: Tensor,
        video: Tensor,
        main_image: Tensor,
        domain_id: int = 0,
        num_steps: Optional[int] = None,
        return_video: bool = False,
        raw_action_dim: int = 26,
        current_latent: Optional[Tensor] = None,
        **_: Any,
    ) -> Any:
        """Policy sampler matching the post-training sequence exactly."""

        device = next(self.parameters()).device
        dtype = self.lance.language_model.model.embed_tokens.weight.dtype
        lm = self.lance.language_model
        image_pad_id = int(lm.config.image_token_id)
        vision_end_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        text_ids = text_ids.to(device).long()
        state = state.to(device).float()
        main_image = main_image.to(device).float()
        current = (
            self._encode_current_observation(video, device)
            if current_latent is None
            else current_latent.to(device=device).float()
        )
        if current.ndim != 4 or current.shape[0] != 1:
            raise ValueError("current_latent must be [1,h,w,C]")
        _, h, w, channels = current.shape
        current_flat = current.reshape(-1, channels)
        current_pos = self._latent_pos_ids(1, h, w, 0, device)
        vit = self._encode_main_image(main_image, device).to(dtype)
        domain = self.domain_embed(torch.tensor(int(domain_id), device=device))

        coarse_prefix_ids = text_ids[:-1]
        close_ids = text_ids[-1:].clone()
        image_ids = torch.cat((
            torch.tensor([self.vision_start_id], device=device),
            torch.full((vit.shape[0],), image_pad_id, device=device),
            torch.tensor([vision_end_id], device=device),
        ))
        prefix_pos, image_pos, close_pos = (
            self._text_image_positions(
                coarse_prefix_ids, image_ids, close_ids
            )
        )
        image_embedding = torch.cat((
            lm.model.embed_tokens(image_ids[:1]),
            vit,
            lm.model.embed_tokens(image_ids[-1:]),
        ))
        current_embed = self._embed_xt(current_flat, torch.zeros(1, device=device), current_pos)
        state_embed = (self.state_in(state.to(dtype)) + domain).view(1, self.hidden_size)
        marker_id = torch.tensor([self.vision_start_id], device=device)
        marker_embed = lm.model.embed_tokens(marker_id).view(1, self.hidden_size)

        n_text = int(coarse_prefix_ids.numel())
        n_image = int(image_ids.numel())
        n_current = h * w
        text_len = int(text_ids.numel())
        video_base = text_len + 1
        future_latents = int(self.config.num_video_latent_frames) - 1
        action_horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        video_max = video_base + max(
            self._tps * future_latents, h - 1, w - 1
        )
        point_cursor = video_max + 1
        action_anchor = point_cursor + 1

        prefix_start = 0
        image_start = prefix_start + n_text
        close_start = image_start + n_image
        marker_start = close_start + 1
        current_start = marker_start + 1
        state_index = current_start + n_current
        fixed_len = state_index + 1
        future_len = future_latents * h * w
        action_len = action_horizon
        future_start = fixed_len
        action_start = future_start + future_len
        total_len = action_start + action_len
        und_idx = torch.arange(0, marker_start, device=device)
        gen_idx = torch.arange(marker_start, total_len, device=device)
        split_lens = [n_text, n_image, 1]
        attn_modes = ["causal", "full", "causal"]
        split_lens.extend((1, n_current, 1))
        attn_modes.extend(("full", "full", "full"))
        split_lens.append(future_len + action_len)
        attn_modes.append("noise")
        pos = torch.cat((
            prefix_pos,
            image_pos,
            close_pos,
            self._mp_point(text_len, 1, device),
            self._mp_grid(
                video_base, torch.zeros(1, device=device, dtype=torch.long),
                h, w, device,
            ),
            self._mp_point(point_cursor, 1, device),
            self._mp_grid(
                video_base,
                torch.arange(1, future_latents + 1, device=device),
                h,
                w,
                device,
            ),
            self._mp_action(
                video_base,
                action_horizon,
                future_latents,
                action_anchor,
                device,
            ),
        ), dim=1).unsqueeze(1)
        if int(pos.shape[-1]) != total_len:
            raise AssertionError("policy inference position length drifted")

        pad = (-total_len) % int(self.config.flex_block_size)
        mask_sample_lens = [total_len] + ([pad] if pad else [])
        mask_split_lens = split_lens + ([pad] if pad else [])
        mask_modes = attn_modes + (["causal"] if pad else [])
        token_valid = torch.ones(total_len + pad, dtype=torch.bool, device=device)
        if pad:
            token_valid[-pad:] = False
        attention_mask = self.lance.process_attention_mask(
            mask_modes, mask_split_lens, mask_sample_lens, device,
            BLOCK_SIZE=int(self.config.flex_block_size), token_valid=token_valid,
        )

        steps = int(num_steps or self.config.num_inference_steps)
        schedule = torch.linspace(1, 0, steps + 1, device=device)
        shift = float(self.config.validation_timestep_shift)
        schedule = shift * schedule / (1 + (shift - 1) * schedule)
        delta = schedule[:-1] - schedule[1:]
        future = torch.randn(future_len, channels, device=device)
        action_valid = torch.arange(action_dim, device=device) < int(raw_action_dim)
        action = torch.randn(action_horizon, action_dim, device=device).masked_fill(
            ~action_valid[None, :], 0
        )
        future_pos = self._latent_pos_ids(future_latents, h, w, 1, device)
        fixed = torch.cat((
            lm.model.embed_tokens(coarse_prefix_ids),
            image_embedding,
            lm.model.embed_tokens(close_ids),
            marker_embed,
            current_embed,
            state_embed,
        )).to(dtype)
        for index in range(steps):
            timestep = schedule[index:index + 1]
            future_embed = self._embed_xt(future, timestep, future_pos)
            action_embed = (
                self.action_in(action.to(dtype)) + self.action_modality_embed
                + self.lance.time_embedder(timestep.expand(action_horizon)) + domain
            )
            sequence = torch.cat((
                fixed, future_embed, action_embed
            )).to(dtype)
            hidden = lm(
                packed_sequence=sequence,
                sample_lens=mask_sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=pos,
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
                mode_forward="validation",
            )
            future = future - self.lance.llm2vae(
                hidden[future_start:future_start + future_len]
            ).float() * delta[index]
            action_velocity = self.action_out(hidden[action_start:]).float().masked_fill(
                ~action_valid[None, :], 0
            )
            action = (action - action_velocity * delta[index]).masked_fill(
                ~action_valid[None, :], 0
            )
        if return_video:
            future_latent = future.view(future_latents, h, w, channels)
            rgb = self.vae.vae_decode([torch.cat((current, future_latent))])[0][:, 1:]
            return action, rgb
        return action


@register_model("ME_U0")
def build_mach_embodied_unified(**kwargs: Any) -> MachEmbodiedUnifiedModel:
    return MachEmbodiedUnifiedModel(MachEmbodiedUnifiedConfig(**kwargs))
