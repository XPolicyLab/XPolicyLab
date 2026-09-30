"""Single configuration for every retained ME_U0 training and inference route.

The configuration reconstructs the official bundled Lance backbone and defines
the default inference video/action geometry. Training derives temporal lengths
from each batch. Embodiment routing is a zero-initialized
domain embedding added to the shared action/state projections, not a collection
of per-domain heads.
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import PretrainedConfig


OFFICIAL_LANCE_CHECKPOINT_ROOT = "/path/to/lance-assets"


class MachEmbodiedUnifiedConfig(PretrainedConfig):
    """ME_U0 configuration with the retained route invariants locked."""

    model_type = "ME_U0"

    def __init__(
        self,
        *,
        checkpoint_root: str = OFFICIAL_LANCE_CHECKPOINT_ROOT,
        load_backbone_weights: bool = True,
        lance_subdir: str = "Lance_3B_Video",
        vit_subdir: str = "Qwen2.5-VL-ViT",
        vae_filename: str = "Wan2.2_VAE.pth",
        layer_module: str = "Qwen2MoTDecoderLayer",
        llm_qk_norm: bool = True,
        llm_qk_norm_und: bool = True,
        llm_qk_norm_gen: bool = True,
        tie_word_embeddings: bool = False,
        vit_type: str = "qwen_2_5_vl_original",
        vit_patch_size: int = 14,
        vit_patch_size_temporal: int = 2,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        latent_patch_size: tuple[int, int, int] = (1, 1, 1),
        max_latent_size: int = 64,
        max_num_frames: int = 41,
        visual_gen: bool = True,
        visual_und: bool = True,
        action_dim: int = 26,
        state_dim: int = 26,
        action_space_schema_id: Optional[str] = None,
        action_horizon: int = 32,
        action_position_mode: str = "legacy_grouped",
        num_video_frames: int = 33,
        num_video_latent_frames: int = 9,
        num_logical_domains: int = 17,
        max_text_len: int = 512,
        flex_block_size: int = 128,
        timestep_shift: float = 4.0,
        time_dist_video: str = "logitnormal",
        action_time_schedule: str = "shared",
        time_dist_action: str = "logitnormal",
        apply_qwen_2_5_vl_pos_emb: bool = True,
        validation_timestep_shift: float = 3.0,
        num_inference_steps: int = 24,
        video_loss_weight: float = 10.0,
        action_loss_weight: float = 10.0,
        freeze_understanding: bool = False,
        freeze_vit: bool = True,
        freeze_vae: bool = True,
        gradient_checkpointing: bool = False,
        pretrained_pth: Optional[str] = None,
        pretrained_strict: bool = True,
        main_image_size: int = 224,
        batch_vae_vit_encoders: bool = True,
        **kwargs: Any,
    ) -> None:
        # Launcher-only fields are serialized so checkpoints record whether
        # training started from the official backbone or a strict warm start.
        self.pretrained_pth = pretrained_pth
        self.pretrained_strict = bool(pretrained_strict)
        self.main_image_size = int(main_image_size)
        self.batch_vae_vit_encoders = bool(batch_vae_vit_encoders)

        self.load_backbone_weights = bool(load_backbone_weights)
        self.checkpoint_root = checkpoint_root
        self.lance_subdir = lance_subdir
        self.vit_subdir = vit_subdir
        self.vae_filename = vae_filename
        self.layer_module = layer_module
        self.llm_qk_norm = llm_qk_norm
        self.llm_qk_norm_und = llm_qk_norm_und
        self.llm_qk_norm_gen = llm_qk_norm_gen
        self.tie_word_embeddings = tie_word_embeddings
        self.vit_type = vit_type
        self.vit_patch_size = vit_patch_size
        self.vit_patch_size_temporal = vit_patch_size_temporal
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.latent_patch_size = tuple(latent_patch_size)
        self.max_latent_size = max_latent_size
        self.max_num_frames = max_num_frames
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.timestep_shift = timestep_shift
        self.time_dist_video = time_dist_video
        self.action_time_schedule = action_time_schedule
        self.time_dist_action = time_dist_action
        self.apply_qwen_2_5_vl_pos_emb = apply_qwen_2_5_vl_pos_emb
        self.validation_timestep_shift = validation_timestep_shift
        self.num_inference_steps = num_inference_steps
        self.video_loss_weight = video_loss_weight
        self.action_loss_weight = action_loss_weight
        self.freeze_understanding = freeze_understanding
        self.freeze_vit = freeze_vit
        self.freeze_vae = freeze_vae
        self.gradient_checkpointing = gradient_checkpointing
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_space_schema_id = action_space_schema_id
        # Default generation lengths when inference has no future targets.
        # Training reads these lengths from its tensors and never mutates them.
        self.action_horizon = action_horizon
        self.action_position_mode = str(action_position_mode)
        self.num_video_frames = num_video_frames
        self.num_video_latent_frames = num_video_latent_frames

        super().__init__(**kwargs)
        self.num_logical_domains = int(num_logical_domains)
        self.max_text_len = int(max_text_len)
        self.flex_block_size = int(flex_block_size)
        self._validate_me_u0_contract()

    def _validate_me_u0_contract(self) -> None:
        schema_dimensions = {
            None: (26, 26),
            "pi05_state_action_v1": (26, 26),
            "unified_eef_action_v1": (48, 26),
        }
        if self.action_space_schema_id not in schema_dimensions:
            raise ValueError(
                "unsupported action_space_schema_id: "
                f"{self.action_space_schema_id!r}"
            )
        expected_action_dim, expected_state_dim = schema_dimensions[
            self.action_space_schema_id
        ]
        locked = {
            "lance_subdir": (self.lance_subdir, "Lance_3B_Video"),
            "vit_subdir": (self.vit_subdir, "Qwen2.5-VL-ViT"),
            "vae_filename": (self.vae_filename, "Wan2.2_VAE.pth"),
            "layer_module": (self.layer_module, "Qwen2MoTDecoderLayer"),
            "llm_qk_norm": (self.llm_qk_norm, True),
            "llm_qk_norm_und": (self.llm_qk_norm_und, True),
            "llm_qk_norm_gen": (self.llm_qk_norm_gen, True),
            "tie_word_embeddings": (self.tie_word_embeddings, False),
            "vit_type": (self.vit_type, "qwen_2_5_vl_original"),
            "vit_patch_size": (self.vit_patch_size, 14),
            "vit_patch_size_temporal": (self.vit_patch_size_temporal, 2),
            "vit_max_num_patch_per_side": (self.vit_max_num_patch_per_side, 70),
            "connector_act": (self.connector_act, "gelu_pytorch_tanh"),
            "interpolate_pos": (self.interpolate_pos, False),
            "latent_patch_size": (tuple(self.latent_patch_size), (1, 1, 1)),
            "max_latent_size": (self.max_latent_size, 64),
            "max_num_frames": (self.max_num_frames, 41),
            "visual_gen": (self.visual_gen, True),
            "visual_und": (self.visual_und, True),
            "action_dim": (self.action_dim, expected_action_dim),
            "state_dim": (self.state_dim, expected_state_dim),
            "num_logical_domains": (self.num_logical_domains, 17),
            "max_text_len": (self.max_text_len, 512),
            "flex_block_size": (self.flex_block_size, 128),
            "validation_timestep_shift": (self.validation_timestep_shift, 3.0),
            "num_inference_steps": (self.num_inference_steps, 24),
            "video_loss_weight": (self.video_loss_weight, 10.0),
            "action_loss_weight": (self.action_loss_weight, 10.0),
            "gradient_checkpointing": (self.gradient_checkpointing, False),
            "pretrained_strict": (self.pretrained_strict, True),
        }
        bad = [f"{name}={value!r} (required {expected!r})"
               for name, (value, expected) in locked.items() if value != expected]
        if bad:
            raise ValueError("ME_U0 video/action geometry is locked: " + ", ".join(bad))

        if self.time_dist_video != "logitnormal":
            raise ValueError("ME_U0 video flow time must be logitnormal")
        if float(self.timestep_shift) != 4.0:
            raise ValueError("ME_U0 training timestep_shift must be 4.0")
        if self.action_time_schedule != "shared":
            raise ValueError("video and action must use one shared timestep")
        if self.time_dist_action != "logitnormal":
            raise ValueError("the locked action time distribution is logitnormal")
        if self.action_position_mode not in {
            "legacy_grouped",
            "group_substep_mrope",
        }:
            raise ValueError(
                "action_position_mode must be legacy_grouped or "
                "group_substep_mrope"
            )
        if self.num_video_frames < 2:
            raise ValueError("num_video_frames must be at least 2")
        if (self.num_video_frames - 1) % 4:
            raise ValueError(
                "num_video_frames must be 1 + 4k for the Wan VAE temporal "
                f"geometry, got {self.num_video_frames}"
            )
        expected_video_latents = 1 + (self.num_video_frames - 1) // 4
        if self.num_video_latent_frames != expected_video_latents:
            raise ValueError(
                "num_video_latent_frames does not match num_video_frames: "
                f"got {self.num_video_latent_frames}, expected "
                f"{expected_video_latents} for {self.num_video_frames} RGB frames"
            )
        future_video_latents = self.num_video_latent_frames - 1
        if self.action_horizon <= 0 or self.action_horizon % future_video_latents:
            raise ValueError(
                "action_horizon must be positive and divisible by the number "
                "of future video latents; got action_horizon="
                f"{self.action_horizon}, future_video_latents={future_video_latents}"
            )
        if not self.freeze_vae:
            raise ValueError("Wan VAE remains frozen on every retained ME_U0 route")
        if self.freeze_understanding:
            raise ValueError("retained ME_U0 routes keep understanding unfrozen")
        if self.main_image_size != 224 or self.main_image_size % 28:
            raise ValueError("ViT main image must be 224 and divisible by 28")
        if not self.apply_qwen_2_5_vl_pos_emb:
            raise ValueError("ME_U0 training keeps the audited Lance mRoPE path enabled")
