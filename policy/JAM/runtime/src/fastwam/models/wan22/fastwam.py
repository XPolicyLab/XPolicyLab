from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .foresight import (
    ActionForesightHead,
    SpatialActionForesightHead,
    ForesightProjector,
    SharedForesightProjector,
    MLPForesightProjector,
    build_foresight_encoder,
    build_foresight_ema_shadow,
    build_random_foresight_encoder,
    build_random_foresight_vae,
    encode_foresight_patch_embedding,
    encode_foresight_hidden,
    encode_foresight_hidden_multi,
    encode_foresight_vae_direct,
    fuse_future_targets,
    lift_foresight_vae_patches,
    normalize_foresight_target_type,
    normalize_foresight_teacher_mode,
    per_frame_targets,
    update_foresight_ema_teacher_,
)
from .helpers.loader import (
    load_wan22_pretrained_video_dit_prefix,
    load_wan22_ti2v_5b_components,
)
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

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
        foresight_encoder=None,
        foresight_vae=None,
        foresight_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        foresight_cfg = dict(foresight_cfg or {})
        self.foresight_enabled = bool(foresight_cfg.get("enabled", False))
        self.foresight_video_cotrain = bool(foresight_cfg.get("video_cotrain", True))
        self.foresight_target_type = normalize_foresight_target_type(
            foresight_cfg.get("target_type", None)
        )
        self.foresight_teacher_mode = normalize_foresight_teacher_mode(
            foresight_cfg.get("teacher_mode", None)
        )
        self.foresight_teacher_random_seed = int(
            foresight_cfg.get("teacher_random_seed", 0)
        )
        self.foresight_teacher_ema_decay = float(
            foresight_cfg.get("teacher_ema_decay", 0.995)
        )
        if not 0.0 <= self.foresight_teacher_ema_decay < 1.0:
            raise ValueError(
                "`foresight.teacher_ema_decay` must be in [0, 1), got "
                f"{self.foresight_teacher_ema_decay}."
            )
        self._foresight_ema_shadow = None
        self._foresight_ema_updates = 0
        # `layers` (list) supersedes `layer` (single int); each student layer L_i is paired
        # with encoder layer encoder_layers[i] at the same relative depth by default.
        if "layers" in foresight_cfg and foresight_cfg["layers"] is not None:
            self.foresight_layers = [int(x) for x in foresight_cfg["layers"]]
        else:
            self.foresight_layers = [int(foresight_cfg.get("layer", 15))]
        enc_layers = foresight_cfg.get("encoder_layers", foresight_cfg.get("encoder_layer", None))
        if enc_layers is None:
            self.foresight_encoder_layers = list(self.foresight_layers)
        elif isinstance(enc_layers, (list, tuple)):
            self.foresight_encoder_layers = [int(x) for x in enc_layers]
        else:
            self.foresight_encoder_layers = [int(enc_layers)] * len(self.foresight_layers)
        if len(self.foresight_encoder_layers) != len(self.foresight_layers):
            raise ValueError(
                f"`foresight.encoder_layers` ({self.foresight_encoder_layers}) must match "
                f"length of `foresight.layers` ({self.foresight_layers})."
            )
        offsets = foresight_cfg.get("offsets", None)
        self.foresight_offsets = None if offsets is None else [int(o) for o in offsets]
        self.loss_lambda_foresight = float(foresight_cfg.get("lambda_foresight", 0.2))
        # Optional per-offset (per future-frame) weighting for per_frame foresight: aligned by
        # POSITION to `offsets`. Lets you tilt the foresight budget toward farther-future frames
        # (long-horizon world modeling, e.g. [0.2,0.3,0.5]) instead of equal-weighting them.
        # Normalized to mean 1.0 so the overall loss scale (and the meaning of lambda) is
        # unchanged; only the relative split across offsets moves. None -> equal weights.
        ow = foresight_cfg.get("offset_weights", None)
        if ow is None:
            self.foresight_offset_weights = None
        else:
            ow = [float(w) for w in ow]
            if self.foresight_offsets is not None and len(ow) != len(self.foresight_offsets):
                raise ValueError(
                    f"`foresight.offset_weights` ({ow}) must match length of `foresight.offsets` "
                    f"({self.foresight_offsets})."
                )
            mean_w = sum(ow) / len(ow)
            if mean_w <= 0:
                raise ValueError(f"`foresight.offset_weights` must have positive mean, got {ow}.")
            self.foresight_offset_weights = [w / mean_w for w in ow]
        # Shared projector backbone + lightweight per-target heads (avoids the
        # num_layers x num_offsets full-projector explosion under per_frame).
        self.foresight_shared_projector = bool(foresight_cfg.get("shared_projector", False))
        # Idea 1: per-frame multi-target foresight (predict each future latent frame separately
        # instead of fusing into one averaged target — preserves temporal structure).
        self.foresight_per_frame = bool(foresight_cfg.get("per_frame", False))
        # Idea 2: action-stream foresight (align action-expert hidden states to future visual
        # features; action group i -> latent frame i+1, spatially pooled).
        action_layers = foresight_cfg.get("action_layers", None)
        self.foresight_action_layers = None if action_layers is None else [int(x) for x in action_layers]
        self.loss_lambda_action_foresight = float(
            foresight_cfg.get("lambda_action_foresight", self.loss_lambda_foresight)
        )
        # action_per_token: align the action group to the future frame's FULL spatial token map
        # (per-token, like the video stream) instead of a single spatially-pooled vector. Restores
        # the spatial richness that made video-stream foresight work. Default False = pooled (legacy).
        self.foresight_action_per_token = bool(foresight_cfg.get("action_per_token", False))
        # action_per_step: do NOT mean-pool the ~H/num_groups action steps of a transition into one
        # vector; instead each action step's hidden separately predicts that transition's future
        # frame target. Restores the temporal richness lost by the group averaging (the action-input
        # analog of per-token). Uses the pooled frame target; mutually exclusive with action_per_token.
        self.foresight_action_per_step = bool(foresight_cfg.get("action_per_step", False))
        if self.foresight_action_per_step and self.foresight_action_per_token:
            raise ValueError(
                "`foresight.action_per_step` and `foresight.action_per_token` are mutually exclusive "
                "(per-step un-pools the action input; per-token un-pools the frame target). Pick one."
            )
        # The frozen encoder is intentionally kept out of the module registry so it is
        # never trained, checkpointed, or sharded; it is reloaded from pretrained weights
        # at construction time instead.
        self._foresight_encoder_ref = None
        self._foresight_vae_ref = None
        self.foresight_projectors = None
        # When targets come from the precomputed bf16+zstd cache, the frozen encoder
        # is not instantiated at all; batches must then carry `foresight_enc_cache`.
        self.foresight_use_target_cache = bool(foresight_cfg.get("target_cache_roots"))
        cache_roots = foresight_cfg.get("target_cache_roots")
        evaluation_teacher_disabled = (
            isinstance(cache_roots, (list, tuple))
            and list(cache_roots) == ["__evaluation_only_no_teacher__"]
        )
        cacheable_target_types = {"encoder_hidden", "vae_direct"}
        if (
            self.foresight_enabled
            and self.foresight_use_target_cache
            and not evaluation_teacher_disabled
            and (
                self.foresight_teacher_mode != "frozen_pretrained"
                or self.foresight_target_type not in cacheable_target_types
            )
        ):
            raise ValueError(
                "Cached foresight targets support only frozen-pretrained hidden states or "
                "frozen-pretrained VAE-direct patches; "
                f"they cannot be used with teacher_mode={self.foresight_teacher_mode!r}, "
                f"target_type={self.foresight_target_type!r}."
            )
        if self.foresight_enabled:
            target_needs_encoder = self.foresight_target_type in {
                "encoder_hidden", "patch_embedding"
            }
            if (
                target_needs_encoder
                and foresight_encoder is None
                and not self.foresight_use_target_cache
            ):
                raise ValueError(
                    f"`foresight_encoder` is required for target_type={self.foresight_target_type!r}."
                )
            if self.foresight_target_type == "patch_embedding":
                if self.foresight_teacher_mode != "frozen_random":
                    raise ValueError(
                        "Random Layer0 (`target_type=patch_embedding`) requires "
                        "`teacher_mode=frozen_random`."
                    )
                if any(layer != 0 for layer in self.foresight_encoder_layers):
                    raise ValueError(
                        "`target_type=patch_embedding` requires `encoder_layers=[0]`."
                    )
            elif self.foresight_target_type == "vae_direct":
                if any(layer != 0 for layer in self.foresight_encoder_layers):
                    raise ValueError(
                        "`target_type=vae_direct` requires `encoder_layers=[0]`."
                    )
                if foresight_encoder is not None:
                    raise ValueError("VAE-direct targets must not instantiate a foresight encoder.")
                if self.foresight_teacher_mode == "online_ema":
                    raise ValueError(
                        "VAE-direct targets do not support teacher_mode=online_ema; "
                        "use frozen_pretrained or frozen_random."
                    )
                if (
                    self.foresight_teacher_mode == "frozen_random"
                    and foresight_vae is None
                    and not evaluation_teacher_disabled
                ):
                    raise ValueError(
                        "Random VAE-direct targets require a frozen random teacher VAE."
                    )
            num_blocks = len(video_expert.blocks)
            for L in self.foresight_layers:
                if not (1 <= L <= num_blocks):
                    raise ValueError(f"`foresight.layers` entry must be in [1, {num_blocks}], got {L}")
            if self.foresight_target_type == "encoder_hidden":
                enc_num_blocks = len(foresight_encoder.blocks) if foresight_encoder is not None else num_blocks
                for L in self.foresight_encoder_layers:
                    if not (1 <= L <= enc_num_blocks):
                        raise ValueError(
                            f"`foresight.encoder_layers` entry must be in [1, {enc_num_blocks}], got {L}"
                        )
            self._foresight_encoder_ref = [foresight_encoder] if foresight_encoder is not None else None
            self._foresight_vae_ref = [foresight_vae] if foresight_vae is not None else None
            # Encoder is a frozen copy of the video expert, so its hidden dim equals the
            # expert's; resolve explicitly so cache mode (encoder=None) can build projectors.
            enc_out_dim = int(foresight_encoder.hidden_dim) if foresight_encoder is not None else int(video_expert.hidden_dim)
            if self.foresight_per_frame and self.foresight_offsets is None:
                raise ValueError("`foresight.per_frame=true` requires explicit `foresight.offsets` (e.g. [0,1,2]).")
            # Projector keys: per-frame mode uses "{layer}_{offset}" so each (depth, future
            # frame) gets its own head; fused mode uses "{layer}".
            if self.foresight_per_frame:
                proj_keys = [f"{L}_{o}" for L in self.foresight_layers for o in self.foresight_offsets]
            else:
                proj_keys = [str(L) for L in self.foresight_layers]
            if self.foresight_shared_projector:
                # One shared backbone + a cheap per-key head. projector_type selects the backbone:
                #   "transformer" (default) = SharedForesightProjector (token-mixing DiT trunk);
                #   "mlp" = MLPForesightProjector (REPA / Efficient-WAM style: pointwise low-dim
                #           projection, no attention, ~10-40M instead of ~0.45B).
                projector_type = str(foresight_cfg.get("projector_type", "transformer")).lower()
                if projector_type == "mlp":
                    shared_projector = MLPForesightProjector(
                        in_dim=int(video_expert.hidden_dim),
                        out_dim=enc_out_dim,
                        keys=proj_keys,
                        hidden=int(foresight_cfg.get("projector_mlp_hidden", 2048)),
                    ).to(torch_dtype)
                elif projector_type == "transformer":
                    shared_projector = SharedForesightProjector(
                        in_dim=int(video_expert.hidden_dim),
                        out_dim=enc_out_dim,
                        keys=proj_keys,
                        hidden_dim=foresight_cfg.get("projector_hidden_dim", None),
                        num_layers=int(foresight_cfg.get("projector_num_layers", 3)),
                        num_heads=int(foresight_cfg.get("projector_num_heads", video_expert.num_heads)),
                        ffn_dim=foresight_cfg.get("projector_ffn_dim", None),
                    ).to(torch_dtype)
                else:
                    raise ValueError(
                        f"`foresight.projector_type` must be 'transformer' or 'mlp', got {projector_type!r}"
                    )
                self.foresight_projectors = nn.ModuleDict({"shared": shared_projector})
            else:
                self.foresight_projectors = nn.ModuleDict(
                    {
                        k: ForesightProjector(
                            in_dim=int(video_expert.hidden_dim),
                            out_dim=enc_out_dim,
                            hidden_dim=foresight_cfg.get("projector_hidden_dim", None),
                            num_layers=int(foresight_cfg.get("projector_num_layers", 3)),
                            num_heads=int(foresight_cfg.get("projector_num_heads", video_expert.num_heads)),
                            ffn_dim=foresight_cfg.get("projector_ffn_dim", None),
                        ).to(torch_dtype)
                        for k in proj_keys
                    }
                )
            # Action-stream foresight heads (Idea 2): one per supervised action layer.
            self.foresight_action_heads = None
            if self.foresight_action_layers is not None:
                for L in self.foresight_action_layers:
                    if not (1 <= L <= len(action_expert.blocks)):
                        raise ValueError(
                            f"`foresight.action_layers` entry must be in [1, {len(action_expert.blocks)}], got {L}"
                        )
                _ActionHead = (
                    SpatialActionForesightHead if self.foresight_action_per_token else ActionForesightHead
                )
                self.foresight_action_heads = nn.ModuleDict(
                    {
                        str(L): _ActionHead(
                            in_dim=int(action_expert.hidden_dim),
                            out_dim=enc_out_dim,
                        ).to(torch_dtype)
                        for L in self.foresight_action_layers
                    }
                )
            logger.info(
                "Foresight: layers=%s enc=%s target=%s offsets=%s offset_weights=%s per_frame=%s action_layers=%s action_per_token=%s action_per_step=%s lambda=%.3f/%.3f cotrain=%s teacher=%s",
                self.foresight_layers,
                self.foresight_encoder_layers,
                self.foresight_target_type,
                "all" if self.foresight_offsets is None else self.foresight_offsets,
                "equal" if self.foresight_offset_weights is None
                else [round(w, 3) for w in self.foresight_offset_weights],
                self.foresight_per_frame,
                self.foresight_action_layers,
                self.foresight_action_per_token,
                self.foresight_action_per_step,
                self.loss_lambda_foresight,
                self.loss_lambda_action_foresight,
                self.foresight_video_cotrain,
                self.foresight_teacher_mode,
            )
            if (
                self.foresight_teacher_mode == "online_ema"
                and not self.foresight_video_cotrain
            ):
                logger.info(
                    "Online EMA teacher tracks the policy video expert under the "
                    "foresight-only objective (video_cotrain=false); no generative "
                    "video loss has been added."
                )

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
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)
        if self.foresight_teacher_mode == "online_ema" and self.foresight_encoder is not None:
            self._initialize_foresight_ema_shadow()

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
        skip_video_dit_load_from_pretrain: Optional[bool] = None,
        video_dit_random_seed: Optional[int] = None,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        foresight_cfg: Optional[dict] = None,
        action_lora_config: Optional[dict] = None,
        action_success_distillation: Optional[dict] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        # The legacy flag continues to skip both experts, which evaluation uses
        # before loading a complete policy checkpoint. The video-only override
        # enables a clean random-student ablation while retaining the pretrained
        # ActionDiT backbone.
        skip_video_pretrain = (
            bool(skip_dit_load_from_pretrain)
            if skip_video_dit_load_from_pretrain is None
            else bool(skip_video_dit_load_from_pretrain)
        )
        skip_action_pretrain = bool(skip_dit_load_from_pretrain)

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_video_pretrain,
            dit_random_seed=video_dit_random_seed,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_action_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        logger.info(
            "FastWAM expert initialization: video=%s action=%s.",
            (
                f"random(seed={int(video_dit_random_seed)})"
                if skip_video_pretrain and video_dit_random_seed is not None
                else ("random(run_rng)" if skip_video_pretrain else "pretrained")
            ),
            "random(run_rng)" if skip_action_pretrain else "pretrained",
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        foresight_cfg = dict(foresight_cfg or {})
        foresight_encoder = None
        foresight_vae = None
        foresight_teacher_path = None
        if foresight_cfg.get("enabled", False):
            if foresight_cfg.get("target_cache_roots"):
                # Targets come from the precomputed bf16+zstd cache: skip the 5B
                # frozen-encoder copy entirely (saves ~10GB/GPU and load time).
                logger.info(
                    "Foresight target cache enabled (%s): frozen encoder NOT instantiated.",
                    foresight_cfg["target_cache_roots"],
                )
            else:
                target_type = normalize_foresight_target_type(
                    foresight_cfg.get("target_type", None)
                )
                teacher_mode = normalize_foresight_teacher_mode(
                    foresight_cfg.get("teacher_mode", None)
                )
                student_layers = foresight_cfg.get("layers")
                if student_layers is None:
                    student_layers = [foresight_cfg.get("layer", 15)]
                elif not isinstance(student_layers, (list, tuple)):
                    student_layers = [student_layers]
                encoder_layers = foresight_cfg.get(
                    "encoder_layers", foresight_cfg.get("encoder_layer", None)
                )
                if encoder_layers is None:
                    encoder_layers = student_layers
                elif not isinstance(encoder_layers, (list, tuple)):
                    encoder_layers = [encoder_layers]
                max_encoder_layer = max(int(layer) for layer in encoder_layers)

                # Copy/build now, before any fine-tuning or checkpoint loading.
                if target_type == "vae_direct":
                    if teacher_mode == "online_ema":
                        raise ValueError(
                            "VAE-direct targets do not support teacher_mode=online_ema."
                        )
                    if teacher_mode == "frozen_random":
                        foresight_vae = build_random_foresight_vae(
                            components.vae,
                            seed=int(foresight_cfg.get("teacher_random_seed", 0)),
                            device=device,
                            torch_dtype=torch_dtype,
                        )
                        foresight_teacher_path = (
                            "RANDOM_VAE(seed="
                            f"{int(foresight_cfg.get('teacher_random_seed', 0))})"
                        )
                    else:
                        logger.info(
                            "Pretrained VAE-direct foresight target enabled: no Wan teacher "
                            "is instantiated."
                        )
                elif target_type == "patch_embedding":
                    if teacher_mode != "frozen_random" or max_encoder_layer != 0:
                        raise ValueError(
                            "Random Layer0 requires target_type=patch_embedding, "
                            "teacher_mode=frozen_random, encoder_layers=[0]."
                        )
                    foresight_encoder = build_random_foresight_encoder(
                        video_dit_config,
                        max_layers=0,
                        seed=int(foresight_cfg.get("teacher_random_seed", 0)),
                        device=device,
                        torch_dtype=torch_dtype,
                    )
                elif teacher_mode == "frozen_random":
                    foresight_encoder = build_random_foresight_encoder(
                        video_dit_config,
                        max_layers=max_encoder_layer,
                        seed=int(foresight_cfg.get("teacher_random_seed", 0)),
                        device=device,
                        torch_dtype=torch_dtype,
                    )
                elif teacher_mode == "frozen_pretrained" and skip_video_pretrain:
                    pretrained_teacher, foresight_teacher_path = (
                        load_wan22_pretrained_video_dit_prefix(
                            device=device,
                            torch_dtype=torch_dtype,
                            model_id=model_id,
                            dit_config=video_dit_config,
                            max_layers=max_encoder_layer,
                        )
                    )
                    foresight_encoder = build_foresight_encoder(
                        pretrained_teacher,
                        max_layers=max_encoder_layer,
                        copy_model=False,
                        source="independently loaded pretrained",
                    )
                else:
                    if skip_video_pretrain:
                        logger.info(
                            "Online EMA foresight teacher is initialized from the random "
                            "policy video expert."
                        )
                    foresight_encoder = build_foresight_encoder(
                        video_expert,
                        max_layers=(max_encoder_layer if teacher_mode == "online_ema" else None),
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
            foresight_encoder=foresight_encoder,
            foresight_vae=foresight_vae,
            foresight_cfg=foresight_cfg,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_action_pretrain else action_dit_pretrained_path
            ),
            "foresight_teacher": foresight_teacher_path,
        }
        if action_lora_config or action_success_distillation:
            from .success_distill import attach_action_success_distillation

            attach_action_success_distillation(
                model,
                lora_cfg=action_lora_config,
                distill_cfg=action_success_distillation,
            )
        return model

    @property
    def foresight_encoder(self):
        if self._foresight_encoder_ref is None:
            return None
        return self._foresight_encoder_ref[0]

    @property
    def foresight_vae(self):
        if self._foresight_vae_ref is None:
            return None
        return self._foresight_vae_ref[0]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if self.foresight_encoder is not None:
            self.foresight_encoder.to(*args, **kwargs)
        if self.foresight_vae is not None:
            self.foresight_vae.to(*args, **kwargs)
        if self._foresight_ema_shadow is not None and self.foresight_encoder is not None:
            teacher_device = next(self.foresight_encoder.parameters()).device
            self._foresight_ema_shadow = {
                name: value.to(device=teacher_device, dtype=torch.float32)
                for name, value in self._foresight_ema_shadow.items()
            }
        return self

    def _initialize_foresight_ema_shadow(self):
        if self.foresight_encoder is None:
            raise ValueError("Cannot initialize EMA shadow without a foresight encoder.")
        self._foresight_ema_shadow = build_foresight_ema_shadow(self.foresight_encoder)
        self._foresight_ema_updates = 0
        shadow_numel = sum(value.numel() for value in self._foresight_ema_shadow.values())
        logger.info(
            "Initialized fp32 foresight EMA shadow: %.2f B values, %.2f GiB, decay=%.6f.",
            shadow_numel / 1e9,
            shadow_numel * 4 / (1024**3),
            self.foresight_teacher_ema_decay,
        )

    @torch.no_grad()
    def update_foresight_teacher(self) -> bool:
        """Update the online EMA target after one successful optimizer step."""
        if self.foresight_teacher_mode != "online_ema":
            return False
        if self.foresight_encoder is None or self._foresight_ema_shadow is None:
            raise RuntimeError("Online EMA teacher is enabled but its encoder/shadow is absent.")
        update_foresight_ema_teacher_(
            self.foresight_encoder,
            self.video_expert,
            self._foresight_ema_shadow,
            decay=self.foresight_teacher_ema_decay,
        )
        self._foresight_ema_updates += 1
        if self._foresight_ema_updates == 1:
            logger.info("Applied first online foresight-teacher EMA update.")
        return True

    @torch.no_grad()
    def reset_foresight_teacher_from_student(self) -> bool:
        """Restart EMA from current student weights after a weight/state resume.

        The unregistered teacher is intentionally absent from normal checkpoints.
        This gives a coherent restart, but it cannot reconstruct pre-resume EMA
        history; callers emit an explicit warning when taking this path.
        """
        if self.foresight_teacher_mode != "online_ema" or self.foresight_encoder is None:
            return False
        student_parameters = dict(self.video_expert.named_parameters())
        for name, teacher_parameter in self.foresight_encoder.named_parameters():
            source = student_parameters.get(name)
            if source is None or source.shape != teacher_parameter.shape:
                raise ValueError(f"Cannot reset EMA teacher: incompatible student parameter {name!r}.")
            teacher_parameter.copy_(source)
        self._initialize_foresight_ema_shadow()
        return True

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
        # FIXME: original implementation's zero padding is visible in cross-attn.
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
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
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

    def build_inputs(self, sample, tiled: bool = False):
        video = sample.get("video")
        cached_input_latents = sample.get("input_latents_cache")
        if cached_input_latents is not None and video is not None:
            raise ValueError(
                "A sample must provide either RGB `video` or `input_latents_cache`, not both."
            )
        if cached_input_latents is None and video is None:
            raise ValueError(
                "A sample must provide RGB `video` or the exact frozen-VAE "
                "`input_latents_cache`."
            )
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if cached_input_latents is not None:
            if cached_input_latents.ndim != 5:
                raise ValueError(
                    "`sample['input_latents_cache']` must be [B,C,T,H,W], got "
                    f"shape {tuple(cached_input_latents.shape)}."
                )
            batch_size, latent_channels, num_latent_frames, latent_height, latent_width = (
                cached_input_latents.shape
            )
            if latent_channels != int(self.video_expert.in_dim):
                raise ValueError(
                    "Cached input latent channel mismatch: "
                    f"got {latent_channels}, expected {self.video_expert.in_dim}."
                )
            pt, ph, pw = (int(value) for value in self.video_expert.patch_size)
            if (
                num_latent_frames != 1
                or num_latent_frames % pt
                or latent_height % ph
                or latent_width % pw
            ):
                raise ValueError(
                    "Cached policy input must contain one patch-compatible latent frame, "
                    f"got {tuple(cached_input_latents.shape)} with patch size {(pt, ph, pw)}."
                )
            num_frames = 1
        else:
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[1] != 3:
                raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

            batch_size, _, num_frames, height, width = video.shape
            if height % 16 != 0 or width % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
                )
            if num_frames % 4 != 1:
                raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        # first_frame_only (Plan-B + target cache): pixels carry ONE frame; the video
        # branch consumes latents[:, :, 0:1] either way. Any other config needs T > 1.
        _ff_ok = num_frames == 1 and (
            (
                self.foresight_enabled
                and not self.foresight_video_cotrain
                and self.foresight_use_target_cache
            )
            # golden-noise self-distillation consumes only latent frame 0 (deployment prefix),
            # so first_frame_only pixels are exactly what it needs and no video target exists.
            or str(getattr(self, "success_distill_mode", "")) in ("golden", "oracle")
        )
        if num_frames <= 1 and not _ff_ok:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        # In no-cotrain foresight the video branch consumes only latent frame 0 and the
        # clip may extend past the action window (video_future_frames teacher targets),
        # so the action÷transitions divisibility only binds under video co-training.
        _use_video_cotrain = self.foresight_video_cotrain if self.foresight_enabled else True
        if _use_video_cotrain and not _ff_ok and action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
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
            # first_frame_only pipelines carry pixels for 1 frame but pad METADATA for the
            # full video-frame grid; the mask logic consumes the grid, so allow longer-than-video.
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] < num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, >= {num_frames})"
                )

        input_video = (
            None
            if video is None
            else video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        )
        random_vae_direct = (
            self.foresight_enabled
            and self.foresight_target_type == "vae_direct"
            and self.foresight_teacher_mode == "frozen_random"
            and self.foresight_vae is not None
        )
        foresight_target_latents = None
        if cached_input_latents is not None:
            if random_vae_direct:
                raise ValueError(
                    "Cached policy input latents are not supported with a random VAE teacher."
                )
            input_latents = cached_input_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        elif random_vae_direct:
            # The random teacher sees the complete clean clip.  Under the
            # foresight-only objective the student consumes only latent frame
            # zero, so encode only the first RGB frame with the pretrained
            # policy VAE.  Wan's causal VAE makes this first latent identical
            # to encoding the complete clip and slicing [:, :, 0:1].
            with torch.no_grad():
                foresight_target_latents = self.foresight_vae.encode(
                    input_video,
                    device=self.device,
                    tiled=tiled,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
            policy_video = (
                input_video
                if self.foresight_video_cotrain
                else input_video[:, :, 0:1].contiguous()
            )
            input_latents = self._encode_video_latents(policy_video, tiled=tiled)
        else:
            input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "foresight_target_latents": foresight_target_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
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
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
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
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _foresight_frame_valid_mask(
        self,
        image_is_pad: Optional[torch.Tensor],
        num_latent_steps: int,
    ) -> Optional[torch.Tensor]:
        """Map per-frame padding [B, T] to per-latent-step validity [B, f]."""
        if image_is_pad is None:
            return None
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        latent_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        if latent_is_pad.shape[1] != num_latent_steps:
            raise ValueError(
                "Foresight pad-mask shape mismatch: "
                f"mask steps={latent_is_pad.shape[1]}, latent steps={num_latent_steps}."
            )
        return ~latent_is_pad

    @staticmethod
    def _cos_align_loss(pred: torch.Tensor, target: torch.Tensor, sample_valid: torch.Tensor) -> torch.Tensor:
        """1 - cosine over the last dim, mean over tokens, masked mean over the batch."""
        cos = F.cosine_similarity(pred.float(), target.detach().float(), dim=-1)  # [B, tokens]
        per_sample = 1.0 - cos.mean(dim=1)
        valid = sample_valid.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)

    def _compute_foresight_loss(
        self,
        video_hidden_by_layer: dict[int, torch.Tensor],
        action_hidden_by_layer: Optional[dict[int, torch.Tensor]],
        tokens_per_frame: int,
        input_latents: torch.Tensor,
        foresight_target_latents: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        image_is_pad: Optional[torch.Tensor],
        precomputed_enc_cache: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Foresight alignment loss. Returns (video_stream_loss, action_stream_loss).

        Modes (independent, can combine):
          - fused (default): obs tokens align to the offset-averaged encoder target.
          - per_frame: obs tokens align to EACH future frame separately (own head), keeping
            temporal structure (Idea 1).
          - action stream: action-group reps align to spatially-pooled future-frame features,
            group i -> latent frame i+1 (Idea 2).
        Encoder hidden is run once per distinct encoder layer and reused.
        """
        # With precomputed targets the policy may carry only the first latent frame
        # (first_frame_only); the TARGET latent-step count then comes from the cache.
        target_latents = (
            input_latents
            if foresight_target_latents is None
            else foresight_target_latents
        )
        if precomputed_enc_cache is not None:
            _any = next(iter(precomputed_enc_cache.values()))
            num_latent_steps = int(_any.shape[1]) // int(tokens_per_frame)
        else:
            num_latent_steps = int(target_latents.shape[2])
        frame_valid = self._foresight_frame_valid_mask(
            image_is_pad=image_is_pad,
            num_latent_steps=num_latent_steps,
        )
        # Targets either come precomputed (bf16+zstd cache, bit-identical to the
        # online encoder) or from running the frozen encoder per distinct depth.
        enc_cache: dict[int, torch.Tensor] = {}
        if precomputed_enc_cache is not None:
            for enc_layer in set(self.foresight_encoder_layers):
                enc_hidden = precomputed_enc_cache[enc_layer]
                if enc_hidden.shape[1] % tokens_per_frame != 0:
                    raise ValueError(
                        "Cached foresight target token grid mismatch: "
                        f"seq_len={enc_hidden.shape[1]} not divisible by tokens_per_frame={tokens_per_frame}."
                    )
                enc_hidden = enc_hidden.to(
                    device=target_latents.device,
                    dtype=target_latents.dtype,
                )
                if self.foresight_target_type == "vae_direct":
                    enc_hidden = lift_foresight_vae_patches(
                        enc_hidden,
                        output_dim=int(self.video_expert.hidden_dim),
                    )
                enc_cache[enc_layer] = enc_hidden
        else:
            with torch.no_grad():
                if self.foresight_target_type == "encoder_hidden":
                    if self.foresight_encoder is None:
                        raise RuntimeError(
                            "foresight target cache is enabled but the batch has no "
                            "`foresight_enc_cache` (dataset not wrapped with ForesightCacheDataset?)"
                        )
                    # Single pass to max(encoder_layers), capturing every needed depth on
                    # the way — bit-identical to per-layer calls, but e.g. [12,14,16,18]
                    # costs 18 block passes instead of 12+14+16+18=60.
                    enc_cache, enc_tpf = encode_foresight_hidden_multi(
                        encoder=self.foresight_encoder,
                        clean_latents=target_latents,
                        context=context,
                        context_mask=context_mask,
                        capture_layers=set(self.foresight_encoder_layers),
                        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    )
                elif self.foresight_target_type == "patch_embedding":
                    if self.foresight_encoder is None:
                        raise RuntimeError("Random Layer0 target encoder is missing.")
                    enc_hidden, enc_tpf = encode_foresight_patch_embedding(
                        self.foresight_encoder, target_latents
                    )
                    enc_cache = {0: enc_hidden}
                elif self.foresight_target_type == "vae_direct":
                    enc_hidden, enc_tpf = encode_foresight_vae_direct(
                        target_latents,
                        patch_size=self.video_expert.patch_size,
                        output_dim=int(self.video_expert.hidden_dim),
                    )
                    enc_cache = {0: enc_hidden}
                else:  # normalize_foresight_target_type makes this unreachable.
                    raise RuntimeError(
                        f"Unhandled foresight target type {self.foresight_target_type!r}."
                    )
                if enc_tpf != tokens_per_frame:
                    raise ValueError(
                        "Foresight encoder token grid mismatch: "
                        f"encoder={enc_tpf} vs policy={tokens_per_frame}."
                    )

        # ---- video-stream foresight ----
        video_losses = []
        video_weights = []
        for student_layer, enc_layer in zip(self.foresight_layers, self.foresight_encoder_layers):
            h_obs = video_hidden_by_layer[student_layer][:, :tokens_per_frame]
            enc_hidden = enc_cache[enc_layer]
            # Shared mode runs one backbone per student layer, reused across its frame heads.
            shared = self.foresight_projectors["shared"] if self.foresight_shared_projector else None
            feat = shared.backbone(h_obs) if shared is not None else None

            def _predict(key):
                if shared is not None:
                    return shared.head(feat, key)
                return self.foresight_projectors[key](h_obs)

            if self.foresight_per_frame:
                with torch.no_grad():
                    targets, valid = per_frame_targets(enc_hidden, tokens_per_frame, frame_valid)
                for idx, o in enumerate(self.foresight_offsets):
                    pred = _predict(f"{student_layer}_{o}")
                    video_losses.append(self._cos_align_loss(pred, targets[:, o], valid[:, o]))
                    if self.foresight_offset_weights is not None:
                        video_weights.append(self.foresight_offset_weights[idx])
                    else:
                        video_weights.append(1.0)
            else:
                with torch.no_grad():
                    target, sample_valid = fuse_future_targets(
                        enc_hidden, tokens_per_frame, self.foresight_offsets, frame_valid
                    )
                pred = _predict(str(student_layer))
                video_losses.append(self._cos_align_loss(pred, target, sample_valid))
                video_weights.append(1.0)
        losses_v = torch.stack(video_losses)
        w_v = losses_v.new_tensor(video_weights)
        loss_video_stream = (losses_v * w_v).sum() / w_v.sum()

        # ---- action-stream foresight (Idea 2) ----
        if self.foresight_action_heads is None or action_hidden_by_layer is None:
            return loss_video_stream, loss_video_stream.new_zeros(())

        # Action group i (of num_groups = num_latent_frames - 1) aligns to latent frame i+1.
        # per_token=False: align to the frame's spatially-pooled single vector (legacy).
        # per_token=True : align to the frame's FULL spatial token map (richer, like video stream).
        num_latent_frames = int(input_latents.shape[2])
        num_groups = num_latent_frames - 1
        per_token = self.foresight_action_per_token
        per_step = self.foresight_action_per_step
        action_losses = []
        with torch.no_grad():
            # per-token target: [B, f, tokens_per_frame, D_enc]; pooled: [B, f, D_enc]; valid: [B, f].
            targets_by_enc = {}
            for enc_layer in set(self.foresight_encoder_layers):
                tg, vd = per_frame_targets(enc_cache[enc_layer], tokens_per_frame, frame_valid)
                targets_by_enc[enc_layer] = (tg, tg.mean(dim=2), vd)
        for a_layer in self.foresight_action_layers:
            a_hidden = action_hidden_by_layer[a_layer]  # [B, action_horizon, D_action]
            ah = a_hidden.shape[1]
            if ah % num_groups != 0:
                raise ValueError(
                    f"action_horizon {ah} not divisible by num_groups {num_groups} for action foresight."
                )
            steps_per_group = ah // num_groups
            # use the first configured encoder layer as the action target source
            enc_layer = self.foresight_encoder_layers[0]
            tg_full, pooled, vd = targets_by_enc[enc_layer]
            head = self.foresight_action_heads[str(a_layer)]
            if per_step:
                # No within-group averaging: each of the steps_per_group action steps predicts its
                # transition's (pooled) future frame. Treat the steps as the "tokens" dim of the loss.
                a_grouped = a_hidden.view(a_hidden.shape[0], num_groups, steps_per_group, a_hidden.shape[2])
                for i in range(num_groups):
                    pred_i = head(a_grouped[:, i])                                   # [B, S, D_enc]
                    tgt_i = pooled[:, i + 1].unsqueeze(1).expand(-1, steps_per_group, -1)  # [B, S, D_enc]
                    action_losses.append(self._cos_align_loss(pred_i, tgt_i, vd[:, i + 1]))
            else:
                grp = a_hidden.view(a_hidden.shape[0], num_groups, steps_per_group, a_hidden.shape[2]).mean(dim=2)
                for i in range(num_groups):
                    if per_token:
                        pred_i = head(grp[:, i], tokens_per_frame)  # [B, tokens_per_frame, D_enc]
                        tgt_i = tg_full[:, i + 1]                    # [B, tokens_per_frame, D_enc]
                    else:
                        pred_i = head(grp[:, i]).unsqueeze(1)        # [B, 1, D_enc]
                        tgt_i = pooled[:, i + 1].unsqueeze(1)        # [B, 1, D_enc]
                    action_losses.append(self._cos_align_loss(pred_i, tgt_i, vd[:, i + 1]))
        loss_action_stream = torch.stack(action_losses).mean()
        return loss_video_stream, loss_action_stream

    def training_loss(self, sample, tiled: bool = False):
        if getattr(self, "action_success_distillation_enabled", False):
            if str(getattr(self, "success_distill_mode", "oracle")) == "golden":
                return self._training_loss_golden_distillation(sample, tiled=tiled)
            return self._training_loss_success_distillation(sample, tiled=tiled)
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # Plan B (foresight without video co-training): the video branch sees only the
        # clean first frame at t=0, exactly matching the `infer_action` deployment path.
        use_video_cotrain = self.foresight_video_cotrain if self.foresight_enabled else True

        target_video = None
        if use_video_cotrain:
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]
            video_input = latents
            video_action_cond = action
        else:
            if inputs["first_frame_latents"] is not None:
                video_input = inputs["first_frame_latents"]
            else:
                video_input = input_latents[:, :, 0:1]
            timestep_video = torch.zeros(
                (batch_size,), dtype=input_latents.dtype, device=self.device
            )
            video_action_cond = None

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=video_input,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=video_action_cond,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        mot_kwargs = dict(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        video_hidden_by_layer = None
        action_hidden_by_layer = None
        if self.foresight_enabled:
            if self.foresight_action_layers is not None:
                tokens_out, video_hidden_by_layer, action_hidden_by_layer = self.mot(
                    **mot_kwargs,
                    return_video_hidden_layers=self.foresight_layers,
                    return_action_hidden_layers=self.foresight_action_layers,
                )
            else:
                tokens_out, video_hidden_by_layer = self.mot(
                    **mot_kwargs,
                    return_video_hidden_layers=self.foresight_layers,
                )
        else:
            tokens_out = self.mot(**mot_kwargs)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if use_video_cotrain:
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]

            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
            )
            loss_video = (loss_video_per_sample * video_weight).mean()
        else:
            loss_video = torch.zeros((), device=self.device, dtype=torch.float32)

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

        if self.foresight_enabled:
            # Precomputed targets (bf16+zstd cache): batch carries [B, L, seq, D] in
            # the order of self.foresight_encoder_layers (ForesightCacheDataset
            # guarantees the ordering); split into the per-layer dict the loss expects.
            precomputed_enc_cache = None
            cached = sample.get("foresight_enc_cache")
            if cached is not None:
                if cached.ndim != 4 or cached.shape[1] != len(self.foresight_encoder_layers):
                    raise ValueError(
                        f"`foresight_enc_cache` must be [B, {len(self.foresight_encoder_layers)}, seq, D], "
                        f"got {tuple(cached.shape)}"
                    )
                precomputed_enc_cache = {
                    L: cached[:, i] for i, L in enumerate(self.foresight_encoder_layers)
                }
            loss_foresight, loss_action_foresight = self._compute_foresight_loss(
                video_hidden_by_layer=video_hidden_by_layer,
                action_hidden_by_layer=action_hidden_by_layer,
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                input_latents=input_latents,
                foresight_target_latents=inputs["foresight_target_latents"],
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                image_is_pad=image_is_pad,
                precomputed_enc_cache=precomputed_enc_cache,
            )
            loss_total = loss_total + self.loss_lambda_foresight * loss_foresight
            loss_dict["loss_foresight"] = self.loss_lambda_foresight * float(loss_foresight.detach().item())
            if self.foresight_action_heads is not None:
                loss_total = loss_total + self.loss_lambda_action_foresight * loss_action_foresight
                loss_dict["loss_action_foresight"] = self.loss_lambda_action_foresight * float(
                    loss_action_foresight.detach().item()
                )

        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @staticmethod
    def _action_loss_per_sample(pred_action, target_action, action_is_pad):
        token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=token.device, dtype=token.dtype)
            return (token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return token.mean(dim=1)

    def _training_loss_success_distillation(self, sample, tiled: bool = False):
        """Test-adapted objective: correct released-policy failures, keep its successes.

        Mirrors the deployed `infer_action` path exactly: the frozen video expert
        prefills the current-frame prefix once at t=0, the action expert is
        trained on the inference timestep grid (deployment shift and step count)
        with the deployment seed's initial action noise.  Oracle failure episodes
        receive the expert flow-matching target; every other sample (original
        replay and released-success anchors) is distilled toward the frozen
        released vector field, obtained by running the same action expert with
        its LoRA adapters disabled.
        """
        from .lora import lora_disabled

        for key in ("sample_index", "dataset_index", "episode_index"):
            if key not in sample:
                raise RuntimeError(f"success distillation needs mixed-root provenance field `{key}`")
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_horizon, action_dim = int(action.shape[1]), int(action.shape[2])

        if inputs["first_frame_latents"] is not None:
            video_input = inputs["first_frame_latents"]
        else:
            video_input = input_latents[:, :, 0:1]
        timestep_video = torch.zeros((batch_size,), dtype=input_latents.dtype, device=self.device)
        with torch.no_grad():
            video_pre = self.video_expert.pre_dit(
                x=video_input,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )

        # Deployment samples one CPU float32 noise tensor per replan from a
        # generator seeded with the run seed; reproduce it exactly.
        noise_key = (action_horizon, action_dim, str(self.device), str(action.dtype), int(self.success_distill_noise_seed))
        if getattr(self, "_success_distill_noise_key", None) != noise_key:
            generator = torch.Generator(device="cpu").manual_seed(int(self.success_distill_noise_seed))
            self._success_distill_noise = torch.randn(
                (1, action_horizon, action_dim), generator=generator, device="cpu", dtype=torch.float32
            ).to(device=self.device, dtype=action.dtype)
            self._success_distill_noise_key = noise_key
        noise_action = self._success_distill_noise.expand(batch_size, -1, -1)

        num_steps = int(self.success_distill_num_inference_steps)
        infer_timesteps, _ = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=self.success_distill_sigma_shift,
        )
        sample_index = sample["sample_index"].to(device=self.device, dtype=torch.long)
        timestep_action = infer_timesteps.index_select(0, torch.remainder(sample_index, num_steps))
        noisy_action = self.infer_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.infer_action_scheduler.training_target(action, noise_action, timestep_action)

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        def run_action_branch():
            tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            return self.action_expert.post_dit(tokens, action_pre)

        with torch.no_grad(), lora_disabled(self.action_expert):
            pred_base = run_action_branch().detach()
        pred_action = run_action_branch()

        if str(getattr(self, "success_distill_correction_sampling", "deployment_grid")) == "random":
            # Standard flow matching for the expert (correction) target: fresh Gaussian noise and a
            # training-distribution timestep per sample, so new skills are learned for ANY
            # deployment noise/seed rather than only along the fixed deployment noise path.
            noise_rand = torch.randn_like(action)
            t_rand = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=action.dtype
            )
            noisy_rand = self.train_action_scheduler.add_noise(action, noise_rand, t_rand)
            target_rand = self.train_action_scheduler.training_target(action, noise_rand, t_rand)
            pre_rand = self.action_expert.pre_dit(
                action_tokens=noisy_rand, timestep=t_rand, context=context, context_mask=context_mask
            )
            tokens_rand = self.mot.forward_action_with_video_cache(
                action_tokens=pre_rand["tokens"],
                action_freqs=pre_rand["freqs"],
                action_t_mod=pre_rand["t_mod"],
                action_context_payload={"context": pre_rand["context"], "mask": pre_rand["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_rand = self.action_expert.post_dit(tokens_rand, pre_rand)
            weight_rand = self.train_action_scheduler.training_weight(t_rand).to(
                device=pred_rand.device, dtype=torch.float32
            )
            correction = self._action_loss_per_sample(pred_rand, target_rand, action_is_pad) * weight_rand
        else:
            correction = self._action_loss_per_sample(pred_action, target_action, action_is_pad)
        proximity = self._action_loss_per_sample(pred_action, pred_base, action_is_pad)

        dataset_index = sample["dataset_index"].to(device=self.device, dtype=torch.long)
        episode_index = sample["episode_index"].to(device=self.device, dtype=torch.long)
        if getattr(self, "_success_distill_lookup_device", None) != str(self.device):
            self._success_distill_failure_lookup_dev = self._success_distill_failure_lookup.to(self.device)
            self._success_distill_balance_lookup_dev = self._success_distill_balance_lookup.to(self.device)
            self._success_distill_lookup_device = str(self.device)
        failure_lookup = self._success_distill_failure_lookup_dev
        balance_lookup = self._success_distill_balance_lookup_dev
        is_oracle = dataset_index.ne(int(self.success_distill_original_dataset_index))
        safe_episode = episode_index.clamp(0, int(failure_lookup.numel()) - 1)
        failure_mask = is_oracle & failure_lookup.index_select(0, safe_episode)
        keep_mask = ~failure_mask
        balance = balance_lookup.index_select(0, safe_episode).to(dtype=correction.dtype)

        def masked_mean(values, mask, weights=None):
            mask_f = mask.to(device=values.device, dtype=values.dtype)
            if weights is not None:
                mask_f = mask_f * weights.to(device=values.device, dtype=values.dtype)
            return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        loss_fix = masked_mean(correction, failure_mask, balance)
        loss_keep = masked_mean(proximity, keep_mask)
        loss_prox = masked_mean(proximity, failure_mask)
        loss_total = (
            float(self.success_distill_correction_weight) * loss_fix
            + float(self.success_distill_keep_weight) * loss_keep
            + float(self.success_distill_failure_proximity_weight) * loss_prox
        )
        loss_dict = {
            "loss_fix": loss_fix.detach(),
            "loss_keep": loss_keep.detach(),
            "loss_prox": loss_prox.detach(),
            "loss_action": correction.mean().detach(),
            "loss_video": torch.zeros((), device=self.device),
            "oracle_failure_fraction": failure_mask.float().mean().detach(),
            "inference_timestep_mean": timestep_action.float().mean().detach(),
        }
        return loss_total, loss_dict

    @staticmethod
    def _select_golden_target(teacher_actions, select, action_is_pad, action_dim_is_pad=None):
        """Reduce K teacher rollouts (K, B, H, D) to one target per state.

        ``single``: the only rollout (original golden recipe); ``mean``: per-state mean;
        ``medoid``: per-state rollout with the smallest summed squared distance to the
        others.  Distances ignore padded horizon steps / padded action dims.
        """
        num_noises, batch_size, _, _ = teacher_actions.shape
        stats: dict = {}
        if num_noises == 1 or select == "single":
            return teacher_actions[0], stats
        acts = teacher_actions.float()
        valid = torch.ones_like(acts[0])
        if action_is_pad is not None:
            valid = valid * (~action_is_pad).to(device=acts.device, dtype=acts.dtype).unsqueeze(-1)
        if action_dim_is_pad is not None:
            valid = valid * (~action_dim_is_pad).to(device=acts.device, dtype=acts.dtype).unsqueeze(1)
        valid_count = valid.sum(dim=(1, 2)).clamp(min=1.0)
        mean = acts.mean(dim=0)
        spread = torch.sqrt((((acts - mean.unsqueeze(0)) ** 2) * valid.unsqueeze(0)).sum(dim=(2, 3)) / valid_count)
        stats["golden_teacher_spread_rmse"] = spread.mean().detach()
        if select == "mean":
            chosen = mean
        elif select == "medoid":
            diff = acts.unsqueeze(0) - acts.unsqueeze(1)
            dist = ((diff ** 2) * valid.unsqueeze(0).unsqueeze(0)).sum(dim=(3, 4))
            medoid_idx = dist.sum(dim=1).argmin(dim=0)
            chosen = acts[medoid_idx, torch.arange(batch_size, device=acts.device)]
            stats["golden_medoid_vs_mean_rmse"] = torch.sqrt(
                (((chosen - mean) ** 2) * valid).sum(dim=(1, 2)) / valid_count
            ).mean().detach()
        else:
            raise ValueError(f"unknown golden_select={select!r}; expected single|mean|medoid")
        return chosen.to(dtype=teacher_actions.dtype), stats

    def _training_loss_golden_distillation(self, sample, tiled: bool = False):
        """Golden-noise self-distillation (HELD OUT; see success_distill.py).

        Teacher: the frozen released ActionDiT (LoRA disabled) run through the
        deployed solver (same step count / shift as `infer_action`) from the
        deployment's fixed action noise z* (CPU float32 generator seeded with the
        run seed, exactly as `infer_action` re-creates it at every replan) on the
        current-frame prefix of ordinary training states -> a*(s).  Student: the
        LoRA ActionDiT flow-matched toward a*(s) under random noise and timesteps.
        """
        from .lora import lora_disabled

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_horizon, action_dim = int(action.shape[1]), int(action.shape[2])

        if inputs["first_frame_latents"] is not None:
            video_input = inputs["first_frame_latents"]
        else:
            video_input = input_latents[:, :, 0:1]
        timestep_video = torch.zeros((batch_size,), dtype=input_latents.dtype, device=self.device)
        with torch.no_grad():
            video_pre = self.video_expert.pre_dit(
                x=video_input,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )

        def action_forward(latents, timestep):
            pre = self.action_expert.pre_dit(
                action_tokens=latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )
            tokens = self.mot.forward_action_with_video_cache(
                action_tokens=pre["tokens"],
                action_freqs=pre["freqs"],
                action_t_mod=pre["t_mod"],
                action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            return self.action_expert.post_dit(tokens, pre)

        # Teacher noise(s): the deployment construction (CPU float32 randn from a generator
        # seeded with the run seed, then cast to the model dtype).  K > 1 draws follow.
        num_noises = max(1, int(getattr(self, "golden_num_noises", 1)))
        golden_select = str(getattr(self, "golden_select", "single"))
        golden_key = (action_horizon, action_dim, str(self.device), str(action.dtype), int(self.golden_noise_seed), num_noises)
        if getattr(self, "_golden_noise_cache_key", None) != golden_key:
            generator = torch.Generator(device="cpu").manual_seed(int(self.golden_noise_seed))
            self._golden_noise = torch.randn(
                (num_noises, action_horizon, action_dim), generator=generator, device="cpu", dtype=torch.float32
            ).to(device=self.device, dtype=action.dtype)
            self._golden_noise_cache_key = golden_key
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(self.golden_num_inference_steps),
            device=self.device,
            dtype=action.dtype,
            shift_override=self.golden_sigma_shift,
        )
        with torch.no_grad(), lora_disabled(self.action_expert):
            teacher_actions = []
            for noise_idx in range(num_noises):
                latents = self._golden_noise[noise_idx : noise_idx + 1].expand(batch_size, -1, -1).clone()
                for step_t, step_delta in zip(infer_timesteps, infer_deltas):
                    timestep = step_t.expand(batch_size).to(dtype=action.dtype, device=self.device)
                    pred = action_forward(latents, timestep)
                    latents = self.infer_action_scheduler.step(pred, step_delta, latents)
                teacher_actions.append(latents.detach())
            a_star, select_stats = self._select_golden_target(
                torch.stack(teacher_actions, dim=0), golden_select, action_is_pad, None
            )
            a_star = a_star.detach()

        # Student: ordinary flow matching toward a*(s) under random noise / timestep.
        noise = torch.randn_like(action)
        t_train = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy = self.train_action_scheduler.add_noise(a_star, noise, t_train)
        target = self.train_action_scheduler.training_target(a_star, noise, t_train)
        pred_student = action_forward(noisy, t_train)
        per_sample = self._action_loss_per_sample(pred_student, target, action_is_pad)
        weight = self.train_action_scheduler.training_weight(t_train).to(per_sample.device, dtype=per_sample.dtype)
        loss_golden = (per_sample * weight).mean()
        loss_total = float(self.golden_target_weight) * loss_golden
        loss_dict = {
            "loss_golden": loss_golden.detach(),
            "loss_action": loss_golden.detach(),
            "loss_video": torch.zeros((), device=self.device),
            "golden_vs_data_action_rmse": torch.sqrt(F.mse_loss(a_star.float(), action.float())).detach(),
            "student_timestep_mean": t_train.float().mean().detach(),
        }
        loss_dict.update(select_stats)
        if float(self.golden_data_weight) > 0.0:
            noisy_data = self.train_action_scheduler.add_noise(action, noise, t_train)
            target_data = self.train_action_scheduler.training_target(action, noise, t_train)
            pred_data = action_forward(noisy_data, t_train)
            per_sample_data = self._action_loss_per_sample(pred_data, target_data, action_is_pad)
            loss_data = (per_sample_data * weight).mean()
            loss_total = loss_total + float(self.golden_data_weight) * loss_data
            loss_dict["loss_data"] = loss_data.detach()
        return loss_total, loss_dict

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
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
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
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
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
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
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
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        mot_state = self.mot.state_dict()
        if getattr(self, "action_lora_save_merged", False):
            from .lora import lora_merged_state_dict, lora_modules

            if lora_modules(self.mot):
                # Render the trained adapters into plain Linear weights so the
                # checkpoint keeps the released key set and deployment topology.
                mot_state = lora_merged_state_dict(self.mot)
        payload = {
            "mot": mot_state,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if bool(getattr(self, "action_success_distillation_enabled", False)) and str(
            getattr(self, "success_distill_mode", "oracle")
        ) == "golden":
            payload["golden_noise_distillation"] = {
                "enabled": True,
                "held_out": True,
                "label": "GOLDEN_NOISE_SELF_DISTILLATION_NO_TEST_DATA",
                "golden_noise_seed": int(self.golden_noise_seed),
                "num_inference_steps": int(self.golden_num_inference_steps),
                "sigma_shift": self.golden_sigma_shift,
                "golden_target_weight": float(self.golden_target_weight),
                "golden_data_weight": float(self.golden_data_weight),
                "golden_num_noises": int(getattr(self, "golden_num_noises", 1)),
                "golden_select": str(getattr(self, "golden_select", "single")),
                "action_lora_rank": int(getattr(self, "action_lora_rank", 0)),
                "action_lora_merged": bool(getattr(self, "action_lora_save_merged", False)),
            }
        if self.foresight_enabled and self.foresight_target_type != "encoder_hidden":
            payload["foresight_target_config"] = {
                "type": self.foresight_target_type,
                "encoder_layers": tuple(self.foresight_encoder_layers),
            }
        if self.foresight_teacher_mode != "frozen_pretrained":
            payload["foresight_teacher_config"] = {
                "mode": self.foresight_teacher_mode,
                "random_seed": self.foresight_teacher_random_seed,
                "ema_decay": self.foresight_teacher_ema_decay,
                "ema_updates": self._foresight_ema_updates,
                "encoder_layers": tuple(self.foresight_encoder_layers),
                "video_cotrain": self.foresight_video_cotrain,
            }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.foresight_projectors is not None:
            payload["foresight_projectors"] = self.foresight_projectors.state_dict()
        if getattr(self, "foresight_action_heads", None) is not None:
            payload["foresight_action_heads"] = self.foresight_action_heads.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, mmap: bool = False):
        load_kwargs = {"map_location": "cpu"}
        if mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(path, **load_kwargs)
        if "mot" in payload:
            mot_state = payload["mot"]
            from .lora import lora_modules, remap_plain_linear_keys_to_lora_base

            if lora_modules(self.mot):
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            unexpected = [k for k in load_result.unexpected_keys]
            missing = [k for k in load_result.missing_keys if ".lora_A" not in k and ".lora_B" not in k]
            if missing or unexpected:
                logger.warning(
                    "MoT checkpoint load: %d missing (non-LoRA) keys, %d unexpected keys; first missing=%s first unexpected=%s",
                    len(missing), len(unexpected), missing[:3], unexpected[:3],
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        teacher_metadata = payload.get("foresight_teacher_config")
        if teacher_metadata is not None:
            saved_mode = normalize_foresight_teacher_mode(teacher_metadata.get("mode"))
            if saved_mode != self.foresight_teacher_mode:
                logger.warning(
                    "Checkpoint teacher mode %s differs from configured mode %s. "
                    "This is harmless for action inference, but invalid for training resume.",
                    saved_mode,
                    self.foresight_teacher_mode,
                )
            elif saved_mode == "frozen_random":
                saved_seed = int(teacher_metadata.get("random_seed", 0))
                if saved_seed != self.foresight_teacher_random_seed:
                    logger.warning(
                        "Random-teacher seed differs: checkpoint=%d config=%d. "
                        "This is harmless for action inference, but invalid for training resume.",
                        saved_seed,
                        self.foresight_teacher_random_seed,
                    )
        target_metadata = payload.get("foresight_target_config")
        if target_metadata is not None:
            saved_target_type = normalize_foresight_target_type(target_metadata.get("type"))
            if saved_target_type != self.foresight_target_type:
                logger.warning(
                    "Checkpoint foresight target type %s differs from configured type %s. "
                    "This is harmless for action inference, but invalid for training resume.",
                    saved_target_type,
                    self.foresight_target_type,
                )
        if self.reset_foresight_teacher_from_student():
            logger.warning(
                "Weight-only EMA resume restarted the unregistered teacher from the loaded "
                "student; pre-checkpoint EMA history is not reconstructible."
            )
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if self.foresight_projectors is not None:
            if "foresight_projectors" in payload:
                self.foresight_projectors.load_state_dict(payload["foresight_projectors"], strict=True)
            elif "foresight_projector" in payload and len(self.foresight_layers) == 1:
                # Back-compat: load a single-layer checkpoint into the sole projector.
                only = self.foresight_projectors[str(self.foresight_layers[0])]
                only.load_state_dict(payload["foresight_projector"], strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `foresight_projectors` weights; keeping current projector params."
                )
        elif "foresight_projectors" in payload or "foresight_projector" in payload:
            logger.warning(
                "Checkpoint contains foresight projector weights but foresight is disabled; ignoring "
                "(inference does not use the projector)."
            )

        if getattr(self, "foresight_action_heads", None) is not None and "foresight_action_heads" in payload:
            self.foresight_action_heads.load_state_dict(payload["foresight_action_heads"], strict=True)

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
