from typing import Any, Optional, Sequence, Union

import torch

from wam.utils.logging_config import get_logger

from .modules.experts.action_dit import ActionDiT
from .modules.memory import ProprioContextEncoder
from .modules.mot.mixture_of_transformers import MoT
from .modules.understanding.qwen_vl_encoder import QwenVLUnderstandingEncoder
from .backbones.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

logger = get_logger(__name__)


class WAM(torch.nn.Module):
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
        understanding: Optional[dict[str, Any]] = None,
        memory: Optional[dict[str, Any]] = None,
        future_delta: Optional[dict[str, Any]] = None,
        proprio_encoding: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.mot = mot
        if (
            self.video_expert is not video_expert
            or self.action_expert is not action_expert
        ):
            raise ValueError(
                "WAM experts must be the same modules registered in `mot.mixtures`."
            )

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
        self.proprio_encoding_cfg = dict(proprio_encoding or {})
        self.proprio_encoder = None
        self.action_proprio_encoder = None

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
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.understanding = None
        self.understanding_interval_chunks = 1
        self.understanding_cfg = dict(understanding or {})
        self._init_understanding(self.understanding_cfg)
        if self.proprio_dim is not None:
            self.proprio_encoder = ProprioContextEncoder(
                self.proprio_dim,
                self.text_dim,
                valid_input_scale=float(
                    self.proprio_encoding_cfg.get("valid_input_scale", 1.0)
                ),
                dimension_valid_mask=self.proprio_encoding_cfg.get(
                    "dimension_valid_mask"
                ),
            ).to(torch_dtype)
        self.memory_cfg = dict(memory or {})
        self._memory_enabled = False
        self.memory_video_enabled = False
        self.memory_video_anchor_frames = 0
        self.memory_video_recent_frames = 0
        self._init_memory(self.memory_cfg)
        self.future_delta_cfg = dict(future_delta or {})
        self.future_delta_enabled = False
        self.future_delta_action_video_freq_ratio = 4
        self.future_delta_num_frames = 1
        self.future_delta_num_tokens = 0
        self.loss_lambda_future_delta = 0.0
        self.semantic_future_alignment_enabled = False
        self.semantic_future_student_layer = 0
        self.semantic_future_teacher_layer = 0
        self.semantic_future_max_sigma = 0.0
        self.loss_lambda_semantic_future = 0.0
        self._init_future_delta(self.future_delta_cfg)

        self.to(self.device)

    @property
    def video_expert(self):
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    def set_runtime_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        # ZeRO-3 builds the model on CPU and only owns parameters that enter its
        # optimizer. Frozen inference modules therefore need an explicit move
        # after ``accelerator.prepare``. Do not move ``mot`` or the trainable
        # understanding adapter here: DeepSpeed owns those parameters.
        set_vae_runtime = getattr(self.vae, "set_runtime_context", None)
        if callable(set_vae_runtime):
            set_vae_runtime(device=self.device, dtype=self.torch_dtype)
        else:
            self.vae.to(device=self.device, dtype=self.torch_dtype)
            if hasattr(self.vae, "mean") and torch.is_tensor(self.vae.mean):
                self.vae.mean = self.vae.mean.to(device=self.device, dtype=self.torch_dtype)
            if hasattr(self.vae, "std") and torch.is_tensor(self.vae.std):
                self.vae.std = self.vae.std.to(device=self.device, dtype=self.torch_dtype)
            if hasattr(self.vae, "mean") and hasattr(self.vae, "std"):
                self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        if self.text_encoder is not None:
            self.text_encoder.to(device=self.device, dtype=self.torch_dtype)
        if self.understanding is not None:
            self.understanding.vlm.device = self.device
            if not self.understanding.train_vlm:
                self.understanding.vlm.model.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )

    def offload_deferred_vae(self) -> bool:
        offload = getattr(self.vae, "offload", None)
        return bool(offload()) if callable(offload) else False

    def _init_understanding(self, cfg: dict[str, Any]) -> None:
        if not bool(cfg.get("enabled", False)):
            return
        vlm_model_path = cfg.get("vlm_model_path")
        if not vlm_model_path:
            raise ValueError(
                "`model.understanding.vlm_model_path` is required when understanding is enabled."
            )
        if self.proprio_dim is None:
            raise ValueError(
                "Action VLM conditioning requires a configured proprio dimension."
            )
        self.understanding_interval_chunks = max(1, int(cfg.get("interval_chunks", 1)))
        self.understanding = QwenVLUnderstandingEncoder(
            vlm_model_path=str(vlm_model_path),
            device=self.device,
            dtype=self.torch_dtype,
            prompt=cfg.get("prompt"),
            train_vlm=bool(cfg.get("train_vlm", False)),
            vlm_gradient_checkpointing=bool(
                cfg.get("vlm_gradient_checkpointing", False)
            ),
            save_vlm_weights=cfg.get("save_vlm_weights"),
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            vlm_batch_size=int(cfg.get("vlm_batch_size", 0) or 0),
            max_pixels=int(cfg.get("max_pixels", 65536)),
            precompute_metadata=bool(cfg.get("precompute_metadata", True)),
            prompt_cache_size=int(cfg.get("prompt_cache_size", 256)),
            use_view_valid_mask=bool(
                cfg.get("use_view_valid_mask", True)
            ),
        )
        context_dim = int(self.understanding.context_dim)
        self.action_proprio_encoder = ProprioContextEncoder(
            self.proprio_dim,
            context_dim,
            valid_input_scale=float(
                self.proprio_encoding_cfg.get("valid_input_scale", 1.0)
            ),
            dimension_valid_mask=self.proprio_encoding_cfg.get(
                "dimension_valid_mask"
            ),
        ).to(self.torch_dtype)
        replaced_action_cross_attn = self.action_expert.configure_vlm_conditioning(
            context_dim=context_dim
        )
        logger.info(
            "Initialized action Qwen/state conditioning: qwen_dim=%d "
            "state_dim=%d action_layers=%d interval=%d train_vlm=%s",
            context_dim,
            int(self.proprio_dim),
            replaced_action_cross_attn,
            self.understanding_interval_chunks,
            bool(cfg.get("train_vlm", False)),
        )

    @property
    def understanding_enabled(self) -> bool:
        return self.understanding is not None

    def _init_future_delta(self, cfg: dict[str, Any]) -> None:
        if not bool(cfg.get("enabled", False)):
            return
        self.future_delta_action_video_freq_ratio = int(
            cfg.get("action_video_freq_ratio", 4)
        )
        if bool(getattr(self.video_expert, "action_conditioned", False)):
            raise ValueError(
                "Video-side Future Delta queries require "
                "video_dit_config.action_conditioned=false to prevent noisy "
                "action leakage."
            )
        patch_size = tuple(int(v) for v in self.video_expert.patch_size)
        latent_channels = int(self.video_expert.in_dim)
        num_tokens_value = cfg.get("num_tokens")
        if num_tokens_value is None:
            height, width = map(int, cfg["video_size"])
            num_views = int(cfg.get("num_views", 1))
            vae_factor = int(self.vae.upsampling_factor)
            latent_h, latent_w = height // vae_factor, width // vae_factor
            patch_h, patch_w = patch_size[1], patch_size[2]
            num_tokens = (
                (latent_h // patch_h)
                * (latent_w // patch_w)
                * num_views
            )
        else:
            num_tokens = int(num_tokens_value)
        loss_weight = float(cfg.get("loss_weight", 0.5))
        self.future_delta_num_frames = (
            (int(cfg["num_frames"]) - 1)
            // self.future_delta_action_video_freq_ratio
            // int(self.vae.temporal_downsample_factor)
        )
        patch_dim = latent_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.video_expert.configure_future_delta_queries(
            num_tokens=num_tokens,
            patch_dim=patch_dim,
            num_future_delta_frames=self.future_delta_num_frames,
        )
        self.future_delta_enabled = True
        self.future_delta_num_tokens = num_tokens
        self.loss_lambda_future_delta = loss_weight
        logger.info(
            "Initialized video-side Future Delta queries: tokens_per_frame=%d "
            "future_frames=%d patch_dim=%d action_video_freq_ratio=%d "
            "loss_weight=%.3f",
            num_tokens,
            self.future_delta_num_frames,
            patch_dim,
            self.future_delta_action_video_freq_ratio,
            loss_weight,
        )
        semantic_cfg = cfg.get("semantic_alignment") or {}
        if not bool(semantic_cfg.get("enabled", False)):
            return

        student_layer = int(semantic_cfg.get("student_layer", 8))
        teacher_layer = int(semantic_cfg.get("teacher_layer", 20))
        semantic_weight = float(semantic_cfg.get("loss_weight", 0.1))
        max_sigma = float(semantic_cfg.get("max_sigma", 0.2))
        self.semantic_future_alignment_enabled = True
        self.semantic_future_student_layer = student_layer
        self.semantic_future_teacher_layer = teacher_layer
        self.semantic_future_max_sigma = max_sigma
        self.loss_lambda_semantic_future = semantic_weight
        logger.info(
            "Initialized Semantic Future Alignment: student_layer=%d "
            "teacher_layer=%d max_sigma=%.3f loss_weight=%.3f",
            student_layer,
            teacher_layer,
            max_sigma,
            semantic_weight,
        )

    def _init_memory(self, cfg: dict[str, Any]) -> None:
        """Configure the two frame roles; persistent memory is intentionally off."""
        video_cfg = dict(cfg.get("video") or {})
        self.memory_video_enabled = bool(video_cfg.get("enabled", True))
        self.memory_video_anchor_frames = max(
            0, int(video_cfg.get("num_anchor_frames", 1) or 0)
        )
        self.memory_video_recent_frames = max(
            0, int(video_cfg.get("num_recent_frames", 1) or 0)
        )
        self.deduplicate_identical_anchor_batch = bool(
            video_cfg.get("deduplicate_identical_anchor_batch", True)
        )
        if self.memory_video_anchor_frames != 1:
            raise ValueError(
                "Frame conditioning requires exactly one anchor frame."
            )
        if self.memory_video_recent_frames != 1:
            raise ValueError(
                "Frame conditioning requires exactly one recent frame."
            )
        self._memory_enabled = False
        logger.info(
            "Initialized frame conditioning: anchor=1 recent=1 "
            "persistent_kv_memory=false deduplicate_anchor_batch=%s",
            self.deduplicate_identical_anchor_batch,
        )

    @property
    def memory_enabled(self) -> bool:
        return bool(self._memory_enabled)

    def _action_token_seq_len_for_mask(self, action_seq_len: int) -> int:
        return int(action_seq_len)

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
        prompt_emb = prompt_emb.masked_fill(~mask.unsqueeze(-1), 0)
        return prompt_emb.to(device=self.device), torch.ones_like(mask)

    def set_trainable_modules(self):
        self.eval()
        self.requires_grad_(False)
        self.mot.train()
        self.mot.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        if self.action_proprio_encoder is not None:
            self.action_proprio_encoder.train()
            self.action_proprio_encoder.requires_grad_(True)
        if self.understanding is not None:
            self.understanding.set_trainable()

    def trainable_parameters(self):
        params = list(self.mot.parameters())
        if self.proprio_encoder is not None:
            params.extend(list(self.proprio_encoder.parameters()))
        if self.action_proprio_encoder is not None:
            params.extend(list(self.action_proprio_encoder.parameters()))
        if self.understanding is not None:
            params.extend(list(self.understanding.trainable_parameters()))
        return [param for param in params if param.requires_grad]
