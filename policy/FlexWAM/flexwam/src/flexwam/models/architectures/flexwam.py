from flexwam.models.architectures.builder import PretrainedBuilderMixin
from flexwam.training.checkpoints import CheckpointMixin
from flexwam.training.counterfactual import CounterfactualMixin
from flexwam.training.objective import TrainingObjectiveMixin
from flexwam.inference.pipeline import InferenceMixin
from flexwam.models.architectures.structures import _video_visual_view_slices
from flexwam.config import action_attention_config
from flexwam.models.architectures.state_conditioning import conditioning_config
from typing import Any, Optional, Sequence, Union
import math

import torch
import torch.nn as nn
from PIL import Image

from flexwam.utils.camera_metrics import resolve_camera_metric_names
from flexwam.utils.logging_config import get_logger

from flexwam.models.architectures.mot import MoT
from flexwam.models.schedulers.flow_matching import WanContinuousFlowMatchScheduler
from flexwam.models.architectures.view_scoring_head import (
    HEAD_NETWORK_KEYS,
    SpatialQueryHead,
    normalize_head_arch,
)
from flexwam.models.architectures.critic import CameraCritic
from flexwam.models.architectures.router import CameraRouter

logger = get_logger(__name__)


class FlexWAM(PretrainedBuilderMixin, CounterfactualMixin, TrainingObjectiveMixin, InferenceMixin, CheckpointMixin, torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: nn.Module,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        context_input_dim: Optional[int] = None,
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
        action_router_config: Optional[dict[str, Any]] = None,
        action_critic_config: Optional[dict[str, Any]] = None,
        action_context_as_prefix: bool = True,
        router_cf_config: Optional[dict[str, Any]] = None,
        video_state_config: Optional[dict[str, Any]] = None,
        state_conditioning: Optional[dict[str, Any]] = None,
        action_attention: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        if not getattr(video_expert, "cross_view_layer_indices", ()):
            raise ValueError(
                "FlexWAM requires at least one cross-view attention layer; set "
                "`cross_view_attention_interval` > 0 or `cross_view_attention_layers`."
            )
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.context_input_dim = self.text_dim if context_input_dim is None else int(context_input_dim)
        self.context_adapter = None
        if self.context_input_dim != self.text_dim:
            self.context_adapter = nn.Linear(self.context_input_dim, self.text_dim).to(torch_dtype)
        if proprio_dim is None or int(proprio_dim) <= 0:
            raise ValueError("FlexWAM requires a positive `proprio_dim`; state-free mode was removed.")
        self.proprio_dim = int(proprio_dim)
        conditioning = conditioning_config(state_conditioning)
        self.state_conditioning_mode = conditioning["mode"]
        attention_cfg = action_attention_config(action_attention)
        self.action_attention_mode = attention_cfg["mode"]
        self.action_attention_bias_strength = float(attention_cfg["bias_strength"])
        self.state_drop_prob = float(conditioning["drop_prob"])
        shared = self.state_conditioning_mode == "shared"
        # Preserve the legacy parameter name and initialization order in separate mode.
        state_hidden = self._video_token_dim(video_expert if shared else action_expert)
        self.state_proj = nn.Linear(self.proprio_dim, state_hidden).to(torch_dtype)
        video_expert.configure_state_conditioning(self.proprio_dim, video_state_config, shared=shared)

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
        # Compatibility flag; both experts now use only state as action prefix.
        # WAN language remains separate cross-attention context.
        self.action_context_as_prefix = bool(action_context_as_prefix)
        router_cf_config = {} if router_cf_config is None else dict(router_cf_config)
        self.router_cf_lambda = 1.0
        self.router_cf_interval = int(router_cf_config.get("interval", 10))
        if self.router_cf_interval <= 0:
            raise ValueError("`router_cf.interval` must be positive.")
        self.router_cf_pixel_drop_min = float(router_cf_config.get("pixel_drop_min", 1.0))
        self.router_cf_pixel_drop_max = float(router_cf_config.get("pixel_drop_max", 1.0))
        if not 0.0 <= self.router_cf_pixel_drop_min <= self.router_cf_pixel_drop_max <= 1.0:
            raise ValueError(
                "`router_cf.pixel_drop_min/max` must satisfy 0 <= min <= max <= 1, "
                f"got min={self.router_cf_pixel_drop_min}, max={self.router_cf_pixel_drop_max}."
            )
        self.router_cf_zscore_eps = float(router_cf_config.get("zscore_eps", 1e-6))
        if self.router_cf_zscore_eps <= 0.0:
            raise ValueError(
                f"`router_cf.zscore_eps` must be positive, got {self.router_cf_zscore_eps}."
            )
        self.router_cf_pixel_fill = float(router_cf_config.get("pixel_fill", 0.0))
        self.router_cf_kl_temperature_start = float(router_cf_config.get("kl_temperature_start", 2.0))
        self.router_cf_kl_temperature_end = float(router_cf_config.get("kl_temperature_end", 1.0))
        if min(self.router_cf_kl_temperature_start, self.router_cf_kl_temperature_end) <= 0.0:
            raise ValueError("`router_cf.kl_temperature_*` must be positive.")
        self.router_cf_kl_lambda_start = float(router_cf_config.get("kl_lambda_start", 0.25))
        self.router_cf_kl_lambda_end = float(router_cf_config.get("kl_lambda_end", 0.33))
        if min(self.router_cf_kl_lambda_start, self.router_cf_kl_lambda_end) < 0.0:
            raise ValueError("`router_cf.kl_lambda_*` must be non-negative.")
        self.router_cf_uniform_probe = bool(router_cf_config.get("uniform_probe", False))
        # Accept the old key when loading saved configurations.
        intervention = str(
            router_cf_config.get("intervention", router_cf_config.get("probe", "pixel"))
        ).strip().lower()
        allowed_interventions = {"pixel", "router_zero"}
        if intervention not in allowed_interventions:
            raise ValueError(
                f"`router_cf.intervention` must be one of {sorted(allowed_interventions)}, got {intervention}."
            )
        self.router_cf_intervention = intervention
        teacher = str(router_cf_config.get("teacher", "critic")).strip().lower()
        allowed_teachers = {"critic", "cf"}
        if teacher not in allowed_teachers:
            raise ValueError(
                f"`router_cf.teacher` must be one of {sorted(allowed_teachers)}, got {teacher}."
            )
        self.router_cf_teacher = teacher
        if teacher == "cf":
            logger.info(
                "Router CF teacher=cf: z-scored counterfactual deltas supervise the router "
                "directly from stage 4; the critic is not trained or used as a teacher."
            )
        action_sigma_min = router_cf_config.get("action_sigma_min", None)
        action_sigma_max = router_cf_config.get("action_sigma_max", None)
        if action_sigma_min is None and action_sigma_max is None:
            self.router_cf_action_sigma_min = None
            self.router_cf_action_sigma_max = None
        else:
            if action_sigma_min is None or action_sigma_max is None:
                raise ValueError(
                    "`router_cf.action_sigma_min` and `router_cf.action_sigma_max` must be set together."
                )
            self.router_cf_action_sigma_min = float(action_sigma_min)
            self.router_cf_action_sigma_max = float(action_sigma_max)
            if not (
                0.0
                <= self.router_cf_action_sigma_min
                <= self.router_cf_action_sigma_max
                <= 1.0
            ):
                raise ValueError(
                    "`router_cf.action_sigma_min/max` must satisfy 0 <= min <= max <= 1, "
                    f"got min={self.router_cf_action_sigma_min}, max={self.router_cf_action_sigma_max}."
                )
        self.router_cf_train_step = 0
        if self.router_cf_intervention == "router_zero":
            logger.info(
                "Critic CF intervention=router_zero: baseline mixer 1:1:1, then set each "
                "camera scale to 0; those counterfactual forwards do not backprop."
            )
        elif self.router_cf_uniform_probe:
            logger.info(
                "Pixel-critic CF uses a uniform 1:1:1 router on the unmasked baseline "
                "and on each camera drop; those counterfactual forwards do not backprop."
            )
        if self.router_cf_action_sigma_min is not None:
            logger.info(
                "Pixel-critic CF resamples action sigma ~ U[%.3f, %.3f]; "
                "main action/video training still uses the full schedule.",
                self.router_cf_action_sigma_min,
                self.router_cf_action_sigma_max,
            )
        self.use_cuda_graph = False
        self._action_denoise_cuda_graph: Optional["_ActionDenoiseCudaGraph"] = None
        self._prompt_embed_cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_camera_heads(action_router_config, action_critic_config, torch_dtype)

        self.training_stage: Optional[int] = None
        self._progress_step = 0
        self._stage4_epoch = 0
        self._stage4_epochs = 1
        self.action_camera_router_train_enabled = bool(getattr(self, "camera_router_enabled", True))
        self._deferred_pixel_cf_ctx: Optional[dict[str, Any]] = None

        self.to(self.device)

    @staticmethod
    def _video_token_dim(video_expert) -> int:
        dim = getattr(video_expert, "hidden_dim", None)
        return int(dim if dim is not None else video_expert.hidden_size)

    @staticmethod
    def _video_layer_count(video_expert) -> int:
        blocks = getattr(video_expert, "blocks", None)
        if blocks is not None:
            return int(len(blocks))
        depth = getattr(video_expert, "num_layers", None)
        return int(depth if depth is not None else video_expert.depth)

    def _head_init_kwargs(self, config: dict[str, Any], token_dim: int, num_views: int) -> dict[str, Any]:
        return {
            "token_dim": int(token_dim),
            "num_views": int(num_views),
            "feature_dim": int(config.get("feature_dim", 256)),
            "pool_grid": config.get("pool_grid", [4, 4]),
            "num_queries": int(config.get("num_queries", 4)),
            "num_heads": int(config.get("num_heads", 4)),
            "spatial_hidden_dim": int(config.get("spatial_hidden_dim", 1024)),
            "branch_output_dim": int(config.get("branch_output_dim", 512)),
            "head_hidden_dim": int(config.get("head_hidden_dim", 512)),
            "proprio_dim": self.proprio_dim,
            "state_conditioning": config.get("state_conditioning"),
        }

    @staticmethod
    def _copy_network_config(config: dict[str, Any]) -> dict[str, Any]:
        copied = {key: config[key] for key in HEAD_NETWORK_KEYS if key in config}
        if "pool_grid" in copied:
            copied["pool_grid"] = [int(copied["pool_grid"][0]), int(copied["pool_grid"][1])]
        copied["arch"] = normalize_head_arch(copied.get("arch"))
        return copied

    def _check_head_layer(self, label: str, layer_index: int) -> int:
        layer_index = int(layer_index)
        depth = self._video_layer_count(self.video_expert)
        if not 0 <= layer_index < depth:
            raise ValueError(f"{label}.feature_layer_index={layer_index} is outside video depth={depth}.")
        return layer_index

    def _effective_head_config(self, module: SpatialQueryHead, layer_index: int) -> dict[str, Any]:
        config = module.network_config()
        config["feature_layer_index"] = int(layer_index)
        config["num_views"] = int(module.num_views)
        return config

    def _init_camera_heads(
        self,
        action_router_config: Optional[dict[str, Any]],
        action_critic_config: Optional[dict[str, Any]],
        torch_dtype: torch.dtype,
    ) -> None:
        router_config = {} if action_router_config is None else dict(action_router_config)
        self.camera_router_enabled = bool(router_config.pop("enabled", True))
        if action_critic_config is None:
            critic_config = self._copy_network_config(router_config)
            self.camera_critic_enabled = self.camera_router_enabled
        else:
            critic_config = dict(action_critic_config)
            critic_config.pop("same_router", None)
            critic_config["arch"] = normalize_head_arch(critic_config.get("arch"))
            self.camera_critic_enabled = bool(critic_config.pop("enabled", True))

        num_views = int(router_config.get("num_views", getattr(self.video_expert, "num_views", 1)))
        token_dim = int(router_config.get("token_dim", self._video_token_dim(self.video_expert)))
        self.action_router_feature_layer_index = self._check_head_layer(
            "router",
            router_config.get("feature_layer_index", 18),
        )
        self.action_critic_feature_layer_index = self._check_head_layer(
            "critic",
            critic_config.get("feature_layer_index", self.action_router_feature_layer_index),
        )
        self.action_router_arch = normalize_head_arch(router_config.get("arch"))
        self.action_critic_arch = normalize_head_arch(critic_config.get("arch"))

        self.action_camera_router = CameraRouter(
            **self._head_init_kwargs(router_config, token_dim, num_views)
        ).to(torch_dtype)
        self.action_camera_critic_train_enabled = False
        self.action_camera_critic = CameraCritic(
            **self._head_init_kwargs(critic_config, token_dim, num_views)
        ).to(torch_dtype)
        self.action_router_config = self._effective_head_config(
            self.action_camera_router,
            self.action_router_feature_layer_index,
        )
        self.action_critic_config = self._effective_head_config(
            self.action_camera_critic,
            self.action_critic_feature_layer_index,
        )
        logger.info(
            "Camera heads: router=%s critic=%s router_enabled=%s critic_enabled=%s",
            self.action_router_config,
            self.action_critic_config,
            self.camera_router_enabled,
            self.camera_critic_enabled,
        )
        if not self.camera_router_enabled:
            self.action_camera_router_train_enabled = False
            self.action_camera_router.requires_grad_(False)
            self.action_camera_router.eval()
        if not self.camera_critic_enabled:
            self.action_camera_critic_train_enabled = False
            self.action_camera_critic.requires_grad_(False)
            self.action_camera_critic.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        self._prompt_embed_cache.clear()
        return self

    def _validate_video_spatial_size(self, height: int, width: int) -> None:
        # Pixel-space divisibility must cover both VAE compression and DiT
        # patching: WAN2.2 requires 32, LingBot requires 16 with current presets.
        factor = int(self.vae.upsampling_factor)
        multiple_h = factor * int(self.video_expert.patch_size[1])
        multiple_w = factor * int(self.video_expert.patch_size[2])
        if min(height, width) <= 0 or height % multiple_h or width % multiple_w:
            raise ValueError(
                f"Video/image spatial dims must be positive multiples of {multiple_h} (H) "
                f"and {multiple_w} (W), got H={height}, W={width}. Resize before encoding."
            )

    @staticmethod
    def _prompt_cache_key(prompt: Union[str, Sequence[str]]) -> Optional[Any]:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)) and all(isinstance(item, str) for item in prompt):
            return tuple(prompt)
        return None

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        cache_key = self._prompt_cache_key(prompt)
        cached = None if cache_key is None else self._prompt_embed_cache.get(cache_key)
        if cached is not None:
            prompt_emb, mask = cached
            return prompt_emb.clone(), mask.clone()
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        prompt_emb = prompt_emb.to(device=self.device)
        if cache_key is not None:
            self._prompt_embed_cache[cache_key] = (prompt_emb.detach(), mask.detach())
            logger.info("Cached UMT5 prompt embedding (%d tokens).", int(mask.shape[-1]))
        return prompt_emb, mask

    def _action_prefix_len(self, action_pre: dict[str, Any]) -> int:
        return int(action_pre.get("meta", {}).get("action_prefix_len", 0))

    def _sample_state_keep_mask(self, proprio: torch.Tensor) -> Optional[torch.Tensor]:
        enabled = (
            getattr(self.video_expert, "state_proj", None) is not None
            or self.action_camera_router.state_encoder is not None
            or self.action_camera_critic.state_encoder is not None
        )
        # Trainer keeps the wrapper/VAE in eval mode and trains model.dit (MoT).
        if not self.mot.training or self.state_drop_prob == 0.0 or not enabled:
            return None
        return torch.rand((proprio.shape[0], 1), device=proprio.device) >= self.state_drop_prob

    def _video_expert_pre_dit(self, *, proprio=None, state_keep_mask=None, **kwargs):
        if self.state_conditioning_mode == "shared":
            # One mandatory state token, cached with video K/V and never dropped.
            kwargs["state_tokens"] = self._project_state(proprio)
        elif self.video_expert.state_proj is not None:
            kwargs.update(proprio=proprio, state_keep_mask=state_keep_mask)
        return self.video_expert.pre_dit(**kwargs)

    def _project_state(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio is None:
            raise ValueError("FlexWAM requires `proprio` state input.")
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self.state_proj(
            proprio.to(device=self.state_proj.weight.device, dtype=self.torch_dtype)
        ).unsqueeze(1)

    def _action_state_tokens(self, proprio: torch.Tensor) -> Optional[torch.Tensor]:
        if self.state_conditioning_mode == "shared":
            return None
        return self._project_state(proprio)

    def _action_expert_pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        kwargs = {
            "action_tokens": action_tokens,
            "timestep": timestep,
            "context": action_context,
            "context_mask": action_context_mask,
        }
        kwargs["state_tokens"] = self._action_state_tokens(proprio)
        return self.action_expert.pre_dit(**kwargs)

    def _adapt_context_for_experts(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.context_input_dim:
            raise ValueError(
                f"`context` last dim must be {self.context_input_dim}, got {context.shape[-1]}."
            )
        if self.context_adapter is None:
            return context
        return self.context_adapter(context.to(device=self.device, dtype=self.torch_dtype))

    def _build_video_action_contexts(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del proprio
        adapted_context = self._adapt_context_for_experts(context)
        if getattr(self.action_expert, "uses_text_cross_attention", False):
            action_context, action_mask = adapted_context, context_mask
        else:
            # LingBot actions read language through the shared video prefix.
            action_context = context.new_empty((context.shape[0], 0, self.context_input_dim))
            action_mask = context_mask.new_empty((context_mask.shape[0], 0))
        return {
            "video_context": adapted_context,
            "video_context_mask": context_mask,
            "action_context": action_context,
            "action_context_mask": action_mask,
        }

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if video_tensor.ndim == 6:
            batch_size, num_views = video_tensor.shape[:2]
            flat_video = video_tensor.reshape(batch_size * num_views, *video_tensor.shape[2:])
            flat_latents = self.vae.encode(
                flat_video,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            return flat_latents.reshape(batch_size, num_views, *flat_latents.shape[1:])
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @staticmethod
    def _canonicalize_video_tensor(video: torch.Tensor, expected_num_views: int | None = None) -> torch.Tensor:
        if video.ndim == 5:
            if video.shape[1] != 3:
                raise ValueError(
                    f"5D `video` must be [B, 3, T, H, W], got shape {tuple(video.shape)}"
                )
            return video
        if video.ndim != 6:
            raise ValueError(
                "`video` must be 5D [B, 3, T, H, W], 6D [B, V, 3, T, H, W], "
                f"or 6D [B, T, V, 3, H, W], got shape {tuple(video.shape)}"
            )
        if expected_num_views is not None:
            if video.shape[1] == expected_num_views and video.shape[2] == 3:
                return video
            if video.shape[2] == expected_num_views and video.shape[3] == 3:
                return video.permute(0, 2, 3, 1, 4, 5).contiguous()
        if video.shape[3] == 3 and video.shape[2] != 3:
            return video.permute(0, 2, 3, 1, 4, 5).contiguous()
        if video.shape[2] == 3 and video.shape[3] != 3:
            return video
        if video.shape[3] == 3:
            return video.permute(0, 2, 3, 1, 4, 5).contiguous()
        raise ValueError(
            "6D `video` must be [B, V, 3, T, H, W] or [B, T, V, 3, H, W], "
            f"got shape {tuple(video.shape)}"
        )

    @staticmethod
    def _latent_num_frames(latents: torch.Tensor) -> int:
        if latents.ndim == 5:
            return int(latents.shape[2])
        if latents.ndim == 6:
            return int(latents.shape[3])
        raise ValueError(f"Expected 5D/6D latents, got shape {tuple(latents.shape)}")

    @staticmethod
    def _latent_first_frame(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            return latents[:, :, 0:1]
        if latents.ndim == 6:
            return latents[:, :, :, 0:1]
        raise ValueError(f"Expected 5D/6D latents, got shape {tuple(latents.shape)}")

    @staticmethod
    def _latent_tail(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            return latents[:, :, 1:]
        if latents.ndim == 6:
            return latents[:, :, :, 1:]
        raise ValueError(f"Expected 5D/6D latents, got shape {tuple(latents.shape)}")

    @staticmethod
    def _copy_first_frame_(latents: torch.Tensor, first_frame_latents: torch.Tensor) -> None:
        if latents.ndim == 5:
            latents[:, :, 0:1] = first_frame_latents
            return
        if latents.ndim == 6:
            latents[:, :, :, 0:1] = first_frame_latents
            return
        raise ValueError(f"Expected 5D/6D latents, got shape {tuple(latents.shape)}")

    @staticmethod
    def _spatial_grid_hw(video_meta: dict[str, Any]) -> tuple[int, int]:
        grid = video_meta.get("grid_size")
        if grid is not None and len(grid) >= 3:
            height, width = int(grid[1]), int(grid[2])
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid spatial grid_size {grid!r}.")
            return height, width
        raise ValueError("`video_meta['grid_size']` is required for camera-head visual routing.")

    def _extract_first_frame_view_tokens(
        self,
        video_tokens: torch.Tensor,
        video_meta: dict[str, Any],
    ) -> torch.Tensor:
        view_slices = _video_visual_view_slices(int(video_tokens.shape[1]), video_meta)
        first_frame_tokens = int(video_meta.get("video_tokens_per_frame", video_meta.get("tokens_per_frame", 0)))
        height, width = self._spatial_grid_hw(video_meta)
        if first_frame_tokens != height * width:
            raise ValueError(
                "First-frame token count does not match grid: "
                f"video_tokens_per_frame={first_frame_tokens}, grid=({height},{width})."
            )
        return torch.stack(
            [
                video_tokens[:, start:start + first_frame_tokens, :]
                for start, _ in view_slices
            ],
            dim=1,
        )

    def _forward_video_branch(
        self,
        video_pre: dict[str, Any],
        attention_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        snapshot_layer_indices: Sequence[int] = (),
    ) -> dict[str, Any]:
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_mask = attention_mask[:video_seq_len, :video_seq_len]
        cross_view_mask = self.video_expert.build_cross_view_mask(video_mask, video_pre["meta"])
        return self.mot.forward_video(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_mask,
            snapshot_layer_indices=snapshot_layer_indices,
            cross_view_attention_mask=cross_view_mask,
            cross_view_layer_indices=() if cross_view_mask is None else self.video_expert.cross_view_layer_indices,
        )

    def _forward_action_with_video_cache(
        self,
        action_pre: dict[str, Any],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        video_meta: dict[str, Any],
        router_output: dict[str, torch.Tensor],
        action_attention_stats: Optional[list[dict[str, float]]] = None,
        action_attention_heatmaps: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        mot_kwargs = self._action_router_mot_kwargs(router_output)
        return self.mot.forward_action_with_video_cache(
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
            action_prefix_len=self._action_prefix_len(action_pre),
            video_meta=video_meta,
            action_attention_mode=self.action_attention_mode,
            action_attention_bias_strength=self.action_attention_bias_strength,
            action_attention_stats=action_attention_stats,
            action_attention_heatmaps=action_attention_heatmaps,
            **mot_kwargs,
        )

    def _call_camera_head(
        self,
        module: SpatialQueryHead,
        *,
        first_frame_features: torch.Tensor,
        video_meta: dict[str, Any],
        return_logits: bool = False,
        proprio: Optional[torch.Tensor] = None,
        state_keep_mask: Optional[torch.Tensor] = None,
    ):
        if first_frame_features is None:
            raise ValueError("Visual router/critic requires first-frame features.")
        return module(
            first_frame_features,
            self._spatial_grid_hw(video_meta),
            return_logits=return_logits,
            proprio=proprio,
            state_keep_mask=state_keep_mask,
        )

    def _uniform_action_router_output(
        self,
        *,
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        weights = torch.full(
            (batch_size, num_views),
            1.0 / float(num_views),
            device=device,
            dtype=dtype,
        )
        return {"weights": weights, "scales": torch.ones_like(weights)}

    @staticmethod
    def _router_output_zero_view(
        router_output: dict[str, torch.Tensor],
        view_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Copy a mixer and set one camera's weight/scale to 0, e.g. 1:1:1 -> 1:1:0."""
        weights = router_output["weights"].clone()
        scales = router_output["scales"].clone()
        if view_idx < 0 or view_idx >= int(scales.shape[-1]):
            raise ValueError(f"Invalid camera index {view_idx} for {tuple(scales.shape)} router.")
        weights[:, view_idx] = 0
        scales[:, view_idx] = 0
        result = {"weights": weights, "scales": scales}
        if "log_scales" in router_output:
            log_scales = router_output["log_scales"].clone()
            log_scales[:, view_idx] = float("-inf")
            result["log_scales"] = log_scales
        return result

    def _build_action_router_output(
        self,
        video_pre: dict[str, Any],
        first_frame_features: Optional[torch.Tensor] = None,
        video_meta: Optional[dict[str, Any]] = None,
        force_uniform_router: bool = False,
        proprio: Optional[torch.Tensor] = None,
        state_keep_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        meta = video_pre["meta"] if video_meta is None else video_meta
        video_tokens = video_pre["tokens"]
        num_views = int(meta.get("num_views", getattr(self.action_camera_router, "num_views", 1)))
        batch_size = int(video_tokens.shape[0])
        if force_uniform_router or not bool(getattr(self, "action_camera_router_train_enabled", True)):
            return self._uniform_action_router_output(
                batch_size=batch_size,
                num_views=num_views,
                device=video_tokens.device,
                dtype=video_tokens.dtype,
            )
        if first_frame_features is None:
            raise ValueError("Action router requires first-frame features from Video DiT.")
        if self.action_attention_mode == "logit_bias":
            logits = self._call_camera_head(
                self.action_camera_router,
                first_frame_features=first_frame_features,
                video_meta=meta,
                return_logits=True,
                proprio=proprio,
                state_keep_mask=state_keep_mask,
            )
            # Keep the attention path in log space: low-precision probabilities
            # can round to zero, and taking their logarithm loses both the prior
            # and stable gradients. Probabilities remain available for logging.
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            weights = log_probs.exp().to(dtype=first_frame_features.dtype)
            return {
                "weights": weights,
                "scales": weights * float(num_views),
                "log_scales": log_probs + math.log(num_views),
            }
        weights, scales = self._call_camera_head(
            self.action_camera_router,
            first_frame_features=first_frame_features,
            video_meta=meta,
            return_logits=False,
            proprio=proprio,
            state_keep_mask=state_keep_mask,
        )
        return {
            "weights": weights,
            "scales": scales,
        }

    def _action_router_mot_kwargs(
        self,
        router_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Router scales are V * softmax. additive multiplies them onto each camera
        # output; logit_bias adds bias_strength * log(scale) inside one softmax.
        kwargs = {
            "action_video_router_weights": router_output["scales"],
        }
        if "log_scales" in router_output:
            kwargs["action_video_router_log_scales"] = router_output["log_scales"]
        return kwargs

    def _head_snapshot_layers(self, *, include_critic: bool) -> tuple[int, ...]:
        layers = [int(self.action_router_feature_layer_index)]
        if include_critic:
            layers.append(int(self.action_critic_feature_layer_index))
        return tuple(sorted(set(layers)))

    def _first_frame_features_from_layers(
        self,
        layer_tokens: dict[int, torch.Tensor],
        video_meta: dict[str, Any],
        layer_index: int,
    ) -> torch.Tensor:
        if layer_index not in layer_tokens:
            raise RuntimeError(f"Video forward did not snapshot layer {layer_index}.")
        return self._extract_first_frame_view_tokens(layer_tokens[layer_index], video_meta).detach()

    def _run_mot_video_action(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        proprio: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        force_uniform_router: bool = False,
        include_critic: bool = True,
        state_keep_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run video then action. Heads read first-frame visual tokens at configured layers."""
        router_layer = int(self.action_router_feature_layer_index)
        critic_layer = int(self.action_critic_feature_layer_index)
        video_out = self._forward_video_branch(
            video_pre=video_pre,
            attention_mask=attention_mask,
            proprio=proprio,
            snapshot_layer_indices=self._head_snapshot_layers(include_critic=include_critic),
        )
        router_features = self._first_frame_features_from_layers(
            video_out["layer_tokens"],
            video_pre["meta"],
            router_layer,
        )
        head_features = {"router": router_features}
        if include_critic:
            if critic_layer == router_layer:
                head_features["critic"] = router_features
            else:
                head_features["critic"] = self._first_frame_features_from_layers(
                    video_out["layer_tokens"],
                    video_pre["meta"],
                    critic_layer,
                )
        router_output = self._build_action_router_output(
            video_pre=video_pre,
            first_frame_features=router_features,
            force_uniform_router=force_uniform_router,
            proprio=proprio,
            state_keep_mask=state_keep_mask,
        )
        action_tokens = self._forward_action_with_video_cache(
            action_pre=action_pre,
            video_kv_cache=video_out["kv_cache"],
            attention_mask=attention_mask,
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_meta=video_pre["meta"],
            router_output=router_output,
        )
        return (
            {"video": video_out["tokens"], "action": action_tokens},
            router_output,
            head_features,
        )



















    def _camera_view_names(self, num_views: int) -> tuple[str, ...]:
        return resolve_camera_metric_names(num_views, getattr(self, "camera_metric_names", None))

    def _video_gt_action_condition(self, action: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Keep Wan behavior unchanged, but avoid leaking GT action into LingBot video tokens."""
        video_expert = getattr(self, "video_expert", None)
        module_name = "" if video_expert is None else str(video_expert.__class__.__module__)
        if module_name.startswith("flexwam.models.backbone.lingbot_video."):
            return None
        return action

    @staticmethod
    def _router_weights_to_cpu(router_output: dict[str, torch.Tensor]) -> torch.Tensor:
        return router_output["weights"].detach().to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _summarize_action_attention_stats(
        raw_stats: list[dict[str, float]],
        num_views: int,
        view_names: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        if not raw_stats:
            return {}
        view_names = resolve_camera_metric_names(num_views, view_names)
        last_step = max((int(item["step"]) for item in raw_stats if "step" in item), default=None)

        def _collect(
            records: list[dict[str, float]],
            *,
            layers: Optional[set[int]] = None,
            step: Optional[int] = None,
        ) -> dict[str, float]:
            filtered = []
            for item in records:
                if layers is not None and int(item.get("layer", -1)) not in layers:
                    continue
                if step is not None and int(item.get("step", -1)) != step:
                    continue
                filtered.append(item)
            profile: dict[str, float] = {"num_records": float(len(filtered))}
            if not filtered:
                return profile
            for view_idx in range(int(num_views)):
                raw_key = f"view{view_idx}"
                values = [float(item[raw_key]) for item in filtered if raw_key in item]
                if values:
                    name = view_names[view_idx] if view_idx < len(view_names) else raw_key
                    profile[name] = float(sum(values) / len(values))
            self_values = [float(item["action_self"]) for item in filtered if "action_self" in item]
            if self_values:
                profile["action_self"] = float(sum(self_values) / len(self_values))
            return profile

        all_profile = _collect(raw_stats)
        mid_profile = _collect(raw_stats, layers=set(range(12, 19)), step=last_step)
        late_profile = _collect(raw_stats, layers=set(range(18, 25)), step=last_step)

        summary: dict[str, Any] = {
            "profiles": {
                "all": all_profile,
                "layers_12_18": mid_profile,
                "layers_18_24": late_profile,
            },
            "num_records": float(len(raw_stats)),
        }
        for name in view_names:
            if name in all_profile:
                summary[f"attn_{name}"] = all_profile[name]
            if name in mid_profile:
                summary[f"attn_12_18_{name}"] = mid_profile[name]
            if name in late_profile:
                summary[f"attn_18_24_{name}"] = late_profile[name]
        if "action_self" in all_profile:
            summary["attn_action_self"] = all_profile["action_self"]
        if "action_self" in mid_profile:
            summary["attn_12_18_action_self"] = mid_profile["action_self"]
        if "action_self" in late_profile:
            summary["attn_18_24_action_self"] = late_profile["action_self"]
        return summary

    @staticmethod
    def _summarize_action_attention_heatmaps(
        raw_heatmaps: dict[str, Any],
        num_views: int,
        view_names: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        view_names = resolve_camera_metric_names(num_views, view_names)

        def _average_records(records: list[dict[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {"num_records": float(len(records))}
            if not records:
                return result
            layers = sorted({int(item["layer"]) for item in records if "layer" in item})
            steps = sorted({int(item["step"]) for item in records if "step" in item})
            result["layers"] = layers
            result["steps"] = steps
            for view_idx in range(int(num_views)):
                raw_key = f"view{view_idx}"
                maps = [
                    item["views"][raw_key]
                    for item in records
                    if isinstance(item.get("views"), dict) and raw_key in item["views"]
                ]
                if not maps:
                    continue
                name = view_names[view_idx] if view_idx < len(view_names) else raw_key
                result[name] = torch.stack(maps, dim=0).mean(dim=0)
            return result

        return {
            "primary": _average_records(list(raw_heatmaps.get("primary", []))),
            "late": _average_records(list(raw_heatmaps.get("late", []))),
            "primary_policy": {
                "denoise_step": "last",
                "layers": list(raw_heatmaps.get("primary_layers", [])),
                "heads": "mean",
                "action_tokens": "mean",
            },
            "late_policy": {
                "denoise_step": "last",
                "layers": list(raw_heatmaps.get("late_layers", [])),
                "heads": "mean",
                "action_tokens": "mean",
            },
        }

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        input_image = self._canonicalize_input_image_tensor(input_image)
        if input_image.ndim == 5:
            batch_size, num_views, channels, height, width = input_image.shape
            if channels != 3:
                raise ValueError(
                    f"Multiview `input_image` must have shape [B,V,3,H,W], got {tuple(input_image.shape)}"
                )
            flat_images = input_image.to(device=self.device).reshape(batch_size * num_views, channels, height, width)
            image_videos = [image.unsqueeze(1) for image in flat_images]
            z = self.vae.encode(
                image_videos,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if isinstance(z, list):
                z = torch.stack(z, dim=0)
            return z.reshape(batch_size, num_views, *z.shape[1:])

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

    @staticmethod
    def _canonicalize_input_image_tensor(input_image: torch.Tensor) -> torch.Tensor:
        if input_image.ndim == 3:
            return input_image.unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] == 1 and input_image.shape[1] == 3:
                return input_image
            if input_image.shape[1] == 3:
                return input_image.unsqueeze(0)
        if input_image.ndim == 5 and input_image.shape[0] == 1 and input_image.shape[2] == 3:
            return input_image
        raise ValueError(
            "`input_image` must have shape [3,H,W], [1,3,H,W], [V,3,H,W], "
            f"or [1,V,3,H,W], got {tuple(input_image.shape)}"
        )

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if latents.ndim == 6:
            batch_size, num_views = latents.shape[:2]
            if batch_size != 1:
                raise ValueError(f"Multiview decode currently expects batch size 1, got {batch_size}")
            flat_latents = latents.reshape(batch_size * num_views, *latents.shape[2:])
            video_tensor = self.vae.decode(
                flat_latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            video_tensor = video_tensor.reshape(batch_size, num_views, *video_tensor.shape[1:])
            video_tensor = video_tensor[0].detach().float().clamp(-1, 1)
            video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
            videos = []
            for view_idx in range(video_tensor.shape[0]):
                frames = []
                for t in range(video_tensor.shape[2]):
                    frame = video_tensor[view_idx, :, t].permute(1, 2, 0).numpy()
                    frames.append(Image.fromarray(frame))
                videos.append(frames)
            return videos

        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames


    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_prefix_len: int = 0,
        num_views: int = 1,
        video_meta: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        action_prefix_len = int(action_prefix_len)
        if action_prefix_len < 0 or action_prefix_len > int(action_seq_len):
            raise ValueError(
                f"`action_prefix_len` must be in [0,{action_seq_len}], got {action_prefix_len}."
            )

        # video -> video
        prefix_layout = str((video_meta or {}).get("prefix_layout", "per_view"))
        video_mask_tokens_per_frame = int(video_tokens_per_frame)
        if prefix_layout == "global":
            video_mask_tokens_per_frame = int(
                (video_meta or {}).get("video_tokens_per_frame", video_tokens_per_frame)
            )
            video_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_mask_tokens_per_frame,
                device=device,
                num_views=num_views,
                prefix_len=int((video_meta or {}).get("prefix_len", 0)),
                prefix_layout="global",
            )
        else:
            video_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_mask_tokens_per_frame,
                device=device,
                num_views=num_views,
            )
        if self.state_conditioning_mode == "shared" and not self.video_expert.state_conditioning["enabled"]:
            # Keep state available to ActionDiT without leaking through LingBot language.
            video_mask[1:, 0] = False
        mask[:video_seq_len, :video_seq_len] = video_mask
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # Action's own state prefix does not read noisy actions. It reads the
        # same language and first-frame video tokens as action queries, with
        # per-view router weighting, but not the video state token.
        if action_prefix_len > 0:
            prefix_start = video_seq_len
            prefix_end = video_seq_len + action_prefix_len
            mask[prefix_start:prefix_end, :] = False
            prefix_mask = torch.zeros(
                (action_prefix_len, action_prefix_len),
                dtype=torch.bool,
                device=device,
            )
            prefix_mask[0, 0] = True
            mask[prefix_start:prefix_end, prefix_start:prefix_end] = prefix_mask
        # Share layout validation and visual ranges with camera heads and MoT.
        meta = dict(video_meta or {})
        meta.setdefault("num_views", int(num_views))
        view_slices = _video_visual_view_slices(video_seq_len, meta)
        prefix_len = int(meta.get("prefix_len", 0))
        language_len = int(meta.get("context_len", 0))
        language_start = int(meta.get("context_start", 0))
        first_frame_tokens = int(meta.get("video_tokens_per_frame", video_tokens_per_frame))
        if prefix_layout != "global" and "prefix_len" not in meta:
            first_frame_tokens = min(first_frame_tokens, view_slices[0][1] - view_slices[0][0])
        if not 0 < first_frame_tokens <= view_slices[0][1] - view_slices[0][0]:
            raise ValueError("First-frame tokens must fit within each view's visual range.")

        action_query_start = video_seq_len + action_prefix_len
        if prefix_layout == "global" and language_len:
            mask[action_query_start:, language_start:language_start + language_len] = True
        if self.state_conditioning_mode == "shared":
            mask[action_query_start:, 0] = True
        for start, _ in view_slices:
            if prefix_layout != "global" and language_len:
                prefix_start_in_view = start - prefix_len
                mask[action_query_start:, prefix_start_in_view:prefix_start_in_view + language_len] = True
            mask[action_query_start:, start:start + first_frame_tokens] = True
        if action_prefix_len > 0 and action_query_start < total_seq_len:
            mask[prefix_start:prefix_end, :video_seq_len] = mask[action_query_start, :video_seq_len]
        return mask












    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
