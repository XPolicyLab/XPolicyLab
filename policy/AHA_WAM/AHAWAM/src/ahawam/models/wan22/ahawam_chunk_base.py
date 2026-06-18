from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ahawam.utils.logging_config import get_logger

from .base_wam import BaseWAM

logger = get_logger(__name__)


class MultiQueryChunkObsEncoder(nn.Module):
    """Encode multiple obs-conditioned latent queries per action chunk."""

    def __init__(self, *, text_dim: int, num_queries: int):
        super().__init__()
        self.text_dim = int(text_dim)
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}")
        self.base_queries = nn.Parameter(
            torch.randn(1, 1, self.num_queries, self.text_dim)
            / (float(self.text_dim) ** 0.5)
        )
        self.obs_key_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self.obs_value_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        )

    def forward(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if obs_context.ndim != 4:
            raise ValueError(
                "`obs_context` must be [B, N, L, D], "
                f"got shape {tuple(obs_context.shape)}"
            )
        if obs_context_mask.ndim != 3:
            raise ValueError(
                "`obs_context_mask` must be [B, N, L], "
                f"got shape {tuple(obs_context_mask.shape)}"
            )
        batch_size, num_chunks, _, _ = obs_context.shape
        queries = self.base_queries.expand(batch_size, num_chunks, -1, -1)
        keys = self.obs_key_proj(obs_context)
        values = self.obs_value_proj(obs_context)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (
            float(self.text_dim) ** 0.5
        )
        scores = scores.masked_fill(~obs_context_mask.unsqueeze(2), float("-inf"))
        obs_weights = torch.softmax(scores, dim=-1)
        obs_guided_queries = torch.matmul(obs_weights, values)
        return self.query_proj(obs_guided_queries)


class AHAWAMChunkBase(BaseWAM):
    obs_context_causal: bool = False
    ACTION_TRAIN_TIMESTEP_MODES = ("per_chunk", "shared")

    def configure_action_chunking(
        self,
        *,
        action_horizon: int,
        action_chunk_size: int,
        action_train_timestep_mode: str = "per_chunk",
    ) -> None:
        """Set chunk-action hyperparameters after model construction."""
        self.action_horizon = int(action_horizon)
        self.action_chunk_size = int(action_chunk_size)
        self.action_train_timestep_mode = self._normalize_action_train_timestep_mode(
            action_train_timestep_mode
        )
        self._validate_action_chunking_configuration()

    @staticmethod
    def _normalize_action_train_timestep_mode(mode: str) -> str:
        key = str(mode).strip().lower()
        if key not in AHAWAMChunkBase.ACTION_TRAIN_TIMESTEP_MODES:
            raise ValueError(
                "`action_train_timestep_mode` must be one of "
                f"{AHAWAMChunkBase.ACTION_TRAIN_TIMESTEP_MODES}, got {mode}"
            )
        return key

    def _validate_action_chunking_configuration(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be > 0, got {self.action_horizon}")
        if self.action_chunk_size <= 0:
            raise ValueError(
                f"`action_chunk_size` must be > 0, got {self.action_chunk_size}"
            )
        if self.action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        expert_chunk_size = int(
            getattr(self.action_expert, "action_chunk_size", self.action_chunk_size)
        )
        if expert_chunk_size != self.action_chunk_size:
            raise ValueError(
                f"`action_expert.action_chunk_size` ({expert_chunk_size}) must match "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        if not bool(
            getattr(self.action_expert, "autoregressive_teacher_forcing", False)
        ):
            raise ValueError(
                "Chunk action training requires "
                "`action_expert.autoregressive_teacher_forcing=true`."
            )

    def _validate_runtime_action_horizon(self, action_horizon: int) -> None:
        if int(action_horizon) != self.action_horizon:
            raise ValueError(
                f"`action_horizon` must equal configured action horizon {self.action_horizon}, "
                f"got {action_horizon}."
            )

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the joint video/action mask with chunk-action visibility."""
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

        if action_self_attn_mask is None:
            raise ValueError("`action_self_attn_mask` is required for AHAWAMChunkBase.")
        if action_self_attn_mask.ndim != 2:
            raise ValueError(
                "`action_self_attn_mask` must be 2D [S, S], "
                f"got shape {tuple(action_self_attn_mask.shape)}"
            )
        if action_self_attn_mask.shape != (action_seq_len, action_seq_len):
            raise ValueError(
                "`action_self_attn_mask` shape mismatch: "
                f"got {tuple(action_self_attn_mask.shape)} vs expected {(action_seq_len, action_seq_len)}"
            )
        mask[video_seq_len:, video_seq_len:] = action_self_attn_mask.to(device=device)

        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _sample_action_training_timestep(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample shared or per-chunk action timesteps for training."""
        if self.action_train_timestep_mode == "shared":
            return self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=dtype,
            )
        num_chunks = self.action_horizon // self.action_chunk_size
        chunk_timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size * num_chunks,
            device=self.device,
            dtype=dtype,
        ).view(batch_size, num_chunks)
        return chunk_timestep.repeat_interleave(self.action_chunk_size, dim=1)

    def _build_prefilled_action_attention_mask(
        self,
        *,
        current_action_seq_len: int,
        action_history_seq_len: int,
        chunk_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the action-only mask used when earlier chunks already live in KV cache."""
        if action_history_seq_len == 0:
            return self.action_expert.build_single_branch_chunk_causal_mask(
                seq_len=current_action_seq_len,
                chunk_size=self.action_chunk_size,
                device=device,
            )
        return self.action_expert.build_action_self_attention_mask(
            noisy_seq_len=current_action_seq_len,
            chunk_size=self.action_chunk_size,
            device=device,
            clean_seq_len=action_history_seq_len,
            noisy_position_offset=chunk_start,
        )

    @torch.no_grad()
    def _prepare_action_inference(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "AHAWAMChunkBase uses the chunk-local prefill_video/infer_action path; "
            "BaseWAM._prepare_action_inference is not compatible with chunk action masks."
        )

    @staticmethod
    def _filter_state_dict_by_shape(
        source_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        *,
        module_name: str,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        filtered_state = {}
        skipped_keys = []
        for key, value in source_state.items():
            if key not in target_state:
                filtered_state[key] = value
                continue
            if tuple(value.shape) == tuple(target_state[key].shape):
                filtered_state[key] = value
                continue
            skipped_keys.append(
                f"{key}: ckpt {list(value.shape)} vs model {list(target_state[key].shape)}"
            )
        if skipped_keys:
            logger.warning(
                "%s: skipped shape-mismatched params during adaptive checkpoint load:\n  %s",
                module_name,
                "\n  ".join(skipped_keys[:50]),
            )
        return filtered_state, skipped_keys

    @staticmethod
    def _adapt_linear_weight_input_dim(
        source_weight: torch.Tensor, target_weight: torch.Tensor
    ) -> torch.Tensor:
        if source_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError(
                "Linear weight adaptation expects 2D tensors, got "
                f"{tuple(source_weight.shape)} and {tuple(target_weight.shape)}."
            )
        if int(source_weight.shape[0]) != int(target_weight.shape[0]):
            raise ValueError(
                "Cannot adapt linear weight with different output dims: "
                f"{tuple(source_weight.shape)} vs {tuple(target_weight.shape)}."
            )
        if int(target_weight.shape[1]) == int(source_weight.shape[1]) + 1:
            adapted = target_weight.detach().clone().float()
            source = source_weight.float()
            adapted[:, : source.shape[1]] = source
            adapted[:, source.shape[1] :] = source.mean(dim=1, keepdim=True)
            return adapted.to(dtype=target_weight.dtype)
        source = source_weight.float().unsqueeze(1)
        adapted = F.adaptive_avg_pool1d(source, int(target_weight.shape[1])).squeeze(1)
        return adapted.to(dtype=target_weight.dtype)

    def _adapt_action_branch_embedding_state(
        self, mot_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        key = "action_branch_embedding"
        if key not in mot_state:
            return mot_state
        target = self.mot.state_dict().get(key)
        source = mot_state[key]
        if target is None or tuple(source.shape) == tuple(target.shape):
            return mot_state
        if source.ndim == 2 and target.ndim == 2 and int(source.shape[1]) == int(target.shape[1]):
            adapted = target.detach().clone()
            if int(source.shape[0]) == 1 and int(target.shape[0]) >= 2:
                adapted[1] = source[0].to(device=target.device, dtype=target.dtype)
            elif int(source.shape[0]) >= 2 and int(target.shape[0]) == 1:
                adapted[0] = source[1].to(device=target.device, dtype=target.dtype)
            else:
                rows = min(int(source.shape[0]), int(target.shape[0]))
                adapted[:rows] = source[:rows].to(device=target.device, dtype=target.dtype)
            mot_state = dict(mot_state)
            mot_state[key] = adapted.cpu()
            logger.info(
                "Adapted mot.action_branch_embedding shape %s -> %s for checkpoint compatibility.",
                tuple(source.shape),
                tuple(target.shape),
            )
        return mot_state

    def _load_proprio_encoder_state(
        self, ckpt_state: dict[str, torch.Tensor], *, adapt_shapes: bool
    ) -> None:
        assert self.proprio_encoder is not None
        if not adapt_shapes:
            self.proprio_encoder.load_state_dict(ckpt_state, strict=True)
            return

        model_state = self.proprio_encoder.state_dict()
        filtered_state = {}
        skipped_keys = []
        for key, value in ckpt_state.items():
            if key not in model_state:
                filtered_state[key] = value
                continue
            target = model_state[key]
            if tuple(value.shape) == tuple(target.shape):
                filtered_state[key] = value
                continue
            if key == "weight" and value.ndim == 2 and target.ndim == 2:
                filtered_state[key] = self._adapt_linear_weight_input_dim(value, target)
                logger.info(
                    "proprio_encoder: adapted weight input dim %d -> %d by adaptive average pooling.",
                    int(value.shape[1]),
                    int(target.shape[1]),
                )
            else:
                skipped_keys.append(
                    f"{key}: ckpt {list(value.shape)} vs model {list(target.shape)}"
                )

        if skipped_keys:
            logger.warning(
                "proprio_encoder: skipped shape-mismatched params during adaptive checkpoint load:\n  %s",
                "\n  ".join(skipped_keys),
            )
        incompatible = self.proprio_encoder.load_state_dict(filtered_state, strict=False)
        if incompatible.missing_keys:
            logger.warning(
                "proprio_encoder adaptive load missing keys: %s",
                incompatible.missing_keys,
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "proprio_encoder adaptive load unexpected keys: %s",
                incompatible.unexpected_keys,
            )

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
        from .base_wam import BaseWAM

        if self.proprio_encoder is not None:
            self.proprio_encoder = self.proprio_encoder.to(
                device=self.device,
                dtype=self.torch_dtype,
            )

        return BaseWAM.evaluate_validation(
            self,
            sample,
            eval_num_inference_steps=eval_num_inference_steps,
            eval_dir=eval_dir,
            global_step=global_step,
            process_index=process_index,
        )

    def configure_chunk_obs_context(
        self,
        *,
        action_obs_downsample_factor: int = 2,
        chunk_kv_editor_num_queries: int = 32,
        loss_lambda_action_prior: float = 1.0,
        query_init_from_ckpt: str = "random",
    ) -> None:
        self.action_obs_downsample_factor = int(action_obs_downsample_factor)
        if self.action_obs_downsample_factor <= 0:
            raise ValueError(
                "`action_obs_downsample_factor` must be > 0, "
                f"got {self.action_obs_downsample_factor}"
            )

        vae_latent_channels = getattr(self.vae, "latent_dim", None)
        if vae_latent_channels is None:
            vae_latent_channels = getattr(self.vae, "z_dim", None)
        if vae_latent_channels is None:
            raise ValueError("Could not infer VAE latent channels from `self.vae`.")

        self.action_obs_visual_proj = nn.Sequential(
            nn.LayerNorm(int(vae_latent_channels)),
            nn.Linear(int(vae_latent_channels), self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.loss_lambda_action_prior = float(loss_lambda_action_prior)
        self._query_init_from_ckpt = str(query_init_from_ckpt)
        self.chunk_obs_query_encoder = MultiQueryChunkObsEncoder(
            text_dim=self.text_dim,
            num_queries=int(chunk_kv_editor_num_queries),
        ).to(device=self.device, dtype=self.torch_dtype)
        self.mot.configure_chunk_kv_cache_editor(
            query_dim=self.text_dim,
            use_delta_gate=True,
        )

    def get_additional_trainable_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        parent_getter = getattr(super(), "get_additional_trainable_modules", None)
        if callable(parent_getter):
            parent_modules = parent_getter()
            if parent_modules is not None:
                if not isinstance(parent_modules, dict):
                    raise TypeError(
                        "`get_additional_trainable_modules()` must return dict[str, nn.Module], "
                        f"got {type(parent_modules)}"
                    )
                modules.update(parent_modules)
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized via `configure_chunk_obs_context()`."
            )
        modules["action_obs_visual_proj"] = action_obs_visual_proj
        modules["chunk_obs_query_encoder"] = self.chunk_obs_query_encoder
        return modules

    def _build_chunk_kv_queries(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.chunk_obs_query_encoder(
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
        )

    def _split_visual_obs_and_proprio_context(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()
        if proprio_tokens_per_chunk != 1:
            raise ValueError(
                "Chunk KV editor currently expects one proprio token per chunk, "
                f"got {proprio_tokens_per_chunk}."
            )
        if obs_context.shape[2] <= proprio_tokens_per_chunk:
            raise ValueError(
                "`obs_context` must contain visual obs tokens plus proprio tokens."
            )
        visual_context = obs_context[:, :, :-proprio_tokens_per_chunk]
        visual_mask = obs_context_mask[:, :, :-proprio_tokens_per_chunk]
        proprio_context = obs_context[:, :, -proprio_tokens_per_chunk:]
        proprio_mask = obs_context_mask[:, :, -proprio_tokens_per_chunk:]
        return visual_context, visual_mask, proprio_context, proprio_mask

    def _build_action_obs_context(
        self,
        obs_latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized for required obs context."
            )
        if obs_latents is None:
            raise ValueError(
                "`obs_latents` is required to build mandatory obs context."
            )
        normalized_obs_latents = obs_latents

        if normalized_obs_latents.ndim == 4:
            normalized_obs_latents = normalized_obs_latents.unsqueeze(0)
        if normalized_obs_latents.ndim == 6:
            batch_size, num_chunks = (
                int(normalized_obs_latents.shape[0]),
                int(normalized_obs_latents.shape[1]),
            )
            flat_latents = normalized_obs_latents.reshape(
                batch_size * num_chunks, *normalized_obs_latents.shape[2:]
            )
            flat_tokens, flat_mask = self._build_action_obs_context(flat_latents)
            return (
                flat_tokens.reshape(
                    batch_size, num_chunks, flat_tokens.shape[1], flat_tokens.shape[2]
                ),
                flat_mask.reshape(batch_size, num_chunks, flat_mask.shape[1]),
            )
        if normalized_obs_latents.ndim != 5:
            raise ValueError(
                "`obs_latents` must be [C,1,H,W], [B,C,1,H,W], or [B,N,C,1,H,W], "
                f"got shape {tuple(normalized_obs_latents.shape)}"
            )
        if normalized_obs_latents.shape[2] != 1:
            raise ValueError(
                "`obs_latents` time dim (dim 2) must be 1, "
                f"got shape {tuple(normalized_obs_latents.shape)}"
            )

        spatial_latents = normalized_obs_latents[:, :, 0].to(
            device=self.device, dtype=self.torch_dtype
        )
        if self.action_obs_downsample_factor > 1:
            spatial_latents = F.avg_pool2d(
                spatial_latents,
                kernel_size=self.action_obs_downsample_factor,
                stride=self.action_obs_downsample_factor,
            )
        obs_tokens = spatial_latents.flatten(2).transpose(1, 2).contiguous()
        obs_tokens = action_obs_visual_proj(obs_tokens)
        obs_mask = torch.ones(
            (obs_tokens.shape[0], obs_tokens.shape[1]),
            dtype=torch.bool,
            device=obs_tokens.device,
        )
        return obs_tokens, obs_mask

    def _compute_chunk_obs_video_indices(
        self,
        *,
        action_horizon: int,
        num_video_frames: int,
    ) -> torch.Tensor:
        if num_video_frames <= 1:
            raise ValueError(
                f"`num_video_frames` must be > 1 for chunk-aligned obs, got {num_video_frames}"
            )
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by action_chunk_size ({self.action_chunk_size})."
            )
        video_transitions = num_video_frames - 1
        if action_horizon % video_transitions != 0:
            raise ValueError(
                "Cannot derive action/video alignment for chunk obs context: "
                f"action_horizon={action_horizon}, num_video_frames={num_video_frames}."
            )
        actions_per_video_step = action_horizon // video_transitions
        if self.action_chunk_size % actions_per_video_step != 0:
            raise ValueError(
                "`action_chunk_size` must align with sampled video frames for chunk obs context: "
                f"action_chunk_size={self.action_chunk_size}, actions_per_video_step={actions_per_video_step}."
            )
        sampled_video_step = self.action_chunk_size // actions_per_video_step
        num_chunks = action_horizon // self.action_chunk_size
        chunk_indices = torch.arange(num_chunks, dtype=torch.long) * sampled_video_step
        if int(chunk_indices[-1].item()) >= num_video_frames:
            raise ValueError(
                "Chunk-aligned obs frame index exceeds available sampled video frames: "
                f"max_index={int(chunk_indices[-1].item())}, num_video_frames={num_video_frames}."
            )
        return chunk_indices

    def _compute_chunk_video_cache_indices(
        self,
        *,
        action_horizon: int,
        num_video_frames: int,
    ) -> torch.Tensor:
        if num_video_frames <= 1:
            raise ValueError(
                f"`num_video_frames` must be > 1 for chunk video cache indices, got {num_video_frames}"
            )
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by action_chunk_size ({self.action_chunk_size})."
            )
        num_chunks = action_horizon // self.action_chunk_size
        video_transitions = num_video_frames - 1
        chunk_starts = torch.arange(num_chunks, dtype=torch.float32) * float(
            self.action_chunk_size
        )
        frame_indices = torch.floor(
            chunk_starts * float(video_transitions) / float(action_horizon)
        ).to(dtype=torch.long)
        return frame_indices.clamp(max=num_video_frames - 1)

    def _build_chunk_aligned_obs_context_from_video(
        self,
        *,
        video: torch.Tensor,
        action_horizon: int,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized for required obs context."
            )
        if video.ndim != 5:
            raise ValueError(
                f"`video` must be 5D [B,3,T,H,W], got shape {tuple(video.shape)}"
            )
        batch_size, channels, num_video_frames, height, width = video.shape
        if channels != 3:
            raise ValueError(
                f"`video` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        chunk_indices = self._compute_chunk_obs_video_indices(
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
        ).to(device=video.device)
        selected_images = video.index_select(2, chunk_indices)
        selected_images = selected_images.permute(0, 2, 1, 3, 4).reshape(
            batch_size * int(chunk_indices.shape[0]),
            channels,
            height,
            width,
        )
        obs_latents = self._encode_input_image_latents_tensor(
            input_image=selected_images,
            tiled=tiled,
        )
        if obs_latents.ndim != 5 or obs_latents.shape[2] != 1:
            raise ValueError(
                "Encoded chunk observation latents must have shape [B,C,1,H,W], "
                f"got {tuple(obs_latents.shape)}"
            )
        obs_latents = obs_latents.reshape(
            batch_size,
            int(chunk_indices.shape[0]),
            *obs_latents.shape[1:],
        )
        return self._build_action_obs_context(obs_latents)

    def _normalize_chunk_obs_images(
        self,
        *,
        chunk_obs_images: Optional[torch.Tensor],
        batch_size: int,
        action_horizon: int,
    ) -> Optional[torch.Tensor]:
        if chunk_obs_images is None:
            return None
        normalized_chunk_obs_images = chunk_obs_images
        if normalized_chunk_obs_images.ndim == 4:
            if batch_size != 1:
                raise ValueError(
                    "Unbatched `chunk_obs_images` can only be used when batch_size=1, "
                    f"got batch_size={batch_size} and shape {tuple(normalized_chunk_obs_images.shape)}"
                )
            normalized_chunk_obs_images = normalized_chunk_obs_images.unsqueeze(0)
        if normalized_chunk_obs_images.ndim != 5:
            raise ValueError(
                "`chunk_obs_images` must be [N,3,H,W] or [B,N,3,H,W], "
                f"got shape {tuple(normalized_chunk_obs_images.shape)}"
            )
        expected_num_chunks = action_horizon // self.action_chunk_size
        if normalized_chunk_obs_images.shape[0] != batch_size:
            raise ValueError(
                f"`chunk_obs_images` batch mismatch: {normalized_chunk_obs_images.shape[0]} vs expected {batch_size}"
            )
        if (
            normalized_chunk_obs_images.shape[1] != expected_num_chunks
            or normalized_chunk_obs_images.shape[2] != 3
        ):
            raise ValueError(
                "`chunk_obs_images` must have shape [B, num_chunks, 3, H, W], "
                f"got {tuple(normalized_chunk_obs_images.shape)} vs expected num_chunks={expected_num_chunks}"
            )
        return normalized_chunk_obs_images

    def _build_chunk_aligned_obs_context_from_images(
        self,
        *,
        chunk_obs_images: torch.Tensor,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized for required obs context."
            )
        batch_size, num_chunks, channels, height, width = chunk_obs_images.shape
        if channels != 3:
            raise ValueError(
                "`chunk_obs_images` channel dimension must be 3, "
                f"got shape {tuple(chunk_obs_images.shape)}"
            )
        flat_images = chunk_obs_images.reshape(
            batch_size * num_chunks, channels, height, width
        )
        obs_latents = self._encode_input_image_latents_tensor(
            input_image=flat_images,
            tiled=tiled,
        )
        if obs_latents.ndim != 5 or obs_latents.shape[2] != 1:
            raise ValueError(
                "Encoded chunk observation latents must have shape [B,C,1,H,W], "
                f"got {tuple(obs_latents.shape)}"
            )
        obs_latents = obs_latents.reshape(
            batch_size, num_chunks, *obs_latents.shape[1:]
        )
        return self._build_action_obs_context(obs_latents)

    def _extract_chunk_start_proprio(
        self,
        proprio: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        if proprio.ndim != 3:
            raise ValueError(
                f"`proprio` must be 3D [B, T, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
            )
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by action_chunk_size ({self.action_chunk_size})."
            )
        num_chunks = action_horizon // self.action_chunk_size
        chunk_start_indices = (
            torch.arange(num_chunks, device=proprio.device, dtype=torch.long)
            * self.action_chunk_size
        )
        if int(chunk_start_indices[-1].item()) >= int(proprio.shape[1]):
            raise ValueError(
                "Chunk-start proprio index exceeds available proprio sequence length: "
                f"max_index={int(chunk_start_indices[-1].item())}, proprio_steps={int(proprio.shape[1])}."
            )
        return proprio.index_select(1, chunk_start_indices)

    def _normalize_infer_chunk_start_proprio(
        self,
        *,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        action_horizon: int,
    ) -> torch.Tensor:
        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            raise ValueError(
                "Chunk-start proprio is required for AHAWAMChunkBase but `proprio_encoder` is not configured "
                "(set `proprio_dim` to enable proprio encoding)."
            )
        if proprio is None:
            raise ValueError(
                "`proprio` must provide chunk-start proprio with shape [N,D] or [B,N,D] for AHAWAMChunkBase inference."
            )
        normalized_proprio = proprio
        if normalized_proprio.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    "Unbatched chunk-start `proprio` can only be used when batch_size=1, "
                    f"got batch_size={batch_size} and shape {tuple(normalized_proprio.shape)}"
                )
            normalized_proprio = normalized_proprio.unsqueeze(0)
        if normalized_proprio.ndim != 3:
            raise ValueError(
                "Chunk-start `proprio` must be [N,D] or [B,N,D], "
                f"got shape {tuple(normalized_proprio.shape)}"
            )
        num_chunks = action_horizon // self.action_chunk_size
        if normalized_proprio.shape[0] != batch_size:
            raise ValueError(
                f"Chunk-start `proprio` batch mismatch: {normalized_proprio.shape[0]} vs expected {batch_size}"
            )
        if normalized_proprio.shape[1] != num_chunks:
            raise ValueError(
                f"Chunk-start `proprio` chunk mismatch: {normalized_proprio.shape[1]} vs expected {num_chunks}"
            )
        if self.proprio_dim is None or normalized_proprio.shape[2] != self.proprio_dim:
            raise ValueError(
                f"Chunk-start `proprio` last dim must be {self.proprio_dim}, got {normalized_proprio.shape[2]}"
            )
        return normalized_proprio.to(device=self.device, dtype=self.torch_dtype)

    def _append_chunk_start_proprio_to_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        chunk_start_proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            raise ValueError(
                "`chunk_start_proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
            )
        if chunk_start_proprio.ndim != 3:
            raise ValueError(
                "`chunk_start_proprio` must be [B, num_chunks, D], "
                f"got shape {tuple(chunk_start_proprio.shape)}"
            )
        encoder_device = next(self.proprio_encoder.parameters()).device
        proprio_tokens = self.proprio_encoder(
            chunk_start_proprio.to(device=encoder_device, dtype=self.torch_dtype)
        ).to(device=context.device, dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], chunk_start_proprio.shape[1]),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, proprio_tokens], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _append_proprio_to_obs_context(
        self,
        *,
        obs_context: torch.Tensor,
        obs_context_mask: torch.Tensor,
        chunk_start_proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append chunk-start proprio tokens to obs_context, one per chunk."""
        if self.proprio_encoder is None:
            raise ValueError(
                "Cannot append proprio to obs_context when `proprio_encoder` is not configured."
            )
        if chunk_start_proprio.ndim != 3:
            raise ValueError(
                "`chunk_start_proprio` must be [B, N, D], "
                f"got shape {tuple(chunk_start_proprio.shape)}"
            )
        encoder_device = next(self.proprio_encoder.parameters()).device
        proprio_tokens = self.proprio_encoder(
            chunk_start_proprio.to(device=encoder_device, dtype=self.torch_dtype)
        ).to(device=obs_context.device, dtype=obs_context.dtype)
        proprio_tokens = proprio_tokens.unsqueeze(2)
        new_obs_context = torch.cat([obs_context, proprio_tokens], dim=2)
        num_chunks = chunk_start_proprio.shape[1]
        proprio_mask = torch.ones(
            (obs_context_mask.shape[0], num_chunks, 1),
            dtype=torch.bool,
            device=obs_context_mask.device,
        )
        new_obs_mask = torch.cat([obs_context_mask, proprio_mask], dim=2)
        return new_obs_context, new_obs_mask

    def _should_update_action_history(self) -> bool:
        return True

    def _require_obs_proprio_tokens_per_chunk(self) -> int:
        if self.proprio_encoder is None:
            raise ValueError(
                f"{type(self).__name__} requires `proprio_encoder` for chunk obs conditioning."
            )
        return 1

    def _build_training_obs_context(
        self,
        sample: dict[str, Any],
        action_horizon: int,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Leaf chunk models must override `_build_training_obs_context()`."
        )

    def build_inputs(self, sample, tiled: bool = False):
        inputs = self._build_standard_inputs(sample, tiled=tiled)
        self._validate_runtime_action_horizon(int(inputs["action"].shape[1]))
        if not inputs["fuse_vae_embedding_in_latents"]:
            raise ValueError(
                "AHAWAMChunkBase requires `fuse_vae_embedding_in_latents=True` "
                "in the video expert config, got False."
            )
        action_horizon = int(inputs["action"].shape[1])
        obs_context, obs_context_mask = self._build_training_obs_context(
            sample, action_horizon, tiled
        )

        if self.proprio_encoder is not None:
            if "proprio" not in sample or sample["proprio"] is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            base_context = inputs["context"]
            base_context_mask = inputs["context_mask"]
            if base_context.shape[1] < 1 or base_context_mask.shape[1] < 1:
                raise ValueError(
                    "Expected parent context to contain the appended first-frame proprio token."
                )
            inputs["context"] = base_context[:, :-1]
            inputs["context_mask"] = base_context_mask[:, :-1]
            chunk_start_proprio = self._extract_chunk_start_proprio(
                sample["proprio"].to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                ),
                action_horizon,
            )
            obs_context, obs_context_mask = self._append_proprio_to_obs_context(
                obs_context=obs_context,
                obs_context_mask=obs_context_mask,
                chunk_start_proprio=chunk_start_proprio,
            )
        inputs["obs_context"] = obs_context
        inputs["obs_context_mask"] = obs_context_mask
        return inputs

    def _compute_losses_from_pre_states(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        target_video: torch.Tensor,
        target_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        image_is_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute video + action losses from pre_dit states."""
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            action_self_attn_mask=action_pre["self_attn_mask"],
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
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_token.device, dtype=action_loss_token.dtype
        )
        if action_weight.ndim == 1:
            if action_is_pad is not None:
                valid = (~action_is_pad).to(
                    device=action_loss_token.device, dtype=action_loss_token.dtype
                )
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(
                    dim=1
                ) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)
            loss_action = (action_loss_per_sample * action_weight).mean()
        else:
            if action_weight.shape != action_loss_token.shape:
                raise ValueError(
                    "`action_weight` shape mismatch: "
                    f"got {tuple(action_weight.shape)} vs expected {tuple(action_loss_token.shape)}"
                )
            if action_is_pad is not None:
                valid = (~action_is_pad).to(
                    device=action_loss_token.device, dtype=action_loss_token.dtype
                )
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (
                    action_loss_token * action_weight * valid
                ).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = (action_loss_token * action_weight).mean(dim=1)
            loss_action = action_loss_per_sample.mean()
        loss_total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        obs_context = inputs["obs_context"]
        obs_context_mask = inputs["obs_context_mask"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )

        latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self._sample_action_training_timestep(
            batch_size=batch_size,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        clean_timestep_action = torch.zeros_like(
            timestep_action, dtype=action.dtype, device=self.device
        )
        obs_proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            clean_action_tokens=action,
            clean_timestep=clean_timestep_action,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=0,
            single_branch_chunk_causal=False,
            obs_chunk_offset=0,
            obs_context_causal=self.obs_context_causal,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
        )
        return self._compute_losses_from_pre_states(
            video_pre=video_pre,
            action_pre=action_pre,
            target_video=target_video,
            target_action=target_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
        )

    @torch.no_grad()
    def prefill_video(
        self,
        *,
        prompt: Optional[str] = None,
        input_image: torch.Tensor,
        action_horizon: int,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        video_frame_index: Optional[int] = None,
    ) -> dict[str, Any]:
        del video_frame_index
        self.eval()
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` ({action_horizon}) must be divisible by "
                f"`action_chunk_size` ({self.action_chunk_size})."
            )
        video_mask_mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if video_mask_mode not in {"first_frame_causal", "per_frame_causal"}:
            raise ValueError(
                "Two-phase inference requires `video_attention_mask_mode` to be "
                "'first_frame_causal' or 'per_frame_causal'."
            )

        latents_action, batch_size = self._prepare_action_start_latents(
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=None,
            seed=seed,
            rand_device=rand_device,
        )
        context, context_mask = self._prepare_action_context(
            prompt=prompt,
            batch_size=batch_size,
            proprio=None,
            context=context,
            context_mask=context_mask,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )

        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        if not fuse_flag:
            raise ValueError(
                "AHAWAMChunkBase requires `fuse_vae_embedding_in_latents=True`."
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
            "video_kv_cache": video_kv_cache,
            "action_history_kv_cache": None,
            "action_history_seq_len": 0,
        }

    def _prepare_inference_chunk_conditioning(
        self,
        *,
        chunk_obs_image: torch.Tensor,
        chunk_proprio: Optional[torch.Tensor],
        chunk_index: int,
        inference_state: dict[str, Any],
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare obs/proprio conditioning for a single inference chunk."""
        del chunk_obs_image, chunk_proprio, chunk_index, inference_state, tiled
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_prepare_inference_chunk_conditioning`."
        )

    def _encode_cross_attn_context_tokens(
        self,
        *,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_expert.text_embedding(context_tokens)

    def _compute_cross_attn_kv_cache_for_tokens(
        self,
        *,
        context_tokens: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        return self.mot.compute_cross_attn_kv(
            self._encode_cross_attn_context_tokens(context_tokens=context_tokens)
        )

    def _prepare_inference_cross_attn_kv_cache(
        self,
        *,
        inference_state: dict[str, Any],
        context: torch.Tensor,
        obs_context: Optional[torch.Tensor],
    ) -> Optional[list[dict[str, torch.Tensor]]]:
        del inference_state, context, obs_context
        raise NotImplementedError(
            "Leaf chunk models must override `_prepare_inference_cross_attn_kv_cache()`."
        )

    @torch.no_grad()
    def infer_action_chunk(
        self,
        *,
        inference_state: dict[str, Any],
        chunk_obs_image: torch.Tensor,
        chunk_proprio: Optional[torch.Tensor] = None,
        chunk_index: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()

        chunk_start = chunk_index * self.action_chunk_size
        chunk_end = chunk_start + self.action_chunk_size
        action_horizon = int(inference_state["start_latents"].shape[1])
        total_chunks = action_horizon // self.action_chunk_size

        if chunk_index < 0 or chunk_index >= total_chunks:
            raise ValueError(
                f"`chunk_index` must be in [0, {total_chunks}), got {chunk_index}."
            )

        context = inference_state["context"]
        context_mask = inference_state["context_mask"]
        video_kv_cache = inference_state["video_kv_cache"]
        video_seq_len = inference_state["video_seq_len"]
        video_tokens_per_frame = inference_state["video_tokens_per_frame"]
        action_history_kv_cache = inference_state["action_history_kv_cache"]
        action_history_seq_len = inference_state["action_history_seq_len"]

        batch_size = int(inference_state["batch_size"])
        chunk_obs_image = chunk_obs_image.to(device=self.device, dtype=self.torch_dtype)
        if chunk_obs_image.ndim == 3:
            chunk_obs_image = chunk_obs_image.unsqueeze(0)
        if chunk_obs_image.ndim != 4 or chunk_obs_image.shape[0] != batch_size:
            raise ValueError(
                f"`chunk_obs_image` must be [3,H,W] or [{batch_size},3,H,W], "
                f"got shape {tuple(chunk_obs_image.shape)}"
            )
        if self.proprio_encoder is None:
            raise ValueError(
                f"{type(self).__name__} requires `proprio_encoder` for chunk inference."
            )
        if chunk_proprio is None:
            raise ValueError(
                f"`chunk_proprio` is required for `infer_action_chunk`. "
                f"(chunk_index={chunk_index})"
            )
        (
            chunk_conditioning_context,
            chunk_conditioning_mask,
            chunk_conditioning_offset,
        ) = self._prepare_inference_chunk_conditioning(
            chunk_obs_image=chunk_obs_image,
            chunk_proprio=chunk_proprio,
            chunk_index=chunk_index,
            inference_state=inference_state,
            tiled=tiled,
        )
        video_kv_cache = inference_state["_chunk_video_kv_cache"]
        video_seq_len = inference_state["video_seq_len"]
        cross_attn_kv_cache = self._prepare_inference_cross_attn_kv_cache(
            inference_state=inference_state,
            context=context,
            obs_context=chunk_conditioning_context,
        )
        obs_proprio_tokens_per_chunk = self._require_obs_proprio_tokens_per_chunk()

        current_latents = inference_state["start_latents"]
        noisy_chunk = current_latents[:, chunk_start:chunk_end].clone()

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=int(num_inference_steps),
                device=self.device,
                dtype=noisy_chunk.dtype,
                shift_override=sigma_shift,
            )
        )

        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action
        ):
            timestep_action = step_t_action.expand(noisy_chunk.shape[0]).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            pred_action = self._predict_action_chunk_with_clean_cache(
                noisy_action_chunk=noisy_chunk,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                obs_context=chunk_conditioning_context,
                obs_context_mask=chunk_conditioning_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                action_history_kv_cache=action_history_kv_cache,
                action_history_seq_len=action_history_seq_len,
                cross_attn_kv_cache=cross_attn_kv_cache,
                chunk_start=chunk_start,
                obs_chunk_offset=chunk_conditioning_offset,
                obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            )
            noisy_chunk = self.infer_action_scheduler.step(
                pred_action, step_delta_action, noisy_chunk
            )

        current_latents[:, chunk_start:chunk_end] = noisy_chunk

        if self._should_update_action_history():
            action_history_kv_cache = self._prefill_clean_action_chunk_cache(
                clean_action_chunk=noisy_chunk,
                context=context,
                context_mask=context_mask,
                obs_context=chunk_conditioning_context,
                obs_context_mask=chunk_conditioning_mask,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                action_history_kv_cache=action_history_kv_cache,
                action_history_seq_len=action_history_seq_len,
                cross_attn_kv_cache=cross_attn_kv_cache,
                chunk_start=chunk_start,
                obs_chunk_offset=chunk_conditioning_offset,
                obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            )
            action_history_seq_len += self.action_chunk_size

        updated_state = {
            **inference_state,
            "start_latents": current_latents,
            "action_history_kv_cache": action_history_kv_cache,
            "action_history_seq_len": action_history_seq_len,
            "cross_attn_kv_cache": cross_attn_kv_cache,
        }

        return {
            "action_chunk": noisy_chunk[0]
            .detach()
            .to(device="cpu", dtype=torch.float32),
            "final_latents_chunk": noisy_chunk.detach().clone(),
            "chunk_index": int(chunk_index),
            "inference_state": updated_state,
        }

    @torch.no_grad()
    def _build_prefilled_action_pre_and_mask(
        self,
        *,
        action_chunk: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: Optional[torch.Tensor] = None,
        obs_context_mask: Optional[torch.Tensor] = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_chunk,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=self.action_chunk_size,
            noisy_position_offset=chunk_start,
            single_branch_chunk_causal=True,
            obs_chunk_offset=obs_chunk_offset,
            obs_context_causal=self.obs_context_causal,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            skip_context_embedding=cross_attn_kv_cache is not None,
        )
        total_action_seq_len = action_history_seq_len + action_pre["tokens"].shape[1]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=total_action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=action_pre["tokens"].device,
            action_self_attn_mask=self._build_prefilled_action_attention_mask(
                current_action_seq_len=action_pre["tokens"].shape[1],
                action_history_seq_len=action_history_seq_len,
                chunk_start=chunk_start,
                device=action_pre["tokens"].device,
            ),
        )
        return action_pre, attention_mask

    @torch.no_grad()
    def _predict_action_chunk_with_clean_cache(
        self,
        *,
        noisy_action_chunk: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_kv_cache: Optional[list[dict[str, torch.Tensor]]],
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: Optional[torch.Tensor] = None,
        obs_context_mask: Optional[torch.Tensor] = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        action_pre, attention_mask = self._build_prefilled_action_pre_and_mask(
            action_chunk=noisy_action_chunk,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_history_seq_len=action_history_seq_len,
            chunk_start=chunk_start,
            obs_chunk_offset=obs_chunk_offset,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        prior_embed = self.mot.action_branch_embedding[1].to(
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        action_tokens = action_pre["tokens"] + prior_embed.view(1, 1, -1)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_history_kv_cache=action_history_kv_cache,
            action_history_seq_len=action_history_seq_len,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _prefill_clean_action_chunk_cache(
        self,
        *,
        clean_action_chunk: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
        action_history_kv_cache: Optional[list[dict[str, torch.Tensor]]],
        action_history_seq_len: int,
        chunk_start: int,
        obs_context: Optional[torch.Tensor] = None,
        obs_context_mask: Optional[torch.Tensor] = None,
        obs_chunk_offset: int = 0,
        obs_proprio_tokens_per_chunk: int = 0,
        cross_attn_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        clean_timestep = torch.zeros(
            (clean_action_chunk.shape[0], clean_action_chunk.shape[1]),
            device=self.device,
            dtype=clean_action_chunk.dtype,
        )
        action_pre, attention_mask = self._build_prefilled_action_pre_and_mask(
            action_chunk=clean_action_chunk,
            timestep_action=clean_timestep,
            context=context,
            context_mask=context_mask,
            obs_context=obs_context,
            obs_context_mask=obs_context_mask,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            action_history_seq_len=action_history_seq_len,
            chunk_start=chunk_start,
            obs_chunk_offset=obs_chunk_offset,
            obs_proprio_tokens_per_chunk=obs_proprio_tokens_per_chunk,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        prior_embed = self.mot.action_branch_embedding[1].to(
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        action_tokens = action_pre["tokens"] + prior_embed.view(1, 1, -1)
        delta_kv_cache = self.mot.prefill_action_history_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_history_kv_cache=action_history_kv_cache,
            action_history_seq_len=action_history_seq_len,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        return self.mot.append_kv_cache(action_history_kv_cache, delta_kv_cache)

    @torch.no_grad()
    def infer_action_stream(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "AHAWAMChunkBase does not implement `infer_action_stream`. "
            "Use `infer_action` with `phase='video'` / `phase='action'` for deployment, "
            "or `infer` for evaluation."
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        chunk_obs_image: Optional[torch.Tensor] = None,
        chunk_proprio: Optional[torch.Tensor] = None,
        video_frame_index: Optional[int] = None,
        phase: str = "video",
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        if phase == "video":
            if input_image is None:
                raise ValueError("`input_image` is required for phase='video'.")
            if action_horizon is None:
                action_horizon = self.action_horizon

            prefill_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "input_image": input_image,
                "action_horizon": action_horizon,
                "context": context,
                "context_mask": context_mask,
                "seed": seed,
                "rand_device": rand_device,
                "tiled": tiled,
            }
            if video_frame_index is not None:
                prefill_kwargs["video_frame_index"] = video_frame_index
            self._inference_state = self.prefill_video(**prefill_kwargs)
            self._inference_state["next_chunk_index"] = 0
            return {"phase": "video", "chunk_index": 0}

        if phase == "action":
            if not hasattr(self, "_inference_state") or self._inference_state is None:
                raise RuntimeError(
                    "Must call `infer_action(phase='video')` before `phase='action'`."
                )
            if chunk_obs_image is None:
                raise ValueError("`chunk_obs_image` is required for phase='action'.")

            state = self._inference_state
            chunk_index = state["next_chunk_index"]

            if self.proprio_encoder is not None and chunk_proprio is None:
                raise ValueError(
                    f"`chunk_proprio` is required for phase='action' when `proprio_dim` is enabled. "
                    f"Pass the current chunk's proprio obtained from the environment. "
                    f"(chunk_index={chunk_index})"
                )

            result = self.infer_action_chunk(
                inference_state=state,
                chunk_obs_image=chunk_obs_image,
                chunk_proprio=chunk_proprio,
                chunk_index=chunk_index,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
            )

            self._inference_state = result["inference_state"]
            self._inference_state["next_chunk_index"] = chunk_index + 1

            return {
                "action_chunk": result["action_chunk"],
                "chunk_index": int(chunk_index),
                "phase": "action",
            }

        raise ValueError(f"`phase` must be 'video' or 'action', got '{phase}'.")

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
        chunk_obs_images: Optional[torch.Tensor] = None,
    ):
        del num_frames, action, action_cfg_scale, negative_prompt
        if action_horizon is None:
            action_horizon = self.action_horizon

        if chunk_obs_images is None:
            raise ValueError(
                "`chunk_obs_images` is required for AHAWAMChunkBase evaluation. "
                "Expected shape [num_chunks, 3, H, W] or [B, num_chunks, 3, H, W]."
            )

        self.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            phase="video",
        )

        batch_size = int(self._inference_state["batch_size"])
        normalized = self._normalize_chunk_obs_images(
            chunk_obs_images=chunk_obs_images,
            batch_size=batch_size,
            action_horizon=action_horizon,
        )
        if normalized is None:
            raise ValueError(
                "`chunk_obs_images` normalization failed. "
                "Expected shape [num_chunks, 3, H, W] or [B, num_chunks, 3, H, W]."
            )

        num_chunks = action_horizon // self.action_chunk_size
        # Pre-extract per-chunk proprio for eval loop
        chunk_proprios = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`proprio` is required for evaluation when `proprio_dim` is enabled."
                )
            if proprio.ndim == 2:
                # [num_chunks, D_p] -> treat as batch_size=1 for eval
                chunk_proprios = proprio.unsqueeze(0)
            elif proprio.ndim == 3:
                # [B, num_chunks, D_p] -> keep batch dimension for chunk-wise eval
                chunk_proprios = proprio
            else:
                raise ValueError(
                    f"`proprio` must be [num_chunks, D_p] or [B, num_chunks, D_p], "
                    f"got shape {tuple(proprio.shape)}"
                )

        chunk_outputs = []
        for chunk_index in range(num_chunks):
            obs_image = normalized[:, chunk_index]
            result = self.infer_action(
                chunk_obs_image=obs_image,
                chunk_proprio=(
                    chunk_proprios[:, chunk_index, :]
                    if chunk_proprios is not None
                    else None
                ),
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
                phase="action",
            )
            chunk_outputs.append(result["action_chunk"])

        action = torch.cat(chunk_outputs, dim=0)
        return {"action": action}

    def _build_eval_infer_kwargs(
        self,
        *,
        sample: dict[str, Any],
        eval_num_inference_steps: int,
    ) -> dict[str, Any]:
        kwargs = super()._build_eval_infer_kwargs(
            sample=sample,
            eval_num_inference_steps=eval_num_inference_steps,
        )
        video0 = sample["video"][0]
        _, num_frames, _, _ = video0.shape
        action_horizon = int(kwargs.get("action_horizon", self.action_horizon))

        sample_chunk_obs_images = sample.get("chunk_obs_images")
        if sample_chunk_obs_images is not None:
            if sample_chunk_obs_images.ndim == 5:
                chunk_obs_images = sample_chunk_obs_images[0]
            else:
                chunk_obs_images = sample_chunk_obs_images
            kwargs["chunk_obs_images"] = chunk_obs_images
        else:
            chunk_obs_indices = self._compute_chunk_obs_video_indices(
                action_horizon=action_horizon,
                num_video_frames=num_frames,
            )
            chunk_obs_images = video0[:, chunk_obs_indices].permute(1, 0, 2, 3)
            kwargs["chunk_obs_images"] = chunk_obs_images

        if self.proprio_encoder is not None:
            if "proprio" not in sample or sample["proprio"] is None:
                raise ValueError(
                    "`sample['proprio']` is required for AHAWAMChunkBase evaluation when `proprio_dim` is enabled."
                )
            chunk_start_proprio = self._extract_chunk_start_proprio(
                sample["proprio"][0:1].to(device=self.device, dtype=self.torch_dtype),
                action_horizon,
            )
            kwargs["proprio"] = chunk_start_proprio.detach().cpu()
        return kwargs

    def save_checkpoint(self, path, optimizer=None, step=None):
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized before saving a checkpoint."
            )
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        payload["action_obs_visual_proj"] = action_obs_visual_proj.state_dict()
        payload["chunk_obs_query_encoder"] = self.chunk_obs_query_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        action_obs_visual_proj = getattr(self, "action_obs_visual_proj", None)
        if action_obs_visual_proj is None:
            raise ValueError(
                "`action_obs_visual_proj` must be initialized before loading a checkpoint."
            )
        payload = torch.load(path, map_location="cpu")
        adapt_shapes = bool(getattr(self, "checkpoint_shape_adapt", False))
        if "mot" in payload:
            if "action_branch_embedding" in payload["mot"]:
                logger.info(
                    "Checkpoint contains `mot.action_branch_embedding` with shape=%s.",
                    tuple(payload["mot"]["action_branch_embedding"].shape),
                )
            else:
                logger.warning(
                    "Checkpoint `mot` state is missing `action_branch_embedding`; "
                    "current randomly initialized branch embedding will be kept."
                )
            mot_state = self._adapt_action_branch_embedding_state(payload["mot"])
            if adapt_shapes:
                mot_state, _ = self._filter_state_dict_by_shape(
                    mot_state,
                    self.mot.state_dict(),
                    module_name="mot",
                )
            incompatible = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Loaded `mot` checkpoint with missing_keys=%d unexpected_keys=%d. "
                    "First missing=%s first unexpected=%s",
                    len(missing_keys),
                    len(unexpected_keys),
                    missing_keys[:20],
                    unexpected_keys[:20],
                )
            else:
                logger.info("Loaded `mot` checkpoint with no missing/unexpected keys.")
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self._load_proprio_encoder_state(
                    payload["proprio_encoder"],
                    adapt_shapes=adapt_shapes,
                )
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )

        if "action_obs_visual_proj" in payload:
            action_obs_visual_proj.load_state_dict(
                payload["action_obs_visual_proj"], strict=True
            )
        else:
            logger.warning(
                "Checkpoint has no `action_obs_visual_proj` weights; keeping current `action_obs_visual_proj` params."
            )

        if "chunk_obs_query_encoder" in payload:
            ckpt_state = payload["chunk_obs_query_encoder"]
            model_state = self.chunk_obs_query_encoder.state_dict()
            query_init = getattr(self, "_query_init_from_ckpt", "random")
            # Filter out shape-mismatched parameters, optionally interpolate
            filtered_state = {}
            skipped_keys = []
            for k, v in ckpt_state.items():
                if k in model_state and v.shape != model_state[k].shape:
                    target_shape = model_state[k].shape
                    if query_init == "interpolate" and k == "base_queries":
                        # Interpolate queries: [1, 1, old_q, dim] → [1, 1, new_q, dim]
                        old_q, new_q = v.shape[2], target_shape[2]
                        # F.interpolate expects [N, C, L]: treat dim as channels, queries as length
                        src = v.reshape(1, old_q, -1).permute(0, 2, 1).float()  # [1, dim, old_q]
                        interpolated = F.interpolate(src, size=new_q, mode="linear", align_corners=False)
                        interpolated = interpolated.permute(0, 2, 1).reshape(target_shape).to(v.dtype)
                        # Small noise to break symmetry when queries are repeated (e.g. 32→64)
                        noise_scale = 0.02 * interpolated.std()
                        interpolated = interpolated + torch.randn_like(interpolated) * noise_scale
                        filtered_state[k] = interpolated
                        logger.info(
                            "chunk_obs_query_encoder: interpolated %s from %d → %d queries",
                            k, old_q, new_q,
                        )
                    else:
                        skipped_keys.append(
                            f"{k}: ckpt {list(v.shape)} vs model {list(target_shape)}"
                        )
                else:
                    filtered_state[k] = v
            if skipped_keys:
                logger.warning(
                    "chunk_obs_query_encoder: skipped shape-mismatched params (kept random init):\n  %s",
                    "\n  ".join(skipped_keys),
                )
            incompatible = self.chunk_obs_query_encoder.load_state_dict(
                filtered_state, strict=False
            )
            if incompatible.unexpected_keys:
                logger.warning(
                    "chunk_obs_query_encoder unexpected keys (ignored): %s",
                    incompatible.unexpected_keys,
                )
        else:
            logger.warning(
                "Checkpoint has no `chunk_obs_query_encoder` weights; "
                "keeping current randomly initialized `chunk_obs_query_encoder` params."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
