"""Typed configuration for native Video-Action Pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

from wall_x.config.registry import register_model_config
from wall_x.model.vga.config import VGAModelConfig


def _video_flow_defaults() -> Dict[str, Any]:
    return {
        "shift": 5.0,
        "train_timesteps": 1000,
        "inference_steps": 50,
        "sample_mode": "uniform",
    }


def _action_flow_defaults() -> Dict[str, Any]:
    return {
        "shift": 1.0,
        "train_timesteps": 1000,
        "inference_steps": 8,
        "sample_mode": "uniform",
    }


@register_model_config("vap")
@dataclass
class VAPModelConfig(VGAModelConfig):
    """VAP config while retaining compatible WAN/data fields.

    The inherited WAN fields keep refactor_2604 configs loadable. Fields below
    control only the native two-stream model and never change model_type=wan.
    """

    base_model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    # Opt-in text plugin. Frozen Qwen features are cached or encoded online; only the input
    # projector is trained/saved with VAP. T5 remains the exact default path.
    text_encoder_backend: str = "t5"
    text_feature_cache_id: str | None = None
    text_feature_hidden_dim: int = 0
    text_feature_seq_len: int = 512
    text_projector_output_dim: int = 4096
    # Optional online frozen encoder; never registered with the trainable core.
    qwen35_model_path: str | None = None
    qwen35_model_id: str | None = None
    qwen35_text_recipe: str = "plain_text_eos_last_hidden_v1"
    qwen35_encode_micro_batch: int = 1
    # Serving-only hybrid text path. Cache hits retain the zero-T5 fast path;
    # cache misses may use the frozen Wan UMT5 encoder and are memoized in the
    # process. Training and checkpoint configs keep this disabled by default.
    text_feature_cache_online_fallback: bool = False
    # Paired IVAVA remains fixed to I-A0-V0-A1-V1. Video-only pretraining may
    # use a longer fixed chain I->V0->...->V{E-1}; ``num_interleave_chunks``
    # is still one immutable shape for the entire run.
    num_interleave_chunks: int = 2
    video_frames_per_chunk: int = 20
    action_steps_per_chunk: int = 40
    vae_temporal_stride: int = 4
    action_expert_dim: int = 1536
    # Optional chunk-start proprioceptive conditioning. New chunk-local data
    # provides [S0,S1,...] and applies state[e] to Action chunk Ae. Legacy
    # [S0] batches remain supported and broadcast S0 across every chunk.
    state_film_enabled: bool = False
    state_film_hidden_dim: int = 512
    # Per-sample conditioning dropout. One mask is shared by both Action
    # chunks, and a dropped sample receives exactly zero FiLM modulation.
    state_film_dropout_probability: float = 0.0
    # Optional episode-start visual anchor.  This is an external, read-only
    # condition bank: it never enters the native I-A0-V0-A1-V1 joint sequence.
    # Disabled is a strict compatibility path (no parameters, decode, RNG, or
    # batch-field requirements).  When enabled the data backend must provide a
    # real ``episode_first_obs``; falling back to the current window I is banned.
    first_frame_memory_enabled: bool = False
    # ``gated_cross_attention`` preserves the original external Anchor Bank.
    # ``prefix_self_attention`` prepends Memory to the native condition-video
    # stream, so every Video and Action query can read the complete bank using
    # the ordinary joint self-attention mask.  This is the parameter-free path
    # used by the native joint-self-attention Memory checkpoints.
    first_frame_memory_injection_mode: str = "gated_cross_attention"
    first_frame_memory_grid: tuple[int, int] = (4, 4)
    history_offsets: tuple[int, ...] = ()
    first_frame_memory_reader_layers: tuple[int, ...] = (3, 9, 15, 21, 27)
    # Empty preserves the original all-view Anchor Bank.  A non-empty tuple
    # selects stable VGA ``viewN`` slots before VAE encoding; the native
    # Video stream remains multi-view regardless of this Memory-only filter.
    first_frame_memory_view_keys: tuple[str, ...] = ()
    # ``window_origin`` preserves the original MVP, where the episode anchor
    # and the current window I both occupied temporal coordinate zero.
    # ``relative_age`` keeps the native window-local IVAVA positions intact
    # and places the episode anchor at ``-query_local_idx`` on the chunk axis,
    # exposing its real temporal distance without changing model parameters.
    first_frame_memory_position_mode: str = "window_origin"
    joint_num_heads: int = 24
    joint_head_dim: int = 128
    # Window-local (chunk_start, local_time, height, width). The 44-channel Wan
    # temporal budget is split 8/36 while preserving its 42/42 H/W widths.
    rope_axes_dim: tuple[int, int, int, int] = (8, 36, 42, 42)
    # Exclusive upper bound for physical camera IDs.
    max_views: int = 64
    attention_backend: str = "flex_attention"
    # Isolated two-stage pretraining mode. ``paired_ivava`` is the historical
    # default and must remain behavior-identical. ``video_only`` retains the
    # full checkpoint schema but trains the causal I->V0->V1 marginal without
    # exposing placeholder Action tokens to Video attention.
    training_mode: str = "paired_ivava"
    video_only_loss_scale: float = 1.0
    # Duplicate clean context and noisy target streams so all four IVAVA stages
    # are supervised in one block-sparse trajectory-level forward.
    parallel_teacher_forcing: bool = True
    # Route only noisy A1 through an additional corrupted A0 context copy.
    # V0 and the clean A1->V1 teacher-forcing chain retain clean A0. Disabled
    # is the exact historical layout: no extra tokens, mask, compute, or RNG.
    robust_a0: bool = False
    # Optional robustness ablation for the teacher-forced Action prefix.
    # Only clean local A0 context is corrupted; diffusion inputs/targets stay
    # unchanged.  Defaults preserve the historical training path exactly.
    action_prefix_noise_enabled: bool = False
    action_prefix_noise_probability: float = 0.0
    action_prefix_noise_std: float = 0.0
    action_prefix_noise_warmup_steps: int = 0
    action_prefix_noise_exclude_dims: tuple[int, ...] = ()
    # Sample-gated locked dropout on the encoded clean A0 prefix.  A triggered
    # sample shares one channel mask across every A0 time step, preserving
    # temporal coherence while discouraging brittle feature-level shortcuts.
    action_prefix_embedding_dropout_sample_probability: float = 0.0
    action_prefix_embedding_dropout_rate: float = 0.0
    action_prefix_embedding_dropout_warmup_steps: int = 0
    # Optional per-DOF Action loss weights.  The reduction divides by the
    # weighted valid-element count, so these values change relative gradient
    # allocation without changing the nominal scalar Action-loss scale.
    # Empty preserves the historical uniform reduction exactly.
    action_loss_dimension_weights: tuple[float, ...] = ()
    # ``weighted_mean`` divides by the weighted valid-element count and only
    # reallocates relative gradient mass. ``valid_mean`` divides by the raw
    # valid-element count, so weights also change the scalar Action-loss scale.
    action_loss_dimension_weight_normalization: str = "weighted_mean"
    # Forward Explorative Modeling for the Action expert.  K=1 is a strict
    # baseline no-op.  K>1 samples K independent Action-noise trajectories at
    # the same per-stage sigma, scores A0+A1 jointly per example, and
    # backpropagates only through the lowest-loss candidate.  Video noise and
    # Video supervision are not explored.
    action_xm_best_of_k: int = 1
    # Optional SOAR correction for the Action denoising trajectory. ``off`` is
    # deliberately a strict no-op: it must not sample RNG, run another model
    # pass, or alter the historical loss. ``light`` runs on a synchronized
    # optimizer-step cadence; ``full`` runs on every eligible step.
    action_soar_mode: str = "off"
    action_soar_every_n_steps: int = 4
    action_soar_correction_weight: float = 0.2
    action_soar_num_aux_points: int = 1
    action_soar_warmup_steps: int = 1000
    action_soar_ramp_steps: int = 4000
    action_soar_min_aux_sigma: float = 0.05
    action_soar_apply_stages: tuple[str, ...] = ("a0", "a1")
    # Uniformly sampling one of four AR stages would halve the expected loss
    # relative to the old two-modality joint objective; 2.0 preserves scale.
    ar_loss_scale: float = 2.0
    video_flow: Dict[str, Any] = field(default_factory=_video_flow_defaults)
    action_flow: Dict[str, Any] = field(default_factory=_action_flow_defaults)

    def __post_init__(self) -> None:
        from .text_conditioning import validate_text_config

        validate_text_config(self)
        if self.num_interleave_chunks <= 0:
            raise ValueError("num_interleave_chunks must be positive")
        if self.training_mode == "paired_ivava" and self.num_interleave_chunks != 2:
            raise ValueError(
                "paired_ivava requires num_interleave_chunks=2 (I-A-V-A-V)"
            )
        if self.training_mode == "video_only" and self.num_interleave_chunks < 2:
            raise ValueError("video_only requires at least two video chunks")
        if self.base_model_id != "Wan-AI/Wan2.2-TI2V-5B":
            raise ValueError("VAP milestone 1 requires the Wan2.2 TI2V 5B base")
        if self.action_expert_dim <= 0:
            raise ValueError("action_expert_dim must be positive")
        if self.state_film_hidden_dim <= 0:
            raise ValueError("state_film_hidden_dim must be positive")
        if not 0.0 <= self.state_film_dropout_probability < 1.0:
            raise ValueError("state_film_dropout_probability must be in [0, 1)")
        if not self.state_film_enabled and self.state_film_dropout_probability:
            raise ValueError(
                "state_film_dropout_probability requires state_film_enabled=true"
            )
        self.first_frame_memory_grid = tuple(
            int(value) for value in self.first_frame_memory_grid
        )
        from .history import validate_history_offsets
        self.history_offsets = validate_history_offsets(self.history_offsets)
        if self.history_offsets and not (
            self.first_frame_memory_enabled
            and self.first_frame_memory_injection_mode == "prefix_self_attention"
            and self.first_frame_memory_position_mode == "relative_age"
            and tuple(self.first_frame_memory_view_keys) == ("view1",)
        ):
            raise ValueError("history_offsets requires ego view1 prefix_self_attention / relative_age Memory")
        if len(self.first_frame_memory_grid) != 2 or any(
            value <= 0 for value in self.first_frame_memory_grid
        ):
            raise ValueError(
                "first_frame_memory_grid must contain two positive integers"
            )
        self.first_frame_memory_reader_layers = tuple(
            int(index) for index in self.first_frame_memory_reader_layers
        )
        if (
            not self.first_frame_memory_reader_layers
            or len(set(self.first_frame_memory_reader_layers))
            != len(self.first_frame_memory_reader_layers)
            or tuple(sorted(self.first_frame_memory_reader_layers))
            != self.first_frame_memory_reader_layers
            or any(index < 0 for index in self.first_frame_memory_reader_layers)
        ):
            raise ValueError(
                "first_frame_memory_reader_layers must be a non-empty, sorted "
                "tuple of unique non-negative layer indices"
            )
        self.first_frame_memory_view_keys = tuple(
            str(key) for key in self.first_frame_memory_view_keys
        )
        if (
            len(set(self.first_frame_memory_view_keys))
            != len(self.first_frame_memory_view_keys)
            or any(
                not key.startswith("view") or not key[4:].isdigit() or int(key[4:]) <= 0
                for key in self.first_frame_memory_view_keys
            )
            or tuple(
                sorted(
                    self.first_frame_memory_view_keys,
                    key=lambda key: int(key[4:]),
                )
            )
            != self.first_frame_memory_view_keys
        ):
            raise ValueError(
                "first_frame_memory_view_keys must be an empty or sorted "
                "tuple of unique viewN keys"
            )
        if self.first_frame_memory_position_mode not in {
            "window_origin",
            "relative_age",
        }:
            raise ValueError(
                "first_frame_memory_position_mode must be window_origin or "
                "relative_age"
            )
        if self.first_frame_memory_injection_mode not in {
            "gated_cross_attention",
            "prefix_self_attention",
        }:
            raise ValueError(
                "first_frame_memory_injection_mode must be "
                "gated_cross_attention or prefix_self_attention"
            )
        if (
            self.first_frame_memory_enabled
            and self.first_frame_memory_injection_mode == "prefix_self_attention"
            and self.first_frame_memory_position_mode != "relative_age"
        ):
            raise ValueError(
                "prefix_self_attention first-frame memory requires "
                "first_frame_memory_position_mode=relative_age"
            )
        if self.first_frame_memory_enabled:
            if self.training_mode != "paired_ivava":
                raise ValueError(
                    "first-frame memory is currently supported only for "
                    "training_mode=paired_ivava"
                )
            if not self.parallel_teacher_forcing:
                raise ValueError(
                    "first-frame memory requires parallel_teacher_forcing=true"
                )
        if self.video_frames_per_chunk <= 0 or self.action_steps_per_chunk <= 0:
            raise ValueError("per-chunk video/action horizons must be positive")
        if self.action_steps_per_chunk % self.video_frames_per_chunk:
            raise ValueError(
                "action_steps_per_chunk must be an integer multiple of "
                "video_frames_per_chunk for shared temporal coordinates"
            )
        if self.max_views <= 0:
            raise ValueError("max_views must be a positive physical-view ID bound")
        if self.video_frames_per_chunk % self.vae_temporal_stride:
            raise ValueError(
                "video_frames_per_chunk must be divisible by vae_temporal_stride "
                "so I + all future chunks has Wan's 4N+1 frame structure"
            )
        if self.joint_num_heads * self.joint_head_dim != 3072:
            raise ValueError(
                "joint_num_heads * joint_head_dim must preserve Wan2.2 5B dim=3072"
            )
        self.rope_axes_dim = tuple(int(value) for value in self.rope_axes_dim)
        if len(self.rope_axes_dim) != 4:
            raise ValueError("rope_axes_dim must contain (chunk_start,local_time,h,w)")
        if sum(self.rope_axes_dim) != self.joint_head_dim:
            raise ValueError(
                "sum(rope_axes_dim) must equal joint_head_dim, got "
                f"{sum(self.rope_axes_dim)} != {self.joint_head_dim}"
            )
        if any(value <= 0 or value % 2 for value in self.rope_axes_dim):
            raise ValueError("every rope axis width must be positive and even")
        if self.attention_backend not in {"sdpa", "flex_attention"}:
            raise ValueError("attention_backend must be 'sdpa' or 'flex_attention'")
        if self.training_mode not in {"paired_ivava", "video_only"}:
            raise ValueError(
                "training_mode must be 'paired_ivava' or 'video_only', got "
                f"{self.training_mode!r}"
            )
        if self.video_only_loss_scale <= 0:
            raise ValueError("video_only_loss_scale must be positive")
        if self.training_mode == "video_only":
            if not self.parallel_teacher_forcing:
                raise ValueError("video_only requires parallel_teacher_forcing=true")
            if self.action_loss_weight != 0:
                raise ValueError("video_only requires action_loss_weight=0")
            if self.state_film_enabled:
                raise ValueError(
                    "State FiLM conditions the Action expert and is not valid for "
                    "training_mode=video_only"
                )
        if self.robust_a0:
            if self.training_mode != "paired_ivava":
                raise ValueError("robust_a0 is only valid for paired_ivava training")
            if not self.parallel_teacher_forcing:
                raise ValueError("robust_a0 requires parallel_teacher_forcing=true")
        self.action_prefix_noise_exclude_dims = tuple(
            int(index) for index in self.action_prefix_noise_exclude_dims
        )
        if len(set(self.action_prefix_noise_exclude_dims)) != len(
            self.action_prefix_noise_exclude_dims
        ) or any(
            index < 0 or index >= self.action_dim
            for index in self.action_prefix_noise_exclude_dims
        ):
            raise ValueError(
                "action_prefix_noise_exclude_dims must contain unique "
                "indices inside the configured Action dimension"
            )
        if not 0.0 <= float(self.action_prefix_noise_probability) <= 1.0:
            raise ValueError("action_prefix_noise_probability must be in [0,1]")
        if float(self.action_prefix_noise_std) < 0:
            raise ValueError("action_prefix_noise_std must be non-negative")
        if int(self.action_prefix_noise_warmup_steps) < 0:
            raise ValueError("action_prefix_noise_warmup_steps must be non-negative")
        if self.action_prefix_noise_enabled:
            if self.training_mode != "paired_ivava":
                raise ValueError(
                    "action prefix noise is only valid for paired_ivava training"
                )
            if not self.parallel_teacher_forcing:
                raise ValueError(
                    "action prefix noise requires parallel_teacher_forcing=true"
                )
            if self.action_prefix_noise_probability <= 0:
                raise ValueError(
                    "enabled action prefix noise requires positive probability"
                )
            if self.action_prefix_noise_std <= 0:
                raise ValueError("enabled action prefix noise requires positive std")
        embedding_dropout_sample_probability = float(
            self.action_prefix_embedding_dropout_sample_probability
        )
        embedding_dropout_rate = float(self.action_prefix_embedding_dropout_rate)
        embedding_dropout_warmup_steps = int(
            self.action_prefix_embedding_dropout_warmup_steps
        )
        if not 0.0 <= embedding_dropout_sample_probability <= 1.0:
            raise ValueError(
                "action_prefix_embedding_dropout_sample_probability must be in [0,1]"
            )
        if not 0.0 <= embedding_dropout_rate < 1.0:
            raise ValueError("action_prefix_embedding_dropout_rate must be in [0,1)")
        if embedding_dropout_warmup_steps < 0:
            raise ValueError(
                "action_prefix_embedding_dropout_warmup_steps must be non-negative"
            )
        embedding_dropout_enabled = (
            embedding_dropout_sample_probability > 0.0 or embedding_dropout_rate > 0.0
        )
        if embedding_dropout_enabled:
            if embedding_dropout_sample_probability <= 0.0:
                raise ValueError(
                    "action prefix embedding dropout requires positive sample probability"
                )
            if embedding_dropout_rate <= 0.0:
                raise ValueError(
                    "action prefix embedding dropout requires positive dropout rate"
                )
            if self.training_mode != "paired_ivava":
                raise ValueError(
                    "action prefix embedding dropout is only valid for paired_ivava training"
                )
            if not self.parallel_teacher_forcing:
                raise ValueError(
                    "action prefix embedding dropout requires parallel_teacher_forcing=true"
                )
        if self.robust_a0 and not (
            self.action_prefix_noise_enabled or embedding_dropout_enabled
        ):
            raise ValueError(
                "robust_a0 requires action prefix noise and/or embedding dropout"
            )
        self.action_loss_dimension_weights = tuple(
            float(weight) for weight in self.action_loss_dimension_weights
        )
        if self.action_loss_dimension_weights:
            if len(self.action_loss_dimension_weights) != self.action_dim:
                raise ValueError(
                    "action_loss_dimension_weights must have one entry per "
                    f"Action dimension ({self.action_dim}), got "
                    f"{len(self.action_loss_dimension_weights)}"
                )
            if any(weight <= 0.0 for weight in self.action_loss_dimension_weights):
                raise ValueError(
                    "action_loss_dimension_weights entries must be positive"
                )
        if self.action_loss_dimension_weight_normalization not in {
            "weighted_mean",
            "valid_mean",
        }:
            raise ValueError(
                "action_loss_dimension_weight_normalization must be "
                "weighted_mean or valid_mean"
            )
        self.action_xm_best_of_k = int(self.action_xm_best_of_k)
        if self.action_xm_best_of_k <= 0:
            raise ValueError("action_xm_best_of_k must be a positive integer")
        if self.action_xm_best_of_k > 1:
            if self.training_mode != "paired_ivava":
                raise ValueError("Action XM is only valid for paired_ivava training")
            if not self.parallel_teacher_forcing:
                raise ValueError("Action XM requires parallel_teacher_forcing=true")
            if self.state_film_dropout_probability > 0.0:
                raise ValueError(
                    "Action XM requires state_film_dropout_probability=0 so "
                    "candidate branches differ only in Action noise"
                )
        self.action_soar_mode = str(self.action_soar_mode).lower()
        if self.action_soar_mode not in {"off", "light", "full"}:
            raise ValueError(
                "action_soar_mode must be one of off/light/full, got "
                f"{self.action_soar_mode!r}"
            )
        if int(self.action_soar_every_n_steps) <= 0:
            raise ValueError("action_soar_every_n_steps must be positive")
        if float(self.action_soar_correction_weight) < 0.0:
            raise ValueError("action_soar_correction_weight must be non-negative")
        if int(self.action_soar_num_aux_points) != 1:
            raise ValueError("Action SOAR currently requires num_aux_points=1")
        if int(self.action_soar_warmup_steps) < 0:
            raise ValueError("action_soar_warmup_steps must be non-negative")
        if int(self.action_soar_ramp_steps) < 0:
            raise ValueError("action_soar_ramp_steps must be non-negative")
        if not 0.0 < float(self.action_soar_min_aux_sigma) < 1.0:
            raise ValueError("action_soar_min_aux_sigma must be inside (0,1)")
        self.action_soar_apply_stages = tuple(
            str(stage).lower() for stage in self.action_soar_apply_stages
        )
        if not self.action_soar_apply_stages:
            raise ValueError("action_soar_apply_stages cannot be empty")
        if len(set(self.action_soar_apply_stages)) != len(
            self.action_soar_apply_stages
        ) or any(stage not in {"a0", "a1"} for stage in self.action_soar_apply_stages):
            raise ValueError(
                "action_soar_apply_stages must contain unique a0/a1 entries"
            )
        if self.action_soar_mode != "off":
            if self.training_mode != "paired_ivava":
                raise ValueError("Action SOAR is only valid for paired_ivava training")
            if not self.parallel_teacher_forcing:
                raise ValueError("Action SOAR requires parallel_teacher_forcing=true")
            if self.action_soar_correction_weight <= 0.0:
                raise ValueError(
                    "enabled Action SOAR requires positive correction weight"
                )
            if self.action_xm_best_of_k > 1:
                raise ValueError("Action XM and Action SOAR cannot be enabled together")
        if self.ar_loss_scale <= 0:
            raise ValueError("ar_loss_scale must be positive")
        for name, flow in (
            ("video_flow", self.video_flow),
            ("action_flow", self.action_flow),
        ):
            missing = {
                "shift",
                "train_timesteps",
                "inference_steps",
                "sample_mode",
            } - set(flow)
            if missing:
                raise ValueError(
                    f"{name} is missing required fields: {sorted(missing)}"
                )
            if float(flow["shift"]) <= 0:
                raise ValueError(f"{name}.shift must be positive")
            if int(flow["train_timesteps"]) <= 0 or int(flow["inference_steps"]) <= 0:
                raise ValueError(f"{name} timestep counts must be positive")
            sample_mode = str(flow["sample_mode"])
            if sample_mode not in {"uniform", "logit_normal"}:
                raise ValueError(
                    f"{name}.sample_mode must be uniform or logit_normal, "
                    f"got {sample_mode!r}"
                )
            if sample_mode == "logit_normal":
                logit_mean = float(flow.get("logit_mean", 0.0))
                logit_std = float(flow.get("logit_std", 1.0))
                if not math.isfinite(logit_mean):
                    raise ValueError(f"{name}.logit_mean must be finite")
                if not math.isfinite(logit_std) or logit_std <= 0:
                    raise ValueError(f"{name}.logit_std must be finite and positive")


__all__ = ["VAPModelConfig"]
