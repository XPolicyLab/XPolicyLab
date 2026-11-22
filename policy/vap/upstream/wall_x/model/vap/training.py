"""End-to-end VAP training loss from a refactor_2604 VGA batch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from wall_x.model.vga.data_utils import build_prompt_from_data

from .config import VAPModelConfig
from .data import split_vga_batch
from .loss import (
    ARStage,
    build_action_soar_batch,
    build_ar_flow_batch,
    build_parallel_teacher_forcing_flow_batch,
    corrupt_clean_action_prefix,
    flow_mse_per_sample,
    sample_ar_stages,
    sample_shifted_sigma,
)
from .layout import (
    ParallelTeacherForcingLayout,
    VideoOnlyParallelTeacherForcingLayout,
)
from .memory import FirstFrameMemoryTokens
from .model import VAPTokenModel
from .rope import action_position_ids
from .wan_bridge import (
    EncodedFirstFrameMemory,
    EncodedVideoSequence,
    PatchedVideoSequence,
    encode_first_frame_memory,
    encode_video_sequence,
    patchify_video_sequence,
    patchify_video_sequence_with_memory,
    unpatchify_video_sequence,
)


logger = logging.getLogger(__name__)


def _prepend_first_frame_memory_prefix(
    patched: PatchedVideoSequence,
    memory: FirstFrameMemoryTokens,
) -> PatchedVideoSequence:
    """Prepend read-only Memory to the native condition-video token segment."""

    if memory.tokens.shape[0] != patched.tokens.shape[0]:
        raise ValueError("Memory and Video prefix batch sizes must match")
    if memory.tokens.shape[2] != patched.tokens.shape[2]:
        raise ValueError("Memory and Video prefix embedding dimensions must match")
    if memory.position_ids.device != patched.position_ids.device:
        raise ValueError("Memory and Video prefix position IDs must share a device")
    if memory.camera_ids.device != patched.camera_ids.device:
        raise ValueError("Memory and Video prefix camera IDs must share a device")

    memory_token_count = memory.tokens.shape[1]
    memory_condition_mask = torch.ones(
        patched.tokens.shape[0],
        memory_token_count,
        device=patched.condition_token_mask.device,
        dtype=torch.bool,
    )
    return replace(
        patched,
        tokens=torch.cat((memory.tokens, patched.tokens), dim=1),
        position_ids=torch.cat((memory.position_ids, patched.position_ids), dim=1),
        camera_ids=torch.cat((memory.camera_ids, patched.camera_ids), dim=1),
        condition_token_mask=torch.cat(
            (memory_condition_mask, patched.condition_token_mask), dim=1
        ),
        condition_token_count=patched.condition_token_count + memory_token_count,
        memory_valid=memory.valid,
    )


def _sample_flow_sigma(
    batch_size: int,
    flow_config: dict,
    *,
    device: torch.device | str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample sigma according to one modality's explicit training contract."""

    return sample_shifted_sigma(
        batch_size,
        shift=float(flow_config["shift"]),
        sample_mode=str(flow_config.get("sample_mode", "uniform")),
        logit_mean=float(flow_config.get("logit_mean", 0.0)),
        logit_std=float(flow_config.get("logit_std", 1.0)),
        device=device,
        generator=generator,
    )


@dataclass(frozen=True)
class VAPTrainingOutput:
    total_loss: torch.Tensor
    video_loss: torch.Tensor
    action_loss: torch.Tensor
    video_sigma: torch.Tensor
    action_sigma: torch.Tensor
    stage_ids: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        video_active = self.stage_ids.remainder(2) == 0
        action_active = ~video_active
        active_video_sigma = (
            self.video_sigma[video_active].mean()
            if video_active.any()
            else self.video_sigma.sum() * 0
        )
        active_action_sigma = (
            self.action_sigma[action_active].mean()
            if action_active.any()
            else self.action_sigma.sum() * 0
        )
        output = {
            "total_loss": self.total_loss,
            "video_loss": self.video_loss.detach().float(),
            "action_loss": self.action_loss.detach().float(),
            "video_sigma": active_video_sigma.detach().float(),
            "action_sigma": active_action_sigma.detach().float(),
        }
        for stage in ARStage:
            output[f"ar_stage_{stage.name.lower()}_fraction"] = (
                (self.stage_ids == int(stage)).float().mean()
            )
        return output


def _sample_locked_action_prefix_embedding_dropout(
    *,
    batch_size: int,
    embedding_dim: int,
    sample_probability: float,
    dropout_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a per-example channel mask shared across all clean A0 tokens."""

    trigger = torch.rand(batch_size, 1, 1, device=device, generator=generator) < float(
        sample_probability
    )
    keep = torch.rand(
        batch_size, 1, embedding_dim, device=device, generator=generator
    ) >= float(dropout_rate)
    triggered_scale = keep.to(dtype=dtype) / (1.0 - float(dropout_rate))
    scale = torch.where(trigger, triggered_scale, torch.ones_like(triggered_scale))
    dropped = trigger & ~keep
    return (
        scale,
        trigger.float().mean(),
        dropped.float().mean(),
    )


def _scheduled_action_soar_weight(
    config: VAPModelConfig,
    *,
    global_step: int,
    training: bool,
) -> float:
    """Return a rank-synchronous SOAR weight without touching RNG state."""

    mode = str(getattr(config, "action_soar_mode", "off")).lower()
    if not training or mode == "off":
        return 0.0
    warmup = int(getattr(config, "action_soar_warmup_steps", 0))
    if global_step < warmup:
        return 0.0
    if mode == "light" and global_step % int(
        getattr(config, "action_soar_every_n_steps", 4)
    ):
        return 0.0
    target = float(getattr(config, "action_soar_correction_weight", 0.0))
    ramp_steps = int(getattr(config, "action_soar_ramp_steps", 0))
    if ramp_steps <= 0:
        return target
    progress = min(1.0, max(0.0, (global_step - warmup) / ramp_steps))
    return target * progress


def _select_action_xm_winners(
    chunk_losses: list[torch.Tensor],
    *,
    batch_size: int,
    best_of_k: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Select one coherent Action-noise candidate for all Action stages.

    Each input is a flattened ``[B*K]`` per-sample loss for one Action stage.
    The winner is chosen from the mean A0/A1 loss, rather than independently
    per stage (which would silently turn K candidates into a combinatorial
    hybrid trajectory).  ``gather`` leaves autograd connected only to the
    winning candidate for each example.
    """

    if not chunk_losses:
        raise ValueError("Action XM requires at least one Action-stage loss")
    expected = batch_size * best_of_k
    if best_of_k <= 0 or any(loss.numel() != expected for loss in chunk_losses):
        raise ValueError("Action XM chunk losses must each contain B*K values")
    candidates = torch.stack(
        [loss.reshape(batch_size, best_of_k) for loss in chunk_losses], dim=-1
    )
    joint_candidates = candidates.mean(dim=-1)
    winner = joint_candidates.detach().argmin(dim=1)
    gather_index = winner[:, None, None].expand(-1, 1, candidates.shape[-1])
    selected = candidates.gather(1, gather_index).squeeze(1)
    return list(selected.unbind(dim=-1)), winner, joint_candidates


class VAPTrainingModel(nn.Module):
    """Wan VAE + native VAP core, without a VLM planner.

    ``text_context`` is optional cached/raw T5 output. Camera ``view_ids`` are
    explicit because silently assigning the wrong physical view identity would
    corrupt the learned camera identity while leaving MRoPE window-local.
    """

    def __init__(
        self,
        *,
        vae: nn.Module,
        core: VAPTokenModel,
        config: VAPModelConfig,
        prompter=None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.core = core
        self.config = config
        if getattr(config, "text_encoder_backend", "t5") in {"qwen35_cache", "qwen35_online"}:
            from .text_conditioning import TextInputProjector, validate_text_config

            validate_text_config(config)
            reference = next(core.parameters())
            core.text_input_projector = TextInputProjector(
                config.text_feature_hidden_dim,
                config.text_projector_output_dim,
                config.text_feature_seq_len,
            ).to(device=reference.device, dtype=reference.dtype)
        action_loss_dimension_weights = getattr(
            config, "action_loss_dimension_weights", ()
        )
        self.register_buffer(
            "_action_loss_dimension_weights",
            (
                torch.tensor(
                    action_loss_dimension_weights, dtype=torch.float32
                ).reshape(1, 1, -1)
                if action_loss_dimension_weights
                else None
            ),
            persistent=False,
        )
        self._action_loss_dimension_weight_normalization = getattr(
            config,
            "action_loss_dimension_weight_normalization",
            "weighted_mean",
        )
        # Keep the live-forward dtype independent from DMuon's fp32 master
        # parameter storage. This mirrors the Wan baseline's pipe.torch_dtype:
        # frozen VAE/T5 inputs and trainable model activations stay in bf16.
        self.torch_dtype = torch_dtype
        # WanPrompter is intentionally not an nn.Module. It owns either a frozen
        # T5 reference or a CPU text-feature cache, neither of which belongs in
        # VAP optimizer/checkpoint state.
        self.prompter = prompter
        self.vae.requires_grad_(False)
        self.vae.eval()
        text_encoder = self._text_encoder()
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

    @classmethod
    def from_wan_pipeline(
        cls,
        pipeline,
        config: VAPModelConfig,
        *,
        action_dim: int,
        state_dim: int,
    ) -> "VAPTrainingModel":
        if pipeline.dit is None or pipeline.vae is None:
            raise ValueError("Wan pipeline must contain both DiT and VAE")
        core = VAPTokenModel.from_wan_dit(
            pipeline.dit,
            action_input_dim=action_dim,
            action_dim=config.action_expert_dim,
            num_chunks=config.num_interleave_chunks,
            rope_axes_dim=config.rope_axes_dim,
            max_views=config.max_views,
            expected_num_heads=config.joint_num_heads,
            expected_head_dim=config.joint_head_dim,
            attention_backend=config.attention_backend,
            state_input_dim=(
                state_dim if getattr(config, "state_film_enabled", False) else None
            ),
            state_film_hidden_dim=int(getattr(config, "state_film_hidden_dim", 512)),
            state_film_dropout_probability=float(
                getattr(config, "state_film_dropout_probability", 0.0)
            ),
            first_frame_memory_enabled=(
                bool(getattr(config, "first_frame_memory_enabled", False))
                and getattr(
                    config,
                    "first_frame_memory_injection_mode",
                    "gated_cross_attention",
                )
                == "gated_cross_attention"
            ),
            first_frame_memory_reader_layers=tuple(
                getattr(
                    config,
                    "first_frame_memory_reader_layers",
                    (3, 9, 15, 21, 27),
                )
            ),
        )
        return cls(
            vae=pipeline.vae,
            core=core,
            config=config,
            prompter=pipeline.prompter,
            torch_dtype=getattr(pipeline, "torch_dtype", torch.bfloat16),
        )

    def _text_encoder(self) -> nn.Module | None:
        if self.prompter is None:
            return None
        encoder = getattr(self.prompter, "text_encoder", None)
        return encoder if isinstance(encoder, nn.Module) else None

    def _move_text_encoder(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Online Qwen owns rank-local placement and preserves FP32 recurrent
        # buffers; generic .to(bf16) must not cast those or offload it per step.
        if getattr(self.prompter, "manages_placement", False):
            return
        encoder = self._text_encoder()
        if encoder is None:
            return
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        encoder.to(**kwargs)
        encoder.eval()

    @torch.no_grad()
    def _encode_text_context(
        self,
        batch,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.prompter is None:
            raise RuntimeError(
                "VAP text conditioning is required but no WanPrompter exists"
            )
        prompt = build_prompt_from_data(
            batch,
            use_qwen25=bool(getattr(self.config, "use_qwen25", False)),
            prompt_template=getattr(self.config, "prompt_template", None),
            qwen25_prompt_template=getattr(self.config, "qwen25_prompt_template", None),
            prompt_priority_order=getattr(self.config, "prompt_priority_order", None),
        )
        # A hybrid serving process keeps T5 on CPU for cache hits. An actual
        # miss moves the frozen encoder to the inference device, memoizes the
        # result, and then offloads T5 again so it cannot permanently consume
        # the 5B policy's inference memory budget.
        cache_probe = getattr(self.prompter, "has_cached_prompt", None)
        # Older WanPrompter-compatible implementations do not expose a cache
        # probe. Treat them as a miss so their frozen encoder is placed on the
        # requested device before encode_prompt() instead of failing here.
        cache_miss = (
            not bool(cache_probe(prompt, positive=True))
            if callable(cache_probe)
            else True
        )
        online_fallback = bool(
            getattr(self.config, "text_feature_cache_online_fallback", False)
        )
        if cache_miss:
            self._move_text_encoder(device=device, dtype=dtype)
        try:
            context = self.prompter.encode_prompt(
                prompt,
                positive=True,
                device=device,
            )
        finally:
            if cache_miss and online_fallback:
                self._move_text_encoder(device=torch.device("cpu"))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise RuntimeError(
                "WanPrompter must return [B,L,C] text context; "
                f"expected B={batch_size}, got {tuple(context.shape)}"
            )
        return context.to(device=device, dtype=dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        text_encoder = self._text_encoder()
        if text_encoder is not None:
            text_encoder.eval()
        return self

    def _forward_parallel_teacher_forcing(
        self,
        *,
        encoded: EncodedVideoSequence,
        first_frame_memory: EncodedFirstFrameMemory | None = None,
        clean_actions: torch.Tensor,
        state_condition: torch.Tensor | None = None,
        dof_mask: torch.Tensor,
        view_ids: torch.Tensor,
        text_context: torch.Tensor | None,
        generator: torch.Generator | None,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Train all ``A0,V0,A1,V1`` stages in one masked trajectory pass."""

        config = self.config
        memory_enabled = bool(getattr(config, "first_frame_memory_enabled", False))
        if memory_enabled != (first_frame_memory is not None):
            raise ValueError(
                "first-frame memory config and encoded Anchor Bank must agree"
            )
        source_batch_size = clean_actions.shape[0]
        device, dtype = clean_actions.device, clean_actions.dtype
        num_chunks = config.num_interleave_chunks
        action_xm_best_of_k = int(getattr(config, "action_xm_best_of_k", 1))
        if not self.training:
            # Validation remains one deterministic-cost estimate of the normal
            # diffusion objective; exploration is a training-only optimizer.
            action_xm_best_of_k = 1
        video_chunk_sigma = _sample_flow_sigma(
            source_batch_size * num_chunks,
            config.video_flow,
            device=device,
            generator=generator,
        ).unflatten(0, (source_batch_size, num_chunks))
        action_sigma = _sample_flow_sigma(
            source_batch_size * num_chunks,
            config.action_flow,
            device=device,
            generator=generator,
        ).unflatten(0, (source_batch_size, num_chunks))

        # Preserve the exact historical RNG order for K=1: timesteps, then one
        # Video noise tensor, then Action noise.  For K>1 only Action noise gains
        # a candidate axis.  Repeating the Video noise makes this an Action-only
        # exploration objective rather than a best-of-K joint Video objective.
        video_noise = torch.randn(
            encoded.latents.shape,
            device=encoded.latents.device,
            dtype=encoded.latents.dtype,
            generator=generator,
        )
        action_noise = torch.randn(
            source_batch_size,
            action_xm_best_of_k,
            *clean_actions.shape[1:],
            device=clean_actions.device,
            dtype=clean_actions.dtype,
            generator=generator,
        ).flatten(0, 1)

        def expand_candidates(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or action_xm_best_of_k == 1:
                return tensor
            return tensor.repeat_interleave(action_xm_best_of_k, dim=0)

        encoded = replace(encoded, latents=expand_candidates(encoded.latents))
        if first_frame_memory is not None:
            first_frame_memory = replace(
                first_frame_memory,
                latents=expand_candidates(first_frame_memory.latents),
                age_steps=expand_candidates(first_frame_memory.age_steps),
                valid=expand_candidates(first_frame_memory.valid),
            )
        clean_actions_source = clean_actions
        dof_mask_source = dof_mask
        clean_actions = expand_candidates(clean_actions)
        dof_mask = expand_candidates(dof_mask)
        view_ids = expand_candidates(view_ids)
        text_context = expand_candidates(text_context)
        state_condition = expand_candidates(state_condition)
        video_chunk_sigma = expand_candidates(video_chunk_sigma)
        action_sigma = expand_candidates(action_sigma)
        video_noise = expand_candidates(video_noise)
        batch_size = source_batch_size * action_xm_best_of_k
        flow = build_parallel_teacher_forcing_flow_batch(
            encoded.latents,
            clean_actions,
            num_chunks=num_chunks,
            future_video_frames_per_chunk=encoded.future_latent_frames_per_chunk,
            action_steps_per_chunk=config.action_steps_per_chunk,
            video_chunk_sigma=video_chunk_sigma,
            action_sigma=action_sigma,
            video_noise=video_noise,
            action_noise=action_noise,
            generator=generator,
        )

        # Patch clean/noisy streams in one module call.  Besides being faster,
        # this keeps the DMuon video_patch_embedding hook boundary single-use
        # within one outer forward.
        combined_encoded = EncodedVideoSequence(
            latents=torch.cat((encoded.latents, flow.noisy_video), dim=0),
            view_keys=encoded.view_keys,
            raw_frames=encoded.raw_frames,
            num_chunks=encoded.num_chunks,
            future_latent_frames_per_chunk=encoded.future_latent_frames_per_chunk,
        )
        combined_view_ids = torch.cat((view_ids, view_ids), dim=0)
        temporal_stride = (
            config.vae_temporal_stride
            * config.action_steps_per_chunk
            // config.video_frames_per_chunk
        )
        first_frame_memory_tokens = None
        memory_prefix_token_count = 0
        if first_frame_memory is None:
            combined_patched = patchify_video_sequence(
                self.core,
                combined_encoded,
                view_ids=combined_view_ids,
                chunk_stride=config.action_steps_per_chunk,
                temporal_stride=temporal_stride,
            )
        else:
            combined_memory = replace(
                first_frame_memory,
                valid=(torch.cat((first_frame_memory.valid, first_frame_memory.valid), dim=0)
                       if first_frame_memory.valid is not None else None),
                latents=torch.cat(
                    (first_frame_memory.latents, first_frame_memory.latents), dim=0
                ),
                age_steps=(
                    torch.cat(
                        (first_frame_memory.age_steps, first_frame_memory.age_steps),
                        dim=0,
                    )
                    if first_frame_memory.age_steps is not None
                    else None
                ),
            )
            combined_patched, combined_memory_tokens = (
                patchify_video_sequence_with_memory(
                    self.core,
                    combined_encoded,
                    combined_memory,
                    view_ids=combined_view_ids,
                    memory_grid=tuple(config.first_frame_memory_grid),
                    chunk_stride=config.action_steps_per_chunk,
                    temporal_stride=temporal_stride,
                    position_mode=getattr(
                        config,
                        "first_frame_memory_position_mode",
                        "window_origin",
                    ),
                )
            )
            memory_injection_mode = getattr(
                config,
                "first_frame_memory_injection_mode",
                "gated_cross_attention",
            )
            if memory_injection_mode == "prefix_self_attention":
                memory_prefix_token_count = combined_memory_tokens.tokens.shape[1]
                combined_main_patched = combined_patched
                combined_patched = _prepend_first_frame_memory_prefix(
                    combined_patched,
                    combined_memory_tokens,
                )
            else:
                first_frame_memory_tokens = replace(
                    combined_memory_tokens,
                    tokens=combined_memory_tokens.tokens[:batch_size],
                    position_ids=combined_memory_tokens.position_ids[:batch_size],
                    camera_ids=combined_memory_tokens.camera_ids[:batch_size],
                )
        if memory_prefix_token_count == 0:
            combined_main_patched = combined_patched
        clean_main_patched = replace(
            combined_main_patched,
            tokens=combined_main_patched.tokens[:batch_size],
            position_ids=combined_main_patched.position_ids[:batch_size],
            camera_ids=combined_main_patched.camera_ids[:batch_size],
            condition_token_mask=(
                combined_main_patched.condition_token_mask[:batch_size]
            ),
        )
        clean_patched = replace(
            combined_patched,
            tokens=combined_patched.tokens[:batch_size],
            position_ids=combined_patched.position_ids[:batch_size],
            camera_ids=combined_patched.camera_ids[:batch_size],
            condition_token_mask=combined_patched.condition_token_mask[:batch_size],
            memory_valid=(combined_patched.memory_valid[:batch_size] if combined_patched.memory_valid is not None else None),
        )
        noisy_patched = replace(
            combined_patched,
            tokens=combined_patched.tokens[batch_size:],
            position_ids=combined_patched.position_ids[batch_size:],
            camera_ids=combined_patched.camera_ids[batch_size:],
            condition_token_mask=combined_patched.condition_token_mask[batch_size:],
            memory_valid=(combined_patched.memory_valid[batch_size:] if combined_patched.memory_valid is not None else None),
        )
        video_only = getattr(config, "training_mode", "paired_ivava") == "video_only"
        layout_cls = (
            VideoOnlyParallelTeacherForcingLayout
            if video_only
            else ParallelTeacherForcingLayout
        )
        layout = layout_cls(
            num_chunks=num_chunks,
            condition_video_tokens=clean_patched.condition_token_count,
            video_tokens_per_chunk=clean_patched.video_tokens_per_chunk,
            action_tokens_per_chunk=config.action_steps_per_chunk,
            robust_a0=(
                False if video_only else bool(getattr(config, "robust_a0", False))
            ),
        )
        packed_video = layout.expand_video(clean_patched.tokens, noisy_patched.tokens)
        packed_video_positions = layout.expand_video(
            clean_patched.position_ids, noisy_patched.position_ids
        )
        packed_camera_ids = layout.expand_video(
            clean_patched.camera_ids[..., None], noisy_patched.camera_ids[..., None]
        ).squeeze(-1)
        clean_video_timestep = torch.zeros(
            batch_size,
            clean_patched.tokens.shape[1],
            device=device,
            dtype=video_chunk_sigma.dtype,
        )
        noisy_video_timestep = noisy_patched.per_token_timestep(
            video_chunk_sigma * float(config.video_flow["train_timesteps"])
        )
        packed_video_timestep = layout.expand_video(
            clean_video_timestep[..., None], noisy_video_timestep[..., None]
        ).squeeze(-1)

        robust_a0_enabled = bool(getattr(config, "robust_a0", False))
        prefix_actions_source = clean_actions_source
        robust_a0_source = (
            clean_actions_source[:, : config.action_steps_per_chunk]
            if robust_a0_enabled
            else None
        )
        prefix_noise_fraction = clean_actions.new_zeros((), dtype=torch.float32)
        prefix_noise_rms = clean_actions.new_zeros((), dtype=torch.float32)
        if (
            self.training
            and bool(getattr(config, "action_prefix_noise_enabled", False))
            and global_step
            >= int(getattr(config, "action_prefix_noise_warmup_steps", 0))
        ):
            corruption = corrupt_clean_action_prefix(
                clean_actions_source,
                dof_mask_source,
                action_steps_per_chunk=config.action_steps_per_chunk,
                probability=float(config.action_prefix_noise_probability),
                std=float(config.action_prefix_noise_std),
                exclude_dims=tuple(config.action_prefix_noise_exclude_dims),
                generator=generator,
            )
            if robust_a0_enabled:
                # Keep the physical Action context clean for V0 and V1. Only
                # noisy A1 receives this separately packed corrupted A0 view.
                robust_a0_source = corruption.actions[
                    :, : config.action_steps_per_chunk
                ]
            else:
                # Strict compatibility: the historical switch corrupts the
                # shared clean A0 prefix consumed by every later stage.
                prefix_actions_source = corruption.actions
            prefix_noise_fraction = corruption.sample_fraction
            prefix_noise_rms = corruption.rms
        prefix_actions = expand_candidates(prefix_actions_source)
        robust_a0_actions = (
            None if robust_a0_source is None else expand_candidates(robust_a0_source)
        )
        if robust_a0_enabled:
            packed_action = layout.expand_action(
                prefix_actions,
                flow.noisy_action,
                robust_a0=robust_a0_actions,
            )
        else:
            # Preserve the historical call and physical storage exactly.
            packed_action = layout.expand_action(prefix_actions, flow.noisy_action)
        base_action_positions = action_position_ids(
            batch_size,
            num_chunks,
            config.action_steps_per_chunk,
            device=device,
            chunk_stride=config.action_steps_per_chunk,
        )
        packed_action_positions = layout.expand_action(
            base_action_positions, base_action_positions
        )
        clean_action_timestep = torch.zeros(
            batch_size,
            clean_actions.shape[1],
            device=device,
            dtype=action_sigma.dtype,
        )
        noisy_action_timestep = action_sigma.repeat_interleave(
            config.action_steps_per_chunk, dim=1
        ) * float(config.action_flow["train_timesteps"])
        packed_action_timestep = layout.expand_action(
            clean_action_timestep[..., None], noisy_action_timestep[..., None]
        ).squeeze(-1)

        prefix_embedding_dropout_scale = None
        prefix_embedding_dropout_sample_fraction = clean_actions.new_zeros(
            (), dtype=torch.float32
        )
        prefix_embedding_dropout_channel_fraction = clean_actions.new_zeros(
            (), dtype=torch.float32
        )
        prefix_embedding_dropout_sample_probability = float(
            getattr(
                config,
                "action_prefix_embedding_dropout_sample_probability",
                0.0,
            )
        )
        prefix_embedding_dropout_rate = float(
            getattr(config, "action_prefix_embedding_dropout_rate", 0.0)
        )
        if (
            self.training
            and prefix_embedding_dropout_sample_probability > 0.0
            and prefix_embedding_dropout_rate > 0.0
            and global_step
            >= int(
                getattr(
                    config,
                    "action_prefix_embedding_dropout_warmup_steps",
                    0,
                )
            )
        ):
            configured_embedding_dim = getattr(config, "action_expert_dim", None)
            action_embedding_dim = int(
                self.core.action_dim
                if configured_embedding_dim is None
                else configured_embedding_dim
            )
            (
                prefix_embedding_dropout_scale,
                prefix_embedding_dropout_sample_fraction,
                prefix_embedding_dropout_channel_fraction,
            ) = _sample_locked_action_prefix_embedding_dropout(
                batch_size=source_batch_size,
                embedding_dim=action_embedding_dim,
                sample_probability=prefix_embedding_dropout_sample_probability,
                dropout_rate=prefix_embedding_dropout_rate,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            prefix_embedding_dropout_scale = expand_candidates(
                prefix_embedding_dropout_scale
            )

        output = self.core(
            packed_video,
            packed_action,
            video_timestep=packed_video_timestep,
            action_timestep=packed_action_timestep,
            video_position_ids=packed_video_positions,
            condition_video_tokens=clean_patched.condition_token_count,
            action_position_ids=packed_action_positions,
            video_camera_ids=packed_camera_ids,
            text_context=(
                None
                if text_context is None
                else text_context.to(device=device, dtype=dtype)
            ),
            layout=layout,
            action_prefix_embedding_dropout_scale=(prefix_embedding_dropout_scale),
            state_condition=state_condition,
            first_frame_memory=first_frame_memory_tokens,
            memory_prefix_valid=clean_patched.memory_valid,
        )

        noisy_video_velocity = layout.noisy_video(output.video_velocity)
        condition_video_velocity = output.video_velocity[
            :,
            memory_prefix_token_count : clean_patched.condition_token_count,
        ]
        standard_video_velocity = torch.cat(
            (
                condition_video_velocity,
                noisy_video_velocity,
            ),
            dim=1,
        )
        video_velocity = unpatchify_video_sequence(
            self.core, standard_video_velocity, clean_main_patched
        )
        action_velocity = layout.noisy_action(output.action_velocity)

        soar_weight = _scheduled_action_soar_weight(
            config,
            global_step=global_step,
            training=self.training and not video_only,
        )
        soar_batch = None
        soar_action_velocity = None
        if soar_weight > 0.0:
            soar_batch = build_action_soar_batch(
                clean_actions,
                flow.noisy_action,
                flow.action_noise,
                action_velocity,
                action_sigma=action_sigma,
                action_steps_per_chunk=config.action_steps_per_chunk,
                shift=float(config.action_flow["shift"]),
                inference_steps=int(config.action_flow["inference_steps"]),
                min_aux_sigma=float(config.action_soar_min_aux_sigma),
                generator=generator,
            )
            if robust_a0_enabled:
                soar_packed_action = layout.expand_action(
                    prefix_actions,
                    soar_batch.noisy_action,
                    robust_a0=robust_a0_actions,
                )
            else:
                soar_packed_action = layout.expand_action(
                    prefix_actions, soar_batch.noisy_action
                )
            soar_noisy_action_timestep = soar_batch.aux_sigma.repeat_interleave(
                config.action_steps_per_chunk, dim=1
            ) * float(config.action_flow["train_timesteps"])
            soar_packed_action_timestep = layout.expand_action(
                clean_action_timestep[..., None],
                soar_noisy_action_timestep[..., None],
            ).squeeze(-1)
            soar_output = self.core(
                packed_video,
                soar_packed_action,
                video_timestep=packed_video_timestep,
                action_timestep=soar_packed_action_timestep,
                video_position_ids=packed_video_positions,
                condition_video_tokens=clean_patched.condition_token_count,
                action_position_ids=packed_action_positions,
                video_camera_ids=packed_camera_ids,
                text_context=(
                    None
                    if text_context is None
                    else text_context.to(device=device, dtype=dtype)
                ),
                layout=layout,
                action_prefix_embedding_dropout_scale=(prefix_embedding_dropout_scale),
                state_condition=state_condition,
                first_frame_memory=first_frame_memory_tokens,
                memory_prefix_valid=clean_patched.memory_valid,
            )
            soar_action_velocity = layout.noisy_action(soar_output.action_velocity)

        action_losses = []
        action_candidate_losses = []
        optimized_action_losses = []
        soar_action_losses = []
        video_losses = []
        soar_stages = set(
            str(stage).lower()
            for stage in getattr(config, "action_soar_apply_stages", ("a0", "a1"))
        )
        for chunk in range(num_chunks):
            action_slice = slice(
                chunk * config.action_steps_per_chunk,
                (chunk + 1) * config.action_steps_per_chunk,
            )
            action_mask = flow.action_loss_mask[:, action_slice] & dof_mask[
                :, action_slice
            ].to(dtype=torch.bool)
            base_action_loss_per_candidate = flow_mse_per_sample(
                action_velocity[:, action_slice],
                flow.action_target[:, action_slice],
                mask=action_mask,
                element_weights=self._action_loss_dimension_weights,
                element_weight_normalization=self._action_loss_dimension_weight_normalization,
            )
            action_candidate_losses.append(base_action_loss_per_candidate)
            base_action_loss = base_action_loss_per_candidate.mean()
            action_losses.append(base_action_loss)
            stage_name = f"a{chunk}"
            if (
                soar_batch is not None
                and soar_action_velocity is not None
                and stage_name in soar_stages
            ):
                correction_loss = flow_mse_per_sample(
                    soar_action_velocity[:, action_slice],
                    soar_batch.correction_target[:, action_slice],
                    mask=action_mask,
                    element_weights=self._action_loss_dimension_weights,
                    element_weight_normalization=self._action_loss_dimension_weight_normalization,
                ).mean()
                soar_action_losses.append(correction_loss)
                optimized_action_losses.append(
                    (base_action_loss + soar_weight * correction_loss)
                    / (1.0 + soar_weight)
                )
            else:
                soar_action_losses.append(base_action_loss.detach() * 0.0)
                optimized_action_losses.append(base_action_loss)
            frame_start = 1 + chunk * encoded.future_latent_frames_per_chunk
            frame_slice = slice(
                frame_start,
                frame_start + encoded.future_latent_frames_per_chunk,
            )
            video_loss_per_candidate = flow_mse_per_sample(
                video_velocity[:, :, :, frame_slice],
                flow.video_target[:, :, :, frame_slice],
                mask=flow.video_loss_mask[:, :, :, frame_slice],
            )
            # All K branches share one Video corruption.  Candidate zero owns
            # Video supervision so Action exploration cannot cherry-pick or
            # multiply the Video objective.
            video_losses.append(
                video_loss_per_candidate.reshape(
                    source_batch_size, action_xm_best_of_k
                )[:, 0].mean()
            )

        (
            selected_action_losses_per_sample,
            action_xm_winner,
            action_xm_joint_candidates,
        ) = _select_action_xm_winners(
            action_candidate_losses,
            batch_size=source_batch_size,
            best_of_k=action_xm_best_of_k,
        )
        action_losses = [loss.mean() for loss in selected_action_losses_per_sample]
        if action_xm_best_of_k > 1:
            # Config validation makes SOAR mutually exclusive with Action XM.
            optimized_action_losses = list(action_losses)
            soar_action_losses = [loss.detach() * 0.0 for loss in action_losses]
        action_xm_selected_joint = action_xm_joint_candidates.gather(
            1, action_xm_winner[:, None]
        ).squeeze(1)
        action_xm_candidate0_joint = action_xm_joint_candidates[:, 0]

        if video_only:
            # Optimize the true video marginal I->V0->V1. Do not dilute its
            # scale by averaging two zero Action losses into four stages.
            # The zero-valued anchor keeps every Action-expert autograd path
            # present for deterministic DMuon/FSDP hooks without updating it.
            action_zero_anchor = torch.stack(action_losses).sum() * 0.0
            total_loss = (
                float(config.video_only_loss_scale)
                * float(config.video_loss_weight)
                * torch.stack(video_losses).mean()
                + action_zero_anchor
            )
        else:
            weighted_stage_losses = []
            for action_loss, video_loss in zip(optimized_action_losses, video_losses):
                weighted_stage_losses.extend(
                    (
                        float(config.action_loss_weight) * action_loss,
                        float(config.video_loss_weight) * video_loss,
                    )
                )
            total_loss = (
                float(config.ar_loss_scale) * torch.stack(weighted_stage_losses).mean()
            )
        video_loss = torch.stack(video_losses).mean()
        action_loss = torch.stack(action_losses).mean()
        optimized_action_loss = torch.stack(optimized_action_losses).mean()
        soar_correction_loss = torch.stack(soar_action_losses).mean()
        one = total_loss.detach().new_ones(())
        zero = total_loss.detach().new_zeros((), dtype=torch.float32)
        metrics = {
            "total_loss": total_loss,
            "video_loss": video_loss.detach().float(),
            "action_loss": action_loss.detach().float(),
            "action_optimized_loss": optimized_action_loss.detach().float(),
            "action_xm_best_of_k": one * float(action_xm_best_of_k),
            "action_xm_candidate0_loss": (
                action_xm_candidate0_joint.mean().detach().float()
            ),
            "action_xm_selected_loss": (
                action_xm_selected_joint.mean().detach().float()
            ),
            "action_xm_improvement": (
                (action_xm_candidate0_joint - action_xm_selected_joint)
                .mean()
                .detach()
                .float()
            ),
            "action_xm_nonzero_winner_fraction": (
                (action_xm_winner != 0).float().mean().detach()
            ),
            "video_sigma": video_chunk_sigma.mean().detach().float(),
            "action_sigma": action_sigma.mean().detach().float(),
            "action_prefix_noise_fraction": prefix_noise_fraction.detach().float(),
            "action_prefix_noise_rms": prefix_noise_rms.detach().float(),
            "action_prefix_embedding_dropout_sample_fraction": (
                prefix_embedding_dropout_sample_fraction.detach().float()
            ),
            "action_prefix_embedding_dropout_channel_fraction": (
                prefix_embedding_dropout_channel_fraction.detach().float()
            ),
            "action_soar_active": one * float(soar_batch is not None),
            "action_soar_weight": one * float(soar_weight),
            "action_soar_correction_loss": (
                soar_correction_loss.detach().float()
                if soar_batch is not None
                else zero
            ),
            "action_soar_a0_loss": (
                soar_action_losses[0].detach().float()
                if soar_batch is not None and "a0" in soar_stages
                else zero
            ),
            "action_soar_a1_loss": (
                soar_action_losses[1].detach().float()
                if soar_batch is not None and "a1" in soar_stages
                else zero
            ),
            "action_soar_aux_sigma": (
                soar_batch.aux_sigma.mean().detach().float()
                if soar_batch is not None
                else zero
            ),
            "action_soar_rollout_rms": (
                soar_batch.rollout_rms.detach().float()
                if soar_batch is not None
                else zero
            ),
        }
        for chunk, (chunk_action_loss, chunk_video_loss) in enumerate(
            zip(action_losses, video_losses)
        ):
            metrics[f"video_sigma_v{chunk}"] = (
                video_chunk_sigma[:, chunk].mean().detach().float()
            )
            metrics[f"ar_stage_a{chunk}_loss"] = chunk_action_loss.detach().float()
            metrics[f"ar_stage_v{chunk}_loss"] = chunk_video_loss.detach().float()
            metrics[f"ar_stage_a{chunk}_fraction"] = one * (0.0 if video_only else 1.0)
            metrics[f"ar_stage_v{chunk}_fraction"] = one
        return metrics

    def convert_to_mix_precision(self):
        self.core.to(dtype=self.torch_dtype)
        self.vae.to(dtype=self.torch_dtype)
        self._move_text_encoder(dtype=self.torch_dtype)
        return self

    def convert_to_pure_dmuon_mix_precision(self):
        """Set Wan live-forward dtypes before DMuon creates fp32 masters.

        This matches the Wan baseline policy: frozen modules and forward
        activations use bf16, while ``convert_to_dmuon`` subsequently promotes
        trainable parameters to fp32 master storage.
        """

        self.core.to(dtype=self.torch_dtype)
        self.vae.to(dtype=self.torch_dtype)
        self._move_text_encoder(dtype=self.torch_dtype)
        return self

    def gradient_checkpointing_enable(self, num_layers: int | None = None) -> None:
        self.core.gradient_checkpointing_enable(num_layers=num_layers)

    def gradient_checkpointing_disable(self) -> None:
        self.core.gradient_checkpointing_disable()

    @staticmethod
    def _is_dmuon_target_param(name: str, parameter: torch.Tensor) -> bool:
        return bool(
            parameter.requires_grad
            and parameter.ndim == 2
            and name.endswith(".weight")
            and "video_patch_embedding" not in name
            and "video_identity.camera" not in name
        )

    @staticmethod
    def _is_dmuon_trainable_param(name: str, parameter: torch.Tensor) -> bool:
        """Make pure DMuon responsible for every trainable VAP parameter."""

        del name
        return bool(parameter.requires_grad)

    @classmethod
    def _pure_dmuon_route_hint(cls, name: str, parameter: torch.Tensor) -> str | None:
        """Route transformer matrices to Muon and everything else to AdamW.

        Wan's Conv3d patch embedding, zero-init categorical camera/modality
        embeddings, window-origin anchor, expert modulation tensors, norm weights
        and biases keep AdamW semantics instead of being flattened into
        artificial Muon matrices.
        """

        if not parameter.requires_grad:
            return None
        if cls._is_dmuon_target_param(name, parameter):
            return "muon"
        return "adamw"

    @classmethod
    def _pure_dmuon_param_policy(cls, name: str, parameter: torch.Tensor):
        """Define DMuon storage/compute policy at each VAP hook boundary.

        ``video_patch_embedding`` is called after ``patchify`` observes the
        fp32 master placeholder. Its boundary must therefore recast the already
        prepared activation after DMuon materializes bf16 live parameters.
        Other boundaries keep their existing input dtypes, notably fp32 RoPE.
        """

        route = cls._pure_dmuon_route_hint(name, parameter)
        if route is None:
            return None
        return {
            "route": route,
            "param_dtype": torch.bfloat16,
            "cast_forward_inputs": "video_patch_embedding" in name,
        }

    def _pure_dmuon_hook_boundary(
        self, param_fqn: str, parameter: nn.Parameter
    ) -> nn.Module | None:
        """Resolve communication boundaries that match VAP's real forward.

        A whole VAP block is one boundary so all Video/Action projections for
        that layer can be prepared together. The modules outside the block
        stack are kept at the coarsest boundary whose ``forward`` is actually
        called; ``ExpertTimeConditioner`` itself is deliberately not used
        because the model calls its ``embed`` and ``project`` methods directly.
        """

        del parameter
        if not param_fqn.startswith("core."):
            return None

        parts = param_fqn.split(".")
        if len(parts) >= 4 and parts[1] == "blocks" and parts[2].isdigit():
            block_index = int(parts[2])
            if block_index < len(self.core.blocks):
                return self.core.blocks[block_index]

        boundary_paths = (
            "core.action_encoder",
            "core.state_film",
            "core.video_identity",
            "core.action_identity",
            "core.global_time_anchor",
            "core.first_frame_memory",
            "core.video_time.time_embedding",
            "core.video_time.time_projection",
            "core.action_time.time_embedding",
            "core.action_time.time_projection",
            "core.video_head",
            "core.action_head",
            "core.text_embedding",
            "core.text_input_projector",
            "core.video_patch_embedding",
        )
        for module_path in boundary_paths:
            if param_fqn == module_path or param_fqn.startswith(f"{module_path}."):
                module = self.get_submodule(module_path)
                if module is not None:
                    return module
        return None

    def convert_to_fsdp(
        self,
        *,
        mesh,
        mp_policy,
        offload_policy=None,
        reshard_after_forward: bool = True,
        use_dmuon: bool = False,
        dmuon_owner_kwargs=None,
    ):
        """FSDP2-wrap VAP blocks one at a time to bound peak GPU memory."""

        import torch.distributed as dist
        from torch.distributed.fsdp import fully_shard

        if not torch.cuda.is_available():
            raise RuntimeError("VAP FSDP2 requires CUDA")
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        for name, parameter in list(self.named_parameters()):
            if parameter.ndim == 0:
                parts = name.rsplit(".", 1)
                parent = self.get_submodule(parts[0]) if len(parts) == 2 else self
                setattr(
                    parent,
                    parts[-1],
                    nn.Parameter(
                        parameter.data.unsqueeze(0),
                        requires_grad=parameter.requires_grad,
                    ),
                )
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()

        if use_dmuon:
            import dmuon
            from wall_x.trainer.optimizer.dmuon.utils import (
                dmuon_dedicate_mesh_args,
            )

            dmuon_mesh, dmuon_mesh_kwargs = dmuon_dedicate_mesh_args(mesh)
            dmuon.dedicate_params(
                self,
                dmuon_mesh,
                predicate=self._is_dmuon_target_param,
                compute_dtype=torch.bfloat16,
                reshard_after_forward=reshard_after_forward,
                **dmuon_mesh_kwargs,
                **(dmuon_owner_kwargs or {}),
            )

        ignored_params = {
            parameter for parameter in self.parameters() if not parameter.requires_grad
        }
        for block in self.core.blocks:
            block.to(device)
            fully_shard(
                block,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
                ignored_params=ignored_params,
            )
        self.vae.to(device=device, dtype=self.torch_dtype)
        for parameter in self.parameters():
            if parameter.device.type != "cuda":
                parameter.data = parameter.data.to(device)
        for buffer in self.buffers():
            if buffer.device.type != "cuda":
                buffer.data = buffer.data.to(device)
        self._move_text_encoder(device=device, dtype=self.torch_dtype)
        fully_shard(
            self,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=None,
            offload_policy=offload_policy,
            ignored_params=ignored_params,
        )
        self._wallx_fsdp2_root_wrapped = True
        self._dp_mesh = mesh
        torch.cuda.empty_cache()
        return self

    def convert_to_dmuon(
        self,
        *,
        mesh,
        reshard_after_forward: bool = False,
        dmuon_owner_kwargs=None,
    ):
        """Install the pure DMuon backend without applying FSDP2.

        All trainable VAP parameters are owned by DMuon: two-dimensional
        projection/FFN/head matrices use Muon, while convolution, norm, bias
        and modulation parameters use DMuon's AdamW route. Frozen Wan VAE
        parameters remain replicated in bf16.
        """

        import torch.distributed as dist

        import dmuon
        from wall_x.trainer.optimizer.dmuon.utils import dmuon_dedicate_mesh_args

        if torch.cuda.is_available():
            local_rank = (
                dist.get_rank() % torch.cuda.device_count()
                if dist.is_available() and dist.is_initialized()
                else torch.cuda.current_device()
            )
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        named_trainable = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        unresolved = [
            name
            for name, parameter in named_trainable
            if self._pure_dmuon_hook_boundary(name, parameter) is None
        ]
        if unresolved:
            examples = ", ".join(unresolved[:8])
            raise RuntimeError(
                "VAP pure DMuon has no forward hook boundary for "
                f"{len(unresolved)} trainable parameters; examples: {examples}"
            )

        route_counts = {"muon": 0, "adamw": 0}
        for name, parameter in named_trainable:
            route = self._pure_dmuon_route_hint(name, parameter)
            if route not in route_counts:
                raise RuntimeError(
                    f"unsupported VAP DMuon route {route!r} for parameter {name}"
                )
            route_counts[route] += 1
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()

        dmuon_mesh, dmuon_mesh_kwargs = dmuon_dedicate_mesh_args(mesh)
        assignment = dmuon.dedicate_params(
            self,
            dmuon_mesh,
            predicate=self._is_dmuon_trainable_param,
            param_policy_fn=self._pure_dmuon_param_policy,
            hook_boundary_resolver=self._pure_dmuon_hook_boundary,
            hook_boundary_strict=True,
            reshard_after_forward=reshard_after_forward,
            **dmuon_mesh_kwargs,
            **(dmuon_owner_kwargs or {}),
        )
        if len(assignment) != len(named_trainable):
            raise RuntimeError(
                "VAP pure DMuon did not assign every trainable parameter: "
                f"assigned={len(assignment)}, trainable={len(named_trainable)}"
            )

        self.vae.to(device=device, dtype=self.torch_dtype)
        for parameter in self.parameters():
            if parameter.device != device:
                parameter.data = parameter.data.to(device)
        for buffer in self.buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)
        self._move_text_encoder(device=device, dtype=self.torch_dtype)

        self._dp_mesh = mesh
        logger.info(
            "[VAP Pure DMuon] assigned %d trainable params; routes=%s; "
            "reshard_after_forward=%s",
            len(assignment),
            route_counts,
            reshard_after_forward,
        )
        return self

    def forward(
        self,
        batch,
        *,
        view_ids: torch.Tensor,
        text_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        stage_ids: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        effective_global_step = 0 if global_step is None else int(global_step)
        chunks = split_vga_batch(
            batch,
            num_chunks=config.num_interleave_chunks,
            video_frames_per_chunk=config.video_frames_per_chunk,
            action_steps_per_chunk=config.action_steps_per_chunk,
            require_episode_first_obs=bool(
                getattr(config, "first_frame_memory_enabled", False)
            ),
        )
        view_count = len(chunks.video_chunks)
        batch_size = chunks.actions.shape[0]
        if tuple(view_ids.shape) != (batch_size, view_count):
            raise ValueError(
                f"view_ids must be [B,V]=({batch_size},{view_count}), "
                f"got {tuple(view_ids.shape)}"
            )
        if view_ids.min().item() < 0 or view_ids.max().item() >= config.max_views:
            raise ValueError(
                f"physical view ids must be in [0,{config.max_views}), "
                f"got min={view_ids.min().item()} max={view_ids.max().item()}"
            )

        parameter = next(self.core.parameters())
        device, dtype = parameter.device, self.torch_dtype
        if text_context is None:
            text_context = self._encode_text_context(
                batch,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        with torch.no_grad():
            encoded = encode_video_sequence(
                self.vae,
                chunks,
                device=device,
                dtype=dtype,
                vae_temporal_stride=config.vae_temporal_stride,
            )
            first_frame_memory = (
                encode_first_frame_memory(
                    self.vae,
                    chunks,
                    device=device,
                    dtype=dtype,
                    view_keys=tuple(
                        getattr(config, "first_frame_memory_view_keys", ())
                    ),
                )
                if bool(getattr(config, "first_frame_memory_enabled", False))
                else None
            )
            if bool(getattr(config, "history_offsets", ())) != (first_frame_memory is not None and first_frame_memory.valid is not None):
                raise ValueError("Checkpoint history_offsets and batch Memory metadata must agree")
            if getattr(config, "history_offsets", ()):
                if first_frame_memory is None or first_frame_memory.valid is None or first_frame_memory.valid.shape[1] != 1 + len(config.history_offsets):
                    raise ValueError("history_offsets requires matching per-frame Memory valid/age metadata")

        clean_actions = chunks.actions.flatten(1, 2).to(device=device, dtype=dtype)
        state_condition = (
            chunks.chunk_start_states(num_chunks=config.num_interleave_chunks).to(
                device=device, dtype=dtype
            )
            if getattr(config, "state_film_enabled", False)
            else None
        )
        dof_mask = chunks.dof_mask.flatten(1, 2).to(device=device)
        if bool(getattr(config, "parallel_teacher_forcing", False)):
            if stage_ids is not None:
                raise ValueError(
                    "stage_ids are only valid for the single-stage debug path"
                )
            return self._forward_parallel_teacher_forcing(
                encoded=encoded,
                first_frame_memory=first_frame_memory,
                clean_actions=clean_actions,
                state_condition=state_condition,
                dof_mask=dof_mask,
                view_ids=view_ids.to(device),
                text_context=text_context,
                generator=generator,
                global_step=effective_global_step,
            )
        if bool(getattr(config, "first_frame_memory_enabled", False)):
            raise RuntimeError(
                "first-frame memory is implemented only for the parallel "
                "teacher-forcing training path"
            )
        if stage_ids is None:
            stage_ids = sample_ar_stages(
                batch_size,
                num_chunks=config.num_interleave_chunks,
                device=device,
                generator=generator,
            )
        else:
            stage_ids = stage_ids.to(device=device)
        video_sigma = _sample_flow_sigma(
            batch_size,
            config.video_flow,
            device=device,
            generator=generator,
        )
        action_sigma = _sample_flow_sigma(
            batch_size,
            config.action_flow,
            device=device,
            generator=generator,
        )
        flow = build_ar_flow_batch(
            encoded.latents,
            clean_actions,
            num_chunks=config.num_interleave_chunks,
            future_video_frames_per_chunk=encoded.future_latent_frames_per_chunk,
            action_steps_per_chunk=config.action_steps_per_chunk,
            stage_ids=stage_ids,
            video_sigma=video_sigma,
            action_sigma=action_sigma,
            generator=generator,
        )
        noisy_encoded = EncodedVideoSequence(
            latents=flow.noisy_video,
            view_keys=encoded.view_keys,
            raw_frames=encoded.raw_frames,
            num_chunks=encoded.num_chunks,
            future_latent_frames_per_chunk=encoded.future_latent_frames_per_chunk,
        )
        patched = patchify_video_sequence(
            self.core,
            noisy_encoded,
            view_ids=view_ids.to(device),
            chunk_stride=config.action_steps_per_chunk,
            temporal_stride=(
                config.vae_temporal_stride
                * config.action_steps_per_chunk
                // config.video_frames_per_chunk
            ),
        )
        video_timestep = patched.per_token_timestep(
            flow.video_chunk_sigma * float(config.video_flow["train_timesteps"])
        )
        action_timestep = flow.action_chunk_sigma.repeat_interleave(
            config.action_steps_per_chunk, dim=1
        ) * float(config.action_flow["train_timesteps"])
        action_positions = action_position_ids(
            batch_size,
            config.num_interleave_chunks,
            config.action_steps_per_chunk,
            device=device,
            chunk_stride=config.action_steps_per_chunk,
        )
        output = self.core(
            patched.tokens,
            flow.noisy_action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            video_position_ids=patched.position_ids,
            condition_video_tokens=patched.condition_token_count,
            action_position_ids=action_positions,
            video_camera_ids=patched.camera_ids,
            text_context=(
                None
                if text_context is None
                else text_context.to(device=device, dtype=dtype)
            ),
            state_condition=state_condition,
        )
        video_velocity = unpatchify_video_sequence(
            self.core, output.video_velocity, patched
        )
        video_loss_per_sample = flow_mse_per_sample(
            video_velocity,
            flow.video_target,
            mask=flow.video_loss_mask,
        )
        action_loss_per_sample = flow_mse_per_sample(
            output.action_velocity,
            flow.action_target,
            mask=flow.action_loss_mask & dof_mask.to(dtype=torch.bool),
            element_weights=self._action_loss_dimension_weights,
            element_weight_normalization=self._action_loss_dimension_weight_normalization,
        )
        video_active = stage_ids.remainder(2) == 0
        action_active = ~video_active
        video_loss = (
            video_loss_per_sample[video_active].mean()
            if video_active.any()
            else video_loss_per_sample.sum() * 0
        )
        action_loss = (
            action_loss_per_sample[action_active].mean()
            if action_active.any()
            else action_loss_per_sample.sum() * 0
        )
        weighted_per_sample = (
            float(config.video_loss_weight) * video_loss_per_sample
            + float(config.action_loss_weight) * action_loss_per_sample
        )
        total_loss = float(config.ar_loss_scale) * weighted_per_sample.mean()
        return VAPTrainingOutput(
            total_loss=total_loss,
            video_loss=video_loss,
            action_loss=action_loss,
            video_sigma=video_sigma,
            action_sigma=action_sigma,
            stage_ids=stage_ids,
        ).as_dict()

    @torch.no_grad()
    def sample_latents_sequential(self, **kwargs):
        """Correctness-first ``A0->V0->A1->V1`` sampler without KV cache."""

        from .sampling import sample_latents_sequential

        return sample_latents_sequential(self, **kwargs)

    @torch.no_grad()
    def sample_video_given_actions(self, **kwargs):
        """Generate V0/V1 while revealing GT A0/A1 at their causal stages."""

        from .sampling import sample_video_given_actions

        return sample_video_given_actions(self, **kwargs)

    @torch.no_grad()
    def sample_first_video_chunk_given_action(self, **kwargs):
        """Generate only local V0 for the first rollout chunk."""

        from .sampling import sample_first_video_chunk_given_action

        return sample_first_video_chunk_given_action(self, **kwargs)

    @torch.no_grad()
    def sample_next_video_chunk_given_prefix(self, **kwargs):
        """Generate local V1 from a clean overlapping V0 rollout prefix."""

        from .sampling import sample_next_video_chunk_given_prefix

        return sample_next_video_chunk_given_prefix(self, **kwargs)

    @torch.no_grad()
    def sample_video_only_window(self, **kwargs):
        """Generate the native Stage-1 ``I -> V0 -> V1`` TI2V window."""

        from .sampling import sample_video_only_window

        return sample_video_only_window(self, **kwargs)

    @torch.no_grad()
    def sample_first_video_only_chunk(self, **kwargs):
        """Generate only local V0 for Stage-1 video-only rollout."""

        from .sampling import sample_first_video_only_chunk

        return sample_first_video_only_chunk(self, **kwargs)

    @torch.no_grad()
    def sample_next_video_only_chunk_given_prefix(self, **kwargs):
        """Generate local V1 from a clean V0 video-only prefix."""

        from .sampling import sample_next_video_only_chunk_given_prefix

        return sample_next_video_only_chunk_given_prefix(self, **kwargs)

    @torch.no_grad()
    def sample_actions_given_video(self, **kwargs):
        """Predict A0/A1 while revealing only I and then GT V0."""

        from .sampling import sample_actions_given_video

        return sample_actions_given_video(self, **kwargs)

    @torch.no_grad()
    def sample_actions_from_batch(
        self,
        batch,
        *,
        view_ids: torch.Tensor,
        generator: torch.Generator | None = None,
        use_kv_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the causal two-chunk action open-loop contract on a VGA batch.

        The returned actions are in the exact normalized, chunk-local space used
        by training.  A0 observes only I; A1 observes I, predicted A0 and clean
        GT V0.  GT V1 is encoded for shape consistency but is never visible to
        either generated action chunk.
        """

        config = self.config
        if bool(getattr(config, "first_frame_memory_enabled", False)):
            raise NotImplementedError(
                "first-frame memory open-loop sampling is not enabled yet; "
                "training remains available and Memory-off sampling is unchanged"
            )
        chunks = split_vga_batch(
            batch,
            num_chunks=config.num_interleave_chunks,
            video_frames_per_chunk=config.video_frames_per_chunk,
            action_steps_per_chunk=config.action_steps_per_chunk,
        )
        batch_size = chunks.actions.shape[0]
        parameter = next(self.core.parameters())
        device = parameter.device
        text_context = self._encode_text_context(
            batch,
            batch_size=batch_size,
            device=device,
            dtype=self.torch_dtype,
        )
        encoded = encode_video_sequence(
            self.vae,
            chunks,
            device=device,
            dtype=self.torch_dtype,
            vae_temporal_stride=config.vae_temporal_stride,
        )
        sample = self.sample_actions_given_video(
            clean_video_latents=encoded.latents,
            view_ids=view_ids,
            text_context=text_context,
            generator=generator,
            use_kv_cache=use_kv_cache,
            state_condition=(
                chunks.chunk_start_states(num_chunks=config.num_interleave_chunks).to(
                    device=device, dtype=self.torch_dtype
                )
                if getattr(config, "state_film_enabled", False)
                else None
            ),
        )
        return {
            "predicted_actions": sample.actions,
            "target_actions": chunks.actions.flatten(1, 2).to(
                device=device, dtype=sample.actions.dtype
            ),
            "dof_mask": chunks.dof_mask.flatten(1, 2).to(device=device),
        }

    @torch.no_grad()
    def sample_first_action_chunk_given_video(self, **kwargs):
        """Generate A0 from only the current observation I."""

        from .sampling import sample_first_action_chunk_given_video

        return sample_first_action_chunk_given_video(self, **kwargs)

    @torch.no_grad()
    def sample_next_action_chunk_given_prefix(self, **kwargs):
        """Generate local A1 after pre-filling clean local A0/V0."""

        from .sampling import sample_next_action_chunk_given_prefix

        return sample_next_action_chunk_given_prefix(self, **kwargs)


__all__ = ["VAPTrainingModel", "VAPTrainingOutput"]
