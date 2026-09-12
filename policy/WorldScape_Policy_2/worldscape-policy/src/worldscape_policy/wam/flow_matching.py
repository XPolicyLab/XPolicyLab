from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn

from worldscape_policy.types import (
    WAMInferenceState,
    WanI2VCondition,
    WorldActionOutput,
)
from worldscape_policy.wam.wan22.distributed import (
    Wan22DistributedConfig,
    Wan22DistributedContext,
    default_distributed_context,
)
from worldscape_policy.wam.wan22.scheduler import (
    FlowMatchScheduler,
    FlowUniPCMultistepScheduler,
)


@dataclass(frozen=True)
class Wan22KernelConfig:
    num_train_timesteps: int
    num_inference_steps: int
    num_frame_per_block: int
    action_horizon: int
    num_action_per_block: int = 24
    num_state_per_block: int = 1
    cfg_scale: float = 1.0
    sigma_shift: float = 5.0
    decouple_inference_noise: bool = False
    video_inference_final_noise: float = 0.8
    dynamic_cache_schedule: bool = False
    dit_step_mask: tuple[bool, ...] = ()
    decouple_video_action_noise: bool = False
    video_noise_beta_alpha: float = 3.0
    video_noise_beta_beta: float = 1.0
    use_high_noise_emphasis: bool = False
    high_noise_beta_alpha: float = 3.0
    high_noise_beta_beta: float = 1.0
    kv_cache_fifo: bool = False
    training_sigma_shift: float = 5.0
    distributed: Wan22DistributedConfig = field(
        default_factory=Wan22DistributedConfig
    )

    def __post_init__(self) -> None:
        expected = {
            "num_frame_per_block": 2,
            "num_state_per_block": 1,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(
                    f"Native Wan2.2 requires {name}={required}"
                )
        if self.num_action_per_block not in {24, 48}:
            raise ValueError(
                "Native Wan2.2 num_action_per_block must be 24 or 48"
            )
        if self.action_horizon != self.num_action_per_block:
            raise ValueError(
                "Native Wan2.2 action_horizon must match num_action_per_block"
            )


@dataclass(frozen=True)
class Wan22TrainingInputs:
    noisy_video: Tensor
    noisy_action: Tensor
    video_timestep: Tensor
    action_timestep: Tensor
    video_velocity_target: Tensor
    action_velocity_target: Tensor
    video_weight: Tensor | None = None
    action_weight: Tensor | None = None


class Wan22LegacyExactKernel:
    """Single-rank legacy-exact Wan2.2 flow integration and causal caching."""

    def __init__(
        self,
        config: Wan22KernelConfig,
        *,
        distributed_context: Wan22DistributedContext | None = None,
    ) -> None:
        if config.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if config.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        self.config = config
        if distributed_context is not None:
            self._distributed = distributed_context
        elif config.distributed.enabled:
            self._distributed = default_distributed_context(config.distributed)

    @property
    def distributed_context(self) -> Wan22DistributedContext:
        return getattr(self, "_distributed", Wan22DistributedContext.single_rank())

    def prepare_training_inputs(
        self,
        *,
        clean_video_latents: Tensor,
        clean_action: Tensor,
        generator: torch.Generator,
    ) -> Wan22TrainingInputs:
        """Apply the legacy flow-matching noise and target construction."""

        batch_size = clean_video_latents.shape[0]
        device = clean_video_latents.device
        dtype = clean_video_latents.dtype
        if clean_action.device != device:
            raise ValueError("clean video and action must be on the same device")
        generator = self._clone_generator(generator, device=device)
        video_noise = torch.randn(
            clean_video_latents.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        num_frames = clean_video_latents.shape[2]
        if (
            num_frames > 1
            and (num_frames - 1) % self.config.num_frame_per_block
        ):
            raise ValueError(
                "generated latent frames must divide into causal blocks"
            )
        generated_blocks = (num_frames - 1) // self.config.num_frame_per_block
        expected_actions = generated_blocks * self.config.num_action_per_block
        if clean_action.shape[1] != expected_actions:
            raise ValueError(
                "training geometry requires "
                f"{self.config.num_action_per_block} actions per 2-frame block: "
                f"expected {expected_actions}, got {clean_action.shape[1]}"
            )
        timestep_ids = self._sample_video_timestep_ids(
            batch_size=batch_size,
            num_frames=num_frames,
            device=device,
            generator=generator,
        )
        if num_frames > 1:
            generated_ids = timestep_ids[:, 1:].reshape(
                batch_size, -1, self.config.num_frame_per_block
            )
            generated_ids[:, :, 1:] = generated_ids[:, :, :1]
            timestep_ids = torch.cat(
                [timestep_ids[:, :1], generated_ids.flatten(1)],
                dim=1,
            )
        action_noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
        if self.config.decouple_video_action_noise:
            action_timestep_ids = torch.randint(
                0,
                self.config.num_train_timesteps,
                (batch_size, clean_action.shape[1]),
                device=device,
                generator=generator,
            )
        else:
            generated_frames = num_frames - 1
            if generated_frames <= 0 or clean_action.shape[1] % generated_frames:
                raise ValueError(
                    "action horizon must be divisible by generated latent frames"
                )
            actions_per_frame = clean_action.shape[1] // generated_frames
            action_timestep_ids = (
                timestep_ids[:, 1:]
                .unsqueeze(-1)
                .expand(-1, -1, actions_per_frame)
                .reshape(batch_size, -1)
            )
        scheduler = FlowMatchScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=self.config.training_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        scheduler.set_timesteps(
            self.config.num_train_timesteps,
            training=True,
        )
        training_timesteps = scheduler.timesteps.to(device)
        video_timestep = training_timesteps[timestep_ids]
        action_timestep = training_timesteps[action_timestep_ids]
        clean_video_time_major = clean_video_latents.transpose(1, 2)
        video_noise_time_major = video_noise.transpose(1, 2)
        noisy_video = scheduler.add_noise(
            clean_video_time_major.flatten(0, 1),
            video_noise_time_major.flatten(0, 1),
            video_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        video_target = scheduler.training_target(
            clean_video_time_major,
            video_noise_time_major,
            video_timestep,
        ).transpose(1, 2)
        noisy_action = scheduler.add_noise(
            clean_action.flatten(0, 1),
            action_noise.flatten(0, 1),
            action_timestep.flatten(0, 1),
        ).unflatten(0, clean_action.shape[:2])
        video_weight = scheduler.training_weight(
            video_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames)).to(
            device=device,
            dtype=video_target.dtype,
        )
        action_weight = scheduler.training_weight(
            action_timestep.flatten(0, 1),
        ).unflatten(0, clean_action.shape[:2]).to(
            device=device,
            dtype=clean_action.dtype,
        )
        return Wan22TrainingInputs(
            noisy_video=noisy_video.transpose(1, 2),
            noisy_action=noisy_action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            video_velocity_target=video_target,
            action_velocity_target=scheduler.training_target(
                clean_action,
                action_noise,
                action_timestep,
            ),
            video_weight=video_weight,
            action_weight=action_weight,
        )

    def prepare_inference_state(
        self,
        *,
        core: nn.Module,
        inference_state: WAMInferenceState,
        observation_num_frames: int,
    ) -> WAMInferenceState:
        if inference_state.current_start_frame == 0:
            return inference_state
        if int(getattr(core, "max_chunk_size", 0)) == 1:
            # A one-chunk checkpoint only trained visual positions 1-2 and
            # action/state block index 0. Start each replan from a fresh local
            # anchor instead of advancing to the unseen second block.
            return WAMInferenceState()
        if observation_num_frames == 1:
            return WAMInferenceState()
        local_attn_size = int(getattr(core, "local_attn_size", 0))
        if (
            local_attn_size <= 0
            or inference_state.current_start_frame < local_attn_size
        ):
            return inference_state
        # Rebuild a local window from the real rolling
        # observation instead of extrapolating RoPE positions beyond training.
        return WAMInferenceState(
            current_start_frame=1 + self.config.num_frame_per_block,
            rebase_observation_window=True,
        )

    def training_forward(
        self,
        *,
        core: nn.Module,
        i2v_condition: WanI2VCondition,
        clean_video_latents: Tensor,
        clean_action: Tensor,
        noisy_video: Tensor,
        noisy_action: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        recent_visual_prefill: Tensor | None,
    ) -> WorldActionOutput:
        if self.distributed_context.world_size > 1:
            raise RuntimeError(
                "Wan22 multi-rank training is not implemented; distributed WAM "
                "support is inference-only"
            )
        self._validate_core_geometry(
            core,
            video_frames=noisy_video.shape[2],
            action_steps=noisy_action.shape[1],
            state_steps=state.shape[1],
        )
        del negative_cross_attention_tokens, recent_visual_prefill
        seq_len = self._sequence_length(core, noisy_video)
        result = core(
            noisy_video,
            timestep=video_timestep,
            clip_feature=i2v_condition.clip_features,
            y=i2v_condition.masked_latent_y,
            context=cross_attention_tokens,
            seq_len=seq_len,
            state=state,
            embodiment_id=embodiment_id,
            action=noisy_action,
            timestep_action=action_timestep,
            clean_x=clean_video_latents,
            clean_ctx=persistent_prefill,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("Wan22 core training call must return video/action velocity")
        return WorldActionOutput(
            video_velocity=result[0],
            action_velocity=result[1],
        )

    def _validate_core_geometry(
        self,
        core: nn.Module,
        *,
        video_frames: int,
        action_steps: int,
        state_steps: int,
    ) -> None:
        for name, required in (
            ("num_frame_per_block", self.config.num_frame_per_block),
            ("num_action_per_block", self.config.num_action_per_block),
            ("num_state_per_block", self.config.num_state_per_block),
        ):
            if int(getattr(core, name, required)) != required:
                raise ValueError(
                    f"Native Wan2.2 core requires {name}={required}"
                )
        if (video_frames - 1) % self.config.num_frame_per_block:
            raise ValueError(
                "video latent frames must be one anchor plus complete blocks"
            )
        blocks = (video_frames - 1) // self.config.num_frame_per_block
        if (
            action_steps != blocks * self.config.num_action_per_block
            or state_steps != blocks * self.config.num_state_per_block
        ):
            raise ValueError(
                "core input geometry does not match the configured "
                "video/action/state tokens per causal block"
            )

    @torch.no_grad()
    def sample(
        self,
        *,
        core: nn.Module,
        i2v_condition: WanI2VCondition,
        anchor_latent: Tensor | None,
        chunk_latents: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        inference_state: WAMInferenceState,
        generator: torch.Generator,
    ) -> tuple[WorldActionOutput, WAMInferenceState]:
        distributed = self.distributed_context
        distributed.coordinate("kernel.sample.enter")
        distributed.validate_state_owner(
            inference_state.cache_owner_rank,
            inference_state.cache_world_size,
        )
        if chunk_latents.ndim != 5:
            raise ValueError("chunk_latents must have shape [B,C,T,H,W]")
        self._validate_core_geometry(
            core,
            video_frames=1 + self.config.num_frame_per_block,
            action_steps=self.config.action_horizon,
            state_steps=state.shape[1],
        )
        batch_size, channels, _, height, width = chunk_latents.shape
        device = chunk_latents.device
        dtype = chunk_latents.dtype
        frame_seqlen = self._frame_sequence_length(core, height, width)
        permanent_ctx_len = self._permanent_context_length(core, persistent_prefill)

        positive_kv = inference_state.positive_kv_cache
        negative_kv = inference_state.negative_kv_cache
        positive_cross = inference_state.positive_cross_attention_cache
        negative_cross = inference_state.negative_cross_attention_cache
        if positive_kv is None:
            positive_kv = self._create_kv_cache(core, batch_size, dtype, device)
        if positive_cross is None:
            positive_cross = self._create_cross_attention_cache(
                core, batch_size, dtype, device
            )
        use_guidance = (
            self.config.cfg_scale != 1.0
            and negative_cross_attention_tokens is not None
        )
        if use_guidance and negative_kv is None:
            negative_kv = self._create_kv_cache(core, batch_size, dtype, device)
        if use_guidance and negative_cross is None:
            negative_cross = self._create_cross_attention_cache(
                core, batch_size, dtype, device
            )

        contexts = [cross_attention_tokens]
        kv_caches = [positive_kv]
        cross_caches = [positive_cross]
        if use_guidance:
            contexts.append(negative_cross_attention_tokens)
            kv_caches.append(negative_kv)
            cross_caches.append(negative_cross)

        start_frame = inference_state.current_start_frame
        rebase_observation_window = inference_state.rebase_observation_window
        anchor = anchor_latent
        if start_frame == 0 or rebase_observation_window:
            if anchor is None:
                raise ValueError("A new causal window requires anchor_latent")
            self._predict_branches(
                core=core,
                noisy_input=anchor,
                timestep=torch.zeros(
                    batch_size, 1, device=device, dtype=torch.int64
                ),
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                contexts=contexts,
                seq_len=frame_seqlen,
                y=i2v_condition.masked_latent_y[:, :, :1],
                clip_features=i2v_condition.clip_features,
                kv_caches=kv_caches,
                cross_caches=cross_caches,
                start_frame=0,
                update_kv_cache=True,
                persistent_prefill=persistent_prefill,
                permanent_ctx_len=permanent_ctx_len,
            )
            if start_frame == 0:
                start_frame = 1
        if start_frame != 1:
            self._prefill_chunk_if_needed(
                core=core,
                chunk_latents=chunk_latents,
                state_start_frame=start_frame,
                i2v_condition=i2v_condition,
                contexts=contexts,
                kv_caches=kv_caches,
                cross_caches=cross_caches,
                persistent_prefill=persistent_prefill,
                permanent_ctx_len=permanent_ctx_len,
            )

        video_generator = self._clone_generator(generator, device=device)
        action_generator = self._clone_generator(generator, device=device)
        video_shape = (
            batch_size,
            channels,
            self.config.num_frame_per_block,
            height,
            width,
        )
        noisy_video = distributed.owned_tensor(
            lambda: torch.randn(
                video_shape,
                generator=video_generator,
                device=device,
                dtype=dtype,
            ),
            shape=video_shape,
            dtype=dtype,
            device=device,
            tag="kernel.noise.video",
        )
        action_shape = (batch_size, self.config.action_horizon, int(core.action_dim))
        noisy_action = distributed.owned_tensor(
            lambda: torch.randn(
                action_shape,
                generator=action_generator,
                device=device,
                dtype=dtype,
            ),
            shape=action_shape,
            dtype=dtype,
            device=device,
            tag="kernel.noise.action",
        )
        video_scheduler = self._make_scheduler(device)
        action_scheduler = self._make_scheduler(device)
        if self.config.decouple_inference_noise:
            sigma_max = video_scheduler.sigmas[0]
            final_noise = self.config.video_inference_final_noise
            video_scheduler.sigmas = (
                video_scheduler.sigmas * (sigma_max - final_noise) / sigma_max
                + final_noise
            )
            video_scheduler.timesteps = (
                video_scheduler.sigmas[:-1] * self.config.num_train_timesteps
            ).to(torch.int64)

        previous_predictions: list[
            tuple[Tensor, Tensor, Tensor]
        ] = []
        skip_countdown = 0
        for index, video_timestep_value in enumerate(video_scheduler.timesteps):
            action_timestep_value = action_scheduler.timesteps[index]
            should_run, skip_countdown = self._should_run_model(
                index,
                previous_predictions,
                skip_countdown,
            )
            if should_run:
                y = self._slice_i2v_y(
                    i2v_condition.masked_latent_y,
                    start_frame,
                    self.config.num_frame_per_block,
                )
                predictions = self._predict_branches(
                    core=core,
                    noisy_input=noisy_video,
                    timestep=torch.full(
                        (batch_size, self.config.num_frame_per_block),
                        video_timestep_value,
                        device=device,
                        dtype=torch.int64,
                    ),
                    action=noisy_action,
                    timestep_action=torch.full(
                        (batch_size, self.config.action_horizon),
                        action_timestep_value,
                        device=device,
                        dtype=torch.int64,
                    ),
                    state=state,
                    embodiment_id=embodiment_id,
                    contexts=contexts,
                    seq_len=self.config.num_frame_per_block * frame_seqlen,
                    y=y,
                    clip_features=i2v_condition.clip_features,
                    kv_caches=kv_caches,
                    cross_caches=cross_caches,
                    start_frame=start_frame,
                    update_kv_cache=False,
                    persistent_prefill=persistent_prefill,
                    permanent_ctx_len=permanent_ctx_len,
                )
                video_velocity, action_velocity = predictions[0]
                if len(predictions) == 2:
                    uncond_video, _ = predictions[1]
                    video_velocity = uncond_video + self.config.cfg_scale * (
                        video_velocity - uncond_video
                    )
                previous_predictions.append(
                    (video_timestep_value, video_velocity, action_velocity)
                )
                previous_predictions = previous_predictions[-2:]
            else:
                _, video_velocity, action_velocity = previous_predictions[-1]

            noisy_video = video_scheduler.step(
                model_output=video_velocity,
                timestep=video_timestep_value,
                sample=noisy_video,
                step_index=index,
                return_dict=False,
            )[0]
            noisy_action = action_scheduler.step(
                model_output=action_velocity,
                timestep=action_timestep_value,
                sample=noisy_action,
                step_index=index,
                return_dict=False,
            )[0]

        predicted_video = noisy_video
        if inference_state.current_start_frame == 0 and anchor is not None:
            predicted_video = torch.cat([anchor, predicted_video], dim=2)
        next_state = replace(
            inference_state,
            cache_owner_rank=distributed.rank,
            cache_world_size=distributed.world_size,
            current_start_frame=start_frame + self.config.num_frame_per_block,
            positive_kv_cache=positive_kv,
            negative_kv_cache=negative_kv,
            positive_cross_attention_cache=positive_cross,
            negative_cross_attention_cache=negative_cross,
            rebase_observation_window=False,
        )
        distributed.coordinate("kernel.sample.exit")
        return (
            WorldActionOutput(
                action=noisy_action,
                video=predicted_video,
            ),
            next_state,
        )

    def _prefill_chunk_if_needed(
        self,
        *,
        core: nn.Module,
        chunk_latents: Tensor,
        state_start_frame: int,
        i2v_condition: WanI2VCondition,
        contexts: list[Tensor],
        kv_caches: list,
        cross_caches: list,
        persistent_prefill: Tensor | None,
        permanent_ctx_len: int,
    ) -> None:
        if state_start_frame == 1:
            return
        num_frames = self.config.num_frame_per_block
        reference = chunk_latents[:, :, -num_frames:]
        self._predict_branches(
            core=core,
            noisy_input=reference,
            timestep=torch.zeros(
                reference.shape[0],
                num_frames,
                device=reference.device,
                dtype=torch.int64,
            ),
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            contexts=contexts,
            seq_len=self._sequence_length(core, reference),
            y=self._slice_i2v_y(
                i2v_condition.masked_latent_y,
                state_start_frame - num_frames,
                num_frames,
            ),
            clip_features=i2v_condition.clip_features,
            kv_caches=kv_caches,
            cross_caches=cross_caches,
            start_frame=state_start_frame - num_frames,
            update_kv_cache=True,
            persistent_prefill=persistent_prefill,
            permanent_ctx_len=permanent_ctx_len,
        )

    @staticmethod
    def _predict_branches(
        *,
        core: nn.Module,
        noisy_input: Tensor,
        timestep: Tensor,
        action: Tensor | None,
        timestep_action: Tensor | None,
        state: Tensor | None,
        embodiment_id: Tensor | None,
        contexts: list[Tensor],
        seq_len: int,
        y: Tensor,
        clip_features: Tensor,
        kv_caches: list,
        cross_caches: list,
        start_frame: int,
        update_kv_cache: bool,
        persistent_prefill: Tensor | None,
        permanent_ctx_len: int,
    ) -> list[tuple[Tensor, Tensor]]:
        predictions = []
        for context, kv_cache, cross_cache in zip(
            contexts, kv_caches, cross_caches, strict=True
        ):
            result = core(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=context,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_features,
                kv_cache=kv_cache,
                crossattn_cache=cross_cache,
                current_start_frame=start_frame,
                clean_ctx=persistent_prefill,
                permanent_ctx_len=permanent_ctx_len,
            )
            if not isinstance(result, tuple) or len(result) < 2:
                raise TypeError("Wan22 core inference call returned an invalid result")
            video_velocity = result[0].clone()
            action_velocity = result[1]
            if action_velocity is None:
                action_velocity = torch.zeros((), device=video_velocity.device)
            else:
                action_velocity = action_velocity.clone()
            if update_kv_cache:
                if len(result) < 3:
                    raise TypeError("Wan22 core did not return updated KV caches")
                for block_index, updated in enumerate(result[2]):
                    kv_cache[block_index] = updated.clone()
            predictions.append((video_velocity, action_velocity))
        return predictions

    def _make_scheduler(self, device: torch.device) -> FlowUniPCMultistepScheduler:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            self.config.num_inference_steps,
            device=device,
            shift=self.config.sigma_shift,
        )
        return scheduler

    def _should_run_model(
        self,
        index: int,
        previous_predictions: list[tuple[Tensor, Tensor, Tensor]],
        skip_countdown: int,
    ) -> tuple[bool, int]:
        if not self.config.dynamic_cache_schedule:
            if not self.config.dit_step_mask:
                return True, skip_countdown
            return self.config.dit_step_mask[index], skip_countdown
        if len(previous_predictions) < 2:
            return True, skip_countdown
        if skip_countdown > 1:
            return False, skip_countdown - 1
        if skip_countdown == 1:
            return True, 0
        last = previous_predictions[-1][1].flatten(1).float()
        previous = previous_predictions[-2][1].flatten(1).float()
        similarity = torch.nn.functional.cosine_similarity(
            last, previous, dim=1
        ).mean()
        for threshold, countdown in ((0.95, 4), (0.93, 2)):
            if similarity > threshold:
                return False, countdown
        return True, 0

    @staticmethod
    def _create_kv_cache(
        core: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Tensor]:
        head_dim = int(core.dim) // int(core.num_heads)
        return [
            torch.zeros(
                2,
                batch_size,
                0,
                int(core.num_heads),
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(int(core.num_layers))
        ]

    @staticmethod
    def _create_cross_attention_cache(
        core: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Tensor]:
        head_dim = int(core.dim) // int(core.num_heads)
        return [
            torch.zeros(
                2,
                batch_size,
                512,
                int(core.num_heads),
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(int(core.num_layers))
        ]

    @staticmethod
    def _slice_i2v_y(y: Tensor, start: int, length: int) -> Tensor:
        if start + length <= y.shape[2]:
            return y[:, :, max(0, start) : max(0, start) + length]
        return y[:, :, -length:]

    @staticmethod
    def _frame_sequence_length(core: nn.Module, height: int, width: int) -> int:
        _, patch_h, patch_w = tuple(getattr(core, "patch_size", (1, 2, 2)))
        return (height // patch_h) * (width // patch_w)

    @classmethod
    def _sequence_length(cls, core: nn.Module, video: Tensor) -> int:
        return video.shape[2] * cls._frame_sequence_length(
            core, video.shape[3], video.shape[4]
        )

    @staticmethod
    def _permanent_context_length(
        core: nn.Module, persistent_prefill: Tensor | None
    ) -> int:
        if persistent_prefill is None:
            return 0
        patch_t, patch_h, patch_w = tuple(
            getattr(core, "patch_size", (1, 2, 2))
        )
        return (
            (persistent_prefill.shape[2] // patch_t)
            * (persistent_prefill.shape[3] // patch_h)
            * (persistent_prefill.shape[4] // patch_w)
        )

    @staticmethod
    def _clone_generator(
        generator: torch.Generator,
        *,
        device: torch.device,
    ) -> torch.Generator:
        source_device = torch.device(generator.device)
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if source_device.type == "cuda" and source_device.index is None:
            source_device = torch.device("cuda", torch.cuda.current_device())

        # Reuse a compatible generator so every random draw advances the state
        # owned (and checkpointed) by the caller. Generator states are
        # device-specific, so a cross-device copy cannot use set_state().
        if source_device == target_device:
            return generator

        # For a necessary device conversion, derive a fresh seed from the
        # source generator. This advances its checkpointable state while
        # producing a deterministic stream on the target device.
        seed = torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (),
            dtype=torch.int64,
            device=source_device,
            generator=generator,
        ).item()
        return torch.Generator(device=target_device).manual_seed(seed)

    def _sample_video_timestep_ids(
        self,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> Tensor:
        if not (
            self.config.decouple_video_action_noise
            or self.config.use_high_noise_emphasis
        ):
            return torch.randint(
                0,
                self.config.num_train_timesteps,
                (batch_size, num_frames),
                device=device,
                generator=generator,
            )
        if self.config.decouple_video_action_noise:
            alpha = self.config.video_noise_beta_alpha
            beta = self.config.video_noise_beta_beta
        else:
            alpha = self.config.high_noise_beta_alpha
            beta = self.config.high_noise_beta_beta
        alpha_values = torch.full(
            (batch_size, num_frames),
            alpha,
            device=device,
            dtype=torch.float32,
        )
        beta_values = torch.full(
            (batch_size, num_frames),
            beta,
            device=device,
            dtype=torch.float32,
        )
        x = torch._standard_gamma(alpha_values, generator=generator)
        y = torch._standard_gamma(beta_values, generator=generator)
        noise_ratio = x / (x + y)
        return ((1 - noise_ratio) * self.config.num_train_timesteps).long().clamp(
            0,
            self.config.num_train_timesteps - 1,
        )
