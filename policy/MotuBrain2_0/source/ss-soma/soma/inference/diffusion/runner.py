"""Non-autoregressive diffusion trajectory execution."""

from __future__ import annotations

import einops
import torch

from soma.inference.config import EmbodiedConfig
from soma.inference.diffusion.solvers import BaseSolver
from soma.inference.guidance.rtc import RTCGuidance
from soma.inference.types import DiffusionCondition
from soma.inference.types import DiffusionSample
from soma.inference.types import DiTInput
from soma.inference.types import VelocityPredictor


def _solver_step(
    solver: BaseSolver,
    x: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Advance one stream with an ODE or re-noising solver step."""
    if not solver.uses_renoising:
        return solver.step(x, velocity, (next_timestep - timestep).float())
    noise = torch.randn(
        x.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    return solver.step(
        x,
        velocity,
        t=timestep.float(),
        s=next_timestep.float(),
        noise=noise,
    )


class DiffusionRunner:
    """Advance joint video-action noise through the Non-AR solver loop."""

    def __init__(
        self,
        config: EmbodiedConfig,
        predictor: VelocityPredictor,
        *,
        action_mask: torch.Tensor | None,
    ) -> None:
        observation = config.policy.observation
        inference = config.inference
        model = config.model_kwargs

        self._predictor = predictor
        self._device = predictor.device
        self._num_frames = observation.num_frames
        self._action_dim = model["action_dim"]
        self._action_mask_concat = model.get(
            "action_mask_concat",
            False,
        )
        if not isinstance(self._action_mask_concat, bool):
            raise ValueError("model action_mask_concat must be a boolean, got " f"{self._action_mask_concat!r}")
        self._vae_latent_channels = observation.vae_latent_channels
        self._patch_size = config.policy.patch_size or model.get("patch_size", 1)
        self._action_length = (self._num_frames - 1) * config.policy.action.video_action_freq
        self._num_views = config.policy.multiview.num_views
        self._num_steps = inference.steps
        self._timeshift = inference.timeshift
        self._action_timeshift = inference.action_timeshift or self._timeshift
        self._video_steps = inference.video_steps
        self._action_mask = action_mask.to(self._device, dtype=torch.bool) if action_mask is not None else None
        if self._action_mask is not None and self._action_mask.shape != (self._action_dim,):
            raise ValueError(
                f"action mask must have shape ({self._action_dim},), got " f"{tuple(self._action_mask.shape)}"
            )

        self._video_solver = inference.solver()
        self._action_solver = inference.solver()
        self._rtc = RTCGuidance(inference.simple_rtc, self._device)

    def run(
        self,
        condition: DiffusionCondition,
        *,
        seed: int | None,
        action_constraint: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        rtc_delay: int = 0,
    ) -> DiffusionSample:
        """Run one complete Non-AR diffusion trajectory in model space."""
        current_latents = [latent[:, -1:].to(self._device, dtype=torch.float32) for latent in condition.video_latents]
        if len(current_latents) != self._num_views:
            raise ValueError(f"expected {self._num_views} condition latents, " f"got {len(current_latents)}")
        condition_latents, memory_frame_count = self._prepend_sparse_memory(current_latents, condition)
        text_embedding = condition.text_embedding.to(self._device)
        if action_constraint is not None:
            action_constraint = action_constraint.to(
                self._device,
                dtype=torch.float32,
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        video_latents = self._initialize_video_latents(
            current_latents,
            generator,
        )
        if condition.sparse_memory is not None:
            video_latents = [
                torch.cat((prepared[:, :memory_frame_count], latent), dim=1)
                for prepared, latent in zip(
                    condition_latents,
                    video_latents,
                    strict=True,
                )
            ]
        run_action_mask = self._resolve_action_mask(action_mask)
        action = self._apply_action_mask(
            torch.randn(
                self._action_dim,
                self._action_length,
                device=self._device,
                dtype=torch.float32,
                generator=generator,
            ),
            run_action_mask,
        )
        rtc_state = self._rtc.prepare(
            action_constraint,
            action_length=self._action_length,
            rtc_delay=rtc_delay,
        )
        action = self._apply_action_mask(
            rtc_state.initialize(action),
            run_action_mask,
        )

        action_timesteps = torch.tensor(
            self._timesteps(self._action_timeshift),
            device=self._device,
            dtype=torch.bfloat16,
        )
        video_timesteps = torch.tensor(
            self._timesteps(self._timeshift),
            device=self._device,
            dtype=torch.bfloat16,
        )
        self._video_solver.reset()
        self._action_solver.reset()
        self._predictor.reset()

        video_steps = self._video_steps
        if video_steps is None:
            video_steps = len(action_timesteps)

        for step_index in range(len(action_timesteps) - 1):
            timestep = action_timesteps[step_index]
            next_timestep = action_timesteps[step_index + 1]
            video_active = step_index < video_steps
            action, action_timestep = rtc_state.before_prediction(
                action,
                timestep,
            )
            action = self._apply_action_mask(action, run_action_mask)

            video_patches, video_tokens = self._patchify_video(video_latents)
            video_timestep = (
                video_timesteps[step_index : step_index + 1]
                if video_active
                else torch.tensor(
                    [1.0],
                    device=self._device,
                    dtype=torch.bfloat16,
                )
            )
            video_velocity, action_velocity = self._predictor.predict(
                DiTInput(
                    video_tokens=video_tokens,
                    video_patches=video_patches,
                    action=self._model_action(action, run_action_mask),
                    text_embedding=text_embedding,
                    video_timestep=video_timestep,
                    action_timestep=action_timestep,
                    action_conditions=condition.action_conditions,
                ),
                timestep=timestep,
                step_index=step_index,
                skip_video=not video_active,
            )

            expected_action_velocity_shape = (
                self._action_length,
                self._action_dim,
            )
            if action_velocity.shape != expected_action_velocity_shape:
                raise ValueError(
                    "action velocity must have shape "
                    f"{expected_action_velocity_shape}, got "
                    f"{tuple(action_velocity.shape)}"
                )
            action = _solver_step(
                self._action_solver,
                action,
                einops.rearrange(
                    action_velocity,
                    "a c -> c a",
                ).float(),
                timestep,
                next_timestep,
                generator,
            )
            action = self._apply_action_mask(
                rtc_state.after_solver(action),
                run_action_mask,
            )

            if video_active:
                if video_velocity is None:
                    raise RuntimeError("video velocity is required while video denoising is active")
                video_latents = self._update_video_latents(
                    video_tokens,
                    video_velocity,
                    video_timesteps[step_index],
                    video_timesteps[step_index + 1],
                    generator,
                    condition_latents,
                )

        self._predictor.finish()
        return DiffusionSample(
            action=einops.rearrange(action, "c a -> a c").float(),
            video_latents=([latent[:, memory_frame_count:] for latent in video_latents] if video_steps > 0 else None),
        )

    def reset(self) -> None:
        """Clear trajectory-scoped predictor state."""
        self._predictor.reset()

    def _initialize_video_latents(
        self,
        condition_latents: list[torch.Tensor],
        generator: torch.Generator | None,
    ) -> list[torch.Tensor]:
        latent_frame_count = self._num_latents(self._num_frames)
        video_latents = []
        for condition in condition_latents:
            latent = torch.randn(
                self._vae_latent_channels,
                latent_frame_count,
                condition.shape[2],
                condition.shape[3],
                device=self._device,
                dtype=torch.float32,
                generator=generator,
            )
            latent[:, :1] = condition[:, :1]
            video_latents.append(latent)
        return video_latents

    def _prepend_sparse_memory(
        self,
        current_latents: list[torch.Tensor],
        condition: DiffusionCondition,
    ) -> tuple[list[torch.Tensor], int]:
        """Build the clean history prefix followed by the current latent."""
        memory = condition.sparse_memory
        if memory is None:
            return current_latents, 0
        if len(memory.latents) != self._num_views:
            raise ValueError(f"expected {self._num_views} sparse-memory views, " f"got {len(memory.latents)}")
        prepared = []
        for view_index, (history, current) in enumerate(zip(memory.latents, current_latents, strict=True)):
            if history.ndim != 4 or history.shape[1] != 3:
                raise ValueError(
                    "each sparse-memory view must contain three compressed history "
                    f"latent frames, got view {view_index} shape "
                    f"{tuple(history.shape)}"
                )
            history = history.to(self._device, dtype=torch.float32)
            if history.shape[0] != current.shape[0] or history.shape[2:] != current.shape[2:]:
                raise ValueError(
                    "sparse-memory and current latent shapes must match, got "
                    f"{tuple(history.shape)} and {tuple(current.shape)}"
                )
            prepared.append(torch.cat((history, current), dim=1))
        return prepared, 3

    def _patchify_video(
        self,
        video_latents: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        patches_by_view = []
        tokens_by_view = []
        for latent in video_latents:
            patches = einops.rearrange(
                latent,
                "c t (h ph) (w pw) -> (ph pw c) t h w",
                ph=self._patch_size,
                pw=self._patch_size,
            )
            patches_by_view.append(patches.cpu())
            tokens_by_view.append(einops.rearrange(patches, "c t h w -> (t h w) c"))
        return patches_by_view, torch.cat(tokens_by_view, dim=0)

    def _update_video_latents(
        self,
        video_tokens: torch.Tensor,
        video_velocity: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        generator: torch.Generator | None,
        condition_latents: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        new_tokens = _solver_step(
            self._video_solver,
            video_tokens.float(),
            video_velocity.float(),
            timestep,
            next_timestep,
            generator,
        )
        latent_frame_count = self._num_latents(self._num_frames)
        if condition_latents:
            latent_frame_count += condition_latents[0].shape[1] - 1
        updated_latents = []
        offset = 0
        for condition in condition_latents:
            patch_height = condition.shape[2] // self._patch_size
            patch_width = condition.shape[3] // self._patch_size
            token_count = latent_frame_count * patch_height * patch_width
            view_tokens = new_tokens[offset : offset + token_count]
            offset += token_count
            patches = einops.rearrange(
                view_tokens,
                "(t h w) c -> c t h w",
                t=latent_frame_count,
                h=patch_height,
                w=patch_width,
            )
            latent = einops.rearrange(
                patches,
                "(ph pw c) t h w -> c t (h ph) (w pw)",
                ph=self._patch_size,
                pw=self._patch_size,
                c=self._vae_latent_channels,
            )
            latent[:, : condition.shape[1]] = condition
            updated_latents.append(latent)
        return updated_latents

    def _resolve_action_mask(
        self,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if action_mask is None:
            return self._action_mask[:, None] if self._action_mask is not None else None
        action_mask = action_mask.to(self._device, dtype=torch.bool)
        if action_mask.shape == (self._action_dim,):
            return action_mask[:, None]
        if action_mask.shape == (self._action_length, self._action_dim):
            return action_mask.t().contiguous()
        raise ValueError(
            "action mask must have shape "
            f"({self._action_dim},) or "
            f"({self._action_length}, {self._action_dim}), got "
            f"{tuple(action_mask.shape)}"
        )

    @staticmethod
    def _apply_action_mask(
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_mask is None:
            return action
        return torch.where(
            action_mask,
            action,
            torch.zeros_like(action),
        )

    def _model_action(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._action_mask_concat:
            return action
        if action_mask is None:
            concat_mask = torch.ones_like(action)
        else:
            concat_mask = action_mask.expand_as(action).to(action.dtype)
        return torch.cat([action, concat_mask], dim=0)

    def _timesteps(self, timeshift: float) -> list[float]:
        timesteps = torch.linspace(1.0, 0.0, self._num_steps + 1).tolist()
        return [timestep * timeshift / (1 + (timeshift - 1) * timestep) for timestep in timesteps]

    @staticmethod
    def _num_latents(num_frames: int) -> int:
        return 0 if num_frames <= 0 else (num_frames - 1) // 4 + 1
