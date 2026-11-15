"""One-step DiT prediction with metadata, CFG, and velocity caching."""

from __future__ import annotations

import torch
from falcon import distributed
from soma.data import conditions

from soma.inference.acceleration.dit_cache import DiTCache
from soma.inference.config import InferenceConfig
from soma.inference.diffusion.metadata import get_audio_meta_dict
from soma.inference.diffusion.metadata import get_caption_meta_dict
from soma.inference.diffusion.metadata import get_multiview_video_meta_dict
from soma.inference.guidance.cfg import apply_cfg
from soma.inference.types import DiTForwardRuntime
from soma.inference.types import DiTInput
from soma.inference.types import RemoteActionCondition
from soma.inference.types import Velocity
from soma.inference.types import VelocityForward


class DiTPredictor:
    """Run one DiT prediction behind CFG and DiTCache composition."""

    def __init__(
        self,
        config: InferenceConfig,
        runtime: DiTForwardRuntime,
        *,
        num_views: int,
        rope_mode: str,
        action_length: int,
        view_ids: torch.Tensor | None = None,
    ) -> None:
        self._runtime = runtime
        self._num_views = num_views
        self._rope_mode = rope_mode
        self._action_length = action_length
        self._view_ids = view_ids
        if view_ids is not None and view_ids.shape != (num_views,):
            raise ValueError("view_ids must contain one ID per input view")
        self._cfg_scale = config.cfg_scale
        self._metadata: dict[str, object] | None = None
        self._cache = DiTCache(
            threshold=config.dit_cache_threshold,
            threshold_2=config.dit_cache_threshold_2,
            skip_steps=config.dit_cache_skip_steps,
            skip_steps_2=config.dit_cache_skip_steps_2,
            damping=config.dit_cache_damping,
        )

    @property
    def device(self) -> torch.device:
        """Device on which prediction inputs must be allocated."""
        return self._runtime.device

    def reset(self) -> None:
        """Clear DiTCache state before a new diffusion trajectory."""
        self._cache.reset()
        self._metadata = None

    def predict(
        self,
        model_input: DiTInput,
        *,
        timestep: torch.Tensor,
        step_index: int,
        skip_video: bool,
        forward: VelocityForward | None = None,
    ) -> Velocity:
        """Predict velocity, optionally using a custom model-forward function."""
        prediction_forward = forward
        if prediction_forward is None:
            remote_condition = any(
                isinstance(value, RemoteActionCondition) for value in (model_input.action_conditions or {}).values()
            )
            if remote_condition:
                if self._metadata is None:
                    self._metadata = self._forward_metadata(model_input)
                metadata = self._metadata
            else:
                metadata = self._forward_metadata(model_input)

            def default_forward(unconditional: bool) -> Velocity:
                forward_metadata = dict(metadata)
                if unconditional:
                    forward_metadata["force_drop_ids"] = True
                return self._runtime.forward_dit(
                    video_tokens=model_input.video_tokens,
                    text_embedding=model_input.text_embedding,
                    video_timestep=model_input.video_timestep,
                    action=model_input.action,
                    action_timestep=model_input.action_timestep,
                    metadata=forward_metadata,
                )

            prediction_forward = default_forward

        def predict_with_guidance() -> Velocity:
            return apply_cfg(
                prediction_forward,
                self._cfg_scale,
            )

        return self._cache.step(
            timestep,
            predict_with_guidance,
            step_index,
            skip_video,
        )

    def finish(self) -> None:
        """Log DiTCache statistics for the completed trajectory."""
        self._cache.log_stats()

    def _forward_metadata(self, model_input: DiTInput) -> dict[str, object]:
        metadata: dict[str, object] = {}
        metadata.update(
            get_multiview_video_meta_dict(
                list(model_input.video_patches),
                view_counts=[self._num_views],
                rope_mode=self._rope_mode,
            )
        )
        metadata.update(
            get_caption_meta_dict(
                [model_input.text_embedding],
                caption_lengths=[model_input.text_embedding.shape[0]],
            )
        )
        metadata.update(
            get_audio_meta_dict(
                [model_input.action],
                audio_lengths=[self._action_length],
            )
        )
        metadata["micro_batch_size"] = 1
        metadata["f_seqlen_offsets"] = None
        if self._view_ids is not None:
            metadata["view_ids_view"] = self._view_ids
        if model_input.video_key_mask is not None:
            metadata["video_key_mask"] = model_input.video_key_mask
        if model_input.action_conditions:
            remote = [
                value for value in model_input.action_conditions.values() if isinstance(value, RemoteActionCondition)
            ]
            if remote:
                if len(remote) != 1 or len(model_input.action_conditions) != 1:
                    raise ValueError("remote geometry requires exactly one action condition")
                metadata["action_condition_cache_key"] = remote[0].cache_key
                return {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in metadata.items()
                }
            metadata["action_conditions"] = {
                name: conditions.pack_action_condition(
                    {layer: [value] for layer, value in features.items()}, distributed.get_cp_size()
                )
                for name, features in model_input.action_conditions.items()
            }
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in metadata.items()
        }
