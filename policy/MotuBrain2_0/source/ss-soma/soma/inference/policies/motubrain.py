"""Thin policy assembly for Motubrain non-autoregressive inference."""

from collections.abc import Mapping
import dataclasses
import logging
import time
from typing import Any

import torch

from soma.data import views
from soma.inference.config import EmbodiedConfig
from soma.inference.diffusion.predictor import DiTPredictor
from soma.inference.diffusion.runner import DiffusionRunner
from soma.inference.io.action import ActionCodec
from soma.inference.io.observation import ObservationWrapper
from soma.inference.policies.base import BasePolicy
from soma.inference.runtime.backend import ModelInferenceRuntime
from soma.inference.types import ImageInput

logger = logging.getLogger(__name__)


class MotubrainPolicy(BasePolicy):
    """Compose observation encoding with non-AR diffusion execution."""

    def __init__(
        self,
        cfg: EmbodiedConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        cfg.validate_action_conditions()
        self._runtime = ModelInferenceRuntime(
            cfg,
            device,
            policy_name=type(self).__name__,
        )
        self.geometry_encoder = self._runtime.build_geometry_encoder()
        self.observation_wrapper = ObservationWrapper(
            cfg.policy.observation,
            cfg.policy.multiview,
            self._runtime,
            sparse_memory=cfg.policy.sparse_memory,
        )
        self.action_codec = ActionCodec(
            cfg.policy.action,
            self._runtime.device,
            smoothing_window=cfg.inference.action_smoothing_window,
        )
        action_length = (cfg.policy.observation.num_frames - 1) * cfg.policy.action.video_action_freq
        predictor = DiTPredictor(
            cfg.inference,
            self._runtime,
            num_views=cfg.policy.multiview.num_views,
            rope_mode=cfg.policy.multiview.rope_mode,
            action_length=action_length,
            view_ids=views.get_view_embed_ids(
                cfg.policy.multiview.view_names,
                cfg.model_kwargs,
                cfg.policy.multiview.view_embed_aliases,
            ),
        )
        self.diffusion_runner = DiffusionRunner(
            cfg,
            predictor,
            action_mask=self.action_codec.model_mask,
        )
        self._prompt_builder = cfg.policy.prompt_builder
        self.view_names = self.observation_wrapper.view_names
        self._decode_video = cfg.inference.decode_video
        self._incremental_vae_decode = cfg.policy.vae.incremental_encode and not cfg.policy.sparse_memory.enabled
        self._head_split_num = cfg.policy.head_split_num
        logger.info("MotubrainPolicy: view_names=%s", self.view_names)

    @torch.no_grad()
    def infer(
        self,
        images: ImageInput,
        instruction: str,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode one observation and run non-AR diffusion."""
        start = time.time()
        prompt = instruction
        if self._prompt_builder is not None:
            action_mask = self.action_codec.model_mask
            if action_mask is None:
                raise ValueError("prompt_builder requires an action profile with a model mask")
            prompt = self._prompt_builder.build(
                {
                    "caption": instruction,
                    "action_mask": action_mask,
                }
            )
        if self.geometry_encoder is not None and self.geometry_encoder.config.required_image_names:
            if not isinstance(images, Mapping):
                raise TypeError("DA3 inference requires a camera-name mapping")
            required = self.geometry_encoder.config.required_image_names
            missing = [name for name in required if name not in images]
            if missing:
                raise ValueError("DA3 input is missing required wire cameras: " f"{missing}; available={list(images)}")
        prepared_views = self.observation_wrapper.prepare(
            images,
            video_views=kwargs.get("video_views"),
        )
        condition = self.observation_wrapper.encode_prepared(
            prepared_views,
            prompt,
        )
        if self.geometry_encoder is not None:
            source_views = dict(
                zip(
                    self.observation_wrapper.view_names,
                    prepared_views,
                    strict=True,
                )
            )
            condition = dataclasses.replace(
                condition,
                action_conditions={
                    self.geometry_encoder.config.condition_name: self.geometry_encoder.encode(source_views)
                },
            )
        action_constraint = self.action_codec.encode_constraint(
            kwargs.get("action_constraint"),
            kwargs,
        )
        prepared = time.time()
        sample = self.diffusion_runner.run(
            condition,
            seed=seed,
            action_constraint=action_constraint,
            rtc_delay=int(kwargs.get("rtc_d", 0)),
        )
        sampled = time.time()
        action_output = self.action_codec.decode(sample.action, kwargs)
        result: dict[str, Any] = {
            "raw_action": action_output.raw_action,
            "action": action_output.action,
        }
        if self._decode_video and sample.video_latents is not None:
            result["video"] = self.observation_wrapper.decode_video_latents(
                sample.video_latents,
                ground_truth_latents=(condition.video_latents if self._incremental_vae_decode else None),
                head_split_num=self._head_split_num,
            )
        finished = time.time()
        logger.info(
            "[timing] input=%.2fs | diffusion=%.2fs | output=%.2fs | " "total=%.2fs",
            prepared - start,
            sampled - prepared,
            finished - sampled,
            finished - start,
        )
        return result

    def reset(self) -> None:
        """Reset observation and diffusion episode state."""
        self.observation_wrapper.reset()
        self.diffusion_runner.reset()

    def close(self) -> None:
        """Close runtime resources."""
        self._runtime.close()

    def save_compile_cache(self) -> None:
        """Persist compiler artifacts created by the runtime."""
        self._runtime.save_compile_cache()
