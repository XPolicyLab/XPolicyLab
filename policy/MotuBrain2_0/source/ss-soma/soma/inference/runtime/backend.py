"""Construction and lifecycle of models used by policy inference."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from contextlib import nullcontext
import logging
from typing import cast

import torch
import torch.nn as nn
from falcon.utils.locate import locate

from soma.inference.acceleration import embedding_cache
from soma.inference.acceleration.model import ModelAcceleration
from soma.inference.config import EmbodiedConfig
from soma.inference.runtime.encoders import build_text_encoder
from soma.inference.runtime.encoders import build_vae
from soma.inference.runtime.encoders import encode_text
from soma.inference.runtime.geometry import DA3ConditionEncoder
from soma.inference.runtime.incremental_vae import IncrementalWanVAEDecoder
from soma.inference.runtime.incremental_vae import IncrementalWanVAEEncoder
from soma.inference.types import DiTModel
from soma.inference.types import DiTOutput
from soma.inference.types import TextEncoder
from soma.inference.types import VideoVAE

logger = logging.getLogger(__name__)


class ModelInferenceRuntime:
    """Own model components for direct local inference."""

    def __init__(
        self,
        cfg: EmbodiedConfig,
        device: str | torch.device,
        *,
        policy_name: str,
    ) -> None:
        self.cfg: EmbodiedConfig = cfg
        self.device: torch.device = torch.device(device)
        self.vae: VideoVAE
        self.text_encoder: TextEncoder
        self._dit: DiTModel
        self.embedding_cache = embedding_cache.EmbeddingCache(cfg.policy.text_encoder.cache_bytes)
        self.incremental_vae_encode: bool = bool(cfg.policy.vae.incremental_encode)
        self._incremental_vae_encoders: dict[
            str,
            IncrementalWanVAEEncoder,
        ] = {}
        self._incremental_vae_decoders: dict[
            str,
            IncrementalWanVAEDecoder,
        ] = {}
        self._acceleration: ModelAcceleration = ModelAcceleration(
            cfg,
            policy_name=policy_name,
        )

        context_parallel_size = int(getattr(cfg.inference, "context_parallel_size", 1))
        if context_parallel_size > 1:
            raise NotImplementedError(
                "Soma direct inference currently requires "
                "context_parallel_size=1; use the pipeline server for "
                "multi-GPU inference"
            )

        self._build_local_components()

    def build_geometry_encoder(self) -> DA3ConditionEncoder | None:
        """Construct the local geometry encoder."""
        geometry_config = self.cfg.policy.geometry
        if geometry_config is None:
            return None
        return DA3ConditionEncoder(
            geometry_config,
            feature_dim=int(self.cfg.model_kwargs["action_conditions"][geometry_config.condition_name]["feature_dim"]),
            device=self.device,
        )

    def _build_local_components(self) -> None:
        self.vae = build_vae(self.cfg.policy.vae, self.device)
        self._acceleration.prepare_vae(self.vae)
        self.text_encoder = build_text_encoder(
            self.cfg.policy.text_encoder,
            self.device,
        )
        logger.info("Encoders loaded")

        factory_spec = self.cfg.policy.dit_factory
        if isinstance(factory_spec, str):
            factory_spec = locate(factory_spec)
        factory = cast(Callable[..., nn.Module], factory_spec)
        dit = factory(
            self.cfg.checkpoint_path,
            **self.cfg.model_kwargs,
        ).to(self.device)
        parameter_count = sum(parameter.numel() for parameter in dit.parameters()) / 1e9
        logger.info("DIT loaded. Params: %.3fB", parameter_count)
        self._acceleration.prepare_weights(dit)
        self._dit = cast(DiTModel, self._acceleration.prepare(dit))

    def forward_dit(
        self,
        *,
        video_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        video_timestep: torch.Tensor,
        action: torch.Tensor,
        action_timestep: torch.Tensor,
        metadata: Mapping[str, object],
    ) -> DiTOutput:
        """Execute one local DiT model invocation."""
        self._acceleration.mark_step_begin()
        execution_context = torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
        )

        with torch.inference_mode(), execution_context:
            output = self._dit(
                v=video_tokens.bfloat16(),
                caption=text_embedding.bfloat16().clone(),
                t=video_timestep,
                a=action.t().bfloat16(),
                t_a=action_timestep,
                **dict(metadata),
            )
        return output[0].clone(), output[1].clone()

    def encode_video(
        self,
        video: torch.Tensor,
        *,
        stream_id: str,
        incremental: bool | None = None,
    ) -> torch.Tensor:
        """Encode one complete observation chunk for a causal video stream."""
        vae_dtype = getattr(self.vae, "dtype", torch.float16)
        video = video.unsqueeze(0).to(self.device, dtype=vae_dtype)
        use_incremental = self.incremental_vae_encode if incremental is None else incremental
        if use_incremental:
            encode_incremental = getattr(self.vae, "encode_incremental", None)
            if encode_incremental is not None:
                latent = encode_incremental(video, stream_id=stream_id)
            else:
                encoder = self._incremental_vae_encoders.get(stream_id)
                if encoder is None:
                    encoder = IncrementalWanVAEEncoder(self.vae)
                    self._incremental_vae_encoders[stream_id] = encoder
                latent = encoder.encode(video)
        else:
            with torch.amp.autocast("cuda", dtype=vae_dtype):
                latent = self.vae.encode(video)
        latent = latent.squeeze(0).float()
        # Locally compiled CUDA graphs may reuse output storage for the next view.
        return latent.clone()

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode one video latent into an RGB uint8 tensor on CPU."""
        vae_dtype = getattr(self.vae, "dtype", torch.float16)
        latent = latent.unsqueeze(0).to(self.device, dtype=vae_dtype)
        with torch.amp.autocast("cuda", dtype=vae_dtype):
            video = self.vae.decode(latent)
        return self._format_decoded_video(video)

    def decode_video_incremental(
        self,
        ground_truth_latent: torch.Tensor,
        prediction_latent: torch.Tensor,
        *,
        stream_id: str,
        include_condition: bool = True,
    ) -> torch.Tensor:
        """Decode predictions from a cache advanced only by GT observations."""
        vae_dtype = getattr(self.vae, "dtype", torch.float16)
        ground_truth_latent = ground_truth_latent.unsqueeze(0).to(
            self.device,
            dtype=vae_dtype,
        )
        prediction_latent = prediction_latent.unsqueeze(0).to(
            self.device,
            dtype=vae_dtype,
        )
        decode_incremental = getattr(self.vae, "decode_incremental", None)
        if decode_incremental is not None:
            video = decode_incremental(
                ground_truth_latent,
                prediction_latent,
                stream_id=stream_id,
                include_condition=include_condition,
            )
        else:
            decoder = self._incremental_vae_decoders.get(stream_id)
            if decoder is None:
                decoder = IncrementalWanVAEDecoder(self.vae)
                self._incremental_vae_decoders[stream_id] = decoder
            autocast = (
                torch.amp.autocast("cuda", dtype=vae_dtype)
                if self.device.type == "cuda" and vae_dtype != torch.float32
                else nullcontext()
            )
            with autocast:
                video = decoder.decode(
                    ground_truth_latent,
                    prediction_latent,
                    include_condition=include_condition,
                )
        return self._format_decoded_video(video)

    @staticmethod
    def _format_decoded_video(video: torch.Tensor) -> torch.Tensor:
        """Convert a decoded BCHWT video to THWC uint8 on CPU."""
        video = video.squeeze(0).float().permute(1, 2, 3, 0)
        return ((video.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Reuse exact text within this runtime's fixed encoder instance."""
        if self.embedding_cache is None:
            embedding = encode_text(self.text_encoder, prompt)
        else:
            key = embedding_cache.text_embedding_key(
                prompt,
                max_length=self.cfg.policy.text_encoder.max_length,
                dtype="native",
            )
            embedding = self.embedding_cache.get_or_compute(
                key,
                lambda: encode_text(self.text_encoder, prompt),
            )
        return embedding.to(self.device)

    def reset_observation(self) -> None:
        """Clear locally owned causal observation state."""
        self._incremental_vae_encoders.clear()
        self._incremental_vae_decoders.clear()

    def save_compile_cache(self) -> None:
        """Persist compiler artifacts created during model execution."""
        self._acceleration.save_cache()

    def close(self) -> None:
        """Direct inference does not own a separate transport."""
        return None


class ModelFSDPRuntime:
    """Borrow a caller-owned model, including FSDP, for inference calls."""

    def __init__(
        self,
        model: nn.Module,
        device: str | int | torch.device,
    ) -> None:
        self._dit = model
        if isinstance(device, int):
            self.device = torch.device("cuda", device)
        else:
            self.device = torch.device(device)

    def forward_dit(
        self,
        *,
        video_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        video_timestep: torch.Tensor,
        action: torch.Tensor,
        action_timestep: torch.Tensor,
        metadata: Mapping[str, object],
    ) -> DiTOutput:
        """Run one forward without taking ownership of the FSDP model."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self._dit(
                v=video_tokens.bfloat16(),
                caption=text_embedding.bfloat16().clone(),
                t=video_timestep,
                a=action.t().bfloat16(),
                t_a=action_timestep,
                **dict(metadata),
            )
        return output[0].clone(), output[1].clone()
