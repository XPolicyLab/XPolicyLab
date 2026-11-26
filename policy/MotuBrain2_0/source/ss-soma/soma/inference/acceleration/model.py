"""Inference model acceleration lifecycle."""

from __future__ import annotations

import hashlib
import os

import torch.nn as nn
from morph.convs.image import ResizeConfig

from soma.inference import types
from soma.inference.acceleration.compiling import InferenceCompiler
from soma.inference.config import EmbodiedConfig


class ModelAcceleration:
    """Prepare loaded weights and compile inference model execution."""

    def __init__(
        self,
        cfg: EmbodiedConfig,
        *,
        policy_name: str,
    ) -> None:
        if cfg.inference.int8 and cfg.inference.compile.dit and cfg.inference.compile.fullgraph:
            raise ValueError("Beacon INT8 CUDA bindings require compile.fullgraph=False")
        self.cfg: EmbodiedConfig = cfg
        self.policy_name: str = policy_name
        self._weights_prepared: bool = False
        self._execution_prepared: bool = False
        self._compiler: InferenceCompiler = InferenceCompiler(
            cfg.inference.compile,
            cache_key=self._cache_key(),
        )

        if "expandable_segments" not in os.environ.get(
            "PYTORCH_ALLOC_CONF",
            "",
        ):
            previous = os.environ.get("PYTORCH_ALLOC_CONF", "")
            os.environ["PYTORCH_ALLOC_CONF"] = f"{previous},expandable_segments:True".lstrip(",")

    def prepare(self, model: nn.Module) -> nn.Module:
        """Compile model execution once after final device placement."""
        return self.prepare_execution(model)

    def prepare_weights(self, model: nn.Module) -> nn.Module:
        """Apply optional quantization before final storage or compilation."""
        if self._weights_prepared:
            return model
        if self.cfg.inference.int8:
            raise ValueError("This standalone release supports BF16, not INT8")
        self._weights_prepared = True
        return model

    def prepare_execution(self, model: nn.Module) -> nn.Module:
        """Compile execution modules after their final device placement."""
        if not self._execution_prepared:
            self._compiler.prepare(model)
            self._execution_prepared = True
        return model

    def prepare_vae(self, vae: types.VideoVAE) -> types.VideoVAE:
        """Apply the shared compilation policy to a locally owned VAE."""
        return self._compiler.prepare_vae(
            vae,
            incremental_encode=self.cfg.policy.vae.incremental_encode,
        )

    def mark_step_begin(self) -> None:
        """Mark one real compiled-model forward for CUDA graph reuse."""
        self._compiler.mark_step_begin()

    def save_cache(self) -> None:
        """Save compiler artifacts produced by model preparation and inference."""
        self._compiler.save_cache()

    def _cache_key(self) -> str:
        observation = self.cfg.policy.observation
        multiview = self.cfg.policy.multiview
        model = self.cfg.model_kwargs
        resize_signatures = [
            f"{name}:{_resize_signature(multiview.view_resizes.get(name, observation.resize))}"
            for name in multiview.view_names
        ]
        parts = [
            f"layers={model.get('num_layers', 24)}",
            f"heads={model.get('num_attention_heads', 32)}",
            f"hidden={model.get('hidden_size', 4096)}",
            f"action_dim={model['action_dim']}",
            f"action_mask_concat={model.get('action_mask_concat', False)}",
            f"resize={','.join(resize_signatures)}",
            f"frames={observation.num_frames}",
            f"patch={self.cfg.policy.patch_size or model.get('patch_size', 1)}",
            f"compile_mode={self.cfg.inference.compile.mode}",
            f"compile_fullgraph={self.cfg.inference.compile.fullgraph}",
            f"compile_dynamic={self.cfg.inference.compile.dynamic}",
            f"policy={self.policy_name}",
        ]
        key_string = "_".join(parts)
        short_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        return (
            f"{self.policy_name}_{model['action_dim']}d_"
            f"{_resize_label(observation.resize)}_"
            f"{short_hash}"
        )


def _resize_signature(config: ResizeConfig) -> str:
    """Return all geometry and interpolation fields for cache identity."""

    return ":".join(
        str(value)
        for value in (
            config.mode,
            config.width,
            config.height,
            config.target_area,
            config.multiple,
            config.interpolation,
        )
    )


def _resize_label(config: ResizeConfig) -> str:
    """Return a concise human-readable resize label."""

    if config.mode == "area":
        return f"area{config.target_area}m{config.multiple}"
    return f"{config.height}x{config.width}"
