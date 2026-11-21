"""Inference-only compilation of locally owned DiT and VAE components."""

from __future__ import annotations

import logging
import os
import tempfile

import torch
import torch.nn as nn

from soma.inference import types
from soma.inference.config import CompileConfig

logger = logging.getLogger(__name__)


class InferenceCompiler:
    """Prepare the DiT and VAE owned by one inference runtime."""

    def __init__(self, config: CompileConfig, *, cache_key: str) -> None:
        self._config: CompileConfig = config
        self._cache_key: str = cache_key
        self._compiler_artifacts_path: str | None = None
        self._prepared: bool = False
        self._vae_prepared: bool = False

    @property
    def enabled(self) -> bool:
        """Whether DiT compilation is enabled."""
        return self._config.dit

    def prepare(self, model: nn.Module) -> nn.Module:
        """Compile the model's execution modules in-place."""
        if not self.enabled or self._prepared:
            return model

        self._setup_cache()
        self._compile_modules(model)
        self._prepared = True
        return model

    def prepare_vae(self, vae: types.VideoVAE, *, incremental_encode: bool) -> types.VideoVAE:
        """Compile whole-video methods once, independently of DiT compilation."""
        if not self._config.vae or self._vae_prepared:
            return vae
        if incremental_encode:
            logger.info("Skipping whole-video VAE compile because incremental encode is enabled")
            return vae
        # Preserve the existing VAE compilation policy. fullgraph/dynamic are
        # DiT settings; VAE keeps torch.compile's existing defaults.
        vae.encode = torch.compile(vae.encode, mode=self._config.mode)
        vae.decode = torch.compile(vae.decode, mode=self._config.mode)
        self._vae_prepared = True
        return vae

    def mark_step_begin(self) -> None:
        """Mark the start of one inference request for CUDA Graph Trees."""
        if self.enabled and self._prepared:
            torch.compiler.cudagraph_mark_step_begin()

    def save_cache(self) -> None:
        """Persist compiler artifacts produced by preparation and execution."""
        if not self.enabled or self._compiler_artifacts_path is None:
            return
        try:
            result = torch.compiler.save_cache_artifacts()
            if result is None:
                return
            data, _ = result
            with open(self._compiler_artifacts_path, "wb") as cache_file:
                cache_file.write(data)
            logger.info(
                "Compiler artifacts saved (%.1fMB)",
                len(data) / 1024 / 1024,
            )
        except Exception as error:  # Compiler cache failures are non-fatal.
            logger.warning("Failed to save compiler artifacts: %s", error)

    def _compile_modules(self, model: nn.Module) -> None:
        # TODO(pytorch>=2.10): remove once mixed-dtype pad_mm is fixed.
        try:
            from torch._inductor.fx_passes import pad_mm

            pad_mm.should_pad_bench = lambda *args, **kwargs: False
        except ImportError:
            pass

        blocks = getattr(model, "vidu_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("compiled DiT must expose vidu_blocks as ModuleList")
        modules = [(f"vidu_blocks.{index}", block) for index, block in enumerate(blocks) if block is not None]
        final_layer = getattr(model, "final_layer", None)
        if final_layer is not None:
            if not isinstance(final_layer, nn.Module):
                raise TypeError("compiled DiT final_layer must be an nn.Module")
            modules.append(("final_layer", final_layer))
        if not modules:
            raise ValueError("compiled DiT has no blocks or final layer")

        logger.info(
            "Compiling %d DiT modules (mode=%r, fullgraph=%s, dynamic=%s)",
            len(modules),
            self._config.mode,
            self._config.fullgraph,
            self._config.dynamic,
        )
        for name, module in modules:
            compiled = torch.compile(
                module,
                mode=self._config.mode,
                fullgraph=self._config.fullgraph,
                dynamic=self._config.dynamic,
            )
            if name == "final_layer":
                model.final_layer = compiled
            else:
                blocks[int(name.rpartition(".")[2])] = compiled

    def _setup_cache(self) -> None:
        cache_dir = self._config.cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(
                tempfile.gettempdir(),
                "soma-inference",
            )
        cache_dir = os.path.join(
            cache_dir,
            "compile_cache",
            self._cache_key,
        )
        os.makedirs(cache_dir, exist_ok=True)

        inductor_dir = os.path.join(cache_dir, "inductor")
        os.makedirs(inductor_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir

        self._compiler_artifacts_path = os.path.join(
            cache_dir,
            "compiler_artifacts.bin",
        )
        if os.path.exists(self._compiler_artifacts_path):
            logger.info("Loading compiler cache: %s", self._cache_key)
            with open(self._compiler_artifacts_path, "rb") as cache_file:
                torch.compiler.load_cache_artifacts(cache_file.read())
        else:
            logger.info(
                "No compiler cache found for: %s (will compile from scratch)",
                self._cache_key,
            )
