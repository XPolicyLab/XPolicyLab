"""Base class and construction entry point for the CogWAM framework.

CogWAM ships exactly one framework, so there is no registry and no dynamic
module discovery: ``build_model`` imports the single class and validates that
the config asks for it. This is a deliberate simplification of the upstream
design, whose registry eagerly imported every framework in the tree just to
resolve one name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PretrainedConfig, PreTrainedModel

FRAMEWORK_NAME = "CogWAM"


class CogWAMBase(PreTrainedModel):
    """Thin ``PreTrainedModel`` base carrying the train/infer contract.

    Subclasses register their submodules in ``__init__``. The attribute names
    they choose become the top-level checkpoint key prefixes, so renaming one
    invalidates every published checkpoint -- see README.md.
    """

    def __init__(self, hf_config: PretrainedConfig | None = None) -> None:
        super().__init__(hf_config if hf_config is not None else PretrainedConfig())

    def forward(self, examples: list[dict], **kwargs) -> dict:
        """Training step. Returns a dict containing at least ``action_loss``."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward(examples)")

    def predict_action(self, examples: list[dict], **kwargs) -> dict:
        """Inference. Returns a dict containing at least ``normalized_actions``."""
        raise NotImplementedError(f"{type(self).__name__} must implement predict_action(examples)")

    def compute_loss(self, tag: str, batch: Any, loss_scale: dict | None = None) -> dict[str, torch.Tensor] | None:
        """Route a tagged batch to the right forward pass.

        The trainer drives two streams: ``"vla"`` (physical: action + world
        denoising) and ``"semantic"`` (the KEEP/UPDATE next-token objective),
        which the framework handles inside ``forward``. Unknown tags return
        ``None`` so the trainer can skip them.
        """
        if tag != "vla":
            return None
        scale = (loss_scale or {}).get(tag, 1.0)
        out = self.forward(batch)
        return {key: value * scale for key, value in out.items() if isinstance(value, torch.Tensor)}


def build_model(cfg) -> CogWAMBase:
    """Instantiate the framework named by ``cfg.framework.name``."""
    framework = getattr(cfg, "framework", None)
    name = getattr(framework, "name", None) if framework is not None else None
    if name is None:
        raise ValueError("Missing `cfg.framework.name`")
    if str(name) != FRAMEWORK_NAME:
        raise ValueError(f"This repository only implements `{FRAMEWORK_NAME}`, got {name!r}")

    from cogwam.models.cogwam import CogWAM

    return CogWAM(cfg)


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint, accepting both release (.safetensors) and training (.pt) formats."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)
