"""Model builder utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from leap.core.config import instantiate


def is_accelerate_deepspeed_checkpoint(path: str) -> bool:
    """Whether ``path`` is one raw Accelerate/DeepSpeed checkpoint directory."""

    root = Path(path).expanduser()
    if not root.is_dir():
        return False
    tag_file = root / "latest"
    if tag_file.is_file():
        try:
            tag = tag_file.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if tag and (root / tag / "mp_rank_00_model_states.pt").is_file():
            return True
    return (root / "pytorch_model" / "mp_rank_00_model_states.pt").is_file()


def load_deepspeed_model_checkpoint(
    model: nn.Module,
    path: str,
    *,
    strict: bool = True,
) -> None:
    """Load only model weights from a raw checkpoint into a prepared engine."""

    if not hasattr(model, "load_checkpoint"):
        raise TypeError("prepared model is not a DeepSpeed engine")
    root = Path(path).expanduser().resolve(strict=True)
    tag_file = root / "latest"
    tag = tag_file.read_text(encoding="utf-8").strip() if tag_file.is_file() else "pytorch_model"
    if not tag or not (root / tag / "mp_rank_00_model_states.pt").is_file():
        raise FileNotFoundError(f"invalid Accelerate/DeepSpeed checkpoint: {root}")
    loaded, _client_state = model.load_checkpoint(
        str(root),
        tag=tag,
        load_module_strict=bool(strict),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    if not loaded:
        raise RuntimeError(f"DeepSpeed did not load model checkpoint {root}/{tag}")


def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """Build a model from a config dict.

    Supports:
    - Direct _target_ instantiation
    - Loading pretrained weights

    Args:
        cfg: Model config dict with _target_ and optional pretrained_pth.

    Returns:
        The instantiated model.
    """
    model = instantiate(cfg)

    # Load pretrained weights if specified
    pretrained_pth = cfg.get("pretrained_pth", None)
    if pretrained_pth:
        load_pretrained_weights(model, pretrained_pth)

    return model


def load_pretrained_weights(
    model: nn.Module,
    path: str,
    strict: bool = False,
) -> None:
    """Load pretrained weights from a checkpoint file.

    Supports:
    - .pth / .pt files (PyTorch state dicts)
    - .safetensors files
    - HuggingFace model directories

    Args:
        model: The model to load weights into.
        path: Path to checkpoint file or directory.
        strict: Whether to require exact key matching.
    """
    import os

    if os.path.isdir(path):
        # Accelerate writes a DeepSpeed tag (normally ``pytorch_model``) below
        # the checkpoint root.  Loading that replicated ZeRO-2 model-state file
        # is a raw state-dict load: no exporter, tensor slicing, or key remap.
        root = Path(path)
        if is_accelerate_deepspeed_checkpoint(path):
            tag_file = root / "latest"
            tag = tag_file.read_text(encoding="utf-8").strip() if tag_file.is_file() else "pytorch_model"
            path = str(root / tag / "mp_rank_00_model_states.pt")
        # HuggingFace model directory
        elif hasattr(model, "from_pretrained"):
            model.from_pretrained(path)
            return
        # Try loading from safetensors or pytorch_model.bin
        else:
            safetensors_path = os.path.join(path, "model.safetensors")
            pytorch_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                path = safetensors_path
            elif os.path.exists(pytorch_path):
                path = pytorch_path
            else:
                raise FileNotFoundError(f"No model weights found in {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
        # Extract model weights from various checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict and isinstance(state_dict["module"], dict):
            # DeepSpeed checkpoint: weights under "module" key
            state_dict = state_dict["module"]

    # Handle key prefix mismatches
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    if not model_keys & loaded_keys:
        # Try removing common prefixes
        for prefix in ["module.", "model.", "base_vlm."]:
            stripped = {k.removeprefix(prefix): v for k, v in state_dict.items()}
            if model_keys & set(stripped.keys()):
                state_dict = stripped
                break

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        import logging
        logging.getLogger(__name__).warning(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        import logging
        logging.getLogger(__name__).warning(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
