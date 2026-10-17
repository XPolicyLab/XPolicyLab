"""Inference-only Qwen ownership, shared by policy-side text consumers.

The plain handle is deliberately outside the policy's registered modules: adding
an Interactive inference helper must not change checkpoint keys or training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen_checkpoint import bundled_qwen_path


@dataclass(frozen=True)
class RuntimeQwen:
    model: Any
    processor: Any
    source: str


def get_runtime_qwen(
    policy: Any, *, checkpoint: Path, device: Any, verify_hashes: bool = True,
) -> RuntimeQwen:
    """Reuse internal Qwen or load the packaged component once; no external path."""
    cached = getattr(policy, "runtime_qwen", None)
    if cached is not None:
        return cached
    router = getattr(policy, "condition_router", None)
    planner = getattr(getattr(router, "auto", None), "vlm", None)
    interface = getattr(planner, "vlm", None)
    if interface is not None:
        if not callable(getattr(interface.model, "generate", None)):
            raise ValueError("The policy's internal Qwen does not support text generation")
        shared = RuntimeQwen(interface.model, interface.processor, "policy_vlm")
        policy.runtime_qwen = shared
        return shared
    model_path = bundled_qwen_path(checkpoint, verify_hashes=verify_hashes)
    if model_path is None:
        raise FileNotFoundError(f"Checkpoint has no packaged Qwen component: {checkpoint}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().requires_grad_(False).to(device)
    shared = RuntimeQwen(model, processor, str(model_path))
    policy.runtime_qwen = shared
    return shared
