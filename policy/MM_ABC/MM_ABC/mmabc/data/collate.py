"""Batch samples and optionally preprocess vision-language inputs in workers."""

from __future__ import annotations

import numpy as np
import torch

# Keys carried through as Python lists rather than tensors.
META_KEYS = ("prompt", "profile", "embodiment_tag", "action_type")


def collate(samples: list[dict]) -> dict:
    out: dict[str, object] = {}
    for key in ("state", "state_mask", "target", "target_mask", "view_mask"):
        out[key] = torch.from_numpy(np.stack([s[key] for s in samples])).float()

    out["aux_active"] = torch.tensor([float(s["aux_active"]) for s in samples])
    # (B, V, 1 + n_future, H, W, 3) uint8: current frame plus one frame per segment.
    out["images"] = torch.from_numpy(np.stack([s["images"] for s in samples]))
    if "future_valid" in samples[0]:
        out["future_valid"] = torch.from_numpy(
            np.stack([s["future_valid"] for s in samples])
        ).float()
    # Per-dim action scale (real-unit de-normalisation), carried so the model
    # can report a decoded-action error that is comparable across flow
    # parametrisations. Optional: only decoupled/native embodiments emit it.
    if "action_scale" in samples[0]:
        out["action_scale"] = torch.from_numpy(
            np.stack([s["action_scale"] for s in samples])
        ).float()

    for key in META_KEYS:
        out[key] = [s[key] for s in samples]
    return out


class Collator:
    """``collate`` plus backbone inputs built in the worker.

    The processor is loaded lazily, once per worker process, so the object
    pickles cheaply into every worker.
    """

    def __init__(self, processor_path: str | None) -> None:
        self.processor_path = processor_path
        self._processor = None

    def __getstate__(self) -> dict:
        return {"processor_path": self.processor_path, "_processor": None}

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            torch.set_num_threads(1)
            self._processor = AutoProcessor.from_pretrained(self.processor_path, use_fast=True)
        return self._processor

    def __call__(self, samples: list[dict]) -> dict:
        out = collate(samples)
        if self.processor_path:
            from mmabc.models.backbone import build_inputs

            current = np.stack([s["images"][:, 0] for s in samples])
            out["vlm"] = build_inputs(
                self._get_processor(),
                out["prompt"],
                current,
                out["view_mask"].numpy(),
                device=None,
            )
        return out


def to_device(batch: dict, device: torch.device, *, non_blocking: bool = True) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            out[k] = to_device(v, device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out
