"""PrismWAM Stage 2: shared RGB/depth blocks from Stage 1 and a separate action expert."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf


from .rgb_depth_action import PrismWAMRGBDepthAction


@dataclass(frozen=True)
class DepthOnlyParentProvenance:
    """Validated identity of the Stage-1 (depth-only) checkpoint."""

    path: str
    step: int
    depth_parameter_keys: int


def validate_depth_only_parent_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_step: int,
) -> DepthOnlyParentProvenance:
    """Check that Stage 2 starts from the final, depth-only Stage-1 weights."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {path}.")
    payload = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Stage-1 checkpoint must be a dictionary: {path}.")
    step = int(payload.get("step", -1))
    if step != int(expected_step):
        raise ValueError(
            f"Stage-1 checkpoint must be step {expected_step}, got {step}: {path}."
        )
    mot_state = payload.get("mot")
    if not isinstance(mot_state, dict):
        raise ValueError(f"Stage-1 checkpoint has no `mot` state dictionary: {path}.")

    depth_prefix = "mixtures.video."
    depth_keys = [key for key in mot_state if key.startswith(depth_prefix)]
    foreign_mixtures = sorted(
        key
        for key in mot_state
        if key.startswith("mixtures.") and not key.startswith(depth_prefix)
    )
    if foreign_mixtures:
        raise ValueError(
            "The Stage-1 checkpoint must contain only depth weights; found "
            f"parameters: {foreign_mixtures[:20]}."
        )
    if not depth_keys:
        raise ValueError(f"Stage-1 checkpoint contains no depth weights: {path}.")
    if not all(isinstance(mot_state[key], torch.Tensor) for key in depth_keys):
        raise TypeError(f"Stage-1 checkpoint contains non-tensor state: {path}.")
    return DepthOnlyParentProvenance(
        path=str(path),
        step=step,
        depth_parameter_keys=len(depth_keys),
    )


def warmup_cosine_lr_factor(step: int, *, schedule_steps: int) -> float:
    """Linear warm-up over 5% of the steps, then cosine decay to 1% of the LR.

    Equals ``LinearLR`` followed by ``CosineAnnealingLR``, including step zero.
    """
    if schedule_steps < 20:
        raise ValueError("The LR schedule requires at least 20 steps.")
    warmup = int(schedule_steps * 0.05)
    step = min(max(int(step), 0), schedule_steps)
    if step < warmup:
        return 1.0 / warmup + (1.0 - 1.0 / warmup) * step / warmup
    progress = (step - warmup) / (schedule_steps - warmup)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


# Checkpoint key of the Stage-1 initialization record.
STAGE1_RECORD_KEY = "depth_only_stage1"


class PrismWAMSharedDepthParent(PrismWAMRGBDepthAction):
    """Two parameter backbones: shared RGB/Depth blocks and independent Action."""

    def build_lr_lambdas(
        self, *, param_groups: list[dict], total_steps: int, warmup_steps: int
    ) -> list[Callable[[int], float]]:
        del warmup_steps
        horizon = self.lr_schedule_steps
        if not 0 < total_steps <= horizon:
            raise ValueError("Training must stop within the recorded LR schedule.")

        def factor(step: int) -> float:
            return warmup_cosine_lr_factor(step, schedule_steps=horizon)

        return [factor for _ in param_groups]

    def _extra_checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._extra_checkpoint_payload()
        provenance = getattr(self, "stage1_provenance", None)
        if provenance is None:
            raise RuntimeError("Missing the Stage-1 initialization record.")
        payload["rgb_depth_action_config"][STAGE1_RECORD_KEY] = dict(provenance)
        return payload

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        provenance = payload.get("rgb_depth_action_config", {}).get(STAGE1_RECORD_KEY)
        if not isinstance(provenance, dict):
            raise ValueError("Checkpoint has no Stage-1 initialization record.")
        expected = {
            "parent_step": self.depth_parent_expected_step,
            "action_init_source": self.action_init_source,
            "lr_schedule_steps": self.lr_schedule_steps,
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(
                    f"Stage-2 checkpoint does not match this recipe: {key}."
                )
        self.stage1_provenance = dict(provenance)
        return payload


def create_prismwam_rgb_depth_action(
    *,
    model_dtype: torch.dtype,
    device: str,
    depth_parent_expected_step: int = 15000,
    action_init_source: str = "wan",
    lr_schedule_steps: int = 15000,
    _model_class: type[PrismWAMSharedDepthParent] = PrismWAMSharedDepthParent,
    **kwargs,
) -> PrismWAMSharedDepthParent:
    """Initialize the shared visual backbone from the complete Stage-1 checkpoint.

    The action expert starts from the Wan action initializer; both choices are
    recorded with the checkpoint. Loading a final policy uses
    checkpoint_init_only and does not reopen either initializer.
    """
    if action_init_source != "wan":
        raise ValueError(
            "action_init_source must be wan: the action expert starts from the Wan action initializer."
        )
    if (
        isinstance(depth_parent_expected_step, bool)
        or not isinstance(depth_parent_expected_step, int)
        or depth_parent_expected_step <= 0
    ):
        raise ValueError("depth_parent_expected_step must be a positive integer.")
    if (
        isinstance(lr_schedule_steps, bool)
        or int(lr_schedule_steps) != lr_schedule_steps
        or int(lr_schedule_steps) < 20
    ):
        raise ValueError(
            "The LR schedule requires an integer horizon of at least 20 steps."
        )
    if not kwargs.get("share_rgb_depth_blocks"):
        raise ValueError("RGB and Depth must share all transformer blocks.")
    checkpoint_only = bool(kwargs.get("checkpoint_init_only", False))
    parent = None
    if not checkpoint_only:
        required = {
            "skip_depth_dit_load_from_pretrain": True,
            "skip_rgb_dit_load_from_pretrain": False,
            "skip_action_dit_load_from_pretrain": False,
        }
        if any((kwargs.get(key) != value for (key, value) in required.items())):
            raise ValueError(
                "Use the prescribed Depth/RGB/Action initialization paths."
            )
        if not kwargs.get("action_dit_pretrained_path"):
            raise ValueError("The action backbone initializer is required.")
        parent = validate_depth_only_parent_checkpoint(
            kwargs["depth_dit_checkpoint_path"],
            expected_step=depth_parent_expected_step,
        )
    for key in (
        "depth_dit_config",
        "rgb_dit_config",
        "action_dit_config",
        "rgb_scheduler",
        "depth_scheduler",
        "action_scheduler",
        "loss",
    ):
        if OmegaConf.is_config(kwargs[key]):
            kwargs[key] = OmegaConf.to_container(kwargs[key], resolve=True)
    if not issubclass(_model_class, PrismWAMSharedDepthParent):
        raise TypeError("The Stage-2 model must start from a Stage-1 checkpoint.")
    model = _model_class.from_wan22_pretrained(
        device=device, torch_dtype=model_dtype, **kwargs
    )
    model.lr_schedule_steps = int(lr_schedule_steps)
    model.depth_parent_expected_step = int(depth_parent_expected_step)
    model.action_init_source = action_init_source
    model.stage1_provenance = None
    if checkpoint_only:
        return model
    if model.rgb_expert.blocks is not model.video_expert.blocks:
        raise RuntimeError("RGB/Depth blocks are not aliases of the same backbone.")
    model.stage1_provenance = {
        "parent_path": parent.path,
        "parent_step": parent.step,
        "parent_type": "depth_only",
        "shared_visual_blocks_source": "stage1_depth",
        "rgb_private_modules_source": "wan",
        "action_init_source": action_init_source,
        "wan_action_initializer": kwargs["action_dit_pretrained_path"],
        "action_init_policy": "linear_interp_alpha_scale",
        "action_private_modules_preserved": True,
        "lr_schedule_steps": int(lr_schedule_steps),
    }
    return model
