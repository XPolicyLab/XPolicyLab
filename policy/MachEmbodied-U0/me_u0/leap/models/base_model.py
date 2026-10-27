"""Training batch, model registry, and base model interfaces."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# batch dataclass
# ---------------------------------------------------------------------------


@dataclass
class LeapBatch:
    """Training batch.

    The continuous-action path uses chunk-axis tensors; one packed VLM sequence
    can carry several chunks, so ``C >= B``. In the non-packing case ``C == B``
    and ``action_chunk_vlm_ranges[c] == (0, L_c)``.

    Attributes:
        proprio_states: ``(C, S, D_s)``. Per-chunk proprioceptive history.
        action_chunks: ``(C, H, D_a)``. Per-chunk ground-truth actions.
        action_mask: ``(C, H, D_a)`` or ``(C, 1, D_a)``. Per-dim validity.
        action_chunks_valid: ``(C,)`` bool. Filters padded / invalid chunks.
        action_chunk_vlm_ranges: ``(C, 2)`` long. ``(start, end)`` slice into
            the VLM sequence dim that the continuous head's cross-attn may
            see for chunk ``c``. In Stage 2 this range excludes the FAST
            token positions, so the continuous head cannot peek at the
            discrete prediction.
        labels: ``(B, L)`` long. LM CE targets (Stage 1/2 only). ``-100``
            on positions that are not supervised.
        action_label_mask: ``(B, L)`` bool. Marks which label positions are
            FAST action tokens (for separated lang/act CE).

    Other VLM fields (``input_ids`` / ``pixel_values`` / ``attention_mask`` /
    ...) are passed through ``**extras`` rather than being typed here, since
    each VLM backbone has its own surface.
    """

    proprio_states: Optional[Tensor] = None
    action_chunks: Optional[Tensor] = None
    action_mask: Optional[Tensor] = None
    action_chunks_valid: Optional[Tensor] = None
    action_chunk_vlm_ranges: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    action_label_mask: Optional[Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSpec:
    """Self-description of which batch fields a model needs.

    Used by the data layer to route fields and by tests/diagnostics to
    catch missing-field bugs early.
    """

    required: Set[str] = field(default_factory=set)
    optional: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


MODEL_REGISTRY: Dict[str, Callable[..., "LeapVLABase"]] = {}


def register_model(name: str) -> Callable[[Callable[..., "LeapVLABase"]], Callable[..., "LeapVLABase"]]:
    """Decorator: register a builder under ``name``.

    """

    def _decorator(builder: Callable[..., "LeapVLABase"]) -> Callable[..., "LeapVLABase"]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model {name!r} already registered (existing builder: "
                f"{MODEL_REGISTRY[name].__module__}.{MODEL_REGISTRY[name].__name__})."
            )
        MODEL_REGISTRY[name] = builder
        return builder

    return _decorator


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LeapVLABase(PreTrainedModel):
    """Model base interface.

    Subclasses must:

    * implement :meth:`forward` returning a ``Dict[str, Tensor]`` of loss
      components (key ``"loss"`` is the scalar to backprop);
    * implement :meth:`predict_actions(*, mode, **batch)` returning a
      predicted-action tensor;
    * implement :attr:`batch_spec` so the data layer can validate inputs.
    """

    @property
    @abstractmethod
    def batch_spec(self) -> BatchSpec:
        """Required + optional batch field names for this model."""

    @abstractmethod
    def forward(self, **batch: Any) -> Dict[str, Tensor]:
        """Compute losses for one batch.

        Returns:
            Dict with at least ``"loss"`` key. Other keys (e.g. ``"lm_ce"``,
            ``"fm_mse"``) are component losses for logging.
        """

    @abstractmethod
    def predict_actions(self, *, mode: str = "auto", **batch: Any) -> Tensor:
        """Run inference. ``mode`` interpretation is implementation-defined."""

    @property
    def prompt_extras(self) -> List[str]:
        """Optional extra prompt-template fields. Default: none."""
        return []
