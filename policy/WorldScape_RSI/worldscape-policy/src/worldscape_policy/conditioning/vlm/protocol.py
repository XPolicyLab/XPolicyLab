"""Public contracts for WorldScape-owned planning VLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from worldscape_policy.memory.visual.normalization import VisualInputRange


@dataclass
class AutoPlanningFeatures:
    """Outputs from one shared VLM perception/planning pass."""

    perception_features: Tensor
    planning_features: Tensor | None = None
    negative_perception_features: Tensor | None = None
    negative_planning_features: Tensor | None = None
    task_embedding: Tensor | None = None
    semantic_target: Tensor | None = None
    planning_logits: Tensor | None = None
    planning_labels: Tensor | None = None
    history_perception_features: Tensor | None = None
    history_planning_features: Tensor | None = None
    history_mask: Tensor | None = None


class PlanningVLM(Protocol):
    def encode_planning(
        self,
        *,
        images: Tensor,
        planning_text: list[str],
        negative_text: list[str] | None,
        training: bool,
        planning_labels_text: list[str | None] | None = None,
        visual_input_range: VisualInputRange = "zero_one",
        planning_supervision: bool = False,
        history_images: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> AutoPlanningFeatures: ...


__all__ = ["AutoPlanningFeatures", "PlanningVLM"]
