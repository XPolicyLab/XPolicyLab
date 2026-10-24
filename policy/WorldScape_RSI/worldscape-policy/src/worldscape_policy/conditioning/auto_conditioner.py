from __future__ import annotations

import torch
from torch import nn

from worldscape_policy.conditioning.semantic_forcing import (
    LEGACY_SEMANTIC_TARGET_BUILDER,
    SemanticTargetBuilder,
)
from worldscape_policy.conditioning.vlm.protocol import AutoPlanningFeatures
from worldscape_policy.memory.event.memory import EventMemoryManager
from worldscape_policy.memory.visual.normalization import VisualInputRange
from worldscape_policy.types import (
    Conditioning,
    EventMemoryState,
    ObservationBatch,
    PromptBatch,
)


class AutoConditioner(nn.Module):
    """Build WAM cross-attention tokens from shared VLM planning features."""

    def __init__(
        self,
        *,
        vlm: nn.Module,
        token_pooler: nn.Module,
        projector: nn.Module,
        event_memory: nn.Module,
        output_norm: nn.Module | None = None,
        max_history_steps: int = 16,
        detach_inference_history: bool = True,
        visual_input_range: VisualInputRange = "zero_one",
        semantic_target_builder: SemanticTargetBuilder = (
            LEGACY_SEMANTIC_TARGET_BUILDER
        ),
        semantic_gate_only: bool = False,
        semantic_grad_clip_norm: float = 0.5,
    ) -> None:
        super().__init__()
        if max_history_steps <= 0:
            raise ValueError("max_history_steps must be positive")
        if not hasattr(vlm, "encode_planning"):
            raise TypeError("vlm must implement encode_planning(...)")
        self.vlm = vlm
        self.token_pooler = token_pooler
        self.projector = projector
        self.event_memory = event_memory
        self.output_norm = output_norm or nn.Identity()
        self.max_history_steps = int(max_history_steps)
        self.detach_inference_history = bool(detach_inference_history)
        self.visual_input_range = visual_input_range
        self.semantic_target_builder = semantic_target_builder
        self.semantic_gate_only = bool(semantic_gate_only)
        self.semantic_grad_clip_norm = float(semantic_grad_clip_norm)
        if self.semantic_grad_clip_norm < 0:
            raise ValueError("semantic_grad_clip_norm must be non-negative")
        self._event_memory_manager = EventMemoryManager(
            max_history_steps=self.max_history_steps,
            detach_inference_history=self.detach_inference_history,
        )

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        """Match VLM features to the projector precision under DeepSpeed."""
        parameter = next(self.projector.parameters(), None)
        if parameter is not None and tokens.is_floating_point():
            tokens = tokens.to(device=parameter.device, dtype=parameter.dtype)
        return self.projector(tokens)

    @staticmethod
    def _require_matching_token_widths(
        perception: torch.Tensor,
        planning: torch.Tensor | None,
    ) -> None:
        """Require perception and AR planning tokens to share one VLM space."""
        if planning is not None and perception.shape[-1] != planning.shape[-1]:
            raise ValueError(
                "VLM perception and autoregressive planning token widths differ: "
                f"{perception.shape[-1]} != {planning.shape[-1]}. Configure "
                "qformer_output_dim to match the VLM hidden/token dimension before "
                "the shared WAM condition projector."
            )

    def forward(
        self,
        *,
        observation: ObservationBatch,
        prompts: PromptBatch,
        event_memory: EventMemoryState | None,
        training: bool,
        planning_supervision: bool = False,
    ) -> Conditioning:
        planning_text = prompts.vlm_planning_text
        if planning_text is None:
            raise ValueError("AutoConditioner requires vlm_planning_text")
        encode_kwargs = {
            "images": observation.head_view,
            "planning_text": planning_text,
            "negative_text": prompts.negative_vlm_text,
            "training": training,
            "visual_input_range": self.visual_input_range,
            "planning_supervision": planning_supervision,
        }
        if observation.vlm_history_images is not None:
            encode_kwargs["history_images"] = observation.vlm_history_images
            encode_kwargs["history_mask"] = observation.vlm_history_mask
        if prompts.planning_labels_text is not None:
            encode_kwargs["planning_labels_text"] = prompts.planning_labels_text
        features = self.vlm.encode_planning(**encode_kwargs)
        if not isinstance(features, AutoPlanningFeatures):
            raise TypeError(
                "encode_planning must return AutoPlanningFeatures, "
                f"got {type(features).__name__}"
            )
        if features.perception_features.ndim != 3:
            raise ValueError("perception_features must have shape [B, L, D]")
        if features.perception_features.shape[0] != observation.images.shape[0]:
            raise ValueError("VLM perception batch size does not match observations")
        if features.planning_features is not None:
            if features.planning_features.ndim != 3:
                raise ValueError("planning_features must have shape [B, P, D]")
            if features.planning_features.shape[0] != observation.images.shape[0]:
                raise ValueError("VLM planning batch size does not match observations")

        perception = self.token_pooler(features.perception_features)
        planning = features.planning_features
        self._require_matching_token_widths(perception, planning)
        current_tokens = (
            torch.cat([perception, planning], dim=1)
            if planning is not None
            else perception
        )
        projected = self._project(current_tokens)
        projected_perception = self._project(perception)
        projected_planning = self._project(planning) if planning is not None else None
        projected_task = (
            self._project(features.task_embedding)
            if features.task_embedding is not None
            else None
        )
        projected_history = (
            self._project(features.history_perception_features)
            if features.history_perception_features is not None
            else None
        )
        projected_history_planning = (
            self._project(features.history_planning_features)
            if features.history_planning_features is not None
            else None
        )
        negative_tokens = features.negative_perception_features
        if negative_tokens is not None:
            negative_tokens = self.token_pooler(negative_tokens)
            if features.negative_planning_features is not None:
                negative_tokens = torch.cat(
                    [negative_tokens, features.negative_planning_features],
                    dim=1,
                )
            projected_negative = self._project(negative_tokens)
        else:
            projected_negative = projected

        prompt_signature = tuple(planning_text) + tuple(
            prompts.negative_vlm_text or ()
        )
        previous = self._event_memory_manager.begin(
            event_memory,
            prompt_signature=prompt_signature,
            has_planning=planning is not None,
            training=training,
        )
        history_tokens = (
            projected_history
            if projected_history is not None
            else previous.perception_tokens
        )
        history_planning = (
            projected_history_planning
            if projected_history_planning is not None
            else previous.planning_tokens
        )
        history_mask = (
            features.history_mask
            if projected_history is not None
            else previous.valid_mask
        )
        fused, _ = self.event_memory(
            projected,
            history_tokens=history_tokens,
            history_planning_tokens=history_planning,
            history_mask=history_mask,
            task_embeddings=projected_task,
        )
        next_memory = self._event_memory_manager.stage(
            previous,
            projected_perception,
            projected_planning,
            training=training,
        )
        positive_condition = self.output_norm(fused)
        semantic_prediction = positive_condition
        if training and self.semantic_gate_only:
            detached_perception = perception.detach()
            detached_planning = planning.detach() if planning is not None else None
            detached_current = (
                torch.cat([detached_perception, detached_planning], dim=1)
                if detached_planning is not None
                else detached_perception
            )
            semantic_tokens = self._project(detached_current)
            semantic_history = (
                self._project(features.history_perception_features.detach())
                if features.history_perception_features is not None
                else None
            )
            semantic_history_planning = (
                self._project(features.history_planning_features.detach())
                if features.history_planning_features is not None
                else None
            )
            semantic_prediction, _ = self.event_memory(
                semantic_tokens,
                history_tokens=semantic_history,
                history_planning_tokens=semantic_history_planning,
                history_mask=features.history_mask,
                task_embeddings=(
                    self._project(features.task_embedding.detach())
                    if features.task_embedding is not None
                    else None
                ),
            )
            semantic_prediction = self.output_norm(semantic_prediction)
        negative_condition = self.output_norm(projected_negative)
        if not training:
            positive_condition, negative_condition = (
                self._event_memory_manager.cached_conditions(
                    previous,
                    positive_condition,
                    negative_condition,
                )
            )
            next_memory = self._event_memory_manager.cache_conditions(
                next_memory,
                positive=positive_condition,
                negative=negative_condition,
                prompt_signature=prompt_signature,
            )
        return Conditioning(
            cross_attention_tokens=positive_condition,
            negative_cross_attention_tokens=negative_condition,
            event_memory=next_memory,
            semantic_prediction=semantic_prediction if training else None,
            semantic_target=self.semantic_target_builder(
                training=training,
                explicit_target=features.semantic_target,
                # Leave the fallback empty so WorldScapePolicy can build the
                # teacher target from the scheduled T5 subtask text.
                fallback_target=None,
            ),
            planning_logits=features.planning_logits if training else None,
            planning_labels=features.planning_labels if training else None,
        )

    def promote_pending(
        self, previous: EventMemoryState | None
    ) -> EventMemoryState | None:
        return self._event_memory_manager.promote_pending(previous)
