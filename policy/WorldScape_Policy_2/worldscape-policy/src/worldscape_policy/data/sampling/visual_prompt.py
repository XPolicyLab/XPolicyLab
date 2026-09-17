from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from worldscape_policy.data.schema import EventSample, VisualPromptMetadata
from worldscape_policy.types import InteractionMode


class PromptModality(str, Enum):
    TEXT = "text"
    GOAL = "goal"
    DEMO = "demo"
    TEXT_GOAL = "text+goal"
    TEXT_DEMO = "text+demo"

    @classmethod
    def parse(cls, value: PromptModality | str) -> PromptModality:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as exc:
            choices = ", ".join(modality.value for modality in cls)
            raise ValueError(
                f"unknown prompt modality {value!r}; expected one of: {choices}"
            ) from exc


@dataclass(frozen=True)
class VisualPrompt:
    goal_image: np.ndarray | None = None
    demo_video: np.ndarray | None = None
    goal_metadata: VisualPromptMetadata | None = None
    demo_metadata: VisualPromptMetadata | None = None


@dataclass(frozen=True)
class AuditedVisualPromptOverride:
    """Explicit authorization for otherwise incompatible visual prompts."""

    enabled: bool = False
    audit_reason: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and (
            not isinstance(self.audit_reason, str) or not self.audit_reason.strip()
        ):
            raise ValueError(
                "enabled visual prompt override requires a non-empty audit_reason"
            )


@dataclass(frozen=True)
class VisualPromptSampler:
    modality: str = "none"
    override: AuditedVisualPromptOverride | None = None

    def sample(
        self,
        sample: EventSample,
        *,
        rng: np.random.Generator | None = None,
    ) -> VisualPrompt:
        sample.validate()
        modality = self.modality.lower()
        if modality not in {"goal", "demo", "none"}:
            raise ValueError("modality must be goal, demo, or none")
        if modality == "none":
            return VisualPrompt()
        if modality == "goal":
            if sample.goal_image is None:
                raise ValueError("sample does not contain a goal image")
            return VisualPrompt(
                goal_image=sample.goal_image.copy(),
                goal_metadata=self._validate_compatibility(
                    sample, "goal", sample.goal_prompt_metadata
                ),
            )
        if sample.demo_video is None:
            raise ValueError("sample does not contain a demo video")
        return VisualPrompt(
            demo_video=sample.demo_video.copy(),
            demo_metadata=self._validate_compatibility(
                sample, "demo", sample.demo_prompt_metadata
            ),
        )

    def _validate_compatibility(
        self,
        sample: EventSample,
        prompt_name: str,
        metadata: VisualPromptMetadata | None,
    ) -> VisualPromptMetadata:
        if metadata is None:
            raise ValueError(f"{prompt_name} prompt is missing provenance metadata")
        mismatches = []
        if metadata.embodiment != sample.embodiment:
            mismatches.append(
                f"embodiment {metadata.embodiment!r} != {sample.embodiment!r}"
            )
        if metadata.task_id != sample.task_id:
            mismatches.append(f"task {metadata.task_id!r} != {sample.task_id!r}")
        if not mismatches:
            return metadata
        if self.override is not None and self.override.enabled:
            return replace(
                metadata,
                override_audit_reason=self.override.audit_reason,
            )
        raise ValueError(
            f"incompatible {prompt_name} visual prompt: {', '.join(mismatches)}; "
            "an enabled AuditedVisualPromptOverride with audit_reason is required"
        )


@dataclass(frozen=True)
class ModeSampler:
    auto_probability: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.auto_probability <= 1:
            raise ValueError("auto_probability must be in [0, 1]")

    def sample(self, *, rng: np.random.Generator | None = None) -> InteractionMode:
        generator = rng or np.random.default_rng()
        if float(generator.random()) < self.auto_probability:
            return InteractionMode.AUTO
        return InteractionMode.INTERACTIVE


@dataclass(frozen=True)
class PromptModalitySampler:
    modalities: tuple[PromptModality, ...] = (PromptModality.TEXT,)
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        parsed = tuple(PromptModality.parse(value) for value in self.modalities)
        object.__setattr__(self, "modalities", parsed)
        if not parsed:
            raise ValueError("modalities cannot be empty")
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.modalities):
                raise ValueError("probabilities must match modalities")
            if any(probability < 0 for probability in self.probabilities):
                raise ValueError("probabilities cannot be negative")
            if not np.isclose(sum(self.probabilities), 1.0):
                raise ValueError("probabilities must sum to one")

    def sample(self, *, rng: np.random.Generator | None = None) -> PromptModality:
        generator = rng or np.random.default_rng()
        index = int(generator.choice(len(self.modalities), p=self.probabilities))
        return self.modalities[index]


__all__ = [
    "AuditedVisualPromptOverride",
    "ModeSampler",
    "PromptModality",
    "PromptModalitySampler",
    "VisualPrompt",
    "VisualPromptSampler",
]
