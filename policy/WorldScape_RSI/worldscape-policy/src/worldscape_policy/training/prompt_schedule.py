"""Deterministic prompt-mode schedules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Stage:
    """A schedule interval ending at ``end`` training progress."""

    end: float
    auto_ratio: float
    name: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.end <= 1:
            raise ValueError("stage end must be in (0, 1]")
        if not 0 <= self.auto_ratio <= 1:
            raise ValueError("auto_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PromptScheduleResult:
    """Selected stage and per-example Auto/Interactive mode mask."""

    mode_mask: Tensor
    stage_index: int
    stage: Stage
    progress: float

    @property
    def auto_count(self) -> int:
        return int(self.mode_mask.sum().item())

    @property
    def interactive_count(self) -> int:
        return self.mode_mask.numel() - self.auto_count


class PromptSchedule:
    """Sample prompt modes per example or once for the complete batch.

    ``mode_mask`` is boolean and uses ``True`` for Auto (VLM) and ``False`` for
    Interactive (T5). Supplying a ``torch.Generator`` makes selection exactly
    reproducible across save/resume boundaries when its state is checkpointed.
    """

    def __init__(self, stages: list[Stage] | tuple[Stage, ...]) -> None:
        if not stages:
            raise ValueError("PromptSchedule requires at least one stage")
        self.stages = tuple(stages)
        previous = 0.0
        for stage in self.stages:
            if stage.end <= previous:
                raise ValueError("stage ends must be strictly increasing")
            previous = stage.end
        if self.stages[-1].end != 1.0:
            raise ValueError("the final stage must end at 1.0")

    def stage_at(self, progress: float) -> tuple[int, Stage]:
        if not 0 <= progress <= 1:
            raise ValueError("progress must be in [0, 1]")
        for index, stage in enumerate(self.stages):
            if progress < stage.end or index == len(self.stages) - 1:
                return index, stage
        raise AssertionError("validated schedule must contain progress")

    def sample(
        self,
        batch_size: int,
        progress: float,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        per_sample: bool = True,
    ) -> PromptScheduleResult:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        index, stage = self.stage_at(progress)
        draws = batch_size if per_sample else min(batch_size, 1)
        mask = torch.rand(
            draws,
            generator=generator,
            device=device,
        ) < stage.auto_ratio
        if not per_sample and batch_size > 1:
            mask = mask.expand(batch_size).clone()
        return PromptScheduleResult(
            mode_mask=mask,
            stage_index=index,
            stage=stage,
            progress=float(progress),
        )

    __call__ = sample

    @classmethod
    def plan_default(cls) -> PromptSchedule:
        """Construct the schedule specified in the public refactor plan."""

        return cls(
            [
                Stage(end=0.05, auto_ratio=0.5, name="stage_a"),
                Stage(end=0.30, auto_ratio=0.5, name="stage_b"),
                Stage(end=1.00, auto_ratio=0.7, name="stage_c"),
            ]
        )

    @classmethod
    def legacy_dual_prompt(
        cls,
        *,
        warmup_end: float,
        stage_b_end: float,
    ) -> PromptSchedule:
        """Match the legacy T5/VLM stage ratios.

        Legacy stages used T5 ratios 0.7, 0.5, and 0.3, corresponding to Auto
        ratios 0.3, 0.5, and 0.7.
        """

        return cls(
            [
                Stage(end=warmup_end, auto_ratio=0.3, name="stage_a"),
                Stage(end=stage_b_end, auto_ratio=0.5, name="stage_b"),
                Stage(end=1.0, auto_ratio=0.7, name="stage_c"),
            ]
        )


__all__ = ["PromptSchedule", "PromptScheduleResult", "Stage"]
