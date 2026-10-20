"""Tensor contracts internal to multi-teacher video distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoBatch:
    frames: torch.Tensor
    frame_positions: torch.Tensor
    patch_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class AdapterStates:
    local: tuple[torch.Tensor, ...]
    inter: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StudentOutput:
    tokens: torch.Tensor
    states: AdapterStates


@dataclass(frozen=True)
class VGGTTargets:
    caches: tuple[torch.Tensor | None, ...]
    depth: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class VGGTDiagnosticTargets(VGGTTargets):
    """Training targets plus camera output used only by offline diagnostics."""

    pose: torch.Tensor


@dataclass(frozen=True)
class VJEPATargets:
    tokens: torch.Tensor
