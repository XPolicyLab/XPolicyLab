"""Declarative state/action layouts and supervised target dimensions.

Rotation storage uses six dimensions; rotation increments supervise three."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Kinds whose target is an increment against the anchor frame's state.
DELTA_KINDS = frozenset({"joint", "eef_pos", "eef_rot", "base_pose"})
# Kinds that are copied through as absolute values. ``cmd`` is a raw recorded
# command: pass-through target, no increment.
ABSOLUTE_KINDS = frozenset({"gripper", "base_twist", "discrete", "hand", "cmd"})


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    end: int
    kind: str
    group: str
    norm: str
    target_dim: int | None = None

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def target_width(self) -> int:
        """How many of this segment's dims carry a supervised target."""
        return self.width if self.target_dim is None else self.target_dim

    @property
    def target_slice(self) -> slice:
        return slice(self.start, self.start + self.target_width)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def is_delta(self) -> bool:
        return self.kind in DELTA_KINDS


@dataclass(frozen=True)
class Head:
    name: str
    start: int
    end: int
    groups: tuple[str, ...]

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


class CanonicalLayout:
    """Index bookkeeping for one canonical layout definition."""

    def __init__(self, cfg) -> None:
        self.total_dim: int = int(cfg.total_dim)
        self.segments: tuple[Segment, ...] = tuple(
            Segment(
                name=str(s.name),
                start=int(s.range[0]),
                end=int(s.range[1]),
                kind=str(s.kind),
                group=str(s.group),
                norm=str(s.norm),
                target_dim=None if s.get("target_dim") is None else int(s.target_dim),
            )
            for s in cfg.segments
        )
        self.heads: tuple[Head, ...] = tuple(
            Head(
                name=str(name),
                start=int(h.range[0]),
                end=int(h.range[1]),
                groups=tuple(str(g) for g in h.groups),
            )
            for name, h in cfg.heads.items()
        )
        # A separate state layout permits proprioception and action vectors to differ.
        state_cfg = cfg.get("state") if hasattr(cfg, "get") else None
        if state_cfg is not None:
            self.state_dim: int = int(state_cfg.total_dim)
            self.state_segments: tuple[Segment, ...] = tuple(
                Segment(
                    name=str(s.name),
                    start=int(s.range[0]),
                    end=int(s.range[1]),
                    kind=str(s.get("kind", "state")),
                    group=str(s.get("group", "state")),
                    norm=str(s.norm),
                    target_dim=None,
                )
                for s in state_cfg.segments
            )
        else:
            self.state_dim = self.total_dim
            self.state_segments = self.segments
        self.action_types: dict[str, dict[str, tuple[str, ...]]] = {
            str(name): {
                "keep_groups": tuple(str(g) for g in spec.get("keep_groups", [])),
                "drop_groups": tuple(str(g) for g in spec.get("drop_groups", [])),
            }
            for name, spec in cfg.action_types.items()
        }
        self.reference_frames: tuple[str, ...] = tuple(str(f) for f in cfg.reference_frames)
        self._validate()

    def _validate(self) -> None:
        covered = np.zeros(self.total_dim, dtype=bool)
        for seg in self.segments:
            if seg.end > self.total_dim or seg.start < 0 or seg.start >= seg.end:
                raise ValueError(f"segment {seg.name} has out-of-range span [{seg.start},{seg.end})")
            if covered[seg.slice].any():
                raise ValueError(f"segment {seg.name} overlaps an earlier segment")
            covered[seg.slice] = True
            if seg.target_dim is not None and seg.target_dim > seg.width:
                raise ValueError(f"segment {seg.name} target_dim exceeds its width")
        if not covered.all():
            missing = np.flatnonzero(~covered).tolist()
            raise ValueError(f"canonical layout leaves dims uncovered: {missing}")

        head_cover = np.zeros(self.total_dim, dtype=bool)
        for head in self.heads:
            if head_cover[head.slice].any():
                raise ValueError(f"head {head.name} overlaps another head")
            head_cover[head.slice] = True
        if not head_cover.all():
            raise ValueError("heads must jointly cover the canonical vector")

        # A decoupled state layout must tile its own width exactly.
        if self.state_segments is not self.segments:
            scov = np.zeros(self.state_dim, dtype=bool)
            for seg in self.state_segments:
                if seg.end > self.state_dim or seg.start < 0 or seg.start >= seg.end:
                    raise ValueError(
                        f"state segment {seg.name} out-of-range [{seg.start},{seg.end})"
                    )
                if scov[seg.start : seg.end].any():
                    raise ValueError(f"state segment {seg.name} overlaps an earlier segment")
                scov[seg.start : seg.end] = True
            if not scov.all():
                raise ValueError(
                    f"state layout leaves dims uncovered: {np.flatnonzero(~scov).tolist()}"
                )

        # Every segment must fall entirely inside one head, otherwise a head
        # would own a fraction of a semantic quantity.
        for seg in self.segments:
            owner = [h for h in self.heads if h.start <= seg.start and seg.end <= h.end]
            if len(owner) != 1:
                raise ValueError(f"segment {seg.name} is not contained in exactly one head")
            if seg.group not in owner[0].groups:
                raise ValueError(
                    f"segment {seg.name} (group {seg.group}) sits in head "
                    f"{owner[0].name} which does not declare that group"
                )

    # lookups
    def segment(self, name: str) -> Segment:
        for seg in self.segments:
            if seg.name == name:
                return seg
        raise KeyError(name)

    def segments_of_kind(self, kind: str) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.kind == kind)

    def segments_of_group(self, group: str) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.group == group)

    def head(self, name: str) -> Head:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(name)

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(h.name for h in self.heads)

    # masks
    @lru_cache(maxsize=1)
    def target_support(self) -> np.ndarray:
        """Dims that can ever carry a target, before per-embodiment validity.

        Excludes the trailing dims of rotation segments (absolute 6D collapses
        to a 3-dim increment) and anything in a ``reserved`` segment.
        """
        support = np.zeros(self.total_dim, dtype=bool)
        for seg in self.segments:
            if seg.kind == "reserved":
                continue
            support[seg.target_slice] = True
        return support

    def action_type_mask(self, action_type: str) -> np.ndarray:
        """Dims kept when supervising under one control convention."""
        if action_type not in self.action_types:
            raise KeyError(f"unknown action type {action_type!r}")
        drop = set(self.action_types[action_type]["drop_groups"])
        keep = np.ones(self.total_dim, dtype=bool)
        for seg in self.segments:
            if seg.group in drop:
                keep[seg.slice] = False
        return keep

    def available_action_types(self, action_valid: np.ndarray) -> list[str]:
        """Which conventions this embodiment actually has labels for."""
        out = []
        for name, spec in self.action_types.items():
            needed = np.zeros(self.total_dim, dtype=bool)
            for group in spec["keep_groups"]:
                for seg in self.segments_of_group(group):
                    needed[seg.target_slice] = True
            if bool((action_valid & needed).any()):
                out.append(name)
        return out

    def head_slices(self) -> dict[str, slice]:
        return {h.name: h.slice for h in self.heads}

    def group_mask(self, groups: tuple[str, ...] | list[str]) -> np.ndarray:
        mask = np.zeros(self.total_dim, dtype=bool)
        wanted = set(groups)
        for seg in self.segments:
            if seg.group in wanted:
                mask[seg.slice] = True
        return mask


@lru_cache(maxsize=8)
def load_layout(path: str | Path) -> CanonicalLayout:
    cfg = OmegaConf.load(str(path))
    return CanonicalLayout(cfg)
