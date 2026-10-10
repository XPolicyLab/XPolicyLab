"""Translate RGB observations and mobile m92uw actions to the canonical layout.

State keys accept singular runtime and plural storage names. Actions return
one dict per step, with poses in [x, y, z, qw, qx, qy, qz] order."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mmabc.embodiments import mobile
from mmabc.eval.adapters.base import BenchmarkAdapter, register


def _leaf(name: str) -> str:
    return str(name).replace("/", ".").split(".")[-1]


@register("mobile")
class MobileAdapter(BenchmarkAdapter):
    view_map = {cam: slot for slot, cam in mobile.CAMERAS.items()}

    def __init__(
        self,
        *,
        allow_missing_state: bool = False,
        action_key_style: str = "singular",
        variant: str = "m75",
    ) -> None:
        self.allow_missing_state = allow_missing_state
        self.action_key_style = action_key_style
        if variant not in ("m75", "c80"):
            raise ValueError(f"variant must be m75 or c80, got {variant!r}")
        # m75: from-scratch Mobile layout; c80: pretrain-compatible canonical layout.
        self.variant = variant

    def to_canonical_state(self, raw_obs: dict) -> np.ndarray:
        state = raw_obs.get("state")
        if not isinstance(state, Mapping):
            state = raw_obs
        vec = mobile.pack(state, allow_missing=self.allow_missing_state)
        return mobile.to_c80(vec) if self.variant == "c80" else vec

    def to_canonical_images(self, raw_obs: dict) -> dict[str, np.ndarray]:
        vision = {_leaf(k): v for k, v in (raw_obs.get("vision") or {}).items()}
        out = {}
        for cam, slot in self.view_map.items():
            entry = None
            for name in (cam, *mobile.CAMERA_ALIASES.get(cam, ())):
                if name in vision:
                    entry = vision[name]
                    break
            if entry is None:
                raise KeyError(f"missing camera {cam!r}; observation has {sorted(vision)}")
            if isinstance(entry, Mapping):
                entry = next((entry[k] for k in ("color", "rgb", "image") if k in entry), None)
                if entry is None:
                    raise KeyError(f"camera {cam!r} carries no color field")
            img = np.asarray(entry)
            if img.ndim != 3 or img.shape[-1] not in (3, 4):
                raise ValueError(f"camera {cam!r}: expected HxWx3 image, got {img.shape}")
            img = img[..., :3]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            out[slot] = np.ascontiguousarray(img)
        return out

    def from_canonical_action(self, canonical: np.ndarray) -> list[dict[str, np.ndarray]]:
        """(T, 75) or (T, 80) absolute actions -> one per-key dict per control step."""
        canonical = np.asarray(canonical)
        if self.variant == "c80":
            canonical = mobile.from_c80(canonical)
        keyed = mobile.unpack(canonical, key_style=self.action_key_style)
        steps = np.asarray(canonical).shape[0]
        return [{k: v[t] for k, v in keyed.items()} for t in range(steps)]
