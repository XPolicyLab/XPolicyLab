"""Per-profile dataset and the weighted mixture that trains on all of them.

Sampling is step-based, not epoch-based. A weighted sampler draws a profile per
sample for as many steps as training runs, so a 680-hour corpus may be seen less
than once while a 6-hour one repeats many times. That is the intent: what
matters is the ratio each platform contributes to a gradient step, not that
every frame is visited.
"""

from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset, get_worker_info

from mmabc.canonical.layout import CanonicalLayout, load_layout
from mmabc.canonical.normalize import NormStats, Normalizer, identity_stats
from mmabc.canonical.transforms import EmbodimentSpec, TargetBuilder
from mmabc.data.lerobot_v3 import LeRobotV3Profile
from mmabc.data.prompt import PromptSpec, build_prompt
from mmabc.paths import HOME, home_path

CANONICAL_VIEWS = ("primary", "wrist_left", "wrist_right")

# Periodically collect PyAV reference cycles; MMABC_GC_EVERY=0 disables collection.
_GC_EVERY = int(os.environ.get("MMABC_GC_EVERY", "32"))

# Prefix of profile paths that MMABC_DATA_ROOT may redirect to node-local disk.
_DATA_SRC_PREFIX_DEFAULT = str(HOME / "data")


def remap_data_root(path: str) -> str:
    """Map a shared dataset to MMABC_DATA_ROOT when a local meta/ exists.

    MMABC_DATA_SRC_PREFIX selects the replaced prefix (default: HOME/data).
    Missing local copies keep their original path.
    """
    local_root = os.environ.get("MMABC_DATA_ROOT")
    if not local_root:
        return path
    src_prefix = os.environ.get("MMABC_DATA_SRC_PREFIX", _DATA_SRC_PREFIX_DEFAULT).rstrip("/")
    p = str(path)
    if not p.startswith(src_prefix + "/"):
        return path
    candidate = local_root.rstrip("/") + p[len(src_prefix):]
    if (Path(candidate) / "meta").is_dir():
        return candidate
    return path


@dataclass
class ProfileConfig:
    """One entry from configs/embodiments/*.yaml."""

    name: str
    path: str
    embodiment_tag: str
    fps: float
    action_type: str
    available_action_types: tuple[str, ...]
    switch_prob: float
    reference_frame: str
    views: dict[str, str | None]
    hours: float
    weight_multiplier: float

    @classmethod
    def load(cls, path: str | Path) -> ProfileConfig:
        cfg = OmegaConf.load(str(path))
        views = {v: cfg.views.get(v) for v in CANONICAL_VIEWS}
        return cls(
            name=str(cfg.name),
            path=remap_data_root(str(home_path(cfg.path))),
            embodiment_tag=str(cfg.embodiment_tag),
            fps=float(cfg.fps),
            action_type=str(cfg.action_type),
            available_action_types=tuple(str(a) for a in cfg.available_action_types),
            switch_prob=float(cfg.get("switch_prob", 0.0)),
            reference_frame=str(cfg.get("reference_frame", "base")),
            views={k: (None if v in (None, "null") else str(v)) for k, v in views.items()},
            hours=float(cfg.get("hours", 0.0)),
            weight_multiplier=float(cfg.get("weight_multiplier", 1.0)),
        )


class ProfileDataset:
    """Samples from one dataset profile, producing model-space targets."""

    def __init__(
        self,
        cfg: ProfileConfig,
        layout: CanonicalLayout,
        *,
        chunk_size: int,
        image_size: int,
        future_offsets: tuple[int, ...],
        norm_stats: NormStats | None = None,
        prompt_dropout: float = 0.0,
        prompt_header: bool = False,
    ) -> None:
        self.cfg = cfg
        self.layout = layout
        self.chunk_size = chunk_size
        self.image_size = image_size
        # One offset per future segment: the frame each segment's prediction
        # targets. Frame 0 is always the current observation.
        self.future_offsets = tuple(int(o) for o in future_offsets)
        self.prompt_dropout = prompt_dropout
        self.prompt_header = prompt_header

        self.profile = LeRobotV3Profile(cfg.path, cfg.views)
        embodiment = json.loads((Path(cfg.path) / "meta" / "embodiment.json").read_text())
        self.spec = EmbodimentSpec.from_meta(embodiment, fps=cfg.fps)
        self.builder = TargetBuilder(layout, self.spec, reference_frame=cfg.reference_frame)

        allowed = [
            a for a in cfg.available_action_types if a in self.builder.available_action_types
        ]
        if not allowed:
            raise ValueError(f"{cfg.name}: configured action types unsupported by its labels")
        self.action_types = allowed
        self.primary_action_type = (
            cfg.action_type if cfg.action_type in allowed else allowed[0]
        )

        stats = norm_stats or identity_stats(
            cfg.embodiment_tag, layout.total_dim, layout.state_dim
        )
        self.normalizer = Normalizer(layout, stats)

        # Sampling an anchor uniformly over frames (not episodes) keeps long
        # episodes from being under-represented relative to their content.
        self.episode_cumsum = np.cumsum(self.profile.length)
        self.total_frames = int(self.episode_cumsum[-1]) if len(self.episode_cumsum) else 0

    def pick_action_type(self, rng: random.Random) -> str:
        if len(self.action_types) < 2 or self.cfg.switch_prob <= 0.0:
            return self.primary_action_type
        if rng.random() < self.cfg.switch_prob:
            others = [a for a in self.action_types if a != self.primary_action_type]
            return rng.choice(others)
        return self.primary_action_type

    def sample(self, rng: random.Random) -> dict:
        global_frame = rng.randrange(self.total_frames)
        ep = int(np.searchsorted(self.episode_cumsum, global_frame, side="right"))
        ep_start = int(self.episode_cumsum[ep - 1]) if ep > 0 else 0
        frame = global_frame - ep_start

        action_type = self.pick_action_type(rng)
        state, actions, valid = self.profile.sample_window(ep, frame, self.chunk_size)
        target, mask = self.builder.build(
            state, actions, action_type=action_type, valid_steps=valid
        )

        target = self.normalizer.normalize_action(target).astype(np.float32)
        target = np.where(mask, target, 0.0).astype(np.float32)
        bad = ~np.isfinite(target)
        if bad.any():
            target = np.where(bad, 0.0, target).astype(np.float32)
            mask = np.where(bad, 0.0, mask)
        state_norm = self.normalizer.normalize_state(state).astype(np.float32)
        state_norm = np.nan_to_num(state_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        n = int(self.profile.length[ep])
        frames = [frame] + [frame + o for o in self.future_offsets]
        # Never seek past the last real frame. The teacher loss is gated by
        # `future_valid` so a clamped (i.e. repeated last) frame is not a target.
        decode_frames = [int(min(f, max(n - 1, 0))) for f in frames]
        future_valid = np.array(
            [1.0 if (frame + o) < n else 0.0 for o in self.future_offsets],
            dtype=np.float32,
        )
        images, present = self.profile.read_views(ep, decode_frames, self.image_size)

        view_stack = np.zeros(
            (len(CANONICAL_VIEWS), len(frames), self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        view_mask = np.zeros(len(CANONICAL_VIEWS), dtype=bool)
        for i, slot in enumerate(CANONICAL_VIEWS):
            if present.get(slot):
                view_stack[i] = images[slot]
                view_mask[i] = True

        prompt = build_prompt(
            PromptSpec(
                embodiment=self.cfg.embodiment_tag,
                fps=self.cfg.fps,
                action_type=action_type,
                reference_frame=self.cfg.reference_frame,
                instruction=self.profile.task(ep),
            ),
            dropout=self.prompt_dropout,
            rng=rng,
            header=self.prompt_header,
        )

        aux = self.layout.head("aux")
        return {
            "state": state_norm,
            "state_mask": self.spec.state_valid.astype(np.float32),
            "target": target,
            "target_mask": mask.astype(np.float32),
            # Per-dim scale to map a normalised action back to real recorded-
            # command units, for a parametrisation-invariant decoded-action metric.
            "action_scale": self.normalizer.action_scale.astype(np.float32),
            "aux_active": np.float32(mask[:, aux.slice].any()),
            "images": view_stack,
            "future_valid": future_valid,
            "view_mask": view_mask.astype(np.float32),
            "prompt": prompt,
            "profile": self.cfg.name,
            "embodiment_tag": self.cfg.embodiment_tag,
            "action_type": action_type,
        }


class MixtureDataset(IterableDataset):
    """Infinite weighted stream over every profile in a mixture."""

    def __init__(
        self,
        mixture_path: str | Path,
        *,
        chunk_size: int,
        image_size: int,
        future_offsets: tuple[int, ...],
        prompt_dropout: float = 0.15,
        prompt_header: bool = False,
        norm_stats_dir: str | Path | None = None,
        seed: int = 0,
        repo_root: str | Path | None = None,
        max_retries: int = 8,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(str(mixture_path))
        self.layout = load_layout(str(self._resolve(cfg.canonical)))
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.future_offsets = tuple(int(o) for o in future_offsets)
        self.prompt_dropout = prompt_dropout
        self.prompt_header = prompt_header
        self.seed = seed
        self.max_retries = max_retries
        self.norm_stats_dir = Path(norm_stats_dir) if norm_stats_dir else None

        self.profile_configs: list[ProfileConfig] = []
        weights: list[float] = []
        for entry in cfg.datasets:
            pc = ProfileConfig.load(self._resolve(entry.config))
            self.profile_configs.append(pc)
            weights.append(float(entry.weight) * pc.weight_multiplier)

        total = sum(weights)
        if total <= 0:
            raise ValueError("mixture weights sum to zero")
        self.weights = np.asarray([w / total for w in weights], dtype=np.float64)
        self._datasets: list[ProfileDataset] | None = None
        self.skipped = 0

    def _resolve(self, rel: str) -> Path:
        p = Path(str(rel))
        return p if p.is_absolute() else self.repo_root / p

    def _norm_stats(self, tag: str) -> NormStats | None:
        if self.norm_stats_dir is None:
            return None
        path = self.norm_stats_dir / f"{tag}.json"
        return NormStats.load(path) if path.exists() else None

    def _ensure_built(self) -> list[ProfileDataset]:
        """Open profiles lazily, inside the worker that will use them.

        Episode indices and parquet handles do not survive fork cleanly and
        would otherwise be duplicated into every worker by the parent.
        """
        if self._datasets is None:
            built = []
            for pc in self.profile_configs:
                ds = ProfileDataset(
                    pc,
                    self.layout,
                    chunk_size=self.chunk_size,
                    image_size=self.image_size,
                    future_offsets=self.future_offsets,
                    norm_stats=self._norm_stats(pc.embodiment_tag),
                    prompt_dropout=self.prompt_dropout,
                    prompt_header=self.prompt_header,
                )
                built.append(ds)
            self._datasets = built
        return self._datasets

    def describe(self) -> str:
        lines = ["mixture:"]
        for pc, w in zip(self.profile_configs, self.weights):
            lines.append(
                f"  {w:7.4f}  {pc.name}  tag={pc.embodiment_tag} fps={pc.fps:g} "
                f"action={pc.action_type} views={sum(v is not None for v in pc.views.values())}"
            )
        return "\n".join(lines)

    def __iter__(self):
        datasets = self._ensure_built()
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0

        # Distinct stream per (rank, worker) so no two producers repeat samples.
        stream_id = rank * num_workers + worker_id
        rng = random.Random(self.seed * 1_000_003 + stream_id)
        nprng = np.random.default_rng(self.seed * 7_919 + stream_id)

        since_gc = 0
        while True:
            idx = int(nprng.choice(len(datasets), p=self.weights))
            for _ in range(self.max_retries):
                try:
                    yield datasets[idx].sample(rng)
                    break
                except Exception:
                    # Resample unreadable shards and count failures.
                    self.skipped += 1
                    idx = int(nprng.choice(len(datasets), p=self.weights))
            # Reclaim PyAV's decoded-frame reference cycles (see _GC_EVERY);
            # otherwise the worker RSS climbs every sample until the node OOMs.
            since_gc += 1
            if _GC_EVERY > 0 and since_gc >= _GC_EVERY:
                gc.collect()
                since_gc = 0
