"""Model-only serving runtime for stateful VAP action rollout.

This module deliberately has no XPolicyLab or RoboDojo dependency.  It owns
the heavyweight, model-specific part of serving:

* load the typed VAP config, x2robot_v1/v2 normalizer and strict checkpoint;
* encode one RGB observation or one continuous ``I + V`` sequence;
* sample ``I -> A0`` and ``I + A0 + V0 -> A1``;
* convert normalized model output back to physical 20-D action values.

RoboDojo observation/action dictionaries and episode state live in the
integration adapter under ``integrations/robodojo``.
"""

from __future__ import annotations

import gc
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _resolve_checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAP checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_normalizer_key(normalizer: Any, requested: str | None) -> str:
    keys = list(getattr(normalizer, "delta", {}).keys())
    if not keys:
        raise KeyError("VAP action normalizer has no dataset keys")
    if requested is not None:
        if requested not in keys:
            raise KeyError(
                f"normalizer key {requested!r} is unavailable; available={keys}"
            )
        return requested
    if len(keys) != 1:
        raise KeyError(
            "VAP serving requires an explicit normalizer key when statistics "
            f"contain multiple entries; available={keys}"
        )
    return keys[0]


def _validate_v1_action_stats(
    raw_stats: dict[str, Any],
    *,
    normalizer_key: str,
    dof_config: dict[str, int],
    min_key: str,
    delta_key: str,
) -> None:
    """Reject missing/defaulted x2robot_v1 action statistics.

    The legacy ``Normalizer`` can replace missing DOFs with identity values.
    That behavior is useful for some training mixtures but unsafe for serving:
    a policy could emit numerically valid actions in the wrong physical scale.
    """

    entry = raw_stats.get(normalizer_key)
    if not isinstance(entry, dict):
        raise KeyError(f"x2robot_v1 stats have no mapping for {normalizer_key!r}")
    from wall_x.utils.constant import _ACTION_KEY_FULL_MAPPING

    for model_key, dimension in dof_config.items():
        candidates = [model_key]
        follow_key = model_key.replace("master_", "follow_", 1)
        if follow_key not in candidates:
            candidates.append(follow_key)
        raw_key = _ACTION_KEY_FULL_MAPPING.get(model_key)
        if raw_key and raw_key not in candidates:
            candidates.append(raw_key)
        selected = next((key for key in candidates if key in entry), None)
        if selected is None:
            raise KeyError(
                f"x2robot_v1 stats {normalizer_key!r} are missing action "
                f"{model_key!r}; tried {candidates}"
            )
        stat = entry[selected]
        if not isinstance(stat, dict):
            raise TypeError(
                f"stats[{normalizer_key!r}][{selected!r}] must be a mapping"
            )
        for field in (min_key, delta_key):
            values = stat.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != int(dimension):
                width = len(values) if isinstance(values, (list, tuple)) else None
                raise ValueError(
                    f"stats[{normalizer_key!r}][{selected!r}][{field!r}] "
                    f"width {width} != configured {dimension}"
                )


def _build_action_normalizer(cfg: Any, requested_key: str | None) -> tuple[Any, str]:
    """Build the training-isomorphic action normalizer for v1 or v2 data."""

    norm_cfg = cfg.data.normalizer_config or {}
    stats_file = norm_cfg.get("customized_action_statistic_dof")
    if not stats_file:
        raise ValueError(
            "data.normalizer_config.customized_action_statistic_dof is required"
        )
    stats_path = Path(stats_file).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"action statistics not found: {stats_path}")

    dataset_type = cfg.data.dataset_type
    if dataset_type == "x2robot_v1":
        from wall_x.model.core.action.normalizer import create_normalizers

        with stats_path.open("r") as handle:
            raw_stats = json.load(handle)
        if not isinstance(raw_stats, dict):
            raise TypeError("x2robot_v1 action statistics must be a mapping")
        available = list(raw_stats)
        if requested_key is None:
            if len(available) != 1:
                raise KeyError(
                    "x2robot_v1 serving requires an explicit normalizer key; "
                    f"available={available}"
                )
            resolved_key = available[0]
        elif requested_key not in raw_stats:
            raise KeyError(
                f"normalizer key {requested_key!r} is unavailable; "
                f"available={available}"
            )
        else:
            resolved_key = requested_key
        min_key = norm_cfg.get("min_key", "min")
        delta_key = norm_cfg.get("delta_key", "delta")
        _validate_v1_action_stats(
            raw_stats,
            normalizer_key=resolved_key,
            dof_config=cfg.task.dof_config,
            min_key=min_key,
            delta_key=delta_key,
        )
        merged = {
            "dof_config": cfg.task.dof_config,
            "agent_pos_config": cfg.task.agent_pos_config,
            "normalizer_config": {
                "customized_action_statistic_dof": str(stats_path),
                "min_key": min_key,
                "delta_key": delta_key,
            },
        }
        if norm_cfg.get("clip") is not None:
            merged["normalizer_config"]["clip"] = norm_cfg["clip"]
        normalizer_action, _, _ = create_normalizers(
            merged, action_statistic_dof=raw_stats
        )
        return normalizer_action, resolved_key

    if dataset_type == "x2robot_v2":
        # Serving only needs the flat action min/delta vectors.  Parse them
        # strictly here instead of importing x2robot_dataset_v2: the training
        # data package is intentionally not a deployment dependency, and its
        # permissive default-range behavior would be unsafe for robot control.
        from wall_x.model.core.action.normalizer import Normalizer

        normalizer_type = norm_cfg.get("type", "min_max")
        if normalizer_type not in (None, "min_max"):
            raise NotImplementedError(
                "VAP serving only supports x2robot_v2 min_max normalization, "
                f"got {normalizer_type!r}"
            )
        min_key = norm_cfg.get("min_key", "q001")
        delta_key = norm_cfg.get("delta_key", "delta_001")
        with stats_path.open("r") as handle:
            raw_stats = json.load(handle)
        if not isinstance(raw_stats, dict) or not raw_stats:
            raise TypeError("x2robot_v2 action statistics must be a non-empty mapping")

        # Canonical normalizer catalogs are intentionally heterogeneous: most
        # robot families do not expose the X5 20-DoF EE schema. Serving one
        # checkpoint must parse only its explicit training key instead of
        # requiring every unrelated catalog entry to match this task. When no
        # key is supplied, retain strict single-entry behavior and never guess.
        if requested_key is None:
            if len(raw_stats) != 1:
                raise KeyError(
                    "x2robot_v2 serving requires an explicit normalizer key "
                    "when the statistics catalog has multiple datasets; "
                    f"available={list(raw_stats)}"
                )
            selected_stats = list(raw_stats.items())
        else:
            root_key = requested_key.split("/", 1)[0]
            if root_key not in raw_stats:
                raise KeyError(
                    f"normalizer key {requested_key!r} is unavailable; "
                    f"available={list(raw_stats)}"
                )
            selected_stats = [(root_key, raw_stats[root_key])]

        min_by_key: dict[str, torch.Tensor] = {}
        delta_by_key: dict[str, torch.Tensor] = {}
        for dataset_name, dataset_stats in selected_stats:
            if not isinstance(dataset_stats, dict) or not dataset_stats:
                raise TypeError(f"stats[{dataset_name!r}] must be a non-empty mapping")

            children = list(dataset_stats.values())
            is_flat = any(
                isinstance(child, dict) and (min_key in child or delta_key in child)
                for child in children
            )
            variants = (
                [(str(dataset_name), dataset_stats)]
                if is_flat
                else [
                    (f"{dataset_name}/{end_effector_type}", action_stats)
                    for end_effector_type, action_stats in dataset_stats.items()
                ]
            )
            for norm_key, action_stats in variants:
                if not isinstance(action_stats, dict) or not action_stats:
                    raise TypeError(f"stats[{norm_key!r}] must be a non-empty mapping")
                min_parts: list[np.ndarray] = []
                delta_parts: list[np.ndarray] = []
                for model_key, dimension in cfg.task.dof_config.items():
                    candidates = [model_key]
                    follow_key = model_key.replace("master_", "follow_", 1)
                    if follow_key not in candidates:
                        candidates.append(follow_key)
                    selected = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate in action_stats
                        ),
                        None,
                    )
                    if selected is None:
                        raise KeyError(
                            f"x2robot_v2 stats {norm_key!r} are missing action "
                            f"{model_key!r}; tried {candidates}"
                        )
                    stat = action_stats[selected]
                    if not isinstance(stat, dict):
                        raise TypeError(
                            f"stats[{norm_key!r}][{selected!r}] must be a mapping"
                        )
                    width = int(dimension)
                    min_values = stat.get(min_key)
                    delta_values = stat.get(delta_key)
                    parsed = []
                    for field, values in (
                        (min_key, min_values),
                        (delta_key, delta_values),
                    ):
                        try:
                            array = np.asarray(values, dtype=np.float64)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"stats[{norm_key!r}][{selected!r}][{field!r}] "
                                "must be numeric [D] or [T,D]"
                            ) from exc
                        if array.ndim not in (1, 2) or array.shape[-1] != width:
                            raise ValueError(
                                f"stats[{norm_key!r}][{selected!r}][{field!r}] "
                                f"shape {array.shape} must end in configured width {width}"
                            )
                        if not np.isfinite(array).all():
                            raise ValueError(
                                f"stats[{norm_key!r}][{selected!r}][{field!r}] "
                                "contains non-finite values"
                            )
                        parsed.append(array)
                    min_parts.append(parsed[0])
                    delta_parts.append(parsed[1])

                temporal_lengths = {
                    part.shape[0]
                    for part in (*min_parts, *delta_parts)
                    if part.ndim == 2
                }
                if len(temporal_lengths) > 1:
                    raise ValueError(
                        f"stats[{norm_key!r}] has inconsistent per-step lengths: "
                        f"{sorted(temporal_lengths)}"
                    )
                if temporal_lengths:
                    steps = temporal_lengths.pop()

                    def per_step(part: np.ndarray) -> np.ndarray:
                        if part.ndim == 2:
                            return part
                        return np.broadcast_to(part[None, :], (steps, part.shape[0]))

                    min_array = np.concatenate(
                        [per_step(part) for part in min_parts], axis=-1
                    )
                    delta_array = np.concatenate(
                        [per_step(part) for part in delta_parts], axis=-1
                    )
                else:
                    min_array = np.concatenate(min_parts, axis=-1)
                    delta_array = np.concatenate(delta_parts, axis=-1)

                min_tensor = torch.as_tensor(min_array, dtype=torch.float32)
                delta_tensor = torch.as_tensor(delta_array, dtype=torch.float32)
                if not torch.isfinite(min_tensor).all():
                    raise ValueError(
                        f"x2robot_v2 stats {norm_key!r} contain non-finite minima"
                    )
                if not torch.isfinite(delta_tensor).all() or bool(
                    (delta_tensor <= 0).any().item()
                ):
                    raise ValueError(
                        f"x2robot_v2 stats {norm_key!r} require finite positive deltas"
                    )
                min_by_key[norm_key] = min_tensor
                delta_by_key[norm_key] = delta_tensor

        normalizer_action = Normalizer.from_vectors(
            min_by_key,
            delta_by_key,
            clamp_normalized=bool(norm_cfg.get("clip", True)),
        )
        return normalizer_action, _resolve_normalizer_key(
            normalizer_action, requested_key
        )

    raise NotImplementedError(
        "VAP serving supports dataset_type x2robot_v1 or x2robot_v2, got "
        f"{dataset_type!r}"
    )


def _validate_serving_normalizer_contract(
    cfg: Any, normalizer: Any, normalizer_key: str
) -> None:
    """Fail before model load when checkpoint stats cannot serve one AR chunk."""

    minimum_by_key = getattr(normalizer, "min", {})
    delta_by_key = getattr(normalizer, "delta", {})
    if normalizer_key not in minimum_by_key or normalizer_key not in delta_by_key:
        raise KeyError(f"normalizer tensors are unavailable for {normalizer_key!r}")
    minimum = minimum_by_key[normalizer_key]
    delta = delta_by_key[normalizer_key]
    if tuple(minimum.shape) != tuple(delta.shape):
        raise ValueError(
            f"normalizer min/delta shapes differ for {normalizer_key!r}: "
            f"{tuple(minimum.shape)} vs {tuple(delta.shape)}"
        )
    action_dim = sum(int(width) for width in cfg.task.dof_config.values())
    if minimum.ndim == 1:
        expected = (action_dim,)
    elif minimum.ndim == 2:
        expected = (
            int(cfg.model.num_interleave_chunks)
            * int(cfg.model.action_steps_per_chunk),
            action_dim,
        )
    else:
        raise ValueError(
            f"normalizer {normalizer_key!r} must be [D] or [T,D], "
            f"got {tuple(minimum.shape)}"
        )
    if tuple(minimum.shape) != expected:
        raise ValueError(
            f"normalizer {normalizer_key!r} shape {tuple(minimum.shape)} != "
            f"checkpoint contract {expected}"
        )
    if minimum.ndim == 2:
        chunk_steps = int(cfg.model.action_steps_per_chunk)
        chunks = int(cfg.model.num_interleave_chunks)
        minimum_chunks = minimum.reshape(chunks, chunk_steps, action_dim)
        delta_chunks = delta.reshape(chunks, chunk_steps, action_dim)
        if any(
            not torch.equal(minimum_chunks[0], minimum_chunks[index])
            or not torch.equal(delta_chunks[0], delta_chunks[index])
            for index in range(1, chunks)
        ):
            raise ValueError(
                f"normalizer {normalizer_key!r} has different per-chunk "
                "statistics; one-chunk AR serving cannot choose a training "
                "slot implicitly"
            )
    if not torch.isfinite(minimum).all() or not torch.isfinite(delta).all():
        raise ValueError(f"normalizer {normalizer_key!r} contains non-finite values")
    if bool((delta <= 0).any().item()):
        raise ValueError(f"normalizer {normalizer_key!r} contains non-positive deltas")

    norm_cfg = cfg.data.normalizer_config or {}
    expected_clip = bool(norm_cfg.get("clip", True))
    actual_clip = bool(getattr(normalizer, "clamp_normalized", True))
    if actual_clip != expected_clip:
        raise ValueError(
            "serving normalizer clamp differs from checkpoint config: "
            f"normalizer={actual_clip}, config={expected_clip}"
        )


def _build_x2robot_v2_proprio_normalizer(
    cfg: Any, *, stats_path: str | Path, normalizer_key: str
) -> Any:
    """Rebuild the exact state normalizer from the EE-v3 training JSON."""

    from wall_x.model.core.action.normalizer import Normalizer

    path = Path(stats_path).expanduser().resolve()
    with path.open("r") as handle:
        raw_stats = json.load(handle)
    root_key = normalizer_key.split("/", 1)[0]
    if not isinstance(raw_stats, dict) or root_key not in raw_stats:
        raise KeyError(
            f"proprio stats key {root_key!r} is unavailable in {path}; "
            f"available={list(raw_stats) if isinstance(raw_stats, dict) else None}"
        )
    selected = raw_stats[root_key]
    if "/" in normalizer_key:
        variant = normalizer_key.split("/", 1)[1]
        if not isinstance(selected, dict) or variant not in selected:
            raise KeyError(
                f"proprio stats variant {normalizer_key!r} is unavailable in {path}"
            )
        selected = selected[variant]
    if not isinstance(selected, dict):
        raise TypeError(f"proprio stats {normalizer_key!r} must be a mapping")

    norm_cfg = cfg.data.normalizer_config or {}
    min_key = norm_cfg.get("min_key", "q001")
    delta_key = norm_cfg.get("delta_key", "delta_001")
    min_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    for key, width_value in cfg.task.agent_pos_config.items():
        width = int(width_value)
        stat = selected.get(key)
        if not isinstance(stat, dict):
            raise KeyError(
                f"x2robot_v2 stats {normalizer_key!r} are missing state {key!r}"
            )
        parsed: list[np.ndarray] = []
        for field in (min_key, delta_key):
            array = np.asarray(stat.get(field), dtype=np.float64)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(
                    f"stats[{normalizer_key!r}][{key!r}][{field!r}] must be "
                    f"finite shape ({width},), got {array.shape}"
                )
            parsed.append(array)
        min_parts.append(parsed[0])
        delta_parts.append(parsed[1])
    minimum = torch.as_tensor(np.concatenate(min_parts), dtype=torch.float32)
    delta = torch.as_tensor(np.concatenate(delta_parts), dtype=torch.float32)
    if bool((delta <= 0).any().item()):
        raise ValueError("proprio normalizer requires strictly positive deltas")
    return Normalizer.from_vectors(
        {normalizer_key: minimum},
        {normalizer_key: delta},
        clamp_normalized=bool(norm_cfg.get("clip", True)),
    )


def _build_checkpoint_proprio_normalizer(
    cfg: Any, *, state_path: str | Path, normalizer_key: str
) -> Any:
    """Restore state normalization when global action stats omit state keys.

    Some EE-v3 global-normalization runs intentionally persist only action
    statistics in the JSON selected at serving time.  Their state normalizer
    is still saved in the checkpoint and is the exact object used in training.
    This loader is opt-in at the evaluation-profile boundary and validates the
    selected tensors before constructing a serving-only normalizer.
    """

    from wall_x.model.core.action.normalizer import Normalizer

    path = Path(state_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint normalizer_propri.pth not found: {path}")
    saved_state = torch.load(path, map_location="cpu", weights_only=True)
    minimum = saved_state.get(f"min.{normalizer_key}")
    delta = saved_state.get(f"delta.{normalizer_key}")
    state_dim = sum(int(width) for width in cfg.task.agent_pos_config.values())
    for label, tensor in (("min", minimum), ("delta", delta)):
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(
                f"checkpoint normalizer_propri.pth is missing "
                f"{label}.{normalizer_key}"
            )
        if tensor.shape != (state_dim,) or not torch.isfinite(tensor).all():
            raise ValueError(
                f"checkpoint proprio {label}.{normalizer_key} must be finite "
                f"shape ({state_dim},), got {tuple(tensor.shape)}"
            )
    if bool((delta <= 0).any().item()):
        raise ValueError("checkpoint proprio normalizer requires positive deltas")
    norm_cfg = cfg.data.normalizer_config or {}
    return Normalizer.from_vectors(
        {normalizer_key: minimum.detach().float().cpu()},
        {normalizer_key: delta.detach().float().cpu()},
        clamp_normalized=bool(norm_cfg.get("clip", True)),
    )


def _load_matching_normalizer_state(
    normalizer: Any, state_path: str | Path, *, label: str
) -> None:
    """Load selected saved tensors only when they equal config statistics.

    ``load_state_dict(strict=True)`` alone only checks keys and shapes. A stale
    checkpoint normalizer with the same shape would otherwise silently replace
    the statistics named by ``config.yml``. Exact equality is intentional:
    both tensors are created from the same JSON values during training.

    A training checkpoint may contain a superset of robot-family normalizers
    from a heterogeneous catalog. Serving deliberately reconstructs only the
    explicit evaluation key. Unrelated saved keys are ignored, while every
    selected min/delta tensor must exist and match exactly.
    """

    state_path = Path(state_path).expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint {label} not found: {state_path}")
    saved_state = torch.load(state_path, map_location="cpu", weights_only=True)
    expected_state = normalizer.state_dict()
    missing = sorted(set(expected_state) - set(saved_state))
    if missing:
        raise RuntimeError(
            f"checkpoint {label} is missing config-derived statistics: "
            f"missing={missing}"
        )
    for key, expected in expected_state.items():
        observed = saved_state[key]
        if not isinstance(observed, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            if observed != expected:
                raise RuntimeError(
                    f"checkpoint {label} value {key!r} differs from config"
                )
            continue
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise RuntimeError(
                f"checkpoint {label} tensor {key!r} differs from config: "
                f"saved={tuple(observed.shape)}/{observed.dtype}, "
                f"expected={tuple(expected.shape)}/{expected.dtype}"
            )
        if not torch.equal(observed.cpu(), expected.detach().cpu()):
            max_abs = float(
                (observed.float().cpu() - expected.detach().float().cpu())
                .abs()
                .max()
                .item()
            )
            raise RuntimeError(
                f"checkpoint {label} tensor {key!r} differs from the statistics "
                f"recorded by config.yml (max_abs_diff={max_abs:.9g})"
            )
    selected_state = {key: saved_state[key] for key in expected_state}
    normalizer.load_state_dict(selected_state, strict=True)


class VAPServingRuntime:
    """Single-process VAP inference runtime.

    Input video is RGB ``[V,T,H,W,3]``.  ``T=1`` is valid for the first action
    chunk.  Later chunks require exactly ``T=1+video_frames_per_chunk`` and
    are encoded in one causal VAE call; separately encoding the anchor and
    executed video would not reproduce the training contract.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        model: Any,
        normalizer_action: Any,
        normalizer_propri: Any | None,
        normalizer_key: str,
        device: str | torch.device,
        view_ids: Sequence[int] = (1, 0, 2),
        use_kv_cache: bool = True,
        vae_tiled: bool = False,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.normalizer_action = normalizer_action
        self.normalizer_propri = normalizer_propri
        self.normalizer_key = str(normalizer_key)
        self.device = torch.device(device)
        self.view_ids = tuple(int(value) for value in view_ids)
        self.use_kv_cache = bool(use_kv_cache)
        self.vae_tiled = bool(vae_tiled)

        model_cfg = cfg.model
        self.video_frames_per_chunk = int(model_cfg.video_frames_per_chunk)
        self.action_steps_per_chunk = int(model_cfg.action_steps_per_chunk)
        self.vae_temporal_stride = int(model_cfg.vae_temporal_stride)
        self.action_dim = int(cfg.action_dim)
        self.state_conditioning_enabled = bool(
            getattr(model_cfg, "state_film_enabled", False)
        )
        self.first_frame_memory_enabled = bool(
            getattr(model_cfg, "first_frame_memory_enabled", False)
        )
        self.history_offsets = tuple(getattr(model_cfg, "history_offsets", ()))
        self.first_frame_memory_view_keys = tuple(
            str(key) for key in getattr(model_cfg, "first_frame_memory_view_keys", ())
        )
        self.first_frame_memory_position_mode = str(
            getattr(
                model_cfg,
                "first_frame_memory_position_mode",
                "window_origin",
            )
        )
        self.state_dim = sum(int(width) for width in cfg.task.agent_pos_config.values())
        if int(model_cfg.num_interleave_chunks) != 2:
            raise ValueError("VAP serving requires exactly two interleave slots")
        if self.video_frames_per_chunk <= 0 or self.action_steps_per_chunk <= 0:
            raise ValueError("VAP serving requires positive action/video horizons")
        if self.action_steps_per_chunk % self.video_frames_per_chunk:
            raise ValueError(
                "VAP serving requires an integral action/video temporal ratio"
            )
        if self.video_frames_per_chunk % self.vae_temporal_stride:
            raise ValueError(
                "VAP serving requires video_frames_per_chunk divisible by the "
                "VAE temporal stride"
            )
        if self.action_dim != 20:
            raise ValueError("RoboDojo EE serving currently requires action_dim=20")
        if self.state_conditioning_enabled:
            if self.normalizer_propri is None:
                raise ValueError("State FiLM serving requires a proprio normalizer")
            minimum = self.normalizer_propri.min[self.normalizer_key]
            delta = self.normalizer_propri.delta[self.normalizer_key]
            expected_state = (self.state_dim,)
            if (
                tuple(minimum.shape) != expected_state
                or tuple(delta.shape) != expected_state
            ):
                raise ValueError(
                    "proprio normalizer shape must match the checkpoint state "
                    f"contract {expected_state}, got {tuple(minimum.shape)} and "
                    f"{tuple(delta.shape)}"
                )
            if not torch.isfinite(minimum).all() or not torch.isfinite(delta).all():
                raise ValueError("proprio normalizer contains non-finite values")
            if bool((delta <= 0).any().item()):
                raise ValueError("proprio normalizer contains non-positive deltas")
        if len(self.view_ids) != 3:
            raise ValueError("RoboDojo VAP serving requires three physical view IDs")
        if min(self.view_ids) < 0 or max(self.view_ids) >= int(model_cfg.max_views):
            raise ValueError(
                f"view_ids={self.view_ids} must lie in [0,{model_cfg.max_views})"
            )
        _validate_serving_normalizer_contract(
            cfg, self.normalizer_action, self.normalizer_key
        )

        norm_cfg = cfg.data.normalizer_config or {}
        self.normalizer_stats_file = str(
            Path(norm_cfg["customized_action_statistic_dof"]).expanduser().resolve()
        )
        self.normalizer_min_key = str(norm_cfg.get("min_key", "q001"))
        self.normalizer_delta_key = str(norm_cfg.get("delta_key", "delta_001"))
        self.normalizer_clip = bool(norm_cfg.get("clip", True))

    @classmethod
    def from_paths(
        cls,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        normalizer_key: str | None = None,
        view_ids: Sequence[int] = (1, 0, 2),
        use_kv_cache: bool = True,
        vae_tiled: bool = False,
        action_inference_steps: int | None = None,
        action_inference_shift: float | None = None,
        state_conditioning: bool = False,
        normalizer_stats_path: str | Path | None = None,
        text_feature_cache_path: str | Path | None = None,
        text_feature_cache_online_fallback: bool = False,
        require_checkpoint_normalizer_state: bool = True,
        allow_checkpoint_proprio_normalizer: bool = False,
    ) -> "VAPServingRuntime":
        """Load a strict x2robot_v1/v2 VAP checkpoint without a dataset."""

        from wall_x.config.loader import load_config

        cfg = load_config(str(Path(config_path).expanduser().resolve()))
        if cfg.model_type != "vap":
            raise ValueError(f"expected model_type='vap', got {cfg.model_type!r}")
        if cfg.data.dataset_type not in {"x2robot_v1", "x2robot_v2"}:
            raise ValueError(
                "RoboDojo serving requires an x2robot_v1/v2 VAP config, got "
                f"{cfg.data.dataset_type!r}"
            )
        if normalizer_stats_path is not None:
            stats_override = Path(normalizer_stats_path).expanduser().resolve()
            if not stats_override.is_file():
                raise FileNotFoundError(
                    f"normalizer statistics override not found: {stats_override}"
                )
            norm_cfg = dict(cfg.data.normalizer_config or {})
            norm_cfg["customized_action_statistic_dof"] = str(stats_override)
            cfg.data.normalizer_config = norm_cfg
        if text_feature_cache_path is not None:
            text_cache_override = Path(text_feature_cache_path).expanduser().resolve()
            if not text_cache_override.is_file():
                raise FileNotFoundError(
                    f"text feature cache override not found: {text_cache_override}"
                )
            # Serving-only override: it avoids loading one UMT5 encoder per
            # policy process and never mutates the checkpoint config on disk.
            cfg.model.text_feature_cache_path = str(text_cache_override)
        if text_feature_cache_online_fallback:
            if not getattr(cfg.model, "text_feature_cache_path", None):
                raise ValueError(
                    "online text cache fallback requires a text feature cache"
                )
            # Serving-only override. It is intentionally absent from the
            # checkpoint's training contract and does not mutate config.yml.
            cfg.model.text_feature_cache_online_fallback = True
        if action_inference_steps is not None:
            if int(action_inference_steps) <= 0:
                raise ValueError("action_inference_steps must be positive")
            cfg.model.action_flow["inference_steps"] = int(action_inference_steps)
        if action_inference_shift is not None:
            if (
                not math.isfinite(float(action_inference_shift))
                or float(action_inference_shift) <= 0
            ):
                raise ValueError("action_inference_shift must be positive")
            # This is an inference-only ODE-grid override.  The checkpoint's
            # training-time sigma sampling distribution remains unchanged.
            cfg.model.action_flow["shift"] = float(action_inference_shift)

        checkpoint_state_film = bool(getattr(cfg.model, "state_film_enabled", False))
        if bool(state_conditioning) != checkpoint_state_film:
            raise ValueError(
                "VAP_STATE_CONDITIONING must exactly match checkpoint "
                f"model.state_film_enabled: switch={bool(state_conditioning)} "
                f"checkpoint={checkpoint_state_film}"
            )

        normalizer_action, resolved_norm_key = _build_action_normalizer(
            cfg, normalizer_key
        )

        checkpoint = _resolve_checkpoint_file(checkpoint_path)
        action_state_path = checkpoint.parent / "normalizer_action.pth"
        if action_state_path.is_file():
            _load_matching_normalizer_state(
                normalizer_action,
                action_state_path,
                label="normalizer_action.pth",
            )
        elif require_checkpoint_normalizer_state:
            raise FileNotFoundError(
                f"checkpoint normalizer_action.pth not found: {action_state_path}"
            )
        _validate_serving_normalizer_contract(cfg, normalizer_action, resolved_norm_key)

        normalizer_propri = None
        if checkpoint_state_film:
            proprio_path = checkpoint.parent / "normalizer_propri.pth"
            if allow_checkpoint_proprio_normalizer:
                normalizer_propri = _build_checkpoint_proprio_normalizer(
                    cfg,
                    state_path=proprio_path,
                    normalizer_key=resolved_norm_key,
                )
            else:
                normalizer_propri = _build_x2robot_v2_proprio_normalizer(
                    cfg,
                    stats_path=cfg.data.normalizer_config[
                        "customized_action_statistic_dof"
                    ],
                    normalizer_key=resolved_norm_key,
                )
                if proprio_path.is_file():
                    _load_matching_normalizer_state(
                        normalizer_propri,
                        proprio_path,
                        label="normalizer_propri.pth",
                    )
                elif require_checkpoint_normalizer_state:
                    raise FileNotFoundError(
                        f"checkpoint normalizer_propri.pth not found: {proprio_path}"
                    )

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA VAP serving requested but CUDA is unavailable")
        if requested_device.type == "cuda":
            torch.cuda.set_device(requested_device)

        from wall_x.trainer.adapters.vap_model_adapter import VAPAdapter

        adapter = VAPAdapter(cfg=cfg, logger=None, model_type="vap")
        adapter.normalizer_action = normalizer_action
        # ``checkpoint.resume_from`` records how the training run was
        # initialized.  A self-contained serving checkpoint must not require
        # that source checkpoint to remain mounted: the strict load below is
        # the deployment contract and also verifies the learned Qwen input
        # projector is present.
        model = adapter.create_model(
            None,
            None,
            cfg.model,
            validate_checkpoint_initialization=False,
        )

        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint), device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict VAP checkpoint load failed: {incompatible}")
        del state_dict
        gc.collect()

        model.to(device=requested_device)
        if text_feature_cache_online_fallback:
            # WanPrompter is deliberately not an nn.Module, but make the
            # serving residency contract explicit and future-proof: online
            # fallback starts with frozen T5 on CPU and only borrows the
            # inference device for a cache miss.
            model._move_text_encoder(device=torch.device("cpu"))
        model.eval()
        normalizer_action.to("cpu")
        if normalizer_propri is not None:
            normalizer_propri.to("cpu")
        return cls(
            cfg=cfg,
            model=model,
            normalizer_action=normalizer_action,
            normalizer_propri=normalizer_propri,
            normalizer_key=resolved_norm_key,
            device=requested_device,
            view_ids=view_ids,
            use_kv_cache=use_kv_cache,
            vae_tiled=vae_tiled,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.model.torch_dtype

    def encode_text(self, instruction: str) -> torch.Tensor:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("VAP serving requires a non-empty instruction")
        return self.model._encode_text_context(
            {"instruction": [instruction]},
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def encode_video(self, video: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode RGB ``[V,T,H,W,3]`` to ``[1,V,C,F,H',W']``."""

        pixels = torch.as_tensor(video)
        if pixels.ndim != 5 or pixels.shape[-1] != 3:
            raise ValueError(f"video must be [V,T,H,W,3], got {tuple(pixels.shape)}")
        views, frames, height, width, channels = pixels.shape
        if views != len(self.view_ids):
            raise ValueError(f"video views {views} != configured {len(self.view_ids)}")
        valid_frames = {1, 1 + self.video_frames_per_chunk}
        if frames not in valid_frames:
            raise ValueError(
                f"video frames {frames} must be one of {sorted(valid_frames)}"
            )
        if (frames - 1) % self.vae_temporal_stride:
            raise ValueError(
                f"video length {frames} violates Wan 4N+1 temporal contract"
            )
        if not torch.isfinite(pixels.float()).all():
            raise ValueError("video contains non-finite pixels")

        flat = pixels.reshape(views, frames, height, width, channels).to(
            device=self.device, dtype=torch.float32
        )
        if flat.max().item() <= 1.0:
            flat = flat * 2.0 - 1.0
        else:
            flat = flat * (2.0 / 255.0) - 1.0
        flat = flat.to(dtype=self.dtype).permute(0, 4, 1, 2, 3)
        latents = self.model.vae.encode(
            flat,
            device=self.device,
            tiled=self.vae_tiled,
        ).to(device=self.device, dtype=self.dtype)
        expected_latent_frames = (frames - 1) // self.vae_temporal_stride + 1
        if latents.shape[2] != expected_latent_frames:
            raise RuntimeError(
                f"Wan VAE returned {latents.shape[2]} latent frames; "
                f"expected {expected_latent_frames}"
            )
        return latents.unsqueeze(0)

    @torch.no_grad()
    def encode_first_frame_memory(self, anchor_video: np.ndarray | torch.Tensor):
        """Encode the configured episode first-frame views exactly as in training."""

        if not self.first_frame_memory_enabled:
            raise RuntimeError("checkpoint does not enable first-frame Memory")
        latents = self.encode_video(anchor_video)
        if latents.shape[3] != 1:
            raise ValueError(
                "first-frame Memory anchor must encode to one latent frame"
            )
        from .wan_bridge import EncodedFirstFrameMemory

        # ``viewN`` names the stable input slot used by training data
        # (view1=head, view2=left wrist, view3=right wrist). Physical camera
        # identity is carried separately by ``view_ids`` and must never be
        # used to reorder Memory latents.
        available = {
            f"view{input_index + 1}": input_index
            for input_index in range(len(self.view_ids))
        }
        selected_view_keys = self.first_frame_memory_view_keys or tuple(
            sorted(available, key=lambda key: int(key[4:]))
        )
        missing = tuple(key for key in selected_view_keys if key not in available)
        if missing:
            raise ValueError(
                "first-frame Memory selected views are unavailable at serving: "
                f"{missing}; available={tuple(available)}"
            )
        selected_indices = [available[key] for key in selected_view_keys]
        latents = latents[:, selected_indices]

        return EncodedFirstFrameMemory(
            latents=latents,
            view_keys=selected_view_keys,
        )

    def assemble_history_memory(self, anchors, valid, ages):
        """Snapshot independently encoded observations into a fixed-slot bank.

        Each valid item is encode_first_frame_memory(single image). Latents may
        be cached across requests; age/PE and downstream K/V may not.
        """
        from .wan_bridge import EncodedFirstFrameMemory
        if not self.history_offsets or len(anchors) != 1 + len(self.history_offsets):
            raise ValueError("History bank slot count differs from checkpoint")
        if not valid[0] or anchors[0] is None or len(valid) != len(anchors) or len(ages) != len(anchors):
            raise ValueError("History bank requires an anchor and matching valid/age")
        template = anchors[0]
        frames = []
        for item, ok in zip(anchors, valid):
            if ok:
                if item is None or item.view_keys != template.view_keys or item.latents.shape != template.latents.shape:
                    raise ValueError("History frame geometry/view mismatch")
                frames.append(item.latents)
            else:
                frames.append(torch.zeros_like(template.latents))
        return EncodedFirstFrameMemory(
            latents=torch.cat(frames, dim=3), view_keys=template.view_keys,
            valid=torch.tensor([valid], dtype=torch.bool, device=template.latents.device),
            age_steps=torch.tensor([ages], dtype=torch.long, device=template.latents.device),
        )

    def _validate_first_frame_memory(
        self,
        value: Any,
        *,
        age_steps: int | None,
    ) -> Any:
        if self.first_frame_memory_enabled != (value is not None):
            raise ValueError(
                "checkpoint first-frame Memory setting and supplied episode "
                "Anchor Bank must agree"
            )
        if value is None:
            return None
        if getattr(self, "history_offsets", ()):
            expected = (value.latents.shape[0], 1 + len(self.history_offsets))
            if value.valid is None or value.age_steps is None or tuple(value.valid.shape) != expected or tuple(value.age_steps.shape) != expected:
                raise ValueError("History Memory requires explicit per-frame valid/age")
            if value.valid.dtype != torch.bool or value.age_steps.is_floating_point() or bool((value.age_steps < 0).any()) or not bool(value.valid[:, 0].all()):
                raise ValueError("Invalid History Memory validity/ages")
            # Scalar legacy age cannot overwrite per-frame snapshot ages.
            return value
        if getattr(value, "valid", None) is not None:
            raise ValueError("History bank supplied to a checkpoint with empty history_offsets")
        if self.first_frame_memory_position_mode == "window_origin":
            if age_steps is not None:
                raise ValueError(
                    "first_frame_memory_age_steps is valid only for relative_age PE"
                )
            return value
        if self.first_frame_memory_position_mode != "relative_age":
            raise ValueError(
                "unsupported first-frame Memory position mode: "
                f"{self.first_frame_memory_position_mode!r}"
            )
        if isinstance(age_steps, bool) or not isinstance(age_steps, int):
            raise TypeError(
                "relative-age first-frame Memory requires integer age_steps"
            )
        if age_steps < 0:
            raise ValueError("first-frame Memory age_steps must be non-negative")
        from dataclasses import replace

        return replace(
            value,
            age_steps=torch.full(
                (value.latents.shape[0],),
                age_steps,
                device=value.latents.device,
                dtype=torch.long,
            ),
        )

    def _view_id_tensor(self) -> torch.Tensor:
        return torch.tensor([self.view_ids], dtype=torch.long, device=self.device)

    def _generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def normalize_state_condition(
        self, physical_state: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Normalize one or more absolute chunk-start states as in training."""

        if not self.state_conditioning_enabled or self.normalizer_propri is None:
            raise RuntimeError(
                "state conditioning is disabled for this serving runtime"
            )
        state = torch.as_tensor(physical_state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                "physical state must be [D] or [E,D] with D="
                f"{self.state_dim}, got {tuple(state.shape)}"
            )
        if state.shape[0] not in (1, int(self.cfg.model.num_interleave_chunks)):
            raise ValueError(
                "physical state chunk axis must be 1 or model.num_interleave_chunks="
                f"{self.cfg.model.num_interleave_chunks}, got {state.shape[0]}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("physical state contains non-finite values")
        normalized = self.normalizer_propri.normalize_data(
            state, dataset_names=[self.normalizer_key] * state.shape[0]
        )
        if tuple(normalized.shape) != tuple(state.shape):
            raise RuntimeError(
                "proprio normalizer returned unexpected state shape "
                f"{tuple(normalized.shape)}"
            )
        if not torch.isfinite(normalized).all():
            raise RuntimeError("normalized state contains non-finite values")
        normalized = normalized.to(device=self.device, dtype=self.dtype)
        return normalized if normalized.shape[0] == 1 else normalized.unsqueeze(0)

    @torch.no_grad()
    def sample_first_action(
        self,
        anchor_video: np.ndarray | torch.Tensor,
        *,
        text_context: torch.Tensor,
        seed: int,
        physical_state: np.ndarray | torch.Tensor | None = None,
        first_frame_memory: Any | None = None,
        first_frame_memory_age_steps: int | None = None,
    ) -> torch.Tensor:
        latents = self.encode_video(anchor_video)
        if latents.shape[3] != 1:
            raise ValueError("first-action anchor must encode to one latent frame")
        state_condition = (
            self.normalize_state_condition(physical_state)
            if physical_state is not None
            else None
        )
        sample = self.model.sample_first_action_chunk_given_video(
            condition_video_latents=latents,
            view_ids=self._view_id_tensor(),
            text_context=text_context,
            generator=self._generator(seed),
            use_kv_cache=self.use_kv_cache,
            state_condition=state_condition,
            first_frame_memory=self._validate_first_frame_memory(
                first_frame_memory,
                age_steps=first_frame_memory_age_steps,
            ),
        )
        result = sample.generated_chunk.detach().float().cpu()
        expected = (1, self.action_steps_per_chunk, self.action_dim)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"first action shape {tuple(result.shape)} != {expected}"
            )
        return result[0]

    @torch.no_grad()
    def sample_next_action(
        self,
        anchor_and_executed_video: np.ndarray | torch.Tensor,
        previous_normalized_action: np.ndarray | torch.Tensor,
        *,
        text_context: torch.Tensor,
        seed: int,
        physical_state: np.ndarray | torch.Tensor | None = None,
        first_frame_memory: Any | None = None,
        first_frame_memory_age_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample the next chunk from one fully executed 40/20 prefix."""

        latents = self.encode_video(anchor_and_executed_video)
        expected_latent_frames = self.video_frames_per_chunk // self.vae_temporal_stride
        if latents.shape[3] != 1 + expected_latent_frames:
            raise RuntimeError(
                f"executed video encoded to {latents.shape[3]} latent frames; "
                f"expected {1 + expected_latent_frames}"
            )
        previous = torch.as_tensor(previous_normalized_action).to(
            device=self.device, dtype=self.dtype
        )
        expected_action = (self.action_steps_per_chunk, self.action_dim)
        if tuple(previous.shape) != expected_action:
            raise ValueError(
                f"previous action shape {tuple(previous.shape)} != {expected_action}"
            )
        state_condition = (
            self.normalize_state_condition(physical_state)
            if physical_state is not None
            else None
        )
        sample = self.model.sample_next_action_chunk_given_prefix(
            condition_video_latents=latents[:, :, :, :1],
            clean_prefix_video_latents=latents[:, :, :, 1:],
            clean_prefix_action=previous.unsqueeze(0),
            view_ids=self._view_id_tensor(),
            text_context=text_context,
            generator=self._generator(seed),
            use_kv_cache=self.use_kv_cache,
            state_condition=state_condition,
            first_frame_memory=self._validate_first_frame_memory(
                first_frame_memory,
                age_steps=first_frame_memory_age_steps,
            ),
        )
        result = sample.generated_chunk.detach().float().cpu()
        expected = (1, self.action_steps_per_chunk, self.action_dim)
        if tuple(result.shape) != expected:
            raise RuntimeError(f"next action shape {tuple(result.shape)} != {expected}")
        return result[0]

    def prepare_action_for_execution(
        self, normalized_action: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp controller-bound proposals and decode into physical action space.

        ``cfg.data.normalizer_config.clip`` belongs to the training data
        transform.  In particular, IVAVA must keep it disabled while it
        decodes and rebases A1.  It must not control the independent safety
        boundary used when a model proposal is sent to the robot: the action
        statistics define ``[-1, 1]`` as the supported clean-action range, so
        serving always clamps model extrapolation before unnormalization.
        """

        values = torch.as_tensor(normalized_action, dtype=torch.float32).cpu()
        expected = (self.action_steps_per_chunk, self.action_dim)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"normalized action shape {tuple(values.shape)} != {expected}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("normalized VAP action contains non-finite values")

        # This is deliberately independent from ``self.normalizer_clip``.
        # That flag records the training transform contract; using it here
        # allowed out-of-distribution model values to extrapolate into large
        # Cartesian targets whenever IVAVA training correctly used
        # ``clip=false``.
        outside = (values < -1.0) | (values > 1.0)
        outside_count = int(outside.sum().item())
        if outside_count:
            logger.warning(
                "Clamping %d/%d normalized VAP action values from range "
                "[%.6f, %.6f] to [-1, 1] before unnormalization",
                outside_count,
                values.numel(),
                float(values.min().item()),
                float(values.max().item()),
            )
        issued = values.clamp(-1.0, 1.0)
        physical = self.normalizer_action.unnormalize_data(
            issued.unsqueeze(0), dataset_names=[self.normalizer_key]
        )[0]
        if not torch.isfinite(physical).all():
            raise ValueError("unnormalized VAP action contains non-finite values")
        return (
            issued.float().cpu().numpy(),
            physical.float().cpu().numpy(),
        )

    def normalize_action_prefix(
        self, physical_action: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Encode final issued physical targets as a clean AR action prefix.

        The input must already include every deterministic controller-boundary
        transform performed by the policy adapter, such as SO(3) projection and
        gripper clipping.  Reusing the serving normalizer here gives the next
        window the same normalized action representation used during training.
        """

        values = torch.as_tensor(physical_action, dtype=torch.float32).cpu()
        expected = (self.action_steps_per_chunk, self.action_dim)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"physical action prefix shape {tuple(values.shape)} != {expected}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("physical VAP action prefix contains non-finite values")

        normalized = self.normalizer_action.normalize_data(
            values.unsqueeze(0), dataset_names=[self.normalizer_key]
        )[0]
        if tuple(normalized.shape) != expected:
            raise RuntimeError(
                f"normalized action prefix shape {tuple(normalized.shape)} != "
                f"{expected}"
            )
        if not torch.isfinite(normalized).all():
            raise ValueError("normalized VAP action prefix contains non-finite values")
        return normalized.float().cpu().numpy()


__all__ = ["VAPServingRuntime"]
