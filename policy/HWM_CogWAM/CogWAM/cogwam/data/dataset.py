"""Dual-query LeRobot dataset wrappers.

The implementation composes the single/mixture datasets, the RoboDojo data
config, and the collator without modifying the LeRobot data path underneath.

Pruned relative to upstream, which served several benchmarks from one module:

* The offline DINOv3 latent store (``dino_target_latents``, the ``latents/``
  memmap and per-episode ``.pth`` readers, the strict precomputed-target
  validator, and ``online_dino: auto``).  This recipe sets
  ``framework.dino.force_online`` and ``online_dino: true``, and no latent
  store ships with the release.
* ``fastwam_direct_frame_sampling`` and the aggregate-RoboTwin dataset adapter.
* The LIBERO composite layout and the per-camera (``separate_views``) layout.
* ``future_target_view_keys`` -- the future world target is the same tri-view
  composite as the current observation.
* Planner RGB/text history.  Event-driven semantic memory is the no-history
  path and both the dataset and the model refuse the combination.
* The correlated-action-noise Cholesky estimator; the action head in this
  release has no ``set_action_correlation`` hook.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cogwam.data.collate import collate_fn
from cogwam.data.composite import (
    FASTWAM_COMPOSITE_LAYOUT,
    FASTWAM_COMPOSITE_VIEW_KEY,
    TRI_VIEW_COMPOSITE_LAYOUT,
    TRI_VIEW_COMPOSITE_SIZE,
    TRI_VIEW_COMPOSITE_VIEW_KEY,
    build_per_view_images,
    build_tri_view_composite,
)
from cogwam.data.event_memory import (
    SemanticBoundaryBatchSampler,
    build_or_load_semantic_index,
    event_memory_from_trajectory,
)
from cogwam.data.lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from cogwam.data.lerobot.embodiment_tags import EmbodimentTag
from cogwam.data.lerobot.transform.base import ComposedModalityTransform
from cogwam.data.lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from cogwam.data.mixtures import resolve_data_mix
from cogwam.data.robodojo import ROBOT_TYPE_CONFIG_MAP


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _composite_contract(image_layout: str):
    """Return ``(size, view_key, builder)`` for a configured layout.

    ``fastwam_composite`` is the older name for the same stitched layout and
    the same builder; released checkpoint metadata may carry either string, and
    the evaluation adapter treats the two contracts as equivalent.
    """

    if image_layout == TRI_VIEW_COMPOSITE_LAYOUT:
        return (
            TRI_VIEW_COMPOSITE_SIZE,
            TRI_VIEW_COMPOSITE_VIEW_KEY,
            build_tri_view_composite,
        )
    if image_layout == FASTWAM_COMPOSITE_LAYOUT:
        return (
            TRI_VIEW_COMPOSITE_SIZE,
            FASTWAM_COMPOSITE_VIEW_KEY,
            build_tri_view_composite,
        )
    raise ValueError(f"Unknown composite image layout: {image_layout!r}")


def _text_value(value: Any) -> str:
    """Normalize one parquet text cell without inventing a label."""

    if value is None:
        return ""
    try:
        if bool(np.asarray(value).ndim == 0 and np.asarray(value).dtype.kind == "f" and np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def text_annotation_from_row(row, config) -> dict[str, str]:
    """Read required current/completed-subtask labels from one parquet row."""

    fields = _cfg_get(
        config,
        "fields",
        {"subtask_text": "subtask_text", "completed_subtask_text": "complete_text"},
    )
    fields = dict(fields) if hasattr(fields, "items") else {}
    if not fields:
        raise ValueError("text_annotations.fields must contain at least one mapping")
    result = {
        output_name: _text_value(row.get(str(column_name), ""))
        for output_name, column_name in fields.items()
    }
    missing = sorted(name for name, value in result.items() if not value)
    if missing:
        raise ValueError(
            "Text-supervised RoboDojo data requires every configured annotation "
            f"to be non-empty; missing={missing}"
        )
    return result


def _append_state_norm_if_needed(base_transforms, state_keys: list[str], state_norm_modes: dict | None):
    if not state_keys:
        return base_transforms
    if not isinstance(base_transforms, ComposedModalityTransform):
        base_transforms = ComposedModalityTransform(transforms=[base_transforms])

    existing = []
    for transform in base_transforms.transforms:
        existing.extend(list(getattr(transform, "apply_to", []) or []))
    if any(key in existing for key in state_keys):
        return base_transforms

    modes = state_norm_modes or {}
    if not modes:
        modes = {key: ("binary" if "gripper" in key else "q99") for key in state_keys}
    base_transforms.transforms.append(StateActionToTensor(apply_to=state_keys))
    base_transforms.transforms.append(StateActionTransform(apply_to=state_keys, normalization_modes=modes))
    return base_transforms


def _drop_video_transforms(base_transforms):
    if not isinstance(base_transforms, ComposedModalityTransform):
        return base_transforms

    kept = []
    for transform in base_transforms.transforms:
        apply_to = list(getattr(transform, "apply_to", []) or [])
        if not apply_to:
            kept.append(transform)
            continue

        non_video_keys = [key for key in apply_to if not str(key).startswith("video.")]
        if not non_video_keys:
            continue

        if len(non_video_keys) != len(apply_to):
            transform = copy.deepcopy(transform)
            transform.apply_to = non_video_keys
        kept.append(transform)

    base_transforms.transforms = kept
    return base_transforms


class JointFlowDataset(LeRobotSingleDataset):
    """One LeRobot episode store packed into the dual-query sample ABI.

    Each sample carries the current tri-view composite, the future composite at
    ``world_model.future_stride``, the normalized action chunk, the state, the
    language instruction, and -- when text supervision is on -- the parquet
    subtask labels plus the event-memory KEEP/UPDATE target.
    """

    def __init__(
        self,
        *args,
        online_dino: bool = True,
        **kwargs,
    ):
        if not bool(online_dino):
            raise ValueError(
                "online_dino=false selects the precomputed DINOv3 latent store, which this "
                "release does not ship; the world branch always encodes RGB online."
            )
        self.online_dino = True
        self._last_trajectory_id = None
        self._last_base_index = None
        self._last_video_frames = None
        data_cfg = kwargs.get("data_cfg")
        self._action_pack_dtype = np.dtype(str(_cfg_get(data_cfg, "action_pack_dtype", "float32")))
        self._text_config = _cfg_get(data_cfg, "text_annotations", {})
        if self._action_pack_dtype not in {np.dtype("float16"), np.dtype("float32")}:
            raise ValueError(f"action_pack_dtype must be float16 or float32, got {self._action_pack_dtype}")
        super().__init__(*args, **kwargs)

        text_enabled = bool(_cfg_get(self._text_config, "enabled", False))
        if text_enabled:
            fields = dict(_cfg_get(self._text_config, "fields", {}))
            if not fields:
                raise ValueError(
                    "text_annotations.enabled=true requires non-empty fields mapping"
                )
            available = set(self.lerobot_info_meta.get("features", {}))
            missing = sorted(str(column) for column in fields.values() if str(column) not in available)
            if missing:
                raise ValueError(
                    "Text supervision requires parquet columns declared in meta/info.json; "
                    f"missing={missing}, dataset={self.dataset_path}"
                )
            history = _cfg_get(self._text_config, "history", {})
            event_memory = _cfg_get(self._text_config, "event_memory", {})
            if bool(_cfg_get(event_memory, "enabled", False)):
                if bool(_cfg_get(history, "enabled", False)):
                    raise ValueError(
                        "Event-driven semantic memory and planner RGB history are "
                        "mutually exclusive in the no-history recipe"
                    )
                semantic_offset = int(
                    _cfg_get(event_memory, "semantic_offset", -10)
                )
                replan_interval = int(
                    _cfg_get(event_memory, "replan_interval", 10)
                )
                replan_phase = int(_cfg_get(event_memory, "replan_phase", 0))
                if semantic_offset >= 0 or abs(semantic_offset) != replan_interval:
                    raise ValueError(
                        "event_memory requires semantic_offset=-replan_interval, "
                        f"got offset={semantic_offset}, interval={replan_interval}"
                    )
                if not 0 <= replan_phase < replan_interval:
                    raise ValueError(
                        "event_memory.replan_phase must lie in [0, interval), "
                        f"got phase={replan_phase}, interval={replan_interval}"
                    )
            if bool(_cfg_get(history, "enabled", False)):
                raise ValueError(
                    "Planner RGB/text history was removed with the no-history recipe; "
                    "set text_annotations.history.enabled=false"
                )

    def _future_dino_stride(self) -> int:
        action_horizon = int(
            _cfg_get(self.data_cfg, "action_horizon", _cfg_get(self.data_cfg, "future_action_window_size", 8))
        )
        world_model_cfg = _cfg_get(self.data_cfg, "world_model", None)
        stride = _cfg_get(world_model_cfg, "future_stride", None)
        if stride is None:
            stride = _cfg_get(self.data_cfg, "future_dino_stride", action_horizon)
        return max(int(stride), 0)

    def _future_dino_index(self, trajectory_id: int, base_index: int) -> tuple[int, int, int]:
        traj_pos = int(self.get_trajectory_index(trajectory_id))
        length = int(self.trajectory_lengths[traj_pos])
        requested_stride = self._future_dino_stride()
        remaining_steps = max(length - 1 - int(base_index), 0)
        actual_stride = min(requested_stride, remaining_steps)
        return int(base_index) + actual_stride, actual_stride, requested_stride

    def get_step_data(self, trajectory_id: int, base_index: int, decode_video: bool = True) -> dict:
        self._last_trajectory_id = int(trajectory_id)
        self._last_base_index = int(base_index)
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)

        if decode_video:
            self._last_video_frames = {}
            for key in self.modality_keys.get("video", []):
                self._last_video_frames[key] = self.get_data_by_modality(trajectory_id, "video", key, base_index)

        for modality in ("state", "action", "language"):
            for key in self.modality_keys.get(modality, []):
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        return self._pack_sample(data)

    def _pack_sample(self, data: dict) -> dict:
        if self._last_trajectory_id is None or self._last_base_index is None:
            raise RuntimeError("JointFlowDataset._pack_sample requires get_step_data to be called first.")
        trajectory_id = int(self._last_trajectory_id)
        base_index = int(self._last_base_index)
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(_to_numpy(data[action_key]))
        # Standard LeRobotSingleDataset packs normalized action labels as
        # float16. WAM ablations can opt into the same label quantization while
        # other experiments retain their historical float32 default.
        action = np.concatenate(action, axis=1).astype(self._action_pack_dtype, copy=False)

        future_index, future_steps, future_stride = self._future_dino_index(trajectory_id, base_index)
        require_full_future = bool(_cfg_get(self.data_cfg, "future_valid_requires_full_stride", False))
        future_valid = future_steps == future_stride if require_full_future else future_steps > 0
        sample = {
            "action": action,
            "lang": data[self.modality_keys["language"][0]][0],
            "robot_tag": self.tag,
            "dataset_name": self.dataset_name,
            "trajectory_id": np.int64(trajectory_id),
            "base_index": np.int64(base_index),
            "future_index": np.int64(future_index),
            "future_valid": bool(future_valid),
            "future_valid_steps": np.int64(future_steps),
            "future_stride": np.int64(future_stride),
        }

        if bool(_cfg_get(self._text_config, "enabled", False)):
            row = self.curr_traj_data.iloc[int(self._last_base_index)]
            sample.update(text_annotation_from_row(row, self._text_config))
            sample.update(
                event_memory_from_trajectory(
                    self.curr_traj_data,
                    base_index,
                    self._text_config,
                )
            )

        source_view_keys = list(self.modality_keys.get("video", []))
        if self._last_video_frames is None:
            raise RuntimeError("JointFlowDataset requires get_step_data to decode video first.")
        decode_future = bool(_cfg_get(self.data_cfg, "decode_future_video", True))
        image_layout = str(_cfg_get(self.data_cfg, "image_layout", TRI_VIEW_COMPOSITE_LAYOUT)).lower()
        current_views, future_views = [], []
        video_offsets = [
            int(offset) for offset in self.delta_indices[source_view_keys[0]]
        ]
        for key in source_view_keys[1:]:
            key_offsets = [int(offset) for offset in self.delta_indices[key]]
            if key_offsets != video_offsets:
                raise ValueError(
                    "All video views must share planner delta indices; "
                    f"{source_view_keys[0]}={video_offsets}, {key}={key_offsets}"
                )
        if 0 not in video_offsets:
            raise ValueError(
                f"Decoded RGB offsets must contain the current frame, got {video_offsets}"
            )
        current_position = video_offsets.index(0)
        future_position = (
            video_offsets.index(future_stride)
            if decode_future and future_stride in video_offsets
            else None
        )
        for key in source_view_keys:
            frames = np.asarray(self._last_video_frames[key])  # [T,H,W,C]
            current_views.append(frames[current_position])
            if decode_future:
                if future_position is None:
                    raise ValueError(
                        "Decoded future video offset is missing from modality "
                        f"indices: future_stride={future_stride}, offsets={video_offsets}"
                    )
                future_views.append(frames[future_position])

        expected_source_keys = list(_cfg_get(self.data_cfg, "composite_source_view_keys", []))
        if expected_source_keys and source_view_keys != expected_source_keys:
            raise ValueError(
                "Composite camera order mismatch: "
                f"dataset={source_view_keys}, configured={expected_source_keys}"
            )
        composite_size, default_view_key, composite_builder = (
            _composite_contract(image_layout)
        )
        configured_size = tuple(
            int(value)
            for value in _cfg_get(
                self.data_cfg, "obs_image_size", composite_size
            )
        )
        if configured_size != composite_size:
            raise ValueError(
                f"{image_layout} must be configured as {composite_size} "
                "(width,height), "
                f"got {configured_size}"
            )
        composite_view_key = str(
            _cfg_get(
                self.data_cfg,
                "composite_view_key",
                default_view_key,
            )
        )
        img0 = [np.asarray(composite_builder(current_views), dtype=np.uint8)]
        view_keys = [composite_view_key]
        # Optional parallel per-camera stream, consumed by the VLM only.
        # The composite in img0 is left untouched, so the DINO/world
        # branch keeps its 384x320 input and its 12x10/120 token grid --
        # weight shapes therefore do not move.  Absent config key means
        # absent sample key, so existing recipes are unaffected.
        vlm_view_size = _cfg_get(self.data_cfg, "vlm_view_size", None)
        if vlm_view_size is not None:
            vlm_size = tuple(int(value) for value in vlm_view_size)
            sample["image_0_vlm"] = np.stack(
                [
                    np.asarray(view, dtype=np.uint8)
                    for view in build_per_view_images(
                        current_views,
                        vlm_size,
                    )
                ],
                axis=0,
            )
            sample["vlm_view_keys"] = list(source_view_keys)
        if decode_future:
            img1 = [
                np.asarray(
                    composite_builder(future_views),
                    dtype=np.uint8,
                )
            ]
            target_view_keys = [composite_view_key]
        else:
            img1 = []
            target_view_keys = []

        sample["image_0"] = np.stack(img0, axis=0)
        if img1:
            sample["image_1"] = np.stack(img1, axis=0)
        sample["image_view_keys"] = view_keys
        sample["dino_view_keys"] = view_keys
        if img1:
            sample["dino_target_view_keys"] = target_view_keys

        if self.data_cfg is not None and _cfg_get(self.data_cfg, "include_state", True) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                if state_key in data:
                    state.append(_to_numpy(data[state_key]))
            if state:
                sample["state"] = np.concatenate(state, axis=1).astype(np.float32)
        return sample


def _make_joint_single_dataset(
    dataset_path: Path, robot_type: str, data_cfg, online_dino: bool | None = None
) -> JointFlowDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = copy.deepcopy(data_config.modality_config())

    action_horizon = int(_cfg_get(data_cfg, "action_horizon", _cfg_get(data_cfg, "future_action_window_size", 8)))
    if online_dino is None:
        online_dino = _resolve_online_dino_value(_cfg_get(data_cfg, "online_dino", True))
    world_model_cfg = _cfg_get(data_cfg, "world_model", None)
    future_stride = _cfg_get(world_model_cfg, "future_stride", None)
    if future_stride is None:
        future_stride = _cfg_get(data_cfg, "future_dino_stride", action_horizon)
    future_stride = max(int(future_stride), 1)
    if "video" in modality_config:
        decode_future_video = bool(_cfg_get(data_cfg, "decode_future_video", True))
        video_delta = [0, future_stride] if decode_future_video else [0]
        modality_config["video"] = ModalityConfig(
            delta_indices=video_delta,
            modality_keys=modality_config["video"].modality_keys,
        )
    if "language" in modality_config:
        modality_config["language"] = ModalityConfig(
            delta_indices=[0], modality_keys=modality_config["language"].modality_keys
        )
    if "state" in modality_config:
        modality_config["state"] = ModalityConfig(
            delta_indices=[0], modality_keys=modality_config["state"].modality_keys
        )
    if "action" in modality_config:
        modality_config["action"] = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=modality_config["action"].modality_keys,
        )

    transforms = _drop_video_transforms(data_config.transform())
    state_norm_modes = _cfg_get(data_cfg, "state_norm_modes", None)
    state_config = modality_config.get("state", ModalityConfig(delta_indices=[], modality_keys=[]))
    transforms = _append_state_norm_if_needed(transforms, state_config.modality_keys, state_norm_modes)

    embodiment_tag = getattr(data_config, "embodiment_tag", None) or EmbodimentTag.NEW_EMBODIMENT
    return JointFlowDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=_cfg_get(data_cfg, "video_backend", "torchvision_av"),
        delete_pause_frame=bool(_cfg_get(data_cfg, "delete_pause_frame", False)),
        data_cfg=data_cfg,
        online_dino=online_dino,
    )


def _resolve_online_dino_value(raw) -> bool:
    """Coerce ``datasets.vla_data.online_dino`` to a bool.

    Upstream also accepted ``"auto"``, which probed every dataset for a
    precomputed DINOv3 latent store and only fell back to online features when
    one was missing or had the wrong dimension.  This release ships no latent
    store and pins ``framework.dino.force_online``, so ``"auto"`` is rejected
    rather than silently resolved to a path that no longer exists.
    """
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value == "auto":
            raise ValueError(
                "datasets.vla_data.online_dino='auto' selects the precomputed DINOv3 latent "
                "store, which this release does not ship; set online_dino=true."
            )
        return value in {"1", "true", "yes", "on"}
    return bool(raw)


def build_joint_dataset(cfg, mode: str = "train"):
    vla_cfg = cfg.datasets.vla_data
    mixture_spec = resolve_data_mix(vla_cfg.data_mix)
    online_dino = _resolve_online_dino_value(_cfg_get(vla_cfg, "online_dino", True))
    dataset_mixture = []
    seen = set()
    for data_name, weight, robot_type in mixture_spec:
        key = (data_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        dataset_path = Path(vla_cfg.data_root_dir) / data_name
        dataset_mixture.append(
            (_make_joint_single_dataset(dataset_path, robot_type, vla_cfg, online_dino=online_dino), weight)
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=vla_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=vla_cfg.get("balance_trajectory_weights", False),
        seed=int(getattr(cfg, "seed", 42)),
        data_cfg=vla_cfg,
    )


def build_joint_dataloader(cfg, mode: str = "train") -> DataLoader:
    dataset = build_joint_dataset(cfg, mode=mode)
    workers = int(cfg.datasets.vla_data.get("num_workers", 16))
    prefetch_factor = int(cfg.datasets.vla_data.get("prefetch_factor", 2)) if workers > 0 else None
    persistent_workers = bool(cfg.datasets.vla_data.get("persistent_workers", workers > 0)) if workers > 0 else False
    pin_memory = bool(cfg.datasets.vla_data.get("pin_memory", True))
    return DataLoader(
        dataset,
        batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def build_event_memory_dataloader(
    cfg,
    *,
    output_dir: str | Path | None = None,
) -> DataLoader | None:
    """Build the opt-in semantic NTP stream used by event-memory training.

    The physical dataloader remains untouched.  This second stream decodes only
    the current RGB observation and uses an exact 2/2/2 boundary sampler.  A
    single dataset is required because sampler indices address one episode-local
    frame table directly; the RoboDojo text recipe intentionally satisfies this
    contract.
    """

    vla_cfg = cfg.datasets.vla_data
    text_cfg = _cfg_get(vla_cfg, "text_annotations", {})
    event_cfg = _cfg_get(text_cfg, "event_memory", {})
    if not bool(_cfg_get(event_cfg, "enabled", False)):
        return None

    mixture_spec = resolve_data_mix(vla_cfg.data_mix)
    unique_entries: list[tuple[str, str]] = []
    seen = set()
    for data_name, _weight, robot_type in mixture_spec:
        key = (str(data_name), str(robot_type))
        if key not in seen:
            seen.add(key)
            unique_entries.append(key)
    if len(unique_entries) != 1:
        raise ValueError(
            "Event-memory semantic sampling currently requires exactly one "
            f"dataset/embodiment entry, got {unique_entries}"
        )

    # Do not mutate the physical recipe.  The semantic stream has no future
    # world target and needs only the current observation plus parquet labels.
    if OmegaConf.is_config(vla_cfg):
        semantic_payload = OmegaConf.to_container(vla_cfg, resolve=True)
    elif callable(getattr(vla_cfg, "to_dict", None)):
        semantic_payload = vla_cfg.to_dict(resolve=True)
    elif hasattr(vla_cfg, "items"):
        semantic_payload = dict(vla_cfg.items())
    else:
        raise TypeError(
            "datasets.vla_data must be an OmegaConf/config mapping for the "
            "event-memory semantic stream"
        )
    semantic_data_cfg = OmegaConf.create(semantic_payload)
    semantic_data_cfg.online_dino = True
    semantic_data_cfg.decode_future_video = False
    semantic_data_cfg.action_horizon = 1
    semantic_data_cfg.future_action_window_size = 1

    data_name, robot_type = unique_entries[0]
    dataset = _make_joint_single_dataset(
        Path(semantic_data_cfg.data_root_dir) / data_name,
        robot_type,
        semantic_data_cfg,
        online_dino=True,
    )

    cache_root = Path(output_dir or getattr(cfg, "output_dir", "."))
    index_name = str(_cfg_get(event_cfg, "index_cache_name", "semantic_index_v1.npz"))
    pools = build_or_load_semantic_index(
        dataset,
        semantic_data_cfg.text_annotations,
        cache_root / index_name,
    )
    sampler_cfg = _cfg_get(event_cfg, "sampler", {})
    sampler = SemanticBoundaryBatchSampler(
        pools,
        seed=int(_cfg_get(sampler_cfg, "seed", getattr(cfg, "seed", 42))),
        update_per_batch=int(_cfg_get(sampler_cfg, "update_per_batch", 2)),
        hard_keep_per_batch=int(_cfg_get(sampler_cfg, "hard_keep_per_batch", 2)),
        random_keep_per_batch=int(_cfg_get(sampler_cfg, "random_keep_per_batch", 2)),
    )
    expected_batch = int(_cfg_get(sampler_cfg, "per_device_batch_size", 6))
    if sampler.batch_size != expected_batch:
        raise ValueError(
            "Semantic sampler category counts must sum to per_device_batch_size: "
            f"counts={sampler.batch_size}, configured={expected_batch}"
        )

    workers = int(_cfg_get(sampler_cfg, "num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(_cfg_get(sampler_cfg, "pin_memory", True)),
        "persistent_workers": bool(
            _cfg_get(sampler_cfg, "persistent_workers", workers > 0)
        )
        if workers > 0
        else False,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            _cfg_get(sampler_cfg, "prefetch_factor", 2)
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


__all__ = [
    "JointFlowDataset",
    "build_event_memory_dataloader",
    "build_joint_dataloader",
    "build_joint_dataset",
    "text_annotation_from_row",
]
