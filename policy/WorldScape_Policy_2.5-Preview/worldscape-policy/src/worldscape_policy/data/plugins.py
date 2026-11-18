"""Built-in production dataset registrations.

Importing this module is intentionally cheap. Optional HDF5/Parquet readers are
only imported when the corresponding dataset is opened or indexed.
"""

from __future__ import annotations

from functools import partial

from worldscape_policy.data.adapters import NativeHDF5Dataset, NativeLeRobotDataset
from worldscape_policy.data.mixture import NativeShardedMixtureDataset
from worldscape_policy.data.registry import DATASETS

HDF5_DEMO_DATASET = "worldscape_hdf5_demo"
HDF5_GOAL_DATASET = "worldscape_hdf5_goal"
LEROBOT_DEMO_DATASET = "worldscape_lerobot_demo"
LEROBOT_GOAL_DATASET = "worldscape_lerobot_goal"
LEROBOT_TEXT_DATASET = "worldscape_lerobot_text"
HDF5_TEXT_DATASET = "worldscape_hdf5_text"
HDF5_MIXED_PRETRAIN_DATASET = "worldscape_hdf5_mixed_pretrain"

_DEMO_KWARGS = {
    "visual_prompt": "demo",
    "context_sampling_mode": "uniform",
    "context_video_len": 50,
    "temporal_packing": True,
    "max_chunk_size": 4,
    "wo_norm": True,
}
_MIXTURE_KEYS = {
    "shard_size",
    "shard_sampling_rate",
    "num_shards_to_sample",
    "seed",
    "training",
    "exec_early_sampling_enabled",
    "exec_early_ratio",
    "exec_early_weight",
    "rank",
    "world_size",
}
_PARITY = {"temporal_packing": True, "max_chunk_size": 4, "wo_norm": True}


def _register(name: str, factory: object) -> None:
    # Reloads are harmless, while genuine duplicate registrations still fail in
    # DatasetRegistry.register.
    if name not in DATASETS.names():
        DATASETS.register(name, factory)  # type: ignore[arg-type]


_register(LEROBOT_DEMO_DATASET, partial(NativeLeRobotDataset, visual_prompt="demo"))
_register(LEROBOT_GOAL_DATASET, partial(NativeLeRobotDataset, visual_prompt="goal"))
def _hdf5_text_factory(data_root=None, **kwargs):
    """Text-only HDF5 data, optionally using the legacy-style shard schedule."""

    mixture_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in _MIXTURE_KEYS and value is not None
    }
    child_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _MIXTURE_KEYS and value is not None
    }
    if "seed" in mixture_kwargs:
        child_kwargs.setdefault("seed", mixture_kwargs["seed"])
    child_kwargs.setdefault("visual_prompt", "none")
    child_kwargs.setdefault("context_sampling_mode", "none")
    for key, value in _PARITY.items():
        child_kwargs.setdefault(key, value)
    child = NativeHDF5Dataset(data_root, **child_kwargs)
    if not mixture_kwargs:
        return child
    return NativeShardedMixtureDataset([child], **mixture_kwargs)


_register(HDF5_TEXT_DATASET, _hdf5_text_factory)
_register(
    LEROBOT_TEXT_DATASET,
    partial(
        NativeLeRobotDataset,
        visual_prompt="none",
        context_sampling_mode="none",
        **_PARITY,
    ),
)
def _hdf5_goal_factory(data_root=None, **kwargs):
    """Goal-image HDF5 data with optional legacy-style shard sampling."""

    mixture_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in _MIXTURE_KEYS and value is not None
    }
    child_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _MIXTURE_KEYS and value is not None
    }
    if "seed" in mixture_kwargs:
        child_kwargs.setdefault("seed", mixture_kwargs["seed"])
    child_kwargs.setdefault("visual_prompt", "goal")
    child_kwargs.setdefault("context_sampling_mode", "last")
    child_kwargs.setdefault("context_video_len", 1)
    for key, value in _PARITY.items():
        child_kwargs.setdefault(key, value)
    child = NativeHDF5Dataset(data_root, **child_kwargs)
    if not mixture_kwargs:
        return child
    return NativeShardedMixtureDataset([child], **mixture_kwargs)


_register(HDF5_GOAL_DATASET, _hdf5_goal_factory)


def _hdf5_demo_factory(
    data_root=None,
    dataset_roots=None,
    *,
    mixture_weights=None,
    source_names=None,
    **kwargs,
):
    """Uniform-50 demo visual prefill; optional sharded multi-source mixture."""

    child_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _MIXTURE_KEYS and value is not None
    }
    mixture_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in _MIXTURE_KEYS and value is not None
    }
    if "seed" in mixture_kwargs:
        child_kwargs.setdefault("seed", mixture_kwargs["seed"])
    for key in _DEMO_KWARGS:
        child_kwargs.setdefault(key, _DEMO_KWARGS[key])

    if dataset_roots is not None:
        roots = [root for root in dataset_roots if root is not None and str(root)]
        dataset_roots = roots or None

    use_mixture = (
        dataset_roots is not None
        or mixture_weights is not None
        or mixture_kwargs.get("shard_size") is not None
        or mixture_kwargs.get("exec_early_sampling_enabled") is not None
    )
    if not use_mixture:
        if data_root is None:
            raise ValueError("worldscape_hdf5_demo requires data_root")
        return NativeHDF5Dataset(data_root, **child_kwargs)

    if dataset_roots is None:
        if data_root is None:
            raise ValueError("worldscape_hdf5_demo mixture requires data_root or dataset_roots")
        roots = [data_root]
    else:
        roots = list(dataset_roots)
    names = list(source_names or [f"source_{index}" for index in range(len(roots))])
    if not roots:
        raise ValueError("worldscape_hdf5_demo mixture requires at least one data root")
    if len(names) != len(roots):
        raise ValueError("source_names must contain one name for each dataset root")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("source_names must contain non-empty strings")

    children = [
        NativeHDF5Dataset(root, **child_kwargs)
        for root in roots
    ]
    return NativeShardedMixtureDataset(
        children,
        mixture_weights=mixture_weights,
        **mixture_kwargs,
    )


_register(HDF5_DEMO_DATASET, _hdf5_demo_factory)


def _mixed_pretrain_factory(
    text_data_root,
    goal_data_root,
    video_data_root,
    *,
    mixture_weights=(1.0, 1.0, 1.0),
    goal_context_video_len=1,
    video_context_video_len=50,
    **kwargs,
):
    """Build T2VA/goal-image/video samples for one shared native WAM."""

    mixture_keys = {
        "shard_size",
        "shard_sampling_rate",
        "num_shards_to_sample",
        "seed",
        "training",
        "exec_early_sampling_enabled",
        "exec_early_ratio",
        "exec_early_weight",
        "rank",
        "world_size",
    }
    mixture_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in mixture_keys
    }
    child_seed = int(mixture_kwargs.get("seed", 42))
    common = dict(kwargs)
    common.pop("visual_prompt", None)
    common.pop("context_sampling_mode", None)
    common.pop("context_video_len", None)
    common.setdefault("temporal_packing", True)
    common.setdefault("max_chunk_size", 4)
    common.setdefault("wo_norm", True)
    children = (
        NativeHDF5Dataset(
            text_data_root,
            visual_prompt="none",
            context_sampling_mode="none",
            seed=child_seed,
            **common,
        ),
        NativeHDF5Dataset(
            goal_data_root,
            visual_prompt="goal",
            context_sampling_mode="last",
            context_video_len=goal_context_video_len,
            seed=child_seed + 1,
            **common,
        ),
        NativeHDF5Dataset(
            video_data_root,
            visual_prompt="demo",
            context_sampling_mode="uniform",
            context_video_len=video_context_video_len,
            seed=child_seed + 2,
            **common,
        ),
    )
    return NativeShardedMixtureDataset(
        children,
        mixture_weights=mixture_weights,
        **mixture_kwargs,
    )


_register(HDF5_MIXED_PRETRAIN_DATASET, _mixed_pretrain_factory)


__all__ = [
    "HDF5_DEMO_DATASET",
    "HDF5_GOAL_DATASET",
    "HDF5_MIXED_PRETRAIN_DATASET",
    "HDF5_TEXT_DATASET",
    "LEROBOT_DEMO_DATASET",
    "LEROBOT_GOAL_DATASET",
    "LEROBOT_TEXT_DATASET",
]
