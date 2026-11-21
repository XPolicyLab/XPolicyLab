"""Shared readers for official RoboDojo ARX-X5 HDF5 preparation tools.

Only state/action arrays are read; image payloads are never decoded or rewritten.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from worldscape_policy.data.adapters.common import _read_jsonl, _resolve_episode_path
from worldscape_policy.data.adapters.robodojo import (
    _ACTION_JOINT_PATHS, _STATE_JOINT_PATHS, _pack_joint_array,
)
from worldscape_policy.robodojo_joint import load_joint_statistics


def discover_episodes(data_root: Path) -> list[Path]:
    """Use the same manifest/discovery order as the training HDF5 loader."""
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    metadata = root / "meta/episodes.jsonl"
    if metadata.is_file():
        paths = [_resolve_episode_path(item, [root]) for item in _read_jsonl(metadata)]
    else:
        paths = sorted(p for p in root.glob("**/episode_*.hdf5") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No official episode_*.hdf5 files under {root}")
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError("Duplicate episodes in the dataset manifest")
    return paths


def validate_joint_shapes(handle: h5py.File) -> None:
    lengths = []
    for paths in (_STATE_JOINT_PATHS, _ACTION_JOINT_PATHS):
        for key, width in zip(paths, (6, 1, 6, 1), strict=True):
            if key not in handle:
                raise ValueError(f"{handle.filename}: missing official RoboDojo field {key}")
            shape = handle[key].shape
            if not ((len(shape) == 2 and shape[1] == width) or (len(shape) == 1 and width == 1)):
                raise ValueError(f"{handle.filename}: {key} must have shape [T,{width}], got {shape}")
            lengths.append(shape[0])
    if len(set(lengths)) != 1 or lengths[0] <= 0:
        raise ValueError(f"{handle.filename}: state/action fields must have the same nonzero length")


def read_joints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        validate_joint_shapes(handle)
        state = _pack_joint_array(handle, _STATE_JOINT_PATHS)
        action = _pack_joint_array(handle, _ACTION_JOINT_PATHS)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: nonfinite state/action values")
    return state, action


def map_jobs(function, jobs, workers: int):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        yield from map(function, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            yield from pool.map(function, jobs)


def output_path(path: Path, *, overwrite: bool = False) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use --overwrite to replace it")
    if path.exists() and not path.is_file():
        raise IsADirectoryError(path)
    return path


def write_statistics(path: Path, artifact: dict, *, overwrite: bool = False) -> None:
    path = output_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
        load_joint_statistics(temporary, normalization=artifact["normalization"])
        output_path(path, overwrite=overwrite)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
