#!/usr/bin/env python3
"""Local LeRobot reader for pretraining data.

Each source path is relative to ``data_root``.  The on-disk layout is the
standard LeRobot tree:

    /path/to/data/<dataset>/meta/info.json
    /path/to/data/<dataset>/data/chunk-000/episode_000000.parquet
    /path/to/data/<dataset>/videos/chunk-000/<video_key>/episode_000000.mp4

Example:

    python -m wam.datasets.pretrain_lerobot_loader \
      --dataset-config configs/pretrain/dataset.yaml \
      --index 0 --with-video
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import heapq
import io
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from wam.datasets.urdf import load_chain as _load_urdf_chain
from wam.datasets.lerobot.utils.rotation import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
V3_TASKS_PATH = "meta/tasks.parquet"
V3_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

_PROFILE_ENABLED = False
_PROFILE_EVENTS: list[tuple[str, float, dict[str, Any]]] = []
DEFAULT_DATA_ROOT = os.environ.get("WAM_DATA_ROOT", "data")
DEFAULT_CACHE_ROOT = os.environ.get("WAM_CACHE_ROOT", ".cache/internw0")


def _profile_record(name: str, elapsed_s: float, **info: Any) -> None:
    if _PROFILE_ENABLED:
        _PROFILE_EVENTS.append((name, float(elapsed_s), info))


def _profile_print() -> None:
    if not _PROFILE_ENABLED:
        return
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, elapsed_s, _ in _PROFILE_EVENTS:
        totals[name] = totals.get(name, 0.0) + elapsed_s
        counts[name] = counts.get(name, 0) + 1
    print("[profile] summary", flush=True)
    for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"[profile] {name}: total={total:.3f}s count={counts[name]}", flush=True)
    print("[profile] events", flush=True)
    for name, elapsed_s, info in _PROFILE_EVENTS:
        info_text = " ".join(f"{k}={v}" for k, v in info.items() if v is not None)
        print(f"[profile] {name}: {elapsed_s:.3f}s {info_text}".rstrip(), flush=True)

JSON_DECODE_RETRIES = 3
JSON_RETRY_BASE_DELAY_SECONDS = 1.0


def _is_file_accessible(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _local_path_candidates(path: str | Path | None) -> list[Path]:
    if path is None:
        return []
    return [Path(str(path)).expanduser()]


def path_join(base_url: str, *parts: str) -> str:
    url = str(base_url).rstrip("/")
    for part in parts:
        url = url + "/" + str(part).lstrip("/")
    return url


def _is_remote_url(path: str) -> bool:
    text = str(path)
    if "://" in text:
        return True
    first_segment = text.split("/", 1)[0]
    return ":" in first_segment


def _is_local_filesystem_path(path: str) -> bool:
    text = str(path or "").strip()
    if not text:
        return False
    return not _is_remote_url(text)


def _resolve_local_dataset_root(path: str, data_root: str | None = None) -> str:
    value = os.path.expandvars(str(path or "").strip())
    if not value:
        raise ValueError("dataset root is empty")
    if _is_remote_url(value):
        raise ValueError("Download the dataset locally and set its root to a filesystem path.")
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = Path(data_root or os.environ.get("WAM_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser() / target
    return str(target.resolve())


def _json_loads(raw: bytes) -> Any:
    t0 = time.perf_counter()
    out = json.loads(raw.decode("utf-8"))
    _profile_record("json.loads", time.perf_counter() - t0, bytes=len(raw))
    return out


def _jsonl_loads(raw: bytes) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    records = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    _profile_record("jsonl.loads", time.perf_counter() - t0, bytes=len(raw), records=len(records))
    return records


def _shape_tuple_inplace(info: dict[str, Any]) -> dict[str, Any]:
    for feature in info.get("features", {}).values():
        if "shape" in feature:
            feature["shape"] = tuple(feature["shape"])
    return info


def _merge_mapping(parent: Any, child: Any) -> dict[str, Any]:
    out = dict(parent) if isinstance(parent, dict) else {}
    if not isinstance(child, dict):
        return out
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_mapping(out[key], value)
        else:
            out[key] = value
    return out


def _modality_key(modalities: dict[str, Any], name: str, default: str) -> str:
    spec = modalities.get(name)
    if spec is None:
        return default
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        key = spec.get("key", spec.get("source_key"))
        if key in (None, "", "auto"):
            return default
        return str(key)
    return default


def _modality_keys(modalities: dict[str, Any], name: str, default: str) -> list[str]:
    spec = modalities.get(name)
    if spec is None:
        return [default]
    if isinstance(spec, str):
        return [spec] if spec not in ("", "auto") else [default]
    if isinstance(spec, dict):
        keys = spec.get("keys", spec.get("source_keys"))
        if isinstance(keys, (list, tuple)):
            out = [str(key) for key in keys if key not in (None, "", "auto")]
            return out or [default]
        key = spec.get("key", spec.get("source_key"))
        if key in (None, "", "auto"):
            return [default]
        return [str(key)]
    return [default]


def _modality_is_auto(modalities: dict[str, Any], name: str) -> bool:
    spec = modalities.get(name)
    if isinstance(spec, str):
        return spec == "auto"
    if isinstance(spec, dict):
        if "keys" in spec or "source_keys" in spec:
            keys = spec.get("keys", spec.get("source_keys"))
            return keys in (None, "", "auto", [])
        return spec.get("key", spec.get("source_key")) in (None, "", "auto")
    return False


def _feature_dtype(info: dict[str, Any], key: str) -> str:
    return str(info.get("features", {}).get(key, {}).get("dtype", "")).lower()


def _is_numeric_feature(info: dict[str, Any], key: str) -> bool:
    dtype = _feature_dtype(info, key)
    return dtype not in {"video", "image", "string", "str"}


def _feature_dim(info: dict[str, Any], key: str) -> int | None:
    shape = info.get("features", {}).get(key, {}).get("shape")
    if not shape:
        return None
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        return int(shape[0])
    if isinstance(shape, (list, tuple)) and _is_numeric_feature(info, key):
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    return None


def _feature_dims(info: dict[str, Any], keys: list[str]) -> int | None:
    total = 0
    for key in keys:
        dim = _feature_dim(info, key)
        if dim is None:
            return None
        total += int(dim)
    return total


def _pad_last_dim(x: torch.Tensor, target_dim: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if target_dim is None:
        target_dim = int(x.shape[-1])
    if int(x.shape[-1]) > int(target_dim):
        raise ValueError(f"Cannot pad tensor with last dim {x.shape[-1]} to smaller target_dim={target_dim}")
    pad_dim = int(target_dim) - int(x.shape[-1])
    if pad_dim > 0:
        x = torch.nn.functional.pad(x, (0, pad_dim))
    mask = torch.zeros(int(target_dim), dtype=torch.bool, device=x.device)
    if pad_dim > 0:
        mask[-pad_dim:] = True
    return x, mask


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text).strip("/"):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "_".join("".join(out).split("/"))


def _safe_rel_path(text: str) -> str:
    parts = [_safe_name(part) for part in str(text).strip("/").split("/") if part.strip("/")]
    return "/".join(part for part in parts if part)


_REMOTE_CACHE_STRIP_PREFIXES = (
    f"{DEFAULT_DATA_ROOT.rstrip('/')}/",
    "/path/to/data/",
)


def _relative_remote_cache_path(remote_root: str | None) -> str:
    text = str(remote_root or "").strip()
    for prefix in _REMOTE_CACHE_STRIP_PREFIXES:
        if text.startswith(prefix):
            return _safe_rel_path(text[len(prefix) :])
    if _is_remote_url(text):
        raise ValueError("Use a local dataset path")
    return _safe_rel_path(text)

def _format_template(
    template: str | None,
    *,
    name: str,
    path: str,
    remote: str | None = None,
    source: str | None = None,
    relative_remote: str | None = None,
    embodiment: str | None = None,
    cache_root: str | None = None,
    **extra: Any,
) -> str | None:
    if template in (None, ""):
        return None
    clean_path = str(path).strip("/")
    clean_remote = _safe_rel_path(relative_remote or _relative_remote_cache_path(remote) or remote or name or "dataset")
    values = {
        "name": _safe_name(name),
        "path": clean_path,
        "safe_path": _safe_name(clean_path),
        "remote": clean_remote,
        "relative_remote": clean_remote,
        "safe_remote": _safe_name(remote or clean_remote or "dataset"),
        "source": _safe_name(source or remote or "source"),
        "embodiment": _safe_name(embodiment or "default"),
        "cache_root": str(cache_root or extra.get("cache_root") or DEFAULT_CACHE_ROOT).rstrip("/"),
    }
    for key, value in extra.items():
        if key == "cache_root":
            continue
        values[str(key)] = _safe_name(value) if value is not None else ""
    return str(template).format(**values)


def _weight_override_value(value: Any) -> tuple[float | None, bool | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        weight = value.get("dataset_weight", value.get("weight"))
        distribute = value.get("distribute_weights")
        return (None if weight is None else float(weight), None if distribute is None else bool(distribute))
    return float(value), None


def _apply_weight_override(weight: float, distribute: bool, override: Any) -> tuple[float, bool]:
    new_weight, new_distribute = _weight_override_value(override)
    if new_weight is not None:
        weight = float(new_weight)
    if new_distribute is not None:
        distribute = bool(new_distribute)
    return float(weight), bool(distribute)


def _lookup_weight_override(overrides: Any, keys: list[Any] | tuple[Any, ...]) -> Any:
    if not isinstance(overrides, dict):
        return None
    for key in keys:
        if key is None:
            continue
        text = str(key)
        if text in overrides:
            return overrides[text]
        safe = _safe_name(text)
        if safe in overrides:
            return overrides[safe]
    return None


def _table_to_tensors(table: Any) -> dict[str, torch.Tensor | list[Any]]:
    out: dict[str, torch.Tensor | list[Any]] = {}
    for name in table.column_names:
        column = table[name]
        values = column.to_pylist()
        if not values:
            out[name] = torch.empty(0)
            continue
        first = values[0]
        if isinstance(first, str):
            out[name] = values
            continue
        arr = np.asarray(values)
        if arr.dtype == object:
            arr = np.stack(values)
        if arr.dtype.kind in {"U", "S", "O"}:
            out[name] = values
        else:
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32, copy=False)
            out[name] = torch.from_numpy(np.asarray(arr))
    return out


def _select_rows(data: dict[str, torch.Tensor | list[Any]], key: str, indices: list[int]) -> torch.Tensor | list[Any]:
    if key not in data:
        raise KeyError(f"Missing column {key!r}; available columns: {sorted(data.keys())}")
    value = data[key]
    if isinstance(value, torch.Tensor):
        return value[torch.as_tensor(indices, dtype=torch.long)]
    return [value[i] for i in indices]


def _as_feature_matrix(value: torch.Tensor, key: str) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    if value.ndim > 2:
        return value.flatten(1)
    if value.ndim == 0:
        raise ValueError(f"Column {key!r} produced a scalar tensor; expected one value per frame.")
    return value


def _select_rows_concat(data: dict[str, torch.Tensor | list[Any]], keys: list[str], indices: list[int]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for key in keys:
        selected = _select_rows(data, key, indices)
        if not isinstance(selected, torch.Tensor):
            raise TypeError(f"Column {key!r} is not numeric and cannot be used as state/action.")
        parts.append(_as_feature_matrix(selected, key))
    if not parts:
        raise ValueError("No feature keys configured for state/action.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _parquet_num_rows(data: dict[str, torch.Tensor | list[Any]]) -> int:
    lengths: list[int] = []
    for value in data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            lengths.append(int(value.shape[0]))
        elif isinstance(value, list):
            lengths.append(len(value))
    if not lengths:
        raise RuntimeError("Could not infer parquet row count from loaded episode columns.")
    return min(lengths)


CANONICAL_LAYOUT_DIMS = {
    "robot": 80,
}


def _canonical_adapter_dim(adapter: dict[str, Any]) -> int | None:
    if not isinstance(adapter, dict):
        return None
    dim = adapter.get("dim", adapter.get("target_dim"))
    if dim is not None:
        return int(dim)
    layout = adapter.get("layout")
    if layout is None:
        return None
    return CANONICAL_LAYOUT_DIMS.get(str(layout))


def _slice_last_dim(x: torch.Tensor, raw_slice: list[Any] | tuple[Any, ...] | None) -> torch.Tensor:
    x = _as_feature_matrix(x, "canonical")
    if raw_slice is None:
        return x
    if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
        raise ValueError(f"raw_slice must be [start, end], got {raw_slice!r}")
    start = raw_slice[0]
    end = raw_slice[1]
    start = None if start is None else int(start)
    end = None if end is None else int(end)
    return x[..., start:end]


def _select_feature(data: dict[str, torch.Tensor | list[Any]], key: str, indices: list[int], raw_slice: list[Any] | tuple[Any, ...] | None = None) -> torch.Tensor:
    value = _select_rows(data, key, indices)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Column {key!r} is not numeric and cannot be used by canonical_adapter.")
    return _slice_last_dim(value, raw_slice).to(torch.float32)


def _component_raw_slice(component: dict[str, Any], kind: str) -> list[Any] | tuple[Any, ...] | None:
    return component.get(f"{kind}_raw_slice") or component.get("raw_slice")


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_to_rotvec_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q.to(torch.float32))
    sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
    q = q * sign
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half > 1e-8, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    return xyz * scale


def _quat_to_rotvec_xyzw(q: torch.Tensor) -> torch.Tensor:
    return _quat_to_rotvec_wxyz(torch.cat([q[..., 3:4], q[..., :3]], dim=-1))


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q.to(torch.float32))
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _normalize_quat(a.to(torch.float32))
    b = _normalize_quat(b.to(torch.float32))
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return _normalize_quat(
        torch.cat(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=-1,
        )
    )


def _rpy_to_rotvec(rpy: torch.Tensor) -> torch.Tensor:
    # Roll-pitch-yaw in XYZ convention, converted through quaternion.
    half = rpy.to(torch.float32) * 0.5
    cr, cp, cy = torch.cos(half[..., 0:1]), torch.cos(half[..., 1:2]), torch.cos(half[..., 2:3])
    sr, sp, sy = torch.sin(half[..., 0:1]), torch.sin(half[..., 1:2]), torch.sin(half[..., 2:3])
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _quat_to_rotvec_wxyz(torch.cat([qw, qx, qy, qz], dim=-1))


def _pose_to_xyz_rotvec(value: torch.Tensor, pose_input: str) -> torch.Tensor:
    value = _as_feature_matrix(value, "pose").to(torch.float32)
    if pose_input == "xyz_rotvec":
        if value.shape[-1] != 6:
            raise ValueError(f"xyz_rotvec pose expects 6 dims, got {value.shape[-1]}")
        return value
    if pose_input == "xyz_quat_wxyz":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_wxyz pose expects 7 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _quat_to_rotvec_wxyz(value[..., 3:7])], dim=-1)
    if pose_input == "xyz_quat_xyzw":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_xyzw pose expects 7 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _quat_to_rotvec_xyzw(value[..., 3:7])], dim=-1)
    if pose_input in {"xyz_rpy", "delta_xyz_rpy"}:
        if value.shape[-1] != 6:
            raise ValueError(f"{pose_input} pose expects 6 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _rpy_to_rotvec(value[..., 3:6])], dim=-1)
    raise ValueError(f"Unsupported canonical_adapter pose_input={pose_input!r}")


def _pose_output_format(kind: str, adapter: dict[str, Any], component: dict[str, Any]) -> str:
    return str(
        component.get(f"{kind}_pose_output")
        or component.get("pose_output")
        or adapter.get(f"{kind}_pose_output")
        or adapter.get("pose_output")
        or "xyz_rotvec"
    )


def _rotvec_to_rot6d(rotvec: torch.Tensor) -> torch.Tensor:
    rotation = _rotvec_to_matrix(rotvec)
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def _rot6d_columns_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Invert the compact wrist producer's two-column rotation encoding."""
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:6]
    first = torch.nn.functional.normalize(first, dim=-1)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _interpolate_xyz_rot6d_pose(
    current: torch.Tensor,
    future: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Interpolate camera-local xyz+rot6d poses at fractional source time."""
    xyz = torch.lerp(current[..., :3], future[..., :3], alpha)
    current_rotation = _rot6d_columns_to_matrix(current[..., 3:9])
    future_rotation = _rot6d_columns_to_matrix(future[..., 3:9])
    relative_rotation = current_rotation.transpose(-1, -2) @ future_rotation
    relative_rotvec = _matrix_to_rotvec(relative_rotation)
    rotation = current_rotation @ _rotvec_to_matrix(alpha * relative_rotvec)
    rotation_6d = rotation[..., :, :2].transpose(-1, -2).reshape(
        *rotation.shape[:-2], 6
    )
    return torch.cat((xyz, rotation_6d), dim=-1)


def _fractional_eef_action_delta(
    first_delta: torch.Tensor,
    second_delta: torch.Tensor,
    alpha_start: torch.Tensor,
    alpha_end: torch.Tensor,
    crosses_boundary: torch.Tensor,
) -> torch.Tensor:
    """Evaluate EEF-local SE(3) motion between two fractional source times."""
    first_translation = first_delta[..., :3]
    first_rotvec = first_delta[..., 3:6]
    second_translation = second_delta[..., :3]
    second_rotvec = second_delta[..., 3:6]

    start_rotation = _rotvec_to_matrix(alpha_start * first_rotvec)
    start_translation = alpha_start * first_translation
    same_rotation = _rotvec_to_matrix(alpha_end * first_rotvec)
    same_translation = alpha_end * first_translation

    first_rotation = _rotvec_to_matrix(first_rotvec)
    second_partial_rotation = _rotvec_to_matrix(alpha_end * second_rotvec)
    crossed_rotation = first_rotation @ second_partial_rotation
    crossed_translation = first_translation + torch.einsum(
        "...ij,...j->...i",
        first_rotation,
        alpha_end * second_translation,
    )

    crossed = crosses_boundary[..., None]
    end_rotation = torch.where(crossed[..., None], crossed_rotation, same_rotation)
    end_translation = torch.where(crossed, crossed_translation, same_translation)
    start_inverse_rotation = start_rotation.transpose(-1, -2)
    delta_translation = torch.einsum(
        "...ij,...j->...i",
        start_inverse_rotation,
        end_translation - start_translation,
    )
    delta_rotation = start_inverse_rotation @ end_rotation
    return torch.cat((delta_translation, _matrix_to_rotvec(delta_rotation)), dim=-1)


def _format_xyz_rotvec_pose(pose: torch.Tensor, pose_output: str) -> torch.Tensor:
    pose_output = str(pose_output)
    if pose_output == "xyz_rotvec":
        return pose
    if pose_output == "xyz_rot6d":
        return torch.cat([pose[..., :3], _rotvec_to_rot6d(pose[..., 3:6])], dim=-1)
    raise ValueError(f"Unsupported canonical_adapter pose_output={pose_output!r}")


def _pose_xyz_quat_wxyz(value: torch.Tensor, pose_input: str) -> tuple[torch.Tensor, torch.Tensor]:
    value = _as_feature_matrix(value, "pose").to(torch.float32)
    if pose_input == "xyz_quat_wxyz":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_wxyz pose expects 7 dims, got {value.shape[-1]}")
        return value[..., :3], _normalize_quat(value[..., 3:7])
    if pose_input == "xyz_quat_xyzw":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_xyzw pose expects 7 dims, got {value.shape[-1]}")
        return value[..., :3], _normalize_quat(torch.cat([value[..., 6:7], value[..., 3:6]], dim=-1))
    raise ValueError(f"state-derived action requires quaternion pose input, got {pose_input!r}")


def _pose_delta_xyz_rotvec(current: torch.Tensor, future: torch.Tensor, pose_input: str) -> torch.Tensor:
    cur_xyz, cur_quat = _pose_xyz_quat_wxyz(current, pose_input)
    next_xyz, next_quat = _pose_xyz_quat_wxyz(future, pose_input)
    delta_quat = _quat_mul_wxyz(_quat_conj_wxyz(cur_quat), next_quat)
    return torch.cat([next_xyz - cur_xyz, _quat_to_rotvec_wxyz(delta_quat)], dim=-1)


def _skew_from_vector(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    rv = rotvec.to(torch.float32)
    theta = rv.norm(dim=-1, keepdim=True)
    axis = rv / theta.clamp_min(1e-8)
    k = _skew_from_vector(axis)
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(*rv.shape[:-1], 3, 3)
    theta_m = theta.unsqueeze(-1)
    return eye + torch.sin(theta_m) * k + (1.0 - torch.cos(theta_m)) * (k @ k)


def _matrix_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle without collapsing pi turns.

    The trace/vee formula is singular at 180 degrees: the skew part is zero,
    so exact pi rotations were previously projected to a zero rotation vector.
    The quaternion branch implementation is well-conditioned both near zero
    and near pi and uses the same WXYZ convention as the rest of this loader.
    """
    r = rotation.to(torch.float32)
    return quaternion_to_axis_angle(matrix_to_quaternion(r))


def _rotation_matrix_from_config(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.eye(3, dtype=dtype, device=device)
    if isinstance(value, dict):
        value = value.get("matrix", value.get("rotation", value.get("name", value.get("preset"))))
    if isinstance(value, str):
        name = value.strip().lower()
        presets = {
            "identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "opencv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ros_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "z_up_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "x_forward_y_left_z_up_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_y_left_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_y_right_to_opencv": [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
            "opencv_to_ros": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot_y_left": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot_y_right": [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
            # EEF-local transforms. These matrices are source_ee_from_canonical_ee:
            # their columns are canonical OpenCV-EEF axes expressed in the source EEF frame.
            # Use the preset whose source EEF axis should become canonical +Z
            # (gripper/finger direction) before entering the network.
            "ee_source_x_to_canonical_z": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "ee_source_y_to_canonical_z": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            "ee_source_z_to_canonical_z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ee_source_neg_x_to_canonical_z": [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            "ee_source_neg_y_to_canonical_z": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "ee_source_neg_z_to_canonical_z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        }
        if name not in presets:
            raise ValueError(f"Unknown frame transform preset {value!r}")
        value = presets[name]
    mat = torch.as_tensor(value, dtype=dtype, device=device)
    if mat.shape != (3, 3):
        raise ValueError(f"frame transform matrix must be 3x3, got {tuple(mat.shape)}")
    return mat


def _merge_frame_transform(adapter: dict[str, Any], component: dict[str, Any], kind: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for source in (
        adapter.get("frame_transform"),
        adapter.get(f"{kind}_frame_transform"),
        component.get("frame_transform"),
        component.get(f"{kind}_frame_transform"),
    ):
        if isinstance(source, dict):
            cfg.update(source)
    return cfg


def _pose_is_delta(kind: str, pose_input: str, adapter: dict[str, Any], component: dict[str, Any]) -> bool:
    semantic = _pose_semantics(kind, adapter, component)
    if semantic is not None:
        return semantic in {"delta", "relative", "velocity"}
    return kind == "action" and str(pose_input).startswith("delta_")


def _pose_semantics(kind: str, adapter: dict[str, Any], component: dict[str, Any]) -> str | None:
    semantic = (
        component.get(f"{kind}_pose_semantics")
        or component.get("pose_semantics")
        or adapter.get(f"{kind}_pose_semantics")
        or adapter.get("pose_semantics")
    )
    if semantic is not None:
        return str(semantic).strip().lower()
    return None


def _uses_state_delta_common(
    adapter: dict[str, Any], component: dict[str, Any]
) -> bool:
    """Whether EEF actions must be regenerated from adjacent canonical states."""

    return bool(component.get("action_from_state_delta")) or (
        _pose_semantics("action", adapter, component) == "state_delta_common"
    )


def _transform_xyz_rotvec_pose(pose: torch.Tensor, frame_transform: dict[str, Any], *, is_delta: bool) -> torch.Tensor:
    if not frame_transform:
        return pose
    dtype, device = pose.dtype, pose.device
    world_from_source = _rotation_matrix_from_config(
        frame_transform.get("world_from_source", frame_transform.get("canonical_world_from_source_world")),
        dtype=dtype,
        device=device,
    )
    canonical_ee_from_source_cfg = frame_transform.get(
        "canonical_ee_from_source_ee",
        frame_transform.get("canonical_local_from_source_local"),
    )
    if canonical_ee_from_source_cfg is not None:
        canonical_ee_from_source = _rotation_matrix_from_config(canonical_ee_from_source_cfg, dtype=dtype, device=device)
        source_ee_from_canonical = canonical_ee_from_source.transpose(-1, -2)
    else:
        source_ee_from_canonical = _rotation_matrix_from_config(
            frame_transform.get("source_ee_from_canonical_ee", frame_transform.get("source_local_from_canonical_local")),
            dtype=dtype,
            device=device,
        )
        canonical_ee_from_source = source_ee_from_canonical.transpose(-1, -2)

    xyz = pose[..., :3]
    rot = _rotvec_to_matrix(pose[..., 3:6])
    if is_delta:
        delta_frame = str(
            frame_transform.get(
                "delta_frame",
                frame_transform.get("action_delta_frame", frame_transform.get("delta_pose_frame", "world")),
            )
        ).lower()
        if delta_frame in {"eef_local", "ee_local", "local", "source_ee", "source_local"}:
            xyz_out = torch.einsum("ij,...j->...i", canonical_ee_from_source, xyz)
            rot_out = canonical_ee_from_source @ rot @ canonical_ee_from_source.transpose(-1, -2)
        else:
            xyz_out = torch.einsum("ij,...j->...i", world_from_source, xyz)
            rot_out = world_from_source @ rot @ world_from_source.transpose(-1, -2)
    else:
        xyz_out = torch.einsum("ij,...j->...i", world_from_source, xyz)
        rot_out = world_from_source @ rot @ source_ee_from_canonical
    return torch.cat([xyz_out, _matrix_to_rotvec(rot_out)], dim=-1)


def _pose_delta_from_transformed_xyz_rotvec(current_pose: torch.Tensor, future_pose: torch.Tensor) -> torch.Tensor:
    current_rot = _rotvec_to_matrix(current_pose[..., 3:6])
    future_rot = _rotvec_to_matrix(future_pose[..., 3:6])
    current_from_world = current_rot.transpose(-1, -2)
    delta_xyz_world = future_pose[..., :3] - current_pose[..., :3]
    delta_xyz_local = torch.einsum("...ij,...j->...i", current_from_world, delta_xyz_world)
    delta_rot = current_from_world @ future_rot
    return torch.cat([delta_xyz_local, _matrix_to_rotvec(delta_rot)], dim=-1)


def _pose_delta_world_from_transformed_xyz_rotvec(current_pose: torch.Tensor, future_pose: torch.Tensor) -> torch.Tensor:
    current_rot = _rotvec_to_matrix(current_pose[..., 3:6])
    future_rot = _rotvec_to_matrix(future_pose[..., 3:6])
    delta_xyz_world = future_pose[..., :3] - current_pose[..., :3]
    delta_rot = current_rot.transpose(-1, -2) @ future_rot
    return torch.cat([delta_xyz_world, _matrix_to_rotvec(delta_rot)], dim=-1)


def _xyz_rot6d_to_xyz_rotvec(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 9:
        raise ValueError(f"xyz+rot6d pose expects 9 dims, got {pose.shape[-1]}")
    rotation = _rot6d_columns_to_matrix(pose[..., 3:9])
    return torch.cat((pose[..., :3], _matrix_to_rotvec(rotation)), dim=-1)


def _assign_target(out: torch.Tensor, dim_mask: torch.Tensor, target_slice: list[Any] | tuple[Any, ...], values: torch.Tensor) -> None:
    if not isinstance(target_slice, (list, tuple)) or len(target_slice) != 2:
        raise ValueError(f"target_slice must be [start, end], got {target_slice!r}")
    start, end = int(target_slice[0]), int(target_slice[1])
    expected = end - start
    values = _as_feature_matrix(values, "target").to(out.dtype)
    if values.shape[-1] != expected:
        raise ValueError(f"target_slice {target_slice!r} expects {expected} dims, got {values.shape[-1]}")
    out[..., start:end] = values
    dim_mask[start:end] = False


def _adapter_columns(adapter: dict[str, Any] | None) -> list[str]:
    if not adapter:
        return []
    columns: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in columns:
            columns.append(value)

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith("_key"):
                    add(value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(adapter)
    return columns


def _decode_pyav_timestamp_group(
    container: Any,
    stream: Any,
    timestamps: list[float],
    *,
    tolerance_s: float,
) -> tuple[torch.Tensor, int]:
    """Decode one timestamp group from an already-open PyAV container."""

    def decode_once(query_timestamps: list[float]) -> tuple[torch.Tensor | None, int]:
        first_ts = float(min(query_timestamps))
        if stream.time_base is not None:
            seek_offset = int(first_ts / float(stream.time_base))
            container.seek(
                seek_offset, any_frame=False, backward=True, stream=stream
            )
        else:
            container.seek(0, any_frame=False, backward=True)

        query_ts = np.asarray(query_timestamps, dtype=np.float64)
        ordered_indices = np.argsort(query_ts, kind="stable").tolist()
        best_frames: list[np.ndarray | None] = [None] * len(query_timestamps)
        best_errors = np.full(
            len(query_timestamps), np.inf, dtype=np.float64
        )
        next_query = 0
        candidate_frames = 0
        previous_frame = None
        previous_ts = None
        fallback_fps = (
            float(stream.average_rate)
            if stream.average_rate is not None
            else 30.0
        )
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and stream.time_base is not None:
                ts = float(frame.pts * stream.time_base)
            elif frame.time is not None:
                ts = float(frame.time)
            else:
                ts = frame_idx / fallback_fps
            if ts + tolerance_s < first_ts:
                previous_frame = frame
                previous_ts = ts
                continue
            candidate_frames += 1
            current_rgb = None
            previous_rgb = None
            while next_query < len(ordered_indices):
                index = ordered_indices[next_query]
                target_ts = query_ts[index]
                if ts < target_ts:
                    break
                if (
                    previous_ts is not None
                    and abs(previous_ts - target_ts) <= abs(ts - target_ts)
                ):
                    if previous_rgb is None:
                        previous_rgb = previous_frame.to_ndarray(
                            format="rgb24"
                        )
                    best_frames[index] = previous_rgb
                    best_errors[index] = abs(previous_ts - target_ts)
                else:
                    if current_rgb is None:
                        current_rgb = frame.to_ndarray(format="rgb24")
                    best_frames[index] = current_rgb
                    best_errors[index] = abs(ts - target_ts)
                next_query += 1
            previous_frame = frame
            previous_ts = ts
            if next_query == len(ordered_indices):
                break

        if candidate_frames == 0:
            return None, 0
        if next_query < len(ordered_indices):
            previous_rgb = previous_frame.to_ndarray(format="rgb24")
            for index in ordered_indices[next_query:]:
                best_frames[index] = previous_rgb
                best_errors[index] = abs(previous_ts - query_ts[index])
        max_err = float(best_errors.max())
        if max_err > max(tolerance_s, 1.0 / 30.0 + tolerance_s):
            _profile_record("video.timestamp_warning", 0.0, max_err=max_err)
        arr = np.stack(best_frames, axis=0)
        frames = (
            torch.from_numpy(arr)
            .permute(0, 3, 1, 2)
            .to(torch.float32)
            / 255.0
        )
        return frames, candidate_frames

    frames, candidate_frames = decode_once(timestamps)
    if frames is not None:
        return frames, candidate_frames
    if float(min(timestamps)) > 0.0:
        fallback, fallback_candidates = decode_once([0.0, *timestamps])
        if fallback is not None:
            return fallback[1:], fallback_candidates
    raise RuntimeError("PyAV decoded no frames from local mp4")


def _decode_mp4_groups_lerobot(
    source: bytes | io.RawIOBase,
    timestamp_groups: dict[Any, list[float]],
    *,
    fps: float,
    tolerance_s: float = 0.02,
    backend: str | None = None,
) -> dict[Any, torch.Tensor]:
    """Decode several independently-seeked groups from one open video."""
    nonempty_groups = {
        key: list(timestamps)
        for key, timestamps in timestamp_groups.items()
        if timestamps
    }
    decoded: dict[Any, torch.Tensor] = {
        key: torch.empty(0, 3, 0, 0)
        for key, timestamps in timestamp_groups.items()
        if not timestamps
    }
    if not nonempty_groups:
        return decoded

    if backend is None:
        from wam.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec

        requested_backend = get_safe_default_codec()
    else:
        requested_backend = str(backend).lower()
    if requested_backend == "torchcodec":
        try:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(
                source, device="cpu", seek_mode="approximate"
            )
            for key, timestamps in nonempty_groups.items():
                t0 = time.perf_counter()
                frame_indices = [
                    max(0, round(float(timestamp) * float(fps)))
                    for timestamp in timestamps
                ]
                decoded[key] = (
                    decoder.get_frames_at(indices=frame_indices)
                    .data.to(torch.float32)
                    / 255.0
                )
                _profile_record(
                    "video.decode",
                    time.perf_counter() - t0,
                    frames=len(timestamps),
                    backend="torchcodec_shared",
                )
            return decoded
        except Exception as exc:
            _profile_record(
                "video.torchcodec_fallback",
                0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
            decoded.clear()
            if hasattr(source, "seek"):
                source.seek(0)

    t0 = time.perf_counter()
    import av

    _profile_record("video.import_av", time.perf_counter() - t0)
    container = av.open(source, mode="r")
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "NONE"
        except Exception:
            pass
        try:
            stream.codec_context.thread_count = 1
        except Exception:
            pass

        ordered_groups = sorted(
            nonempty_groups.items(),
            key=lambda item: float(min(item[1])),
        )
        for key, timestamps in ordered_groups:
            group_t0 = time.perf_counter()
            frames, candidate_frames = _decode_pyav_timestamp_group(
                container,
                stream,
                timestamps,
                tolerance_s=tolerance_s,
            )
            decoded[key] = frames
            _profile_record(
                "video.decode",
                time.perf_counter() - group_t0,
                frames=len(timestamps),
                loaded=candidate_frames,
                backend="pyav_shared",
            )
    finally:
        container.close()
    return decoded


def _decode_mp4_lerobot(
    source: bytes | io.RawIOBase,
    timestamps: list[float],
    *,
    fps: float,
    tolerance_s: float = 0.02,
    backend: str | None = None,
) -> torch.Tensor:
    return _decode_mp4_groups_lerobot(
        source,
        {"frames": timestamps},
        fps=fps,
        tolerance_s=tolerance_s,
        backend=backend,
    )["frames"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    remote_root: str
    stats_path: str | None = "dataset_stats.json"
    local_stats_path: str | None = None
    path_index_dir: str | None = None
    data_root: str = "data"
    require_path_index: bool = False
    local_data_dir: str | None = None
    local_text_embedding_cache_dir: str | None = None
    context_len: int = 128
    text_encoder_id: str = "wan22ti2v5b"
    prompt_template: str = DEFAULT_PROMPT
    task_index_authority: str = "parquet"
    slow_motion_factor: float = 1.0
    action_loss_weight: float = 1.0
    action_gripper_loss_weight: float = 1.0
    action_loss_normalization: str = "legacy_per_sample"
    sampling_family: str | None = None
    allow_padding_at_end: bool | None = None
    modalities: dict[str, Any] | None = None
    canonical_adapter: dict[str, Any] | None = None
    format_version: int | None = None
    embodiment: str = "default"
    control_schema: str | None = None
    stats_group: str | None = None
    normalization_stats: str | None = None
    dataset_weight: float = 1.0
    distribute_weights: bool = False
    action_target_dim: int | None = None
    state_target_dim: int | None = None
    source_name: str | None = None
    group_id: int = 0


def _eef_local_translation_to_common(
    delta_xyz_rotvec: torch.Tensor,
    start_xyz_rot6d: torch.Tensor,
) -> torch.Tensor:
    """Express an EEF-local translation delta in the start observation frame.

    The compact wrist release stores translation and rotation as the relative
    transform ``inv(world_from_eef[t]) @ world_from_eef[t+1]``.  Its rotation
    is already the robot contract's EEF-local delta.  The translation must be
    left-multiplied by the absolute start EEF orientation to match the robot
    contract's common-frame translation delta.

    ``start_xyz_rot6d`` is camera/observation-relative in the compact release,
    because the exported Parquet intentionally does not retain camera world
    poses.  Consequently "common" here is the synchronized observation frame
    at the action start, not a recoverable fixed world frame.  This conversion
    is exact for the information present in the release and deliberately does
    not alter the local SO(3) rotation delta.
    """

    if delta_xyz_rotvec.shape[-1] != 6:
        raise ValueError(
            "EEF action conversion expects xyz+rotvec (6 dims), got "
            f"{delta_xyz_rotvec.shape[-1]}"
        )
    if start_xyz_rot6d.shape[-1] != 9:
        raise ValueError(
            "EEF action conversion expects start xyz+rot6d (9 dims), got "
            f"{start_xyz_rot6d.shape[-1]}"
        )
    start_rotation = _rot6d_columns_to_matrix(start_xyz_rot6d[..., 3:9])
    common_translation = torch.einsum(
        "...ij,...j->...i", start_rotation, delta_xyz_rotvec[..., :3]
    )
    return torch.cat((common_translation, delta_xyz_rotvec[..., 3:6]), dim=-1)

def _urdf_fk_xyz_rotvec(
    joint_values: torch.Tensor,
    fk: dict[str, Any],
) -> torch.Tensor:
    """Batched FK using URDF origin/axis semantics, returned as xyz+rotvec."""

    values = _as_feature_matrix(joint_values, "urdf_fk_joint_values").to(torch.float32)
    joint_names = tuple(str(name) for name in (fk.get("joint_names") or ()))
    if len(joint_names) != int(values.shape[-1]):
        raise ValueError(
            f"URDF FK joint_names has {len(joint_names)} entries but the source has "
            f"{values.shape[-1]} dimensions"
        )
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("URDF FK joint_names must be unique")
    joint_scale = torch.as_tensor(
        fk.get("joint_scale", [1.0] * len(joint_names)),
        dtype=values.dtype,
        device=values.device,
    )
    joint_offset = torch.as_tensor(
        fk.get("joint_offset", [0.0] * len(joint_names)),
        dtype=values.dtype,
        device=values.device,
    )
    if joint_scale.shape != (len(joint_names),) or joint_offset.shape != (
        len(joint_names),
    ):
        raise ValueError(
            "URDF FK joint_scale and joint_offset must match joint_names; "
            f"got {tuple(joint_scale.shape)} and {tuple(joint_offset.shape)}"
        )
    values = values * joint_scale + joint_offset

    urdf_path = str(fk.get("urdf_path") or "")
    base_link = str(fk.get("base_link") or "")
    end_link = str(fk.get("end_link") or "")
    if not urdf_path or not base_link or not end_link:
        raise ValueError("URDF FK requires urdf_path, base_link, and end_link")
    chain = _load_urdf_chain(urdf_path, base_link, end_link)
    source_values = {name: values[..., index] for index, name in enumerate(joint_names)}
    defaults = {
        str(name): float(value)
        for name, value in (fk.get("default_joint_positions") or {}).items()
    }

    batch_shape = values.shape[:-1]
    dtype, device = values.dtype, values.device
    rotation = torch.eye(3, dtype=dtype, device=device).expand(*batch_shape, 3, 3).clone()
    translation = torch.zeros(*batch_shape, 3, dtype=dtype, device=device)
    for joint in chain:
        origin_xyz = torch.as_tensor(joint.xyz, dtype=dtype, device=device)
        origin_rpy = torch.as_tensor(joint.rpy, dtype=dtype, device=device)
        origin_rotation = _rotvec_to_matrix(_rpy_to_rotvec(origin_rpy))
        translation = translation + torch.einsum("...ij,j->...i", rotation, origin_xyz)
        rotation = rotation @ origin_rotation

        if joint.kind not in {"revolute", "continuous", "prismatic", "fixed"}:
            raise ValueError(f"Unsupported URDF joint type {joint.kind!r} for {joint.name!r}")
        if joint.kind == "fixed":
            continue
        if joint.name in source_values:
            position = source_values[joint.name]
        elif joint.name in defaults:
            position = torch.full(batch_shape, defaults[joint.name], dtype=dtype, device=device)
        else:
            raise ValueError(
                f"URDF chain joint {joint.name!r} is not mapped and has no default position"
            )
        axis = torch.as_tensor(joint.axis, dtype=dtype, device=device)
        if joint.kind == "prismatic":
            translation = translation + torch.einsum(
                "...ij,...j->...i", rotation, position[..., None] * axis
            )
        else:
            rotation = rotation @ _rotvec_to_matrix(position[..., None] * axis)

    return torch.cat([translation, _matrix_to_rotvec(rotation)], dim=-1)


class PretrainLeRobotDataset(torch.utils.data.Dataset):
    """Map-style local reader for one LeRobot v2.1/v3.0-style dataset."""

    def __init__(
        self,
        spec: DatasetSpec,
        *,
        episodes: list[int] | None = None,
        num_frames: int = 33,
        action_size: int | None = None,
        global_sample_stride: int = 1,
        path_index: list[dict[str, Any]] | None = None,
        use_path_index_cache: bool = True,
    ):
        self.spec = spec
        self.name = spec.name
        self.prompt_template = str(spec.prompt_template)
        self.action_loss_normalization = str(spec.action_loss_normalization).strip().lower()
        if self.action_loss_normalization not in {"legacy_per_sample", "weighted_valid_cells"}:
            raise ValueError(f"Unsupported action loss normalization: {self.action_loss_normalization}")
        for value in (spec.action_loss_weight, spec.action_gripper_loss_weight):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError("Action loss weights must be finite and positive")
        self.task_index_authority = str(spec.task_index_authority)
        if self.task_index_authority not in {"parquet", "episode_metadata"}:
            raise ValueError(
                f"Dataset {self.name} has unsupported task_index_authority="
                f"{self.task_index_authority!r}; use 'parquet' or 'episode_metadata'."
            )
        self.slow_motion_factor = float(spec.slow_motion_factor)
        self.remote_root = spec.remote_root.rstrip("/")
        self.modalities = spec.modalities or {}
        self.canonical_adapter = spec.canonical_adapter or {}
        video_cfg = self.modalities.get("video")
        if isinstance(video_cfg, dict):
            self.video_timestamp_source = str(
                video_cfg.get("timestamp_source", "timestamp")
            ).strip().lower()
        else:
            self.video_timestamp_source = "timestamp"
        if self.video_timestamp_source not in {"timestamp", "frame_index"}:
            raise ValueError(
                f"Dataset {self.name} has unsupported video timestamp_source="
                f"{self.video_timestamp_source!r}; use 'timestamp' or 'frame_index'."
            )
        self.canonical_dim = _canonical_adapter_dim(self.canonical_adapter)
        self.use_canonical_adapter = self.canonical_dim is not None
        self.state_keys = _modality_keys(self.modalities, "state", "observation.state")
        self.action_keys = _modality_keys(self.modalities, "action", "action")
        self.state_keys_config = list(self.state_keys)
        self.action_keys_config = list(self.action_keys)
        self.state_key = self.state_keys[0]
        self.action_key = self.action_keys[0]
        self.timestamp_key = _modality_key(self.modalities, "timestamp", "timestamp")
        self.frame_index_key = _modality_key(self.modalities, "frame_index", "frame_index")
        self.task_index_key = _modality_key(self.modalities, "task_index", "task_index")
        self.video_key = _modality_key(self.modalities, "video", "")
        self.video_keys_config = _modality_keys(self.modalities, "video", "")
        self.num_frames_per_sample = int(num_frames)
        self.action_size = int(action_size if action_size is not None else num_frames - 1)
        self.global_sample_stride = int(global_sample_stride)
        if not math.isfinite(self.slow_motion_factor) or self.slow_motion_factor < 1.0:
            raise ValueError(
                f"Dataset {self.name} slow_motion_factor must be finite and >= 1.0, "
                f"got {self.slow_motion_factor}."
            )
        if self.slow_motion_factor != 1.0:
            if self.global_sample_stride != 1:
                raise ValueError(
                    f"Dataset {self.name} slow motion requires global_sample_stride=1, "
                    f"got {self.global_sample_stride}."
                )
            if not self.use_canonical_adapter:
                raise ValueError(
                    f"Dataset {self.name} slow motion requires a canonical_adapter."
                )
        self._v3_data_file_range_cache: dict[int, tuple[int, int] | None] = {}
        self._v3_data_file_range_cache_max = 1024
        self.use_path_index_cache = bool(use_path_index_cache)
        t0 = time.perf_counter()
        self.info = self._read_json("meta/info.json")
        self.info = _shape_tuple_inplace(self.info)
        self.format_major = self._detect_format_major()
        available_video_keys = [key for key, ft in self.info.get("features", {}).items() if ft.get("dtype") == "video"]
        configured_video_keys = [key for key in self.video_keys_config if key]
        if configured_video_keys:
            missing_video_keys = [key for key in configured_video_keys if key not in available_video_keys]
            if missing_video_keys:
                raise KeyError(
                    f"Dataset {self.name} configured video keys not found in meta/info.json: "
                    f"missing={missing_video_keys}, available={available_video_keys}"
                )
            self.video_keys = configured_video_keys
        else:
            self.video_keys = available_video_keys
        if not self.video_key and self.video_keys:
            self.video_key = self.video_keys[0]
        self.fps = float(self.info.get("fps", 30.0))
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self._resolve_feature_keys()
        self.state_dim = _feature_dims(self.info, self.state_keys)
        self.action_dim = _feature_dims(self.info, self.action_keys)
        if self.use_canonical_adapter:
            self.state_target_dim = int(self.canonical_dim)
            self.action_target_dim = int(self.canonical_dim)
        else:
            self.state_target_dim = int(spec.state_target_dim) if spec.state_target_dim is not None else self.state_dim
            self.action_target_dim = int(spec.action_target_dim) if spec.action_target_dim is not None else self.action_dim
        _profile_record(
            "dataset.read_info",
            time.perf_counter() - t0,
            dataset=self.name,
            version=self.info.get("codebase_version"),
            format_major=self.format_major,
            state_key="+".join(self.state_keys),
            action_key="+".join(self.action_keys),
        )
        t0 = time.perf_counter()
        self.tasks = self._read_tasks()
        self.task_to_index = {task: task_index for task_index, task in self.tasks.items()}
        _profile_record("dataset.read_tasks", time.perf_counter() - t0, dataset=self.name, count=len(self.tasks))
        self.segment_action_text_enabled = self._should_use_segment_action_text()
        self.subtask_action_segments: dict[int, tuple[tuple[int, int, str], ...]] = {}

        loaded_path_index = path_index if path_index is not None else self._load_local_path_index()
        if loaded_path_index is not None:
            self._apply_path_index(loaded_path_index, episodes=episodes)
            self._load_subtask_action_segments()
            _profile_record("dataset.path_index_hit", 0.0, dataset=self.name)
            return

        if self.spec.require_path_index:
            candidates = ", ".join(str(path) for path in self._path_index_paths())
            raise FileNotFoundError(
                f"Dataset {self.name} requires the precomputed path index, "
                f"but path_index.jsonl was not loaded. candidates=[{candidates}]"
            )

        t0 = time.perf_counter()
        self.episodes_dict = self._read_episodes()
        _profile_record("dataset.read_episodes", time.perf_counter() - t0, dataset=self.name, count=len(self.episodes_dict))

        path_index = self.export_path_index()
        self._write_local_path_index(path_index)
        self._apply_path_index(path_index, episodes=episodes)
        self._load_subtask_action_segments()

    def _path_index_paths(self) -> list[Path]:
        if not self.use_path_index_cache or not self.spec.path_index_dir:
            return []
        return [path / "path_index.jsonl" for path in _local_path_candidates(self.spec.path_index_dir)]

    def _path_index_path(self) -> Path | None:
        paths = self._path_index_paths()
        return paths[0] if paths else None

    def _path_index_row_is_usable(self, row: dict[str, Any]) -> bool:
        if "episode_index" not in row:
            return False
        if "frame_count" not in row and "length" not in row:
            return False
        if not row.get("data_path"):
            return False
        video_paths = row.get("video_paths")
        if self.video_keys and not isinstance(video_paths, dict):
            return False
        return all(key in video_paths for key in self.video_keys)

    def _load_local_path_index(self) -> list[dict[str, Any]] | None:
        for cache_path in self._path_index_paths():
            if not _is_file_accessible(cache_path):
                continue
            rows: list[dict[str, Any]] = []
            with cache_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or not self._path_index_row_is_usable(row):
                        raise ValueError(f"Invalid path-index row: {cache_path}:{line_number}")
                    rows.append(row)
            if not rows:
                raise ValueError(
                    f"Dataset {self.name} has no episodes in {cache_path}."
                )
            return rows
        return None


    def _write_local_path_index(self, rows: list[dict[str, Any]]) -> None:
        for cache_path in self._path_index_paths():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=cache_path.parent,
                    prefix=cache_path.name + ".tmp.",
                    delete=False,
                ) as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tmp_path = Path(handle.name)
                tmp_path.replace(cache_path)
                _profile_record("dataset.path_index_write", 0.0, dataset=self.name, path=str(cache_path), episodes=len(rows))
                return
            except Exception:
                continue

    def _apply_path_index(self, rows: list[dict[str, Any]], *, episodes: list[int] | None = None) -> None:
        episodes_dict: dict[int, dict[str, Any]] = {}
        for row in rows:
            episode_index = int(row["episode_index"])
            item: dict[str, Any] = {
                "episode_index": episode_index,
                "length": int(row.get("frame_count", row.get("length"))),
                "data_path": str(row["data_path"]),
                "video_paths": {str(key): str(value) for key, value in dict(row.get("video_paths") or {}).items()},
            }
            for key in (
                "task_index",
                "data_from_index",
                "data_to_index",
                "data_file_ordinal",
                "data_row_group",
                "start_frame",
                "source_episode_index",
            ):
                if key in row:
                    item[key] = int(row[key])
            if isinstance(row.get("video_timestamp_offsets"), dict):
                item["video_timestamp_offsets"] = {
                    str(key): float(value) for key, value in row["video_timestamp_offsets"].items()
                }
            episodes_dict[episode_index] = item
        self.episodes_dict = episodes_dict
        self.selected_episodes = list(episodes) if episodes is not None else sorted(self.episodes_dict)
        self.episode_lengths = [int(self.episodes_dict[ep]["length"]) for ep in self.selected_episodes]
        self.cumulative_lengths = np.cumsum(self.episode_lengths).tolist()
        self.trajectory_ids = np.asarray(self.selected_episodes, dtype=np.int64)
        self.trajectory_lengths = np.asarray(self.episode_lengths, dtype=np.int64)
        # ``action_size`` actions require their right endpoint, so an H32
        # action window spans 32 logical steps.  Slow-motion wrist data maps
        # those logical steps to fractional source time (H32/slow2 -> 16 raw
        # frame transitions).  This single physical span is shared by start
        # eligibility and episode boundaries.
        self.max_delta_index = self._physical_window_span()

    def _physical_window_span(self) -> int:
        logical_span = max(self.num_frames_per_sample - 1, self.action_size)
        return int(math.ceil(logical_span / self.slow_motion_factor)) * int(
            self.global_sample_stride
        )

    def source_episode_index(self, episode_index: int) -> int:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict) and "source_episode_index" in episode:
            return int(episode["source_episode_index"])
        return int(episode_index)

    def episode_start_frame(self, episode_index: int) -> int:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            return max(0, int(episode.get("start_frame", 0) or 0))
        return 0

    def source_frame_index(self, episode_index: int, frame_idx: int) -> int:
        return self.episode_start_frame(int(episode_index)) + int(frame_idx)

    def source_frame_indices(self, episode_index: int, frame_indices: list[int] | torch.Tensor) -> list[int]:
        start = self.episode_start_frame(int(episode_index))
        if isinstance(frame_indices, torch.Tensor):
            values = frame_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            values = list(frame_indices)
        return [start + int(value) for value in values]

    def _should_use_segment_action_text(self) -> bool:
        source = str(self.spec.source_name or "").lower()
        root = str(self.remote_root or "").lower()
        name = str(self.name or "").lower()
        return (
            source.startswith("agibotworld")
            or source.startswith("galaxea")
            or source.startswith("abc")
            or source.startswith("rh20t")
            or source.startswith("robomind_benchmark")
            or "agibotworld" in root
            or "/galaxea/" in root
            or ":abc" in root
            or "rh20t" in root
            or "/robomind/" in root
            or name.startswith("task_")
        )

    def _load_subtask_action_segments(self) -> None:
        self.subtask_action_segments = {}
        if not bool(getattr(self, "segment_action_text_enabled", False)):
            return
        try:
            records = self._read_jsonl("meta/episodes_lang_mem.jsonl")
        except FileNotFoundError:
            return
        except Exception:
            return

        loaded: dict[int, tuple[tuple[int, int, str], ...]] = {}
        for row in records:
            if not isinstance(row, dict) or "episode_index" not in row:
                continue
            segments: list[tuple[int, int, str]] = []
            for segment in row.get("action_config") or []:
                if not isinstance(segment, dict):
                    continue
                text = str(segment.get("action_text") or "").strip()
                if not text:
                    continue
                try:
                    start = int(segment.get("start_frame"))
                    end = int(segment.get("end_frame"))
                except Exception:
                    continue
                if end > start:
                    segments.append((start, end, text))
            if segments:
                segments.sort(key=lambda item: (item[0], item[1]))
                loaded[int(row["episode_index"])] = tuple(segments)
        self.subtask_action_segments = loaded
        _profile_record(
            "dataset.subtask_action_segments",
            0.0,
            dataset=self.name,
            episodes=len(loaded),
        )

    def _subtask_segments_for_episode(self, episode_index: int) -> tuple[tuple[int, int, str], ...]:
        if not self.subtask_action_segments:
            return tuple()
        source_episode = self.source_episode_index(int(episode_index))
        return self.subtask_action_segments.get(int(source_episode), tuple())

    def _subtask_for_source_frame(self, episode_index: int, source_frame_idx: int) -> tuple[int, int, str] | None:
        segments = self._subtask_segments_for_episode(int(episode_index))
        if not segments:
            return None
        frame = int(source_frame_idx)
        for start, end, text in segments:
            if start <= frame < end:
                return start, end, text
        return None

    def _valid_start_intervals_for_episode(
        self,
        episode_index: int,
        episode_len: int,
        allow_padding_at_end: bool = False,
        *,
        extra_span: int = 0,
    ) -> tuple[tuple[int, int], ...]:
        episode_len = max(0, int(episode_len))
        required_delta = (
            0 if allow_padding_at_end else int(self.max_delta_index)
        ) + max(0, int(extra_span))
        segments = self._subtask_segments_for_episode(int(episode_index))
        if not segments:
            end = episode_len - 1 - required_delta
            return ((0, end),) if end >= 0 else tuple()
        source_start = self.episode_start_frame(int(episode_index))
        intervals = []
        for start, end, _ in segments:
            local_start = max(0, int(start) - source_start)
            local_end = min(episode_len, int(end) - source_start) - 1 - required_delta
            if local_end >= local_start:
                intervals.append((local_start, local_end))
        return tuple(intervals)

    def _window_segment_end(
        self,
        episode_index: int,
        episode_len: int,
        frame_idx: int,
    ) -> int:
        for start, end in self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            True,
        ):
            if start <= frame_idx <= end:
                return end + 1
        raise IndexError(
            f"frame {frame_idx} is outside every valid segment in episode {episode_index}"
        )

    def valid_start_intervals_for_trajectory_pos(
        self,
        trajectory_pos: int,
        allow_padding_at_end: bool = False,
    ) -> tuple[tuple[int, int], ...]:
        """Return physical, contiguous valid-start intervals for a trajectory."""
        trajectory_pos = int(trajectory_pos)
        episode_index = int(self.trajectory_ids[trajectory_pos])
        episode_len = int(self.trajectory_lengths[trajectory_pos])
        return self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            allow_padding_at_end,
        )

    def sample_valid_start_for_trajectory_pos(
        self,
        trajectory_pos: int,
        rng: np.random.Generator,
        allow_padding_at_end: bool = False,
        *,
        extra_span: int = 0,
    ) -> int | None:
        episode_index = int(self.trajectory_ids[int(trajectory_pos)])
        episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
        intervals = self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            allow_padding_at_end,
            extra_span=extra_span,
        )
        if not intervals:
            return None
        counts = np.asarray([end - start + 1 for start, end in intervals], dtype=np.float64)
        seg_idx = int(rng.choice(len(intervals), p=counts / counts.sum()))
        start, end = intervals[seg_idx]
        return int(rng.integers(int(start), int(end) + 1))

    def valid_start_for_trajectory_rank(
        self,
        trajectory_pos: int,
        rank: int,
        allow_padding_at_end: bool = False,
        *,
        extra_span: int = 0,
    ) -> int:
        """Map a dense valid-window rank to its episode frame.

        This is the deterministic counterpart of
        :meth:`sample_valid_start_for_trajectory_pos` and is used by the
        coverage sampler to visit valid starts without replacement.
        """
        episode_index = int(self.trajectory_ids[int(trajectory_pos)])
        episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
        intervals = self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            allow_padding_at_end,
            extra_span=extra_span,
        )
        remaining = int(rank)
        if remaining < 0:
            raise IndexError(f"valid-start rank must be non-negative, got {rank}")
        for start, end in intervals:
            count = int(end) - int(start) + 1
            if remaining < count:
                return int(start) + remaining
            remaining -= count
        total = sum(int(end) - int(start) + 1 for start, end in intervals)
        raise IndexError(
            f"valid-start rank {rank} is out of range for dataset={self.name} "
            f"trajectory_pos={trajectory_pos}, count={total}"
        )


    def _episode_task_index_from_metadata(self, episode: dict[str, Any]) -> int | None:
        for key in ("task_index", "tasks_index", "task_indices"):
            value = self._scalar(episode.get(key))
            if value is not None:
                return int(value)

        tasks = episode.get("tasks")
        task: str | None = None
        if isinstance(tasks, str):
            task = tasks
        elif isinstance(tasks, (list, tuple)) and tasks:
            task = str(tasks[0])
        if task is None:
            return None
        return int(self.task_to_index[task]) if task in self.task_to_index else None

    def _index_relative_path(self, path: str) -> str:
        """Keep an index portable when metadata uses absolute local paths."""
        local_path = self._local_data_path(path)
        root = self._dataset_local_root()
        if local_path is None or root is None:
            raise ValueError(f"Cannot index a non-local dataset path: {path!r}")
        # Do not resolve symlinks: a dataset may link to files on another volume.
        local_path = Path(os.path.abspath(local_path))
        root = Path(os.path.abspath(root))
        try:
            return local_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Indexed files must be within the dataset root {root}: {path}") from exc

    def _path_index_row(self, episode_index: int, episode: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "episode_index": int(episode_index),
            "frame_count": int(episode["length"]),
            "start_frame": 0,
            "data_path": self._index_relative_path(self._data_rel_path(int(episode_index))),
            "video_paths": {
                key: self._index_relative_path(self._video_rel_path(int(episode_index), key))
                for key in self.video_keys
            },
        }
        task_index = self._episode_task_index_from_metadata(episode)
        if task_index is not None:
            entry["task_index"] = task_index
        if self.is_v3:
            bounds = self._v3_episode_data_bounds(int(episode_index))
            if bounds is not None:
                entry["data_from_index"] = int(bounds[0])
                entry["data_to_index"] = int(bounds[1])
            entry["data_file_ordinal"] = int(self._v3_data_file_ordinal_from_episode(int(episode_index)))
            video_offsets: dict[str, float] = {}
            for key in self.video_keys:
                offset = self._video_timestamp_offset(int(episode_index), key)
                if offset:
                    video_offsets[key] = float(offset)
            if video_offsets:
                entry["video_timestamp_offsets"] = video_offsets
        return entry

    def export_path_index(self) -> list[dict[str, Any]]:
        return [self._path_index_row(int(key), self.episodes_dict[key]) for key in sorted(self.episodes_dict)]

    @property
    def is_v3(self) -> bool:
        return self.format_major >= 3

    def _detect_format_major(self) -> int:
        if self.spec.format_version is not None:
            return int(self.spec.format_version)
        version = str(self.info.get("codebase_version", "v2.1")).strip().lower().lstrip("v")
        try:
            return int(version.split(".", 1)[0])
        except ValueError:
            return 2

    def set_target_dims(self, *, action_target_dim: int | None = None, state_target_dim: int | None = None) -> None:
        if action_target_dim is not None:
            self.action_target_dim = int(action_target_dim)
        if state_target_dim is not None:
            self.state_target_dim = int(state_target_dim)

    def _feature_sort_key(self, key: str) -> tuple[int, int, str]:
        lowered = key.lower()
        if ".left_" in lowered or lowered.endswith(".left") or "left" in lowered:
            side = 0
        elif ".right_" in lowered or lowered.endswith(".right") or "right" in lowered:
            side = 1
        else:
            side = 2
        if "joint" in lowered or "qpos" in lowered or "arm" in lowered:
            kind = 0
        elif "end_effector" in lowered or "eef" in lowered or ".ee" in lowered:
            kind = 1
        elif "gripper" in lowered:
            kind = 2
        else:
            kind = 3
        return side, kind, lowered

    def _infer_vector_keys(self, name: str, candidates: tuple[str, ...]) -> list[str]:
        features = self.info.get("features", {})
        for key in candidates:
            if key in features and _is_numeric_feature(self.info, key):
                return [key]

        skip_tokens = ("timestamp", "frame_index", "episode_index", "task_index", "index")
        keys: list[str] = []
        if name == "action":
            prefixes = ("action", "actions")
        else:
            prefixes = ("observation.state", "observation.states", "states", "state")

        for key in features:
            lowered = key.lower()
            if any(token == lowered or lowered.endswith("." + token) for token in skip_tokens):
                continue
            if not _is_numeric_feature(self.info, key):
                continue
            if any(lowered == prefix or lowered.startswith(prefix + ".") for prefix in prefixes):
                keys.append(key)

        if keys:
            return sorted(keys, key=self._feature_sort_key)

        lowered_name = name.lower()
        for key in features:
            lower_key = key.lower()
            if lowered_name in lower_key and _is_numeric_feature(self.info, key):
                return [key]
        return []

    def _resolve_feature_key(self, name: str, current: str, candidates: tuple[str, ...]) -> str:
        features = self.info.get("features", {})
        if current in features:
            return current
        if not _modality_is_auto(self.modalities, name) and current not in ("", "auto"):
            return current
        inferred = self._infer_vector_keys(name, candidates)
        return inferred[0] if inferred else current

    def _resolve_vector_keys(self, name: str, current: list[str], candidates: tuple[str, ...]) -> list[str]:
        features = self.info.get("features", {})
        if current and all(key in features for key in current):
            return current
        if not _modality_is_auto(self.modalities, name) and current and current != [""]:
            return current
        inferred = self._infer_vector_keys(name, candidates)
        return inferred or current

    def _resolve_feature_keys(self) -> None:
        self.state_keys = self._resolve_vector_keys(
            "state",
            self.state_keys,
            (
                "observation.state",
                "observation.states",
                "observation.robot_state",
                "observation.joint_state",
                "observation.ee_state",
            ),
        )
        self.action_keys = self._resolve_vector_keys("action", self.action_keys, ("action", "actions"))
        self.state_key = self.state_keys[0]
        self.action_key = self.action_keys[0]
        self.timestamp_key = self._resolve_feature_key("timestamp", self.timestamp_key, ("timestamp",))
        self.frame_index_key = self._resolve_feature_key("frame_index", self.frame_index_key, ("frame_index",))
        self.task_index_key = self._resolve_feature_key("task_index", self.task_index_key, ("task_index",))

    def _remote_path(self, rel_or_abs_path: str) -> str:
        text = str(rel_or_abs_path)
        if text.startswith("@data/"):
            relative = Path(text[len("@data/"):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid dataset path: {text!r}")
            return str((Path(self.spec.data_root).expanduser() / relative).resolve())
        if _is_remote_url(text):
            raise ValueError(f"Expected a local filesystem path, got {text!r}")
        path = Path(text).expanduser()
        return str(path if path.is_absolute() else Path(self.remote_root) / path)

    def _dataset_local_root(self) -> Path | None:
        if self.spec.local_data_dir:
            return Path(self.spec.local_data_dir).expanduser()
        if _is_local_filesystem_path(self.remote_root):
            return Path(self.remote_root).expanduser()
        return None

    def _read_bytes(self, rel_path: str) -> bytes:
        data_path = self._local_data_path(rel_path)
        local_paths = [data_path] if data_path is not None else []
        seen: set[str] = set()
        for local_path in local_paths:
            key = str(local_path)
            if key in seen:
                continue
            seen.add(key)
            if not _is_file_accessible(local_path):
                continue
            t0 = time.perf_counter()
            raw = local_path.read_bytes()
            _profile_record("local.read_bytes", time.perf_counter() - t0, bytes=len(raw), path=local_path.name)
            return raw
        raise FileNotFoundError(
            f"Dataset {self.name} local file not found: {rel_path} "
            f"(looked in {[str(path) for path in local_paths]})"
        )

    def _read_decoded_json_with_retries(
        self,
        rel_path: str,
        decoder: Callable[[bytes], Any],
    ) -> Any:
        total_attempts = JSON_DECODE_RETRIES + 1
        for attempt in range(total_attempts):
            raw = self._read_bytes(rel_path)
            try:
                return decoder(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if attempt >= JSON_DECODE_RETRIES:
                    raise
                delay = JSON_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                print(
                    f"[pretrain-json] decode failed for {rel_path} "
                    f"(attempt {attempt + 1}/{total_attempts}, bytes={len(raw)}, "
                    f"{type(exc).__name__}: {exc}); retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _read_json(self, rel_path: str) -> dict[str, Any]:
        return self._read_decoded_json_with_retries(rel_path, _json_loads)

    def load_dataset_stats(self) -> dict[str, Any]:
        local_stats_path = Path(self.spec.local_stats_path).expanduser() if self.spec.local_stats_path else None
        local_stats_paths = _local_path_candidates(local_stats_path)
        for candidate in local_stats_paths:
            if _is_file_accessible(candidate):
                return json.loads(candidate.read_text(encoding="utf-8"))

        if not self.spec.stats_path:
            if self.is_v3:
                stats = _json_loads(self._read_bytes("meta/stats.json"))
            else:
                raise FileNotFoundError(f"dataset {self.name} has no stats_path or local_stats_path configured")
        else:
            stats_path = str(self.spec.stats_path)
            if _is_remote_url(stats_path):
                raise ValueError(
                    f"dataset {self.name} stats_path must be a local file, got {stats_path}"
                )
            local_candidate = None
            for candidate in _local_path_candidates(stats_path):
                if _is_file_accessible(candidate):
                    local_candidate = candidate
                    break
            if local_candidate is not None:
                stats = json.loads(local_candidate.read_text(encoding="utf-8"))
            else:
                stats = _json_loads(self._read_bytes(stats_path))
        return stats

    def _read_jsonl(self, rel_path: str) -> list[dict[str, Any]]:
        return self._read_decoded_json_with_retries(rel_path, _jsonl_loads)

    @staticmethod
    def _task_text_from_metadata(item: dict[str, Any]) -> str | None:
        def normalize(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, float) and math.isnan(value):
                return None
            if isinstance(value, (list, tuple)):
                for child in value:
                    text = normalize(child)
                    if text:
                        return text
                return None
            if isinstance(value, dict):
                for key in ("task", "instruction", "language_instruction", "prompt", "description", "text"):
                    text = normalize(value.get(key))
                    if text:
                        return text
                return None
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "nan"}:
                return None
            return text

        for key in (
            "task",
            "language_instruction",
            "instruction",
            "prompt",
            "description",
            "text",
            "__index_level_0__",
        ):
            text = normalize(item.get(key))
            if text:
                return text
        return None

    def _fallback_task_text(self) -> str:
        return ""

    def _read_tasks(self) -> dict[int, str]:
        if self.is_v3:
            try:
                raw = self._read_bytes(V3_TASKS_PATH)
            except FileNotFoundError:
                records = self._read_jsonl("meta/tasks.jsonl")
            else:
                t0 = time.perf_counter()
                table = pq.read_table(BytesIO(raw))
                records = table.to_pylist()
                _profile_record("tasks.read_parquet", time.perf_counter() - t0, records=len(records), bytes=len(raw))
        else:
            records = self._read_jsonl("meta/tasks.jsonl")

        t0 = time.perf_counter()
        out: dict[int, str] = {}
        for item in records:
            if "task_index" not in item:
                raise KeyError(f"task metadata row missing task_index: keys={list(item)}")
            task = self._task_text_from_metadata(item)
            task_index = int(item["task_index"])
            if task is None:
                task = self._fallback_task_text()
            out[task_index] = task
        out = dict(sorted(out.items()))
        _profile_record("tasks.sort_index", time.perf_counter() - t0, records=len(records))
        return out

    def _text_embedding_cache_paths(self, prompt: str) -> list[Path]:
        if not self.spec.local_text_embedding_cache_dir:
            raise ValueError(f"dataset {self.name} has no local_text_embedding_cache_dir configured")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{hashed}.t5_len{int(self.spec.context_len)}.{self.spec.text_encoder_id}.pt"
        return [path / filename for path in _local_path_candidates(self.spec.local_text_embedding_cache_dir)]

    def _text_embedding_cache_path(self, prompt: str) -> Path:
        paths = self._text_embedding_cache_paths(prompt)
        for path in paths:
            if _is_file_accessible(path):
                return path
        return paths[0]

    def iter_text_tasks_for_cache(self) -> list[str]:
        tasks: list[str] = []
        seen: set[str] = set()
        for task in self.tasks.values():
            text = str(task).strip()
            if text not in seen:
                seen.add(text)
                tasks.append(text)
        for segments in self.subtask_action_segments.values():
            for _, _, task in segments:
                text = str(task).strip()
                if text and text not in seen:
                    seen.add(text)
                    tasks.append(text)
        return tasks

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._text_embedding_cache_path(prompt)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run tools/pretrain_text_cache.py first or set local_text_embedding_cache_dir."
            )
        t0 = time.perf_counter()
        payload = torch.load(cache_path, map_location="cpu")
        _profile_record("text_cache.torch_load", time.perf_counter() - t0, path=cache_path.name)
        context = payload["context"].clone()
        context_mask = payload["mask"].bool().clone()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L, D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != int(self.spec.context_len):
            raise ValueError(f"Cached context_len mismatch: expected {self.spec.context_len}, got {context.shape[0]} in {cache_path}")
        if context_mask.shape[0] != int(self.spec.context_len):
            raise ValueError(f"Cached mask_len mismatch: expected {self.spec.context_len}, got {context_mask.shape[0]} in {cache_path}")
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    def _read_episodes(self) -> dict[int, dict[str, Any]]:
        if self.is_v3:
            records = self._read_v3_episode_records()
        else:
            records = self._read_jsonl("meta/episodes.jsonl")
        t0 = time.perf_counter()
        out = {int(item["episode_index"]): item for item in sorted(records, key=lambda x: int(x["episode_index"]))}
        _profile_record("episodes.sort_index", time.perf_counter() - t0, records=len(records))
        return out

    def _read_v3_episode_records(self) -> list[dict[str, Any]]:
        target = int(self.info.get("total_episodes") or 0)
        template = str(self.info.get("episodes_path") or V3_EPISODES_PATH)
        records: list[dict[str, Any]] = []
        file_ordinal = 0
        while True:
            if target > 0 and len(records) >= target:
                break
            if target > 0 and file_ordinal >= target:
                raise FileNotFoundError(
                    f"Could not collect all v3 episode metadata for {self.name}: "
                    f"got {len(records)}/{target} records before scanning {file_ordinal} files"
                )
            chunk_index = file_ordinal // self.chunks_size
            file_index = file_ordinal % self.chunks_size
            rel_path = template.format(chunk_index=chunk_index, file_index=file_index, episode_chunk=chunk_index)
            try:
                raw = self._read_bytes(rel_path)
            except FileNotFoundError:
                if records and target == 0:
                    break
                if records and target > 0 and len(records) >= target:
                    break
                raise
            t0 = time.perf_counter()
            table = pq.read_table(BytesIO(raw))
            chunk_records = table.to_pylist()
            records.extend(chunk_records)
            _profile_record(
                "episodes.read_parquet",
                time.perf_counter() - t0,
                records=len(chunk_records),
                bytes=len(raw),
                file=rel_path.rsplit("/", 1)[-1],
            )
            file_ordinal += 1
        return records[:target] if target > 0 else records

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0].item() if value.size else default
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def _v3_data_rel_path_for_file_ordinal(self, episode_index: int, file_ordinal: int) -> str:
        template = self.info["data_path"]
        file_ordinal = max(0, int(file_ordinal))
        chunk_index = file_ordinal // self.chunks_size
        file_index = file_ordinal % self.chunks_size
        return template.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )

    def _v3_episode_data_bounds(self, episode_index: int) -> tuple[int, int] | None:
        episode = self.episodes_dict[int(episode_index)]
        if "data_from_index" in episode and "data_to_index" in episode:
            return int(episode["data_from_index"]), int(episode["data_to_index"])
        start = self._scalar(episode.get("dataset_from_index"), episode.get("data/from_index", episode.get("data/from")))
        end = self._scalar(episode.get("dataset_to_index"), episode.get("data/to_index", episode.get("data/to")))
        if start is None or end is None:
            return None
        return int(start), int(end)

    def _cache_v3_data_file_range(self, file_ordinal: int, value: tuple[int, int] | None) -> None:
        self._v3_data_file_range_cache[int(file_ordinal)] = value
        while len(self._v3_data_file_range_cache) > self._v3_data_file_range_cache_max:
            self._v3_data_file_range_cache.pop(next(iter(self._v3_data_file_range_cache)))

    def _v3_data_file_ordinal_from_episode(self, episode_index: int) -> int:
        episode = self.episodes_dict[int(episode_index)]
        if "data_file_ordinal" in episode:
            return int(episode["data_file_ordinal"])
        chunk_index = int(self._scalar(episode.get("data/chunk_index"), 0))
        file_index = int(self._scalar(episode.get("data/file_index"), 0))
        return chunk_index * self.chunks_size + file_index

    def _v3_data_file_index_range(self, episode_index: int, file_ordinal: int) -> tuple[int, int] | None:
        file_ordinal = int(file_ordinal)
        if file_ordinal in self._v3_data_file_range_cache:
            return self._v3_data_file_range_cache[file_ordinal]
        rel_path = self._v3_data_rel_path_for_file_ordinal(episode_index, file_ordinal)
        try:
            raw = self._read_bytes(rel_path)
        except FileNotFoundError:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        parquet_file = pq.ParquetFile(BytesIO(raw))
        if "index" not in parquet_file.schema_arrow.names:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        table = parquet_file.read(columns=["index"])
        if table.num_rows == 0:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        values = table["index"].to_numpy(zero_copy_only=False)
        out = (int(values.min()), int(values.max()) + 1)
        self._cache_v3_data_file_range(file_ordinal, out)
        return out

    def _v3_find_data_rel_path_by_index_range(self, episode_index: int, *, max_scan_files: int = 128) -> str | None:
        bounds = self._v3_episode_data_bounds(episode_index)
        if bounds is None:
            return None
        start, end = bounds
        base_ordinal = self._v3_data_file_ordinal_from_episode(episode_index)
        for file_ordinal in range(base_ordinal, base_ordinal + max_scan_files):
            file_range = self._v3_data_file_index_range(episode_index, file_ordinal)
            if file_range is None:
                if file_ordinal > base_ordinal:
                    break
                continue
            low, high = file_range
            if high <= start:
                continue
            if low >= end:
                break
            return self._v3_data_rel_path_for_file_ordinal(episode_index, file_ordinal)
        return None

    def _data_rel_path(self, episode_index: int) -> str:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            if episode.get("data_path"):
                return str(episode["data_path"])
            if episode.get("data_rel_path"):
                return str(episode["data_rel_path"])
        template = self.info["data_path"]
        if self.is_v3:
            if episode is None:
                raise KeyError(f"Missing episode metadata for {episode_index}")
            chunk_index = int(self._scalar(episode.get("data/chunk_index"), 0))
            file_index = int(self._scalar(episode.get("data/file_index"), 0))
            return template.format(
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index,
            )
        return template.format(episode_chunk=self._episode_chunk(episode_index), episode_index=episode_index)

    def _local_data_path(self, rel_or_remote_path: str) -> Path | None:
        root = self._dataset_local_root()
        if root is None:
            return None
        path = str(rel_or_remote_path)
        if path.startswith("@data/"):
            return Path(self._remote_path(path))
        remote_prefix = self.remote_root.rstrip("/") + "/"
        if path.startswith(remote_prefix):
            path = path[len(remote_prefix) :]
        elif _is_remote_url(path):
            raise ValueError(f"Expected a local filesystem path, got {path!r}")
        local_path = Path(path).expanduser()
        return local_path if local_path.is_absolute() else root / local_path

    def _video_rel_path(self, episode_index: int, video_key: str) -> str:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            paths = episode.get("video_paths")
            if isinstance(paths, dict) and video_key in paths:
                return str(paths[video_key])
            rel_paths = episode.get("video_rel_paths")
            if isinstance(rel_paths, dict) and video_key in rel_paths:
                return str(rel_paths[video_key])
        template = self.info["video_path"]
        if self.is_v3:
            if episode is None:
                raise KeyError(f"Missing episode metadata for {episode_index}")
            chunk_index = int(self._scalar(episode.get(f"videos/{video_key}/chunk_index"), episode.get("video/chunk_index", 0)))
            file_index = int(self._scalar(episode.get(f"videos/{video_key}/file_index"), episode.get("video/file_index", 0)))
            return template.format(
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index,
                video_key=video_key,
            )
        return template.format(
            episode_chunk=self._episode_chunk(episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )

    def _needed_parquet_columns(self) -> list[str]:
        columns = [
            *self.state_keys,
            *self.action_keys,
            *_adapter_columns(self.canonical_adapter),
            self.timestamp_key,
            self.frame_index_key,
            self.task_index_key,
        ]
        if self.is_v3:
            columns.extend(["episode_index", "index"])
        semantics = self.canonical_adapter.get("required_semantics")
        if isinstance(semantics, dict):
            columns.append(str(semantics.get("key", "ego_coordinate_semantics")))
        out: list[str] = []
        for col in columns:
            if col and col not in out:
                out.append(col)
        return out

    def _slice_episode_table(self, table: Any, episode_index: int) -> Any:
        if not self.is_v3:
            return table

        source_episode_index = self.source_episode_index(int(episode_index))
        bounds = self._v3_episode_data_bounds(int(episode_index))
        if bounds is not None and "index" in table.column_names:
            start, end = bounds
            mask = pc.and_(pc.greater_equal(table["index"], int(start)), pc.less(table["index"], int(end)))
            episode_table = table.filter(mask)
            if episode_table.num_rows > 0:
                return episode_table

        if "episode_index" in table.column_names:
            return table.filter(pc.equal(table["episode_index"], int(source_episode_index)))
        return table

    def _load_episode_parquet(self, episode_index: int) -> dict[str, torch.Tensor | list[Any]]:
        rel_path = self._data_rel_path(episode_index)

        def read_episode_table(path: str) -> tuple[Any, int, int]:
            local_path = self._local_data_path(path)
            t0 = time.perf_counter()
            if local_path is None or not _is_file_accessible(local_path):
                raise FileNotFoundError(
                    f"Dataset {self.name} local parquet not found: rel={path} local={local_path}"
                )
            parquet_file = pq.ParquetFile(local_path)
            num_bytes = int(local_path.stat().st_size)
            available_columns = set(parquet_file.schema_arrow.names)
            columns = [col for col in self._needed_parquet_columns() if col in available_columns]
            _profile_record(
                "parquet.open",
                time.perf_counter() - t0,
                episode=episode_index,
                bytes=num_bytes,
                columns=len(columns),
                path=path,
            )
            t0 = time.perf_counter()
            row_group = self.episodes_dict[int(episode_index)].get("data_row_group")
            if row_group is None:
                table = parquet_file.read(columns=columns)
            else:
                row_group = int(row_group)
                if not 0 <= row_group < parquet_file.num_row_groups:
                    raise IndexError(
                        f"Dataset {self.name} episode {episode_index} references "
                        f"row group {row_group}, but {local_path} contains "
                        f"{parquet_file.num_row_groups} row groups."
                    )
                table = parquet_file.read_row_group(row_group, columns=columns)
            table = self._slice_episode_table(table, int(episode_index))
            _profile_record(
                "parquet.read_table",
                time.perf_counter() - t0,
                episode=episode_index,
                bytes=num_bytes,
                columns=len(columns),
                rows=table.num_rows,
                path=path,
            )
            return table, num_bytes, len(columns)

        table, _, _ = read_episode_table(rel_path)
        if table.num_rows == 0 and self.is_v3:
            resolved_rel_path = self._v3_find_data_rel_path_by_index_range(int(episode_index))
            if resolved_rel_path and resolved_rel_path != rel_path:
                rel_path = resolved_rel_path
                table, _, _ = read_episode_table(rel_path)

        if table.num_rows == 0:
            raise RuntimeError(f"No rows for episode {episode_index} in {rel_path}")
        t0 = time.perf_counter()
        data = _table_to_tensors(table)
        _profile_record("parquet.to_tensors", time.perf_counter() - t0, episode=episode_index, rows=table.num_rows)
        return data

    def _open_episode_video(
        self,
        episode_index: int,
        video_key: str,
        *,
        block_cache=None,
    ) -> io.BufferedReader:
        del block_cache
        rel_path = self._video_rel_path(episode_index, video_key)
        local_path = self._local_data_path(rel_path)
        if local_path is None or not _is_file_accessible(local_path):
            raise FileNotFoundError(
                f"Dataset {self.name} local video not found: rel={rel_path} local={local_path}"
            )
        return local_path.open("rb")

    def _episode_video_block_cache(
        self, episode_index: int, video_key: str
    ):
        del episode_index, video_key
        return None

    def _episode_task(self, episode_index: int, frame_idx: int, parquet: dict[str, torch.Tensor | list[Any]]) -> tuple[int, str]:
        subtask = self._subtask_for_source_frame(int(episode_index), int(frame_idx))
        if subtask is not None:
            _, _, task = subtask
            return int(self.task_to_index.get(task, -1)), task

        episode = self.episodes_dict[int(episode_index)]
        if self.task_index_authority == "episode_metadata":
            value = episode.get("task_index")
            if value is not None:
                task_index = int(self._scalar(value))
                return task_index, self.tasks[task_index]
            if self.task_index_key in parquet:
                task_index = int(_select_rows(parquet, self.task_index_key, [frame_idx])[0].item())
                return task_index, self.tasks[task_index]
        else:
            if self.task_index_key in parquet:
                task_index = int(_select_rows(parquet, self.task_index_key, [frame_idx])[0].item())
                return task_index, self.tasks[task_index]
            value = episode.get("task_index")
            if value is not None:
                task_index = int(self._scalar(value))
                return task_index, self.tasks[task_index]
        task = episode.get("task")
        if isinstance(task, str):
            return int(self.task_to_index.get(task, -1)), task
        for key in ("tasks_index", "task_indices"):
            value = self._scalar(episode.get(key))
            if value is not None:
                task_index = int(value)
                return task_index, self.tasks[task_index]

        tasks = episode.get("tasks")
        if isinstance(tasks, str):
            return int(self.task_to_index.get(tasks, -1)), tasks
        if isinstance(tasks, (list, tuple)) and tasks:
            task = str(tasks[0])
            return int(self.task_to_index.get(task, -1)), task
        raise KeyError(f"Could not resolve task for episode {episode_index}")

    def _video_timestamp_offset(self, episode_index: int, video_key: str) -> float:
        episode = self.episodes_dict[int(episode_index)]
        offsets = episode.get("video_timestamp_offsets")
        if isinstance(offsets, dict) and video_key in offsets:
            return float(offsets[video_key])
        if not self.is_v3:
            return 0.0
        for key in (f"videos/{video_key}/from_timestamp", f"videos/{video_key}/start_timestamp", "from_timestamp"):
            value = self._scalar(episode.get(key))
            if value is not None:
                return float(value)
        return 0.0

    def resolve_video_timestamps(
        self,
        item: dict[str, Any],
        video_key: str,
        sample_indices: list[int] | tuple[int, ...] | None = None,
    ) -> list[float]:
        """Resolve video-relative timestamps without changing frame semantics.

        Most sources contain usable episode-relative timestamps and retain the
        historical ``timestamp`` behavior.  A few converted sources explicitly
        opt into ``frame_index`` because their parquet timestamp was stored as a
        float32 Unix epoch: sub-second increments were destroyed and seeking by
        that value scans the whole MP4.  For those sources ``frame_index / fps``
        is the authoritative LeRobot frame mapping.
        """

        if self.video_timestamp_source == "frame_index":
            frame_index_key = (
                "retimed_frame_index"
                if "retimed_frame_index" in item
                else "frame_index"
            )
            if frame_index_key not in item:
                raise KeyError(
                    f"Dataset {self.name} requires frame_index for video decode."
                )
            frame_indices = [
                float(x) for x in item[frame_index_key].reshape(-1).tolist()
            ]
            timestamps = [idx / float(self.fps) for idx in frame_indices]
        elif "timestamp" in item:
            timestamps = [
                float(x) for x in item["timestamp"].reshape(-1).tolist()
            ]
        else:
            frame_indices = [
                int(x) for x in item["frame_index"].reshape(-1).tolist()
            ]
            timestamps = [idx / float(self.fps) for idx in frame_indices]

        if sample_indices is not None:
            timestamps = [
                timestamps[int(idx)]
                for idx in sample_indices
                if 0 <= int(idx) < len(timestamps)
            ]
        offset = self._video_timestamp_offset(
            int(item["episode_index"].item()), video_key
        )
        if offset:
            timestamps = [offset + ts for ts in timestamps]
        return timestamps


    def _global_to_episode_frame(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of bounds for dataset length {len(self)}")
        pos = bisect.bisect_right(self.cumulative_lengths, idx)
        start = 0 if pos == 0 else self.cumulative_lengths[pos - 1]
        return self.selected_episodes[pos], idx - start

    def _query_indices(self, frame_idx: int, episode_len: int, horizon: int) -> tuple[list[int], torch.Tensor]:
        raw = [frame_idx + i * self.global_sample_stride for i in range(horizon)]
        clamped = [max(0, min(episode_len - 1, i)) for i in raw]
        is_pad = torch.as_tensor([i < 0 or i >= episode_len for i in raw], dtype=torch.bool)
        return clamped, is_pad

    def _query_indices_batch(
        self,
        starts: torch.Tensor,
        episode_len: int,
        horizon: int,
    ) -> tuple[list[int], torch.Tensor]:
        offsets = torch.arange(int(horizon), dtype=torch.long) * int(self.global_sample_stride)
        raw = starts.to(torch.long)[:, None] + offsets[None, :]
        is_pad = (raw < 0) | (raw >= int(episode_len))
        clamped = raw.clamp(min=0, max=max(0, int(episode_len) - 1))
        return clamped.reshape(-1).tolist(), is_pad

    def _slow_motion_time_query(
        self,
        starts: torch.Tensor,
        segment_end: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = torch.arange(int(horizon), dtype=torch.float64)
        raw = starts.to(torch.float64)[:, None] + steps[None, :] / self.slow_motion_factor
        is_pad = (raw < 0.0) | (raw > float(segment_end - 1))
        clamped = raw.clamp(min=0.0, max=float(segment_end - 1))
        lower = torch.floor(clamped).to(torch.long)
        alpha = clamped - lower.to(torch.float64)
        alpha = torch.where(alpha.abs() < 1e-10, torch.zeros_like(alpha), alpha)
        upper = lower + (alpha > 0.0).to(torch.long)
        return raw, clamped, lower, upper, alpha.to(torch.float32), is_pad

    def _project_retimed_canonical_states(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.canonical_adapter.get("source_layout") == "wrist_only_compact_v1":
            return self._project_retimed_wrist_states(
                parquet, episode_index, lower, upper, alpha
            )
        return self._project_retimed_general_states(
            parquet, episode_index, lower, upper, alpha
        )

    def _project_retimed_canonical_actions(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        first_local: torch.Tensor,
        second_local: torch.Tensor,
        alpha_start: torch.Tensor,
        alpha_end: torch.Tensor,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        start_state_mask: torch.Tensor,
        end_state_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.canonical_adapter.get("source_layout") == "wrist_only_compact_v1":
            return self._project_retimed_wrist_actions(
                parquet,
                episode_index,
                first_local,
                second_local,
                alpha_start,
                alpha_end,
                start_state,
                end_state,
                start_state_mask,
                end_state_mask,
            )
        return self._project_retimed_general_actions(
            parquet,
            episode_index,
            first_local,
            second_local,
            alpha_start,
            alpha_end,
            start_state,
            end_state,
            start_state_mask,
            end_state_mask,
        )

    def _slow_motion_canonical_windows(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        starts: torch.Tensor,
        segment_end: int,
        episode_len: int,
    ) -> dict[str, torch.Tensor]:
        _, source_times, lower, upper, alpha, state_pad = self._slow_motion_time_query(
            starts, segment_end, self.num_frames_per_sample
        )
        num_windows = int(starts.numel())
        state, state_dim_is_pad, state_mask = self._project_retimed_canonical_states(
            parquet,
            episode_index,
            lower.reshape(-1),
            upper.reshape(-1),
            alpha.reshape(-1),
        )
        state = state.reshape(num_windows, self.num_frames_per_sample, -1)
        state_mask = state_mask.reshape(num_windows, self.num_frames_per_sample, -1)

        first_local = lower[:, : self.action_size]
        second_local = lower[:, 1 : self.action_size + 1]
        action, action_dim_is_pad, action_mask = self._project_retimed_canonical_actions(
            parquet,
            episode_index,
            first_local.reshape(-1),
            second_local.reshape(-1),
            alpha[:, : self.action_size].reshape(-1),
            alpha[:, 1 : self.action_size + 1].reshape(-1),
            state[:, : self.action_size].reshape(-1, state.shape[-1]),
            state[:, 1 : self.action_size + 1].reshape(-1, state.shape[-1]),
            state_mask[:, : self.action_size].reshape(-1, state_mask.shape[-1]),
            state_mask[:, 1 : self.action_size + 1].reshape(-1, state_mask.shape[-1]),
        )
        action = action.reshape(num_windows, self.action_size, -1)
        action_mask = action_mask.reshape(num_windows, self.action_size, -1)
        action_pad = state_pad[:, : self.action_size] | state_pad[:, 1 : self.action_size + 1]


        return {
            "source_times": source_times,
            "lower": lower,
            "upper": upper,
            "alpha": alpha,
            "state": state,
            "state_mask": state_mask,
            "state_is_pad": state_pad,
            "state_dim_is_pad": state_dim_is_pad,
            "action": action,
            "action_mask": action_mask,
            "action_is_pad": action_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    def _interpolate_parquet_column(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        key: str,
        episode_index: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        current = _select_rows(
            parquet, key, self.source_frame_indices(episode_index, lower)
        )
        future = _select_rows(
            parquet, key, self.source_frame_indices(episode_index, upper)
        )
        if not isinstance(current, torch.Tensor) or not isinstance(future, torch.Tensor):
            raise TypeError(f"Column {key!r} must be numeric for slow motion.")
        blend = alpha.to(torch.float64)
        while blend.ndim < current.ndim:
            blend = blend.unsqueeze(-1)
        return torch.lerp(current.to(torch.float64), future.to(torch.float64), blend)

    def _slow_motion_unique_stat_rows(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        starts: torch.Tensor,
        segment_end: int,
        episode_len: int,
        *,
        include_state: bool,
    ) -> dict[str, torch.Tensor | None]:
        _, _, lower, upper, alpha, state_pad = self._slow_motion_time_query(
            starts, segment_end, self.num_frames_per_sample
        )
        num_windows = int(starts.numel())
        out: dict[str, torch.Tensor | None] = {
            "num_windows": torch.tensor(num_windows, dtype=torch.long)
        }

        if include_state:
            state_steps = torch.arange(self.num_frames_per_sample, dtype=torch.long)
            state_steps = state_steps[None, :].expand(num_windows, -1)
            state_signatures = torch.stack((lower, upper, state_steps), dim=-1)
            unique_state, state_weights = torch.unique(
                state_signatures[~state_pad],
                dim=0,
                return_counts=True,
            )
            state_alpha = alpha[0].index_select(0, unique_state[:, 2])
            state, state_dim_is_pad, state_mask = self._project_retimed_canonical_states(
                parquet,
                episode_index,
                unique_state[:, 0],
                unique_state[:, 1],
                state_alpha,
            )
            state_time_is_pad = ~torch.isfinite(state).all(dim=-1)
            state = torch.nan_to_num(state)
            state_mask &= ~state_time_is_pad[:, None]
            state_mask &= ~state_dim_is_pad[None, :]
            state[~state_mask] = 0.0
            out.update(
                {
                    "state": state,
                    "state_mask": state_mask,
                    "state_is_pad": state_time_is_pad,
                    "state_dim_is_pad": state_dim_is_pad,
                    "state_weights": state_weights,
                }
            )

        first_local = lower[:, : self.action_size]
        second_local = lower[:, 1 : self.action_size + 1]
        action_pad = state_pad[:, : self.action_size] | state_pad[:, 1 : self.action_size + 1]

        action_steps = torch.arange(self.action_size, dtype=torch.long)
        action_steps = action_steps[None, :].expand(num_windows, -1)
        action_signatures = torch.stack(
            (first_local, second_local, action_steps), dim=-1
        )
        unique_action, action_weights = torch.unique(
            action_signatures[~action_pad],
            dim=0,
            return_counts=True,
        )
        action_step = unique_action[:, 2]
        alpha_start = alpha[0].index_select(0, action_step)
        alpha_end = alpha[0].index_select(0, action_step + 1)
        start_upper = unique_action[:, 0] + (alpha_start > 0.0).to(torch.long)
        end_upper = unique_action[:, 1] + (alpha_end > 0.0).to(torch.long)
        start_state, _, start_state_mask = self._project_retimed_canonical_states(
            parquet,
            episode_index,
            unique_action[:, 0],
            start_upper,
            alpha_start,
        )
        end_state, _, end_state_mask = self._project_retimed_canonical_states(
            parquet,
            episode_index,
            unique_action[:, 1],
            end_upper,
            alpha_end,
        )
        action, action_dim_is_pad, action_mask = self._project_retimed_canonical_actions(
            parquet,
            episode_index,
            unique_action[:, 0],
            unique_action[:, 1],
            alpha_start,
            alpha_end,
            start_state,
            end_state,
            start_state_mask,
            end_state_mask,
        )

        action_time_is_pad = ~torch.isfinite(action).all(dim=-1).cpu()
        action = torch.nan_to_num(action)
        action_mask &= ~action_time_is_pad[:, None]
        action_mask &= ~action_dim_is_pad[None, :]
        action[~action_mask] = 0.0
        pair_indices = torch.stack(
            (
                torch.as_tensor(
                    self.source_frame_indices(episode_index, unique_action[:, 0]),
                    dtype=torch.long,
                ),
                torch.as_tensor(
                    self.source_frame_indices(episode_index, unique_action[:, 1]),
                    dtype=torch.long,
                ),
            ),
            dim=-1,
        )
        out.update(
            {
                "action": action,
                "action_mask": action_mask,
                "action_is_pad": action_time_is_pad,
                "action_dim_is_pad": action_dim_is_pad,
                "action_weights": action_weights,
                "action_pair_indices": pair_indices,
            }
        )
        return out

    def _canonical_component_tensor(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
    ) -> torch.Tensor | None:
        key = component.get(f"{kind}_key") or component.get("key")
        if not key:
            return None
        value = _select_feature(parquet, str(key), indices, _component_raw_slice(component, kind))
        scale = component.get(f"{kind}_scale")
        offset = component.get(f"{kind}_offset")
        if scale is not None:
            value = value * torch.as_tensor(scale, dtype=value.dtype, device=value.device)
        if offset is not None:
            value = value + torch.as_tensor(offset, dtype=value.dtype, device=value.device)
        return value

    def _canonical_pose_input(self, component: dict[str, Any], kind: str) -> str:
        return str(
            component.get(f"{kind}_pose_input")
            or component.get("pose_input")
            or self.canonical_adapter.get(f"{kind}_pose_input")
            or (self.canonical_adapter.get("action_pose_input") if kind == "action" else None)
            or self.canonical_adapter.get("pose_input")
            or "xyz_rotvec"
        )

    def _canonical_frame_transform(self, component: dict[str, Any], kind: str) -> dict[str, Any]:
        return _merge_frame_transform(self.canonical_adapter, component, kind)

    def _canonical_transform_pose(
        self,
        pose: torch.Tensor,
        component: dict[str, Any],
        kind: str,
        pose_input: str,
        *,
        is_delta: bool | None = None,
    ) -> torch.Tensor:
        if is_delta is None:
            is_delta = _pose_is_delta(kind, pose_input, self.canonical_adapter, component)
        return _transform_xyz_rotvec_pose(
            pose,
            self._canonical_frame_transform(component, kind),
            is_delta=bool(is_delta),
        )

    def _canonical_absolute_pose_from_component(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
    ) -> torch.Tensor | None:
        pose_input = self._canonical_pose_input(component, kind)
        fk = component.get("fk")
        if isinstance(fk, dict):
            key = fk.get(f"{kind}_joint_key") or fk.get("joint_key")
            if not key:
                return None
            raw_slice = fk.get(f"{kind}_raw_slice") or fk.get("raw_slice")
            joint_values = _select_feature(parquet, str(key), indices, raw_slice)
            pose = _urdf_fk_xyz_rotvec(joint_values, fk)
            return self._canonical_transform_pose(
                pose,
                component,
                kind,
                "xyz_rotvec",
                is_delta=False,
            )

        direct = self._canonical_component_tensor(parquet, component, indices, kind)
        if direct is not None:
            pose = _pose_to_xyz_rotvec(direct, pose_input)
            return self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)

        pos_key = component.get(f"{kind}_position_key") or component.get("position_key")
        rot_key = (
            component.get(f"{kind}_rotation_key")
            or component.get(f"{kind}_orientation_key")
            or component.get("rotation_key")
            or component.get("orientation_key")
        )
        if not pos_key or not rot_key:
            return None

        pos_slice = component.get(f"{kind}_position_slice") or component.get("position_slice")
        rot_slice = (
            component.get(f"{kind}_rotation_slice")
            or component.get(f"{kind}_orientation_slice")
            or component.get("rotation_slice")
            or component.get("orientation_slice")
        )
        pos = _select_feature(parquet, str(pos_key), indices, pos_slice)
        rot = _select_feature(parquet, str(rot_key), indices, rot_slice)
        if pos.shape[-1] != 3:
            raise ValueError(f"Canonical ee position expects 3 dims from {pos_key!r}, got {pos.shape[-1]}")
        if rot.shape[-1] == 4:
            rotvec = _quat_to_rotvec_xyzw(rot) if "xyzw" in pose_input else _quat_to_rotvec_wxyz(rot)
        elif rot.shape[-1] == 3:
            rotvec = _rpy_to_rotvec(rot)
        else:
            raise ValueError(f"Canonical ee rotation expects 3 or 4 dims from {rot_key!r}, got {rot.shape[-1]}")
        pose = torch.cat([pos, rotvec], dim=-1)
        return self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)

    def _canonical_pose_from_component(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
        next_indices: list[int] | None = None,
        *,
        regenerate_state_delta_common: bool = True,
    ) -> torch.Tensor | None:
        pose_input = self._canonical_pose_input(component, kind)
        selected_indices = indices
        if kind == "action" and component.get("action_from_next_action"):
            if next_indices is None:
                raise ValueError(f"canonical_adapter for {self.name} uses action_from_next_action without next_indices")
            selected_indices = next_indices

        if kind == "action":
            if (
                regenerate_state_delta_common
                and _uses_state_delta_common(self.canonical_adapter, component)
            ):
                if next_indices is None:
                    raise ValueError(
                        f"canonical_adapter for {self.name} uses "
                        "state_delta_common without next_indices"
                    )
                current_pose = self._canonical_absolute_pose_from_component(parquet, component, indices, "state")
                future_pose = self._canonical_absolute_pose_from_component(parquet, component, next_indices, "state")
                if current_pose is None or future_pose is None:
                    return None
                pose = _pose_delta_world_from_transformed_xyz_rotvec(current_pose, future_pose)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

            semantic = _pose_semantics(kind, self.canonical_adapter, component)
            if semantic in {"absolute", "absolute_target", "target", "target_pose"}:
                current_pose = self._canonical_absolute_pose_from_component(parquet, component, indices, "state")
                target_pose = self._canonical_absolute_pose_from_component(parquet, component, selected_indices, "action")
                if current_pose is None or target_pose is None:
                    return None
                pose = _pose_delta_world_from_transformed_xyz_rotvec(current_pose, target_pose)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

            state_delta_key = component.get("state_delta_key")
            if state_delta_key:
                if next_indices is None:
                    raise ValueError(f"canonical_adapter for {self.name} uses state_delta_key without next_indices")
                current = _select_feature(parquet, str(state_delta_key), indices, _component_raw_slice(component, "state"))
                future = _select_feature(parquet, str(state_delta_key), next_indices, _component_raw_slice(component, "state"))
                pose = _pose_delta_xyz_rotvec(current, future, pose_input)
                pose = self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=True)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

            state_future_key = component.get("state_future_key")
            if state_future_key:
                if next_indices is None:
                    raise ValueError(f"canonical_adapter for {self.name} uses state_future_key without next_indices")
                future = _select_feature(parquet, str(state_future_key), next_indices, _component_raw_slice(component, "state"))
                pose = _pose_to_xyz_rotvec(future, pose_input)
                pose = self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

        if isinstance(component.get("fk"), dict):
            pose = self._canonical_absolute_pose_from_component(
                parquet,
                component,
                selected_indices,
                kind,
            )
            if pose is None:
                return None
            return _format_xyz_rotvec_pose(
                pose,
                _pose_output_format(kind, self.canonical_adapter, component),
            )

        direct = self._canonical_component_tensor(parquet, component, selected_indices, kind)
        if direct is not None:
            pose = _pose_to_xyz_rotvec(direct, pose_input)
            pose = self._canonical_transform_pose(pose, component, kind, pose_input)
            return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

        pos_key = component.get(f"{kind}_position_key") or component.get("position_key")
        rot_key = (
            component.get(f"{kind}_rotation_key")
            or component.get(f"{kind}_orientation_key")
            or component.get("rotation_key")
            or component.get("orientation_key")
        )
        if not pos_key or not rot_key:
            return None

        pos_slice = component.get(f"{kind}_position_slice") or component.get("position_slice")
        rot_slice = (
            component.get(f"{kind}_rotation_slice")
            or component.get(f"{kind}_orientation_slice")
            or component.get("rotation_slice")
            or component.get("orientation_slice")
        )
        pos = _select_feature(parquet, str(pos_key), selected_indices, pos_slice)
        rot = _select_feature(parquet, str(rot_key), selected_indices, rot_slice)
        if pos.shape[-1] != 3:
            raise ValueError(f"Canonical ee position expects 3 dims from {pos_key!r}, got {pos.shape[-1]}")
        if rot.shape[-1] == 4:
            pose_input = str(self.canonical_adapter.get("pose_input") or "split_xyz_xyzw")
            rotvec = _quat_to_rotvec_xyzw(rot) if "xyzw" in pose_input else _quat_to_rotvec_wxyz(rot)
        elif rot.shape[-1] == 3:
            rotvec = _rpy_to_rotvec(rot)
        else:
            raise ValueError(f"Canonical ee rotation expects 3 or 4 dims from {rot_key!r}, got {rot.shape[-1]}")
        pose = torch.cat([pos, rotvec], dim=-1)
        pose = self._canonical_transform_pose(pose, component, kind, pose_input)
        return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

    def _canonical_project(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        indices: list[int],
        kind: str,
        next_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, dim_mask, _ = self._canonical_project_with_validity(
            parquet,
            indices,
            kind,
            next_indices,
        )
        return out, dim_mask

    def _action_dimension_loss_weight(self) -> torch.Tensor:
        weight = torch.full(
            (int(self.canonical_dim),),
            float(self.spec.action_loss_weight),
            dtype=torch.float32,
        )
        for slot in self.canonical_adapter["slots"].values():
            gripper = slot.get("gripper")
            if gripper is None:
                continue
            target_slice = (
                gripper.get("action_target_slice") or gripper["target_slice"]
            )
            start, end = map(int, target_slice)
            weight[start:end] = self.spec.action_gripper_loss_weight
        return weight

    def _canonical_project_with_validity(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        indices: list[int],
        kind: str,
        next_indices: list[int] | None = None,
        *,
        convert_wrist_action_translation: bool = True,
        regenerate_state_delta_common: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.use_canonical_adapter:
            raise RuntimeError("canonical projection requested without canonical_adapter")
        dim = int(self.canonical_dim)
        out = torch.zeros((len(indices), dim), dtype=torch.float32)
        dim_mask = torch.ones(dim, dtype=torch.bool)
        exact_validity = torch.zeros((len(indices), dim), dtype=torch.bool)
        slots = self.canonical_adapter.get("slots")
        if not isinstance(slots, dict):
            raise ValueError(f"canonical_adapter for {self.name} is missing slots")

        has_exact_validity = any(
            isinstance(component, dict)
            and bool(
                component.get(f"{kind}_validity_key")
                or component.get("validity_key")
            )
            for slot_cfg in slots.values()
            if isinstance(slot_cfg, dict)
            for component in slot_cfg.values()
        )

        skip_component_keys = {
            "pad",
            "target_slice",
            "state_target_slice",
            "action_target_slice",
            "notes",
            "note",
        }
        for slot_name, slot_cfg in slots.items():
            if not isinstance(slot_cfg, dict) or slot_cfg.get("pad") is True:
                continue
            for component_name, component in slot_cfg.items():
                if component_name in skip_component_keys:
                    continue
                component = slot_cfg.get(component_name)
                if not isinstance(component, dict) or component.get("pad") is True:
                    continue
                if component.get(f"{kind}_enabled", True) is False:
                    continue
                target_slice = component.get(f"{kind}_target_slice") or component.get("target_slice")
                if target_slice is None:
                    continue
                if component_name == "ee":
                    values = self._canonical_pose_from_component(
                        parquet,
                        component,
                        indices,
                        kind,
                        next_indices,
                        regenerate_state_delta_common=(
                            regenerate_state_delta_common
                        ),
                    )
                elif kind == "action" and component.get("action_from_next_state"):
                    if next_indices is None:
                        raise ValueError(f"canonical_adapter for {self.name} uses action_from_next_state without next_indices")
                    values = self._canonical_component_tensor(parquet, component, next_indices, "state")
                elif kind == "action" and component.get("action_from_next_action"):
                    if next_indices is None:
                        raise ValueError(f"canonical_adapter for {self.name} uses action_from_next_action without next_indices")
                    values = self._canonical_component_tensor(parquet, component, next_indices, "action")
                else:
                    values = self._canonical_component_tensor(parquet, component, indices, kind)
                if values is None:
                    continue
                _assign_target(out, dim_mask, target_slice, values)
                if has_exact_validity:
                    validity_key = component.get(
                        f"{kind}_validity_key"
                    ) or component.get("validity_key")
                    if validity_key:
                        validity_slice = (
                            component.get(f"{kind}_validity_raw_slice")
                            or component.get("validity_raw_slice")
                            or _component_raw_slice(component, kind)
                        )
                        selected = _select_rows(parquet, str(validity_key), indices)
                        if not isinstance(selected, torch.Tensor):
                            raise TypeError(
                                f"Column {validity_key!r} is not numeric and cannot be used as validity."
                            )
                        validity = _as_feature_matrix(
                            _slice_last_dim(selected, validity_slice), "validity"
                        ).to(torch.bool)
                        # A row-valid flag covers every feature of this slot.
                        if validity.shape[-1] == 1:
                            validity = validity.expand(-1, values.shape[-1])
                    else:
                        validity = torch.ones_like(values, dtype=torch.bool)
                    validity_dim_mask = torch.ones(dim, dtype=torch.bool)
                    _assign_target(
                        exact_validity,
                        validity_dim_mask,
                        target_slice,
                        validity,
                    )
        validity = exact_validity if has_exact_validity else None
        if (
            kind == "action"
            and str(
                self.canonical_adapter.get(
                    "action_pose_semantics", "stored_delta"
                )
            ).strip().lower()
            == "state_delta_common"
            and regenerate_state_delta_common
        ):
            if next_indices is None:
                raise ValueError(
                    f"Dataset {self.name} state_delta_common action semantics "
                    "requires next_indices."
                )
            start_state, _, start_state_validity = self._canonical_project_with_validity(
                parquet, indices, "state"
            )
            end_state, _, end_state_validity = self._canonical_project_with_validity(
                parquet, next_indices, "state"
            )
            out, validity = self._replace_wrist_actions_from_states(
                out,
                start_state,
                end_state,
                start_state_validity,
                end_state_validity,
                validity,
            )
        elif (
            kind == "action"
            and convert_wrist_action_translation
            and getattr(
                self,
                "action_translation_frame",
                str(
                    self.canonical_adapter.get(
                        "action_translation_frame", "source"
                    )
                ).strip().lower(),
            )
            == "start_observation_common"
        ):
            start_state, _, start_state_validity = self._canonical_project_with_validity(
                parquet,
                indices,
                "state",
            )
            out, validity = self._convert_wrist_action_translation(
                out,
                start_state,
                start_state_validity,
                validity,
            )
        return out, dim_mask, validity

    def _canonical_action_delta_from_states(
        self,
        current_state: torch.Tensor,
        future_state: torch.Tensor,
        dim_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "_canonical_action_delta_from_states is deprecated. "
            "Use _canonical_project(..., kind='action') so state/action slices can differ."
        )

    def __len__(self) -> int:
        return int(sum(self.episode_lengths))

    def valid_start_counts(self, allow_padding_at_end: bool = False) -> np.ndarray:
        cache = getattr(self, "_valid_start_counts_cache", None)
        if cache is None:
            cache = {}
            self._valid_start_counts_cache = cache
        cache_key = bool(allow_padding_at_end)
        if cache_key in cache:
            return cache[cache_key]

        if not self.subtask_action_segments:
            if allow_padding_at_end:
                counts = np.maximum(self.trajectory_lengths, 0).astype(np.int64)
            else:
                counts = np.maximum(
                    self.trajectory_lengths - int(self.max_delta_index), 0
                ).astype(np.int64)
        else:
            counts = []
            for trajectory_pos, episode_index in enumerate(self.trajectory_ids):
                episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
                intervals = self._valid_start_intervals_for_episode(
                    int(episode_index),
                    episode_len,
                    allow_padding_at_end,
                )
                counts.append(sum(end - start + 1 for start, end in intervals))
            counts = np.asarray(counts, dtype=np.int64)
        cache[cache_key] = counts
        return counts

    def max_start_for_trajectory_pos(self, trajectory_pos: int, allow_padding_at_end: bool = False) -> int:
        if not self.subtask_action_segments:
            length = int(self.trajectory_lengths[int(trajectory_pos)])
            if allow_padding_at_end:
                return length - 1
            return length - 1 - int(self.max_delta_index)
        episode_index = int(self.trajectory_ids[int(trajectory_pos)])
        episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
        intervals = self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            allow_padding_at_end,
        )
        return max((end for _, end in intervals), default=-1)

    def get_episode_data(self, selected_episode_pos: int) -> dict[str, torch.Tensor | list[Any]]:
        episode_index = self.selected_episodes[selected_episode_pos]
        return self._load_episode_parquet(episode_index)

    def get_step_item(
        self,
        episode_index: int,
        frame_idx: int,
        *,
        parquet: dict[str, torch.Tensor | list[Any]] | None = None,
        text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        episode_meta_len = int(self.episodes_dict[episode_index]["length"])
        if parquet is None:
            parquet = self._load_episode_parquet(episode_index)
        episode_start = self.episode_start_frame(int(episode_index))
        parquet_rows = _parquet_num_rows(parquet)
        episode_len = min(episode_meta_len, max(0, parquet_rows - episode_start))
        if episode_len <= 0:
            raise RuntimeError(f"Episode {episode_index} has no readable rows.")
        frame_idx = max(0, min(episode_len - 1, int(frame_idx)))

        segment_end = self._window_segment_end(
            int(episode_index), episode_len, frame_idx
        )

        retimed = None
        if self.slow_motion_factor != 1.0:
            retimed = self._slow_motion_canonical_windows(
                parquet,
                int(episode_index),
                torch.tensor([frame_idx], dtype=torch.long),
                segment_end,
                episode_len,
            )
            state = retimed["state"][0]
            action = retimed["action"][0]
            state_pad = retimed["state_is_pad"][0]
            action_pad = retimed["action_is_pad"][0]
            state_dim_is_pad = retimed["state_dim_is_pad"]
            action_dim_is_pad = retimed["action_dim_is_pad"]
            state_mask = retimed["state_mask"][0]
            action_mask = retimed["action_mask"][0]
            nearest_local = torch.floor(retimed["source_times"][0] + 0.5).to(
                torch.long
            )
            source_state_indices = self.source_frame_indices(
                episode_index, nearest_local
            )
        else:
            state_indices, state_pad = self._query_indices(
                frame_idx, segment_end, self.num_frames_per_sample
            )
            action_indices, action_pad = self._query_indices(
                frame_idx, segment_end, self.action_size
            )
            source_state_indices = self.source_frame_indices(episode_index, state_indices)
            source_action_indices = self.source_frame_indices(episode_index, action_indices)
            action_next_raw_indices = [
                idx + self.global_sample_stride for idx in action_indices
            ]
            action_next_pad = torch.as_tensor(
                [idx >= segment_end for idx in action_next_raw_indices],
                dtype=torch.bool,
            )
            action_next_local_indices = [
                min(segment_end - 1, idx) for idx in action_next_raw_indices
            ]
            source_action_next_indices = self.source_frame_indices(episode_index, action_next_local_indices)

            action_pad = action_pad | action_next_pad

            if self.use_canonical_adapter:
                state, state_dim_is_pad, state_mask = self._canonical_project_with_validity(
                    parquet, source_state_indices, "state"
                )
                action, action_dim_is_pad, action_mask = self._canonical_project_with_validity(
                    parquet,
                    source_action_indices,
                    "action",
                    source_action_next_indices,
                )
            else:
                state, state_dim_is_pad = _pad_last_dim(_select_rows_concat(parquet, self.state_keys, source_state_indices), self.state_target_dim)
                action, action_dim_is_pad = _pad_last_dim(_select_rows_concat(parquet, self.action_keys, source_action_indices), self.action_target_dim)
                state_mask = None
                action_mask = None

        state_nonfinite = ~torch.isfinite(state).all(dim=-1)
        action_nonfinite = ~torch.isfinite(action).all(dim=-1)
        state_pad = state_pad | state_nonfinite
        action_pad = action_pad | action_nonfinite
        state = torch.nan_to_num(state)
        action = torch.nan_to_num(action)
        if state_mask is not None:
            state_mask &= ~state_pad[:, None]
            state_mask &= ~state_dim_is_pad[None, :]
            state[~state_mask] = 0.0
        if action_mask is not None:
            action_mask &= ~action_pad[:, None]
            action_mask &= ~action_dim_is_pad[None, :]
            action[~action_mask] = 0.0

        item: dict[str, Any] = {
            "dataset_name": self.name,
            "embodiment": self.spec.embodiment,
            "control_schema": self.spec.control_schema or self.spec.embodiment,
            "stats_group": self.spec.stats_group or self.name,
            "canonical_action_layout": self.canonical_adapter.get("layout", ""),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "sample_frame_index": torch.tensor(frame_idx, dtype=torch.long),
            "source_episode_index": torch.tensor(self.source_episode_index(episode_index), dtype=torch.long),
            "source_frame_index": torch.tensor(self.source_frame_index(episode_index, frame_idx), dtype=torch.long),
            "observation.state": state,
            "action": action,
            "observation.state_is_pad": state_pad,
            "action_is_pad": action_pad,
            "observation.state_dim_is_pad": state_dim_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }
        if (
            self.spec.action_loss_weight != 1.0
            or self.spec.action_gripper_loss_weight != 1.0
        ):
            item["action_dim_loss_weight"] = (
                self._action_dimension_loss_weight()
            )
        if self.spec.action_loss_weight != 1.0:
            item["action_loss_weight"] = torch.tensor(
                self.spec.action_loss_weight, dtype=torch.float32
            )
        if self.action_loss_normalization == "weighted_valid_cells":
            item["action_loss_weighted_valid_cells"] = torch.tensor(True)
        if state_mask is not None:
            item["observation.state_mask"] = state_mask
        if action_mask is not None:
            item["action_mask"] = action_mask
        if self.timestamp_key in parquet:
            if retimed is None:
                item["timestamp"] = _select_rows(parquet, self.timestamp_key, source_state_indices)
            else:
                item["timestamp"] = self._interpolate_parquet_column(
                    parquet,
                    self.timestamp_key,
                    int(episode_index),
                    retimed["lower"][0],
                    retimed["upper"][0],
                    retimed["alpha"][0],
                )
        if self.frame_index_key in parquet:
            item["frame_index"] = _select_rows(parquet, self.frame_index_key, source_state_indices)
        else:
            item["frame_index"] = torch.as_tensor(source_state_indices, dtype=torch.long)
        if retimed is not None:
            item["retimed_frame_index"] = (
                retimed["source_times"][0] + float(episode_start)
            )
        task_index, task = self._episode_task(episode_index, self.source_frame_index(episode_index, frame_idx), parquet)
        item["task_index"] = torch.tensor(task_index, dtype=torch.long)
        item["task"] = task
        prompt = self.prompt_template.format(task=task)
        item["prompt"] = prompt
        if self.spec.local_text_embedding_cache_dir:
            cached_context = (
                text_context_cache.get(prompt)
                if text_context_cache is not None
                else None
            )
            if cached_context is None:
                cached_context = self._get_cached_text_context(
                    prompt,
                )
                if text_context_cache is not None:
                    text_context_cache[prompt] = cached_context
            context, context_mask = cached_context
            item["context"] = context
            item["context_mask"] = context_mask
        return item

    def iter_episode_stat_batches(
        self,
        episode_index: int,
        *,
        allow_padding_at_end: bool = False,
        max_windows_per_batch: int = 4096,
        max_start: int | None = None,
    ):
        """Yield canonical state/action windows for one episode in vectorized batches.

        This is intentionally image-free and prompt-free; it mirrors get_step_item's
        state/action construction for normalization stats without recomputing
        overlapping windows one start frame at a time.
        """

        episode_meta_len = int(self.episodes_dict[int(episode_index)]["length"])
        parquet = self._load_episode_parquet(int(episode_index))
        episode_start = self.episode_start_frame(int(episode_index))
        episode_len = min(episode_meta_len, max(0, _parquet_num_rows(parquet) - episode_start))
        if episode_len <= 0:
            raise RuntimeError(f"Episode {episode_index} has no readable rows.")

        valid_intervals = self._valid_start_intervals_for_episode(
            int(episode_index),
            episode_len,
            allow_padding_at_end,
        )
        if max_start is not None:
            limit = min(int(max_start), episode_len - 1)
            valid_intervals = tuple(
                (int(start), min(int(end), limit))
                for start, end in valid_intervals
                if int(start) <= limit
            )
        if not valid_intervals:
            return

        batch_size = max(1, int(max_windows_per_batch))
        for interval_start, interval_end in valid_intervals:
            segment_end = self._window_segment_end(
                int(episode_index), episode_len, int(interval_start)
            )
            for start0 in range(int(interval_start), int(interval_end) + 1, batch_size):
                start1 = min(int(interval_end) + 1, start0 + batch_size)
                starts = torch.arange(start0, start1, dtype=torch.long)
                num_windows = int(starts.numel())

                if self.slow_motion_factor != 1.0:
                    retimed_stats = self._slow_motion_unique_stat_rows(
                        parquet,
                        int(episode_index),
                        starts,
                        segment_end,
                        episode_len,
                        include_state=True,
                    )
                    yield {
                        "dataset_name": self.name,
                        "stats_group": self.spec.stats_group or self.name,
                        "episode_index": torch.full(
                            (num_windows,), int(episode_index), dtype=torch.long
                        ),
                        "sample_frame_index": starts,
                        "observation.state": retimed_stats["state"],
                        "action": retimed_stats["action"],
                        "observation.state_is_pad": retimed_stats["state_is_pad"],
                        "action_is_pad": retimed_stats["action_is_pad"],
                        "observation.state_dim_is_pad": retimed_stats["state_dim_is_pad"],
                        "action_dim_is_pad": retimed_stats["action_dim_is_pad"],
                        "observation.state_mask": retimed_stats["state_mask"],
                        "action_mask": retimed_stats["action_mask"],
                        "_stats_state_row_weights": retimed_stats["state_weights"],
                        "_stats_action_row_weights": retimed_stats["action_weights"],
                        "_stats_num_samples": num_windows,
                    }
                    continue

                state_indices, state_pad = self._query_indices_batch(
                    starts, segment_end, self.num_frames_per_sample
                )
                action_indices, action_pad = self._query_indices_batch(
                    starts, segment_end, self.action_size
                )
                source_state_indices = self.source_frame_indices(int(episode_index), state_indices)
                source_action_indices = self.source_frame_indices(int(episode_index), action_indices)
                action_next_raw = torch.as_tensor(
                    action_indices, dtype=torch.long
                ).add(int(self.global_sample_stride))
                action_next_pad = action_next_raw.ge(segment_end).reshape_as(
                    action_pad
                )
                action_next_indices = action_next_raw.clamp(
                    max=segment_end - 1
                ).tolist()
                source_action_next_indices = self.source_frame_indices(int(episode_index), action_next_indices)
                action_pad = action_pad | action_next_pad

                if self.use_canonical_adapter:
                    # Adjacent windows repeat almost every state/action row.
                    # Project each distinct source row (or action/next-action
                    # pair) once and attach its exact multiplicity.  The stats
                    # accumulator consumes these weights, so count, moments,
                    # extrema and reservoir quantiles are identical to the
                    # expanded window stream without doing ~16x redundant EEF
                    # rotation conversions.
                    state_source = np.asarray(source_state_indices, dtype=np.int64)
                    state_valid = ~state_pad.reshape(-1).detach().cpu().numpy().astype(bool)
                    unique_state, state_weights = np.unique(
                        state_source[state_valid],
                        return_counts=True,
                    )
                    state, state_dim_is_pad, state_mask = self._canonical_project_with_validity(
                        parquet,
                        unique_state.tolist(),
                        "state",
                    )

                    action_source = np.asarray(source_action_indices, dtype=np.int64)
                    action_next_source = np.asarray(source_action_next_indices, dtype=np.int64)
                    action_valid = ~action_pad.reshape(-1).detach().cpu().numpy().astype(bool)
                    action_pairs = np.stack(
                        [action_source[action_valid], action_next_source[action_valid]],
                        axis=1,
                    )
                    unique_action_pairs, action_weights = np.unique(
                        action_pairs,
                        axis=0,
                        return_counts=True,
                    )
                    action, action_dim_is_pad, action_mask = self._canonical_project_with_validity(
                        parquet,
                        unique_action_pairs[:, 0].tolist(),
                        "action",
                        unique_action_pairs[:, 1].tolist(),
                    )
                    state_row_weights = torch.as_tensor(state_weights, dtype=torch.long)
                    action_row_weights = torch.as_tensor(action_weights, dtype=torch.long)
                    state_time_is_pad = torch.zeros(len(state_weights), dtype=torch.bool)

                    action_time_is_pad = ~torch.isfinite(action).all(dim=-1).cpu()
                else:
                    state_flat, state_dim_is_pad = _pad_last_dim(
                        _select_rows_concat(parquet, self.state_keys, source_state_indices),
                        self.state_target_dim,
                    )
                    action_flat, action_dim_is_pad = _pad_last_dim(
                        _select_rows_concat(parquet, self.action_keys, source_action_indices),
                        self.action_target_dim,
                    )
                    state = state_flat.reshape(num_windows, self.num_frames_per_sample, -1)
                    action = action_flat.reshape(num_windows, self.action_size, -1)
                    state_row_weights = None
                    action_row_weights = None
                    state_time_is_pad = state_pad
                    action_time_is_pad = action_pad
                    state_mask = None
                    action_mask = None

                batch = {
                    "dataset_name": self.name,
                    "stats_group": self.spec.stats_group or self.name,
                    "episode_index": torch.full((num_windows,), int(episode_index), dtype=torch.long),
                    "sample_frame_index": starts,
                    "observation.state": state,
                    "action": action,
                    "observation.state_is_pad": state_time_is_pad,
                    "action_is_pad": action_time_is_pad,
                    "observation.state_dim_is_pad": state_dim_is_pad,
                    "action_dim_is_pad": action_dim_is_pad,
                    "_stats_state_row_weights": state_row_weights,
                    "_stats_action_row_weights": action_row_weights,
                    "_stats_num_samples": num_windows,
                }
                if state_mask is not None:
                    batch["observation.state_mask"] = state_mask
                if action_mask is not None:
                    batch["action_mask"] = action_mask
                yield batch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_index, frame_idx = self._global_to_episode_frame(int(idx))
        return self.get_step_item(episode_index, frame_idx)

    def get_video_for_item(
        self,
        item: dict[str, Any],
        video_key: str | None = None,
        sample_indices: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if not self.video_keys:
            raise RuntimeError("dataset has no video keys")
        video_key = video_key or self.video_key or self.video_keys[0]
        episode_index = int(item["episode_index"].item())
        timestamps = self.resolve_video_timestamps(
            item, video_key, sample_indices
        )
        with self._open_episode_video(episode_index, video_key) as source:
            return _decode_mp4_lerobot(source, timestamps, fps=self.fps)


    def _project_retimed_wrist_states(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_lower = self.source_frame_indices(episode_index, lower)
        source_upper = self.source_frame_indices(episode_index, upper)
        current, state_dim_is_pad, current_mask = self._canonical_project_with_validity(
            parquet, source_lower, "state"
        )
        future, _, future_mask = self._canonical_project_with_validity(
            parquet, source_upper, "state"
        )
        if current_mask is None or future_mask is None:
            raise ValueError(
                f"Dataset {self.name} slow motion requires exact state validity."
            )

        blend = alpha.reshape(-1, 1).to(current.dtype)
        state = torch.lerp(current, future, blend)
        slots = self.canonical_adapter["slots"]
        for slot in slots.values():
            pose = slot["pose"]
            start, end = map(int, pose["state_target_slice"])
            state[:, start:end] = _interpolate_xyz_rot6d_pose(
                current[:, start:end], future[:, start:end], blend
            )
        return state, state_dim_is_pad, current_mask & future_mask

    def _project_retimed_wrist_actions(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        first_local: torch.Tensor,
        second_local: torch.Tensor,
        alpha_start: torch.Tensor,
        alpha_end: torch.Tensor,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        start_state_mask: torch.Tensor,
        end_state_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_source = self.source_frame_indices(episode_index, first_local)
        second_source = self.source_frame_indices(episode_index, second_local)
        first, action_dim_is_pad, first_mask = self._canonical_project_with_validity(
            parquet,
            first_source,
            "action",
            convert_wrist_action_translation=False,
            regenerate_state_delta_common=False,
        )
        second, _, second_mask = self._canonical_project_with_validity(
            parquet,
            second_source,
            "action",
            convert_wrist_action_translation=False,
            regenerate_state_delta_common=False,
        )
        if first_mask is None or second_mask is None:
            raise ValueError(
                f"Dataset {self.name} slow motion requires exact action validity."
            )

        crosses = second_local != first_local
        needs_second = crosses & (alpha_end > 0.0)
        action = first.clone()
        action_mask = first_mask & torch.where(
            needs_second[:, None], second_mask, torch.ones_like(second_mask)
        )
        alpha_start_column = alpha_start[:, None].to(action.dtype)
        alpha_end_column = alpha_end[:, None].to(action.dtype)

        slots = self.canonical_adapter["slots"]
        action_pose_semantics = str(
            self.canonical_adapter.get("action_pose_semantics", "stored_delta")
        ).strip().lower()
        for slot in slots.values():
            pose = slot["pose"]
            state_pose_start, state_pose_end = map(
                int, pose["state_target_slice"]
            )
            action_pose_start, action_pose_end = map(
                int, pose["action_target_slice"]
            )
            if action_pose_semantics == "state_delta_common":
                current_pose = _xyz_rot6d_to_xyz_rotvec(
                    start_state[:, state_pose_start:state_pose_end]
                )
                future_pose = _xyz_rot6d_to_xyz_rotvec(
                    end_state[:, state_pose_start:state_pose_end]
                )
                action[:, action_pose_start:action_pose_end] = (
                    _pose_delta_world_from_transformed_xyz_rotvec(
                        current_pose, future_pose
                    )
                )
            elif action_pose_semantics == "stored_delta":
                action[:, action_pose_start:action_pose_end] = (
                    _fractional_eef_action_delta(
                        first[:, action_pose_start:action_pose_end],
                        second[:, action_pose_start:action_pose_end],
                        alpha_start_column,
                        alpha_end_column,
                        crosses,
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {self.name} has unsupported action_pose_semantics="
                    f"{action_pose_semantics!r}."
                )
            pose_valid = (
                start_state_mask[:, state_pose_start:state_pose_end].all(dim=-1)
                & end_state_mask[:, state_pose_start:state_pose_end].all(dim=-1)
            )
            action_mask[:, action_pose_start:action_pose_end] &= pose_valid[:, None]

            gripper = slot["gripper"]
            state_gripper_start, state_gripper_end = map(
                int, gripper["state_target_slice"]
            )
            action_gripper_start, action_gripper_end = map(
                int, gripper["action_target_slice"]
            )
            action[:, action_gripper_start:action_gripper_end] = end_state[
                :, state_gripper_start:state_gripper_end
            ]
            gripper_valid = (
                start_state_mask[:, state_gripper_start:state_gripper_end].all(dim=-1)
                & end_state_mask[:, state_gripper_start:state_gripper_end].all(dim=-1)
            )
            action_mask[:, action_gripper_start:action_gripper_end] &= gripper_valid[:, None]

        if action_pose_semantics == "stored_delta":
            action, action_mask = self._convert_wrist_action_translation(
                action,
                start_state,
                start_state_mask,
                action_mask,
            )

        return action, action_dim_is_pad, action_mask

    def _project_retimed_general_states(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retimes a multi-key canonical state without wrist-release assumptions."""

        source_lower = self.source_frame_indices(episode_index, lower)
        source_upper = self.source_frame_indices(episode_index, upper)
        current, state_dim_is_pad, current_mask = self._canonical_project_with_validity(
            parquet, source_lower, "state"
        )
        future, _, future_mask = self._canonical_project_with_validity(
            parquet, source_upper, "state"
        )
        structural_mask = (~state_dim_is_pad).unsqueeze(0).expand(
            current.shape[0], -1
        )
        if current_mask is None:
            current_mask = structural_mask
        if future_mask is None:
            future_mask = structural_mask

        blend = alpha.reshape(-1, 1).to(current.dtype)
        state = torch.lerp(current, future, blend)
        for slot_name, slot in self.canonical_adapter["slots"].items():
            pose = slot.get("pose") or slot.get("ee")
            if not isinstance(pose, dict) or pose.get("pad") is True:
                continue
            target_slice = pose.get("state_target_slice") or pose.get(
                "target_slice"
            )
            if target_slice is None:
                continue
            start, end = map(int, target_slice)
            if end - start != 9:
                raise ValueError(
                    f"Dataset {self.name} slow-motion pose slot {slot_name!r} "
                    f"must map state xyz+rot6d to 9 dimensions, got {end - start}."
                )
            state[:, start:end] = _interpolate_xyz_rot6d_pose(
                current[:, start:end], future[:, start:end], blend
            )
        return state, state_dim_is_pad, current_mask & future_mask

    def _project_retimed_general_actions(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        episode_index: int,
        first_local: torch.Tensor,
        second_local: torch.Tensor,
        alpha_start: torch.Tensor,
        alpha_end: torch.Tensor,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        start_state_mask: torch.Tensor,
        end_state_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retimes Human2Robot-style EEF, joint, and gripper action slots."""

        first_source = self.source_frame_indices(episode_index, first_local)
        second_source = self.source_frame_indices(episode_index, second_local)
        first, action_dim_is_pad, first_mask = self._canonical_project_with_validity(
            parquet,
            first_source,
            "action",
            convert_wrist_action_translation=False,
            regenerate_state_delta_common=False,
        )
        second, _, second_mask = self._canonical_project_with_validity(
            parquet,
            second_source,
            "action",
            convert_wrist_action_translation=False,
            regenerate_state_delta_common=False,
        )
        structural_mask = (~action_dim_is_pad).unsqueeze(0).expand(
            first.shape[0], -1
        )
        if first_mask is None:
            first_mask = structural_mask
        if second_mask is None:
            second_mask = structural_mask

        crosses = second_local != first_local
        needs_second = crosses & (alpha_end > 0.0)
        action = first.clone()
        action_mask = first_mask & torch.where(
            needs_second[:, None], second_mask, torch.ones_like(second_mask)
        )
        alpha_start_column = alpha_start[:, None].to(action.dtype)
        alpha_end_column = alpha_end[:, None].to(action.dtype)

        for slot_name, slot in self.canonical_adapter["slots"].items():
            pose = slot.get("pose") or slot.get("ee")
            if isinstance(pose, dict) and pose.get("pad") is not True:
                state_target_slice = pose.get("state_target_slice") or pose.get(
                    "target_slice"
                )
                action_target_slice = pose.get("action_target_slice") or pose.get(
                    "target_slice"
                )
                if state_target_slice is None or action_target_slice is None:
                    raise ValueError(
                        f"Dataset {self.name} slow-motion pose slot {slot_name!r} "
                        "requires state and action target slices."
                    )
                state_pose_start, state_pose_end = map(int, state_target_slice)
                action_pose_start, action_pose_end = map(int, action_target_slice)
                if state_pose_end - state_pose_start != 9:
                    raise ValueError(
                        f"Dataset {self.name} slow-motion pose slot {slot_name!r} "
                        "must map state xyz+rot6d to 9 dimensions."
                    )
                if action_pose_end - action_pose_start != 6:
                    raise ValueError(
                        f"Dataset {self.name} slow-motion pose slot {slot_name!r} "
                        "must map action xyz+rotvec to 6 dimensions."
                    )
                if _uses_state_delta_common(self.canonical_adapter, pose):
                    current_pose = _xyz_rot6d_to_xyz_rotvec(
                        start_state[:, state_pose_start:state_pose_end]
                    )
                    future_pose = _xyz_rot6d_to_xyz_rotvec(
                        end_state[:, state_pose_start:state_pose_end]
                    )
                    action[:, action_pose_start:action_pose_end] = (
                        _pose_delta_world_from_transformed_xyz_rotvec(
                            current_pose, future_pose
                        )
                    )
                else:
                    action[:, action_pose_start:action_pose_end] = (
                        _fractional_eef_action_delta(
                            first[:, action_pose_start:action_pose_end],
                            second[:, action_pose_start:action_pose_end],
                            alpha_start_column,
                            alpha_end_column,
                            crosses,
                        )
                    )
                pose_valid = (
                    start_state_mask[:, state_pose_start:state_pose_end].all(dim=-1)
                    & end_state_mask[:, state_pose_start:state_pose_end].all(dim=-1)
                )
                action_mask[:, action_pose_start:action_pose_end] &= pose_valid[:, None]

            for component_name in ("joint", "gripper"):
                component = slot.get(component_name)
                if not isinstance(component, dict) or component.get("pad") is True:
                    continue
                state_target_slice = component.get(
                    "state_target_slice"
                ) or component.get("target_slice")
                action_target_slice = component.get(
                    "action_target_slice"
                ) or component.get("target_slice")
                if state_target_slice is None or action_target_slice is None:
                    continue
                state_start, state_end = map(int, state_target_slice)
                action_start, action_end = map(int, action_target_slice)
                if state_end - state_start != action_end - action_start:
                    raise ValueError(
                        f"Dataset {self.name} slow-motion {component_name} slot "
                        f"{slot_name!r} has mismatched state/action widths."
                    )
                action[:, action_start:action_end] = end_state[:, state_start:state_end]
                component_valid = (
                    start_state_mask[:, state_start:state_end].all(dim=-1)
                    & end_state_mask[:, state_start:state_end].all(dim=-1)
                )
                action_mask[:, action_start:action_end] &= component_valid[:, None]

        return action, action_dim_is_pad, action_mask

    def _replace_wrist_actions_from_states(
        self,
        action: torch.Tensor,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        start_state_validity: torch.Tensor | None,
        end_state_validity: torch.Tensor | None,
        action_validity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate robot-semantic pose actions from absolute common-frame states."""

        slots = self.canonical_adapter.get("slots")
        if not isinstance(slots, dict):
            raise ValueError(f"canonical_adapter for {self.name} is missing slots")
        converted = action.clone()
        converted_validity = (
            None if action_validity is None else action_validity.clone()
        )
        for slot_name, slot in slots.items():
            if not isinstance(slot, dict) or slot.get("pad") is True:
                continue
            pose = slot.get("pose")
            if not isinstance(pose, dict) or pose.get("pad") is True:
                continue
            state_slice = pose.get("state_target_slice") or pose.get("target_slice")
            action_slice = pose.get("action_target_slice") or pose.get("target_slice")
            if (
                not isinstance(state_slice, (list, tuple))
                or len(state_slice) != 2
                or not isinstance(action_slice, (list, tuple))
                or len(action_slice) != 2
            ):
                raise ValueError(
                    f"Dataset {self.name} slot {slot_name!r} lacks pose slices."
                )
            state_start, state_end = map(int, state_slice)
            action_start, action_end = map(int, action_slice)
            current_pose = _xyz_rot6d_to_xyz_rotvec(
                start_state[:, state_start:state_end]
            )
            future_pose = _xyz_rot6d_to_xyz_rotvec(
                end_state[:, state_start:state_end]
            )
            converted[:, action_start:action_end] = (
                _pose_delta_world_from_transformed_xyz_rotvec(
                    current_pose, future_pose
                )
            )
            if (
                converted_validity is not None
                and start_state_validity is not None
                and end_state_validity is not None
            ):
                pose_valid = (
                    start_state_validity[:, state_start:state_end].all(dim=-1)
                    & end_state_validity[:, state_start:state_end].all(dim=-1)
                )
                converted_validity[:, action_start:action_end] &= pose_valid[:, None]
        return converted, converted_validity

    def _convert_wrist_action_translation(
        self,
        action: torch.Tensor,
        start_state: torch.Tensor,
        start_state_validity: torch.Tensor | None,
        action_validity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the production compact-wrist translation contract once."""

        action_translation_frame = getattr(
            self,
            "action_translation_frame",
            str(
                self.canonical_adapter.get("action_translation_frame", "source")
            ).strip().lower(),
        )
        if action_translation_frame != "start_observation_common":
            return action, action_validity
        slots = self.canonical_adapter.get("slots")
        if not isinstance(slots, dict):
            raise ValueError(f"canonical_adapter for {self.name} is missing slots")
        converted = action.clone()
        converted_validity = (
            None if action_validity is None else action_validity.clone()
        )
        for slot_name, slot in slots.items():
            if not isinstance(slot, dict) or slot.get("pad") is True:
                continue
            pose = slot.get("pose")
            if not isinstance(pose, dict) or pose.get("pad") is True:
                continue
            state_slice = pose.get("state_target_slice") or pose.get("target_slice")
            action_slice = pose.get("action_target_slice") or pose.get("target_slice")
            if (
                not isinstance(state_slice, (list, tuple))
                or len(state_slice) != 2
                or not isinstance(action_slice, (list, tuple))
                or len(action_slice) != 2
            ):
                raise ValueError(
                    f"Dataset {self.name} slot {slot_name!r} requires state/action "
                    "target slices for start_observation_common conversion."
                )
            state_start, state_end = map(int, state_slice)
            action_start, action_end = map(int, action_slice)
            converted[:, action_start:action_end] = _eef_local_translation_to_common(
                converted[:, action_start:action_end],
                start_state[:, state_start:state_end],
            )
            if converted_validity is not None and start_state_validity is not None:
                pose_valid = start_state_validity[:, state_start:state_end].all(
                    dim=-1
                )
                converted_validity[:, action_start:action_end] &= pose_valid[:, None]
        return converted, converted_validity


def safe_hash(input_tuple: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(input_tuple).encode("utf-8")).hexdigest()
    return int(digest, 16) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


from .pretrain_sampling import PretrainLeRobotMixture


def _distributed_dataset_weights(
    specs: list[DatasetSpec],
    datasets: list[PretrainLeRobotDataset],
    *,
    allow_padding_at_end: bool = False,
) -> list[float]:
    weights = [float(spec.dataset_weight) for spec in specs]
    valid_lengths = [int(ds.valid_start_counts(allow_padding_at_end).sum()) for ds in datasets]
    for group_id in sorted({spec.group_id for spec in specs if spec.distribute_weights}):
        indices = [i for i, spec in enumerate(specs) if spec.group_id == group_id]
        total_len = sum(valid_lengths[i] for i in indices)
        if total_len <= 0:
            continue
        group_weight = float(specs[indices[0]].dataset_weight)
        for i in indices:
            weights[i] = group_weight * (valid_lengths[i] / total_len)
    return weights


def _mixture_dataset_weights(
    specs: list[DatasetSpec],
    datasets: list[PretrainLeRobotDataset],
    mixture_cfg: dict[str, Any],
    *,
    allow_padding_at_end: bool = False,
) -> list[float]:
    mode = str(mixture_cfg.get("dataset_weight_mode", "configured")).strip().lower()
    if mode in {"valid_start_count", "valid_window_count"}:
        return [
            float(_dataset_valid_start_count(ds, allow_padding_at_end))
            if float(spec.dataset_weight) > 0
            else 0.0
            for spec, ds in zip(specs, datasets)
        ]
    if mode in {
        "family_valid_start_count",
        "family_valid_window_count",
        "family_grouped",
    }:
        raw_family_weights = mixture_cfg.get("family_weights")
        if not isinstance(raw_family_weights, dict) or not raw_family_weights:
            raise ValueError(
                "mixture.family_weights must be a non-empty mapping when "
                f"dataset_weight_mode={mode!r}"
            )
        family_weights = {
            str(name): float(weight)
            for name, weight in raw_family_weights.items()
        }
        if any(weight < 0 for weight in family_weights.values()) or not any(
            weight > 0 for weight in family_weights.values()
        ):
            raise ValueError(
                "mixture.family_weights must be non-negative with at least "
                "one positive entry"
            )

        valid_lengths = np.asarray(
            [
                _dataset_valid_start_count(ds, allow_padding_at_end)
                if float(spec.dataset_weight) > 0
                else 0
                for spec, ds in zip(specs, datasets)
            ],
            dtype=np.float64,
        )
        weights = np.zeros(len(specs), dtype=np.float64)
        assigned = np.zeros(len(specs), dtype=np.bool_)
        raw_family_groups = mixture_cfg.get("family_groups", {})
        if raw_family_groups is None:
            raw_family_groups = {}
        if not isinstance(raw_family_groups, dict):
            raise ValueError("mixture.family_groups must be a mapping")
        if mode != "family_grouped" and raw_family_groups:
            raise ValueError(
                "mixture.family_groups requires "
                "dataset_weight_mode='family_grouped'"
            )

        for family, family_weight in family_weights.items():
            indices = np.asarray(
                [
                    idx
                    for idx, spec in enumerate(specs)
                    if spec.sampling_family == family
                ],
                dtype=np.int64,
            )
            if indices.size == 0:
                # A local release may select only a subset of the configured sources.
                continue
            assigned[indices] = True
            family_capacity = float(valid_lengths[indices].sum())
            if family_weight > 0 and family_capacity <= 0:
                raise ValueError(
                    f"mixture family {family!r} has positive weight but no "
                    "valid training starts"
                )
            family_groups = raw_family_groups.get(family)
            if family_groups is None:
                if family_capacity > 0:
                    weights[indices] = (
                        family_weight * valid_lengths[indices] / family_capacity
                    )
                continue
            if not isinstance(family_groups, dict) or not family_groups:
                raise ValueError(
                    f"mixture.family_groups[{family!r}] must be a non-empty "
                    "mapping"
                )

            family_names = {specs[int(idx)].name: int(idx) for idx in indices}
            if len(family_names) != int(indices.size):
                raise ValueError(
                    f"mixture family {family!r} has duplicate dataset names"
                )
            # Retain quality-group ratios among the explicitly selected datasets.
            family_groups = {
                name: dict(group, datasets=[n for n in group.get("datasets", []) if n in family_names])
                for name, group in family_groups.items()
                if isinstance(group, dict)
                and any(n in family_names for n in group.get("datasets", []))
            }
            group_weights = {
                str(group_name): float(group_cfg.get("weight", 0.0))
                for group_name, group_cfg in family_groups.items()
                if isinstance(group_cfg, dict)
            }
            if len(group_weights) != len(family_groups):
                raise ValueError(
                    f"mixture.family_groups[{family!r}] entries must be mappings"
                )
            if any(weight < 0 for weight in group_weights.values()) or not any(
                weight > 0 for weight in group_weights.values()
            ):
                raise ValueError(
                    f"mixture.family_groups[{family!r}] weights must be "
                    "non-negative with at least one positive entry"
                )
            group_weight_total = float(sum(group_weights.values()))
            grouped_indices: set[int] = set()
            for group_name, group_cfg in family_groups.items():
                raw_names = group_cfg.get("datasets")
                if not isinstance(raw_names, list) or not raw_names:
                    raise ValueError(
                        f"mixture family group {family}.{group_name} must "
                        "list datasets"
                    )
                names = [str(name) for name in raw_names]
                if len(set(names)) != len(names):
                    raise ValueError(
                        f"mixture family group {family}.{group_name} lists "
                        "a dataset more than once"
                    )
                unknown = sorted(set(names) - set(family_names))
                if unknown:
                    raise ValueError(
                        f"mixture family group {family}.{group_name} has "
                        f"unknown datasets: {unknown}"
                    )
                group_indices = np.asarray(
                    [family_names[name] for name in names], dtype=np.int64
                )
                duplicate = grouped_indices.intersection(
                    int(idx) for idx in group_indices
                )
                if duplicate:
                    duplicate_names = sorted(specs[idx].name for idx in duplicate)
                    raise ValueError(
                        f"mixture family {family!r} assigns datasets to "
                        f"multiple groups: {duplicate_names}"
                    )
                grouped_indices.update(int(idx) for idx in group_indices)
                allocation = str(
                    group_cfg.get("allocation", "valid_start_count")
                ).strip().lower()
                if allocation in {"valid_start_count", "valid_window_count"}:
                    allocation_weights = valid_lengths[group_indices]
                elif allocation in {"uniform", "equal"}:
                    allocation_weights = (
                        valid_lengths[group_indices] > 0
                    ).astype(np.float64)
                else:
                    raise ValueError(
                        f"Unsupported allocation={allocation!r} for mixture "
                        f"family group {family}.{group_name}; use "
                        "valid_start_count or uniform"
                    )
                allocation_total = float(allocation_weights.sum())
                group_weight = float(group_weights[str(group_name)])
                if group_weight > 0 and allocation_total <= 0:
                    raise ValueError(
                        f"mixture family group {family}.{group_name} has "
                        "positive weight but no valid training starts"
                    )
                if allocation_total > 0:
                    weights[group_indices] = (
                        family_weight
                        * group_weight
                        / group_weight_total
                        * allocation_weights
                        / allocation_total
                    )
            missing = sorted(
                specs[int(idx)].name
                for idx in indices
                if int(idx) not in grouped_indices and valid_lengths[int(idx)] > 0
            )
            if missing:
                raise ValueError(
                    f"mixture family {family!r} has active datasets missing "
                    f"from family_groups: {missing}"
                )

        active_unassigned = [
            spec.name
            for idx, spec in enumerate(specs)
            if valid_lengths[idx] > 0 and not bool(assigned[idx])
        ]
        if active_unassigned:
            raise ValueError(
                "active datasets are missing a configured sampling_family: "
                + ", ".join(active_unassigned[:20])
            )
        return weights.tolist()
    if mode != "configured":
        raise ValueError(
            f"Unsupported mixture.dataset_weight_mode={mode!r}; "
            "use configured, valid_start_count, family_valid_start_count, "
            "or family_grouped"
        )
    return _distributed_dataset_weights(
        specs,
        datasets,
        allow_padding_at_end=allow_padding_at_end,
    )

def _read_yaml_or_json(path: Path) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _resolve_include_path(path: str | Path, config_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = config_dir / out
    return out


def _merge_config_includes(payload: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    projection_files = payload.get("projection_files", payload.get("projection_file", []))
    if projection_files in (None, ""):
        projection_files = []
    if isinstance(projection_files, (str, Path)):
        projection_files = [projection_files]
    if not isinstance(projection_files, list):
        raise TypeError("projection_file/projection_files must be a path or list of paths")

    for include in projection_files:
        include_path = _resolve_include_path(include, config_dir)
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue
        if not isinstance(fragment, dict):
            raise TypeError(f"Projection include must be a mapping: {include_path}")
        for key in ("canonical_action_space", "control_schemas"):
            if key in fragment:
                existing = payload.get(key)
                value = fragment[key]
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged = dict(existing)
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value

    embodiments = dict(payload.get("embodiments", {}) or {})
    for include in payload.get("embodiment_files", []) or []:
        include_path = _resolve_include_path(include, config_dir)
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue
        if not isinstance(fragment, dict):
            raise TypeError(f"Embodiment include must be a mapping: {include_path}")
        if isinstance(fragment.get("embodiments"), dict):
            embodiments.update(fragment["embodiments"])
        elif "name" in fragment:
            fragment = dict(fragment)
            name = str(fragment.pop("name"))
            embodiments[name] = fragment
        else:
            embodiments.update(fragment)
    if embodiments:
        payload["embodiments"] = embodiments

    ego_cfg = payload.get("ego_data", {}) or {}
    if isinstance(ego_cfg, dict):
        ego_enabled = bool(ego_cfg.get("enabled", True))
        ego_source_names = {str(name) for name in ego_cfg.get("source_files", ["egodex"])}
    else:
        ego_enabled = bool(ego_cfg)
        ego_source_names = {"egodex"}

    def iter_source_includes(source_files: Any) -> list[str]:
        if source_files in (None, ""):
            return []
        if isinstance(source_files, (list, tuple)):
            return [str(item) for item in source_files]
        if not isinstance(source_files, dict):
            raise TypeError("source_files must be a list or a mapping")

        includes: list[str] = []
        for name, value in source_files.items():
            include = str(name)
            enabled = True
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, (int, float)):
                enabled = bool(value)
            elif isinstance(value, str):
                text = value.strip().lower()
                if text in {"false", "off", "no", "0", "disabled"}:
                    enabled = False
                elif text in {"true", "on", "yes", "1", "enabled"}:
                    enabled = True
                else:
                    include = value
            elif isinstance(value, dict):
                enabled = bool(value.get("enabled", True))
                include = str(value.get("path", value.get("file", value.get("include", include))))
            elif value is None:
                enabled = False
            else:
                raise TypeError(f"Unsupported source_files.{name!s} value type: {type(value).__name__}")
            if not enabled:
                continue
            if "/" not in include and not Path(include).suffix:
                include = f"sources/{include}.yaml"
            includes.append(include)
        return includes

    sources = list(payload.get("sources", []) or [])
    for include in iter_source_includes(payload.get("source_files", [])):
        include_path = _resolve_include_path(include, config_dir)
        source_file = include_path.stem
        if not ego_enabled and source_file in ego_source_names:
            continue
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue

        def annotate_source(item: Any) -> Any:
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("_source_file", source_file)
                if isinstance(fragment, dict) and fragment.get("normalization_stats"):
                    item.setdefault("normalization_stats", fragment["normalization_stats"])
            return item

        if isinstance(fragment, dict) and isinstance(fragment.get("sources"), list):
            sources.extend(annotate_source(item) for item in fragment["sources"])
        elif isinstance(fragment, list):
            sources.extend(annotate_source(item) for item in fragment)
        elif isinstance(fragment, dict):
            sources.append(annotate_source(fragment))
        else:
            raise TypeError(f"Source include must be a mapping or list: {include_path}")
    if sources:
        payload["sources"] = sources

    return payload


def _read_dataset_config(path: Path | None, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"Dataset configuration inheritance cycle: {path}")
    payload = _read_yaml_or_json(path)
    if not isinstance(payload, dict):
        return {}
    config_dir = path.parent
    parent = payload.pop("extends", None)
    base = {} if parent is None else _read_dataset_config(
        _resolve_include_path(parent, config_dir), (*_parents, path)
    )
    payload = _merge_config_includes(payload, config_dir)
    if base:
        from omegaconf import OmegaConf
        payload = OmegaConf.to_container(OmegaConf.merge(base, payload), resolve=True)
    payload["_config_dir"] = str(config_dir)
    for key, env in (("data_root", "WAM_DATA_ROOT"), ("cache_root", "WAM_CACHE_ROOT")):
        if os.environ.get(env):
            payload[key] = os.environ[env]
    return payload


def _get_nested(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    return default


def _resolved_arg(args_value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return args_value if args_value is not None else _get_nested(config, section, key, default)


def _optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None or value == "":
        return default
    return str(value)


def _parse_format_major(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().lstrip("v")
    if text.startswith("lerobot"):
        text = text.replace("lerobot", "").strip("_-v ")
    try:
        return int(text.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"Unsupported format_version={value!r}; use v2 or v3") from exc


def _path_segments_for_group(value: str) -> list[str]:
    text = str(value).replace("://", "/")
    return [part for part in text.replace("\\", "/").split("/") if part]


def _infer_stats_group(remote_root: str, dataset_name: str, embodiment: str, control_schema: str | None) -> str:
    segments = _path_segments_for_group(remote_root)
    lowered = [part.lower() for part in segments]
    remote_text = "/".join(lowered)

    for pos, part in enumerate(lowered):
        if "interndata-a1" in part:
            if pos + 2 < len(segments):
                return f"a1/{_safe_name(segments[pos + 2])}"
            return "a1/unknown"

    if "agibotworld" in lowered:
        return "agibot/a2d"

    if "galaxea" in lowered:
        return "galaxea/r1lite"


    emb = _safe_name(embodiment or "unknown")
    schema = _safe_name(control_schema or "")
    return f"{emb}/{schema}" if schema else emb

def _make_spec(
    item: dict[str, Any] | str,
    args: argparse.Namespace,
    *,
    idx: int,
    group_id: int,
    inherited: dict[str, Any] | None = None,
) -> DatasetSpec:
    inherited = inherited or {}
    if isinstance(item, str):
        item = {"dir": item}

    remotes = inherited.get("_remotes", {}) if isinstance(inherited.get("_remotes"), dict) else {}
    remote_name = str(item.get("remote") or inherited.get("remote") or "")
    remote_base = item.get("remote_base", inherited.get("remote_base"))
    if not remote_base and remote_name:
        if remote_name not in remotes:
            raise KeyError(f"Unknown remote alias {remote_name!r}; available={sorted(remotes)}")
        remote_base = remotes[remote_name]

    base_path = str(item.get("base_path", inherited.get("base_path", ""))).strip("/")
    rel_path = item.get("dir", item.get("path", item.get("relative_path", "")))
    rel_path = str(rel_path).strip("/") if rel_path not in (None, "") else ""

    remote_root = item.get("remote_root", item.get("address", item.get("root")))
    if not remote_root and remote_base:
        parts = [str(remote_base)]
        if base_path:
            parts.append(base_path)
        if rel_path:
            parts.append(rel_path)
        remote_root = path_join(parts[0], *parts[1:])
    if not remote_root:
        raise KeyError(f"dataset spec {idx} is missing remote_root or remote/base_path/dir")
    remote_root = _resolve_local_dataset_root(str(remote_root), inherited.get("data_root"))

    dataset_path = "/".join(part for part in (base_path, rel_path) if part)
    dataset_name = str(item.get("name") or _safe_name(dataset_path or str(remote_root).rstrip('/').rsplit('/', 1)[-1]) or f"dataset_{idx}")
    embodiment = str(item.get("adapter") or item.get("embodiment") or inherited.get("adapter") or inherited.get("embodiment") or "default")

    stats_value = item.get("stats_path", inherited.get("stats_path", "dataset_stats.json"))
    local_stats_value = item.get("local_stats_path", inherited.get("local_stats_path"))
    path_index_value = item.get("path_index_dir", inherited.get("path_index_dir"))
    local_text_cache_value = item.get(
        "local_text_embedding_cache_dir",
        item.get("text_embedding_cache_dir", inherited.get("local_text_embedding_cache_dir", inherited.get("text_embedding_cache_dir"))),
    )

    source_name = item.get("source_name") or inherited.get("source_name") or inherited.get("_source_file") or remote_name
    relative_remote = _relative_remote_cache_path(str(remote_root))
    cache_root = str(
        inherited.get("cache_root") or os.environ.get("WAM_CACHE_ROOT") or DEFAULT_CACHE_ROOT
    ).rstrip("/")
    local_stats_value = local_stats_value or _format_template(
        inherited.get("local_stats_path_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        cache_root=cache_root,
    )
    path_index_value = path_index_value or _format_template(
        inherited.get("path_index_dir_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        cache_root=cache_root,
    )
    local_text_cache_value = local_text_cache_value or _format_template(
        inherited.get("local_text_embedding_cache_dir_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        cache_root=cache_root,
    )

    dataset_weight = float(item.get("dataset_weight", inherited.get("dataset_weight", 1.0)))
    distribute_weights = bool(item.get("distribute_weights", inherited.get("distribute_weights", False)))
    dataset_override = _lookup_weight_override(
        inherited.get("dataset_weight_overrides"),
        [dataset_name, dataset_path, rel_path, remote_root],
    )
    dataset_weight, distribute_weights = _apply_weight_override(dataset_weight, distribute_weights, dataset_override)

    modalities = _merge_mapping(inherited.get("modalities"), item.get("modalities"))
    control_schema = _optional_str(item.get("control_schema", inherited.get("control_schema")))
    stats_group_template = item.get("stats_group_template", inherited.get("stats_group_template"))
    stats_group = item.get("stats_group", inherited.get("stats_group"))
    if not stats_group and stats_group_template:
        body = _infer_stats_group(str(remote_root), dataset_name, embodiment, control_schema).split("/", 1)[-1]
        stats_group = _format_template(
            stats_group_template,
            name=dataset_name,
            path=dataset_path,
            remote=str(remote_root),
            source=source_name,
            relative_remote=relative_remote,
            embodiment=embodiment,
            cache_root=cache_root,
            body=body,
            control_schema=control_schema or "",
        )
    if not stats_group:
        stats_group = _infer_stats_group(str(remote_root), dataset_name, embodiment, control_schema)

    return DatasetSpec(
        name=dataset_name,
        remote_root=str(remote_root),
        stats_path=_optional_str(stats_value),
        local_stats_path=_optional_str(local_stats_value),
        path_index_dir=_optional_str(path_index_value),
        data_root=str(inherited.get("data_root") or os.environ.get("WAM_DATA_ROOT", "data")),
        require_path_index=bool(item.get("require_path_index", inherited.get("require_path_index", False))),
        local_data_dir=_optional_str(item.get("local_data_dir", inherited.get("local_data_dir"))),
        local_text_embedding_cache_dir=_optional_str(local_text_cache_value),
        context_len=int(item.get("context_len", inherited.get("context_len", 128))),
        text_encoder_id=str(item.get("text_encoder_id", inherited.get("text_encoder_id", "wan22ti2v5b"))),
        prompt_template=str(item.get("prompt_template", inherited.get("prompt_template", DEFAULT_PROMPT))),
        task_index_authority=str(item.get("task_index_authority", inherited.get("task_index_authority", "parquet"))),
        slow_motion_factor=float(item.get("slow_motion_factor", inherited.get("slow_motion_factor", 1.0))),
        action_loss_weight=float(
            item.get(
                "action_loss_weight",
                inherited.get("action_loss_weight", 1.0),
            )
        ),
        sampling_family=item.get("sampling_family", inherited.get("sampling_family", None)),
        allow_padding_at_end=item.get("allow_padding_at_end", inherited.get("allow_padding_at_end", None)),
        action_loss_normalization=item.get("action_loss_normalization", inherited.get("action_loss_normalization", "legacy_per_sample")),
        action_gripper_loss_weight=float(
            item.get(
                "action_gripper_loss_weight",
                inherited.get("action_gripper_loss_weight", 1.0),
            )
        ),
        modalities=modalities,
        canonical_adapter=_merge_mapping(inherited.get("canonical_adapter"), item.get("canonical_adapter")),
        format_version=_parse_format_major(item.get("format_version", item.get("format", inherited.get("format_version", inherited.get("format"))))),
        embodiment=embodiment,
        control_schema=control_schema,
        stats_group=_optional_str(stats_group),
        normalization_stats=_optional_str(item.get("normalization_stats", inherited.get("normalization_stats"))),
        source_name=_optional_str(source_name),
        dataset_weight=dataset_weight,
        distribute_weights=distribute_weights,
        action_target_dim=item.get("action_target_dim", inherited.get("action_target_dim")),
        state_target_dim=item.get("state_target_dim", inherited.get("state_target_dim")),
        group_id=int(group_id),
    )


def _resolve_config_path(path: str | Path, config_dir: str | None) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute() and config_dir:
        out = Path(config_dir) / out
    return out


def _read_dataset_entries_file(path: str | Path, config_dir: str | None) -> list[dict[str, Any] | str]:
    file_path = _resolve_config_path(path, config_dir)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        if suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            import yaml

            payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("datasets", payload.get("paths", payload.get("items")))
        if not isinstance(payload, list):
            raise TypeError(f"Dataset list file must contain a list: {file_path}")
        return payload

    entries: list[dict[str, Any] | str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _dataset_entries_from_group(group: dict[str, Any], config_dir: str | None, data_root: str | None = None) -> list[dict[str, Any] | str]:
    entries: list[dict[str, Any] | str] = []
    files = group.get("datasets_files", group.get("datasets_file"))
    if files is not None:
        if isinstance(files, (str, Path)):
            files = [files]
        if not isinstance(files, list):
            raise TypeError("datasets_file/datasets_files must be a path or list of paths")
        for file_path in files:
            entries.extend(_read_dataset_entries_file(file_path, config_dir))

    inline = group.get("datasets")
    if inline is not None:
        if not isinstance(inline, list):
            raise TypeError("datasets must be a list")
        entries.extend(inline)

    pattern = group.get("datasets_glob")
    if pattern:
        root = Path(_resolve_local_dataset_root(group["root"], data_root))
        matches = sorted(path for path in root.glob(str(pattern)) if (path / "meta/info.json").is_file())
        if not matches:
            raise FileNotFoundError(f"No local LeRobot datasets match {root / str(pattern)}")
        for path in matches:
            relative = path.relative_to(root).as_posix()
            name = str(group["name"]) if relative == "." else str(group["name"]) + "__" + _safe_name(relative)
            entries.append({"name": name, "root": str(path.resolve())})
    return entries or [group]


def _flatten_mixture_specs(args: argparse.Namespace, config: dict[str, Any]) -> list[DatasetSpec] | None:
    mixture = config.get("mixture")
    if not isinstance(mixture, dict) or not isinstance(mixture.get("specs"), list):
        return None

    remotes = config.get("remotes", {}) if isinstance(config.get("remotes"), dict) else {}
    cache_templates = config.get("cache_templates", {}) if isinstance(config.get("cache_templates"), dict) else {}
    dataset_defaults = config.get("dataset_defaults", {}) if isinstance(config.get("dataset_defaults"), dict) else {}
    weight_config = config.get("dataset_weights", {}) if isinstance(config.get("dataset_weights"), dict) else {}
    default_dataset_weight = float(dataset_defaults.get("dataset_weight", config.get("dataset_weight", 1.0)))
    default_distribute_weights = bool(dataset_defaults.get("distribute_weights", config.get("distribute_weights", False)))
    default_canonical_adapter = dataset_defaults.get("canonical_adapter")
    source_weight_overrides = weight_config.get("sources", {}) if isinstance(weight_config, dict) else {}
    dataset_weight_overrides = weight_config.get("datasets", {}) if isinstance(weight_config, dict) else {}
    config_dir = config.get("_config_dir")

    specs: list[DatasetSpec] = []
    for group_id, group in enumerate(mixture["specs"]):
        if not isinstance(group, dict):
            raise TypeError(f"mixture.specs[{group_id}] must be a mapping")
        if group.get("enabled", True) is False:
            continue
        source_name = str(group.get("name") or group.get("_source_file") or "")
        inherited = {
            "_remotes": remotes,
            "_source_file": group.get("_source_file"),
            "source_name": source_name,
            "remote": group.get("remote"),
            "remote_base": group.get("remote_base"),
            "base_path": group.get("base_path", ""),
            "data_root": config.get("data_root") or os.environ.get("WAM_DATA_ROOT", DEFAULT_DATA_ROOT),
            "cache_root": config.get("cache_root") or os.environ.get("WAM_CACHE_ROOT", DEFAULT_CACHE_ROOT),
            "stats_path": group.get("stats_path", "dataset_stats.json"),
            "local_stats_path": group.get("local_stats_path"),
            "normalization_stats": group.get("normalization_stats"),
            "path_index_dir": group.get("path_index_dir"),
            "require_path_index": group.get("require_path_index", dataset_defaults.get("require_path_index", False)),
            "local_data_dir": group.get("local_data_dir"),
            "local_text_embedding_cache_dir": group.get("local_text_embedding_cache_dir", group.get("text_embedding_cache_dir")),
            "path_index_dir_template": group.get("path_index_dir_template", cache_templates.get("path_index_dir")),
            "local_stats_path_template": group.get("local_stats_path_template", cache_templates.get("local_stats_path")),
            "local_text_embedding_cache_dir_template": group.get(
                "local_text_embedding_cache_dir_template",
                cache_templates.get("local_text_embedding_cache_dir"),
            ),
            "context_len": group.get("context_len", 128),
            "text_encoder_id": group.get("text_encoder_id", "wan22ti2v5b"),
            "prompt_template": group.get("prompt_template", DEFAULT_PROMPT),
            "task_index_authority": group.get(
                "task_index_authority",
                dataset_defaults.get("task_index_authority", "parquet"),
            ),
            "slow_motion_factor": group.get("slow_motion_factor", 1.0),
            "action_loss_weight": group.get(
                "action_loss_weight",
                dataset_defaults.get("action_loss_weight", 1.0),
            ),
            "action_gripper_loss_weight": group.get(
                "action_gripper_loss_weight",
                dataset_defaults.get("action_gripper_loss_weight", 1.0),
            ),
            "sampling_family": group.get("sampling_family", dataset_defaults.get("sampling_family", None)),
            "allow_padding_at_end": group.get("allow_padding_at_end", dataset_defaults.get("allow_padding_at_end", None)),
            "action_loss_normalization": group.get("action_loss_normalization", dataset_defaults.get("action_loss_normalization", "legacy_per_sample")),
            "modalities": group.get("modalities"),
            "canonical_adapter": _merge_mapping(
                default_canonical_adapter,
                group.get("canonical_adapter"),
            ),
            "format_version": group.get("format_version", group.get("format")),
            "embodiment": group.get("embodiment", "default"),
            "control_schema": group.get("control_schema"),
            "stats_group": group.get("stats_group"),
            "stats_group_template": group.get("stats_group_template"),
            "dataset_weight": _apply_weight_override(
                float(group.get("dataset_weight", default_dataset_weight)),
                bool(group.get("distribute_weights", default_distribute_weights)),
                _lookup_weight_override(source_weight_overrides, [group.get("name"), group.get("_source_file"), group.get("remote_base", group.get("address"))]),
            )[0],
            "distribute_weights": _apply_weight_override(
                float(group.get("dataset_weight", default_dataset_weight)),
                bool(group.get("distribute_weights", default_distribute_weights)),
                _lookup_weight_override(source_weight_overrides, [group.get("name"), group.get("_source_file"), group.get("remote_base", group.get("address"))]),
            )[1],
            "dataset_weight_overrides": dataset_weight_overrides,
            "action_target_dim": group.get("action_target_dim"),
            "state_target_dim": group.get("state_target_dim"),
        }
        children = _dataset_entries_from_group(group, config_dir, config.get("data_root"))
        for child in children:
            if not isinstance(child, (dict, str)):
                raise TypeError(f"mixture.specs[{group_id}] dataset entry must be a mapping or string")
            if isinstance(child, dict) and child.get("enabled", True) is False:
                continue
            specs.append(_make_spec(child, args, idx=len(specs), group_id=group_id, inherited=inherited))
    return specs


def _flatten_source_specs(args: argparse.Namespace, config: dict[str, Any]) -> list[DatasetSpec] | None:
    sources = config.get("sources")
    if not isinstance(sources, list):
        return None

    remotes = config.get("remotes", {}) if isinstance(config.get("remotes"), dict) else {}
    cache_templates = config.get("cache_templates", {}) if isinstance(config.get("cache_templates"), dict) else {}
    dataset_defaults = config.get("dataset_defaults", {}) if isinstance(config.get("dataset_defaults"), dict) else {}
    weight_config = config.get("dataset_weights", {}) if isinstance(config.get("dataset_weights"), dict) else {}
    default_dataset_weight = float(dataset_defaults.get("dataset_weight", config.get("dataset_weight", 1.0)))
    default_distribute_weights = bool(dataset_defaults.get("distribute_weights", config.get("distribute_weights", False)))
    default_canonical_adapter = dataset_defaults.get("canonical_adapter")
    source_weight_overrides = weight_config.get("sources", {}) if isinstance(weight_config, dict) else {}
    source_file_weight_overrides = weight_config.get("source_files", {}) if isinstance(weight_config, dict) else {}
    embodiment_weight_overrides = weight_config.get("embodiments", {}) if isinstance(weight_config, dict) else {}
    dataset_weight_overrides = weight_config.get("datasets", {}) if isinstance(weight_config, dict) else {}
    embodiment_defaults = config.get("embodiments", {}) if isinstance(config.get("embodiments"), dict) else {}
    config_dir = config.get("_config_dir")
    specs: list[DatasetSpec] = []
    source_file_group_ids: dict[str, int] = {}
    next_group_id = 0

    for source_idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TypeError(f"sources[{source_idx}] must be a mapping")
        if source.get("enabled", True) is False:
            continue
        source_file = str(source.get("_source_file") or source.get("name") or f"source_{source_idx}")
        source_file_weight_override = _lookup_weight_override(
            source_file_weight_overrides, [source_file]
        )
        if source_file_weight_override is not None:
            if source_file not in source_file_group_ids:
                source_file_group_ids[source_file] = next_group_id
                next_group_id += 1
            group_id = source_file_group_ids[source_file]
        else:
            group_id = next_group_id
            next_group_id += 1
        embodiment = str(source.get("adapter", source.get("embodiment", "default")))
        emb_cfg = embodiment_defaults.get(embodiment, {})
        if emb_cfg is None:
            emb_cfg = {}
        if not isinstance(emb_cfg, dict):
            raise TypeError(f"embodiments.{embodiment} must be a mapping")
        address = source.get("address", source.get("root", source.get("remote_root")))
        source_name = str(source.get("name") or source.get("_source_file") or "")
        base_source_weight = float(source.get("dataset_weight", emb_cfg.get("dataset_weight", default_dataset_weight)))
        base_source_distribute = bool(source.get("distribute_weights", emb_cfg.get("distribute_weights", default_distribute_weights)))
        if source_file_weight_override is not None:
            source_weight, _ = _apply_weight_override(
                base_source_weight,
                base_source_distribute,
                source_file_weight_override,
            )
            source_distribute = True
        else:
            source_weight, source_distribute = _apply_weight_override(
                base_source_weight,
                base_source_distribute,
                _lookup_weight_override(source_weight_overrides, [source_name, source.get("_source_file"), address, embodiment]),
            )
            source_weight, source_distribute = _apply_weight_override(
                source_weight,
                source_distribute,
                _lookup_weight_override(embodiment_weight_overrides, [embodiment]),
            )
        if source_weight <= 0:
            continue
        source_modalities = _merge_mapping(emb_cfg.get("modalities"), source.get("modalities"))
        source_canonical_adapter = _merge_mapping(
            default_canonical_adapter,
            _merge_mapping(
                emb_cfg.get("canonical_adapter"),
                source.get("canonical_adapter"),
            ),
        )
        inherited = {
            "_remotes": remotes,
            "_source_file": source.get("_source_file"),
            "source_name": source_name,
            "remote": source.get("remote", emb_cfg.get("remote")),
            "remote_base": source.get("remote_base", address if address is not None else emb_cfg.get("remote_base")),
            "base_path": source.get("base_path", emb_cfg.get("base_path", "")),
            "stats_path": source.get("stats_path", emb_cfg.get("stats_path", "dataset_stats.json")),
            "local_stats_path": source.get("local_stats_path", emb_cfg.get("local_stats_path")),
            "normalization_stats": source.get("normalization_stats", emb_cfg.get("normalization_stats")),
            "path_index_dir": source.get("path_index_dir", emb_cfg.get("path_index_dir")),
            "require_path_index": source.get(
                "require_path_index",
                emb_cfg.get("require_path_index", dataset_defaults.get("require_path_index", False)),
            ),
            "local_data_dir": source.get("local_data_dir", emb_cfg.get("local_data_dir")),
            "data_root": config.get("data_root") or os.environ.get("WAM_DATA_ROOT", DEFAULT_DATA_ROOT),
            "cache_root": config.get("cache_root") or os.environ.get("WAM_CACHE_ROOT", DEFAULT_CACHE_ROOT),
            "local_text_embedding_cache_dir": source.get(
                "local_text_embedding_cache_dir",
                source.get("text_embedding_cache_dir", emb_cfg.get("local_text_embedding_cache_dir", emb_cfg.get("text_embedding_cache_dir"))),
            ),
            "path_index_dir_template": source.get("path_index_dir_template", emb_cfg.get("path_index_dir_template", cache_templates.get("path_index_dir"))),
            "local_stats_path_template": source.get("local_stats_path_template", emb_cfg.get("local_stats_path_template", cache_templates.get("local_stats_path"))),
            "local_text_embedding_cache_dir_template": source.get(
                "local_text_embedding_cache_dir_template",
                emb_cfg.get("local_text_embedding_cache_dir_template", cache_templates.get("local_text_embedding_cache_dir")),
            ),
            "context_len": source.get("context_len", emb_cfg.get("context_len", 128)),
            "text_encoder_id": source.get("text_encoder_id", emb_cfg.get("text_encoder_id", "wan22ti2v5b")),
            "prompt_template": source.get("prompt_template", emb_cfg.get("prompt_template", DEFAULT_PROMPT)),
            "task_index_authority": source.get(
                "task_index_authority",
                emb_cfg.get(
                    "task_index_authority",
                    dataset_defaults.get("task_index_authority", "parquet"),
                ),
            ),
            "slow_motion_factor": source.get("slow_motion_factor", emb_cfg.get("slow_motion_factor", 1.0)),
            "action_loss_weight": source.get(
                "action_loss_weight",
                emb_cfg.get(
                    "action_loss_weight",
                    dataset_defaults.get("action_loss_weight", 1.0),
                ),
            ),
            "action_gripper_loss_weight": source.get(
                "action_gripper_loss_weight",
                emb_cfg.get(
                    "action_gripper_loss_weight",
                    dataset_defaults.get("action_gripper_loss_weight", 1.0),
                ),
            ),
            "sampling_family": source.get("sampling_family", dataset_defaults.get("sampling_family", None)),
            "allow_padding_at_end": source.get("allow_padding_at_end", dataset_defaults.get("allow_padding_at_end", None)),
            "action_loss_normalization": source.get("action_loss_normalization", dataset_defaults.get("action_loss_normalization", "legacy_per_sample")),
            "modalities": source_modalities,
            "canonical_adapter": source_canonical_adapter,
            "format_version": source.get("format_version", source.get("format", emb_cfg.get("format_version", emb_cfg.get("format")))),
            "embodiment": embodiment,
            "control_schema": source.get("control_schema", emb_cfg.get("control_schema")),
            "stats_group": source.get("stats_group", emb_cfg.get("stats_group")),
            "stats_group_template": source.get("stats_group_template", emb_cfg.get("stats_group_template")),
            "dataset_weight": source_weight,
            "distribute_weights": source_distribute,
            "dataset_weight_overrides": dataset_weight_overrides,
            "action_target_dim": source.get("action_target_dim", emb_cfg.get("action_target_dim")),
            "state_target_dim": source.get("state_target_dim", emb_cfg.get("state_target_dim")),
        }
        children = _dataset_entries_from_group(source, config_dir, config.get("data_root"))
        for child in children:
            if not isinstance(child, (dict, str)):
                raise TypeError(f"sources[{source_idx}] dataset entry must be a mapping or string")
            if isinstance(child, dict) and child.get("enabled", True) is False:
                continue
            specs.append(_make_spec(child, args, idx=len(specs), group_id=group_id, inherited=inherited))
    return specs


def _load_specs(args: argparse.Namespace, config: dict[str, Any]) -> list[DatasetSpec]:
    if args.dataset_specs_json:
        text = args.dataset_specs_json
        maybe_path = Path(text)
        if maybe_path.exists():
            text = maybe_path.read_text(encoding="utf-8")
        raw_specs = json.loads(text)
        return [_make_spec(item, args, idx=idx, group_id=idx) for idx, item in enumerate(raw_specs)]

    source_specs = _flatten_source_specs(args, config)
    if source_specs is not None:
        return source_specs

    mixture_specs = _flatten_mixture_specs(args, config)
    if mixture_specs is not None:
        return mixture_specs

    if isinstance(config.get("datasets"), list):
        return [_make_spec(item, args, idx=idx, group_id=idx) for idx, item in enumerate(config["datasets"])]

    if not args.remote_root:
        raise ValueError("provide --remote-root, --dataset-specs-json, or --dataset-config")
    return [DatasetSpec(name=args.name, remote_root=args.remote_root)]


def _summarize_item(item: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, str):
            summary[key] = value[:200]
        else:
            summary[key] = str(type(value).__name__)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=None, help="YAML config describing one or more local mix datasets.")
    parser.add_argument("--remote-root", default=None, help="Local LeRobot dataset root, e.g. /path/to/data/ego/egodex")
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--dataset-specs-json", default=None, help="JSON string or path containing a list of {name, remote_root, stats_path} specs.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--action-size", type=int, default=None)
    parser.add_argument("--global-sample-stride", type=int, default=None)
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument("--load-stats", action="store_true", help="Also read each configured dataset_stats.json and print its top-level keys.")
    parser.add_argument("--epoch", type=int, default=0, help="Coverage cycle used by deterministic random-segment training sampling.")
    parser.add_argument("--eval-mode", action="store_true", help="Force deterministic eval subset mode instead of training random sampling.")
    parser.add_argument("--eval-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--eval-frames-per-episode", type=int, default=None)
    parser.add_argument("--eval-frame-policy", choices=["first", "middle", "last", "uniform", "random"], default=None)
    parser.add_argument("--eval-scope", choices=["dataset", "source", "source_type", "type", "source_entry"], default=None)
    parser.add_argument("--eval-segments-per-source", type=int, default=None)
    parser.add_argument("--eval-total-samples", type=int, default=None)
    parser.add_argument("--eval-fixed-samples-json", default=None, help="JSON string/path for fixed eval samples.")
    parser.add_argument("--eval-fixed-per-dataset", action="store_true", help="Use one fixed eval sample from every expanded dataset.")
    parser.add_argument("--eval-exclude-from-training", choices=["none", "sample", "episode"], default=None)
    parser.add_argument("--profile", action="store_true", help="Print coarse timing breakdown for local loading.")
    parser.add_argument("--save-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    global _PROFILE_ENABLED
    args = parse_args()
    _PROFILE_ENABLED = bool(args.profile)
    t_main = time.perf_counter()
    config = _read_dataset_config(args.dataset_config)
    specs = _load_specs(args, config)
    num_frames = int(_resolved_arg(args.num_frames, config, "sampling", "num_frames", 33))
    action_size = _resolved_arg(args.action_size, config, "sampling", "action_size", None)
    if action_size is not None:
        action_size = int(action_size)
    global_sample_stride = int(_resolved_arg(args.global_sample_stride, config, "sampling", "global_sample_stride", 1))

    t0 = time.perf_counter()
    datasets = [
        PretrainLeRobotDataset(
            spec,
            num_frames=num_frames,
            action_size=action_size,
            global_sample_stride=global_sample_stride,
        )
        for spec in specs
    ]
    _profile_record("main.build_datasets", time.perf_counter() - t0, num_datasets=len(datasets))
    if args.load_stats:
        for ds in datasets:
            t0 = time.perf_counter()
            stats = ds.load_dataset_stats()
            _profile_record("main.load_stats", time.perf_counter() - t0, dataset=ds.name)
            print(f"[stats] dataset={ds.name} stats_path={ds.spec.stats_path} keys={list(stats.keys())}", flush=True)

    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    eval_cfg = mixture_cfg.get("eval_subset", mixture_cfg.get("eval", {}))
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    dataset_weights = _mixture_dataset_weights(
        specs,
        datasets,
        mixture_cfg,
        allow_padding_at_end=allow_padding_at_end,
    )
    epoch_length = mixture_cfg.get("epoch_length")
    training = bool(mixture_cfg.get("training", True)) and not bool(args.eval_mode)
    eval_episodes_per_dataset = args.eval_episodes_per_dataset
    if eval_episodes_per_dataset is None:
        eval_episodes_per_dataset = eval_cfg.get("episodes_per_dataset")
    eval_frames_per_episode = args.eval_frames_per_episode
    if eval_frames_per_episode is None:
        eval_frames_per_episode = eval_cfg.get("frames_per_episode")
    eval_frame_policy = args.eval_frame_policy or str(eval_cfg.get("frame_policy", "first"))
    eval_scope = args.eval_scope or str(eval_cfg.get("scope", "dataset"))
    eval_fixed_samples = eval_cfg.get("fixed_samples")
    if args.eval_fixed_samples_json:
        fixed_text = args.eval_fixed_samples_json
        fixed_path = Path(fixed_text)
        if fixed_path.exists():
            fixed_text = fixed_path.read_text(encoding="utf-8")
        eval_fixed_samples = json.loads(fixed_text)
    eval_fixed_per_dataset = bool(args.eval_fixed_per_dataset or eval_cfg.get("fixed_per_dataset", False))
    eval_fixed_episode_pos = eval_cfg.get("episode_pos", 0)
    eval_fixed_frame_index = int(eval_cfg.get("frame_index", 0))
    eval_exclude_from_training = args.eval_exclude_from_training
    if eval_exclude_from_training is None:
        eval_exclude_from_training = eval_cfg.get("exclude_from_training")
    eval_segments_per_source = args.eval_segments_per_source
    if eval_segments_per_source is None:
        eval_segments_per_source = eval_cfg.get("segments_per_source")
    eval_total_samples = args.eval_total_samples
    if eval_total_samples is None:
        eval_total_samples = eval_cfg.get("total_samples", eval_cfg.get("eval_total_samples"))
    dataset = PretrainLeRobotMixture(
        datasets,
        dataset_weights=dataset_weights,
        training=training,
        balance_dataset_weights=bool(mixture_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(mixture_cfg.get("balance_trajectory_weights", True)),
        seed=int(mixture_cfg.get("seed", 42)),
        allow_padding_at_end=allow_padding_at_end,
        epoch_length=None if epoch_length in (None, "") else int(epoch_length),
        eval_episodes_per_dataset=None if eval_episodes_per_dataset in (None, "") else int(eval_episodes_per_dataset),
        eval_frames_per_episode=None if eval_frames_per_episode in (None, "") else int(eval_frames_per_episode),
        eval_frame_policy=eval_frame_policy,
        eval_scope=eval_scope,
        eval_segments_per_source=None if eval_segments_per_source in (None, "") else int(eval_segments_per_source),
        eval_total_samples=None if eval_total_samples in (None, "") else int(eval_total_samples),
        eval_fixed_samples=eval_fixed_samples if isinstance(eval_fixed_samples, list) else None,
        eval_fixed_per_dataset=eval_fixed_per_dataset,
        eval_fixed_episode_pos=None if eval_fixed_episode_pos in (None, "") else int(eval_fixed_episode_pos),
        eval_fixed_frame_index=eval_fixed_frame_index,
        eval_exclude_from_training=eval_exclude_from_training,
        training_block_size=int(mixture_cfg.get("training_block_size", 4)),
    )
    dataset.set_epoch(args.epoch)
    t0 = time.perf_counter()
    item = dataset[args.index]
    _profile_record("main.get_item", time.perf_counter() - t0, index=args.index)
    summary = _summarize_item(item)
    payload = {
        "num_datasets": len(datasets),
        "length": len(dataset),
        "dataset_sampling_weights": dataset.dataset_sampling_weights.tolist(),
        "item": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    if args.with_video:
        t0 = time.perf_counter()
        video = dataset.get_video_for_item(item)
        _profile_record("main.get_video", time.perf_counter() - t0)
        print(f"[video] shape={tuple(video.shape)} dtype={video.dtype} range=({video.min().item():.4f},{video.max().item():.4f})", flush=True)
        summary["video"] = {"shape": list(video.shape), "dtype": str(video.dtype)}

    if args.save_summary is not None:
        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.save_summary}", flush=True)

    _profile_record("main.total", time.perf_counter() - t_main)
    _profile_print()


def _dataset_allows_end_padding(
    dataset: PretrainLeRobotDataset,
    default: bool,
) -> bool:
    override = getattr(dataset.spec, "allow_padding_at_end", None)
    return bool(default if override is None else override)

def _dataset_valid_start_count(
    dataset: PretrainLeRobotDataset,
    default_allow_padding_at_end: bool,
) -> int:
    return int(
        dataset.valid_start_counts(
            _dataset_allows_end_padding(dataset, default_allow_padding_at_end)
        ).sum()
    )

if __name__ == "__main__":
    main()
