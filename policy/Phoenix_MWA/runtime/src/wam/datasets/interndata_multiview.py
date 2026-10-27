"""Certified camera variants from one Interndata episode's seven video streams."""

from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath


INTERNDATA_SOURCES = frozenset({"generic_pick", "task_generalization"})
INTERNDATA_BASE_KEYS = (
    "observation.images.franka_head",
    *(f"observation.images.franka_external_{index:02d}" for index in range(5)),
)
INTERNDATA_WRIST_KEY = "observation.images.franka_hand"
INTERNDATA_MULTIVIEW_POLICY = {
    "version": "six_base_one_wrist_v1",
    "sources": sorted(INTERNDATA_SOURCES),
    "base_keys": list(INTERNDATA_BASE_KEYS),
    "wrist_key": INTERNDATA_WRIST_KEY,
}
_EXPECTED_FPS = 30.0
_TIME_TOLERANCE = 1e-6


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Interndata {label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Interndata {label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"Interndata {label} must be a finite number")
    return result


def _nonnegative_integer(value, label):
    number = _finite_number(value, label)
    if number < 0 or not number.is_integer():
        raise ValueError(f"Interndata {label} must be a nonnegative integer")
    return int(number)


def _absolute_path(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Interndata {label} must be an absolute POSIX path")
    result = PurePosixPath(value)
    if not result.is_absolute() or ".." in result.parts:
        raise ValueError(f"Interndata {label} must be an absolute POSIX path")
    return result


def interndata_camera_variants(record):
    """Return six base-plus-wrist variants without changing the source record.

    Every variant uses the same episode's numeric rows and original text routes.
    Video offsets may differ between camera files; each must match its own
    episode metadata. This validates metadata only and performs no file I/O.
    The episode-level language3 binding remains on the caller's outer record.
    """
    try:
        return _camera_variants(record)
    except KeyError as exc:
        raise ValueError(
            f"Incomplete Interndata multiview metadata: {exc.args[0]}"
        ) from exc


def _camera_variants(record):
    source = record["source_id"]
    if source not in INTERNDATA_SOURCES:
        raise ValueError(f"Unsupported Interndata multiview source: {source}")
    root = _absolute_path(record["source_root"], "source_root")
    if root.name != source:
        raise ValueError("Interndata source_root does not match source_id")
    fps = _finite_number(record["fps"], "episode fps")
    if fps != _EXPECTED_FPS:
        raise ValueError("Interndata multiview requires episode fps 30")
    length = _nonnegative_integer(record["length"], "episode length")
    if length == 0:
        raise ValueError("Interndata episode length must be positive")

    views = record["selected_views"]
    expected_keys = set(INTERNDATA_BASE_KEYS) | {INTERNDATA_WRIST_KEY}
    if not isinstance(views, list) or len(views) != len(expected_keys):
        raise ValueError("Interndata requires exactly seven selected video views")
    keys = [view["key"] for view in views]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError(
            "Interndata selected video keys must be the seven unique cameras"
        )

    paths = set()
    for view in views:
        key = view["key"]
        if view["available"] is not True:
            raise ValueError(f"Interndata camera is unavailable: {key}")
        view_root = _absolute_path(view["source_root"], f"{key} source_root")
        if view_root != root:
            raise ValueError(f"Interndata camera has a different source_root: {key}")
        path = _absolute_path(view["path"], f"{key} video path")
        if path in paths:
            raise ValueError("Interndata camera video paths must be distinct")
        paths.add(path)
        if _finite_number(view["fps"], f"{key} fps") != fps:
            raise ValueError(f"Interndata camera fps differs from episode fps: {key}")

        prefix = f"videos/{key}/"
        chunk = _nonnegative_integer(record[prefix + "chunk_index"], "chunk_index")
        file_index = _nonnegative_integer(record[prefix + "file_index"], "file_index")
        expected_path = (
            root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        )
        if path != expected_path:
            raise ValueError(
                f"Interndata video path differs from episode metadata: {key}"
            )

        start = _finite_number(view["start_s"], f"{key} start_s")
        end = _finite_number(view["end_s"], f"{key} end_s")
        meta_start = _finite_number(
            record[prefix + "from_timestamp"], f"{key} from_timestamp"
        )
        meta_end = _finite_number(
            record[prefix + "to_timestamp"], f"{key} to_timestamp"
        )
        if min(start, end, meta_start, meta_end) < 0 or end <= start:
            raise ValueError(
                f"Interndata video timestamps must be nonnegative and ordered: {key}"
            )
        if not (
            math.isclose(start, meta_start, rel_tol=0, abs_tol=_TIME_TOLERANCE)
            and math.isclose(end, meta_end, rel_tol=0, abs_tol=_TIME_TOLERANCE)
        ):
            raise ValueError(
                f"Interndata camera timestamps differ from episode metadata: {key}"
            )
        if not math.isclose(
            end - start, length / fps, rel_tol=0, abs_tol=_TIME_TOLERANCE
        ):
            raise ValueError(
                f"Interndata video duration differs from episode length/fps: {key}"
            )

    by_key = {view["key"]: view for view in views}
    return [
        {
            "cameras": [base_key, "", INTERNDATA_WRIST_KEY],
            "camera_roles": ["main", "pad", "right_wrist"],
            "selected_views": copy.deepcopy(
                [by_key[base_key], by_key[INTERNDATA_WRIST_KEY]]
            ),
            "t5_task_bindings": copy.deepcopy(record["t5_task_bindings"]),
        }
        for base_key in INTERNDATA_BASE_KEYS
    ]
