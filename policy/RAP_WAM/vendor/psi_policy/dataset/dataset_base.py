from __future__ import annotations

import bisect
from collections.abc import MutableMapping
import errno
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from zarr.storage import DirectoryStore


logger = logging.getLogger(__name__)

REQUIRED_INDEX_COLUMNS: Tuple[str, ...] = (
    "task",
    "volume_id",
    "start_idx",
    "end_idx",
)
SUPPORTED_ZARR_FORMAT = 2
_RETRYABLE_OS_ERRNOS = {
    code
    for code in (
        getattr(errno, "ENODATA", None),
        getattr(errno, "EIO", None),
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EINTR", None),
    )
    if code is not None
}


@dataclass(frozen=True)
class ClipIndexRecord:
    task: str
    volume_id: str
    start_idx: int
    end_idx: int

    @property
    def clip_length(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass(frozen=True)
class SampleWindow:
    clip_idx: int
    buffer_start: int
    buffer_end: int
    sample_start: int
    sample_end: int


@dataclass(frozen=True)
class ClipWindowSpan:
    clip_idx: int
    min_start: int
    max_start: int
    global_start: int

    @property
    def count(self) -> int:
        return self.max_start - self.min_start + 1

    @property
    def global_end(self) -> int:
        return self.global_start + self.count


class _InjectedZArrayStore(MutableMapping):
    """Read-only store wrapper that injects synthetic `.zarray` metadata."""

    def __init__(self, store: DirectoryStore, zarray_bytes: bytes):
        self.store = store
        self.zarray_bytes = zarray_bytes

    def __getitem__(self, key):
        if key == ".zarray":
            return self.zarray_bytes
        return self.store[key]

    def __setitem__(self, key, value):  # pragma: no cover - read-only wrapper
        raise TypeError("_InjectedZArrayStore is read-only")

    def __delitem__(self, key):  # pragma: no cover - read-only wrapper
        raise TypeError("_InjectedZArrayStore is read-only")

    def __iter__(self):
        yielded_zarray = False
        for key in self.store:
            if key == ".zarray":
                yielded_zarray = True
            yield key
        if not yielded_zarray:
            yield ".zarray"

    def __len__(self):
        return len(self.store) + (0 if ".zarray" in self.store else 1)

    def __contains__(self, key):
        return key == ".zarray" or key in self.store


class _LooseArrayGroup:
    """Filesystem-backed fallback for legacy layouts missing `data/.zgroup`."""

    def __init__(self, root: Path, *, expected_shapes: Mapping[str, Tuple[int, ...]] | None = None):
        self.root = Path(root)
        self._cache: Dict[str, object] = {}
        self._borrowed_metadata_cache: Dict[str, bytes | None] = {}
        self._warned_borrowed_metadata: set[str] = set()
        self.expected_shapes = {
            str(key): tuple(int(dim) for dim in value)
            for key, value in (expected_shapes or {}).items()
        }

    def __contains__(self, key: str) -> bool:
        child_path = self._resolve_child_path(key)
        if child_path is None:
            return False
        if (child_path / ".zarray").exists() or (child_path / ".zgroup").exists():
            return True
        return self._borrow_zarray_metadata_bytes(str(key), warn=False) is not None

    def __getitem__(self, key: str):
        child_path = self._resolve_child_path(key)
        if child_path is None:
            raise KeyError(key)

        cache_key = str(child_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if (child_path / ".zarray").exists():
            value = zarr.open(str(child_path), mode="r")
        elif (child_path / ".zgroup").exists():
            value = zarr.open_group(str(child_path), mode="r")
        else:
            zarray_bytes = self._borrow_zarray_metadata_bytes(str(key), warn=True)
            if zarray_bytes is None:
                raise KeyError(key)
            value = zarr.open_array(
                store=_InjectedZArrayStore(DirectoryStore(str(child_path)), zarray_bytes),
                mode="r",
            )
        self._cache[cache_key] = value
        return value

    def keys(self) -> list[str]:
        if not self.root.exists():
            return []

        keys: list[str] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            if child.name in self:
                keys.append(child.name)
        return keys

    def _resolve_child_path(self, key: str) -> Path | None:
        child = self.root / str(key)
        if not child.is_dir():
            return None
        return child

    def _borrow_zarray_metadata_bytes(self, key: str, *, warn: bool) -> bytes | None:
        cache_key = str(key)
        if cache_key in self._borrowed_metadata_cache:
            return self._borrowed_metadata_cache[cache_key]

        expected_shape = self.expected_shapes.get(cache_key)
        for candidate_name in self._iter_metadata_candidates(cache_key):
            candidate_bytes = self._load_candidate_metadata_bytes(
                candidate_name,
                expected_shape=expected_shape,
            )
            if candidate_bytes is None:
                continue
            self._borrowed_metadata_cache[cache_key] = candidate_bytes
            if warn and cache_key not in self._warned_borrowed_metadata:
                logger.warning(
                    "Borrowing .zarray metadata for modality '%s' from '%s' under %s",
                    cache_key,
                    candidate_name,
                    self.root,
                )
                self._warned_borrowed_metadata.add(cache_key)
            return candidate_bytes

        self._borrowed_metadata_cache[cache_key] = None
        return None

    def _iter_metadata_candidates(self, key: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def append(candidate: str | None) -> None:
            if candidate is None:
                return
            text = str(candidate).strip()
            if not text or text == key or text in seen:
                return
            seen.add(text)
            candidates.append(text)

        key_no_next = key[5:] if key.startswith("next_") else key
        key_with_next = key if key.startswith("next_") else f"next_{key}"
        swap_plain = _swap_left_right_token(key_no_next)
        swap_next = None if swap_plain is None else f"next_{swap_plain}"

        append(key_with_next)
        append(key_no_next)
        append(swap_plain)
        append(swap_next)

        if key.startswith("next_"):
            append(_swap_left_right_token(key))
        else:
            append(_swap_left_right_token(key))

        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            append(child.name)
        return candidates

    def _load_candidate_metadata_bytes(
        self,
        candidate_name: str,
        *,
        expected_shape: Tuple[int, ...] | None,
    ) -> bytes | None:
        candidate_zarray = self.root / candidate_name / ".zarray"
        if not candidate_zarray.exists():
            return None

        payload = json.loads(_read_text_with_retry(candidate_zarray, encoding="utf-8"))
        raw_shape = payload.get("shape")
        if expected_shape is not None:
            if not isinstance(raw_shape, Sequence) or len(raw_shape) <= 1:
                return None
            candidate_tail = tuple(int(dim) for dim in raw_shape[1:])
            if tuple(expected_shape) != candidate_tail:
                return None
        return candidate_zarray.read_bytes()


def _swap_left_right_token(value: str) -> str | None:
    if "left_" in value:
        return value.replace("left_", "right_", 1)
    if "right_" in value:
        return value.replace("right_", "left_", 1)
    return None


class DatasetBase(torch.utils.data.Dataset, ABC):
    """Manifest 驱动的数据集基类。"""

    def __init__(
        self,
        *,
        data_root: str | Path,
        datahouse_id: str,
        view_id: str,
        n_action_steps: int = 1,
        n_obs_steps: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        split: str = "train",
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.datahouse_id = _non_empty(datahouse_id, "datahouse_id")
        self.view_id = _non_empty(view_id, "view_id")
        self.action_horizon = _positive_int(n_action_steps, "n_action_steps")
        self.n_obs_steps = _positive_int(n_obs_steps, "n_obs_steps")
        self.pad_before = _clamp_pad(pad_before, self.action_horizon, "pad_before")
        self.pad_after = _clamp_pad(pad_after, self.action_horizon, "pad_after")
        self.split = _non_empty(split, "split")

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split 必须为 train/val/test")
        if self.n_obs_steps > self.action_horizon:
            raise ValueError("n_obs_steps 不能大于 action_horizon")

        self._view_root = self.data_root / self.datahouse_id / "views" / self.view_id
        self._tasks_root = self.data_root / self.datahouse_id / "tasks"
        self._zarr_cache: Dict[str, object] = {}
        self._validated_zarr_paths: set[str] = set()
        self._warned_legacy_data_layout = False

        self._modalities_need = _unique_non_empty(self.modalities_need, "modalities_need")
        self._manifest_modalities = self._load_manifest_modalities()

        self._clips = self._load_clip_index()
        self._modality_shapes = self._resolve_modality_shapes()
        (
            self._window_spans,
            self._window_span_ends,
            self._window_span_by_clip,
            self._num_windows,
        ) = self._build_window_index()

    @property
    @abstractmethod
    def modalities_need(self) -> Sequence[str]:
        """派生类声明的必需模态列表。"""
        raise NotImplementedError

    @abstractmethod
    def get_normalizer(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def modality_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return dict(self._modality_shapes)

    def __len__(self) -> int:
        return int(self._num_windows)

    def _resolve_window(self, idx: int) -> SampleWindow:
        total = len(self)
        if total == 0:
            raise IndexError("dataset is empty")
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"index out of range: {idx}")

        span_idx = bisect.bisect_right(self._window_span_ends, idx)
        span = self._window_spans[span_idx]
        offset = int(span.min_start + (idx - span.global_start))
        return self._make_window_for_offset(clip_idx=span.clip_idx, offset=offset)

    def _raise_data_error(self, message: str, exc_type: type[Exception] = ValueError) -> None:
        raise exc_type(message)

    def _load_manifest_modalities(self) -> Dict[str, Tuple[int, ...]]:
        manifest_path = self._view_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

        payload = json.loads(_read_text_with_retry(manifest_path, encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest 内容必须是 JSON object")

        spec = payload.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("manifest.spec 缺失或格式错误")

        raw_modalities = spec.get("modalities")
        if not isinstance(raw_modalities, Mapping):
            raise ValueError("manifest.spec.modalities 必须是映射")

        parsed: Dict[str, Tuple[int, ...]] = {}
        for raw_name, raw_desc in raw_modalities.items():
            name = str(raw_name).strip()
            if not name:
                continue
            if not isinstance(raw_desc, Mapping):
                self._raise_data_error(f"模态 {name} 的描述必须是映射")
                continue

            raw_shape = raw_desc.get("shape")
            if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
                self._raise_data_error(f"模态 {name} 缺少 shape 列表")
                continue

            shape: List[int] = []
            invalid = False
            for dim in raw_shape:
                dim_i = int(dim)
                if dim_i <= 0:
                    invalid = True
                    self._raise_data_error(f"模态 {name} 的 shape 维度必须为正整数")
                    break
                shape.append(dim_i)
            if invalid:
                continue

            parsed[name] = tuple(shape)

        if not parsed:
            raise ValueError("manifest.spec.modalities 为空")
        return parsed

    def _load_clip_index(self) -> List[ClipIndexRecord]:
        index_path = self._view_root / f"clip_index_{self.split}.parquet"
        if not index_path.exists():
            if self.split == "val":
                return []
            raise FileNotFoundError(f"clip index 不存在: {index_path}")

        table = pq.read_table(index_path)
        columns = set(table.column_names)
        missing = [name for name in REQUIRED_INDEX_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"clip index 缺少必要列: {', '.join(sorted(missing))}")

        data = table.to_pydict()
        records: List[ClipIndexRecord] = []
        for row in range(table.num_rows):
            task = _non_empty(str(data["task"][row]), "task")
            volume_id = _non_empty(str(data["volume_id"][row]), "volume_id")
            start_idx = int(data["start_idx"][row])
            end_idx = int(data["end_idx"][row])
            if end_idx < start_idx:
                raise ValueError(f"clip 索引非法: end_idx({end_idx}) < start_idx({start_idx})")
            records.append(
                ClipIndexRecord(
                    task=task,
                    volume_id=volume_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
        return records

    def _resolve_modality_shapes(self) -> Dict[str, Tuple[int, ...]]:
        resolved: Dict[str, Tuple[int, ...]] = {}
        missing: List[str] = []

        for key in self._modalities_need:
            if key in self._manifest_modalities:
                resolved[key] = self._manifest_modalities[key]
            else:
                missing.append(key)

        if not missing:
            return resolved

        infer_failed: List[str] = []
        for key in missing:
            inferred = self._infer_modality_shape_from_data(key)
            if inferred is None:
                infer_failed.append(key)
                resolved[key] = (1,)
            else:
                resolved[key] = inferred

        if infer_failed:
            self._raise_data_error(
                "manifest.spec.modalities 缺少必需模态: " + ", ".join(sorted(infer_failed))
            )

        return resolved

    def _infer_modality_shape_from_data(self, modality: str) -> Tuple[int, ...] | None:
        for clip in self._clips:
            group = self._open_clip_group(clip)
            if modality not in group:
                continue
            shape = tuple(int(v) for v in group[modality].shape[1:])
            return shape
        return None

    def _build_window_index(
        self,
    ) -> tuple[list[ClipWindowSpan], list[int], list[ClipWindowSpan | None], int]:
        spans: list[ClipWindowSpan] = []
        span_ends: list[int] = []
        span_by_clip: list[ClipWindowSpan | None] = [None] * len(self._clips)
        total = 0

        for clip_idx, clip in enumerate(self._clips):
            bounds = self._compute_window_offset_bounds(clip)
            if bounds is None:
                continue
            min_start, max_start = bounds
            span = ClipWindowSpan(
                clip_idx=clip_idx,
                min_start=min_start,
                max_start=max_start,
                global_start=total,
            )
            spans.append(span)
            span_ends.append(span.global_end)
            span_by_clip[clip_idx] = span
            total = span.global_end
        return spans, span_ends, span_by_clip, total

    def _compute_window_offset_bounds(self, clip: ClipIndexRecord) -> tuple[int, int] | None:
        clip_len = int(clip.clip_length)
        if clip_len <= 0:
            return None

        min_start = -self.pad_before
        max_start = clip_len - self.action_horizon + self.pad_after
        if max_start < min_start:
            return None
        return int(min_start), int(max_start)

    def _make_window_for_offset(self, *, clip_idx: int, offset: int) -> SampleWindow:
        clip = self._clips[int(clip_idx)]
        clip_len = int(clip.clip_length)
        offset_i = int(offset)
        buffer_start = max(offset_i, 0) + int(clip.start_idx)
        buffer_end = min(offset_i + self.action_horizon, clip_len) + int(clip.start_idx)
        sample_start = buffer_start - (offset_i + int(clip.start_idx))
        sample_end = self.action_horizon - (
            (offset_i + self.action_horizon + int(clip.start_idx)) - buffer_end
        )

        if sample_end - sample_start != buffer_end - buffer_start:
            raise ValueError("window length mismatch")

        return SampleWindow(
            clip_idx=int(clip_idx),
            buffer_start=int(buffer_start),
            buffer_end=int(buffer_end),
            sample_start=int(sample_start),
            sample_end=int(sample_end),
        )

    def _load_modality_window(
        self,
        *,
        window: SampleWindow,
        modality: str,
        steps: int,
    ) -> np.ndarray:
        if modality not in self._modality_shapes:
            raise KeyError(f"modality '{modality}' 不在 modalities_need 中")

        clip = self._clips[window.clip_idx]
        group = self._open_clip_group(clip)
        if modality not in group:
            self._raise_data_error(
                f"模态 {modality} 在 zarr 不存在: task={clip.task} volume={clip.volume_id}",
                exc_type=KeyError,
            )
            return self._make_zeros(modality=modality, target_steps=steps)

        arr = np.asarray(group[modality][window.buffer_start:window.buffer_end])
        if arr.shape[0] == 0:
            return self._make_zeros(modality=modality, target_steps=steps)

        arr = _pad_array_time_edge(
            arr,
            sample_start=window.sample_start,
            sample_end=window.sample_end,
            target_len=self.action_horizon,
        )
        if steps != self.action_horizon:
            arr = arr[:steps]
        return arr

    def _load_obs_prefix_window(
        self,
        *,
        window: SampleWindow,
        modality: str,
        steps: int,
    ) -> np.ndarray:
        if modality not in self._modality_shapes:
            raise KeyError(f"modality '{modality}' 不在 modalities_need 中")
        if steps <= 0 or steps > self.action_horizon:
            raise ValueError(
                f"obs prefix steps 非法: steps={steps}, action_horizon={self.action_horizon}"
            )

        clip = self._clips[window.clip_idx]
        group = self._open_clip_group(clip)
        if modality not in group:
            self._raise_data_error(
                f"模态 {modality} 在 zarr 不存在: task={clip.task} volume={clip.volume_id}",
                exc_type=KeyError,
            )
            return self._make_zeros(modality=modality, target_steps=steps)

        raw_len = max(window.buffer_end - window.buffer_start, 0)
        if raw_len == 0:
            return self._make_zeros(modality=modality, target_steps=steps)

        if steps <= window.sample_start:
            need_read = 1
        elif steps < window.sample_end:
            need_read = steps - window.sample_start
        else:
            need_read = raw_len

        arr = np.asarray(group[modality][window.buffer_start : window.buffer_start + need_read])
        if arr.shape[0] == 0:
            return self._make_zeros(modality=modality, target_steps=steps)

        if steps <= window.sample_start:
            return np.repeat(arr[:1], steps, axis=0)

        left_pad = min(window.sample_start, steps)
        actual_len = min(arr.shape[0], max(steps - left_pad, 0))

        out = np.empty((steps,) + arr.shape[1:], dtype=arr.dtype)
        if left_pad > 0:
            out[:left_pad] = arr[0]
        if actual_len > 0:
            out[left_pad : left_pad + actual_len] = arr[:actual_len]

        filled = left_pad + actual_len
        if filled < steps:
            out[filled:] = arr[-1]
        return out

    def _make_zeros(self, *, modality: str, target_steps: int) -> np.ndarray:
        tail_shape = self._modality_shapes.get(modality, (1,))
        return np.zeros((target_steps,) + tail_shape, dtype=np.float32)

    def read_clip_modality(self, clip: ClipIndexRecord, modality: str) -> np.ndarray:
        group = self._open_clip_group(clip)
        if modality not in group:
            self._raise_data_error(
                f"模态 {modality} 在 zarr 不存在: task={clip.task} volume={clip.volume_id}",
                exc_type=KeyError,
            )
            return self._make_zeros(modality=modality, target_steps=clip.clip_length)

        arr = np.asarray(group[modality][clip.start_idx : clip.end_idx + 1])
        if arr.shape[0] == clip.clip_length:
            return arr
        if arr.shape[0] == 0:
            return self._make_zeros(modality=modality, target_steps=clip.clip_length)
        if arr.shape[0] > clip.clip_length:
            return arr[: clip.clip_length]

        padded = np.empty((clip.clip_length,) + arr.shape[1:], dtype=arr.dtype)
        padded[: arr.shape[0]] = arr
        padded[arr.shape[0] :] = arr[-1]
        return padded

    def _open_clip_group(self, clip: ClipIndexRecord):
        cache_key = f"{clip.task}/{clip.volume_id}"
        if cache_key in self._zarr_cache:
            return self._zarr_cache[cache_key]

        zarr_path = self._tasks_root / clip.task / f"{clip.volume_id}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"zarr 不存在: {zarr_path}")

        self._assert_zarr_format(zarr_path)

        root = _open_zarr_group_with_retry(zarr_path)
        if "data" in root:
            group = root["data"]
        else:
            data_path = zarr_path / "data"
            if data_path.is_dir():
                group = _LooseArrayGroup(data_path, expected_shapes=self._modality_shapes)
                if not self._warned_legacy_data_layout:
                    logger.warning(
                        "Detected legacy zarr layout without data/.zgroup under %s; "
                        "falling back to filesystem-backed modality lookup.",
                        zarr_path,
                    )
                    self._warned_legacy_data_layout = True
            else:
                group = root
        self._zarr_cache[cache_key] = group
        return group

    def _assert_zarr_format(self, path: Path) -> None:
        key = str(path)
        if key in self._validated_zarr_paths:
            return

        zarr_v3_marker = path / "zarr.json"
        if zarr_v3_marker.exists():
            raise ValueError(
                f"detected zarr v3 at {path}; current stage only supports zarr v{SUPPORTED_ZARR_FORMAT}"
            )

        zgroup_path = path / ".zgroup"
        if not zgroup_path.exists():
            raise ValueError(f"zarr v{SUPPORTED_ZARR_FORMAT} metadata '.zgroup' not found: {path}")

        payload = json.loads(_read_text_with_retry(zgroup_path, encoding="utf-8"))
        zarr_format = int(payload.get("zarr_format", -1))
        if zarr_format != SUPPORTED_ZARR_FORMAT:
            raise ValueError(
                f"unsupported zarr format at {path}: expected {SUPPORTED_ZARR_FORMAT}, got {zarr_format}"
            )

        self._validated_zarr_paths.add(key)


def _pad_array_time_edge(
    array: np.ndarray,
    sample_start: int,
    sample_end: int,
    target_len: int,
) -> np.ndarray:
    current_len = array.shape[0]
    if sample_start == 0 and sample_end == target_len and current_len == target_len:
        return array
    if current_len != (sample_end - sample_start):
        raise ValueError("array length mismatch for window padding")

    padded = np.empty((target_len,) + array.shape[1:], dtype=array.dtype)
    padded[sample_start:sample_end] = array
    if sample_start > 0:
        padded[:sample_start] = array[0]
    if sample_end < target_len:
        padded[sample_end:] = array[-1]
    return padded


def _is_retryable_os_error(exc: OSError) -> bool:
    if exc.errno in _RETRYABLE_OS_ERRNOS:
        return True
    message = str(exc).lower()
    return (
        "no data available" in message
        or "stale file handle" in message
        or "resource temporarily unavailable" in message
    )


def _read_text_with_retry(
    path: Path,
    *,
    encoding: str = "utf-8",
    retries: int = 8,
    retry_delay_sec: float = 0.2,
) -> str:
    attempts = max(1, int(retries))
    for idx in range(attempts):
        try:
            return path.read_text(encoding=encoding)
        except OSError as exc:
            if (idx + 1) >= attempts or not _is_retryable_os_error(exc):
                raise
            logger.warning(
                "读取文件失败，将重试 (%s/%s): path=%s err=%r",
                idx + 1,
                attempts,
                path,
                exc,
            )
            if retry_delay_sec > 0:
                time.sleep(float(retry_delay_sec))
    raise RuntimeError(f"读取文件失败: {path}")


def _open_zarr_group_with_retry(
    path: Path,
    *,
    retries: int = 8,
    retry_delay_sec: float = 0.2,
) -> zarr.Group:
    attempts = max(1, int(retries))
    for idx in range(attempts):
        try:
            return zarr.open_group(str(path), mode="r")
        except OSError as exc:
            if (idx + 1) >= attempts or not _is_retryable_os_error(exc):
                raise
            logger.warning(
                "打开 zarr 失败，将重试 (%s/%s): path=%s err=%r",
                idx + 1,
                attempts,
                path,
                exc,
            )
            if retry_delay_sec > 0:
                time.sleep(float(retry_delay_sec))
    raise RuntimeError(f"打开 zarr 失败: {path}")


def _non_empty(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} 不能为空")
    return cleaned


def _positive_int(value: int, name: str) -> int:
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value_i


def _non_negative_int(value: int, name: str) -> int:
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{name} 必须为非负整数")
    return value_i


def _clamp_pad(value: int, action_horizon: int, name: str) -> int:
    value_i = _non_negative_int(value, name)
    return min(value_i, max(action_horizon - 1, 0))


def _unique_non_empty(values: Sequence[str], name: str) -> List[str]:
    uniq: List[str] = []
    seen = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    if not uniq:
        raise ValueError(f"{name} 不能为空")
    return uniq


__all__ = [
    "DatasetBase",
    "ClipIndexRecord",
    "SampleWindow",
]
