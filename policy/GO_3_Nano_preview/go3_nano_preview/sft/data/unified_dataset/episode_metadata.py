from __future__ import annotations
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import jsonlines
from go3_nano_preview.sft.data.snapshot_index import SnapshotLogicalView, SnapshotRecapMap, SnapshotRecordStore
from go3_nano_preview.sft.data.utils.WAM_policy_camera import infer_camera_view_ids

class _SourceProjection(Sequence[Any]):

    def __init__(self, metadata: EpisodeMetadata, source_values: Sequence[Any]) -> None:
        self._metadata = metadata
        self._source_values = tuple(source_values)
        if len(self._source_values) != metadata.source_count:
            raise ValueError('source value count does not match metadata source count')

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> Any:
        value = self._source_values[self._metadata.source_id(int(index))]
        return list(value) if isinstance(value, list) else value

class EpisodeMetadata(Sequence[dict[str, Any]]):

    def __init__(self, records: Sequence[dict[str, Any]], *, source_names: Sequence[str], source_configs: Sequence[dict[str, Any]], repeats: Sequence[int], physical_episode_count: int, eager_source_ids: Sequence[int] | None=None, manifest_path: str | None=None, identity: str | None=None) -> None:
        self._records = records
        self.source_names = tuple((str(name) for name in source_names))
        self.source_configs = tuple((dict(config) for config in source_configs))
        self.repeats = tuple((int(repeat) for repeat in repeats))
        self.physical_episode_count = int(physical_episode_count)
        self.manifest_path = manifest_path
        self.identity = identity
        self._eager_source_ids = tuple((int(source_id) for source_id in eager_source_ids)) if eager_source_ids is not None else None
        if len(self.source_names) != len(self.source_configs):
            raise ValueError('metadata source names and configs have different lengths')
        if len(self.repeats) != len(self.source_names):
            raise ValueError('metadata repeat count does not match source count')
        if self._eager_source_ids is not None and len(self._eager_source_ids) != len(self._records):
            raise ValueError('eager source IDs do not match metadata record count')

    @property
    def source_count(self) -> int:
        return len(self.source_names)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[int(index)]

    def source_id(self, index: int) -> int:
        if self._eager_source_ids is None:
            return int(self._records.source_id(int(index)))
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._eager_source_ids[index]

    def project(self, source_values: Sequence[Any]) -> Sequence[Any]:
        return _SourceProjection(self, source_values)

    def open_recap(self, name: str, eager_path: str | None, *, cache_size: int) -> Any:
        if self.manifest_path is not None:
            return SnapshotRecapMap(self.manifest_path, name, cache_size=cache_size)
        if eager_path is None:
            raise ValueError(f'{name} recap path is required')
        with Path(eager_path).open('r', encoding='UTF-8') as handle:
            return json.load(handle)

    def cache_metrics(self) -> dict[str, int]:
        metrics = getattr(self._records, 'cache_metrics', None)
        return metrics() if metrics is not None else {}

    def close(self) -> None:
        close = getattr(self._records, 'close', None)
        if close is not None:
            close()

def load_episode_metadata(*, jsonl_paths: Sequence[str], video_folders: Sequence[str], dataset_sources: Sequence[str], dataset_names: Sequence[str], fps_values: Sequence[float] | None, camera_names: Sequence[Sequence[str]], view_ids: Sequence[Sequence[int]] | None, action_spaces: Sequence[str], stat_files: Sequence[str], repeats: Sequence[int] | None, manifest_path: str | None, episode_cache_size: int, mmap_cache_size: int, max_samples_per_source: int | None) -> EpisodeMetadata:
    if manifest_path:
        return _load_indexed_metadata(manifest_path=manifest_path, repeats=repeats, episode_cache_size=episode_cache_size, mmap_cache_size=mmap_cache_size)
    return _load_eager_metadata(jsonl_paths=jsonl_paths, video_folders=video_folders, dataset_sources=dataset_sources, dataset_names=dataset_names, fps_values=fps_values, camera_names=camera_names, view_ids=view_ids, action_spaces=action_spaces, stat_files=stat_files, repeats=repeats, max_samples_per_source=max_samples_per_source)

def _load_indexed_metadata(*, manifest_path: str, repeats: Sequence[int] | None, episode_cache_size: int, mmap_cache_size: int) -> EpisodeMetadata:
    physical_store = SnapshotRecordStore(manifest_path, cache_size=episode_cache_size, mmap_cache_size=mmap_cache_size)
    if repeats is None:
        repeat_values = [1] * physical_store.source_count
    elif isinstance(repeats, list | tuple):
        repeat_values = list(repeats)
    else:
        raise ValueError('indexed metadata requires one repeat value per source')
    if len(repeat_values) != physical_store.source_count:
        raise ValueError(f'metadata requires one repeat value per source: repeats={len(repeat_values)}, sources={physical_store.source_count}')
    records = SnapshotLogicalView(physical_store, repeats=repeat_values)
    source_configs = [dict(physical_store.source_config(source_id)) for source_id in range(physical_store.source_count)]
    source_names = []
    for (source_id, source_config) in enumerate(source_configs):
        source_name = str(source_config.get('dataset_name') or physical_store.source_names[source_id])
        source_config.setdefault('dataset_name', source_name)
        source_names.append(source_name)
    return EpisodeMetadata(records, source_names=source_names, source_configs=source_configs, repeats=repeat_values, physical_episode_count=physical_store.episode_count, manifest_path=str(manifest_path), identity=physical_store.snapshot_id)

def _load_eager_metadata(*, jsonl_paths: Sequence[str], video_folders: Sequence[str], dataset_sources: Sequence[str], dataset_names: Sequence[str], fps_values: Sequence[float] | None, camera_names: Sequence[Sequence[str]], view_ids: Sequence[Sequence[int]] | None, action_spaces: Sequence[str], stat_files: Sequence[str], repeats: Sequence[int] | None, max_samples_per_source: int | None) -> EpisodeMetadata:
    source_count = len(jsonl_paths)
    _require_source_count('video_folders', video_folders, source_count)
    _require_source_count('dataset_sources', dataset_sources, source_count)
    _require_source_count('dataset_names', dataset_names, source_count)
    _require_source_count('camera_names', camera_names, source_count)
    normalized_fps = [30.0] * source_count if fps_values is None else [float(value) for value in fps_values]
    _require_source_count('fps_values', normalized_fps, source_count)
    normalized_view_ids = [infer_camera_view_ids(list(names)) for names in camera_names] if view_ids is None else [list(values) for values in view_ids]
    _require_source_count('view_ids', normalized_view_ids, source_count)
    repeat_values = [1] * source_count
    if repeats is not None:
        for source_id in range(min(source_count, len(repeats))):
            repeat_values[source_id] = int(repeats[source_id])
    source_configs = []
    records: list[dict[str, Any]] = []
    source_ids: list[int] = []
    physical_episode_count = 0
    for (source_id, jsonl_path) in enumerate(jsonl_paths):
        cameras = list(camera_names[source_id])
        views = list(normalized_view_ids[source_id])
        if len(views) != len(cameras):
            raise ValueError(f'view_ids length {len(views)} does not match cam_use length {len(cameras)} for dataset index {source_id}')
        source_config = {'dataset_name': str(dataset_names[source_id]), 'video_folder': str(video_folders[source_id]), 'dataset_source': str(dataset_sources[source_id]), 'fps': normalized_fps[source_id], 'cam_use': cameras, 'view_ids': views}
        if source_id < len(action_spaces):
            source_config['action_space'] = action_spaces[source_id]
        if source_id < len(stat_files):
            source_config['stat_file'] = stat_files[source_id]
        source_configs.append(source_config)
        with jsonlines.open(jsonl_path, 'r') as jsonl_file:
            raw_records = list(jsonl_file)
        if max_samples_per_source is not None and max_samples_per_source > 0:
            raw_records = raw_records[:max_samples_per_source]
        physical_episode_count += len(raw_records)
        repeat = repeat_values[source_id]
        processed_records = raw_records * repeat if repeat > 1 else raw_records
        records.extend(processed_records)
        source_ids.extend([source_id] * len(processed_records))
    return EpisodeMetadata(records, source_names=[str(name) for name in dataset_names], source_configs=source_configs, repeats=repeat_values, physical_episode_count=physical_episode_count, eager_source_ids=source_ids)

def _require_source_count(name: str, values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f'{name} must have one value per metadata source: {len(values)} != {expected}')
