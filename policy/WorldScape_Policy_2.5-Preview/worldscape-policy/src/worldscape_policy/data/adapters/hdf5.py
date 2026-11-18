from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence

from torch.utils.data import Dataset

from worldscape_policy.data.adapters.common import (
    Any,
    AuditedVisualPromptOverride,
    EventSample,
    HistorySampler,
    LanguageTemporalPacker,
    Path,
    _build_sample,
    _effective_length,
    _eligible_temporal_anchors,
    _first,
    _image_sequence,
    _metadata_task,
    _optional_dependency,
    _read_jsonl,
    _resolve_episode_path,
    _selected_chunk_length,
    _state_and_action,
    _task_records,
    _temporal_labels,
    np,
)

class NativeHDF5Dataset(Dataset[EventSample]):
    """Read repository raw/GEAR HDF5 episodes as validated native samples."""

    INDEX_CACHE_VERSION = 1
    INDEX_CACHE_FILENAME = f"native_hdf5_index_cache_v{INDEX_CACHE_VERSION}.npz"

    def __init__(
        self,
        data_root: str | Path,
        *,
        data_roots: Sequence[str | Path] | str | Path | None = None,
        visual_prompt: str = "demo",
        file_pattern: str = "**/*.hdf5",
        embodiment: str = "agilex",
        action_horizon: int = 24,
        max_event_chunks: int = 16,
        history_num_frames: int = 8,
        history_stride: int = 24,
        history_window: int | None = None,
        temporal_packing: bool = False,
        temporal_anchor_index: int = 0,
        max_chunk_size: int = 4,
        respect_trajectory_segments: bool = True,
        context_sampling_mode: str | None = None,
        context_video_len: int = 50,
        ctx_head_only: bool = False,
        wo_norm: bool = True,
        seed: int = 0,
        max_episodes: int | None = None,
        step_stride: int = 1,
        index_cache: bool = True,
        index_cache_dir: str | Path | None = None,
        cache_video_resize_width: int | None = None,
        cache_video_resize_height: int | None = None,
        cache_video_resize_interpolation: str = "area",
        allow_incompatible_visual_prompts: bool = False,
        visual_prompt_override_audit_reason: str | None = None,
    ) -> None:
        if data_root is None or not str(data_root):
            raise ValueError("data_root is required")
        roots = [Path(data_root)]
        if data_roots is not None:
            if isinstance(data_roots, (str, Path)):
                roots.append(Path(data_roots))
            else:
                roots.extend(Path(item) for item in data_roots)
        self.roots = tuple(dict.fromkeys(roots))
        if step_stride <= 0:
            raise ValueError("step_stride must be positive")
        if (cache_video_resize_width is None) ^ (cache_video_resize_height is None):
            raise ValueError(
                "cache_video_resize_width and cache_video_resize_height "
                "must be both set or both None"
            )
        if cache_video_resize_width is not None and cache_video_resize_width <= 0:
            raise ValueError("cache_video_resize_width must be positive")
        if cache_video_resize_height is not None and cache_video_resize_height <= 0:
            raise ValueError("cache_video_resize_height must be positive")
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided")
        metadata = next(
            (
                root / "meta" / "episodes.jsonl"
                for root in self.roots
                if (root / "meta" / "episodes.jsonl").is_file()
            ),
            None,
        )
        if metadata is not None:
            self.metadata_path = metadata
            self.episodes = _read_jsonl(metadata)
            if max_episodes is not None:
                self.episodes = self.episodes[:max_episodes]
            self.paths = [
                _resolve_episode_path(item, self.roots) for item in self.episodes
            ]
        else:
            self.metadata_path = None
            self.paths = sorted(
                {
                    path
                    for root in self.roots
                    for path in root.glob(file_pattern)
                    if path.is_file()
                }
            )
            if max_episodes is not None:
                self.paths = self.paths[:max_episodes]
            self.episodes = [
                {"episode_index": index, "path": str(path)}
                for index, path in enumerate(self.paths)
            ]
        if not self.paths:
            raise FileNotFoundError(f"no HDF5 episodes found under {self.roots}")
        tasks_path = next(
            (
                root / "meta" / "tasks.jsonl"
                for root in self.roots
                if (root / "meta" / "tasks.jsonl").is_file()
            ),
            None,
        )
        self.tasks = _task_records(tasks_path)
        self.visual_prompt = visual_prompt
        self.embodiment = embodiment
        self.action_horizon = action_horizon
        self.max_event_chunks = max_event_chunks
        self.history_sampler = HistorySampler(history_num_frames, history_stride)
        self.history_window = history_window
        self.temporal_packing = temporal_packing
        self.temporal_anchor_index = temporal_anchor_index
        self.temporal_max_chunk_size = max_chunk_size
        self.respect_trajectory_segments = bool(respect_trajectory_segments)
        self.context_sampling_mode = context_sampling_mode
        self.context_video_len = context_video_len
        self.ctx_head_only = ctx_head_only
        self.wo_norm = wo_norm
        self.seed = seed
        self.step_stride = int(step_stride)
        self.index_cache = bool(index_cache)
        self.index_cache_dir = (
            None if index_cache_dir is None else Path(index_cache_dir)
        )
        self.cache_video_resize_width = cache_video_resize_width
        self.cache_video_resize_height = cache_video_resize_height
        self.cache_video_resize_interpolation = cache_video_resize_interpolation
        self.visual_prompt_override = (
            AuditedVisualPromptOverride(
                enabled=True,
                audit_reason=visual_prompt_override_audit_reason,
            )
            if allow_incompatible_visual_prompts
            else None
        )
        self._index_arrays: dict[int, dict[str, np.ndarray]] = {}
        self._sampling_progress = np.empty(0, dtype=np.float64)
        self.index = self._build_index()

    def __len__(self) -> int:
        return len(self.index)

    def sampling_progress(self, indices: np.ndarray) -> np.ndarray:
        """Return each sample anchor's ratio within its is_exec-filtered episode."""

        return self._sampling_progress[np.asarray(indices, dtype=np.int64)]

    def _read_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        h5py = _optional_dependency("h5py", "train")
        arrays: dict[str, np.ndarray] = {}
        with h5py.File(self.paths[episode_index], "r") as handle:
            def collect(name: str, value: Any) -> None:
                if isinstance(value, h5py.Dataset):
                    arrays[name.replace("/", ".")] = np.asarray(value[()])

            handle.visititems(collect)
            for key, value in handle.attrs.items():
                arrays.setdefault(str(key), np.asarray(value))
        _metadata_task(arrays, self.episodes[episode_index], self.tasks)
        return arrays

    def shard_ranges(self, shard_size: int) -> tuple[tuple[int, int], ...]:
        """Group complete episodes into legacy-compatible ~shard_size shards."""

        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        # ``self.index`` may already be thinned by step_stride. Keep the public
        # shard_size in legacy raw-step units so enabling stride does not merge
        # many more complete episodes into each decoded shard.
        indexed_shard_size = max(1, int(np.ceil(shard_size / self.step_stride)))
        counts = np.bincount(
            np.asarray([episode for episode, _ in self.index], dtype=np.int64),
            minlength=len(self.paths),
        )
        total = int(counts.sum())
        num_shards = int(np.ceil(total / indexed_shard_size))
        cutoffs = np.linspace(0, total, num_shards + 1)[1:]
        ranges: list[tuple[int, int]] = []
        start = 0
        cursor = 0
        cutoff_index = 0
        for count in counts.tolist():
            cursor += int(count)
            if cutoff_index < num_shards - 1 and cursor >= cutoffs[cutoff_index]:
                ranges.append((start, cursor))
                start = cursor
                cutoff_index += 1
        if start < cursor:
            ranges.append((start, cursor))
        return tuple(ranges)

    def _cache_resize_target(self) -> tuple[int, int] | None:
        if self.cache_video_resize_width is None or self.cache_video_resize_height is None:
            return None
        return int(self.cache_video_resize_width), int(self.cache_video_resize_height)

    def _cache_resize_interpolation(
        self, src_h: int, src_w: int, dst_h: int, dst_w: int
    ) -> int:
        cv2 = _optional_dependency("cv2", "train")
        mode = str(self.cache_video_resize_interpolation).lower()
        if mode == "nearest":
            return cv2.INTER_NEAREST
        if mode == "linear":
            return cv2.INTER_LINEAR
        if mode == "cubic":
            return cv2.INTER_CUBIC
        if mode == "lanczos":
            return cv2.INTER_LANCZOS4
        if mode == "auto":
            return cv2.INTER_AREA if (dst_h < src_h or dst_w < src_w) else cv2.INTER_LINEAR
        return cv2.INTER_AREA

    def _maybe_resize_cached_frames(self, frames: np.ndarray) -> np.ndarray:
        target = self._cache_resize_target()
        if target is None or frames.ndim != 4:
            return frames
        dst_w, dst_h = target
        src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
        if src_h == dst_h and src_w == dst_w:
            return frames
        cv2 = _optional_dependency("cv2", "train")
        interpolation = self._cache_resize_interpolation(src_h, src_w, dst_h, dst_w)
        resized = [
            cv2.resize(frame, (dst_w, dst_h), interpolation=interpolation)
            for frame in frames
        ]
        return np.stack(resized, axis=0).astype(np.uint8, copy=False)

    def load_shard(self, start: int, stop: int) -> dict[int, dict[str, np.ndarray]]:
        """Materialize and decode every complete HDF5 episode in one shard."""

        profile = os.environ.get("WSP_DATALOADER_PROFILE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        shard_start = time.perf_counter()
        episode_indices = dict.fromkeys(
            episode_index for episode_index, _ in self.index[start:stop]
        )
        shard: dict[int, dict[str, np.ndarray]] = {}
        camera_prefixes = (
            "observations.images.",
            "observation.images.",
            "observations.camera.",
            "observation.camera.",
        )
        decoded_frames = 0
        read_decode_time = 0.0
        for episode_index in episode_indices:
            episode_start = time.perf_counter()
            arrays = self._read_arrays(int(episode_index))
            # Legacy get_shard decoded every selected camera once up front and
            # retained RGB arrays for all samples drawn from this shard.  Keep
            # that contract: caching only base64/JPEG strings makes each sample
            # re-decode the complete episode before temporal packing.
            for key, value in tuple(arrays.items()):
                if key.startswith(camera_prefixes):
                    arrays[key] = self._maybe_resize_cached_frames(
                        _image_sequence(value)
                    )
                    decoded_frames += int(arrays[key].shape[0])
            state, action = _state_and_action(arrays, None)
            arrays["observation.eef6d"] = state
            arrays["action.eef6d"] = action
            shard[int(episode_index)] = arrays
            read_decode_time += time.perf_counter() - episode_start
        if profile:
            print(
                json.dumps(
                    {
                        "dataloader_profile": "hdf5_load_shard",
                        "start": int(start),
                        "stop": int(stop),
                        "episodes": len(episode_indices),
                        "decoded_frames": decoded_frames,
                        "read_decode_s": round(read_decode_time, 6),
                        "total_s": round(time.perf_counter() - shard_start, 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return shard

    def _read_index_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        """Read only lengths and temporal labels; never materialize image payloads."""

        cached = self._index_arrays.get(episode_index)
        if cached is not None:
            return cached
        h5py = _optional_dependency("h5py", "train")
        arrays: dict[str, np.ndarray] = {}
        metadata_names = {
            "task_index",
            "meta.task_index",
            "annotation.task_index",
            "high_level_instruction",
            "language",
            "task",
            "annotation.task",
            "event_instruction",
            "subtask",
            "annotation.subtask",
            "is_exec",
            "meta.is_exec",
        }
        source_names = {
            "action.eef6d",
            "actions.eef6d",
            "actions",
            "action",
            "robot_state",
            "observation.state",
            "observations.state",
            "observation.eef6d",
            "state",
            "observations.end_pose",
            "observation.end_pose",
        }
        source_lengths: list[int] = []
        image_lengths: list[int] = []
        with h5py.File(self.paths[episode_index], "r") as handle:
            def collect(name: str, value: Any) -> None:
                if not isinstance(value, h5py.Dataset):
                    return
                key = name.replace("/", ".")
                if key in metadata_names:
                    arrays[key] = np.asarray(value[()])
                if key in source_names and value.shape:
                    source_lengths.append(int(value.shape[0]))
                if (
                    key.startswith(
                        (
                            "observations.images.",
                            "observation.images.",
                            "observations.camera.",
                            "observation.camera.",
                        )
                    )
                    or key in {"observations.video", "observation.video", "video"}
                ) and value.shape:
                    image_lengths.append(int(value.shape[0]))

            handle.visititems(collect)
            for key, value in handle.attrs.items():
                if str(key) in metadata_names:
                    arrays.setdefault(str(key), np.asarray(value))
        fallback_length = min(source_lengths) if source_lengths else 0
        episode_length = int(
            self.episodes[episode_index].get("length", fallback_length)
        )
        known_lengths = [
            value
            for value in (*source_lengths, episode_length, *image_lengths)
            if value > 0
        ]
        if not known_lengths:
            raise ValueError(
                f"episode {self.paths[episode_index]} has no indexable temporal length"
            )
        length = min(known_lengths)
        arrays.setdefault("action", np.empty((length, 0), dtype=np.float32))
        _metadata_task(
            arrays,
            self.episodes[episode_index],
            self.tasks,
            length=length,
        )
        self._index_arrays[episode_index] = arrays
        return arrays

    def _index_cache_path(self) -> Path | None:
        if not self.index_cache or self.metadata_path is None:
            return None
        if self.index_cache_dir is not None:
            cache_dir = self.index_cache_dir
        else:
            env_cache = os.environ.get("WSP_DATA_CACHE_DIR") or os.environ.get(
                "WSP_CACHE_DIR"
            )
            cache_dir = (
                Path(env_cache) / "data-index"
                if env_cache
                else Path(os.environ.get("TMPDIR", "/tmp"))
                / "worldscape-policy-data-index"
            )
        dataset_id = hashlib.sha256(
            str(self.metadata_path.parent.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:16]
        return cache_dir / f"{dataset_id}-{self.INDEX_CACHE_FILENAME}"

    def _index_cache_key(self) -> str:
        path_fingerprints = []
        for path in self.paths:
            stat = path.stat()
            path_fingerprints.append(
                {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        metadata_stat = self.metadata_path.stat() if self.metadata_path is not None else None
        payload = {
            "version": self.INDEX_CACHE_VERSION,
            "metadata": (
                None
                if metadata_stat is None
                else {
                    "path": str(self.metadata_path),
                    "size": int(metadata_stat.st_size),
                    "mtime_ns": int(metadata_stat.st_mtime_ns),
                }
            ),
            "paths": path_fingerprints,
            "tasks": self.tasks,
            "temporal_packing": self.temporal_packing,
            "temporal_anchor_index": self.temporal_anchor_index,
            "max_chunk_size": self.temporal_max_chunk_size,
            "respect_trajectory_segments": self.respect_trajectory_segments,
            "action_horizon": self.action_horizon,
            "max_event_chunks": self.max_event_chunks,
            "step_stride": self.step_stride,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_index_cache(self) -> list[tuple[int, int]] | None:
        cache_path = self._index_cache_path()
        if cache_path is None or not cache_path.is_file():
            return None
        expected_key = self._index_cache_key()
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_key_raw = cached["cache_key"]
                cache_key = str(
                    cache_key_raw.item()
                    if hasattr(cache_key_raw, "item")
                    else cache_key_raw
                )
                if cache_key != expected_key:
                    return None
                episode_indices = cached["episode_indices"].astype(np.int64)
                anchors = cached["anchors"].astype(np.int64)
                sampling_progress = cached["sampling_progress"].astype(np.float64)
        except Exception:
            return None
        if (
            episode_indices.shape != anchors.shape
            or sampling_progress.shape != anchors.shape
            or episode_indices.ndim != 1
            or not np.isfinite(sampling_progress).all()
            or np.any((sampling_progress < 0) | (sampling_progress > 1))
        ):
            return None
        self._sampling_progress = sampling_progress
        return [
            (int(episode_index), int(anchor))
            for episode_index, anchor in zip(episode_indices, anchors, strict=True)
        ]

    def _save_index_cache(
        self,
        index: list[tuple[int, int]],
        sampling_progress: np.ndarray,
    ) -> None:
        cache_path = self._index_cache_path()
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            episode_indices = np.asarray(
                [episode_index for episode_index, _ in index],
                dtype=np.int64,
            )
            anchors = np.asarray([anchor for _, anchor in index], dtype=np.int64)
            tmp_path = cache_path.with_name(
                f"{cache_path.stem}.tmp.{os.getpid()}{cache_path.suffix}"
            )
            np.savez_compressed(
                tmp_path,
                cache_key=np.asarray(self._index_cache_key()),
                episode_indices=episode_indices,
                anchors=anchors,
                sampling_progress=np.asarray(sampling_progress, dtype=np.float64),
            )
            tmp_path.replace(cache_path)
        except Exception:
            # Index caching is an optimization; dataset construction remains valid
            # if the shared metadata directory is read-only or concurrently written.
            return

    def _build_index(self) -> list[tuple[int, int]]:
        cached = self._load_index_cache()
        if cached is not None:
            return cached
        if not self.temporal_packing:
            result = [
                (episode_index, self.temporal_anchor_index)
                for episode_index in range(len(self.paths))
            ]
            self._sampling_progress = np.asarray(
                [
                    min(
                        1.0,
                        float(self.temporal_anchor_index)
                        / max(1, _effective_length(self._read_index_arrays(episode_index))),
                    )
                    for episode_index in range(len(self.paths))
                ],
                dtype=np.float64,
            )
            self._save_index_cache(result, self._sampling_progress)
            return result
        result: list[tuple[int, int]] = []
        progress: list[float] = []
        for episode_index in range(len(self.paths)):
            arrays = self._read_index_arrays(episode_index)
            anchors = _eligible_temporal_anchors(
                arrays,
                self.temporal_max_chunk_size,
                action_horizon=self.action_horizon,
                respect_trajectory_segments=self.respect_trajectory_segments,
                anchor_start=self.temporal_anchor_index,
                step_stride=self.step_stride,
            )
            if self.temporal_anchor_index in anchors:
                anchors = [self.temporal_anchor_index] + [
                    anchor for anchor in anchors if anchor != self.temporal_anchor_index
                ]
            result.extend(
                (episode_index, anchor)
                for anchor in anchors
            )
            exec_length = max(1, _effective_length(arrays))
            mask_value = _first(arrays, ("meta.is_exec", "is_exec"))
            mask = (
                None
                if mask_value is None
                else np.asarray(mask_value).astype(bool).reshape(-1)
            )
            labels, segments = _temporal_labels(arrays, exec_length, mask)
            packer = LanguageTemporalPacker(
                max_chunk_size=self.temporal_max_chunk_size,
                chunk_size=self.action_horizon,
            )
            for anchor in anchors:
                packed = packer.indices(
                    anchor,
                    labels,
                    trajectory_ids=(
                        segments if self.respect_trajectory_segments else None
                    ),
                )
                progress.append(float(packed.video[0]) / exec_length)
        if not result:
            raise ValueError("dataset has no eligible temporal packing anchors")
        self._sampling_progress = np.asarray(progress, dtype=np.float64)
        self._save_index_cache(result, self._sampling_progress)
        return result

    def length_signature(self, index: int) -> tuple[int, ...]:
        episode_index, anchor = self.index[index]
        arrays = self._read_index_arrays(episode_index)
        if self.temporal_packing:
            length = _effective_length(arrays)
            mask_value = _first(arrays, ("meta.is_exec", "is_exec"))
            mask = (
                None
                if mask_value is None
                else np.asarray(mask_value).astype(bool).reshape(-1)
            )
            labels, segments = _temporal_labels(arrays, length, mask)
            packed = LanguageTemporalPacker(
                max_chunk_size=self.temporal_max_chunk_size,
                chunk_size=self.action_horizon,
            ).indices(
                anchor,
                labels,
                trajectory_ids=(
                    segments if self.respect_trajectory_segments else None
                ),
            )
            return (len(packed.action), len(packed.state), len(packed.video))
        length = _selected_chunk_length(
            _effective_length(arrays),
            action_horizon=self.action_horizon,
            max_event_chunks=self.max_event_chunks,
            seed=self.seed + index,
        )
        return (length, length, length)

    def __getitem__(self, index: int) -> EventSample:
        episode_index, anchor = self.index[index]
        arrays = self._read_arrays(episode_index)
        return self._build_indexed_sample(index, arrays, episode_index, anchor)

    def get_from_shard(
        self,
        index: int,
        shard: dict[int, dict[str, np.ndarray]],
    ) -> EventSample:
        """Build one sample from a fully materialized shard."""

        episode_index, anchor = self.index[index]
        return self._build_indexed_sample(
            index, shard[episode_index], episode_index, anchor
        )

    def _build_indexed_sample(
        self,
        index: int,
        arrays: dict[str, np.ndarray],
        episode_index: int,
        anchor: int,
    ) -> EventSample:
        episode_id = str(self.episodes[episode_index].get("episode_index", episode_index))
        session_id = str(self.episodes[episode_index].get("session_id", episode_id))
        return _build_sample(
            arrays,
            episode_id=episode_id,
            session_id=session_id,
            visual_prompt=self.visual_prompt,
            embodiment=self.embodiment,
            action_horizon=self.action_horizon,
            max_event_chunks=self.max_event_chunks,
            history_sampler=self.history_sampler,
            history_window=self.history_window,
            seed=self.seed + index,
            visual_prompt_override=self.visual_prompt_override,
            temporal_packing=self.temporal_packing,
            temporal_anchor_index=anchor,
            max_chunk_size=self.temporal_max_chunk_size,
            respect_trajectory_segments=self.respect_trajectory_segments,
            context_sampling_mode=self.context_sampling_mode,
            context_video_len=self.context_video_len,
            ctx_head_only=self.ctx_head_only,
            wo_norm=self.wo_norm,
        )
