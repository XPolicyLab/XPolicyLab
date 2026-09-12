from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

from torch.utils.data import Dataset

from worldscape_policy.data.adapters.common import (
    AuditedVisualPromptOverride,
    EventSample,
    HistorySampler,
    LanguageTemporalPacker,
    Path,
    _build_sample,
    _effective_length,
    _eligible_temporal_anchors,
    _first,
    _metadata_task,
    _optional_dependency,
    _read_jsonl,
    _read_video,
    _resolve_episode_path,
    _selected_chunk_length,
    _task_records,
    _temporal_labels,
    json,
    np,
)
from worldscape_policy.data.adapters.robotwin import (
    apply_robotwin_labels,
    discover_lerobot_roots,
    load_episode_map,
)


class NativeLeRobotDataset(Dataset[EventSample]):
    """Read LeRobot v2 parquet episodes without importing the legacy pipeline."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        visual_prompt: str = "demo",
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
        allow_incompatible_visual_prompts: bool = False,
        visual_prompt_override_audit_reason: str | None = None,
        subtask_label_root: str | Path | None = None,
        subtask_episode_map_path: str | Path | None = None,
        strict_subtask_labels: bool = False,
        single_high_level_instruction_per_episode: bool = False,
        episode_cache_size: int = 0,
        cache_video_resize_width: int | None = None,
        cache_video_resize_height: int | None = None,
    ) -> None:
        if data_root is None or not str(data_root):
            raise ValueError("data_root is required")
        if episode_cache_size < 0:
            raise ValueError("episode_cache_size must be non-negative")
        if (cache_video_resize_width is None) ^ (
            cache_video_resize_height is None
        ):
            raise ValueError(
                "cache video resize width and height must be both set or both None"
            )
        if cache_video_resize_width is not None and cache_video_resize_width <= 0:
            raise ValueError("cache_video_resize_width must be positive")
        if cache_video_resize_height is not None and cache_video_resize_height <= 0:
            raise ValueError("cache_video_resize_height must be positive")
        self.roots = discover_lerobot_roots(data_root)
        self.infos: list[dict] = []
        self.tasks_by_episode: list[dict[int, str]] = []
        self.episode_roots: list[Path] = []
        self.episodes: list[dict] = []
        self.paths: list[Path] = []
        repository_infos: list[dict] = []
        repository_tasks: list[dict[int, str]] = []
        for root in self.roots:
            info = json.loads((root / "meta" / "info.json").read_text())
            tasks_path = root / "meta" / "tasks.jsonl"
            tasks = _task_records(tasks_path if tasks_path.is_file() else None)
            repository_infos.append(info)
            repository_tasks.append(tasks)
            episodes_path = root / "meta" / "episodes.jsonl"
            episodes = (
                _read_jsonl(episodes_path)
                if episodes_path.is_file()
                else [
                    {"episode_index": index}
                    for index in range(int(info.get("total_episodes", 0)))
                ]
            )
            for episode in episodes:
                self.infos.append(info)
                self.tasks_by_episode.append(tasks)
                self.episode_roots.append(root)
                self.episodes.append(episode)
                self.paths.append(_resolve_episode_path(episode, (root,), info))
        # Retain these public attributes for single-repository callers.
        self.info = repository_infos[0]
        self.tasks = repository_tasks[0]
        self.subtask_label_root = (
            None if subtask_label_root is None else Path(subtask_label_root)
        )
        self.subtask_episode_map = load_episode_map(subtask_episode_map_path)
        self.strict_subtask_labels = bool(strict_subtask_labels)
        self.single_high_level_instruction_per_episode = bool(
            single_high_level_instruction_per_episode
        )
        self.episode_cache_size = int(episode_cache_size)
        self.cache_video_resize_width = cache_video_resize_width
        self.cache_video_resize_height = cache_video_resize_height
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
        self.visual_prompt_override = (
            AuditedVisualPromptOverride(
                enabled=True,
                audit_reason=visual_prompt_override_audit_reason,
            )
            if allow_incompatible_visual_prompts
            else None
        )
        self._index_arrays: dict[int, dict[str, np.ndarray]] = {}
        self._episode_arrays_cache: OrderedDict[
            int, dict[str, np.ndarray]
        ] = OrderedDict()
        self.index = self._build_index()

    def __len__(self) -> int:
        return len(self.index)

    def sample_group_key(self, index: int) -> int:
        """Return the source episode so the batch sampler preserves locality."""

        episode_index, _ = self.index[index]
        return int(episode_index)

    def _read_tabular_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        pandas = _optional_dependency("pandas", "train")
        frame = pandas.read_parquet(self.paths[episode_index])
        arrays = {str(name): frame[name].to_numpy() for name in frame.columns}
        episode = self.episodes[episode_index]
        _metadata_task(arrays, episode, self.tasks_by_episode[episode_index])
        apply_robotwin_labels(
            arrays,
            dataset_root=self.episode_roots[episode_index],
            episode_index=int(episode.get("episode_index", episode_index)),
            trajectory_length=len(frame),
            subtask_label_root=self.subtask_label_root,
            episode_map=self.subtask_episode_map,
            strict=self.strict_subtask_labels,
        )
        self._apply_single_episode_instruction(
            arrays,
            episode=episode,
            trajectory_length=len(frame),
        )
        return arrays

    def _read_tabular_index_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._index_arrays.get(episode_index)
        if cached is not None:
            return cached
        pandas = _optional_dependency("pandas", "train")
        candidates = {
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
        try:
            parquet = _optional_dependency("pyarrow.parquet", "train")
            available = set(
                parquet.ParquetFile(self.paths[episode_index]).schema_arrow.names
            )
            columns = sorted(candidates & available)
            frame = pandas.read_parquet(self.paths[episode_index], columns=columns)
        except (ImportError, OSError, TypeError, ValueError):
            # Metadata-only fallback for alternate parquet engines. A test or
            # plugin dataframe reader that does not accept ``columns`` may
            # provide an already materialized frame.
            try:
                frame = pandas.read_parquet(self.paths[episode_index], columns=[])
            except TypeError:
                frame = pandas.read_parquet(self.paths[episode_index])
        arrays = {str(name): frame[name].to_numpy() for name in frame.columns if name in candidates}
        length = int(self.episodes[episode_index].get("length", len(frame)))
        arrays["action"] = np.empty((length, 0), dtype=np.float32)
        _metadata_task(
            arrays,
            self.episodes[episode_index],
            self.tasks_by_episode[episode_index],
            length=length,
        )
        apply_robotwin_labels(
            arrays,
            dataset_root=self.episode_roots[episode_index],
            episode_index=int(
                self.episodes[episode_index].get("episode_index", episode_index)
            ),
            trajectory_length=length,
            subtask_label_root=self.subtask_label_root,
            episode_map=self.subtask_episode_map,
            strict=self.strict_subtask_labels,
        )
        self._apply_single_episode_instruction(
            arrays,
            episode=self.episodes[episode_index],
            trajectory_length=length,
        )
        self._index_arrays[episode_index] = arrays
        return arrays

    def _apply_single_episode_instruction(
        self,
        arrays: dict[str, np.ndarray],
        *,
        episode: dict[str, Any],
        trajectory_length: int,
    ) -> None:
        if not self.single_high_level_instruction_per_episode:
            return
        candidates = [
            str(value).strip()
            for value in episode.get("tasks", ())
            if str(value).strip()
        ]
        if not candidates:
            raw = arrays.get("high_level_instruction")
            candidates = [
                str(value).strip()
                for value in np.asarray(raw if raw is not None else ()).reshape(-1)
                if str(value).strip()
            ]
        if not candidates:
            raise ValueError(
                "single high-level instruction mode requires episode task text"
            )
        instruction = candidates[0]
        values = np.full(trajectory_length, instruction, dtype=object)
        arrays["high_level_instruction"] = values.copy()
        arrays["event_instruction"] = values.copy()
        arrays["_task_index_is_real"] = np.asarray(False, dtype=np.bool_)
        arrays["_task_segment_id"] = np.zeros(trajectory_length, dtype=np.int64)

    def _build_index(self) -> list[tuple[int, int]]:
        if not self.temporal_packing:
            return [(episode_index, self.temporal_anchor_index) for episode_index in range(len(self.paths))]
        if self.single_high_level_instruction_per_episode:
            return [
                (episode_index, anchor)
                for episode_index, episode in enumerate(self.episodes)
                for anchor in range(
                    max(
                        0,
                        int(episode.get("length", 0)) - self.action_horizon,
                    )
                )
            ]
        result: list[tuple[int, int]] = []
        for episode_index in range(len(self.paths)):
            arrays = self._read_tabular_index_arrays(episode_index)
            anchors = _eligible_temporal_anchors(
                arrays,
                self.temporal_max_chunk_size,
                action_horizon=self.action_horizon,
                respect_trajectory_segments=self.respect_trajectory_segments,
            )
            if self.temporal_anchor_index in anchors:
                anchors = [self.temporal_anchor_index] + [
                    anchor for anchor in anchors if anchor != self.temporal_anchor_index
                ]
            result.extend(
                (episode_index, anchor)
                for anchor in anchors
            )
        if not result:
            raise ValueError("dataset has no eligible temporal packing anchors")
        return result

    def length_signature(self, index: int) -> tuple[int, ...]:
        episode_index, anchor = self.index[index]
        arrays = self._read_tabular_index_arrays(episode_index)
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

    def _resize_cached_video(self, frames: np.ndarray) -> np.ndarray:
        if (
            self.cache_video_resize_width is None
            or self.cache_video_resize_height is None
            or frames.ndim != 4
        ):
            return frames
        width = int(self.cache_video_resize_width)
        height = int(self.cache_video_resize_height)
        if frames.shape[1:3] == (height, width):
            return frames
        cv2 = _optional_dependency("cv2", "train")
        resized = [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
        return np.stack(resized, axis=0).astype(np.uint8, copy=False)

    def _load_episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        arrays = self._read_tabular_arrays(episode_index)
        episode = self.episodes[episode_index]
        info = self.infos[episode_index]
        episode_root = self.episode_roots[episode_index]
        for name, feature in info.get("features", {}).items():
            if name in arrays or feature.get("dtype") not in {"video", "image"}:
                continue
            pattern = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4",
            )
            video_episode_index = int(episode.get("episode_index", episode_index))
            relative = str(pattern).format(
                episode_index=video_episode_index,
                episode_chunk=int(
                    episode.get("episode_chunk", video_episode_index // 1000)
                ),
                video_key=name,
            )
            video_path = episode_root / relative
            if not video_path.is_file():
                raise FileNotFoundError(f"LeRobot video does not exist: {relative}")
            arrays[str(name)] = self._resize_cached_video(_read_video(video_path))
        return arrays

    def _episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        if self.episode_cache_size == 0:
            return self._load_episode_arrays(episode_index)
        cached = self._episode_arrays_cache.pop(episode_index, None)
        if cached is not None:
            self._episode_arrays_cache[episode_index] = cached
            return cached
        arrays = self._load_episode_arrays(episode_index)
        self._episode_arrays_cache[episode_index] = arrays
        while len(self._episode_arrays_cache) > self.episode_cache_size:
            self._episode_arrays_cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> EventSample:
        episode_index, anchor = self.index[index]
        arrays = self._episode_arrays(episode_index)
        episode = self.episodes[episode_index]
        episode_root = self.episode_roots[episode_index]
        local_episode_id = str(episode.get("episode_index", episode_index))
        episode_id = (
            local_episode_id
            if len(self.roots) == 1
            else f"{episode_root.name}:{local_episode_id}"
        )
        session_id = str(self.episodes[episode_index].get("session_id", episode_id))
        sample = _build_sample(
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
            use_native_history_sampler=True,
        )
        return replace(
            sample,
            provenance={
                **sample.provenance,
                "dataset_index": int(index),
                "dataset_episode_index": int(episode_index),
                "temporal_anchor_index": int(anchor),
            },
        )


__all__ = ["NativeLeRobotDataset"]
