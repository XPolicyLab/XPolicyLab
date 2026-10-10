"""LeRobot v2.0/v2.1 map-style data provider.

Reads native episode parquet files, task metadata, and per-camera videos.
Observation and action windows are decoded for the requested anchors."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F

from lerobot.datasets.video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
)

from ._utils import (
    get_logger,
    supervised_anchor_lengths,
)
from .processors.base_processor import BaseProcessor
from .base_lerobot_dataset import sliding_window_with_replication

logger = get_logger(__name__)

_TOLERANCE_S = 1.0


def _lerobot_key(meta: Dict[str, Any], prefix: str, default: str) -> str:
    """Map a shape_meta entry to its v2 parquet/video column name.

    An explicit ``lerobot_key`` in the entry is used verbatim (for datasets
    whose columns don't follow the standard naming, e.g. RH20T's
    ``observation.action``); otherwise ``key: default`` → the bare group column
    (``observation.state`` / ``action`` / ``observation.images``), an
    already-dotted key is used verbatim, and any other key is prefixed
    (e.g. ``cam_head_rgb`` → ``observation.images.cam_head_rgb``).
    """
    override = meta.get("lerobot_key")
    if override:
        return override
    key = meta["key"]
    if key == "default":
        return default
    if key.startswith(prefix):
        return key
    return f"{prefix}{key}"


class ShardedLerobotV2Dataset:
    """Shard provider for a single LeRobot v2.x dataset directory."""

    def __init__(
        self,
        dataset_dir: str,
        shape_meta: Dict[str, Any],
        num_frames: int,
        # v2-specific
        decode_resize: Optional[List[int]] = None,
        # shared with ShardedLerobotV3Dataset
        global_sample_stride: int = 1,
        val_set_proportion: float = 0.05,
        is_training_set: bool = True,
        seed: int = 42,
        decode_workers: int = 1,
        skip_corrupt_data: bool = False,
        dataset_name: Optional[str] = None,
        minimum_future_offset: int = 1,
        main_image_key: Optional[str] = None,
        main_image_size: Optional[List[int]] = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.action_size = num_frames - 1
        self.decode_workers = int(decode_workers)
        self.minimum_future_offset = int(minimum_future_offset)
        self._dataset_name = dataset_name or Path(dataset_dir).name
        # decode-time working resolution [H, W]; None keeps native frame size.
        self.decode_resize = list(decode_resize) if decode_resize is not None else None
        # When True, a per-episode decode/stats failure is logged and the episode
        # dropped instead of raising. Propagated from the owning group / mixture
        # (default False = fail-fast).
        self.skip_corrupt = bool(skip_corrupt_data)
        self.processor: Optional[BaseProcessor] = None
        self.main_image_key = main_image_key
        self.main_image_size = list(main_image_size or [224, 224])

        self._root = Path(dataset_dir)

        # Parse shape_meta → per-group meta with resolved v2 column/video keys.
        self.image_meta = []
        for m in shape_meta["images"]:
            m = dict(m)
            m["lerobot_key"] = _lerobot_key(m, "observation.images.", "observation.images")
            self.image_meta.append(m)
        self.state_meta = []
        for m in shape_meta["state"]:
            m = dict(m)
            m["lerobot_key"] = _lerobot_key(m, "observation.state.", "observation.state")
            self.state_meta.append(m)
        self.action_meta = []
        for m in shape_meta["action"]:
            m = dict(m)
            m["lerobot_key"] = _lerobot_key(m, "action.", "action")
            self.action_meta.append(m)

        # --- metadata: info.json + episodes.jsonl --- #
        info = json.loads((self._root / "meta" / "info.json").read_text())
        self.fps = int(info["fps"])
        self._chunks_size = int(info["chunks_size"])
        self._data_path_tmpl = info["data_path"]
        self._video_path_tmpl = info["video_path"]
        features = info.get("features") or {}
        self._has_frame_task_index = "task_index" in features
        self._task_text_by_index = (
            self._read_task_texts() if self._has_frame_task_index else {}
        )

        # Some LeRobot v2 datasets contain children whose flat proprio vectors
        # have the same named coordinates in different orders.  Reorder only
        # fields that declare an audited canonical name contract in shape_meta;
        # all other providers retain their original behavior.  This lives at
        # the parquet boundary so training decode and normalization-stat
        # construction consume exactly the same semantic layout.
        self._canonical_permutations: Dict[str, torch.Tensor] = {}
        vector_contracts: Dict[str, tuple[str, ...]] = {}
        features = info.get("features")
        if not isinstance(features, dict):
            features = {}
        for meta in self.state_meta + self.action_meta:
            canonical_names = tuple(meta.get("canonical_names", ()))
            lerobot_key = meta["lerobot_key"]
            previous_contract = vector_contracts.get(lerobot_key)
            if previous_contract is not None and previous_contract != canonical_names:
                raise ValueError(
                    f"Conflicting canonical name contracts for {lerobot_key!r} "
                    f"under {self._root}"
                )
            vector_contracts[lerobot_key] = canonical_names
            if not canonical_names:
                continue
            if meta.get("select_indices"):
                raise ValueError(
                    f"canonical_names and select_indices cannot both configure "
                    f"{lerobot_key!r} under {self._root}"
                )
            if any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in canonical_names
            ) or len(set(canonical_names)) != len(canonical_names):
                raise ValueError(
                    f"canonical_names for {lerobot_key!r} must be unique, "
                    f"non-empty exact strings"
                )
            raw_shape = meta.get("raw_shape")
            shape = meta.get("shape")
            if raw_shape != len(canonical_names) or shape != len(canonical_names):
                raise ValueError(
                    f"canonical_names width for {lerobot_key!r} under {self._root} "
                    f"must equal raw_shape and shape: {len(canonical_names)} != "
                    f"{raw_shape!r}, {shape!r}"
                )
            feature = features.get(lerobot_key)
            if not isinstance(feature, dict):
                raise ValueError(
                    f"meta/info.json under {self._root} has no feature metadata "
                    f"for canonical field {lerobot_key!r}"
                )
            source_names = feature.get("names")
            if not isinstance(source_names, list) or any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in source_names
            ) or len(set(source_names)) != len(source_names):
                raise ValueError(
                    f"meta/info.json names for {lerobot_key!r} under {self._root} "
                    f"must be unique, non-empty exact strings"
                )
            feature_shape = feature.get("shape")
            if (
                not isinstance(feature_shape, list)
                or len(feature_shape) != 1
                or feature_shape[0] != len(source_names)
                or len(source_names) != len(canonical_names)
                or set(source_names) != set(canonical_names)
            ):
                raise ValueError(
                    f"meta/info.json layout for {lerobot_key!r} under {self._root} "
                    f"does not match its canonical name contract"
                )
            index_by_name = {name: index for index, name in enumerate(source_names)}
            permutation = torch.tensor(
                [index_by_name[name] for name in canonical_names], dtype=torch.long
            )
            previous_permutation = self._canonical_permutations.get(lerobot_key)
            if previous_permutation is not None and not torch.equal(
                previous_permutation, permutation
            ):
                raise ValueError(
                    f"Conflicting canonical permutations for {lerobot_key!r} "
                    f"under {self._root}"
                )
            self._canonical_permutations[lerobot_key] = permutation

        all_records = self._build_records(self._read_episodes_jsonl())
        self._records = self._split_episodes(
            all_records, val_set_proportion, is_training_set, seed
        )
        self._records = self._reattach_paths(self._records)

        self.episode_lengths = [r["ep_len"] for r in self._records]
        dense_anchor_lengths = supervised_anchor_lengths(
            self.episode_lengths, self.minimum_future_offset
        )
        self.anchor_lengths = list(map(int, dense_anchor_lengths))
        self.num_loaded_episodes = len(self._records)
        assert self.num_loaded_episodes > 0, f"No usable episodes found under {self._root}"

        # Integer frame offsets for sampling windows.
        self.obs_offsets = np.array(
            [t * global_sample_stride for t in range(num_frames)], dtype=np.int64
        )
        self.action_offsets = np.array(
            [t * global_sample_stride for t in range(num_frames - 1)], dtype=np.int64
        )


    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _read_episodes_jsonl(self) -> List[Dict[str, Any]]:
        path = self._root / "meta" / "episodes.jsonl"
        episodes: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes

    def _read_task_texts(self) -> Dict[int, str]:
        """Read the standard LeRobot-v2 frame task lookup table."""

        path = self._root / "meta" / "tasks.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"LeRobot v2 dataset {self._root} declares task_index but is "
                f"missing {path}"
            )
        texts: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "task_index" not in row or "task" not in row:
                    raise ValueError(
                        f"Invalid {path}:{line_number}: expected task_index and task"
                    )
                task_index = int(row["task_index"])
                task = str(row["task"]).strip()
                if task_index < 0 or not task:
                    raise ValueError(
                        f"Invalid {path}:{line_number}: task_index must be "
                        "non-negative and task must be non-empty"
                    )
                previous = texts.get(task_index)
                if previous is not None and previous != task:
                    raise ValueError(
                        f"Conflicting task text for index {task_index} in {path}: "
                        f"{previous!r} != {task!r}"
                    )
                texts[task_index] = task
        if not texts:
            raise ValueError(f"LeRobot task table is empty: {path}")
        return texts

    def _episode_paths(self, ep_idx: int) -> Dict[str, Any]:
        chunk = ep_idx // self._chunks_size
        parquet_path = self._root / self._data_path_tmpl.format(
            episode_chunk=chunk, episode_index=ep_idx
        )
        video_paths = {
            m["key"]: self._root / self._video_path_tmpl.format(
                episode_chunk=chunk, video_key=m["lerobot_key"], episode_index=ep_idx
            )
            for m in self.image_meta
        }
        return {"parquet_path": parquet_path, "video_paths": video_paths}

    def _build_records(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enumerate-ordered records: identity + length + task (paths attached
        later by ``_reattach_paths``)."""
        records: List[Dict[str, Any]] = []
        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            tasks = ep.get("tasks") or [""]
            records.append(
                {
                    "ep_idx": ep_idx,
                    "ep_len": int(ep["length"]),
                    "task": tasks[0],
                }
            )
        logger.info("Enumerated %d LeRobot v2 episodes under %s", len(records), self._root)
        return records

    # ------------------------------------------------------------------ #
    # Manifest cache
    # ------------------------------------------------------------------ #

    def _reattach_paths(self, records):
        """Attach parquet_path / video_paths from the ep_idx identity."""
        for rec in records:
            paths = self._episode_paths(rec["ep_idx"])
            rec["parquet_path"] = paths["parquet_path"]
            rec["video_paths"] = paths["video_paths"]
        return records


    @staticmethod
    def _split_episodes(records, val_set_proportion, is_training_set, seed):
        total = len(records)
        if val_set_proportion < 1e-6:
            return records
        split_idx = int(total * (1 - val_set_proportion))
        indices = list(range(total))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        keep = indices[:split_idx] if is_training_set else indices[split_idx:]
        return [records[i] for i in sorted(keep)]

    # ------------------------------------------------------------------ #
    # Anchor range decoding
    # ------------------------------------------------------------------ #


    @staticmethod
    def _chunk_descriptor(spec: Any, ep_len: int) -> tuple[int, int, int]:
        if isinstance(spec, (int, np.integer)):
            return int(spec), 0, int(ep_len)
        pos, anchor_start, anchor_stop = spec
        return int(pos), int(anchor_start), int(anchor_stop)


    # ------------------------------------------------------------------ #
    # Direct parquet reading (one episode per parquet in v2)
    # ------------------------------------------------------------------ #
    def _read_proprio(
        self,
        rec,
        frame_start: int = 0,
        frame_stop: Optional[int] = None,
    ) -> tuple:
        """Read one frame range of state/action columns from parquet."""
        cols = [m["lerobot_key"] for m in self.state_meta]
        cols += [m["lerobot_key"] for m in self.action_meta]
        table = pq.read_table(str(rec["parquet_path"]), columns=list(dict.fromkeys(cols)))
        frame_stop = len(table) if frame_stop is None else int(frame_stop)
        if not 0 <= frame_start < frame_stop <= len(table):
            raise ValueError(
                f"Invalid frame range [{frame_start}, {frame_stop}) for "
                f"{rec['parquet_path']} with {len(table)} rows"
            )
        table = table.slice(frame_start, frame_stop - frame_start)

        def _col(meta: Dict[str, Any]) -> torch.Tensor:
            lerobot_key = meta["lerobot_key"]
            t = torch.tensor(table.column(lerobot_key).to_pylist(), dtype=torch.float32)
            if t.ndim == 1:  # scalar-per-frame column (e.g. a lone gripper)
                t = t.unsqueeze(-1)
            permutation = self._canonical_permutations.get(lerobot_key)
            if permutation is not None:
                if t.ndim != 2 or t.shape[-1] != len(permutation):
                    raise ValueError(
                        f"Parquet column {lerobot_key!r} under {self._root} has "
                        f"shape {tuple(t.shape)}, expected [T,{len(permutation)}]"
                    )
                t = t.index_select(-1, permutation)
            select_indices = meta.get("select_indices")
            if select_indices:
                t = t.index_select(
                    -1, torch.as_tensor(select_indices, dtype=torch.long)
                )
            return t

        state = {m["key"]: _col(m) for m in self.state_meta}
        action = {m["key"]: _col(m) for m in self.action_meta}
        return state, action

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    def _decode_episode(self, pos: int, decoder_cache) -> Dict[str, Any]:
        return self._decode_chunk(pos, decoder_cache)

    def _decode_chunk(self, spec: Any, decoder_cache) -> Dict[str, Any]:
        pos = int(spec if isinstance(spec, (int, np.integer)) else spec[0])
        rec = self._records[pos]
        ep_len = int(rec["ep_len"])
        anchor_len = int(self.anchor_lengths[pos])
        pos, anchor_start, anchor_stop = self._chunk_descriptor(spec, anchor_len)
        if not 0 <= anchor_start < anchor_stop <= anchor_len:
            raise ValueError(
                f"Invalid anchor chunk {(pos, anchor_start, anchor_stop)} "
                f"for anchor_len={anchor_len}, ep_len={ep_len}"
            )
        all_offsets = np.concatenate((self.obs_offsets, self.action_offsets))
        anchor_origin = int(rec.get("window_start", 0))
        local_anchor_indices = np.arange(
            anchor_start, anchor_stop, dtype=np.int64
        )
        actual_anchor_indices = local_anchor_indices + anchor_origin
        actual_anchor_start = int(actual_anchor_indices[0])
        actual_anchor_stop = int(actual_anchor_indices[-1]) + 1
        image_frame_indices = np.unique(
            np.clip(
                actual_anchor_indices[:, None] + self.obs_offsets[None, :],
                0,
                ep_len - 1,
            ).reshape(-1)
        )
        frame_start = max(0, actual_anchor_start + int(all_offsets.min(initial=0)))
        frame_stop = min(
            ep_len,
            actual_anchor_stop + int(all_offsets.max(initial=0)),
        )
        state, action = self._read_proprio(rec, frame_start, frame_stop)

        images: Dict[str, torch.Tensor] = {}
        main_images: Optional[torch.Tensor] = None
        for m in self.image_meta:
            key = m["key"]
            frames = self._decode_frames(
                str(rec["video_paths"][key]),
                image_frame_indices,
                decoder_cache,
            )
            if self.main_image_key == key:
                main_images = (transforms_F.resize(
                    frames,
                    size=self.main_image_size,
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                ).clamp(0, 1).mul(255).round().to(torch.uint8).contiguous())
            if self.decode_resize is not None:
                frames = transforms_F.resize(
                    frames,
                    size=self.decode_resize,
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T, C, H, W] float [0,1]
            images[key] = (frames * 255).to(torch.uint8)

        payload = {
            "ep_idx": rec["ep_idx"],
            "ep_len": ep_len,
            "num_anchors": anchor_stop - anchor_start,
            "anchor_start": actual_anchor_start,
            "anchor_indices": actual_anchor_indices,
            "frame_start": frame_start,
            "image_frame_indices": image_frame_indices,
            "task": rec["task"],
            "state": state,
            "action": action,
            "images": images,
            "main_images": main_images,
        }
        payload.update(
            self._load_standard_task_metadata(rec, frame_start, frame_stop)
        )
        payload.update(
            self._load_chunk_metadata(rec, frame_start, frame_stop)
        )
        return payload

    def _load_standard_task_metadata(
        self,
        rec: Dict[str, Any],
        frame_start: int,
        frame_stop: int,
    ) -> Dict[str, Any]:
        """Load frame-aligned LeRobot ``task_index`` values."""

        if not self._has_frame_task_index:
            return {}
        table = pq.read_table(str(rec["parquet_path"]), columns=["task_index"])
        if not 0 <= frame_start < frame_stop <= len(table):
            raise ValueError(
                f"Invalid task_index range [{frame_start}, {frame_stop}) for "
                f"{rec['parquet_path']} with {len(table)} rows"
            )
        values = table.slice(frame_start, frame_stop - frame_start).column(
            "task_index"
        ).to_pylist()
        indices = []
        for value in values:
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(
                        f"task_index in {rec['parquet_path']} must be scalar or "
                        f"width one, got {value!r}"
                    )
                value = value[0]
            indices.append(-1 if value is None else int(value))
        return {"lerobot_task_indices": torch.tensor(indices, dtype=torch.long)}

    def _load_chunk_metadata(
        self,
        rec: Dict[str, Any],
        frame_start: int,
        frame_stop: int,
    ) -> Dict[str, Any]:
        """Extension hook for dataset-specific per-frame annotations."""
        return {}

    @staticmethod
    def _decode_frames(
        video_path: str,
        frame_indices: np.ndarray,
        decoder_cache,
    ) -> torch.Tensor:
        """Decode only the requested absolute frames from an episode mp4."""
        decoder = decoder_cache.get_decoder(video_path)
        meta = decoder.metadata
        fps = meta.average_fps
        max_ts = (meta.num_frames - 1) / fps
        decode_ts = [min(int(i) / fps, max_ts) for i in frame_indices]
        try:
            return decode_video_frames_torchcodec(
                video_path, decode_ts, _TOLERANCE_S, decoder_cache=decoder_cache
            )
        except Exception as exc:
            logger.warning(
                "Batched decode of %s failed (%s); retrying per-frame.", video_path, exc
            )
            batched_error = exc
        retry_cache = VideoDecoderCache()
        try:
            frames: List[torch.Tensor] = []
            for frame_index, ts in zip(frame_indices, decode_ts):
                try:
                    frame = decode_video_frames_torchcodec(
                        video_path, [ts], _TOLERANCE_S, decoder_cache=retry_cache
                    )[0]
                except Exception as frame_exc:
                    raise RuntimeError(
                        f"Frame {int(frame_index)} at timestamp {ts:.9f}s failed "
                        f"to decode for "
                        f"{video_path}; batched decode also failed: {batched_error}"
                    ) from frame_exc
                frames.append(frame)
        finally:
            retry_cache.clear()
        return torch.stack(frames)




    # ------------------------------------------------------------------ #
    # In-RAM sample reconstruction
    # ------------------------------------------------------------------ #
    def get_sample(self, ep_payload: Dict[str, Any], anchor: int) -> Dict[str, Any]:
        ep_len = ep_payload["ep_len"]
        anchor_indices = ep_payload.get("anchor_indices")
        global_anchor = (
            int(anchor_indices[int(anchor)])
            if anchor_indices is not None
            else int(ep_payload.get("anchor_start", 0)) + int(anchor)
        )
        obs_q = global_anchor + self.obs_offsets
        act_q = global_anchor + self.action_offsets
        obs_idx = np.clip(obs_q, 0, ep_len - 1)
        act_idx = np.clip(act_q, 0, ep_len - 1)
        frame_start = int(ep_payload.get("frame_start", 0))
        obs_local_idx = obs_idx - frame_start
        act_local_idx = act_idx - frame_start
        image_frame_indices = ep_payload.get("image_frame_indices")
        image_local_idx = np.searchsorted(image_frame_indices, obs_idx)
        if (
            np.any(image_local_idx >= len(image_frame_indices))
            or not np.array_equal(image_frame_indices[image_local_idx], obs_idx)
        ):
            raise RuntimeError("Decoded image-frame index is missing from shard payload")
        obs_pad = torch.from_numpy(obs_q >= ep_len)
        act_pad = torch.from_numpy(act_q >= ep_len)

        sample: Dict[str, Any] = {
            "idx": ep_payload["ep_idx"] * 1_000_000 + global_anchor,
            "task": ep_payload["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        sample.update(
            self._standard_task_annotations(
                ep_payload, global_anchor, frame_start
            )
        )
        sample.update(
            self._sample_annotations(ep_payload, global_anchor, frame_start)
        )
        for meta in self.state_meta:
            win = ep_payload["state"][meta["key"]][obs_local_idx]
            if win.ndim == 1:
                win = win.unsqueeze(-1)
            sample["state"][meta["key"]] = win
        for meta in self.action_meta:
            win = ep_payload["action"][meta["key"]][act_local_idx]
            if win.ndim == 1:
                win = win.unsqueeze(-1)
            sample["action"][meta["key"]] = win
        for meta in self.image_meta:
            sample["images"][meta["key"]] = ep_payload["images"][meta["key"]][
                image_local_idx
            ]

        sample["action_is_pad"] = act_pad
        sample["state_is_pad"] = obs_pad
        sample["image_is_pad"] = obs_pad

        processed = self.processor.preprocess(sample)
        if ep_payload.get("main_images") is not None:
            processed["main_image"] = (
                ep_payload["main_images"][image_local_idx[0]].float().div(255)
            )
        processed["metadata"] = {
            "provider_dataset_dir": str(self.dataset_dir),
            "episode_id": int(ep_payload["ep_idx"]),
            "anchor_index": global_anchor,
        }
        return processed

    def _standard_task_annotations(
        self,
        ep_payload: Dict[str, Any],
        global_anchor: int,
        frame_start: int,
    ) -> Dict[str, Any]:
        """Resolve this anchor's task from parquet through ``tasks.jsonl``."""

        values = ep_payload.get("lerobot_task_indices")
        local_anchor = int(global_anchor) - int(frame_start)
        if values is None or not 0 <= local_anchor < len(values):
            return {}
        task_index = int(values[local_anchor].item())
        if task_index < 0:
            raise ValueError(
                f"Missing task_index for episode {ep_payload['ep_idx']} "
                f"frame {global_anchor} under {self._root}"
            )
        task = self._task_text_by_index.get(task_index)
        if task is None:
            raise KeyError(
                f"task_index={task_index} for episode {ep_payload['ep_idx']} "
                f"frame {global_anchor} is absent from "
                f"{self._root / 'meta' / 'tasks.jsonl'}"
            )
        return {"task": task, "task_index": task_index}

    def _sample_annotations(
        self,
        ep_payload: Dict[str, Any],
        global_anchor: int,
        frame_start: int,
    ) -> Dict[str, Any]:
        """Extension hook for dataset-specific anchor annotations."""
        return {}

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        return self

    def __len__(self) -> int:
        return sum(self.anchor_lengths)

    # ------------------------------------------------------------------ #
    # Normalization stats (reads parquet only; numerically matches
    # BaseLerobotDataset.get_dataset_stats without the LeRobot IO).
    # Copied from ShardedAgiBotBetaDataset (keep in sync).
    # ------------------------------------------------------------------ #
    def _accumulate_stats(self, preprocessor: BaseProcessor, skip_corrupt_data: bool = False) -> Dict[str, Any]:
        acc = {
            group: {stat: defaultdict(list) for stat in ("min", "max", "mean", "var", "q01", "q99")}
            for group in ("state", "action")
        }

        def process_episode(pos):
            record = self._records[pos]
            state, action = self._read_proprio(record)
            batch = {"state": {}, "action": {}}
            for meta in self.state_meta:
                batch["state"][meta["key"]] = (
                    state[meta["key"]].unsqueeze(1).float()
                )
            for meta in self.action_meta:
                batch["action"][meta["key"]] = sliding_window_with_replication(
                    action[meta["key"]].float(), self.action_size
                )
            return preprocessor.action_state_transform(batch)

        num_episodes = self.num_loaded_episodes
        max_workers = max(1, self.decode_workers) * 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            from tqdm import tqdm

            positions = iter(range(num_episodes))
            pending = {
                executor.submit(process_episode, pos)
                for pos in islice(positions, max_workers * 2)
            }
            progress = tqdm(total=num_episodes, desc="LeRobot v2 stats")
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    progress.update()
                    try:
                        pending.add(executor.submit(process_episode, next(positions)))
                    except StopIteration:
                        pass
                    try:
                        batch = future.result()
                        # Reduce inside the try so a degenerate batch (e.g. a
                        # zero-length episode -> amin over an empty dim) is skipped
                        # too, not just a failed future. Stage per-episode results
                        # and commit atomically so a mid-reduction failure never
                        # leaves some keys with an extra entry (keeps every per-key
                        # list length equal to the accepted-episode count).
                        staged = []
                        for group, metas in (
                            ("state", preprocessor.shape_meta["state"]),
                            ("action", preprocessor.shape_meta["action"]),
                        ):
                            for meta in metas:
                                key = meta["key"]
                                cur = batch[group][key]
                                staged.append((
                                    group, key,
                                    cur.amin(0), cur.amax(0), cur.mean(0),
                                    cur.var(0, unbiased=False),
                                    torch.quantile(cur, 0.01, dim=0, keepdim=False),
                                    torch.quantile(cur, 0.99, dim=0, keepdim=False),
                                ))
                    except Exception as exc:
                        if not skip_corrupt_data:
                            raise
                        logger.warning(
                            "skip_corrupt_data: dropping one episode from stats in %s: %s",
                            self.dataset_dir, exc,
                        )
                        continue
                    for group, key, mn, mx, mean, var, q01, q99 in staged:
                        acc[group]["min"][key].append(mn)
                        acc[group]["max"][key].append(mx)
                        acc[group]["mean"][key].append(mean)
                        acc[group]["var"][key].append(var)
                        acc[group]["q01"][key].append(q01)
                        acc[group]["q99"][key].append(q99)
            progress.close()
        return acc

    @staticmethod
    def _reduce_stats(acc, metas_by_group, num_episodes: int, num_transition: int) -> Dict[str, Any]:
        def get_mean_std(means, vars_):
            means = torch.stack(means)
            vars_ = torch.stack(vars_)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars_ + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars_ + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {
            "state": defaultdict(dict),
            "action": defaultdict(dict),
            "num_episodes": num_episodes,
            "num_transition": num_transition,
        }
        for group, metas in metas_by_group.items():
            for meta in metas:
                key = meta["key"]
                if not acc[group]["min"][key]:
                    raise RuntimeError(
                        f"No usable episodes contributed to '{group}/{key}' normalization "
                        f"stats (every episode was skipped as corrupt). Refusing to emit "
                        f"empty stats; fix the data or disable skip_corrupt_data."
                    )
                stats[group][key]["stepwise_min"] = torch.stack(acc[group]["min"][key]).amin(0)
                stats[group][key]["stepwise_max"] = torch.stack(acc[group]["max"][key]).amax(0)
                stats[group][key]["global_min"] = stats[group][key]["stepwise_min"].amin(0)
                stats[group][key]["global_max"] = stats[group][key]["stepwise_max"].amax(0)
                stats[group][key]["stepwise_q01"] = torch.stack(acc[group]["q01"][key]).amin(0)
                stats[group][key]["stepwise_q99"] = torch.stack(acc[group]["q99"][key]).amax(0)
                stats[group][key]["global_q01"] = stats[group][key]["stepwise_q01"].amin(0)
                stats[group][key]["global_q99"] = stats[group][key]["stepwise_q99"].amax(0)
                (
                    stats[group][key]["stepwise_mean"],
                    stats[group][key]["stepwise_std"],
                    stats[group][key]["global_mean"],
                    stats[group][key]["global_std"],
                ) = get_mean_std(acc[group]["mean"][key], acc[group]["var"][key])
        return stats

    def get_dataset_stats(self, preprocessor: BaseProcessor, skip_corrupt_data: bool = False) -> Dict[str, Any]:
        acc = self._accumulate_stats(preprocessor, skip_corrupt_data=skip_corrupt_data)
        return self._reduce_stats(
            acc,
            {
                "state": preprocessor.shape_meta["state"],
                "action": preprocessor.shape_meta["action"],
            },
            self.num_loaded_episodes,
            int(sum(self.episode_lengths)),
        )

    @staticmethod
    def pooled_dataset_stats(providers, preprocessor: BaseProcessor, skip_corrupt_data: bool = False) -> Dict[str, Any]:
        """Stats over the union of several providers' episodes.

        Pools each provider's per-episode accumulators before a single
        reduction, matching a single multi-dir ``BaseLerobotDataset`` pass.
        """
        assert providers, "pooled_dataset_stats needs at least one provider."
        merged = {
            group: {stat: defaultdict(list) for stat in ("min", "max", "mean", "var", "q01", "q99")}
            for group in ("state", "action")
        }
        num_episodes = 0
        num_transition = 0
        for provider in providers:
            acc = provider._accumulate_stats(preprocessor, skip_corrupt_data=skip_corrupt_data)
            for group in ("state", "action"):
                for stat in ("min", "max", "mean", "var", "q01", "q99"):
                    for key, vals in acc[group][stat].items():
                        merged[group][stat][key].extend(vals)
            num_episodes += provider.num_loaded_episodes
            num_transition += int(sum(provider.episode_lengths))
        return ShardedLerobotV2Dataset._reduce_stats(
            merged,
            {
                "state": preprocessor.shape_meta["state"],
                "action": preprocessor.shape_meta["action"],
            },
            num_episodes,
            num_transition,
        )
