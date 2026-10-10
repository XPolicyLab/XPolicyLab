"""LeRobot v3.0 map-style data provider.

Reads episode parquet data on demand. Packed video timestamps include each
episode's from_timestamp. Padded observation/action windows carry validity masks."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _resolve_video_reference(video_path: str) -> str:
    """Resolve MP4 reference files.

    A reference is a small UTF-8 file containing the absolute path of an MP4.
    Ordinary MP4 files pass through untouched.
    """

    path = Path(video_path)
    try:
        if path.stat().st_size >= 4096:
            return video_path
        reference = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return video_path
    target = Path(reference)
    if not target.is_absolute() or target.suffix.lower() != ".mp4":
        return video_path
    try:
        resolved_target = target.resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        return video_path
    if not resolved_target.is_file():
        return video_path
    return str(resolved_target)


class ShardedLerobotV3Dataset:
    """Shard provider for a single lerobot-v3.0 dataset directory.

    Reads episodes metadata at init (fast), then reads parquet data per-shard
    on demand (no full dataset preload).
    """

    def __init__(
        self,
        dataset_dir: str,
        shape_meta: Dict[str, Any],
        num_frames: int,
        global_sample_stride: int = 1,
        val_set_proportion: float = 0.05,
        is_training_set: bool = True,
        seed: int = 42,
        decode_workers: int = 1,
        decode_resize: Optional[List[int]] = None,
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
        self.decode_resize = (
            None if decode_resize is None else [int(v) for v in decode_resize]
        )
        if self.decode_resize is not None and (
            len(self.decode_resize) != 2 or any(v <= 0 for v in self.decode_resize)
        ):
            raise ValueError("`decode_resize` must contain positive [height, width]")
        self._dataset_name = dataset_name or Path(dataset_dir).name
        # When True, a per-episode decode failure in _get_shard_payload is
        # logged and the episode dropped instead of raising. Propagated from the
        # owning group / mixture (default False = fail-fast).
        self.skip_corrupt = bool(skip_corrupt_data)
        self.processor: Optional[BaseProcessor] = None
        self.main_image_key = main_image_key
        self.main_image_size = list(main_image_size or [224, 224])

        # Parse shape_meta to build key mappings (same logic as BaseLerobotDataset).
        self.image_meta = []
        for meta in shape_meta["images"]:
            m = dict(meta)
            key = m["key"]
            if m.get("lerobot_key"):
                pass  # explicit override for non-standard column names
            elif key.startswith("observation.images."):
                m["lerobot_key"] = key
            elif key == "default":
                m["lerobot_key"] = "observation.images"
            else:
                m["lerobot_key"] = f"observation.images.{key}"
            self.image_meta.append(m)

        self.state_meta = []
        for meta in shape_meta["state"]:
            m = dict(meta)
            key = m["key"]
            if m.get("lerobot_key"):
                pass
            elif key.startswith("observation.state."):
                m["lerobot_key"] = key
            elif key == "default":
                m["lerobot_key"] = "observation.state"
            else:
                m["lerobot_key"] = f"observation.state.{key}"
            self.state_meta.append(m)

        self.action_meta = []
        for meta in shape_meta["action"]:
            m = dict(meta)
            key = m["key"]
            if m.get("lerobot_key"):
                pass
            elif key.startswith("action."):
                m["lerobot_key"] = key
            elif key == "default":
                m["lerobot_key"] = "action"
            else:
                m["lerobot_key"] = f"action.{key}"
            self.action_meta.append(m)

        # Load episodes metadata (fast: reads small parquet with episode info).
        # Decode uses this table for video paths and timestamps.
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self.meta = LeRobotDatasetMetadata(repo_id=dataset_dir, root=Path(dataset_dir))
        self.fps = self.meta.fps
        self._dataset_root = Path(dataset_dir)

        # Full per-episode arrays (length / chunk / file / task) in original
        # episode-index order, read directly from the source metadata.
        all_len, all_chunk, all_file, all_task = self._episodes_from_meta()
        total_episodes = len(all_len)

        # Train/val split (same logic as BaseLerobotDataset).
        if val_set_proportion < 1e-6:
            selected_episodes = list(range(total_episodes))
        else:
            split_idx = int(total_episodes * (1 - val_set_proportion))
            episode_indices = list(range(total_episodes))
            rng = np.random.default_rng(seed)
            rng.shuffle(episode_indices)
            if is_training_set:
                selected_episodes = sorted(episode_indices[:split_idx])
            else:
                selected_episodes = sorted(episode_indices[split_idx:])

        # Store episode info for selected episodes.
        self._selected_episodes = selected_episodes  # original episode indices
        self.episode_lengths = [all_len[i] for i in selected_episodes]
        self.anchor_lengths = supervised_anchor_lengths(
            self.episode_lengths, self.minimum_future_offset
        )
        self.num_loaded_episodes = len(self.episode_lengths)

        # Per-episode parquet file location (for on-demand reading).
        self._ep_chunk_idx = [all_chunk[i] for i in selected_episodes]
        self._ep_file_idx = [all_file[i] for i in selected_episodes]

        # Integer frame offsets for sampling windows.
        self.obs_offsets = np.array(
            [t * global_sample_stride for t in range(num_frames)], dtype=np.int64
        )
        self.action_offsets = np.array(
            [t * global_sample_stride for t in range(num_frames - 1)], dtype=np.int64
        )


        # Some merged datasets ship episodes metadata whose ``data/file_index``
        # is off-by-one for a handful of episodes; cache the corrected
        # (chunk, file) per position once located so we don't re-search.
        self._ep_file_correction: Dict[int, tuple] = {}

    # ------------------------------------------------------------------ #
    # Source episode metadata in original episode-index order.
    # ------------------------------------------------------------------ #

    def _episodes_from_meta(self):
        """(length, chunk, file, task) per original episode from the meta table."""
        ep_meta = self.meta.episodes
        all_len, all_chunk, all_file, all_task = [], [], [], []
        for i in range(len(ep_meta)):
            row = ep_meta[i]
            all_len.append(int(row["length"]))
            all_chunk.append(int(row["data/chunk_index"]))
            all_file.append(int(row["data/file_index"]))
            tasks = row.get("tasks") or [""]
            all_task.append(tasks[0] if len(tasks) else "")
        return all_len, all_chunk, all_file, all_task


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
    # Direct parquet reading (replaces LeRobotDataset.hf_dataset)
    # ------------------------------------------------------------------ #
    _FILE_SEARCH_RADIUS = 4

    @staticmethod
    def _find_ep_rows(table, original_ep: int):
        """Return (start, end) row range for ``original_ep`` in ``table``, or
        None if absent. Robust regardless of where the episode sits in the file."""
        ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
        idx = np.nonzero(ep_col == original_ep)[0]
        if idx.size == 0:
            return None
        return int(idx[0]), int(idx[-1]) + 1

    def _read_episode_data(
        self,
        pos: int,
        frame_start: int = 0,
        frame_stop: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read one episode's action/state/metadata directly from parquet.

        ``pos`` is the position index within selected_episodes (not the
        original episode index).
        """
        original_ep = self._selected_episodes[pos]
        chunk_idx = self._ep_chunk_idx[pos]
        file_idx = self._ep_file_idx[pos]

        # Determine which columns we need.
        needed_cols = ["episode_index", "timestamp", "task_index"]
        for meta in self.state_meta:
            needed_cols.append(meta["lerobot_key"])
        for meta in self.action_meta:
            needed_cols.append(meta["lerobot_key"])
        # Several canonical fields may intentionally select different slices
        # from one physical parquet column.  PyArrow rejects duplicate column
        # requests, so read every source column only once.
        needed_cols = list(dict.fromkeys(needed_cols))

        # Locate the episode. The claimed (chunk, file) is correct for almost
        # all episodes, but merged datasets occasionally have an off-by-one
        # ``data/file_index``; search neighbouring files and cache any fix.
        corrected = self._ep_file_correction.get(pos)
        if corrected is not None:
            candidates = [corrected]
        else:
            candidates = [(chunk_idx, file_idx)]
            for d in range(1, self._FILE_SEARCH_RADIUS + 1):
                candidates.append((chunk_idx, file_idx - d))
                candidates.append((chunk_idx, file_idx + d))

        ep_table = None
        for c, f in candidates:
            if f < 0:
                continue
            parquet_path = self._dataset_root / f"data/chunk-{c:03d}/file-{f:03d}.parquet"
            if not parquet_path.exists():
                continue
            table = pq.read_table(str(parquet_path), columns=needed_cols)
            rows = self._find_ep_rows(table, original_ep)
            if rows is not None:
                if (c, f) != (chunk_idx, file_idx):
                    self._ep_file_correction[pos] = (c, f)
                    logger.warning(
                        "Episode %d not in claimed chunk-%03d/file-%03d; found in "
                        "chunk-%03d/file-%03d (metadata file_index off by %+d)",
                        original_ep, chunk_idx, file_idx, c, f, f - file_idx,
                    )
                start, end = rows
                ep_table = table.slice(start, end - start)
                break

        if ep_table is None:
            raise RuntimeError(
                f"Episode {original_ep} not found in chunk-{chunk_idx:03d}/"
                f"file-{file_idx:03d} or within ±{self._FILE_SEARCH_RADIUS} files"
            )

        frame_stop = len(ep_table) if frame_stop is None else int(frame_stop)
        if not 0 <= frame_start < frame_stop <= len(ep_table):
            raise ValueError(
                f"Invalid frame range [{frame_start}, {frame_stop}) for "
                f"episode {original_ep} with {len(ep_table)} rows"
            )
        ep_table = ep_table.slice(frame_start, frame_stop - frame_start)

        result = {}
        for col_name in needed_cols:
            col_data = ep_table.column(col_name).to_pylist()
            if col_name in ("episode_index", "task_index"):
                result[col_name] = torch.tensor(col_data, dtype=torch.long)
            elif col_name == "timestamp":
                result[col_name] = torch.tensor(col_data, dtype=torch.float64)
            else:
                # action / state columns: list of lists → tensor
                result[col_name] = torch.tensor(col_data, dtype=torch.float32)
        return result

    # ------------------------------------------------------------------ #
    # Batch decode
    # ------------------------------------------------------------------ #

    def _decode_episode(self, pos: int, decoder_cache) -> Dict[str, Any]:
        return self._decode_chunk(pos, decoder_cache)

    def _decode_chunk(self, spec: Any, decoder_cache) -> Dict[str, Any]:
        """Read one bounded anchor range plus its observation/action halo."""

        pos = int(spec if isinstance(spec, (int, np.integer)) else spec[0])
        ep_len = int(self.episode_lengths[pos])
        anchor_len = int(self.anchor_lengths[pos])
        pos, anchor_start, anchor_stop = self._chunk_descriptor(spec, anchor_len)
        if not 0 <= anchor_start < anchor_stop <= anchor_len:
            raise ValueError(
                f"Invalid anchor chunk {(pos, anchor_start, anchor_stop)} "
                f"for anchor_len={anchor_len}, ep_len={ep_len}"
            )
        all_offsets = np.concatenate((self.obs_offsets, self.action_offsets))
        anchor_indices = np.arange(anchor_start, anchor_stop, dtype=np.int64)
        image_frame_indices = np.unique(
            np.clip(
                anchor_indices[:, None] + self.obs_offsets[None, :],
                0,
                ep_len - 1,
            ).reshape(-1)
        )
        frame_start = max(0, anchor_start + int(all_offsets.min(initial=0)))
        frame_stop = min(
            ep_len,
            anchor_stop + int(all_offsets.max(initial=0)),
        )
        cols = self._read_episode_data(pos, frame_start, frame_stop)
        original_ep = self._selected_episodes[pos]
        timestamps = cols["timestamp"].numpy().astype(np.float64)
        image_row_indices = image_frame_indices - frame_start
        if (
            np.any(image_row_indices < 0)
            or np.any(image_row_indices >= len(timestamps))
        ):
            raise RuntimeError(
                "Selected image-frame indices fall outside the loaded parquet "
                f"range [{frame_start}, {frame_stop}): "
                f"{image_frame_indices.tolist()}"
            )

        state = {}
        for meta in self.state_meta:
            state[meta["key"]] = self._select_dimensions(
                cols[meta["lerobot_key"]], meta
            )

        action = {}
        for meta in self.action_meta:
            action[meta["key"]] = self._select_dimensions(
                cols[meta["lerobot_key"]], meta
            )

        images: Dict[str, torch.Tensor] = {}
        main_images: Optional[torch.Tensor] = None
        ep_row = self.meta.episodes[original_ep]
        for meta in self.image_meta:
            vid_key = meta["lerobot_key"]
            from_ts = float(ep_row[f"videos/{vid_key}/from_timestamp"])
            decode_ts = (timestamps[image_row_indices] + from_ts).tolist()
            video_path = self.meta.root / self.meta.get_video_file_path(original_ep, vid_key)
            frames = self._decode_frames(str(video_path), decode_ts, decoder_cache)
            if self.main_image_key == meta["key"]:
                main_images = (
                    transforms_F.resize(
                        frames,
                        size=self.main_image_size,
                        interpolation=transforms_F.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .contiguous()
                )
            if self.decode_resize is not None:
                # Resize once while materializing the episode. Performing the
                # same resize in the processor would repeat it for heavily
                # overlapping 33-frame sample windows.
                frames = transforms_F.resize(
                    frames,
                    size=self.decode_resize,
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            images[meta["key"]] = (frames * 255).to(torch.uint8)

        return {
            "original_ep": original_ep,
            "ep_len": ep_len,
            "num_anchors": anchor_stop - anchor_start,
            "anchor_start": anchor_start,
            "frame_start": frame_start,
            "image_frame_indices": image_frame_indices,
            # Task ids are frame-level in LeRobot.  Preserve the decoded range
            # and resolve the instruction at each anchor instead of assigning
            # the first row's task to an entire chunk.
            "task_indices": cols["task_index"],
            "state": state,
            "action": action,
            "images": images,
            "main_images": main_images,
        }

    @staticmethod
    def _select_dimensions(value: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Select canonical coordinates from a non-standard source column.

        Some converted RoboMIND v3 datasets store the gripper as the last
        coordinate of ``joint_position`` *and* expose it in a dedicated
        ``gripper`` column.  ``select_indices`` lets the data config keep only
        the arm joints from the former before the canonical fields are
        concatenated, avoiding a duplicated gripper coordinate.
        """
        indices = meta.get("select_indices")
        if indices is None:
            return value
        if not indices:
            raise ValueError(f"select_indices must not be empty for {meta['key']!r}")
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        if index.min().item() < 0 or index.max().item() >= value.shape[-1]:
            raise IndexError(
                f"select_indices={list(indices)} exceed source width "
                f"{value.shape[-1]} for {meta['lerobot_key']!r}"
            )
        return value.index_select(-1, index)

    @staticmethod
    def _decode_frames(video_path: str, decode_ts: List[float], decoder_cache) -> torch.Tensor:
        """Decode all ``decode_ts`` of one episode in one batched call."""
        video_path = _resolve_video_reference(video_path)
        # v3.0 packs many episodes into one mp4. For the last episode in a file
        # the final frame's absolute timestamp can map via ``round(ts*fps)`` to
        # ``num_frames`` (one past the last valid index ``num_frames-1``) due to
        # float rounding, which makes the batched ``get_frames_at`` call raise.
        # Clamp to the last valid frame's timestamp so the fast path is kept.
        try:
            decoder = decoder_cache.get_decoder(video_path)
        except Exception as exc:
            try:
                video_bytes: int | str = Path(video_path).stat().st_size
            except OSError as stat_error:
                video_bytes = f"unavailable: {stat_error}"
            raise RuntimeError(
                f"Failed to open video {video_path!r} (bytes={video_bytes})"
            ) from exc
        meta = decoder.metadata
        max_ts = (meta.num_frames - 1) / meta.average_fps
        decode_ts = [min(ts, max_ts) for ts in decode_ts]
        try:
            return decode_video_frames_torchcodec(
                video_path, decode_ts, _TOLERANCE_S, decoder_cache=decoder_cache
            )
        except Exception as exc:
            logger.warning(
                "Batched decode of %s failed (%s); retrying per-frame.",
                video_path,
                exc,
            )
            batched_error = exc
        retry_cache = VideoDecoderCache()
        try:
            frames: List[torch.Tensor] = []
            for i, ts in enumerate(decode_ts):
                try:
                    frame = decode_video_frames_torchcodec(
                        video_path, [ts], _TOLERANCE_S, decoder_cache=retry_cache
                    )[0]
                except Exception as frame_exc:
                    raise RuntimeError(
                        f"Frame {i} at timestamp {ts:.9f}s failed to decode for "
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
        """Rebuild the nested dict for a single anchor, then run processor."""
        ep_len = ep_payload["ep_len"]
        global_anchor = int(ep_payload.get("anchor_start", 0)) + int(anchor)
        obs_q = global_anchor + self.obs_offsets
        act_q = global_anchor + self.action_offsets
        obs_idx = np.clip(obs_q, 0, ep_len - 1)
        act_idx = np.clip(act_q, 0, ep_len - 1)
        frame_start = int(ep_payload.get("frame_start", 0))
        obs_local_idx = obs_idx - frame_start
        act_local_idx = act_idx - frame_start
        image_frame_indices = ep_payload["image_frame_indices"]
        image_local_idx = np.searchsorted(image_frame_indices, obs_idx)
        if (
            np.any(image_local_idx >= len(image_frame_indices))
            or not np.array_equal(image_frame_indices[image_local_idx], obs_idx)
        ):
            raise RuntimeError(
                "Decoded image-frame index is missing from shard payload"
            )
        obs_pad = torch.from_numpy(obs_q >= ep_len)
        act_pad = torch.from_numpy(act_q >= ep_len)

        sample: Dict[str, Any] = {
            "idx": ep_payload["original_ep"] * 1_000_000 + global_anchor,
            "task": self.meta.tasks.iloc[
                int(ep_payload["task_indices"][global_anchor - frame_start].item())
            ].name,
            "action": {},
            "state": {},
            "images": {},
        }
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
            "episode_id": int(ep_payload["original_ep"]),
            "anchor_index": global_anchor,
        }
        return processed

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        return self

    def __len__(self) -> int:
        return sum(self.anchor_lengths)

    # ------------------------------------------------------------------ #
    # Normalization statistics.  Runtime and stats intentionally share
    # _read_episode_data() + _select_dimensions(); using the map-style
    # BaseLerobotDataset here would silently ignore select_indices.
    # ------------------------------------------------------------------ #
    def _accumulate_stats(
        self, preprocessor: BaseProcessor, skip_corrupt_data: bool = False
    ) -> Dict[str, Any]:
        acc = {
            group: {
                stat: defaultdict(list)
                for stat in ("min", "max", "mean", "var", "q01", "q99")
            }
            for group in ("state", "action")
        }

        def process_episode(pos: int):
            cols = self._read_episode_data(pos)
            batch = {"state": {}, "action": {}}
            for meta in self.state_meta:
                value = self._select_dimensions(cols[meta["lerobot_key"]], meta)
                batch["state"][meta["key"]] = value.unsqueeze(1).float()
            for meta in self.action_meta:
                value = self._select_dimensions(cols[meta["lerobot_key"]], meta)
                batch["action"][meta["key"]] = sliding_window_with_replication(
                    value.float(), self.action_size
                )
            return preprocessor.action_state_transform(batch)

        max_workers = max(1, self.decode_workers) * 4
        accepted = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_episode, pos)
                for pos in range(self.num_loaded_episodes)
            ]
            from tqdm import tqdm

            for future in tqdm(
                as_completed(futures),
                total=self.num_loaded_episodes,
                desc="LeRobot v3 stats",
            ):
                try:
                    batch = future.result()
                    staged = []
                    for group, metas in (
                        ("state", self.state_meta),
                        ("action", self.action_meta),
                    ):
                        for meta in metas:
                            key = meta["key"]
                            cur = batch[group][key]
                            if cur.numel() == 0 or not torch.isfinite(cur).all():
                                raise ValueError(
                                    f"non-finite/empty values in {group}/{key}"
                                )
                            staged.append(
                                (
                                    group,
                                    key,
                                    cur.amin(0),
                                    cur.amax(0),
                                    cur.mean(0),
                                    cur.var(0),
                                    torch.quantile(cur, 0.01, dim=0),
                                    torch.quantile(cur, 0.99, dim=0),
                                )
                            )
                except Exception as exc:
                    if not skip_corrupt_data:
                        raise
                    logger.warning(
                        "offline quarantine candidate: dropping one episode from "
                        "stats in %s: %s",
                        self.dataset_dir,
                        exc,
                    )
                    continue
                accepted += 1
                for group, key, mn, mx, mean, var, q01, q99 in staged:
                    acc[group]["min"][key].append(mn)
                    acc[group]["max"][key].append(mx)
                    acc[group]["mean"][key].append(mean)
                    acc[group]["var"][key].append(var)
                    acc[group]["q01"][key].append(q01)
                    acc[group]["q99"][key].append(q99)
        if accepted != self.num_loaded_episodes and not skip_corrupt_data:
            raise RuntimeError(
                f"stats accepted {accepted}/{self.num_loaded_episodes} episodes"
            )
        return acc

    def get_dataset_stats(
        self, preprocessor: BaseProcessor, skip_corrupt_data: bool = False
    ) -> Dict[str, Any]:
        from .sharded_lerobot_v2_dataset import ShardedLerobotV2Dataset

        acc = self._accumulate_stats(
            preprocessor, skip_corrupt_data=skip_corrupt_data
        )
        return ShardedLerobotV2Dataset._reduce_stats(
            acc,
            {"state": self.state_meta, "action": self.action_meta},
            self.num_loaded_episodes,
            int(sum(self.episode_lengths)),
        )

    @staticmethod
    def pooled_dataset_stats(
        providers,
        preprocessor: BaseProcessor,
        skip_corrupt_data: bool = False,
    ) -> Dict[str, Any]:
        if not providers:
            raise ValueError("pooled_dataset_stats needs at least one provider")
        from .sharded_lerobot_v2_dataset import ShardedLerobotV2Dataset

        head = providers[0]
        merged = {
            group: {
                stat: defaultdict(list)
                for stat in ("min", "max", "mean", "var", "q01", "q99")
            }
            for group in ("state", "action")
        }
        num_episodes = 0
        num_transition = 0
        for provider in providers:
            acc = provider._accumulate_stats(
                preprocessor, skip_corrupt_data=skip_corrupt_data
            )
            for group in ("state", "action"):
                for stat in ("min", "max", "mean", "var", "q01", "q99"):
                    for key, values in acc[group][stat].items():
                        merged[group][stat][key].extend(values)
            num_episodes += provider.num_loaded_episodes
            num_transition += int(sum(provider.episode_lengths))
        return ShardedLerobotV2Dataset._reduce_stats(
            merged,
            {"state": head.state_meta, "action": head.action_meta},
            num_episodes,
            num_transition,
        )
