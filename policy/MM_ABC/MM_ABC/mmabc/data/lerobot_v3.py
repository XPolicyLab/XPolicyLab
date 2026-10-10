"""Lazy reader for LeRobot v3.0 parquet/video profiles with embodiment metadata.

Episode indexes are held in memory; row groups and RGB frames are cached per
worker. Instructions may be embedded strings or references to tasks.parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mmabc.data import parquet_window
from mmabc.data import video as videolib


@dataclass
class ViewSpec:
    """One canonical view slot bound to a concrete video key."""

    slot: str
    video_key: str
    chunk_index: np.ndarray
    file_index: np.ndarray
    from_timestamp: np.ndarray


class LeRobotV3Profile:
    """Episode index and lazy row/frame access for one dataset profile."""

    def __init__(self, root: str | Path, view_map: dict[str, str | None]) -> None:
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.modality = json.loads((self.root / "meta" / "modality.json").read_text())
        self.embodiment = json.loads((self.root / "meta" / "embodiment.json").read_text())
        self.fps = float(self.info["fps"])
        self.data_path_tmpl = self.info["data_path"]
        self.video_path_tmpl = self.info["video_path"]

        self._load_episode_index(view_map)
        self._rows = parquet_window.reader(("observation.state", "action"))
        self._file_base: dict[tuple[int, int], int] = {}

    def _load_episode_index(self, view_map: dict[str, str | None]) -> None:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no episode metadata under {self.root}/meta/episodes")

        video_keys = {
            slot: self.modality["video"][key]["original_key"]
            for slot, key in view_map.items()
            if key is not None
        }
        want = [
            "episode_index",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "data/chunk_index",
            "data/file_index",
        ]
        for vk in video_keys.values():
            want += [
                f"videos/{vk}/chunk_index",
                f"videos/{vk}/file_index",
                f"videos/{vk}/from_timestamp",
            ]

        tables = []
        task_cols: list[str] = []
        for f in files:
            schema = pq.ParquetFile(f).schema_arrow
            cols = [c for c in want if c in schema.names]
            if not task_cols:
                task_cols = [c for c in ("task_texts", "tasks") if c in schema.names]
            tables.append(pq.read_table(f, columns=cols + task_cols))

        import pyarrow as pa

        tbl = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
        col = lambda n: np.asarray(tbl.column(n).to_numpy(zero_copy_only=False))

        self.episode_index = col("episode_index").astype(np.int64)
        self.length = col("length").astype(np.int64)
        self.dataset_from = col("dataset_from_index").astype(np.int64)
        self.data_chunk = col("data/chunk_index").astype(np.int64)
        self.data_file = col("data/file_index").astype(np.int64)
        self.num_episodes = len(self.episode_index)

        self.views: dict[str, ViewSpec] = {}
        for slot, vk in video_keys.items():
            base = f"videos/{vk}"
            if f"{base}/chunk_index" not in tbl.schema.names:
                continue
            self.views[slot] = ViewSpec(
                slot=slot,
                video_key=vk,
                chunk_index=col(f"{base}/chunk_index").astype(np.int64),
                file_index=col(f"{base}/file_index").astype(np.int64),
                from_timestamp=col(f"{base}/from_timestamp").astype(np.float64),
            )

        self._tasks = self._resolve_tasks(tbl, task_cols)

    def _resolve_tasks(self, tbl, task_cols: list[str]) -> list[str]:
        """One instruction string per episode, whichever dialect is in use."""
        lookup: dict[int, str] | None = None
        tasks_parquet = self.root / "meta" / "tasks.parquet"
        if tasks_parquet.exists():
            t = pq.read_table(tasks_parquet)
            names = t.schema.names
            if "task" in names:
                idx_col = "task_index" if "task_index" in names else names[0]
                lookup = {
                    int(i): str(s)
                    for i, s in zip(
                        t.column(idx_col).to_pylist(), t.column("task").to_pylist()
                    )
                }

        for name in ("task_texts", "tasks"):
            if name not in task_cols:
                continue
            raw = tbl.column(name).to_pylist()
            out: list[str] = []
            ok = True
            for entry in raw:
                if not entry:
                    out.append("")
                    continue
                first = entry[0]
                if isinstance(first, str):
                    out.append(first)
                elif lookup is not None:
                    out.append(lookup.get(int(first), ""))
                else:
                    ok = False
                    break
            if ok and len(out) == self.num_episodes:
                return out
        return [""] * self.num_episodes

    def _row_offset(self, ep: int) -> int:
        """First row of this episode inside its own data shard.

        Shards hold anywhere from one episode to several thousand, so the
        episode's offset is its global start minus the shard's global start.
        The per-shard minimum is computed once and memoised.
        """
        key = (int(self.data_chunk[ep]), int(self.data_file[ep]))
        base = self._file_base.get(key)
        if base is None:
            same = (self.data_chunk == key[0]) & (self.data_file == key[1])
            base = int(self.dataset_from[same].min())
            self._file_base[key] = base
        return int(self.dataset_from[ep]) - base

    def _shard_path(self, ep: int) -> str:
        return str(
            self.root
            / self.data_path_tmpl.format(
                chunk_index=int(self.data_chunk[ep]), file_index=int(self.data_file[ep])
            )
        )

    def task(self, ep: int) -> str:
        return self._tasks[ep]

    def sample_window(
        self, ep: int, frame: int, chunk: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """State at `frame` plus `chunk` action rows, padded past episode end.

        Returns (state[80], actions[chunk,80], valid_steps). Padding repeats the
        final row so the tensor shape stays static; `valid_steps` tells the
        caller how much of it may be supervised.
        """
        n = int(self.length[ep])
        frame = int(np.clip(frame, 0, max(n - 1, 0)))
        offset = self._row_offset(ep)

        # One windowed read covers the anchor state and the whole action chunk.
        lo = offset + frame
        hi = offset + min(frame + chunk, n)
        state_rows, action_rows = self._rows.read(self._shard_path(ep), lo, hi)
        if len(state_rows) == 0:
            raise RuntimeError(f"empty window for episode {ep} frame {frame} in {self.root}")

        state = state_rows[0]
        valid = len(action_rows)
        if valid < chunk:
            pad = np.repeat(action_rows[-1:], chunk - valid, axis=0)
            action_rows = np.concatenate([action_rows, pad], axis=0)
        return state, action_rows, valid

    def read_views(
        self, ep: int, frames: list[int], size: int
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        """Decode `frames` for every view slot this profile binds."""
        images: dict[str, np.ndarray] = {}
        present: dict[str, bool] = {}
        for slot, spec in self.views.items():
            path = self.root / self.video_path_tmpl.format(
                video_key=spec.video_key,
                chunk_index=int(spec.chunk_index[ep]),
                file_index=int(spec.file_index[ep]),
            )
            times = [
                videolib.timestamp_of(float(spec.from_timestamp[ep]), f, self.fps) for f in frames
            ]
            decoded = videolib.pool().read(str(path), times)
            images[slot] = np.stack([videolib.to_square(f, size) for f in decoded])
            present[slot] = True
        return images, present
