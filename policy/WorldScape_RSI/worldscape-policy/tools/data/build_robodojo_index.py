#!/usr/bin/env python3
"""Build a RoboDojo temporal index on CPU before distributed training."""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools.data.robodojo_common import output_path
from worldscape_policy.data.adapters.common import (
    LanguageTemporalPacker,
    _effective_length,
    _eligible_temporal_anchors,
    _first,
    _temporal_labels,
)
from worldscape_policy.data.adapters.robodojo import RoboDojoJointHDF5Dataset
from worldscape_policy.data.robodojo_index import (
    INDEX_VERSION,
    index_options,
    read_index,
)


def _read_episode_metadata(path, episode, tasks):
    """Use the adapter's metadata reader without recursively building an index."""
    reader = object.__new__(RoboDojoJointHDF5Dataset)
    reader.paths = [path]
    reader.episodes = [episode]
    reader.tasks = tasks
    reader.include_joint_velocity = False
    reader._index_arrays = {}
    return reader._read_index_arrays(0)


class IndexBuilder(RoboDojoJointHDF5Dataset):
    """Match the native packed index order while releasing per-episode scratch."""

    def __init__(self, *args, metadata_workers=1, **kwargs):
        if metadata_workers < 1:
            raise ValueError("metadata_workers must be positive")
        self.metadata_workers = metadata_workers
        super().__init__(*args, **kwargs)

    def _metadata_stream(self):
        if self.metadata_workers == 1:
            for index in range(len(self.paths)):
                yield self._read_index_arrays(index)
            return
        # h5py serializes threads. Separate read-only processes overlap remote
        # metadata I/O; a bounded queue preserves episode order and memory use.
        with ProcessPoolExecutor(
            max_workers=self.metadata_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            pending = deque()
            jobs = iter(zip(self.paths, self.episodes))

            def submit_next():
                job = next(jobs, None)
                if job is not None:
                    pending.append(pool.submit(_read_episode_metadata, *job, self.tasks))

            for _ in range(self.metadata_workers * 2):
                submit_next()
            while pending:
                arrays = pending.popleft().result()
                submit_next()
                yield arrays

    def _build_index(self) -> np.ndarray:
        if not self.temporal_packing or self.action_horizon not in {24, 48}:
            raise ValueError(
                "Offline RoboDojo indexing requires temporal packing and horizon 24 or 48"
            )
        result, progress, lengths = [], [], []
        self.episode_lengths = []
        print(f"Indexing 0/{len(self.paths)} episodes", flush=True)
        for episode_index, arrays in enumerate(self._metadata_stream()):
            max_chunks = self._episode_max_chunk_size(episode_index)
            packer = LanguageTemporalPacker(
                max_chunk_size=max_chunks, chunk_size=self.action_horizon
            )
            anchors = _eligible_temporal_anchors(
                arrays,
                max_chunks,
                action_horizon=self.action_horizon,
                respect_trajectory_segments=self.respect_trajectory_segments,
                anchor_start=self.temporal_anchor_index,
                step_stride=self.step_stride,
            )
            if self.temporal_anchor_index in anchors:
                anchors = [self.temporal_anchor_index] + [
                    anchor for anchor in anchors if anchor != self.temporal_anchor_index
                ]
            length = _effective_length(arrays)
            self.episode_lengths.append(length)
            mask_value = _first(arrays, ("meta.is_exec", "is_exec"))
            mask = (
                None
                if mask_value is None
                else np.asarray(mask_value).astype(bool).reshape(-1)
            )
            labels, segments = _temporal_labels(arrays, length, mask)
            for anchor in anchors:
                packed = packer.indices(
                    anchor,
                    labels,
                    trajectory_ids=segments
                    if self.respect_trajectory_segments
                    else None,
                )
                result.append((episode_index, anchor))
                progress.append(float(packed.video[0]) / max(1, length))
                lengths.append(
                    (len(packed.action), len(packed.state), len(packed.video))
                )
            self._index_arrays.pop(episode_index, None)
            if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(self.paths):
                print(
                    f"Indexing {episode_index + 1}/{len(self.paths)} episodes; {len(result)} samples",
                    flush=True,
                )
        if not result:
            raise ValueError("dataset has no eligible temporal packing anchors")
        self._sampling_progress = np.asarray(progress, dtype=np.float64)
        self.index_lengths = np.asarray(lengths, dtype=np.int64)
        return np.asarray(result, dtype=np.int64)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunk-size", type=int, default=1)
    parser.add_argument(
        "--chunk-policy", choices=("uniform", "task_aware"), default="uniform",
        help="task_aware uses memory/other limits 24/4 at horizon 24, or 12/1 at horizon 48",
    )
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--max-event-chunks", type=int, default=16)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--temporal-anchor-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--metadata-workers", type=int, default=1,
                        help="Parallel read-only HDF5 metadata processes; preserves index order")
    parser.add_argument("--normalization", choices=("zscore", "q99"), default=None,
                        help="Default: infer from the statistics artifact")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--respect-trajectory-segments", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the saved index and source size/mtime; do not rebuild.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if not args.verify_only:
        output_path(output, overwrite=args.overwrite)
    if args.step_stride < 1 or args.max_chunk_size < 1 or args.temporal_anchor_index < 0:
        raise ValueError("stride/chunk size must be positive and anchor index nonnegative")
    if args.verify_only:
        artifact = read_index(output, verify_files=True)
        if artifact["manifest"]["options"]["data_roots"] != [
            str(args.data_root.expanduser().resolve())
        ]:
            raise ValueError("Index data_root mismatch")
        print(
            f"Verified {output}: {len(artifact['manifest']['paths'])} episodes",
            flush=True,
        )
        return 0
    print(
        "Scanning episode paths (CPU only; image payloads are not read)...", flush=True
    )
    dataset = IndexBuilder(
        args.data_root.expanduser().resolve(),
        stats_path=args.stats_path,
        relative_joint_action=False,
        normalization=args.normalization,
        temporal_packing=True,
        max_chunk_size=args.max_chunk_size,
        chunk_policy=args.chunk_policy,
        action_horizon=args.action_horizon,
        max_event_chunks=args.max_event_chunks,
        step_stride=args.step_stride,
        temporal_anchor_index=args.temporal_anchor_index,
        respect_trajectory_segments=args.respect_trajectory_segments,
        max_episodes=args.max_episodes,
        history_num_frames=0,
        index_cache=False,
        metadata_workers=args.metadata_workers,
    )
    sources = list(dataset.paths)
    if dataset.metadata_path is not None:
        sources.append(dataset.metadata_path)
    sources.extend(
        root / "meta/tasks.jsonl"
        for root in dataset.roots
        if (root / "meta/tasks.jsonl").is_file()
    )
    records = []
    for path in sources:
        stat = path.stat()
        records.append(
            {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    manifest = {
        "version": INDEX_VERSION,
        "kind": "robodojo_joint",
        "options": index_options(dataset),
        "paths": [str(path) for path in dataset.paths],
        "episodes": dataset.episodes,
        "tasks": dataset.tasks,
        "episode_lengths": dataset.episode_lengths,
        "source_files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest)),
                index=dataset.index,
                sampling_progress=dataset._sampling_progress,
                lengths=dataset.index_lengths,
            )
        read_index(temporary)  # Validate the artifact before replacing a previous index.
        output_path(output, overwrite=args.overwrite)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Wrote {output}: {len(dataset.paths)} episodes, {len(dataset)} samples",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
