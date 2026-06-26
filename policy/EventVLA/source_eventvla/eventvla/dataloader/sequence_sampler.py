from __future__ import annotations

from bisect import bisect_left
import hashlib
from random import Random
from typing import Iterator, List, Set, Tuple

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


EpisodeSampleIndex = Tuple[int, object, int, bool, bool, int, int, bool]


class SequentialEpisodeBatchSampler(Sampler[List[EpisodeSampleIndex]]):
    """
    Stream-style episode sampler.

    It expands each trajectory into a sparse temporal anchor stream and then
    chunks the per-rank stream into fixed-size batches.

    Each yielded tuple is:
        (
            dataset_index,
            trajectory_id,
            step_index,
            is_new_episode,
            is_last_sampled_step,
            anchor_index,
            prev_anchor_step,
            is_keyframe_approx,
        )
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_trajectories: bool = False,
        seed: int = 42,
        sampling_interval: int = 1,
        action_horizon: int = 1,
        balance_dataset_step_counts: bool = False,
        rank: int | None = None,
        num_replicas: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_trajectories = bool(shuffle_trajectories)
        self.seed = int(seed)
        self.sampling_interval = max(1, int(sampling_interval))
        self.action_horizon = max(1, int(action_horizon))
        self.balance_dataset_step_counts = bool(balance_dataset_step_counts)

        self.epoch = 0
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

        all_trajectory_pool: List[Tuple[int, object, int]] = []
        for dataset_index, ds in enumerate(self.dataset.datasets):
            traj_ids = list(ds.trajectory_ids)
            traj_lens = list(ds.trajectory_lengths)
            for traj_id, traj_len in zip(traj_ids, traj_lens):
                traj_len = int(traj_len)
                if traj_len > 0:
                    all_trajectory_pool.append((dataset_index, traj_id, traj_len))
        self._all_trajectory_pool = all_trajectory_pool
        if self.balance_dataset_step_counts and self.rank == 0:
            base_counts = self._count_steps_by_dataset(self._all_trajectory_pool)
            valid_counts = [count for count in base_counts if count > 0]
            target_count = max(valid_counts) if len(valid_counts) > 0 else 0
            print(
                "SequentialEpisodeBatchSampler task step balancing enabled: "
                f"base sampled steps per dataset={base_counts}, target per dataset={target_count}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _normalize_trajectory_id(trajectory_id: object) -> object:
        if hasattr(trajectory_id, "item"):
            return trajectory_id.item()
        return trajectory_id

    def _build_anchor_seed(self, dataset_index: int, trajectory_id: object) -> int:
        normalized_traj_id = self._normalize_trajectory_id(trajectory_id)
        payload = (
            f"{self.seed}|{self.epoch}|{int(dataset_index)}|"
            f"{type(normalized_traj_id).__name__}|{repr(normalized_traj_id)}"
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _build_sparse_anchors(
        self,
        dataset_index: int,
        trajectory_id: object,
        trajectory_length: int,
    ) -> List[int]:
        max_valid_step = int(trajectory_length) - self.action_horizon
        if max_valid_step < 0:
            return []

        if self.sampling_interval <= 1:
            return list(range(max_valid_step + 1))

        anchors = [0]
        if max_valid_step >= 1:
            upper = min(self.sampling_interval, max_valid_step)
            rng = Random(self._build_anchor_seed(dataset_index=dataset_index, trajectory_id=trajectory_id))
            second_anchor = rng.randint(1, upper)
            anchors.append(second_anchor)
            while anchors[-1] + self.sampling_interval <= max_valid_step:
                anchors.append(anchors[-1] + self.sampling_interval)

        # Try to keep a tail-side anchor unless the preceding anchor's action
        # chunk already covers the raw trajectory end.
        last_step = int(trajectory_length) - 1
        tail_anchor = min(last_step - self.sampling_interval, max_valid_step)
        if tail_anchor < 0 or tail_anchor in anchors:
            return anchors

        prev_candidates = [anchor for anchor in anchors if anchor < tail_anchor]
        if len(prev_candidates) > 0:
            prev_anchor = prev_candidates[-1]
            if prev_anchor + self.action_horizon > last_step:
                return anchors

        anchors.append(int(tail_anchor))
        anchors.sort()
        return anchors

    def _get_keyframe_steps(self, dataset_index: int, trajectory_id: object) -> List[int]:
        ds = self.dataset.datasets[int(dataset_index)]
        getter = getattr(ds, "get_keyframe_steps", None)
        if getter is None:
            getter = getattr(ds, "get_inspect_keyframe_steps", None)
        if getter is None:
            return []

        try:
            raw_steps = getter(trajectory_id)
        except Exception:
            return []

        if raw_steps is None:
            return []

        normalized_steps: List[int] = []
        for raw_step in raw_steps:
            if raw_step is None:
                continue
            if isinstance(raw_step, float) and np.isnan(raw_step):
                continue
            normalized_steps.append(int(raw_step))
        return normalized_steps

    def _get_inspect_keyframe_steps(self, dataset_index: int, trajectory_id: object) -> List[int]:
        return self._get_keyframe_steps(dataset_index=dataset_index, trajectory_id=trajectory_id)

    @staticmethod
    def _nearest_sampled_keyframes(anchors: List[int], inspect_keyframe_steps: List[int]) -> Set[int]:
        if len(anchors) == 0 or len(inspect_keyframe_steps) == 0:
            return set()

        nearest_anchor_steps: Set[int] = set()
        for keyframe_step in inspect_keyframe_steps:
            pos = bisect_left(anchors, int(keyframe_step))
            candidates: List[int] = []
            if pos < len(anchors):
                candidates.append(anchors[pos])
            if pos > 0:
                candidates.append(anchors[pos - 1])
            if len(candidates) == 0:
                continue
            nearest_anchor = min(candidates, key=lambda step: (abs(step - int(keyframe_step)), step))
            nearest_anchor_steps.add(int(nearest_anchor))
        return nearest_anchor_steps

    def _build_rank_trajectory_pool(self) -> List[Tuple[int, object, int]]:
        trajectories = self._build_epoch_trajectory_pool()
        return trajectories[self.rank :: self.num_replicas]

    def _count_sampled_steps(
        self,
        dataset_index: int,
        trajectory_id: object,
        trajectory_length: int,
    ) -> int:
        return len(
            self._build_sparse_anchors(
                dataset_index=dataset_index,
                trajectory_id=trajectory_id,
                trajectory_length=trajectory_length,
            )
        )

    def _count_trajectory_pool_steps(self, trajectories: List[Tuple[int, object, int]]) -> int:
        sampled_steps = 0
        for dataset_index, traj_id, traj_len in trajectories:
            sampled_steps += self._count_sampled_steps(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
                trajectory_length=int(traj_len),
            )
        return int(sampled_steps)

    def _count_steps_by_dataset(self, trajectories: List[Tuple[int, object, int]]) -> List[int]:
        counts = [0 for _ in range(len(self.dataset.datasets))]
        for dataset_index, traj_id, traj_len in trajectories:
            dataset_index = int(dataset_index)
            if 0 <= dataset_index < len(counts):
                counts[dataset_index] += self._count_sampled_steps(
                    dataset_index=dataset_index,
                    trajectory_id=traj_id,
                    trajectory_length=int(traj_len),
                )
        return [int(count) for count in counts]

    def _repeat_trajectory_pool_to_step_target(
        self,
        trajectories: List[Tuple[int, object, int]],
        target_count: int,
    ) -> List[Tuple[int, object, int]]:
        if target_count <= 0 or len(trajectories) == 0:
            return []

        trajectories_with_counts = []
        pool_step_count = 0
        for trajectory in trajectories:
            dataset_index, traj_id, traj_len = trajectory
            sampled_steps = self._count_sampled_steps(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
                trajectory_length=int(traj_len),
            )
            if sampled_steps <= 0:
                continue
            trajectories_with_counts.append((trajectory, int(sampled_steps)))
            pool_step_count += int(sampled_steps)

        if pool_step_count <= 0:
            return []

        repeated: List[Tuple[int, object, int]] = []
        repeated_steps = 0
        full_repeats = int(target_count // pool_step_count)
        for _ in range(full_repeats):
            repeated.extend(trajectories)
        repeated_steps += full_repeats * pool_step_count

        while repeated_steps < target_count:
            added_any = False
            for trajectory, sampled_steps in trajectories_with_counts:
                repeated.append(trajectory)
                repeated_steps += int(sampled_steps)
                added_any = True
                if repeated_steps >= target_count:
                    break
            if not added_any:
                break

        return repeated

    def _balance_trajectory_pool_by_dataset_steps(
        self,
        trajectories: List[Tuple[int, object, int]],
    ) -> List[Tuple[int, object, int]]:
        if not self.balance_dataset_step_counts or len(trajectories) == 0:
            return trajectories

        grouped_trajectories: List[List[Tuple[int, object, int]]] = [
            [] for _ in range(len(self.dataset.datasets))
        ]
        dataset_order: List[int] = []
        seen_dataset_indices = set()
        for trajectory in trajectories:
            dataset_index = int(trajectory[0])
            if not (0 <= dataset_index < len(grouped_trajectories)):
                continue
            grouped_trajectories[dataset_index].append(trajectory)
            if dataset_index not in seen_dataset_indices:
                dataset_order.append(dataset_index)
                seen_dataset_indices.add(dataset_index)

        base_counts = [
            self._count_trajectory_pool_steps(dataset_trajectories)
            for dataset_trajectories in grouped_trajectories
        ]
        valid_counts = [count for count in base_counts if count > 0]
        if len(valid_counts) <= 1:
            return trajectories

        target_count = max(valid_counts)
        balanced_groups = {}
        for dataset_index, dataset_trajectories in enumerate(grouped_trajectories):
            if base_counts[dataset_index] <= 0:
                continue
            balanced_groups[dataset_index] = self._repeat_trajectory_pool_to_step_target(
                trajectories=dataset_trajectories,
                target_count=target_count,
            )

        balanced_trajectories: List[Tuple[int, object, int]] = []
        if self.shuffle_trajectories:
            for dataset_trajectories in balanced_groups.values():
                balanced_trajectories.extend(dataset_trajectories)
            rng = np.random.default_rng(self.seed + self.epoch + 104729)
            rng.shuffle(balanced_trajectories)
            return balanced_trajectories

        for dataset_index in dataset_order:
            balanced_trajectories.extend(balanced_groups.get(dataset_index, []))
        return balanced_trajectories

    def _build_epoch_trajectory_pool(self) -> List[Tuple[int, object, int]]:
        trajectories = list(self._all_trajectory_pool)
        if self.shuffle_trajectories:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(trajectories)
        return self._balance_trajectory_pool_by_dataset_steps(trajectories)

    def _build_flat_step_stream(
        self,
        trajectories: List[Tuple[int, object, int]],
    ) -> List[EpisodeSampleIndex]:
        stream: List[EpisodeSampleIndex] = []
        for dataset_index, traj_id, traj_len in trajectories:
            anchors = self._build_sparse_anchors(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
                trajectory_length=int(traj_len),
            )
            keyframe_steps = self._get_keyframe_steps(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
            )
            nearest_keyframe_steps = self._nearest_sampled_keyframes(
                anchors=anchors,
                inspect_keyframe_steps=keyframe_steps,
            )
            for anchor_index, step in enumerate(anchors):
                prev_anchor_step = anchors[anchor_index - 1] if anchor_index > 0 else -1
                stream.append(
                    (
                        int(dataset_index),
                        traj_id,
                        int(step),
                        anchor_index == 0,
                        anchor_index == len(anchors) - 1,
                        int(anchor_index),
                        int(prev_anchor_step),
                        bool(int(step) in nearest_keyframe_steps),
                    )
                )
        return stream

    def _compute_target_num_batches(self, trajectories: List[Tuple[int, object, int]]) -> int:
        # Deterministic across all ranks (no collectives): all ranks have the same
        # trajectory list + same shuffling seed, and rank assignment is via slicing.
        steps_per_rank: List[int] = []
        for r in range(self.num_replicas):
            rank_pool = trajectories[r :: self.num_replicas]
            sampled_steps = 0
            for dataset_index, traj_id, traj_len in rank_pool:
                sampled_steps += self._count_sampled_steps(
                    dataset_index=int(dataset_index),
                    trajectory_id=traj_id,
                    trajectory_length=int(traj_len),
                )
            steps_per_rank.append(int(sampled_steps))
        if len(steps_per_rank) == 0:
            return 0
        # Use max + local cycling to keep all ranks at identical batch count.
        return int(np.ceil(max(steps_per_rank) / self.batch_size))

    def __len__(self) -> int:
        trajectories = self._build_epoch_trajectory_pool()
        return self._compute_target_num_batches(trajectories)

    def _build_fallback_stream_sample(
        self,
        trajectories: List[Tuple[int, object, int]],
    ) -> EpisodeSampleIndex | None:
        for dataset_index, traj_id, traj_len in trajectories:
            anchors = self._build_sparse_anchors(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
                trajectory_length=int(traj_len),
            )
            if len(anchors) == 0:
                continue
            keyframe_steps = self._get_keyframe_steps(
                dataset_index=int(dataset_index),
                trajectory_id=traj_id,
            )
            nearest_keyframe_steps = self._nearest_sampled_keyframes(
                anchors=anchors,
                inspect_keyframe_steps=keyframe_steps,
            )
            return (
                int(dataset_index),
                traj_id,
                int(anchors[0]),
                True,
                len(anchors) == 1,
                0,
                -1,
                bool(int(anchors[0]) in nearest_keyframe_steps),
            )
        return None

    def __iter__(self) -> Iterator[List[EpisodeSampleIndex]]:
        all_trajectories = self._build_epoch_trajectory_pool()
        trajectories = all_trajectories[self.rank :: self.num_replicas]
        local_stream = self._build_flat_step_stream(trajectories)

        target_num_batches = self._compute_target_num_batches(all_trajectories)
        total_needed = target_num_batches * self.batch_size

        if total_needed == 0:
            return

        if len(local_stream) == 0:
            # Fallback to prevent DDP length mismatch when a rank receives no trajectory.
            fallback_sample = self._build_fallback_stream_sample(all_trajectories)
            if fallback_sample is None:
                return
            local_stream = [fallback_sample]

        if len(local_stream) < total_needed:
            repeats = int(np.ceil(total_needed / len(local_stream)))
            local_stream = (local_stream * repeats)[:total_needed]
        else:
            local_stream = local_stream[:total_needed]

        for i in range(0, total_needed, self.batch_size):
            yield local_stream[i : i + self.batch_size]
