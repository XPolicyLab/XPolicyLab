"""Deterministic family quotas and temporal-block sampling for pretraining."""
from __future__ import annotations
import bisect
import hashlib
import math
import multiprocessing as mp
import os
from typing import Any
import numpy as np
import torch


def safe_hash(values):
    return int(hashlib.sha256(repr(values).encode()).hexdigest(), 16) & ((1 << 128) - 1)


def _dataset_allows_end_padding(dataset, default):
    override = dataset.spec.allow_padding_at_end
    return bool(default if override is None else override)


class PretrainLeRobotMixture(torch.utils.data.Dataset):
    """Deterministic family-balanced temporal blocks over local LeRobot datasets.

    Real trajectories use coverage without replacement; cyclic families can
    repeat, and UMI samples temporal blocks with replacement.
    """
    _OPTIMIZER_STRATIFIED_RETRY_BASE = 1 << 62

    def __init__(self, datasets: list[Any], dataset_weights: list[float] | None=None, *, training: bool=True, balance_dataset_weights: bool=False, balance_trajectory_weights: bool=True, seed: int=42, allow_padding_at_end: bool=False, epoch_length: int | None=None, eval_episodes_per_dataset: int | None=None, eval_frames_per_episode: int | None=None, eval_frame_policy: str='first', eval_scope: str='dataset', eval_segments_per_source: int | None=None, eval_total_samples: int | None=None, eval_fixed_samples: list[dict[str, Any]] | None=None, eval_fixed_per_dataset: bool=False, eval_fixed_episode_pos: int | None=0, eval_fixed_frame_index: int=0, eval_exclude_from_training: str | None=None, training_block_size: int=1, training_batch_size_per_rank: int | None=None, training_frame_stride: int=1, training_sampling_strategy: str='optimizer_stratified_without_replacement', episode_block_tile_size: int=64, bucket_wave_steps: int=16, optimizer_gradient_accumulation_steps: int=1, bucket_wave_apply_dataset_weights: bool=False, bucket_wave_epoch_anchor_family: str | None=None, bucket_wave_cyclic_families: list[str] | tuple[str, ...] | None=None, coverage_min_samples_per_dataset: int=0):
        if not datasets:
            raise ValueError('at least one dataset is required')
        self.datasets = datasets
        self.training = bool(training)
        self.balance_dataset_weights = bool(balance_dataset_weights)
        self.balance_trajectory_weights = bool(balance_trajectory_weights)
        self.seed = int(seed)
        self.allow_padding_at_end = bool(allow_padding_at_end)
        self.training_block_size = max(1, int(training_block_size)) if self.training else 1
        self.training_batch_size_per_rank = max(self.training_block_size, int(training_batch_size_per_rank or self.training_block_size))
        if self.training_batch_size_per_rank % self.training_block_size:
            raise ValueError(f'training_batch_size_per_rank must be divisible by training_block_size, got {self.training_batch_size_per_rank} and {self.training_block_size}')
        self.training_frame_stride = max(1, int(training_frame_stride))
        self.training_sampling_strategy = str(training_sampling_strategy).strip().lower()
        self.episode_block_tile_size = max(1, int(episode_block_tile_size))
        self.bucket_wave_steps = max(1, int(bucket_wave_steps))
        self.optimizer_gradient_accumulation_steps = max(1, int(optimizer_gradient_accumulation_steps))
        self.bucket_wave_apply_dataset_weights = bool(bucket_wave_apply_dataset_weights)
        self.bucket_wave_epoch_anchor_family = None if bucket_wave_epoch_anchor_family in (None, '') else str(bucket_wave_epoch_anchor_family).strip()
        self.bucket_wave_cyclic_families = frozenset((str(family).strip() for family in bucket_wave_cyclic_families or () if str(family).strip()))
        active_families = {getattr(ds.spec, "sampling_family", None) for ds in datasets}
        self.bucket_wave_cyclic_families &= active_families
        if self.bucket_wave_epoch_anchor_family not in active_families:
            self.bucket_wave_epoch_anchor_family = None
        if self.bucket_wave_epoch_anchor_family is not None and (not self.bucket_wave_apply_dataset_weights):
            raise ValueError('bucket_wave_epoch_anchor_family requires bucket_wave_apply_dataset_weights=true')
        if self.bucket_wave_cyclic_families and (not self.bucket_wave_apply_dataset_weights):
            raise ValueError('bucket_wave_cyclic_families requires bucket_wave_apply_dataset_weights=true')
        self.coverage_min_samples_per_dataset = max(0, int(coverage_min_samples_per_dataset))
        self.prefer_sequential_indices = self.training
        self._shared_epoch = mp.Value('q', 0, lock=True)
        max_action_dim = max((int(ds.action_dim or 0) for ds in datasets), default=0) or None
        max_state_dim = max((int(ds.state_dim or 0) for ds in datasets), default=0) or None
        for ds in datasets:
            ds.set_target_dims(action_target_dim=ds.action_target_dim or max_action_dim, state_target_dim=ds.state_target_dim or max_state_dim)
        eval_candidate_steps: list[tuple[int, int, int]] = []
        if eval_fixed_samples:
            eval_candidate_steps = self._build_fixed_eval_steps(eval_fixed_samples)
        elif eval_fixed_per_dataset:
            eval_candidate_steps = self._build_per_dataset_fixed_eval_steps(episode_pos=eval_fixed_episode_pos, frame_index=eval_fixed_frame_index)
        exclude_mode = str(eval_exclude_from_training or '').lower()
        self.excluded_eval_steps: set[tuple[int, int, int]] = set()
        self.excluded_eval_episodes: set[tuple[int, int]] = set()
        if self.training and eval_candidate_steps and (exclude_mode not in {'', 'none', 'false', 'off'}):
            if exclude_mode in {'episode', 'episodes', 'trajectory', 'trajectories'}:
                self.excluded_eval_episodes = {(ds_pos, episode_index) for ds_pos, episode_index, _ in eval_candidate_steps}
            elif exclude_mode in {'sample', 'samples', 'step', 'steps'}:
                self.excluded_eval_steps = set(eval_candidate_steps)
            else:
                raise ValueError(f'Unsupported eval_exclude_from_training={eval_exclude_from_training!r}; use episode/sample/none')
        if self.training:
            self.dataset_lengths = np.asarray([int(self._training_valid_start_counts(ds_pos).sum()) for ds_pos in range(len(datasets))], dtype=np.float64)
        else:
            self.dataset_lengths = np.asarray([len(ds) for ds in datasets], dtype=np.float64)
        weights = np.asarray(dataset_weights if dataset_weights is not None else [1.0] * len(datasets), dtype=np.float64)
        if weights.shape != self.dataset_lengths.shape:
            raise ValueError(f'dataset_weights shape {weights.shape} does not match datasets {self.dataset_lengths.shape}')
        weights = weights * (self.dataset_lengths > 0)
        if self.balance_dataset_weights:
            weights = weights * self.dataset_lengths
        if float(weights.sum()) <= 0:
            raise ValueError('dataset sampling weights must sum to a positive value')
        self.dataset_sampling_weights = weights / weights.sum()
        self.trajectory_sampling_weights: list[np.ndarray] = []
        default_epoch_length = int((self.dataset_lengths * self.dataset_sampling_weights).sum()) if self.training else int(self.dataset_lengths.sum())
        self.epoch_length = int(epoch_length) if epoch_length is not None else max(default_epoch_length, 1)
        self.coverage_dataset_quotas: np.ndarray | None = None
        self.coverage_cumulative_quotas: list[int] | None = None
        self.coverage_trajectory_cumulative_counts: list[list[int]] | None = None
        self.coverage_affine_params: list[tuple[int, int]] | None = None
        self.frame_block_total_samples = 0
        self.frame_block_dataset_cumulative_counts: np.ndarray | None = None
        self.frame_block_trajectory_cumulative_counts: list[np.ndarray] | None = None
        self._frame_block_affine_cache: dict[int, tuple[int, int]] = {}
        self.frame_block_world_size = max(1, int(os.environ.get('WORLD_SIZE', '1')))
        self.frame_block_group_width = min(4, self.frame_block_world_size)
        self.episode_block_total_samples = 0
        self.episode_block_dropped_samples = 0
        self.episode_block_candidate_blocks = 0
        self.episode_block_full_blocks = 0
        self.episode_block_residual_samples = 0
        self.episode_block_dataset_full_cumulative = np.empty(0, dtype=np.int64)
        self.episode_block_dataset_residual_cumulative = np.empty(0, dtype=np.int64)
        self.episode_block_trajectory_full_cumulative: list[np.ndarray] = []
        self.episode_block_trajectory_residual_cumulative: list[np.ndarray] = []
        self.episode_block_valid_counts: list[np.ndarray] = []
        self._episode_block_affine_cache: dict[int, tuple[int, int]] = {}
        self.decode_bucket_keys: list[tuple[Any, ...]] = []
        self.decode_bucket_dataset_positions: list[np.ndarray] = []
        self.decode_bucket_dataset_full_cumulative: list[np.ndarray] = []
        self.decode_bucket_dataset_residual_cumulative: list[np.ndarray] = []
        self.decode_bucket_batch_cumulative = np.empty(0, dtype=np.int64)
        self.decode_bucket_episode_full_blocks = np.empty(0, dtype=np.int64)
        self.decode_bucket_full_blocks = np.empty(0, dtype=np.int64)
        self.decode_bucket_usable_blocks = np.empty(0, dtype=np.int64)
        self.decode_bucket_candidate_batches = 0
        self.decode_bucket_total_samples = 0
        self.decode_bucket_dropped_samples = 0
        self._decode_bucket_batch_affine_cache: dict[int, tuple[int, int]] = {}
        self._decode_bucket_block_affine_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self.bucket_wave_world_size = max(1, int(os.environ.get('WORLD_SIZE', '1')))
        self.bucket_wave_rank = int(os.environ.get('RANK', '0'))
        self.bucket_wave_local_world_size = max(1, int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('PROC_PER_NODE', str(self.bucket_wave_world_size)))))
        if self.bucket_wave_world_size % self.bucket_wave_local_world_size:
            raise ValueError('WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE for node-source wave sampling')
        self.bucket_wave_num_nodes = self.bucket_wave_world_size // self.bucket_wave_local_world_size
        self.bucket_wave_available_samples = 0
        self.bucket_wave_total_samples = 0
        self.bucket_wave_dropped_samples = 0
        self.bucket_wave_dataset_positions: list[np.ndarray] = []
        self.bucket_wave_dataset_cumulative_counts: list[np.ndarray] = []
        self.bucket_wave_trajectory_cumulative_counts: list[np.ndarray] = []
        self.bucket_wave_dataset_capacities = np.zeros(len(self.datasets), dtype=np.int64)
        self.bucket_wave_dataset_quotas = np.zeros(len(self.datasets), dtype=np.int64)
        self.bucket_wave_dataset_is_cyclic = np.zeros(len(self.datasets), dtype=np.bool_)
        self.bucket_wave_family_quotas: dict[str, int] = {}
        self.bucket_wave_scheduled_family_samples: dict[str, int] = {}
        self.bucket_wave_source_quotas: dict[str, int] = {}
        self.bucket_wave_scheduled_source_samples: dict[str, int] = {}
        self.bucket_wave_ids = np.empty((0, self.bucket_wave_num_nodes), dtype=np.int32)
        self.bucket_wave_offsets = np.empty((0, self.bucket_wave_num_nodes), dtype=np.int64)
        self.bucket_wave_num_steps = np.empty(0, dtype=np.int32)
        self.bucket_wave_cumulative_steps = np.empty(0, dtype=np.int64)
        self.optimizer_stratified_steps = 0
        self.optimizer_stratified_blocks_per_rank_batch = 0
        self.optimizer_stratified_blocks_per_step = 0
        self.optimizer_stratified_minor_family: str | None = None
        self.optimizer_stratified_major_family: str | None = None
        self.optimizer_stratified_family_names: tuple[str, ...] = ()
        self.optimizer_stratified_minor_families: tuple[str, ...] = ()
        self.optimizer_stratified_family_blocks: dict[str, int] = {}
        self.optimizer_stratified_stream_sources: list[str] = []
        self.optimizer_stratified_stream_families: list[str] = []
        self.optimizer_stratified_stream_cost_keys: list[tuple[Any, ...]] = []
        self.optimizer_stratified_stream_full_blocks = np.empty(0, dtype=np.int64)
        self.optimizer_stratified_stream_scheduled_blocks = np.empty(0, dtype=np.int64)
        self.optimizer_stratified_stream_episode_full_blocks = np.empty(0, dtype=np.int64)
        self.optimizer_stratified_stream_dataset_full_cumulative: list[np.ndarray] = []
        self.optimizer_stratified_stream_dataset_residual_cumulative: list[np.ndarray] = []
        self.optimizer_stratified_family_stream_ids: dict[str, np.ndarray] = {}
        self.optimizer_stratified_family_stream_cumulative: dict[str, np.ndarray] = {}
        self.optimizer_stratified_stream_usable_cumulative = np.empty(0, dtype=np.int64)
        self._optimizer_stratified_family_affine_cache: dict[tuple[int, str], tuple[int, int]] = {}
        self._optimizer_stratified_stream_affine_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self._optimizer_stratified_lane_affine_cache: dict[tuple[int, int, int], tuple[int, int]] = {}
        if self.training:
            self._init_bucket_wave_sampling(epoch_length)
        self.cumulative_lengths = np.cumsum(self.dataset_lengths.astype(np.int64)).tolist()
        self.eval_steps: list[tuple[int, int, int]] | None = None
        if not self.training and eval_candidate_steps:
            self.eval_steps = eval_candidate_steps
            if not self.eval_steps:
                raise ValueError('fixed eval subset is empty')
        elif not self.training and (eval_episodes_per_dataset is not None or eval_frames_per_episode is not None or eval_segments_per_source is not None or (eval_total_samples is not None) or (str(eval_scope or 'dataset').lower() != 'dataset')):
            self.eval_steps = self._build_eval_steps(episodes_per_dataset=eval_episodes_per_dataset, frames_per_episode=eval_frames_per_episode, frame_policy=eval_frame_policy, scope=eval_scope, segments_per_source=eval_segments_per_source, total_samples=eval_total_samples)
            if not self.eval_steps:
                raise ValueError('eval subset is empty; check episode/frame limits and allow_padding_at_end')

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        with self._shared_epoch.get_lock():
            return int(self._shared_epoch.value)

    @staticmethod
    def _allocate_coverage_quotas(weights: np.ndarray, capacities: np.ndarray, total: int, minimum_per_positive: int=0) -> np.ndarray:
        """Largest-remainder allocation with hard per-dataset capacities."""
        capacities = capacities.astype(np.int64, copy=True)
        quotas = np.zeros_like(capacities)
        minimum_per_positive = max(0, int(minimum_per_positive))
        if minimum_per_positive:
            positive = (capacities > 0) & (weights > 0)
            quotas[positive] = np.minimum(capacities[positive], minimum_per_positive)
        remaining = int(total) - int(quotas.sum())
        if remaining < 0 or remaining > int(capacities.sum()):
            raise ValueError(f'coverage epoch_length={total} cannot satisfy minimum_per_positive={minimum_per_positive} within total available capacity={int(capacities.sum())}')
        while remaining > 0:
            room = capacities - quotas
            active = (room > 0) & (weights > 0)
            if not bool(active.any()):
                raise ValueError(f'could not allocate {remaining} coverage samples within available capacities')
            active_weights = np.where(active, weights, 0.0)
            raw = active_weights / float(active_weights.sum()) * float(remaining)
            base = np.minimum(np.floor(raw).astype(np.int64), room)
            added = int(base.sum())
            if added > 0:
                quotas += base
                remaining -= added
                continue
            fractions = raw - np.floor(raw)
            order = sorted(np.flatnonzero(active).tolist(), key=lambda idx: (-float(fractions[idx]), -float(weights[idx]), int(idx)))
            take = min(remaining, len(order))
            quotas[np.asarray(order[:take], dtype=np.int64)] += 1
            remaining -= take
        return quotas

    @classmethod
    def _allocate_family_quotas(cls, weights, capacities, total):
        """Keep largest-remainder family quotas, correcting capacity overflow only."""
        quotas = cls._allocate_weighted_quotas(weights, total)
        if not bool((quotas > capacities).any()):
            return quotas
        quotas = np.minimum(quotas, capacities)
        remaining = int(total) - int(quotas.sum())
        raw = weights / float(weights.sum()) * float(total)
        while remaining > 0:
            room = capacities - quotas
            candidates = np.flatnonzero(room > 0)
            if candidates.size == 0:
                raise ValueError("Family quotas exceed available complete-block capacity")
            residual = raw - quotas
            order = sorted(candidates.tolist(), key=lambda i: (-float(residual[i]), -float(weights[i]), i))
            for i in order:
                take = min(int(room[i]), remaining)
                quotas[i] += take
                remaining -= take
                if remaining == 0:
                    break
        return quotas

    @staticmethod
    def _allocate_weighted_quotas(weights: np.ndarray, total: int) -> np.ndarray:
        """Largest-remainder allocation without per-dataset capacity caps."""
        weights = np.asarray(weights, dtype=np.float64)
        total = int(total)
        if total < 0:
            raise ValueError(f'weighted quota total must be non-negative, got {total}')
        if np.any(weights < 0):
            raise ValueError('weighted quota weights must be non-negative')
        quotas = np.zeros(weights.shape, dtype=np.int64)
        if total == 0:
            return quotas
        weight_total = float(weights.sum())
        if not math.isfinite(weight_total) or weight_total <= 0:
            raise ValueError('positive weighted quota total requires positive finite weights')
        raw = weights / weight_total * float(total)
        quotas = np.floor(raw).astype(np.int64)
        remainder = total - int(quotas.sum(dtype=np.int64))
        if remainder:
            fractions = raw - quotas
            order = sorted(np.flatnonzero(weights > 0).tolist(), key=lambda idx: (-float(fractions[idx]), -float(weights[idx]), int(idx)))
            quotas[np.asarray(order[:remainder], dtype=np.int64)] += 1
        return quotas

    def _allocate_anchored_bucket_wave_quotas(self, capacities: np.ndarray, epoch_length: int | None) -> np.ndarray:
        """Anchor an epoch on one full family and cycle configured families.

        The anchor family receives its complete valid-window capacity as a
        no-replacement quota. The other family totals are derived from the
        configured sampling weights. A cyclic family may receive a quota
        larger than its finite capacity; all other families retain hard
        no-replacement capacity limits. Downstream node-wave alignment may
        discard terminal slots exactly as in the existing Real sampler.
        """
        anchor = self.bucket_wave_epoch_anchor_family
        if anchor is None:
            raise RuntimeError('anchored bucket-wave allocation lacks an anchor')
        families = np.asarray([getattr(dataset.spec, 'sampling_family', None) for dataset in self.datasets], dtype=object)
        known_families = {str(family) for family, capacity in zip(families.tolist(), capacities.tolist()) if family is not None and int(capacity) > 0}
        unknown_cyclic = sorted(self.bucket_wave_cyclic_families - known_families)
        if unknown_cyclic:
            raise ValueError(f'bucket_wave_cyclic_families contains unknown active families: {unknown_cyclic}')
        if anchor in self.bucket_wave_cyclic_families:
            raise ValueError(f'bucket-wave anchor family {anchor!r} cannot also be cyclic')
        anchor_mask = (families == anchor) & (capacities > 0)
        if not bool(anchor_mask.any()):
            raise ValueError(f'bucket-wave anchor family {anchor!r} has no valid windows')
        active = capacities > 0
        missing_family = [self.datasets[idx].name for idx in np.flatnonzero(active) if families[int(idx)] is None]
        if missing_family:
            raise ValueError('anchored bucket-wave datasets are missing sampling_family: ' + ', '.join(missing_family[:20]))
        weights = self.dataset_sampling_weights
        anchor_weight = float(weights[anchor_mask].sum())
        if anchor_weight <= 0:
            raise ValueError(f'bucket-wave anchor family {anchor!r} has zero sampling weight')
        anchor_capacity = int(capacities[anchor_mask].sum(dtype=np.int64))
        expected_anchor_distribution = capacities[anchor_mask].astype(np.float64) / float(anchor_capacity)
        actual_anchor_distribution = weights[anchor_mask] / anchor_weight
        if not np.allclose(actual_anchor_distribution, expected_anchor_distribution, rtol=0.0, atol=1e-12):
            raise ValueError(f'bucket-wave anchor family {anchor!r} must be weighted by valid-start capacity to preserve its existing full no-replacement traversal')
        family_weights = {family: float(weights[families == family].sum()) for family in sorted(known_families)}
        other_families = [family for family in sorted(family_weights) if family != anchor and family_weights[family] > 0]
        other_weight = float(sum((family_weights[family] for family in other_families)))
        other_total = int(math.floor(float(anchor_capacity) * other_weight / anchor_weight + 0.5))
        other_family_quotas = self._allocate_weighted_quotas(np.asarray([family_weights[family] for family in other_families], dtype=np.float64), other_total)
        quotas = np.zeros_like(capacities)
        quotas[anchor_mask] = capacities[anchor_mask]
        for family, family_total in zip(other_families, other_family_quotas.tolist()):
            family_mask = (families == family) & active
            family_indices = np.flatnonzero(family_mask)
            if family in self.bucket_wave_cyclic_families:
                family_quotas = self._allocate_weighted_quotas(weights[family_indices], int(family_total))
            else:
                family_quotas = self._allocate_coverage_quotas(weights[family_indices], capacities[family_indices], int(family_total))
            quotas[family_indices] = family_quotas
        anchored_total = int(quotas.sum(dtype=np.int64))
        if epoch_length is not None and (not 0 < int(epoch_length) <= anchored_total):
            raise ValueError(f'bucket-wave epoch_length must be a positive aligned prefix of the quota determined by anchor family {anchor!r}: configured={int(epoch_length)}, anchored={anchored_total}')
        self.bucket_wave_dataset_is_cyclic = np.asarray([str(family) in self.bucket_wave_cyclic_families for family in families.tolist()], dtype=np.bool_)
        return quotas

    @staticmethod
    def _coprime_affine_multiplier(size: int, seed_value: int) -> int:
        size = int(size)
        if size <= 1:
            return 1
        candidate = int(seed_value % size) or 1
        for _ in range(size):
            if math.gcd(candidate, size) == 1:
                return candidate
            candidate += 1
            if candidate >= size:
                candidate = 1
        raise RuntimeError(f'failed to find affine multiplier coprime to {size}')

    def _episode_tiled_rank_in_trajectory(self, *, ds_pos: int, traj_pos: int, episode_block: int, block_offset: int, coverage_cycle: int) -> int:
        """Map a block slot to a dispersed, unique rank in an episode.

        Valid starts are partitioned into fixed-size local tiles.  A block
        takes one start from each of ``training_block_size`` temporal strata
        inside its tile.  The same affine permutation is applied to every
        stratum, so full 64-window tiles have an exact 16/16/16 valid-rank
        spacing for block size four.  The permutation changes each coverage
        cycle while preserving a bijection: every full-block valid start is
        visited exactly once.
        """
        block_size = self.training_block_size
        tile_size = self.episode_block_tile_size
        blocks_per_full_tile = tile_size // block_size
        tile_index, block_in_tile = divmod(int(episode_block), blocks_per_full_tile)
        count = int(self.episode_block_valid_counts[ds_pos][traj_pos])
        tile_start = tile_index * tile_size
        tile_count = min(tile_size, count - tile_start)
        blocks_in_tile = tile_count // block_size
        if not 0 <= block_in_tile < blocks_in_tile:
            raise IndexError(f'episode tiled block is out of range: dataset={ds_pos} trajectory={traj_pos} block={episode_block} tile={tile_index} block_in_tile={block_in_tile} blocks_in_tile={blocks_in_tile} count={count}')
        stratum = int(block_offset)
        multiplier = self._coprime_affine_multiplier(blocks_in_tile, safe_hash((self.seed, int(coverage_cycle), int(ds_pos), int(traj_pos), int(tile_index), 'episode_tile_multiplier')))
        offset = int(safe_hash((self.seed, int(coverage_cycle), int(ds_pos), int(traj_pos), int(tile_index), 'episode_tile_offset')) % blocks_in_tile)
        rank_in_stratum = (multiplier * block_in_tile + offset) % blocks_in_tile
        return tile_start + stratum * blocks_in_tile + rank_in_stratum

    @staticmethod
    def _video_codec_family(codec: Any) -> str:
        name = str(codec or 'unknown').lower()
        if name in {'av1', 'av01'}:
            return 'av1'
        if name in {'h264', 'avc', 'avc1', 'libx264'}:
            return 'h264'
        if name in {'h265', 'hevc', 'hev1'}:
            return 'hevc'
        return name

    def _dataset_decode_bucket(self, ds: Any) -> tuple[Any, ...]:
        codecs: list[str] = []
        total_pixels = 0
        for video_key in ds.video_keys:
            feature = ds.info.get('features', {}).get(video_key, {})
            video_info = feature.get('info', {})
            shape = feature.get('shape', ())
            height = int(video_info.get('video.height', shape[0] if len(shape) >= 2 else 1))
            width = int(video_info.get('video.width', shape[1] if len(shape) >= 2 else 1))
            total_pixels += max(1, height * width)
            codecs.append(self._video_codec_family(video_info.get('video.codec')))
        pixel_bin = int(round(2.0 * math.log2(max(total_pixels, 1) / (640 * 480))))
        storage = 'local'
        return (len(ds.video_keys), storage, tuple(sorted(codecs)), pixel_bin)

    def _init_bucket_wave_sampling(self, epoch_length: int | None) -> None:
        counts_by_dataset = [self._training_valid_start_counts(ds_pos).astype(np.int64) for ds_pos in range(len(self.datasets))]
        capacities = np.asarray([int(counts.sum(dtype=np.int64)) for counts in counts_by_dataset], dtype=np.int64)
        self.bucket_wave_dataset_capacities = capacities
        quotas = capacities.copy()
        if self.bucket_wave_apply_dataset_weights:
            if self.bucket_wave_epoch_anchor_family is not None:
                quotas = self._allocate_anchored_bucket_wave_quotas(capacities, epoch_length)
            else:
                positive = (capacities > 0) & (self.dataset_sampling_weights > 0)
                if not bool(positive.any()):
                    raise ValueError('weighted bucket-wave sampling has no positive capacity')
                proportional_scale = float(np.min(capacities[positive] / self.dataset_sampling_weights[positive]))
                max_proportional_total = int(math.floor(proportional_scale * float(self.dataset_sampling_weights[positive].sum())))
                requested_total = max_proportional_total if epoch_length is None else int(epoch_length)
                if requested_total > max_proportional_total:
                    raise ValueError(f'weighted bucket-wave epoch_length exceeds the largest no-replacement epoch that preserves configured dataset ratios: requested={requested_total}, maximum={max_proportional_total}, capacities={capacities.tolist()}')
                quotas = self._allocate_coverage_quotas(self.dataset_sampling_weights, capacities, requested_total)
        self.bucket_wave_dataset_quotas = quotas
        family_quotas: dict[str, int] = {}
        for ds_pos, quota in enumerate(quotas.tolist()):
            family = getattr(self.datasets[ds_pos].spec, 'sampling_family', None)
            if family is None:
                continue
            family_quotas[family] = family_quotas.get(family, 0) + int(quota)
        self.bucket_wave_family_quotas = family_quotas
        source_quotas: dict[str, int] = {}
        for ds_pos, quota in enumerate(quotas.tolist()):
            source = str(self.datasets[ds_pos].spec.source_name or self.datasets[ds_pos].spec.group_id)
            source_quotas[source] = source_quotas.get(source, 0) + int(quota)
        self.bucket_wave_source_quotas = source_quotas
        stream_groups: dict[tuple[tuple[Any, ...], str], list[int]] = {}
        for ds_pos, ds in enumerate(self.datasets):
            if int(quotas[ds_pos]) <= 0:
                continue
            cost_key = self._dataset_decode_bucket(ds)
            source = str(ds.spec.source_name or ds.spec.group_id)
            stream_groups.setdefault((cost_key, source), []).append(ds_pos)
        self.bucket_wave_trajectory_cumulative_counts = [np.cumsum(counts).astype(np.int64) for counts in counts_by_dataset]
        stream_cost_keys: list[tuple[Any, ...]] = []
        stream_sources: list[str] = []
        stream_families: list[str | None] = []
        stream_totals: list[int] = []
        for cost_key, source in sorted(stream_groups, key=repr):
            dataset_positions = np.asarray(stream_groups[cost_key, source], dtype=np.int32)
            dataset_counts = np.asarray([int(quotas[int(ds_pos)]) for ds_pos in dataset_positions], dtype=np.int64)
            self.bucket_wave_dataset_positions.append(dataset_positions)
            self.bucket_wave_dataset_cumulative_counts.append(np.cumsum(dataset_counts).astype(np.int64))
            stream_cost_keys.append(cost_key)
            stream_sources.append(source)
            families = {getattr(self.datasets[int(ds_pos)].spec, 'sampling_family', None) for ds_pos in dataset_positions}
            if len(families) != 1:
                raise ValueError(f'bucket-wave source stream mixes sampling families: source={source!r}, families={sorted(map(str, families))}')
            stream_families.append(next(iter(families)))
            stream_totals.append(int(dataset_counts.sum()))
        self._init_optimizer_stratified_sampling(epoch_length=epoch_length, counts_by_dataset=counts_by_dataset, stream_cost_keys=stream_cost_keys, stream_sources=stream_sources, stream_families=stream_families, stream_totals=stream_totals)
        return

    def _init_optimizer_stratified_sampling(self, *, epoch_length: int | None, counts_by_dataset: list[np.ndarray], stream_cost_keys: list[tuple[Any, ...]], stream_sources: list[str], stream_families: list[str | None], stream_totals: list[int]) -> None:
        """Build a block-randomized, optimizer-step-stratified traversal.

        The four-sample temporal block is the scheduling atom, matching the
        pretraining sampler.  A physical rank batch may therefore mix
        four independently permuted sources.  The LeRobot collator pads the VLM
        view axis, while every model-facing video tensor already has a fixed
        canvas, so cross-source blocks remain shape-safe.
        """
        if any((family is None for family in stream_families)):
            raise ValueError('optimizer-stratified streams require sampling_family on every active dataset')
        family_names = sorted({str(family) for family in stream_families})
        block_size = self.training_block_size
        blocks_per_rank_batch = self.training_batch_size_per_rank // block_size
        blocks_per_step = self.bucket_wave_world_size * self.optimizer_gradient_accumulation_steps * blocks_per_rank_batch
        optimizer_batch_size = blocks_per_step * block_size
        self.episode_block_valid_counts = [counts.astype(np.int64, copy=True) for counts in counts_by_dataset]
        self.episode_block_trajectory_full_cumulative = [np.cumsum(counts // block_size, dtype=np.int64) for counts in counts_by_dataset]
        self.episode_block_trajectory_residual_cumulative = [np.cumsum(counts % block_size, dtype=np.int64) for counts in counts_by_dataset]
        stream_episode_full_blocks: list[int] = []
        stream_dataset_full_cumulative: list[np.ndarray] = []
        stream_dataset_residual_cumulative: list[np.ndarray] = []
        full_stream_blocks: list[int] = []
        for stream_id, dataset_positions in enumerate(self.bucket_wave_dataset_positions):
            dataset_full_blocks = np.asarray([int((counts_by_dataset[int(ds_pos)] // block_size).sum(dtype=np.int64)) for ds_pos in dataset_positions], dtype=np.int64)
            dataset_residual_samples = np.asarray([int((counts_by_dataset[int(ds_pos)] % block_size).sum(dtype=np.int64)) for ds_pos in dataset_positions], dtype=np.int64)
            episode_full_blocks = int(dataset_full_blocks.sum(dtype=np.int64))
            stream_dataset_full_cumulative.append(np.cumsum(dataset_full_blocks, dtype=np.int64))
            stream_dataset_residual_cumulative.append(np.cumsum(dataset_residual_samples, dtype=np.int64))
            stream_episode_full_blocks.append(episode_full_blocks)
            family = str(stream_families[stream_id])
            if family in self.bucket_wave_cyclic_families:
                total_blocks = int(stream_totals[stream_id]) // block_size
            else:
                total_blocks = episode_full_blocks + int(dataset_residual_samples.sum(dtype=np.int64)) // block_size
            full_stream_blocks.append(total_blocks)
        full_stream_blocks_array = np.asarray(full_stream_blocks, dtype=np.int64)
        if not bool((full_stream_blocks_array > 0).any()):
            raise ValueError('optimizer-stratified sampling found no complete block')
        family_full_blocks = np.asarray([int(full_stream_blocks_array[np.asarray([str(family) == family_name for family in stream_families], dtype=np.bool_)].sum(dtype=np.int64)) for family_name in family_names], dtype=np.int64)
        family_weights = np.asarray([self.bucket_wave_family_quotas[name] for name in family_names], dtype=np.float64)
        requested_samples = int(sum(stream_totals)) if epoch_length is None else int(epoch_length)
        optimizer_steps = requested_samples // optimizer_batch_size
        if optimizer_steps <= 0:
            raise ValueError('optimizer-stratified epoch does not contain one complete optimizer step')
        total_blocks = optimizer_steps * blocks_per_step
        scheduled_family_blocks_array = self._allocate_family_quotas(family_weights, family_full_blocks, total_blocks)
        if bool((scheduled_family_blocks_array > family_full_blocks).any()):
            raise ValueError(f'optimizer-stratified epoch cannot satisfy the configured family weights within complete block capacities: steps={optimizer_steps}, family_names={family_names}, family_capacities={family_full_blocks.tolist()}')
        scheduled_family_blocks = {name: int(count) for name, count in zip(family_names, scheduled_family_blocks_array.tolist())}
        if any((count <= 0 for count in scheduled_family_blocks.values())):
            raise ValueError(f'optimizer-stratified sampling requires every family in the epoch, got {scheduled_family_blocks}')
        minor_family = min(family_names, key=lambda name: (self.bucket_wave_family_quotas[name], name))
        major_family = max(family_names, key=lambda name: (self.bucket_wave_family_quotas[name], name))
        minor_families = tuple(sorted((name for name in family_names if name != major_family), key=lambda name: (self.bucket_wave_family_quotas[name], name)))
        scheduled_stream_blocks = np.zeros_like(full_stream_blocks_array)
        family_stream_ids: dict[str, np.ndarray] = {}
        family_stream_cumulative: dict[str, np.ndarray] = {}
        for family_name in family_names:
            stream_ids = np.asarray([stream_id for stream_id, family in enumerate(stream_families) if str(family) == family_name and int(full_stream_blocks_array[stream_id]) > 0], dtype=np.int32)
            capacities = full_stream_blocks_array[stream_ids]
            stream_quotas = self._allocate_coverage_quotas(capacities.astype(np.float64), capacities, scheduled_family_blocks[family_name])
            scheduled_stream_blocks[stream_ids] = stream_quotas
            positive = stream_quotas > 0
            stream_ids = stream_ids[positive]
            stream_quotas = stream_quotas[positive]
            family_stream_ids[family_name] = stream_ids
            family_stream_cumulative[family_name] = np.cumsum(stream_quotas, dtype=np.int64)
        scheduled_source_samples: dict[str, int] = {}
        for stream_id, num_blocks in enumerate(scheduled_stream_blocks.tolist()):
            if num_blocks <= 0:
                continue
            source = stream_sources[stream_id]
            scheduled_source_samples[source] = scheduled_source_samples.get(source, 0) + int(num_blocks) * block_size
        self.optimizer_stratified_steps = optimizer_steps
        self.optimizer_stratified_blocks_per_rank_batch = blocks_per_rank_batch
        self.optimizer_stratified_blocks_per_step = blocks_per_step
        self.optimizer_stratified_minor_family = minor_family
        self.optimizer_stratified_major_family = major_family
        self.optimizer_stratified_family_names = tuple(family_names)
        self.optimizer_stratified_minor_families = minor_families
        self.optimizer_stratified_family_blocks = scheduled_family_blocks
        self.optimizer_stratified_stream_sources = list(stream_sources)
        self.optimizer_stratified_stream_families = [str(family) for family in stream_families]
        self.optimizer_stratified_stream_cost_keys = list(stream_cost_keys)
        self.optimizer_stratified_stream_full_blocks = full_stream_blocks_array
        self.optimizer_stratified_stream_scheduled_blocks = scheduled_stream_blocks
        self.optimizer_stratified_stream_episode_full_blocks = np.asarray(stream_episode_full_blocks, dtype=np.int64)
        self.optimizer_stratified_stream_dataset_full_cumulative = stream_dataset_full_cumulative
        self.optimizer_stratified_stream_dataset_residual_cumulative = stream_dataset_residual_cumulative
        self.optimizer_stratified_family_stream_ids = family_stream_ids
        self.optimizer_stratified_family_stream_cumulative = family_stream_cumulative
        self.optimizer_stratified_stream_usable_cumulative = np.cumsum(full_stream_blocks_array * block_size, dtype=np.int64)
        self.bucket_wave_available_samples = int(full_stream_blocks_array.sum(dtype=np.int64) * block_size)
        self.bucket_wave_total_samples = optimizer_steps * optimizer_batch_size
        self.bucket_wave_scheduled_family_samples = {family: blocks * block_size for family, blocks in scheduled_family_blocks.items()}
        self.bucket_wave_scheduled_source_samples = scheduled_source_samples
        self.bucket_wave_dropped_samples = int(sum(stream_totals)) - self.bucket_wave_total_samples
        self.epoch_length = self.bucket_wave_total_samples

    def _optimizer_stratified_family_affine_params(self, family: str) -> tuple[int, int]:
        key = (self.epoch, str(family))
        cached = self._optimizer_stratified_family_affine_cache.get(key)
        if cached is not None:
            return cached
        total = self.optimizer_stratified_family_blocks[str(family)]
        multiplier = self._coprime_affine_multiplier(total, safe_hash((self.seed, self.epoch, str(family), 'optimizer_stratified_family_multiplier')))
        offset = int(safe_hash((self.seed, self.epoch, str(family), 'optimizer_stratified_family_offset')) % total)
        if len(self._optimizer_stratified_family_affine_cache) >= 8:
            self._optimizer_stratified_family_affine_cache.clear()
        self._optimizer_stratified_family_affine_cache[key] = (multiplier, offset)
        return (multiplier, offset)

    def _optimizer_stratified_stream_affine_params(self, stream_id: int) -> tuple[int, int]:
        key = (self.epoch, int(stream_id))
        cached = self._optimizer_stratified_stream_affine_cache.get(key)
        if cached is not None:
            return cached
        total = int(self.optimizer_stratified_stream_full_blocks[stream_id])
        multiplier = self._coprime_affine_multiplier(total, safe_hash((self.seed, self.epoch, self.optimizer_stratified_stream_sources[stream_id], self.optimizer_stratified_stream_cost_keys[stream_id], 'optimizer_stratified_stream_multiplier')))
        offset = int(safe_hash((self.seed, self.epoch, self.optimizer_stratified_stream_sources[stream_id], self.optimizer_stratified_stream_cost_keys[stream_id], 'optimizer_stratified_stream_offset')) % total)
        if len(self._optimizer_stratified_stream_affine_cache) >= 256:
            self._optimizer_stratified_stream_affine_cache.clear()
        self._optimizer_stratified_stream_affine_cache[key] = (multiplier, offset)
        return (multiplier, offset)

    def _optimizer_stratified_lane_affine_params(self, optimizer_step: int, micro_batch: int) -> tuple[int, int]:
        key = (self.epoch, int(optimizer_step), int(micro_batch))
        cached = self._optimizer_stratified_lane_affine_cache.get(key)
        if cached is not None:
            return cached
        size = self.bucket_wave_world_size * self.optimizer_stratified_blocks_per_rank_batch
        multiplier = self._coprime_affine_multiplier(size, safe_hash((self.seed, self.epoch, int(optimizer_step), int(micro_batch), 'optimizer_stratified_lane_multiplier')))
        offset = int(safe_hash((self.seed, self.epoch, int(optimizer_step), int(micro_batch), 'optimizer_stratified_lane_offset')) % size)
        if len(self._optimizer_stratified_lane_affine_cache) >= 64:
            self._optimizer_stratified_lane_affine_cache.clear()
        self._optimizer_stratified_lane_affine_cache[key] = (multiplier, offset)
        return (multiplier, offset)

    def _optimizer_stratified_minor_micro_count(self, *, optimizer_step: int, micro_batch: int, step_count: int) -> int:
        accum = self.optimizer_gradient_accumulation_steps
        base, remainder = divmod(int(step_count), accum)
        extra_start = int(optimizer_step) % accum
        extra = int((int(micro_batch) - extra_start) % accum < remainder)
        return base + extra

    def _optimizer_stratified_multifamily_assignment(self, *, optimizer_step: int, micro_batch: int, permuted_block_slot: int, blocks_per_micro_batch: int) -> tuple[str, int]:
        """Assign one permuted block slot while preserving all family quotas.

        The existing two-family path remains byte-for-byte unchanged below.
        For three or more families, every non-major family follows the same
        cumulative-floor schedule previously used by the single minor family;
        the largest family receives the exact residual in each microbatch.
        """
        major_family = self.optimizer_stratified_major_family
        minor_families = self.optimizer_stratified_minor_families
        if major_family is None or not minor_families:
            raise RuntimeError('optimizer-stratified families are uninitialized')
        steps = self.optimizer_stratified_steps
        blocks_before_step = int(optimizer_step) * self.optimizer_stratified_blocks_per_step
        minor_before_step_total = 0
        minor_before_micro_total = 0
        slot_start = 0
        for family in minor_families:
            family_total = self.optimizer_stratified_family_blocks[family]
            before_step = int(optimizer_step) * family_total // steps
            after_step = (int(optimizer_step) + 1) * family_total // steps
            step_count = after_step - before_step
            before_micro = sum((self._optimizer_stratified_minor_micro_count(optimizer_step=optimizer_step, micro_batch=previous_micro, step_count=step_count) for previous_micro in range(micro_batch)))
            micro_count = self._optimizer_stratified_minor_micro_count(optimizer_step=optimizer_step, micro_batch=micro_batch, step_count=step_count)
            slot_end = slot_start + micro_count
            if slot_start <= permuted_block_slot < slot_end:
                return (family, before_step + before_micro + permuted_block_slot - slot_start)
            slot_start = slot_end
            minor_before_step_total += before_step
            minor_before_micro_total += before_micro
        if slot_start > blocks_per_micro_batch:
            raise RuntimeError(f'optimizer-stratified minor-family allocation exceeds one microbatch: allocated={slot_start}, capacity={blocks_per_micro_batch}')
        major_before_step = blocks_before_step - minor_before_step_total
        major_before_micro = int(micro_batch) * blocks_per_micro_batch - minor_before_micro_total
        return (major_family, major_before_step + major_before_micro + int(permuted_block_slot) - slot_start)

    def _optimizer_stratified_stream_block_for_index(self, index: int) -> tuple[int, int, int]:
        retry_base = self._OPTIMIZER_STRATIFIED_RETRY_BASE
        if int(index) >= retry_base:
            flat_rank = int(index) - retry_base
            if flat_rank < 0 or flat_rank >= int(self.optimizer_stratified_stream_usable_cumulative[-1]):
                raise IndexError(f'optimizer-stratified retry rank is out of range: {index}')
            stream_id = int(np.searchsorted(self.optimizer_stratified_stream_usable_cumulative, flat_rank, side='right'))
            stream_start = 0 if stream_id == 0 else int(self.optimizer_stratified_stream_usable_cumulative[stream_id - 1])
            stream_block, block_offset = divmod(flat_rank - stream_start, self.training_block_size)
            return (stream_id, stream_block, block_offset)
        if not 0 <= int(index) < self.epoch_length:
            raise IndexError(f'optimizer-stratified sample index {index} is out of range 0..{self.epoch_length - 1}')
        rank_batch_id, offset_in_rank_batch = divmod(int(index), self.training_batch_size_per_rank)
        block_in_rank_batch, block_offset = divmod(offset_in_rank_batch, self.training_block_size)
        global_micro_batch, lane = divmod(rank_batch_id, self.bucket_wave_world_size)
        optimizer_step, micro_batch = divmod(global_micro_batch, self.optimizer_gradient_accumulation_steps)
        lane_multiplier, lane_offset = self._optimizer_stratified_lane_affine_params(optimizer_step, micro_batch)
        block_slot = lane * self.optimizer_stratified_blocks_per_rank_batch + block_in_rank_batch
        blocks_per_micro_batch = self.bucket_wave_world_size * self.optimizer_stratified_blocks_per_rank_batch
        permuted_block_slot = (lane_multiplier * block_slot + lane_offset) % blocks_per_micro_batch
        if len(self.optimizer_stratified_family_names) == 1:
            family = self.optimizer_stratified_family_names[0]
            family_occurrence = (optimizer_step * self.optimizer_stratified_blocks_per_step
                                 + micro_batch * blocks_per_micro_batch + permuted_block_slot)
        elif len(self.optimizer_stratified_family_names) > 2:
            family, family_occurrence = self._optimizer_stratified_multifamily_assignment(optimizer_step=optimizer_step, micro_batch=micro_batch, permuted_block_slot=permuted_block_slot, blocks_per_micro_batch=blocks_per_micro_batch)
        else:
            minor_family = self.optimizer_stratified_minor_family
            major_family = self.optimizer_stratified_major_family
            if minor_family is None or major_family is None:
                raise RuntimeError('optimizer-stratified families are uninitialized')
            minor_total = self.optimizer_stratified_family_blocks[minor_family]
            steps = self.optimizer_stratified_steps
            minor_before_step = optimizer_step * minor_total // steps
            minor_after_step = (optimizer_step + 1) * minor_total // steps
            minor_step_count = minor_after_step - minor_before_step
            minor_before_micro = sum((self._optimizer_stratified_minor_micro_count(optimizer_step=optimizer_step, micro_batch=previous_micro, step_count=minor_step_count) for previous_micro in range(micro_batch)))
            minor_micro_count = self._optimizer_stratified_minor_micro_count(optimizer_step=optimizer_step, micro_batch=micro_batch, step_count=minor_step_count)
            blocks_before_step = optimizer_step * self.optimizer_stratified_blocks_per_step
            if permuted_block_slot < minor_micro_count:
                family = minor_family
                family_occurrence = minor_before_step + minor_before_micro + permuted_block_slot
            else:
                family = major_family
                major_before_step = blocks_before_step - minor_before_step
                major_before_micro = micro_batch * blocks_per_micro_batch - minor_before_micro
                family_occurrence = major_before_step + major_before_micro + permuted_block_slot - minor_micro_count
        family_total = self.optimizer_stratified_family_blocks[family]
        family_multiplier, family_offset = self._optimizer_stratified_family_affine_params(family)
        family_slot = (family_multiplier * family_occurrence + family_offset) % family_total
        cumulative = self.optimizer_stratified_family_stream_cumulative[family]
        stream_pos = int(np.searchsorted(cumulative, family_slot, side='right'))
        stream_slot_start = 0 if stream_pos == 0 else int(cumulative[stream_pos - 1])
        stream_id = int(self.optimizer_stratified_family_stream_ids[family][stream_pos])
        logical_stream_block = family_slot - stream_slot_start
        stream_multiplier, stream_offset = self._optimizer_stratified_stream_affine_params(stream_id)
        stream_block = (stream_multiplier * logical_stream_block + stream_offset) % int(self.optimizer_stratified_stream_full_blocks[stream_id])
        return (stream_id, stream_block, block_offset)

    def _sample_optimizer_stratified_training_step(self, index: int) -> tuple[int, int, int]:
        stream_id, stream_block, block_offset = self._optimizer_stratified_stream_block_for_index(index)
        return self._sample_optimizer_stratified_stream_block(stream_id, stream_block, block_offset)

    def optimizer_stratified_replacement_index(self, index: int, attempt: int) -> int:
        """Return a retry from the same family, source, and decode bucket."""
        stream_id, stream_block, block_offset = self._optimizer_stratified_stream_block_for_index(int(index))
        full_blocks = int(self.optimizer_stratified_stream_full_blocks[stream_id])
        if full_blocks > 1:
            shift = 1 + int(safe_hash((self.seed, self.epoch, int(index), int(attempt), 'optimizer_stratified_retry')) % (full_blocks - 1))
            target_block = (stream_block + shift) % full_blocks
            target_rank = target_block * self.training_block_size + block_offset
        else:
            target_rank = (block_offset + max(1, int(attempt))) % self.training_block_size
        stream_start = 0 if stream_id == 0 else int(self.optimizer_stratified_stream_usable_cumulative[stream_id - 1])
        return int(self._OPTIMIZER_STRATIFIED_RETRY_BASE + stream_start + target_rank)

    def _sample_optimizer_stratified_stream_block(self, stream_id: int, stream_block: int, block_offset: int) -> tuple[int, int, int]:
        if self.optimizer_stratified_stream_families[int(stream_id)] == "umi":
            return self._sample_umi_block(stream_id=stream_id, stream_block=stream_block, block_offset=block_offset)
        family = self.optimizer_stratified_stream_families[stream_id]
        if family in self.bucket_wave_cyclic_families:
            stream_rank = int(stream_block) * self.training_block_size + int(block_offset)
            return self._sample_bucket_wave_stream_rank(stream_id, stream_rank)
        episode_full_blocks = int(self.optimizer_stratified_stream_episode_full_blocks[stream_id])
        dataset_positions = self.bucket_wave_dataset_positions[stream_id]
        if int(stream_block) < episode_full_blocks:
            dataset_cumulative = self.optimizer_stratified_stream_dataset_full_cumulative[stream_id]
            dataset_pos_in_stream = int(np.searchsorted(dataset_cumulative, stream_block, side='right'))
            dataset_block_start = 0 if dataset_pos_in_stream == 0 else int(dataset_cumulative[dataset_pos_in_stream - 1])
            ds_pos = int(dataset_positions[dataset_pos_in_stream])
            dataset_block = int(stream_block) - dataset_block_start
            trajectory_cumulative = self.episode_block_trajectory_full_cumulative[ds_pos]
            traj_pos = int(np.searchsorted(trajectory_cumulative, dataset_block, side='right'))
            trajectory_block_start = 0 if traj_pos == 0 else int(trajectory_cumulative[traj_pos - 1])
            episode_block = dataset_block - trajectory_block_start
            rank_in_trajectory = self._episode_tiled_rank_in_trajectory(ds_pos=ds_pos, traj_pos=traj_pos, episode_block=episode_block, block_offset=int(block_offset), coverage_cycle=self.epoch)
        else:
            residual_rank = (int(stream_block) - episode_full_blocks) * self.training_block_size + int(block_offset)
            dataset_cumulative = self.optimizer_stratified_stream_dataset_residual_cumulative[stream_id]
            dataset_pos_in_stream = int(np.searchsorted(dataset_cumulative, residual_rank, side='right'))
            dataset_residual_start = 0 if dataset_pos_in_stream == 0 else int(dataset_cumulative[dataset_pos_in_stream - 1])
            ds_pos = int(dataset_positions[dataset_pos_in_stream])
            dataset_residual_rank = residual_rank - dataset_residual_start
            trajectory_cumulative = self.episode_block_trajectory_residual_cumulative[ds_pos]
            traj_pos = int(np.searchsorted(trajectory_cumulative, dataset_residual_rank, side='right'))
            trajectory_residual_start = 0 if traj_pos == 0 else int(trajectory_cumulative[traj_pos - 1])
            residual_offset = dataset_residual_rank - trajectory_residual_start
            count = int(self.episode_block_valid_counts[ds_pos][traj_pos])
            rank_in_trajectory = count // self.training_block_size * self.training_block_size + residual_offset
        ds = self.datasets[ds_pos]
        episode_index = int(ds.trajectory_ids[traj_pos])
        frame_idx = ds.valid_start_for_trajectory_rank(traj_pos, rank_in_trajectory, self._allow_padding_for_dataset(ds_pos))
        return (ds_pos, episode_index, frame_idx)

    def _sample_bucket_wave_stream_rank(self, stream_id: int, stream_rank: int) -> tuple[int, int, int]:
        dataset_cumulative = self.bucket_wave_dataset_cumulative_counts[stream_id]
        dataset_pos_in_bucket = int(np.searchsorted(dataset_cumulative, stream_rank, side='right'))
        dataset_start = 0 if dataset_pos_in_bucket == 0 else int(dataset_cumulative[dataset_pos_in_bucket - 1])
        ds_pos = int(self.bucket_wave_dataset_positions[stream_id][dataset_pos_in_bucket])
        rank_in_dataset = stream_rank - dataset_start
        if self.bucket_wave_apply_dataset_weights:
            capacity = int(self.bucket_wave_dataset_capacities[ds_pos])
            quota = int(self.bucket_wave_dataset_quotas[ds_pos])
            if not 0 <= rank_in_dataset < quota:
                raise IndexError(f'weighted bucket-wave rank {rank_in_dataset} exceeds dataset quota {quota} for {self.datasets[ds_pos].name}')
            if capacity <= 0:
                raise RuntimeError(f'weighted bucket-wave dataset {self.datasets[ds_pos].name} has a non-positive capacity')
            cycle_id = 0
            if bool(self.bucket_wave_dataset_is_cyclic[ds_pos]):
                cycle_id, rank_in_dataset = divmod(rank_in_dataset, capacity)
                rotation_key = (self.seed, self.epoch, self.datasets[ds_pos].name, cycle_id, 'bucket_wave_quota_rotation')
            elif quota > capacity:
                raise RuntimeError(f'non-cyclic weighted bucket-wave quota {quota} exceeds capacity {capacity} for {self.datasets[ds_pos].name}')
            else:
                rotation_key = (self.seed, self.epoch, self.datasets[ds_pos].name, 'bucket_wave_quota_rotation')
            offset = int(safe_hash(rotation_key) % capacity)
            rank_in_dataset = (rank_in_dataset + offset) % capacity
        trajectory_cumulative = self.bucket_wave_trajectory_cumulative_counts[ds_pos]
        traj_pos = int(np.searchsorted(trajectory_cumulative, rank_in_dataset, side='right'))
        trajectory_start = 0 if traj_pos == 0 else int(trajectory_cumulative[traj_pos - 1])
        ds = self.datasets[ds_pos]
        episode_index = int(ds.trajectory_ids[traj_pos])
        frame_idx = ds.valid_start_for_trajectory_rank(traj_pos, rank_in_dataset - trajectory_start, self._allow_padding_for_dataset(ds_pos))
        return (ds_pos, episode_index, frame_idx)

    def _allow_padding_for_dataset(self, ds_pos: int) -> bool:
        return _dataset_allows_end_padding(self.datasets[int(ds_pos)], self.allow_padding_at_end)

    def _training_valid_start_counts(self, ds_pos: int) -> np.ndarray:
        ds = self.datasets[int(ds_pos)]
        counts = ds.valid_start_counts(self._allow_padding_for_dataset(ds_pos)).astype(np.int64, copy=True)
        if self.excluded_eval_episodes:
            for traj_pos, episode_index in enumerate(ds.trajectory_ids):
                if (int(ds_pos), int(episode_index)) in self.excluded_eval_episodes:
                    counts[traj_pos] = 0
        if self.excluded_eval_steps:
            for excluded_ds_pos, episode_index, _ in self.excluded_eval_steps:
                if int(excluded_ds_pos) != int(ds_pos):
                    continue
                matches = np.where(ds.trajectory_ids == int(episode_index))[0]
                if len(matches):
                    traj_pos = int(matches[0])
                    counts[traj_pos] = max(0, int(counts[traj_pos]) - 1)
        return counts

    def _resolve_fixed_eval_dataset_pos(self, sample: dict[str, Any]) -> int:
        if 'spec_index' in sample:
            pos = int(sample['spec_index'])
            if pos < 0 or pos >= len(self.datasets):
                raise IndexError(f'fixed eval spec_index={pos} out of range for {len(self.datasets)} datasets')
            return pos
        if 'dataset_index' in sample:
            pos = int(sample['dataset_index'])
            if pos < 0 or pos >= len(self.datasets):
                raise IndexError(f'fixed eval dataset_index={pos} out of range for {len(self.datasets)} datasets')
            return pos
        name = sample.get('dataset', sample.get('name'))
        if name is None:
            raise KeyError('fixed eval sample needs one of spec_index, dataset_index, dataset, or name')
        matches = [idx for idx, ds in enumerate(self.datasets) if ds.name == str(name)]
        if not matches:
            raise KeyError(f'fixed eval dataset {name!r} not found')
        if len(matches) > 1:
            raise ValueError(f'fixed eval dataset {name!r} is ambiguous; use spec_index instead')
        return matches[0]

    def _build_fixed_eval_steps(self, samples: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        for sample_idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f'fixed eval sample {sample_idx} must be a mapping')
            ds_pos = self._resolve_fixed_eval_dataset_pos(sample)
            ds = self.datasets[ds_pos]
            if 'episode_index' in sample:
                episode_index = int(sample['episode_index'])
            else:
                episode_pos = int(sample.get('episode_pos', sample.get('trajectory_pos', 0)))
                if episode_pos < 0 or episode_pos >= len(ds.trajectory_ids):
                    raise IndexError(f'fixed eval episode_pos={episode_pos} out of range for dataset {ds.name}')
                episode_index = int(ds.trajectory_ids[episode_pos])
            if episode_index not in ds.episodes_dict:
                raise KeyError(f'fixed eval episode_index={episode_index} not found in dataset {ds.name}')
            traj_matches = np.where(ds.trajectory_ids == episode_index)[0]
            if len(traj_matches) == 0:
                raise KeyError(f'fixed eval episode_index={episode_index} is not selected in dataset {ds.name}')
            traj_pos = int(traj_matches[0])
            max_start = ds.max_start_for_trajectory_pos(traj_pos, self._allow_padding_for_dataset(ds_pos))
            frame_idx = int(sample.get('frame_index', sample.get('start_frame', 0)))
            if frame_idx < 0 or frame_idx > max_start:
                raise IndexError(f'fixed eval frame_index={frame_idx} invalid for dataset={ds.name} episode={episode_index}; valid range is 0..{max_start}')
            steps.append((ds_pos, episode_index, frame_idx))
        return steps

    def _build_per_dataset_fixed_eval_steps(self, *, episode_pos: int | None=0, frame_index: int=0) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        for ds_pos, ds in enumerate(self.datasets):
            valid_counts = ds.valid_start_counts(self._allow_padding_for_dataset(ds_pos))
            valid_positions = [pos for pos, count in enumerate(valid_counts) if int(count) > 0]
            if not valid_positions:
                raise ValueError(f'dataset {ds.name} has no valid trajectory for fixed per-dataset eval')
            if episode_pos is None:
                traj_pos = valid_positions[0]
            else:
                pos = int(episode_pos)
                if pos < 0 or pos >= len(valid_positions):
                    raise IndexError(f'fixed eval episode_pos={pos} out of valid range for dataset {ds.name}')
                traj_pos = valid_positions[pos]
            episode_index = int(ds.trajectory_ids[traj_pos])
            max_start = ds.max_start_for_trajectory_pos(traj_pos, self._allow_padding_for_dataset(ds_pos))
            frame_idx = int(frame_index)
            if frame_idx < 0 or frame_idx > max_start:
                raise IndexError(f'fixed eval frame_index={frame_idx} invalid for dataset={ds.name} episode={episode_index}; valid range is 0..{max_start}')
            steps.append((ds_pos, episode_index, frame_idx))
        return steps

    def _eval_frame_indices(self, max_start: int, frames_per_episode: int | None, frame_policy: str, rng: np.random.Generator | None=None) -> list[int]:
        if max_start < 0:
            return []
        n = max(1, int(frames_per_episode or 1))
        policy = str(frame_policy or 'first').lower()
        if policy == 'first':
            return list(range(0, min(n, max_start + 1)))
        if policy == 'middle':
            if n == 1:
                return [max_start // 2]
            values = np.linspace(0, max_start, num=min(n, max_start + 1))
            return sorted({int(round(v)) for v in values})
        if policy == 'last':
            start = max(0, max_start - n + 1)
            return list(range(start, max_start + 1))
        if policy == 'uniform':
            values = np.linspace(0, max_start, num=min(n, max_start + 1))
            return sorted({int(round(v)) for v in values})
        if policy == 'random':
            rng = rng or np.random.default_rng(self.seed)
            count = min(n, max_start + 1)
            return sorted((int(v) for v in rng.choice(max_start + 1, size=count, replace=False)))
        raise ValueError(f'Unsupported eval_frame_policy={frame_policy!r}; use first/middle/last/uniform/random')

    @staticmethod
    def _allocate_eval_counts(weights: list[float], total: int) -> list[int]:
        weights_arr = np.asarray(weights, dtype=np.float64)
        counts = np.zeros(len(weights_arr), dtype=np.int64)
        total = max(0, int(total))
        positive = weights_arr > 0
        if total <= 0 or not bool(positive.any()):
            return counts.tolist()
        remaining_total = total
        if total >= int(positive.sum()):
            counts[positive] = 1
            remaining_total -= int(positive.sum())
        if remaining_total > 0:
            active_weights = np.where(positive, weights_arr, 0.0)
            raw = active_weights / float(active_weights.sum()) * float(remaining_total)
            add = np.floor(raw).astype(np.int64)
            counts += add
            remainder = remaining_total - int(add.sum())
            if remainder > 0:
                fractions = raw - add
                order = np.argsort(-fractions)
                for idx in order[:remainder]:
                    if positive[int(idx)]:
                        counts[int(idx)] += 1
        return counts.tolist()

    def _sample_one_source_eval_step(self, group_id: Any, ds_positions: list[int], sample_index: int) -> tuple[int, int, int] | None:
        valid_totals = []
        for ds_pos in ds_positions:
            counts = self.datasets[ds_pos].valid_start_counts(self._allow_padding_for_dataset(ds_pos)).astype(np.float64)
            valid_totals.append(float(counts.sum()))
        weights = np.asarray(valid_totals, dtype=np.float64)
        if float(weights.sum()) <= 0:
            return None
        rng = np.random.default_rng(safe_hash(('eval_source', self.seed, group_id, int(sample_index))))
        ds_pos = int(rng.choice(ds_positions, p=weights / weights.sum()))
        ds = self.datasets[ds_pos]
        valid_counts = ds.valid_start_counts(self._allow_padding_for_dataset(ds_pos)).astype(np.float64)
        if float(valid_counts.sum()) <= 0:
            return None
        traj_pos = int(rng.choice(len(ds.trajectory_ids), p=valid_counts / valid_counts.sum()))
        episode_index = int(ds.trajectory_ids[traj_pos])
        frame_idx = ds.sample_valid_start_for_trajectory_pos(traj_pos, rng, self._allow_padding_for_dataset(ds_pos))
        if frame_idx is None:
            return None
        return (ds_pos, episode_index, frame_idx)

    def _build_eval_steps(self, *, episodes_per_dataset: int | None=None, frames_per_episode: int | None=None, frame_policy: str='first', scope: str='dataset', segments_per_source: int | None=None, total_samples: int | None=None) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        scope = str(scope or 'dataset').lower()
        grouped_scopes = {'source', 'group', 'source_file', 'source_type', 'type', 'source_entry'}
        total_samples_int = None if total_samples in (None, '') else int(total_samples)
        if scope in grouped_scopes or (total_samples_int is not None and total_samples_int > 0):
            groups: dict[Any, list[int]] = {}
            for ds_pos, ds in enumerate(self.datasets):
                if scope in {'source_type', 'type', 'source_entry'}:
                    key = str(ds.spec.source_name or ds.spec.group_id)
                elif scope in {'dataset', 'leaf', 'dataset_name'}:
                    key = f'{ds_pos}:{ds.name}'
                else:
                    key = int(ds.spec.group_id)
                groups.setdefault(key, []).append(ds_pos)
            group_items = sorted(groups.items(), key=lambda item: str(item[0]))
            if total_samples_int is not None and total_samples_int > 0:
                group_weights = [sum((float(self.dataset_sampling_weights[ds_pos]) for ds_pos in ds_positions)) for _, ds_positions in group_items]
                counts = self._allocate_eval_counts(group_weights, total_samples_int)
                for (group_id, ds_positions), count in zip(group_items, counts):
                    for sample_index in range(int(count)):
                        step = self._sample_one_source_eval_step(group_id, ds_positions, sample_index)
                        if step is not None:
                            steps.append(step)
                return steps
            n = max(1, int(segments_per_source or 1))
            for group_id, ds_positions in group_items:
                for sample_index in range(n):
                    step = self._sample_one_source_eval_step(group_id, ds_positions, sample_index)
                    if step is not None:
                        steps.append(step)
            return steps
        if scope != 'dataset':
            raise ValueError(f'Unsupported eval scope={scope!r}; use dataset/source/source_type')
        for ds_pos, ds in enumerate(self.datasets):
            valid_counts = ds.valid_start_counts(self._allow_padding_for_dataset(ds_pos))
            valid_positions = [pos for pos, count in enumerate(valid_counts) if int(count) > 0]
            if episodes_per_dataset is not None and int(episodes_per_dataset) > 0:
                valid_positions = valid_positions[:int(episodes_per_dataset)]
            for traj_pos in valid_positions:
                episode_index = int(ds.trajectory_ids[traj_pos])
                max_start = ds.max_start_for_trajectory_pos(traj_pos, self._allow_padding_for_dataset(ds_pos))
                rng = np.random.default_rng(safe_hash(('eval_dataset', self.seed, int(ds_pos), int(episode_index))))
                for frame_idx in self._eval_frame_indices(max_start, frames_per_episode, frame_policy, rng):
                    steps.append((ds_pos, episode_index, int(frame_idx)))
        return steps

    def _is_training_excluded(self, ds_pos, episode_index, frame_idx):
        return ((ds_pos, episode_index) in self.excluded_eval_episodes
                or (ds_pos, episode_index, frame_idx) in self.excluded_eval_steps)

    def __len__(self) -> int:
        if self.eval_steps is not None:
            return len(self.eval_steps)
        return self.epoch_length if self.training else int(self.dataset_lengths.sum())

    def _sample_training_step(self, index: int) -> tuple[int, int, int]:
        return self._sample_optimizer_stratified_training_step(int(index))

    def _sample_step(self, index: int) -> tuple[int, int, int]:
        if self.eval_steps is not None:
            if index < 0 or index >= len(self.eval_steps):
                raise IndexError(f'index {index} out of bounds for eval subset length {len(self.eval_steps)}')
            return self.eval_steps[index]
        if self.training:
            return self._sample_training_step(int(index))
        ds_pos = bisect.bisect_right(self.cumulative_lengths, index)
        start = 0 if ds_pos == 0 else self.cumulative_lengths[ds_pos - 1]
        episode_index, frame_idx = self.datasets[ds_pos]._global_to_episode_frame(index - start)
        return (ds_pos, int(episode_index), int(frame_idx))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_pos, episode_index, frame_idx = self._sample_step(int(idx))
        item = self.datasets[ds_pos].get_step_item(episode_index, frame_idx)
        item['dataset_index'] = torch.tensor(ds_pos, dtype=torch.long)
        return item

    def get_items(self, indices: list[int]) -> list[dict[str, Any]]:
        steps = [self._sample_step(int(index)) for index in indices]
        grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for output_pos, (ds_pos, episode_index, frame_idx) in enumerate(steps):
            grouped.setdefault((ds_pos, episode_index), []).append((output_pos, frame_idx))
        items: list[dict[str, Any] | None] = [None] * len(indices)
        for (ds_pos, episode_index), positions in grouped.items():
            dataset = self.datasets[ds_pos]
            parquet = dataset._load_episode_parquet(episode_index)
            text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for output_pos, frame_idx in positions:
                item = dataset.get_step_item(episode_index, frame_idx, parquet=parquet, text_context_cache=text_context_cache)
                item['dataset_index'] = torch.tensor(ds_pos, dtype=torch.long)
                items[output_pos] = item
        return [item for item in items if item is not None]

    def get_video_for_item(self, item: dict[str, Any]) -> torch.Tensor:
        ds_pos = int(item['dataset_index'].item())
        return self.datasets[ds_pos].get_video_for_item(item)
    def _sample_umi_block(
        self: Any,
        *,
        stream_id: int,
        stream_block: int,
        block_offset: int,
    ) -> tuple[int, int, int]:
        """Sample one UMI temporal block exactly like the base random sampler."""

        positions = np.asarray(
            self.bucket_wave_dataset_positions[int(stream_id)], dtype=np.int32
        )
        if positions.size <= 0:
            raise RuntimeError(f"UMI stream {stream_id} has no dataset positions")

        # The rank is part of the seed because optimizer-stratified global
        # indices can otherwise map to the same source/block on multiple ranks.
        rank = int(getattr(self, "bucket_wave_rank", 0))
        rng = np.random.default_rng(
            safe_hash(
                (
                    self.seed,
                    self.epoch,
                    rank,
                    int(stream_id),
                    int(stream_block),
                    "umi_random_with_replacement",
                )
            )
        )

        configured_weights = np.asarray(
            [float(self.dataset_sampling_weights[int(ds_pos)]) for ds_pos in positions],
            dtype=np.float64,
        )
        if not np.isfinite(configured_weights).all() or configured_weights.sum() <= 0:
            configured_weights = np.ones(positions.size, dtype=np.float64)
        configured_weights /= configured_weights.sum()

        block_size = int(self.training_block_size)
        span = (block_size - 1) * int(self.training_frame_stride)
        for _ in range(128):
            ds_pos = int(rng.choice(positions, p=configured_weights))
            # The frozen loader intentionally does not materialize trajectory
            # weights for ``optimizer_stratified_without_replacement``.  UMI is
            # the one family for which we deliberately restore random-with-
            # replacement sampling, so build the same valid-start weights lazily
            # instead of indexing the empty base list.
            cache = getattr(self, "_umi_trajectory_weights", None)
            if cache is None:
                cache = self._umi_trajectory_weights = {}
            traj_weights = cache.get(ds_pos)
            if traj_weights is None:
                ds = self.datasets[ds_pos]
                valid_start_counts = self._training_valid_start_counts(ds_pos).astype(
                    np.float64
                )
                traj_weights = (valid_start_counts > 0).astype(np.float64)
                if self.balance_trajectory_weights:
                    traj_weights *= valid_start_counts
                total = float(traj_weights.sum())
                if total > 0:
                    traj_weights = traj_weights / total
                cache[ds_pos] = traj_weights
            if float(np.asarray(traj_weights).sum()) <= 0:
                continue
            traj_pos = int(
                rng.choice(len(self.datasets[ds_pos].trajectory_ids), p=traj_weights)
            )
            ds = self.datasets[ds_pos]
            base_frame = ds.sample_valid_start_for_trajectory_pos(
                traj_pos,
                rng,
                self._allow_padding_for_dataset(ds_pos),
                extra_span=span,
            )
            if base_frame is None:
                continue
            frame_idx = int(base_frame) + int(block_offset) * int(
                self.training_frame_stride
            )
            episode_index = int(ds.trajectory_ids[traj_pos])
            if not self._is_training_excluded(ds_pos, episode_index, frame_idx):
                return ds_pos, episode_index, frame_idx
        raise RuntimeError(
            "could not sample a UMI random-with-replacement block after 128 attempts"
        )

