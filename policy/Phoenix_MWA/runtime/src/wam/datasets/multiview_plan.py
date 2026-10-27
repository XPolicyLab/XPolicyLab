"""Deterministic, complete block traversal of a frozen multiview sidecar plan."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from wam.datasets.interndata_multiview import (
    INTERNDATA_MULTIVIEW_POLICY,
    INTERNDATA_SOURCES,
)


PLAN_SCHEMA = "phoenix.robot37.multiview_plan.v1"
SAMPLER_SCHEMA = "phoenix.robot37.multiview_sampler.v1"
ARRAY_DTYPES = {
    "run_specs": "int64",
    "run_counts": "int64",
    "run_pair_ids": "int8",
    "run_padding": "uint8",
    "episode_clocks": "int64",
}


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_pair_acceptance(plan_root):
    root = Path(plan_root)
    report = json.loads((root / "PAIR_ACCEPTANCE.json").read_text())
    if (
        report.get("passed") is not True
        or report.get("plan_sha256") != file_sha256(root / "manifest.json")
        or report.get("all_rh_paired_raw_supervision_checked") is not True
        or report.get("strict_source_equality") is not True
        or report.get("errors")
        or not report.get("rgb_readback")
    ):
        raise ValueError("Multiview paired supervision/readback acceptance is missing or stale")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("interndata_multiview"):
        evidence = report.get("interndata_review", {})
        audit_path = root / "INTERNDATA_ACCEPTANCE.json"
        if (
            manifest["interndata_multiview"] != INTERNDATA_MULTIVIEW_POLICY
            or not audit_path.is_file()
            or evidence.get("sha256") != file_sha256(audit_path)
        ):
            raise ValueError("InternData multiview acceptance is missing or stale")
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("passed") is not True
            or audit.get("plan_sha256") != file_sha256(root / "manifest.json")
            or audit.get("package_sha256") != manifest["package_sha256"]
            or audit.get("norm_sha256") != manifest["norm_sha256"]
            or audit.get("all_episode_view_metadata_checked") is not True
        ):
            raise ValueError("InternData multiview acceptance does not cover this plan")
    return report


def affine_permutation(size, seed, epoch):
    """Return a bijection over blocks, including non-prime block counts."""
    if size <= 0:
        raise ValueError("A sampling plan must contain a block")
    value = int(stable_hash([seed, epoch, "multiview_blocks_v1"]), 16)
    multiplier = value % size or 1
    while math.gcd(multiplier, size) != 1:
        multiplier += 1
    return multiplier, (value // max(1, size)) % size


class MultiViewPlanSampler:
    """Address all plan samples once before any global-batch padding replay.

    The sample index is global. Accelerate must shard it exactly once in full
    per-rank batches; four adjacent indices are always one indivisible block.
    Padding is explicit metadata and remains supervised. It is never counted as
    newly covered data. Epoch and draw identities make worker prefetch stateless.
    """

    training_block_size = 4
    training_frame_stride = 1

    def __init__(self, plan_root, *, package, batch_size, seed,
                 rgb_selection_root=None, rgb_selection_sha256=None):
        self.root = Path(plan_root).resolve()
        self.package = package
        self.seed = int(seed)
        self.training_batch_size_per_rank = int(batch_size)
        self.bucket_wave_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.training_batch_size_per_rank <= 0 or batch_size % 4:
            raise ValueError("Multiview batches must contain complete blocks of four")
        if self.bucket_wave_world_size <= 0:
            raise ValueError("Invalid multiview world size")
        self._epoch = mp.Value("q", 0, lock=False)
        self._permutation_epoch = None
        self._permutation_parameters = None
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.plan_sha256 = file_sha256(self.root / "manifest.json")
        completion = json.loads((self.root / "_PLAN_COMPLETE").read_text())
        if completion.get("manifest_sha256") != self.plan_sha256:
            raise ValueError("Multiview plan completion fingerprint mismatch")
        manifest = self.manifest
        if (
            "interndata_multiview" in manifest
            and manifest["interndata_multiview"] != INTERNDATA_MULTIVIEW_POLICY
        ):
            raise ValueError("Unknown InternData camera variant policy")
        if manifest.get("schema") != PLAN_SCHEMA or manifest.get("anchor_stride") != 1:
            raise ValueError("Expected an immutable stride-one multiview plan")
        if manifest.get("package_sha256") != package.fingerprint:
            raise ValueError("Multiview plan belongs to a different data package")
        if manifest.get("norm_sha256") != file_sha256(package.root / "norm.json"):
            raise ValueError("Multiview plan normalization identity mismatch")
        self.array_root = self.root
        selection_sha = None
        if rgb_selection_root is not None:
            from wam.datasets.robot37_rgb_variants import load_selection
            selection, selection_sha = load_selection(
                rgb_selection_root, expected_sha256=rgb_selection_sha256,
                package_sha256=package.fingerprint, parent_plan_sha256=self.plan_sha256,
            )
            self.array_root = Path(rgb_selection_root)
            # Parent pairing acceptance remains bound to the original plan. The
            # derived arrays remove only explicitly excluded ordinary episodes.
            self.manifest = dict(manifest, **{key: selection[key] for key in (
                'files', 'num_runs', 'num_blocks', 'unique_samples', 'padded_samples')})
            manifest = self.manifest
        elif rgb_selection_sha256 is not None:
            raise ValueError('RGB selection hash requires its artifact root')
        self._open_arrays(verify_hashes=True)
        self._validate_arrays()
        blocks_per_update = (
            self.training_batch_size_per_rank * self.bucket_wave_world_size // 4
        )
        self.scheduled_blocks = (
            (self.num_blocks + blocks_per_update - 1) // blocks_per_update
        ) * blocks_per_update
        self.epoch_length = self.scheduled_blocks * 4
        self.sampling_contract = {
            "schema": SAMPLER_SCHEMA,
            "plan_schema": PLAN_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "package_sha256": package.fingerprint,
            "norm_sha256": manifest["norm_sha256"],
            "anchor_stride": 1,
            "block_size": 4,
            "batch_size_per_rank": self.training_batch_size_per_rank,
            "world_size": self.bucket_wave_world_size,
            "seed": self.seed,
            "unique_samples": self.unique_samples,
            "plan_padding_samples": self.plan_padding_samples,
            "global_padding_samples": self.epoch_length - self.num_blocks * 4,
            "samples_per_epoch": self.epoch_length,
            "shuffle": "bijective_block_affine_v1",
            "padding_policy": "explicit_supervised_replay_after_complete_coverage",
        }
        if selection_sha is not None:
            self.sampling_contract['rgb_selection_sha256'] = selection_sha

    def _open_arrays(self, *, verify_hashes):
        for name, dtype in ARRAY_DTYPES.items():
            filename = name + ".npy"
            path = self.array_root / filename
            metadata = self.manifest["files"][filename]
            if path.stat().st_size != metadata["bytes"]:
                raise ValueError(f"Multiview array length changed: {filename}")
            if verify_hashes and file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Multiview array hash changed: {filename}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.dtype(dtype):
                raise ValueError(f"Multiview array dtype changed: {filename}")
            if list(array.shape) != metadata["shape"]:
                raise ValueError(f"Multiview array shape changed: {filename}")
            setattr(self, name, array)
        self._pid = os.getpid()

    def _validate_arrays(self):
        runs = int(self.manifest["num_runs"])
        expected = {
            "run_specs": (runs, 4, 4),
            "run_counts": (runs,),
            "run_pair_ids": (runs, 4),
            "run_padding": (runs, 4),
        }
        if runs <= 0:
            raise ValueError("Empty multiview plan")
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Invalid {name} geometry")
        if self.episode_clocks.ndim != 2 or self.episode_clocks.shape[1] != 2:
            raise ValueError("Invalid episode clock geometry")
        if len(self.episode_clocks) and (
            np.any(np.diff(self.episode_clocks[:, 0]) <= 0)
            or np.any(self.episode_clocks[:, 0] < 0)
        ):
            raise ValueError("Episode clock IDs must be sorted and unique")
        if np.any(self.run_counts <= 0):
            raise ValueError("Multiview runs must contain positive block counts")
        if (
            np.any(self.run_specs[:, :, 0] < 0)
            or np.any(self.run_specs[:, :, 1] < 0)
            or np.any(self.run_specs[:, :, 2] < -1)
            or np.any(self.run_specs[:, :, 3] < 0)
            or np.any(self.run_padding > 1)
            or np.any(self.run_pair_ids < -1)
            or np.any(self.run_pair_ids > 3)
        ):
            raise ValueError("Invalid multiview run descriptor")
        self._cumulative = np.cumsum(self.run_counts, dtype=np.int64)
        if np.any(np.diff(self._cumulative) <= 0):
            raise ValueError("Multiview run count overflow")
        self.num_blocks = int(self._cumulative[-1])
        self.plan_padding_samples = int(
            np.dot(self.run_counts, self.run_padding.sum(axis=1, dtype=np.int64))
        )
        self.unique_samples = self.num_blocks * 4 - self.plan_padding_samples
        if (
            self.num_blocks != self.manifest["num_blocks"]
            or self.unique_samples != self.manifest["unique_samples"]
            or self.plan_padding_samples != self.manifest["padded_samples"]
        ):
            raise ValueError("Multiview coverage counts do not match the manifest")

    @property
    def epoch(self):
        return int(self._epoch.value)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Multiview epoch must be a nonnegative integer")
        self._epoch.value = epoch

    def __len__(self):
        return self.epoch_length

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in ARRAY_DTYPES:
            state[name] = None
        state["_cumulative"] = None
        state["_pid"] = None
        return state

    def _ensure_arrays(self):
        if self._pid != os.getpid() or self.run_counts is None:
            # Hashes were checked before workers started. The immutable file
            # identity is part of the resume contract; workers reopen read-only.
            self._open_arrays(verify_hashes=False)
            self._cumulative = np.cumsum(self.run_counts, dtype=np.int64)

    def _clock(self, episode_id):
        index = int(np.searchsorted(self.episode_clocks[:, 0], episode_id))
        if index == len(self.episode_clocks) or self.episode_clocks[index, 0] != episode_id:
            raise ValueError(f"RH20T episode has no validated physical clock: {episode_id}")
        return int(self.episode_clocks[index, 1])

    def sample(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        self._ensure_arrays()
        block_id, lane = divmod(int(index), 4)
        epoch = self.epoch
        if self._permutation_epoch != epoch:
            self._permutation_parameters = affine_permutation(
                self.num_blocks, self.seed, epoch
            )
            self._permutation_epoch = epoch
        multiplier, offset = self._permutation_parameters
        global_padding = block_id >= self.num_blocks
        plan_block = (multiplier * (block_id % self.num_blocks) + offset) % self.num_blocks
        run_id = int(np.searchsorted(self._cumulative, plan_block, side="right"))
        before = int(self._cumulative[run_id - 1]) if run_id else 0
        within_run = plan_block - before
        episode_id, base, variant, step = map(int, self.run_specs[run_id, lane])
        start = base + within_run * step
        episode = self.package.episode(episode_id)
        profile = self.package.manifest["profiles"][episode["profile"]]
        record = episode["record"]
        group_id = stable_hash([profile["source"], profile["id"], episode["unit"]])
        if profile["source"] == "rh20t_v6_20261003":
            physical_start = self._clock(episode_id) + start * 100
            view_id = "rh20t:" + str(record["pair_id"])
        elif variant >= 0:
            if profile["source"] in INTERNDATA_SOURCES:
                if not self.manifest.get("interndata_multiview"):
                    raise ValueError("InternData variants require an explicit plan policy")
                if variant >= len(INTERNDATA_MULTIVIEW_POLICY["base_keys"]):
                    raise ValueError("InternData plan camera variant is out of range")
                group_id = stable_hash([group_id, episode_id])
            physical_start = start
            view_id = "variant:" + str(variant)
        else:
            # Other sources are not paired merely because names happen to match.
            group_id = stable_hash([group_id, episode_id])
            physical_start = start
            view_id = "episode:" + str(episode_id)
        physical_chunk_uid = stable_hash([group_id, physical_start, 33])
        pair = int(self.run_pair_ids[run_id, lane])
        pair_id = f"pair:{pair}" if pair >= 0 else f"single:{lane}"
        return {
            "episode_id": episode_id,
            "start": start,
            "view_variant": variant,
            "group_id": group_id,
            "physical_chunk_uid": physical_chunk_uid,
            "view_id": view_id,
            "block_id": block_id,
            "pair_id": pair_id,
            "is_coverage_padding": bool(global_padding or self.run_padding[run_id, lane]),
        }
