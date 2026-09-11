from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from worldscape_policy.data import DATASETS
from worldscape_policy.data.adapters.common import (
    _decode_image,
    _state_and_action,
)
from worldscape_policy.data.adapters.hdf5 import NativeHDF5Dataset
from worldscape_policy.geometry import quaternion_to_rotation6d
from worldscape_policy.training.data import (
    _LengthBucketBatchSampler,
    build_registered_loader,
)


def test_native_image_decoder_accepts_raw_and_base64_encoded_bytes() -> None:
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    stream = io.BytesIO()
    Image.fromarray(source).save(stream, format="PNG")
    raw = stream.getvalue()
    expected = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    for payload in (raw, base64.b64encode(raw), base64.b64encode(raw).decode()):
        np.testing.assert_array_equal(_decode_image(payload), expected)
    np.testing.assert_array_equal(_decode_image(source), source)


def test_legacy_two_arm_pose_conversion_has_exact_eef6d_order() -> None:
    half = np.sqrt(0.5)
    end_pose = np.asarray(
        [[1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, half, half]],
        dtype=np.float32,
    )
    qpos = np.arange(14, dtype=np.float32)[None]
    state, action = _state_and_action(
        {"observations.end_pose": end_pose, "observations.qpos": qpos}, None
    )
    assert state.shape == action.shape == (1, 20)
    np.testing.assert_allclose(state[0, :3], [1, 2, 3])
    np.testing.assert_allclose(state[0, 3:9], [1, 0, 0, 1, 0, 0], atol=1e-6)
    assert state[0, 9] == 6
    np.testing.assert_allclose(state[0, 10:13], [4, 5, 6])
    np.testing.assert_allclose(state[0, 13:19], [0, -1, 1, 0, 0, 0], atol=1e-6)
    assert state[0, 19] == 13


def test_owned_quaternion_rotation6d_known_rotations() -> None:
    half = np.sqrt(0.5)
    result = quaternion_to_rotation6d(
        np.asarray([[0, 0, 0, 1], [0, 0, half, half]], dtype=np.float32)
    )
    np.testing.assert_allclose(result[0], [1, 0, 0, 1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result[1], [0, -1, 1, 0, 0, 0], atol=1e-6)


def test_hdf5_temporal_map_exposes_all_eligible_anchors_and_task_metadata(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "tasks.jsonl").write_text(
        '{"task_index": 7, "task": "stack the blocks"}\n'
    )
    path = tmp_path / "episode.hdf5"
    length = 30
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((length, 1, 2, 2, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((length, 2), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((length, 2), dtype=np.float32))
        handle.create_dataset("task_index", data=np.asarray(7))
    dataset = NativeHDF5Dataset(
        tmp_path,
        visual_prompt="none",
        temporal_packing=True,
        max_chunk_size=1,
        history_num_frames=0,
    )
    assert len(dataset) == 6
    assert {anchor for _, anchor in dataset.index} == set(range(6))
    assert dataset[0].high_level_instruction == "stack the blocks"


def test_hdf5_whole_shard_cache_loads_each_episode_once(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    for episode in range(2):
        with h5py.File(tmp_path / f"episode_{episode}.hdf5", "w") as handle:
            handle.create_dataset(
                "observation/camera/head",
                data=np.zeros((4, 2, 2, 3), dtype=np.uint8),
            )
            handle.create_dataset(
                "observation/state", data=np.zeros((4, 2), dtype=np.float32)
            )
            handle.create_dataset("action", data=np.zeros((4, 2), dtype=np.float32))
    dataset = NativeHDF5Dataset(tmp_path, visual_prompt="none")
    assert dataset.shard_ranges(10_000) == ((0, 2),)
    shard = dataset.load_shard(0, 2)
    assert set(shard) == {0, 1}
    assert shard[0]["observation.camera.head"].dtype == np.uint8
    first = dataset.get_from_shard(0, shard)
    second = dataset.get_from_shard(1, shard)
    assert (first.episode_id, second.episode_id) == ("0", "1")


def test_hdf5_shard_ranges_never_split_or_emit_empty_episode_groups() -> None:
    dataset = object.__new__(NativeHDF5Dataset)
    dataset.paths = [object(), object(), object()]
    dataset.index = [
        *((0, anchor) for anchor in range(6)),
        *((1, anchor) for anchor in range(6)),
        *((2, anchor) for anchor in range(3)),
    ]
    ranges = dataset.shard_ranges(10)
    assert ranges == ((0, 12), (12, 15))
    assert all(start < stop for start, stop in ranges)
    for start, stop in ranges:
        episodes = {episode for episode, _ in dataset.index[start:stop]}
        for episode in episodes:
            assert all(
                start <= index < stop
                for index, (owner, _) in enumerate(dataset.index)
                if owner == episode
            )


@pytest.mark.parametrize(
    ("dataset_name", "visual_prompt", "context_sampling_mode", "context_video_len"),
    [
        ("worldscape_hdf5_text", "none", "none", 1),
        ("worldscape_hdf5_goal", "goal", "last", 1),
        ("worldscape_hdf5_demo", "demo", "uniform", 50),
    ],
)
def test_hdf5_factories_accept_explicit_profile_kwargs(
    tmp_path,
    dataset_name,
    visual_prompt,
    context_sampling_mode,
    context_video_len,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "episode.hdf5", "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((30, 1, 2, 2, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((30, 2), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((30, 2), dtype=np.float32))
        handle.create_dataset("language", data=np.bytes_("move"))
    dataset = DATASETS.create(
        dataset_name,
        data_root=tmp_path,
        temporal_packing=True,
        max_chunk_size=4,
        wo_norm=True,
        visual_prompt=visual_prompt,
        context_sampling_mode=context_sampling_mode,
        context_video_len=context_video_len,
        ctx_head_only=False,
        shard_size=10_000,
        shard_sampling_rate=0.1,
        num_shards_to_sample=1,
        training=True,
        seed=62,
    )
    assert len(dataset.shards) == 1
    assert dataset.shards[0].length == len(dataset.datasets[0])


def test_length_bucket_sampler_separates_mixed_24_and_96() -> None:
    class Sample:
        def __init__(self, length: int):
            self.actions = torch.zeros(length, 20)
            self.robot_state = torch.zeros(length // 24, 20)
            self.observations = {
                "video": torch.zeros(length // 3 + 1, 1, 2, 2, 3)
            }

    class Samples:
        def __init__(self):
            self.values = [Sample(24), Sample(96), Sample(24), Sample(96)]

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    # Exercise the batching algorithm without relying on model/GPU geometry.
    dataset = Samples()
    sampler = object.__new__(_LengthBucketBatchSampler)
    sampler.batch_size = 2
    sampler.shuffle = False
    sampler.seed = 0
    sampler.epoch = 0
    sampler.offset = 0
    sampler.distributed = torch.utils.data.DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=False
    )
    sampler.signatures = {
        index: (
            len(sample.actions),
            len(sample.robot_state),
            *(len(value) for value in sample.observations.values()),
        )
        for index, sample in enumerate(dataset.values)
    }
    batches = list(sampler)
    assert batches == [[0, 2], [1, 3]]


def test_map_bucket_loader_uses_metadata_without_startup_decode(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    for episode in range(2):
        with h5py.File(tmp_path / f"episode_{episode}.hdf5", "w") as handle:
            handle.create_dataset(
                "observation/video",
                data=np.zeros((30, 1, 2, 2, 3), dtype=np.uint8),
            )
            handle.create_dataset(
                "observation/state", data=np.zeros((30, 2), dtype=np.float32)
            )
            handle.create_dataset("action", data=np.zeros((30, 2), dtype=np.float32))
            handle.create_dataset("language", data=np.bytes_("move"))

    decodes = 0

    class CountingDataset(NativeHDF5Dataset):
        def __getitem__(self, index):
            nonlocal decodes
            decodes += 1
            return super().__getitem__(index)

    name = f"audit-counting-hdf5-{tmp_path.name}"
    DATASETS.register(
        name,
        lambda: CountingDataset(
            tmp_path,
            visual_prompt="none",
            temporal_packing=True,
            max_chunk_size=1,
            history_num_frames=0,
        ),
    )
    loader = build_registered_loader(
        dataset_name=name,
        mode="auto",
        batch_size=2,
        num_workers=0,
        bucket_by_length=True,
    )

    assert decodes == 0
    next(iter(loader))
    assert decodes == 2


def test_hdf5_multitask_segments_keep_labels_and_packing_in_bounds(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(
        '{"task_index": 7, "task": "stack blocks", "event_instruction": "place red"}\n'
        '{"task_index": 8, "task": "sort shapes", "event_instruction": "move circle"}\n'
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 60, "path": "episode.hdf5", '
        '"task_segments": [{"start": 0, "end": 30, "task_index": 7}, '
        '{"start": 30, "end": 60, "task_index": 8}]}\n'
    )
    with h5py.File(tmp_path / "episode.hdf5", "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((60, 1, 2, 2, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((60, 2), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((60, 2), dtype=np.float32))
        handle.create_dataset(
            "task_index",
            data=np.asarray([7] * 30 + [8] * 30, dtype=np.int64),
        )

    dataset = NativeHDF5Dataset(
        tmp_path,
        visual_prompt="none",
        temporal_packing=True,
        max_chunk_size=1,
        history_num_frames=0,
    )

    assert {anchor for _, anchor in dataset.index} == set(range(6)) | set(
        range(30, 36)
    )
    first = dataset[dataset.index.index((0, 0))]
    second = dataset[dataset.index.index((0, 30))]
    assert (first.high_level_instruction, first.event_instruction) == (
        "stack blocks",
        "place red",
    )
    assert (second.high_level_instruction, second.event_instruction) == (
        "sort shapes",
        "move circle",
    )
    assert (first.task_id, second.task_id) == ("7", "8")
    assert int(first.source_indices["action"].max()) < 30
    assert int(second.source_indices["action"].min()) >= 30
