from __future__ import annotations

import json

import numpy as np
import pytest

from worldscape_policy.data import (
    ContextSampler,
    EventSample,
    LanguageTemporalPacker,
    NativeHDF5Dataset,
    NativeShardedMixtureDataset,
    NativeVideoAugmentation,
    VLMHistorySampler,
)


def _event(index: int) -> EventSample:
    return EventSample(
        episode_id=f"episode-{index}",
        event_id=f"event-{index}",
        observations={"video": np.full((1, 1, 2, 2, 3), index, np.uint8)},
        actions=np.full((1, 1), index, np.float32),
        robot_state=np.full((1, 1), index, np.float32),
        high_level_instruction="task",
        event_instruction="task",
        goal_image=None,
        demo_video=None,
        history_head_frames=None,
        embodiment="eef",
        task_id="task",
        session_id=f"session-{index}",
    )


def test_language_temporal_packing_exact_four_chunk_formula():
    labels = np.asarray(["other"] * 24 + ["task"] * 121, dtype=object)
    packed = LanguageTemporalPacker(max_chunk_size=4).indices(72, labels)
    anchors = np.asarray([24, 48, 72, 96])
    np.testing.assert_array_equal(packed.anchors, anchors)
    np.testing.assert_array_equal(packed.state, anchors)
    np.testing.assert_array_equal(
        packed.action,
        np.concatenate([np.arange(anchor, anchor + 24) for anchor in anchors]),
    )
    expected_video = np.concatenate(
        [anchor + np.arange(0, 24, 3) for anchor in anchors]
    )
    expected_video = np.append(expected_video, expected_video[-1] + 3)
    np.testing.assert_array_equal(packed.video, expected_video)
    assert (len(packed.video), len(packed.action), len(packed.state)) == (33, 96, 4)


def test_language_temporal_packing_horizon48_uses_nine_stride6_frames():
    labels = np.asarray(["task"] * 97, dtype=object)
    packed = LanguageTemporalPacker(
        max_chunk_size=1,
        chunk_size=48,
    ).indices(24, labels)

    np.testing.assert_array_equal(packed.anchors, [24])
    np.testing.assert_array_equal(packed.state, [24])
    np.testing.assert_array_equal(packed.action, np.arange(24, 72))
    np.testing.assert_array_equal(packed.video, np.arange(24, 73, 6))
    assert (len(packed.video), len(packed.action), len(packed.state)) == (9, 48, 1)


def test_temporal_packing_stops_at_language_and_trajectory_boundaries():
    labels = np.asarray(["a"] * 48 + ["b"] * 72, dtype=object)
    trajectory = np.asarray([0] * 73 + [1] * 47)
    packed = LanguageTemporalPacker().indices(48, labels, trajectory_ids=trajectory)
    np.testing.assert_array_equal(packed.anchors, [48])
    assert packed.video.shape == (9,)
    assert packed.action.shape == (24,)
    assert packed.state.shape == (1,)


def test_temporal_packing_rejects_incomplete_single_chunk():
    with pytest.raises(ValueError, match="no complete"):
        LanguageTemporalPacker().indices(0, np.asarray(["a"] * 24))


def test_context_modes_have_exact_legacy_indices():
    frames = np.arange(7)[:, None, None, None]
    assert ContextSampler("none").sample(frames) is None
    np.testing.assert_array_equal(ContextSampler("last", 1).sample(frames), frames[-1:])
    uniform = ContextSampler("uniform", 50).sample(frames)
    expected = np.linspace(0, 6, 50).round().astype(np.int64)
    np.testing.assert_array_equal(uniform[:, 0, 0, 0], expected)
    with pytest.raises(ValueError, match="real context"):
        ContextSampler("last", 1).sample(frames[:0])


@pytest.mark.parametrize(
    ("sampler", "anchor", "expected"),
    [
        (VLMHistorySampler(8, 24, 192), 200, [8, 32, 56, 80, 104, 128, 152, 176]),
        (VLMHistorySampler(2, 24, 48), 40, [0, 16]),
        (VLMHistorySampler(8, 24, 192), 10, [0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_vlm_history_profiles_and_left_padding(sampler, anchor, expected):
    np.testing.assert_array_equal(sampler.indices(anchor, 300), expected)


def test_eval_video_pipeline_is_deterministic_and_resizes():
    video = np.arange(2 * 20 * 40 * 3, dtype=np.uint8).reshape(2, 20, 40, 3)
    transform = NativeVideoAugmentation(training=False)
    first = transform(video, rng=np.random.default_rng(1))
    second = transform(video, rng=np.random.default_rng(99))
    assert first.shape == (2, 160, 320, 3)
    np.testing.assert_array_equal(first, second)


def test_training_video_pipeline_replays_parameters_across_frames():
    frame = np.arange(20 * 40 * 3, dtype=np.uint8).reshape(20, 40, 3)
    video = np.stack([frame, frame])
    output = NativeVideoAugmentation(training=True)(
        video, rng=np.random.default_rng(5)
    )
    np.testing.assert_array_equal(output[0], output[1])


def test_native_hdf5_reader_composes_packing_context_and_history(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode.hdf5"
    length = 124
    frames = np.arange(length, dtype=np.uint8)[:, None, None, None]
    frames = np.broadcast_to(frames, (length, 2, 3, 3)).copy()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("is_exec", data=[False] * 3 + [True] * 121)
        handle.create_dataset("observation/camera/head", data=frames)
        handle.create_dataset(
            "observation/state", data=np.arange(length, dtype=np.float32)[:, None]
        )
        handle.create_dataset("action", data=np.arange(length, dtype=np.float32)[:, None])
        handle.create_dataset(
            "language", data=np.asarray([b"task"] * length, dtype="S4")
        )

    sample = NativeHDF5Dataset(
        tmp_path,
        temporal_packing=True,
        temporal_anchor_index=48,
        max_chunk_size=4,
        visual_prompt="demo",
        context_sampling_mode="uniform",
        context_video_len=50,
        history_num_frames=2,
        history_stride=24,
        history_window=48,
    )[0]
    assert sample.observations["video"].shape[:2] == (33, 1)
    assert sample.actions.shape == (96, 1)
    assert sample.robot_state.shape == (4, 1)
    assert sample.demo_video.shape == (50, 1, 2, 3, 3)
    assert sample.history_head_frames.shape == (2, 2, 3, 3)
    assert sample.source_indices is not None
    np.testing.assert_array_equal(sample.source_indices["anchors"], [0, 24, 48, 72])


def test_native_hdf5_index_cache_and_step_stride(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode.hdf5"
    length = 124
    frames = np.arange(length, dtype=np.uint8)[:, None, None, None]
    frames = np.broadcast_to(frames, (length, 2, 3, 3)).copy()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observation/camera/head", data=frames)
        handle.create_dataset(
            "observation/state", data=np.arange(length, dtype=np.float32)[:, None]
        )
        handle.create_dataset("action", data=np.arange(length, dtype=np.float32)[:, None])
        handle.create_dataset(
            "language", data=np.asarray([b"task"] * length, dtype="S4")
        )
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "path": path.name, "length": length}) + "\n",
        encoding="utf-8",
    )

    unstrided = NativeHDF5Dataset(
        tmp_path,
        temporal_packing=True,
        visual_prompt="none",
        history_num_frames=0,
        index_cache=False,
    )
    strided = NativeHDF5Dataset(
        tmp_path,
        temporal_packing=True,
        visual_prompt="none",
        history_num_frames=0,
        step_stride=4,
        index_cache_dir=meta,
    )

    assert len(strided) < len(unstrided)
    assert all(anchor % 4 == 0 for _, anchor in strided.index)
    labels = np.asarray(["task"] * length, dtype=object)
    packer = LanguageTemporalPacker(max_chunk_size=4)
    expected_progress = np.asarray(
        [
            float(packer.indices(anchor, labels).video[0]) / length
            for _, anchor in strided.index
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        strided.sampling_progress(np.arange(len(strided))),
        expected_progress,
    )
    anchor_progress = {
        anchor: progress
        for (_, anchor), progress in zip(
            strided.index,
            expected_progress,
            strict=True,
        )
    }
    assert anchor_progress[48] == 0.0
    cache_path = strided._index_cache_path()
    assert cache_path is not None
    assert cache_path.is_file()

    cached = NativeHDF5Dataset(
        tmp_path,
        temporal_packing=True,
        visual_prompt="none",
        history_num_frames=0,
        step_stride=4,
        index_cache_dir=meta,
    )
    assert cached.index == strided.index
    np.testing.assert_allclose(
        cached.sampling_progress(np.arange(len(cached))),
        expected_progress,
    )


def test_native_mixture_schedule_resume_and_rank_partition():
    children = [[_event(index) for index in range(20)], [_event(100 + index) for index in range(20)]]
    kwargs = {
        "mixture_weights": [1, 3],
        "shard_size": 10,
        "shard_sampling_rate": 0.1,
        "num_shards_to_sample": 12,
        "seed": 7,
    }
    uninterrupted = NativeShardedMixtureDataset(children, **kwargs)
    iterator = iter(uninterrupted)
    prefix = [next(iterator).event_id for _ in range(5)]
    state = uninterrupted.state_dict()
    suffix = [sample.event_id for sample in iterator]

    resumed = NativeShardedMixtureDataset(children, **kwargs)
    resumed.load_state_dict(state)
    assert [sample.event_id for sample in resumed] == suffix
    assert len(prefix) + len(suffix) == len(uninterrupted)

    rank0 = NativeShardedMixtureDataset(children, rank=0, world_size=2, **kwargs)
    rank1 = NativeShardedMixtureDataset(children, rank=1, world_size=2, **kwargs)
    combined = len(list(rank0)) + len(list(rank1))
    assert combined == len(NativeShardedMixtureDataset(children, **kwargs))


def test_native_mixture_reads_samples_only_from_whole_shard_cache():
    class CachedChild:
        def __init__(self):
            self.samples = [_event(index) for index in range(12)]
            self.loads = []

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            raise AssertionError("sharded HDF5 samples must come from the shard cache")

        def shard_ranges(self, shard_size):
            assert shard_size == 10
            # Complete episodes: 0..6 and 7..11 are never split.
            return ((0, 7), (7, 12))

        def load_shard(self, start, stop):
            self.loads.append((start, stop))
            return self.samples[start:stop]

        def get_from_shard(self, index, shard):
            start = 0 if index < 7 else 7
            return shard[index - start]

    child = CachedChild()
    mixture = NativeShardedMixtureDataset(
        [child],
        shard_size=10,
        shard_sampling_rate=0.2,
        num_shards_to_sample=4,
        seed=9,
    )
    samples = list(mixture)
    assert len(samples) == 8
    assert len(child.loads) == 4
    assert set(child.loads) <= {(0, 7), (7, 12)}


def test_native_mixture_weights_are_source_level_for_unequal_lengths():
    children = [
        [_event(index) for index in range(15)],
        [_event(100 + index) for index in range(35)],
    ]
    mixture = NativeShardedMixtureDataset(
        children,
        mixture_weights=[1, 1],
        shard_size=10,
        num_shards_to_sample=10,
    )
    source_probabilities = np.zeros(2, dtype=np.float64)
    for shard, probability in zip(
        mixture.shards, mixture.shard_weights, strict=True
    ):
        source_probabilities[shard.dataset_index] += probability
    np.testing.assert_allclose(source_probabilities, [0.5, 0.5])


def test_exec_early_sampling_is_deterministic():
    child = [[_event(index) for index in range(100)]]
    kwargs = {
        "shard_size": 100,
        "shard_sampling_rate": 0.1,
        "num_shards_to_sample": 1,
        "seed": 11,
        "exec_early_sampling_enabled": True,
        "exec_early_ratio": 0.2,
        "exec_early_weight": 10,
    }
    first = [sample.event_id for sample in NativeShardedMixtureDataset(child, **kwargs)]
    second = [sample.event_id for sample in NativeShardedMixtureDataset(child, **kwargs)]
    assert first == second
    assert sum(int(value.split("-")[-1]) < 20 for value in first) >= 5


def test_exec_early_sampling_uses_sample_progress_not_shard_position():
    class ProgressDataset(list):
        def sampling_progress(self, indices):
            progress = np.concatenate(
                [
                    np.full(80, 0.8, dtype=np.float64),
                    np.full(20, 0.1, dtype=np.float64),
                ]
            )
            return progress[indices]

    child = ProgressDataset([_event(index) for index in range(100)])
    sampled = list(
        NativeShardedMixtureDataset(
            [child],
            shard_size=100,
            shard_sampling_rate=0.1,
            num_shards_to_sample=1,
            seed=11,
            exec_early_sampling_enabled=True,
            exec_early_ratio=0.2,
            exec_early_weight=100,
        )
    )
    assert sum(int(sample.event_id.split("-")[-1]) >= 80 for sample in sampled) >= 9
