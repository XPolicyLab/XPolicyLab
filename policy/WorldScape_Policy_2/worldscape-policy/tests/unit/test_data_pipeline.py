from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from worldscape_policy.data import (
    DATASETS,
    AuditedVisualPromptOverride,
    ConditionMode,
    DatasetRegistry,
    EventChunkSampler,
    EventSample,
    HistorySampler,
    LegacyContextAdapter,
    ModeSampler,
    NativeEventTransform,
    NativeHDF5Dataset,
    NativeLeRobotDataset,
    NativeTrainingCollator,
    NativeVideoAugmentation,
    PromptModalitySampler,
    VisualPromptMetadata,
    VisualPromptSampler,
)
from worldscape_policy.data.adapters.common import _record_text
from worldscape_policy.training.data import build_registered_loader
from worldscape_policy.cli.train import _fast_forward_data
from worldscape_policy.types import InteractionMode


def _sample(
    *,
    event_id: str = "event-1",
    length: int = 2,
    goal: bool = False,
    demo: bool = False,
    task_id: str = "tidy-table",
    embodiment: str = "eef",
    prompt_task_id: str | None = None,
    prompt_embodiment: str | None = None,
) -> EventSample:
    prompt_metadata = VisualPromptMetadata(
        task_id=prompt_task_id or task_id,
        embodiment=prompt_embodiment or embodiment,
        source_episode_id="episode-1",
        source_session_id="session-1",
    )
    return EventSample(
        episode_id="episode-1",
        event_id=event_id,
        observations={"video": np.zeros((length, 1, 4, 4, 3), dtype=np.uint8)},
        actions=np.ones((length, 3), dtype=np.float32),
        robot_state=np.ones((length, 4), dtype=np.float32),
        high_level_instruction="tidy the table",
        event_instruction="pick up cup",
        goal_image=np.ones((4, 4, 3), dtype=np.uint8) if goal else None,
        demo_video=np.ones((3, 4, 4, 3), dtype=np.uint8) if demo else None,
        history_head_frames=np.ones((3, 4, 4, 3), dtype=np.uint8),
        embodiment=embodiment,
        task_id=task_id,
        session_id="session-1",
        goal_prompt_metadata=prompt_metadata if goal else None,
        demo_prompt_metadata=prompt_metadata if demo else None,
    )


def test_combined_agilex_task_record_splits_high_level_and_subtask() -> None:
    record = {
        "task": (
            "task: Fold two shirts neatly., "
            "sub_task: Grasp the left sleeve., "
            "embodiment_tag: agilex_eef"
        )
    }

    assert _record_text(record, event=False) == "Fold two shirts neatly."
    assert _record_text(record, event=True) == "Grasp the left sleeve."


@pytest.mark.parametrize(
    ("legacy_mode", "has_goal", "has_demo"),
    [("uniform", False, True), ("last", True, False)],
)
def test_legacy_context_adapter_maps_to_explicit_prompt(
    legacy_mode, has_goal, has_demo
):
    context = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    sample = LegacyContextAdapter().adapt(
        {
            "trajectory_id": 7,
            "chunk_id": 2,
            "video": np.zeros((2, 1, 2, 2, 3), dtype=np.uint8),
            "video_ctx": context,
            "action": np.zeros((2, 3), dtype=np.float32),
            "state": np.zeros((2, 4), dtype=np.float32),
            "language": "move",
            "embodiment_tag": "eef",
        },
        context_sampling_mode=legacy_mode,
    )

    assert (sample.goal_image is not None) is has_goal
    assert (sample.demo_video is not None) is has_demo
    if has_goal:
        np.testing.assert_array_equal(sample.goal_image, context[-1])
    if has_demo:
        np.testing.assert_array_equal(sample.demo_video, context)
    metadata = sample.goal_prompt_metadata if has_goal else sample.demo_prompt_metadata
    assert metadata is not None
    assert metadata.trusted_same_sample
    assert metadata.source_episode_id == sample.episode_id
    assert metadata.source_session_id == sample.session_id


def test_schema_rejects_malformed_history():
    sample = _sample()
    malformed = EventSample(
        **{
            **sample.__dict__,
            "history_head_frames": np.zeros((4, 4, 3), dtype=np.uint8),
        }
    )
    with pytest.raises(ValueError, match="history_head_frames"):
        malformed.validate()


def test_explicit_samplers_are_bounded_and_deterministic():
    chunks = [_sample() for _ in range(5)]
    selected = EventChunkSampler(max_chunks=2).sample(
        chunks, rng=np.random.default_rng(4)
    )
    assert len(selected) == 2

    frames = np.arange(10, dtype=np.uint8)[:, None, None, None]
    history = HistorySampler(num_frames=3, stride=2).sample(frames)
    assert history[:, 0, 0, 0].tolist() == [5, 7, 9]

    assert (
        ModeSampler(auto_probability=1).sample(rng=np.random.default_rng(0))
        is InteractionMode.AUTO
    )
    prompt = VisualPromptSampler("goal").sample(_sample(goal=True, demo=True))
    assert prompt.goal_image is not None
    assert prompt.demo_video is None
    assert prompt.goal_metadata == _sample(goal=True).goal_prompt_metadata
    with pytest.raises(ValueError, match="goal, demo, or none"):
        VisualPromptSampler("available").sample(_sample(goal=True, demo=True))
    modality_sampler = PromptModalitySampler(
        modalities=("text", "goal", "demo"),
        probabilities=(0.2, 0.3, 0.5),
    )
    first_sequence = [
        modality_sampler.sample(rng=np.random.default_rng(seed))
        for seed in range(10)
    ]
    second_sequence = [
        modality_sampler.sample(rng=np.random.default_rng(seed))
        for seed in range(10)
    ]
    assert first_sequence == second_sequence


@pytest.mark.parametrize(
    ("field", "modality", "sample"),
    [
        (
            "embodiment",
            "goal",
            _sample(goal=True, prompt_embodiment="different-robot"),
        ),
        ("task", "demo", _sample(demo=True, prompt_task_id="different-task")),
    ],
)
def test_visual_prompt_sampler_rejects_incompatible_provenance(
    field, modality, sample
):
    with pytest.raises(ValueError, match=f"incompatible.*{field}"):
        VisualPromptSampler(modality).sample(sample)


def test_visual_prompt_override_must_be_audited_and_explicit():
    with pytest.raises(ValueError, match="audit_reason"):
        AuditedVisualPromptOverride(enabled=True)

    sample = _sample(goal=True, prompt_embodiment="different-robot")
    prompt = VisualPromptSampler(
        "goal",
        override=AuditedVisualPromptOverride(
            enabled=True, audit_reason="approved migration audit WSP-148"
        ),
    ).sample(sample)
    assert prompt.goal_image is not None
    assert (
        prompt.goal_metadata.override_audit_reason
        == "approved migration audit WSP-148"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("task_id", "other-task", "task"),
        ("embodiment", "other-robot", "embodiment"),
    ],
)
def test_collator_enforces_visual_prompt_provenance(field, value, match):
    transformed = NativeEventTransform()(_sample(goal=True), mode="auto")
    metadata = replace(transformed.goal_prompt_metadata, **{field: value})
    incompatible = replace(transformed, goal_prompt_metadata=metadata)

    with pytest.raises(ValueError, match=f"incompatible goal.*{match}"):
        NativeTrainingCollator()([incompatible])

    audited = replace(
        incompatible,
        goal_prompt_metadata=replace(
            metadata, override_audit_reason="approved exception WSP-201"
        ),
    )
    assert NativeTrainingCollator()([audited]).goal_images is not None


def test_collator_requires_metadata_and_validates_trusted_legacy_marker():
    transformed = NativeEventTransform()(_sample(goal=True), mode="auto")
    with pytest.raises(ValueError, match="prompt metadata"):
        NativeTrainingCollator()(
            [replace(transformed, goal_prompt_metadata=None)]
        )

    legacy = replace(
        transformed,
        goal_prompt_metadata=replace(
            transformed.goal_prompt_metadata,
            trusted_same_sample=True,
        ),
    )
    assert NativeTrainingCollator()([legacy]).goal_images is not None

    forged = replace(
        legacy,
        goal_prompt_metadata=replace(
            legacy.goal_prompt_metadata,
            source_episode_id="different-episode",
        ),
    )
    with pytest.raises(ValueError, match="trusted legacy goal prompt marker"):
        NativeTrainingCollator()([forged])


def test_native_transform_and_collator_pad_temporal_fields():
    transform = NativeEventTransform()
    first = transform(_sample(length=2, goal=True), mode="auto")
    second = transform(_sample(length=3, demo=True), mode="interactive")

    batch = NativeTrainingCollator()([first, second])

    assert batch.actions.shape == (2, 3, 3)
    assert batch.action_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.mode_mask.tolist() == [True, False]
    assert batch.condition_modes == (
        ConditionMode.GOAL_IMAGE_TO_VA,
        ConditionMode.VIDEO_TO_VA,
    )
    assert batch.condition_ids.tolist() == [1, 2]
    assert batch.goal_image_mask.tolist() == [True, False]
    assert batch.demo_video_mask.tolist() == [
        [False, False, False],
        [True, True, True],
    ]
    assert batch.observations["video"].dtype is torch.uint8
    assert isinstance(batch.goal_prompt_metadata, tuple)
    assert batch.goal_prompt_metadata[0] == first.goal_prompt_metadata
    assert batch.demo_prompt_metadata[1] == second.demo_prompt_metadata


def test_native_transform_preserves_unresized_video_for_diffusion():
    sample = replace(
        _sample(length=2),
        observation_valid_masks={"video": np.array([True, False])},
    )
    transform = NativeEventTransform(
        fixed_mode="auto",
        video_augmentation=NativeVideoAugmentation(
            training=False,
            width=8,
            height=2,
        ),
        raw_video_observation_key="video_raw",
    )

    transformed = transform(sample)

    assert transformed.observations["video"].shape == (2, 1, 2, 8, 3)
    np.testing.assert_array_equal(
        transformed.observations["video_raw"].numpy(),
        sample.observations["video"],
    )
    assert transformed.observation_valid_masks is not None
    torch.testing.assert_close(
        transformed.observation_valid_masks["video_raw"],
        transformed.observation_valid_masks["video"],
    )
    batch = NativeTrainingCollator()([transformed])
    assert batch.observation_masks["video_raw"].tolist() == [[True, False]]


def test_native_transform_requires_explicit_or_fixed_mode():
    with pytest.raises(ValueError, match="explicit mode or fixed_mode"):
        NativeEventTransform()(_sample())

    transformed = NativeEventTransform(fixed_mode="auto")(_sample())
    assert transformed.mode is InteractionMode.AUTO


def test_registered_loader_buckets_variable_temporal_lengths():
    name = "test_variable_length_bucket"

    class VariableLengthSamples(IterableDataset):
        def __iter__(self):
            yield from (
                _sample(event_id="two-a", length=2),
                _sample(event_id="three-a", length=3),
                _sample(event_id="two-b", length=2),
                _sample(event_id="three-b", length=3),
            )

    if name not in DATASETS.names():
        DATASETS.register(name, VariableLengthSamples)
    def build_loader():
        return build_registered_loader(
            dataset_name=name,
            mode="auto",
            batch_size=2,
            num_workers=0,
            bucket_by_length=True,
        )

    loader = build_loader()

    batches = list(loader)

    assert len(batches) == 2
    assert {batch.actions.shape[1] for batch in batches} == {2, 3}
    assert all(bool(batch.action_mask.all()) for batch in batches)
    assert all(
        bool(batch.observation_masks["video"].all()) for batch in batches
    )
    assert len(list(loader)) == 2

    interrupted = build_loader()
    first = next(iter(interrupted))
    state = interrupted.dataset.state_dict()
    assert first.event_ids == ("two-a", "two-b")
    resumed = build_loader()
    resumed.dataset.load_state_dict(state)
    second = next(iter(resumed))
    assert second.event_ids == ("three-a", "three-b")


def test_native_transform_samples_explicit_wam_condition_per_sample():
    sample = _sample(goal=True, demo=True)
    transform = NativeEventTransform(
        fixed_mode="auto",
        prompt_modality_sampler=PromptModalitySampler(
            modalities=("text", "goal", "demo"),
            probabilities=(1.0, 0.0, 0.0),
        ),
        seed=17,
    )
    text = transform(sample)
    assert text.condition_mode is ConditionMode.T2VA
    assert text.goal_image is None
    assert text.demo_video is None

    goal = NativeEventTransform(
        fixed_mode="auto",
        prompt_modality_sampler=PromptModalitySampler(
            modalities=("goal",),
        ),
    )(sample)
    assert goal.condition_mode is ConditionMode.GOAL_IMAGE_TO_VA
    assert goal.goal_image is not None
    assert goal.demo_video is None


def test_dataset_registry_is_explicit_and_rejects_duplicates():
    registry = DatasetRegistry()

    @registry.register("tiny")
    def build_tiny():
        return iter([_sample()])

    assert next(iter(registry.create("TINY"))).event_id == "event-1"
    with pytest.raises(ValueError, match="already registered"):
        registry.register("tiny", build_tiny)


def test_builtin_mixture_factories_wire_three_modes_and_demo_uniform(monkeypatch):
    from worldscape_policy.data import plugins

    class FakeDataset:
        def __init__(self, data_root, **kwargs):
            self.data_root = str(data_root)
            self.kwargs = kwargs

        def __len__(self):
            return 1

    monkeypatch.setattr(plugins, "NativeHDF5Dataset", FakeDataset)
    mixed = DATASETS.create(
        "worldscape_hdf5_mixed_pretrain",
        text_data_root="/text",
        goal_data_root="/goal",
        video_data_root="/video",
        num_shards_to_sample=1,
    )
    assert [child.kwargs["visual_prompt"] for child in mixed.datasets] == [
        "none",
        "goal",
        "demo",
    ]
    assert [
        child.kwargs["context_sampling_mode"] for child in mixed.datasets
    ] == ["none", "last", "uniform"]

    single = DATASETS.create(
        "worldscape_hdf5_demo",
        data_root="/demo",
        num_shards_to_sample=1,
    )
    assert single.data_root == "/demo"
    assert single.kwargs["context_sampling_mode"] == "uniform"

    mixture = DATASETS.create(
        "worldscape_hdf5_demo",
        data_root="/shell",
        shard_size=1000,
        num_shards_to_sample=1,
    )
    assert [child.data_root for child in mixture.datasets] == ["/shell"]

    goal = DATASETS.create(
        "worldscape_hdf5_goal",
        data_root="/goal",
        shard_size=10_000,
        shard_sampling_rate=0.1,
        num_shards_to_sample=3,
        seed=62,
    )
    assert [child.data_root for child in goal.datasets] == ["/goal"]
    assert goal.datasets[0].kwargs["visual_prompt"] == "goal"
    assert goal.shard_sampling_rate == 0.1

    text = DATASETS.create(
        "worldscape_hdf5_text",
        data_root="/text",
        shard_size=10_000,
        shard_sampling_rate=0.1,
        num_shards_to_sample=7,
        seed=62,
        training=True,
    )
    assert len(text.datasets) == 1
    assert text.datasets[0].data_root == "/text"
    assert text.datasets[0].kwargs["seed"] == 62
    assert text.shard_size == 10_000
    assert text.shard_sampling_rate == 0.1
    assert text.num_shards_to_sample == 7


def test_native_hdf5_dataset_reads_repository_episode_format(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode_000.hdf5"
    frames = np.arange(8 * 3 * 4 * 3, dtype=np.uint8).reshape(8, 3, 4, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "is_exec", data=[False, False, True, True, True, True, True, True]
        )
        handle.create_dataset("observation/camera/head", data=frames)
        handle.create_dataset("observation/camera/wrist", data=frames + 1)
        handle.create_dataset(
            "observation/state", data=np.ones((8, 4), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.ones((8, 3), dtype=np.float32))
        handle.create_dataset("language", data=np.bytes_("build a tower"))

    dataset = NativeHDF5Dataset(
        tmp_path,
        visual_prompt="demo",
        action_horizon=2,
        max_event_chunks=2,
        history_num_frames=3,
        history_stride=2,
        seed=4,
    )
    sample = dataset[0]

    sample.validate()
    assert sample.observations["video"].shape == (4, 2, 3, 4, 3)
    assert sample.actions.shape == (4, 3)
    np.testing.assert_array_equal(sample.demo_video, frames[:2])
    assert sample.goal_image is None
    assert sample.history_head_frames.shape == (3, 3, 4, 3)


def test_dataset_adapter_rejects_cross_embodiment_prompt(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((2, 1, 3, 4, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((2, 4), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((2, 3), dtype=np.float32))
        handle.create_dataset("goal_image", data=np.ones((3, 4, 3), dtype=np.uint8))
        handle.create_dataset("task_id", data=np.bytes_("stack-blocks"))
        handle.create_dataset("prompt/goal/embodiment", data=np.bytes_("other-robot"))

    dataset = NativeHDF5Dataset(
        tmp_path, visual_prompt="goal", embodiment="eef", history_num_frames=1
    )
    with pytest.raises(ValueError, match="incompatible goal.*embodiment"):
        dataset[0]


def test_dataset_adapter_allows_only_audited_override(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((2, 1, 3, 4, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((2, 4), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((2, 3), dtype=np.float32))
        handle.create_dataset("goal_image", data=np.ones((3, 4, 3), dtype=np.uint8))
        handle.create_dataset("task_id", data=np.bytes_("stack-blocks"))
        handle.create_dataset("prompt/goal/task_id", data=np.bytes_("sort-blocks"))

    with pytest.raises(ValueError, match="incompatible goal.*task"):
        NativeHDF5Dataset(tmp_path, visual_prompt="goal")[0]

    with pytest.raises(ValueError, match="audit_reason"):
        NativeHDF5Dataset(
            tmp_path,
            visual_prompt="goal",
            allow_incompatible_visual_prompts=True,
        )

    sample = NativeHDF5Dataset(
        tmp_path,
        visual_prompt="goal",
        allow_incompatible_visual_prompts=True,
        visual_prompt_override_audit_reason="reviewed exception WSP-149",
        history_num_frames=1,
    )[0]
    assert sample.goal_prompt_metadata.task_id == "sort-blocks"


def test_builtin_registry_and_loader_need_no_external_plugin(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "episode.hdf5"
    length = 30
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observation/video",
            data=np.zeros((length, 1, 3, 4, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observation/state", data=np.zeros((length, 4), dtype=np.float32)
        )
        handle.create_dataset("action", data=np.zeros((length, 3), dtype=np.float32))
        handle.create_dataset(
            "language", data=np.asarray([b"stack blocks"] * length, dtype="S12")
        )
        handle.create_dataset(
            "is_exec",
            data=np.asarray([False, False, True] + [True] * (length - 3), dtype=bool),
        )
        handle.create_dataset("goal_image", data=np.ones((3, 4, 3), dtype=np.uint8))

    assert {
        "worldscape_hdf5_demo",
        "worldscape_hdf5_goal",
        "worldscape_hdf5_text",
        "worldscape_lerobot_demo",
        "worldscape_lerobot_goal",
        "worldscape_lerobot_text",
    }.issubset(DATASETS.names())
    loader = build_registered_loader(
        dataset_name="worldscape_hdf5_goal",
        dataset_kwargs={"data_root": tmp_path, "history_num_frames": 1},
        mode="auto",
        batch_size=1,
    )

    batch = next(iter(loader))
    assert batch.goal_image_mask.tolist() == [True]
    assert batch.demo_videos is None


def test_native_lerobot_dataset_reads_v2_parquet_episode(tmp_path, monkeypatch):
    pandas = pytest.importorskip("pandas")
    meta = tmp_path / "meta"
    data = tmp_path / "data" / "chunk-000"
    meta.mkdir()
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        """{
          "total_episodes": 1,
          "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
          "features": {
            "observation.images.head": {"dtype": "image"},
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"}
          }
        }"""
    )
    (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 4}\n')
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "stack blocks"}\n')
    parquet = data / "episode_000000.parquet"
    parquet.touch()
    images = [np.full((3, 4, 3), index, dtype=np.uint8) for index in range(4)]
    episode_frame = pandas.DataFrame(
        {
            "observation.images.head": images,
            "observation.state": [
                np.full(4, index, dtype=np.float32) for index in range(4)
            ],
            "action": [np.full(3, index, dtype=np.float32) for index in range(4)],
            "is_exec": [False, True, True, True],
            "task_index": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(pandas, "read_parquet", lambda _: episode_frame)

    sample = NativeLeRobotDataset(
        tmp_path, visual_prompt="goal", history_num_frames=2, history_stride=1
    )[0]

    sample.validate()
    assert sample.high_level_instruction == "stack blocks"
    assert sample.observations["video"].shape == (3, 1, 3, 4, 3)
    np.testing.assert_array_equal(sample.goal_image, images[0])
    assert sample.demo_video is None


def test_lerobot_single_episode_instruction_uses_first_task() -> None:
    dataset = object.__new__(NativeLeRobotDataset)
    dataset.single_high_level_instruction_per_episode = True
    arrays = {
        "high_level_instruction": np.asarray(["frame task"] * 3, dtype=object),
        "event_instruction": np.asarray(["frame event"] * 3, dtype=object),
    }

    dataset._apply_single_episode_instruction(
        arrays,
        episode={
            "episode_index": 7,
            "tasks": ["canonical first task", "unused paraphrase"],
        },
        trajectory_length=3,
    )

    assert arrays["high_level_instruction"].tolist() == ["canonical first task"] * 3
    assert arrays["event_instruction"].tolist() == ["canonical first task"] * 3


def test_registered_loader_imports_plugin_before_dataset_creation(monkeypatch):
    dataset_name = "unit-plugin-map"

    def import_plugin(module_name):
        assert module_name == "tests.fake_native_dataset_plugin"
        DATASETS.register(dataset_name, lambda: [_sample(event_id="plugin")])

    monkeypatch.setattr(
        "worldscape_policy.training.data.importlib.import_module", import_plugin
    )

    loader = build_registered_loader(
        dataset_plugin="tests.fake_native_dataset_plugin",
        dataset_name=dataset_name,
        mode="auto",
        batch_size=1,
    )

    assert next(iter(loader)).event_ids == ("plugin",)


def test_registered_loader_reports_plugin_that_did_not_register(monkeypatch):
    monkeypatch.setattr(
        "worldscape_policy.training.data.importlib.import_module",
        lambda _: object(),
    )

    with pytest.raises(RuntimeError, match="did not register.*missing-native"):
        build_registered_loader(
            dataset_plugin="tests.empty_native_dataset_plugin",
            dataset_name="missing-native",
            mode="auto",
            batch_size=1,
        )


def test_shuffled_map_loader_resume_reproduces_epoch_permutations(monkeypatch):
    dataset_name = "unit-shuffled-map"
    samples = [_sample(event_id=f"event-{index}") for index in range(5)]

    def import_plugin(_):
        if dataset_name not in DATASETS.names():
            DATASETS.register(dataset_name, lambda: samples)

    monkeypatch.setattr(
        "worldscape_policy.training.data.importlib.import_module", import_plugin
    )

    def make_loader():
        return build_registered_loader(
            dataset_plugin="tests.shuffled_native_dataset_plugin",
            dataset_name=dataset_name,
            mode="auto",
            batch_size=1,
            shuffle=True,
            seed=314,
        )

    uninterrupted_loader = make_loader()
    uninterrupted_iterator = iter(uninterrupted_loader)
    expected = []
    for _ in range(13):
        try:
            batch = next(uninterrupted_iterator)
        except StopIteration:
            uninterrupted_iterator = iter(uninterrupted_loader)
            batch = next(uninterrupted_iterator)
        expected.append(batch.event_ids[0])

    resumed_loader = make_loader()
    resumed_iterator = _fast_forward_data(resumed_loader, iter(resumed_loader), 8)
    actual = []
    for _ in range(5):
        try:
            batch = next(resumed_iterator)
        except StopIteration:
            resumed_iterator = iter(resumed_loader)
            batch = next(resumed_iterator)
        actual.append(batch.event_ids[0])

    assert actual == expected[8:]


def test_iterable_loader_shards_samples_across_workers(monkeypatch):
    dataset_name = "unit-worker-sharded-iterable"

    class SampleIterable(IterableDataset):
        def __iter__(self):
            return iter([_sample(event_id=f"event-{index}") for index in range(12)])

    def import_plugin(_):
        DATASETS.register(dataset_name, SampleIterable)

    monkeypatch.setattr(
        "worldscape_policy.training.data.importlib.import_module", import_plugin
    )
    loader = build_registered_loader(
        dataset_plugin="tests.iterable_native_dataset_plugin",
        dataset_name=dataset_name,
        mode="interactive",
        batch_size=1,
        num_workers=2,
        shuffle=False,
        seed=7,
    )

    event_ids = [batch.event_ids[0] for batch in loader]

    assert sorted(event_ids) == sorted(f"event-{index}" for index in range(12))
    assert len(event_ids) == len(set(event_ids)) == 12
