from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from worldscape_policy.data import ConditionMode, TrainingBatch, VisualPromptMetadata
from worldscape_policy.cli.train import (
    _apply_freezing,
    _checkpoint_config_metadata,
    _checkpoint_path,
    _fast_forward_data,
    _prepare_resume_data_loader_config,
    _persist_freeze_report,
    _reject_compat_targets,
    _validate_planning_vlm_trainable,
    _validate_resume_data_loader,
)
from worldscape_policy.checkpoint.weights_io import load_checkpoint_state_dict
from worldscape_policy.training.objective import CompositeObjective
from worldscape_policy.training.prompt_schedule import PromptSchedule, Stage
from worldscape_policy.visual_mosaic import (
    prepare_diffusion_mosaic,
    prepare_diffusion_robotwin_concat,
)
from worldscape_policy.training.trainer import (
    ModelReadyTrainingBatch,
    NativeTrainer,
    NativeWan22BatchAdapter,
    validate_homogeneous_mode,
)
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)
from worldscape_policy.wam.wan22 import Wan22TrainingInputs


class _TinyPolicy(nn.Module):
    configured_mode = InteractionMode.INTERACTIVE

    def __init__(self) -> None:
        super().__init__()
        self.video_scale = nn.Parameter(torch.tensor(0.0))
        self.action_scale = nn.Parameter(torch.tensor(0.0))

    def training_forward(self, **kwargs) -> WorldActionOutput:
        return WorldActionOutput(
            video_velocity=self.video_scale.expand_as(kwargs["noisy_video"]),
            action_velocity=self.action_scale.expand_as(kwargs["noisy_action"]),
        )


class _NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class _NonfiniteGradientPolicy(_TinyPolicy):
    def training_forward(self, **kwargs) -> WorldActionOutput:
        return WorldActionOutput(
            video_velocity=self.video_scale.expand_as(kwargs["noisy_video"]),
            action_velocity=_NonfiniteBackward.apply(self.action_scale).expand_as(
                kwargs["noisy_action"]
            ),
        )


class _LargeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, 10_000.0)


class _LargeGradientPolicy(_TinyPolicy):
    def training_forward(self, **kwargs) -> WorldActionOutput:
        return WorldActionOutput(
            video_velocity=self.video_scale.expand_as(kwargs["noisy_video"]),
            action_velocity=_LargeBackward.apply(self.action_scale).expand_as(
                kwargs["noisy_action"]
            ),
        )


class _RecordingMixedPolicy(nn.Module):
    configured_mode = None
    training_supported_modes = (
        InteractionMode.AUTO,
        InteractionMode.INTERACTIVE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.calls: list[tuple[InteractionMode, str, int]] = []

    def training_forward(self, **kwargs) -> WorldActionOutput:
        prompts = kwargs["prompts"]
        if prompts.goal_images is not None:
            condition = "goal"
            marker = 1.0
        elif prompts.demo_videos is not None:
            condition = "video"
            marker = 2.0
        else:
            condition = "text"
            marker = 0.0
        self.calls.append(
            (
                InteractionMode.parse(kwargs["mode"]),
                condition,
                kwargs["clean_action"].shape[0],
            )
        )
        return WorldActionOutput(
            video_velocity=(self.scale + marker).expand_as(kwargs["noisy_video"]),
            action_velocity=(self.scale + marker).expand_as(kwargs["noisy_action"]),
        )


class _TinyNoiseKernel:
    def prepare_training_inputs(
        self,
        *,
        clean_video_latents,
        clean_action,
        generator,
    ) -> Wan22TrainingInputs:
        video_target = torch.rand(
            clean_video_latents.shape, generator=generator
        ) + 0.5
        action_target = torch.rand(clean_action.shape, generator=generator) + 0.5
        return Wan22TrainingInputs(
            noisy_video=torch.zeros_like(clean_video_latents),
            noisy_action=torch.zeros_like(clean_action),
            video_timestep=torch.zeros(
                clean_video_latents.shape[0], clean_video_latents.shape[2]
            ),
            action_timestep=torch.zeros(clean_action.shape[:2]),
            video_velocity_target=video_target,
            action_velocity_target=action_target,
        )


class _FakeDeepSpeedEngine:
    def __init__(self, policy: nn.Module) -> None:
        self.policy = policy
        self.save_calls: list[tuple[str, str]] = []
        self.load_calls: list[tuple[str, str]] = []
        self.export_calls: list[tuple[str, str]] = []

    def save_checkpoint(self, directory: str, *, tag: str):
        self.save_calls.append((directory, tag))
        target = Path(directory) / tag
        target.mkdir(parents=True)
        torch.save({"optimizer_shard": True}, target / "zero-rank-0.pt")
        return True

    def load_checkpoint(self, directory: str, *, tag: str, **kwargs):
        assert kwargs == {
            "load_module_strict": True,
            "load_optimizer_states": True,
            "load_lr_scheduler_states": False,
        }
        self.load_calls.append((directory, tag))
        return str(Path(directory) / tag), {}

    def save_16bit_model(self, directory: str, *, save_filename: str):
        self.export_calls.append((directory, save_filename))
        target = Path(directory)
        target.mkdir(parents=True)
        torch.save(
            {
                f"policy.{key}": value.detach().cpu().clone()
                for key, value in self.policy.state_dict().items()
            },
            target / save_filename,
        )
        return True


class _NativeFreezeTree(nn.Module):
    configured_mode = InteractionMode.AUTO

    def __init__(self) -> None:
        super().__init__()
        self.condition_router = nn.Module()
        self.condition_router.auto = nn.Module()
        self.condition_router.auto.vlm = nn.Linear(1, 1)
        self.condition_router.auto.projector = nn.Linear(1, 1)
        self.condition_router.auto.event_memory = nn.Linear(1, 1)
        self.condition_router.interactive = nn.Module()
        self.condition_router.interactive.t5 = nn.Linear(1, 1)
        self.condition_router.interactive.projector = nn.Linear(1, 1)
        self.visual_memory = nn.Module()
        self.visual_memory.codec = nn.Module()
        self.visual_memory.codec.vae = nn.Linear(1, 1)
        self.wam = nn.Module()
        self.wam.image_encoder = nn.Linear(1, 1)
        self.wam.core = nn.Module()
        self.wam.core.action_encoder = nn.Linear(1, 1)
        self.wam.core.state_encoder = nn.Linear(1, 1)
        self.wam.core.action_decoder = nn.Linear(1, 1)


def _batch() -> ModelReadyTrainingBatch:
    images = torch.zeros(2, 2, 1, 3, 2, 2)
    return ModelReadyTrainingBatch(
        mode=InteractionMode.INTERACTIVE,
        observation=ObservationBatch(
            images=images,
            head_view=images[:, :1, 0],
            proprioception=torch.zeros(2, 2, 3),
            embodiment_id=torch.zeros(2, dtype=torch.long),
        ),
        prompts=PromptBatch(language_instruction=["move", "move"]),
        clean_video=images[:, :, 0],
        clean_video_latents=torch.zeros(2, 1, 2, 1, 1),
        clean_action=torch.zeros(2, 2, 2),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
    )


def _trainer(seed: int = 19) -> NativeTrainer:
    policy = _TinyPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return NativeTrainer(
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        objective=CompositeObjective(),
        noise_kernel=_TinyNoiseKernel(),
        generator=torch.Generator().manual_seed(seed),
        config_metadata={"recipe": "tiny", "mode": "interactive"},
    )


def _scheduled_trainer(seed: int = 19) -> NativeTrainer:
    trainer = _trainer(seed)
    trainer.policy.configured_mode = None
    trainer.prompt_schedule = PromptSchedule(
        [Stage(end=0.5, auto_ratio=0.0), Stage(end=1.0, auto_ratio=1.0)]
    )
    trainer.max_steps = 2
    return trainer


def test_native_trainer_short_overfit_decreases_loss() -> None:
    trainer = _trainer()
    first = trainer.train_step(_batch())["loss"]
    losses = [trainer.train_step(_batch())["loss"] for _ in range(30)]

    # The fake kernel deliberately samples a different target every step, so
    # its irreducible variance prevents convergence to zero.  The gate is a
    # material loss reduction, not memorisation of one fixed random target.
    assert sum(losses[-5:]) / 5 < first * 0.4
    assert trainer.step == 31
    assert trainer.last_training_step_time > 0
    assert 0 < trainer.last_model_forward_time <= trainer.last_training_step_time


def test_native_trainer_skips_nonfinite_gradient_step(caplog) -> None:
    policy = _NonfiniteGradientPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    trainer = NativeTrainer(
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        objective=CompositeObjective(),
        noise_kernel=_TinyNoiseKernel(),
        generator=torch.Generator().manual_seed(19),
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    initial_lr = optimizer.param_groups[0]["lr"]

    with caplog.at_level("ERROR"):
        metrics = trainer.train_step(_batch())

    assert metrics["gradient_finite"] == 0.0
    assert metrics["optimizer_update_applied"] == 0.0
    assert metrics["optimizer_step_skipped_nonfinite"] == 1.0
    assert optimizer.param_groups[0]["lr"] == initial_lr
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name])
        assert parameter.grad is None
    assert "parameter=action_scale" in caplog.text


def test_native_trainer_skips_finite_gradient_above_threshold(caplog) -> None:
    policy = _LargeGradientPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    trainer = NativeTrainer(
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        objective=CompositeObjective(),
        noise_kernel=_TinyNoiseKernel(),
        generator=torch.Generator().manual_seed(19),
        skip_grad_norm_threshold=1.0,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    initial_lr = optimizer.param_groups[0]["lr"]

    with caplog.at_level("ERROR"):
        metrics = trainer.train_step(_batch())

    assert metrics["gradient_finite"] == 1.0
    assert metrics["gradient_norm_threshold_active"] == 1.0
    assert metrics["gradient_norm_threshold_exceeded"] == 1.0
    assert metrics["optimizer_step_skipped_large_gradient"] == 1.0
    assert metrics["optimizer_update_applied"] == 0.0
    assert metrics["grad_norm"] > 1.0
    assert optimizer.param_groups[0]["lr"] == initial_lr
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name])
        assert parameter.grad is None
    assert "finite global gradient norm" in caplog.text
    assert "exceeds threshold 1" in caplog.text


def test_native_trainer_delays_finite_gradient_threshold() -> None:
    policy = _LargeGradientPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.08)
    trainer = NativeTrainer(
        policy=policy,
        optimizer=optimizer,
        objective=CompositeObjective(),
        noise_kernel=_TinyNoiseKernel(),
        generator=torch.Generator().manual_seed(19),
        max_steps=2,
        skip_grad_norm_threshold=1.0,
        skip_grad_norm_threshold_start_ratio=0.5,
    )

    warmup_metrics = trainer.train_step(_batch())
    after_warmup = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    guarded_metrics = trainer.train_step(_batch())

    assert warmup_metrics["gradient_norm_threshold_active"] == 0.0
    assert warmup_metrics["optimizer_update_applied"] == 1.0
    assert guarded_metrics["gradient_norm_threshold_active"] == 1.0
    assert guarded_metrics["gradient_norm_threshold_exceeded"] == 1.0
    assert guarded_metrics["optimizer_update_applied"] == 0.0
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, after_warmup[name])


def test_native_training_batch_uses_explicit_wan_adapter() -> None:
    native = TrainingBatch(
        episode_ids=("episode",),
        event_ids=("event",),
        observations={"video": torch.zeros(1, 2, 2, 2, 3)},
        observation_masks={"video": torch.ones(1, 2, dtype=torch.bool)},
        actions=torch.zeros(1, 2, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        robot_state=torch.zeros(1, 2, 3),
        robot_state_mask=torch.ones(1, 2, dtype=torch.bool),
        high_level_instructions=("move",),
        event_instructions=(None,),
        embodiments=("toy",),
        modes=(InteractionMode.INTERACTIVE,),
        mode_mask=torch.zeros(1, dtype=torch.bool),
    )
    trainer = _trainer()
    trainer.batch_adapter = NativeWan22BatchAdapter(
        video_latent_encoder=lambda video: video.movedim(2, 1),
        diffusion_video_preprocessor=lambda views: views[:, :, 0].float(),
        embodiment_ids={"toy": 0},
    )

    metrics = trainer.train_step(native)

    assert metrics["step"] == 1


def test_native_adapter_rejects_padding_and_accepts_mixed_prompts() -> None:
    native = TrainingBatch(
        episode_ids=("one", "two"),
        event_ids=("one", "two"),
        observations={"video": torch.zeros(2, 2, 2, 2, 3)},
        observation_masks={
            "video": torch.tensor([[True, True], [True, False]])
        },
        actions=torch.zeros(2, 2, 2),
        action_mask=torch.ones(2, 2, dtype=torch.bool),
        robot_state=torch.zeros(2, 2, 3),
        robot_state_mask=torch.ones(2, 2, dtype=torch.bool),
        high_level_instructions=("task one", "task two"),
        event_instructions=("subtask one", "subtask two"),
        embodiments=("toy", "toy"),
        task_ids=("move", "move"),
        session_ids=("session", "session"),
        modes=(InteractionMode.INTERACTIVE, InteractionMode.INTERACTIVE),
        mode_mask=torch.zeros(2, dtype=torch.bool),
    )
    adapter = NativeWan22BatchAdapter(
        video_latent_encoder=lambda video: video.movedim(2, 1),
        diffusion_video_preprocessor=lambda views: views[:, :, 0].float(),
        embodiment_ids={"toy": 0},
    )

    with pytest.raises(ValueError, match="cannot propagate"):
        adapter(native)

    native.observation_masks["video"].fill_(True)
    native.demo_videos = torch.zeros(2, 2, 2, 2, 3)
    native.demo_video_mask = torch.tensor(
        [[True, True], [False, False]], dtype=torch.bool
    )
    native.demo_prompt_metadata = (
        VisualPromptMetadata(
            task_id="move",
            embodiment="toy",
            source_episode_id="one",
            source_session_id="session",
        ),
        None,
    )
    ready = adapter(native)
    assert ready.condition_modes == (
        ConditionMode.VIDEO_TO_VA,
        ConditionMode.T2VA,
    )
    assert ready.condition_ids.tolist() == [2, 0]
    assert ready.prompts.vlm_planning_text == ["task one", "task two"]
    assert ready.prompts.language_instruction == ["subtask one", "subtask two"]
    assert ready.prompts.planning_labels_text == ["subtask one", "subtask two"]
    assert ready.prompts.demo_videos.shape == (2, 2, 3, 2, 2)
    goal_layout = adapter._visual_prompt_to_policy_layout(
        torch.zeros(2, 2, 2, 3),
        is_goal=True,
    )
    assert goal_layout.shape == (2, 1, 3, 2, 2)


def test_native_adapter_uses_mosaic_for_diffusion_and_head_for_vlm() -> None:
    videos = torch.zeros((1, 2, 3, 4, 8, 3), dtype=torch.uint8)
    videos[:, :, 0] = 10
    videos[:, :, 1] = 20
    videos[:, :, 2] = 30
    native = TrainingBatch(
        episode_ids=("episode",),
        event_ids=("event",),
        observations={"video": videos},
        observation_masks={"video": torch.ones(1, 2, dtype=torch.bool)},
        actions=torch.zeros(1, 2, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        robot_state=torch.zeros(1, 2, 3),
        robot_state_mask=torch.ones(1, 2, dtype=torch.bool),
        high_level_instructions=("move",),
        event_instructions=("reach",),
        embodiments=("toy",),
        modes=(InteractionMode.INTERACTIVE,),
        mode_mask=torch.zeros(1, dtype=torch.bool),
    )
    encoded = {}

    def encode(video):
        encoded["video"] = video
        return video.movedim(2, 1)

    adapter = NativeWan22BatchAdapter(
        video_latent_encoder=encode,
        diffusion_video_preprocessor=lambda views: prepare_diffusion_mosaic(
            views, input_range="uint8"
        ),
        embodiment_ids={"toy": 0},
    )

    ready = adapter(native)

    assert ready.clean_video_normalized is True
    assert ready.clean_video.shape == (1, 2, 3, 4, 8)
    assert torch.equal(encoded["video"], ready.clean_video)
    assert ready.observation.images.shape == (1, 2, 3, 3, 4, 8)
    assert torch.equal(
        ready.observation.head_view,
        torch.full((1, 1, 3, 4, 8), 10, dtype=torch.uint8),
    )


def test_native_adapter_uses_raw_views_only_for_robotwin_diffusion() -> None:
    processed = torch.zeros((1, 2, 3, 2, 8, 3), dtype=torch.uint8)
    raw = torch.zeros((1, 2, 3, 6, 10, 3), dtype=torch.uint8)
    processed[:, :, 0] = 10
    raw[:, :, 0] = 255
    native = TrainingBatch(
        episode_ids=("episode",),
        event_ids=("event",),
        observations={"video": processed, "video_raw": raw},
        observation_masks={
            "video": torch.ones(1, 2, dtype=torch.bool),
            "video_raw": torch.ones(1, 2, dtype=torch.bool),
        },
        actions=torch.zeros(1, 2, 2),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        robot_state=torch.zeros(1, 2, 3),
        robot_state_mask=torch.ones(1, 2, dtype=torch.bool),
        high_level_instructions=("move",),
        event_instructions=("reach",),
        embodiments=("toy",),
        modes=(InteractionMode.INTERACTIVE,),
        mode_mask=torch.zeros(1, dtype=torch.bool),
    )
    adapter = NativeWan22BatchAdapter(
        video_latent_encoder=lambda video: video.movedim(2, 1),
        diffusion_video_preprocessor=lambda views: (
            prepare_diffusion_robotwin_concat(views, input_range="uint8")
        ),
        embodiment_ids={"toy": 0},
        diffusion_image_key="video_raw",
        diffusion_view_layout="robotwin_concat",
    )

    ready = adapter(native)

    assert ready.observation.images.shape == (1, 2, 3, 3, 2, 8)
    assert ready.observation.head_view.shape == (1, 1, 3, 2, 8)
    assert ready.clean_video.shape == (1, 2, 3, 384, 320)
    assert torch.all(ready.clean_video[..., :256, :] == 1)


def test_native_trainer_routes_three_conditions_through_one_policy() -> None:
    policy = _RecordingMixedPolicy()
    trainer = NativeTrainer(
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters(), lr=0.01),
        objective=CompositeObjective(),
        noise_kernel=_TinyNoiseKernel(),
        generator=torch.Generator().manual_seed(7),
    )
    images = torch.zeros(3, 2, 1, 3, 2, 2)
    ready = ModelReadyTrainingBatch(
        mode=InteractionMode.INTERACTIVE,
        mode_mask=torch.tensor([False, True, False]),
        condition_modes=(
            ConditionMode.T2VA,
            ConditionMode.GOAL_IMAGE_TO_VA,
            ConditionMode.VIDEO_TO_VA,
        ),
        condition_ids=torch.tensor([0, 1, 2]),
        observation=ObservationBatch(
            images=images,
            head_view=images[:, :1, 0],
            proprioception=torch.zeros(3, 2, 3),
            embodiment_id=torch.zeros(3, dtype=torch.long),
        ),
        prompts=PromptBatch(
            vlm_planning_text=["text", "goal", "video"],
            language_instruction=["text", "goal", "video"],
            goal_images=torch.zeros(3, 3, 2, 2),
            goal_image_mask=torch.tensor([False, True, False]),
            demo_videos=torch.zeros(3, 2, 3, 2, 2),
            demo_video_mask=torch.tensor(
                [[False, False], [False, False], [True, True]]
            ),
            visual_prompt="mixed",
        ),
        clean_video=images[:, :, 0],
        clean_video_latents=torch.zeros(3, 1, 2, 1, 1),
        clean_action=torch.zeros(3, 2, 2),
        action_mask=torch.ones(3, 2, dtype=torch.bool),
    )

    trainer.train_step(ready)

    assert sorted(policy.calls, key=lambda item: item[1]) == [
        (InteractionMode.AUTO, "goal", 1),
        (InteractionMode.INTERACTIVE, "text", 1),
        (InteractionMode.INTERACTIVE, "video", 1),
    ]
    assert policy.scale.item() != 0.0


def test_data_fast_forward_restores_cyclic_position() -> None:
    loader = ["a", "b", "c"]
    iterator = _fast_forward_data(loader, iter(loader), 5)
    assert next(iterator) == "c"


def test_native_resume_rejects_worker_rng_and_prefetch_state() -> None:
    loader = torch.utils.data.DataLoader([1, 2], num_workers=1)

    with pytest.raises(ValueError, match="num_workers=0.*cannot be reconstructed"):
        _validate_resume_data_loader(loader)


def test_resume_preparation_disables_prefetch_workers() -> None:
    config = OmegaConf.create(
        {
            "training": {
                "resume": "/tmp/step-20.pt",
                "resume_mode": "exact",
            },
            "data_loader": {"num_workers": 2},
        }
    )

    _prepare_resume_data_loader_config(config)

    assert config.data_loader.num_workers == 0


def test_fast_resume_preserves_prefetch_workers() -> None:
    config = OmegaConf.create(
        {
            "training": {
                "resume": "/tmp/checkpoint-20",
                "resume_mode": "fast",
            },
            "data_loader": {"num_workers": 2},
        }
    )

    _prepare_resume_data_loader_config(config)

    assert config.data_loader.num_workers == 2


def test_checkpoint_metadata_ignores_resume_and_worker_runtime_choices() -> None:
    base = OmegaConf.create(
        {
            "pretrained_adapter_source_rows": {"agilex": 2},
            "model": {
                "checkpoint_dir": "/models/pretrained",
                "initialization": "native",
                "pretrained_action_adapter_index": 2,
                "shape": {"hidden": 4},
            },
            "training": {
                "resume": None,
                "resume_mode": "fast",
                "checkpoint_dir": "/output/first",
                "max_steps": 20,
            },
            "data_loader": {"num_workers": 2, "batch_size": 1},
        }
    )
    resumed = OmegaConf.create(
        {
            "pretrained_adapter_source_rows": {"agilex": 2},
            "model": {
                "checkpoint_dir": None,
                "initialization": "components",
                "pretrained_action_adapter_index": None,
                "shape": {"hidden": 4},
            },
            "training": {
                "resume": "/output/checkpoint-10",
                "resume_mode": "exact",
                "checkpoint_dir": "/output/second",
                "max_steps": 20,
            },
            "data_loader": {"num_workers": 0, "batch_size": 1},
        }
    )

    metadata = _checkpoint_config_metadata(base)
    assert metadata == _checkpoint_config_metadata(resumed)
    assert "pretrained_adapter_source_rows" not in metadata


def test_fresh_run_preserves_configured_workers() -> None:
    config = OmegaConf.create(
        {
            "training": {"resume": None},
            "data_loader": {"num_workers": 2},
        }
    )

    _prepare_resume_data_loader_config(config)

    assert config.data_loader.num_workers == 2


def test_native_save_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted = _trainer()
    uninterrupted.train_step(_batch())
    expected_metrics = uninterrupted.train_step(_batch())

    interrupted = _trainer()
    interrupted.train_step(_batch())
    checkpoint = interrupted.save_checkpoint(tmp_path / "step-1.pt")
    resumed = _trainer(seed=999)
    metadata = resumed.load_checkpoint(checkpoint)
    actual_metrics = resumed.train_step(_batch())

    assert metadata == {"recipe": "tiny", "mode": "interactive"}
    assert actual_metrics == pytest.approx(expected_metrics)
    assert resumed.step == uninterrupted.step == 2
    assert resumed.data_batches_consumed == uninterrupted.data_batches_consumed == 2
    for expected, actual in zip(
        uninterrupted.policy.parameters(), resumed.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_directory_checkpoint_is_portable_and_exactly_resumable(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    trainer.train_step(_batch())
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint-1")

    assert checkpoint.is_dir()
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "trainer_state.pt").is_file()
    assert (checkpoint / "rank-00000.pt").is_file()
    assert (checkpoint / ".complete").is_file()

    resumed = _trainer(seed=999)
    metadata = resumed.load_checkpoint(checkpoint)

    assert metadata == {"recipe": "tiny", "mode": "interactive"}
    assert resumed.step == trainer.step == 1
    for expected, actual in zip(
        trainer.policy.parameters(), resumed.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_directory_checkpoint_shards_and_strictly_resumes(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.checkpoint_max_shard_size = 1
    trainer.train_step(_batch())
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint-1")

    assert (checkpoint / "model.safetensors.index.json").is_file()
    assert len(list(checkpoint.glob("model-*-of-*.safetensors"))) == 2
    resumed = _trainer(seed=999)
    resumed.load_checkpoint(checkpoint)
    for expected, actual in zip(
        trainer.policy.parameters(), resumed.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_restores_attached_rank_local_data_state(tmp_path: Path) -> None:
    class StatefulData:
        def __init__(self, cursor: int = 0) -> None:
            self.cursor = cursor

        def state_dict(self):
            return {"version": 1, "cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = int(state["cursor"])

    trainer = _trainer()
    source = StatefulData(cursor=7)
    trainer.attach_data_source(source)
    checkpoint = trainer.save_checkpoint(tmp_path / "data-state.pt")

    resumed = _trainer(seed=999)
    restored = StatefulData()
    resumed.attach_data_source(restored)
    resumed.load_checkpoint(checkpoint)

    assert restored.cursor == 7
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["distributed"]["format_version"] == 1
    assert payload["distributed"]["rank_states"][0]["data"] == {
        "version": 1,
        "cursor": 7,
    }


def test_fast_resume_skips_attached_data_state(tmp_path: Path) -> None:
    class StatefulData:
        def __init__(self, cursor: int = 0) -> None:
            self.cursor = cursor

        def state_dict(self):
            return {"version": 1, "cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = int(state["cursor"])

    trainer = _trainer()
    trainer.attach_data_source(StatefulData(cursor=7))
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint-0")

    resumed = _trainer(seed=999)
    restored = StatefulData()
    resumed.attach_data_source(restored)
    resumed.load_checkpoint(checkpoint, restore_data_state=False)

    assert restored.cursor == 0


def test_deepspeed_checkpoint_validation_accepts_zero_optimizer_shape(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    checkpoint = trainer.save_checkpoint(tmp_path / "native.pt")
    payload = torch.load(checkpoint, weights_only=True)
    payload["optimizer"] = {
        "zero_stage": 2,
        "partition_count": 2,
        "single_partition_of_fp32_groups": [torch.ones(1)],
    }
    trainer.runtime.engine = object()
    trainer._validate_checkpoint(payload)


def test_deepspeed_directory_checkpoint_uses_engine_and_portable_policy(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer()
    engine = _FakeDeepSpeedEngine(trainer.policy)
    trainer.runtime.backend = "deepspeed"
    trainer.runtime.engine = engine
    trainer.checkpoint_max_shard_size = 1
    monkeypatch.setattr(
        trainer.runtime,
        "gather_rank_state",
        lambda state: pytest.fail("DeepSpeed must not gather Python rank objects"),
    )

    checkpoint = trainer.save_checkpoint(tmp_path / "step-0")

    assert checkpoint.is_dir()
    assert engine.save_calls == [
        (str(checkpoint / "deepspeed"), "checkpoint")
    ]
    assert engine.export_calls == [
        (str(checkpoint / ".policy-export"), "policy.pt")
    ]
    assert (checkpoint / ".complete").is_file()
    assert (checkpoint / "rank-00000.pt").is_file()
    metadata = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    sidecar = torch.load(
        checkpoint / "rank-00000.pt", map_location="cpu", weights_only=True
    )
    assert "optimizer" not in metadata
    assert "optimizer" not in sidecar

    assert (checkpoint / "model.safetensors.index.json").is_file()
    portable = load_checkpoint_state_dict(checkpoint)
    assert set(portable) == set(trainer.policy.state_dict())
    assert not any(key.startswith("policy.") for key in portable)
    standalone = _TinyPolicy()
    standalone.load_state_dict(portable, strict=True)
    for key, target in standalone.state_dict().items():
        assert target.shape == trainer.policy.state_dict()[key].shape
        assert target.dtype == trainer.policy.state_dict()[key].dtype

    metadata_result = trainer.load_checkpoint(checkpoint)
    assert metadata_result == {"recipe": "tiny", "mode": "interactive"}
    assert engine.load_calls == [
        (str(checkpoint / "deepspeed"), "checkpoint")
    ]


def test_deepspeed_checkpoint_rejects_incomplete_marker(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.runtime.backend = "deepspeed"
    trainer.runtime.engine = _FakeDeepSpeedEngine(trainer.policy)
    checkpoint = trainer.save_checkpoint(tmp_path / "step-0")
    (checkpoint / ".complete").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        trainer.load_checkpoint(checkpoint)


def test_deepspeed_checkpoint_rejects_world_size_and_schema_mismatch(
    tmp_path: Path,
) -> None:
    trainer = _trainer()
    trainer.runtime.backend = "deepspeed"
    engine = _FakeDeepSpeedEngine(trainer.policy)
    trainer.runtime.engine = engine
    checkpoint = trainer.save_checkpoint(tmp_path / "step-0")
    metadata_path = checkpoint / "trainer_state.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    metadata["world_size"] = 2
    torch.save(metadata, metadata_path)

    with pytest.raises(ValueError, match="world_size"):
        trainer.load_checkpoint(checkpoint)
    assert engine.load_calls == []

    metadata["world_size"] = 1
    metadata["unknown"] = True
    torch.save(metadata, metadata_path)
    with pytest.raises(ValueError, match="unexpected=.*unknown"):
        trainer.load_checkpoint(checkpoint)
    assert engine.load_calls == []


def test_checkpoint_naming_is_backend_independent(tmp_path: Path) -> None:
    assert _checkpoint_path(tmp_path, "checkpoint-3") == (
        tmp_path / "checkpoint-3"
    )


def test_prompt_schedule_switches_modes_and_checkpoints_position(
    tmp_path: Path,
) -> None:
    trainer = _scheduled_trainer()
    first = trainer.train_step(_batch())
    checkpoint = trainer.save_checkpoint(tmp_path / "scheduled.pt")
    payload = torch.load(checkpoint, weights_only=False)

    resumed = _scheduled_trainer(seed=999)
    resumed.load_checkpoint(checkpoint)
    second = resumed.train_step(_batch())

    assert first["prompt_schedule/auto"] == 0
    assert second["prompt_schedule/auto"] == 1
    assert payload["prompt_schedule"] == {"position": 1, "max_steps": 2}


def test_prompt_schedule_uses_rank_zero_routing_mask(monkeypatch) -> None:
    trainer = _scheduled_trainer()
    broadcasts: list[torch.Tensor] = []

    def broadcast(mask: torch.Tensor) -> torch.Tensor:
        broadcasts.append(mask.clone())
        return torch.ones_like(mask)

    monkeypatch.setattr(trainer.runtime, "broadcast_from_rank_zero", broadcast)
    scheduled = trainer._apply_prompt_schedule(_batch())["batch"]

    assert len(broadcasts) == 1
    assert scheduled.mode_mask is not None
    assert scheduled.mode_mask.tolist() == [True, True]


def test_optional_t5_prompt_template_wraps_subtask() -> None:
    trainer = _scheduled_trainer()
    trainer.t5_high_level_prompt_prob = 0.0
    trainer.t5_prompt_template = "before {instruction} after {instruction}"
    ready = replace(
        _batch(),
        prompts=PromptBatch(
            vlm_planning_text=["Fold Shirt", "Fold Shirt"],
            language_instruction=["Lift Left Sleeve", "Smooth Fabric"],
        ),
    )

    scheduled = trainer._apply_prompt_schedule(ready)["batch"]

    assert scheduled.prompts.vlm_planning_text == ["Fold Shirt", "Fold Shirt"]
    assert scheduled.prompts.language_instruction == [
        "before lift left sleeve after lift left sleeve",
        "before smooth fabric after smooth fabric",
    ]


def test_prompt_text_logging_shows_actual_auto_inputs(capsys) -> None:
    trainer = _scheduled_trainer()
    trainer.step = 1
    trainer.t5_high_level_prompt_prob = 0.0
    trainer.t5_prompt_template = "t5 {instruction}"
    trainer.vlm_prompt_template = "plan {task}"
    trainer.log_prompt_text = True
    ready = replace(
        _batch(),
        prompts=PromptBatch(
            vlm_planning_text=["Fold Shirt", "Fold Shirt"],
            language_instruction=["Lift Left Sleeve", "Smooth Fabric"],
        ),
    )

    trainer._apply_prompt_schedule(ready)

    output = capsys.readouterr().out
    assert '"mode": "auto"' in output
    assert '"vlm_planning": "plan Fold Shirt"' in output
    assert '"vlm_goal": "Fold Shirt"' in output
    assert '"t5": "t5 lift left sleeve"' in output


def test_projector_only_stage_forces_auto_and_restricts_trainability() -> None:
    trainer = _scheduled_trainer()
    trainer.projector_only_end = 0.5
    trainer.projector_module_paths = ("video_scale",)

    metrics = trainer.train_step(_batch())

    assert metrics["prompt_schedule/auto"] == 1
    assert metrics["prompt_schedule/projector_only"] == 1
    assert trainer.policy.video_scale.requires_grad
    assert not trainer.policy.action_scale.requires_grad


def test_semantic_forcing_skips_interactive_only_batch() -> None:
    trainer = _trainer()
    trainer.objective = CompositeObjective(semantic_forcing_weight=1.0)
    metrics = trainer.train_step(_batch())
    assert metrics["semantic_forcing/skipped"] == 1.0


def test_auxiliary_objectives_fail_closed_on_auto_when_targets_are_missing() -> None:
    trainer = _scheduled_trainer()
    trainer.step = 1
    trainer.objective = CompositeObjective(semantic_forcing_weight=1.0)
    with pytest.raises(ValueError, match="semantic prediction and target"):
        trainer.train_step(_batch())

    trainer = _trainer()
    trainer.objective = CompositeObjective(planning_ce_weight=1.0)
    with pytest.raises(ValueError, match="planning logits and labels"):
        trainer.train_step(_batch())


def test_native_resume_fails_closed_on_unknown_keys(tmp_path: Path) -> None:
    trainer = _trainer()
    checkpoint = trainer.save_checkpoint(tmp_path / "valid.pt")
    payload = torch.load(checkpoint, weights_only=False)
    payload["unknown"] = "must fail"
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)

    with pytest.raises(ValueError, match="unexpected=.*unknown"):
        trainer.load_checkpoint(invalid)


def test_mode_validation_rejects_mixed_and_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="homogeneous"):
        validate_homogeneous_mode(
            [InteractionMode.AUTO, InteractionMode.INTERACTIVE]
        )
    batch = _batch()
    object.__setattr__(batch, "mode", InteractionMode.AUTO)
    with pytest.raises(ValueError, match="checkpoint mode"):
        _trainer().train_step(batch)


def test_native_entrypoint_rejects_compat_action_head_target() -> None:
    config = OmegaConf.create(
        {
            "model": {
                "_target_": (
                    "worldscape_policy"
                    ".compat.training.WorldScapeWan22ActionHead"
                )
            }
        }
    )
    with pytest.raises(ValueError, match="forbidden compatibility"):
        _reject_compat_targets(OmegaConf.to_container(config))


def test_planning_supervision_requires_trainable_vlm() -> None:
    vlm = nn.Linear(2, 2)
    model = SimpleNamespace(
        condition_router=SimpleNamespace(
            auto=SimpleNamespace(vlm=SimpleNamespace(vlm=vlm))
        )
    )
    objective = CompositeObjective(planning_ce_weight=1.0)
    vlm.requires_grad_(False)
    with pytest.raises(ValueError, match="requires an unfrozen VLM"):
        _validate_planning_vlm_trainable(model, objective)

    vlm.requires_grad_(True)
    _validate_planning_vlm_trainable(model, objective)


def test_entrypoint_applies_and_persists_freeze_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _NativeFreezeTree()
    config = OmegaConf.create(
        {
            "freeze": {
                "strict": True,
                "config": {
                    "_target_": (
                        "worldscape_policy.training.freezing.NativeFreezeConfig"
                    ),
                    "t5": False,
                },
            }
        }
    )

    report = _apply_freezing(config, model, dual_prompt=False)
    _persist_freeze_report(report.as_dict(), tmp_path)

    assert not model.condition_router.auto.vlm.weight.requires_grad
    assert model.wam.core.action_encoder.weight.requires_grad
    assert "condition_router.interactive.t5.weight" in report.unused_trainable_names
    assert capsys.readouterr().out == ""
    assert (tmp_path / "trainability-report.json").is_file()
