from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from worldscape_policy.cli.train import run_config
from worldscape_policy.types import InteractionMode, WorldActionOutput
from worldscape_policy.wam.wan22 import Wan22TrainingInputs


class _Codec(nn.Module):
    def encode_visual(self, video: torch.Tensor) -> torch.Tensor:
        return video.float().movedim(2, 1) / 255.0

    def prepare_diffusion_video(self, views: torch.Tensor) -> torch.Tensor:
        return views[:, :, 0].float() / 255.0

    def encode_normalized(self, video: torch.Tensor) -> torch.Tensor:
        return video.movedim(2, 1)


class _VisualMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codec = _Codec()


class FakeNumericalPolicy(nn.Module):
    configured_mode = InteractionMode.AUTO
    training_supported_modes = (
        InteractionMode.AUTO,
        InteractionMode.INTERACTIVE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.video_scale = nn.Parameter(torch.tensor(0.1))
        self.action_scale = nn.Parameter(torch.tensor(0.1))
        self.semantic = nn.Parameter(torch.zeros(1, 3, 4))
        self.planning = nn.Parameter(torch.zeros(1, 2, 5))
        self.visual_memory = _VisualMemory()
        self.register_buffer("auto_examples", torch.zeros((), dtype=torch.long))
        self.register_buffer("interactive_examples", torch.zeros((), dtype=torch.long))

    def training_forward(self, **kwargs) -> WorldActionOutput:
        batch_size = kwargs["clean_action"].shape[0]
        mode = InteractionMode.parse(kwargs["mode"])
        metrics = {}
        if mode is InteractionMode.AUTO:
            self.auto_examples.add_(batch_size)
            labels_text = kwargs["prompts"].planning_labels_text
            if labels_text is None or len(labels_text) != batch_size:
                raise AssertionError("planning label text did not reach Auto conditioning")
            metrics = {
                "semantic_prediction": self.semantic.expand(batch_size, -1, -1),
                "planning_logits": self.planning.expand(batch_size, -1, -1),
            }
        else:
            self.interactive_examples.add_(batch_size)
        return WorldActionOutput(
            video_velocity=self.video_scale.expand_as(kwargs["noisy_video"]),
            action_velocity=self.action_scale.expand_as(kwargs["noisy_action"]),
            metrics=metrics,
        )


class FakeNoiseKernel:
    def prepare_training_inputs(
        self,
        *,
        clean_video_latents: torch.Tensor,
        clean_action: torch.Tensor,
        generator: torch.Generator,
    ) -> Wan22TrainingInputs:
        video_target = torch.rand(
            clean_video_latents.shape, generator=generator
        )
        action_target = torch.rand(clean_action.shape, generator=generator)
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


def _write_episode(path: Path, offset: int) -> None:
    frames = (
        np.arange(4 * 2 * 4 * 5 * 3, dtype=np.uint16)
        .reshape(4, 2, 4, 5, 3)
        .astype(np.uint8)
        + offset
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observation/video", data=frames)
        handle.create_dataset(
            "observation/state",
            data=np.arange(16, dtype=np.float32).reshape(4, 4),
        )
        handle.create_dataset(
            "action", data=np.arange(12, dtype=np.float32).reshape(4, 3)
        )
        handle.create_dataset("event_instruction", data=np.bytes_("place the cup"))
        handle.create_dataset("planning_labels_text", data=np.bytes_("reach; grasp; place"))
        handle.create_dataset("planning_labels", data=np.asarray([1, 3], dtype=np.int64))
        handle.create_dataset(
            "semantic_target",
            data=np.arange(12, dtype=np.float32).reshape(3, 4) / 12,
        )
        handle.create_dataset(
            "semantic_mask", data=np.asarray([True, True, True])
        )
        handle.create_dataset("goal_image", data=frames[-1, 0])


def _config(data_root: Path, checkpoint_dir: Path, resume: Path | None = None):
    target = "tests.integration.test_train_step"
    return OmegaConf.create(
        {
            "model": {"_target_": f"{target}.FakeNumericalPolicy"},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.05},
            "noise_kernel": {"_target_": f"{target}.FakeNoiseKernel"},
            "batch_adapter": {
                "_target_": (
                    "worldscape_policy.training.trainer.NativeWan22BatchAdapter"
                ),
                "embodiment_ids": {"agilex": 0},
            },
            "data_loader": {
                "_target_": (
                    "worldscape_policy.training.data.build_registered_loader"
                ),
                "dataset_name": "worldscape_hdf5_goal",
                "dataset_kwargs": {
                    "data_root": str(data_root),
                    "action_horizon": 2,
                    "max_event_chunks": 1,
                    "history_num_frames": 1,
                    "temporal_packing": False,
                    "context_sampling_mode": None,
                },
                "mode": "auto",
                "batch_size": 2,
                "num_workers": 0,
            },
            "objective": {
                "_target_": (
                    "worldscape_policy.training.objective.CompositeObjective"
                ),
                "semantic_forcing_weight": 0.2,
                "planning_ce_weight": 0.3,
            },
            "prompt_schedule": {
                "enabled": True,
                "projector_only_end": 0.0,
                "schedule": {
                    "_target_": (
                        "worldscape_policy.training.prompt_schedule.PromptSchedule"
                    ),
                    "stages": [
                        {
                            "_target_": (
                                "worldscape_policy.training.prompt_schedule.Stage"
                            ),
                            "end": 1.0,
                            "auto_ratio": 0.5,
                        }
                    ],
                },
            },
            "freeze": {
                "strict": False,
                "config": {
                    "_target_": (
                        "worldscape_policy.training.freezing.NativeFreezeConfig"
                    ),
                    "vlm": False,
                    "t5": False,
                    "vae": False,
                },
            },
            "training": {
                "max_steps": 2,
                "save_every": 1,
                "save_at_end": True,
                "checkpoint_dir": str(checkpoint_dir),
                "resume": None if resume is None else str(resume),
                "resume_mode": "exact",
            },
        }
    )


def test_native_cpu_train_step_and_exact_save_resume(tmp_path: Path) -> None:
    data_root = tmp_path / "episodes"
    data_root.mkdir()
    _write_episode(data_root / "episode_000.hdf5", 0)
    _write_episode(data_root / "episode_001.hdf5", 7)
    checkpoint_dir = tmp_path / "checkpoints"

    torch.manual_seed(0)
    uninterrupted = run_config(_config(data_root, checkpoint_dir))
    expected = {
        name: value.detach().clone()
        for name, value in uninterrupted.policy.state_dict().items()
    }
    step_one = checkpoint_dir / "checkpoint-1"
    assert step_one.is_dir()
    assert uninterrupted.step == 2
    assert uninterrupted.policy.configured_mode is InteractionMode.AUTO
    assert uninterrupted.policy.auto_examples.item() > 0
    assert uninterrupted.policy.interactive_examples.item() > 0
    assert uninterrupted.policy.video_scale.grad is not None
    assert uninterrupted.policy.semantic.grad is not None
    assert uninterrupted.policy.planning.grad is not None

    torch.manual_seed(999)
    resumed_checkpoint_dir = tmp_path / "resumed-checkpoints"
    resumed = run_config(_config(data_root, resumed_checkpoint_dir, step_one))

    assert resumed.step == 2
    assert resumed.data_batches_consumed == 2
    for name, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    final_step = resumed_checkpoint_dir / "checkpoint-2"
    assert final_step.is_dir()
    assert (final_step / "model.safetensors").is_file()
    assert not (resumed_checkpoint_dir / "final").exists()
    payload = torch.load(
        final_step / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    assert payload["step"] == 2
    assert payload["prompt_schedule"]["position"] == 2

    completed = run_config(_config(data_root, resumed_checkpoint_dir, final_step))
    assert completed.step == 2
    assert not (resumed_checkpoint_dir / "final").exists()
