"""Training-faithful WAM bridge for the InternW0_delta RoboDojo evaluation."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import OmegaConf

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

from ._adapter_base import StatefulWAMAdapter, as_bool, optional_float, optional_int

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"


def _local_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else POLICY_DIR / path


def _require_file(value: Any, name: str) -> Path:
    path = _local_path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _require_dir(value: Any, name: str) -> Path:
    path = _local_path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _activate_runtime(runtime_root: Path) -> None:
    source = str(runtime_root / "src")
    sys.path[:] = [source] + [entry for entry in sys.path if entry != source]
    for module_name in list(sys.modules):
        if module_name == "wam" or module_name.startswith("wam."):
            del sys.modules[module_name]


def _camera_tensor(image: Any) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float32)
        .div(255.0)
    )


def build_training_exact_canvas(
    head: Any,
    left: Any,
    right: Any,
    *,
    processor_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> torch.Tensor:
    """Reproduce training's resize-only three-camera T canvas, without crop."""

    processor_h, processor_w = map(int, processor_size)
    canvas_h, canvas_w = map(int, canvas_size)
    cameras = [
        transforms_F.resize(
            _camera_tensor(image),
            size=[processor_h, processor_w],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        for image in (head, left, right)
    ]
    top_h = (canvas_h * 2) // 3
    bottom_h = canvas_h - top_h
    left_w = canvas_w // 2
    targets = ((top_h, canvas_w), (bottom_h, left_w), (bottom_h, canvas_w - left_w))
    packed = [
        transforms_F.resize(
            camera,
            size=list(size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        for camera, size in zip(cameras, targets, strict=True)
    ]
    canvas = torch.cat([packed[0], torch.cat(packed[1:], dim=-1)], dim=-2)
    if tuple(canvas.shape) != (1, 3, canvas_h, canvas_w):
        raise RuntimeError(f"Unexpected canvas shape: {tuple(canvas.shape)}")
    return canvas.mul(2.0).sub(1.0)


class Model(ModelTemplate):
    """XPolicyLab model adapter with one stateful WAM session per RoboDojo environment."""

    __init__ = StatefulWAMAdapter.__init__
    _adapt_obs = StatefulWAMAdapter._adapt_obs
    _ingest = StatefulWAMAdapter._ingest
    update_obs = StatefulWAMAdapter.update_obs
    _session_for = StatefulWAMAdapter._session_for
    update_obs_batch = StatefulWAMAdapter.update_obs_batch
    _dummy_actions = StatefulWAMAdapter._dummy_actions
    _predict = StatefulWAMAdapter._predict
    get_action = StatefulWAMAdapter.get_action
    get_action_batch = StatefulWAMAdapter.get_action_batch
    get_timing_rollout = StatefulWAMAdapter.get_timing_rollout
    reset = StatefulWAMAdapter.reset

    def _load_real_policy(self) -> None:
        runtime_root = _require_dir(
            self.model_cfg.get("wam_root", "wam_runtime"), "wam_root"
        )
        checkpoint_path = resolve_checkpoint_root(
            self.model_cfg,
            CHECKPOINTS_DIR,
            policy_dir=POLICY_DIR,
            explicit_keys=("checkpoint_path", "ckpt_path", "model_path"),
            must_exist=True,
        )
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "robodojo.pt"
        checkpoint_path = _require_file(checkpoint_path, "checkpoint_path")
        stats_path = _require_file(
            self.model_cfg.get("dataset_stats_path", "config/dataset_stats.json"),
            "dataset_stats_path",
        )
        config_path = _require_file(
            self.model_cfg.get("train_config_path", "config/eval_model.yaml"),
            "train_config_path",
        )
        base_model_dir = _require_dir(
            self.model_cfg.get("base_model_dir", "assets/Wan-AI/Wan2.2-TI2V-5B"),
            "base_model_dir",
        )
        vlm_model_path = _require_dir(
            self.model_cfg.get(
                "vlm_model_path", "assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B"
            ),
            "vlm_model_path",
        )

        _activate_runtime(runtime_root)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(base_model_dir.parent.parent)

        from wam.runtime.robodojo_policy import (
            RobotWinPolicySession,
            RobotWinSessionConfig,
            _mixed_precision_to_model_dtype,
            _normalize_device,
            _resolve_training_video_contract,
            build_robotwin_shared_runtime,
        )

        module_path = Path(sys.modules["wam"].__file__).resolve()
        if not module_path.is_relative_to(runtime_root):
            raise RuntimeError(f"WAM source isolation failed: {module_path}")

        saved_cfg = OmegaConf.load(config_path)
        model_cfg = OmegaConf.create(OmegaConf.to_container(saved_cfg.model, resolve=True))
        processor_cfg = OmegaConf.create(
            OmegaConf.to_container(saved_cfg.data.train.processor, resolve=True)
        )
        model_cfg.model_id = str(base_model_dir)
        model_cfg.tokenizer_model_id = str(base_model_dir)
        model_cfg.redirect_common_files = False
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.action_dit_use_interpolated_init = False
        model_cfg.understanding.vlm_model_path = str(vlm_model_path)
        model_cfg.load_text_encoder = True
        model_cfg.action_dit_config.physical_time_rope_enabled = False
        model_cfg.action_dit_config.dimension_valid_mask = None
        model_cfg.action_dit_config.valid_input_scale = 1.0
        model_cfg.proprio_encoding = OmegaConf.create(
            {"dimension_valid_mask": None, "valid_input_scale": 1.0}
        )

        device = _normalize_device(str(self.model_cfg.get("device") or "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real WAM evaluation")
        dtype = _mixed_precision_to_model_dtype(
            str(self.model_cfg.get("mixed_precision") or "bf16")
        )
        self.runtime = build_robotwin_shared_runtime(
            model_cfg=model_cfg,
            processor_cfg=processor_cfg,
            checkpoint_path=str(checkpoint_path),
            dataset_stats_path=stats_path,
            device=device,
            model_dtype=dtype,
        )
        if bool(self.runtime.model.action_expert.physical_time_rope_enabled):
            raise RuntimeError("Physical-time Action RoPE must be disabled")
        if hasattr(self.runtime.model.action_expert.action_encoder, "feature_scale"):
            raise RuntimeError("Action fan-in calibration must be disabled")
        for name in ("proprio_encoder", "action_proprio_encoder"):
            encoder = getattr(self.runtime.model, name, None)
            if encoder is not None and getattr(encoder, "feature_scale", None) is not None:
                raise RuntimeError(f"{name} fan-in calibration must be disabled")

        image_meta = list(saved_cfg.data.train.processor.shape_meta.images)
        image_shapes = [tuple(map(int, item.shape)) for item in image_meta]
        if [str(item.key) for item in image_meta] != [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]:
            raise ValueError("Evaluation camera order does not match training")
        if len(set(image_shapes)) != 1:
            raise ValueError(f"Camera processor shapes differ: {image_shapes}")
        processor_size = image_shapes[0][-2:]
        video_layout, view_names = _resolve_training_video_contract(saved_cfg.data.train)
        num_frames = int(saved_cfg.data.train.num_frames)
        frequency_ratio = int(saved_cfg.data.train.action_video_freq_ratio)
        session_cfg = RobotWinSessionConfig(
            action_horizon=self.action_horizon,
            action_hz=float(self.model_cfg.get("action_hz") or 25.0),
            replan_steps=self.replan_steps,
            num_inference_steps=int(self.model_cfg.get("num_inference_steps") or 10),
            sigma_shift=optional_float(self.model_cfg.get("sigma_shift")),
            seed=optional_int(self.model_cfg.get("seed")),
            text_cfg_scale=float(self.model_cfg.get("text_cfg_scale") or 1.0),
            negative_prompt=str(self.model_cfg.get("negative_prompt") or ""),
            rand_device=str(self.model_cfg.get("rand_device") or "cpu"),
            tiled=as_bool(self.model_cfg.get("tiled", False)),
            timing_enabled=as_bool(self.model_cfg.get("timing_enabled", False)),
            num_video_frames=(num_frames - 1) // frequency_ratio + 1,
            video_size=tuple(map(int, saved_cfg.data.train.video_size)),
            video_layout=video_layout,
            video_view_names=view_names,
        )

        class TrainingExactSession(RobotWinPolicySession):
            def _build_robotwin_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
                images = observation["observation"]
                canvas = build_training_exact_canvas(
                    images["head_camera"]["rgb"],
                    images["left_camera"]["rgb"],
                    images["right_camera"]["rgb"],
                    processor_size=processor_size,
                    canvas_size=(self.video_height, self.video_width),
                )
                return canvas.to(
                    device=self.model.device,
                    dtype=self.model.torch_dtype,
                    non_blocking=True,
                )

        self.session_factory = partial(TrainingExactSession, self.runtime, session_cfg)
        self.session = self.session_factory()
        print(
            "[InternW0_delta] initialized "
            f"checkpoint={checkpoint_path} horizon={self.action_horizon} "
            f"replan={self.replan_steps} denoise={session_cfg.num_inference_steps} "
            "physical_time_rope=false fanin=false crop=none",
            flush=True,
        )


__all__ = ["Model", "build_training_exact_canvas"]
