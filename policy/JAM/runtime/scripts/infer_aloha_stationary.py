#!/usr/bin/env python3
"""Preflight, offline, replay, and guarded live FastWAM inference for ALOHA.

Offline mode accepts either a canonical observation NPZ or one frame from a
LeRobot v2.1 dataset and writes a denormalized 14-D action chunk.  Live mode is
adapter-based.  The bundled replay adapter captures commands without ROS.  The
version-locked Interbotix ALOHA 2.0 adapter provides observations and consumes
follower joint-position targets.  Hardware commands remain disabled unless the
deployment manifest passes every safety gate and ``--enable-hardware`` is
supplied.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

LOGGER = logging.getLogger("aloha_inference")
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
CANONICAL_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
TASK_DEFAULTS = {
    "fold_towel": {
        "task_config": "aloha_fold_towel_vae_direct_off12_3cam_384_1e-4",
        "instruction": "Fold the towel.",
        "stats": "/original-workspace/real_robot_data/aloha/stats/fold_towel_seed42_train90.json",
    },
    "open_drawer_store_marker": {
        "task_config": "aloha_open_drawer_store_marker_vae_direct_off12_3cam_384_1e-4",
        "instruction": "Open the drawer and store the marker.",
        "stats": "/original-workspace/real_robot_data/aloha/stats/open_drawer_store_marker_seed42_train90.json",
    },
}


class AlohaHardwareAdapter(Protocol):
    """Laboratory adapter contract used by live inference."""

    hardware_commands_enabled: bool

    def read_observation(self) -> dict[str, Any]:
        """Return qpos and RGB cameras in the source order declared by the manifest."""

    def send_joint_position_target(self, action: np.ndarray) -> None:
        """Send one 14-D follower joint-position target in hardware source order."""

    def emergency_stop(self, reason: str) -> None:
        """Stop or torque-disable the robot using the verified laboratory mechanism."""

    def close(self) -> None:
        """Release cameras and robot resources."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "offline", "replay", "live"), required=True
    )
    parser.add_argument("--task", choices=tuple(TASK_DEFAULTS), required=True)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--offline-warmup-repeats", type=int, default=0)
    parser.add_argument("--offline-benchmark-repeats", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--rand-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--observation-npz", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=None)

    parser.add_argument("--deployment-config", type=Path, default=None)
    parser.add_argument("--adapter-module", default=None)
    parser.add_argument("--adapter-factory", default="create_adapter")
    parser.add_argument("--max-commands", type=int, default=None)
    parser.add_argument("--enable-hardware", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _swap_halves(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[-1] != 14:
        raise ValueError(f"Expected a 14-D vector, got {array.shape}")
    return np.concatenate([array[..., 7:], array[..., :7]], axis=-1)


def _to_canonical(value: np.ndarray, source_order: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if source_order == "right_then_left":
        array = _swap_halves(array)
    elif source_order != "left_then_right":
        raise ValueError(f"Unsupported vector order: {source_order}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _from_canonical(value: np.ndarray, target_order: str) -> np.ndarray:
    # Swapping two equal halves is its own inverse.
    return _to_canonical(value, target_order)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Camera image must contain finite numeric values")
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0.0) <= 1.0:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(array, mode="RGB").resize(size_wh, Image.BILINEAR))


def _pack_cameras(observation: dict[str, Any]) -> np.ndarray:
    """Build a lightweight RGB preview with the policy's spatial layout."""
    high = _resize_rgb(np.asarray(observation["cam_high"]), (320, 256))
    left = _resize_rgb(np.asarray(observation["cam_left_wrist"]), (160, 128))
    right = _resize_rgb(np.asarray(observation["cam_right_wrist"]), (160, 128))
    return np.concatenate([high, np.concatenate([left, right], axis=1)], axis=0)


def _image_to_unit_tensor(image: np.ndarray, torch):
    array = np.asarray(image)
    if array.dtype == np.uint8:
        owned = np.array(array, dtype=np.uint8, copy=True, order="C")
        tensor = torch.from_numpy(owned).permute(2, 0, 1).to(torch.float32)
        return tensor / 255.0
    owned = np.array(array, dtype=np.float32, copy=True, order="C")
    tensor = torch.from_numpy(owned).permute(2, 0, 1)
    if not torch.isfinite(tensor).all():
        raise ValueError("Camera image contains NaN or Inf")
    if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise ValueError("Floating-point camera images must be in [0,1]")
    return tensor


def _pack_cameras_for_policy(observation: dict[str, Any], torch):
    """Reproduce FastWAMProcessor + RobotVideoDataset's two-stage resize exactly."""
    import torchvision.transforms.functional as transforms_f

    interpolation = transforms_f.InterpolationMode.BILINEAR

    def resize(image: np.ndarray, size: list[int]):
        tensor = _image_to_unit_tensor(image, torch)
        # The task configs first resize every view to 240x320.  The robotwin
        # packing then maps overhead/wrists to 256x320 and 128x160.
        tensor = transforms_f.resize(
            tensor, size=[240, 320], interpolation=interpolation, antialias=True
        )
        return transforms_f.resize(
            tensor, size=size, interpolation=interpolation, antialias=True
        )

    high = resize(np.asarray(observation["cam_high"]), [256, 320])
    left = resize(np.asarray(observation["cam_left_wrist"]), [128, 160])
    right = resize(np.asarray(observation["cam_right_wrist"]), [128, 160])
    packed = torch.cat([high, torch.cat([left, right], dim=-1)], dim=-2)
    if tuple(packed.shape) != (3, 384, 320):
        raise RuntimeError(f"Packed camera tensor has unexpected shape {tuple(packed.shape)}")
    return packed.mul(2.0).sub(1.0)


def _validate_observation(
    observation: dict[str, Any], build_preview: bool = True
) -> dict[str, Any]:
    missing = [key for key in ("cam_high", "cam_left_wrist", "cam_right_wrist", "qpos") if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing keys: {missing}")
    qpos = np.asarray(observation["qpos"], dtype=np.float32)
    if qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise ValueError(f"qpos must be finite [14], got {qpos.shape}")
    summary = {"qpos": qpos.tolist(), "cameras": {}}
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        summary["cameras"][key] = {"shape": list(image.shape), "dtype": str(image.dtype)}
    if build_preview:
        packed = _pack_cameras(observation)
        summary["packed_shape"] = list(packed.shape)
    return summary


def _load_npz_observation(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _episode_parquet(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / str(info["data_path"]).format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )


def _episode_video(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    return root / str(info["video_path"]).format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
        video_key=video_key,
    )


def _find_video_key(info: dict[str, Any], role: str) -> str:
    aliases = {
        "cam_high": (".cam_high", ".camera_high"),
        "cam_left_wrist": (".cam_left_wrist", ".camera_wrist_left"),
        "cam_right_wrist": (".cam_right_wrist", ".camera_wrist_right"),
    }[role]
    matches = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video" and any(key.endswith(alias) for alias in aliases)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve one {role} video feature, found {matches}")
    return matches[0]


def _decode_video_frame(path: Path, frame_index: int) -> np.ndarray:
    import av

    with av.open(str(path), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Video {path} has no decoded frame {frame_index}")


def _load_lerobot_observation(root: Path, episode_index: int, frame_index: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise RuntimeError(f"Expected LeRobot v2.1, found {info.get('codebase_version')}")
    if not 0 <= episode_index < int(info["total_episodes"]):
        raise IndexError(f"episode_index={episode_index} outside dataset")
    parquet = _episode_parquet(root, info, episode_index)
    table = pq.read_table(parquet, columns=["observation.state"])
    if not 0 <= frame_index < table.num_rows:
        raise IndexError(f"frame_index={frame_index} outside episode length {table.num_rows}")
    column = table["observation.state"].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False).reshape(len(column), 14)
    observation: dict[str, Any] = {"qpos": np.asarray(values[frame_index], dtype=np.float32)}
    for role in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        video_key = _find_video_key(info, role)
        observation[role] = _decode_video_frame(
            _episode_video(root, info, episode_index, video_key),
            frame_index,
        )
    return observation


def _resolve_offline_observation(args: argparse.Namespace) -> dict[str, Any]:
    if args.observation_npz is not None:
        return _load_npz_observation(args.observation_npz.resolve())
    if args.dataset_root is not None:
        return _load_lerobot_observation(
            args.dataset_root.resolve(), args.episode_index, args.frame_index
        )
    raise ValueError("preflight/offline mode requires --observation-npz or --dataset-root")


def _compose_task(task_config: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"task={task_config}"])


def _model_dtype(name: str, device: str):
    import torch

    if device == "cpu" or name == "fp32":
        return torch.float32
    return torch.bfloat16 if name == "bf16" else torch.float16


class AlohaFastWAMPolicy:
    def __init__(
        self,
        task_config: str,
        checkpoint: Path,
        dataset_stats: Path,
        device: str,
        mixed_precision: str,
        num_inference_steps: int,
        action_horizon: int | None,
        sigma_shift: float | None,
        seed: int,
        text_cfg_scale: float,
        rand_device: str,
        tiled: bool,
    ) -> None:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        cfg = _compose_task(task_config)
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.mot_checkpoint_mixed_attn = False
        dtype = _model_dtype(mixed_precision, device)
        self.model = instantiate(model_cfg, model_dtype=dtype, device=device)
        self.model.load_checkpoint(str(checkpoint))
        self.model = self.model.to(device).eval()

        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats)))
        self.action_horizon = int(action_horizon or (int(cfg.data.train.num_frames) - 1))
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = int(seed)
        self.text_cfg_scale = float(text_cfg_scale)
        self.rand_device = rand_device
        self.tiled = tiled
        self._torch = torch

    def _normalize_qpos(self, qpos: np.ndarray):
        torch = self._torch
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise RuntimeError("ALOHA policy expects one merged state field")
        key = state_meta[0]["key"]
        state = torch.from_numpy(np.array(qpos, dtype=np.float32, copy=True)).unsqueeze(0)
        batch = {"state": {key: state}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action) -> np.ndarray:
        torch = self._torch
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise RuntimeError("ALOHA policy expects one merged action field")
        key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.detach().to(dtype=torch.float32, device="cpu")).numpy()[0]

    def prepare(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[dict[str, Any], float]:
        """Prepare CPU policy inputs and return their observation-time origin."""
        torch = self._torch
        request_started = time.perf_counter()
        _validate_observation(observation, build_preview=False)
        image = _pack_cameras_for_policy(observation, torch).unsqueeze(0)
        proprio = self._normalize_qpos(np.asarray(observation["qpos"], dtype=np.float32))
        kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": "",
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        signature = inspect.signature(self.model.infer_action)
        if "num_video_frames" in signature.parameters:
            kwargs["num_video_frames"] = 9
        return kwargs, request_started

    def predict_prepared(
        self, kwargs: dict[str, Any], request_started: float
    ) -> tuple[np.ndarray, float]:
        torch = self._torch
        prepared = dict(kwargs)
        prepared["input_image"] = prepared["input_image"].to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        with torch.inference_mode():
            prediction = self.model.infer_action(**prepared)["action"]
        elapsed = time.perf_counter() - request_started
        action = self._denormalize_action(prediction)
        if action.shape != (self.action_horizon, 14) or not np.isfinite(action).all():
            raise RuntimeError(f"Model returned invalid action chunk {action.shape}")
        return action, elapsed

    def predict(self, observation: dict[str, Any], instruction: str) -> tuple[np.ndarray, float]:
        prepared, request_started = self.prepare(observation, instruction)
        return self.predict_prepared(prepared, request_started)


def _load_deployment_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Deployment config must be a YAML mapping")
    return payload


def _require_verified_deployment(
    config: dict[str, Any],
    checkpoint: Path,
    stats_path: Path,
    task: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"replay", "live"}:
        raise ValueError(f"Unsupported deployment mode: {mode}")
    if mode == "live":
        required_true = (
            "camera_calibration_verified",
            "action_semantics_verified",
            "joint_limits_verified",
            "control_timing_verified",
            "tracking_error_verified",
            "replay_verified",
            "dry_run_verified",
            "emergency_stop_verified",
        )
        missing_true = [
            name
            for name in required_true
            if config.get("verification", {}).get(name) is not True
        ]
        if missing_true:
            raise RuntimeError(f"Deployment verification gates are not true: {missing_true}")
    hardware = config.get("hardware", {})
    safety = config.get("safety", {})
    for name in ("installed_revision", "controller_sdk", "units"):
        value = hardware.get(name)
        if value in (None, "", "TODO"):
            raise RuntimeError(f"Deployment hardware.{name} is not frozen")
    source_order = hardware.get("source_vector_order")
    if source_order not in {"left_then_right", "right_then_left"}:
        raise RuntimeError("hardware.source_vector_order must be explicit")
    if config.get("task") != task:
        raise RuntimeError(f"Deployment task {config.get('task')!r} does not match {task!r}")
    command_rate = float(safety.get("command_rate_hz") or 0.0)
    joint_min = np.asarray(safety.get("joint_min"), dtype=np.float32)
    joint_max = np.asarray(safety.get("joint_max"), dtype=np.float32)
    max_delta = np.asarray(safety.get("max_delta_per_command"), dtype=np.float32)
    max_tracking_error = np.asarray(safety.get("max_tracking_error"), dtype=np.float32)
    max_initial_inference_latency = float(
        safety.get("max_initial_inference_latency_s") or 0.0
    )
    max_replan_inference_latency = float(
        safety.get("max_replan_inference_latency_s") or 0.0
    )
    inference_margin_steps = safety.get("inference_margin_steps")
    log_flush_interval = safety.get("log_flush_interval_commands")
    max_cycle_overrun = float(safety.get("max_control_cycle_overrun_s") or 0.0)
    if (
        command_rate <= 0
        or max_initial_inference_latency <= 0
        or max_replan_inference_latency <= 0
        or isinstance(inference_margin_steps, bool)
        or not isinstance(inference_margin_steps, int)
        or inference_margin_steps < 1
        or isinstance(log_flush_interval, bool)
        or not isinstance(log_flush_interval, int)
        or log_flush_interval < 1
        or max_cycle_overrun <= 0
        or joint_min.shape != (14,)
        or joint_max.shape != (14,)
        or max_delta.shape != (14,)
        or max_tracking_error.shape != (14,)
    ):
        raise RuntimeError(
            "Safety rate/latency/margin/overrun and all 14-D bounds must be fully specified"
        )
    if not (
        np.isfinite(joint_min).all()
        and np.isfinite(joint_max).all()
        and np.isfinite(max_delta).all()
        and np.isfinite(max_tracking_error).all()
    ):
        raise RuntimeError("Safety arrays contain NaN/Inf")
    if (
        np.any(joint_min >= joint_max)
        or np.any(max_delta <= 0)
        or np.any(max_tracking_error <= 0)
    ):
        raise RuntimeError("Safety bounds are invalid")
    artifacts = config.get("artifacts", {})
    expected_checkpoint = str(artifacts.get("checkpoint_sha256") or "")
    expected_stats = str(artifacts.get("dataset_stats_sha256") or "")
    if len(expected_checkpoint) != 64 or _sha256(checkpoint) != expected_checkpoint:
        raise RuntimeError("Checkpoint SHA256 does not match the deployment manifest")
    if len(expected_stats) != 64 or _sha256(stats_path) != expected_stats:
        raise RuntimeError("Dataset-stats SHA256 does not match the deployment manifest")
    return {
        "source_order": source_order,
        "command_rate_hz": command_rate,
        "max_initial_inference_latency_s": max_initial_inference_latency,
        "max_replan_inference_latency_s": max_replan_inference_latency,
        "inference_margin_steps": inference_margin_steps,
        "log_flush_interval_commands": log_flush_interval,
        "max_control_cycle_overrun_s": max_cycle_overrun,
        "joint_min": joint_min,
        "joint_max": joint_max,
        "max_delta": max_delta,
        "max_tracking_error": max_tracking_error,
        "camera_keys": hardware.get("camera_keys", {}),
    }


def _canonical_adapter_observation(raw: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    camera_keys = deployment["camera_keys"]
    if not isinstance(camera_keys, dict):
        raise RuntimeError("hardware.camera_keys must map canonical roles to adapter keys")
    observation = {
        "qpos": _to_canonical(np.asarray(raw["qpos"]), deployment["source_order"]),
    }
    for role in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        source_key = camera_keys.get(role)
        if not source_key or source_key not in raw:
            raise RuntimeError(f"Adapter observation lacks mapped camera {role}: {source_key!r}")
        observation[role] = np.asarray(raw[source_key])
    _validate_observation(observation, build_preview=False)
    return observation


def _safe_target(target: np.ndarray, reference: np.ndarray, deployment: dict[str, Any]) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if (
        target.shape != (14,)
        or reference.shape != (14,)
        or not np.isfinite(target).all()
        or not np.isfinite(reference).all()
    ):
        raise RuntimeError("Non-finite or malformed action target")
    outside = (target < deployment["joint_min"]) | (target > deployment["joint_max"])
    if np.any(outside):
        bad = np.flatnonzero(outside).tolist()
        raise RuntimeError(f"Predicted target violates hard joint bounds at dimensions {bad}")
    too_fast = np.abs(target - reference) > deployment["max_delta"]
    if np.any(too_fast):
        bad = np.flatnonzero(too_fast).tolist()
        raise RuntimeError(f"Predicted target violates per-command delta at dimensions {bad}")
    return target


def _create_adapter(
    module_name: str,
    factory_name: str,
    config: dict[str, Any],
    expect_hardware: bool,
) -> AlohaHardwareAdapter:
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RuntimeError(f"Adapter factory {module_name}.{factory_name} is not callable")
    adapter = factory(config)
    hardware_enabled = getattr(adapter, "hardware_commands_enabled", None)
    if not isinstance(hardware_enabled, bool):
        raise RuntimeError("Adapter must declare boolean hardware_commands_enabled")
    if hardware_enabled is not expect_hardware:
        expected = "hardware" if expect_hardware else "non-hardware replay"
        raise RuntimeError(f"Adapter does not match requested {expected} mode")
    return adapter


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _submit_policy_prediction(
    executor: ThreadPoolExecutor,
    policy: AlohaFastWAMPolicy,
    observation: dict[str, Any],
    instruction: str,
) -> Future[tuple[np.ndarray, float]]:
    # Keep camera resize/packing away from the command thread.  This thread
    # still shares a Python process, so powered deployment must pass the
    # control-computer timing gate or replace it with a process/RPC worker.
    return executor.submit(policy.predict, _copy_observation(observation), instruction)


def _validated_chunk(
    prediction: tuple[np.ndarray, float],
    deployment: dict[str, Any],
    replan_steps: int,
    latency_key: str,
) -> tuple[np.ndarray, float]:
    chunk, inference_s = prediction
    chunk = np.asarray(chunk, dtype=np.float32)
    inference_s = float(inference_s)
    if chunk.ndim != 2 or chunk.shape[1] != 14 or not np.isfinite(chunk).all():
        raise RuntimeError(f"Policy returned malformed action chunk {chunk.shape}")
    if chunk.shape[0] < replan_steps:
        raise RuntimeError(
            f"Policy horizon {chunk.shape[0]} is shorter than replan_steps={replan_steps}"
        )
    if not math.isfinite(inference_s) or inference_s < 0:
        raise RuntimeError(f"Policy returned invalid inference latency {inference_s}")
    latency_limit = deployment[latency_key]
    if inference_s > latency_limit:
        raise RuntimeError(
            f"Inference latency {inference_s:.3f}s exceeds the verified limit "
            f"{latency_limit:.3f}s ({latency_key})"
        )
    return np.array(chunk[:replan_steps], dtype=np.float32, copy=True), inference_s


def _check_measured_state(
    qpos: np.ndarray,
    last_command: np.ndarray | None,
    deployment: dict[str, Any],
) -> None:
    measured = np.asarray(qpos, dtype=np.float32)
    if measured.shape != (14,) or not np.isfinite(measured).all():
        raise RuntimeError("Measured qpos is non-finite or malformed")
    outside = (measured < deployment["joint_min"]) | (measured > deployment["joint_max"])
    if np.any(outside):
        raise RuntimeError(
            "Measured qpos violates verified joint bounds at dimensions "
            f"{np.flatnonzero(outside).tolist()}"
        )
    if last_command is not None:
        tracking_error = np.abs(measured - last_command)
        bad = tracking_error > deployment["max_tracking_error"]
        if np.any(bad):
            indices = np.flatnonzero(bad).tolist()
            values = tracking_error[bad].tolist()
            raise RuntimeError(
                f"Follower tracking error exceeds limit at dimensions {indices}: {values}"
            )


def _run_live(
    args: argparse.Namespace,
    policy: AlohaFastWAMPolicy,
    instruction: str,
    config: dict[str, Any],
    deployment: dict[str, Any],
) -> None:
    assert args.adapter_module is not None and args.max_commands is not None
    period = 1.0 / deployment["command_rate_hz"]
    deadline_steps = (
        math.ceil(
            deployment["max_replan_inference_latency_s"]
            * deployment["command_rate_hz"]
        )
        + deployment["inference_margin_steps"]
    )
    # A newly predicted chunk has already aged by up to one inference deadline
    # when it is activated.  Its remaining tail must cover the following
    # inference deadline, hence the factor of two.
    if args.replan_steps < 2 * deadline_steps:
        raise RuntimeError(
            f"replan_steps={args.replan_steps} cannot cover the verified inference deadline: "
            f"need at least {2 * deadline_steps} steps at "
            f"{deployment['command_rate_hz']:.3f} Hz"
        )
    log_path = args.output.resolve()
    if log_path.exists():
        raise FileExistsError(f"Control log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    expect_hardware = args.mode == "live"
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fastwam-policy")
    old_signal_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)
    }

    def terminate_from_signal(signum, _frame):
        raise RuntimeError(f"Control loop received termination signal {signum}")

    for sig in old_signal_handlers:
        signal.signal(sig, terminate_from_signal)
    try:
        adapter = _create_adapter(
            args.adapter_module,
            args.adapter_factory,
            config,
            expect_hardware=expect_hardware,
        )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        for sig, handler in old_signal_handlers.items():
            signal.signal(sig, handler)
        raise
    next_prediction: Future[tuple[np.ndarray, float]] | None = None
    next_prediction_start: int | None = None
    current_chunk: np.ndarray
    current_chunk_start = 0
    last_command: np.ndarray | None = None
    chunk_index = 0
    gc_was_enabled = gc.isenabled()
    try:
        initial_raw = adapter.read_observation()
        initial_observation = _canonical_adapter_observation(initial_raw, deployment)
        _check_measured_state(initial_observation["qpos"], None, deployment)
        current_chunk, active_inference_s = _validated_chunk(
            policy.predict(initial_observation, instruction),
            deployment,
            args.replan_steps,
            "max_initial_inference_latency_s",
        )
        if gc_was_enabled:
            gc.disable()
        with log_path.open("x", encoding="utf-8") as log_stream:
            for command_index in range(int(args.max_commands)):
                cycle_started = time.monotonic()
                raw = adapter.read_observation()
                observation = _canonical_adapter_observation(raw, deployment)
                _check_measured_state(observation["qpos"], last_command, deployment)

                activated_chunk = False
                activation_offset = None
                if next_prediction is not None and next_prediction.done():
                    assert next_prediction_start is not None
                    candidate, candidate_inference_s = _validated_chunk(
                        next_prediction.result(),
                        deployment,
                        args.replan_steps,
                        "max_replan_inference_latency_s",
                    )
                    candidate_offset = command_index - next_prediction_start
                    if not 0 <= candidate_offset < len(candidate):
                        raise RuntimeError(
                            f"Completed action chunk is already expired at offset {candidate_offset}"
                        )
                    current_chunk = candidate
                    current_chunk_start = next_prediction_start
                    active_inference_s = candidate_inference_s
                    next_prediction = None
                    next_prediction_start = None
                    chunk_index += 1
                    activated_chunk = True
                    activation_offset = candidate_offset

                action_offset = command_index - current_chunk_start
                if not 0 <= action_offset < len(current_chunk):
                    if next_prediction is None or not next_prediction.done():
                        raise RuntimeError(
                            "Asynchronous policy worker missed the time-aligned action horizon"
                        )
                    raise RuntimeError("Action-chunk activation invariant failed")

                reference = observation["qpos"] if last_command is None else last_command
                canonical_target = _safe_target(
                    current_chunk[action_offset], reference, deployment
                )
                hardware_target = _from_canonical(canonical_target, deployment["source_order"])
                adapter.send_joint_position_target(hardware_target)
                command_sent_time = time.monotonic()
                last_command = canonical_target

                submitted_replan = False
                if next_prediction is None and command_index + 1 < int(args.max_commands):
                    next_prediction = _submit_policy_prediction(
                        executor, policy, observation, instruction
                    )
                    next_prediction_start = command_index
                    submitted_replan = True

                record = {
                    "command_index": command_index,
                    "monotonic_time": cycle_started,
                    "command_monotonic_time": command_sent_time,
                    "qpos_canonical": observation["qpos"].tolist(),
                    "target_canonical": canonical_target.tolist(),
                    "chunk_index": chunk_index,
                    "chunk_start_command": current_chunk_start,
                    "action_offset": action_offset,
                    "chunk_inference_s": active_inference_s,
                    "chunk_actions_remaining": len(current_chunk) - action_offset - 1,
                    "activated_chunk": activated_chunk,
                    "activation_offset": activation_offset,
                    "submitted_replan": submitted_replan,
                }
                log_stream.write(json.dumps(record) + "\n")
                if (
                    (command_index + 1)
                    % deployment["log_flush_interval_commands"]
                    == 0
                ):
                    log_stream.flush()
                remaining = period - (time.monotonic() - cycle_started)
                if remaining < -deployment["max_control_cycle_overrun_s"]:
                    raise RuntimeError(
                        f"Control cycle overran by {-remaining:.6f}s, beyond the verified limit "
                        f"{deployment['max_control_cycle_overrun_s']:.6f}s"
                    )
                if remaining > 0:
                    time.sleep(remaining)
    except BaseException as error:
        try:
            adapter.emergency_stop(f"FastWAM inference aborted: {error}")
        except Exception:
            LOGGER.exception("Adapter emergency stop raised while handling %r", error)
        raise
    finally:
        try:
            adapter.close()
        finally:
            if gc_was_enabled:
                gc.enable()
            if next_prediction is not None:
                next_prediction.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            for sig, handler in old_signal_handlers.items():
                signal.signal(sig, handler)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.num_inference_steps <= 0 or args.replan_steps <= 0:
        raise ValueError("num-inference-steps and replan-steps must be positive")
    if args.offline_warmup_repeats < 0 or args.offline_benchmark_repeats <= 0:
        raise ValueError("offline warmups must be non-negative and benchmark repeats positive")
    if args.mode != "offline" and (
        args.offline_warmup_repeats != 0 or args.offline_benchmark_repeats != 1
    ):
        raise ValueError("Offline benchmark repeat options are valid only in offline mode")
    if args.action_horizon is not None and args.action_horizon <= 0:
        raise ValueError("action-horizon must be positive")
    defaults = TASK_DEFAULTS[args.task]
    task_config = args.task_config or defaults["task_config"]
    instruction = args.instruction or defaults["instruction"]
    stats_path = (args.dataset_stats or Path(defaults["stats"])).expanduser().resolve()

    if args.mode in {"preflight", "offline"}:
        observation = _resolve_offline_observation(args)
        summary = _validate_observation(observation)
        if args.preview is not None:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_pack_cameras(observation), mode="RGB").save(args.preview)
            summary["preview"] = str(args.preview.resolve())
        if args.mode == "preflight":
            summary.update(
                {
                    "task": args.task,
                    "instruction": instruction,
                    "task_config": task_config,
                    "status": "ALOHA_INFERENCE_PREFLIGHT_OK",
                }
            )
            _atomic_json(args.output.resolve(), summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
    else:
        observation = None

    if args.checkpoint is None:
        raise ValueError("offline/live mode requires --checkpoint")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint or stats: {checkpoint} / {stats_path}")

    live_config: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    if args.mode in {"replay", "live"}:
        if args.mode == "live" and not args.enable_hardware:
            raise RuntimeError(
                "Live mode is command-disabled; pass --enable-hardware after all safety checks"
            )
        if args.mode == "replay" and args.enable_hardware:
            raise RuntimeError("Replay mode rejects --enable-hardware")
        if args.mode == "replay" and args.adapter_module is None:
            args.adapter_module = "experiments.real_robot.aloha_replay_adapter"
        if (
            args.deployment_config is None
            or args.adapter_module is None
            or args.max_commands is None
            or args.max_commands <= 0
        ):
            raise RuntimeError(
                "Replay/live mode requires deployment config, adapter module, and positive max-commands"
            )
        live_config = _load_deployment_config(args.deployment_config.resolve())
        deployment = _require_verified_deployment(
            live_config, checkpoint, stats_path, args.task, args.mode
        )

    if args.mode == "offline" and args.output.suffix != ".npz":
        raise ValueError("Offline --output must end in .npz")
    policy = AlohaFastWAMPolicy(
        task_config=task_config,
        checkpoint=checkpoint,
        dataset_stats=stats_path,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        rand_device=args.rand_device,
        tiled=args.tiled,
    )
    if args.mode == "offline":
        assert observation is not None
        warmup_timings = []
        for _ in range(args.offline_warmup_repeats):
            _, warmup_s = policy.predict(observation, instruction)
            warmup_timings.append(warmup_s)
        timings = []
        action = None
        for _ in range(args.offline_benchmark_repeats):
            action, inference_s = policy.predict(observation, instruction)
            timings.append(inference_s)
        assert action is not None
        timing_array = np.asarray(timings, dtype=np.float64)
        timing_summary = {
            "repeats": len(timings),
            "warmup_repeats": args.offline_warmup_repeats,
            "min": float(timing_array.min()),
            "median": float(np.median(timing_array)),
            "p95": float(np.quantile(timing_array, 0.95)),
            "max": float(timing_array.max()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            action=action,
            qpos=np.asarray(observation["qpos"], dtype=np.float32),
            inference_seconds=np.asarray(timings[-1], dtype=np.float64),
            inference_seconds_all=timing_array,
            warmup_inference_seconds_all=np.asarray(warmup_timings, dtype=np.float64),
        )
        print(
            json.dumps(
                {
                    "status": "ALOHA_OFFLINE_INFERENCE_OK",
                    "output": str(args.output.resolve()),
                    "action_shape": list(action.shape),
                    "action_min": action.min(axis=0).tolist(),
                    "action_max": action.max(axis=0).tolist(),
                    "inference_seconds": timings[-1],
                    "warmup_inference_seconds": warmup_timings,
                    "inference_timing": timing_summary,
                },
                indent=2,
            )
        )
        return 0

    assert live_config is not None and deployment is not None
    _run_live(args, policy, instruction, live_config, deployment)
    print(f"ALOHA_{args.mode.upper()}_INFERENCE_COMPLETE commands={args.max_commands}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
