from __future__ import annotations
import bisect
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence
import h5py
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from prismwam.rgb_history import RGBHistory
from .lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from .robotwin_rgb_dataset import PROMPT_TEMPLATE

LIBERO4_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
RGBD_SCHEMA_VERSION = "libero-rgbd-v1"
RGBD_VARIANT = "full"
RGBD_SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
RGBD_TEMPORAL_SEMANTICS = "states[t] -- actions[t] --> RGB/proprio/robot_states[t]; depth[t] = render(states[t+1])"
ACTION_CONVENTION = "libero_delta6_gripper_open01_v1"
DATASET_CONTRACT_VERSION = "libero_rgbd_full_depth_joint_v1"


def rgb_history_offsets(num_frames: int, stride: int) -> tuple[int, ...]:
    """Return causal RGB history indices relative to the current observation."""
    if num_frames not in (1, 5) or stride not in (2, 4):
        raise ValueError("LIBERO RGB history requires 1 or 5 frames at stride 2 or 4.")
    return RGBHistory.dense(num_frames, stride).offsets()


def _compose_views(frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """Agent view and wrist view side by side, each resized to 224x224 (NCHW)."""
    if len(frames) != 2 or any(frame.ndim != 4 for frame in frames):
        raise ValueError("Expected two NCHW LIBERO views.")
    return torch.cat(
        [
            F.interpolate(frame, size=(224, 224), mode="bilinear", align_corners=False)
            for frame in frames
        ],
        dim=-1,
    )


def _numeric_stats(values: np.ndarray) -> dict[str, torch.Tensor]:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Stats input must be finite [N,D], got {values.shape}.")
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }
    return {
        f"global_{name}": torch.from_numpy(np.asarray(value, dtype=np.float32))
        for (name, value) in stats.items()
    }


class LiberoRGBDDepthDataset(torch.utils.data.Dataset):
    """Paired LIBERO RGB/depth windows from the non-aligned full RGB-D corpus."""

    def __init__(
        self,
        root_dir: str,
        processor: Any,
        suites: Sequence[str] = LIBERO4_SUITES,
        cameras: Sequence[str] = ("agentview", "eye_in_hand"),
        num_video_frames: int = 9,
        frame_stride: int = 4,
        sample_stride: int = 1,
        action_horizon: int = 32,
        video_size: Sequence[int] = (224, 448),
        depth_max_meters: float = 1.5,
        expected_tasks_per_suite: int = 10,
        expected_demos_per_task: int = 50,
        expected_source_revision: str = RGBD_SOURCE_REVISION,
        pretrained_norm_stats: str | None = None,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        rgb_target_mode: str | None = None,
        rgb_history_num_frames: int = 1,
        rgb_history_frame_stride: int = 4,
        dataset_uid: str | None = None,
        prompt_policy_id: str = "libero_language_instruction_v1",
        max_open_files: int = 8,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.suites = tuple((str(suite) for suite in suites))
        self.cameras = tuple((str(camera) for camera in cameras))
        self.num_video_frames = int(num_video_frames)
        self.frame_stride = int(frame_stride)
        self.sample_stride = int(sample_stride)
        self.action_horizon = int(action_horizon)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.depth_max_meters = float(depth_max_meters)
        self.expected_tasks_per_suite = int(expected_tasks_per_suite)
        self.expected_demos_per_task = int(expected_demos_per_task)
        self.expected_source_revision = str(expected_source_revision)
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser()
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.rgb_target_mode = (
            None if rgb_target_mode is None else str(rgb_target_mode).strip().lower()
        )
        if self.rgb_target_mode not in {None, "f4", "f8"}:
            raise ValueError("LIBERO supports the RGB endpoint modes f4 and f8.")
        self.rgb_history_num_frames = int(rgb_history_num_frames)
        self.rgb_history_frame_stride = int(rgb_history_frame_stride)
        rgb_history_offsets(self.rgb_history_num_frames, self.rgb_history_frame_stride)
        self.rgb_history = RGBHistory.dense(
            self.rgb_history_num_frames, self.rgb_history_frame_stride
        )
        self.dataset_uid = str(dataset_uid or self.root_dir)
        self.prompt_policy_id = str(prompt_policy_id)
        self.max_open_files = max(int(max_open_files), 1)
        self._open_files: OrderedDict[str, h5py.File] = OrderedDict()
        self._owner_pid = os.getpid()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"LIBERO RGB-D root not found: {self.root_dir}")
        if "aligned" in self.root_dir.name.lower():
            raise ValueError(
                f"Aligned LIBERO data is forbidden for this task: {self.root_dir}"
            )
        if self.suites != LIBERO4_SUITES:
            raise ValueError(f"Expected exactly {LIBERO4_SUITES}, got {self.suites}.")
        if self.cameras != ("agentview", "eye_in_hand"):
            raise ValueError(
                f"Camera order is fixed to ('agentview', 'eye_in_hand'), got {self.cameras}."
            )
        if (
            self.action_horizon not in (8, 16, 32)
            or self.frame_stride not in (2, 4)
            or self.sample_stride != 1
        ):
            raise ValueError(
                "LIBERO requires H8/H16/H32, frame_stride=2/4 and sample_stride=1."
            )
        if self.num_video_frames not in (5, 9):
            raise ValueError("LIBERO Depth VAE requires 5 or 9 input frames.")
        if (self.num_video_frames - 1) * self.frame_stride != self.action_horizon:
            raise ValueError("Video frames and stride must cover the action horizon.")
        if (
            self.rgb_target_mode is not None
            and self.rgb_target_mode != f"f{self.num_video_frames - 1}"
        ):
            raise ValueError(
                "RGB endpoint mode must match the number of future video frames."
            )
        if self.video_size != (224, 448):
            raise ValueError(
                f"LIBERO views are composed into 224x448, got {self.video_size}."
            )
        if self.depth_max_meters != 1.5:
            raise ValueError("Depth preprocessing must clip at exactly 1.5 meters.")
        if self.expected_tasks_per_suite <= 0 or self.expected_demos_per_task <= 0:
            raise ValueError("Expected task/demo counts must be positive.")
        self.frame_offsets = [
            index * self.frame_stride for index in range(self.num_video_frames)
        ]
        self.episodes: list[dict[str, Any]] = []
        self.cumulative_sizes: list[int] = []
        self.instructions: list[str] = []
        self._build_index()
        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        if processor is None:
            raise ValueError("LIBERO RGB-D training requires a PrismWAMProcessor.")
        self.processor = processor.train()
        stats_path = (
            None
            if pretrained_norm_stats is None
            else Path(pretrained_norm_stats).expanduser()
        )
        if stats_path is None:
            dataset_stats = self._compute_dataset_stats()
        else:
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"LIBERO normalization stats not found: {stats_path}"
                )
            dataset_stats = load_dataset_stats_from_json(str(stats_path))
            self._validate_stats_provenance(dataset_stats, stats_path)
        self.dataset_stats = dataset_stats
        self.processor.set_normalizer_from_stats(dataset_stats)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_open_files"] = OrderedDict()
        return state

    def __del__(self):
        for handle in getattr(self, "_open_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "contract_version": DATASET_CONTRACT_VERSION,
            "root_dir": str(self.root_dir),
            "rgbd_variant": RGBD_VARIANT,
            "rgbd_source_revision": self.expected_source_revision,
            "suites": list(self.suites),
            "num_tasks": self.expected_tasks_per_suite * len(self.suites),
            "num_demos": len(self.episodes),
            "num_valid_depth_anchors": len(self),
            "action_convention": ACTION_CONVENTION,
        }

    def save_dataset_stats(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            save_dataset_stats_to_json(self.dataset_stats, str(temporary))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def _build_index(self) -> None:
        total = 0
        instructions: list[str] = []
        for suite in self.suites:
            suite_dir = self.root_dir / suite
            task_paths = sorted(suite_dir.glob("*_demo.hdf5"))
            if len(task_paths) != self.expected_tasks_per_suite:
                raise ValueError(
                    f"{suite} must contain {self.expected_tasks_per_suite} task files, found {len(task_paths)} under {suite_dir}."
                )
            for task_path in task_paths:
                with h5py.File(task_path, "r") as h5_file:
                    data = h5_file.get("data")
                    if data is None:
                        raise KeyError(f"Missing /data group in {task_path}")
                    attrs = data.attrs
                    expected_attrs = {
                        "rgbd_schema_version": RGBD_SCHEMA_VERSION,
                        "rgbd_variant": RGBD_VARIANT,
                        "rgbd_last_frame_policy": "invalid_sentinel",
                        "rgbd_source_hf_revision": self.expected_source_revision,
                        "rgbd_temporal_semantics": RGBD_TEMPORAL_SEMANTICS,
                    }
                    for name, expected in expected_attrs.items():
                        actual = attrs.get(name)
                        if actual != expected:
                            raise ValueError(
                                f"{task_path}: {name}={actual!r}, expected {expected!r}."
                            )
                    instruction = str(
                        json.loads(attrs["problem_info"])["language_instruction"]
                    )
                    demo_names = sorted(
                        data.keys(), key=lambda name: int(name.rsplit("_", 1)[1])
                    )
                    if len(demo_names) != self.expected_demos_per_task:
                        raise ValueError(
                            f"{task_path} must contain {self.expected_demos_per_task} demos, found {len(demo_names)}."
                        )
                    for demo_name in demo_names:
                        demo = data[demo_name]
                        length = self._validate_demo(task_path, demo_name, demo)
                        valid_length = length - 1
                        episode_index = len(self.episodes)
                        self.episodes.append(
                            {
                                "path": task_path,
                                "group": f"data/{demo_name}",
                                "suite": suite,
                                "task": task_path.stem.removesuffix("_demo"),
                                "instruction": instruction,
                                "episode_index": episode_index,
                                "demo_index": int(demo_name.rsplit("_", 1)[1]),
                                "length": length,
                                "valid_length": valid_length,
                                "num_samples": valid_length,
                            }
                        )
                        total += valid_length
                        self.cumulative_sizes.append(total)
                instructions.append(instruction)
        if not self.episodes or total <= 0:
            raise ValueError(
                f"No valid LIBERO depth anchors found under {self.root_dir}."
            )
        self.instructions = list(dict.fromkeys(instructions))
        expected_tasks = self.expected_tasks_per_suite * len(self.suites)
        expected_demos = expected_tasks * self.expected_demos_per_task
        if (
            len(self.instructions) != expected_tasks
            or len(self.episodes) != expected_demos
        ):
            raise ValueError(
                f"LIBERO4 coverage mismatch: instructions={len(self.instructions)}/{expected_tasks}, demos={len(self.episodes)}/{expected_demos}."
            )

    def _validate_demo(self, path: Path, demo_name: str, demo: h5py.Group) -> int:
        required = {"actions": 7, "obs/ee_states": 6, "obs/gripper_states": 2}
        lengths: list[int] = []
        for key, width in required.items():
            if key not in demo:
                raise KeyError(f"Missing {key} in {path}:{demo_name}")
            shape = demo[key].shape
            if len(shape) != 2 or shape[1] != width:
                raise ValueError(
                    f"{path}:{demo_name}/{key} must be [N,{width}], got {shape}."
                )
            lengths.append(int(shape[0]))
        for camera in self.cameras:
            for modality, tail in (("rgb", (128, 128, 3)), ("depth", (128, 128, 1))):
                key = f"obs/{camera}_{modality}"
                if key not in demo or tuple(demo[key].shape[1:]) != tail:
                    actual = None if key not in demo else tuple(demo[key].shape)
                    raise ValueError(
                        f"{path}:{demo_name}/{key} has invalid shape {actual}."
                    )
                lengths.append(int(demo[key].shape[0]))
        if len(set(lengths)) != 1 or lengths[0] < 2:
            raise ValueError(
                f"{path}:{demo_name} has inconsistent or too-short lengths {lengths}."
            )
        length = lengths[0]
        valid = np.asarray(demo["depth_frame_valid"], dtype=np.uint8)
        rgb_index = np.asarray(demo["source_rgb_index"], dtype=np.int64)
        state_index = np.asarray(demo["source_state_index"], dtype=np.int64)
        expected_valid = np.concatenate([np.ones(length - 1, dtype=np.uint8), [0]])
        expected_rgb = np.concatenate([np.arange(length - 1, dtype=np.int64), [-1]])
        expected_state = np.concatenate([np.arange(1, length, dtype=np.int64), [-1]])
        if not np.array_equal(valid, expected_valid):
            raise ValueError(
                f"{path}:{demo_name} depth_frame_valid violates full-data sentinel policy."
            )
        if not np.array_equal(rgb_index, expected_rgb) or not np.array_equal(
            state_index, expected_state
        ):
            raise ValueError(
                f"{path}:{demo_name} source RGB/state indices violate full-data semantics."
            )
        for camera in self.cameras:
            depth = demo[f"obs/{camera}_depth"]
            if np.any(depth[-1]) or not np.any(depth[0]):
                raise ValueError(
                    f"{path}:{demo_name}/{camera} has an invalid depth sentinel/valid range."
                )
        return length

    def _compute_dataset_stats(self) -> dict[str, Any]:
        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for episode in self.episodes:
            demo = self._open(episode["path"])[episode["group"]]
            actions.append(
                self._convert_action(np.asarray(demo["actions"], dtype=np.float32))
            )
            states.append(
                np.concatenate(
                    [
                        np.asarray(demo["obs/ee_states"], dtype=np.float32),
                        np.asarray(demo["obs/gripper_states"], dtype=np.float32),
                    ],
                    axis=-1,
                )
            )
        action_values = np.concatenate(actions, axis=0)
        state_values = np.concatenate(states, axis=0)
        return {
            "action": {"default": _numeric_stats(action_values)},
            "state": {"default": _numeric_stats(state_values)},
            "_provenance": self.provenance,
        }

    def _validate_stats_provenance(self, stats: dict[str, Any], path: Path) -> None:
        expected = self.provenance
        actual = stats.get("_provenance")
        if not isinstance(actual, dict):
            raise ValueError(f"Normalization stats lack _provenance: {path}")
        for key in (
            "contract_version",
            "root_dir",
            "rgbd_variant",
            "rgbd_source_revision",
            "suites",
            "num_tasks",
            "num_demos",
            "num_valid_depth_anchors",
            "action_convention",
        ):
            if actual.get(key) != expected[key]:
                raise ValueError(
                    f"Normalization stats provenance mismatch for {key}: {actual.get(key)!r} != {expected[key]!r} in {path}."
                )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def task_index_groups(self) -> dict[str, list[int]]:
        """Return exact dataset-index ownership for the 40 LIBERO tasks."""
        groups: dict[str, list[int]] = {}
        previous = 0
        for episode, cumulative in zip(self.episodes, self.cumulative_sizes):
            task = f"{episode['suite']}/{episode['task']}"
            groups.setdefault(task, []).extend(range(previous, int(cumulative)))
            previous = int(cumulative)
        if previous != len(self):
            raise RuntimeError(
                f"LIBERO task index groups do not cover the dataset length: covered={previous}, length={len(self)}."
            )
        return groups

    def _locate(self, idx: int) -> tuple[dict[str, Any], int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}."
            )
        episode_position = bisect.bisect_right(self.cumulative_sizes, idx)
        previous = (
            0 if episode_position == 0 else self.cumulative_sizes[episode_position - 1]
        )
        return (
            self.episodes[episode_position],
            int((idx - previous) * self.sample_stride),
        )

    def _open(self, path: Path) -> h5py.File:
        if self._owner_pid != os.getpid():
            self._open_files.clear()
            self._owner_pid = os.getpid()
        key = str(path)
        handle = self._open_files.pop(key, None)
        if handle is None:
            handle = h5py.File(path, "r")
        self._open_files[key] = handle
        while len(self._open_files) > self.max_open_files:
            (_, old_handle) = self._open_files.popitem(last=False)
            old_handle.close()
        return handle

    @staticmethod
    def _convert_action(action: np.ndarray) -> np.ndarray:
        if action.ndim != 2 or action.shape[1] != 7 or (not np.isfinite(action).all()):
            raise ValueError(
                f"LIBERO actions must be finite [T,7], got {action.shape}."
            )
        result = np.asarray(action, dtype=np.float32).copy()
        raw_gripper = result[:, 6]
        if not np.all(np.isin(raw_gripper, (-1.0, 1.0))):
            raise ValueError(
                f"LIBERO raw gripper must be -1/+1, got {np.unique(raw_gripper)}."
            )
        result[:, 6] = (1.0 - raw_gripper) * 0.5
        return result

    @staticmethod
    def _padded_indices(
        start: int, count: int, length: int
    ) -> tuple[np.ndarray, torch.Tensor]:
        requested = np.arange(start, start + count, dtype=np.int64)
        return (
            np.minimum(requested, length - 1),
            torch.from_numpy(requested >= length),
        )

    def _read_vectors(
        self, demo: h5py.Group, start: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(demo["actions"].shape[0])
        (_, pad) = self._padded_indices(start, self.action_horizon, length)
        stop = min(start + self.action_horizon, length)
        action = np.asarray(demo["actions"][start:stop], dtype=np.float32)
        ee_state = np.asarray(demo["obs/ee_states"][start:stop], dtype=np.float32)
        gripper_state = np.asarray(
            demo["obs/gripper_states"][start:stop], dtype=np.float32
        )
        if len(action) < self.action_horizon:
            padding = self.action_horizon - len(action)
            action = np.concatenate(
                [action, np.repeat(action[-1:], padding, axis=0)], axis=0
            )
            ee_state = np.concatenate(
                [ee_state, np.repeat(ee_state[-1:], padding, axis=0)], axis=0
            )
            gripper_state = np.concatenate(
                [gripper_state, np.repeat(gripper_state[-1:], padding, axis=0)], axis=0
            )
        action = self._convert_action(action)
        state = np.concatenate([ee_state, gripper_state], axis=-1)
        action[pad.numpy(), :6] = 0.0
        batch = {
            "action": {"default": torch.from_numpy(action)},
            "state": {"default": torch.from_numpy(state)},
            "action_is_pad": pad.clone(),
            "state_is_pad": pad.clone(),
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return (batch["action"], batch["state"], pad, pad.clone())

    def _read_depth_video(
        self, demo: h5py.Group, start: int, valid_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(
            [start + offset for offset in self.frame_offsets], dtype=np.int64
        )
        indices = np.minimum(requested, valid_length - 1)
        (unique_indices, inverse) = np.unique(indices, return_inverse=True)
        camera_videos = []
        for camera in self.cameras:
            depth_mm = np.asarray(
                demo[f"obs/{camera}_depth"][unique_indices], dtype=np.uint16
            )[inverse, ..., 0]
            depth_m = torch.from_numpy(depth_mm.astype(np.float32) * 0.001).unsqueeze(1)
            valid = depth_m > 0.0
            depth_m = torch.where(valid, depth_m.clamp(0.0, self.depth_max_meters), 0.0)
            normalized = depth_m.mul(2.0 / self.depth_max_meters).sub(1.0)
            camera_videos.append(normalized)
        video = _compose_views(camera_videos).permute(1, 0, 2, 3).contiguous()
        return (video, torch.from_numpy(requested >= valid_length))

    def _read_rgb_image(self, demo: h5py.Group, start: int) -> torch.Tensor:
        images = []
        for camera in self.cameras:
            image = torch.from_numpy(
                np.asarray(demo[f"obs/{camera}_rgb"][start], dtype=np.uint8).copy()
            ).permute(2, 0, 1)
            image = image.float().div(127.5).sub(1.0).unsqueeze(0)
            images.append(image)
        return _compose_views(images)[0].contiguous()

    def _read_rgb_history(self, demo: h5py.Group, start: int) -> torch.Tensor:
        frames = [
            self._read_rgb_image(demo, index)
            for index in self.rgb_history.indices(start)
        ]
        return torch.stack(frames, dim=1)

    def _read_rgb_endpoint(
        self, demo: h5py.Group, episode: dict[str, Any], start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The RGB goal frame at the end of the window, clamped to the last frame."""
        requested_index = int(start) + int(self.frame_offsets[-1])
        frame_index = min(requested_index, int(episode["length"]) - 1)
        return (
            self._read_rgb_image(demo, frame_index),
            torch.tensor(requested_index < int(episode["length"]), dtype=torch.bool),
        )

    def _get_cached_text_context(
        self, prompt: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.text_embedding_cache_dir is None:
            return (None, None)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = (
            self.text_embedding_cache_dir
            / f"{digest}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing LIBERO text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context must be [{self.context_len},D], got {context.shape} in {path}."
            )
        if mask.shape != (self.context_len,):
            raise ValueError(
                f"Cached context mask must be [{self.context_len}], got {mask.shape} in {path}."
            )
        context = context.clone()
        context[~mask] = 0.0
        return (context, torch.ones_like(mask))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        (episode, start) = self._locate(idx)
        demo = self._open(episode["path"])[episode["group"]]
        (video, image_is_pad) = self._read_depth_video(
            demo, start, episode["valid_length"]
        )
        (action, proprio, action_is_pad, proprio_is_pad) = self._read_vectors(
            demo, start
        )
        prompt = PROMPT_TEMPLATE.format(task=episode["instruction"])
        (context, context_mask) = self._get_cached_text_context(prompt)
        sample: dict[str, Any] = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "prompt": prompt,
            "idx": int(idx),
            "task": episode["task"],
            "phase": episode["suite"],
            "episode_index": int(episode["episode_index"]),
            "start_index": int(start),
        }
        if context is not None:
            sample["context"] = context
            sample["context_mask"] = context_mask
        if self.rgb_target_mode is not None:
            history = self._read_rgb_history(demo, start)
            (endpoint, endpoint_is_valid) = self._read_rgb_endpoint(
                demo, episode, start
            )
            sample["rgb_video"] = torch.cat((history, endpoint.unsqueeze(1)), dim=1)
            sample["rgb_target_is_valid"] = endpoint_is_valid
            sample["rgb_target_mode"] = self.rgb_target_mode
        return sample
