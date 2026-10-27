# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HDF5 data loading for Awomo-0.5I RoboDojo SFT."""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from logging import getLogger as get_logger
from typing import Any, Optional, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from awomo.awomo.model import compose_robodojo_rgb_mosaic, resize_mosaic_for_vae
from awomo.awomo.datasets.robodojo.vlm_prompt import history_offsets_frames
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from awomo.awomo.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json

logger = get_logger(__name__)

_INDEX_CACHE: dict[str, Any] = {}


def _load_index(path: str):
    idx = _INDEX_CACHE.get(path)
    if idx is None:
        idx = torch.load(path, map_location="cpu", weights_only=False)
        directory = os.path.dirname(os.path.abspath(path))
        idx["episodes"] = [
            dict(ep, path=os.path.normpath(os.path.join(directory, ep["path"])))
            if not os.path.isabs(ep["path"]) else ep
            for ep in idx["episodes"]
        ]
        _INDEX_CACHE[path] = idx
    return idx


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _decode_jpeg_rgb(buf) -> Image.Image:
    """Decode both supported JPEG formats through XPolicyLab's shared helper."""
    from XPolicyLab.utils.process_data import decode_image_bit
    return Image.fromarray(decode_image_bit(buf))


DEFAULT_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
RAW_FPS = 25.0
GRIPPER_OPEN_IS_ONE = "open_is_one"
GRIPPER_CLOSED_IS_ONE = "closed_is_one"
_GRIPPER_SEMANTICS = {GRIPPER_OPEN_IS_ONE, GRIPPER_CLOSED_IS_ONE}


def _build_dense_window_index(episodes: Sequence[dict], window_stride: int = 1) -> np.ndarray:
    """Pool every episode's window starts; stride only the start, not its contents."""
    stride = int(window_stride)
    if stride <= 0:
        raise ValueError(f"window_stride must be positive, got {window_stride}")
    blocks = []
    for ep_idx, ep in enumerate(episodes):
        starts = np.arange(0, int(ep["n_frames"]) - 1, stride, dtype=np.int64)
        if starts.size:
            blocks.append(np.stack([np.full(starts.size, ep_idx, dtype=np.int64), starts], axis=1))
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, 2), dtype=np.int64)


def convert_normalized_gripper_semantics(
    action: torch.Tensor,
    *,
    gripper_indices: Sequence[int],
    source_semantics: str,
    target_semantics: str,
) -> torch.Tensor:
    """Flip only z-scored gripper channels once per direction: open=1 ↔ closed=1."""
    source_semantics = str(source_semantics)
    target_semantics = str(target_semantics)
    if source_semantics not in _GRIPPER_SEMANTICS:
        raise ValueError(f"unsupported source gripper semantics: {source_semantics!r}")
    if target_semantics not in _GRIPPER_SEMANTICS:
        raise ValueError(f"unsupported target gripper semantics: {target_semantics!r}")

    indices = tuple(int(v) for v in gripper_indices)
    if len(set(indices)) != len(indices):
        raise ValueError(f"gripper_indices contains duplicates: {indices}")
    if any(v < 0 or v >= action.shape[-1] for v in indices):
        raise ValueError(f"gripper_indices must be in [0, {action.shape[-1]}), got {indices}")
    if source_semantics == target_semantics or not indices:
        return action

    converted = action.clone()
    converted[..., list(indices)] = -converted[..., list(indices)]
    return converted


def scatter_action_dims(
    action: torch.Tensor,
    *,
    output_dim: int,
    output_indices: Optional[Sequence[int]],
) -> tuple[torch.Tensor, torch.BoolTensor]:
    if output_indices is None:
        if action.shape[-1] != int(output_dim):
            raise ValueError(f"action has {action.shape[-1]} dims but output_dim={output_dim} without a mapping")
        return action, torch.zeros(int(output_dim), dtype=torch.bool, device=action.device)
    indices = tuple(int(v) for v in output_indices)
    if len(indices) != action.shape[-1]:
        raise ValueError(f"mapping has {len(indices)} entries for action dim {action.shape[-1]}")
    output = action.new_zeros(*action.shape[:-1], int(output_dim))
    output[..., list(indices)] = action
    dim_is_pad = torch.ones(int(output_dim), dtype=torch.bool, device=action.device)
    dim_is_pad[list(indices)] = False
    return output, dim_is_pad


class DenseHistorySampler:
    """One past frame per slot, slots ordered oldest -> newest."""

    def __init__(
        self,
        offset_bins_sec,
        fps=RAW_FPS,
        slot_dropout_prob=0.2,
        no_history_prob=0.2,
        random_num_slots=True,
    ):
        bins = []
        for lo, hi in offset_bins_sec:
            lo_f = max(1, int(round(float(lo) * fps)))
            hi_f = max(lo_f, int(round(float(hi) * fps)))
            bins.append((lo_f, hi_f))
        order = sorted(range(len(bins)), key=lambda i: bins[i][0], reverse=True)
        self.bins_frames = [bins[i] for i in order]
        self.num_slots = len(self.bins_frames)
        self.slot_dropout_prob = float(slot_dropout_prob)
        self.no_history_prob = float(no_history_prob)
        self.random_num_slots = bool(random_num_slots)

    def sample(self, current_t: int):
        n = self.num_slots
        if self.no_history_prob > 0.0 and torch.rand(1).item() < self.no_history_prob:
            return [None] * n
        keep_newest = int(torch.randint(0, n + 1, (1,)).item()) if self.random_num_slots else None
        out = []
        for lo_f, hi_f in self.bins_frames:
            hi_eff = min(hi_f, current_t)
            if hi_eff < lo_f:
                out.append(None)
                continue
            if self.slot_dropout_prob > 0.0 and torch.rand(1).item() < self.slot_dropout_prob:
                out.append(None)
                continue
            d = int(torch.randint(lo_f, hi_eff + 1, (1,)).item())
            out.append(current_t - d)
        if keep_newest is not None:
            valid = [i for i, v in enumerate(out) if v is not None]
            for i in valid[: max(0, len(valid) - keep_newest)]:
                out[i] = None
        return out


class VLMHistorySampler:
    """Sample available past frames near a fixed grid, oldest-to-newest; omit missing frames."""

    def __init__(
        self,
        num_slots=20,
        period_s=1.0,
        fps=RAW_FPS,
        jitter_frames=1,
        camera="cam_head",
        image_size=(448, 448),
        pool=(4, 4),
    ):
        self.num_slots = int(num_slots)
        self.period_s = float(period_s)
        self.period_frames = float(period_s) * float(fps)
        self.jitter = int(jitter_frames)
        self.camera = str(camera)
        self.image_size = tuple(int(v) for v in image_size)
        self.pool = (
            int(pool[0]),
            int(pool[1]),
        )
        self.offsets = history_offsets_frames(self.num_slots, self.period_frames)
        if self.jitter * 2 >= min(b - a for a, b in zip(self.offsets[1:], self.offsets[:-1])):
            raise ValueError(f"jitter_frames={self.jitter} could reorder frames {self.period_frames} frames apart")
        self.max_lookback = max(self.offsets)

    def sample(self, current_t: int):
        out = []
        for d in self.offsets:
            if self.jitter > 0:
                d += int(torch.randint(-self.jitter, self.jitter + 1, (1,)).item())
            d = max(d, 1)
            if d > current_t:
                continue
            out.append(current_t - d)
        return out


class RoboDojoHDF5Dataset(torch.utils.data.Dataset):
    """RoboDojo HDF5 windows for image/action/state conditioning."""

    def __init__(
        self,
        index_file: str,
        pretrained_norm_stats: str,
        num_frames: int = 17,
        video_size: Sequence[int] = (256, 256),
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        concat_multi_camera: Optional[str] = None,
        image_layout: str = "separate",
        image_color_order: str = "rgb",
        is_training_set: bool = True,
        val_set_proportion: float = 0.01,
        split_seed: int = 42,
        windows_per_episode: Optional[int] = None,
        window_stride: int = 1,
        norm_default_mode: str = "z-score",
        norm_exception_mode: Optional[dict] = None,
        use_stepwise_action_norm: bool = False,
        action_dim: int = 14,
        action_output_indices: Optional[Sequence[int]] = None,
        action_gripper_indices: Sequence[int] = (),
        action_gripper_source_semantics: str = GRIPPER_OPEN_IS_ONE,
        action_gripper_target_semantics: str = GRIPPER_OPEN_IS_ONE,
        state_dim: int = 14,
        state_output_indices: Optional[Sequence[int]] = None,
        state_gripper_indices: Sequence[int] = (),
        state_gripper_source_semantics: str = GRIPPER_OPEN_IS_ONE,
        state_gripper_target_semantics: str = GRIPPER_OPEN_IS_ONE,
        history: Optional[dict] = None,
        video_augmentation: Optional[Any] = None,
        handle_cache_size: int = 48,
        override_instruction: Optional[str] = None,
        max_retry: int = 4,
        vlm: Optional[dict] = None,
        verify_paths: bool = True,
        max_missing_fraction: float = 0.02,
    ):
        import h5py

        self._h5py = h5py
        if num_frames < 2:
            raise ValueError(f"`num_frames` must be >= 2, got {num_frames}")
        self.num_frames = int(num_frames)
        self.action_horizon = self.num_frames - 1
        self.target_offset = self.num_frames - 1
        self.video_size = tuple(int(v) for v in video_size)
        self.cameras = list(cameras)
        self.concat_multi_camera = concat_multi_camera
        self.image_layout = image_layout
        self.image_color_order = image_color_order
        if image_color_order not in ("rgb", "bgr"):
            raise ValueError(f"Unsupported image_color_order={image_color_order!r}")
        if image_color_order == "bgr" and image_layout != "hero_top":
            raise ValueError("BGR input requires hero_top mosaic")
        if image_layout not in ("separate", "hero_top"):
            raise ValueError(f"Unsupported RoboDojo image_layout={image_layout!r}")
        if image_layout == "hero_top" and (self.cameras != list(DEFAULT_CAMERAS) or concat_multi_camera is not None):
            raise ValueError("hero_top requires the three canonical cameras and concat_multi_camera=None")
        self.train = bool(is_training_set)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.handle_cache_size = int(handle_cache_size)
        self.override_instruction = override_instruction
        self.max_retry = int(max_retry)

        if self.concat_multi_camera not in (None, "horizontal", "vertical"):
            raise ValueError(
                f"Unsupported concat_multi_camera={self.concat_multi_camera!r}; "
                "expected None (independent views), 'horizontal' or 'vertical'."
            )

        index = _load_index(index_file)
        episodes = index["episodes"]
        self.actions = index["actions"]
        self.states = index["states"]
        self.raw_action_dim = int(self.actions.shape[1])
        if action_output_indices is None:
            if self.raw_action_dim != self.action_dim:
                raise ValueError(
                    f"index action dim {self.raw_action_dim} differs from output action_dim={self.action_dim}; "
                    "set action_output_indices explicitly"
                )
            self.action_output_indices = None
        else:
            indices = tuple(int(v) for v in action_output_indices)
            if len(indices) != self.raw_action_dim:
                raise ValueError(
                    f"action_output_indices has {len(indices)} entries, expected raw action dim {self.raw_action_dim}"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"action_output_indices contains duplicates: {indices}")
            if any(v < 0 or v >= self.action_dim for v in indices):
                raise ValueError(f"action_output_indices must be in [0, {self.action_dim}), got {indices}")
            self.action_output_indices = indices
        self.action_gripper_indices = tuple(int(v) for v in action_gripper_indices)
        self.action_gripper_source_semantics = str(action_gripper_source_semantics)
        self.action_gripper_target_semantics = str(action_gripper_target_semantics)
        convert_normalized_gripper_semantics(
            torch.zeros(self.raw_action_dim),
            gripper_indices=self.action_gripper_indices,
            source_semantics=self.action_gripper_source_semantics,
            target_semantics=self.action_gripper_target_semantics,
        )
        self.raw_state_dim = int(self.states.shape[1])
        if state_output_indices is None:
            if self.raw_state_dim != self.state_dim:
                raise ValueError(
                    f"index state dim {self.raw_state_dim} differs from output state_dim={self.state_dim}; "
                    "set state_output_indices explicitly"
                )
            self.state_output_indices = None
        else:
            indices = tuple(int(v) for v in state_output_indices)
            if len(indices) != self.raw_state_dim:
                raise ValueError(
                    f"state_output_indices has {len(indices)} entries, expected raw state dim {self.raw_state_dim}"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"state_output_indices contains duplicates: {indices}")
            if any(v < 0 or v >= self.state_dim for v in indices):
                raise ValueError(f"state_output_indices must be in [0, {self.state_dim}), got {indices}")
            self.state_output_indices = indices
        self.state_gripper_indices = tuple(int(v) for v in state_gripper_indices)
        self.state_gripper_source_semantics = str(state_gripper_source_semantics)
        self.state_gripper_target_semantics = str(state_gripper_target_semantics)
        convert_normalized_gripper_semantics(
            torch.zeros(self.raw_state_dim),
            gripper_indices=self.state_gripper_indices,
            source_semantics=self.state_gripper_source_semantics,
            target_semantics=self.state_gripper_target_semantics,
        )

        n_ep = len(episodes)
        perm = torch.randperm(n_ep, generator=torch.Generator().manual_seed(int(split_seed))).tolist()
        n_val = int(round(float(val_set_proportion) * n_ep))
        keep = perm[n_val:] if self.train else perm[:n_val]
        if not keep:
            keep = perm[:1]
        self.episodes = [episodes[i] for i in sorted(keep)]

        if verify_paths:
            present, missing = [], []
            for ep in self.episodes:
                (present if os.path.exists(ep["path"]) else missing).append(ep)
            if missing:
                by_task: dict[str, int] = {}
                by_prefix: dict[str, int] = {}
                for ep in missing:
                    by_task[ep.get("task", "?")] = by_task.get(ep.get("task", "?"), 0) + 1
                    pref = "/".join(ep["path"].split("/")[:6])
                    by_prefix[pref] = by_prefix.get(pref, 0) + 1
                frac = len(missing) / max(len(self.episodes), 1)
                detail = (
                    f"{len(missing)}/{len(self.episodes)} episodes ({frac:.1%}) listed in "
                    f"{index_file} no longer exist on disk.\n"
                    f"  by task: {by_task}\n"
                    f"  by path prefix: {dict(sorted(by_prefix.items(), key=lambda x: -x[1])[:5])}\n"
                    f"  first missing: {missing[0]['path']}"
                )
                if frac > float(max_missing_fraction):
                    raise FileNotFoundError(
                        detail + "\n\nThis exceeds max_missing_fraction="
                        f"{max_missing_fraction:.1%}. Rebuild the index with "
                        "the training index against the current layout, or restore missing files. Raise "
                        "max_missing_fraction only if training on the reduced set is intended."
                    )
                logger.warning("RoboDojoHDF5Dataset: dropping missing episodes -- %s", detail)
                self.episodes = present

        self.window_stride = int(window_stride)
        if self.window_stride <= 0:
            raise ValueError(f"window_stride must be positive, got {window_stride}")
        self.one_per_episode = windows_per_episode == 1
        if self.one_per_episode:
            if self.window_stride != 1:
                raise ValueError("window_stride is only valid for dense windows")
            self.windows = np.stack(
                [
                    np.arange(len(self.episodes), dtype=np.int64),
                    np.full(len(self.episodes), -1, dtype=np.int64),
                ],
                axis=1,
            )
        else:
            self.windows = _build_dense_window_index(self.episodes, self.window_stride)

        self.task_names = sorted({ep["task"] for ep in self.episodes})
        _tid = {n: i for i, n in enumerate(self.task_names)}
        _ep_tid = np.array([_tid[ep["task"]] for ep in self.episodes], dtype=np.int64)
        self.window_task = _ep_tid[self.windows[:, 0]]

        shape_meta = {
            "action": [
                {
                    "key": "default",
                    "raw_shape": self.raw_action_dim,
                    "shape": self.raw_action_dim,
                }
            ],
            "state": [{"key": "default", "raw_shape": self.state_dim, "shape": self.state_dim}],
        }
        stats = load_dataset_stats_from_json(pretrained_norm_stats)
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=bool(use_stepwise_action_norm),
            default_mode=norm_default_mode,
            exception_mode=OmegaConf.to_container(norm_exception_mode, resolve=True)
            if isinstance(norm_exception_mode, DictConfig)
            else norm_exception_mode,
            stats=stats,
        )

        self.history_sampler = None
        self.history_num_slots = 0
        self.history_cameras = [self.cameras[0]]
        if history is not None and not isinstance(history, dict):
            history = OmegaConf.to_container(history, resolve=True)
        if history and history.get("enabled", False):
            self.history_sampler = DenseHistorySampler(
                offset_bins_sec=history["offset_bins_sec"],
                fps=float(history.get("fps", RAW_FPS)),
                slot_dropout_prob=float(history.get("slot_dropout_prob", 0.2)),
                no_history_prob=float(history.get("no_history_prob", 0.2)),
                random_num_slots=bool(history.get("random_num_slots", True)),
            )
            self.history_num_slots = self.history_sampler.num_slots
            cam = history.get("camera", self.cameras[0])
            self.history_cameras = [cam] if isinstance(cam, str) else list(cam)
            max_slots = history.get("max_slots")
            if max_slots is not None and int(max_slots) != self.history_num_slots:
                raise ValueError(
                    f"history.max_slots={max_slots} disagrees with len(offset_bins_sec)="
                    f"{self.history_num_slots}; the model's per-slot positions depend on it."
                )
            if not self.train:
                self.history_sampler.slot_dropout_prob = 0.0
                self.history_sampler.no_history_prob = 0.0
                self.history_sampler.random_num_slots = False

        self.video_augmentation = None
        if video_augmentation is not None and self.train:
            self.video_augmentation = (
                video_augmentation
                if isinstance(video_augmentation, torch.nn.Module)
                else instantiate(video_augmentation)
            )

        self.vlm_codec = None
        self.vlm_history = None
        self.vlm_image_size = (448, 448)
        if vlm is not None and not isinstance(vlm, dict):
            vlm = OmegaConf.to_container(vlm, resolve=True)
        if vlm and vlm.get("enabled", False):
            from awomo.awomo.datasets.robodojo.vlm_prompt import VLMPromptCodec
            from transformers import AutoProcessor

            self.vlm_image_size = tuple(int(v) for v in vlm.get("image_size", (448, 448)))
            hist = vlm.get("history") or {}
            if hist.get("enabled", True):
                self.vlm_history = VLMHistorySampler(
                    num_slots=int(hist.get("num_slots", 20)),
                    period_s=float(hist.get("period_s", 1.0)),
                    fps=float(hist.get("fps", RAW_FPS)),
                    jitter_frames=int(hist.get("jitter_frames", 1)) if self.train else 0,
                    camera=str(hist.get("camera", self.cameras[0])),
                    image_size=tuple(int(v) for v in hist.get("image_size", (448, 448))),
                    pool=tuple(int(v) for v in hist.get("pool", (4, 4))),
                )
            self.vlm_codec = VLMPromptCodec(
                AutoProcessor.from_pretrained(vlm["model_path"]),
                self.vlm_image_size,
                history_image_size=(self.vlm_history.image_size if self.vlm_history else (448, 448)),
                history_pool=(self.vlm_history.pool if self.vlm_history else (4, 4)),
                history_period_s=(self.vlm_history.period_s if self.vlm_history else 1.0),
                max_history=(self.vlm_history.num_slots if self.vlm_history else 0),
            )
            logger.info("RoboDojoHDF5Dataset: VLM conditioning enabled; image=%s history=%s",
                        self.vlm_image_size, self.vlm_history is not None)

        self._handles = OrderedDict()
        self._vanished: set = set()
        self._vanished_budget = 50

        by_task: dict[str, int] = {}
        for ep in self.episodes:
            by_task[ep.get("task", "?")] = by_task.get(ep.get("task", "?"), 0) + 1
        logger.info(
            "RoboDojoHDF5Dataset[%s]: %d windows / %d episodes (%s), views=%s concat=%s "
            "size=%s horizon=%d history=%s",
            "train" if self.train else "val",
            len(self.windows),
            len(self.episodes),
            ", ".join(f"{k}:{v}" for k, v in sorted(by_task.items())),
            self.cameras,
            self.concat_multi_camera,
            self.video_size,
            self.action_horizon,
            f"{self.history_num_slots} slots@{self.history_cameras}" if self.history_sampler else "off",
        )

    def __len__(self):
        return len(self.windows)

    def state_dict(self):
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        random.setstate(state["python"])
        algorithm, keys, position, has_gauss, cached_gaussian = state["numpy"]
        np.random.set_state((algorithm, np.frombuffer(keys, dtype=np.uint32).copy(), position, has_gauss, cached_gaussian))
        torch.set_rng_state(state["torch"])

    def _handle(self, path):
        h = self._handles.get(path)
        if h is None:
            h = self._h5py.File(path, "r")
            self._handles[path] = h
            if len(self._handles) > self.handle_cache_size:
                _, old = self._handles.popitem(last=False)
                old.close()
        else:
            self._handles.move_to_end(path)
        return h

    def _frame(self, handle, camera: str, t: int) -> torch.Tensor:
        if self.image_layout == "hero_top":
            return resize_mosaic_for_vae(self._rgb_mosaic(handle, t), self.video_size)
        buf = handle[f"vision/{camera}/colors"][t]
        img = _decode_jpeg_rgb(buf)
        h, w = self.video_size
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
        return x.permute(2, 0, 1).float() / 255.0

    def _frame_pil(self, handle, camera: str, t: int, size=None):
        size = tuple(int(v) for v in (size or self.vlm_image_size))
        if self.image_layout == "hero_top":
            return Image.fromarray(self._rgb_mosaic(handle, t)).resize(size, Image.BILINEAR)
        buf = handle[f"vision/{camera}/colors"][t]
        return _decode_jpeg_rgb(buf).resize(size, Image.BILINEAR)

    def _rgb_mosaic(self, handle, t: int) -> np.ndarray:
        frames = [
            np.asarray(_decode_jpeg_rgb(handle[f"vision/{camera}/colors"][t]), dtype=np.uint8)
            for camera in self.cameras
        ]
        return compose_robodojo_rgb_mosaic(frames, color_order=self.image_color_order)

    def _build_vlm(self, handle, ep, t: int):
        history = []
        if self.vlm_history is not None:
            cam, size = self.vlm_history.camera, self.vlm_history.image_size
            history = [self._frame_pil(handle, cam, ht, size) for ht in self.vlm_history.sample(t)]
        return self.vlm_codec.encode_sample(
            (self.override_instruction or ep["instruction"]).strip(),
            self._frame_pil(handle, self.cameras[0], t, self.vlm_image_size),
            history,
        )

    def _sample_start(self, ep) -> int:
        n = ep["n_frames"]
        hi = max(n - self.target_offset, 1) if n > self.target_offset + 1 else max(n - 1, 1)
        return int(torch.randint(0, hi, (1,)).item()) if self.train else hi // 2

    def _get(self, idx: int):
        ep_idx, t = (int(v) for v in self.windows[idx])
        ep = self.episodes[ep_idx]
        if t < 0:
            t = self._sample_start(ep)
        n_frames = ep["n_frames"]
        handle = self._handle(ep["path"])
        tgt_t = min(t + self.target_offset, n_frames - 1)

        if self.image_layout == "hero_top":
            views = torch.stack(
                [
                    self._frame(handle, self.cameras[0], t),
                    self._frame(handle, self.cameras[0], tgt_t),
                ]
            ).unsqueeze(0)
        else:
            views = torch.stack(
                [torch.stack([self._frame(handle, cam, t), self._frame(handle, cam, tgt_t)]) for cam in self.cameras]
            )
        image_is_pad = torch.zeros(2, dtype=torch.bool)
        image_is_pad[1] = (t + self.target_offset) > (n_frames - 1)

        history = None
        if self.history_sampler is not None:
            history = self._build_history(handle, t)

        if self.video_augmentation is not None:
            views = self.video_augmentation(views)
            if history is not None and history[0].numel():
                history = (self.video_augmentation(history[0].unsqueeze(0))[0],) + history[1:]

        views = views * 2.0 - 1.0
        if self.concat_multi_camera == "horizontal":
            video = torch.cat([views[v] for v in range(views.shape[0])], dim=-1)
        elif self.concat_multi_camera == "vertical":
            video = torch.cat([views[v] for v in range(views.shape[0])], dim=-2)
        else:
            video = views

        if self.concat_multi_camera is None:
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            if video.shape[0] == 1 and self.image_layout == "separate":
                video = video[0]
        else:
            video = video.permute(1, 0, 2, 3).contiguous()

        o = ep["offset"]
        hi = min(t + self.action_horizon, n_frames)
        action = self.actions[o + t : o + hi].clone().float()
        proprio = self.states[o + t : o + hi].clone().float()
        action_is_pad = torch.zeros(self.action_horizon, dtype=torch.bool)
        n_valid = action.shape[0]
        if n_valid < self.action_horizon:
            pad = self.action_horizon - n_valid
            action_is_pad[n_valid:] = True
            action = torch.cat([action, action[-1:].repeat(pad, 1)], dim=0)
            proprio = torch.cat([proprio, proprio[-1:].repeat(pad, 1)], dim=0)
        proprio_is_pad = action_is_pad.clone()

        batch = {"action": {"default": action}, "state": {"default": proprio}}
        batch = self.normalizer.forward(batch)
        action = batch["action"]["default"].float()
        proprio = batch["state"]["default"].float()

        action = convert_normalized_gripper_semantics(
            action,
            gripper_indices=self.action_gripper_indices,
            source_semantics=self.action_gripper_source_semantics,
            target_semantics=self.action_gripper_target_semantics,
        )
        proprio = convert_normalized_gripper_semantics(
            proprio,
            gripper_indices=self.state_gripper_indices,
            source_semantics=self.state_gripper_source_semantics,
            target_semantics=self.state_gripper_target_semantics,
        )

        action, action_dim_is_pad = scatter_action_dims(
            action,
            output_dim=self.action_dim,
            output_indices=self.action_output_indices,
        )
        proprio, proprio_dim_is_pad = scatter_action_dims(
            proprio,
            output_dim=self.state_dim,
            output_indices=self.state_output_indices,
        )

        task = self.override_instruction or ep["instruction"]
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "task": ep["task"],
        }
        if history is not None:
            frames, valid, age = history
            data["history_frames"] = frames * 2.0 - 1.0
            data["history_valid_mask"] = valid
            data["history_age_steps"] = age
            data["history_slot_index"] = torch.arange(self.history_num_slots, dtype=torch.long)
        if self.vlm_codec is not None:
            data["vlm"] = self._build_vlm(handle, ep, t)
        return data

    def collate_fn(self, batch):
        from awomo.awomo.datasets.robodojo.vlm_prompt import collate_vlm
        from torch.utils.data._utils.collate import default_collate

        if "vlm" not in batch[0]:
            return default_collate(batch)
        items = [b.pop("vlm") for b in batch]
        out = default_collate(batch)
        out["vlm"] = collate_vlm(items)
        return out

    def _build_history(self, handle, current_t: int):
        n = self.history_num_slots
        h, w = self.video_size
        cam = self.history_cameras[0]
        frames = torch.zeros(n, 3, h, w)
        valid = torch.zeros(n, dtype=torch.bool)
        age = torch.zeros(n, dtype=torch.long)
        for slot, ht in enumerate(self.history_sampler.sample(current_t)):
            if ht is None:
                continue
            frames[slot] = self._frame(handle, cam, ht)
            valid[slot] = True
            age[slot] = current_t - ht
        return frames, valid, age

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retry + 1):
            try:
                return self._get(idx)
            except Exception as exc:
                ep_idx = int(self.windows[idx][0])
                path = self.episodes[ep_idx]["path"]
                if isinstance(exc, (FileNotFoundError, OSError)) and not os.path.exists(path):
                    self._vanished.add(path)
                    if len(self._vanished) > self._vanished_budget:
                        raise FileNotFoundError(
                            f"{len(self._vanished)} indexed episodes have disappeared while this run "
                            f"was training (budget {self._vanished_budget}); the dataset is being "
                            f"modified underneath it. Latest: {path}. Rebuild the index against the "
                            "current layout before restarting."
                        ) from exc
                    if len(self._vanished) <= 5:
                        logger.warning(
                            "RoboDojoHDF5Dataset: episode vanished mid-run (%d so far): %s",
                            len(self._vanished),
                            path,
                        )
                else:
                    logger.warning(
                        "RoboDojoHDF5Dataset: sample %d (%s) failed: %s -- retry %d/%d with a random sample",
                        idx,
                        path,
                        exc,
                        attempt + 1,
                        self.max_retry,
                    )
                if attempt >= self.max_retry:
                    raise
                idx = int(np.random.randint(len(self.windows)))
