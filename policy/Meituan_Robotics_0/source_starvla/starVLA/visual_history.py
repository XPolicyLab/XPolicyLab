"""Shared causal visual-history contract for dataset and online observations."""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from PIL import Image


class HistoryContractError(ValueError):
    """Missing source observations must not silently become a different sample."""


@dataclass(frozen=True)
class VisualHistoryConfig:
    window_sec: float = 16.0
    sample_fps: float = 0.5
    include_first: bool = True
    main_camera: str = "cam_high"
    image_size_hw: tuple | None = (256, 352)
    min_pixels: int | None = None
    max_pixels: int | None = None

    @classmethod
    def from_data_config(cls, data_cfg, dataset_name=None, *, for_inference=False):
        cfg = (data_cfg or {}).get("visual_history", {}) or {}
        if not cfg.get("enabled", False):
            return None
        cfg = dict(cfg)
        overrides = cfg.pop("dataset_overrides", {}) or {}
        inference_dataset = cfg.pop("inference_dataset", None)
        if for_inference:
            if overrides and inference_dataset not in overrides:
                raise HistoryContractError("eval with dataset_overrides requires an explicit inference_dataset override key")
            dataset_name = inference_dataset
        if dataset_name in overrides:
            cfg.update(dict(overrides[dataset_name]))
        cfg.pop("enabled", None)
        cfg.pop("src_fps", None)  # inference clock; dataset validates metadata separately
        if "image_size_hw" not in cfg and ("min_pixels" in cfg or "max_pixels" in cfg):
            cfg["image_size_hw"] = None
        if cfg.get("image_size_hw") is not None:
            cfg["image_size_hw"] = tuple(int(v) for v in cfg["image_size_hw"])
        spec = cls(**cfg)
        if spec.window_sec <= 0 or spec.sample_fps < 0:
            raise HistoryContractError("history window must be positive and sample_fps nonnegative")
        if (Fraction(str(spec.window_sec)) * Fraction(str(spec.sample_fps))).denominator != 1:
            raise HistoryContractError("window_sec * sample_fps must be an integer")
        if spec.image_size_hw is not None:
            if len(spec.image_size_hw) != 2 or any(int(v) <= 0 or int(v) % 32 for v in spec.image_size_hw):
                raise HistoryContractError("image_size_hw is [height,width], both positive multiples of 32")
            if spec.min_pixels is not None or spec.max_pixels is not None:
                raise HistoryContractError("choose fixed image_size_hw OR smart-resize pixel limits")
        if spec.min_pixels is not None and spec.min_pixels <= 0:
            raise HistoryContractError("min_pixels must be positive")
        if spec.max_pixels is not None and spec.max_pixels <= 0:
            raise HistoryContractError("max_pixels must be positive")
        if spec.min_pixels and spec.max_pixels and spec.min_pixels > spec.max_pixels:
            raise HistoryContractError("history min_pixels exceeds max_pixels")
        return spec

    @staticmethod
    def source_fps(data_cfg, dataset_name=None, *, for_inference=False):
        """Resolve the configured source clock; loaders still verify dataset metadata."""
        cfg = (data_cfg or {}).get("visual_history", {}) or {}
        key = cfg.get("inference_dataset") if for_inference else dataset_name
        override = (cfg.get("dataset_overrides", {}) or {}).get(key, {})
        value = override.get("src_fps", cfg.get("src_fps"))
        return None if value is None else float(value)

    def indices(self, frame_idx, src_fps):
        if frame_idx < 0 or src_fps <= 0 or self.sample_fps > src_fps:
            raise HistoryContractError("invalid source clock or history sampling rate")
        count = int(Fraction(str(self.window_sec)) * Fraction(str(self.sample_fps)))
        indices = set()
        if count:
            ratio = Fraction(str(src_fps)) / Fraction(str(self.sample_fps))
            indices = {frame_idx - math.ceil(j * ratio) for j in range(1, count + 1)}
        indices = {i for i in indices if 0 <= i < frame_idx}
        if self.include_first and frame_idx > 0:
            indices.add(0)
        return sorted(indices)


def image_metadata(spec, frame_idx, src_fps, views, timestamps=None):
    indices = spec.indices(frame_idx, src_fps)
    time_at = (lambda i: float(timestamps[i])) if timestamps is not None else (lambda i: i / src_fps)
    current_time = time_at(frame_idx)
    if not math.isfinite(current_time):
        raise HistoryContractError("nonfinite current timestamp")
    result = []
    for i, view, current in ([(i, spec.main_camera, False) for i in indices] + [(frame_idx, v, True) for v in views]):
        timestamp = time_at(i)
        age = current_time - timestamp
        if not math.isfinite(age) or age < -1e-6:
            raise HistoryContractError("history timestamps must be finite and causal")
        first = i == 0 and view == spec.main_camera and spec.include_first
        result.append(dict(frame_idx=i, timestamp_sec=timestamp, age_sec=max(0.0, age),
                           role="current" if current else ("first" if first else "history"),
                           view=view, is_episode_first=first))
    return result


def history_label(meta, main_camera="cam_high"):
    view = {main_camera: "main camera", "cam_left_wrist": "left wrist", "cam_right_wrist": "right wrist"}.get(meta["view"], meta["view"])
    if meta["role"] == "current":
        extra = " (episode start)" if meta.get("is_episode_first") else ""
        return f"Current, {view}{extra}:"
    role = "First frame" if meta["role"] == "first" else "History"
    return f'{role}, {view}, {meta["age_sec"]:.2f} seconds ago:'


class VisualHistoryBank:
    """CPU ring. Store all source ticks, sample only when policy replans."""
    def __init__(self, spec, src_fps):
        self.spec, self.src_fps = spec, float(src_fps)
        self.reset()

    def reset(self):
        self.frames = OrderedDict()
        self.first = None
        self.latest = -1

    def append(self, frame_idx, image, timestamp_sec=None):
        frame_idx = int(frame_idx)
        if frame_idx not in (self.latest, self.latest + 1):
            raise HistoryContractError(f"non-contiguous observation: {self.latest} -> {frame_idx}")
        if frame_idx < 0:
            raise HistoryContractError("negative observation index")
        timestamp = frame_idx / self.src_fps if timestamp_sec is None else float(timestamp_sec)
        if abs(timestamp - frame_idx / self.src_fps) > 1.0 / self.src_fps + 1e-6:
            raise HistoryContractError("observation clock differs from source FPS")
        entry = (np.array(image, copy=True), timestamp)
        self.frames[frame_idx] = entry
        if frame_idx == 0:
            self.first = entry
        self.latest = frame_idx
        oldest = frame_idx - math.ceil(self.spec.window_sec * self.src_fps)
        while self.frames and next(iter(self.frames)) < oldest:
            self.frames.popitem(last=False)

    def select(self, current_images, views):
        n = self.latest
        if n < 0:
            raise HistoryContractError("episode has no observation")
        indices = self.spec.indices(n, self.src_fps)
        entries = {}
        for i in indices + [n]:
            entry = self.first if i == 0 else self.frames.get(i)
            if entry is None:
                raise HistoryContractError(f"missing exact history frame {i} at {n}")
            entries[i] = entry
        metas = image_metadata(self.spec, n, self.src_fps, views, {i: e[1] for i, e in entries.items()})
        return [entries[i][0] for i in indices] + list(current_images), metas


def prepare_history_images(images, metas, spec, processor, native_size, current_size_wh=None):
    """Resolve each role's geometry once; HF normalization/patchification follows.

    native_size is captured BEFORE dataset current-frame pixel limits are applied.
    All callers use this same CPU path. No global processor settings are mutated.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    if len(images) != len(metas):
        raise HistoryContractError("image/metadata count mismatch")
    result = []
    for image, meta in zip(images, metas):
        image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        if meta["role"] != "current" and spec.image_size_hw is not None:
            height, width = map(int, spec.image_size_hw)
        else:
            if meta["role"] == "current":
                if current_size_wh:
                    image = image.resize(tuple(current_size_wh))
                limits = processor.image_processor.size
                minimum, maximum = limits["shortest_edge"], limits["longest_edge"]
            else:
                minimum = spec.min_pixels if spec.min_pixels is not None else native_size["shortest_edge"]
                maximum = spec.max_pixels if spec.max_pixels is not None else native_size["longest_edge"]
                if minimum > maximum:
                    raise HistoryContractError("resolved history min_pixels exceeds max_pixels")
            width0, height0 = image.size
            height, width = smart_resize(height0, width0, factor=32, min_pixels=minimum, max_pixels=maximum)
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
        result.append(image)
    return result
