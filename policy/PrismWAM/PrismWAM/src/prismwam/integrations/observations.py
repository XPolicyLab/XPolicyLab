"""Checkpoint-matched image preprocessing for decoded XPolicyLab observations.

This module does not decode JPEG, change RGB channel order, infer depth units,
estimate depth, normalize robot state, or own temporal history. Those contracts
are deliberately separate from spatial image preparation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from prismwam.datasets.real_world_dataset import (
    REAL_WORLD_TASK_SPECS,
    compose_real_world_views,
)
from prismwam.datasets.robotwin_depth_dataset import RobotWinDepthVideoDataset
from prismwam.deployment.observations import rgb_from_uint8
from prismwam.rgb_history import RGBHistory


_DUAL_ARM_RECIPES = ("robodojo", "robotwin")
_STANDARD_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_CAMERA_ROLES = ("external", "left_wrist", "right_wrist")


class Preprocessor:
    """Prepare float32 RGB CHW and scalar depth 1HW in [-1, 1].

    ``camera_map`` maps external/left_wrist/right_wrist roles to
    ``obs['vision']`` keys; explicit training camera names are also accepted,
    but cannot be mixed with role keys.
    ``depth_scale`` converts the explicitly selected input depth field to the
    units in the training config, before clipping or resizing. It is not guessed
    from dtype or magnitude. ``depth_key`` never falls back to another source:
    raw sensor depth and refined/approximate depth are not interchangeable.

    Real-world models must retain per-task source geometry and specify a task.
    LIBERO uses a different action contract and is not an XPolicyLab recipe.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        recipe: str,
        camera_map: Mapping[str, str] | None = None,
        depth_scale: float = 1.0,
        task_name: str | None = None,
        depth_key: str = "depth",
    ) -> None:
        self.recipe = str(recipe)
        self.depth_scale = float(depth_scale)
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive.")
        self.depth_key = str(depth_key)
        if not self.depth_key:
            raise ValueError("depth_key must explicitly name a depth field.")
        try:
            train = config["data"]["train"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Expected a config with data.train geometry.") from exc

        self.is_real_world = self.recipe == "realworld"
        if self.is_real_world:
            if task_name not in REAL_WORLD_TASK_SPECS:
                raise ValueError("A known real-world task_name is required.")
            sources = train.get("sources", [train])
            matches = [s for s in sources if task_name in s.get("task_names", [])]
            if len(matches) != 1:
                raise ValueError(
                    "The model must retain exactly one matching data.train source "
                    f"for task_name={task_name!r}."
                )
            self.train_config = matches[0]
            spec = REAL_WORLD_TASK_SPECS[task_name]
            self.cameras = tuple(view.name for view in spec.views)
            self.layout = spec.layout
            defaults = dict(zip(self.cameras, self.cameras))
        elif self.recipe in _DUAL_ARM_RECIPES:
            self.train_config = train
            self.cameras = tuple(str(k) for k in train["cameras"])
            if len(self.cameras) != 3:
                raise ValueError("Dual-arm recipes require three ordered cameras.")
            self.layout = str(train["layout"])
            defaults = dict(zip(self.cameras, _STANDARD_CAMERAS))
        else:
            raise ValueError(f"Recipe {self.recipe!r} is not supported by XPolicyLab.")

        self.rgb_history = RGBHistory.from_config(
            self.train_config, config.get("model")
        )
        self.video_size = tuple(int(x) for x in self.train_config["video_size"])
        if len(self.video_size) != 2 or min(self.video_size) <= 0:
            raise ValueError("video_size must contain positive height and width.")
        self.depth_min = float(self.train_config["depth_min"])
        self.depth_max = float(self.train_config["depth_max"])
        if not (
            np.isfinite(self.depth_min)
            and np.isfinite(self.depth_max)
            and self.depth_max > self.depth_min
        ):
            raise ValueError("Expected an increasing finite training depth range.")
        if camera_map is not None:
            roles = _CAMERA_ROLES[: len(self.cameras)]
            keys = set(camera_map)
            if keys <= set(roles):
                defaults.update(
                    {
                        camera: camera_map[role]
                        for role, camera in zip(roles, self.cameras)
                        if role in camera_map
                    }
                )
            elif keys <= set(self.cameras):
                defaults.update(camera_map)
            else:
                raise ValueError(
                    "camera_map must use only camera roles or only training camera "
                    "names; unknown or mixed keys are not allowed."
                )
        self.camera_map = {str(k): str(v) for k, v in defaults.items()}
        if len(set(self.camera_map.values())) != len(self.cameras):
            raise ValueError("Each training view must map to a distinct input camera.")

        # Reuse the dataset's pure layout routine without constructing a dataset,
        # opening indexes, loading statistics or writing any training artifacts.
        self._layout_config = SimpleNamespace(
            video_size=self.video_size,
            layout=self.layout,
            _resize=RobotWinDepthVideoDataset._resize,
        )

    def _read_views(self, obs: Mapping[str, Any]):
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("Observation requires a vision mapping.")
        rgb_views, depth_views = [], []
        for camera in self.cameras:
            source_name = self.camera_map[camera]
            source = vision.get(source_name)
            if not isinstance(source, Mapping):
                raise ValueError(f"Missing camera vision[{source_name!r}].")
            if "color" not in source:
                raise ValueError(f"Camera {source_name!r} has no decoded RGB color.")
            rgb = np.asarray(source["color"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("RGB color must already be decoded HWC uint8 RGB.")
            if self.depth_key not in source:
                raise ValueError(
                    f"Camera {source_name!r} has no {self.depth_key!r} depth field; "
                    "no alternative depth source is substituted."
                )
            depth = np.asarray(source[self.depth_key])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif (
                depth.ndim == 3
                and depth.shape[-1] == 3
                and (depth == depth[..., :1]).all()
            ):
                # A single-channel depth replicated across three channels.
                depth = depth[..., 0]
            if depth.ndim != 2 or depth.dtype.kind not in "uif":
                raise ValueError("Depth must be a decoded HW or HW1 numeric array.")
            if depth.shape != rgb.shape[:2]:
                raise ValueError("RGB and depth must be pixel-aligned for each camera.")
            if min(depth.shape) < 1:
                raise ValueError("Camera image cannot be empty.")
            # WS decoding can expose read-only views of the message buffer.
            # Own writable storage before torch.from_numpy, including when the
            # input was already contiguous.
            rgb_views.append(np.array(rgb, copy=True, order="C"))
            depth_views.append(depth.astype(np.float32) * self.depth_scale)
        return rgb_views, depth_views

    def prepare(self, obs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare one observation; no random augmentations are applied."""
        rgb_views, depth_views = self._read_views(obs)
        if self.is_real_world:
            if not all(np.isfinite(view).all() for view in depth_views):
                raise ValueError("Real-world deployment requires finite refined depth.")
            rgb = compose_real_world_views(
                rgb_views,
                layout=self.layout,
                size=self.video_size,
                interpolation="area",
            )
            depth = compose_real_world_views(
                depth_views,
                layout=self.layout,
                size=self.video_size,
                interpolation="nearest",
            )
            depth = np.clip(depth, self.depth_min, self.depth_max)
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min) * 2 - 1
            return (
                rgb_from_uint8(torch.from_numpy(rgb)).contiguous(),
                torch.from_numpy(depth).unsqueeze(0).contiguous(),
            )

        # The native simulation evaluator multiplies by 2/255 before resizing.
        # Keep that exact arithmetic here; real-world training above uses
        # division by 127.5, which has slightly different float32 rounding.
        rgb_tensors = [
            torch.from_numpy(view).permute(2, 0, 1).contiguous().float() * (2.0 / 255.0)
            - 1.0
            for view in rgb_views
        ]
        rgb = RobotWinDepthVideoDataset._layout_video(
            self._layout_config, [view[None] for view in rgb_tensors]
        )[0]
        depths = [
            RobotWinDepthVideoDataset._normalize_depth(self, view[None])
            for view in depth_views
        ]
        depth = RobotWinDepthVideoDataset._layout_video(self._layout_config, depths)[0]
        return rgb.contiguous(), depth.contiguous()
