"""Per-environment full-frame head history for websocket inference."""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

IMG_HW = 224


@lru_cache(maxsize=4)
def _source_from_deploy(policy_dir: Path) -> str | None:
    config_path = policy_dir / "deploy.yml"
    if not config_path.is_file():
        return None
    import yaml

    config = yaml.safe_load(config_path.read_text()) or {}
    return config.get("focus_vlwa_source")


def _configure_client_source() -> None:
    policy_dir = Path(__file__).resolve().parent
    configured = os.environ.get("FOCUS_VLWA_SOURCE") or _source_from_deploy(policy_dir)
    source = Path(configured).expanduser() if configured else policy_dir / "focus-vlwa" / "src"
    if configured and not source.is_absolute():
        source = policy_dir / source
    if configured and not source.is_dir():
        raise FileNotFoundError(f"Focus-VLWA source directory not found: {source}")
    if source.is_dir():
        resolved = str(source.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((IMG_HW, IMG_HW, 3), np.uint8)
    arr = np.asarray(img)
    if arr.dtype == object or arr.ndim == 0:
        return np.zeros((IMG_HW, IMG_HW, 3), np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        return np.zeros((IMG_HW, IMG_HW, 3), np.uint8)
    return np.asarray(arr, np.uint8)


def _decode_raw_image(image) -> np.ndarray:
    """Read client-side camera fields through the shared RGB decoder."""
    if isinstance(image, dict):
        for key in ("color", "rgb", "image"):
            if key in image:
                return _decode_raw_image(image[key])
        raise KeyError("image dict missing color/rgb")
    from XPolicyLab.utils.process_data import decode_image_bit

    return chw_to_hwc(decode_image_bit(image))


def extract_images_from_obs(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Raw Isaac / already-encoded obs → HWC uint8 for the 3 policy cams."""
    images = observation.get("images") if isinstance(observation.get("images"), dict) else {}
    vision = observation.get("vision") if isinstance(observation.get("vision"), dict) else {}
    names = {
        "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
        "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
        "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
    }
    out: dict[str, np.ndarray] = {}
    for dest, cands in names.items():
        found = None
        for key in cands:
            if key in images:
                found = images[key]
                break
            if key in vision:
                found = vision[key]
                break
        if found is None:
            raise KeyError(f"missing camera for {dest}: {cands}")
        out[dest] = _decode_raw_image(found)
    return out


def hist_from_obs(observation: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Validate client-stamped full head history."""
    _configure_client_source()
    from focus_vlwa.data.head_history import HISTORY_FRAMES, IMAGE_SIZE, pack_history

    if "hist_cells" in observation:
        raise ValueError("History cells are not supported; provide full head frames")
    images = observation.get("hist_images")
    if images is None:
        return None
    if np.shape(images) != (HISTORY_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("Expected 20 full head-history frames")
    packed = pack_history(images, observation.get("hist_mask"))
    if "hist_t0" in observation:
        packed["hist_t0"] = np.int32(observation["hist_t0"])
    return packed


class LiveHeadHistoryTracker:
    """Collect full head frames on every control step, independently per environment."""

    def __init__(self):
        self.buffers = {}

    def reset_all(self):
        self.buffers.clear()

    def stamp_obs(self, observation, env_idx=0):
        _configure_client_source()
        from focus_vlwa.data.head_history import HeadHistoryBuffer

        buffer = self.buffers.setdefault(int(env_idx), HeadHistoryBuffer())
        images = extract_images_from_obs(observation)
        step = buffer.next_step
        buffer.observe(step, images["cam_high"])
        observation.update(buffer.pack(step))
        observation["hist_t0"] = step
        return observation

    def stamp_obs_list(self, observations, env_idx_list=None):
        ids = env_idx_list or [int(obs.get("env_idx", index)) for index, obs in enumerate(observations)]
        for observation, env_idx in zip(observations, ids, strict=True):
            self.stamp_obs(observation, env_idx)
        return observations
