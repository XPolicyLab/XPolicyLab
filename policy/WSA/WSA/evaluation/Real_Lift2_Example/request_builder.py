from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_CAMERA_MAP = {
    "head": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def to_chw_uint8(image: Any) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = np.transpose(image_np, (2, 0, 1))
    elif image_np.ndim == 3 and image_np.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Invalid image shape: {image_np.shape}")

    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image_np)) <= 1.5 else 1.0
            image_np = np.clip(image_np * scale, 0.0, 255.0).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image_np)


def resize_chw_uint8_nearest(
    image: np.ndarray,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    if target_height is None or target_width is None:
        return np.ascontiguousarray(image)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"send_image_height/send_image_width must be positive when provided, got "
            f"{target_height}x{target_width}."
        )
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got shape={image.shape}")

    _, src_height, src_width = image.shape
    if src_height == target_height and src_width == target_width:
        return np.ascontiguousarray(image)

    y_idx = np.rint(np.linspace(0, src_height - 1, target_height)).astype(np.int32)
    x_idx = np.rint(np.linspace(0, src_width - 1, target_width)).astype(np.int32)
    resized = image[:, y_idx[:, None], x_idx[None, :]]
    return np.ascontiguousarray(resized)


def prepare_history_frame(
    image: Any,
    *,
    send_image_height: int | None = None,
    send_image_width: int | None = None,
) -> np.ndarray:
    chw_image = to_chw_uint8(image)
    return resize_chw_uint8_nearest(
        chw_image,
        target_height=send_image_height,
        target_width=send_image_width,
    )


def build_history_stack(history_frames: list[np.ndarray], image_history_interval: int) -> np.ndarray:
    if not history_frames:
        raise ValueError("history_frames must contain at least one frame.")

    past_idx = max(len(history_frames) - image_history_interval - 1, 0)
    current = history_frames[-1]
    past = history_frames[past_idx]
    return np.stack([past, current], axis=0)


def build_wsa_base_request(
    *,
    qpos: np.ndarray,
    image_histories: Mapping[str, list[np.ndarray]],
    prompt: str,
    timestep: int,
    image_history_interval: int = 15,
    state_dim: int = 14,
    camera_name_map: Mapping[str, str] | None = None,
    send_image_height: int | None = None,
    send_image_width: int | None = None,
    inference_delay: int | None = None,
    prev_chunk_left_over: np.ndarray | None = None,
    prev_chunk_left_over_processed: np.ndarray | None = None,
) -> dict[str, Any]:
    if camera_name_map is None:
        camera_name_map = DEFAULT_CAMERA_MAP

    state = np.zeros((state_dim,), dtype=np.float32)
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    state[: min(state_dim, qpos.size)] = qpos[:state_dim]

    images: dict[str, np.ndarray] = {}
    blank_height = send_image_height or 480
    blank_width = send_image_width or 640
    for local_name, remote_name in camera_name_map.items():
        history = image_histories.get(local_name, [])
        if not history:
            blank = np.zeros((3, blank_height, blank_width), dtype=np.uint8)
            images[remote_name] = np.stack([blank, blank], axis=0)
            continue

        chw_history = [
            prepare_history_frame(
                frame,
                send_image_height=send_image_height,
                send_image_width=send_image_width,
            )
            for frame in history
        ]
        images[remote_name] = build_history_stack(chw_history, image_history_interval=image_history_interval)

    request = {
        "images": images,
        "state": state,
        "prompt": prompt,
        "timestep": int(timestep),
        "reset": bool(timestep == 0),
    }
    if inference_delay is not None:
        request["inference_delay"] = int(inference_delay)
    if prev_chunk_left_over is not None:
        prev_chunk = np.asarray(prev_chunk_left_over, dtype=np.float32)
        request["prev_chunk_left_over"] = np.ascontiguousarray(prev_chunk)
    if prev_chunk_left_over_processed is not None:
        prev_chunk_processed = np.asarray(prev_chunk_left_over_processed, dtype=np.float32)
        request["prev_chunk_left_over_processed"] = np.ascontiguousarray(prev_chunk_processed)
    return request
