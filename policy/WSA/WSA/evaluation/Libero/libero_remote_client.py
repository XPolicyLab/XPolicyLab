from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in [THIS_DIR, SRC_ROOT, REPO_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from evaluation.Libero.websocket_client import WebsocketClientPolicy
except ImportError:
    from .websocket_client import WebsocketClientPolicy


def to_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a single RGB frame, got shape={array.shape}")

    if array.shape[-1] == 3:
        hwc = array
    elif array.shape[0] == 3:
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}. Expected HWC or CHW with 3 channels.")

    if np.issubdtype(hwc.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        hwc = np.clip(hwc * scale, 0.0, 255.0)
    else:
        hwc = np.clip(hwc, 0, 255)

    return np.ascontiguousarray(hwc.astype(np.uint8))


def build_history_stack(history_frames: list[np.ndarray], image_history_interval: int) -> np.ndarray:
    if not history_frames:
        raise ValueError("history_frames must contain at least one frame.")

    past_idx = max(len(history_frames) - image_history_interval - 1, 0)
    current = to_hwc_uint8(history_frames[-1])
    past = to_hwc_uint8(history_frames[past_idx])
    return np.stack([past, current], axis=0)


class LiberoRemoteClient:
    """Small helper for LIBERO evaluation loops that query an out-of-process WSABase policy server."""

    def __init__(self, ws_url: str, image_history_interval: int = 15) -> None:
        self._policy = WebsocketClientPolicy(host=ws_url)
        self._metadata = self._policy.get_server_metadata()
        self._image_history_interval = image_history_interval

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer_step(
        self,
        *,
        head_history: list[np.ndarray],
        wrist_history: list[np.ndarray],
        state: np.ndarray,
        prompt: str,
        timestep: int,
    ) -> dict:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        request = {
            "images": {
                "head": build_history_stack(head_history, self._image_history_interval),
                "left_wrist": build_history_stack(wrist_history, self._image_history_interval),
            },
            "state": np.ascontiguousarray(state_np),
            "prompt": prompt,
            "timestep": int(timestep),
            "reset": bool(timestep == 0),
        }
        return self._policy.infer(request)

    def close(self) -> None:
        self._policy.close()
