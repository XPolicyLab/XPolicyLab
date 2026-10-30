"""RGB geometry and per-environment visual/semantic episode memory."""

import re
from collections import deque

import cv2
import numpy as np

from ..config import ObservationConfig


def resize_head(image: np.ndarray, cfg: ObservationConfig) -> np.ndarray:
    """Symmetric aspect padding and bilinear resize; never change RGB order."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected decoded RGB uint8 [H,W,3]")
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("Empty camera image")
    from math import gcd

    divisor = gcd(cfg.width, cfg.height)
    aw, ah = cfg.width // divisor, cfg.height // divisor
    unit = max((width + aw - 1) // aw, (height + ah - 1) // ah)
    dw, dh = unit * aw - width, unit * ah - height
    padded = cv2.copyMakeBorder(
        image,
        dh // 2,
        dh - dh // 2,
        dw // 2,
        dw - dw // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return cv2.resize(np.ascontiguousarray(padded), (cfg.width, cfg.height))


def composite_image(
    vision: dict, cfg: ObservationConfig, cameras: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Current head plus two wrists; normalize before downsampling the wrists."""
    import torch
    from torch.nn import functional as F

    frames = [resize_head(vision[name]["color"], cfg) for name in cameras]
    values = 2.0 * (np.stack(frames).astype(np.float32) / 255.0) - 1.0
    values = torch.from_numpy(values).permute(0, 3, 1, 2).contiguous()
    wrists = [
        F.interpolate(
            values[i : i + 1],
            size=(cfg.height // 2, cfg.width // 2),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        for i in (1, 2)
    ]
    mosaic = torch.cat((values[0], torch.cat(wrists, dim=-1)), dim=-2)
    image = (
        ((mosaic.permute(1, 2, 0).numpy() + 1.0) * 127.5)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return image, frames[0]


def merge_memory(previous: str, delta: str) -> str:
    items, seen = [], set()
    for value in (previous, delta):
        normalized = re.sub(r"\s+", " ", value).strip(" \t.;")
        if normalized.lower() in ("", "none", "nothing", "n/a"):
            continue
        for part in re.split(r"(?:\r?\n|\s*[;|]\s*|(?<=[.!?])\s+)", value.strip()):
            item = re.sub(r"\s+", " ", part).strip(" \t.;")
            key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
            if key and key not in seen:
                items.append(item)
                seen.add(key)
    return ". ".join(items) + "." if items else "None."


class EpisodeMemory:
    """Sample the configured history window on the policy's control-step clock."""

    def __init__(self, config: ObservationConfig, *, capture_stride: int = 1):
        self.config = config
        if (
            type(capture_stride) is not int
            or capture_stride <= 0
            or config.interval % capture_stride
        ):
            raise ValueError(
                "History capture stride must be a positive divisor of the history interval"
            )
        self.capture_stride = capture_stride
        self.step = -capture_stride
        self.frames = deque(
            maxlen=config.max_frames * config.interval // capture_stride + 1
        )
        self.semantic = "None."
        self.subtask = "None."
        self.cache_valid = False
        self.instruction = None
        self.current = None
        self.state = None

    def configure_capture(self, replan_steps: int):
        """Negotiate sparse capture after the first server reply, at step zero."""
        from math import gcd

        if (
            type(replan_steps) is not int
            or not 1 <= replan_steps <= self.config.horizon
        ):
            raise ValueError(
                "replan_steps must be an integer within the model action horizon"
            )
        if self.step != 0 or len(self.frames) != 1:
            raise RuntimeError("Capture cadence must be configured at episode start")
        self.capture_stride = gcd(replan_steps, self.config.interval)
        self.frames = deque(
            self.frames,
            maxlen=self.config.max_frames * self.config.interval // self.capture_stride
            + 1,
        )

    def observe(self, instruction, current, head, state, *, step: int | None = None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation requires a nonempty instruction")
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError("Instruction changed without an episode reset")
        expected = self.step + self.capture_stride
        if step is not None and step != expected:
            raise ValueError(f"Expected history capture at step {expected}, got {step}")
        self.step = expected
        self.instruction, self.current, self.state = instruction, current, state
        self.frames.append(head.copy())

    def history(self):
        offsets = self.config.offsets(self.step)
        frames = list(self.frames)
        return [
            frames[offset // self.capture_stride - 1] for offset in offsets
        ], offsets

    def commit(self, decision, delta, subtask):
        if decision == "UPDATE":
            self.semantic = merge_memory(self.semantic, delta)
            self.subtask = subtask
            self.cache_valid = True
        elif decision != "KEEP" or not self.cache_valid:
            raise ValueError(f"Invalid semantic transition: {decision}")


def encode_histories(network, currents, histories):
    import torch

    with torch.no_grad():
        return _encode_histories(network, currents, histories)


def _encode_histories(network, currents, histories):
    import torch

    cfg = network.config
    stride = cfg.temporal_stride
    prepared = []
    groups = {}
    for i, (current, (history, offsets)) in enumerate(
        zip(currents, histories, strict=True)
    ):
        if not history:
            prepared.append(None)
            continue
        count = stride * (len(history) // stride)
        frames = [*history, current[: cfg.observation.height]]
        positions = [*offsets, 0]
        frames = [frames[0], *frames[-count:]] if count else [frames[0]]
        positions = [positions[0], *positions[-count:]] if count else [positions[0]]
        prepared.append((np.stack(frames), [positions[0], *positions[stride::stride]]))
        groups.setdefault(len(frames), []).append(i)
    result = [None] * len(currents)
    device = network.world_queries.device
    for indices in groups.values():
        pixels = torch.from_numpy(np.stack([prepared[i][0] for i in indices])).to(
            device=device, dtype=torch.bfloat16
        )
        pixels = pixels.permute(0, 4, 1, 2, 3).div(127.5).sub(1)
        latents = network.encode_video(pixels)
        for row, i in enumerate(indices):
            times = (
                torch.tensor(prepared[i][1], device=device, dtype=torch.float32)
                / cfg.latent_time_steps
            )
            result[i] = (latents[row], times)
    return result


def encode_history(network, current, history, offsets):
    """Single-environment convenience wrapper for grouped history encoding."""
    return encode_histories(network, [current], [(history, offsets)])[0]
