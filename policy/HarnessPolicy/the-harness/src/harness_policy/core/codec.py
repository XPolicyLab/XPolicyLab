"""Codecs shared by benchmark workers: episode cursors, action bounds, image data URLs.

Importable without a simulator, torch or a GPU.
"""

import base64
import math
import struct
import zlib


class EpisodeCursor:
    """Reject a repeated motion without guessing whether the simulator applied it."""

    def __init__(self):
        self.episode_id = None
        self.step = None

    def start(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.step = 0

    def check(self, episode_id: str, step: int) -> None:
        if self.episode_id is None or (episode_id, step) != (self.episode_id, self.step):
            raise ValueError("Stale or missing observation; action was not executed")

    def advance(self) -> int:
        self.step += 1
        return self.step

    def invalidate(self) -> None:
        self.episode_id = None
        self.step = None



def finite_bounds(low, high):
    bounds = []
    for start, stop in zip(low, high, strict=True):
        lower, upper = float(start), float(stop)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("Action bounds must be finite and ordered")
        bounds.append((lower, upper))
    return bounds



def validate_position(arguments, bounds):
    if set(arguments) != {"position"}:
        raise ValueError("Expected joint_position(position=[...])")
    values = arguments["position"]
    if not isinstance(values, list) or len(values) != len(bounds):
        raise ValueError(f"position must contain {len(bounds)} finite numbers")
    command = []
    for value, (lower, upper) in zip(values, bounds, strict=True):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("position values must be finite numbers")
        number = float(value)
        if number < lower or number > upper:
            raise ValueError(f"position value {number} is outside [{lower}, {upper}]")
        command.append(number)
    return command



def data_url(payload, mime="image/png"):
    if not payload:
        raise ValueError("Image payload is empty")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def encode_png(width, height, rgb):
    """Encode tightly packed RGB bytes as a PNG."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("RGB byte count does not match the image size")
    raw = b"".join(
        b"\x00" + rgb[row * width * 3:(row + 1) * width * 3]
        for row in range(height)
    )

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

