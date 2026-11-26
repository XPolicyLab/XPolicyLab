"""SAM3's private HTTP PNG protocol, separate from XPolicyLab camera transport.

Implements the exact Segmented/segment interface expected by RoboDojo perception tools
without loading an in-process 6 GB SAM3 checkpoint per policy server.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Segmented:
    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    reason: str | None = None


def _encode_rgb_png(pixels: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask_png(data_url: str) -> np.ndarray:
    """Read only the SAM3 response mask, never a benchmark observation buffer."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw), formats=("PNG",)) as img:
        return np.asarray(img.convert("L")) > 0


class HttpSam3:
    """Thin client for our persistent SAM3 HTTP service (default http://127.0.0.1:8810)."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 60.0) -> None:
        url = base_url or os.environ.get("HARNESS_SAM3_URL") or "http://127.0.0.1:8810"
        self.base_url = url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Segmented:
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        if has_text == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        min_score = float(min_score)
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        img_url = _encode_rgb_png(pixels)

        if has_text:
            assert text_prompt is not None
            data = self._post(
                "/segment",
                {"image": img_url, "prompt": text_prompt.strip(), "threshold": 0.0},
            )
        else:
            assert point is not None
            row, col = int(point[0]), int(point[1])
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {list(point)} is outside image shape [{height}, {width}]"
                )
            data = self._post(
                "/segment_point",
                {"image": img_url, "points": [[col, row]], "labels": [1]},
            )

        instances = data.get("instances") or []
        if not instances:
            return Segmented(False, reason="SAM3 service returned no candidate")
        top = max(instances, key=lambda item: float(item.get("score", -1.0)))
        score = float(top["score"])
        box = [float(v) for v in top["box"]] if top.get("box") is not None else None
        if score < min_score:
            return Segmented(
                False,
                score=score,
                box=box,
                reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
            )
        mask = _decode_mask_png(top["mask"])
        if mask.ndim != 2 or not mask.any():
            return Segmented(False, score=score, box=box, reason="SAM3 service returned an empty mask")
        return Segmented(True, score=score, box=box, mask=mask)
