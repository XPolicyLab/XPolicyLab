"""SAM3 HTTP client using the endpoint selected by the runtime manifest."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .sam3_segment_text import Sam3SegmentationResult

DEFAULT_TIMEOUT_S = 120.0


def _endpoint() -> str:
    return os.environ["ROBODOJO_SAM3_URL"].rstrip("/")


def _image_png_b64(image_rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask(item: dict) -> np.ndarray:
    shape = item.get("shape")
    payload = item.get("mask_base64")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise RuntimeError("SAM3 HTTP mask shape must be [H, W]")
    if not isinstance(payload, str) or not payload:
        raise RuntimeError("SAM3 HTTP mask_base64 missing")
    height, width = int(shape[0]), int(shape[1])
    raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    if raw.size != height * width:
        raise RuntimeError("SAM3 HTTP mask bytes do not match shape")
    return raw.reshape(height, width) > 0


def _post_segment_many(
    image_b64: str,
    prompt_specs: Sequence[tuple[str, float]],
    *,
    timeout_s: float,
    box_cxcywh_normalized=None,
) -> list[dict]:
    body = json.dumps(
        {
            "image_base64": image_b64,
            "box_cxcywh_normalized": box_cxcywh_normalized,
            "prompts": [
                {
                    "text": str(prompt),
                    "confidence_threshold": float(confidence_threshold),
                }
                for prompt, confidence_threshold in prompt_specs
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        f"{_endpoint()}/segment_many",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"SAM3 HTTP {exc.code} for segment_many") from exc
    except URLError as exc:
        raise RuntimeError(f"SAM3 HTTP unreachable {_endpoint()}: {exc}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SAM3 HTTP segment_many failed: {exc}") from exc
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("SAM3 HTTP response missing results")
    return results


def segment_many_http(
    image_rgb: np.ndarray,
    prompt_specs: Sequence[tuple[str, float]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    box_cxcywh_normalized=None,
) -> list[Sam3SegmentationResult]:
    """Segment several prompts through the resident CapTwin SAM3 server."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RuntimeError("SAM3 HTTP image must be uint8 HxWx3")
    height, width = image.shape[:2]
    image_b64 = _image_png_b64(image)
    prompt_specs = tuple(prompt_specs)
    if not prompt_specs:
        raise RuntimeError("SAM3 HTTP prompt_specs must not be empty")
    raw_groups = _post_segment_many(
        image_b64,
        prompt_specs,
        timeout_s=timeout_s,
        box_cxcywh_normalized=box_cxcywh_normalized,
    )
    if len(raw_groups) != len(prompt_specs):
        raise RuntimeError(
            "SAM3 HTTP result group count does not match prompt count"
        )

    results: list[Sam3SegmentationResult] = []
    for (prompt, _threshold), raw_group in zip(prompt_specs, raw_groups):
        if not isinstance(raw_group, dict) or raw_group.get("prompt") != str(prompt):
            raise RuntimeError("SAM3 HTTP result prompt order mismatch")
        raw_items = raw_group.get("instances")
        if not isinstance(raw_items, list):
            raise RuntimeError("SAM3 HTTP result group missing instances")
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        scores: list[float] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            masks.append(_decode_mask(item))
            box = item.get("box") or [0.0, 0.0, 0.0, 0.0]
            boxes.append([float(v) for v in box])
            scores.append(float(item.get("score") or 0.0))
        if masks:
            mask_arr = np.stack(masks, axis=0)
            box_arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
            score_arr = np.asarray(scores, dtype=float).reshape(-1)
        else:
            mask_arr = np.zeros((0, height, width), dtype=bool)
            box_arr = np.zeros((0, 4), dtype=float)
            score_arr = np.zeros((0,), dtype=float)
        results.append(
            Sam3SegmentationResult(
                prompt=str(prompt),
                masks=mask_arr,
                boxes_xyxy_px=box_arr,
                scores=score_arr,
                image_size_wh=(width, height),
                model_id=f"sam3_http:{_endpoint()}",
            )
        )
    return results
