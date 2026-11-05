"""Resident SAM3 HTTP service for RoboDojo CAP perception.

One request carries one RGB image and all text prompts.  SAM3 therefore encodes
the image once, then evaluates each prompt against the resident feature map.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from typing import Any

import numpy as np
from PIL import Image

from .sam3_segment_text import (
    Sam3SegmentationResult,
    get_sam3_text_segmenter,
)


def _decode_request(payload: dict[str, Any]) -> tuple[np.ndarray, list[tuple[str, float]]]:
    encoded = payload.get("image_base64")
    raw_prompts = payload.get("prompts")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 must be a non-empty string")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("prompts must be a non-empty list")
    image = np.asarray(
        Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"),
        dtype=np.uint8,
    )
    prompt_specs: list[tuple[str, float]] = []
    for item in raw_prompts:
        if not isinstance(item, dict):
            raise ValueError("each prompt must be an object")
        text = item.get("text")
        threshold = item.get("confidence_threshold")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prompt text must be non-empty")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("confidence_threshold must be numeric")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        prompt_specs.append((text, threshold))
    return image, prompt_specs


def _encode_result(result: Sam3SegmentationResult) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    for mask, box, score in zip(
        result.masks,
        result.boxes_xyxy_px,
        result.scores,
    ):
        packed = np.asarray(mask, dtype=np.uint8)
        instances.append(
            {
                "score": float(score),
                "box": [float(value) for value in box],
                "shape": [int(packed.shape[0]), int(packed.shape[1])],
                "mask_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
            }
        )
    return {"prompt": result.prompt, "instances": instances}


class _Handler(BaseHTTPRequestHandler):
    server_version = "RoboDojoSAM3/1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        segmenter = get_sam3_text_segmenter()
        self._json(
            200,
            {
                "status": "ready",
                "device": segmenter.device,
                "checkpoint": str(segmenter.checkpoint_path),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/segment_many":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image, prompt_specs = _decode_request(payload)
            results = get_sam3_text_segmenter().segment_many(image, prompt_specs, box_cxcywh_normalized=payload.get("box_cxcywh_normalized"))
            self._json(200, {"results": [_encode_result(item) for item in results]})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[sam3] {self.address_string()} {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    args = parser.parse_args()
    segmenter = get_sam3_text_segmenter()
    print(
        f"[sam3] ready http://{args.host}:{args.port} "
        f"device={segmenter.device} checkpoint={segmenter.checkpoint_path}",
        flush=True,
    )
    HTTPServer((args.host, args.port), _Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
