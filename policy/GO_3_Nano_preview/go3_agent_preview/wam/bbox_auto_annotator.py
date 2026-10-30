from __future__ import annotations

import atexit
import functools
import http.server
import json
import os
import re
import socket
import tempfile
import threading
import time
import uuid
import base64
import io
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image


BBOX_GROUNDING_PROMPT = """You are a precise visual grounding model for a household manipulation robot.

TASK: Given a head-camera image and a natural-language instruction, return a bounding box around the SINGLE visible bbox anchor for the current action step.

The bbox anchor is usually a physical object or container. For relative PLACE instructions such as "place A to the right/left of B", the bbox anchor is the local visible table/desk landing region on the instructed side of B. Do NOT box airspace, robot arms, grippers, human body parts, or the whole table.

INSTRUCTION:
"{instruction}"

OUTPUT (single-line JSON, NO markdown, NO commentary):
{{"target": "<concise English noun phrase>", "bbox": [x1, y1, x2, y2], "found": true}}

If the requested action step is already completed and the robot is only resetting/retracting, or if no valid bbox anchor can be identified, return:
{{"target": "<concise English noun phrase>", "bbox": null, "found": false}}

Rules:
- Coordinates are integers in [0, 1000], normalized pixel position. (0, 0) = top-left, (1000, 1000) = bottom-right.
- Always use x1 < x2 and y1 < y2.
- PICK / GRASP / FIND / SEARCH: box the named object.
- PLACE to the left/right/front/back/near side of a named reference object: box the compact local visible landing region on the instructed side of the reference object, not the reference object itself.
- PLACE into/on a container or named object: box that destination container/object.
- HANDOVER / TRANSFER / PASS: box the object being transferred.
- Do not box robot arms, grippers, human hands, body parts, airspace, the whole table, or arbitrary empty space.
"""


DEFAULT_BBOX_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "base_bbox.md"
)
_PROMPT_CACHE: dict[str, str] = {}


def _load_bbox_grounding_prompt() -> str:
    prompt_path = os.environ.get("BBOX_ANNOTATOR_PROMPT_PATH") or str(
        DEFAULT_BBOX_PROMPT_PATH
    )
    if not prompt_path:
        return BBOX_GROUNDING_PROMPT
    if prompt_path not in _PROMPT_CACHE:
        path = Path(prompt_path)
        if path.exists():
            _PROMPT_CACHE[prompt_path] = path.read_text(encoding="utf-8").strip()
        else:
            _PROMPT_CACHE[prompt_path] = BBOX_GROUNDING_PROMPT
    return _PROMPT_CACHE[prompt_path]


def get_bbox_grounding_prompt_path() -> str:
    return os.environ.get("BBOX_ANNOTATOR_PROMPT_PATH") or str(
        DEFAULT_BBOX_PROMPT_PATH
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start >= 0:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
        raise ValueError(f"VLM response must be a JSON object, got {type(parsed)}")
    return parsed


def _norm_bbox_1000(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    max_abs = max(abs(value) for value in values)
    if max_abs <= 1.5:
        norm = values
    elif max_abs <= 1000.5:
        norm = [value / 1000.0 for value in values]
    else:
        norm = [
            values[0] / max(1, width),
            values[1] / max(1, height),
            values[2] / max(1, width),
            values[3] / max(1, height),
        ]
    norm = [max(0.0, min(1.0, value)) for value in norm]
    if norm[2] - norm[0] <= 1e-3 or norm[3] - norm[1] <= 1e-3:
        return None
    return [int(round(value * 1000.0)) for value in norm]


def _local_ip_for(host: str, port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((host, port))
        return str(sock.getsockname()[0])


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


class ImageUrlPublisher:
    def __init__(
        self,
        *,
        upstream_url: str,
        host: str | None = None,
        port: int = 0,
    ) -> None:
        raise NotImplementedError('External bbox API is not implemented in this release')

    def publish(self, image: Image.Image) -> str:
        name = f"{uuid.uuid4().hex}.jpg"
        image.save(Path(self._tmp.name) / name, format="JPEG", quality=85)
        return f"{self.base_url}/{name}"

    def close(self) -> None:
        if getattr(self, "_server", None) is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        self._tmp.cleanup()
        self._server = None


class BBoxAutoAnnotator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
        image_host: str | None = None,
        image_port: int = 0,
        cache_path: str | None = None,
        image_mode: str = "url",
        ca_bundle: str | None = None,
    ) -> None:
        raise NotImplementedError('External bbox API is not implemented in this release')

    @classmethod
    def from_env(cls) -> "BBoxAutoAnnotator":
        raise NotImplementedError('External bbox API is not implemented in this release')

    def annotate(
        self,
        image: Image.Image,
        instruction: str,
        *,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        t0 = time.perf_counter()
        parsed, raw = self._call_json(image, instruction)
        bbox = _norm_bbox_1000(parsed.get("bbox"), *image.size)
        found = bool(parsed.get("found", bbox is not None)) and bbox is not None
        result = {
            "target": str(parsed.get("target") or ""),
            "bbox": bbox,
            "found": found,
            "raw": raw,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        if cache_key:
            self._store_cached(cache_key, result)
        return result

    def _call_json(self, image: Image.Image, instruction: str) -> tuple[dict[str, Any], str]:
        import urllib.error
        import urllib.request

        if self.image_mode == "data_uri":
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            image_url = f"data:image/jpeg;base64,{b64}"
        else:
            image_url = self._publisher.publish(image)
        prompt = _load_bbox_grounding_prompt().replace("{instruction}", instruction)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if _as_bool(os.environ.get("BBOX_ANNOTATOR_RESPONSE_FORMAT"), True):
            body["response_format"] = {"type": "json_object"}

        payload = self._post_chat_completion_with_fallbacks(body)
        raw = str(payload["choices"][0]["message"]["content"] or "").strip()
        return _parse_json_object(raw), raw

    def _post_chat_completion_with_fallbacks(self, body: dict[str, Any]) -> dict[str, Any]:
        variants = [body]
        if "response_format" in body:
            v = dict(body)
            v.pop("response_format", None)
            variants.append(v)
        if "chat_template_kwargs" in body:
            v = dict(body)
            v.pop("chat_template_kwargs", None)
            variants.append(v)
        if "response_format" in body or "chat_template_kwargs" in body:
            v = dict(body)
            v.pop("response_format", None)
            v.pop("chat_template_kwargs", None)
            variants.append(v)

        last_error: RuntimeError | None = None
        seen: set[str] = set()
        for variant in variants:
            key = json.dumps(sorted(variant.keys()))
            if key in seen:
                continue
            seen.add(key)
            try:
                return self._post_chat_completion(variant)
            except RuntimeError as exc:
                last_error = exc
                text = str(exc)
                if (
                    "Unknown parameter" not in text
                    and "InvalidParameter" not in text
                    and "invalid_request_error" not in text
                ):
                    raise
        assert last_error is not None
        raise last_error

    def _post_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError('External bbox API is not implemented in this release')

    def _load_cache(self) -> None:
        if self._cache_loaded or self.cache_path is None or not self.cache_path.exists():
            self._cache_loaded = True
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("key") or "")
                result = record.get("result")
                if key and isinstance(result, dict):
                    self._cache[key] = result
        self._cache_loaded = True

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            self._load_cache()
            return self._cache.get(key)

    def _store_cached(self, key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._load_cache()
            self._cache[key] = result
            if self.cache_path is None:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"key": key, "result": result},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


_ANNOTATOR: BBoxAutoAnnotator | None = None
_ANNOTATOR_LOCK = threading.Lock()


def get_bbox_auto_annotator() -> BBoxAutoAnnotator:
    global _ANNOTATOR
    with _ANNOTATOR_LOCK:
        if _ANNOTATOR is None:
            _ANNOTATOR = BBoxAutoAnnotator.from_env()
        return _ANNOTATOR


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"expected frame [C,H,W], got {tuple(frame.shape)}")
    import torch
    image = ((frame.detach().cpu().float() + 1.0) / 2.0 * 255.0)
    image = image.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


def annotate_bbox_from_video_tensor(
    video: torch.Tensor,
    instruction: str,
    *,
    cache_key: str | None = None,
    head_view_index: int = 0,
    frame_index: int = 0,
) -> dict[str, Any]:
    if video.ndim != 5:
        raise ValueError(f"expected video [C,V,T,H,W], got {tuple(video.shape)}")
    if not 0 <= int(head_view_index) < video.shape[1]:
        raise ValueError(f"head_view_index {head_view_index} out of range")
    if not 0 <= int(frame_index) < video.shape[2]:
        raise ValueError(f"frame_index {frame_index} out of range")
    image = tensor_frame_to_pil(video[:, int(head_view_index), int(frame_index)])
    return get_bbox_auto_annotator().annotate(
        image,
        instruction,
        cache_key=cache_key,
    )
