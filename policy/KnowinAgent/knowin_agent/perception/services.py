"""Segmentation and depth services: HTTP clients (stdlib only), native depth and LRU caches.

HttpSegmenter talks to a SAM3-style text segmenter, HttpPointDepth to a learned point-cloud
model (for example MoGe-2). Both send the RGB frame as a PNG in a multipart form. Base URLs
come from KNOWIN_AGENT_SAM3_URL and KNOWIN_AGENT_DEPTH_URL.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
from PIL import Image

from knowin_agent.perception.types import DepthSource, Mask, PerceptionError, Segmenter
from knowin_agent.runtime.backend import CameraFrame

DEFAULT_SAM3_URL = "http://127.0.0.1:8035"
DEFAULT_DEPTH_URL = "http://127.0.0.1:8034"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def encode_png(rgb: np.ndarray) -> bytes:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise PerceptionError(f"expected an HxWx3 RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array[..., :3])).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """multipart/form-data body and content type; files maps name -> (filename, bytes, content type)."""
    boundary = "knowin_agent" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, content_type) in files.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                      f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _opener(url: str) -> urllib.request.OpenerDirector:
    host = urllib.parse.urlsplit(url).hostname or ""
    # Environment proxies must not intercept a loopback model server.
    if host in LOOPBACK_HOSTS:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def post_form(url: str, fields: dict, files: dict, *, timeout: float, service: str) -> tuple[bytes, dict]:
    body, content_type = multipart(fields, files)
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with _opener(url).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            headers = {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace").strip()
        raise PerceptionError(f"{service} at {url} returned HTTP {exc.code}: {detail or exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise PerceptionError(f"{service} at {url} timed out after {timeout:g} s") from exc
    except urllib.error.URLError as exc:
        raise PerceptionError(f"{service} at {url} unreachable: {exc.reason}") from exc
    except OSError as exc:
        raise PerceptionError(f"{service} at {url} failed: {exc}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise PerceptionError(f"{service} response exceeds {MAX_RESPONSE_BYTES} bytes")
    return raw, headers


def decode_mask_png(data: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(data, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (ValueError, OSError) as exc:
        raise PerceptionError(f"segmentation mask is not a valid base64 PNG: {exc}") from exc
    mask = np.asarray(image.convert("L")) > 127
    if mask.shape != shape:
        raise PerceptionError(f"segmentation mask is {mask.shape[::-1]}, image is {shape[::-1]} (width, height)")
    return mask


class HttpSegmenter:
    """POST {base}/v1/segment: image (PNG), text, min_score, max_masks -> masks sorted by score."""

    def __init__(self, base_url: Optional[str] = None, *, timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("KNOWIN_AGENT_SAM3_URL") or DEFAULT_SAM3_URL).rstrip("/")
        self.timeout = float(timeout)

    def segment(self, rgb: np.ndarray, text: str, *, max_masks: int = 8, min_score: float = 0.5) -> list[Mask]:
        if not isinstance(text, str) or not text.strip():
            raise PerceptionError("segmentation text must be a non-empty string")
        h, w = np.asarray(rgb).shape[:2]
        raw, _ = post_form(f"{self.base_url}/v1/segment",
                           {"text": text, "min_score": float(min_score), "max_masks": int(max_masks)},
                           {"image": ("observation.png", encode_png(rgb), "image/png")},
                           timeout=self.timeout, service="segmentation service")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PerceptionError(f"segmentation service returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("masks"), list):
            raise PerceptionError("segmentation response has no 'masks' list")
        if (payload.get("width"), payload.get("height")) != (w, h):
            raise PerceptionError(f"segmentation response is {payload.get('width')}x{payload.get('height')}, "
                                  f"image is {w}x{h}")
        masks = []
        for item in payload["masks"]:
            try:
                score = float(item["score"])
                encoded = item["mask_png_base64"]
            except (KeyError, TypeError, ValueError) as exc:
                raise PerceptionError(f"segmentation mask entry is malformed: {exc}") from exc
            if score < min_score:
                continue
            box = tuple(float(v) for v in item.get("box_xyxy") or ())
            masks.append(Mask(score=score, mask=decode_mask_png(encoded, (h, w)), box_xyxy=box))
        masks.sort(key=lambda m: m.score, reverse=True)
        return masks[:max_masks]


class HttpPointDepth:
    """POST {base}/v1/pointcloud -> little-endian .npy HxWx3 camera-optical points (m); returns the Z channel."""

    def __init__(self, base_url: Optional[str] = None, *, timeout: float = 90.0):
        self.base_url = (base_url or os.environ.get("KNOWIN_AGENT_DEPTH_URL") or DEFAULT_DEPTH_URL).rstrip("/")
        self.timeout = float(timeout)

    def depth(self, frame: CameraFrame) -> Optional[np.ndarray]:
        h, w = frame.rgb.shape[:2]
        raw, headers = post_form(f"{self.base_url}/v1/pointcloud", {"output_dtype": "float32", "compression": "none"},
                                 {"image": ("observation.png", encode_png(frame.rgb), "image/png")},
                                 timeout=self.timeout, service="depth service")
        for key, expected in (("x-pointcloud-unit", "metre"), ("x-pointcloud-frame", "camera_optical"),
                              ("x-pointcloud-compression", "none")):
            if key in headers and headers[key] != expected:
                raise PerceptionError(f"depth service contract mismatch: {key}={headers[key]!r}, expected {expected!r}")
        try:
            points = np.load(io.BytesIO(raw), allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            raise PerceptionError(f"depth service returned an invalid .npy body: {exc}") from exc
        if not isinstance(points, np.ndarray) or points.ndim != 3 or points.shape[2] != 3:
            raise PerceptionError(f"depth service returned shape {getattr(points, 'shape', None)}, expected HxWx3")
        if points.shape[:2] != (h, w):
            raise PerceptionError(f"depth service returned {points.shape[1]}x{points.shape[0]} points, image is {w}x{h}")
        z = points[..., 2].astype(np.float64)
        z[~np.isfinite(z) | (z <= 0)] = np.nan
        return z


class NativeDepth:
    """The backend's own metric depth (simulator or RGB-D camera)."""

    def depth(self, frame: CameraFrame) -> Optional[np.ndarray]:
        if frame.depth is None:
            return None
        z = np.array(frame.depth, dtype=np.float64)
        if z.shape != frame.rgb.shape[:2]:
            raise PerceptionError(f"native depth is {z.shape}, image is {frame.rgb.shape[:2]}")
        z[~np.isfinite(z) | (z <= 0)] = np.nan
        return z


def image_digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{a.shape}|{a.dtype}|".encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


class _Lru:
    def __init__(self, maxsize: int):
        self.maxsize = max(1, int(maxsize))
        self.store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, compute):
        if key in self.store:
            self.hits += 1
            self.store.move_to_end(key)
            return self.store[key]
        self.misses += 1
        value = compute()
        self.store[key] = value
        while len(self.store) > self.maxsize:
            self.store.popitem(last=False)
        return value

    def clear(self) -> None:
        self.store.clear()


def _frozen(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if array is not None:
        array = np.array(array)
        array.setflags(write=False)
    return array


class CachedSegmenter:
    """LRU cache over a Segmenter keyed by sha256 of the image, text and request limits. Errors are not cached."""

    def __init__(self, inner: Segmenter, maxsize: int = 64):
        self.inner = inner
        self._lru = _Lru(maxsize)

    @property
    def calls(self) -> int:
        return self._lru.misses

    @property
    def hits(self) -> int:
        return self._lru.hits

    def segment(self, rgb: np.ndarray, text: str, *, max_masks: int = 8, min_score: float = 0.5) -> list[Mask]:
        key = (image_digest(rgb), text, int(max_masks), float(min_score))

        def compute():
            masks = self.inner.segment(rgb, text, max_masks=max_masks, min_score=min_score)
            return tuple(Mask(m.score, _frozen(np.asarray(m.mask, dtype=bool)), tuple(m.box_xyxy)) for m in masks)

        return list(self._lru.get(key, compute))

    def clear(self) -> None:
        self._lru.clear()


class CachedDepth:
    """LRU cache over a DepthSource keyed by sha256 of the image (and of native depth, when present)."""

    def __init__(self, inner: DepthSource, maxsize: int = 16):
        self.inner = inner
        self._lru = _Lru(maxsize)

    @property
    def calls(self) -> int:
        return self._lru.misses

    @property
    def hits(self) -> int:
        return self._lru.hits

    def depth(self, frame: CameraFrame) -> Optional[np.ndarray]:
        key = (image_digest(frame.rgb), None if frame.depth is None else image_digest(frame.depth))
        return self._lru.get(key, lambda: _frozen(self.inner.depth(frame)))

    def clear(self) -> None:
        self._lru.clear()


def default_services() -> tuple[CachedSegmenter, CachedDepth]:
    """Cached HTTP segmenter and learned depth at KNOWIN_AGENT_SAM3_URL / KNOWIN_AGENT_DEPTH_URL."""
    return CachedSegmenter(HttpSegmenter()), CachedDepth(HttpPointDepth())
