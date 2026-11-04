"""Client for the machine's shared SAM3 service (services/sam3_server.py).

    sam = Sam3Client("http://127.0.0.1:18300")
    cups = sam.segment(rgb, text="cup")                          # every cup, best first
    under = sam.segment(rgb, points=[(320, 240)])                # candidates for the object there
    one = sam.segment(rgb, points=[(320, 240), (300, 200), (400, 260)], labels=[1, 1, 0])
    boxed = sam.segment(rgb, box=(250, 180, 390, 300))
    cups[0].mask, cups[0].score, cups[0].box                     # bool HxW, float, (x0, y0, x1, y1)

Pixels are (x, y) = (column, row) from the top-left corner; boxes are inclusive pixel bounds.
A picture is sent once: the client hashes it locally and only uploads ids the service lacks.
Safe to share between threads.
"""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import httpx
import numpy as np
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:18300"


class Sam3Error(RuntimeError):
    """The service refused or failed a request."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Sam3Unavailable(Sam3Error):
    """The service cannot be reached (not running, or not answering)."""


@dataclass(eq=False)
class Segment:
    mask: np.ndarray  # bool, H x W
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1, inclusive pixels
    area: int = 0  # pixels in the mask

    @property
    def center(self) -> tuple[float, float]:
        """Mean (x, y) of the mask's pixels."""
        ys, xs = np.nonzero(self.mask)
        return float(xs.mean()), float(ys.mean())


def image_id(rgb: np.ndarray) -> str:
    """The id the service gives this RGB uint8 picture."""
    h, w = rgb.shape[:2]
    digest = hashlib.sha256(b"%dx%d:" % (h, w))
    digest.update(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    return digest.hexdigest()


def _rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        got = f"{type(image).__name__} {getattr(image, 'shape', '')} {getattr(image, 'dtype', '')}"
        raise ValueError(f"image must be an H x W x 3 uint8 RGB array, got {got}")
    return np.ascontiguousarray(image)


def _xy(points: Any) -> list[list[float]]:
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        arr = np.zeros((0,))
    if arr.shape == (2,):  # a single (x, y)
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError(f"points must be a non-empty list of (x, y), got {points!r}")
    return arr.tolist()


class Sam3Client:
    """Talks to one SAM3 service.  Every call blocks until the masks are back."""

    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 30.0, *, remember: int = 512):
        self.url = url.rstrip("/")
        # trust_env=False: local traffic must not go through any HTTP(S)_PROXY
        self._http = httpx.Client(base_url=self.url, timeout=timeout_s, trust_env=False)
        self._timeout = timeout_s
        self._known: OrderedDict[str, None] = OrderedDict()  # ids the service has (as far as we know)
        self._remember = remember
        self._lock = threading.Lock()
        self.stats = {"segments": 0, "uploads": 0, "reuploads": 0, "retries": 0, "ms": 0.0}

    # public

    def segment(self, image: np.ndarray, *, text: str | None = None, points: Sequence | None = None,
                labels: Sequence[int] | None = None, box: Sequence[float] | None = None, max_masks: int = 8,
                min_score: float = 0.0, multimask: bool | None = None) -> list[Segment]:
        """Masks for a text prompt (every instance) or for points and/or a box (one object).

        points: [(x, y), ...]; labels: 1 = on the object, 0 = not on it (default all 1).
        box: (x0, y0, x1, y1).  Results are best first.  multimask (points/box only): None lets
        the service decide (three candidates for a single click, one mask otherwise).
        """
        t0 = time.perf_counter()
        rgb = _rgb(image)
        body: dict[str, Any] = {"max_masks": int(max_masks), "min_score": float(min_score)}
        if text is not None:
            if points is not None or labels is not None or box is not None:
                raise ValueError("give either text, or points and/or box - not both")
            body["text"] = str(text)
        else:
            if points is None and box is None:
                raise ValueError("give text=, points=, or box=")
            if points is not None:
                body["points"] = _xy(points)
                body["labels"] = [int(v) for v in labels] if labels is not None else [1] * len(body["points"])
            elif labels is not None:
                raise ValueError("labels need points")
            if box is not None:
                b = [float(v) for v in box]
                if len(b) != 4:
                    raise ValueError(f"box must be (x0, y0, x1, y1), got {box!r}")
                body["box"] = b
            if multimask is not None:
                body["multimask"] = bool(multimask)
        key = image_id(rgb)
        body["image_id"] = key
        if not self._has(key):
            self._upload(rgb, key)
        status, data = self._post("/segment", body)
        if status == 404 and data.get("error") == "unknown image":  # the service dropped it or restarted
            self._forget(key)
            self._count("reuploads")
            self._upload(rgb, key)
            status, data = self._post("/segment", body)
        if status != 200:
            raise Sam3Error(f"segment failed ({status}): {data.get('error', data)}", status)
        h, w = rgb.shape[:2]
        out = [self._segment(m, h, w) for m in data.get("masks", [])]
        self._count("segments", (time.perf_counter() - t0) * 1000)
        return out

    def upload(self, image: np.ndarray) -> str:
        """Send a picture ahead of time (optional); returns its id."""
        rgb = _rgb(image)
        key = image_id(rgb)
        if not self._has(key):
            self._upload(rgb, key)
        return key

    def health(self) -> dict:
        """The service's /health answer; raises Sam3Unavailable if it is down."""
        status, data = self._request("GET", "/health", None)
        if status != 200:
            raise Sam3Error(f"health check failed ({status}): {data.get('error', data)}", status)
        return data

    def available(self) -> bool:
        try:
            return bool(self.health().get("ok"))
        except Sam3Error:
            return False

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Sam3Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # internals

    def _count(self, key: str, ms: float = 0.0) -> None:
        with self._lock:
            self.stats[key] += 1
            self.stats["ms"] += ms

    def _has(self, key: str) -> bool:
        with self._lock:
            if key in self._known:
                self._known.move_to_end(key)
                return True
            return False

    def _forget(self, key: str) -> None:
        with self._lock:
            self._known.pop(key, None)

    def _upload(self, rgb: np.ndarray, key: str) -> None:
        # raw bytes: the service is local, and PNG would cost ~35 ms per picture
        body = {"rgb_b64": base64.b64encode(memoryview(rgb).cast("B")).decode("ascii"), "shape": list(rgb.shape[:2])}
        status, data = self._post("/image", body)
        if status != 200:
            raise Sam3Error(f"image upload failed ({status}): {data.get('error', data)}", status)
        if data.get("image_id") != key:
            raise Sam3Error(f"the service named the picture {data.get('image_id')!r}, expected {key!r}")
        self._count("uploads")
        with self._lock:
            self._known[key] = None
            while len(self._known) > self._remember:
                self._known.popitem(last=False)

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        status, data = self._request("POST", path, body)
        if status == 503 and "out of memory" in str(data.get("error", "")):
            time.sleep(1.0)  # the GPU is shared; memory may be back in a moment
            self._count("retries")
            status, data = self._request("POST", path, body)
        return status, data

    def _request(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        for attempt in (1, 2):
            try:
                resp = self._http.request(method, path, json=body)
                break
            except httpx.TimeoutException as exc:
                raise Sam3Unavailable(f"SAM3 service at {self.url} gave no answer within {self._timeout:g}s "
                                      f"({type(exc).__name__})") from exc
            except httpx.TransportError as exc:  # refused, reset, or a stale keep-alive connection
                if attempt == 2:
                    raise Sam3Unavailable(f"SAM3 service at {self.url} is not reachable ({type(exc).__name__}: "
                                          f"{exc}); start it with scripts/sam3_service.sh start GPU PORT") from exc
                self._count("retries")
                time.sleep(0.2)
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text[:300]}
        if not isinstance(data, dict):
            data = {"error": repr(data)[:300]}
        return resp.status_code, data

    @staticmethod
    def _segment(m: dict, h: int, w: int) -> Segment:
        with Image.open(io.BytesIO(base64.b64decode(m["png_b64"]))) as im:
            mask = np.asarray(im.convert("L")) > 0
        if mask.shape != (h, w):
            raise Sam3Error(f"mask is {mask.shape}, expected {(h, w)}")
        return Segment(mask=mask, score=float(m["score"]), box=tuple(int(v) for v in m["box"]),
                       area=int(m.get("area", mask.sum())))
