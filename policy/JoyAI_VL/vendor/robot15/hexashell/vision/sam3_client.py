"""Client for the existing SAM3 service's health, image and segment endpoints."""

import base64
import hashlib
import json
from threading import Lock
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import io
from PIL import Image
import numpy as np

DEFAULT_URL = "http://127.0.0.1:18300"


class Sam3Error(RuntimeError):
    pass


class Segment(NamedTuple):
    mask: np.ndarray
    score: float
    box: tuple[int, int, int, int]
    area: int


class Sam3Client:
    def __init__(self, url=DEFAULT_URL):
        self.url = url.rstrip("/")
        self.http = build_opener(ProxyHandler({}))
        self.images = {}
        self.lock = Lock()

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.url + path, data, {"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=30) as response:
                return json.load(response)
        except HTTPError:
            raise
        except (URLError, OSError, ValueError) as exc:
            raise Sam3Error(f"{path}: {exc}") from exc

    def health(self):
        try:
            return self._request("/health")
        except HTTPError as exc:
            raise Sam3Error(f"/health: HTTP {exc.code}") from exc

    def _upload(self, rgb):
        result = self._request("/image", {"rgb_b64": base64.b64encode(rgb).decode(), "shape": list(rgb.shape[:2])})
        return result["image_id"]

    def _image(self, key, rgb, expired=None):
        with self.lock:
            current = self.images.get(key)
            if current is None or current == expired:
                current = self._upload(rgb)
                if key not in self.images and len(self.images) >= 64:
                    self.images.pop(next(iter(self.images)))
                self.images[key] = current
            return current

    def segment(self, image, **prompt):
        rgb = np.ascontiguousarray(image)
        key = (rgb.shape, hashlib.sha256(rgb).digest())
        try:
            image_id = self._image(key, rgb)
            try:
                result = self._request("/segment", {**prompt, "image_id": image_id})
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                image_id = self._image(key, rgb, expired=image_id)
                result = self._request("/segment", {**prompt, "image_id": image_id})
            segments = []
            for item in result["masks"]:
                with Image.open(io.BytesIO(base64.b64decode(item["png_b64"]))) as picture:
                    if picture.format != "PNG":
                        raise ValueError("SAM3 mask must be PNG")
                    mask = np.array(picture.convert("L"), copy=True)
                if mask is None or mask.shape != rgb.shape[:2]:
                    raise Sam3Error(f"invalid mask for an image of shape {rgb.shape[:2]}")
                pixels = mask > 0
                segments.append(Segment(pixels, float(item["score"]), tuple(item["box"]),
                                        int(np.count_nonzero(pixels))))
            return segments
        except HTTPError as exc:
            raise Sam3Error(f"SAM3 returned HTTP {exc.code}: {exc.read(300).decode(errors='replace')}") from exc
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise Sam3Error(f"invalid SAM3 response: {exc}") from exc

    def close(self):
        self.http.close()
        self.images.clear()
