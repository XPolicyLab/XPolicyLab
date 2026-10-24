"""Clients of the model services that skills call: the segmentation server and
vision-language models behind an OpenAI-compatible endpoint (pointing, questions,
the success verifier)."""

import base64
import json
import re

import numpy as np

from harness_policy.geometry import Mask
from harness_policy.runtime.chat import ChatModel


_POINT_SYSTEM = (
    "You locate things in images. Reply only with JSON: a list of objects "
    '{"point_2d": [x, y], "label": "..."} where x and y are integers from 0 to 1000 '
    "relative to the image width and height. Return an empty list if the thing is not visible."
)


class Segmenter:
    """Client of services/servers/serve_sam3.py."""

    def __init__(self, url, timeout=120.0, on_demand=None):
        self.url = url.rstrip("/") + "/segment"
        self.timeout = timeout
        # A segmenter placed on demand is started by the deploy's broker before a call.
        self.on_demand = on_demand

    def _post(self, route, body):
        from urllib.request import Request, urlopen

        from .broker import ensure

        ensure({"on_demand": self.on_demand})
        request = Request(self.url.rsplit("/", 1)[0] + route, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    @staticmethod
    def _masks(result):
        import cv2

        masks = []
        for item in result["instances"]:
            raw = np.frombuffer(base64.b64decode(item["mask"].split(",", 1)[1]), dtype=np.uint8)
            masks.append(Mask(cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) > 127, item["box"], item["score"]))
        return masks

    def segment_point(self, image, points, labels=None):
        """Candidate masks of the object under the points (SAM 3 interactive predictor),
        best predicted IoU first."""
        points = [[float(u), float(v)] for u, v in points]
        labels = [1] * len(points) if labels is None else [int(v) for v in labels]
        return self._masks(self._post("/segment_point", {"image": image_data_url(image), "points": points,
                                                         "labels": labels}))

    def segment(self, image, prompt, threshold=None):
        """Instances best first. SAM 3 scores depend strongly on phrasing (on one
        image "bowl" scored 0.13 and "white bowl" 0.72 for the same three bowls), so
        by default instances are kept when they score at least half the best score
        (and at least 0.05); pass `threshold` for an absolute cut instead."""
        masks = self._masks(self._post("/segment", {"image": image_data_url(image), "prompt": str(prompt),
                                                    "threshold": 0.0}))
        if not masks:
            return []
        cut = float(threshold) if threshold is not None else max(0.05, 0.5 * masks[0].score)
        return [mask for mask in masks if mask.score >= cut]


class VisionModel:
    """A vision-language model role (vlm, pointer or verifier) on a ChatModel, at
    temperature 0 with 1024 output tokens unless its config says otherwise. `usage` adds
    up its calls, tokens and cost; `point_format` (json or molmo) is how it points."""

    def __init__(self, config):
        config = dict(config)
        # "json": Qwen-style {"point_2d": [x, y]} replies; "molmo": Molmo2's <points coords=...>.
        self.point_format = config.pop("point_format", "json")
        if self.point_format not in ("json", "molmo"):
            raise ValueError("point_format must be json or molmo")
        # Pointing needs full-resolution images; a local server may otherwise downscale them.
        extra = {"max_image_side": 0} if config.get("backend") == "openai_chat" else {}
        self.chat = ChatModel({"max_output_tokens": 1024, "temperature": 0.0, "extra_body": extra, **config})
        # Tokens and cost of every call, reported with the episode.
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def ask(self, images, question, system="Answer briefly and precisely."):
        """The model's text answer to `question` about `images` (RGB arrays, sent as JPEG)
        under `system`; the call is added to `usage`."""
        content = []
        for image in images:
            content.append({"type": "input_image", "image_url": image_data_url(image), "detail": "auto"})
        content.append({"type": "input_text", "text": question})
        reply = self.chat.complete(system, [{"role": "user", "content": content}])
        self.usage["calls"] += 1
        self.usage["input_tokens"] += int(reply.usage.get("input_tokens") or 0)
        self.usage["output_tokens"] += int(reply.usage.get("output_tokens") or 0)
        self.usage["cost_usd"] += float(reply.usage.get("cost") or 0.0)
        return reply.text

    def point(self, image, prompt):
        """The pixel points (x, y) the model gives for `prompt` in `image`, read from its
        0-1000 coordinates (JSON point_2d, or Molmo2 points); [] when it gives none."""
        height, width = np.asarray(image).shape[:2]
        if self.point_format == "molmo":
            # Molmo2's own pointing prompt, without a system message.
            return molmo_points(self.ask([image], f"Point to the {prompt}.", system=""), width, height)
        text = self.ask([image], f"Point to: {prompt}", system=_POINT_SYSTEM)
        results = []
        for item in _json_list(text):
            point = item.get("point_2d") or item.get("point") if isinstance(item, dict) else None
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                continue
            x, y = float(point[0]), float(point[1])
            results.append((x / 1000.0 * width, y / 1000.0 * height))
        return results


def molmo_points(text, width, height):
    """Pixels from a Molmo2 reply: <points coords="1 1 433 424 2 614 465">...</points>,
    i.e. the image number, then index x y per point with x and y scaled to 0-1000
    (Molmo 1's <point x="52.3" y="40.2"> in percent too)."""
    points = []
    for coords in re.finditer(r'coords="([^"]*)"', text or ""):
        for match in re.finditer(r"(\d+) (\d{3,4}) (\d{3,4})", coords.group(1)):
            points.append((float(match.group(2)) / 1000.0 * width, float(match.group(3)) / 1000.0 * height))
    for match in re.finditer(r'<point\b[^>]*?\bx="([\d.]+)"[^>]*?\by="([\d.]+)"', text or ""):
        points.append((float(match.group(1)) / 100.0 * width, float(match.group(2)) / 100.0 * height))
    return [(x, y) for x, y in points if 0 <= x <= width and 0 <= y <= height]


def _json_list(text):
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def image_data_url(image):
    """An HxWx3 RGB array as a JPEG data URL."""
    import cv2

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("images must be HxWx3 RGB arrays")
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode the image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()
