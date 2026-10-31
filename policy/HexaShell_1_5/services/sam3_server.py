"""The machine's shared SAM3 segmentation service: one model on one GPU, used by every episode.

HTTP with JSON bodies on 127.0.0.1 only (stdlib ThreadingHTTPServer).  Requests are handled in
parallel threads, but all GPU work runs one job at a time, first come first served, on a single
GPU thread, so results never mix between clients (and CUDA library handles exist only once).

  POST /image    {"png_b64": ...} or {"jpeg_b64": ...} or {"rgb_b64": raw RGB bytes, "shape": [h, w]}
                 -> {"image_id", "cached", "h", "w", "ms"}
  POST /segment  {"image_id", "text": "red cup", "max_masks": 8, "min_score": 0.0}
                 {"image_id", "points": [[x, y], ...], "labels": [1, 0, ...]}
                 {"image_id", "box": [x0, y0, x1, y1]}            (box and points may be combined)
                 -> {"masks": [{"score", "box", "area", "png_b64"}, ...], "kind", "h", "w", "ms", "wait_ms"}
  GET  /health   -> {"ok": true, "gpu", "mem_mb", "images_cached", "requests", ...}

Conventions:
  - Pixels are (x, y) = (column, row), origin at the top-left corner.  Boxes are [x0, y0, x1, y1]
    in pixels; a returned mask's box is inclusive (its first and last column and row).
  - image_id = sha256(b"<h>x<w>:" + RGB bytes), so a client can compute it before sending.
  - Each mask is a 1-bit PNG of the whole picture.
  - text: every instance found, best score first.
  - points and/or box: SAM's candidate masks for one object, best first.  A single click returns
    SAM's three candidates (e.g. whole object, part, sub-part); several points or a box return
    SAM's single answer, which is the better one for an unambiguous prompt.  Send
    "multimask": true or false to choose yourself.
  - Masks that are empty, or score below min_score, are left out, so "masks" can be [] (e.g. a
    box around bare table: SAM finds nothing there).
  - Errors are {"error": ...} with 400 (bad request), 404 (no such endpoint, or "unknown image":
    send the picture again with /image), 405/501 (method), 411/413 (body), 500, or 503 (GPU out
    of memory: try again; or the service is broken after a CUDA failure: restart it).

Memory: the image and text encoders' Linear and Conv weights are stored in bfloat16.  They only
ever run under bfloat16 autocast, which casts them to exactly these values, so masks and scores
are bit-identical to the fp32 weights while 1.4 GB less is used.  The text encoder's token table
(200 MB) stays in CPU memory; a prompt's few rows are looked up there and copied over (exact).
A cached picture keeps only the ViT trunk's output (1024x72x72 bf16, 10 MB; PyTorch holds it in a
16 MB block of pinned CPU memory) instead of SAM3's full feature state (170 MB of GPU memory).  For each request it is copied back
(under 1 ms) and the small FPN neck is re-run from it (a few ms), giving bit-identical features.
So the cache costs no GPU memory, and the GPU footprint stays flat after the warm-up.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

DEFAULT_PORT = 18300
DEFAULT_CHECKPOINT = "sam3.pt"
MAX_SIDE = 4096
MAX_POINTS = 64
MAX_TEXT = 300
MB = 2 ** 20


def image_id(rgb: np.ndarray) -> str:
    """The id of an RGB uint8 picture; the client computes the same."""
    h, w = rgb.shape[:2]
    digest = hashlib.sha256(b"%dx%d:" % (h, w))
    digest.update(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    return digest.hexdigest()


def log(msg: str) -> None:
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    sys.stderr.flush()


def host_rss_mb() -> int | None:
    """This process's resident CPU memory."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def release_host_memory() -> None:
    """Give freed heap memory (the CPU copy of the model built at start) back to the system."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def bad(message: str) -> HttpError:
    return HttpError(400, message)


# -- the model ---------------------------------------------------------------------------------------


class TrunkTap(nn.Module):
    """Wraps the ViT trunk: can record its output, or return a recorded one instead of running."""

    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.record = False
        self.recorded = None
        self.replay = None

    def forward(self, x):
        if self.replay is not None:
            return self.replay
        out = self.trunk(x)
        if self.record:
            self.recorded = out
        return out


def bf16_weights(model: nn.Module) -> int:
    """Store the image and text encoders' Linear/Conv weights in bf16.  Returns how many values."""
    count = 0
    for part in (model.backbone.vision_backbone, model.backbone.language_backbone):
        for mod in part.modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                names = ("weight", "bias")
            elif isinstance(mod, nn.MultiheadAttention):
                names = ("in_proj_weight", "in_proj_bias")  # its out_proj is a Linear
            else:
                continue
            for name in names:
                p = getattr(mod, name, None)
                if p is not None and p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
                    count += p.numel()
    return count


class CpuEmbedding(nn.Module):
    """A token embedding whose table stays in CPU memory; only the looked-up rows go to the GPU."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = table.detach().cpu()  # a plain attribute, so model.to() leaves it on the CPU

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.cpu(), self.table).to(ids.device)


def to_host(t: torch.Tensor) -> torch.Tensor:
    """A copy of t in pinned CPU memory (quick to copy back), with t's strides: the trunk output is a
    channels-last view, and the neck's convolutions give slightly different numbers for another layout."""
    out = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, pin_memory=True)
    out.copy_(t)  # waits for the GPU
    return out


@dataclass
class Picture:
    """A cached picture: its trunk features and size."""

    trunk: list
    h: int
    w: int
    nbytes: int


Found = list[tuple[np.ndarray, float]]  # (bool HxW mask, score), best first


class Engine:
    """SAM3 on the first visible GPU.  Only the service's one GPU thread calls it."""

    def __init__(self, checkpoint: str, *, fp32_weights: bool = False, text_cache: int = 128):
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device is visible")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sam3.model.data_misc import interpolate
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            # built on the CPU, so the fp32 copy of the converted weights never reaches the GPU
            model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint, load_from_HF=False,
                                           enable_inst_interactivity=True)
        self.bf16_values = 0 if fp32_weights else bf16_weights(model)
        text_encoder = model.backbone.language_backbone.encoder
        text_encoder.token_embedding = CpuEmbedding(text_encoder.token_embedding.weight)
        model.to("cuda")
        model.eval()
        neck = model.backbone.vision_backbone
        self.tap = TrunkTap(neck.trunk)
        neck.trunk = self.tap
        self.model = model
        self.proc = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        self.interpolate = interpolate
        self.gpu_name = torch.cuda.get_device_name(0)
        self._blank: dict[tuple[int, int], torch.Tensor] = {}
        self._text: OrderedDict[str, dict] = OrderedDict()
        self._text_cap = max(0, text_cache)

    @staticmethod
    def _autocast():
        # no weight-cast cache: it would keep bf16 copies of fp32 weights alive (600 MB)
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)

    # pictures

    def encode(self, rgb: np.ndarray) -> Picture:
        h, w = rgb.shape[:2]
        with torch.inference_mode(), self._autocast():
            self.tap.record = True
            try:
                self.proc.set_image(Image.fromarray(rgb))  # only the trunk output is kept
                trunk = [to_host(t.to(torch.bfloat16)) for t in self.tap.recorded]
            finally:
                self.tap.record = False
                self.tap.recorded = None
        return Picture(trunk=trunk, h=h, w=w, nbytes=sum(t.numel() * t.element_size() for t in trunk))

    def _state(self, pic: Picture) -> dict:
        """SAM3's feature state for a cached picture.  Call inside inference mode and autocast."""
        key = (pic.h, pic.w)
        if key not in self._blank:
            # stand-in input: the trunk is replayed, so only the size matters
            self._blank[key] = torch.zeros((3, pic.h, pic.w), dtype=torch.uint8, device="cuda")
        self.tap.replay = [t.to("cuda", non_blocking=True) for t in pic.trunk]
        try:
            return self.proc.set_image(self._blank[key])
        finally:
            self.tap.replay = None

    # prompts

    def _text_features(self, text: str) -> dict:
        feats = self._text.get(text)
        if feats is not None:
            self._text.move_to_end(text)
            return feats
        out = self.model.backbone.forward_text([text], device="cuda")
        feats = {k: out[k] for k in ("language_features", "language_mask")}  # language_embeds is never read
        if self._text_cap:
            self._text[text] = feats
            while len(self._text) > self._text_cap:
                self._text.popitem(last=False)
        return feats

    def by_text(self, pic: Picture, text: str, max_masks: int, min_score: float) -> Found:
        with torch.inference_mode(), self._autocast():
            feats = dict(self._state(pic)["backbone_out"])
            feats.update(self._text_features(text))
            out = self.model.forward_grounding(backbone_out=feats, find_input=self.proc.find_stage,
                                               geometric_prompt=self.model._get_dummy_prompt(), find_target=None)
            # instance score = query score x presence score, as in Sam3Processor
            presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)
            scores = (out["pred_logits"].sigmoid() * presence).squeeze(-1)[0]
            ranked, order = torch.sort(scores, descending=True, stable=True)
            order = order[ranked >= min_score][:max_masks]
            if order.numel() == 0:
                return []
            logits = out["pred_masks"][0, order].unsqueeze(1)
            step = max(1, (16 << 20) // (pic.h * pic.w))  # upsample in chunks of at most ~64 MB (big pictures)
            masks = np.concatenate([
                (self.interpolate(logits[i:i + step], (pic.h, pic.w), mode="bilinear", align_corners=False)
                 .sigmoid() > 0.5)[:, 0].cpu().numpy() for i in range(0, len(logits), step)])
            values = scores[order].float().cpu().tolist()
        return list(zip(masks, values))

    def by_prompt(self, pic: Picture, points, labels, box, multimask: bool, max_masks: int,
                  min_score: float) -> Found:
        with torch.inference_mode(), self._autocast():
            masks, scores, _ = self.model.predict_inst(
                self._state(pic),
                point_coords=None if points is None else np.asarray(points, dtype=np.float32),
                point_labels=None if points is None else np.asarray(labels, dtype=np.int64),
                box=None if box is None else np.asarray(box, dtype=np.float32),
                multimask_output=multimask,
            )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        masks = np.asarray(masks).reshape(len(scores), pic.h, pic.w) > 0
        order = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= min_score][:max_masks]
        return [(masks[i], float(scores[i])) for i in order]

    def warm_up(self) -> None:
        """One request of each kind, so kernels are compiled before the first client comes."""
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[:] = np.linspace(40, 200, 640).astype(np.uint8)[None, :, None]
        rgb[180:300, 250:390] = (200, 30, 30)
        pic = self.encode(rgb)
        self.by_text(pic, "red square", 2, 0.0)
        self.by_prompt(pic, [[320, 240]], [1], None, True, 3, 0.0)
        self.by_prompt(pic, [[320, 240], [100, 100]], [1, 0], [240, 170, 400, 310], False, 3, 0.0)
        self._text.clear()

    @staticmethod
    def memory() -> dict:
        c = torch.cuda
        return {"torch_alloc_mb": round(c.memory_allocated() / MB), "torch_reserved_mb": round(c.memory_reserved() / MB),
                "torch_peak_mb": round(c.max_memory_allocated() / MB)}

    @staticmethod
    def trim(slack_mb: float) -> None:
        """Hand unused allocator memory back to the GPU when more than slack_mb of it is held."""
        c = torch.cuda
        if c.memory_reserved() - c.memory_allocated() > slack_mb * MB:
            c.empty_cache()


# -- the cache ---------------------------------------------------------------------------------------


class Pictures:
    """Least-recently-used cache of encoded pictures by image id."""

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._items: OrderedDict[str, Picture] = OrderedDict()
        self._lock = threading.Lock()
        self.evicted = 0

    def get(self, key: str) -> Picture | None:
        with self._lock:
            pic = self._items.get(key)
            if pic is not None:
                self._items.move_to_end(key)
            return pic

    def put(self, key: str, pic: Picture) -> None:
        with self._lock:
            self._items[key] = pic
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
                self.evicted += 1

    def stats(self) -> dict:
        with self._lock:
            return {"images_cached": len(self._items), "cache_capacity": self.capacity,
                    "cache_mb": round(sum(p.nbytes for p in self._items.values()) / MB, 1),
                    "images_evicted": self.evicted}


# -- requests ----------------------------------------------------------------------------------------


def decode_picture(body: dict) -> np.ndarray:
    keys = [k for k in ("png_b64", "jpeg_b64", "rgb_b64") if k in body]
    if len(keys) != 1:
        raise bad("give the picture as exactly one of png_b64, jpeg_b64 or rgb_b64 (with shape)")
    data = body[keys[0]]
    if not isinstance(data, str) or not data:
        raise bad(f"{keys[0]} must be a non-empty base64 string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise bad(f"{keys[0]} is not valid base64") from None
    if keys[0] == "rgb_b64":
        # raw H x W x 3 uint8 bytes: no PNG encode/decode (30 ms less per new picture)
        shape = body.get("shape")
        if not (isinstance(shape, list) and len(shape) in (2, 3) and shape[2:] in ([], [3])
                and all(isinstance(v, int) and not isinstance(v, bool) for v in shape)):
            raise bad('rgb_b64 needs "shape": [h, w]')
        h, w = shape[:2]
        if not (8 <= w <= MAX_SIDE and 8 <= h <= MAX_SIDE):
            raise bad(f"picture size {w}x{h} is outside 8..{MAX_SIDE} pixels")
        if len(raw) != h * w * 3:
            raise bad(f"rgb_b64 holds {len(raw)} bytes; a {w}x{h} RGB picture has {h * w * 3}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    try:
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if not (8 <= w <= MAX_SIDE and 8 <= h <= MAX_SIDE):
                raise bad(f"picture size {w}x{h} is outside 8..{MAX_SIDE} pixels")
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except HttpError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a bad picture
        raise bad(f"cannot decode the picture: {exc}") from None
    return np.ascontiguousarray(rgb)


def number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise bad(f"{what} must be a finite number, got {value!r}")
    return float(value)


@dataclass
class Prompt:
    kind: str  # "text" | "points" | "box" | "points+box"
    text: str | None = None
    points: list | None = None
    labels: list | None = None
    box: list | None = None
    multimask: bool | None = None
    max_masks: int = 8
    min_score: float = 0.0


def parse_prompt(body: dict) -> Prompt:
    max_masks = body.get("max_masks", 8)
    if isinstance(max_masks, bool) or not isinstance(max_masks, int) or not 1 <= max_masks <= 64:
        raise bad("max_masks must be an integer from 1 to 64")
    min_score = number(body.get("min_score", 0.0), "min_score")
    if not 0.0 <= min_score <= 1.0:
        raise bad("min_score must be between 0 and 1")
    multimask = body.get("multimask")
    if multimask is not None and not isinstance(multimask, bool):
        raise bad("multimask must be true, false or left out")
    text, points, labels, box = (body.get(k) for k in ("text", "points", "labels", "box"))
    if text is not None:
        if points is not None or labels is not None or box is not None:
            raise bad("give either text, or points and/or box - not both")
        if not isinstance(text, str) or not text.strip():
            raise bad("text must be a non-empty string")
        if len(text) > MAX_TEXT:
            raise bad(f"text is longer than {MAX_TEXT} characters")
        return Prompt("text", text=text.strip(), max_masks=max_masks, min_score=min_score)
    if points is None and box is None:
        raise bad("give a prompt: text, points (with labels), or box")
    if points is not None:
        if not isinstance(points, list) or not points or len(points) > MAX_POINTS:
            raise bad(f"points must be a list of 1 to {MAX_POINTS} [x, y] pairs")
        parsed = []
        for p in points:
            if not isinstance(p, list) or len(p) != 2:
                raise bad(f"each point must be [x, y], got {p!r}")
            parsed.append([number(p[0], "point x"), number(p[1], "point y")])
        points = parsed
        if labels is None:
            labels = [1] * len(points)
        if not isinstance(labels, list) or len(labels) != len(points) or any(v not in (0, 1) for v in labels):
            raise bad("labels must be a list of 1 (on the object) or 0 (not on it), one per point")
        labels = [int(v) for v in labels]
        if box is None and 1 not in labels:
            raise bad("at least one point must be positive (label 1) when there is no box")
    elif labels is not None:
        raise bad("labels were given without points")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4:
            raise bad("box must be [x0, y0, x1, y1]")
        box = [number(v, "box coordinate") for v in box]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise bad("box must have x1 > x0 and y1 > y0")
    kind = "points+box" if points is not None and box is not None else ("points" if points is not None else "box")
    return Prompt(kind, points=points, labels=labels, box=box, multimask=multimask, max_masks=max_masks,
                  min_score=min_score)


def check_inside(prompt: Prompt, pic: Picture) -> None:
    # points outside the picture are usually swapped (row, col) - refuse them
    for x, y in prompt.points or []:
        if not (0 <= x < pic.w and 0 <= y < pic.h):
            raise bad(f"point ({x:g}, {y:g}) is outside the {pic.w}x{pic.h} picture; points are (x=column, y=row)")
    if prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        if x1 < 0 or y1 < 0 or x0 >= pic.w or y0 >= pic.h:
            raise bad(f"box {prompt.box} does not overlap the {pic.w}x{pic.h} picture")
        prompt.box = [max(0.0, x0), max(0.0, y0), min(pic.w - 1.0, x1), min(pic.h - 1.0, y1)]


def mask_json(mask: np.ndarray, score: float) -> dict:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")  # a bool array saves as a 1-bit PNG
    return {"score": round(score, 4), "box": [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])],
            "area": int(mask.sum()), "png_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def is_cuda_failure(exc: BaseException) -> bool:
    """A CUDA error that leaves the process unusable (not an out-of-memory)."""
    text = str(exc)
    return not isinstance(exc, torch.cuda.OutOfMemoryError) and (
        "CUDA error" in text or "illegal memory access" in text or "device-side assert" in text)


class Service:
    def __init__(self, engine: Engine, cache_images: int, slack_mb: float, checkpoint: str):
        self.engine = engine
        self.pictures = Pictures(cache_images)
        # the one thread that touches the GPU; every thread using CUDA would get its own library handles
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.slack_mb = slack_mb
        self.checkpoint = checkpoint
        self.started = time.time()
        self.broken: str | None = None  # set after a CUDA failure; the process must be restarted
        self._stats = threading.Lock()
        self.counts = {"image": 0, "image_encoded": 0, "segment": 0, "health": 0, "errors": 0}
        self.gpu_ms = 0.0
        self._smi: tuple[float, int | None] = (0.0, None)

    def count(self, key: str) -> None:
        with self._stats:
            self.counts[key] = self.counts.get(key, 0) + 1

    def _gpu(self, fn, *args):
        """Run fn(*args) on the GPU thread and wait.  Returns (result, ms it waited for its turn)."""
        if self.broken:
            raise HttpError(503, f"service is broken ({self.broken}); restart it")
        t0 = time.perf_counter()

        def job():
            t1 = time.perf_counter()
            try:
                return fn(*args), (t1 - t0) * 1000
            finally:
                self.engine.trim(self.slack_mb)
                with self._stats:
                    self.gpu_ms += (time.perf_counter() - t1) * 1000

        return self.gpu.submit(job).result()

    def post_image(self, body: dict) -> tuple[dict, str]:
        t0 = time.perf_counter()
        rgb = decode_picture(body)
        key = image_id(rgb)
        h, w = rgb.shape[:2]
        cached = self.pictures.get(key) is not None
        if not cached:
            def work():
                if self.pictures.get(key) is None:  # another client may have sent it meanwhile
                    self.pictures.put(key, self.engine.encode(rgb))
                    self.count("image_encoded")
            self._gpu(work)
        ms = (time.perf_counter() - t0) * 1000
        return ({"image_id": key, "cached": cached, "h": h, "w": w, "ms": round(ms, 1)},
                f"{key[:10]} {w}x{h} {'cached' if cached else 'encoded'}")

    def post_segment(self, body: dict) -> tuple[dict, str]:
        t0 = time.perf_counter()
        key = body.get("image_id")
        if not isinstance(key, str) or not key:
            raise bad("image_id must be the string that /image returned")
        prompt = parse_prompt(body)
        pic = self.pictures.get(key)
        if pic is None:
            raise HttpError(404, "unknown image")
        check_inside(prompt, pic)
        if prompt.kind == "text":
            found, waited = self._gpu(self.engine.by_text, pic, prompt.text, prompt.max_masks, prompt.min_score)
        else:
            multimask = prompt.multimask
            if multimask is None:  # a lone click is ambiguous; more points or a box are not
                multimask = prompt.kind == "points" and len(prompt.points) == 1
            found, waited = self._gpu(self.engine.by_prompt, pic, prompt.points, prompt.labels, prompt.box,
                                      multimask, prompt.max_masks, prompt.min_score)
        masks = [mask_json(m, s) for m, s in found if m.any()]
        ms = (time.perf_counter() - t0) * 1000
        what = repr(prompt.text) if prompt.kind == "text" else prompt.kind
        top = f" top {masks[0]['score']:.3f}" if masks else ""
        return ({"masks": masks, "kind": prompt.kind, "h": pic.h, "w": pic.w, "ms": round(ms, 1),
                 "wait_ms": round(waited, 1)}, f"{key[:10]} {what} -> {len(masks)} masks{top}")

    def process_mb(self) -> int | None:
        """This process's GPU memory as nvidia-smi sees it (includes the CUDA context)."""
        stamp, value = self._smi
        if time.time() - stamp < 5.0:
            return value
        value = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(os.getpid()):
                    value = int(parts[1])
        except (OSError, subprocess.SubprocessError, ValueError):
            value = None
        self._smi = (time.time(), value)
        return value

    def get_health(self, body: dict) -> tuple[dict, str]:
        with self._stats:
            counts, gpu_ms = dict(self.counts), self.gpu_ms
        out = {"ok": self.broken is None, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               "gpu_name": self.engine.gpu_name, "mem_mb": self.process_mb(), **self.engine.memory(),
               "cpu_rss_mb": host_rss_mb(),
               **self.pictures.stats(), "requests": counts["image"] + counts["segment"], "counts": counts,
               "gpu_busy_s": round(gpu_ms / 1000, 1), "bf16_weights_m": round(self.engine.bf16_values / 1e6, 1),
               "checkpoint": self.checkpoint, "pid": os.getpid(), "uptime_s": round(time.time() - self.started)}
        if self.broken:
            out["error"] = f"service is broken ({self.broken}); restart it"
        return out, ""


ROUTES = {("POST", "/image"): ("image", Service.post_image),
          ("POST", "/segment"): ("segment", Service.post_segment),
          ("GET", "/health"): ("health", Service.get_health)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so a client reuses its connection
    server_version = "HexaShellSAM3/1.5"
    timeout = 300  # idle connections are dropped after this
    max_body = 100 * MB

    def setup(self) -> None:
        super().setup()
        # headers and body are written separately; without this, Nagle + delayed ACK add 40 ms
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._handle("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        pass  # _handle writes one line per request

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # problems http.server finds itself (bad request line, unknown method, ...): JSON as well
        self.close_connection = True
        message = message or HTTPStatus(code).phrase
        log(f"{int(code)} {getattr(self, 'requestline', '')[:100]!r} {message}")
        self._send(int(code), {"error": message})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log(f"http: {format % args}")

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise HttpError(411, "send a Content-Length, not a chunked body")
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            n = int(length)
        except ValueError:
            self.close_connection = True
            raise HttpError(400, "bad Content-Length") from None
        if n < 0 or n > self.max_body:
            self.close_connection = True  # the body stays unread, so the connection cannot be reused
            raise HttpError(413, f"body over {self.max_body // MB} MB")
        return self.rfile.read(n)

    def _handle(self, method: str) -> None:
        service: Service = self.server.service  # type: ignore[attr-defined]
        t0 = time.perf_counter()
        path = self.path.split("?", 1)[0]
        note = ""
        try:
            raw = self._read_body()
            route = ROUTES.get((method, path))
            if route is None:
                if any(p == path for (_, p) in ROUTES):
                    raise HttpError(405, f"{path} does not take {method}")
                raise HttpError(404, f"no such endpoint: {method} {path}")
            name, fn = route
            service.count(name)
            body: Any = {}
            if method == "POST":
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise bad(f"body is not valid JSON: {exc}") from None
                if not isinstance(body, dict):
                    raise bad("body must be a JSON object")
            payload, note = fn(service, body)
            status = 503 if name == "health" and not payload["ok"] else 200
        except HttpError as exc:
            status, payload = exc.status, {"error": exc.message}
        except Exception as exc:  # noqa: BLE001 - the service must survive any request
            log("request failed:\n" + traceback.format_exc())
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                status, payload = 503, {"error": "GPU out of memory; try again"}
                torch.cuda.empty_cache()
            else:
                status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
                if is_cuda_failure(exc):
                    service.broken = f"{type(exc).__name__}: {str(exc)[:200]}"
        if status >= 400:
            service.count("errors")
            note = f"{note} {payload.get('error')}".strip()
        self._send(status, payload)
        if path != "/health" or status != 200:
            log(f"{status} {method} {path} {(time.perf_counter() - t0) * 1000:.0f}ms {note}")

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return  # a client went away
        log(f"connection from {client_address} failed:\n{traceback.format_exc()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The shared SAM3 segmentation service (127.0.0.1 only).")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--checkpoint", default=os.environ.get("SAM3_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT)
    ap.add_argument("--cache-images", type=int, default=128,
                    help="pictures kept encoded; 16 MB of pinned CPU memory each, no GPU memory")
    ap.add_argument("--text-cache", type=int, default=256, help="text prompts kept encoded (a few KB each)")
    ap.add_argument("--slack-mb", type=float, default=1024.0,
                    help="hand unused allocator memory back to the GPU when more than this is held; requests "
                         "need ~650 MB of working memory, and freeing it every time makes them slower and jittery")
    ap.add_argument("--fp32-weights", action="store_true", help="keep all weights in fp32 (1.4 GB more, same masks)")
    ap.add_argument("--pid-file", default=None, help="write this process's pid here")
    args = ap.parse_args(argv)

    # grow allocator blocks in place instead of keeping many fixed ones (less idle memory); before CUDA starts
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.pid_file:
        Path(args.pid_file).write_text(f"{os.getpid()}\n")
    checkpoint = str(Path(args.checkpoint).expanduser())
    if not Path(checkpoint).is_file():
        log(f"checkpoint not found: {checkpoint}")
        return 2
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    log(f"starting: pid {os.getpid()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"port {args.port}, checkpoint {checkpoint}")
    t0 = time.time()
    engine = Engine(checkpoint, fp32_weights=args.fp32_weights, text_cache=args.text_cache)
    loaded = engine.memory()["torch_alloc_mb"]
    rss = host_rss_mb()
    release_host_memory()
    log(f"CPU memory {rss} MB after load, {host_rss_mb()} MB after giving back freed memory")
    service = Service(engine, args.cache_images, args.slack_mb, checkpoint)
    service.gpu.submit(engine.warm_up).result()  # its working memory stays reserved: the footprint is known now
    torch.cuda.reset_peak_memory_stats()
    log(f"model ready in {time.time() - t0:.0f}s on {engine.gpu_name}: {engine.bf16_values / 1e6:.0f}M weight values "
        f"in bf16, {loaded} MB after load, {engine.memory()} after warm-up")

    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        log(f"cannot listen on 127.0.0.1:{args.port}: {exc}")
        return 3
    server.service = service  # type: ignore[attr-defined]

    def stop(signum: int, frame: Any) -> None:  # noqa: ARG001
        log(f"signal {signum}: shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"listening on http://127.0.0.1:{args.port}; process GPU memory {service.process_mb()} MB")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        service.gpu.shutdown(wait=True, cancel_futures=True)
        if args.pid_file:
            try:
                if Path(args.pid_file).read_text().strip() == str(os.getpid()):
                    Path(args.pid_file).unlink()
            except OSError:
                pass
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
