"""Run one shared SAM3 model on one GPU. All image caching and inference share one worker.

The 6.5 client uploads raw RGB to /image and sends text, points or boxes to /segment.
Masks are returned as PNG. /health reports readiness. The listener binds to loopback.
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
import sys
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

DEFAULT_PORT = 18300
MAX_BODY = 100 * 2 ** 20
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parents[1] / "checkpoints" / "sam3.pt")
MAX_SIDE = 4096
MAX_POINTS = 64
MAX_TEXT = 300
MB = 2 ** 20


def image_id(rgb: np.ndarray) -> str:
    """The stable cache key for an RGB picture."""
    h, w = rgb.shape[:2]
    digest = hashlib.sha256(b"%dx%d:" % (h, w))
    digest.update(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    return digest.hexdigest()


def log(msg: str) -> None:
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    sys.stderr.flush()


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


def bf16_weights(model: nn.Module) -> None:
    """Store the image and text encoders' Linear/Conv weights in bf16."""
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
        if not fp32_weights:
            bf16_weights(model)
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
        return Picture(trunk=trunk, h=h, w=w)

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
    def trim(slack_mb: float) -> None:
        """Hand unused allocator memory back to the GPU when more than slack_mb of it is held."""
        c = torch.cuda
        if c.memory_reserved() - c.memory_allocated() > slack_mb * MB:
            c.empty_cache()


def decode_picture(body: dict) -> np.ndarray:
    shape, data = body.get("shape"), body.get("rgb_b64")
    if not isinstance(shape, list) or len(shape) != 2 or not all(
        type(v) is int and 8 <= v <= MAX_SIDE for v in shape
    ):
        raise bad(f"shape must be [height, width] in 8..{MAX_SIDE}")
    if not isinstance(data, str):
        raise bad("rgb_b64 must contain base64 RGB bytes")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise bad("invalid rgb_b64") from None
    h, w = shape
    if len(raw) != h * w * 3:
        raise bad("RGB byte count does not match shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def number(value: Any, what: str) -> float:
    try:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    except OverflowError:
        pass
    raise bad(f"{what} must be a finite number")


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
    def __init__(self, engine: Engine, gpu: ThreadPoolExecutor, cache_images: int, slack_mb: float):
        self.engine = engine
        self.gpu = gpu
        self.cache_images = cache_images
        self.slack_mb = slack_mb
        self.pictures: OrderedDict[str, Picture] = OrderedDict()
        self.broken = False

    def run(self, operation, *args):
        return self.gpu.submit(self._run, operation, args).result()

    def _run(self, operation, args):
        if self.broken:
            raise HttpError(503, "SAM3 needs a restart")
        try:
            return operation(*args)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HttpError(503, "GPU out of memory; try again") from None
        except Exception as exc:
            if is_cuda_failure(exc):
                self.broken = True
            raise
        finally:
            if not self.broken:
                self.engine.trim(self.slack_mb)

    def image(self, key: str, rgb: np.ndarray):
        if key not in self.pictures:
            self.pictures[key] = self.engine.encode(rgb)
            if len(self.pictures) > self.cache_images:
                self.pictures.popitem(last=False)
        self.pictures.move_to_end(key)
        return {"image_id": key}

    def segment(self, key: str, prompt: Prompt) -> Found:
        pic = self.pictures.get(key)
        if pic is None:
            raise HttpError(404, "unknown image")
        check_inside(prompt, pic)
        self.pictures.move_to_end(key)
        if prompt.kind == "text":
            return self.engine.by_text(pic, prompt.text, prompt.max_masks, prompt.min_score)
        multimask = prompt.multimask
        if multimask is None:
            multimask = prompt.kind == "points" and len(prompt.points) == 1
        return self.engine.by_prompt(pic, prompt.points, prompt.labels, prompt.box,
                                     multimask, prompt.max_masks, prompt.min_score)


class Handler(BaseHTTPRequestHandler):
    timeout = 300

    def do_GET(self):
        if self.path == "/health":
            ok = not self.server.service.broken
            self.reply(200 if ok else 503, {"ok": ok})
        else:
            self.reply(404, {"error": "unknown endpoint"})

    def do_POST(self):
        service = self.server.service
        try:
            if self.path == "/image":
                rgb = decode_picture(self.read_json())
                result = service.run(service.image, image_id(rgb), rgb)
            elif self.path == "/segment":
                body = self.read_json()
                key = body.get("image_id")
                if not isinstance(key, str) or not key:
                    raise bad("image_id must be the string returned by /image")
                found = service.run(service.segment, key, parse_prompt(body))
                result = {"masks": [mask_json(mask, score) for mask, score in found if mask.any()]}
            else:
                raise HttpError(404, "unknown endpoint")
        except HttpError as exc:
            self.reply(exc.status, {"error": exc.message})
        except Exception as exc:
            log(f"SAM3 {self.path} failed: {type(exc).__name__}: {exc}")
            self.reply(500, {"error": "SAM3 request failed"})
        else:
            self.reply(200, result)

    def read_json(self):
        length = self.headers.get("Content-Length")
        if length is None or self.headers.get("Transfer-Encoding"):
            raise HttpError(411, "Content-Length is required")
        try:
            count = int(length)
        except ValueError:
            raise bad("invalid Content-Length") from None
        if not 0 < count <= MAX_BODY:
            raise HttpError(413, "request body size is outside the limit")
        try:
            body = json.loads(self.rfile.read(count))
        except (ValueError, UnicodeDecodeError):
            raise bad("invalid JSON body") from None
        if not isinstance(body, dict):
            raise bad("expected a JSON object")
        return body

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_request(self, code="-", size="-"):
        if str(code).startswith(("4", "5")):
            super().log_request(code, size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", default=os.environ.get("SAM3_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-images", type=int, default=128)
    parser.add_argument("--text-cache", type=int, default=256)
    parser.add_argument("--slack-mb", type=float, default=1024.0)
    parser.add_argument("--fp32-weights", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.cache_images < 1:
        parser.error("--cache-images must be positive")
    if not math.isfinite(args.slack_mb) or args.slack_mb < 0:
        parser.error("--slack-mb must be finite and non-negative")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-gpu") as gpu:
        engine = gpu.submit(Engine, str(checkpoint), fp32_weights=args.fp32_weights,
                            text_cache=args.text_cache).result()
        release_host_memory()
        gpu.submit(engine.warm_up).result()
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.service = Service(engine, gpu, args.cache_images, args.slack_mb)

            def stop(_signum, _frame):
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            log(f"SAM3 ready at http://127.0.0.1:{server.server_port}")
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
