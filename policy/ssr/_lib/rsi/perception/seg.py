from __future__ import annotations

from .. import ASSETS

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import models

DATA = ASSETS
MODEL_NAME = "seg_yolo"
CAMERAS = "head", "left_wrist", "right_wrist"
DEFAULT_PARAMS = {
    "imgsz": 640,
    "conf": 0.35,
    "iou": 0.6,
    "half": True,
    "min_area": 20,
    "agnostic": True,
    "cuda_graph": True,
    "mask_nms": 0.0,
    "e2e": False,
    "gpu_preprocess": False,
    "mask_contain": 0.0,
}
CAMERA_HW = 480, 640


@dataclass
class Instance:
    mask: np.ndarray
    category: str
    score: float
    bbox: tuple[float, float, float, float]
    class_id: int = -1
    area: int = field(default=0)

    def __post_init__(self):
        if not self.area:
            self.area = int(self.mask.sum())


def _offline_env() -> None:
    os.environ["YOLO_OFFLINE"] = "1"
    if not os.environ.get("YOLO_CONFIG_DIR"):
        d = Path(tempfile.gettempdir()) / f"rsi_yolo_cfg_{os.getuid()}"
        d.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(d)
    os.environ.setdefault("YOLO_VERBOSE", "False")


class Segmenter:
    def __init__(self, weights: str | Path | None = None, params: dict | None = None, device: str | None = None):
        if weights is None:
            entry = models.entries().get(MODEL_NAME) or {}
            self.weights = models.path(MODEL_NAME)
            self.version = str(entry.get("version"))
            p = dict(DEFAULT_PARAMS, **entry.get("params") or {})
        else:
            self.weights = Path(weights)
            if not self.weights.is_file():
                raise FileNotFoundError(f"segmentation weights not found: {self.weights}")
            self.version = f"file:{self.weights}"
            p = dict(DEFAULT_PARAMS)
        p.update(params or {})
        unknown = sorted(set(p) - set(DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"unknown segmentation settings {unknown} (known: {sorted(DEFAULT_PARAMS)})")
        self.params = p
        self.imgsz = int(p["imgsz"])
        self.conf = float(p["conf"])
        self.iou = float(p["iou"])
        self.half = bool(p["half"])
        self.min_area = int(p["min_area"])
        self.agnostic = bool(p["agnostic"])
        self.mask_nms = float(p.get("mask_nms") or 0.0)
        self.mask_contain = float(p.get("mask_contain") or 0.0)
        self.e2e = bool(p.get("e2e"))
        _offline_env()
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(self.weights), task="segment")
        names = self.model.names
        self.names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        self._lock = threading.Lock()
        self.cuda_graph = False
        if self.device != "cpu" and bool(p.get("cuda_graph", True)):
            self._install_cuda_graph(*CAMERA_HW)
        self.gpu_preprocess = False
        if self.device != "cpu" and bool(p.get("gpu_preprocess")):
            self._install_gpu_preprocess(*CAMERA_HW)

    def _install_gpu_preprocess(self, h: int, w: int) -> None:
        import torch

        if getattr(self.model, "predictor", None) is None:
            self._predict(np.zeros((h, w, 3), np.uint8), self.conf)
        pred = self.model.predictor
        orig = pred.preprocess
        ref = orig([np.zeros((h, w, 3), np.uint8)])
        H, W = (int(v) for v in ref.shape[2:])
        r = min(self.imgsz / h, self.imgsz / w)
        if (int(round(h * r)), int(round(w * r))) != (H, W):
            return
        dev, dtype = ref.device, ref.dtype

        def preprocess(im):
            if (
                isinstance(im, list)
                and len(im) == 1
                and isinstance(im[0], np.ndarray)
                and (im[0].dtype == np.uint8)
                and (im[0].shape == (h, w, 3))
            ):
                return letterbox_gpu(torch, im[0], (H, W), dev, dtype)
            return orig(im)

        pred.preprocess = preprocess
        self.gpu_preprocess = True

    def _predict(self, bgr: np.ndarray, conf: float):
        extra = {"nms": False} if self.e2e else {}
        return self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            retina_masks=True,
            half=self.half and self.device != "cpu",
            agnostic_nms=self.agnostic,
            device=self.device,
            verbose=False,
            **extra,
        )[0]

    def _install_cuda_graph(self, h: int, w: int) -> None:
        import torch

        self._predict(np.zeros((h, w, 3), np.uint8), self.conf)
        ab = self.model.predictor.model
        net = getattr(getattr(ab, "backend", None), "model", None) or ab.model
        orig = type(net).forward.__get__(net)
        seen = {}

        def probe(im, *a, **k):
            seen.setdefault("x", (tuple(im.shape), im.dtype))
            return orig(im, *a, **k)

        net.forward = probe
        try:
            self._predict(np.zeros((h, w, 3), np.uint8), self.conf)
        finally:
            del net.forward
        shape, dtype = seen.get("x", ((1, 3, h, w), next(net.parameters()).dtype))
        x = torch.zeros(shape, dtype=dtype, device=self.device)
        with torch.inference_mode():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    orig(x)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = orig(x)

        def forward(im, *a, augment=False, embed=None, **k):
            if a or augment or embed or any(k.values()) or (im.shape != x.shape) or (im.dtype != x.dtype):
                return orig(im, *a, augment=augment, embed=embed, **k)
            x.copy_(im)
            graph.replay()
            return out

        net.forward = forward
        self.cuda_graph = True

    def segment(self, img: np.ndarray, camera: str = "head", conf: float | None = None) -> list[Instance]:
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {CAMERAS}")
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB image, got {img.shape}")
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        with self._lock:
            res = self._predict(bgr, self.conf if conf is None else conf)
            out = self._instances(res, img.shape[:2])
        return out

    def _instances(self, res, hw) -> list[Instance]:
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            return []
        md = res.masks.data
        if hasattr(md, "flatten") and tuple(md.shape[-2:]) == tuple(hw):
            import torch

            mb = md > 0.5
            areas = mb.flatten(1).sum(1)
            keep = areas >= self.min_area
            if (self.mask_nms > 0 or self.mask_contain > 0) and len(mb) > 1:
                keep &= _mask_nms_keep(mb, res.boxes.conf, self.mask_nms or 1.0, self.mask_contain)
            meta = (
                torch.cat([res.boxes.xyxy, res.boxes.conf[:, None], res.boxes.cls[:, None], areas[:, None].float()], 1)[
                    keep
                ]
                .cpu()
                .numpy()
            )
            masks = mb[keep].cpu().numpy()
            out = [
                Instance(
                    mask=m,
                    category=self.names[int(r[5])],
                    score=float(r[4]),
                    bbox=tuple(float(v) for v in r[:4]),
                    class_id=int(r[5]),
                    area=int(r[6]),
                )
                for m, r in zip(masks, meta)
            ]
            out.sort(key=lambda i: -i.score)
            return out
        masks = md.cpu().numpy() > 0.5
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        if (self.mask_nms > 0 or self.mask_contain > 0) and len(masks) > 1:
            k = _mask_nms_keep(masks, scores, self.mask_nms or 1.0, self.mask_contain)
            masks, boxes, scores, cls = masks[k], boxes[k], scores[k], cls[k]
        out = []
        for m, b, s, c in zip(masks, boxes, scores, cls):
            if m.shape != tuple(hw):
                import cv2

                m = cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0
            if int(m.sum()) < self.min_area:
                continue
            out.append(
                Instance(
                    mask=m, category=self.names[c], score=float(s), bbox=tuple(float(v) for v in b), class_id=int(c)
                )
            )
        out.sort(key=lambda i: -i.score)
        return out


_default: Segmenter | None = None
_default_lock = threading.Lock()


def letterbox_gpu(torch, bgr: np.ndarray, hw: tuple, device, dtype):
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(bgr)).to(device)
    t = t.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float()
    if tuple(t.shape[2:]) != tuple(hw):
        t = F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False).add_(0.5).floor_()
    t = t.to(dtype)
    t /= 255
    return t


def _mask_nms_keep(masks, scores, thr: float, contain: float = 0.0):
    if hasattr(masks, "flatten") and hasattr(masks, "device"):
        import torch

        f = masks.flatten(1).to(torch.float32)
        inter = f @ f.T
        area = f.sum(1)
        iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1.0)
        cont = (inter / area[:, None].clamp(min=1.0)).float().cpu().numpy()
        order = torch.argsort(scores, descending=True).tolist()
        iou = iou.float().cpu().numpy()
        area = area.float().cpu().numpy()
    else:
        f = np.asarray(masks).reshape(len(masks), -1).astype(np.float32)
        inter = f @ f.T
        area = f.sum(1)
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1.0)
        cont = inter / np.maximum(area[:, None], 1.0)
        order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep = np.zeros(len(order), bool)
    kept: list[int] = []
    for i in order:
        if all(
            iou[i, j] <= thr and (not (contain > 0 and cont[i, j] > contain and (area[j] >= 2 * area[i]))) for j in kept
        ):
            keep[i] = True
            kept.append(i)
    if hasattr(masks, "device"):
        import torch

        return torch.from_numpy(keep).to(masks.device)
    return keep


def get() -> Segmenter:
    global _default
    with _default_lock:
        if _default is None:
            _default = Segmenter()
        return _default


def segment(img: np.ndarray, camera: str = "head", conf: float | None = None) -> list[Instance]:
    return get().segment(img, camera=camera, conf=conf)


_prefetch: threading.Thread | None = None
prefetch_info: dict = {}


def _warm_files(paths, workers: int = 1) -> int:
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seg-warm") as ex:
            return sum(ex.map(lambda p: _warm_files([p]), list(paths)))
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    n += len(b)
        except OSError:
            pass
    return n


def _package_dir(name: str) -> Path | None:
    import importlib.util

    spec = importlib.util.find_spec(name)
    locs = list(spec.submodule_search_locations or []) if spec is not None else []
    return Path(locs[0]) if locs else None


def _prefetch_files() -> list:
    out = []
    torch_dir, nvidia_dir = _package_dir("torch"), _package_dir("nvidia")
    if torch_dir is not None:
        out += sorted((torch_dir / "lib").glob("*.so*"))
    if nvidia_dir is not None:
        out += sorted(nvidia_dir.glob("*/lib/*.so*"))
    for pkg, pattern in (("scipy", "*.so"), ("ultralytics", "*.pyc")):
        d = _package_dir(pkg)
        if d is not None:
            out += sorted(d.rglob(pattern))
    return out


def _prefetch_work(workers: int = 1) -> None:
    t0 = time.perf_counter()
    try:
        files = _prefetch_files()
        prefetch_info["files"] = len(files)
        prefetch_info["bytes"] = _warm_files(files, workers)
        prefetch_info["workers"] = workers
        models.path(MODEL_NAME)
    except BaseException as e:
        prefetch_info["error"] = f"{type(e).__name__}: {e}"
    prefetch_info["seconds"] = round(time.perf_counter() - t0, 2)


def prefetch(workers: int = 1) -> None:
    global _prefetch
    with _default_lock:
        if _prefetch is None and _default is None:
            _prefetch = threading.Thread(target=_prefetch_work, args=(int(workers),), name="seg-prefetch", daemon=True)
            _prefetch.start()


def prefetched(timeout: float = 0.0) -> bool:
    t = _prefetch
    if t is not None and timeout > 0:
        t.join(timeout)
    return t is None or not t.is_alive()


def preload() -> threading.Thread:
    try:
        import torch
    except Exception:
        pass
    t = threading.Thread(target=get, name="seg-preload", daemon=True)
    t.start()
    return t


def of_category(instances: list[Instance], *categories: str) -> list[Instance]:
    want = set(categories)
    return [i for i in instances if i.category in want]


def fit_pose(fitter: Callable, cam, inst: Instance, ignore_others: list[Instance] | None = None, **kw):
    if ignore_others and "ignore" not in kw:
        try:
            import inspect

            if "ignore" in inspect.signature(fitter).parameters:
                ign = np.zeros_like(inst.mask)
                for o in ignore_others:
                    if o is not inst:
                        ign |= o.mask
                kw["ignore"] = ign & ~inst.mask
        except (TypeError, ValueError):
            pass
    return fitter(cam, inst.mask, **kw)
