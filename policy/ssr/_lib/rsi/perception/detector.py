from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import models as M

MODEL = "coco_maskrcnn_v2"
COCO = {
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "bowl": 51,
    "dining table": 67,
    "sink": 81,
    "vase": 86,
    "frisbee": 34,
    "sports ball": 37,
    "donut": 60,
    "cake": 61,
    "clock": 85,
    "toilet": 70,
}
_lock = threading.Lock()
_model = None
_device = None


@dataclass
class Instance:
    label: int
    score: float
    box: np.ndarray
    prob: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.prob >= 0.5


def _load():
    global _model, _device
    if _model is not None:
        return _model
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    weights = M.path(MODEL)
    m = maskrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=91, box_score_thresh=0.05, box_detections_per_img=100
    )
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = m.to(dev)
    _device = dev
    return _model


def _to_cpu() -> bool:
    global _model, _device
    try:
        import torch

        if _model is None or _device is None or _device.type == "cpu":
            return False
        _model = _model.to("cpu")
        _device = torch.device("cpu")
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def detect(img: np.ndarray, labels=None, min_score: float = 0.05) -> list[Instance]:
    with _lock:
        m = _load()
        for attempt in range(2):
            try:
                import torch

                x = torch.from_numpy(np.ascontiguousarray(img)).to(_device).permute(2, 0, 1).float() / 255.0
                with torch.inference_mode():
                    out = m([x])[0]
                boxes = out["boxes"].float().cpu().numpy()
                scores = out["scores"].float().cpu().numpy()
                labs = out["labels"].cpu().numpy()
                probs = out["masks"][:, 0].float().cpu().numpy()
                break
            except Exception:
                if attempt == 0 and _to_cpu():
                    m = _model
                    continue
                raise
    res = []
    for b, s, lab, p in zip(boxes, scores, labs, probs):
        if s < min_score or (labels is not None and int(lab) not in labels):
            continue
        res.append(Instance(int(lab), float(s), b, p.astype(np.float32)))
    res.sort(key=lambda r: -r.score)
    return res


EMBED_SIZE = 128
EMBED_CTX = 0.3
EMBED_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
EMBED_STD = np.array([0.229, 0.224, 0.225], np.float32)
EMBED_LAYERS = {"layer2": "1", "layer3": "2", "layer4": "3"}


def object_crops(img: np.ndarray, masks, size: int = EMBED_SIZE, ctx: float = EMBED_CTX):
    import cv2

    H, W = img.shape[:2]
    out_i, out_m = [], []
    for m in masks:
        vs, us = np.nonzero(m)
        if len(vs) == 0:
            vs, us = np.array([H // 2]), np.array([W // 2])
        u0, u1, v0, v1 = int(us.min()), int(us.max()) + 1, int(vs.min()), int(vs.max()) + 1
        s = max(u1 - u0, v1 - v0) * (1.0 + ctx)
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        a0, a1 = int(round(cu - s / 2)), int(round(cu + s / 2))
        b0, b1 = int(round(cv - s / 2)), int(round(cv + s / 2))
        a1, b1 = max(a1, a0 + 1), max(b1, b0 + 1)
        pad = max(0, -a0, -b0, a1 - W, b1 - H)
        ip = cv2.copyMakeBorder(np.ascontiguousarray(img), pad, pad, pad, pad, cv2.BORDER_REFLECT) if pad else img
        mp = np.pad(np.asarray(m, bool), pad) if pad else np.asarray(m, bool)
        ci = ip[b0 + pad : b1 + pad, a0 + pad : a1 + pad]
        cm = mp[b0 + pad : b1 + pad, a0 + pad : a1 + pad]
        out_i.append(cv2.resize(np.ascontiguousarray(ci), (size, size), interpolation=cv2.INTER_LINEAR))
        out_m.append(cv2.resize(cm.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0)
    return out_i, out_m


def embed(img: np.ndarray, masks, layer: str = "layer3") -> np.ndarray:
    masks = list(masks)
    if not masks:
        return np.zeros((0, 1), np.float32)
    ci, cm = object_crops(img, masks)
    x = (np.stack(ci).astype(np.float32) / 255.0 - EMBED_MEAN) / EMBED_STD
    mk = np.stack(cm).astype(np.float32)[:, None]
    key = EMBED_LAYERS[layer]
    with _lock:
        m = _load()
        for attempt in range(2):
            try:
                import torch
                import torch.nn.functional as F

                xt = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to(_device)
                with torch.inference_mode():
                    f = m.backbone.body(xt)[key].float()
                    mm = F.interpolate(torch.from_numpy(mk).to(_device), size=f.shape[-2:], mode="area")
                    v = (f * mm).sum(dim=(2, 3)) / mm.sum(dim=(2, 3)).clamp(min=0.001)
                    out = F.normalize(v, dim=1).cpu().numpy().astype(np.float32)
                break
            except Exception:
                if attempt == 0 and _to_cpu():
                    m = _model
                    continue
                raise
    return out
