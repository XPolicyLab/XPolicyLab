from __future__ import annotations

import numpy as np

PATCH_M = 0.26, 0.13
PATCH_PX = 104, 52


def patch(img: np.ndarray, cam, xyz, d_xy) -> np.ndarray:
    import cv2

    p0 = np.asarray(xyz, dtype=float).reshape(3)
    d = np.array([float(d_xy[0]), float(d_xy[1]), 0.0])
    d /= max(1e-09, float(np.linalg.norm(d)))
    n = np.array([-d[1], d[0], 0.0])
    w, h = PATCH_PX
    k = w / PATCH_M[0]
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    src = np.stack([p0, p0 + 0.04 * d, p0 - 0.04 * d, p0 + 0.04 * n, p0 - 0.04 * n])
    dst = np.stack([c, c + (0.04 * k, 0.0), c - (0.04 * k, 0.0), c - (0.0, 0.04 * k), c + (0.0, 0.04 * k)])
    uv, _ = cam.project(src)
    A = np.linalg.lstsq(np.c_[uv, np.ones(len(uv))], dst, rcond=None)[0].T
    return cv2.warpAffine(
        np.ascontiguousarray(img[..., :3]), A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def net():
    import torch.nn as nn

    def block(i, o):
        return [
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        ]

    return nn.Sequential(
        *block(3, 16),
        nn.MaxPool2d(2),
        *block(16, 32),
        nn.MaxPool2d(2),
        *block(32, 64),
        nn.MaxPool2d(2),
        *block(64, 96),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(96, 1),
    )


def to_tensor(patches: np.ndarray):
    import torch

    x = torch.from_numpy(np.ascontiguousarray(patches)).float().div_(255.0)
    return (x.permute(0, 3, 1, 2) - 0.5) / 0.25


MODELS = {"car": "pack_car_front", "shoe": "pack_shoe_front"}
_nets: dict = {}


def load(category: str):
    if category not in _nets:
        import torch
        from ...perception import models

        m = net()
        m.load_state_dict(torch.load(models.path(MODELS[category]), map_location="cpu"))
        m.eval()
        _nets[category] = m
    return _nets[category]


def front_score(img: np.ndarray, cam, category: str, xyz, d_xy) -> float:
    import torch

    p = patch(img, cam, xyz, d_xy)
    with torch.no_grad():
        z = load(category)(to_tensor(np.stack([p, p[::-1, ::-1]]))).squeeze(1).numpy()
    return float(z[0] - z[1])
