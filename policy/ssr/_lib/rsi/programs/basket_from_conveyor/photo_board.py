from __future__ import annotations

from ... import ASSETS

from dataclasses import dataclass

import numpy as np

DATA = ASSETS / "board_photos"
THUMB = 24, 36
WINDOW = 250, 380, 260, 380
WHITE_MIN = 170
WHITE_SPREAD = 40
MIN_AREA = 300


@dataclass
class BoardReading:
    asset: str
    score: float
    margin: float
    bbox: tuple
    scores: dict


_REFS = None


def _refs():
    global _REFS
    if _REFS is None:
        _REFS = assets(), [_norm(t) for t in thumbs()]
    return _REFS


def assets() -> list[str]:
    import json

    return list(json.loads(DATA.with_suffix(".json").read_text())["assets"])


def thumbs() -> list[np.ndarray]:
    from PIL import Image

    atlas = np.asarray(Image.open(DATA.with_suffix(".png")).convert("RGB"))
    w = THUMB[0]
    return [atlas[:, i * w : (i + 1) * w] for i in range(atlas.shape[1] // w)]


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a = a - a.mean(axis=(0, 1), keepdims=True)
    return a / (np.linalg.norm(a) + 1e-06)


def read_board(head_rgb: np.ndarray) -> BoardReading | None:
    import cv2

    v0, v1, u0, u1 = WINDOW
    win = np.asarray(head_rgb)[v0:v1, u0:u1].astype(np.int16)
    white = (win.min(axis=2) > WHITE_MIN) & (win.max(axis=2) - win.min(axis=2) < WHITE_SPREAD)
    n, _, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, area = (int(t) for t in st[k])
    if area < MIN_AREA:
        return None
    crop = np.ascontiguousarray(win[y : y + h, x : x + w].astype(np.uint8))
    thumb = cv2.resize(crop, THUMB, interpolation=cv2.INTER_AREA)
    c = _norm(thumb)
    assets, refs = _refs()
    sc = np.array([float((c * r).sum()) for r in refs])
    order = np.argsort(sc)[::-1]
    return BoardReading(
        asset=assets[order[0]],
        score=float(sc[order[0]]),
        margin=float(sc[order[0]] - sc[order[1]]),
        bbox=(u0 + x, v0 + y, u0 + x + w - 1, v0 + y + h - 1),
        scores={a: float(s) for a, s in zip(assets, sc)},
    )
