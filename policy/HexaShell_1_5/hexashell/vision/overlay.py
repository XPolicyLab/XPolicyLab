"""Pictures for the model: coordinate grid, marked points, outlines, enlarged regions, side by side.

Every label drawn is in the full picture's pixel coordinates, also on an enlarged crop, so a
point read off any picture can be passed to the tools as it is.  Drawing is on RGB arrays.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ..errors import Refused

GRID_COLOR = (255, 255, 0)
LABEL_COLOR = (255, 255, 255)
MARK_COLORS = [(255, 40, 40), (40, 220, 40), (60, 140, 255), (255, 170, 0), (230, 60, 230), (0, 230, 230)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _nice_step(span_px: float, lines: int) -> int:
    raw = span_px / max(1, lines)
    for step in (2, 5, 10, 20, 25, 40, 50, 80, 100, 200):
        if step >= raw:
            return step
    return 200


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.4, color=LABEL_COLOR) -> None:
    x, y = org
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x = int(min(max(0, x), img.shape[1] - w - 1))
    y = int(min(max(h + 1, y), img.shape[0] - base - 1))
    cv2.rectangle(img, (x - 1, y - h - 1), (x + w + 1, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


class View:
    """A region of one camera's picture, enlarged by ``zoom``: maps full-picture pixels to it."""

    def __init__(self, image: np.ndarray, crop=None, zoom=None):
        h, w = image.shape[:2]
        if crop is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            try:
                x0, y0, x1, y1 = (float(v) for v in crop)
            except (TypeError, ValueError):
                raise Refused("crop must be (x0, y0, x1, y1) in pixels") from None
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
            x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise Refused("crop is too small (at least 8 x 8 pixels inside the picture)")
        if zoom is None:
            zoom = 1.0 if crop is None else min(6.0, max(1.0, 640.0 / max(x1 - x0, 1), 480.0 / max(y1 - y0, 1)))
        try:
            zoom = float(zoom)
        except (TypeError, ValueError):
            raise Refused("zoom must be a number between 0.25 and 8") from None
        if not (math.isfinite(zoom) and zoom > 0):
            raise Refused("zoom must be a number between 0.25 and 8")
        # beyond 8 a pixel shows no more: cap it rather than lose the cell over it
        zoom = min(max(zoom, 0.25), 8.0)
        # keep the result within ~1600 px on its long side
        zoom = min(zoom, 1600.0 / max(x1 - x0, y1 - y0))
        self.x0, self.y0, self.x1, self.y1, self.zoom = x0, y0, x1, y1, zoom
        part = image[y0:y1, x0:x1]
        size = (max(1, int(round((x1 - x0) * zoom))), max(1, int(round((y1 - y0) * zoom))))
        interp = cv2.INTER_NEAREST if zoom >= 3 else cv2.INTER_LINEAR
        self.image = np.ascontiguousarray(cv2.resize(part, size, interpolation=interp) if zoom != 1.0 else part.copy())

    def to_view(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.x0 + 0.5) * self.zoom - 0.5)), int(round((y - self.y0 + 0.5) * self.zoom - 0.5)))

    def inside(self, x: float, y: float) -> bool:
        return self.x0 - 0.5 <= x <= self.x1 - 0.5 and self.y0 - 0.5 <= y <= self.y1 - 0.5

    def grid(self) -> None:
        img = self.image
        span = max(self.x1 - self.x0, self.y1 - self.y0)
        step = _nice_step(span, 12)
        overlay = img.copy()
        xs = range(int(math.ceil(self.x0 / step) * step), self.x1, step)
        ys = range(int(math.ceil(self.y0 / step) * step), self.y1, step)
        for x in xs:
            vx, _ = self.to_view(x, 0)
            cv2.line(overlay, (vx, 0), (vx, img.shape[0] - 1), GRID_COLOR, 1)
        for y in ys:
            _, vy = self.to_view(0, y)
            cv2.line(overlay, (0, vy), (img.shape[1] - 1, vy), GRID_COLOR, 1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        label_every = 2 if len(xs) > 10 or len(ys) > 8 else 1
        for i, x in enumerate(xs):
            if i % label_every == 0:
                vx, _ = self.to_view(x, 0)
                _text(img, str(x), (vx + 2, 11))
                _text(img, str(x), (vx + 2, img.shape[0] - 3))
        for i, y in enumerate(ys):
            if i % label_every == 0:
                _, vy = self.to_view(0, y)
                _text(img, str(y), (2, vy - 2))
                _text(img, str(y), (img.shape[1] - 30, vy - 2))

    def mark(self, x: float, y: float, label: str | None, color) -> bool:
        if not self.inside(x, y):
            return False
        vx, vy = self.to_view(x, y)
        r = 7
        cv2.circle(self.image, (vx, vy), r, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(self.image, (vx, vy), r, color, 1, cv2.LINE_AA)
        cv2.line(self.image, (vx - r - 4, vy), (vx - 2, vy), color, 1)
        cv2.line(self.image, (vx + 2, vy), (vx + r + 4, vy), color, 1)
        cv2.line(self.image, (vx, vy - r - 4), (vx, vy - 2), color, 1)
        cv2.line(self.image, (vx, vy + 2), (vx, vy + r + 4), color, 1)
        text = f"{label} ({x:.0f},{y:.0f})" if label else f"({x:.0f},{y:.0f})"
        _text(self.image, text, (vx + r + 3, vy - r), 0.42, color)
        return True

    def outline(self, mask: np.ndarray, label: str | None, color) -> bool:
        part = mask[self.y0:self.y1, self.x0:self.x1].astype(np.uint8)
        if not part.any():
            return False
        if self.zoom != 1.0:
            part = cv2.resize(part, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tint = self.image.copy()
        tint[part > 0] = color
        cv2.addWeighted(tint, 0.25, self.image, 0.75, 0, dst=self.image)
        cv2.drawContours(self.image, contours, -1, color, 2 if self.zoom < 3 else 3, cv2.LINE_AA)
        if label:
            ys, xs = np.nonzero(part)
            _text(self.image, label, (int(xs.min()), max(12, int(ys.min()) - 3)), 0.45, color)
        return True


def render(image: np.ndarray, *, cam: str, grid: bool = False, marks: Iterable | None = None,
           outlines: list[tuple[np.ndarray, str]] | None = None, crop=None, zoom=None,
           project: Callable[[Any], tuple[float, float] | None] | None = None) -> tuple[np.ndarray, list[str]]:
    """One annotated picture and notes about what could not be drawn."""
    view = View(image, crop, zoom)
    notes: list[str] = []
    for i, (mask, label) in enumerate(outlines or []):
        if not view.outline(mask, label, MARK_COLORS[i % len(MARK_COLORS)]):
            notes.append(f"{label or 'outline'}: not in this {cam} picture" + (" region" if crop else ""))
    if grid:
        view.grid()
    for i, mark in enumerate(marks or []):
        color = MARK_COLORS[i % len(MARK_COLORS)]
        x, y, label = parse_mark(mark, project)
        if x is None:
            notes.append(f"mark {label or i}: {y}")
            continue
        if not view.mark(x, y, label, color):
            notes.append(f"mark {label or i} at ({x:.0f}, {y:.0f}) is outside the {cam} picture"
                         + (" region" if crop else ""))
    title = cam if crop is None else f"{cam} x{view.zoom:.1f} [{view.x0}..{view.x1}]x[{view.y0}..{view.y1}]"
    _text(view.image, title, (4, view.image.shape[0] - 18 if grid else view.image.shape[0] - 4), 0.5, (255, 255, 0))
    return view.image, notes


def parse_mark(mark, project) -> tuple[float | None, Any, str | None]:
    """(x, y, label) of a mark; (None, reason, label) when it cannot be placed."""
    label = None
    if isinstance(mark, dict):
        if "world" in mark:
            mark = ("world", mark["world"], mark.get("label"))
        else:
            mark = (mark.get("x"), mark.get("y"), mark.get("label"))
    if isinstance(mark, (list, tuple)) and len(mark) >= 2 and mark[0] == "world":
        label = str(mark[2]) if len(mark) > 2 and mark[2] is not None else None
        if project is None:
            return None, "world points need a camera", label
        try:
            px = project(mark[1])
        except (TypeError, ValueError) as exc:
            return None, f"bad world point ({exc})", label
        if px is None:
            return None, "the world point is out of view of this camera", label
        return float(px[0]), float(px[1]), label
    try:
        x, y = float(mark[0]), float(mark[1])
    except (TypeError, ValueError, IndexError):
        return None, "a mark is (x, y), (x, y, 'label') or ('world', (x, y, z), 'label')", None
    if len(mark) > 2 and mark[2] is not None:
        label = str(mark[2])
    return x, y, label


def side_by_side(images: list[np.ndarray], gap: int = 6) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)
            im = np.vstack([im, pad])
        parts.append(im)
        if i < len(images) - 1:
            parts.append(np.full((h, gap, 3), 255, np.uint8))
    return np.hstack(parts)


def jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """JPEG bytes of an RGB picture."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8), "RGB").save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()
