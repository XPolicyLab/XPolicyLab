"""Step 2e2: Own locate / back_project path for HarnessPolicyIncremental.

Calls geometry tools through the local colour toolkit (EpisodeEnv + HttpSam3).
"""

from __future__ import annotations

import math
from typing import Any

from .types import ObjectSight, SeenPoint, _tuple3


def _num(payload: dict[str, Any], key: str) -> float | None:
    return float(payload[key]) if payload.get(key) is not None else None


def locate(
    toolkit: Any,
    text_prompt: str,
    camera: str = "cam_head",
    min_score: float = 0.2,
) -> ObjectSight:
    prompt = str(text_prompt or "").strip()
    if not prompt:
        raise ValueError("text_prompt must not be empty")
    payload = toolkit.execute_tool(
        "locate",
        {"text_prompt": prompt, "camera": camera, "min_score": float(min_score)},
    )
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    centroid = payload.get("centroid_rc") or ()
    center = payload.get("center_xyz")
    note = payload.get("geometry_error") or payload.get("height_guess")
    return ObjectSight(
        found=bool(payload.get("found")),
        center_xyz=_tuple3(center) if center is not None else None,
        top_z=_num(payload, "top_z"),
        height_m=_num(payload, "height_m"),
        length_m=_num(payload, "length_m"),
        width_m=_num(payload, "width_m"),
        long_axis_deg=_num(payload, "long_axis_deg"),
        grasp_yaw=_num(payload, "grasp_yaw"),
        score=_num(payload, "score"),
        centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
        area_px=int(payload.get("area_px") or 0),
        camera=str(payload.get("camera", camera)),
        note=str(note) if note else None,
        views=int(payload.get("views") or 1),
    )


def back_project(toolkit: Any, row: int, col: int, camera: str = "cam_head") -> SeenPoint:
    payload = toolkit.execute_tool(
        "back_project",
        {"row": int(row), "col": int(col), "camera": camera},
    )
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    pixel = payload.get("pixel") or (int(row), int(col))
    return SeenPoint(
        pixel=(int(pixel[0]), int(pixel[1])),
        xyz=_tuple3(payload.get("world_xyz")),
        depth_m=float(payload.get("depth_m", math.nan)),
        camera=str(payload.get("camera", camera)),
        surface=str(payload.get("surface") or "table"),
    )
