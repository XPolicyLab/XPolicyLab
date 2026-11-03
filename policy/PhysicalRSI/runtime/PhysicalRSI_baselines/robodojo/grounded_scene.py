"""RGB-derived geometry for explicitly plane-supported task objects.

Axis-aligned bounds cover observed foreground points and extend to the configured
support plane. They are estimates, not reconstructed hidden surfaces or proof of
complete collision coverage. No simulator object state is consumed.
"""

import math
import time

import numpy as np

from .rgb_grounding import project_boxes


def supported_bounds(
    camera,
    depth,
    box,
    *,
    plane_z,
    minimum_height,
    maximum_height,
    minimum_points,
    padding_m,
    deadline,
):
    if (
        not all(
            math.isfinite(v)
            for v in [plane_z, minimum_height, maximum_height, padding_m]
        )
        or not 0 < minimum_height < maximum_height
        or padding_m < 0
        or type(minimum_points) is not int
        or minimum_points < 3
    ):
        raise ValueError("Explicit supported-object geometry limits required")
    check_deadline(deadline)
    if (
        not isinstance(depth, np.ndarray)
        or depth.shape != (camera.height, camera.width)
        or depth.dtype.kind not in "ifu"
        or not np.isfinite(depth).all()
        or np.any(depth <= 0)
        or depth.size > 1024 * 1024
    ):
        raise ValueError("Calibrated finite positive axial depth required")
    clean = project_boxes(
        [box],
        [1.0],
        query="geometry",
        width=camera.width,
        height=camera.height,
        threshold=0,
        maximum=1,
    )[0]["box_xyxy"]
    x0, y0, x1, y1 = clean
    # Pixel centres strictly inside the detection rectangle; no background crop padding.
    yy, xx = np.mgrid[
        math.ceil(y0) : math.floor(y1) + 1, math.ceil(x0) : math.floor(x1) + 1
    ]
    if xx.size < minimum_points:
        raise ValueError("Insufficient pixels inside target box")
    points = camera.unproject(
        np.column_stack((xx.ravel(), yy.ravel())), depth[yy, xx].ravel()
    )
    heights = points[:, 2] - plane_z
    foreground = points[heights >= minimum_height]
    if len(foreground) < minimum_points:
        raise ValueError("Insufficient above-plane foreground")
    if np.any(foreground[:, 2] - plane_z > maximum_height):
        raise ValueError("Foreground exceeds configured supported-object height")
    low, high = foreground.min(axis=0), foreground.max(axis=0)
    low[2] = plane_z
    low -= padding_m
    high += padding_m
    if np.any(high <= low):
        raise ValueError("Degenerate foreground geometry")
    check_deadline(deadline)
    return dict(
        pose=[*((low + high) / 2).tolist(), 1.0, 0.0, 0.0, 0.0],
        dims=(high - low).tolist(),
    )


def check_deadline(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("Scene estimation deadline reached")


class GroundedScene:
    """Compose injected owned services and alignment on one observation image.

    The public plane mask is frozen calibration input. Detection boxes are removed
    from it before fitting to avoid using detected objects as support-plane pixels.
    The caller owns services and must wrap this estimator in ObservedObstacles for
    current-observation checks before and after inference.
    """

    def __init__(
        self,
        *,
        camera_name,
        camera,
        depth_service,
        target_service,
        alignment,
        plane_mask,
        geometry,
    ):
        if alignment.camera is not camera:
            raise ValueError("Depth and geometry must use the same calibrated camera")
        if (
            not isinstance(plane_mask, np.ndarray)
            or plane_mask.dtype != bool
            or plane_mask.shape != (camera.height, camera.width)
        ):
            raise ValueError("Explicit public plane mask required")
        if set(geometry) != {
            "minimum_height",
            "maximum_height",
            "minimum_points",
            "padding_m",
        }:
            raise ValueError(
                "Explicit supported-object geometry configuration required"
            )
        self.camera_name, self.camera = camera_name, camera
        self.depth_service, self.target_service = depth_service, target_service
        self.alignment, self.plane_mask = alignment, plane_mask.copy()
        self.geometry = dict(geometry)

    def __call__(self, *, snapshot, deadline):
        check_deadline(deadline)
        rgb = snapshot["vision"][self.camera_name]["color"].copy()
        targets = self.target_service.predict(rgb, deadline=deadline)
        if targets["unresolved"] or not targets["selected"]:
            raise ValueError("Unresolved target geometry; no partial scene accepted")
        prediction = self.depth_service.predict(rgb, deadline=deadline)
        mask = self.plane_mask.copy()
        # Exclude all hypotheses, including candidates rejected by CLIP ranking.
        for row in targets["candidates"]:
            x0, y0, x1, y1 = row["box_xyxy"]
            mask[
                math.floor(y0) : math.ceil(y1) + 1, math.floor(x0) : math.ceil(x1) + 1
            ] = False
        depth, _ = self.alignment.align(prediction, mask, deadline=deadline)
        objects = {
            name: supported_bounds(
                self.camera,
                depth,
                row["box_xyxy"],
                plane_z=self.alignment.plane_z,
                deadline=deadline,
                **self.geometry,
            )
            for name, row in targets["selected"].items()
        }
        check_deadline(deadline)
        return objects
