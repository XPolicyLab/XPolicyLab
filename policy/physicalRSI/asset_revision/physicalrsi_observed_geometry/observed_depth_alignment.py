"""Align learned depth to a support plane measured from calibrated RGB views.

The caller supplies measured coefficients z=a*x+b*y+c, never an asset height.
A held-out pixel subset checks affine depth consistency, not absolute accuracy.
"""

import math
import time

import numpy as np


class ObservedPlaneDepthAlignment:
    def __init__(
        self,
        *,
        camera,
        plane,
        target,
        residual_m=0.015,
        minimum_fraction=0.7,
        trials=100,
        max_samples=4096,
    ):
        if target not in {"axial", "inverse_axial"}:
            raise ValueError("Declare axial or inverse_axial calibration target")
        if (
            np.asarray(plane).shape != (3,) or not np.isfinite(plane).all()
            or not math.isfinite(residual_m)
            or residual_m <= 0
            or not 0.5 < minimum_fraction <= 1
            or type(trials) is not int
            or not 1 <= trials <= 1000
            or type(max_samples) is not int
            or not 20 <= max_samples <= 65536
        ):
            raise ValueError("Invalid bounded depth calibration limits")
        self.camera, self.plane, self.target = camera, np.asarray(plane, float), target
        self.residual_m, self.minimum_fraction = residual_m, minimum_fraction
        self.trials, self.max_samples = trials, max_samples

    @staticmethod
    def _check(deadline):
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Depth alignment deadline reached")

    def _depth(self, x, coefficients):
        transformed = coefficients[0] * x + coefficients[1]
        if self.target == "inverse_axial":
            with np.errstate(divide="ignore", invalid="ignore"):
                transformed = 1 / transformed
        return transformed

    def align(self, prediction, plane_mask, *, deadline):
        self._check(deadline)
        prediction = np.asarray(prediction)
        mask = np.asarray(plane_mask)
        shape = (self.camera.height, self.camera.width)
        if (
            prediction.shape != shape
            or prediction.dtype.kind not in "ifu"
            or not np.isfinite(prediction).all()
            or mask.shape != shape
            or mask.dtype != bool
        ):
            raise ValueError(
                "Finite predicted depth and boolean plane mask must match camera"
            )
        pixels = np.column_stack(np.nonzero(mask))[:, ::-1]
        if len(pixels) < 20:
            raise ValueError("Insufficient visible plane pixels")
        if len(pixels) > self.max_samples:
            pixels = pixels[
                np.linspace(0, len(pixels) - 1, self.max_samples, dtype=int)
            ]
        origin = self.camera.transform[:3, 3]
        rays = self.camera.unproject(pixels, np.ones(len(pixels))) - origin
        a, b, c = self.plane
        denominator = rays[:, 2] - a*rays[:, 0] - b*rays[:, 1]
        if np.any(np.abs(denominator) < 1e-9):
            raise ValueError("Support ray is parallel to observed plane")
        expected = (c-origin[2]+a*origin[0]+b*origin[1])/denominator
        if not np.isfinite(expected).all() or np.any(expected <= 0):
            raise ValueError("Observed support plane is behind camera")
        values = prediction[pixels[:, 1], pixels[:, 0]].astype(float)
        holdout = np.arange(len(pixels)) % 4 == 0
        x, y = values[~holdout], expected[~holdout]
        if np.ptp(x) < 1e-8 or np.ptp(y) < 1e-5:
            raise ValueError("Depth/plane range is degenerate for affine alignment")
        response = y if self.target == "axial" else 1 / y
        rng = np.random.default_rng(0)
        best = None
        for _ in range(self.trials):
            self._check(deadline)
            a, b = rng.choice(len(x), size=2, replace=False)
            if abs(x[a] - x[b]) < 1e-8:
                continue
            scale = (response[a] - response[b]) / (x[a] - x[b])
            coefficients = [scale, response[a] - scale * x[a]]
            depth = self._depth(x, coefficients)
            residual = np.abs(depth - y)
            inliers = np.isfinite(depth) & (depth > 0) & (residual <= self.residual_m)
            score = (
                int(inliers.sum()),
                -float(np.median(residual[inliers])) if inliers.any() else -math.inf,
            )
            if best is None or score > best[0]:
                best = (score, inliers)
        if best is None or best[1].mean() < self.minimum_fraction:
            raise ValueError("Insufficient plane consensus for depth alignment")
        inliers = best[1]
        design = np.column_stack((x[inliers], np.ones(inliers.sum())))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, response[inliers], rcond=None
        )
        if rank != 2 or not np.isfinite(coefficients).all():
            raise ValueError("Degenerate fitted depth alignment")
        report = {}
        for name, selected in [("fit", ~holdout), ("holdout", holdout)]:
            predicted = self._depth(values[selected], coefficients)
            error = np.abs(predicted - expected[selected])
            valid = (
                np.isfinite(predicted) & (predicted > 0) & (error <= self.residual_m)
            )
            if valid.mean() < self.minimum_fraction:
                raise ValueError(name + " plane residual failed depth alignment")
            report[name] = dict(
                samples=int(selected.sum()),
                inlier_fraction=float(valid.mean()),
                median_residual_m=float(np.median(error[valid])),
            )
        self._check(deadline)
        depth = self._depth(prediction.astype(float), coefficients)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("Aligned image contains invalid axial depth")
        self._check(deadline)
        return depth, dict(
            target=self.target,
            scale=float(coefficients[0]),
            offset=float(coefficients[1]),
            observed_plane=self.plane.tolist(),
            residual_m=self.residual_m,
            **report,
        )
