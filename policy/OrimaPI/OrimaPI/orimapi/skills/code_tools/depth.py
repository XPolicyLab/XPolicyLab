"""On-demand metric monocular estimates, never simulator depth or ground truth."""

from functools import lru_cache
import threading

import numpy as np

from . import depth_identity


class _Model:
    def __init__(self, path, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.to(device).eval()
        self.device, self.lock = device, threading.Lock()

    def predict(self, rgb):
        import torch

        with self.lock, torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            prediction = self.model(**inputs).predicted_depth
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            return depth[0, 0].float().cpu().numpy().copy()


@lru_cache(maxsize=2)
def _resident_model(path, device, weights_sha256):
    return _Model(path, device)


_load_lock = threading.Lock()


class DepthEstimator:
    def __init__(self, config):
        self.config = config
        self.identity = depth_identity(config)
        self.frames = {}
        self.scales = {}

    def reset(self):
        self.frames.clear()
        self.scales.clear()

    def estimate(self, frame_id, camera, image):
        cached = self.frames.get(camera)
        if cached is not None and cached[0] == frame_id:
            return self.describe(camera)
        if depth_identity(self.config) != self.identity:
            raise ValueError("Depth checkpoint changed during evaluation")
        with _load_lock:
            model = _resident_model(
                self.identity["model_path"],
                self.config["device"],
                self.identity["files"]["model.safetensors"],
            )
        prediction = np.asarray(model.predict(image), dtype=np.float32)
        if (
            prediction.shape != image.shape[:2]
            or not np.isfinite(prediction).all()
            or (prediction <= 0).any()
        ):
            raise ValueError("Depth model returned invalid positive metric estimates")
        self.frames[camera] = (frame_id, prediction)
        self.scales.pop(camera, None)
        return self.describe(camera)

    def describe(self, camera):
        frame_id, prediction = self.frames[camera]
        return {
            "frame_id": frame_id,
            "camera": camera,
            "shape": list(prediction.shape),
            "range_m": [float(prediction.min()), float(prediction.max())],
            "method": "monocular_metric_model",
            "units": "estimated_meters",
            "uncertainty": "Model has no calibrated per-pixel uncertainty; metric scale may be biased",
            "scale_calibration": self.scales.get(camera),
            "sensor_ground_truth": False,
        }

    def sample(self, camera, frame_id, pixel):
        value = self.frames.get(camera)
        if value is None or value[0] != frame_id:
            raise ValueError(
                "Depth estimate is stale or absent; call estimate_depth on the current image"
            )
        x, y = map(lambda v: int(round(v)), pixel)
        depth = value[1]
        if not 1 <= x < depth.shape[1] - 1 or not 1 <= y < depth.shape[0] - 1:
            raise ValueError("Depth sample is outside the image")
        patch = depth[y - 1 : y + 2, x - 1 : x + 2]
        return float(np.median(patch)) * self.scales.get(camera, {}).get("scale", 1.0)

    def calibrate(self, camera, frame_id, anchors):
        """Scale from explicitly selected points on the known public table plane."""
        if not 3 <= len(anchors) <= 12:
            raise ValueError("Select 3..12 well-separated visible table points")
        previous = self.scales.pop(camera, None)
        try:
            pixels = np.asarray([row[0] for row in anchors], dtype=float)
            if any(
                np.linalg.norm(pixels[i] - pixels[j]) < 12
                for i in range(len(pixels))
                for j in range(i)
            ):
                raise ValueError("Table anchors must be at least 12 pixels apart")
            ratios = np.asarray(
                [
                    expected / self.sample(camera, frame_id, pixel)
                    for pixel, expected in anchors
                ]
            )
            scale = float(np.median(ratios))
            disagreement = float(np.max(abs(ratios / scale - 1)))
            if (
                not np.isfinite(ratios).all()
                or not 0.1 <= scale <= 10
                or disagreement > 0.2
            ):
                raise ValueError(
                    "Table anchors do not support one consistent depth scale"
                )
            result = {
                "scale": scale,
                "maximum_relative_disagreement": disagreement,
                "assumption": "Selected visible features really lie on the public table plane",
                "scope": "global scale estimate only; local depth errors remain",
            }
            self.scales[camera] = result
            return result
        except Exception:
            if previous is not None:
                self.scales[camera] = previous
            raise
