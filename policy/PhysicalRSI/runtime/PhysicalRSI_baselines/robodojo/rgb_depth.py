"""Local pretrained RGB depth inference for a supervised perception worker.

Returned values stay in the model's depth domain. PlaneDepthAlignment explicitly
converts that domain to axial metres before geometry consumes it. This backend
must run in an owned worker for enforceable interruption of GPU calls.
"""

import math
import time
from pathlib import Path

import numpy as np


class RGBDepth:
    def __init__(self, *, model_path, device="cuda:0"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        root = Path(model_path).resolve(strict=True)
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            root, local_files_only=True, use_fast=False
        )
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(root, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.closed = False

    @staticmethod
    def _check(deadline):
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("RGB depth inference deadline reached")

    def predict(self, rgb, *, deadline):
        if self.closed:
            raise RuntimeError("RGB depth model is closed")
        self._check(deadline)
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 1
            or rgb.shape[0] * rgb.shape[1] > 4096 * 4096
        ):
            raise ValueError("Bounded uint8 RGB image required")
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        self._check(deadline)
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            result = prediction.float().cpu().numpy().copy()
        self._check(deadline)
        if result.shape != rgb.shape[:2] or not np.isfinite(result).all():
            raise ValueError("Depth model returned invalid image")
        return result

    def close(self):
        if not self.closed:
            del self.model
            del self.processor
            self.closed = True
