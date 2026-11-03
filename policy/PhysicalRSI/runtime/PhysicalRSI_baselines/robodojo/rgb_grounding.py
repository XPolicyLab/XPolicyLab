"""Local open-vocabulary RGB boxes; estimates, not complete scene geometry.

Queries and thresholds are frozen task configuration. Separate query inference
binds each box to an explicit query ID without trusting decoded free-form labels.
This backend belongs in a supervised worker for interruptible GPU execution.
"""

import math
import time
from pathlib import Path

import numpy as np

from PhysicalRSI_core.infra.storage import identifier


def project_boxes(boxes, scores, *, query, width, height, threshold, maximum):
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    if (
        boxes.shape != (len(scores), 4)
        or scores.ndim != 1
        or boxes.dtype.kind not in "ifu"
        or scores.dtype.kind not in "ifu"
        or not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        raise ValueError("Invalid grounded boxes or scores")
    result = []
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x0, y0, x1, y1 = map(float, box)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Grounded box is empty or reversed")
        x0, x1 = max(0.0, x0), min(float(width - 1), x1)
        y0, y1 = max(0.0, y0), min(float(height - 1), y1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Grounded box outside image")
        result.append(dict(query=query, box_xyxy=[x0, y0, x1, y1], score=float(score)))
    if len(result) > maximum:
        raise ValueError("Grounded detections exceed configured capacity")
    return sorted(result, key=lambda row: (-row["score"], row["box_xyxy"]))


class RGBGrounding:
    def __init__(
        self,
        *,
        model_path,
        queries,
        device="cuda:0",
        threshold=0.3,
        text_threshold=0.1,
        maximum=32,
    ):
        if (
            not isinstance(queries, dict)
            or not 1 <= len(queries) <= 8
            or any(
                not isinstance(text, str) or not text.strip() or len(text) > 128
                for text in queries.values()
            )
        ):
            raise ValueError("One to eight bounded grounding queries required")
        for name in queries:
            identifier(name)
        if (
            not math.isfinite(threshold)
            or not 0 < threshold <= 1
            or not math.isfinite(text_threshold)
            or not 0 < text_threshold <= 1
            or type(maximum) is not int
            or not 1 <= maximum <= 256
        ):
            raise ValueError("Invalid grounding thresholds or capacity")
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        root = Path(model_path).resolve(strict=True)
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                root, local_files_only=True
            )
            .to(self.device)
            .eval()
        )
        self.queries = dict(queries)
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.maximum = maximum
        self.closed = False

    @staticmethod
    def _check(deadline):
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("RGB grounding deadline reached")

    def predict(self, rgb, *, deadline):
        if self.closed:
            raise RuntimeError("RGB grounding model is closed")
        self._check(deadline)
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 2
            or rgb.shape[0] * rgb.shape[1] > 1024 * 1024
        ):
            raise ValueError("Bounded uint8 RGB image required")
        result = []
        for name, text in self.queries.items():
            self._check(deadline)
            inputs = self.processor(
                images=rgb, text=text.strip().rstrip(".") + ".", return_tensors="pt"
            ).to(self.device)
            self._check(deadline)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            self._check(deadline)
            predictions = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.threshold,
                text_threshold=self.text_threshold,
                target_sizes=[rgb.shape[:2]],
            )[0]
            result.extend(
                project_boxes(
                    predictions["boxes"].detach().cpu().numpy(),
                    predictions["scores"].detach().cpu().numpy(),
                    query=name,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    threshold=self.threshold,
                    maximum=self.maximum - len(result),
                )
            )
        self._check(deadline)
        return result

    def close(self):
        if not self.closed:
            del self.model
            del self.processor
            self.closed = True
