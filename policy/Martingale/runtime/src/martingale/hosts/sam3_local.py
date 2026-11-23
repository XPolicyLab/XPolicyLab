# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import hashlib
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segmented:
    """One mask, or none: what a SAM3 client returns."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    reason: str | None = None


def _array(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else value.detach().float().cpu().numpy()


def select_top(masks: Any, scores: Any, boxes: Any, min_score: float) -> Segmented:
    """The best-scoring candidate, if it reaches ``min_score`` and its mask is not empty."""

    if masks is None or scores is None:
        return Segmented(False, reason="SAM3 returned no candidate")
    masks, scores = _array(masks), _array(scores).reshape(-1)
    if masks.size == 0 or scores.size == 0:
        return Segmented(False, reason="SAM3 returned no candidate")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"unexpected SAM3 candidate shapes: masks={masks.shape}, scores={scores.shape}"
        )
    index = int(np.argmax(scores))
    score = float(scores[index])
    box: list[float] | None = None
    if boxes is not None:
        boxes = _array(boxes)
        if boxes.ndim >= 2 and index < boxes.shape[0]:
            box = [float(value) for value in boxes[index].reshape(-1)[:4]]
    if score < min_score:
        return Segmented(
            False,
            score=score,
            box=box,
            reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
        )
    mask = np.asarray(masks[index]) > 0
    if mask.ndim != 2 or not mask.any():
        return Segmented(False, score=score, box=box, reason="SAM3 returned an empty mask")
    return Segmented(True, score=score, box=box, mask=mask)


class LocalSam3:
    """SAM3 loaded on one GPU of this process."""

    def __init__(self, checkpoint: str | None = None, cuda_device: int | None = None) -> None:
        checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH")
        if not checkpoint:
            raise RuntimeError("SAM3_CHECKPOINT_PATH is not set")
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        if cuda_device is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "SAM3 in this process needs torch and the sam3 package "
                "(https://github.com/facebookresearch/sam3)"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 in this process needs a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=str(resolved),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        self._torch = torch
        self._model = model
        self._processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        self._lock = threading.Lock()
        self._digest: str | None = None
        self._state: dict[str, Any] | None = None

    def _autocast(self) -> Any:
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for(self, pixels: np.ndarray) -> dict[str, Any]:
        """The image features, computed once for as many prompts as are put to one picture."""

        from PIL import Image

        digest = hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest != self._digest or self._state is None:
            with self._autocast():
                self._state = self._processor.set_image(Image.fromarray(pixels, mode="RGB"))
            self._digest = digest
        return self._state

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Segmented:
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        if has_text == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_score must be a number, got {min_score!r}") from exc
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        with self._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = self._state_for(pixels)
            if has_text:
                assert text_prompt is not None
                with self._autocast():
                    out = self._processor.set_text_prompt(prompt=text_prompt.strip(), state=state)
                return select_top(out.get("masks"), out.get("scores"), out.get("boxes"), min_score)
            assert point is not None
            row, col = int(point[0]), int(point[1])
            height, width = int(state["original_height"]), int(state["original_width"])
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {list(point)} is outside image shape [{height}, {width}]"
                )
            if getattr(self._model, "inst_interactive_predictor", None) is None:
                raise RuntimeError("SAM3 instance interactivity is not enabled")
            with self._autocast():
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=np.asarray([[col, row]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int64),
                    multimask_output=True,
                )
            return select_top(masks, scores, None, min_score)


_SHARED: dict[tuple[str, str], LocalSam3] = {}
_SHARED_LOCK = threading.Lock()


def shared(checkpoint: str | None = None, cuda_device: int | None = None) -> LocalSam3:
    """The process's one :class:`LocalSam3` for a checkpoint and the cards it may see.

    A policy server plays one episode after another: the model is loaded for the first and
    serves the rest (a second copy would cost its load time and the card its memory).
    """

    path = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH") or ""
    key = (path, os.environ.get("CUDA_VISIBLE_DEVICES") or str(cuda_device))
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = LocalSam3(checkpoint, cuda_device)
        return _SHARED[key]


__all__ = ["LocalSam3", "Segmented", "select_top", "shared"]
