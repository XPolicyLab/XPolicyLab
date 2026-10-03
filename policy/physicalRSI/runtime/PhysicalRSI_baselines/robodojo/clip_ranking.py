"""CLIP text/appearance ranking of observable grounding candidates.

Appearance references are explicit caller-supplied memory, never loaded from
historical task runs. Scores are ranking signals, not calibrated probabilities.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from .rgb_grounding import project_boxes


class AmbiguousGrounding(ValueError):
    pass


def rank_scores(rows, text_scores, appearance_scores=None):
    if len(rows) != len(text_scores) or (
        appearance_scores is not None and len(rows) != len(appearance_scores)
    ):
        raise ValueError("Candidate score counts differ")
    result = []
    for i, row in enumerate(rows):
        text = float(text_scores[i])
        appearance = None if appearance_scores is None else appearance_scores[i]
        if not math.isfinite(text) or not -1.00001 <= text <= 1.00001:
            raise ValueError("Invalid CLIP cosine score")
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("Invalid grounding confidence")
        rank = text
        if appearance is not None:
            appearance = float(appearance)
            if not math.isfinite(appearance) or not -1.00001 <= appearance <= 1.00001:
                raise ValueError("Invalid appearance cosine score")
            rank += appearance + 0.5 * row["score"]
        result.append(
            dict(
                deepcopy(row),
                clip_score=text,
                appearance_score=appearance,
                rank_score=rank,
            )
        )
    return sorted(
        result, key=lambda row: (row["query"], -row["rank_score"], row["box_xyxy"])
    )


def select_unique(rows, *, query, minimum_score, minimum_margin, duplicate_iou=0.7):
    if (
        not math.isfinite(minimum_score)
        or not math.isfinite(minimum_margin)
        or minimum_margin <= 0
        or not 0 < duplicate_iou <= 1
    ):
        raise ValueError("Explicit finite selection thresholds required")
    candidates = sorted(
        [row for row in rows if row["query"] == query],
        key=lambda row: -row["rank_score"],
    )
    if (
        not candidates
        or not math.isfinite(candidates[0]["rank_score"])
        or candidates[0]["rank_score"] < minimum_score
    ):
        raise AmbiguousGrounding("No candidate meets the configured score")
    first = candidates[0]
    a = first["box_xyxy"]
    for other in candidates[1:]:
        b = other["box_xyxy"]
        area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
        if union <= 0:
            raise ValueError("Invalid candidate box area")
        if area / union >= duplicate_iou:
            continue
        if (
            not math.isfinite(other["rank_score"])
            or first["rank_score"] - other["rank_score"] < minimum_margin
        ):
            raise AmbiguousGrounding("Distinct candidates remain ambiguous")
    return deepcopy(first)


class CLIPRanking:
    def __init__(self, *, model_path, queries, references=None, device="cuda:0"):
        if (
            not isinstance(queries, dict)
            or not 1 <= len(queries) <= 8
            or any(
                not isinstance(t, str) or not t.strip() or len(t) > 128
                for t in queries.values()
            )
        ):
            raise ValueError("Bounded query descriptions required")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device)
        root = Path(model_path).resolve(strict=True)
        self.processor = CLIPProcessor.from_pretrained(
            root, local_files_only=True, use_fast=False
        )
        self.model = (
            CLIPModel.from_pretrained(root, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.queries = dict(queries)
        self.references = {}
        self.closed = False
        for name, values in (references or {}).items():
            array = np.asarray(values)
            if (
                name not in queries
                or array.ndim != 2
                or not 1 <= len(array) <= 16
                or array.shape[1] != self.model.config.projection_dim
                or array.dtype.kind not in "ifu"
                or not np.isfinite(array).all()
                or np.any(np.linalg.norm(array, axis=1) < 1e-8)
            ):
                raise ValueError("Invalid explicit appearance memory")
            tensor = torch.tensor(array, device=self.device, dtype=torch.float32)
            self.references[name] = tensor / tensor.norm(dim=1, keepdim=True)

    @staticmethod
    def _check(deadline):
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("CLIP ranking deadline reached")

    def rank(self, rgb, rows, *, deadline):
        if self.closed:
            raise RuntimeError("CLIP ranking model is closed")
        self._check(deadline)
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 2
            or rgb.shape[0] * rgb.shape[1] > 1024 * 1024
            or len(rows) > 32
        ):
            raise ValueError("Bounded RGB frame and candidate count required")
        if not rows:
            return []
        clean = []
        crops = []
        texts = []
        for row in rows:
            if row["query"] not in self.queries:
                raise ValueError("Unknown grounding query")
            candidate = project_boxes(
                [row["box_xyxy"]],
                [row["score"]],
                query=row["query"],
                width=rgb.shape[1],
                height=rgb.shape[0],
                threshold=0,
                maximum=1,
            )[0]
            clean.append(candidate)
            x0, y0, x1, y1 = candidate["box_xyxy"]
            crops.append(
                rgb[
                    math.floor(y0) : math.ceil(y1) + 1,
                    math.floor(x0) : math.ceil(x1) + 1,
                ].copy()
            )
            texts.append("a photo of " + self.queries[row["query"]])
        inputs = self.processor(
            images=crops, text=texts, padding=True, return_tensors="pt"
        ).to(self.device)
        self._check(deadline)
        with self.torch.inference_mode():
            visual = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            visual = visual / visual.norm(dim=1, keepdim=True)
            text = text / text.norm(dim=1, keepdim=True)
            scores = (visual * text).sum(dim=1).cpu().tolist()
            appearance = [
                None
                if row["query"] not in self.references
                else float((visual[i] @ self.references[row["query"]].T).max().cpu())
                for i, row in enumerate(clean)
            ]
            embeddings = visual.float().cpu().tolist()
        self._check(deadline)
        enriched = [
            dict(row, embedding=embedding) for row, embedding in zip(clean, embeddings)
        ]
        return rank_scores(enriched, scores, appearance)

    def close(self):
        if not self.closed:
            del self.model
            del self.processor
            self.references.clear()
            self.closed = True
