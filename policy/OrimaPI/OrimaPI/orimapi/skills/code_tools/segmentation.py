"""Prompted current-RGB SAM proposals, never simulator masks or contact evidence.

All masks are hypotheses. Even a single non-clipped high-score mask with RGB
edge support can omit an occluded body part. Nothing here selects a grasp,
creates a metric target, accepts a complete body or schedules a robot action.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
from types import SimpleNamespace
import uuid

import cv2
import numpy as np
import orimapi

from orimapi.evaluation.storage import atomic_json, digest, file_digest
from .envelope import polygon

ROOT = Path(orimapi.__file__).resolve().parent.parent
MODEL_PATH = "checkpoint/vision-tools/sam/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
MODEL_FILES = {
    "config.json": "5ebd0d8643b486f3a716bf17c2a15531eb818b2b96ff0c6c5dcc88fa015161af",
    "model.safetensors": "892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579",
    "preprocessor_config.json": "225545a743c654e3c495ec6f545a0eaba57c8ba3fbbd8483b3cb1c0fc58db517",
}
PACKAGES = (
    "torch", "transformers", "safetensors", "numpy", "opencv-python-headless", "pillow"
)
MAX_PIXELS = 2_000_000
MAX_CANDIDATES = 3


def default_resource(device):
    """Explicit NEW-candidate resource declaration; never an old-runtime default."""
    return validate_config({
        "model_path": MODEL_PATH,
        "device": device,
        "files": dict(MODEL_FILES),
        "software_versions": {name: version(name) for name in PACKAGES},
    })


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {
        "model_path", "device", "files", "software_versions"
    }:
        raise ValueError("SAM requires explicit local path, device, file and software identities")
    name = config["model_path"]
    if not isinstance(name, str) or not 1 <= len(name) <= 300:
        raise ValueError("SAM model_path must be a bounded repository-relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("SAM model_path must be repository-relative without traversal")
    if not isinstance(config["device"], str) or not re.fullmatch(
        r"cpu|cuda:\d+", config["device"]
    ):
        raise ValueError("SAM device must be cpu or cuda:N")
    files = config["files"]
    if not isinstance(files, dict) or set(files) != set(MODEL_FILES) or any(
        not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha)
        for sha in files.values()
    ):
        raise ValueError("SAM requires three explicit SHA-256 file identities")
    software = config["software_versions"]
    if not isinstance(software, dict) or set(software) != set(PACKAGES) or any(
        not isinstance(v, str) or not 1 <= len(v) <= 100 for v in software.values()
    ):
        raise ValueError("SAM requires explicit inference software versions")
    return deepcopy(config)


def resource_identity(config):
    value = validate_config(config)
    root = (ROOT / value["model_path"]).resolve(strict=True)
    actual = {name: file_digest(root / name) for name in MODEL_FILES}
    if actual != value["files"]:
        raise ValueError("SAM checkpoint identity changed")
    software = {name: version(name) for name in PACKAGES}
    if software != value["software_versions"]:
        raise ValueError("SAM inference software identity changed")
    if json.loads((root / "config.json").read_text()).get("model_type") != "sam":
        raise ValueError("SAM proposals require a SAM checkpoint")
    return {
        **value,
        "resolved_model_path": str(root),
        "kind": "prompted_rgb_segmentation_proposals",
        "sensor_ground_truth": False,
    }


def availability(config):
    if config is None:
        return {"available": False, "reason": "No frozen SAM proposal resource configured"}
    try:
        value = validate_config(config)
        root = ROOT / value["model_path"]
        if any(not (root / name).is_file() for name in MODEL_FILES):
            raise ValueError("Pinned local SAM files are missing")
        if {name: version(name) for name in PACKAGES} != value["software_versions"]:
            raise ValueError("Pinned SAM inference software is unavailable")
    except (ValueError, OSError, PackageNotFoundError) as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "reason": "Opt-in local SAM proposals; hashes verified before use; no automatic acceptance",
    }


def prompt(image, box=None, points=None):
    if (
        not isinstance(image, np.ndarray) or image.dtype != np.uint8
        or image.ndim != 3 or image.shape[2] != 3
        or min(image.shape[:2]) < 16 or image.shape[0] * image.shape[1] > MAX_PIXELS
    ):
        raise ValueError("SAM needs a bounded original-size uint8 RGB image")
    h, w = image.shape[:2]
    result = {"box": None, "points": []}
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("SAM box must contain x0,y0,x1,y1")
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in box):
            raise ValueError("SAM box coordinates must be finite numbers")
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("SAM box is outside the original image")
        result["box"] = [x0, y0, x1, y1]
    if points is not None:
        if not isinstance(points, list) or not 1 <= len(points) <= 8:
            raise ValueError("SAM point prompts require 1..8 labelled pixels")
        for row in points:
            if not isinstance(row, dict) or set(row) != {"pixel", "label"}:
                raise ValueError("SAM points require exactly pixel and label")
            pixel, label = row["pixel"], row["label"]
            if (
                not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in pixel)
                or not 0 <= pixel[0] < w or not 0 <= pixel[1] < h
                or type(label) is not int or label not in (0, 1)
            ):
                raise ValueError("SAM point must be a current-image pixel labelled 0 or 1")
            result["points"].append({"pixel": list(map(float, pixel)), "label": label})
    if result["box"] is None and not any(p["label"] == 1 for p in result["points"]):
        raise ValueError("SAM needs an explicit box or at least one positive point")
    return result


def bounded_outline(contour, shape):
    """Simplify one mask boundary, never select/repair another object or RGB edge.

    Preserve the original0.6px representation when it fits. Only an over-complex
    contour may use0.8/1.0px, with at least98% agreement to its filled external
    contour. The unchanged RGB/geometry gates still run separately afterwards.
    """
    reference = np.zeros(shape, np.uint8)
    cv2.drawContours(reference, [contour], -1, 1, cv2.FILLED)
    dense = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    last = None
    for epsilon in (0.6, 0.8, 1.0):
        outline = cv2.approxPolyDP(contour, epsilon, True)
        last = {"epsilon_px": epsilon, "vertices": len(outline)}
        if not 3 <= len(outline) <= 64:
            continue
        deviation = max(
            abs(cv2.pointPolygonTest(outline, tuple(map(float, p)), True))
            for p in dense.reshape(-1, 2)
        )
        raster = np.zeros(shape, np.uint8)
        cv2.fillPoly(raster, [outline], 1)
        union = np.count_nonzero(reference | raster)
        agreement = np.count_nonzero(reference & raster) / union if union else 0.0
        last.update(
            maximum_contour_deviation_px=deviation,
            filled_external_contour_iou=agreement,
            scope="Agreement to model external boundary only, not original-RGB correctness, holes/material or completeness",
        )
        if deviation > 1.0 + 1e-6 or (epsilon > 0.6 and agreement < 0.98):
            # Do not keep simplifying after a representation failed fidelity.
            return None, {**last, "rejected": "Bounded contour fidelity failed"}
        return outline.reshape(-1, 2), last
    return None, {**last, "rejected": "Contour exceeds the unchanged3..64 vertex bound within1px"}


def describe_mask(image, mask, score):
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("SAM mask must be boolean and match original image dimensions")
    if not np.isfinite(score):
        raise ValueError("SAM returned a nonfinite score")
    raster = mask.astype(np.uint8)
    count, _ = cv2.connectedComponents(raster, connectivity=8)
    contours, hierarchy = cv2.findContours(raster, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    external = [
        contour for i, contour in enumerate(contours)
        if hierarchy[0, i, 3] == -1
    ] if hierarchy is not None else []
    ys, xs = np.nonzero(mask)
    clipped = bool(mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any())
    row = {
        "model_score": float(score),
        "score_scope": "Model ranking output, NOT calibrated accuracy/completeness probability",
        "mask_sha256": hashlib.sha256(raster.tobytes()).hexdigest(),
        "dimensions": [image.shape[1], image.shape[0]],
        "area_pixels": int(mask.sum()),
        "components": int(count - 1),
        "holes": int(sum(hierarchy[0, :, 3] >= 0)) if hierarchy is not None else 0,
        "touches_frame": clipped,
        "bounding_box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if len(xs) else None,
        "outline": None,
        "complete_body_verified": False,
        "occlusion_status": "unknown; inspect original RGB",
        "original_edge_gate": "not_checked",
        "geometry_eligible": False,
    }
    if len(external) == 1 and count == 2:
        outline, representation = bounded_outline(external[0], mask.shape)
        row["outline_representation"] = representation
        row["outline_vertices"] = representation["vertices"]
        if outline is not None:
            row["outline"] = outline.tolist()
            if not clipped:
                try:
                    _, support = polygon(
                        image, SimpleNamespace(width=image.shape[1], height=image.shape[0]), outline
                    )
                    row.update(original_edge_gate="pass", rgb_edge_support=support, geometry_eligible=True)
                except ValueError as error:
                    row["original_edge_gate"] = str(error)
        else:
            row["original_edge_gate"] = representation["rejected"]
    if clipped:
        row["original_edge_gate"] = "Frame-clipped proposal is not eligible as a complete body"
    return row


class _Model:
    def __init__(self, path, device):
        from transformers import SamModel, SamProcessor

        self.processor = SamProcessor.from_pretrained(path, local_files_only=True)
        self.model = SamModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).to(device).eval()
        self.device, self.lock = device, threading.Lock()

    def predict(self, image, selection):
        import torch

        arguments = {}
        if selection["box"] is not None:
            arguments["input_boxes"] = [[selection["box"]]]
        if selection["points"]:
            arguments["input_points"] = [[p["pixel"] for p in selection["points"]]]
            arguments["input_labels"] = [[p["label"] for p in selection["points"]]]
        with self.lock, torch.inference_mode():
            data = self.processor(images=image, return_tensors="pt", **arguments).to(self.device)
            output = self.model(**data, multimask_output=True)
            masks = self.processor.post_process_masks(
                output.pred_masks.cpu(), data.original_sizes.cpu(), data.reshaped_input_sizes.cpu()
            )[0][0].numpy()
            scores = output.iou_scores[0, 0].detach().cpu().numpy()
        return masks, scores


_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _resident_model(path, device, resource_digest):
    return _Model(path, device)


class SamSegmenter:
    def __init__(self, config):
        self.config = validate_config(config)
        self.identity = resource_identity(self.config)
        self.trace_root = None

    def bind_trace(self, root):
        self.trace_root = Path(root)

    def reset(self):
        # No scene embeddings or prior masks are used for a later observation.
        pass

    def propose(self, image, *, box=None, points=None, provenance=None):
        selection = prompt(image, box, points)
        if resource_identity(self.config) != self.identity:
            raise ValueError("SAM resources changed during evaluation")
        with _load_lock:
            model = _resident_model(
                self.identity["resolved_model_path"], self.config["device"], digest(self.identity)
            )
        started = time.monotonic()
        masks, scores = model.predict(image, selection)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (
            masks.shape != (MAX_CANDIDATES, *image.shape[:2])
            or masks.dtype != np.bool_ or scores.shape != (MAX_CANDIDATES,)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("SAM must return exactly three finite original-size boolean mask proposals")
        rows = [describe_mask(image, mask, score) for mask, score in zip(masks, scores)]
        report = {
            "resource_identity": deepcopy(self.identity),
            "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "source_dimensions": [image.shape[1], image.shape[0]],
            "prompt": selection,
            "inference_postprocess_s": time.monotonic() - started,
            "provenance": deepcopy(provenance),
            "candidate_metadata": rows,
            "selection": "none; explicit current-RGB review required",
            "sensor_ground_truth": False,
            "official_success": None,
        }
        if self.trace_root is not None:
            folder = self.trace_root / "segmentation_proposals" / uuid.uuid4().hex
            folder.mkdir(parents=True, exist_ok=False)
            original, proposals = folder / "source_rgb.png", folder / "masks.npz"
            from PIL import Image

            Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
                original, format="PNG"
            )
            np.savez_compressed(proposals, masks=masks, scores=scores)
            report["artifacts"] = {
                "original_rgb": {"path": str(original.relative_to(self.trace_root)), "sha256": file_digest(original)},
                "masks": {"path": str(proposals.relative_to(self.trace_root)), "sha256": file_digest(proposals)},
            }
            atomic_json(folder / "provenance.json", report)
        return masks.copy(), report


def overlay(image, mask):
    """Original-size preview, not a substitute for any required original view."""
    result = image.copy()
    result[mask] = np.rint(0.75 * result[mask] + 0.25 * np.array([255, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 0, 255), 1)
    return result
