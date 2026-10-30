from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image


def read_api_key(config):
    raise NotImplementedError('External bbox API is not implemented in this release')


def draw_reference(image, bbox):
    out = np.array(image, dtype=np.uint8, copy=True)
    height, width = out.shape[:2]
    x0, y0 = np.floor(np.asarray(bbox[:2]) * [width, height] / 1000).astype(int)
    x1, y1 = np.ceil(np.asarray(bbox[2:]) * [width, height] / 1000).astype(int)
    l, t, r, b = max(0,x0-5), max(0,y0-5), min(width,x1+5), min(height,y1+5)
    out[t:y0,l:r] = [255,0,0]
    out[y1:b,l:r] = [255,0,0]
    out[y0:y1,l:x0] = [255,0,0]
    out[y0:y1,x1:r] = [255,0,0]
    return Image.fromarray(out)


class BBoxConditioner:
    def __init__(self, config, annotator=None):
        self.cache_mode = str(config.get("bbox_cache_mode", "instruction"))
        self.missing_policy = str(config.get("bbox_missing_policy", "full_image"))
        if self.cache_mode not in {"instruction", "frame"}:
            raise ValueError("bbox_cache_mode must be instruction or frame")
        if self.missing_policy not in {"full_image", "error"}:
            raise ValueError("bbox_missing_policy must be full_image or error")
        if annotator is None:
            raise NotImplementedError("External bbox API is not implemented in this release")
        self.annotator = annotator
        directory = config.get("bbox_artifact_dir")
        self.directory = Path(directory).expanduser() / uuid.uuid4().hex[:12] if directory else None
        self.epoch = 0
        self.serial = 0
        self.cache = {}
        self.context_id = None

    def set_context(self, context_id):
        context_id = str(context_id)
        if context_id != self.context_id:
            self.reset()
            self.context_id = context_id

    def reset(self):
        self.cache.clear()
        self.context_id = None
        self.epoch += 1

    def prepare(self, head_rgb, instruction, env_idx):
        image_array = np.asarray(head_rgb)
        if image_array.ndim != 3 or image_array.shape[-1] != 3 or not np.isfinite(image_array).all():
            raise ValueError("bbox head image must be finite RGB HWC")
        image_array = np.clip(image_array,0,255).astype(np.uint8)
        prompt_key = " ".join(instruction.split()).strip()
        key = (int(env_idx),prompt_key)
        cached = self.cache.get(key) if self.cache_mode == "instruction" else None
        cache_hit = cached is not None
        if cached is None:
            result = self.annotator.annotate(Image.fromarray(image_array),instruction)
            if result.get("found") and result.get("bbox") is not None:
                bbox = np.asarray(result["bbox"], dtype=np.float32)
                if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox < 0).any() or (bbox > 1000).any() or (bbox[2:] <= bbox[:2]).any():
                    raise ValueError("Annotator returned invalid normalized xyxy bbox")
            elif self.missing_policy == "error":
                raise ValueError("BBox annotator did not find a target")
            else:
                bbox = np.asarray([0,0,1000,1000],dtype=np.float32)
            reference = np.array(image_array,copy=True)
            cached = (bbox,reference,result)
            if result.get("found") and self.cache_mode == "instruction":
                self.cache[key] = cached
        bbox, reference, result = cached
        if self.directory:
            self.directory.mkdir(parents=True,exist_ok=True)
            stem = f"reset{self.epoch:03d}_env{int(env_idx):03d}_{self.serial:05d}"
            record = dict(result, env_idx=int(env_idx), reset=self.epoch, instruction=instruction,
                          used_bbox=bbox.tolist(), cache_hit=cache_hit,
                          input_mode="red_reference", image_description=True, step_id=self.context_id)
            if not cache_hit:
                Image.fromarray(reference).save(self.directory / f"{stem}_head.png")
                draw_reference(reference,bbox).save(self.directory / f"{stem}_boxed.png")
                record["head_image"] = f"{stem}_head.png"
                record["boxed_image"] = f"{stem}_boxed.png"
            with (self.directory / "annotations.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
        self.serial += 1
        return {"bbox":bbox[None].copy(),
                "bbox_reference_images":((reference.astype(np.float32)/127.5)-1).transpose(2,0,1)[None]}
