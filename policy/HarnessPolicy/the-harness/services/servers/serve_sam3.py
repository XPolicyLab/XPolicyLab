#!/usr/bin/env python3
"""Serve SAM 3 instance segmentation over HTTP for perception skills.

    CUDA_VISIBLE_DEVICES=3 python services/servers/serve_sam3.py --checkpoint /path/to/sam3.pt --port 8810

POST /segment {"image": <base64 data URL>, "prompt": "red bowl", "threshold": 0.5}
  -> {"instances": [{"score", "box": [x0, y0, x1, y1], "mask": <PNG data URL, 0/255>}]}
POST /segment_point {"image": ..., "points": [[x, y], ...], "labels": [1, ...]}
  -> the same, from SAM 3's interactive (point-prompted) predictor: candidate masks
     of the object under the points, best predicted IoU first. Label 1 marks a
     point on the object, 0 a point on the background.

Uses the official sam3 package (facebookresearch/sam3, installed editable in the
harness-policy environment, see environment/README.md) and its sam3.pt
checkpoint, loaded with torch.load(weights_only=True). One request at a time.
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="sam3.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8810)
    # Memory savers for a GPU shared with the simulator (e.g. a 16 GB laptop GPU).
    parser.add_argument("--bf16-text", action="store_true",
                        help="keep the language backbone in bfloat16 (0.7 GB less; same scores in a check, while a "
                             "bfloat16 vision backbone or decoder breaks the results)")
    parser.add_argument("--free-memory", action="store_true",
                        help="return PyTorch's cached GPU memory after each request (smaller footprint)")
    parser.add_argument("--text-only", action="store_true",
                        help="serve /segment only: skip the point-prompted predictor behind /segment_point")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    # sam3 imports its training data loaders, which import decord (video decoding)
    # at module level. Image inference never calls it; a stub keeps the import light.
    sys.modules.setdefault("decord", types.ModuleType("decord"))
    sys.modules["decord"].cpu = sys.modules["decord"].VideoReader = None
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False, device=device,
                                   enable_inst_interactivity=not args.text_only)
    if args.bf16_text:
        language = model.backbone.language_backbone
        language.to(torch.bfloat16)
        forward = language.forward

        def to_float(value):
            if torch.is_tensor(value):
                return value.float() if value.is_floating_point() else value
            if isinstance(value, dict):
                return {key: to_float(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(to_float(item) for item in value)
            return value

        def language_forward(*args, **kwargs):  # bfloat16 inside, float32 to the rest of the model
            args = tuple(a.to(torch.bfloat16) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
            return to_float(forward(*args, **kwargs))

        language.forward = language_forward
        torch.cuda.empty_cache()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
    lock = threading.Lock()
    print(json.dumps({"serving": "sam3", "port": args.port, "device": device}), flush=True)

    def decode(url):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")

    def encode(mask):
        ok, png = cv2.imencode(".png", ((mask > 0).astype(np.uint8) * 255))
        return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()

    def segment_point(request):
        image = decode(request["image"])
        points = np.asarray(request["points"], dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(request.get("labels") or [1] * len(points), dtype=np.int32)
        if len(points) == 0 or len(labels) != len(points):
            raise ValueError("points must be a nonempty list of [x, y] with one label each")
        with lock, torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            state = processor.set_image(image)
            masks, scores, _ = model.predict_inst(state, point_coords=points, point_labels=labels,
                                                  multimask_output=True)
        instances = []
        for mask, score in zip(np.asarray(masks), np.asarray(scores, dtype=float)):
            rows, cols = np.nonzero(mask > 0)
            if not len(rows):
                continue
            box = [float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())]
            instances.append({"score": float(score), "box": box, "mask": encode(mask)})
        instances.sort(key=lambda item: -item["score"])
        return {"instances": instances}

    def segment(request):
        image = decode(request["image"])
        threshold = float(request.get("threshold", 0.5))
        with lock, torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=str(request["prompt"]))
        masks, boxes, scores = output.get("masks"), output.get("boxes"), output.get("scores")
        instances = []
        if masks is not None and boxes is not None:
            masks = masks.float().cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            for mask, box, score in zip(masks, boxes.float().cpu().numpy(), scores.float().cpu().numpy()):
                if float(score) < threshold:
                    continue
                instances.append({"score": float(score), "box": [float(v) for v in box], "mask": encode(mask)})
        instances.sort(key=lambda item: -item["score"])
        return {"instances": instances}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *items):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            route = {"/segment": segment, "/segment_point": segment_point}.get(self.path.rstrip("/"))
            if route is None:
                self._send(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                started = time.perf_counter()
                body = route(json.loads(self.rfile.read(size)))
                print(json.dumps({"instances": len(body["instances"]),
                                  "seconds": round(time.perf_counter() - started, 2)}), flush=True)
                self._send(200, body)
            except Exception as error:
                self._send(400, {"error": f"{type(error).__name__}: {error}"})
            finally:
                if args.free_memory and torch.cuda.is_available():
                    with lock:
                        torch.cuda.empty_cache()

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
