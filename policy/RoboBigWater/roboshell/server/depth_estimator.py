"""Depth from RGB for the three cameras, in the format of the rendered depth (metres to the image plane).

Selected with ROBOSHELL_DEPTH=<directory of a Hugging Face depth model>; the rendered depth is then not read at all.
The network's output is affine-invariant in practice, so its scale is fixed per frame from the table plane: the camera
pose is known (calibration, or the arm's end pose for a wrist camera) and the table height is a constant of the cell
(ROBOSHELL_TABLE_Z). Pixels that agree with the plane after a first guess give the scale; a camera that sees too
little table keeps its previous scale.
"""
import os
import numpy as np

Z_TABLE = float(os.environ.get("ROBOSHELL_TABLE_Z", "0.768"))


def plane_depth(K, T, shape):
    """Depth at which each pixel's ray meets the table plane (nan where the ray does not go down)."""
    h, w = shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) @ T[:3, :3].T
    down = d[..., 2] < -1e-3
    return np.where(down, (Z_TABLE - T[2, 3]) / np.where(down, d[..., 2], -1.0), np.nan)


def table_scale(pred, K, T):
    plane = plane_depth(K, T, pred.shape)
    ok = np.isfinite(pred) & (pred > 0) & np.isfinite(plane) & (plane > 0.05) & (plane < 3.0)
    if ok.sum() < 2000:
        return None
    s = np.median(plane[ok] / pred[ok])
    for tol in (0.03, 0.02, 0.02):
        sel = ok & (np.abs(s * pred - plane) < tol)
        if sel.sum() < 500:
            return None
        s = np.median(plane[sel] / pred[sel])
    return float(s)


class DepthEstimator:
    def __init__(self, model_dir, device_id=0):
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.device_count() > device_id else "cuda:0")
        self.proc = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_dir).to(self.device).eval()
        self.last_scale = {}
        self.name = os.path.basename(model_dir.rstrip("/"))

    def __call__(self, color_rgb, K, T, camera):
        from PIL import Image
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=Image.fromarray(color_rgb), return_tensors="pt").to(self.device)
            out = self.model(**inp).predicted_depth
            pred = torch.nn.functional.interpolate(out[None], size=color_rgb.shape[:2], mode="bicubic",
                                                   align_corners=False)[0, 0].float().cpu().numpy()
        s = table_scale(pred, np.asarray(K, float), np.asarray(T, float))
        if s is None:
            s = self.last_scale.get(camera, 1.0)
        else:
            self.last_scale[camera] = s
        depth = (pred * s).astype(np.float32)
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
