"""RGB-only monocular depth used by REAL_RSI evaluation.

Camera extrinsics are camera-to-world, looking along -Z with +Y up.
Depth is distance to the image plane; scale is fitted to the table plane.
"""
from real_rsi_assets import model_path as _real_rsi_model_path
import numpy as np
import torch
import torch.nn.functional as F

DA2_LARGE = _real_rsi_model_path('depth-anything__Depth-Anything-V2-Large-hf')


class MonoPlaneDepth:
    """RGB-only metric depth: Depth Anything V2 (relative, affine-invariant inverse depth) per camera, scaled to metres
    by a RANSAC affine fit to the table plane. Pixels whose ray meets the plane z = table_z give exact depths; the fit
    picks the table pixels itself (objects and the robot are outliers). Needs only RGB, K, the camera pose and the
    table height. If a camera's fit fails (table hardly visible), its last good coefficients are reused."""

    def __init__(self, model_dir=DA2_LARGE, table_z=0.765, device=None, tol=0.02, iters=300):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
        self.table_z, self.tol, self.iters = table_z, tol, iters
        self.last = {}
        self.rng = np.random.default_rng(0)

    @torch.no_grad()
    def _raw(self, color):
        from PIL import Image
        img = np.asarray(color)[..., :3]
        inp = self.proc(images=Image.fromarray(img), return_tensors='pt').to(self.device)
        pred = self.model(**inp).predicted_depth
        h, w = img.shape[:2]
        return F.interpolate(pred[:, None], size=(h, w), mode='bicubic', align_corners=False)[0, 0].float().cpu().numpy()

    def _plane(self, k, t, h, w):
        v, u = np.mgrid[0:h, 0:w]
        d = np.stack([(u - k[0, 2]) / k[0, 0], -(v - k[1, 2]) / k[1, 1], -np.ones((h, w))], -1) @ t[:3, :3].T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (self.table_z - t[2, 3]) / d[..., 2]
        s[(s <= 0) | ~np.isfinite(s) | (s > 3.0)] = np.nan
        return s

    def _fit(self, r, inv_p):
        n, best, best_k = len(r), None, -1
        for _ in range(self.iters):
            i, j = self.rng.integers(0, n, 2)
            if abs(r[i] - r[j]) < 1e-9:
                continue
            a = (inv_p[i] - inv_p[j]) / (r[i] - r[j])
            b = inv_p[i] - a * r[i]
            inl = np.abs(a * r + b - inv_p) < self.tol * np.abs(inv_p)
            if inl.sum() > best_k:
                best, best_k = inl, inl.sum()
        if best is None or best_k < 200:
            return None
        return np.polyfit(r[best], inv_p[best], 1)

    def estimate(self, cams, targets=None):
        """cams: {name: {'color', 'intrinsic_matrix', 'extrinsic_matrix'}} -> {name: HxW float32 depth (m)}."""
        out = {}
        for name in (targets or list(cams)):
            c = cams[name]
            raw = self._raw(c['color'])
            h, w = raw.shape
            pd = self._plane(np.asarray(c['intrinsic_matrix'], float), np.asarray(c['extrinsic_matrix'], float), h, w)
            m = np.isfinite(pd)
            idx = np.flatnonzero(m.ravel())
            coef = None
            if len(idx) >= 500:
                idx = self.rng.choice(idx, min(len(idx), 40000), replace=False)
                coef = self._fit(raw.ravel()[idx], 1.0 / pd.ravel()[idx])
            if coef is None:
                coef = self.last.get(name)
            if coef is None:                       # no table in view and no earlier fit: table-plane depth
                out[name] = np.nan_to_num(pd, nan=3.0).astype(np.float32)
                continue
            self.last[name] = coef
            out[name] = np.clip(1.0 / np.clip(coef[0] * raw + coef[1], 1e-3, None), 0.02, 5.0).astype(np.float32)
        return out
