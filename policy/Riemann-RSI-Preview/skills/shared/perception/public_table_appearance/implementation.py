import cv2
import numpy as np

class TableAppearance:

    def __init__(self, region, robot_mask):
        self.region = np.asarray(region, bool)
        self.robot_mask = np.asarray(robot_mask, bool)
        if self.region.shape != self.robot_mask.shape:
            raise ValueError('Public image masks differ')

    def _quad(self, rgb, block=1, subsample=1):
        (h, w) = rgb.shape[:2]
        (yy, xx) = np.mgrid[0:h:block, 0:w:block]
        u = (xx + 0.5 * block) / w * 2.0 - 1.0
        v = (yy + 0.5 * block) / h * 2.0 - 1.0
        return (np.ones_like(u), u, v, u * u, u * v, v * v)

    def fit(self, rgb, ignore=None, k_fit=7, iters=5, trim=0.7, block=2, abs_cap=18.0):
        (h, w) = rgb.shape[:2]
        small = cv2.medianBlur(rgb, k_fit)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        reg = self.region & ~self.robot_mask
        if ignore is not None:
            reg = reg & ~ignore
        (ys, xs) = np.nonzero(reg[::block, ::block])
        if len(xs) < 500:
            return None
        vals = lab[::block, ::block][ys, xs][:, 1:]
        u = (xs * block + block / 2) / w * 2.0 - 1.0
        v = (ys * block + block / 2) / h * 2.0 - 1.0
        A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
        keep = np.ones(len(u), bool)
        coef = None
        for _ in range(iters):
            (coef, *_) = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
            r = np.linalg.norm(vals - A @ coef, axis=1)
            thr = min(max(float(np.quantile(r[keep], trim)), 3.0), abs_cap)
            keep = r <= thr
        r = np.linalg.norm(vals[keep] - A[keep] @ coef, axis=1)
        sigma = float(np.median(r)) * 1.4826 + 2.0
        spread = float(np.quantile(r, 0.9))
        return ((coef, sigma, spread, (h, w)), (ys, xs, block))

    def foreground(self, rgb, fit, k_ab=2.0, k_b=5.0, open_px=3, close_px=11):
        (coef, sigma, spread, (h, w)) = fit
        lab = cv2.cvtColor(cv2.medianBlur(rgb, 5), cv2.COLOR_RGB2LAB).astype(np.float32)
        A = np.stack(self._quad(rgb), -1)
        pred = A @ coef
        d_ab = np.linalg.norm(lab[..., 1:] - pred, axis=2)
        d_b = np.abs(lab[..., 2] - pred[..., 1])
        m = ((d_ab > k_ab * sigma) | (d_b > k_b * sigma)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
        return (m > 0, sigma)
