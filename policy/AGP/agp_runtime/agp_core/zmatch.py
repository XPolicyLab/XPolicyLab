"""Height of a picked surface point without depth: epipolar plane sweep between two calibrated RGB views.

Numpy only, so it runs in the MCP server's environment (.venv-agent: numpy + PIL, no OpenCV/scipy): cv2.pyrDown,
cv2.Sobel and cv2.cvtColor are replaced by equivalent numpy code (border BORDER_REFLECT_101, the same kernels), which
agp_core/tests/test_rgb_only.py checks against OpenCV.

Method (a pixel p picked in view A, matched into view B):
  * Sweep heights h of a horizontal plane z = h (frame: an arm base, table top z = 0). For each h, the window
    around p in A is mapped to B through the homography of that plane and compared with B (anti-aliased: the
    finer view is sampled at the Gaussian-pyramid level of the local scale). Scores: ZNCC on RGB plus a
    gradient channel, or census, each averaged over several window radii, minus a colour-difference penalty.
  * Pick the best height (parabola refinement), its score `peak` and the best score at least 4 mm away
    (`second`); margin = peak - second.
  * Two-way check: project the matched point into B and sweep back along B's ray into A; the two heights must
    agree within 3 mm.
  * Rule ("first LR-consistent (tol 3 mm, margin >= .05) of census, zncc"): try the census
    variant, then the ZNCC variant; the first that passes the two-way check and the margin gate gives the
    height ('match'). If none passes, the result is 'uncertain' with the best guess.

Conventions: pixel indices (u, v) = image[v, u]; continuous image coordinates put the centre of pixel u at
u + 0.5 (the module's p arguments are continuous); K has the principal point in continuous coordinates;
T = frame <- optical (ROS/OpenCV optical axes).
"""
import numpy as np

HS = np.arange(-0.005, 0.2005, 0.0005)       # swept plane heights (m above the table, arm base frame)
EXCLUSION_M = 0.004                           # `second` is the best score at least this far from the peak
LR_TOLERANCE_M = 0.003                        # two-way check
MARGIN_GATE = 0.05                            # peak - second
LR_BORDER_PX = 8                              # the back-projected point must lie this far inside B
CAMERA_CLEARANCE_M = 0.03                     # planes stop this far below the lower camera
VARIANTS = {
    # name: sweep options
    "census": dict(radii=(4, 7, 12), census=True, color_pen=1.0),
    "zncc": dict(radii=(3, 6, 12), use_grad=True, color_pen=2.0),
}
RULE_ORDER = ("census", "zncc")
PYRAMID_LEVELS = 6


# ----------------------------------------------------------------------------- image primitives (numpy)
def _pad101(a, p):
    return np.pad(a, ((p, p), (p, p)) + ((0, 0),) * (a.ndim - 2), mode="reflect")


def pyr_down(img):
    """cv2.pyrDown: 5x5 Gaussian [1 4 6 4 1]^2/256, BORDER_REFLECT_101, keep even rows/columns.

    float32 operations in the order of OpenCV's vectorised path, (s-1 + s1 + s0)*4 + (s-2 + s2 + 2*s0), first
    along rows on the kept columns, then along columns, then * 1/256. The colour channels of levels 1 and 2 of an
    8-bit image come out bit-identical to OpenCV; coarser levels and the gradient channel can differ in the last
    bit (OpenCV may fuse multiply-adds), which only matters for census comparisons of nearly equal neighbours."""
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    oh, ow = (h + 1) // 2, (w + 1) // 2
    p = _pad101(img, 2)
    four = np.float32(4.0)
    c = 2 + 2 * np.arange(ow)                      # padded column of each kept output column
    s0, s1, s2, s3, s4 = p[:, c - 2], p[:, c - 1], p[:, c], p[:, c + 1], p[:, c + 2]
    rows = ((s1 + s3) + s2) * four + ((s0 + s4) + (s2 + s2))
    r = 2 + 2 * np.arange(oh)
    r0, r1, r2, r3, r4 = rows[r - 2], rows[r - 1], rows[r], rows[r + 1], rows[r + 2]
    out = ((r1 + r3) + r2) * four + ((r0 + r4) + (r2 + r2))
    return (out * np.float32(1.0 / 256.0)).astype(np.float32)


def sobel_magnitude(gray):
    """sqrt(Sobel_x^2 + Sobel_y^2) with cv2.Sobel(ksize=3, BORDER_REFLECT_101) kernels (row filter first)."""
    g = _pad101(np.asarray(gray, np.float32), 1)
    h, w = gray.shape
    two = np.float32(2.0)
    dx = g[:, 2:w + 2] - g[:, 0:w]                                  # row filter [-1 0 1]
    gx = (dx[0:h] + dx[1:h + 1] * two) + dx[2:h + 2]                # column filter [1 2 1]
    sx = (g[:, 0:w] + g[:, 1:w + 1] * two) + g[:, 2:w + 2]          # row filter [1 2 1]
    gy = sx[2:h + 2] - sx[0:h]                                      # column filter [-1 0 1]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def features(img):
    """RGB (float32) + gradient magnitude of the grey image (cv2 RGB2GRAY weights)."""
    rgb = np.asarray(img, np.float32)
    gray = rgb[..., 0] * np.float32(0.299) + rgb[..., 1] * np.float32(0.587) + rgb[..., 2] * np.float32(0.114)
    return np.dstack([rgb, sobel_magnitude(gray)[..., None]])


def pyramid(img, levels=PYRAMID_LEVELS):
    p = [np.asarray(img, np.float32)]
    for _ in range(levels - 1):
        p.append(pyr_down(p[-1]))
    return p


def bilinear(img, x, y):
    """img HxWxC sampled at index coordinates x, y (arrays); returns (..., C) and the in-image mask."""
    hh, ww = img.shape[:2]
    valid = (x >= 0) & (x <= ww - 1.001) & (y >= 0) & (y <= hh - 1.001)
    x = np.clip(x, 0, ww - 1.001)
    y = np.clip(y, 0, hh - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    out = (img[y0, x0] * (1 - dx) * (1 - dy) + img[y0, x0 + 1] * dx * (1 - dy)
           + img[y0 + 1, x0] * (1 - dx) * dy + img[y0 + 1, x0 + 1] * dx * dy)
    return out, valid


# ----------------------------------------------------------------------------- geometry
def plane_points(K, T, xs, ys, h):
    """Continuous pixel coordinates of camera (K, T) back-projected onto z = h; points and a lambda > 0 mask."""
    r = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1) @ T[:3, :3].T
    lam = (h - T[2, 3]) / r[..., 2]
    return T[:3, 3] + lam[..., None] * r, lam > 0


def project(K, T, X):
    """Continuous pixel coordinates and optical depth of points X (frame of T)."""
    xc = (X - T[:3, 3]) @ T[:3, :3]
    z = xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = xc[..., 0] / z * K[0, 0] + K[0, 2]
        y = xc[..., 1] / z * K[1, 1] + K[1, 2]
    return x, y, z


def ray_point(K, T, p, h):
    """Point where the ray through continuous pixel p meets z = h."""
    r = T[:3, :3] @ np.array([(p[0] - K[0, 2]) / K[0, 0], (p[1] - K[1, 2]) / K[1, 1], 1.0])
    return T[:3, 3] + (h - T[2, 3]) / r[2] * r


class View:
    """One calibrated image: K, T (frame <- optical), feature pyramid and an optional ignore mask (HxW bool)."""

    def __init__(self, img, K, T, mask=None, levels=PYRAMID_LEVELS):
        self.K = np.asarray(K, float)
        self.T = np.asarray(T, float)
        self.pyr = pyramid(features(img), levels)
        self.mask = None if mask is None else np.asarray(mask, bool)
        self.shape = self.pyr[0].shape[:2]


def local_scale(A, B, p, h):
    """B pixels per A pixel at the plane point under A's pixel p (continuous coordinates)."""
    xs = np.array([p[0], p[0] + 1, p[0]])
    ys = np.array([p[1], p[1], p[1] + 1])
    X, ok = plane_points(A.K, A.T, xs, ys, h)
    x, y, z = project(B.K, B.T, X)
    J = np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])
    return float(np.sqrt(abs(np.linalg.det(J)))), bool(ok.all() and (z > 0.02).all())


# ----------------------------------------------------------------------------- scores (vectorised over heights)
def _zncc(a, b, w):
    """a, b (n, N, C), w (n, N): weighted ZNCC over all channels jointly (per-channel means removed)."""
    sw = w.sum(1)
    safe = np.where(sw > 0, sw, 1.0)
    ma = (a * w[..., None]).sum(1) / safe[:, None]
    mb = (b * w[..., None]).sum(1) / safe[:, None]
    a = a - ma[:, None, :]
    b = b - mb[:, None, :]
    num = (w[..., None] * a * b).sum((1, 2))
    den = np.sqrt((w[..., None] * a * a).sum((1, 2)) * (w[..., None] * b * b).sum((1, 2)))
    out = np.where(den < 1e-6, -1.0, num / np.where(den < 1e-6, 1.0, den))
    return np.where(sw <= 0, -1.0, out)


def _census(a, b, w, rad=2):
    """a, b (n, S, S, 3), w (n, S, S): 1 - weighted mean normalised Hamming distance of census signatures."""
    ga = a.mean(-1)
    gb = b.mean(-1)
    s = ga.shape[1]
    if s <= 2 * rad + 1:
        return np.full(ga.shape[0], np.nan)
    c = slice(rad, s - rad)
    ham = np.zeros((ga.shape[0], s - 2 * rad, s - 2 * rad))
    bits = 0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx == 0 and dy == 0:
                continue
            sa = ga[:, rad + dy:s - rad + dy, rad + dx:s - rad + dx] > ga[:, c, c]
            sb = gb[:, rad + dy:s - rad + dy, rad + dx:s - rad + dx] > gb[:, c, c]
            ham += sa != sb
            bits += 1
    ww = w[:, c, c]
    total = ww.sum((1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - (ww * ham).sum((1, 2)) / (total * bits)
    return np.where(total > 0, out, np.nan)


def sweep(A, B, p, hs, radii=(3, 6, 12), color_sigma=None, use_grad=True, min_valid=0.8, color_pen=0.0,
          census=False):
    """Scores (len(hs), len(radii)) for continuous pixel p of A matched into B; NaN where invalid."""
    hs = np.asarray(hs, float)
    rmax = max(radii)
    gy, gx = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1].astype(np.float64)
    out = np.full((len(hs), len(radii)), np.nan)
    chans = [0, 1, 2, 3] if use_grad else [0, 1, 2]
    templates = {}                     # (round(stepA, 2), LA) -> template of the first height with that key
    groups = {}                        # (template key, LB) -> height indices
    for i, h in enumerate(hs):
        s, ok = local_scale(A, B, p, h)
        if not ok or not np.isfinite(s) or s <= 0:
            continue
        if s >= 1:
            step, la = 1.0, 0
            lb = int(np.clip(np.floor(np.log2(s) + 0.25), 0, len(B.pyr) - 1))
        else:
            step = 1.0 / s
            la = int(np.clip(np.floor(np.log2(step) + 0.25), 0, len(A.pyr) - 1))
            lb = 0
        key = (round(step, 2), la)
        if key not in templates:
            xa = p[0] + gx * step
            ya = p[1] + gy * step
            ta, va = bilinear(A.pyr[la], (xa - 0.5) / 2 ** la, (ya - 0.5) / 2 ** la)
            templates[key] = (xa, ya, ta[..., chans], va)
        groups.setdefault((key, lb), []).append(i)
    weights = {}
    for r in radii:
        sl = slice(rmax - r, rmax + r + 1)
        weights[r] = np.exp(-(gx[sl, sl] ** 2 + gy[sl, sl] ** 2) / (2 * (r / 2.0) ** 2))
    for (key, lb), index in groups.items():
        index = np.asarray(index)
        xa, ya, ta, va = templates[key]
        h = hs[index][:, None, None]
        X, okp = plane_points(A.K, A.T, xa[None], ya[None], h)
        xb, yb, zb = project(B.K, B.T, X)
        tb, vb = bilinear(B.pyr[lb], (xb - 0.5) / 2 ** lb, (yb - 0.5) / 2 ** lb)
        tb = tb[..., chans]
        valid = va[None] & vb & okp & (zb > 0.02)
        if B.mask is not None:
            xi = np.clip(xb.astype(int), 0, B.mask.shape[1] - 1)
            yi = np.clip(yb.astype(int), 0, B.mask.shape[0] - 1)
            valid &= ~B.mask[yi, xi]
        n = len(index)
        for k, r in enumerate(radii):
            sl = slice(rmax - r, rmax + r + 1)
            w0 = weights[r]
            if color_sigma:
                c0 = ta[rmax, rmax, :3]
                w0 = w0 * np.exp(-np.linalg.norm(ta[sl, sl, :3] - c0, axis=-1) / color_sigma)
            vv = valid[:, sl, sl]
            enough = (w0[None] * vv).sum((1, 2)) >= min_valid * w0.sum()
            if not enough.any():
                continue
            w = (w0[None] * vv).reshape(n, -1)
            a = np.broadcast_to(ta[sl, sl], (n,) + ta[sl, sl].shape).reshape(n, -1, ta.shape[-1])
            b = tb[:, sl, sl].reshape(n, -1, tb.shape[-1])
            if census:
                size = 2 * r + 1
                sc = _census(a[..., :3].reshape(n, size, size, 3), b[..., :3].reshape(n, size, size, 3),
                             w.reshape(n, size, size))
            else:
                sc = _zncc(a[..., :3], b[..., :3], w)
                if use_grad:
                    sc = 0.5 * (sc + _zncc(a[..., 3:4], b[..., 3:4], w))
            if color_pen:
                with np.errstate(divide="ignore", invalid="ignore"):     # rows without weight are dropped below
                    sc = sc - color_pen * ((w[..., None] * np.abs(a[..., :3] - b[..., :3])).sum((1, 2))
                                           / (3 * w.sum(1) * 255.0))
            out[index[enough], k] = sc[enough]
    return out


def pick(hs, score, excl=EXCLUSION_M):
    """Best height (parabola refinement), its score and the best score at least `excl` away."""
    s = np.where(np.isfinite(score), score, -2.0)
    if (s <= -2).all():
        return None
    i = int(np.argmax(s))
    best = hs[i]
    if 0 < i < len(hs) - 1 and s[i - 1] > -2 and s[i + 1] > -2:
        den = s[i - 1] - 2 * s[i] + s[i + 1]
        if den < 0:
            best = hs[i] + 0.5 * (s[i - 1] - s[i + 1]) / den * (hs[1] - hs[0])
    far = s[np.abs(hs - hs[i]) >= excl]
    second = float(far.max()) if far.size else -2.0
    return {"h": float(best), "peak": float(s[i]), "second": second}


def curve(A, B, p, hs=HS, **options):
    """Mean score over the radii for each height below both cameras (NaN above)."""
    zmax = min(A.T[2, 3], B.T[2, 3]) - CAMERA_CLEARANCE_M
    ok = hs < zmax
    sc = np.full((len(hs), len(options.get("radii", (1,)))), np.nan)
    if ok.any():
        sc[ok] = sweep(A, B, p, hs[ok], **options)
    if sc.shape[1] == 1:
        return sc[:, 0]
    with np.errstate(all="ignore"):
        valid = np.isfinite(sc)
        count = valid.sum(1)
        mean = np.where(valid, sc, 0.0).sum(1) / np.where(count > 0, count, 1)
    return np.where(count > 0, mean, np.nan)


def back_point(A, B, p, h):
    """The matched point's continuous pixel in B, or None if behind B or within LR_BORDER_PX of its border."""
    X = ray_point(A.K, A.T, p, h)
    x, y, z = project(B.K, B.T, X[None])
    if not z[0] > 0.02:
        return None
    hb, wb = B.shape
    pb = (float(x[0]), float(y[0]))
    if not (LR_BORDER_PX <= pb[0] < wb - LR_BORDER_PX and LR_BORDER_PX <= pb[1] < hb - LR_BORDER_PX):
        return None
    return pb


def lr_check(A, B, p, h, options, hs=HS):
    """Height found by sweeping back from the matched point in B along B's ray into A (None if not possible)."""
    pb = back_point(A, B, p, h)
    if pb is None:
        return None
    pk = pick(hs, curve(B, A, pb, hs, **options))
    return None if pk is None else pk["h"]


def match_height(A, B, p, hs=HS, order=RULE_ORDER):
    """The matching rule (module docstring) for pixel p (continuous) of A matched into B.

    Returns {'status': 'match' | 'uncertain' | 'none', 'h', 'variant', 'peak', 'second', 'margin', 'lr_h',
    'variants': {name: ...}}. 'none': no height had a valid score (the point is outside B, hidden by the mask
    or behind a camera)."""
    tried = {}
    for name in order:
        options = VARIANTS[name]
        pk = pick(hs, curve(A, B, p, hs, **options))
        if pk is None:
            tried[name] = None
            continue
        pk["margin"] = pk["peak"] - pk["second"]
        pk["lr_h"] = lr_check(A, B, p, pk["h"], options, hs)
        pk["consistent"] = pk["lr_h"] is not None and abs(pk["lr_h"] - pk["h"]) <= LR_TOLERANCE_M
        pk["passed"] = bool(pk["consistent"] and pk["margin"] >= MARGIN_GATE)
        tried[name] = pk
        if pk["passed"]:
            return dict(pk, status="match", variant=name, variants=tried)
    found = [(name, pk) for name, pk in tried.items() if pk is not None]
    if not found:
        return {"status": "none", "h": None, "variant": None, "variants": tried}
    # Best guess: the variant whose two-way height agrees best, else the largest margin.
    def quality(item):
        pk = item[1]
        disagreement = abs(pk["lr_h"] - pk["h"]) if pk["lr_h"] is not None else np.inf
        return (disagreement, -pk["margin"])
    name, pk = min(found, key=quality)
    return dict(pk, status="uncertain", variant=name, variants=tried)
