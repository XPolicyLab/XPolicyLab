"""table_plane: world z of the table top, measured once per episode from the three RGB images.

Plane sweep. For a candidate height c, every pixel of a reference camera is cast as a ray onto the horizontal plane
z = c and the plane point is projected into a source camera; intrinsics and extrinsics come from the observation (head:
fixed calibration, wrists: arm kinematics). At the true table height the warped source image lines up with the
reference image wherever the table is seen, so local normalised cross-correlation (NCC, 9 px windows, textured pixels
only, plane points inside the RoboShell xy workspace) peaks there. Candidates span 0.50..1.05 m in 4 mm steps, then
+-6 mm around the best step in 0.5 mm steps with a parabolic peak. Per camera pair: z and
contrast = peak NCC - median NCC over the coarse sweep. Full resolution, about 1-1.5 s single-thread CPU for the
three pairs.

The pairs head<-wrist_l, head<-wrist_r, wrist_l<-wrist_r are merged by consensus: the largest group of pair estimates
(contrast > 0.08) within 4 mm of one member gives the median, if it has at least two pairs; otherwise one pair counts
only if its contrast is above 0.30; otherwise table_z is None (unknown), never a guess. conf = summed contrast of the
pairs used / 1.5, at most 1 (about 1 for a textured table seen by all pairs). No height constant is used anywhere.

Receipt (also the output of `robo table_plane`):
    {"table_z": <m> | null, "conf": <0..1>, "n_agree": <0..3>,
     "pairs": {"head<-wrist_l": {"z": <m> | null, "contrast": <ncc> | null, "used": <bool>}, ... one entry per pair},
     "source": "rgb plane sweep (head/wrist pairs)", "cpu_s": <s>, "cached": <bool>,
     "reason": <text, only when table_z is null>, "note": <text, only when a refresh was inconclusive>}

Per episode: a measured value is kept for the rest of the episode (the cache sits on the episode's arm objects, which a
reset replaces, and is shared by every module that loads this file). An unknown result is measured again once an arm
has moved (a new wrist view); refresh=True (`--refresh 1`) always measures again from the current images, and an
inconclusive re-measurement does not discard a measured value.

Use from a task tool (tasks/<task>/tools/<name>/tool.py):

    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location(
        "table_plane", Path(__file__).resolve().parents[4] / "tools" / "table_plane" / "tool.py")
    table_plane = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(table_plane)

    table_z = table_plane.measure_table_z(api)   # metres, or None when unknown: handle it, do not substitute a number
    receipt = table_plane.table_plane(api)       # the full receipt above

"""
import time
from statistics import median

import cv2
import numpy as np

TOOL = {
    "name": "table_plane",
    "commands": [
        {"name": "table_plane", "budget": False, "help": "measure the world z of the table top from the camera images",
         "args": [
             {"name": "refresh", "type": "int", "default": 0, "choices": [0, 1],
              "help": "1: measure again from the current images instead of returning the value kept for this episode"},
         ]},
    ],
}

SOURCE = "rgb plane sweep (head/wrist pairs)"
CAMERAS = {"cam_head": "head", "cam_left_wrist": "wrist_l", "cam_right_wrist": "wrist_r"}   # RoboShell camera names
PAIRS = (("head", "wrist_l"), ("head", "wrist_r"), ("wrist_l", "wrist_r"))                # (reference, source)
SWEEP_M = (0.50, 1.05)          # candidate heights: a wide band, no prior on where the table is
COARSE_STEP_M, FINE_STEP_M, FINE_SPAN_M = 0.004, 0.0005, 0.006
WORK_X_M, WORK_Y_M = (-0.75, 0.75), (-0.75, 0.60)   # RoboShell workspace in xy: only plane points on the work surface
WIN = (9, 9)                    # NCC window (px)
MIN_VAR = 4.0                   # local grey variance below which a pixel is untextured (gripper black, glare)
MIN_NCC_PX = 3000               # pixels a candidate needs to be scored
MIN_CONTRAST, MIN_PAIR_PX, AGREE_M, SINGLE_CONTRAST = 0.08, 5000, 0.004, 0.30
CONF_FULL = 1.5                 # summed contrast that counts as full confidence
CACHE_ATTR = "table_plane_cache"
EXIT_FAILED = 2                 # RoboShell contract: the command failed


# ---- observation -> views --------------------------------------------------------------------------------------
def _view(gray, intrinsics, extrinsics):
    """Grey image + K + camera-to-world T; None if anything is malformed."""
    img = None if gray is None else np.asarray(gray)
    k, t = np.asarray(intrinsics, dtype=float), np.asarray(extrinsics, dtype=float)
    if (img is None or img.ndim != 2 or min(img.shape) < 32 or k.shape != (3, 3) or t.shape != (4, 4)
            or not np.isfinite(k).all() or not np.isfinite(t).all() or k[0, 0] <= 0 or k[1, 1] <= 0):
        return None
    rot = t[:3, :3]
    if abs((rot[0] * np.cross(rot[1], rot[2])).sum() - 1.0) > 1e-2:       # not a rotation
        return None
    return {"img": img.astype(np.float32), "K": k, "T": t}


def views_from_observation(obs):
    """api.observe() -> {head, wrist_l, wrist_r: view}. Only the colour images and the camera matrices are read."""
    views = {}
    for source, name in CAMERAS.items():
        png, cam = (obs.get("png") or {}).get(source), (obs.get("cameras") or {}).get(source)
        if png is None or cam is None:
            continue
        gray = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        view = _view(gray, cam.get("intrinsics"), cam.get("extrinsics_world"))
        if view is not None:
            views[name] = view
    return views


# ---- plane sweep --------------------------------------------------------------------------------------------------
def _rigid_inverse(t):
    """Inverse of a 4x4 rigid transform, element-wise (no LAPACK: its worker threads spin on and burn CPU)."""
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = [-(t[:3, i] * t[:3, 3]).sum() for i in range(3)]
    return out


def _bound(lo, hi, alpha, beta):
    """Intersect the per-pixel candidate interval (lo, hi) with {c : alpha + c beta > 0}."""
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -alpha / beta
    lo = np.where(beta > 0, np.maximum(lo, root), lo)
    hi = np.where(beta < 0, np.minimum(hi, root), hi)
    return np.where((beta == 0) & (alpha <= 0), np.inf, lo), hi


class _Warp:
    """One camera pair: reference pixels cast onto the plane z = c and projected into the source camera.

    For a downward ray the plane point is X(c) = (ax + c bx, ay + c by, c) and its source camera coordinates are affine
    in c too. Every validity condition (plane point inside the workspace, in front of the source camera, inside its
    image) is therefore linear in c, so each pixel is valid on one interval of c, computed once; so is the condition
    that a whole NCC window is valid (window max of the lower ends, window min of the upper ends). A candidate then
    costs one remap, three box filters and a few element-wise operations, all cropped to the bounding box of its
    covered pixels plus the window radius (exact: no window that is scored crosses the crop)."""

    def __init__(self, ref, src):
        self.a, self.b, kr, ks = ref["img"], src["img"], ref["K"], src["K"]
        hs, ws = self.b.shape
        h, w = self.a.shape
        rot, org = ref["T"][:3, :3], ref["T"][:3, 3]
        self.top = float(org[2])                                  # the plane must lie below the reference camera
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        x, y = (u - kr[0, 2]) / kr[0, 0], (v - kr[1, 2]) / kr[1, 1]
        d = [rot[i, 0] * x + rot[i, 1] * y + rot[i, 2] for i in range(3)]     # ray directions in the world frame
        down = d[2] < -1e-3
        dz = np.where(down, d[2], -1.0)
        bx, by = d[0] / dz, d[1] / dz
        ax, ay = org[0] - org[2] * bx, org[1] - org[2] * by
        e = _rigid_inverse(src["T"])
        p = [e[i, 0] * ax + e[i, 1] * ay + e[i, 3] for i in range(3)]        # source camera coordinates = p + c q
        q = [e[i, 0] * bx + e[i, 1] * by + e[i, 2] for i in range(3)]
        pu, qu = ks[0, 0] * p[0] + ks[0, 2] * p[2], ks[0, 0] * q[0] + ks[0, 2] * q[2]   # homogeneous source pixel
        pv, qv = ks[1, 1] * p[1] + ks[1, 2] * p[2], ks[1, 1] * q[1] + ks[1, 2] * q[2]
        lo, hi = np.where(down, -np.inf, np.inf), np.full((h, w), np.inf)
        for alpha, beta in ((ax - WORK_X_M[0], bx), (WORK_X_M[1] - ax, -bx), (ay - WORK_Y_M[0], by), (WORK_Y_M[1] - ay, -by),
                            (p[2] - 0.05, q[2]),                                          # in front of the source camera
                            (pu, qu), ((ws - 1) * p[2] - pu, (ws - 1) * q[2] - qu),       # 0 <= u < ws - 1
                            (pv, qv), ((hs - 1) * p[2] - pv, (hs - 1) * q[2] - qv)):      # 0 <= v < hs - 1
            lo, hi = _bound(lo, hi, alpha, beta)
        f32 = lambda arr: np.ascontiguousarray(arr, dtype=np.float32)        # noqa: E731
        self.lo, self.hi = f32(lo), f32(hi)
        self.pu, self.qu, self.pv, self.qv, self.pw, self.qw = map(f32, (pu, qu, pv, qv, p[2], q[2]))
        kernel = np.ones(WIN, np.uint8)
        self.ma = cv2.blur(self.a, WIN)
        self.va = cv2.blur(self.a * self.a, WIN) - self.ma * self.ma
        # window fully valid and reference pixel textured <=> win_lo < c < win_hi
        self.win_lo = np.where(self.va > MIN_VAR, cv2.dilate(self.lo, kernel), np.float32(np.inf))
        self.win_hi = cv2.erode(self.hi, kernel)

    def score(self, c):
        """(mean NCC over the textured, fully covered pixels, their count) for the plane z = c; NaN if too few."""
        if c >= self.top:
            return np.nan, 0
        c = np.float32(c)
        candidates = (self.win_lo < c) & (self.win_hi > c)
        if np.count_nonzero(candidates) < MIN_NCC_PX:
            return np.nan, 0
        rows, cols = np.flatnonzero(candidates.any(axis=1)), np.flatnonzero(candidates.any(axis=0))
        r = WIN[0] // 2
        crop = (slice(max(rows[0] - r, 0), rows[-1] + r + 1), slice(max(cols[0] - r, 0), cols[-1] + r + 1))
        candidates = candidates[crop]
        valid = (self.lo[crop] < c) & (self.hi[crop] > c)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            iw = 1 / (self.pw[crop] + c * self.qw[crop])
            us, vs = (self.pu[crop] + c * self.qu[crop]) * iw, (self.pv[crop] + c * self.qv[crop]) * iw
        # invalid pixels are never inside a scored window; an in-image fill keeps remap on its fast path
        fill = np.float32(0)
        warped = cv2.remap(self.b, np.where(valid, us, fill), np.where(valid, vs, fill), cv2.INTER_LINEAR)
        mb = cv2.blur(warped, WIN)
        vb = cv2.blur(warped * warped, WIN) - mb * mb
        ok = candidates & (vb > MIN_VAR)
        n = int(np.count_nonzero(ok))
        if n < MIN_NCC_PX:
            return np.nan, n
        a, ma, va = self.a[crop], self.ma[crop], self.va[crop]
        cov = cv2.blur(a * warped, WIN) - ma * mb
        ncc = cov[ok] / np.sqrt(np.maximum(va[ok] * vb[ok], 1e-6))
        return float(ncc.mean(dtype=np.float64)), n


def sweep_pair(ref, src):
    """Plane sweep of one pair -> {z, peak, base, contrast, px} or None when no candidate could be scored."""
    warp = _Warp(ref, src)
    cs = np.arange(SWEEP_M[0], SWEEP_M[1], COARSE_STEP_M)
    s = np.array([warp.score(c)[0] for c in cs])
    if not np.isfinite(s).any():
        return None
    c0 = cs[int(np.nanargmax(s))]
    fs = np.arange(c0 - FINE_SPAN_M, c0 + FINE_SPAN_M, FINE_STEP_M)
    scored = [warp.score(c) for c in fs]
    sf = np.array([value for value, _ in scored])
    if not np.isfinite(sf).any():
        return None
    j = int(np.nanargmax(sf))
    z = float(fs[j])
    if 0 < j < len(fs) - 1 and np.isfinite(sf[j - 1:j + 2]).all():          # parabolic refinement of the peak
        y0, y1, y2 = sf[j - 1:j + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            z += 0.5 * (y0 - y2) / den * FINE_STEP_M
    peak, base = float(sf[j]), float(np.nanmedian(s))
    return {"z": z, "peak": peak, "base": base, "contrast": peak - base, "px": scored[j][1]}


def sweep_pairs(views):
    """{'head<-wrist_l': sweep result or None, ...} for the pairs whose two cameras are present."""
    return {f"{a}<-{b}": sweep_pair(views[a], views[b]) for a, b in PAIRS if a in views and b in views}


def consensus(pairs):
    """(table_z or None, names of the pairs used): largest group within AGREE_M (>= 2 pairs), else one sharp pair."""
    cands = [(name, p) for name, p in pairs.items()
             if p is not None and p["contrast"] > MIN_CONTRAST and p["px"] > MIN_PAIR_PX]
    best = []
    for _, p0 in cands:
        group = [(name, p) for name, p in cands if abs(p["z"] - p0["z"]) <= AGREE_M]
        if len(group) > len(best) or (len(group) == len(best)
                                      and sum(p["contrast"] for _, p in group) > sum(p["contrast"] for _, p in best)):
            best = group
    if len(best) >= 2:
        return float(median(p["z"] for _, p in best)), [name for name, _ in best]
    sharp = [(name, p) for name, p in cands if p["contrast"] > SINGLE_CONTRAST]
    if sharp:
        name, p = max(sharp, key=lambda item: item[1]["contrast"])
        return float(p["z"]), [name]
    return None, []


class _OneThread:
    """OpenCV single-threaded while measuring: predictable CPU cost inside a busy server process."""

    def __enter__(self):
        self.threads = cv2.getNumThreads()
        cv2.setNumThreads(1)

    def __exit__(self, *exc):
        cv2.setNumThreads(self.threads)


def measure_views(views):
    """Views (see views_from_observation) -> receipt. Pure function, no cache."""
    started = time.thread_time()
    with _OneThread():
        pairs = sweep_pairs(views) if len(views) >= 2 else {}
    z, used = consensus(pairs)
    receipt = {
        "table_z": None if z is None else round(z, 4),
        "conf": 0.0 if z is None else round(min(1.0, sum(pairs[n]["contrast"] for n in used) / CONF_FULL), 2),
        "n_agree": len(used),
        "pairs": {name: {"z": None if p is None else round(p["z"], 4),
                         "contrast": None if p is None else round(p["contrast"], 3), "used": name in used}
                  for name, p in pairs.items()},
        "source": SOURCE,
        "cpu_s": round(time.thread_time() - started, 2),
    }
    if z is None:
        receipt["reason"] = ("fewer than two camera images" if len(views) < 2 else
                             "no two camera pairs agree within 4 mm and no single pair has a sharp peak "
                             "(too little table texture in view)")
    return receipt


# ---- per-episode cache and the robo command ----------------------------------------------------------------------
def _holder(api):
    """An object that lives exactly as long as the episode: its first arm (a reset creates new arm objects)."""
    try:
        return next(iter(api.arms.values()))
    except (AttributeError, StopIteration, TypeError):
        return None


def _arm_poses(api):
    """Rounded TCP poses of both arms: unchanged poses mean unchanged wrist views."""
    try:
        return tuple(np.round(np.asarray(arm.tcp(), dtype=float), 3).tobytes() for arm in api.arms.values())
    except Exception:       # a minimal stand-in api without arms: never reuse an unknown result
        return None


def table_plane(api, refresh=False):
    """Receipt for the current episode: the kept measurement, or a new one from api.observe()."""
    holder = _holder(api)
    kept = getattr(holder, CACHE_ATTR, None) if holder is not None else None
    poses = _arm_poses(api)
    if kept is not None and not refresh and (kept["receipt"]["table_z"] is not None
                                             or (poses is not None and kept["poses"] == poses)):
        return dict(kept["receipt"], cached=True)
    receipt = dict(measure_views(views_from_observation(api.observe())), cached=False)
    if receipt["table_z"] is None and kept is not None and kept["receipt"]["table_z"] is not None:
        return dict(kept["receipt"], cached=True,
                    note="the new measurement from the current images was inconclusive; the earlier value is kept")
    if holder is not None:
        try:
            setattr(holder, CACHE_ATTR, {"receipt": dict(receipt), "poses": poses})
        except (AttributeError, TypeError):
            pass
    return receipt


def measure_table_z(api, refresh=False):
    """World z of the table top in metres for the current episode, or None when it could not be measured."""
    return table_plane(api, refresh)["table_z"]


def run(api, command, args):
    try:
        return table_plane(api, refresh=bool(args.get("refresh") or 0)), 0
    except Exception as error:         # a failed measurement must not break the episode
        return {"table_z": None, "conf": 0.0, "n_agree": 0, "source": SOURCE,
                "reason": f"measurement failed: {type(error).__name__}: {error}"}, EXIT_FAILED
