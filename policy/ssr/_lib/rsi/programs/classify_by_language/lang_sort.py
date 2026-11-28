from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ...planning.kinematics import TABLE_Z
from ...perception import head_background as TT
from ...perception import sort_scene as SS
from ...perception.cameras import HEAD_POS, Camera

PRIORS = {
    "watch": [(0.046, 0.042, 0.027), (0.055, 0.047, 0.021), (0.046, 0.041, 0.025), (0.046, 0.037, 0.024)],
    "car": [
        (0.089, 0.042, 0.026),
        (0.098, 0.042, 0.038),
        (0.1, 0.044, 0.026),
        (0.086, 0.04, 0.025),
        (0.076, 0.05, 0.048),
        (0.084, 0.037, 0.041),
    ],
    "wooden_toy": [
        (0.076, 0.043, 0.072),
        (0.056, 0.05, 0.065),
        (0.067, 0.031, 0.065),
        (0.07, 0.049, 0.056),
        (0.054, 0.036, 0.077),
    ],
    "pepper": [(0.053, 0.052, 0.061), (0.046, 0.046, 0.064), (0.051, 0.049, 0.067)],
    "chocolate_bar": [
        (0.12, 0.024, 0.014),
        (0.097, 0.029, 0.013),
        (0.1, 0.031, 0.012),
        (0.066, 0.037, 0.009),
        (0.093, 0.02, 0.01),
    ],
}
CATEGORIES = tuple(PRIORS)
NADIR = np.array([HEAD_POS[0], HEAD_POS[1]])
CAM_H = HEAD_POS[2] - TABLE_Z
MIN_PX = 60
COUNT_RANGE = 2, 3
COUNT_PENALTY = 6.0
CLIP = -20.0
SEG_PENALTY = 30.0
SEG_COVER = 0.5
SEG_IGNORE = ("basket",)
SEG_COMPLETE_COVER = 0.3
SEG_COMPLETE_MIN_PX = 60


@dataclass
class Item:
    region: TT.Region
    feats: dict
    scores: dict = field(default_factory=dict)
    fits: dict = field(default_factory=dict)
    category: str | None = None
    id: int = 0
    seg_cat: str | None = None
    seg_score: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return self.region.xy

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "cat": self.category,
            "seg": self.seg_cat,
            "seg_score": round(float(self.seg_score), 3),
            "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
            "feats": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.feats.items()},
        }
        d.update(self.region.as_dict())
        return d


SHAPES = {"watch": "ellipse", "pepper": "ellipse", "car": "box", "wooden_toy": "box", "chocolate_bar": "box"}
EXT_SIGMA = 0.008
FILL_SIGMA = 0.12
YAWS = np.radians(np.arange(0.0, 180.0, 7.5))
PX = 0.004


def radial_features(cam: Camera, mask: np.ndarray) -> dict | None:
    import cv2

    rows, cols = np.nonzero(mask)
    if len(rows) < 5:
        return None
    uv = np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)
    p = cam.on_plane_z(uv, TABLE_Z)[:, :2]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 5:
        return None
    c = np.median(p, axis=0)
    r = c - NADIR
    rn = float(np.linalg.norm(r))
    if rn < 1e-06:
        return None
    rh = r / rn
    ah = np.array([-rh[1], rh[0]])
    dist = (p - NADIR) @ rh
    acr = (p - NADIR) @ ah
    d_near, d_far = float(np.percentile(dist, 1)), float(np.percentile(dist, 99))
    a0, a1 = float(np.percentile(acr, 1)), float(np.percentile(acr, 99))
    area = float(cv2.contourArea(cv2.convexHull(p.astype(np.float32))))
    return {
        "d_near": d_near,
        "d_far": d_far,
        "e_al": d_far - d_near + PX,
        "e_ac": a1 - a0 + PX,
        "fill": area / max(1e-06, (d_far - d_near) * (a1 - a0)),
        "dir": math.atan2(rh[1], rh[0]),
        "a_mid": 0.5 * (a0 + a1),
    }


def geometry_fit(f: dict, dims, shape: str = "box") -> tuple[float, float]:
    L, W, h = dims
    k = 1.0 - h / CAM_H
    e_al = f["d_far"] * k - f["d_near"]
    e_ac = f["e_ac"] * k
    c, s = np.abs(np.cos(YAWS)), np.abs(np.sin(YAWS))
    if shape == "box":
        ac, al = L * c + W * s, L * s + W * c
        base = L * W
    else:
        ac, al = np.hypot(L * c, W * s), np.hypot(L * s, W * c)
        base = math.pi * L * W / 4.0
    sweep = max(0.0, f["e_al"] - PX - e_al)
    fill = (base + sweep * ac) / np.maximum(1e-06, ac * (al + sweep))
    err = ((ac - e_ac) / EXT_SIGMA) ** 2 + ((al - e_al) / EXT_SIGMA) ** 2 + ((fill - f["fill"]) / FILL_SIGMA) ** 2
    j = int(np.argmin(err))
    return float(err[j]), float(YAWS[j])


def colour_score(cat: str, col: dict) -> float:
    sat, chroma, hue = col.get("sat", 0.0), col.get("chroma", 0.0), col.get("hue", 0.0)
    s = 0.0
    if cat == "pepper":
        s -= max(0.0, (0.72 - sat) / 0.08) ** 2 + max(0.0, (50.0 - chroma) / 8.0) ** 2
    elif cat == "wooden_toy":
        s -= max(0.0, (sat - 0.72) / 0.06) ** 2 + max(0.0, (10.0 - chroma) / 3.0) ** 2
        s -= max(0.0, (18.0 - hue) / 4.0) ** 2 + max(0.0, (hue - 48.0) / 4.0) ** 2 if hue < 300 else 9.0
    elif cat == "watch":
        s -= max(0.0, (chroma - 24.0) / 6.0) ** 2
    return float(s)


def category_scores(cam: Camera, reg: TT.Region, cats) -> tuple[dict, dict, dict]:
    feats = radial_features(cam, TT.fill_holes(reg.mask))
    scores, fits = {}, {}
    if feats is None:
        return {c: CLIP for c in cats}, {}, {}
    for c in cats:
        best = 1000000000.0, 0, 0.0
        for i, dims in enumerate(PRIORS[c]):
            e, ang = geometry_fit(feats, dims, SHAPES[c])
            if e < best[0]:
                best = e, i, ang
        e, i, ang = best
        scores[c] = -0.5 * e + colour_score(c, reg.color)
        fits[c] = {"inst": i, "ang": ang, "dims": PRIORS[c][i], "geo": round(-0.5 * e, 2)}
    return scores, fits, feats


def base_pose(it: Item, cat: str | None = None) -> tuple[np.ndarray, float, tuple]:
    cat = cat or it.category
    f, fit = it.feats, it.fits.get(cat)
    if not f or fit is None:
        return np.asarray(it.xy, float), float(it.region.yaw), PRIORS[cat][0] if cat in PRIORS else (0.05, 0.05, 0.05)
    L, W, h = fit["dims"]
    k = 1.0 - h / CAM_H
    rh = np.array([math.cos(f["dir"]), math.sin(f["dir"])])
    ah = np.array([-rh[1], rh[0]])
    along = 0.5 * (f["d_near"] + f["d_far"] * k)
    across = f["a_mid"] * (1.0 - 0.5 * h / CAM_H)
    xy = NADIR + rh * along + ah * across
    a_ac = math.atan2(ah[1], ah[0])
    cands = [a_ac + fit["ang"], a_ac - fit["ang"]]
    yaw = min(cands, key=lambda a: abs(math.remainder(a - it.region.yaw, math.pi)))
    return xy, float(math.remainder(yaw, math.pi)), (L, W, h)


def assign(items: list[Item], cats, count_range=COUNT_RANGE) -> float:
    n = len(items)
    if n == 0:
        return 0.0
    S = np.array(
        [
            [
                max(CLIP, it.scores.get(c, CLIP)) - (SEG_PENALTY if it.seg_cat in cats and c != it.seg_cat else 0.0)
                for c in cats
            ]
            for it in items
        ]
    )
    best_total, best_lab = -1e18, None
    if n <= 11:
        for lab in itertools.product(range(len(cats)), repeat=n):
            counts = np.bincount(lab, minlength=len(cats))
            pen = np.sum(np.maximum(0, count_range[0] - counts) + np.maximum(0, counts - count_range[1]))
            tot = float(S[np.arange(n), lab].sum()) - COUNT_PENALTY * float(pen)
            if tot > best_total:
                best_total, best_lab = tot, lab
    else:
        best_lab = tuple(int(j) for j in np.argmax(S, axis=1))
        best_total = float(S[np.arange(n), best_lab].sum())
    for it, j in zip(items, best_lab):
        it.category = cats[j]
    return best_total


def split_judge(cam: Camera, cats):
    def best(reg) -> float:
        sc, _, _ = category_scores(cam, reg, cats)
        return max(max(sc.values()), CLIP)

    def judge(whole, pieces) -> bool:
        sw = best(whole)
        sp = [best(p) for p in pieces]
        return min(sp) > -8.0 and float(np.mean(sp)) > sw + 2.0

    return judge


def detect(
    img: np.ndarray,
    cam: Camera,
    cats,
    q_left=None,
    q_right=None,
    g_left: float = 1.0,
    g_right: float = 1.0,
    baskets=None,
    count_range=COUNT_RANGE,
    instances=None,
    region=None,
) -> tuple[list, list[Item]]:
    if baskets is None:
        baskets = SS.detect_baskets(img, cam)
    insts = None
    if instances is not None:
        insts = [i for i in instances if i.category not in SEG_IGNORE]
    regs = SS.detect_objects(
        img,
        cam,
        q_left,
        q_right,
        g_left,
        g_right,
        baskets=baskets,
        split=split_judge(cam, cats),
        instances=insts or None,
        seg_drop_unseen=insts is not None,
        region=region,
    )
    if insts is not None and (not insts):
        regs = []
    if insts:
        regs = list(regs) + seg_completion(img, cam, regs, insts, baskets, q_left, q_right, g_left, g_right, region)
    items = []
    for r in regs:
        if r.area < MIN_PX:
            continue
        sc, fits, feats = category_scores(cam, r, cats)
        it = Item(region=r, feats=feats or {}, scores=sc, fits=fits, id=len(items))
        if insts:
            seg_evidence(it, insts)
        items.append(it)
    assign(items, list(cats), count_range)
    return baskets, items


def seg_completion(
    img, cam: Camera, regs, insts, baskets, q_left=None, q_right=None, g_left=1.0, g_right=1.0, region=None
) -> list:
    import cv2

    valid = SS.robot_valid(cam, img.shape, q_left, q_right, g_left, g_right)
    near_robot = ~cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    masks = [np.asarray(r.mask, bool) for r in regs]
    out = []
    for inst in insts:
        if inst.category not in PRIORS:
            continue
        m = np.asarray(inst.mask, bool) & valid
        a = int(np.count_nonzero(m))
        if a < SEG_COMPLETE_MIN_PX:
            continue
        grown = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        covered = False
        for rm in masks:
            inter = int(np.count_nonzero(rm & m))
            n_r = max(1, int(np.count_nonzero(rm)))
            if inter >= SEG_COMPLETE_COVER * a or np.count_nonzero(rm & grown) >= 0.5 * n_r:
                covered = True
                break
        if covered:
            continue
        reg = SS.make_region(img, cam, m, m, near_robot, baskets, region)
        if reg is None:
            continue
        reg.extra["seg_completed"] = int(a)
        out.append(reg)
        masks.append(np.asarray(reg.mask, bool))
    return out


def seg_evidence(it: Item, insts) -> None:
    m = np.asarray(it.region.mask, bool)
    a = max(1, int(np.count_nonzero(m)))
    best, cover = None, 0.0
    for inst in insts:
        inter = int(np.count_nonzero(m & inst.mask))
        if not inter:
            continue
        c = max(inter / a, inter / max(1, inst.area))
        if c > cover:
            best, cover = inst, c
    if best is None or cover < SEG_COVER:
        return
    it.seg_cat, it.seg_score = best.category, float(best.score)
