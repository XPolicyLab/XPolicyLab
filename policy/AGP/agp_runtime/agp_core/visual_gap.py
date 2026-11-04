"""The finger pad gap of an X5 gripper, estimated from that arm's wrist RGB image.

Under official leaderboard conditions the observation has no finger joints, only the last gripper command, so the
opening of the fingers has to be read from the image. This module uses nothing but the RGB image and constants of
the public RoboDojo files:

    Assets/Robots/x5/X5A.urdf            the finger joints (origin, axis, limits) and their link meshes
    Assets/Robots/x5/meshes/link7.STL,   the finger meshes (the composed USD uses the same geometry)
                            link8.STL
    Assets/Robots/x5/robot_config.yml    gripper_joints_name, ee_link and the wrist camera mount
    the public camera model              wrist intrinsics and the fixed link6 -> optical transform
                                         (agp_core/public_camera.py, verified against the simulator)

The wrist camera is rigid on link6 and each finger slides along link6 +-y, so the silhouette of a finger in the
wrist image depends only on its own prismatic joint. FingerSilhouettes renders, for every finger and every joint
value on a 0.25 mm grid, the projected mesh (rasterised at 4x), extracts its outer boundary, resamples it every
pixel with outward normals and keeps the points inside the image. GapEstimator scores an image per finger and grid
value by the mean colour contrast across the predicted boundary (|colour 1.5 px outside - colour 1.5 px inside|,
clipped at 60 per point), takes the best grid value per finger and refines it with a parabola. Both fingers are
estimated independently (one can be blocked while the other closes).

Nominal pad gap = q7 + q8 + pad separation at q = 0 (the facing pad planes of the two meshes, 0.807 mm), the same
quantity as gripper_geometry.nominal_pad_gap_m (migration/robot_geometry.py).

Quality: a finger's estimate is "ok" when its best score is >= SCORE_MIN (enough contrast along its silhouette:
it is visible) and no other joint value 2 mm or more away scores within MARGIN_MIN of it (the peak is unique);
the gap is "ok" only when both fingers are. A finger hidden by an object or the other arm, a dark scene (inside a
box) or a background of the fingers' own colour gives "low". Thresholds were set on a labelled set of recorded wrist
images.

No simulator, depth image, measured joint or camera prim is read: the estimator is a function of an image array.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from agp_core import public_camera as pc

Q_STEP_M = 2.5e-4            # joint grid of the silhouette tables
Q_PAD_M = 1e-3               # grid extends this far beyond the joint limits (edge peaks stay interior)
RENDER_SCALE = 4             # rasterisation supersampling
CANVAS_MARGIN_PX = 48        # silhouettes are rendered on a canvas this much larger than the image
BORDER_PX = 3.0              # boundary points closer than this to the image border are not used
NEAR_M = 0.004               # triangles are clipped at this optical depth
DELTA_PX = 1.5               # sampling offset across the boundary
CONTRAST_CLIP = 60.0         # per-point contrast clip (RGB units)
SCORE_MIN = 25.0             # quality: minimum mean boundary contrast of a finger's best joint value
MARGIN_MIN = 4.0             # quality: best score minus the best score 2 mm or more away
MARGIN_AWAY_M = 2e-3
MIN_POINTS = 40              # quality: minimum boundary points inside the image
CLOSED_EMPTY_GAP_M = 1.5e-3  # a closed gripper with a smaller estimated gap has nothing between its pads
CLOSE_TOLERANCE_M = 3e-3     # migration/worker.py: a gripper action reached its endpoint within 3 mm per finger
SOURCE = ("estimate from this arm's wrist RGB image and the public finger meshes, URDF and camera model "
          "(agp_core/visual_gap.py); not a measurement")


# ----------------------------------------------------------------------------- public geometry
def read_stl(path):
    """Triangles (n, 3, 3) of a binary or ASCII STL."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    tris, cur = [], []
    for line in data.decode("ascii", errors="replace").splitlines():
        s = line.split()
        if s and s[0] == "vertex":
            cur.append([float(x) for x in s[1:4]])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    if not tris:
        raise ValueError(f"{path}: no triangles")
    return np.asarray(tris, float)


@dataclass(frozen=True)
class Finger:
    joint: str
    link: str
    T_link6_joint: np.ndarray     # joint origin in the ee link (link6) frame
    axis: np.ndarray              # prismatic axis in the joint frame
    lower: float
    upper: float
    triangles: np.ndarray         # (n, 3, 3) in the finger link frame
    mesh: str


def public_fingers(model):
    """The gripper fingers of the public robot description (robot_config gripper_joints_name, URDF)."""
    robot_cfg = yaml.safe_load(Path(model.sources["robot_config"]["path"]).read_text())
    urdf = Path(model.sources["urdf"]["path"])
    root = ET.parse(urdf).getroot()
    meshes = {}
    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        if mesh is not None:
            meshes[link.get("name")] = (urdf.parent / mesh.get("filename")).resolve()
    fingers = []
    for name in robot_cfg["gripper_joints_name"]:
        joint = model._joints[name]
        if joint["type"] != "prismatic" or joint["parent"] != model.ee_link:
            raise NotImplementedError(f"{name}: expected a prismatic finger on {model.ee_link}")
        limit = root.find(f"joint[@name='{name}']/limit")
        fingers.append(Finger(joint=name, link=joint["child"], T_link6_joint=pc._urdf_origin(joint),
                              axis=np.asarray(joint["axis"], float) / np.linalg.norm(joint["axis"]),
                              lower=float(limit.get("lower")), upper=float(limit.get("upper")),
                              triangles=read_stl(meshes[joint["child"]]), mesh=str(meshes[joint["child"]])))
    return fingers


def pad_gap_at_zero(fingers):
    """Separation of the two facing finger faces at q = 0 along the opening direction (link6 frame)."""
    a, b = fingers
    def extent(f):
        pts = f.triangles.reshape(-1, 3) @ f.T_link6_joint[:3, :3].T + f.T_link6_joint[:3, 3]
        direction = f.T_link6_joint[:3, :3] @ f.axis          # opening direction of this finger in link6
        return pts @ direction, direction
    sa, da = extent(a)
    sb, db = extent(b)
    if not np.allclose(da, -db, atol=1e-9):
        raise NotImplementedError("fingers do not open along opposite directions")
    return float(sa.min() + sb.min())       # inner faces: smallest coordinate along each finger's opening axis


# ----------------------------------------------------------------------------- silhouette tables
class FingerSilhouettes:
    """Boundary points and outward normals of each finger's silhouette in the wrist image, per joint value."""

    def __init__(self, model=None, q_step=Q_STEP_M):
        import cv2   # noqa: F401  (fail early when OpenCV is missing)
        self.model = model if model is not None else pc.default_model()
        m = self.model
        self.fingers = public_fingers(m)
        wrist = [name for name in m.intrinsics if name != "cam_head"]
        k = m.intrinsics[wrist[0]]
        if any(m.intrinsics[n].K.tolist() != k.K.tolist() for n in wrist):
            raise NotImplementedError("the two wrist cameras differ")
        self.width, self.height = k.width, k.height
        self.K = k.K
        self.T_optical_link6 = pc.inverse(m.T_link6_optical)
        self.gap0 = pad_gap_at_zero(self.fingers)
        lo = min(f.lower for f in self.fingers) - Q_PAD_M
        hi = max(f.upper for f in self.fingers) + Q_PAD_M
        self.q = np.round(np.arange(lo, hi + q_step / 2, q_step), 9)
        self.tables = [self._table(f) for f in self.fingers]
        self.sources = {"urdf": m.sources["urdf"], "robot_config": m.sources["robot_config"],
                        **{f"mesh_{f.link}": {"path": f.mesh, "sha256": hashlib.sha256(
                            Path(f.mesh).read_bytes()).hexdigest()} for f in self.fingers},
                        "camera_model": {k: v for k, v in m.sources.items()
                                         if k in ("camera_config", "template", "camera_manager", "rotations")}}

    def triangles_optical(self, finger, q):
        P = finger.triangles.reshape(-1, 3) + finger.axis * q
        P = P @ finger.T_link6_joint[:3, :3].T + finger.T_link6_joint[:3, 3]
        P = P @ self.T_optical_link6[:3, :3].T + self.T_optical_link6[:3, 3]
        return P.reshape(-1, 3, 3)

    def _project(self, P):
        return np.stack([self.K[0, 0] * P[..., 0] / P[..., 2] + self.K[0, 2],
                         self.K[1, 1] * P[..., 1] / P[..., 2] + self.K[1, 2]], -1)

    def polygons(self, finger, q):
        """Projected triangles in continuous image coordinates (pixel i spans [i, i+1)), clipped at NEAR_M."""
        T = self.triangles_optical(finger, q)
        z = T[..., 2]
        full = (z >= NEAR_M).all(1)
        polys = list(self._project(T[full]))
        for tri in T[~full & (z >= NEAR_M).any(1)]:
            out = []
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                if a[2] >= NEAR_M:
                    out.append(a)
                if (a[2] >= NEAR_M) != (b[2] >= NEAR_M):
                    out.append(a + (NEAR_M - a[2]) / (b[2] - a[2]) * (b - a))
            polys.append(self._project(np.asarray(out)))
        return polys

    def mask(self, finger, q):
        import cv2
        s, mg = RENDER_SCALE, CANVAS_MARGIN_PX
        mk = np.zeros(((self.height + 2 * mg) * s, (self.width + 2 * mg) * s), np.uint8)
        for c in self.polygons(finger, q):
            pts = np.clip(np.round(((c + mg) * s - 0.5) * 16), -2 ** 22, 2 ** 22).astype(np.int32)
            cv2.fillConvexPoly(mk, pts, 255, lineType=cv2.LINE_8, shift=4)
        return mk

    def boundary(self, mk, spacing=1.0):
        """Silhouette boundary in pixel-index coordinates (pixel centre at integers) and outward unit normals."""
        import cv2
        s, mg = RENDER_SCALE, CANVAS_MARGIN_PX
        contours, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points, normals = [], []
        for cnt in contours:
            X = cnt[:, 0, :].astype(float)
            if len(X) < 20:
                continue
            C = (X + 0.5) / s - mg                     # continuous image coordinates of boundary pixel centres
            closed = np.vstack([C, C[:1]])
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
            n = max(8, int(arc[-1] / spacing))
            t = np.linspace(0.0, arc[-1], n, endpoint=False)
            P = np.stack([np.interp(t, arc, closed[:, 0]), np.interp(t, arc, closed[:, 1])], -1)
            T = np.roll(P, -2, 0) - np.roll(P, 2, 0)
            T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
            N = np.stack([T[:, 1], -T[:, 0]], -1)
            probe = np.round((P + 2.0 * N + mg) * s - 0.5).astype(int)
            probe[:, 0] = probe[:, 0].clip(0, mk.shape[1] - 1)
            probe[:, 1] = probe[:, 1].clip(0, mk.shape[0] - 1)
            if (mk[probe[:, 1], probe[:, 0]] > 0).mean() > 0.5:
                N = -N
            points.append(P + N * (0.5 / s) - 0.5)     # edge half a fine pixel outside; pixel-index coordinates
            normals.append(N)
        if not points:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.vstack(points), np.vstack(normals)

    def _table(self, finger):
        pts, nrm, seg = [], [], [0]
        b = BORDER_PX
        for q in self.q:
            P, N = self.boundary(self.mask(finger, q))
            keep = ((P[:, 0] >= b) & (P[:, 0] <= self.width - 1 - b)
                    & (P[:, 1] >= b) & (P[:, 1] <= self.height - 1 - b))
            pts.append(P[keep])
            nrm.append(N[keep])
            seg.append(seg[-1] + int(keep.sum()))
        P = np.vstack(pts).astype(np.float32)
        N = np.vstack(nrm).astype(np.float32)
        return {"inside": P - DELTA_PX * N, "outside": P + DELTA_PX * N, "seg": np.asarray(seg),
                "counts": np.diff(np.asarray(seg))}


# ----------------------------------------------------------------------------- estimation
def _sample(image, points, width=1024):
    """Bilinear samples (n, channels) of image at pixel-index coordinates (OpenCV remap, row width <= SHRT_MAX)."""
    import cv2
    n = len(points)
    rows = max(1, -(-n // width))
    pad = rows * width - n
    mx = np.pad(points[:, 0], (0, pad)).reshape(rows, width).astype(np.float32)
    my = np.pad(points[:, 1], (0, pad)).reshape(rows, width).astype(np.float32)
    out = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * width, -1)[:n]


class GapEstimator:
    """Pad gap and finger joints from one wrist RGB image (see the module docstring)."""

    def __init__(self, silhouettes=None):
        self.sil = silhouettes if silhouettes is not None else FingerSilhouettes()

    def scores(self, image, index):
        """Mean clipped boundary contrast of finger `index` for every joint value of the grid."""
        tb = self.sil.tables[index]
        a = _sample(image, tb["inside"])
        b = _sample(image, tb["outside"])
        c = np.minimum(np.linalg.norm(b - a, axis=1), CONTRAST_CLIP)
        sums = np.add.reduceat(np.r_[c, 0.0], tb["seg"][:-1])
        return np.where(tb["counts"] > 0, sums / np.maximum(tb["counts"], 1), 0.0)

    def _finger(self, image, index):
        f = self.sil.fingers[index]
        q = self.sil.q
        S = self.scores(image, index)
        k = int(np.argmax(S))
        best = float(S[k])
        estimate = float(q[k])
        if 0 < k < len(S) - 1:
            d = S[k - 1] - 2 * S[k] + S[k + 1]
            if d < 0:
                estimate += float(np.clip(0.5 * (S[k - 1] - S[k + 1]) / d, -0.5, 0.5)) * (q[1] - q[0])
        away = np.abs(q - q[k]) >= MARGIN_AWAY_M
        margin = best - float(S[away].max()) if away.any() else best
        points = int(self.sil.tables[index]["counts"][k])
        ok = best >= SCORE_MIN and margin >= MARGIN_MIN and points >= MIN_POINTS
        return {"joint": f.joint, "q_m": float(np.clip(estimate, f.lower, f.upper)), "q_raw_m": estimate,
                "score": round(best, 2), "margin": round(margin, 2), "points": points,
                "quality": "ok" if ok else "low"}

    def estimate(self, image):
        """dict with pad_gap_m, finger_joints_m, quality ("ok" | "low") and per-finger details."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[0] != self.sil.height or image.shape[1] != self.sil.width \
                or image.shape[2] < 3:
            raise ValueError(f"expected a {self.sil.height}x{self.sil.width} RGB image, got {image.shape}")
        img = np.ascontiguousarray(image[..., :3], dtype=np.float32)
        fingers = [self._finger(img, i) for i in range(len(self.sil.fingers))]
        q = [f["q_m"] for f in fingers]
        quality = "ok" if all(f["quality"] == "ok" for f in fingers) else "low"
        return {"pad_gap_m": float(sum(q) + self.sil.gap0), "finger_joints_m": q, "quality": quality,
                "fingers": fingers}


@lru_cache(maxsize=None)
def default_estimator():
    return GapEstimator()


def estimate(image):
    return default_estimator().estimate(image)


# ----------------------------------------------------------------------------- what the worker reports
def closed_empty(result, command):
    """True / False after a close (command < 0.5) with an ok estimate; None otherwise."""
    if result is None or result.get("quality") != "ok" or command is None or float(command) >= 0.5:
        return None
    return bool(result["pad_gap_m"] < CLOSED_EMPTY_GAP_M)


def observation_entry(image, command):
    """The per-arm observation field: the estimate, its quality and, after a close, closed_empty."""
    try:
        r = estimate(image)
    except Exception as exc:          # an unusable image never breaks the observation
        return {"estimate": True, "source": SOURCE, "pad_gap_m": None, "finger_joints_m": None,
                "quality": "unavailable", "error": f"{type(exc).__name__}: {exc}", "closed_empty": None}
    return {"estimate": True, "source": SOURCE, "pad_gap_m": round(r["pad_gap_m"], 5),
            "finger_joints_m": [round(v, 5) for v in r["finger_joints_m"]], "quality": r["quality"],
            "closed_empty": closed_empty(r, command),
            "fingers": [{k: f[k] for k in ("joint", "score", "margin", "quality")} for f in r["fingers"]]}


def close_outcome(result, endpoint):
    """Reached / not reached of a gripper action from an estimate, with the rule of migration/worker.py
    (every finger within CLOSE_TOLERANCE_M of its reachable endpoint); None when the estimate is not ok."""
    if result is None or result.get("quality") != "ok":
        return None
    error = float(np.max(np.abs(np.asarray(result["finger_joints_m"]) - np.asarray(endpoint, float))))
    return {"reached": bool(error < CLOSE_TOLERANCE_M), "gripper_error_m": error}
