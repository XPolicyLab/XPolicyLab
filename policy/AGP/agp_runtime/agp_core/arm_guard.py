"""F-g (OURS_ARM_GUARD): clearance between the two X5 arms from public geometry only.

Geometry (no simulator query, no recorded run):
  * links and joints: the public URDF Assets/Robots/x5/X5A.urdf (agp_core/public_camera.py parses it); the arm bases: the
    public base poses the worker already holds (robot.entity_origin_pose of the task's public robot config);
  * shapes: agp_core/x5_link_hulls.json (convex parts of every visual mesh of the public URDF, link frames). The simulator's
    colliders are the convex hull of each link mesh for every link but link3, which is a convex decomposition (ARX.usd,
    read offline from its collider approximations). So every link is one convex
    shape here, the hull of all its parts' vertices (= the hull of the mesh: every face lies inside one part), except
    link3, which keeps its parts. The fingers (link7, link8) are placed at the opening estimated from the official
    observation (see finger_opening); the wrist camera (camera_base, camera) is part of the arm;
  * a held object has no envelope: HELD_RADIUS_M = 0. The guard checks the robot's own shapes only, the fingers
    included. A radius above 0 (sensitivity runs only, ArmGuardGeometry(held_radius=...)) adds that sphere around
    the TCP of a gripper that holds something (gripper commanded closed, fingers apart in the official observation).

Distance: the Gilbert-Johnson-Keerthi algorithm on the convex shapes (exact for convex polytopes; 0 when they overlap),
after a broad phase on bounding spheres. clearance() is the smallest distance between any shape of one arm and any
shape of the other (a sphere's radius is subtracted). The arm's own links are never compared with each other.

The two guard layers that use it are in agp_core/speed_profiles.py (time-synchronised check of a stage before it starts,
per-step check of the measured joints and the next commanded sample).

The declared handover (OURS_HANDOVER; agp_core/handover.py): clearance(..., pairs=f) compares only the pairs of shapes for
which f(link of the first arm's shape, link of the second arm's shape) is true (arms in sorted order, left before
right); the other pairs count as infinitely far apart. Without `pairs` (None, every other caller) every pair is
compared.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HULLS = Path(__file__).with_name("x5_link_hulls.json")
LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "camera_base", "camera", "link7", "link8")
DECOMPOSED = {"link3"}                 # ARX.usd collider approximation convexDecomposition (the others: convexHull)
FINGER_LINKS = ("link7", "link8")
HELD_RADIUS_M = 0.0                    # envelope of a held object: none; > 0 (sensitivity runs only): a sphere of
                                       # this radius
# Official gripper mapping (Assets/Robots/x5/robot_config.yml gripper_scale [-0.01, 0.044]; the observation's
# <arm>_ee_joint_state is the commanded finger target clamped to the measured finger +-20 % of that range
# (env/robot_manager/control_manager.py process_gripper_val, gripper_eps 0.2), mapped to [0, 1]).
GRIPPER_SCALE = (-0.01, 0.044)
GRIPPER_EPS = 0.2


def finger_opening(observed, commanded_open, limits=(0.0, 0.044)):
    """Finger joint (m) to place link7/link8 at, from the official observation value n of one gripper.

    n = (target - scale0) / (scale1 - scale0) where the target is the command clamped to the measured finger +- 10.8 mm.
    While the clamp binds (fingers moving or blocked) the measured finger is n * 0.054 + 0.0008 (closing) or
    n * 0.054 - 0.0208 (opening); unclamped, the finger is within 10.8 mm of the command. The conservative choice is the
    outermost position the fingers can occupy during the stage: the open limit when the gripper is commanded open
    (they keep opening), else the measured finger of the closing relation (they only move inward)."""
    lower, upper = float(limits[0]), float(limits[1])
    if commanded_open:
        return upper
    if observed is None or not np.isfinite(observed):
        return upper
    span = GRIPPER_SCALE[1] - GRIPPER_SCALE[0]
    return float(np.clip(float(observed) * span + GRIPPER_SCALE[0] + GRIPPER_EPS * span, lower, upper))


def _rotation_batch(axis, angles):
    """(T, 3, 3) rotations about a unit axis (Rodrigues)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


class ArmKinematics:
    """Batched forward kinematics of every URDF link of one X5 arm in its base frame (public URDF)."""

    def __init__(self, model=None):
        from agp_core import public_camera as pc
        self.model = model or pc.default_model()
        joints = self.model._joints
        children = {}
        for name, joint in joints.items():
            children.setdefault(joint["parent"], []).append(name)
        order, stack = [], [self.model.base_link]
        while stack:                                         # parents before children
            parent = stack.pop(0)
            for name in children.get(parent, []):
                order.append(name)
                stack.append(joints[name]["child"])
        self.order = [(name, joints[name]["parent"], joints[name]["child"], joints[name]["type"],
                       np.asarray(joints[name]["axis"], float), pc._urdf_origin(joints[name])) for name in order]
        self.arm_index = {name: i for i, name in enumerate(self.model.arm_joint_names)}
        self.T_link6_tcp = np.asarray(self.model.T_link6_tcp, float)

    def links(self, Q, finger):
        """{link: (T, 4, 4)} base <- link for joint configurations Q (T, 6) and finger joints finger (T,)."""
        Q = np.atleast_2d(np.asarray(Q, float))
        T = len(Q)
        finger = np.broadcast_to(np.asarray(finger, float), (T,))
        poses = {self.model.base_link: np.broadcast_to(np.eye(4), (T, 4, 4)).copy()}
        for name, parent, child, kind, axis, origin in self.order:
            M = poses[parent] @ origin
            if kind in ("revolute", "continuous"):
                R = _rotation_batch(axis / np.linalg.norm(axis), Q[:, self.arm_index[name]])
                M = M.copy()
                M[:, :3, :3] = M[:, :3, :3] @ R
            elif kind == "prismatic":
                M = M.copy()
                M[:, :3, 3] = M[:, :3, 3] + (M[:, :3, :3] @ axis) * finger[:, None]
            poses[child] = M
        return poses

    def tcp(self, Q):
        """(T, 4, 4) base <- TCP."""
        return self.links(Q, 0.0)["link6"] @ self.T_link6_tcp


# ----------------------------------------------------------------------------- GJK distance
def _closest_segment(a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 0.0 else -float(a @ ab) / denom
    if t <= 0.0:
        return a, (0,)
    if t >= 1.0:
        return b, (1,)
    return a + t * ab, (0, 1)


def _closest_triangle(a, b, c):
    """Closest point of triangle abc to the origin (Ericson, Real-Time Collision Detection 5.1.5) and the indices of
    the vertices of the feature it lies on."""
    ab, ac = b - a, c - a
    d1, d2 = -float(ab @ a), -float(ac @ a)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (0,)
    d3, d4 = -float(ab @ b), -float(ac @ b)
    if d3 >= 0.0 and d4 <= d3:
        return b, (1,)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, (0, 1)
    d5, d6 = -float(ab @ c), -float(ac @ c)
    if d6 >= 0.0 and d5 <= d6:
        return c, (2,)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, (0, 2)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (1, 2)
    total = va + vb + vc
    if abs(total) < 1e-300:
        return a, (0,)
    v, w = vb / total, vc / total
    return a + ab * v + ac * w, (0, 1, 2)


def _closest_simplex(points):
    """(closest point to the origin, indices of the reduced simplex, origin inside) for 1 to 4 points."""
    n = len(points)
    if n == 1:
        return points[0], (0,), False
    if n == 2:
        p, idx = _closest_segment(points[0], points[1])
        return p, idx, False
    if n == 3:
        p, idx = _closest_triangle(points[0], points[1], points[2])
        return p, idx, False
    best = None
    faces = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0))
    inside = True
    for i, j, k, opposite in faces:
        a, b, c, d = points[i], points[j], points[k], points[opposite]
        normal = np.cross(b - a, c - a)
        side_origin, side_opposite = -float(normal @ a), float(normal @ (d - a))
        if side_origin * side_opposite < 0.0 or abs(side_opposite) < 1e-18:
            inside = False
            p, idx = _closest_triangle(a, b, c)
            dist = float(p @ p)
            if best is None or dist < best[0]:
                best = (dist, p, tuple((i, j, k)[m] for m in idx))
    if inside:
        return np.zeros(3), (0, 1, 2, 3), True
    return best[1], best[2], False


def gjk_distance(A, B, tolerance=1e-10, max_iterations=64):
    """Distance between the convex hulls of the vertex sets A (n, 3) and B (m, 3); 0 when they intersect."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    v = A[0] - B[0]
    simplex = []
    for _ in range(max_iterations):
        vv = float(v @ v)
        if vv <= tolerance * tolerance:
            return 0.0
        w = A[int(np.argmax(A @ -v))] - B[int(np.argmax(B @ v))]
        if vv - float(v @ w) <= max(1e-14, 1e-9 * vv):      # no progress toward the origin: v is the closest point
            return math.sqrt(vv)
        if any(np.array_equal(w, s) for s in simplex):
            return math.sqrt(vv)
        simplex.append(w)
        v, keep, inside = _closest_simplex(simplex)
        if inside:
            return 0.0
        simplex = [simplex[i] for i in keep]
    return math.sqrt(float(v @ v))


# ----------------------------------------------------------------------------- shapes of one arm
class Shape:
    """One convex shape fixed to a link: vertices in the link frame, a bounding sphere, an optional radius."""

    def __init__(self, link, name, vertices, radius=0.0, frame=None):
        self.link, self.name = link, name
        self.vertices = np.asarray(vertices, float).reshape(-1, 3)
        self.center = self.vertices.mean(axis=0)
        self.bound = float(np.max(np.linalg.norm(self.vertices - self.center, axis=1))) + float(radius)
        self.radius = float(radius)
        self.frame = frame          # optional fixed transform link <- shape frame (the TCP sphere)


def load_shapes(path=HULLS):
    """The guard's shapes of one arm (see the module docstring)."""
    from scipy.spatial import ConvexHull
    data = json.loads(Path(path).read_text())
    shapes = []
    for link in LINKS:
        entry = data["links"][link]
        parts = [np.asarray(p, float).reshape(-1, 3) for p in entry["parts"]]
        if link in DECOMPOSED:
            shapes += [Shape(link, f"{link}[{i}]", p) for i, p in enumerate(parts)]
            continue
        allv = np.vstack(parts)
        try:
            hull = allv[ConvexHull(allv).vertices]
        except Exception:                                    # noqa: BLE001 - flat or tiny: keep the points
            hull = allv
        shapes.append(Shape(link, link, hull))
    return shapes


class ArmGuardGeometry:
    """Both arms' shapes in the world frame over a timeline, and the clearance between the arms."""

    def __init__(self, bases, kinematics=None, shapes=None, held_radius=HELD_RADIUS_M):
        self.bases = {arm: np.asarray(T, float) for arm, T in bases.items()}      # world <- base
        self.kin = kinematics or ArmKinematics()
        self.shapes = shapes if shapes is not None else load_shapes()
        self.held_radius = float(held_radius)
        self.tcp_shape = Shape("link6", "held object (TCP sphere)", np.zeros((1, 3)), radius=self.held_radius,
                               frame=self.kin.T_link6_tcp)

    def _arm_frames(self, arm, Q, finger):
        links = self.kin.links(Q, finger)
        base = self.bases[arm]
        return {name: base @ M for name, M in links.items()}

    def _arm_shapes(self, held):
        """The arm's shapes; the held-object sphere only with a radius above 0 (HELD_RADIUS_M = 0: no envelope)."""
        return list(self.shapes) + ([self.tcp_shape] if held and self.held_radius > 0.0 else [])

    def clearance(self, timeline, fingers, held, threshold=None, report=None, stop_below=None, pairs=None):
        """Smallest distance between the arms at every time point.

        timeline: {arm: (T, 6) joints}; fingers: {arm: finger joint (m)}; held: {arm: bool}.
        threshold: distances below it are exact (GJK); above it a time point may carry a lower bound that is already
        >= threshold (bounding spheres, or the last exact distance minus how far the two shapes can have moved since,
        from their centres' paths and rotations). None: exact everywhere.
        stop_below: return as soon as one exact distance is below it (a yes/no question: the minimum returned is then
        below it, the other values are incomplete; detail["stopped_early"] is True).
        pairs (declared handover): None (every pair) or f(link_a, link_b) -> bool choosing the pairs compared (link_a of
        the first arm in sorted order); a pair left out is at distance inf.
        Returns (per_time (T,) minimum distance, detail of the overall minimum {time, links, distance, exact})."""
        arms = sorted(timeline)
        if len(arms) != 2:
            raise ValueError("clearance needs both arms")
        a, b = arms
        T = len(timeline[a])
        frames = {arm: self._arm_frames(arm, timeline[arm], fingers[arm]) for arm in arms}
        shapes = {arm: self._arm_shapes(held.get(arm, False)) for arm in arms}

        def poses(arm, shape):
            M = frames[arm][shape.link]
            return M @ shape.frame if shape.frame is not None else M

        placed = {arm: [(s, poses(arm, s)) for s in shapes[arm]] for arm in arms}
        centers = {arm: np.stack([M[:, :3, :3] @ s.center + M[:, :3, 3] for s, M in placed[arm]], axis=1)
                   for arm in arms}                                   # (T, n, 3)
        bounds = {arm: np.array([s.bound for s, _ in placed[arm]]) for arm in arms}
        # how far any point of a shape can move between consecutive time points (centre path + rotation x bound),
        # accumulated: motion[arm][t, i]
        motion = {}
        for arm in arms:
            steps = np.zeros((T, len(placed[arm])))
            if T > 1:
                for i, (s, M) in enumerate(placed[arm]):
                    R0, R1 = M[:-1, :3, :3], M[1:, :3, :3]
                    cos = (np.einsum("tij,tij->t", R0, R1) - 1.0) / 2.0
                    angle = np.arccos(np.clip(cos, -1.0, 1.0))
                    steps[1:, i] = np.linalg.norm(np.diff(centers[arm][:, i], axis=0), axis=1) + s.bound * angle
            motion[arm] = np.cumsum(steps, axis=0)
        sphere = (np.linalg.norm(centers[a][:, :, None, :] - centers[b][:, None, :, :], axis=3)
                  - bounds[a][None, :, None] - bounds[b][None, None, :])          # (T, na, nb) lower bounds
        if pairs is not None:                       # declared handover: only the chosen pairs are compared
            keep = np.array([[bool(pairs(sa.link, sb.link)) for sb, _ in placed[b]] for sa, _ in placed[a]], bool)
            sphere[:, ~keep] = np.inf
        limit = np.inf if threshold is None else float(threshold)
        dist = sphere.copy()
        exact = set()
        mask = sphere < limit
        stopped = False
        for i, j in zip(*np.nonzero(mask.any(axis=0))):
            if stopped:
                break
            times = np.nonzero(mask[:, i, j])[0]
            moved = motion[a][:, i] + motion[b][:, j]
            sa, Ma = placed[a][i]
            sb, Mb = placed[b][j]
            k = 0
            while k < len(times):
                t = int(times[k])
                Va = sa.vertices @ Ma[t, :3, :3].T + Ma[t, :3, 3]
                Vb = sb.vertices @ Mb[t, :3, :3].T + Mb[t, :3, 3]
                d = gjk_distance(Va, Vb) - sa.radius - sb.radius
                dist[t, i, j] = d
                exact.add((t, int(i), int(j)))
                if stop_below is not None and d < stop_below:
                    stopped = True
                    break
                if d < limit:
                    k += 1
                    continue
                # skip the time points where d minus the motion since t still stays >= the threshold
                later = times[k + 1:]
                bound = d - (moved[later] - moved[t])
                ok = bound >= limit
                skip = int(np.argmin(ok)) if not ok.all() else len(later)
                if skip:
                    dist[later[:skip], i, j] = np.maximum(dist[later[:skip], i, j], bound[:skip])
                k += 1 + skip
        per_time = dist.min(axis=(1, 2))
        t, i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        detail = {"time": int(t), "distance_m": float(per_time[t]),
                  "links": {a: placed[a][i][0].name, b: placed[b][j][0].name},
                  "exact": (int(t), int(i), int(j)) in exact, "stopped_early": stopped}
        if report is not None:
            report["pairs_exact"] = report.get("pairs_exact", 0) + len(exact)
            report["time_points"] = report.get("time_points", 0) + T
        return per_time, detail
