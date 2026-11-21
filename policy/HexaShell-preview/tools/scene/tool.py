"""scene: the objects on the table, from the RGB images and the camera matrices only.

    segment --camera C [--min_area PX]          object regions in one camera image, with an overlay image
    locate  --camera C (--id N | --u U --v V)   3D measurement of one object from two or more camera views
            [--camera_b auto|head|wrist_l|wrist_r]
    pixel_world --camera C --u U --v V [--z Z]  where the ray of a pixel meets the horizontal plane z (default: table)

No depth image, no learned model, nothing known about any object: numpy + OpenCV on the colour images, the camera
matrices of the observation, the table height measured by tools/table_plane and the robot's own geometry.

segment
    1. Table plane: every pixel ray is cut with the plane z = table_z (tools/table_plane). The work surface is where
       that point lies in the RoboShell workspace (x in [-0.75, 0.75], y in [-0.75, 0.60]), inside the table polygon
       (convex hull of the plane points of the head camera's table-coloured pixels, 4 cm margin: the table's rim, the
       frame and the floor are outside) and seen at least 18 degrees from the horizontal (grazing views show
       reflections more than the table colour).
    2. Robot mask: the links are placed by forward kinematics of the robot's URDF joint chain (base pose from the
       executor, joints from the arm state) and checked against the reported end link pose (1 cm / 3 deg); the end
       link and what hangs on it (fingers, wrist camera mount) are anchored to the reported end link pose. The mesh
       vertices of each link (3 mm grid; cuRobo collision spheres for the links without an STL mesh) are projected
       and their convex hull is filled, then dilated a few pixels. A wrist camera masks only its own fingers (the rest
       of its arm is behind it). The finger opening is measured in the arm's own wrist camera: the fingers are dark,
       and the opening (commanded value .. 1) whose finger outlines hold the most dark pixels wins (`gripper_seen`);
       closing on an object stops the fingers early. Without a robot model or with inconsistent kinematics a coarse
       model from the TCP pose is used (gripper box and finger boxes from the end pose, base column, upper arm and
       forearm capsules through an elbow solved in the arm's vertical plane).
    3. Table colour: a trimmed Gaussian in Lab of the box-filtered head image (filter about 12 mm on the table) over
       the work surface; objects are a minority there and drop out of the trimmed estimate even when they are large.
       A pixel is table when its filtered colour is within 4 sigma, also after brightening by 1/0.75 or 1/0.55
       (shadows). The wrist cameras use the head's model (a wrist view can be filled by one object).
    4. Small bright or coloured structures (a key, a pen) that the filter averages away: deviation of the raw colour
       from the local table colour (mean of the table pixels around it) beyond 7 sigma of the table's own grain,
       keeping compact structures that are clearly lighter (45 L units) or of another colour (35 units) somewhere.
    5. Objects: not table, not robot, opened 3x3 (the small structures added after), closed 5x5, connected regions.
       A region is kept when the point of its outline nearest to the camera's foot point lies on the work surface
       and in front of the line of the arm bases + 5 cm (behind it is the robot's station); a region that runs into
       the background is cut at the work-surface border (`clipped`); a thin strip along the robot's outline is a
       mask error and dropped; a small dark region touching a larger one is its contact shadow and merged; small
       holes are filled. A region is split where it narrows to a thin neck, and between two clearly different hues
       (k-means on the a/b chroma of its saturated pixels, hues 40 degrees apart, each part one coherent piece):
       two touching objects of different colours become two regions, a patterned object stays one.
locate
    Plane sweep of two silhouettes. The object's region in camera A and every region of camera B that can be the same
    object (inside the band of B where A's region can appear for heights 0..0.4 m, mean colour within 40 Lab units)
    are cast onto horizontal planes z = table_z + h on a 2 mm xy grid (h in 6 mm steps, then 1 mm around the best,
    parabolic peak). Their intersection over union, counted only where both cameras see the plane (inside the image,
    not on the robot), peaks at the height of the top face for objects with a flat top: there both silhouettes are
    bounded by the same top edges. The best B region must reach IoU 0.25 and B must see at least 30 % of the object.
    With several B cameras the heights within 1.5 cm are averaged. The top face is A's cast at that height, cut by
    each B's cast where B sees the plane; its centroid is `footprint_xy`, its minimum-area rectangle gives the
    extents, its principal axis the yaw. The same object overlaps from its top down to its base, two different
    objects only where their viewing cones cross: a match that stops overlapping above the table is `lifted`
    (held, or standing on something) and needs IoU 0.55 unless the region touches the robot.

Offline:
    python3 tool.py segment OBS_DIR PREFIX [--camera C] [--out DIR]     (PREFIX e.g. 000_reset)
    python3 tool.py locate  OBS_DIR PREFIX --camera C --id N
Robot model offline: env ROBOSHELL_ROBOT_URDF (the robot's URDF; the cuRobo yml next to it gives the sphere links);
in the server it is found from the planner's cuRobo configuration (api.planner(tag).yml_path).
"""
import base64
import importlib.util
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

CAMERA_CHOICES = ["head", "wrist_l", "wrist_r"]
TOOL = {
    "name": "scene",
    "commands": [
        {"name": "segment", "budget": False,
         "help": "find the object regions in one camera image (not table, not robot); writes obs/seg_<camera>.png",
         "args": [
             {"name": "camera", "type": "str", "default": "head", "choices": CAMERA_CHOICES,
              "help": "camera image to segment"},
             {"name": "min_area", "type": "int", "default": 40, "help": "smallest region kept, in pixels"},
         ]},
        {"name": "locate", "budget": False,
         "help": "3D position, size and yaw of one object from two or more camera views",
         "args": [
             {"name": "camera", "type": "str", "default": "head", "choices": CAMERA_CHOICES,
              "help": "camera of the --id (from the latest segment of that camera) or of the pixel --u --v"},
             {"name": "id", "type": "int", "default": None, "help": "object id from `segment --camera C`"},
             {"name": "u", "type": "float", "default": None, "help": "pixel column of a point on the object"},
             {"name": "v", "type": "float", "default": None, "help": "pixel row of a point on the object"},
             {"name": "camera_b", "type": "str", "default": "auto", "choices": ["auto"] + CAMERA_CHOICES,
              "help": "second camera; auto: every other camera that sees the object"},
         ]},
        {"name": "pixel_world", "budget": False,
         "help": "world point where the ray of a pixel meets a horizontal plane",
         "args": [
             {"name": "camera", "type": "str", "default": "head", "choices": CAMERA_CHOICES},
             {"name": "u", "type": "float", "required": True, "help": "pixel column"},
             {"name": "v", "type": "float", "required": True, "help": "pixel row"},
             {"name": "z", "type": "float", "default": None, "help": "plane height in m; default: the table top"},
         ]},
    ],
}

EXIT_FAILED = 2
SOURCES = {"head": "cam_head", "wrist_l": "cam_left_wrist", "wrist_r": "cam_right_wrist"}
WRIST_ARM = {"wrist_l": "left", "wrist_r": "right"}
WORK_X, WORK_Y = (-0.75, 0.75), (-0.75, 0.60)         # RoboShell workspace in xy
CACHE_ATTR = "scene_cache"
TCP_OFFSET_M = 0.145                                    # end link -> TCP along the tool x axis (robot model)
FINGER_TRAVEL = (-0.01, 0.044)                          # finger joint at gripper 0 and 1 (8.8 cm between the fingers)
MAX_HEIGHT_M = 0.40                                     # plane sweep: heights above the table
GRID_M = 0.002                                          # plane sweep: xy cell
ROBOT_DILATE_PX = {"head": 4, "wrist_l": 7, "wrist_r": 7}
OWN_NEAR_M = 0.03                                       # own fingers: parts nearer to the wrist camera are off-image
TABLE_MARGIN_M = 0.04                                   # regions must start this far inside the table's edge
EDGE_STRIP_M = 0.04                                     # thin regions within this much more of the edge: its rim
MIN_ELEVATION_DEG = 18.0                                # table points seen at a flatter angle are not used


def _load_table_plane():
    path = Path(__file__).resolve().parents[1] / "table_plane" / "tool.py"
    spec = importlib.util.spec_from_file_location("table_plane", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table_plane = _load_table_plane()


# ---- small geometry helpers ---------------------------------------------------------------------------------------
def _rigid_inverse(t):
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def _quat_matrix(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def _pose_matrix(pose):
    """[x y z qw qx qy qz] -> 4x4."""
    out = np.eye(4)
    out[:3, :3] = _quat_matrix(*[float(v) for v in pose[3:7]])
    out[:3, 3] = [float(v) for v in pose[:3]]
    return out


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _apply(m, pts):
    """Rigid/affine 4x4 (or 3x4) applied to (N, 3) points, element-wise (no BLAS threads)."""
    p = np.asarray(pts, float)
    return np.stack([m[i, 0] * p[:, 0] + m[i, 1] * p[:, 1] + m[i, 2] * p[:, 2] + m[i, 3] for i in range(3)], axis=1)


def _r3(values, nd=3):
    return [round(float(v), nd) for v in values]


# ---- cameras --------------------------------------------------------------------------------------------------------
class Camera:
    """One colour image with its intrinsics K and camera-to-world T (OpenCV axes: x right, y down, z forward)."""

    def __init__(self, name, bgr, k, t):
        self.name, self.bgr = name, bgr
        self.K, self.T = np.asarray(k, float), np.asarray(t, float)
        self.Tinv = _rigid_inverse(self.T)
        self.h, self.w = bgr.shape[:2]
        self._rays = None
        self._lab = None
        self._smoothed = None

    @property
    def centre(self):
        return self.T[:3, 3]

    def rays(self):
        """Per-pixel world ray directions (h, w, 3), not normalised (camera z component 1)."""
        if self._rays is None:
            u, v = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
            x, y = (u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]
            r = self.T[:3, :3]
            self._rays = np.stack([r[i, 0] * x + r[i, 1] * y + r[i, 2] for i in range(3)], axis=-1)
        return self._rays

    def smoothed(self):
        """Box-filtered image: the table's wood grain averages out. The filter spans about SMOOTH_M on the table
        (5 px for the head camera, more for a wrist camera close to the table)."""
        if self._smoothed is None:
            k = self.smooth_px()
            self._smoothed = cv2.blur(self.bgr, (k, k))
        return self._smoothed

    def lab(self):
        if self._lab is None:
            self._lab = cv2.cvtColor(self.smoothed(), cv2.COLOR_BGR2LAB)
        return self._lab

    def smooth_px(self):
        return int(getattr(self, "_smooth", SMOOTH_PX))

    def set_scale(self, table_z):
        """Pixel size on the table (median depth of the table pixels / focal length) -> filter sizes in pixels."""
        hits = self.plane_hits(table_z)
        ok = np.isfinite(hits[..., 0])
        if ok.sum() < 100:
            return
        d = self.rays()
        c = self.centre
        depth = (table_z - c[2]) / d[..., 2][ok]           # rays have camera-z component 1: s is the depth
        size = float(np.median(depth)) / float(self.K[0, 0])
        k = int(np.clip(round(SMOOTH_M / max(size, 1e-4)), SMOOTH_PX, 31))
        self._smooth = k + 1 - k % 2
        self._lab = None
        self._smoothed = None

    def plane_hits(self, z):
        """(h, w, 2) world xy where each pixel ray meets the plane z; NaN where the ray does not go down to it."""
        d = self.rays()
        c = self.centre
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - c[2]) / d[..., 2]
        s = np.where((d[..., 2] < -1e-6) & (s > 0), s, np.nan)
        return np.stack([c[0] + s * d[..., 0], c[1] + s * d[..., 1]], axis=-1)

    def ray(self, u, v):
        p = np.array([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], 1.0])
        return self.T[:3, :3] @ p

    def pixel_on_plane(self, u, v, z):
        d, c = self.ray(u, v), self.centre
        if d[2] > -1e-6 or (z - c[2]) / d[2] <= 0:
            return None
        s = (z - c[2]) / d[2]
        return c + s * d

    def project(self, points):
        """(N, 3) world -> (N, 2) pixels, (N,) depth along the view."""
        p = _apply(self.Tinv, points)
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.K[0, 0] * p[:, 0] / z + self.K[0, 2], self.K[1, 1] * p[:, 1] / z + self.K[1, 2]], 1)
        return uv, z

    def plane_homography(self, z, origin, cell):
        """3x3 map from grid cells (col, row) of a horizontal plane z (cell (0,0) at world xy `origin`) to pixels."""
        r, t = self.Tinv[:3, :3], self.Tinv[:3, 3]
        # world (x, y, z) = G @ (col, row, 1) with the third row giving z = z * 1
        g = np.array([[cell, 0.0, origin[0]], [0.0, cell, origin[1]], [0.0, 0.0, z]])
        p = r @ g
        p[:, 2] += t
        return self.K @ p


def _decode(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


# ---- robot model ----------------------------------------------------------------------------------------------------
def _read_stl(path):
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        verts = [line.split()[1:4] for line in data.decode("ascii", "ignore").splitlines()
                 if line.strip().startswith("vertex")]
        return np.asarray(verts, float)
    count = int(np.frombuffer(data[80:84], np.uint32)[0])
    rows = np.frombuffer(data, np.uint8, count=50 * count, offset=84).reshape(count, 50)
    return rows[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(float)


class RobotModel:
    """URDF joint chain + per-link point sets (mesh vertices on a 3 mm grid; cuRobo spheres for non-STL links)."""

    _cache = {}

    def __init__(self, urdf_path, spheres=None):
        root = ET.parse(urdf_path).getroot()
        base = os.path.dirname(urdf_path)
        self.joints = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            tf = np.eye(4)
            if origin is not None:
                tf[:3, :3] = _rpy_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()])
                tf[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            axis = joint.find("axis")
            self.joints[joint.find("child").get("link")] = {
                "name": joint.get("name"), "type": joint.get("type"), "parent": joint.find("parent").get("link"),
                "T": tf, "axis": None if axis is None else np.array([float(v) for v in axis.get("xyz").split()])}
        self.points = {}
        for link in root.findall("link"):
            name = link.get("name")
            visual = link.find("visual")
            mesh = None if visual is None else visual.find("geometry/mesh")
            pts = None
            if mesh is not None and mesh.get("filename", "").lower().endswith(".stl"):
                path = os.path.join(base, mesh.get("filename").replace("package://", ""))
                if os.path.exists(path):
                    verts = _read_stl(path)
                    origin = visual.find("origin")
                    if origin is not None:
                        rot = _rpy_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()])
                        verts = verts @ rot.T + np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                    pts = np.unique(np.round(verts / 0.003), axis=0) * 0.003
            if pts is None and spheres and name in spheres:
                pts = _sphere_points(spheres[name])
            if pts is not None and len(pts):
                self.points[name] = pts
        self.arm_joints = [self.joints[c]["name"] for c in self.joints
                           if self.joints[c]["type"] == "revolute"][:6]
        self.finger_joints = [self.joints[c]["name"] for c in self.joints if self.joints[c]["type"] == "prismatic"]

    @classmethod
    def load(cls, urdf_path, yml_path=None):
        key = (urdf_path, yml_path)
        if key not in cls._cache:
            spheres = None
            if yml_path and os.path.exists(yml_path):
                try:
                    import yaml
                    with open(yml_path) as handle:
                        spheres = yaml.safe_load(handle)["robot_cfg"]["kinematics"].get("collision_spheres")
                except Exception:
                    spheres = None
            cls._cache[key] = cls(urdf_path, spheres)
        return cls._cache[key]

    def fk(self, q, opening):
        """{link: 4x4 in the robot base frame} for 6 arm joints and a gripper opening 0..1."""
        values = {name: float(v) for name, v in zip(self.arm_joints, q)}
        finger = float(np.clip(FINGER_TRAVEL[0] + opening * (FINGER_TRAVEL[1] - FINGER_TRAVEL[0]), 0.0, FINGER_TRAVEL[1]))
        for name in self.finger_joints:
            values[name] = finger
        out = {}

        def pose(link):
            if link in out:
                return out[link]
            joint = self.joints.get(link)
            if joint is None:
                out[link] = np.eye(4)
                return out[link]
            m = joint["T"].copy()
            value = values.get(joint["name"], 0.0)
            if joint["type"] in ("revolute", "continuous"):
                r = np.eye(4)
                r[:3, :3] = _axis_angle(joint["axis"], value)
                m = m @ r
            elif joint["type"] == "prismatic":
                p = np.eye(4)
                p[:3, 3] = joint["axis"] * value
                m = m @ p
            out[link] = pose(joint["parent"]) @ m
            return out[link]

        for link in list(self.joints) + list(self.points):
            pose(link)
        return out

    def subtree(self, link):
        """The link and every link that hangs on it."""
        out = {link}
        changed = True
        while changed:
            changed = False
            for child, joint in self.joints.items():
                if joint["parent"] in out and child not in out:
                    out.add(child)
                    changed = True
        return out

    def finger_links(self):
        return [child for child, joint in self.joints.items() if joint["type"] == "prismatic"]

    def end_link(self):
        """The last link of the arm chain (the one the 6th arm joint moves)."""
        for child, joint in self.joints.items():
            if joint["name"] == self.arm_joints[-1]:
                return child
        return None


def _sphere_points(spheres):
    """Points on the surface of cuRobo collision spheres (enough for a convex hull)."""
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
                    + [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = [np.asarray(s["center"], float) + s["radius"] * dirs for s in spheres]
    return np.concatenate(pts) if pts else None


class ArmState:
    """What the robot mask needs of one arm: base pose, joints (or None), end link pose, gripper opening."""

    def __init__(self, tag, base, joints, ee, gripper):
        self.tag, self.base, self.joints, self.ee, self.gripper = tag, base, joints, ee, gripper


def _link_points_world(model, arm):
    """[(link, world points)] from forward kinematics; None if the kinematics disagree with the reported end pose."""
    if model is None or arm.joints is None or arm.base is None:
        return None
    poses = model.fk(arm.joints, arm.gripper)
    end = model.end_link()
    if end is None:
        return None
    ee = arm.base @ poses[end]
    if arm.ee is not None:
        err_m = float(np.linalg.norm(ee[:3, 3] - arm.ee[:3, 3]))
        cos = (np.trace(ee[:3, :3].T @ arm.ee[:3, :3]) - 1) / 2
        if err_m > 0.01 or np.degrees(np.arccos(np.clip(cos, -1, 1))) > 3.0:
            return None
    # the end link and what hangs on it (fingers, wrist camera mount) are placed from the reported end link pose:
    # a joint error of a few milliradians would move the fingers by centimetres in the arm's own wrist camera
    anchor = arm.ee @ _rigid_inverse(poses[end]) if arm.ee is not None else arm.base
    below = model.subtree(end)
    out = []
    for link, pts in model.points.items():
        if link in poses:
            out.append((link, _apply((anchor if link in below else arm.base) @ poses[link], pts)))
    return out


def _box_points(centre_tf, half):
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    return _apply(centre_tf, corners)


def _capsule_points(a, b, radius):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dirs = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.concatenate([a + radius * dirs, b + radius * dirs])


def _coarse_points(arm):
    """Approximate arm without the robot model: gripper body + fingers from the end pose, base column, two capsules.

    Robot geometry: shoulder 0.10 m above the base, upper arm 0.264 m, forearm 0.32 m (X5 arm), gripper body
    about 0.17 x 0.06 m behind the fingers, fingers 0.07 m long."""
    if arm.ee is None:
        return []
    ee = arm.ee
    parts = []
    body = ee.copy()
    body[:3, 3] = ee[:3, :3] @ np.array([0.045, 0.0, 0.0]) + ee[:3, 3]
    parts.append(("gripper", _box_points(body, np.array([0.05, 0.09, 0.035]))))
    travel = float(np.clip(FINGER_TRAVEL[0] + arm.gripper * (FINGER_TRAVEL[1] - FINGER_TRAVEL[0]), 0.0, FINGER_TRAVEL[1]))
    for side in (-1, 1):
        finger = ee.copy()           # X5 finger: 0.086 m .. 0.158 m along the tool x axis, centre 0.019 m + travel aside
        finger[:3, 3] = ee[:3, :3] @ np.array([0.122, side * (0.0194 + travel), 0.0]) + ee[:3, 3]
        parts.append(("finger", _box_points(finger, np.array([0.036, 0.019, 0.031]))))
    if arm.base is not None:
        base = arm.base[:3, 3]
        shoulder = base + np.array([0.0, 0.0, 0.10])
        parts.append(("base", _capsule_points(base, shoulder, 0.05)))
        wrist = ee[:3, 3] - 0.09 * ee[:3, 0]
        l1, l2 = 0.264, 0.32
        d = wrist - shoulder
        dist = float(np.linalg.norm(d))
        if 1e-3 < dist < l1 + l2:
            # elbow above the shoulder-wrist line, in the vertical plane through both
            horiz = np.array([d[0], d[1], 0.0])
            side = np.cross(d, [0.0, 0.0, 1.0]) if np.linalg.norm(horiz) > 1e-3 else np.array([1.0, 0.0, 0.0])
            up = np.cross(side, d)
            up = up / (np.linalg.norm(up) or 1.0)
            if up[2] < 0:
                up = -up
            a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
            hgt = np.sqrt(max(l1 * l1 - a * a, 0.0))
            elbow = shoulder + a * d / dist + hgt * up
        else:
            elbow = 0.5 * (shoulder + wrist)
        parts.append(("upper_arm", _capsule_points(shoulder, elbow, 0.045)))
        parts.append(("forearm", _capsule_points(elbow, wrist, 0.04)))
    return parts


def _finger_parts(model, arm, opening):
    """World points of the two fingers of an arm at a given opening (robot model, else the coarse model)."""
    moved = ArmState(arm.tag, arm.base, arm.joints, arm.ee, opening)
    if model is not None and arm.joints is not None and arm.base is not None:
        parts = _link_points_world(model, moved)
        if parts is not None:
            fingers = set(model.finger_links())
            return [pts for name, pts in parts if name in fingers]
    return [pts for name, pts in _coarse_points(moved) if name == "finger"]


def estimate_opening(cam, arm, model):
    """The fingers are dark: the opening at which the fingers' outlines in the arm's own wrist camera hold the
    darkest pixels. The commanded opening is a lower bound (closing on an object stops the fingers early).
    Returns None when the image does not decide (the commanded value is used then)."""
    low = float(np.clip(arm.gripper, 0.0, 1.0))
    if low >= 0.98:
        return None
    dark = cv2.cvtColor(cam.bgr, cv2.COLOR_BGR2HSV)[..., 2] < 50
    best, best_score = None, -1.0
    for g in np.arange(low, 1.0 + 1e-9, 0.025):
        m = np.zeros((cam.h, cam.w), np.uint8)
        for pts in _finger_parts(model, arm, float(g)):
            uv, z = cam.project(pts)
            keep = z > OWN_NEAR_M
            if keep.sum() >= 3:
                cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.clip(uv[keep], -4 * cam.w, 5 * cam.w)).astype(np.int32)), 1)
        inside = m.astype(bool)
        if inside.sum() < 200:
            continue
        score = float(dark[inside].mean())
        if score > best_score + 0.01:
            best, best_score = float(g), score
    if best is None or best_score < 0.5:
        return None
    return round(best, 3)


def robot_mask(cam, arms, model):
    """Boolean mask of the robot in one camera, and how it was made ('kinematics' / 'coarse' per arm)."""
    mask = np.zeros((cam.h, cam.w), np.uint8)
    how = {}
    fingers = set(model.finger_links()) if model is not None else set()
    for arm in arms:
        parts = _link_points_world(model, arm)
        how[arm.tag] = "kinematics" if parts is not None else "coarse"
        if parts is None:
            parts = _coarse_points(arm)
        own = WRIST_ARM.get(cam.name) == arm.tag
        for name, pts in parts:
            finger = name in fingers or name == "finger"
            if own and not finger:
                continue            # a wrist camera sees its own fingers only; the rest of its arm is behind it
            uv, z = cam.project(pts)
            keep = z > (OWN_NEAR_M if own else 0.05)
            if keep.sum() < 3:
                continue
            uv = np.clip(uv[keep], -4 * cam.w, 5 * cam.w)
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
    k = ROBOT_DILATE_PX.get(cam.name, 3)
    if k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return mask.astype(bool), how


# ---- table colour model ------------------------------------------------------------------------------------------
SMOOTH_PX = 5                  # box filter before the colour test (smallest size, px)
SMOOTH_M = 0.012               # ... spanning about this much table: wood grain averages out, object colours stay
TABLE_D2 = 16.0                # squared Mahalanobis distance (4 sigma) below which a smoothed colour is table
SHADOW_SCALES = (0.75, 0.55)   # a pixel this much darker than table colour is still table (shadows)
LOCAL_D2 = 49.0                # small bright/coloured structures: 7 sigma off the local table colour,
LOCAL_MIN_DL, LOCAL_MIN_DC = 45.0, 35.0   # ... and somewhere this much lighter or this far in colour (Lab units)
SPLIT_MIN_CHROMA = 15.0        # colour splitting: pixels with at least this chroma (Lab a, b) take part
SPLIT_MIN_SEPARATION = 30.0    # ... and the two colours must be this far apart
SPLIT_MIN_HUE_DEG = 40.0       # ... and differ this much in hue


class TableColour:
    """Which (smoothed) Lab colours are table, learned from work-surface pixels.

    A trimmed Gaussian grown from the most common colour: start at the densest cell of a coarse Lab histogram with a
    small spread, then rounds of keeping the samples within 3 sigma and re-estimating mean and covariance (corrected
    for the trimming) until it settles. It describes the table's own colours even when objects of other colours
    cover a large part of the view; the table's spread in lightness covers its grain and soft shadows."""

    def __init__(self, samples_lab):
        lab = samples_lab.reshape(-1, 3).astype(float)
        if len(lab) > 150000:
            lab = lab[np.random.default_rng(0).choice(len(lab), 150000, replace=False)]
        cells = (lab[:, 0] // 8).astype(int) * 4096 + (lab[:, 1] // 4).astype(int) * 64 + (lab[:, 2] // 4).astype(int)
        hist = np.bincount(cells, minlength=32 * 64 * 64).reshape(32, 64, 64).astype(float)
        for axis in range(3):                                   # 3-cell box smoothing
            hist = hist + np.roll(hist, 1, axis) + np.roll(hist, -1, axis)
        peak = np.unravel_index(int(np.argmax(hist)), hist.shape)
        mean = np.array([peak[0] * 8 + 4.0, peak[1] * 4 + 2.0, peak[2] * 4 + 2.0])
        cov = np.diag([10.0 ** 2, 4.0 ** 2, 4.0 ** 2])
        for _ in range(12):
            keep = _mahalanobis2(lab, mean, cov) < 9.0          # 3 sigma
            if keep.sum() < 100:
                break
            new_mean = lab[keep].mean(0)
            new_cov = _cov3(lab[keep]) / 0.918 + np.eye(3) * 0.5   # 3-sigma trimming shrinks the covariance by 0.918
            done = np.abs(new_mean - mean).max() < 0.2 and np.abs(np.diag(new_cov) / np.diag(cov) - 1).max() < 0.02
            mean, cov = new_mean, new_cov
            if done:
                break
        self.mean, self.cov = mean, cov
        self.core_share = float((_mahalanobis2(lab, mean, cov) < TABLE_D2).mean())

    def is_table(self, lab):
        """Within 4 sigma, and not brighter than mean + 3 sigma in lightness: on a glossy table lightness and
        saturation are anti-correlated (highlights), so the ellipsoid reaches toward light grey along its long axis;
        light grey and white things stay objects."""
        flat = lab.reshape(-1, 3).astype(float)
        ok = (_mahalanobis2(flat, self.mean, self.cov) < TABLE_D2) & (flat[:, 0] <= self.l_max)
        return ok.reshape(lab.shape[:2])

    @property
    def l_max(self):
        return float(self.mean[0] + 3.0 * np.sqrt(self.cov[0, 0]))

    def is_table_bgr(self, smoothed_bgr):
        """Table colour, or table colour in a shadow: the pixel brightened by 1/s (s = SHADOW_SCALES) is table."""
        out = None
        for s in (1.0,) + SHADOW_SCALES:
            img = smoothed_bgr if s == 1.0 else np.clip(smoothed_bgr.astype(np.float32) / s, 0, 255).astype(np.uint8)
            t = self.is_table(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            out = t if out is None else out | t
        return out


def _inv3(m):
    """3x3 inverse by the adjugate (no LAPACK: its worker threads keep spinning inside the server process)."""
    a = np.asarray(m, float)
    c = np.array([[a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1], a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2], a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]],
                  [a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2], a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0], a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]],
                  [a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0], a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1], a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]]])
    return c / (a[0, 0] * c[0, 0] + a[0, 1] * c[1, 0] + a[0, 2] * c[2, 0])


def _cov3(x):
    d = x - x.mean(0)
    return np.array([[float((d[:, i] * d[:, j]).mean()) for j in range(3)] for i in range(3)])


def _mahalanobis2(x, mean, cov):
    inv = _inv3(cov)
    d = x - mean
    return sum(inv[i, j] * d[:, i] * d[:, j] for i in range(3) for j in range(3))


# ---- colour names ---------------------------------------------------------------------------------------------------
COLOUR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown", "white", "grey", "black"]


def colour_names(bgr):
    """Per-pixel index into COLOUR_NAMES (coarse, from HSV)."""
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    out = np.full(len(h), COLOUR_NAMES.index("grey"))
    hue = np.select([(h < 6) | (h >= 170), h < 19, h < 34, h < 80, h < 100, h < 130, h < 150],
                    [0, 1, 2, 3, 4, 5, 6], 7)
    out = np.where(s >= 60, hue, out)
    warm = (hue <= 2) & (s >= 60)
    out = np.where(warm & (v < 150) & ((hue > 0) | (v < 110)), COLOUR_NAMES.index("brown"), out)
    out = np.where(((hue == 0) | (hue == 7)) & (s >= 60) & (s < 150) & (v >= 170), COLOUR_NAMES.index("pink"), out)
    low = s < 60
    out = np.where(low & (v >= 180), COLOUR_NAMES.index("white"), out)
    out = np.where(low & (v < 180), COLOUR_NAMES.index("grey"), out)
    out = np.where(v < 45, COLOUR_NAMES.index("black"), out)
    return out


def _colour_summary(pixels_bgr):
    idx = colour_names(pixels_bgr)
    counts = np.bincount(idx, minlength=len(COLOUR_NAMES)).astype(float)
    share = counts / max(counts.sum(), 1)
    order = np.argsort(-share)
    main = COLOUR_NAMES[int(order[0])]
    colours = [COLOUR_NAMES[int(i)] for i in order if share[i] >= 0.15]
    return main, colours


# ---- segmentation ---------------------------------------------------------------------------------------------------
class Scene:
    """Everything one command needs: cameras, table height, arm states, robot model."""

    def __init__(self, cams, table_z, arms, model, signature=None):
        self.cams, self.table_z, self.arms, self.model = cams, table_z, arms, model
        if table_z is not None:
            for cam in cams.values():
                cam.set_scale(table_z)
        self.signature = signature or {}
        self._robot = {}
        self._colour = {}
        self._openings = None
        self._table = {}
        self._segments = {}
        self._polygon = None

    def robot(self, name):
        if name not in self._robot:
            self._robot[name] = robot_mask(self.cams[name], self.mask_arms(), self.model)
        return self._robot[name]

    def openings(self):
        """Finger opening of each arm as seen in its own wrist camera (see estimate_opening)."""
        if self._openings is None:
            self._openings = {}
            for arm in self.arms:
                cam = self.cams.get({v: k for k, v in WRIST_ARM.items()}.get(arm.tag))
                self._openings[arm.tag] = estimate_opening(cam, arm, self.model) if cam is not None else None
        return self._openings

    def mask_arms(self):
        seen = self.openings()
        return [arm if seen.get(arm.tag) is None else ArmState(arm.tag, arm.base, arm.joints, arm.ee, seen[arm.tag])
                for arm in self.arms]

    def table_polygon(self):
        """World xy polygon of the table top: convex hull of the plane points of the largest connected region of
        table-coloured head-camera pixels (the head sees the whole table; its rim, the frame and the floor are not
        table colour)."""
        if self._polygon is None:
            self._polygon = np.zeros((0, 2))
            if "head" in self.cams:
                hits, inside = self.work_surface("head")
                robot, _ = self.robot("head")
                table = self.table_mask("head") & inside & ~robot
                t8 = cv2.morphologyEx(table.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                n, labels, stats, _ = cv2.connectedComponentsWithStats(t8, connectivity=8)
                if n > 1:
                    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    pts = hits[labels == big]
                    if len(pts) >= 2000:
                        poly = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)
                        area = sum(poly[i, 0] * poly[i - 1, 1] - poly[i - 1, 0] * poly[i, 1] for i in range(len(poly)))
                        self._polygon = poly if area < 0 else poly[::-1]      # counter-clockwise
        return self._polygon

    def on_table(self, hits, margin=TABLE_MARGIN_M):
        """Plane points at least `margin` inside the table polygon (everywhere, when the polygon is unknown)."""
        poly = self.table_polygon()
        if len(poly) < 3:
            return np.ones(hits.shape[:2], bool)
        inside = np.ones(hits.shape[:2], bool)
        x, y = hits[..., 0], hits[..., 1]
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i - 1], poly[i]
            length = np.hypot(x1 - x0, y1 - y0)
            if length < 1e-6:
                continue
            dist = ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / length      # > 0: left of the edge = inside
            inside &= np.nan_to_num(dist, nan=-1.0) >= margin
        return inside

    def station_y(self):
        """Regions nearer to the robot than this y are its station (the line of the arm bases + 5 cm)."""
        ys = [float(arm.base[1, 3]) for arm in self.arms if arm.base is not None]
        return (max(ys) if ys else -0.45) + 0.05

    def work_surface(self, name):
        cam = self.cams[name]
        hits = cam.plane_hits(self.table_z)
        with np.errstate(invalid="ignore"):
            inside = ((hits[..., 0] >= WORK_X[0]) & (hits[..., 0] <= WORK_X[1])
                      & (hits[..., 1] >= WORK_Y[0]) & (hits[..., 1] <= WORK_Y[1]))
            # rays that graze the table see reflections more than the table colour: only steeper views count
            d = cam.rays()
            steep = -d[..., 2] >= np.sin(np.radians(MIN_ELEVATION_DEG)) * np.sqrt((d * d).sum(-1))
        return hits, inside & steep

    def table_mask(self, name):
        """Pixels of table colour (shadows included) in one camera."""
        if name not in self._table:
            self._table[name] = self.colour_model(name).is_table_bgr(self.cams[name].smoothed())
        return self._table[name]

    def colour_model(self, name):
        """Table colours (see TableColour), learned once per scene."""
        key = "head" if "head" in self.cams else name
        if key not in self._colour:
            self._colour[key] = self._colour_model(name)
        return self._colour[key]

    def _colour_model(self, name):
        """From the head camera (it sees most of the table, objects are a minority there); from this camera
        itself only when there is no head image."""
        source = "head" if "head" in self.cams else name
        _, inside = self.work_surface(source)
        robot, _ = self.robot(source)
        return TableColour(self.cams[source].lab()[inside & ~robot])

    def segment(self, name, min_area=40):
        key = (name, int(min_area))
        if key not in self._segments:
            self._segments[key] = _segment(self, name, int(min_area))
        return self._segments[key]


def _nadir_distance(cam, hits):
    c = cam.centre
    return np.hypot(hits[..., 0] - c[0], hits[..., 1] - c[1])


def _segment(scene, name, min_area):
    cam = scene.cams[name]
    hits, inside = scene.work_surface(name)
    robot, robot_how = scene.robot(name)
    model = scene.colour_model(name)
    table = scene.table_mask(name)
    inside = inside & scene.on_table(hits)
    finite = np.isfinite(hits[..., 0])
    station_y = scene.station_y()
    fg8 = ((~table) & (~robot) & finite).astype(np.uint8)
    fg8 = cv2.morphologyEx(fg8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg8 |= _small_bright_or_coloured(cam, table & ~robot & finite, ~robot & finite, model)
    fg8 = cv2.morphologyEx(fg8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg8[robot] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg8, connectivity=8)
    dist = _nadir_distance(cam, hits)
    regions = []                                    # [mask, clipped]
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min(min_area, 15):
            continue
        comp = labels == i
        ys, xs = np.nonzero(comp)
        d = dist[ys, xs]
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(d))
        if not inside[ys[j], xs[j]]:
            continue                                   # its nearest point is off the table: background
        if hits[ys[j], xs[j], 1] < station_y:
            continue                                   # behind the line of the arm bases: the robot's station
        clipped = False
        outside = ~inside[ys, xs]
        if outside.any():
            out_mask = np.zeros_like(comp)
            out_mask[ys[outside], xs[outside]] = True
            touches = (out_mask[0].any() or out_mask[-1].any() or out_mask[:, 0].any() or out_mask[:, -1].any())
            if touches or outside.mean() > 0.5:
                comp = comp & inside                   # ran into the background: keep the part on the table
                clipped = True
        comp = _fill_holes(comp, max_hole=max(150, int(0.1 * np.count_nonzero(comp))))
        if np.count_nonzero(comp) >= min(min_area, 15):
            regions.append([comp, clipped])
    regions = [[part, clipped] for comp, clipped in regions for part in _split_necks(comp, min_area)]
    regions = [[part, clipped] for comp, clipped in regions for part in _split_colours(cam, comp, min_area)]
    regions = _merge_shadows(cam, regions)
    near = cv2.dilate(robot.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    objects, masks = [], []
    for comp, clipped in regions:
        if np.count_nonzero(comp) < min_area:
            continue
        if _robot_sliver(comp, near):
            continue
        if _edge_strip(scene, comp, hits):
            continue
        objects.append(_describe(scene, cam, hits, dist, comp, robot, clipped))
        masks.append(comp)
    order = sorted(range(len(objects)), key=lambda k: (objects[k]["bottom_xy"][0], objects[k]["bottom_xy"][1]))
    objects = [objects[k] for k in order]
    masks = [masks[k] for k in order]
    for k, obj in enumerate(objects):
        obj["id"] = k
    return {"camera": name, "objects": objects, "masks": masks, "robot": robot, "robot_model": robot_how,
            "inside": inside, "table_share": round(model.core_share, 3)}


def _small_bright_or_coloured(cam, table, allowed, model=None):
    """Thin or small objects that the smoothed colour test averages away (a key, a pen): pixels that stand out
    from the table around them. The local table colour is the mean of the table pixels in a window 4x the
    smoothing size; the deviations of the table's own pixels from it (wood grain) give a Gaussian; a pixel
    counts when it deviates by more than 7 sigma. Only brighter or differently coloured pixels count: thin dark
    lines are the seams of the wood."""
    lab = cv2.cvtColor(cam.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = 4 * cam.smooth_px() + 1
    w = cv2.blur(table.astype(np.float32), (k, k))
    local = cv2.blur(lab * table[..., None].astype(np.float32), (k, k)) / np.maximum(w, 1e-3)[..., None]
    dev = lab - local
    dev[..., 0] = np.maximum(dev[..., 0], 0.0)
    sample = dev[table & (w > 0.5)]
    if len(sample) < 1000:
        return np.zeros(table.shape, np.uint8)
    if len(sample) > 100000:
        sample = sample[np.random.default_rng(1).choice(len(sample), 100000, replace=False)]
    sample = sample.astype(float)
    mean = np.median(sample, axis=0)
    cov = np.diag((np.median(np.abs(sample - mean), axis=0) * 1.4826 + 0.5) ** 2)
    for _ in range(4):
        keep = _mahalanobis2(sample, mean, cov) < 7.81
        if keep.sum() < 100:
            break
        mean = sample[keep].mean(0)
        cov = _cov3(sample[keep]) / 0.83 + np.eye(3) * 0.25
    d2 = _mahalanobis2(dev.reshape(-1, 3).astype(float), mean, cov).reshape(table.shape)
    strong = ((d2 > LOCAL_D2) & allowed & (w > 0.2)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    if n <= 1:
        return strong
    # a structure counts if it is much brighter somewhere (white, light) or of a clearly other colour somewhere;
    # the bright edges of the wood's seams and grain stay below both
    flat = labels.reshape(-1)
    peak_l = np.zeros(n)
    peak_c = np.zeros(n)
    sel = flat > 0
    np.maximum.at(peak_l, flat[sel], dev[..., 0].reshape(-1)[sel])
    np.maximum.at(peak_c, flat[sel], np.hypot(dev[..., 1], dev[..., 2]).reshape(-1)[sel])
    keep = (stats[:, cv2.CC_STAT_AREA] >= 6) & ((peak_l >= LOCAL_MIN_DL) | (peak_c >= LOCAL_MIN_DC))
    keep[0] = False
    # compact structures only: the bright seams and reflections of the wood are thin lines and stringy networks;
    # and not of table colour on average (a highlight on the wood is a lighter table colour)
    for i in np.nonzero(keep)[0]:
        x, y, bw, bh, area = stats[i]
        sub = labels[y:y + bh, x:x + bw] == i
        pts = np.column_stack(np.nonzero(sub))[:, ::-1].astype(np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        hull = cv2.contourArea(cv2.convexHull(pts))
        if area / max(a, b, 1.0) < 2.5 or area / max(hull, 1.0) < 0.35:
            keep[i] = False
        elif model is not None:
            mean_lab = lab[y:y + bh, x:x + bw][sub].mean(0).astype(np.float32).reshape(1, 1, 3)
            if model.is_table(mean_lab)[0, 0]:
                keep[i] = False
    return keep[labels].astype(np.uint8)


def _split_necks(mask, min_area):
    """Split a region where it narrows to a 1-2 pixel neck (two objects joined by a thin line or a corner): erode
    once; if the core falls into several big parts, every pixel joins the nearest part."""
    ys, xs = np.nonzero(mask)
    y0, x0 = max(ys.min() - 2, 0), max(xs.min() - 2, 0)
    sub = mask[y0:ys.max() + 3, x0:xs.max() + 3].astype(np.uint8)
    core = cv2.erode(sub, np.ones((3, 3), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    total = int(sub.sum())
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(min_area / 2, 0.1 * total)]
    if len(big) < 2:
        return [mask]
    seeds = np.zeros(sub.shape, np.int32)
    for k, i in enumerate(big, 1):
        seeds[labels == i] = k
    # grow the parts back over the region, one pixel ring at a time
    grown = seeds.copy()
    for _ in range(12):
        free = (sub > 0) & (grown == 0)
        if not free.any():
            break
        dil = cv2.dilate(grown.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        grown[free] = dil[free]
    parts = []
    for k in range(1, len(big) + 1):
        part = np.zeros(mask.shape, bool)
        part[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = grown == k
        parts.append(part)
    rest = (sub > 0) & (grown == 0)
    if rest.any():                                   # leftover thin pieces stay with the first part
        parts[0][y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] |= rest
    return parts


def _split_colours(cam, mask, min_area, depth=0):
    """Split a region made of two clearly different colours that each form one coherent part (two objects of
    different colours that touch or stand on each other). Only saturated pixels take part: white, grey and black
    parts and highlights join the nearest part. A region whose colours are interleaved (a pattern) stays whole."""
    n_all = int(np.count_nonzero(mask))
    min_part = max(min_area, 60)
    if n_all < 2 * min_part or depth > 1:
        return [mask]
    ys, xs = np.nonzero(mask)
    lab = cam.lab()[ys, xs].astype(np.float32)
    ab = lab[:, 1:] - 128.0
    chroma = np.hypot(ab[:, 0], ab[:, 1])
    colourful = chroma >= SPLIT_MIN_CHROMA
    if colourful.sum() < 2 * min_part:
        return [mask]
    data = np.ascontiguousarray(ab[colourful])
    cv2.setRNGSeed(0)
    _, labels, centres = cv2.kmeans(data, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    2, cv2.KMEANS_PP_CENTERS)
    hue = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    dhue = abs((hue[0] - hue[1] + 180.0) % 360.0 - 180.0)
    if dhue < SPLIT_MIN_HUE_DEG or float(np.linalg.norm(centres[0] - centres[1])) < SPLIT_MIN_SEPARATION:
        return [mask]                           # one hue: lit and shaded faces of one object differ only in chroma
    labels = labels.ravel()
    y0, x0 = ys.min(), xs.min()
    h, w = ys.max() - y0 + 1, xs.max() - x0 + 1
    seeds = np.zeros((h, w), np.int32)
    seeds[ys[colourful] - y0, xs[colourful] - x0] = labels + 1
    for k in (1, 2):
        part = (seeds == k).astype(np.uint8)
        if part.sum() < min_part:
            return [mask]
        n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        if stats[1:, cv2.CC_STAT_AREA].max() < 0.85 * part.sum():
            return [mask]                       # interleaved colours: one object with a pattern
    # the other pixels of the region join the nearest coloured part
    region = np.zeros((h, w), bool)
    region[ys - y0, xs - x0] = True
    grown = seeds.copy()
    for _ in range(max(h, w)):
        free = region & (grown == 0)
        if not free.any():
            break
        dil = cv2.dilate(grown.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        grown[free] = dil[free]
    parts = []
    for k in (1, 2):
        part = np.zeros(mask.shape, bool)
        part[y0:y0 + h, x0:x0 + w] = (grown == k) & region
        parts.extend(_split_colours(cam, part, min_area, depth + 1))
    return parts


def _edge_strip(scene, mask, hits):
    """A thin strip along the table's edge: the rim of the table catching the light."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (a, b), _ = cv2.minAreaRect(pts)
    if len(xs) / max(a, b, 1.0) >= 5.0:
        return False
    xy = hits[ys, xs].reshape(-1, 1, 2)
    inner = scene.on_table(xy, margin=TABLE_MARGIN_M + EDGE_STRIP_M)
    return float(inner.mean()) < 0.1


def _robot_sliver(mask, near_robot):
    """A thin strip along the robot's outline: the edge of an arm link that the mask misses by a few pixels."""
    ys, xs = np.nonzero(mask)
    if not near_robot[ys, xs].any():
        return False
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (a, b), _ = cv2.minAreaRect(pts)
    return len(xs) / max(a, b, 1.0) < 4.0 and near_robot[ys, xs].mean() > 0.3


def _merge_shadows(cam, regions):
    """A small dark region touching a larger region is its contact shadow or a dark part of it: merge them."""
    if len(regions) < 2:
        return regions
    value = cv2.cvtColor(cam.bgr, cv2.COLOR_BGR2HSV)[..., 2]
    areas = [int(np.count_nonzero(m)) for m, _ in regions]
    order = sorted(range(len(regions)), key=lambda k: areas[k])
    kernel = np.ones((7, 7), np.uint8)
    alive = [True] * len(regions)
    for k in order:
        mask = regions[k][0]
        if float(value[mask].mean()) > 100:
            continue
        grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        best, best_area = None, 0
        for m in range(len(regions)):
            if m == k or not alive[m] or areas[m] < 2.5 * areas[k]:
                continue
            if (grown & regions[m][0]).any() and areas[m] > best_area:
                best, best_area = m, areas[m]
        if best is not None:
            regions[best][0] = regions[best][0] | mask
            regions[best][1] = regions[best][1] or regions[k][1]
            areas[best] += areas[k]
            alive[k] = False
    return [r for r, a in zip(regions, alive) if a]


def _describe(scene, cam, hits, dist, mask, robot, clipped):
    ys, xs = np.nonzero(mask)
    d = dist[ys, xs]
    j = int(np.argmin(np.where(np.isfinite(d), d, np.inf)))
    bottom = hits[ys[j], xs[j]]
    cu, cv_ = float(xs.mean()), float(ys.mean())
    centre_hit = cam.pixel_on_plane(cu, cv_, scene.table_z)
    pixels = cam.bgr[ys, xs]
    main, colours = _colour_summary(pixels)
    near_robot = bool(cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)[robot].any())
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_px": [round(cu, 1), round(cv_, 1)],
        "area_px": int(len(ys)),
        "mean_rgb": [int(round(v)) for v in pixels.mean(0)[::-1]],
        "color": main,
        "colors": colours,
        "touches_border": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == cam.w - 1 or ys.max() == cam.h - 1),
        "near_robot": near_robot,
        "clipped": clipped,
        "table_xy": None if centre_hit is None else _r3(centre_hit[:2]),
        "bottom_xy": _r3(bottom),
    }


def _fill_holes(mask, max_hole):
    m = mask.astype(np.uint8)
    inv = 1 - m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > max_hole or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        out[labels == i] = True
    return out


def overlay(scene, seg):
    cam = scene.cams[seg["camera"]]
    img = cam.bgr.copy()
    dim = ~seg["inside"]
    img[dim] = (img[dim] * 0.45).astype(np.uint8)
    rob = seg["robot"]
    img[rob] = (0.5 * img[rob] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
    palette = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 128, 255), (255, 128, 128)]
    for obj, mask in zip(seg["objects"], seg["masks"]):
        colour = palette[obj["id"] % len(palette)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, 1)
    for obj in seg["objects"]:
        colour = palette[obj["id"] % len(palette)]
        u, v = int(obj["centroid_px"][0]), int(obj["centroid_px"][1])
        text = str(obj["id"])
        cv2.putText(img, text, (u - 4, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (u - 4, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", img)
    return png.tobytes()


# ---- locate (plane sweep of two silhouettes) ---------------------------------------------------------------------
def _grid_for(scene, cam, mask, top=MAX_HEIGHT_M):
    """xy grid (origin, cols, rows) covering where the region can be for heights 0..top."""
    ys, xs = np.nonzero(mask)
    pts = []
    for u, v in ((xs.min(), ys.min()), (xs.max(), ys.min()), (xs.min(), ys.max()), (xs.max(), ys.max()),
                 (xs.mean(), ys.mean())):
        for h in (0.0, top):
            p = cam.pixel_on_plane(float(u), float(v), scene.table_z + h)
            if p is not None:
                pts.append(p[:2])
    if not pts:
        return None
    pts = np.array(pts)
    lo = np.maximum(pts.min(0) - 0.03, [WORK_X[0] - 0.1, WORK_Y[0] - 0.1])
    hi = np.minimum(pts.max(0) + 0.03, [WORK_X[1] + 0.1, WORK_Y[1] + 0.3])
    if np.any(hi <= lo):
        return None
    cols, rows = int(np.ceil((hi[0] - lo[0]) / GRID_M)), int(np.ceil((hi[1] - lo[1]) / GRID_M))
    if cols * rows > 800 * 800:
        return None
    return lo, cols, rows


def _cast(cam, mask8, z, grid):
    origin, cols, rows = grid
    hom = cam.plane_homography(z, origin, GRID_M)
    return cv2.warpPerspective(mask8, hom, (cols, rows), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _peak(heights, scores):
    j = int(np.argmax(scores))
    h = float(heights[j])
    if 0 < j < len(heights) - 1:
        y0, y1, y2 = scores[j - 1:j + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            h += 0.5 * (y0 - y2) / den * (heights[1] - heights[0])
    return h, float(scores[j])


class _Casts:
    """Masks of one camera cast onto the planes z = table_z + h of an xy grid, with the camera's validity there
    (the cell projects into the image and not onto the robot), cached per height."""

    def __init__(self, scene, cam_name, grid):
        self.scene, self.cam, self.grid = scene, scene.cams[cam_name], grid
        robot, _ = scene.robot(cam_name)
        self.free8 = (~robot).astype(np.uint8)
        self._valid = {}

    def cast(self, mask8, h):
        return _cast(self.cam, mask8, self.scene.table_z + h, self.grid).astype(bool)

    def valid(self, h):
        key = round(float(h), 5)
        if key not in self._valid:
            self._valid[key] = self.cast(self.free8, h)
        return self._valid[key]


def _iou_profile(casts_a, a_cast, casts_b, mask_b8, heights):
    """IoU of A's and B's region on each plane, counted only where both cameras see the plane (not the robot,
    inside the image); also the share of A's cast that B can see."""
    ious, seen = np.zeros(len(heights)), np.zeros(len(heights))
    for k, h in enumerate(heights):
        pa = a_cast[k]
        both = casts_a.valid(h) & casts_b.valid(h)
        pb = casts_b.cast(mask_b8, h)
        na = np.count_nonzero(pa)
        a_in, b_in = pa & both, pb & both
        union = np.count_nonzero(a_in | b_in)
        ious[k] = np.count_nonzero(a_in & b_in) / union if union else 0.0
        seen[k] = np.count_nonzero(a_in) / na if na else 0.0
    return ious, seen


def _lab_mean(cam, mask):
    return cam.lab()[mask].reshape(-1, 3).astype(float).mean(0)


def _colour_distance(lab_a, lab_b):
    """Colour difference of two regions' mean (smoothed) Lab; lightness counts less (shading differs by view)."""
    d = lab_a - lab_b
    return float(np.sqrt((d[0] / 2.5) ** 2 + d[1] ** 2 + d[2] ** 2))


def _band(cam_a, mask_a, cam_b, table_z, top):
    """Bounding box in B of where A's region can appear for heights 0..top (corner rays of A's box)."""
    ys, xs = np.nonzero(mask_a)
    pts = []
    for u, v in ((xs.min(), ys.min()), (xs.max(), ys.min()), (xs.min(), ys.max()), (xs.max(), ys.max())):
        for h in np.linspace(0.0, top, 5):
            p = cam_a.pixel_on_plane(float(u), float(v), table_z + h)
            if p is not None:
                pts.append(p)
    if not pts:
        return None
    uv, z = cam_b.project(np.array(pts))
    uv = uv[z > 0.02]
    if not len(uv):
        return None
    return uv[:, 0].min() - 15, uv[:, 1].min() - 15, uv[:, 0].max() + 15, uv[:, 1].max() + 15


MIN_IOU = 0.25            # below: not the same object
MAX_COLOUR = 40.0         # mean colour difference above which two regions are not the same object
MIN_SEEN = 0.3            # share of the object (as cast from A) the other camera must see
LIFTED_M = 0.015          # the views stop agreeing this far above the table: the object does not stand on it
MIN_IOU_LIFTED = 0.55     # ... then a stricter overlap is needed (two different objects can cross in mid-air)


def locate(scene, name_a, mask_a, obj_a, camera_b="auto"):
    started = time.thread_time()
    cam_a = scene.cams[name_a]
    top_a = min(MAX_HEIGHT_M, cam_a.centre[2] - scene.table_z - 0.03)   # planes must lie below the cameras
    if top_a < 0.02:
        return {"plan_ok": False, "reason": f"camera {name_a} is too close to the table to compare heights"}
    top = top_a
    grid = _grid_for(scene, cam_a, mask_a, top)
    if grid is None:
        return {"plan_ok": False, "reason": "the region does not map onto the table"}
    others = [c for c in CAMERA_CHOICES if c != name_a and c in scene.cams] if camera_b == "auto" else [camera_b]
    coarse = np.arange(0.0, top + 1e-9, 0.006)
    casts_a = _Casts(scene, name_a, grid)
    a8 = mask_a.astype(np.uint8)
    a_coarse = [casts_a.cast(a8, h) for h in coarse]
    lab_a = _lab_mean(cam_a, mask_a)
    pairs, notes = [], []
    for name_b in others:
        if name_b == name_a or name_b not in scene.cams:
            continue
        cam_b = scene.cams[name_b]
        top_b = min(top_a, cam_b.centre[2] - scene.table_z - 0.03)
        if top_b < 0.02:
            notes.append(f"{name_b}: too close to the table")
            continue
        n_h = int(np.searchsorted(coarse, top_b + 1e-9))
        band = _band(cam_a, mask_a, cam_b, scene.table_z, top_b)
        if band is None or band[2] < 0 or band[3] < 0 or band[0] >= cam_b.w or band[1] >= cam_b.h:
            notes.append(f"{name_b}: the object is outside this camera's view")
            continue
        seg_b = scene.segment(name_b, min_area=15)
        casts_b = _Casts(scene, name_b, grid)
        best, rejected = None, []
        for obj_b, mask_b in zip(seg_b["objects"], seg_b["masks"]):
            u0, v0, u1, v1 = obj_b["bbox"]
            if u1 < band[0] or u0 > band[2] or v1 < band[1] or v0 > band[3]:
                continue
            b8 = mask_b.astype(np.uint8)
            ious, seen = _iou_profile(casts_a, a_coarse[:n_h], casts_b, b8, coarse[:n_h])
            j = int(np.argmax(ious))
            if ious[j] <= 0.05:
                continue
            dcol = _colour_distance(lab_a, _lab_mean(cam_b, mask_b))
            if dcol > MAX_COLOUR:
                rejected.append(f"id {obj_b['id']} (overlap {ious[j]:.2f}, other colour)")
                continue
            # the same object resting on something overlaps from its top down to its base; two different
            # objects overlap only around the height where their viewing cones cross
            k = j
            while k > 0 and ious[k - 1] > 0.05:
                k -= 1
            base = float(coarse[k])
            fine = np.arange(max(0.0, coarse[j] - 0.008), min(top_b, coarse[j] + 0.008) + 1e-9, 0.001)
            a_fine = [casts_a.cast(a8, h) for h in fine]
            fious, fseen = _iou_profile(casts_a, a_fine, casts_b, b8, fine)
            h, iou = _peak(fine, fious)
            score = iou * (1.0 - 0.5 * dcol / MAX_COLOUR) * (1.0 if base <= LIFTED_M else 0.6)
            cand = {"camera": name_b, "id": obj_b["id"], "height": h, "iou": iou, "colour_diff": dcol,
                    "base": base, "seen": float(fseen[int(np.argmax(fious))]), "mask": mask_b, "casts": casts_b,
                    "score": score}
            if best is None or score > best["score"]:
                best = cand
        if best is None:
            notes.append(f"{name_b}: no region of the same colour overlaps the object at any height"
                         + (f" (rejected: {', '.join(rejected)})" if rejected else ""))
            continue
        if best["iou"] < MIN_IOU:
            notes.append(f"{name_b}: best overlap {best['iou']:.2f} (id {best['id']}) is too low for the same object")
            continue
        if best["base"] > LIFTED_M and best["iou"] < MIN_IOU_LIFTED and not obj_a.get("near_robot"):
            notes.append(f"{name_b}: id {best['id']} agrees only {best['height'] - best['base']:.3f} m below its top and "
                         f"not down to the table (overlap {best['iou']:.2f}); not taken as the same object")
            continue
        if best["seen"] < MIN_SEEN:
            notes.append(f"{name_b}: sees only {100 * best['seen']:.0f}% of the object (image border or robot)")
            continue
        pairs.append(best)
    if not pairs:
        bx, by = obj_a["bottom_xy"]
        arm = "left" if bx < 0 else "right"
        return {"plan_ok": False, "reason": "seen in one camera only: no other camera shows a matching region "
                "(out of view, hidden by the robot or another object, or not separated from the table)",
                "notes": notes, "bottom_xy": obj_a["bottom_xy"],
                "hint": f"let a wrist camera see it: point the {arm} gripper down with the fingers opening along x "
                        f"and put its TCP near ({bx:.2f}, {by + 0.12:.2f}, {scene.table_z + 0.18:.2f}); that wrist "
                        f"camera then sees the table from 0.04 m ahead to 0.30 m behind the TCP (toward the robot). "
                        f"Then run segment and locate again",
                "cpu_s": round(time.thread_time() - started, 2)}
    pairs.sort(key=lambda p: -p["score"])
    heights = np.array([p["height"] for p in pairs])
    h = float(np.median(heights))
    used = [p for p in pairs if abs(p["height"] - h) <= 0.015]
    if len(used) < len(pairs):
        notes.append("cameras disagree on the height: " + ", ".join(f"{p['camera']} {p['height']:.3f}" for p in pairs))
    h = float(np.mean([p["height"] for p in used]))
    z_top = scene.table_z + h
    top_face = casts_a.cast(a8, h)
    for p in used:
        vb = p["casts"].valid(h)
        top_face &= p["casts"].cast(p["mask"].astype(np.uint8), h) | ~vb      # a camera only removes what it sees
    rows, cols = np.nonzero(top_face)
    if len(rows) < 3:
        return {"plan_ok": False, "reason": "the views do not overlap at the measured height", "notes": notes}
    origin = grid[0]
    xy = np.stack([origin[0] + (cols + 0.5) * GRID_M, origin[1] + (rows + 0.5) * GRID_M], 1).astype(np.float32)
    (_, _), (w, l), _ = cv2.minAreaRect(xy)
    if w > l:
        w, l = l, w
    centre = xy.mean(0)
    out = {
        "plan_ok": True,
        "top_z": round(z_top, 3),
        "top_xyz": _r3([centre[0], centre[1], z_top]),
        "footprint_xy": _r3(centre),
        "extents": {"length": round(float(l), 3), "width": round(float(w), 3), "height": round(h, 3)},
        "yaw_deg": round(_long_axis_yaw(xy), 1),
        "iou": round(used[0]["iou"], 2),
        "conf": round(float(min(1.0, max(p["iou"] for p in used) * min(1.0, 0.5 + 0.5 * len(used)))), 2),
        "cameras": [name_a] + [p["camera"] for p in used],
        "matched_ids": {p["camera"]: p["id"] for p in used},
        "pair_heights": {p["camera"]: round(p["height"], 3) for p in pairs},
        "colour_diff": {p["camera"]: round(p["colour_diff"], 1) for p in used},
        "cpu_s": round(time.thread_time() - started, 2),
    }
    if min(p["base"] for p in used) > LIFTED_M:
        out["lifted"] = True
        out["base_height"] = round(min(p["base"] for p in used), 2)
    if any(p["seen"] < 0.9 for p in used):
        out["partly_seen"] = {p["camera"]: round(p["seen"], 2) for p in used}
    if notes:
        out["notes"] = notes
    return out


def _long_axis_yaw(xy):
    c = xy - xy.mean(0)
    sxx, syy, sxy = float((c[:, 0] ** 2).mean()), float((c[:, 1] ** 2).mean()), float((c[:, 0] * c[:, 1]).mean())
    ang = 0.5 * float(np.degrees(np.arctan2(2 * sxy, sxx - syy)))      # principal axis of the 2x2 covariance
    return (ang + 90.0) % 180.0 - 90.0


# ---- episode access -------------------------------------------------------------------------------------------------
def _holder(api):
    try:
        return next(iter(api.arms.values()))
    except (AttributeError, StopIteration, TypeError):
        return None


def _robot_files(api):
    """(urdf, cuRobo yml) of the robot, from the planner's configuration; env ROBOSHELL_ROBOT_URDF overrides."""
    env = os.environ.get("ROBOSHELL_ROBOT_URDF")
    if env:
        yml = os.path.join(os.path.dirname(env), "curobo.yml")
        return env, yml if os.path.exists(yml) else None
    try:
        planner = api.planner(next(iter(api.arms)))
        yml = getattr(planner, "yml_path", None)
        if yml and os.path.exists(yml):
            import yaml
            with open(yml) as handle:
                kin = yaml.safe_load(handle)["robot_cfg"]["kinematics"]
            urdf = kin.get("urdf_path")
            if urdf and not os.path.isabs(urdf):
                urdf = os.path.join(kin.get("asset_root_path") or os.path.dirname(yml), urdf)
            if urdf and os.path.exists(urdf):
                return urdf, yml
    except Exception:
        pass
    return None, None


def _model(urdf, yml):
    if not urdf:
        return None
    try:
        return RobotModel.load(urdf, yml)
    except Exception:
        return None


def scene_from_api(api):
    obs = api.observe()
    cams = {}
    for name, source in SOURCES.items():
        png, cam = (obs.get("png") or {}).get(source), (obs.get("cameras") or {}).get(source)
        if png is None or cam is None:
            continue
        cams[name] = Camera(name, _decode(png), cam["intrinsics"], cam["extrinsics_world"])
    table_z = table_plane.measure_table_z(api)
    arms = []
    for tag, arm in api.arms.items():
        base = None
        try:
            base = _pose_matrix(api.robot_base_pose(tag))
        except Exception:
            base = None
        try:
            joints = np.asarray(arm.joints(), float)[:6]
        except Exception:
            joints = None
        try:
            ee = np.asarray(arm.ee(), float)
        except Exception:
            ee = None
        arms.append(ArmState(tag, base, joints, ee, float(arm.gripper())))
    model = _model(*_robot_files(api))
    signature = {name: _image_signature(cam.bgr) for name, cam in cams.items()}
    return Scene(cams, table_z, arms, model, signature)


def _image_signature(bgr):
    return cv2.resize(bgr, (80, 60), interpolation=cv2.INTER_AREA).astype(np.int16)


def _same_image(a, b):
    return a is not None and b is not None and a.shape == b.shape and float(np.abs(a - b).mean()) < 1.5


def scene_from_dir(obs_dir, prefix, table_z=None, joints=None, urdf=None, base_poses=None):
    """Saved observation files -> Scene. joints: {tag: 6 values} if known; base_poses: {tag: [x y z qw qx qy qz]}."""
    with open(os.path.join(obs_dir, f"{prefix}_state.json")) as handle:
        state = json.load(handle)
    cams = {}
    for name in CAMERA_CHOICES:
        path, cam = os.path.join(obs_dir, f"{prefix}_{name}.png"), state.get("cameras", {}).get(name)
        if cam is None or not os.path.exists(path):
            continue
        cams[name] = Camera(name, cv2.imread(path, cv2.IMREAD_COLOR), cam["intrinsics"], cam["extrinsics_world"])
    if table_z is None:
        table_z = table_plane.measure_dir(obs_dir, prefix)["table_z"]
    base_poses = base_poses or {"left": [-0.3, -0.45, 0.765, 0.7071068, 0, 0, 0.7071068],
                                "right": [0.3, -0.45, 0.765, 0.7071068, 0, 0, 0.7071068]}
    arms = []
    for tag in ("left", "right"):
        if tag not in state:
            continue
        tcp = _pose_matrix(state[tag]["tcp_pos"] + state[tag]["tcp_quat"])
        off = np.eye(4)
        off[:3, 3] = [TCP_OFFSET_M, 0, 0]
        ee = tcp @ _rigid_inverse(off)
        arms.append(ArmState(tag, _pose_matrix(base_poses[tag]) if tag in base_poses else None,
                             None if joints is None or joints.get(tag) is None else np.asarray(joints[tag], float),
                             ee, float(state[tag].get("gripper", 1.0))))
    urdf = urdf or os.environ.get("ROBOSHELL_ROBOT_URDF")
    yml = None
    if urdf:
        cand = os.path.join(os.path.dirname(urdf), "curobo.yml")
        yml = cand if os.path.exists(cand) else None
    return Scene(cams, table_z, arms, _model(urdf, yml))


# ---- commands -------------------------------------------------------------------------------------------------------
def _public(seg):
    return [dict(obj) for obj in seg["objects"]]


def cmd_segment(scene, camera, min_area):
    started = time.thread_time()
    if scene.table_z is None:
        return {"plan_ok": False, "reason": "table height unknown (robo table_plane could not measure it); "
                "move an arm so that its wrist camera sees more of the table and try again"}, EXIT_FAILED
    if camera not in scene.cams:
        return {"plan_ok": False, "reason": f"no image from camera {camera}"}, EXIT_FAILED
    seg = scene.segment(camera, min_area)
    png = overlay(scene, seg)
    seen = {tag: g for tag, g in scene.openings().items() if g is not None}
    out = {"plan_ok": True, "camera": camera, "table_z": round(scene.table_z, 4), "n_objects": len(seg["objects"]),
           "objects": _public(seg), "robot_mask": seg["robot_model"]}
    if seen:
        out["gripper_seen"] = seen
    out["cpu_s"] = round(time.thread_time() - started, 2)
    out["files"] = {f"seg_{camera}.png": base64.b64encode(png).decode()}
    return out, 0


def _pick_region(seg, obj_id, u, v):
    if obj_id is not None:
        for obj, mask in zip(seg["objects"], seg["masks"]):
            if obj["id"] == obj_id:
                return obj, mask
        return None, None
    ui, vi = int(round(u)), int(round(v))
    best, best_d = None, 1e9
    for obj, mask in zip(seg["objects"], seg["masks"]):
        h, w = mask.shape
        if 0 <= vi < h and 0 <= ui < w and mask[vi, ui]:
            return obj, mask
        ys, xs = np.nonzero(mask)
        d = float(np.min(np.hypot(xs - u, ys - v)))
        if d < best_d:
            best, best_d = (obj, mask), d
    if best is not None and best_d <= 6:
        return best
    return None, None


def run(api, command, args):
    try:
        if command == "pixel_world":
            return _run_pixel_world(api, args)
        holder = _holder(api)
        cache = getattr(holder, CACHE_ATTR, None) if holder is not None else None
        if cache is None:
            cache = {}
            if holder is not None:
                try:
                    setattr(holder, CACHE_ATTR, cache)
                except (AttributeError, TypeError):
                    pass
        with table_plane._OneThread():
            scene = scene_from_api(api)
            camera = args.get("camera") or "head"
            if command == "segment":
                out, code = cmd_segment(scene, camera, int(args.get("min_area") or 40))
                if code == 0:
                    seg = scene.segment(camera, int(args.get("min_area") or 40))
                    cache[camera] = {"signature": scene.signature.get(camera), "objects": seg["objects"],
                                     "masks": seg["masks"]}
                return out, code
            if command == "locate":
                return _run_locate(scene, cache, args)
        return {"plan_ok": False, "error": f"unknown command {command}"}, EXIT_FAILED
    except Exception as error:      # a failed measurement must not break the episode
        return {"plan_ok": False, "error": f"{type(error).__name__}: {error}"}, EXIT_FAILED


def _run_locate(scene, cache, args):
    camera = args.get("camera") or "head"
    obj_id, u, v = args.get("id"), args.get("u"), args.get("v")
    if obj_id is None and (u is None or v is None):
        return {"plan_ok": False, "error": "give --id N (from segment) or both --u and --v"}, EXIT_FAILED
    if scene.table_z is None:
        return {"plan_ok": False, "reason": "table height unknown"}, EXIT_FAILED
    if camera not in scene.cams:
        return {"plan_ok": False, "reason": f"no image from camera {camera}"}, EXIT_FAILED
    if obj_id is not None:
        kept = cache.get(camera)
        if kept is None:
            return {"plan_ok": False, "error": f"no segment of camera {camera} in this episode yet: "
                    f"run `robo segment --camera {camera}` first"}, EXIT_FAILED
        if not _same_image(kept["signature"], scene.signature.get(camera)):
            return {"plan_ok": False, "error": f"the {camera} image changed since the last segment; "
                    f"run `robo segment --camera {camera}` again and use its ids"}, EXIT_FAILED
        seg = {"objects": kept["objects"], "masks": kept["masks"]}
    else:
        seg = scene.segment(camera, min_area=15)
    obj, mask = _pick_region(seg, obj_id, u, v)
    if obj is None:
        what = f"id {obj_id}" if obj_id is not None else f"pixel ({u}, {v})"
        return {"plan_ok": False, "error": f"no object region at {what} in camera {camera}"}, EXIT_FAILED
    camera_b = args.get("camera_b") or "auto"
    if camera_b == camera:
        return {"plan_ok": False, "error": "--camera_b must differ from --camera"}, EXIT_FAILED
    out = locate(scene, camera, mask, obj, camera_b)
    out["id"] = obj["id"]
    out["camera"] = camera
    return out, 0 if out.get("plan_ok") else EXIT_FAILED


def _run_pixel_world(api, args):
    obs = api.observe()
    camera = args.get("camera") or "head"
    cam = (obs.get("cameras") or {}).get(SOURCES[camera])
    if cam is None:
        return {"plan_ok": False, "reason": f"no camera {camera}"}, EXIT_FAILED
    z = args.get("z")
    if z is None:
        z = table_plane.measure_table_z(api)
        if z is None:
            return {"plan_ok": False, "reason": "table height unknown; give --z"}, EXIT_FAILED
    k, t = np.asarray(cam["intrinsics"], float), np.asarray(cam["extrinsics_world"], float)
    c = Camera(camera, np.zeros((2, 2, 3), np.uint8), k, t)
    p = c.pixel_on_plane(float(args["u"]), float(args["v"]), float(z))
    if p is None:
        return {"plan_ok": False, "reason": "the ray of this pixel does not go down to that plane"}, EXIT_FAILED
    return {"plan_ok": True, "xyz": _r3(p), "z_plane": round(float(z), 4)}, 0


# ---- offline -------------------------------------------------------------------------------------------------------
def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["segment", "locate"])
    parser.add_argument("obs_dir")
    parser.add_argument("prefix", nargs="?", default="000_reset")
    parser.add_argument("--camera", default="head", choices=CAMERA_CHOICES)
    parser.add_argument("--id", type=int)
    parser.add_argument("--min_area", type=int, default=40)
    parser.add_argument("--table_z", type=float)
    parser.add_argument("--out", default=".")
    a = parser.parse_args(argv)
    scene = scene_from_dir(a.obs_dir, a.prefix, a.table_z)
    if a.command == "segment":
        out, code = cmd_segment(scene, a.camera, a.min_area)
        png = out.pop("files", {})
        for name, data in png.items():
            with open(os.path.join(a.out, name), "wb") as handle:
                handle.write(base64.b64decode(data))
        print(json.dumps(out))
        return code
    seg = scene.segment(a.camera, a.min_area)
    obj, mask = _pick_region(seg, a.id, None, None)
    if obj is None:
        print(json.dumps({"plan_ok": False, "error": "no such id"}))
        return EXIT_FAILED
    print(json.dumps(locate(scene, a.camera, mask, obj)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
