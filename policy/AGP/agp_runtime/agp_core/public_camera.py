"""Camera geometry of the RoboDojo dual-X5 setup, derived only from public RoboDojo files.

The official leaderboard observation contains three 640x480 RGB images, joint states, `left/right_ee_pose`,
the commanded gripper value and the instruction; depth and camera intrinsics/extrinsics are switched off.
Every constant here is parsed from files of the public RoboDojo repository (nothing is read from the
simulator, a depth image or a calibration file):

    env_cfg/arx_x5.yml                     which camera / robot / scene config the benchmark loads
    env_cfg/camera/camera_config.yml       head camera: sensor type, mount pose (pos, Euler ori in degrees)
    env_cfg/camera/template.py             sensor templates (resolution, focal length, apertures), PINHOLE mount
    env/camera_manager/camera_manager.py   REAL_MAP (sensor type -> template), VISUAL_MAP (mesh -> mount template)
    utils/rotations.py                     Euler convention of euler_angles_to_quat (default: intrinsic XYZ)
    env_cfg/robot/dual_x5.yml              arm base poses (first robot = left arm)
    Assets/Robots/x5/robot_config.yml      ee_link, gripper_bias, wrist camera mount (link, type, pos, ori)
    Assets/Robots/x5/X5A.urdf              joints, including the fixed link6 -> camera_base -> camera chain
    env_cfg/scene/default.yml              table centre height and thickness

Conventions (verified against the simulator and all saved calibrations by agp_core/check_public_camera.py):
  * Pixel (u, v) is image[v, u] (u = column, v = row). Its centre is at continuous image coordinates
    (u + 0.5, v + 0.5) and the principal point (width/2, height/2) = (320, 240) is in those coordinates.
    All functions take pixel indices (int or float) and add the 0.5 themselves.
  * fx = fy = width * focal_length / horizontal_aperture (square pixels, as the official
    CameraManager.get_camera_intrinsics computes). The configured vertical aperture is not what the renderer
    uses: Isaac's Camera.get_intrinsics_matrix() returns a wrong fy = 336.46 for the head camera.
  * Optical frame = ROS/OpenCV (+X right, +Y down, +Z forward) = USD camera frame (-Z forward, +Y up)
    times diag(1, -1, -1).
  * World = the env frame of the official observation: `*_ee_pose` and the head camera pose are both
    relative to the env origin.
  * `<arm>_ee_pose` = [x, y, z, qw, qx, qy, qz] of link6 (robot_config ee_link), position relative to the env
    origin (RobotManager.get_real_endpose -> get_link_pose(is_relative=True)).
  * TCP = our runtime's convention (migration/kinematics.x5_tcp_transform): origin gripper_bias along
    link6 +X, axes TCP x = link6 y (closing), TCP y = link6 z, TCP z = link6 x (approach).
  * Arm base frames = our runtime's nominal frames: dual_x5.yml default_root_pos and the normalised
    default_root_rot. The configured quaternion (0.707, 0, 0, 0.707) is not unit length and the simulated
    bases end up at yaw 90.0173 deg instead of 90 (3.0e-4 rad). Camera geometry is unaffected because the
    wrist cameras follow link6, whose observed pose already contains it; `physical_T_world_base` recovers
    the simulated base from one observation (ee_pose and joints) if it is ever needed.

Dependencies: numpy, scipy, PyYAML (plus the standard library).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from agp_core import paths

ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ALIASES = {"top": "cam_head", "head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
PIXEL_CENTER = 0.5
USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])
_PROJECT = Path(__file__).resolve().parents[1]
# without ROBODOJO_ROOT: the upstream tree (agp_core/paths.py: AGP_UPSTREAM_ROOT, else where the checkout's Assets link
# points), then W's copy
DEFAULT_ROOTS = (paths.upstream(), _PROJECT / "source" / "RoboDojo")


@dataclass(frozen=True)
class Intrinsics:
    sensor: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------- small helpers

def transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inverse(T):
    """Inverse of a rigid 4x4 transform."""
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ t)


def pose_matrix(pose):
    """4x4 from a 4x4 or a 7-vector [x, y, z, qw, qx, qy, qz] (official ee_pose and our tcp_pose_base layout)."""
    a = np.asarray(pose, float)
    if a.shape == (4, 4):
        return a.copy()
    if a.shape != (7,):
        raise ValueError(f"pose must be 4x4 or [x, y, z, qw, qx, qy, qz], got shape {a.shape}")
    return transform(Rotation.from_quat(a[[4, 5, 6, 3]]).as_matrix(), a[:3])


def pose_vector(T):
    """7-vector [x, y, z, qw, qx, qy, qz] of a 4x4."""
    T = np.asarray(T, float)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def intersect_z(origin, direction, z):
    """Points where rays origin + s*direction (s > 0) meet the horizontal plane at height z (NaN if none)."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.asarray(z, float) - o[..., 2]) / d[..., 2]
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    return o + s[..., None] * d


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_literals(path):
    """Module-level NAME = <literal> assignments of a Python file (parsed, never imported)."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _module_name_maps(path, names):
    """Module-level dicts {"literal": NAME, ...} (e.g. REAL_MAP) as {"literal": "NAME"}."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names and isinstance(node.value, ast.Dict)):
            out[node.targets[0].id] = {ast.literal_eval(k): v.id for k, v in zip(node.value.keys, node.value.values)
                                       if isinstance(v, ast.Name)}
    missing = set(names) - set(out)
    if missing:
        raise ValueError(f"{path}: cannot find {sorted(missing)}")
    return out


def _euler_order(rotations_py, camera_manager_py):
    """scipy order used for camera 'ori' Euler angles: utils.rotations.euler_angles_to_quat's `extrinsic`
    default ('XYZ' intrinsic unless extrinsic=True), unless CameraManager passes `extrinsic` explicitly."""
    default = None
    for node in ast.walk(ast.parse(Path(rotations_py).read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "euler_angles_to_quat":
            args = node.args.args
            for arg, value in zip(args[len(args) - len(node.args.defaults):], node.args.defaults):
                if arg.arg == "extrinsic":
                    default = ast.literal_eval(value)
    if default is None:
        raise ValueError(f"{rotations_py}: euler_angles_to_quat(extrinsic=...) default not found")
    passed = set()
    for node in ast.walk(ast.parse(Path(camera_manager_py).read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "euler_angles_to_quat":
            passed |= {ast.literal_eval(k.value) for k in node.keywords if k.arg == "extrinsic"}
    if passed - {default}:
        raise ValueError(f"{camera_manager_py} overrides the Euler convention: {passed}")
    return "xyz" if default else "XYZ"


def _parse_urdf(path):
    def vec(element, key, default):
        text = element.get(key, default) if element is not None else default
        return np.array([float(x) for x in text.split()])

    joints = {}
    for j in ET.parse(path).getroot().findall("joint"):
        origin, axis = j.find("origin"), j.find("axis")
        joints[j.get("name")] = {
            "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": vec(origin, "xyz", "0 0 0"), "rpy": vec(origin, "rpy", "0 0 0"), "axis": vec(axis, "xyz", "1 0 0")}
    return joints


def _urdf_origin(joint):
    # URDF rpy: fixed-axis roll, pitch, yaw, i.e. R = Rz(yaw) Ry(pitch) Rx(roll) = scipy extrinsic "xyz".
    return transform(Rotation.from_euler("xyz", joint["rpy"]).as_matrix(), joint["xyz"])


def _urdf_chain(joints, ancestor, link):
    """Joint names from `ancestor` down to `link`."""
    by_child = {j["child"]: name for name, j in joints.items()}
    chain = []
    while link != ancestor:
        if link not in by_child:
            raise ValueError(f"URDF link {link!r} is not below {ancestor!r}")
        chain.append(by_child[link])
        link = joints[by_child[link]]["parent"]
    return chain[::-1]


def _x5_tcp_transform(gripper_bias):
    # Same as migration/kinematics.x5_tcp_transform (checked by agp_core/check_public_camera.py).
    return transform(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [gripper_bias, 0.0, 0.0])


def _find_root(root):
    candidates = [Path(root)] if root else (
        [Path(os.environ["ROBODOJO_ROOT"])] if os.environ.get("ROBODOJO_ROOT") else list(DEFAULT_ROOTS))
    for c in candidates:
        if (c / "env_cfg" / "camera" / "camera_config.yml").is_file():
            return c.resolve()
    raise FileNotFoundError(f"No RoboDojo checkout among {[str(c) for c in candidates]} (set ROBODOJO_ROOT)")


# ----------------------------------------------------------------------------- the model

class PublicCameraModel:
    """All camera constants of the dual-X5 RoboDojo setup, derived from the public files under `root`."""

    def __init__(self, root=None):
        self.root = _find_root(root)
        r = self.root
        top = yaml.safe_load((r / "env_cfg" / "arx_x5.yml").read_text())["config"]
        files = {
            "arx_x5": r / "env_cfg" / "arx_x5.yml",
            "camera_config": r / "env_cfg" / "camera" / f"{top['camera']}.yml",
            "template": r / "env_cfg" / "camera" / "template.py",
            "camera_manager": r / "env" / "camera_manager" / "camera_manager.py",
            "rotations": r / "utils" / "rotations.py",
            "robots": r / "env_cfg" / "robot" / f"{top['robot']}.yml",
            "scene": r / "env_cfg" / "scene" / f"{top['scene']}.yml",
        }
        templates = _module_literals(files["template"])
        maps = _module_name_maps(files["camera_manager"], ("REAL_MAP", "VISUAL_MAP"))
        self.euler_order = _euler_order(files["rotations"], files["camera_manager"])

        # Arm bases (our runtime's nominal frames) and robot description.
        robots = [x for x in yaml.safe_load(files["robots"].read_text())["robots"] if x.get("robot_type") == "arm"]
        if len(robots) != 2 or len({x["robot_name"] for x in robots}) != 1:
            raise NotImplementedError("expected the dual X5 setup (two arms of one robot type)")
        robot_dir = r / "Assets" / "Robots" / robots[0]["robot_name"]
        files["robot_config"] = robot_dir / "robot_config.yml"
        robot_cfg = yaml.safe_load(files["robot_config"].read_text())
        files["urdf"] = (robot_dir / robot_cfg["urdf_path"]).resolve()
        self.T_world_base = {}
        for arm, cfg in zip(ARMS, robots):     # RobotManager: first target arm = left_arm, second = right_arm
            w, x, y, z = cfg["default_root_rot"]
            self.T_world_base[arm] = transform(Rotation.from_quat([x, y, z, w]).as_matrix(), cfg["default_root_pos"])
        self.base_quaternion_norm = float(np.linalg.norm(robots[0]["default_root_rot"]))

        # Kinematics from the URDF.
        self._joints = _parse_urdf(files["urdf"])
        self.ee_link = robot_cfg["ee_link"]
        self.base_link = robot_cfg.get("base_link", "base_link")
        self.arm_joint_names = list(robot_cfg["arm_joints_name"])
        self.gripper_bias = float(robot_cfg["gripper_bias"])
        self.T_link6_tcp = _x5_tcp_transform(self.gripper_bias)
        self._fk_chain = _urdf_chain(self._joints, self.base_link, self.ee_link)

        # Cameras.
        def mount(camera):
            pos = camera.get("pos", [0.0, 0.0, 1.0])
            ori = camera.get("ori")
            if ori is None:
                R = np.eye(3)
            elif len(ori) == 3:
                R = Rotation.from_euler(self.euler_order, ori, degrees=True).as_matrix()
            else:                                   # explicit quaternion (w, x, y, z)
                R = Rotation.from_quat(np.asarray(ori, float)[[1, 2, 3, 0]]).as_matrix()
            visual = templates[maps["VISUAL_MAP"][camera["mesh"]]]
            if "asset_path" in visual:
                raise NotImplementedError(f"camera mesh {camera['mesh']!r} uses a referenced USD asset")
            w, x, y, z = visual["orientation"]
            T_xform_usd = transform(Rotation.from_quat([x, y, z, w]).as_matrix(), visual["position"])
            return transform(R, pos) @ T_xform_usd @ transform(USD_TO_OPTICAL)

        def intrinsics(sensor):
            t = templates[maps["REAL_MAP"][sensor]]
            width, height = t["resolution"]
            f = width * t["focal_length"] / t["horizontal_aperture"]
            return Intrinsics(sensor, int(width), int(height), f, f, width / 2.0, height / 2.0)

        cam_cfg = yaml.safe_load(files["camera_config"].read_text())
        if cam_cfg.get("random"):
            raise NotImplementedError("camera randomisation is configured; camera poses are not fixed")
        self.intrinsics = {}
        head = [name for name, c in cam_cfg.items() if isinstance(c, dict) and "camera" in c]
        if head != ["cam_head"] or cam_cfg["cam_head"]["camera"].get("mount_link") is not None:
            raise NotImplementedError(f"unexpected scene cameras {head}")
        c = cam_cfg["cam_head"]["camera"]
        self.intrinsics["cam_head"] = intrinsics(c["type"])
        self.T_world_head_optical = mount(c)   # xform is a child of the env prim -> env (= world) frame

        wrist = [c for c in robot_cfg.get("camera", []) if c.get("name", "").endswith("_wrist")]
        if len(wrist) != 1:
            raise NotImplementedError("expected exactly one wrist camera per arm")
        c = wrist[0]
        self.camera_link = c["link"]
        chain = _urdf_chain(self._joints, self.ee_link, self.camera_link)
        if any(self._joints[j]["type"] != "fixed" for j in chain):
            raise NotImplementedError(f"camera link {self.camera_link!r} is not rigidly attached to {self.ee_link!r}")
        self.T_link6_camera_link = np.eye(4)
        for j in chain:
            self.T_link6_camera_link = self.T_link6_camera_link @ _urdf_origin(self._joints[j])
        self.T_camera_link_optical = mount(c)
        self.T_link6_optical = self.T_link6_camera_link @ self.T_camera_link_optical
        self.T_tcp_optical = inverse(self.T_link6_tcp) @ self.T_link6_optical
        self.wrist_arm = {}
        for arm in ARMS:                        # RobotManager._load_camera_cfg naming for two target arms
            name = c["name"].replace("cam_", f"cam_{arm}_", 1)
            self.wrist_arm[name] = arm
            self.intrinsics[name] = intrinsics(c["type"])

        # Table top (unit cube scaled by `scale`, centred at default_pos).
        table = yaml.safe_load(files["scene"].read_text())["Table"]
        cx, cy, cz = table["default_pos"]
        sx, sy, sz = table["scale"]
        self.table_top_z = float(cz + sz / 2.0)
        self.table_extent = ((cx - sx / 2.0, cx + sx / 2.0), (cy - sy / 2.0, cy + sy / 2.0))
        self.sources = {k: {"path": str(v), "sha256": _sha256(v)} for k, v in files.items()}

    # ------------------------------------------------------------------ names and frames
    @staticmethod
    def camera_name(camera):
        name = ALIASES.get(camera, camera)
        if name not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}")
        return name

    def K(self, camera):
        return self.intrinsics[self.camera_name(camera)].K

    def world_to_base(self, arm, points):
        T = inverse(self.T_world_base[arm])
        return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]

    def base_to_world(self, arm, points):
        T = self.T_world_base[arm]
        return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]

    def table_top(self, frame="world", arm=None):
        """Height of the table top in the world frame or in an arm's base frame (bases are yaw-only)."""
        if frame == "world":
            return self.table_top_z
        if frame != "base" or arm not in ARMS:
            raise ValueError("frame must be 'world' or 'base' with arm 'left'/'right'")
        return self.table_top_z - float(self.T_world_base[arm][2, 3])

    # ------------------------------------------------------------------ extrinsics
    def T_world_optical(self, camera, ee_pose=None):
        """World (env) <- optical. Wrist cameras need the official `<arm>_ee_pose` (link6) of their arm."""
        name = self.camera_name(camera)
        if name == "cam_head":
            return self.T_world_head_optical.copy()
        if ee_pose is None:
            raise ValueError(f"{name} needs the official {self.wrist_arm[name]}_ee_pose")
        return pose_matrix(ee_pose) @ self.T_link6_optical

    def tcp_in_base(self, arm, ee_pose):
        """The arm's TCP pose in its own base frame (our runtime's tcp_pose_base) from the official ee_pose."""
        return inverse(self.T_world_base[arm]) @ pose_matrix(ee_pose) @ self.T_link6_tcp

    def T_base_optical(self, camera, arm=None, tcp_pose_base=None):
        """Arm base <- optical. Wrist cameras need their own arm's TCP pose in that arm's base frame
        (our runtime's tcp_pose_base); `arm` selects the output base frame (default: the wrist camera's arm)."""
        name = self.camera_name(camera)
        if name == "cam_head":
            if arm not in ARMS:
                raise ValueError("the head camera needs arm='left' or 'right' for a base-frame pose")
            return inverse(self.T_world_base[arm]) @ self.T_world_head_optical
        own = self.wrist_arm[name]
        if tcp_pose_base is None:
            raise ValueError(f"{name} needs the {own} arm's TCP pose in the {own} base frame")
        T = pose_matrix(tcp_pose_base) @ self.T_tcp_optical
        arm = own if arm is None else arm
        if arm != own:
            T = inverse(self.T_world_base[arm]) @ self.T_world_base[own] @ T
        return T

    def _T(self, camera, pose, frame, arm):
        if frame == "world":
            return self.T_world_optical(camera, ee_pose=pose)
        if frame == "base":
            return self.T_base_optical(camera, arm=arm, tcp_pose_base=pose)
        raise ValueError("frame must be 'world' or 'base'")

    # ------------------------------------------------------------------ rays
    def pixel_ray(self, camera, u, v, pose=None, frame="world", arm=None):
        """Ray through the centre of pixel (u, v) (image[v, u]); u, v may be arrays.

        pose (wrist cameras only): frame='world' -> the official `<arm>_ee_pose` (link6, env frame);
        frame='base' -> the camera arm's TCP pose in its own base frame (runtime tcp_pose_base).
        arm (frame='base'): output base frame; required for the head camera.
        Returns (origin (3,), unit direction (..., 3)) in the requested frame."""
        T = self._T(camera, pose, frame, arm)
        k = self.intrinsics[self.camera_name(camera)]
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        d = np.stack([(u + PIXEL_CENTER - k.cx) / k.fx, (v + PIXEL_CENTER - k.cy) / k.fy, np.ones_like(u)], -1)
        d = d @ T[:3, :3].T
        return T[:3, 3].copy(), d / np.linalg.norm(d, axis=-1, keepdims=True)

    def pixel_to_plane(self, camera, u, v, z, pose=None, frame="world", arm=None):
        """3D point(s) where the ray through pixel (u, v) meets the horizontal plane at height z
        (same frame as the result; NaN if the plane is behind the camera)."""
        o, d = self.pixel_ray(camera, u, v, pose=pose, frame=frame, arm=arm)
        return intersect_z(o, d, z)

    def project(self, camera, points, pose=None, frame="world", arm=None):
        """Pixel-index coordinates (u, v) (so image[round(v), round(u)]) and optical depth of 3D points."""
        T = self._T(camera, pose, frame, arm)
        k = self.intrinsics[self.camera_name(camera)]
        p = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
        uv = np.stack([k.fx * p[..., 0] / p[..., 2] + k.cx, k.fy * p[..., 1] / p[..., 2] + k.cy], -1) - PIXEL_CENTER
        return uv, p[..., 2]

    # ------------------------------------------------------------------ kinematics (public URDF)
    def link6_in_base(self, joints):
        """URDF forward kinematics: base_link <- link6 for the six arm joint angles (rad)."""
        q = dict(zip(self.arm_joint_names, np.asarray(joints, float)))
        T = np.eye(4)
        for name in self._fk_chain:
            j = self._joints[name]
            T = T @ _urdf_origin(j)
            if j["type"] in ("revolute", "continuous"):
                T = T @ transform(Rotation.from_rotvec(j["axis"] * q[name]).as_matrix())
            elif j["type"] == "prismatic":
                T = T @ transform(t=j["axis"] * q[name])
        return T

    def physical_T_world_base(self, ee_pose, joints):
        """The simulated base pose recovered from one observation: ee_pose (link6) times inverse FK."""
        return pose_matrix(ee_pose) @ inverse(self.link6_in_base(joints))

    # ------------------------------------------------------------------ export
    def constants(self):
        f = lambda a: np.asarray(a, float).tolist()
        return {
            "root": str(self.root),
            "sources": self.sources,
            "conventions": {"pixel_center": PIXEL_CENTER, "optical": "ROS/OpenCV, = USD camera x diag(1,-1,-1)",
                            "euler_order_scipy": self.euler_order, "world": "env frame of the official observation",
                            "ee_pose": f"{self.ee_link} [x,y,z,qw,qx,qy,qz] relative to the env origin",
                            "base_quaternion_norm": self.base_quaternion_norm},
            "K": {name: f(k.K) for name, k in self.intrinsics.items()},
            "T_world_head_optical": f(self.T_world_head_optical),
            "T_link6_camera_link": f(self.T_link6_camera_link),
            "T_camera_link_optical": f(self.T_camera_link_optical),
            "T_link6_optical": f(self.T_link6_optical),
            "T_link6_tcp": f(self.T_link6_tcp),
            "T_tcp_optical": f(self.T_tcp_optical),
            "T_world_base": {arm: f(T) for arm, T in self.T_world_base.items()},
            "table_top_z_world": self.table_top_z,
            "table_extent_xy_world": self.table_extent,
        }


@lru_cache(maxsize=None)
def default_model(root=None):
    return PublicCameraModel(root)


def pixel_ray(camera, u, v, pose=None, frame="world", arm=None):
    return default_model().pixel_ray(camera, u, v, pose=pose, frame=frame, arm=arm)


def pixel_to_plane(camera, u, v, z, pose=None, frame="world", arm=None):
    return default_model().pixel_to_plane(camera, u, v, z, pose=pose, frame=frame, arm=arm)


if __name__ == "__main__":
    import json
    print(json.dumps(default_model().constants(), indent=1))
