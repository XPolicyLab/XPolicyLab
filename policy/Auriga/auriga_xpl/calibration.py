"""Camera calibration and wrist-camera forward kinematics for the stock RoboDojo arx_x5 cell.

The stock observation config (RoboDojo ``env_cfg/arx_x5.yml``) sets ``intrinsic_matrix`` and
``extrinsic_matrix`` to false, so the calibration that Auriga's ``rgb_proprio_calibrated_v1``
profile needs is supplied here from constants instead of the simulator camera prims:

- ``assets/arx_x5_calibration.json``: the output of Auriga's worker ``_camera_calibration()``
  (``auriga/robot/backends/robodojo/robot_worker.py``), dumped with
  ``tools/dump_calibration.py`` on several tasks (including ``*_random`` ones) and frozen once.
  It holds K for all three cameras (fy := fx, square pixels), the fixed head-camera
  ``T_world_camera``, and for each wrist camera ``T_world_base`` and ``T_link_camera``.
- ``vendor/robodojo_planner/Assets/Robots/x5/X5A.urdf``: the RoboDojo ARX X5 URDF shared with the
  planner and the Auriga adapter (vendored, see ``vendor/robodojo_planner/NOTICE``).
  Wrist extrinsics are ``T_world_base @ FK_urdf(q)[link_name] @ T_link_camera``, which is what
  the worker computes from the simulator link pose (``source_kind = robot_kinematics``).

Everything is pure numpy + the standard library; nothing imports Isaac or RoboDojo.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
CALIBRATION_FILE = ASSETS_DIR / "arx_x5_calibration.json"
STATIC_FILE = ASSETS_DIR / "arx_x5_static.json"
ROBOT_ASSET_DIR = Path(__file__).resolve().parents[1] / "vendor" / "robodojo_planner" / "Assets" / "Robots" / "x5"
URDF_FILE = ROBOT_ASSET_DIR / "X5A.urdf"

HEAD_CAMERA = "cam_head"
WRIST_CAMERAS = {"left": "cam_left_wrist", "right": "cam_right_wrist"}
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAMES = ("joint7", "joint8")
FRAME_ID = "env_origin"


# --------------------------------------------------------------------------------------------
# Frozen constants
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _calibration_document() -> dict:
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(f"{CALIBRATION_FILE} missing; regenerate it with tools/dump_calibration.py")
    document = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    calibration = document["camera_calibration"]
    digest = hashlib.sha256(_canonical(calibration)).hexdigest()
    if document.get("camera_calibration_sha256") not in (None, digest):
        raise ValueError("arx_x5_calibration.json: camera_calibration checksum mismatch")
    return document


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_calibration() -> dict:
    """Return the frozen calibration, structured exactly like the worker's ``camera_calibration``.

    ``{"profile_id", "source_kind", "reference_frame_id", "cameras": {camera_id: {...}}}`` where
    every camera has ``K`` (3x3), ``distortion`` (5 zeros), ``distortion_model``, ``image_size``
    ([width, height]), ``optical_axes``, ``calibration_source``; ``cam_head`` adds
    ``T_world_camera``; the wrist cameras add ``arm``, ``link_name``, ``T_world_base`` and
    ``T_link_camera``. Matrices are nested lists (row-major, metres). A fresh copy is returned.
    """
    return copy.deepcopy(_calibration_document()["camera_calibration"])


def calibration_provenance() -> dict:
    """Dump provenance and the cross-task consistency record stored next to the constants."""
    document = _calibration_document()
    return copy.deepcopy({key: value for key, value in document.items() if key != "camera_calibration"})


def load_static() -> dict:
    """Static robot values (control rate, dt, ticks per action, joint/velocity limits, ...)."""
    if not STATIC_FILE.is_file():
        raise FileNotFoundError(f"{STATIC_FILE} missing; regenerate it with tools/dump_calibration.py")
    return json.loads(STATIC_FILE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# URDF model (same structure as auriga.robot.model.robot_model.urdf_manifest)
# --------------------------------------------------------------------------------------------

def _rpy_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> list:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
            [-sp, cp * sr, cp * cr, xyz[2]],
            [0.0, 0.0, 0.0, 1.0]]


@lru_cache(maxsize=4)
def _manifest(path: str) -> dict:
    data = Path(path).read_bytes()
    root = ET.fromstring(data)
    joints = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        axis = node.find("axis")
        joint = {"name": node.attrib["name"], "type": node.attrib["type"],
                 "parent": node.find("parent").attrib["link"], "child": node.find("child").attrib["link"],
                 "T_parent_joint": _rpy_matrix(xyz, rpy),
                 "axis": [float(x) for x in axis.get("xyz", "1 0 0").split()] if axis is not None else [1.0, 0.0, 0.0]}
        limit = node.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("robot_model_invalid_joint_limits")
            joint["limits"] = {"lower": lower, "upper": upper}
        mimic = node.find("mimic")
        if mimic is not None:
            joint["mimic"] = {"joint": mimic.attrib["joint"], "multiplier": float(mimic.get("multiplier", 1)),
                              "offset": float(mimic.get("offset", 0))}
        joints.append(joint)
    children = {joint["child"] for joint in joints}
    roots = {joint["parent"] for joint in joints} - children
    if len(roots) != 1:
        raise ValueError("robot_model_requires_single_root")
    base = next(iter(roots))
    ordered, resolved = [], {base}
    while joints:
        ready = [joint for joint in joints if joint["parent"] in resolved]
        if not ready:
            raise ValueError("robot_model_cycle_or_disconnected_chain")
        ordered.extend(ready)
        for joint in ready:
            resolved.add(joint["child"])
            joints.remove(joint)
    return {"schema_version": "0.1", "robot_name": root.get("name"), "base_link": base, "joints": ordered,
            "source_kind": "permitted_robot_urdf", "source_sha256": hashlib.sha256(data).hexdigest(),
            "source_filename": Path(path).name,
            "geometry_scope": "joint_frames_and_declared_limits; no task assets or mesh references"}


def robot_model() -> dict:
    """``urdf_manifest`` of the vendored URDF (the worker's observation ``robot_model`` field)."""
    return copy.deepcopy(_manifest(str(URDF_FILE)))


# --------------------------------------------------------------------------------------------
# Forward kinematics
# --------------------------------------------------------------------------------------------

def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return np.array([[c + x * x * v, x * y * v - z * s, x * z * v + y * s],
                     [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
                     [z * x * v - y * s, z * y * v + x * s, c + z * z * v]])


def _named_positions(joint_positions, names: Iterable[str]) -> dict:
    if isinstance(joint_positions, Mapping):
        return {str(k): float(v) for k, v in joint_positions.items()}
    values = [float(v) for v in np.asarray(joint_positions, dtype=float).reshape(-1)]
    names = list(names)
    if len(values) == 6:
        return dict(zip(ARM_JOINT_NAMES, values))
    if len(values) == 7:  # 6 arm joints + one gripper joint (joint8 mimics joint7)
        return {**dict(zip(ARM_JOINT_NAMES, values[:6])), "joint7": values[6], "joint8": values[6]}
    if len(values) == 8:
        return dict(zip(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES, values))
    if len(values) == len(names):
        return dict(zip(names, values))
    raise ValueError(f"expected 6, 7 or 8 joint positions, got {len(values)}")


def link_poses(arm: str, joint_positions, *, base: np.ndarray | None = None) -> dict:
    """World-frame (``env_origin``) 4x4 pose of every URDF link of one arm.

    ``joint_positions``: 6 arm joints (joint1..joint6, rad), optionally followed by the gripper
    joints (joint7, joint8, m), or a name->value mapping. Gripper joints default to 0; they do
    not lie on the camera chain. ``base`` overrides the frozen ``T_world_base`` of that arm.
    """
    model = _manifest(str(URDF_FILE))
    positions = _named_positions(joint_positions, [j["name"] for j in model["joints"] if j["type"] != "fixed"])
    if base is None:
        base = np.asarray(load_calibration()["cameras"][WRIST_CAMERAS[arm]]["T_world_base"], dtype=float)
    poses = {model["base_link"]: np.asarray(base, dtype=float)}
    for joint in model["joints"]:
        origin = np.asarray(joint["T_parent_joint"], dtype=float)
        motion = np.eye(4)
        kind = joint["type"]
        if kind != "fixed":
            mimic = joint.get("mimic")
            if mimic:
                value = positions.get(mimic["joint"], 0.0) * mimic["multiplier"] + mimic["offset"]
            elif joint["name"] in positions:
                value = positions[joint["name"]]
            elif joint["name"] in GRIPPER_JOINT_NAMES:
                value = 0.0
            else:
                raise ValueError(f"missing joint position: {joint['name']}")
            axis = np.asarray(joint["axis"], dtype=float)
            axis = axis / np.linalg.norm(axis)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = _axis_angle(axis, value)
            elif kind == "prismatic":
                motion[:3, 3] = axis * value
            else:
                raise ValueError(f"unsupported joint type {kind}")
        poses[joint["child"]] = poses[joint["parent"]] @ origin @ motion
    return poses


@lru_cache(maxsize=2)
def _wrist_constants(arm: str):
    camera = load_calibration()["cameras"][WRIST_CAMERAS[arm]]
    return (camera["link_name"], np.asarray(camera["T_world_base"], dtype=float),
            np.asarray(camera["T_link_camera"], dtype=float))


def wrist_link_pose(arm: str, joint_positions) -> np.ndarray:
    """``T_world_link`` of the wrist-camera mount link (``camera`` in X5A.urdf)."""
    link, base, _ = _wrist_constants(arm)
    return link_poses(arm, joint_positions, base=base)[link]


def wrist_extrinsics(arm: str, joint_positions) -> np.ndarray:
    """4x4 ``T_world_camera`` (optical axes +x right, +y down, +z forward) of one wrist camera.

    ``arm`` is ``"left"`` or ``"right"``; ``joint_positions`` as for :func:`link_poses`.
    Equals the worker's ``T_world_link @ T_link_camera`` with ``T_world_link`` from URDF FK.
    """
    if arm not in WRIST_CAMERAS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    _, _, link_camera = _wrist_constants(arm)
    return wrist_link_pose(arm, joint_positions) @ link_camera


def head_extrinsics() -> np.ndarray:
    """Fixed 4x4 ``T_world_camera`` of ``cam_head``."""
    return np.asarray(load_calibration()["cameras"][HEAD_CAMERA]["T_world_camera"], dtype=float)


def camera_extrinsics(camera_id: str, joint_positions_by_arm: Mapping | None = None) -> np.ndarray:
    if camera_id == HEAD_CAMERA:
        return head_extrinsics()
    arm = next((side for side, name in WRIST_CAMERAS.items() if name == camera_id), None)
    if arm is None:
        raise ValueError(f"unknown camera {camera_id!r}")
    if joint_positions_by_arm is None or arm not in joint_positions_by_arm:
        raise ValueError(f"{camera_id} needs the {arm} arm joint positions")
    return wrist_extrinsics(arm, joint_positions_by_arm[arm])


def camera_pose_sample(joint_positions_by_arm: Mapping, *, frame_ids: Mapping | None = None,
                       sim_step=None, step_domain: str = "adapter_control_step") -> dict:
    """Build the worker's per-observation ``camera_pose_sample`` from measured arm joints.

    ``joint_positions_by_arm``: ``{"left": q6, "right": q6}`` (or the worker ``robot_state``
    ``arms`` dict, whose ``q_rad`` is used). The head camera keeps the worker's source kind
    (``simulator_camera_transform``, a fixed transform); wrist cameras report
    ``robot_kinematics`` with ``T_world_link`` from URDF FK.
    """
    def arm_q(value):
        return value["q_rad"] if isinstance(value, Mapping) and "q_rad" in value else value

    frame_ids = frame_ids or {}
    cameras = {HEAD_CAMERA: {"T_world_camera": head_extrinsics().tolist(), "source_kind": "simulator_camera_transform"}}
    for arm, camera_id in WRIST_CAMERAS.items():
        link, base, link_camera = _wrist_constants(arm)
        world_link = link_poses(arm, arm_q(joint_positions_by_arm[arm]), base=base)[link]
        cameras[camera_id] = {"T_world_camera": (world_link @ link_camera).tolist(), "source_kind": "robot_kinematics",
                              "arm": arm, "link_name": link, "T_world_link": world_link.tolist(),
                              "T_world_base": base.tolist()}
    for camera_id, item in cameras.items():
        if camera_id in frame_ids:
            item["frame_id"] = frame_ids[camera_id]
    return {"reference_frame_id": FRAME_ID, "sim_step": sim_step, "step_domain": step_domain, "cameras": cameras}


def pose_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(translation error in mm, rotation error in degrees) between two 4x4 transforms."""
    estimate, truth = np.asarray(estimate, dtype=float), np.asarray(truth, dtype=float)
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3])) * 1000.0
    relative = estimate[:3, :3].T @ truth[:3, :3]
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    angle = math.degrees(math.atan2(float(np.linalg.norm(skew)) / 2.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return translation, angle
