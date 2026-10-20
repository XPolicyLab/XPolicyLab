"""Inference-only FK extracted from the exact training sidecar generator.

Upstream SHA256: ac637a8d1abca40d2174ffc7c411af14043d0cd58eba8ee899850f273a8a9d20
"""
from __future__ import annotations

import hashlib

import math

import xml.etree.ElementTree as ET

from pathlib import Path

from typing import Any, Iterable

import torch

from torch import nn

DEFAULT_ALOHA_AGILEX_URDF_SHA256 = (
    "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
)

ROBOTWIN_ARM_NAMES = ("left", "right")

ROBOTWIN_ARM_JOINT_INDICES = ((0, 1, 2, 3, 4, 5), (7, 8, 9, 10, 11, 12))

def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _parse_vector(value: str | None, *, length: int, default: Iterable[float]) -> list[float]:
    values = list(default) if value is None else [float(item) for item in value.split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {values!r}")
    return values

def _rpy_matrix(rpy: Iterable[float], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    roll, pitch, yaw = [float(value) for value in rpy]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    # URDF uses fixed-axis roll, pitch, yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=dtype,
    )

def _transform_matrix(
    xyz: Iterable[float],
    rotation: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    result = torch.eye(4, dtype=dtype)
    result[:3, :3] = rotation.to(dtype=dtype)
    result[:3, 3] = torch.as_tensor(list(xyz), dtype=dtype)
    return result

def _quaternion_wxyz_matrix(
    quaternion: Iterable[float], *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    q = torch.as_tensor(list(quaternion), dtype=dtype)
    if q.shape != (4,):
        raise ValueError(f"Expected wxyz quaternion with shape [4], got {tuple(q.shape)}")
    norm = torch.linalg.vector_norm(q)
    if float(norm) < 1.0e-12:
        raise ValueError("Robot world-pose quaternion must be non-zero")
    w, x, y, z = q / norm
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )

def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return skew matrices for tensors with shape ``[..., 3]``."""

    if vector.shape[-1] != 3:
        raise ValueError(f"Expected shape [...,3], got {tuple(vector.shape)}")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )

def axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors ``[...,3]`` with a stable Rodrigues formula."""

    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle shape [...,3], got {tuple(axis_angle.shape)}")
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    skew = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(axis_angle.shape[:-1] + (3, 3))
    theta2 = theta.square()
    small = theta < 1.0e-4
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    safe_theta2 = safe_theta.square()
    # Taylor branches avoid 0/0 and use only elementary Torch operations that
    # are available on CPU, CUDA, and Ascend NPU backends.
    a_regular = torch.sin(theta) / safe_theta
    b_regular = (1.0 - torch.cos(theta)) / safe_theta2
    a_taylor = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    a = torch.where(small, a_taylor, a_regular)[..., None]
    b = torch.where(small, b_taylor, b_regular)[..., None]
    return eye + a * skew + b * (skew @ skew)

def matrices_to_pose7(matrices: torch.Tensor) -> torch.Tensor:
    """Convert homogeneous transforms to ``[...,xyz,qw,qx,qy,qz]``.

    Pick the best-conditioned quaternion branch, including rotations near pi.
    Quaternion signs are canonicalized to nonnegative w; action representations
    (especially continuous axis-angle) must be derived *after* chunk selection.
    """
    if matrices.shape[-2:] != (4, 4) or not matrices.is_floating_point():
        raise ValueError("matrices_to_pose7 requires floating [...,4,4] transforms")
    if not torch.isfinite(matrices).all():
        raise ValueError("Non-finite transform in matrices_to_pose7")
    r = matrices[..., :3, :3]
    m00, m01, m02 = r[..., 0, :].unbind(-1)
    m10, m11, m12 = r[..., 1, :].unbind(-1)
    m20, m21, m22 = r[..., 2, :].unbind(-1)
    squared = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                           1 - m00 + m11 - m22, 1 - m00 - m11 + m22), dim=-1)
    magnitudes = torch.sqrt(squared.clamp_min(torch.finfo(r.dtype).eps))
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
    ), dim=-2) / (2 * magnitudes.clamp_min(0.1)[..., :, None])
    selection = magnitudes.argmax(dim=-1)[..., None, None].expand(magnitudes.shape[:-1] + (1, 4))
    quaternion = candidates.gather(-2, selection).squeeze(-2)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    return torch.cat((matrices[..., :3, 3], quaternion), dim=-1)

def infer_fk_pose7(joints: torch.Tensor, *, fk: "RobotWinUrdfFK") -> torch.Tensor:
    """Shared offline/online EEF definition from raw absolute 14D drive targets.

    Do not pass normalized values, joint deltas, simulator actual qpos or an
    endpose observation. They have different semantics from the training input.
    """
    return matrices_to_pose7(fk(joints))

class RobotWinUrdfFK(nn.Module):
    """Pure-Torch FK for the two Aloha-AgileX arms used by RoboTwin.

    Input tensors must end in the 14D RoboTwin joint/action layout.  The left
    six arm joints are read from ``0:6`` and the right six from ``7:13``; the
    gripper values at indices 6 and 13 do not participate in FK.  The result is
    ordered ``[left, right]`` and has shape ``[...,2,4,4]``.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        embodiment_config_path: str | Path,
        *,
        expected_urdf_sha256: str | None = DEFAULT_ALOHA_AGILEX_URDF_SHA256,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        self.validate_finite = bool(validate_finite)
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.embodiment_config_path = Path(embodiment_config_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if not self.embodiment_config_path.is_file():
            raise FileNotFoundError(f"Embodiment config not found: {self.embodiment_config_path}")
        self.urdf_sha256 = sha256_file(self.urdf_path)
        if expected_urdf_sha256 and self.urdf_sha256.lower() != expected_urdf_sha256.lower():
            raise ValueError(
                "RoboTwin Aloha-AgileX URDF SHA256 mismatch: "
                f"expected={expected_urdf_sha256}, actual={self.urdf_sha256}, path={self.urdf_path}"
            )

        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - training env always has PyYAML
            raise RuntimeError("RobotWinUrdfFK requires PyYAML to read embodiment config") from exc
        with self.embodiment_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise TypeError(f"Expected YAML mapping: {self.embodiment_config_path}")

        arm_joint_names = config.get("arm_joints_name")
        endpoint_links = config.get("move_group")
        if (
            not isinstance(arm_joint_names, list)
            or len(arm_joint_names) != 2
            or any(not isinstance(names, list) or len(names) != 6 for names in arm_joint_names)
        ):
            raise ValueError("embodiment config arm_joints_name must contain two lists of six joints")
        if not isinstance(endpoint_links, list) or len(endpoint_links) != 2:
            raise ValueError("embodiment config move_group must contain left/right endpoint links")

        robot_pose = config.get("robot_pose")
        if isinstance(robot_pose, list) and len(robot_pose) == 1 and isinstance(robot_pose[0], list):
            robot_pose = robot_pose[0]
        if not isinstance(robot_pose, list) or len(robot_pose) != 7:
            raise ValueError("embodiment config robot_pose must be [xyz,qw,qx,qy,qz]")
        world_root = _transform_matrix(
            robot_pose[:3],
            _quaternion_wxyz_matrix(robot_pose[3:]),
        )
        self.register_buffer("world_root_transform", world_root, persistent=True)

        root = ET.parse(self.urdf_path).getroot()
        joint_records: dict[str, dict[str, Any]] = {}
        child_to_joint: dict[str, str] = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name")
            joint_type = joint.attrib.get("type", "fixed")
            parent_node = joint.find("parent")
            child_node = joint.find("child")
            if not name or parent_node is None or child_node is None:
                raise ValueError(f"Malformed URDF joint in {self.urdf_path}")
            parent = parent_node.attrib["link"]
            child = child_node.attrib["link"]
            if child in child_to_joint:
                raise ValueError(f"URDF child link {child!r} has multiple parent joints")
            origin_node = joint.find("origin")
            xyz = _parse_vector(
                None if origin_node is None else origin_node.attrib.get("xyz"),
                length=3,
                default=(0.0, 0.0, 0.0),
            )
            rpy = _parse_vector(
                None if origin_node is None else origin_node.attrib.get("rpy"),
                length=3,
                default=(0.0, 0.0, 0.0),
            )
            axis_node = joint.find("axis")
            axis = _parse_vector(
                None if axis_node is None else axis_node.attrib.get("xyz"),
                length=3,
                default=(1.0, 0.0, 0.0),
            )
            joint_records[name] = {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": _transform_matrix(xyz, _rpy_matrix(rpy)),
                "axis": torch.tensor(axis, dtype=torch.float64),
            }
            child_to_joint[child] = name

        self.arm_joint_names = tuple(tuple(str(name) for name in names) for names in arm_joint_names)
        self.endpoint_links = tuple(str(link) for link in endpoint_links)
        self.root_links: list[str] = []
        self._chain_buffer_names: list[tuple[str, str, str]] = []
        self._chain_metadata: list[tuple[tuple[int, int], ...]] = []
        for arm_index, (names, endpoint) in enumerate(zip(self.arm_joint_names, self.endpoint_links)):
            chain: list[dict[str, Any]] = []
            link = endpoint
            visited: set[str] = set()
            while link in child_to_joint:
                joint_name = child_to_joint[link]
                if joint_name in visited:
                    raise ValueError(f"Cycle while tracing URDF chain to {endpoint!r}")
                visited.add(joint_name)
                record = joint_records[joint_name]
                chain.append(record)
                link = record["parent"]
            chain.reverse()
            self.root_links.append(link)
            actuated = [record["name"] for record in chain if record["type"] != "fixed"]
            if actuated != [str(name) for name in names]:
                raise ValueError(
                    f"URDF chain for {endpoint!r} has actuated joints {actuated}, expected {names}"
                )
            q_lookup = {str(name): index for index, name in enumerate(names)}
            origins = torch.stack([record["origin"] for record in chain])
            axes = torch.stack([record["axis"] for record in chain])
            type_codes: list[int] = []
            q_indices: list[int] = []
            for record in chain:
                joint_type = record["type"]
                if joint_type == "fixed":
                    type_codes.append(0)
                    q_indices.append(-1)
                elif joint_type in {"revolute", "continuous"}:
                    type_codes.append(1)
                    q_indices.append(q_lookup[record["name"]])
                elif joint_type == "prismatic":
                    type_codes.append(2)
                    q_indices.append(q_lookup[record["name"]])
                else:
                    raise ValueError(f"Unsupported URDF joint type {joint_type!r} in arm chain")
            prefix = ROBOTWIN_ARM_NAMES[arm_index]
            origin_name = f"{prefix}_joint_origins"
            axis_name = f"{prefix}_joint_axes"
            meta_name = f"{prefix}_joint_type_and_q_index"
            self.register_buffer(origin_name, origins, persistent=True)
            self.register_buffer(axis_name, axes, persistent=True)
            self.register_buffer(
                meta_name,
                torch.tensor(list(zip(type_codes, q_indices)), dtype=torch.int64),
                persistent=True,
            )
            self._chain_buffer_names.append((origin_name, axis_name, meta_name))
            self._chain_metadata.append(tuple(zip(type_codes, q_indices)))
        if self.root_links[0] != self.root_links[1]:
            raise ValueError(f"Left/right arm chains have different roots: {self.root_links}")

    @staticmethod
    def _motion_transform(axis: torch.Tensor, value: torch.Tensor, type_code: int) -> torch.Tensor:
        leading = value.shape
        if type_code == 1:
            normalized_axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
            rotation_vector = value[..., None] * normalized_axis
            rotation = axis_angle_to_matrix_torch(rotation_vector)
            translation = torch.zeros(leading + (3, 1), dtype=value.dtype, device=value.device)
        elif type_code == 2:
            rotation = torch.eye(3, dtype=value.dtype, device=value.device).expand(leading + (3, 3))
            normalized_axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
            translation = (value[..., None] * normalized_axis)[..., None]
        else:
            raise ValueError(f"Motion transform requires revolute/prismatic code, got {type_code}")
        upper = torch.cat((rotation, translation), dim=-1)
        bottom = torch.zeros(leading + (1, 4), dtype=value.dtype, device=value.device)
        bottom[..., 0, 3] = 1
        return torch.cat((upper, bottom), dim=-2)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(joints):
            raise TypeError("RobotWinUrdfFK input must be a torch.Tensor")
        if not joints.is_floating_point():
            raise TypeError(f"RobotWinUrdfFK requires floating-point joints, got {joints.dtype}")
        if joints.shape[-1] != 14:
            raise ValueError(f"Expected RoboTwin joint layout [...,14], got {tuple(joints.shape)}")
        if self.validate_finite and not torch.isfinite(joints).all():
            raise ValueError("RobotWinUrdfFK input contains non-finite values")

        leading = joints.shape[:-1]
        world_root = self.world_root_transform.to(dtype=joints.dtype, device=joints.device)
        world_root = world_root.expand(leading + (4, 4))
        outputs: list[torch.Tensor] = []
        for arm_index, (origin_name, axis_name, _) in enumerate(self._chain_buffer_names):
            arm_indices = torch.tensor(
                ROBOTWIN_ARM_JOINT_INDICES[arm_index], dtype=torch.long, device=joints.device
            )
            arm_q = joints.index_select(-1, arm_indices)
            origins = getattr(self, origin_name).to(dtype=joints.dtype, device=joints.device)
            axes = getattr(self, axis_name).to(dtype=joints.dtype, device=joints.device)
            metadata = self._chain_metadata[arm_index]
            transform = world_root
            for segment_index in range(origins.shape[0]):
                transform = transform @ origins[segment_index]
                type_code, q_index = metadata[segment_index]
                if type_code:
                    motion = self._motion_transform(
                        axes[segment_index].to(dtype=joints.dtype, device=joints.device),
                        arm_q[..., q_index],
                        type_code,
                    )
                    transform = transform @ motion
            outputs.append(transform)
        return torch.stack(outputs, dim=-3)

    def extra_repr(self) -> str:
        return (
            f"urdf={self.urdf_path}, endpoints={self.endpoint_links}, "
            f"root={self.root_links[0]}, sha256={self.urdf_sha256[:12]}..., "
            f"validate_finite={self.validate_finite}"
        )
