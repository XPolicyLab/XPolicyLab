import base64
import cv2
import math
import numpy as np
from pathlib import Path
import yaml
import argparse

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
        
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
        
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def rpy_to_quat_wxyz(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.asarray([qw, qx, qy, qz], dtype=np.float32)

def quat_wxyz_to_rpy(quat):
    qw, qx, qy, qz = [float(v) for v in quat]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)

def eef_xyzrpy_to_pose7(eef):
    eef = np.asarray(eef, dtype=np.float32).reshape(-1)
    if eef.shape[0] < 6:
        raise ValueError(f"Expected eef pose with at least 6 dims (xyz+rpy), got {eef.shape}")
    xyz = eef[:3]
    quat = rpy_to_quat_wxyz(eef[3], eef[4], eef[5])
    return np.concatenate([xyz, quat], axis=0)

def pose7_to_move_eef(pose7):
    pose7 = np.asarray(pose7, dtype=np.float32).reshape(-1)
    if pose7.shape[0] != 7:
        raise ValueError(f"Expected 7D pose [xyz qw qx qy qz], got {pose7.shape}")
    xyz = pose7[:3]
    quat = pose7[3:]
    rpy = quat_wxyz_to_rpy(quat)
    return np.concatenate([xyz, rpy], axis=0)

def decode_color_image(image):
    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError(f"Expected image ndarray with 3 dims, got shape {image.shape}")
        return image

    if isinstance(image, (bytes, bytearray)):
        encoded = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode JPEG bytes from camera stream")
        return decoded[:, :, ::-1]

    raise TypeError(f"Unsupported image type: {type(image)}")

def decode_depth_image(depth):
    if depth is None:
        return None

    if not isinstance(depth, np.ndarray):
        raise TypeError(f"Unsupported depth type: {type(depth)}")

    if depth.ndim == 2:
        return depth

    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]

    raise ValueError(f"Expected depth ndarray with shape (H, W) or (H, W, 1), got {depth.shape}")


def default_intrinsic(width, height):
    fx = width * 0.96
    fy = height * 1.28
    cx = width / 2.0
    cy = height / 2.0
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]

def default_extrinsic():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

def camera_meta(camera_cfg, cam_name, cam_data):
    image = decode_color_image(cam_data["color"])
    depth = decode_depth_image(cam_data.get("depth"))
    shape = image.shape[:2]
    width = image.shape[1]
    height = image.shape[0]
    camera_cfg = camera_cfg.get(cam_name, {})
    return {
        "color": image,
        "depth": depth,
        "intrinsic_matrix": camera_cfg.get("intrinsic_matrix", default_intrinsic(width, height)),
        "extrinsics_matrix": camera_cfg.get("extrinsics_matrix", default_extrinsic()),
        "shape": shape,
    }

def decode_base64_text(encoded):
    if not isinstance(encoded, str):
        raise TypeError(f"Expected base64 payload as str, got {type(encoded)}")

    encoded += "=" * (-len(encoded) % 4)
    return base64.b64decode(encoded)

def decode_numpy_payload(name, payload):
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    expected_nbytes = int(np.prod(shape)) * dtype.itemsize

    if payload.get("__numpy_array__"):
        raw_payload = payload["data"]
    else:
        raw_payload = payload["__numpy__"]

    if isinstance(raw_payload, (bytes, bytearray)):
        data = bytes(raw_payload)
    else:
        data = decode_base64_text(raw_payload)

    if len(data) != expected_nbytes:
        raise ValueError(
            f"Action field {name!r} numpy payload size mismatch: "
            f"shape={shape}, dtype={dtype}, expected_nbytes={expected_nbytes}, "
            f"actual_nbytes={len(data)}"
        )
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

def decode_action_array(name, value):
    if isinstance(value, dict):
        if value.get("__numpy_array__") or "__numpy__" in value:
            return decode_numpy_payload(name, value)
        raise TypeError(
            f"Action field {name!r} is a dict, expected a numeric array or numpy JSON payload. "
            f"keys={list(value.keys())}"
        )

    return np.asarray(value, dtype=np.float32)

def create_move_data(action):
    move_data = {"arm": {}}

    has_prefixed_keys = any(
        key.startswith("left_") or key.startswith("right_")
        for key in action.keys()
    )
    if has_prefixed_keys:
        mapping = [("left_arm", "left_"), ("right_arm", "right_")]
    else:
        mapping = [("arm", "")]

    for arm_name, prefix in mapping:
        arm_action = {}

        joint_key    = f"{prefix}arm_joint_state"
        gripper_key  = f"{prefix}ee_joint_state"
        ee_pose_key  = f"{prefix}ee_pose"
        tcp_pose_key = f"{prefix}tcp_pose"
        delta_key    = f"{prefix}delta_ee_pose"

        if joint_key in action:
            arm_action["joint"] = decode_action_array(joint_key, action[joint_key]).astype(np.float32).reshape(-1)

        if gripper_key in action:
            gripper = decode_action_array(gripper_key, action[gripper_key]).astype(np.float32).reshape(-1)
            if gripper.size > 0:
                arm_action["gripper"] = float(gripper[0]) if gripper.size == 1 else gripper

        pose_key = None
        pose_value = None
        if ee_pose_key in action:
            pose_key = ee_pose_key
            pose_value = action[ee_pose_key]
        elif tcp_pose_key in action:
            pose_key = tcp_pose_key
            pose_value = action[tcp_pose_key]

        if pose_value is not None:
            arm_action["eef"] = pose7_to_move_eef(decode_action_array(pose_key, pose_value))

        if delta_key in action and np.any(decode_action_array(delta_key, action[delta_key]).astype(np.float32)):
            raise NotImplementedError(
                f"Real robot client does not support `{delta_key}` yet. "
                "Use joint or absolute ee/tcp actions."
            )

        if arm_action:
            move_data["arm"][arm_name] = arm_action

    if not move_data["arm"]:
        raise ValueError(f"No supported real-robot action keys found in action: {list(action.keys())}")
    return move_data

def build_state(controller_data):
    state = {}

    arm_names = list(controller_data.keys())
    dual_arm = "left_arm" in controller_data and "right_arm" in controller_data
    if dual_arm:
        mapping = [("left_arm", "left_"), ("right_arm", "right_")]
    elif len(arm_names) == 1:
        mapping = [(arm_names[0], "")]
    else:
        raise ValueError(f"Unsupported controller layout for real env: {arm_names}")

    for arm_name, prefix in mapping:
        arm_data = controller_data[arm_name]
        joint = np.asarray(arm_data.get("joint", []), dtype=np.float32).reshape(-1)
        gripper = np.atleast_1d(np.asarray(arm_data.get("gripper", []), dtype=np.float32)).reshape(-1)
        eef_pose = eef_xyzrpy_to_pose7(arm_data.get("eef", np.zeros(6, dtype=np.float32)))

        state[f"{prefix}arm_joint_state"] = joint
        state[f"{prefix}ee_joint_state"] = gripper
        state[f"{prefix}ee_pose"] = eef_pose
        state[f"{prefix}tcp_pose"] = eef_pose.copy()
        state[f"{prefix}delta_ee_pose"] = np.zeros(7, dtype=np.float32)

    return state

def ensure_uint8_bgr(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)
