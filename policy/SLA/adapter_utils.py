"""SLA's XPolicyLab boundary; images here are already decoded RGB arrays."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_POSE = (0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707)


def validate_support(bench_name, env_cfg_type, action_type):
    if (bench_name, env_cfg_type, action_type) != ("RoboTwin", "arx_x5", "ee"):
        raise ValueError("SLA supports bench_name=RoboTwin, env_cfg_type=arx_x5, action_type=ee.")


def validate_robot_dims(dims):
    # This is a compatibility check against the supplied ARX-X5 URDF, not a dimension lookup.
    if dims["arm_dim"] != [6, 6] or dims["ee_dim"] != [1, 1]:
        raise ValueError("SLA's RoboTwin checkpoint requires dual 6-DOF ARX-X5 arms and scalar grippers.")


def finite_array(value, name, shape=None):
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def image_chw(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a decoded RGB image, got shape {image.shape}.")
    if image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating RGB images must be finite and in [0, 1].")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("RGB images must be uint8 or floating point in [0, 1].")
    # Also handles read-only decoded arrays without modifying the caller's image.
    return np.array(image, copy=True, order="C")


def encode_observation(obs, dims):
    camera_aliases = {
        "cam_high": ("cam_head", "cam_high", "head_camera"),
        "cam_left_wrist": ("cam_left_wrist", "left_camera"),
        "cam_right_wrist": ("cam_right_wrist", "right_camera"),
    }
    images = {}
    for target, aliases in camera_aliases.items():
        camera = next((obs["vision"][name] for name in aliases if name in obs["vision"]), None)
        if camera is None or "color" not in camera:
            raise KeyError(f"Missing RGB camera {target}; accepted names: {aliases}.")
        images[target] = image_chw(camera["color"])
    prompt = obs.get("instruction") or obs.get("instructions")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Observation requires a non-empty instruction string.")
    # RobotwinInputs performs joint -> EEF internally, even though the output action type is ee.
    state = pack_robot_state(obs, "joint", dims, source_type="obs")
    state = finite_array(state, "joint state", (sum(dims["arm_dim"]) + sum(dims["ee_dim"]),))
    return {"images": images, "state": state, "prompt": prompt}


def eef_actions_to_xpl(actions, dims, base_pose, common_origin):
    validate_robot_dims(dims)
    actions = finite_array(actions, "SLA actions")
    width = sum(6 + gripper_dim for gripper_dim in dims["ee_dim"])
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != width:
        raise ValueError(f"Expected a nonempty [T, {width}] SLA EEF action chunk, got {actions.shape}.")
    base_pose = finite_array(base_pose, "robot_base_pose", (7,)).astype(np.float64)
    origin = finite_array(common_origin, "common EEF origin", (3,)).astype(np.float64)
    quat = base_pose[3:]
    if np.linalg.norm(quat) < 1e-12:
        raise ValueError("robot_base_pose has a zero quaternion.")
    base_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    parts, offset = [], 0
    for gripper_dim in dims["ee_dim"]:
        arm = actions[:, offset:offset + 6 + gripper_dim].astype(np.float64)
        world_xyz = base_rotation.apply(arm[:, :3] + origin) + base_pose[:3]
        world_rotation = base_rotation * Rotation.from_euler("xyz", arm[:, 3:6])
        xyzw = world_rotation.as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        parts.extend([world_xyz, wxyz, arm[:, 6:]])
        offset += 6 + gripper_dim
    packed = np.concatenate(parts, axis=-1).astype(np.float32)
    return unpack_robot_state(packed, "ee", dims, source_type="obs")


def resolve_model_checkpoint(model_cfg):
    root = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    step = model_cfg.get("checkpoint_step")
    if step is not None:
        if not str(step).isdigit():
            raise ValueError("checkpoint_step must be an exact nonnegative integer.")
        root = root / str(step)
    elif not (root / "params").is_dir() and root.is_dir():
        candidates = [p for p in root.iterdir() if p.name.isdigit() and (p / "params").is_dir()]
        if candidates:
            root = max(candidates, key=lambda p: int(p.name))
    if not (root / "params").is_dir() or not (root / "assets").is_dir():
        raise FileNotFoundError(f"Expected params/ and assets/ in checkpoint: {root}")
    return root


def resolve_norm_asset(checkpoint, asset_id=None):
    assets = checkpoint / "assets"
    if asset_id is not None:
        if Path(str(asset_id)).name != str(asset_id) or str(asset_id) in (".", ".."):
            raise ValueError("norm_stats_asset_id must be one directory name within checkpoint/assets.")
        candidates = [assets / str(asset_id)]
    else:
        candidates = sorted(p.parent for p in assets.glob("*/norm_stats.json"))
    if len(candidates) != 1 or not (candidates[0] / "norm_stats.json").is_file():
        raise ValueError(f"Expected exactly one normalization asset in {assets}; set norm_stats_asset_id explicitly.")
    return candidates[0]
