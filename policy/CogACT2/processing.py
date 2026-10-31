"""Checkpoint feature layout and RoboDojo camera/pose conversions."""

import json
from numbers import Integral
from pathlib import Path

import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from PIL import Image

from .memory import ImageMemory


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TOOL_ROTATION = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
FEATURES = tuple(f"ROBOT_{side}_{kind}" for side in ("LEFT", "RIGHT")
                 for kind in ("TRANS", "ROT_6D", "GRIPPER"))


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def raw_instruction(obs):
    text = obs.get("instruction", obs.get("instructions"))
    if isinstance(text, (list, tuple)) and len(text) == 1:
        text = text[0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Observation must contain the official episode instruction")
    return text


class Checkpoint:
    def __init__(self, directory, unnorm_key=None):
        self.path = Path(directory)
        self.config = yaml.safe_load((self.path / "config.yaml").read_text())
        self.model = self.config["model"]
        params = self.config["train_dataset"]["datasets_params"]
        self.data = OmegaConf.to_container(OmegaConf.merge(
            params["common"], params["UnifiedV2EpisodicDataset"]), resolve=True)
        self.layout = json.loads((self.path / "action_feature.json").read_text())
        blocks = json.loads((self.path / "norm_config.json").read_text())
        if unnorm_key is not None:
            blocks = [b for b in blocks if b["unnorm_key"] == unnorm_key]
        if len(blocks) != 1:
            raise ValueError("Select exactly one normalization block with unnorm_key")
        self.norm = blocks[0]
        self.stats = self.norm["statistics_dict"][self.norm["unnorm_key"]]
        self.prefix_rows = nonnegative_integer(self.config.get("vla_prepend_rows", 0), "vla_prepend_rows")
        self.chunk_size = nonnegative_integer(self.model["action_chunk_size"], "model.action_chunk_size")
        self.robot_chunk_size = self.chunk_size - self.prefix_rows
        self.memory = ImageMemory(self.model.get("first_frame"), self.data.get("image_history"))
        self.resize_interpolation = self.data.get("resize_interpolation", "opencv")
        self.validate()

    def validate(self):
        m, d = self.model, self.data
        if not 0 <= self.prefix_rows < self.chunk_size:
            raise ValueError("vla_prepend_rows must be smaller than model.action_chunk_size")
        lengths = {
            "action_chunk_size": (self.config["action_chunk_size"], self.chunk_size),
            "dataset.action_chunk_size": (d["action_chunk_size"], self.robot_chunk_size),
            "frame_dataset.action_chunk_size": (
                self.config["train_dataset"]["frame_dataset"]["action_chunk_size"], self.robot_chunk_size),
            "trajectory_chunk_size": (
                self.config.get("trajectory_chunk_size", self.robot_chunk_size), self.robot_chunk_size),
            "collator.vla_prepend_rows": (
                self.config["train_dataset"].get("collator", {}).get("vla_prepend_rows", self.prefix_rows),
                self.prefix_rows),
        }
        for name, (value, expected) in lengths.items():
            if nonnegative_integer(value, name) != expected:
                raise ValueError(f"Inconsistent {name}: expected {expected}, found {value!r}")
        head = m["act_head"]
        flow = head["flow_matching_config"]
        net = flow["net"]
        checks = {
            "Qwen35 encoder": len(m["vl_encoders"]) == 1 and
                m["vl_encoders"][0]["name"] == "Qwen35VLEncoder",
            "system prompt": m["vl_encoders"][0].get("use_system_prompt") is True,
            "MoT flow matching": head["type"] == "flow_matching" and net["model_type"] == "MoT",
            "x0 prediction": flow["predict_x0"] and not flow.get("predict_x0_v", False),
            "single MoT pass": net["loop_config"]["max_repeats"] == 1,
            "full VLM weights": not m.get("lora", {}).get("enabled", False),
            "hwt_rope": head["position_embedding"]["pe_type"] == "hwt_rope",
            "state/FOV conditioning": head["state_fov_injection"] == "adaln" and
                m["state_encoder"]["use_state_in_act_head"] and
                not m["state_encoder"]["use_state_in_llm"] and
                not m["state_encoder"].get("set_state_to_zero", False) and
                not m["state_encoder"].get("state_norm", False),
            "FOV encoder": [k for k, v in m["extra_encoders"].items() if v["enabled"]] == ["fov"] and
                m["extra_encoders"]["fov"]["num_tokens"] == 3 and
                not m["extra_encoders"]["fov"]["use_norm"] and
                not m["extra_encoders"]["fov"]["concat_mask"] and
                m["extra_encoders"]["fov"]["inject_to"] == "act_head",
            "no extra learned tokens": m.get("learnable_token_count", 0) == 0 and
                not m["use_cognition_token"] and not head["only_use_cognition_token_as_input"],
            "three camera views": d["view_number"] == 3,
            "absolute camera actions": d["action_space"] == d["state_space"] == "camera" and
                d["translation_action_from_state_type"] == d["rotation_action_from_state_type"] == "abs",
            "6D rotations": d["rotation_type"] == "6d" and d.get("state_rotation_type") in (None, "6d"),
            "gripper state": d["state_include_gripper"] and d["add_state_gripper"],
            "shared robot slots": not d["separate_robot_action_features"],
            "no progress output": not d.get("use_progress", False),
            "native control frequency": d.get("target_fps") in (None, 25),
            "Gaussian normalization": self.norm["norm_type"] == "gaussian" and
                self.norm["use_action_norm"] and self.norm["use_state_norm"],
            "native-frame image history": not d.get("enable_history_image", False),
            "RGB interpolation": self.resize_interpolation in ("opencv", "pillow_lanczos"),
            "layer fusion": isinstance(m["vlm_feature_layer"], list) and
                all(type(n) is int and n > 0 for n in m["vlm_feature_layer"]),
            "stream routing": net["stream_order"] == ["condition", "action"] and
                net["reads_from"] == {"condition": ["condition"], "action": ["condition", "action"]} and
                all(s["class_name"] == "TransformerStream" and not s["args"].get("input_norm", False)
                    and not s["args"].get("causal", False) and
                    not set(s["args"]) - {"input_norm", "causal"} for s in net["streams"].values()),
        }
        unsupported = [k for k, supported in checks.items() if not supported]
        if unsupported:
            raise ValueError(f"Unsupported CogACT2 checkpoint configuration: {', '.join(unsupported)}")

    def dim(self, kind):
        return self.layout[f"{kind.title()}Feature"]["ALL_FEATURES"][1]

    def slot(self, kind, name):
        return slice(*self.layout[f"{kind.title()}Feature"][name])

    def normalize(self, kind, values, inverse=False):
        result = values.copy()
        for name in FEATURES:
            slot = self.slot(kind, name)
            # The training recipe explicitly leaves grippers in [0, 1].
            if name.endswith("GRIPPER") and self.data.get("force_gripper_identity_norm", False):
                continue
            mean = np.asarray(self.stats[kind]["mean"][name], dtype=np.float32)
            std = np.asarray(self.stats[kind]["std"][name], dtype=np.float32)
            result[..., slot] = (values[..., slot] * std + mean if inverse else
                                 (values[..., slot] - mean) / (std + 1e-8))
        return result

    def action_mask(self):
        mask = np.zeros((self.chunk_size, self.dim("action")), dtype=bool)
        for name in FEATURES:
            mask[self.prefix_rows:, self.slot("action", name)] = True
        return mask

    def image(self, rgb):
        if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or
                rgb.dtype != np.uint8 or 0 in rgb.shape):
            raise ValueError("Expected decoded HWC uint8 RGB from the XPolicyLab server")
        h, w = rgb.shape[:2]
        scale = self.data["resize_short_edge"] / min(h, w)
        short = self.data["resize_short_edge"]
        size = ((max(32, round(w * scale / 32) * 32), short) if h <= w else
                (short, max(32, round(h * scale / 32) * 32)))
        if size == (w, h):
            return rgb.copy()
        if self.resize_interpolation == "pillow_lanczos":
            return np.array(Image.fromarray(rgb).resize(size, resample=Image.Resampling.LANCZOS))
        interpolation = cv2.INTER_LINEAR if size[0] > w or size[1] > h else cv2.INTER_AREA
        return cv2.resize(rgb, size, interpolation=interpolation)

    def instruction(self, obs):
        text = self.data.get("instruction_prefix", "") + raw_instruction(obs)
        if self.config["train_dataset"].get("post_processor", {}).get("instruction_lower", False):
            text = text.lower()
        return text


def camera_geometry(obs):
    origin = np.asarray(obs["env_origin"], dtype=np.float64).reshape(3)
    local_to_world = np.eye(4)
    local_to_world[:3, 3] = origin
    intrinsics, extrinsics = [], []
    for name in CAMERAS:
        camera = obs["vision"][name]
        h, w = camera["color"].shape[:2]
        k = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(3, 3).copy()
        k[0] /= w
        k[1] /= h
        raw = camera.get("extrinsic_matrix", camera.get("extrinsics_matrix"))
        if raw is None or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError(f"Invalid calibration for {name}")
        e = np.asarray(raw, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(e).all() or not np.isfinite(origin).all():
            raise ValueError(f"Nonfinite camera pose for {name}")
        intrinsics.append(k.astype(np.float32))
        extrinsics.append((USD_TO_CV @ np.linalg.inv(e) @ local_to_world).astype(np.float32))
    fov = np.array([[2 * np.arctan(0.5 / k[0, 0]), 2 * np.arctan(0.5 / k[1, 1])]
                    for k in intrinsics], dtype=np.float32)
    return extrinsics[0], fov


def encode_state(obs, world_to_camera, checkpoint):
    state = np.zeros(checkpoint.dim("state"), dtype=np.float32)
    mask = np.zeros_like(state, dtype=bool)
    for side in ("left", "right"):
        pose = np.asarray(obs["state"][f"{side}_ee_pose"], dtype=np.float64).reshape(7)
        if not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1, rtol=0, atol=1e-3):
            raise ValueError(f"{side} EEF pose must be finite with a unit quaternion")
        rotation = (Rotation.from_quat(pose[3:], scalar_first=True).as_matrix() @ TOOL_ROTATION).astype(np.float32)
        rotation = world_to_camera[:3, :3] @ rotation
        values = (pose[:3].astype(np.float32) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3],
                  rotation[:2].reshape(6),
                  [1 - float(np.asarray(obs["state"][f"{side}_ee_joint_state"]).reshape(1)[0])])
        for name, value in zip(("TRANS", "ROT_6D", "GRIPPER"), values):
            slot = checkpoint.slot("state", f"ROBOT_{side.upper()}_{name}")
            state[slot], mask[slot] = value, True
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite robot state")
    return checkpoint.normalize("state", state), mask


def rotation_6d_to_matrix(value):
    first, second = np.split(value, 2, axis=-1)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def decode_actions(values, world_to_camera, checkpoint):
    values = np.asarray(values)
    expected = (checkpoint.chunk_size, checkpoint.dim("action"))
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Expected finite model actions with shape {expected}, found {values.shape}")
    values = checkpoint.normalize("action", values[checkpoint.prefix_rows:], inverse=True)
    camera_to_world = np.linalg.inv(world_to_camera)
    actions = [{} for _ in values]
    for side in ("left", "right"):
        prefix = f"ROBOT_{side.upper()}"
        trans = values[:, checkpoint.slot("action", f"{prefix}_TRANS")]
        trans = trans @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        rot = rotation_6d_to_matrix(values[:, checkpoint.slot("action", f"{prefix}_ROT_6D")])
        rot = camera_to_world[:3, :3] @ rot @ TOOL_ROTATION.T
        quat = Rotation.from_matrix(rot).as_quat(scalar_first=True)
        grip = 1 - values[:, checkpoint.slot("action", f"{prefix}_GRIPPER")]
        for i, action in enumerate(actions):
            action[f"{side}_ee_pose"] = np.concatenate((trans[i], quat[i])).astype(np.float32)
            action[f"{side}_ee_joint_state"] = grip[i].astype(np.float32)
    if not all(np.isfinite(value).all() for action in actions for value in action.values()):
        raise ValueError("Model produced nonfinite actions")
    return actions
