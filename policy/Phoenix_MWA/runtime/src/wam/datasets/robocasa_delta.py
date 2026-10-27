"""RoboCasa GR1 raw targets <-> canonical80, with explicit native-step EEF deltas."""
from __future__ import annotations

import torch
from wam.datasets.robocasa_rotations import (
    _rot6d_columns_to_matrix, _matrix_to_rotvec, _rotvec_to_matrix,
)

CONTRACT = {
    "version": "robocasa_gr1_canonical80_native_delta_v1",
    "world_axes": "source_world_A_identity", "ee_axes": "source_wrist_B_identity",
    "translation": "action_xyz[t]-state_xyz[t]",
    "rotation": "Log(state_R[t].T@action_R[t])",
    "rotation6d": "first_two_columns_concatenated",
    "source_action_offset": 0, "joint_hand_waist": "absolute_source_action",
    "controller": "absolute_joint29", "embodiment_id": 13,
}
RAW = {"left_arm": (0, 7), "left_hand": (7, 13), "right_arm": (22, 29),
       "right_hand": (29, 35), "waist": (41, 44), "wrist_l_pos": (44, 47),
       "wrist_l_rot6d": (47, 53), "wrist_r_pos": (98, 101), "wrist_r_rot6d": (101, 107)}
JOINTS = {"left_arm": (0, 7), "left_hand": (17, 23), "right_arm": (40, 47),
          "right_hand": (57, 63), "waist": (29, 32)}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_modality(modality):
    for kind in ("state", "action"):
        for key, (begin, end) in RAW.items():
            field = modality[kind][key]
            require((field["start"], field["end"], field["original_key"]) ==
                    (begin, end, "observation.state" if kind == "state" else "action"),
                    f"Unsupported RoboCasa modality: {kind}.{key}")


def rotation(raw):
    require(raw.shape[-1] == 6 and bool(torch.isfinite(raw).all()), "Invalid wrist rotation")
    a, b = raw[..., :3], raw[..., 3:]
    require(bool((torch.linalg.cross(a, b).norm(dim=-1) > 1e-6).all()), "Degenerate wrist rotation")
    return _rot6d_columns_to_matrix(raw.reshape(-1, 6)).reshape(*raw.shape[:-1], 3, 3)


def masks():
    state, action = torch.zeros(80, dtype=torch.bool), torch.zeros(80, dtype=torch.bool)
    for begin, end in JOINTS.values():
        state[begin:end] = action[begin:end] = True
    for begin in (7, 47):
        state[begin:begin+9] = True
        action[begin:begin+6] = True
    return state, action


def project(state, action=None):
    """Each row's action is relative to THAT row's state, never chunk-start state."""
    state = torch.as_tensor(state, dtype=torch.float32)
    require(state.ndim == 2 and state.shape[1] == 161, "Expected raw state[N,161]")
    out = torch.zeros(len(state), 80)
    act = torch.zeros_like(out) if action is not None else None
    if action is not None:
        action = torch.as_tensor(action, dtype=torch.float32)
        require(action.shape == state.shape, "Action/state rows must align")
    for name, (begin, end) in JOINTS.items():
        a, b = RAW[name]
        out[:, begin:end] = state[:, a:b]
        if act is not None:
            act[:, begin:end] = action[:, a:b]
    for side, begin in (("l", 7), ("r", 47)):
        p, q = RAW[f"wrist_{side}_pos"], RAW[f"wrist_{side}_rot6d"]
        r = rotation(state[:, q[0]:q[1]])
        out[:, begin:begin+3] = state[:, p[0]:p[1]]
        out[:, begin+3:begin+9] = r[:, :, :2].transpose(-1, -2).reshape(-1, 6)
        if act is not None:
            target_r = rotation(action[:, q[0]:q[1]])
            act[:, begin:begin+3] = action[:, p[0]:p[1]] - state[:, p[0]:p[1]]
            act[:, begin+3:begin+6] = _matrix_to_rotvec(r.transpose(-1, -2) @ target_r)
    require(bool(torch.isfinite(out).all()) and (act is None or bool(torch.isfinite(act).all())),
            "Nonfinite required state/action; refusing silent label substitution")
    return out, act


def decode(action, measured_state=None):
    """Absolute joint commands; optional EEF inverse uses a fresh per-step pose."""
    action = torch.as_tensor(action, dtype=torch.float32)
    require(action.shape[-1] == 80 and bool(torch.isfinite(action).all()), "Invalid canonical action")
    result = {"action." + key: action[..., a:b].clone() for key, (a, b) in JOINTS.items()}
    if measured_state is not None:
        state = torch.as_tensor(measured_state, dtype=torch.float32)
        require(state.shape == (*action.shape[:-1], 161), "EEF inverse needs row-aligned measured state")
        for side, begin in (("l", 7), ("r", 47)):
            p, q = RAW[f"wrist_{side}_pos"], RAW[f"wrist_{side}_rot6d"]
            r = rotation(state[..., q[0]:q[1]])
            delta = _rotvec_to_matrix(action[..., begin+3:begin+6].reshape(-1, 3)).reshape(*r.shape)
            target = r @ delta
            result[f"action.wrist_{side}_pos"] = state[..., p[0]:p[1]] + action[..., begin:begin+3]
            result[f"action.wrist_{side}_rot6d"] = target[..., :, :2].transpose(-1, -2).reshape(*action.shape[:-1], 6)
    return result


def observation_state(observations):
    """One GR1 observation in the existing simulator's named-field protocol."""
    state = torch.zeros(1, 161)
    for key, (a, b) in RAW.items():
        value = torch.as_tensor(observations["state." + key], dtype=torch.float32)
        require(value.numel() == b-a, "Only one environment/current observation is supported")
        state[0, a:b] = value.reshape(-1)
    project(state)
    return state


def prompt_for(instruction):
    text = str(instruction).strip()
    for prefix in ("unlocked_waist:", "locked_waist:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    require(bool(text), "Missing coarse-action instruction")
    return "A video recorded from a robot's point of view executing the following instruction: " + text


class RoboCasaNormalizer:
    """Train-only min/max scaling; rot6d passes through; unused dimensions stay zero."""
    contract = CONTRACT

    def __init__(self, stats):
        require(stats["contract"] == self.contract and stats["mode"] == "minmax", "Wrong normalization contract")
        self.transforms = {}
        for kind, valid in zip(("state", "action"), masks()):
            low, high = (torch.tensor(stats[kind][key], dtype=torch.float32) for key in ("min", "max"))
            require(low.shape == high.shape == (80,) and bool(torch.isfinite(low).all())
                    and bool(torch.isfinite(high).all()) and bool((high >= low).all()), "Invalid normalization bounds")
            width = high-low
            scale = torch.where(width >= 1e-4, 2/width.clamp_min(1e-4), 1.)
            offset = torch.where(width >= 1e-4, -1-scale*low, -low)
            if kind == "state":
                for a in (10, 50):
                    scale[a:a+6], offset[a:a+6] = 1., 0.
            self.transforms[kind] = scale, offset, valid

    def forward(self, value, kind):
        scale, offset, valid = self.transforms[kind]
        return (value * scale.to(value) + offset.to(value)).masked_fill(~valid.to(value.device), 0)

    def backward(self, value, kind):
        scale, offset, valid = self.transforms[kind]
        return ((value - offset.to(value)) / scale.to(value)).masked_fill(~valid.to(value.device), 0)
