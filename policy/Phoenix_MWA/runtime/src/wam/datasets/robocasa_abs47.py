"""Legacy RoboCasa absolute joint29 + absolute wrist18 representation."""
import torch

from .robocasa_delta import RAW, require, validate_modality, observation_state, prompt_for

DIM = 47
FIELDS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 20),
          "right_hand": (20, 26), "waist": (26, 29), "wrist_l_pos": (29, 32),
          "wrist_l_rot6d": (32, 38), "wrist_r_pos": (38, 41), "wrist_r_rot6d": (41, 47)}
CONTRACT = {
    "version": "robocasa_gr1_absolute47_zscore_remarks_v1",
    "world_axes": "source_world_A_identity", "ee_axes": "source_wrist_B_identity",
    "translation": "absolute_source_action_xyz", "rotation": "absolute_source_action_rot6d",
    "rotation6d": "first_two_columns_concatenated", "source_action_offset": 0,
    "joint_hand_waist": "absolute_source_action", "controller": "absolute_joint29",
    "embodiment_id": 18, "normalization": "z-score", "prompt_source": "episodes.jsonl:remarks",
}


def masks():
    return torch.ones(DIM, dtype=torch.bool), torch.ones(DIM, dtype=torch.bool)


def project(state, action=None):
    def select(raw):
        raw = torch.as_tensor(raw, dtype=torch.float32)
        require(raw.ndim == 2 and raw.shape[-1] == 161, "Expected raw rows [N,161]")
        value = torch.cat([raw[:, a:b] for a, b in (RAW[name] for name in FIELDS)], dim=-1)
        require(bool(torch.isfinite(value).all()), "Nonfinite absolute47 source")
        return value
    out = select(state)
    act = None if action is None else select(action)
    require(act is None or act.shape == out.shape, "Action/state rows must align")
    return out, act


def decode(action, measured_state=None):
    action = torch.as_tensor(action, dtype=torch.float32)
    require(action.shape[-1] == DIM and bool(torch.isfinite(action).all()), "Invalid absolute47 action")
    # The legacy simulator consumes only the first 29 absolute joint targets.
    return {"action." + name: action[..., a:b].clone() for name, (a, b) in FIELDS.items() if b <= 29}


class RoboCasaNormalizer:
    def __init__(self, stats):
        require(stats["contract"] == CONTRACT and stats["mode"] == "z-score", "Wrong absolute47 normalization")
        self.transforms = {}
        for kind in ("state", "action"):
            mean, std = (torch.tensor(stats[kind][key], dtype=torch.float32) for key in ("mean", "std"))
            require(mean.shape == std.shape == (DIM,) and bool(torch.isfinite(mean).all())
                    and bool(torch.isfinite(std).all()) and bool((std >= 0).all()), "Invalid z-score statistics")
            # Match the old SingleFieldLinearNormalizer, including rot6d fields.
            scale = 1 / (std + 1e-8)
            self.transforms[kind] = scale, -mean * scale

    def forward(self, value, kind):
        scale, offset = self.transforms[kind]
        return value * scale.to(value) + offset.to(value)

    def backward(self, value, kind):
        scale, offset = self.transforms[kind]
        return (value - offset.to(value)) / scale.to(value)
