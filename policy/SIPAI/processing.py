"""RGB, state-token and joint-action transforms for the released checkpoint."""

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F

CAMERAS = (
    ("cam_head", "cam_high", "head_camera", "top_camera"),
    ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
)


# Fixed q01/q99 statistics for the released checkpoint.
STATE_Q01 = np.array(
    [
        -1.0330906894564629,
        -0.01492474360167978,
        -0.013325597870349884,
        -1.5667833752393725,
        -0.5573100898504257,
        -1.66245547246933,
        9.999999999996789e-05,
        -0.43669555014371875,
        -0.011439256957173322,
        -0.007908393166959293,
        -1.6213772659301755,
        -1.248548055434227,
        -1.934650058197975,
        9.999999999996789e-05,
    ],
    dtype=np.float64,
)
STATE_Q99 = np.array(
    [
        0.5420287713170051,
        2.4444837540298705,
        2.45528713234663,
        1.3101755834817883,
        1.2394421816110608,
        1.7071972546577454,
        0.9999,
        1.0593844474196432,
        2.374205224385858,
        2.310035456903279,
        1.175066296386719,
        0.5251381969690323,
        1.4287132223844528,
        0.9999,
    ],
    dtype=np.float64,
)
ACTION_Q01 = np.array(
    [
        -0.9634275891125204,
        -2.0369066297271403,
        -1.7851269753440282,
        -1.301504388523102,
        -0.7509960036337375,
        -1.4765365456938742,
        0.0001,
        -0.8602980139613151,
        -1.9550167570242192,
        -1.6660470775555876,
        -1.3345180193066597,
        -0.8675045107504324,
        -1.598589936375618,
        0.0001,
    ],
    dtype=np.float64,
)
ACTION_Q99 = np.array(
    [
        0.972656575208902,
        1.8347674648755343,
        1.5290213398658672,
        1.38575717458725,
        0.781793492025137,
        1.5668672624945645,
        0.9999,
        0.9190611119389536,
        1.7839940264579375,
        1.4433837208584244,
        1.3045809831261632,
        0.7041258340871048,
        1.4326152338743205,
        0.9999,
    ],
    dtype=np.float64,
)


def camera_image(obs, names):
    for name in names:
        value = obs["vision"].get(name)
        if isinstance(value, Mapping):
            value = next(
                (value[k] for k in ("color", "rgb", "colors", "image") if k in value),
                None,
            )
        if value is None:
            continue
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError("Expected a decoded RGB image")
        if image.shape[-1] not in (1, 3) and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError("Expected a finite RGB image")
        if np.issubdtype(image.dtype, np.floating):
            upper = 255.0 if image.max() > 1.0 else 1.0
            image = np.clip(image, 0.0, upper) * (255.0 / upper)
        return image.astype(np.uint8, copy=False)
    raise KeyError(f"Missing camera: {names}")


def image_tensor(image):
    image = torch.from_numpy(np.array(image, copy=True, order="C")).float() / 255.0 * 2.0 - 1.0
    height, width = image.shape[:2]
    if (height, width) != (224, 224):
        ratio = max(width / 224, height / 224)
        h, w = int(height / ratio), int(width / ratio)
        image = F.interpolate(
            image.permute(2, 0, 1)[None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        pad_h, extra_h = divmod(224 - h, 2)
        pad_w, extra_w = divmod(224 - w, 2)
        image = F.pad(
            image.clamp(-1, 1),
            (pad_w, pad_w + extra_w, pad_h, pad_h + extra_h),
            value=-1,
        )
        image = image[0].permute(1, 2, 0)
    return image


def state_values(obs):
    state, parts = obs["state"], []
    for side in ("left", "right"):
        joints = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
        if joints.shape != (6,) or gripper.size < 1 or not np.isfinite(joints).all() or not np.isfinite(gripper[0]):
            raise ValueError("Expected six finite joints and a gripper per arm")
        parts.extend([joints, np.clip(gripper[:1], 0, 1)])
    return np.concatenate(parts)


def prepare(obs, history, tokenizer, token_length, device):
    state = state_values(obs)
    q01, q99 = STATE_Q01, STATE_Q99
    normalized = (2.0 * (state.astype(np.float64) - q01) / (q99 - q01) - 1.0).astype(np.float32)
    bins = np.digitize(normalized, np.linspace(-1, 1, 257, dtype=np.float64)[:-1]) - 1
    instruction = obs.get("instruction", obs.get("instructions", obs.get("prompt", "")))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    instruction = str(instruction or "").strip().replace("_", " ").replace("\n", " ")
    prompt = f"Task: {instruction}, State: {' '.join(str(int(v)) for v in bins)};\nAction: "
    tokens = tokenizer.encode(prompt, add_bos=True)[:token_length]
    valid = len(tokens)
    return {
        "images": [image_tensor(camera_image(obs, names))[None].to(device) for names in CAMERAS],
        "history": torch.stack(
            [image_tensor(x) if x is not None else torch.full((224, 224, 3), -1.0) for x in history]
        )[None].to(device),
        "history_mask": torch.tensor([[x is not None for x in history]], dtype=torch.bool, device=device),
        "token_ids": torch.tensor([tokens + [0] * (token_length - valid)], dtype=torch.int64, device=device),
        "token_mask": torch.tensor(
            [[True] * valid + [False] * (token_length - valid)],
            dtype=torch.bool,
            device=device,
        ),
    }, state


def decode_actions(normalized, state):
    q01, q99 = ACTION_Q01, ACTION_Q99
    actions = (np.asarray(normalized[..., :14], dtype=np.float64) + 1.0) * (q99 - q01) / 2.0 + q01
    # Preserve the checkpoint contract's FP32 semantic output before composition.
    actions = actions.astype(np.float32).astype(np.float64)
    actions[..., :6] += state[:6]
    actions[..., 7:13] += state[7:13]
    actions[..., [6, 13]] = np.clip(actions[..., [6, 13]], 0.0, 1.0)
    return actions.astype(np.float32)


def action_dict(row, robot_action_dim_info):
    action, offset = {}, 0
    for side, arm_dim, ee_dim in zip(
        ("left", "right"),
        robot_action_dim_info["arm_dim"],
        robot_action_dim_info["ee_dim"],
        strict=True,
    ):
        for name, size in (("arm_joint_state", arm_dim), ("ee_joint_state", ee_dim)):
            action[f"{side}_{name}"] = row[offset : offset + size].copy()
            offset += size
    if offset != len(row):
        raise ValueError("Action size does not match the shared robot dimensions")
    return action
