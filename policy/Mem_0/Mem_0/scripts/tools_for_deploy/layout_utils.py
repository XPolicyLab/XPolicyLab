"""Layout conversion between env action/state and model expectations."""

import numpy as np


def env_to_model_layout(vec: np.ndarray) -> np.ndarray:
    """
    Convert env layout -> model layout.
    Env:   [LA0-6, LGrip, RA0-6, RGrip]
    Model: [LA0-6, RA0-6, LGrip, RGrip]
    """
    flat = np.array(vec, dtype=np.float32).reshape(-1)
    if flat.size < 16:
        padded = np.zeros(16, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    model = np.zeros(16, dtype=np.float32)
    model[0:7] = flat[0:7]          # left arm
    model[7:14] = flat[8:15]        # right arm
    model[14] = flat[7]             # left gripper
    model[15] = flat[15]            # right gripper
    return model


def model_to_env_layout(vec: np.ndarray) -> np.ndarray:
    """
    Convert model layout -> env layout.
    Model: [LA0-6, RA0-6, LGrip, RGrip]
    Env:   [LA0-6, LGrip, RA0-6, RGrip]
    """
    flat = np.array(vec, dtype=np.float32).reshape(-1)
    if flat.size < 16:
        padded = np.zeros(16, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    env = np.zeros(16, dtype=np.float32)
    env[0:7] = flat[0:7]            # left arm
    env[7] = flat[14]               # left gripper
    env[8:15] = flat[7:14]          # right arm
    env[15] = flat[15]              # right gripper
    return env

