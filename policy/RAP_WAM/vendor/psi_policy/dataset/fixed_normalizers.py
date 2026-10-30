from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from psi_policy.common.normalize_util import get_image_range_normalizer
from psi_policy.common.normalize_util import get_normalizer_from_input_min_max
from psi_policy.model.common.normalizer import LinearNormalizer


@dataclass(frozen=True)
class _FixedFieldLimits:
    lower: np.ndarray
    upper: np.ndarray


_DUAL_ARM_WAN_LIMITS = {
    "waist_joints_states": _FixedFieldLimits(
        lower=np.array([0.0, 0.0], dtype=np.float32),
        upper=np.array([0.5, 1.0471976], dtype=np.float32),
    ),
    "left_arm_states": _FixedFieldLimits(
        lower=np.array(
            [-3.1399998, -1.9218, -3.1399998, -1.57, -3.1399998, -1.48, -3.1299999],
            dtype=np.float32,
        ),
        upper=np.array(
            [3.1399998, 1.8151, 3.1399998, 1.22, 3.1399998, 1.48, 3.1299999],
            dtype=np.float32,
        ),
    ),
    "right_arm_states": _FixedFieldLimits(
        lower=np.array(
            [-3.1399998, -1.9218, -3.1399998, -1.22, -3.1399998, -1.48, -3.1299999],
            dtype=np.float32,
        ),
        upper=np.array(
            [3.1399998, 1.8151, 3.1399998, 1.57, 3.1399998, 1.48, 3.1299999],
            dtype=np.float32,
        ),
    ),
    "left_hand_states": _FixedFieldLimits(
        lower=np.zeros(6, dtype=np.float32),
        upper=np.ones(6, dtype=np.float32),
    ),
    "right_hand_states": _FixedFieldLimits(
        lower=np.zeros(6, dtype=np.float32),
        upper=np.ones(6, dtype=np.float32),
    ),
}

_DUAL_ARM_WAN_ACTION_LIMITS = {
    "action_waist_joints": _FixedFieldLimits(
        lower=_DUAL_ARM_WAN_LIMITS["waist_joints_states"].lower,
        upper=_DUAL_ARM_WAN_LIMITS["waist_joints_states"].upper,
    ),
    "action_arm_joints": _FixedFieldLimits(
        lower=np.concatenate(
            [
                _DUAL_ARM_WAN_LIMITS["left_arm_states"].lower,
                _DUAL_ARM_WAN_LIMITS["right_arm_states"].lower,
            ],
            axis=0,
        ),
        upper=np.concatenate(
            [
                _DUAL_ARM_WAN_LIMITS["left_arm_states"].upper,
                _DUAL_ARM_WAN_LIMITS["right_arm_states"].upper,
            ],
            axis=0,
        ),
    ),
    "action_left_hand_joints": _FixedFieldLimits(
        lower=np.zeros(6, dtype=np.float32),
        upper=np.ones(6, dtype=np.float32),
    ),
    "action_right_hand_joints": _FixedFieldLimits(
        lower=np.zeros(6, dtype=np.float32),
        upper=np.ones(6, dtype=np.float32),
    ),
}

_FIXED_NORMALIZER_PRESETS = {
    "wan_dual_arm_joint_limits_v1": {
        "state_limits": _DUAL_ARM_WAN_LIMITS,
        "action_limits": _DUAL_ARM_WAN_ACTION_LIMITS,
    }
}


def build_fixed_wan_normalizer(
    *,
    fixed_preset: str,
    state_keys: Sequence[str],
    action_keys: Sequence[str],
    include_action: bool,
    output_state_key: str,
    obs_keys: Sequence[str],
    image_obs_keys: Sequence[str],
) -> LinearNormalizer:
    preset_key = str(fixed_preset).strip()
    if preset_key not in _FIXED_NORMALIZER_PRESETS:
        raise ValueError(
            f"Unknown Wan fixed normalizer preset: {fixed_preset}. "
            f"Available presets: {', '.join(sorted(_FIXED_NORMALIZER_PRESETS))}"
        )

    preset = _FIXED_NORMALIZER_PRESETS[preset_key]
    state_limits = preset["state_limits"]
    action_limits = preset["action_limits"]

    normalizer = LinearNormalizer()
    if state_keys:
        state_min = _concat_limits(
            field_limits=state_limits,
            keys=state_keys,
            attr="lower",
            kind="state",
            preset_key=preset_key,
        )
        state_max = _concat_limits(
            field_limits=state_limits,
            keys=state_keys,
            attr="upper",
            kind="state",
            preset_key=preset_key,
        )
        normalizer[output_state_key] = get_normalizer_from_input_min_max(
            input_min=state_min,
            input_max=state_max,
        )

    if include_action and action_keys:
        action_min = _concat_limits(
            field_limits=action_limits,
            keys=action_keys,
            attr="lower",
            kind="action",
            preset_key=preset_key,
        )
        action_max = _concat_limits(
            field_limits=action_limits,
            keys=action_keys,
            attr="upper",
            kind="action",
            preset_key=preset_key,
        )
        normalizer["action"] = get_normalizer_from_input_min_max(
            input_min=action_min,
            input_max=action_max,
        )

    for key in obs_keys:
        if key in image_obs_keys:
            normalizer[key] = get_image_range_normalizer()
    return normalizer


def _concat_limits(
    *,
    field_limits: dict[str, _FixedFieldLimits],
    keys: Sequence[str],
    attr: str,
    kind: str,
    preset_key: str,
) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in field_limits:
            raise KeyError(
                f"Fixed normalizer preset '{preset_key}' does not define {kind} key '{key}'"
            )
        parts.append(getattr(field_limits[key], attr))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
