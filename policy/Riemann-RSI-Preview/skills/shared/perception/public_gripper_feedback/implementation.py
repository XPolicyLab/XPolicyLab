"""Evidence from standard public gripper feedback, without private getters."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math


def _scalar(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError('Public gripper feedback requires one normalized scalar')
    value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Public gripper feedback must be numeric')
    value = float(value)
    if not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError('Public gripper feedback must be finite and in [0,1]')
    return value


def _side(side):
    if side.endswith('_arm'):
        side = side[:-4]
    if side not in ('left', 'right'):
        raise ValueError('A public feedback side must be left or right')
    return side


@dataclass(frozen=True)
class GripperEvidence:
    side: str
    normalized_applied_target: float
    requested_target: float | None
    possible_closing_obstruction: bool | None
    contact: str = 'unknown'
    physical_aperture_m: None = None
    finger_joint_positions: None = None
    source: str = 'official_public_ee_joint_state'
    sdk_normalized_target_limit: float = .2

    def as_dict(self):
        return asdict(self)


def observe(observation, side, *, requested_target=None):
    side = _side(side)
    value = _scalar(observation['state'][side + '_ee_joint_state'])
    requested = None if requested_target is None else _scalar([requested_target])
    obstruction = (value > 1e-6) if requested == 0. else None
    return GripperEvidence(side, value, requested, obstruction)


def from_history(recorder, side, *, requested_target=None):
    """Use the already recorded current public sample; never fetch a new frame."""
    side = _side(side)
    values = getattr(recorder, 'state_' + side + '_gripper')
    if not len(values):
        raise ValueError('No current public gripper observation is available')
    return observe({'state': {side + '_ee_joint_state': values[-1]}}, side,
                   requested_target=requested_target)


def command_joint_target(normalized, *, lower=-.01, upper=.044, sign=1):
    """Command-space conversion only; no measured width is reconstructed."""
    value = _scalar([normalized])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper or sign not in (-1, 1):
        raise ValueError('Invalid declared gripper command range')
    return lower + value * (upper-lower) if sign == 1 else upper - value * (upper-lower)


def geometry_joint_candidates(*, lower=-.01, upper=.044):
    """Static bounds for conservative swept occlusion; unrelated to feedback."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError('Invalid declared finger geometry range')
    return (0., max(abs(lower), abs(upper)))
