"""Validate public drive commands without replacing targets with joint positions."""
import hashlib
from pathlib import Path
import numpy as np


def load_command_ranges(config_path, expected_sha256, parameters):
    """Read only an authenticated static public robot configuration."""
    import yaml
    raw = Path(config_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Pinned public robot command configuration changed')
    config = yaml.safe_load(raw)
    scale = np.asarray(config['gripper_scale'], dtype=float)
    move = config['gripper_move']
    if (scale.shape != (2,) or not np.isfinite(scale).all() or scale[0] >= scale[1]
            or scale.tolist() != parameters['gripper_scale']
            or move['sign'] != parameters['sign'] or move['mimic'] != parameters['mimic']):
        raise ValueError('Public command declaration differs from the pinned robot configuration')
    multiplier, offset = np.asarray(move['mimic'][1:], dtype=float)
    if not np.isfinite([multiplier, offset]).all() or multiplier == 0:
        raise ValueError('Finite nonzero public mimic required')
    return np.asarray([scale, np.sort(scale * multiplier + offset)], dtype=float)


def validate_command_pair(robot, pair, ranges):
    pair = np.asarray(pair, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    if (ranges.shape != (2, 2) or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] >= ranges[:, 1])):
        raise ValueError('Two finite declared jaw command ranges required')
    endpoints = np.stack([robot.command_pair(0.), robot.command_pair(1.)])
    expected_ranges = np.stack([endpoints.min(axis=0), endpoints.max(axis=0)], axis=1)
    if not np.allclose(expected_ranges, ranges, rtol=0., atol=1e-8):
        raise ValueError('Robot command geometry differs from the public declaration')
    if (pair.shape != (2,) or not np.isfinite(pair).all()
            or np.any(pair < ranges[:, 0] - 1e-8) or np.any(pair > ranges[:, 1] + 1e-8)):
        raise ValueError('Finite jaw targets within the declared drive range required')
    low, high = robot.gripper_scale
    target = ((pair[0] - low) / (high - low) if robot.gripper_sign == 1
              else (high - pair[0]) / (high - low))
    if not np.allclose(robot.command_pair(target), pair, rtol=0., atol=1e-8):
        raise ValueError('Jaw command pair does not follow the public mimic')


def validated_normalized_target(robot, value, ranges):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Finite normalized jaw intent required')
    validate_command_pair(robot, robot.command_pair(value), ranges)
    return float(value)



def run(robot, normalized_value, *, config_path, config_sha256, parameters):
    """Validate one public command; issue no action and claim no measurement."""
    ranges = load_command_ranges(config_path, config_sha256, parameters)
    target = validated_normalized_target(robot, normalized_value, ranges)
    return {'normalized_target': target, 'drive_targets': robot.command_pair(target).tolist(),
            'actual_aperture': None, 'contact_evidence': 'unknown', 'action_issued': False}
