"""Public mallet/key geometry and RGB-estimated rigid attachment transforms.

Independent implementation inspired by PhysicalRSI PourBallsRGB functional-point
anchoring/rebinding (reviewed code-skills v4); no upstream code or fitted
coefficients copied. Inputs are RGB estimates and observed robot poses only.
This module alone does not establish grasp success, contacts or task success.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .xylophone_keys import KEY_HITS

# Public Assets/Object/RoboDojo/{Rigid/mallet,Geometry/xylophone}/00000/metadata.json.
BEAT_POINTS = np.array([[0., -.001, -.010], [0., -.001, .010]])
KEY_HIT_Z = .014


def rigid_transform(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.all(np.isfinite(value))
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Expected a finite rigid transform")
    return value.copy()


def estimate_attachment(world_ee, world_mallet):
    """Mallet frame in EE coordinates, from an RGB estimate and own EE pose."""
    return np.linalg.inv(rigid_transform(world_ee)) @ rigid_transform(world_mallet)


def attachment_consensus(world_ees, rgb_mallets, translation_tolerance=.008,
                         rotation_tolerance=.10, minimum_motion=.015):
    """Two or more moved observations must agree on a rigid attachment.

    Return unknown for absent/insufficient evidence, inconsistent for disagreement,
    consistent for a geometrical hypothesis. None of these is official success.
    Thresholds are explicit initial engineering bounds, not fitted to outcomes.
    """
    if len(world_ees) != len(rgb_mallets):
        raise ValueError("Pose observation counts disagree")
    if not (translation_tolerance > 0 and rotation_tolerance > 0 and minimum_motion > 0):
        raise ValueError("Positive evidence bounds required")
    if len(world_ees) < 2 or any(p is None for p in rgb_mallets):
        return dict(state="unknown", reason="insufficient_visible_observations")
    ees = [rigid_transform(p) for p in world_ees]
    tools = [rigid_transform(p) for p in rgb_mallets]
    movement = max(np.linalg.norm(p[:3, 3] - ees[0][:3, 3]) for p in ees)
    if movement < minimum_motion:
        return dict(state="unknown", reason="insufficient_actual_motion")
    bindings = [estimate_attachment(ee, tool) for ee, tool in zip(ees, tools)]
    reference = bindings[0]
    position_error = max(np.linalg.norm(b[:3, 3] - reference[:3, 3]) for b in bindings)
    angle_error = max(Rotation.from_matrix(reference[:3, :3].T @ b[:3, :3]).magnitude()
                      for b in bindings)
    accepted = position_error <= translation_tolerance and angle_error <= rotation_tolerance
    return dict(state="consistent" if accepted else "inconsistent",
                reason="rgb_rigid_attachment" if accepted else "relative_pose_changed",
                translation_error_m=float(position_error), rotation_error_rad=float(angle_error),
                actual_ee_motion_m=float(movement),
                ee_mallet=bindings[-1] if accepted else None)


def ordered_key_targets(world_xylophone):
    """Eight public functional locations transformed by an RGB-estimated frame."""
    frame = rigid_transform(world_xylophone)
    local = np.c_[KEY_HITS, np.full(8, KEY_HIT_Z)]
    return local @ frame[:3, :3].T + frame[:3, 3]


def ee_for_beat(target, world_tool_rotation, ee_mallet, beat_index=0):
    """Solve EE pose for a desired hammer functional point, including rotation."""
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Finite world beat target required")
    if type(beat_index) is not int or beat_index not in (0, 1):
        raise ValueError("Unknown public beat point")
    binding = rigid_transform(ee_mallet)
    desired = np.eye(4)
    desired[:3, :3] = world_tool_rotation
    desired[:3, 3] = target - desired[:3, :3] @ BEAT_POINTS[beat_index]
    return rigid_transform(desired) @ np.linalg.inv(binding)


def strike_targets(world_xylophone, world_tool_rotation, ee_mallet,
                   beat_index=0, strike_above_hit=.018, lift=.035):
    """Nominal short-stroke targets. Caller must verify IK, clearance and motion.

    The stroke places the chosen beat point above public hit frames, and lifts
    the whole tool before any lateral transit. Constant orientation makes tool
    origin lift equal functional-point lift. These are plans, never reward checks.
    """
    if not np.isfinite(strike_above_hit) or not .010 < strike_above_hit < .036:
        raise ValueError("Strike height must be inside the public vertical band")
    if not np.isfinite(lift) or lift <= .025:
        raise ValueError("Lift needs margin above the public 25 mm requirement")
    keys = ordered_key_targets(world_xylophone)
    result = []
    for index, point in enumerate(keys):
        hit = point + [0, 0, strike_above_hit]
        high = hit + [0, 0, lift]
        for phase, target in (("hover", high), ("strike", hit), ("lift", high)):
            result.append(dict(key=index, phase=phase, beat_target=target.copy(),
                               ee=ee_for_beat(target, world_tool_rotation, ee_mallet, beat_index)))
    return result
