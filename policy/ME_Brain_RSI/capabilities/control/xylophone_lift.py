"""X07 lift target and arrival confirmation using public geometry and own EE.

No evaluator state or diagnostic files are read. The unchanged X06 RGB binding
estimates the head at the mallet origin; roll remains unobserved, so use a
rotation-independent bound on public beat-point offsets.
"""
import numpy as np
from .geometry import pose_matrix
from .xylophone_geometry import BEAT_POINTS

# Same nominal hit frame as X06's .803 head height / .023 above-hit offset.
# Public static placement prior, not this scene's measured object height.
NOMINAL_HIT_FRAME_Z = .780
PUBLIC_BEAT_UPPER = .036
PUBLIC_REQUIRED_LIFT = .025
LIFT_MARGIN = .008
HEAD_ARRIVAL_TOLERANCE = .003
HEAD_HEIGHT_TOLERANCE = .002
HIGH_DWELL = 3


def lift_head_height(hit_frame_z=NOMINAL_HIT_FRAME_Z):
    if not np.isfinite(hit_frame_z):
        raise ValueError("Finite public hit-plane prior required")
    beat_radius = float(np.max(np.linalg.norm(BEAT_POINTS, axis=1)))
    # At the first acceptable beat, object origin can be as high as
    # hit_z + upper - rotated_beat_z. -rotated_beat_z <= beat_radius.
    return float(hit_frame_z + PUBLIC_BEAT_UPPER + beat_radius
                 + PUBLIC_REQUIRED_LIFT + LIFT_MARGIN)


def high_arrival(current, goal, anchor):
    """Modelled head from actual robot state, not commanded pose or truth.

    This verifies arrival, not that the tool is still attached. Attachment/RGB
    logic is intentionally unchanged from X06.
    """
    current = pose_matrix(current)
    goal = pose_matrix(goal)
    anchor = np.asarray(anchor, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Finite existing bound head offset required")
    actual = current[:3, :3] @ anchor + current[:3, 3]
    target = goal[:3, :3] @ anchor + goal[:3, 3]
    error = float(np.linalg.norm(actual-target))
    height_shortfall = float(target[2]-actual[2])
    return {"accepted": bool(error < HEAD_ARRIVAL_TOLERANCE
                              and height_shortfall <= HEAD_HEIGHT_TOLERANCE),
            "actual_state_modelled_head": actual.tolist(),
            "target_head": target.tolist(),
            "head_error_m": error,
            "height_shortfall_m": height_shortfall,
            "attachment_reverified": False}
