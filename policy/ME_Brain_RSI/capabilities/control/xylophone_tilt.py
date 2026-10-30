"""Bounded held-mallet tilt candidates with nominal table-clearance checks.

Independent geometry using RGB-bound head and the public nominal pinch point.
The handle axis inferred through that point is a grasp-model hypothesis, not
observed full tool pose. These checks are NOT full robot/world collision checks.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .xylophone_geometry import rigid_transform


def bounded_tilt_candidates(rotation, anchor, grasp_offset, tip_offset,
                            table_z, head_z=.803):
    frame=np.eye(4);frame[:3,:3]=rotation
    rigid_transform(frame)
    anchor=np.asarray(anchor,float)
    grasp=np.asarray(grasp_offset,float)
    tip=np.asarray(tip_offset,float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (anchor,grasp,tip)):
        raise ValueError("Finite local tool points required")
    if not np.isfinite([table_z,head_z]).all():
        raise ValueError("Finite public heights required")
    axis=grasp-anchor
    length=np.linalg.norm(axis)
    if not .03 <= length <= .19:
        return []
    axis/=length
    candidates=[]
    # Explicit bounded orientation alternatives, not a changed IK tolerance.
    # Try less tilt first; neither may bypass complete sampled local IK/budget.
    for degrees in (4.,8.):
        tilted=np.asarray(rotation)@Rotation.from_euler('y',degrees,degrees=True).as_matrix()
        # Public shaft: 200.6 mm from head, radius 3.75 mm. Conservatively
        # subtract full radius even though its vertical projection is smaller.
        end_z=head_z+.2006*(tilted@axis)[2]
        shaft_low=min(head_z,end_z)-.00375
        tip_z=head_z+(tilted@(tip-anchor))[2]
        clearance=min(shaft_low,tip_z)-table_z
        if clearance < .005:
            continue
        candidates.append(dict(rotation=tilted,degrees=degrees,
                               nominal_clearance_m=float(clearance),
                               clearance_source="public_handle_and_tip_with_nominal_pinch_axis",
                               collision_checked=False))
    return candidates
