"""Public-base radial tool heading along an upright bottle transport path.

Avoid a large in-place yaw at the source. This is a geometric candidate,
not a collision-free or attachment guarantee. Endpoints come from RGB and
the public fixed bin, never a layout, reward or runtime object pose.
"""
import numpy as np

LEFT_BASE_XY=np.array([-.3,-.45])

def radial_transport(source,drop,max_step=.04,*,side='left'):
    if side not in ('left','right'):raise ValueError('Unknown transport arm')
    base=LEFT_BASE_XY if side=='left' else np.array([.3,-.45])
    source=np.asarray(source,float);drop=np.asarray(drop,float)
    if source.shape!=(2,) or drop.shape!=(2,) or not np.isfinite(np.r_[source,drop]).all():
        raise ValueError('Finite planar source and drop required')
    if not .01<=max_step<=.05:raise ValueError('Unsupported path step')
    count=max(1,int(np.ceil(np.linalg.norm(drop-source)/max_step)))
    points=np.linspace(source,drop,count+1)
    relative=points-base
    if np.min(np.linalg.norm(relative,axis=1))<.08:
        raise ValueError('Radial heading undefined near the public base')
    angles=np.unwrap(np.arctan2(relative[:,1],relative[:,0]))
    if np.max(np.abs(np.diff(angles)))>.25:
        raise ValueError('Heading changes too sharply for this path')
    return [(xy.copy(),float(yaw)) for xy,yaw in zip(points[1:],angles[1:])]
