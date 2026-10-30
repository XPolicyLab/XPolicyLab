"""Bounded stroke-stall evidence from own motion and the existing head binding.

This is a near-target motion plateau, NOT observed contact, hit or attachment.
No evaluator state, scene identity, images or diagnostic files are consumed.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import pose_matrix

HEAD_RADIUS = .013  # public mallet sphere radius; caps vertical accommodation
STALL_SAMPLES = 8
STALL_SPAN = .001
XY_TOLERANCE = .003
MIN_DESCENT = .025
RETRY_LIFT = HEAD_RADIUS
MAX_RETRIES_PER_KEY = 1
MAX_RETRIES_TOTAL = 2


class HitStall:
    def __init__(self):
        self.phase = None
        self.history = deque(maxlen=STALL_SAMPLES)
        self.top = {}

    def clear_window(self):
        self.phase = None
        self.history.clear()

    def observe(self, name, current, goal, anchor):
        if not (name.startswith('xy_key_') and name.endswith('_hit')):
            self.clear_window()
            return {'qualified': False, 'reason': 'not_hit'}
        c,g=pose_matrix(current),pose_matrix(goal)
        anchor=np.asarray(anchor,float)
        if anchor.shape!=(3,) or not np.isfinite(anchor).all():
            self.clear_window()
            return {'qualified': False, 'reason': 'invalid_binding'}
        p=c[:3,:3]@anchor+c[:3,3]
        target=g[:3,:3]@anchor+g[:3,3]
        angle=float(Rotation.from_matrix(c[:3,:3].T@g[:3,:3]).magnitude())
        if name!=self.phase:
            self.history.clear();self.phase=name
        self.top[name]=max(self.top.get(name,float(p[2])),float(p[2]))
        self.history.append((p.copy(),angle))
        points=np.array([v[0] for v in self.history])
        deltas=points-target
        span=float(np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2)))
        descended=float(self.top[name]-p[2])
        qualified=(len(points)==STALL_SAMPLES and span<STALL_SPAN
            and descended>=MIN_DESCENT
            and np.all(np.linalg.norm(deltas[:,:2],axis=1)<XY_TOLERANCE)
            and np.all((deltas[:,2]>=0)&(deltas[:,2]<=HEAD_RADIUS))
            and all(v[1]<.035 for v in self.history))
        return {'qualified':bool(qualified),'samples':len(points),
                'span_m':span,'descent_m':descended,
                'head_error_m':(p-target).tolist(),
                'contact_verified':False,'task_success':False}
