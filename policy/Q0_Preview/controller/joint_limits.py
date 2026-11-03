"""Legal joint representations; never clamp a target and silently change its pose."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from .executor import Replan

POSITION_EPS = 1e-6


class JointLimits:
    def __init__(self, names, kinds, bounds):
        self.names = tuple(names)
        self.kinds = tuple(kinds)
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if (len(self.kinds) != len(self.names) or self.bounds.shape != (len(self.names), 2)
                or np.isnan(self.bounds).any() or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('invalid named joint position limits')
        if any(k not in ('revolute', 'continuous', 'prismatic') for k in self.kinds):
            raise ValueError('unsupported active joint type')

    @classmethod
    def from_urdf(cls, names, runtime_bounds=None):
        joints = {j.get('name'): j for j in ET.parse(
            Path(__file__).with_name('calibration')/'X5A.urdf').getroot().findall('joint')}
        kinds, bounds = [], []
        for name in names:
            j = joints[name]
            kinds.append(j.get('type'))
            limit = j.find('limit')
            bounds.append((-math.inf, math.inf) if j.get('type') == 'continuous'
                          else (float(limit.get('lower')), float(limit.get('upper'))))
        # Optional explicit bounds support documented robot configurations; remote runtime uses the bundled URDF.
        return cls(names, kinds, bounds if runtime_bounds is None else runtime_bounds)

    def _vector(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise Replan('invalid joint target shape or non-finite value')
        return q

    def validate(self, q):
        q = self._vector(q)
        bad = np.flatnonzero((q < self.bounds[:, 0]-POSITION_EPS)
                             | (q > self.bounds[:, 1]+POSITION_EPS))
        if len(bad):
            i = int(bad[0])
            raise Replan(f'{self.names[i]} target outside configured joint limit; change the explicit pose')
        return q

    def nearest(self, q, seed):
        q, seed = self._vector(q).copy(), self._vector(seed)
        for i, kind in enumerate(self.kinds):
            if kind not in ('revolute', 'continuous'):
                continue
            lo, hi = self.bounds[i]
            turns = int(np.rint((seed[i]-q[i])/(2*math.pi)))
            minimum = math.ceil((lo-q[i]-POSITION_EPS)/(2*math.pi)) if math.isfinite(lo) else None
            maximum = math.floor((hi-q[i]+POSITION_EPS)/(2*math.pi)) if math.isfinite(hi) else None
            if minimum is not None and maximum is not None and minimum > maximum:
                raise Replan(f'no legal equivalent angle for {self.names[i]}')
            if minimum is not None: turns = max(turns, minimum)
            if maximum is not None: turns = min(turns, maximum)
            q[i] += 2*math.pi*turns
        return self.validate(q).copy()


def legal_equivalent(backend, arm, q, seed):
    """Unknown/test backends do not license wrapping every coordinate as an angle."""
    q = np.array(q, dtype=float, copy=True)
    if q.shape != np.shape(seed) or not np.isfinite(q).all():
        raise Replan('invalid IK solution')
    if hasattr(backend, 'nearest_joint_equivalent'):
        return backend.nearest_joint_equivalent(arm, q, seed)
    return q
