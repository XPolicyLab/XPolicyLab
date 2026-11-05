"""B19: confirm a visible original patch by repeated measured probe motion.

Independent implementation. Uses original pre-grasp RGB texture, own TCP and
opening only. No scene identifiers, object truth, rewards or fitted coefficients.
The fallback is for an unavailable stationary reference, not contradictory
stationary evidence. Two post-probe observations are mandatory.
"""
import numpy as np


def occluded_source_motion(history, probe):
    if probe is None or len(history) < 4:
        return False
    pre = [h for h in history if h['step'] == probe['step']]
    post = [h for h in history if h['step'] > probe['step']]
    if len(pre) != 1 or len(post) < 2:
        return False
    before = pre[0]
    after = post[-2:]
    observations = [before] + after
    if len({h['step'] for h in observations}) != 3:
        return False
    for h in observations:
        if not h.get('robot_ok') or not np.isfinite(h.get('opening', np.nan)):
            return False
        if not .1 < h['opening'] < .9 or abs(h['opening'] - probe['opening']) > .10:
            return False
    for h in after:
        delta = np.asarray(h['tcp']) - probe['tcp']
        if not np.isfinite(delta).all() or not (.008 <= delta[2] <= .020):
            return False
        if np.linalg.norm(delta[:2]) > .006:
            return False
    for a in before['patches']:
        matches = [next((p for p in h['patches'] if p['id'] == a['id']), None) for h in after]
        if any(b is None for b in matches):
            continue
        patches = [a] + matches
        if not all(p['moved_valid'] and p['moved'] >= .80 for p in patches):
            continue
        # This route does not reinterpret missing stationary scores as evidence.
        # A valid strong stationary match is contrary evidence, never ignored.
        if any(p['stationary_valid'] and p['stationary'] >= p['moved'] - .05 for p in patches):
            continue
        if not all(not p['stationary_valid'] for p in patches):
            continue  # the existing contrasted path handles valid comparisons
        accepted = True
        for b in matches:
            motion = np.asarray(b['predicted_center']) - a['predicted_center']
            residual = np.asarray(b['offset']) - a['offset']
            measured = motion + residual
            if not np.isfinite(np.r_[motion, residual]).all():
                accepted = False
                break
            if not (np.linalg.norm(motion) >= 3 and np.linalg.norm(residual) <= 1.5
                    and measured @ motion > 0
                    and .65 <= np.linalg.norm(measured) / np.linalg.norm(motion) <= 1.35
                    and abs(a['angle'] - b['angle']) <= 12
                    and abs(a['scale'] - b['scale']) <= .12):
                accepted = False
                break
        if accepted:
            return True
    return False
