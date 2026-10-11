"""Observation/action-only release candidates; not a learned failure classifier.

Shadow output never authorizes takeover or asserts that an object is held.
The provisional thresholds require independent calibration before enforcement.
"""
import math
from .contracts import digest


def release_candidate(actions, state):
    if len(state) != 14 or len(actions) != 24 or any(len(a) != 14 for a in actions):
        raise ValueError('Expected native 24x14 actions and 14D state')
    if not all(math.isfinite(float(v)) for row in [state, *actions] for v in row):
        raise ValueError('Non-finite chunk')
    events = []
    for arm, col in [('left', 6), ('right', 13)]:
        for start in range(22):
            window = [float(actions[j][col]) for j in range(start, start + 3)]
            if all(v >= .8 and v - float(state[col]) >= .25 for v in window):
                events.append({'arm': arm, 'first_action_index': start,
                               'current_gripper': float(state[col]), 'targets': window})
                break
    return {'schema': 'rsi.chunk_gate.shadow.v1', 'mode': 'shadow',
            'proposal_sha256': digest(actions), 'state_sha256': digest(state),
            'candidate_release': bool(events), 'events': events,
            'release_authorized': None, 'takeover_authorized': False,
            'interpretation': 'Opening candidate only; may be valid handoff or empty-hand motion.'}
