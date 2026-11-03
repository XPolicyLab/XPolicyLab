"""Termination checks use public task requirements and observed robot state only."""
from .protocol import obj

END_SCHEMA = obj({
    'outcome': {'enum': ['abandoned']},
    'basis': {'enum': ['irrecoverable_state'],
              'description': 'Low remaining budget is never a valid basis; pursue task completion.'},
    'reason': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
    'recovery_attempts': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 20},
})

class TerminationRejected(ValueError):
    """A semantic rejection is feedback, not a trajectory-format failure."""


def validate_termination(value, observation):
    if not isinstance(value, dict) or set(value) != {'outcome','basis','reason','recovery_attempts'}:
        raise ValueError('end_attempt requires outcome, basis, reason and recovery_attempts')
    if value['outcome'] != 'abandoned':
        raise TerminationRejected('end_attempt only abandons an irrecoverable state. Judge apparent completion from task requirements and current observations; use a trajectory for remaining motion or settling.')
    if value['basis'] != 'irrecoverable_state':
        raise TerminationRejected('Budget shortage is not a stopping basis. Continue achievable task actions; abandon only an irrecoverable state with no feasible progress or recovery.')
    if not isinstance(value['reason'], str) or not value['reason'].strip() or len(value['reason']) > 1000:
        raise ValueError('reason must be nonempty evidence of at most 1000 characters')
    recovery = value['recovery_attempts']
    if not isinstance(recovery, list) or len(recovery) > 20 or any(not isinstance(x, str) for x in recovery):
        raise ValueError('recovery_attempts must be a list of at most 20 strings')
    if value['outcome'] == 'abandoned' and not any(x.strip() for x in recovery):
        raise TerminationRejected('Describe the recoveries considered or tried and why no task-progress action remains. A low budget, failed approach or occlusion alone does not justify abandonment.')
    return value
