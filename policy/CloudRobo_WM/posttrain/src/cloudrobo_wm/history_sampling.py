"""Native-step observation selection shared by data and evaluation."""
from numbers import Integral
UNIFORM = 'uniform_stride4_v1'
ANCHOR = 'episode_first_multiscale_v2'
SPLIT = 'episode_anchor_recent_split_v3'

def history_frame_indices(anchor, sampling=UNIFORM):
    if isinstance(anchor, bool) or not isinstance(anchor, Integral) or anchor < 0:
        raise ValueError('Anchor must be a nonnegative native control-step integer')
    anchor = int(anchor)
    if sampling == UNIFORM:
        return [max(0, anchor + d) for d in (-16, -12, -8, -4, 0)]
    if sampling == ANCHOR:
        return [0] + [max(0, anchor + d) for d in (-32, -16, -8, -4)]
    if sampling == SPLIT:
        return [0] + [max(0, anchor + d) for d in (-20, -16, -12, -8, -4)]
    raise ValueError('Unknown history sampling contract')

def selected_capture_step(step, replan_steps=32):
    if replan_steps != 32 or step < 0:
        raise ValueError('This evaluator uses the fixed original 32-action horizon')
    return step % 32 in (0, 12, 16, 20, 24, 28)
