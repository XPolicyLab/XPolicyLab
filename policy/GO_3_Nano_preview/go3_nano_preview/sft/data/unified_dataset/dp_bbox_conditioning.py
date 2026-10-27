import math
MODES = ('original', 'box_only', 'dp_only', 'box_dp')

def validate_probabilities(probabilities):
    if set(probabilities) != set(MODES):
        raise ValueError(f'conditioning probabilities require exactly {MODES}')
    values = {mode: float(probabilities[mode]) for mode in MODES}
    if any((not math.isfinite(v) or v < 0 for v in values.values())):
        raise ValueError('conditioning probabilities must be finite and nonnegative')
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-08):
        raise ValueError('conditioning probabilities must sum to one')
    return values

def select_mode(probabilities, rng):
    draw = rng.random()
    total = 0.0
    for mode in MODES:
        total += probabilities[mode]
        if draw < total:
            return mode
    return next((mode for mode in reversed(MODES) if probabilities[mode] > 0))

def resolve_dense_prompt(step, frame, field='dense_prompt'):
    dense = step.get(field)
    if isinstance(dense, list):
        candidates = []
        for segment in dense:
            if not isinstance(segment, dict) or not isinstance(segment.get('caption'), str):
                continue
            try:
                (start, end) = (int(segment['start_frame']), int(segment['end_frame']))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if start <= frame <= end and segment['caption'].strip():
                candidates.append((abs((start + end) / 2 - frame), start, segment['caption']))
        dense = min(candidates, key=lambda x: x[:2])[2] if candidates else None
    return dense if isinstance(dense, str) and dense.strip() else None

def realize_mode(mode, bbox_available, dense_available):
    if mode not in MODES:
        raise ValueError(f'unknown conditioning mode: {mode}')
    box = mode in ('box_only', 'box_dp') and bbox_available
    dp = mode in ('dp_only', 'box_dp') and dense_available
    return ('box_dp' if dp else 'box_only') if box else 'dp_only' if dp else 'original'
