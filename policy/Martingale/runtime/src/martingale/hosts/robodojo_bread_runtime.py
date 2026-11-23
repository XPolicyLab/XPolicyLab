# Copyright © 2026 Martingale. All Rights Reserved.
import re

from . import robodojo_sight as sight
from .robodojo_bread import fit_bread
from .robodojo_bread_selection import choose


def applies(prompt):
    return bool(re.search(r'\b(?:bread|toast)\b', prompt)
                and not re.search(r'\b(?:shelf|rack|toaster)\b', prompt))


def refine(capture, estimate, prompt, camera):
    _, candidate, info = fit_bread(sight, capture, estimate or {})
    selected = choose(info, sight.SECOND_VIEW_MIN)
    info['original_selected'] = info['selected']
    info['selected'] = selected
    result = capture['result']
    if selected == 'bread_slab':
        result = dict(candidate['result'])
        marks = info['slab_per_view_scores'][1:]
        result['match'] = sum(marks) / len(marks)
    result['geometry_fit'] = info
    return result
