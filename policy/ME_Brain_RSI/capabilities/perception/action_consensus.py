"""Choose a visual action valid in every solvable arithmetic interpretation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def action_key(solution: Mapping[str, Any], slot: int) -> tuple:
    item = solution['selected']
    return slot, solution['selected_piece'], item['label'], item['height'], item['yaw_deg']


def common_action(solutions: Sequence[Mapping[str, Any]],
                  pieces: Sequence[Mapping[str, Any]], slot: int) -> dict | None:
    """Do not choose a convenient reading of an ambiguous glyph or equation.

    Inputs are the existing RGB reader's supported hypotheses. This does not
    introduce a new perceptual threshold or claim unknown readings are ruled out.
    Distinct loose instances with the same digit may implement the same task.
    """
    worlds: dict[tuple, set[tuple]] = {}
    costs: dict[tuple, float] = {}
    for solution in solutions:
        labels = solution['labels']
        if not 0 <= slot < len(labels):
            raise ValueError('Missing slot outside the equation')
        world = tuple(labels[:slot]) + tuple(labels[slot + 1:])
        actions = worlds.setdefault(world, set())
        selected = solution['selected']
        piece = pieces[solution['selected_piece']]
        if {item['label'] for item in piece['supported']} != {selected['label']}:
            continue
        cost = float(selected['cost'])
        if not math.isfinite(cost):
            raise ValueError('Non-finite visual cost')
        key = action_key(solution, slot)
        actions.add(key)
        costs[key] = min(costs.get(key, math.inf), cost)
    if not worlds:
        return None
    common = set.intersection(*worlds.values())
    if not common:
        return None
    chosen = min(common, key=lambda key: (costs[key], key))
    return {'solvable_visual_worlds': len(worlds),
            'common_actions': [list(key) for key in sorted(common)],
            'chosen': list(chosen),
            'rule': 'Same visually unambiguous piece valid in every solvable visual interpretation'}
