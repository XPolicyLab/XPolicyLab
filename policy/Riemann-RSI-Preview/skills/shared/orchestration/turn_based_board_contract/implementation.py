from copy import deepcopy

import numpy as np


SEQUENCE = tuple((turn, stage) for turn in range(5)
                 for stage in ('place', 'home', 'wait') if not (turn == 4 and stage == 'wait'))


def validate_wait_observation(value):
    fields = {'visible_cells', 'expected_count', 'changed_pixels', 'board_region_pixels', 'stable_frames',
              'action_count', 'budget_exhausted', 'history', 'official_success_claimed'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Opponent wait feedback contains only declared public image observations and own action counts')
    visible = value['visible_cells']
    if visible is not None and (not isinstance(visible, list) or len(visible) != 9 or any(type(item) is not bool for item in visible)):
        raise ValueError('Observed cell visibility must contain nine public image booleans or remain unknown')
    if type(value['expected_count']) is not int or value['expected_count'] not in (2, 4, 6, 8):
        raise ValueError('The next expected visibility count comes only from the own turn sequence')
    for name, maximum in (('board_region_pixels', 307200), ('stable_frames', 250), ('action_count', 250)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError('Opponent feedback uses bounded integer public image and action counters')
    if value['board_region_pixels'] < 1 or type(value['budget_exhausted']) is not bool or value['official_success_claimed'] is not False:
        raise ValueError('A nonempty observed region cannot claim native task success')
    changed = value['changed_pixels']
    if changed is not None and (type(changed) is not int or not 0 <= changed <= value['board_region_pixels']):
        raise ValueError('Changed pixels are a measured count inside the declared board region or unknown')
    history = value['history']
    if not isinstance(history, list) or len(history) > 10:
        raise ValueError('Opponent diagnostics retain at most ten public samples')
    previous = -1
    for sample in history:
        if not isinstance(sample, dict) or set(sample) != {'action_count', 'visible_count', 'changed_pixels', 'stable_frames'}:
            raise ValueError('A wait sample cannot carry hidden poses, contact, scorer or scene identifiers')
        if type(sample['action_count']) is not int or not previous < sample['action_count'] <= value['action_count']:
            raise ValueError('Wait samples use increasing own action counters')
        previous = sample['action_count']
        if type(sample['stable_frames']) is not int or not 0 <= sample['stable_frames'] <= 250:
            raise ValueError('A wait sample has a finite stability count')
        for name, maximum in (('visible_count', 9), ('changed_pixels', value['board_region_pixels'])):
            if sample[name] is not None and (type(sample[name]) is not int or not 0 <= sample[name] <= maximum):
                raise ValueError('A sample is a bounded current visibility or image change observation')
    return value


def minimum(stage, turn):
    if (turn, stage) not in SEQUENCE:
        raise ValueError('A board stage must preserve the mature turn and wait order')
    return {'place': 24 if turn == 0 else 19, 'home': 2, 'wait': 12}[stage]


def validate_anchor(value):
    if not isinstance(value, dict) or set(value) != {'cells', 'cell_spacing_m', 'source_homes', 'source_top_plane_m'}:
        raise ValueError('A board anchor contains only initially observed public grid and token surfaces')
    for name, count in (('cells', 9), ('source_homes', 5)):
        coordinates = np.asarray(value[name], dtype=float)
        if coordinates.shape != (count, 3) or not np.isfinite(coordinates).all() or np.max(np.abs(coordinates)) > 3:
            raise ValueError('Public board and source anchors must be bounded finite measured points')
    spacing, height = value['cell_spacing_m'], value['source_top_plane_m']
    if (type(spacing) not in (int, float) or not .02 <= spacing <= .25
            or type(height) not in (int, float) or not np.isfinite(height)):
        raise ValueError('Board spacing and source surfaces are public metric hypotheses')
    return deepcopy(value)


def validate_progress(value):
    if not isinstance(value, dict) or set(value) != {'next_stage', 'own_commanded_cells', 'used_source_slots'}:
        raise ValueError('Board progress contains only bounded own completed command history')
    position = value['next_stage']
    if type(position) is not int or not 0 <= position <= len(SEQUENCE):
        raise ValueError('The own turn cursor must remain within the declared finite sequence')
    count = sum(stage == 'place' for turn, stage in SEQUENCE[:position])
    for name, limit in (('own_commanded_cells', 9), ('used_source_slots', 5)):
        entries = value[name]
        if (not isinstance(entries, list) or len(entries) != count or len(set(entries)) != count
                or any(type(identity) is not int or not 0 <= identity < limit for identity in entries)):
            raise ValueError('Own command history cannot contain unknown or repeated token decisions')
    return deepcopy(value)


def advance(progress, turn, stage, cell, source):
    value = validate_progress(progress)
    if value['next_stage'] >= len(SEQUENCE) or SEQUENCE[value['next_stage']] != (turn, stage):
        raise ValueError('Advance only the next actually issued mature board stage')
    if stage == 'place':
        if (type(cell) is not int or not 0 <= cell < 9 or cell in value['own_commanded_cells']
                or type(source) is not int or not 0 <= source < 5 or source in value['used_source_slots']):
            raise ValueError('A placement must use a new publicly selected cell and source slot')
        value['own_commanded_cells'].append(cell)
        value['used_source_slots'].append(source)
    elif cell != value['own_commanded_cells'][-1] or source != value['used_source_slots'][-1]:
        raise ValueError('Home and opponent waiting must follow the same issued placement')
    value['next_stage'] += 1
    return validate_progress(value)


def empty_progress():
    return {'next_stage': 0, 'own_commanded_cells': [], 'used_source_slots': []}
