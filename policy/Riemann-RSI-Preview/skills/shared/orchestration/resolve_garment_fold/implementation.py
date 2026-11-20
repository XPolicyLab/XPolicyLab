from skills.shared.orchestration.instruction_semantics.implementation import resolver_contract
from functools import partial
import re
from skills.shared.orchestration.instruction_semantics.implementation import matches

from skills.shared.verification.observed_garment_state.implementation import ACTIONS, recorded


NEXT = {'view': 'garment_scene_view', 'approach': 'garment_approach_contact',
        'contact': 'garment_lower_contact', 'close': 'garment_close_grasp',
        'lift': 'garment_lift_fold', 'carry': 'garment_carry_fold',
        'lower': 'garment_settle_fold', 'release': 'garment_release_fold',
        'clear': 'garment_clear_wrist', 'recover': 'garment_recover_contact',
        'home': 'garment_return_home'}


@resolver_contract(partial(matches, actions=('fold',), objects=('garment',)))
def resolve(api, memory, maximum_uncertain_attempts=2):
    empty = {'applicable': matches(api.instruction(), actions=('fold',), objects=('garment',)), 'operations': [], 'relation': None,
             'reason': 'The current instruction or bounded garment evidence does not qualify another fold stage.'}
    if maximum_uncertain_attempts != 2 or not empty['applicable']:
        return empty
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] not in NEXT or current['attempts'] >= 3:
        return empty
    name = NEXT[current['phase']]
    failures = 0
    for event in reversed(memory[-32:]):
        if event.get('skill') not in ACTIONS:
            continue
        if event['skill'] != name or event.get('execution_feedback', {}).get('status') == 'completed':
            break
        failures += 1
    if failures >= maximum_uncertain_attempts:
        return empty
    side = current['route']['arms'][0] if current['route'] else 'auto'
    return {'applicable': True, 'operations': [{'skill': name, 'parameters': {'side': [side]}}],
            'relation': {'kind': 'current_public_garment_fold', 'part': current['part'], 'phase': current['phase']},
            'reason': 'Prefer the next original garment rule stage qualified by current RGB landmarks, own acknowledged commands and bounded visual uncertainty.'}
