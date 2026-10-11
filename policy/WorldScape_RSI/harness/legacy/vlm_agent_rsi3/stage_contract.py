"""Collection-only acceptance of completed toolbox release; no simulator queries."""
import math
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('rsi3_stage_supervisor',Path(__file__).with_name('toolbox_supervisor.py'))
supervision=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(supervision)

def assess(flow, state, feedback):
    event = {'phase': flow.get('phase'), 'tool': flow.get('tool'),
             'native_step': state.get('steps_used'), 'expected_score': None,
             'trigger_status': 'not_at_release_checkpoint',
             'stage_acceptance': 'not_due', 'early_stop_reason': None}
    if not feedback or feedback.get('uses_privileged_reward_feedback') is not True:
        event.update(trigger_status='feedback_disabled', stage_acceptance='unknown')
        return event
    event.update(current_score=feedback.get('current_score'),
                 source_update_step=feedback.get('source_update_step'))
    if flow.get('phase') != 'released':
        return event
    index = len(flow.get('completed_visual_checks') or [])
    if index >= 3:
        event.update(trigger_status='final_reset_required', stage_acceptance='deferred')
        return event
    event['expected_score'] = 25 * (index + 1)
    opened = all(isinstance(state.get(a, {}).get('opening'), (int, float))
                 and state[a]['opening'] > .8 for a in ('left', 'right'))
    score = feedback.get('current_score')
    fresh = (feedback.get('status') == 'fresh'
             and feedback.get('source_update_step') == state.get('steps_used')
             and feedback.get('native_step') == state.get('steps_used')
             and isinstance(score, (int, float)) and math.isfinite(score))
    if not opened or not fresh:
        event.update(trigger_status='unknown_or_jaws_not_open', stage_acceptance='unknown')
        return event
    verdict=supervision.stage_assessment(flow,state,feedback)
    event['supervisor_assessment']=verdict
    if verdict['decision'] not in ('advance_stage','stop_failure') or verdict.get('reason')!='scored_stage_missed' and verdict['decision']=='stop_failure':
        event.update(trigger_status=verdict.get('reason'),stage_acceptance='unknown')
        return event
    event.update(trigger_status='release_withdrawal_complete_jaws_open_native_score_event_confirmed',
                 stage_acceptance='pass' if verdict['decision']=='advance_stage' else 'fail')
    if event['stage_acceptance'] == 'fail':
        event['repair_required'] = True
        event['diagnosis'] = 'Fresh native score missed; inspect placement, allow one bounded settle check, then local repair or abort; never advance.'
    return event
