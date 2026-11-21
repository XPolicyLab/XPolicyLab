"""Tool schemas and argument checks for v2 sessions."""
from __future__ import annotations

CAMERAS = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
PLAN_STATUSES = ['pending', 'in_progress', 'completed']
SELF_ASSESSMENTS = ['success', 'partial', 'fail', 'unclear']
DECISIONS = ['reset_same', 'switch_layout', 'end']


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


_SELECTION = _object({
    'observation_id': {'type': ['string', 'null']},
    'command_id': {'type': ['string', 'null']},
    'phase': {'type': ['string', 'null'], 'enum': ['before', 'after', None]},
    'camera': {'type': 'string', 'enum': CAMERAS},
    'region': {'type': ['array', 'null'], 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}})

DEFINITIONS = {
    'python_exec': (
        'Run Python in your persistent session; variables, functions and imports persist between calls. '
        'Preloaded: robot (the Robot SDK), display(obj, cameras=None) to see an observation or action '
        'report, display_image(img, caption) to see an image you made, and the session helpers '
        'described in your instructions. One call may run several robot commands. The output lists every '
        'robot command as one line, then printed text and displayed items; images arrive when the call ends. '
        'timeout_s limits the call in wall time; simulated motion runs about 9 times slower than real time '
        '(action lines show wall seconds), so allow for it. A running motion is never cut short: it completes, '
        'then the rest of the code is skipped.',
        _object({'code': {'type': 'string'},
                 'timeout_s': {'type': 'integer', 'minimum': 1, 'maximum': 600}})),
    'observe': (
        'Current RGB from the chosen cameras (null for all three) and the robot state line. '
        'Costs no physical steps.',
        _object({'cameras': {'type': ['array', 'null'], 'items': {'type': 'string', 'enum': CAMERAS},
                             'maxItems': 3}})),
    'inspect': (
        'Re-view recorded images without moving the robot. Each selection names an observation_id, or a '
        'command_id with phase before/after, a camera and an optional pixel region [x, y, width, height]. '
        'Up to 6 selections; unused fields are null.',
        _object({'selections': {'type': 'array', 'items': _SELECTION, 'minItems': 1, 'maxItems': 6}})),
    'update_plan': (
        'Keep a short plan for this episode: the full list of steps, each pending, in_progress or completed, '
        'with at most one in_progress. The plan is shown back to you periodically and after context '
        'compaction. explanation is optional (null).',
        _object({'plan': {'type': 'array', 'minItems': 1, 'maxItems': 30, 'items': _object({
                     'step': {'type': 'string'}, 'status': {'type': 'string', 'enum': PLAN_STATUSES}})},
                 'explanation': {'type': ['string', 'null']}})),
    'finish': (
        'End the physical episode; no further motion is possible afterwards. summary: what you did and '
        'observed. self_assessment: your own judgment of the whole task outcome.',
        _object({'summary': {'type': 'string'},
                 'self_assessment': {'type': 'string', 'enum': SELF_ASSESSMENTS}})),
    'plan_next_episode': (
        'Only after the physical episode has ended: choose the next development episode and write the handoff '
        'for it. This ends the session. reset_same restarts this layout (replay_to may name a command of this '
        'episode to replay up to); switch_layout moves to layout_id (one of the development layouts of this task, '
        'listed in the task section); end finishes development of this task when you judge the knowledge ready: '
        'it gets a final review, is frozen, and is evaluated on '
        'layouts you have not seen with the official step budget (you will not see those results). The handoff '
        'tells your next episode what worked, what failed and what to try next.',
        _object({'decision': {'type': 'string', 'enum': DECISIONS},
                 'layout_id': {'type': ['integer', 'null']},
                 'replay_to': {'type': ['string', 'null']},
                 'handoff': {'type': 'string'}})),
    'end_session': (
        'End this session with a short summary of what you changed and why.',
        _object({'summary': {'type': 'string'}})),
}

# A development actor keeps one tool list for the whole session, so the cached
# prefix survives the switch to consolidation; the loop accepts plan_next_episode
# only after the physical episode has ended, and finish only before.
PROFILES = {
    'actor_dev': ('python_exec', 'observe', 'inspect', 'update_plan', 'finish', 'plan_next_episode'),
    'actor_eval': ('python_exec', 'observe', 'inspect', 'update_plan', 'finish'),
    'curation': ('python_exec', 'end_session'),
    'video': ('python_exec', 'update_plan', 'end_session'),
}


def schemas(profile):
    if profile not in PROFILES:
        raise ValueError('unknown tool profile ' + str(profile))
    return [{'type': 'function', 'name': name, 'description': DEFINITIONS[name][0], 'strict': True,
             'parameters': DEFINITIONS[name][1]} for name in PROFILES[profile]]


def _text(args, key, limit, allow_empty=False):
    value = args.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or len(value) > limit:
        raise ValueError(f'{key} must be text of at most {limit} characters')
    return value


def validate(name, args):
    """Return normalized arguments or raise ValueError with a message for the model."""
    if not isinstance(args, dict):
        raise ValueError('arguments must be a JSON object')
    expected = set(DEFINITIONS[name][1]['properties']) if name in DEFINITIONS else None
    if expected is None:
        raise ValueError(f'unknown tool {name}')
    if set(args) != expected:
        raise ValueError(f'{name} expects exactly these fields: {", ".join(sorted(expected))}')
    if name == 'python_exec':
        timeout = args['timeout_s']
        if type(timeout) is not int or not 1 <= timeout <= 600:
            raise ValueError('timeout_s must be an integer from 1 to 600')
        return {'code': _text(args, 'code', 200000), 'timeout_s': timeout}
    if name == 'observe':
        cameras = args['cameras']
        if cameras is not None:
            if not isinstance(cameras, list) or not cameras or any(c not in CAMERAS for c in cameras):
                raise ValueError('cameras must be null or a non-empty list of ' + ', '.join(CAMERAS))
            cameras = list(dict.fromkeys(cameras))
        return {'cameras': cameras}
    if name == 'inspect':
        selections = args['selections']
        if not isinstance(selections, list) or not 1 <= len(selections) <= 6:
            raise ValueError('selections must contain 1 to 6 entries')
        clean = []
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError('each selection must be an object')
            entry = {k: v for k, v in selection.items() if v is not None}
            if bool(entry.get('observation_id')) == bool(entry.get('command_id')):
                raise ValueError('each selection needs exactly one of observation_id or command_id')
            if entry.get('command_id') and entry.get('phase') not in ('before', 'after'):
                raise ValueError('a command_id selection needs phase before or after')
            if entry.get('observation_id') and entry.get('phase'):
                raise ValueError('phase applies only to command_id selections')
            if entry.get('camera') not in CAMERAS:
                raise ValueError('camera must be one of ' + ', '.join(CAMERAS))
            region = entry.get('region')
            if region is not None and (len(region) != 4 or any(type(v) is not int or v < 0 for v in region)
                                       or region[2] == 0 or region[3] == 0):
                raise ValueError('region must be [x, y, width, height] with non-negative integers')
            clean.append(entry)
        return {'selections': clean}
    if name == 'update_plan':
        plan = args['plan']
        if not isinstance(plan, list) or not 1 <= len(plan) <= 30:
            raise ValueError('plan must contain 1 to 30 steps')
        steps = []
        for item in plan:
            if not isinstance(item, dict) or item.get('status') not in PLAN_STATUSES:
                raise ValueError('each step needs text and a status of ' + ', '.join(PLAN_STATUSES))
            steps.append({'step': _text(item, 'step', 300), 'status': item['status']})
        if sum(step['status'] == 'in_progress' for step in steps) > 1:
            raise ValueError('at most one step can be in_progress')
        explanation = args['explanation']
        if explanation is not None and (not isinstance(explanation, str) or len(explanation) > 1000):
            raise ValueError('explanation must be null or text of at most 1000 characters')
        return {'plan': steps, 'explanation': explanation}
    if name == 'finish':
        if args['self_assessment'] not in SELF_ASSESSMENTS:
            raise ValueError('self_assessment must be one of ' + ', '.join(SELF_ASSESSMENTS))
        return {'summary': _text(args, 'summary', 4000), 'self_assessment': args['self_assessment']}
    if name == 'plan_next_episode':
        decision, layout = args['decision'], args['layout_id']
        if decision not in DECISIONS:
            raise ValueError('decision must be one of ' + ', '.join(DECISIONS))
        if decision == 'switch_layout':
            if type(layout) is not int or layout < 0:
                raise ValueError('switch_layout needs a layout_id from the development layouts of this task')
        elif layout is not None:
            raise ValueError('layout_id is only used with switch_layout')
        replay_to = args['replay_to']
        if replay_to is not None and (decision != 'reset_same' or not isinstance(replay_to, str)
                                      or not replay_to.strip()):
            raise ValueError('replay_to is a command ID and is only used with reset_same')
        return {'decision': decision, 'layout_id': layout, 'replay_to': replay_to,
                'handoff': _text(args, 'handoff', 6000)}
    if name == 'end_session':
        return {'summary': _text(args, 'summary', 4000)}
    raise ValueError(f'unknown tool {name}')
