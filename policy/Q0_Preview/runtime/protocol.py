"""Three-point action wire contract. Pure Python/NumPy; never reads simulator task state."""
from __future__ import annotations

import math
import copy
import json
import numpy as np
from ..controller.robot_geometry import jaw_gap_mm

ARMS = ('left', 'right')
MAX_GROUPS = 2
MAX_GROUP_SECONDS = 3.


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': False}


VEC = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
POINTS = {'type': 'array', 'items': VEC, 'minItems': 2, 'maxItems': 2,
          'description': 'ORDERED [p0,p1]: p0 is the -local-Y finger side (link8), p1 is the +local-Y finger side (link7), for either arm. These are pose reference points, not measured fingertips. p0 -> p1 is the signed jaw axis. Keep physical finger identity across observations; never sort by image position or swap to seek an equivalent pose. Reversing the points requests a real 180-degree tool roll, not a quaternion sign change.'}
GRIPPER = obj({'command': {'enum': ['keep', 'open', 'close', 'width']},
               'width_mm': {'type': ['number', 'null'], 'minimum': 0, 'maximum': jaw_gap_mm(1.)},
               'held_width_mm': {'type': ['number', 'null'], 'minimum': 0, 'maximum': jaw_gap_mm(1.),
                   'description': 'Estimated object thickness between the pads after this event. Set when grasping; keep reuses it and open clears it. Not measured opening or proof of possession.'},
               'when': {'enum': ['before', 'after']}}, required=['command', 'when'])
CONTACT = obj({'surface_point': {**VEC, 'description': 'A point on the plane in world coordinates, metres.'},
               'surface_normal': {**VEC, 'description': 'Unit plane normal in world coordinates.'},
               'working_point_local': {**VEC, 'description': 'Offset in metres from the grasp reference (target_points midpoint), in local axes (tool, jaw, tool cross jaw). Not a world position or a flange-relative offset. World point = grasp_reference + frame_rotation @ offset.'},
               'tolerance_mm': {'type': 'number', 'minimum': 0.5, 'maximum': 5}})
TARGET = obj({'target_points': POINTS, 'orientation_point': VEC,
              'contact_constraint': {'anyOf': [CONTACT, {'type': 'null'}]}},
             required=['target_points', 'orientation_point'])
GROUP = obj({'active_arms': {'enum': ['left', 'right', 'both', 'none']},
             'left': {'anyOf': [TARGET, {'type': 'null'}]},
             'right': {'anyOf': [TARGET, {'type': 'null'}]},
             'grippers': obj({a: GRIPPER for a in ARMS}),
             'duration_s': {'type': 'number', 'minimum': 0, 'maximum': MAX_GROUP_SECONDS},
             'checkpoint': {'type': 'boolean', 'description': 'Required true on the first of two groups; omit/false on a single or final group. Group two is gated by an independent low-effort review.'},
             'expectation': {'type': 'string', 'maxLength': 500, 'description': 'For the first of two groups, required nonempty description of the visible intermediate outcome and evidence needed to safely execute the already specified second target.'}},
            required=['active_arms', 'grippers', 'duration_s'])
MEMORY_TARGET_TOKENS = 1200  # Prompt target; no exact local Claude tokenizer is available.
MEMORY_MAX_CHARACTERS = 6000  # Exceptional-output guard, not a token limit.
PLAN_SCHEMA = obj({'memory': {'type': 'string', 'maxLength': MEMORY_MAX_CHARACTERS,
                               'description': 'Latest self-contained task summary, at most 1,200 tokens. The 6,000-character maxLength is an exceptional-output guard, not a token count.'},
                   'groups': {'type': 'array', 'minItems': 1, 'maxItems': MAX_GROUPS, 'items': GROUP, 'description': 'Normally one group. Two may combine confidently planned consecutive actions or pass through an intermediate pose for clearance. Both groups must be specified from the current observation; the first must request a checkpoint and state its expectation.'}},
                  required=['groups'])


class ContractError(ValueError):
    pass


def exact(value, keys, name):
    if not isinstance(value, dict):
        raise ContractError(f'{name}: expected an object')
    missing, unexpected = sorted(set(keys)-set(value)), sorted(set(value)-set(keys))
    if missing or unexpected:
        raise ContractError(f'{name}: missing fields {missing}; unexpected fields {unexpected}')


def normalize_plan(raw):
    """Only restore semantically irrelevant fields. Never invent/alter robot actions."""
    plan, repairs = copy.deepcopy(raw), []
    def add(obj, key, value, path):
        if key not in obj:
            obj[key] = value
            repairs.append({'path': path+'.'+key, 'value': value, 'reason': 'unambiguous omitted field'})
    if not isinstance(plan, dict):
        raise ContractError('plan: expected an object')
    # Descriptive metadata must not invalidate an otherwise valid motion.
    if plan.get('memory') is None:
        if 'memory' in plan:
            plan['memory'] = ''
            repairs.append({'path':'$.memory', 'value':'', 'reason':'empty optional summary'})
        else:
            add(plan, 'memory', '', '$')
    if isinstance(plan.get('memory'), str) and len(plan['memory']) > MEMORY_MAX_CHARACTERS:
        plan['memory'] = plan['memory'][:MEMORY_MAX_CHARACTERS]
        repairs.append({'path':'$.memory', 'reason':'summary exceeded 6000-character guard; copy truncated and incomplete; actions unchanged'})
    if isinstance(plan.get('groups'), str):
        # A valid JSON wrapper can be removed without changing a single action.
        # Never guess/truncate malformed JSON or coerce coordinate strings.
        try:
            decoded = json.loads(plan['groups'])
        except json.JSONDecodeError as exc:
            raise ContractError(f'$.groups: expected a JSON array, got a malformed JSON string ({exc.msg}). '
                                'Send groups as an array, not an escaped/quoted string.') from exc
        if isinstance(decoded, list):
            plan['groups'] = decoded
            repairs.append({'path':'$.groups','reason':'decoded a valid JSON array wrapper; actions unchanged'})
    if not isinstance(plan.get('groups'), list):
        raise ContractError('$.groups: expected a JSON array containing one or two group objects, not a string or object')
    for i, g in enumerate(plan['groups']):
        path = f'$.groups[{i}]'
        if not isinstance(g, dict):
            raise ContractError(path+': expected an object')
        active = {'left': ('left',), 'right': ('right',), 'both': ARMS, 'none': ()}.get(g.get('active_arms'))
        if active is not None:
            for arm in ARMS:
                if arm not in active:
                    add(g, arm, None, path)
                t = g.get(arm)
                if isinstance(t, dict):
                    add(t, 'contact_constraint', None, path+'.'+arm)
        grippers = g.get('grippers')
        if isinstance(grippers, dict):
            for arm, command in grippers.items():
                if isinstance(command, dict) and command.get('command') in ('keep', 'open', 'close'):
                    add(command, 'width_mm', None, path+'.grippers.'+arm)
    validate_plan(plan)
    return plan, repairs


def number(v, low, high, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
        raise ContractError(f'{name}: finite number in [{low}, {high}] required')
    return float(v)


def vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ContractError(f'{name}: expected three coordinates')
    return np.array([number(v, -100, 100, name) for v in value])


def frame(target):
    """Every target has one meaning: midpoint=grasp reference; third point=orientation only.

    Ordered finger points retain finger identity. Never swap jaws or change target pose.
    """
    exact(target, TARGET['properties'], 'target')
    ps = target['target_points']
    if not isinstance(ps, list) or len(ps) != 2:
        raise ContractError('two target points required')
    p, q = (vector(v, 'target point') for v in ps)
    center = (p + q) / 2
    axis = center - vector(target['orientation_point'], 'orientation point')
    if np.linalg.norm(axis) < .005:
        raise ContractError('orientation point must be at least 5 mm from midpoint')
    axis /= np.linalg.norm(axis)
    jaw = q - p
    if not .005 - 1e-9 <= np.linalg.norm(jaw) <= .085 + 1e-9:
        raise ContractError('reference-point separation must be 5–85 mm (not a jaw command)')
    jaw -= axis * (jaw @ axis)
    if np.linalg.norm(jaw) < .001:
        raise ContractError('three points nearly collinear')
    jaw /= np.linalg.norm(jaw)
    rot = np.column_stack([axis, jaw, np.cross(axis, jaw)])
    c = target['contact_constraint']
    if c is not None:
        exact(c, CONTACT['properties'], 'contact_constraint')
        vector(c['surface_point'], 'surface_point')
        n = vector(c['surface_normal'], 'surface_normal')
        if not .99 <= np.linalg.norm(n) <= 1.01:
            raise ContractError('surface_normal must be unit length')
        vector(c['working_point_local'], 'working_point_local')
        number(c['tolerance_mm'], .5, 5, 'contact tolerance')
    return center, rot


def contact_error(target, pose):
    c = target['contact_constraint']
    if c is None:
        return 0.
    p, r = pose
    n = np.asarray(c['surface_normal'], float)
    n /= np.linalg.norm(n)
    working = p + r @ np.asarray(c['working_point_local'], float)
    return abs(float((working - c['surface_point']) @ n))


def validate_plan(plan):
    # Older local smoke callers may omit the optional summary.
    exact(plan, ['groups', 'memory'] if 'memory' in plan else ['groups'], 'plan')
    if 'memory' in plan and (not isinstance(plan['memory'], str) or len(plan['memory']) > MEMORY_MAX_CHARACTERS):
        raise ContractError('memory: string of at most 6000 characters required (not an exact token limit)')
    groups = plan['groups']
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_GROUPS:
        raise ContractError('submit one group, or at most two groups with a mandatory intermediate checkpoint')
    for i, g in enumerate(groups):
        required = set(GROUP['properties']) - {'checkpoint', 'expectation'}
        if isinstance(g, dict):
            required |= set(g) & {'checkpoint', 'expectation'}
        exact(g, required, f'$.groups[{i}]')
        checkpoint = g.get('checkpoint', False)
        expectation = g.get('expectation', '')
        if type(checkpoint) is not bool:
            raise ContractError(f'$.groups[{i}].checkpoint: boolean required')
        if not isinstance(expectation, str) or len(expectation) > 500:
            raise ContractError(f'$.groups[{i}].expectation: string of at most 500 characters required')
        if len(groups) == 2 and i == 0:
            if not checkpoint or not expectation.strip():
                raise ContractError('two groups require checkpoint:true and a nonempty expectation on the first group')
        elif checkpoint:
            raise ContractError('checkpoint:true is only valid between two groups; a single/final group returns to the main planner')
        if g['active_arms'] not in ('left', 'right', 'both', 'none'):
            raise ContractError('invalid active_arms')
        active = ARMS if g['active_arms'] == 'both' else (() if g['active_arms'] == 'none' else (g['active_arms'],))
        for a in ARMS:
            if a in active:
                try:
                    frame(g[a])
                except (KeyError, TypeError, ValueError) as e:
                    raise ContractError(f'$.groups[{i}].{a}: {e}') from e
            elif g[a] is not None:
                raise ContractError('inactive arm target must be null')
        number(g['duration_s'], 0, MAX_GROUP_SECONDS, 'duration_s')
        if g['active_arms'] == 'none' and g['duration_s'] <= 0:
            raise ContractError('none requires positive simulated wait duration')
        exact(g['grippers'], ARMS, 'grippers')
        for a, c in g['grippers'].items():
            keys = set(GRIPPER['properties']) - {'held_width_mm'}
            if 'held_width_mm' in c:
                keys.add('held_width_mm')
            exact(c, keys, f'$.groups[{i}].grippers.{a}')
            if c.get('held_width_mm') is not None:
                number(c['held_width_mm'], 0, jaw_gap_mm(1.), 'held_width_mm')
                if c['command'] == 'open':
                    raise ContractError('open clears held_width_mm; omit this estimate when opening')
            if c['command'] not in ('keep', 'open', 'close', 'width') or c['when'] not in ('before', 'after'):
                raise ContractError('invalid gripper command/timing')
            if c['command'] == 'width':
                number(c['width_mm'], 0, jaw_gap_mm(1.), 'width_mm')
            elif c['width_mm'] is not None:
                raise ContractError('only width command accepts width_mm')
    return plan



DECISION_SCHEMA = obj({'decision': {'enum': ['CONTINUE', 'REPLAN']},
                       'reason': {'type': 'string', 'minLength': 1, 'maxLength': 300}})


def validate_decision(value):
    exact(value, DECISION_SCHEMA['properties'], 'checkpoint_decision')
    if value['decision'] not in ('CONTINUE', 'REPLAN'):
        raise ContractError('checkpoint_decision: expected CONTINUE or REPLAN')
    if not isinstance(value['reason'], str) or not value['reason'].strip() or len(value['reason']) > 300:
        raise ContractError('checkpoint_decision.reason: nonempty string of at most 300 characters required')
    return value
