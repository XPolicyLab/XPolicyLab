"""Bounded adapters to the existing command/trajectory executor, never eval Python."""
import math


def enabled(skill, experimental=()):
    return skill.get('validated') is True or skill.get('id') in experimental


def bind(skill, bindings):
    spec = skill['inputs']
    if set(bindings) != set(spec):
        raise ValueError('Skill inputs must match registered schema')
    values = {}
    for name, rule in spec.items():
        entry = bindings[name]
        allowed = ('current_observation','derived_from_observation') if rule.get('live_only') else ('current_observation','derived_from_observation','constant')
        if entry.get('source') not in allowed or not entry.get('evidence'):
            raise ValueError('Input provenance required')
        value = entry['value']
        if 'enum' in rule:
            if value not in rule['enum']: raise ValueError('Invalid categorical parameter')
        elif type(value) not in (int,float) or not math.isfinite(value) or not rule['min'] <= value <= rule['max']:
            raise ValueError('Out-of-range skill parameter')
        values[name] = value
    if skill['implementation'] == 'trajectory_template':
        return [line.format_map(values) for line in skill['commands']]
    if skill['implementation'] == 'retained_translate':
        if sum(values[k]**2 for k in ('dx','dy','dz')) > 9:
            raise ValueError('Translation norm exceeds 3cm')
        return [f"{values['arm']} move3d {values['dx']} {values['dy']} {values['dz']}"]
    if skill['implementation'] == 'close_probe':
        return [f"{values['arm']} gripper close",f"{values['arm']} move z {values['lift_cm']}"]
    raise ValueError('Unregistered code implementation')


def eligible(skill, stage, evidence, experimental=()):
    if not enabled(skill,experimental) or skill.get('executor') != 'cap': return False
    if not set(stage['completion']).issubset(set(skill['postconditions'])): return False
    if not evidence.satisfies(skill['preconditions']): return False
    try: bind(skill, stage.get('bindings',{}))
    except (KeyError,TypeError,ValueError): return False
    return True
