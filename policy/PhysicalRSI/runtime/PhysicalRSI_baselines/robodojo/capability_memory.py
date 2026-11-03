"""Implementation-grounded capability memory and advisory observation checks."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json

_FORBIDDEN = {'task_id', 'task_name', 'layout_id', 'seed_id', 'score_prior',
              'task_scores', 'preferred_backend', 'best_policy', 'success_rate'}


def _validate(entry):
    if (entry.get('schema') != 'physicalrsi.skill-capability-memory/v1'
            or entry.get('kind') != 'implementation_contract'):
        raise ValueError('Unsupported capability memory schema')
    payload = {key: value for key, value in entry.items() if key != 'capability_id'}
    expected = 'cap-' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    if entry.get('capability_id') != expected:
        raise ValueError('Capability memory identity mismatch')
    if entry.get('provenance') != {'type': 'implementation_inspection', 'rollout_derived': False}:
        raise ValueError('Unsupported capability provenance')
    if (entry.get('input_contract') != 'robodojo.observation-batch/v1'
            or entry.get('output_contract') != 'robodojo.action-chunks/v1'):
        raise ValueError('Unsupported capability interface')
    if entry['implementation']['kind'] == 'code-policy' and not entry.get('operation_memory'):
        raise ValueError('Code capability requires operation memory references')

    def walk(value):
        if isinstance(value, dict):
            if _FORBIDDEN.intersection(value):
                raise ValueError('Capability memory cannot contain benchmark identity or performance priors')
            if {'root', 'path', 'sha256'} <= value.keys():
                path = Path(value['path'])
                if (value['root'] not in {'code-skills', 'policy-runtime'} or path.is_absolute()
                        or '..' in path.parts or len(value['sha256']) != 64
                        or any(c not in '0123456789abcdef' for c in value['sha256'])):
                    raise ValueError('Invalid capability source reference')
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(entry)
    for condition in entry.get('conditions', []):
        if (condition.get('predicate') not in {'array_shape', 'nonempty_string'}
                or condition.get('advisory') is not True
                or not condition.get('path')
                or condition['path'][0] not in {'vision', 'state', 'instruction'}):
            raise ValueError('Unsupported capability condition')


@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).parent / 'configs/skill_capabilities.json'
    document = json.loads(path.read_text())
    if document.get('schema') != 'physicalrsi.capability-library/v1':
        raise ValueError('Invalid capability library schema')
    result = {}
    identities = set()
    for entry in document['entries']:
        _validate(entry)
        if entry['skill_name'] in result or entry['capability_id'] in identities:
            raise ValueError('Duplicate capability identity')
        result[entry['skill_name']] = entry
        identities.add(entry['capability_id'])
    return result


def attach_capability_memory(entry, skill_name):
    """Attach the frozen contract without replacing custom capability metadata."""
    memory = _library().get(skill_name)
    if memory is None:
        return entry
    result = deepcopy(entry)
    result['capability_id'] = memory['capability_id']
    result['capability_memory'] = deepcopy(memory)
    return result


def observed_conditions(entry, observation):
    """Return explicit, advisory observations; unknown never means inapplicable."""
    results = []
    for condition in entry.get('capability_memory', {}).get('conditions', []):
        value = observation
        status = 'unknown'
        try:
            for key in condition['path']:
                value = value[key]
        except (KeyError, TypeError, IndexError):
            pass
        else:
            if condition['predicate'] == 'nonempty_string':
                status = 'satisfied' if isinstance(value, str) and value.strip() else 'violated'
            elif condition['predicate'] == 'array_shape' and hasattr(value, 'shape'):
                status = 'satisfied' if list(value.shape) == condition['shape'] else 'violated'
        results.append(dict(condition_id=condition['id'], status=status, advisory=True,
                            scope=condition['scope'], source=condition['evidence']))
    return results


def catalogue_with_observations(catalogue, observation):
    """Preserve every candidate and its stages while exposing checked conditions."""
    result = deepcopy(catalogue)
    for entry in result.values():
        if 'capability_memory' in entry:
            entry['observed_conditions'] = observed_conditions(entry, observation)
    return result
