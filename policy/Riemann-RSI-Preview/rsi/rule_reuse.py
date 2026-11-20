"""Explicit std-to-random reuse without copying or modifying mature skills."""
import json
from pathlib import Path

def implementation_mode(root, mode):
    root = Path(root)
    plan = json.loads((root / 'tasks' / mode / 'composition.json').read_text())
    if plan.get('schema') != 'riemann-rule-reuse-v1':
        return mode
    source = plan.get('implementation_mode')
    if (not isinstance(source, str) or not source.isidentifier()
            or mode != source + '_random' or plan.get('task_mode') != mode
            or plan.get('backend') != 'rule' or plan.get('source_versioning') != 'git'
            or plan.get('parameter_overrides') not in ({}, {'public_shape_profile': 'random'})
            or (plan.get('parameter_overrides') and mode != 'sort_nesting_dolls_by_size_random')
            or plan.get('workflow_overrides') != {}):
        raise ValueError('Expected an unchanged, explicitly declared std-to-random rule')
    original = json.loads((root / 'tasks' / source / 'composition.json').read_text())
    if original.get('schema') != 'riemann-rule-composition-v1':
        raise ValueError('Rule reuse requires a concrete original implementation')
    from .rule_configuration import settings
    data = settings(root=root)['tasks']
    for name in (mode, source):
        if json.loads((root / 'tasks' / name / 'task.json').read_text()).get('backend') != 'rule':
            raise ValueError('Both reuse declarations must select rule')
    for key in ('official_action_horizon', 'adapter_family'):
        if data[mode][key] != data[source][key]:
            raise ValueError('Rule reuse changes the execution contract: ' + key)
    return source
