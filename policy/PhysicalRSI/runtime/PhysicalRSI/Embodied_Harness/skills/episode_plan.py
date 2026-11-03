"""Transfer explicit agent operation selections into isolated episode workers."""
from copy import deepcopy
import os
from pathlib import Path
from PhysicalRSI_core.infra.storage import atomic_json, digest, read_json

SCHEMA = 'physicalrsi.episode-operation-plan/v1'
PATH_ENV = 'PHYSICALRSI_OPERATION_PLAN'
DIGEST_ENV = 'PHYSICALRSI_OPERATION_PLAN_SHA256'


def catalogue(memory):
    """Expose the operation programs available in immutable memory."""
    if memory.get('schema') != 'physicalrsi.skill-program-memory/v1':
        raise ValueError('Versioned operation memory required')
    programs = memory.get('programs')
    if not isinstance(programs, dict) or not programs:
        raise ValueError('Operation memory requires programs')
    result = {}
    for name, program in programs.items():
        validate_program(memory, name, program.get('steps'))
        bindings = memory.get('bindings', {}).get(name)
        result[name] = {'steps': deepcopy(program['steps'])}
        if bindings:
            result[name].update(input=bindings[0]['input'], output=bindings[-1]['output'],
                                operations=[dict(id=b['operation'], input=b['input'],
                                                 output=b['output']) for b in bindings])
    return {'memory_revision': digest(memory), 'programs': result,
            'composition_scope': 'Use a complete capability composition and preserve its declared stage order.'}


def validate_program(memory, name, steps):
    if not isinstance(steps, list) or not steps:
        raise ValueError('Agent operation program requires explicit steps')
    for step in steps:
        if (not isinstance(step, dict) or set(step) != {'operation', 'revision'}
                or not isinstance(step['operation'], str) or not step['operation']
                or not isinstance(step['revision'], str) or not step['revision']):
            raise ValueError('Operation step requires identity and revision')
    expected = memory['programs'][name]['steps']
    if steps != expected:
        raise ValueError('Operation sequence does not match the exact registered memory order')
    bindings = memory.get('bindings', {}).get(name)
    if bindings is not None:
        if not isinstance(bindings, list) or not bindings:
            raise ValueError('Registered bindings must be nonempty')
        if [b['operation'] for b in bindings] != [s['operation'] for s in steps]:
            raise ValueError('Registered bindings differ from program steps')
        if any(a['output'] != b['input'] for a, b in zip(bindings, bindings[1:])):
            raise ValueError('Registered operation contracts do not compose')
        revision = memory.get('revision')
        if revision is not None and any(s['revision'] != revision for s in steps):
            raise ValueError('Operation revision mismatch')
    return deepcopy(steps)


def validate_episode_plan(plan, memories):
    if not isinstance(plan, dict) or set(plan) != {'schema', 'episode', 'rationale', 'programs'}:
        raise ValueError('Exact episode operation plan fields required')
    if plan['schema'] != SCHEMA or not isinstance(plan['episode'], str) or not plan['episode'].strip():
        raise ValueError('Versioned episode identity required')
    if not isinstance(plan['rationale'], str) or not plan['rationale'].strip():
        raise ValueError('Agent rationale required')
    if not isinstance(plan['programs'], dict) or set(plan['programs']) != set(memories):
        raise ValueError('Plan must cover exactly the configured memory documents')
    for revision, memory in memories.items():
        if digest(memory) != revision:
            raise ValueError('Memory document revision mismatch')
        programs = plan['programs'][revision]
        if not isinstance(programs, dict) or set(programs) != set(memory['programs']):
            raise ValueError('Plan must cover every callable method in its memory document')
        for name, steps in programs.items():
            validate_program(memory, name, steps)
    return deepcopy(plan)


def freeze_episode_plan(path, plan, memories):
    """Write the local immutable launch input."""
    plan = validate_episode_plan(plan, memories)
    path = Path(path).resolve()
    if path.exists():
        if read_json(path) != plan:
            raise ValueError('Refusing to overwrite an episode plan')
    else:
        atomic_json(path, plan)
    return {PATH_ENV: str(path), DIGEST_ENV: digest(plan)}


def apply_launch_plan(memory, environment=None):
    """Apply the episode plan when launch guidance is configured."""
    environment = os.environ if environment is None else environment
    path = environment.get(PATH_ENV)
    expected = environment.get(DIGEST_ENV)
    if not path and not expected:
        return memory
    if not path or not expected:
        raise ValueError('Both episode plan path and digest are required')
    plan = read_json(Path(path))
    if digest(plan) != expected:
        raise ValueError('Episode operation plan changed after launch')
    if (not isinstance(plan, dict)
            or set(plan) != {'schema', 'episode', 'rationale', 'programs'}
            or plan.get('schema') != SCHEMA
            or not isinstance(plan.get('episode'), str) or not plan['episode'].strip()
            or not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip()
            or not isinstance(plan.get('programs'), dict)):
        raise ValueError('Invalid episode operation plan')
    revision = digest(memory)
    programs = plan['programs'].get(revision)
    if not isinstance(programs, dict) or set(programs) != set(memory['programs']):
        raise ValueError('Episode plan has no complete program for this memory revision')
    selected = deepcopy(memory)
    selected['programs'] = {name: {'steps': validate_program(memory, name, steps)}
                            for name, steps in programs.items()}
    selected['agent_plan'] = {'episode': plan['episode'], 'revision': expected,
                              'parent_memory_revision': revision}
    return selected
