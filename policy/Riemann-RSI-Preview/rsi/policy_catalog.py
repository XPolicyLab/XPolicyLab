"""Public task dispatch; no layout, scorer, scene or hidden observation access."""
import json
from pathlib import Path
import re

from .rule_configuration import settings

ROOT = Path(__file__).resolve().parents[1]


def catalog(root=ROOT):
    root = Path(root)
    rule = settings(root=root)
    learned = json.loads((root / 'policies/riemann_1/policy.json').read_text())
    if set(rule['tasks']) & set(learned['tasks']):
        raise ValueError('A task cannot select both rule and learned backends')
    tasks = {mode: dict(record, backend='rule') for mode, record in rule['tasks'].items()}
    tasks.update({mode: dict(record, backend='learned', policy_id='riemann_1')
                  for mode, record in learned['tasks'].items()})
    from .task_contract import metadata
    for mode, route in tasks.items():
        facts = metadata(mode, root=root)
        expected = 'learned_policy' if route['backend'] == 'learned' else 'rule'
        if facts.get('backend') != expected or (expected == 'learned_policy'
                and facts.get('policy_id') != route['policy_id']):
            raise ValueError('Task facts and backend declaration disagree: ' + mode)
    return tasks


# These are public task descriptions, never environment labels or object state.
# Standard/random variants can share an instruction, so ambiguous text requires
# the task_name supplied by the official launcher instead of guessing a layout.
_PHRASES = {
    'plug_in_charger': r'\b(?:plug|insert|connect)\b.*\bcharger\b',
    'insert_key': r'\b(?:insert|put)\b.*\bkey\b',
    'deposit_coin': r'\b(?:deposit|insert|put)\b.*\bcoin\b',
    'general_pickup': r'\b(?:pick up|pickup|grasp)\b.*\b(?:object|item)\b',
    'store_tools_in_toolbox': r'\b(?:store|put|place)\b.*\btoolbox\b',
    'fasten_screws': r'\b(?:fasten|tighten|screw)\b.*\b(?:screw|nut|bolt)s?\b',
    'fill_egg_holder': r'\b(?:fill|put|place)\b.*\begg\b.*\b(?:holder|rack)\b',
    'fill_pen_holder': r'\b(?:fill|put|place)\b.*\bpen\b.*\bholder\b',
    'push_T': r'\bpush\b.*\bt(?:-shaped)?\b',
    'pour_liquid_into_cup': r'\bpour\b.*\b(?:liquid|water)\b.*\bcup\b',
    'pour_balls_into_vase': r'\bpour\b.*\bballs?\b.*\bvase\b',
    'fold_clothes': r'\bfold\b.*\b(?:clothes|cloth|garment|towel)\b',
    'hang_mugs': r'\bhang\b.*\bmugs?\b',
    'stack_bowls': r'\bstack\b.*\bbowls?\b',
    'stack_blocks': r'\bstack\b.*\bblocks?\b',
    'pack_objects_into_box': r'\b(?:pack|put|place)\b.*\b(?:object|item)s?\b.*\bbox\b',
    'build_tower': r'\bbuild\b.*\btower\b',
    'make_kong': r'\b(?:make|build)\b.*\bkong\b',
    'put_bottles_into_dustbin': r'\b(?:put|place|throw)\b.*\bbottles?\b.*\b(?:dustbin|trash)\b',
    'store_laptop_and_headphones': r'\b(?:store|put|place)\b.*\blaptop\b.*\bheadphones\b',
    'sweep_blocks': r'\bsweep\b.*\bblocks?\b',
}


def resolve_task(instruction, task_name=None, *, root=ROOT):
    tasks = catalog(root)
    if instruction is not None and (not isinstance(instruction, str) or len(instruction) > 5000):
        raise ValueError('Expected a bounded official public instruction string')
    if task_name is not None:
        if task_name not in tasks:
            raise ValueError('Unregistered official task: ' + str(task_name))
        # task_name is public launch configuration, not observed/hidden scene
        # state. The unchanged instruction is still passed to the selected actor.
        return task_name, tasks[task_name]
    text = re.sub(r'\s+', ' ', (instruction or '').strip()).lower()
    named = {mode for mode in tasks if text == mode.lower() or text == mode.lower().replace('_', ' ')}
    if not named:
        for mode in tasks:
            base = mode.removesuffix('_random')
            pattern = _PHRASES.get(base)
            if pattern and re.search(pattern, text, re.IGNORECASE):
                named.add(mode)
    if len(named) != 1:
        raise ValueError('Instruction is unknown or ambiguous; supply the official task_name (including its variant)')
    mode = named.pop()
    return mode, tasks[mode]


def runtime_settings(mode=None, *, root=ROOT):
    common = settings(root=root)
    tasks = catalog(root)
    if mode is not None and mode not in tasks:
        raise ValueError('Unregistered official task: ' + str(mode))
    return dict(common, tasks=tasks, policy_source_count=len(tasks),
                expected_qualified_episodes=sum(row.get('official_eval_nums', row.get('official_episodes', 0))
                                               for row in tasks.values()))
