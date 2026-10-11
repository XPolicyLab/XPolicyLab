import hashlib
import json
import math
from dataclasses import dataclass, field

DECISIONS = ('continue', 'complete', 'observe', 'recover', 'escalate', 'stop')
EXECUTORS = ('wam', 'wam_mem', 'cap', 'astra')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def checked_text(value, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError('Expected bounded nonempty text')
    return value

def validate_plan(plan, skills, available):
    """A plan cannot introduce executable Python or silently use unsupported policies."""
    if set(plan) != {'id', 'version', 'goal', 'preconditions', 'stages'}:
        raise ValueError('Invalid plan fields')
    for key in ('id', 'version', 'goal'):
        checked_text(plan[key])
    if not isinstance(plan['preconditions'], list):
        raise ValueError('preconditions must be a list')
    stages = plan['stages']
    if not isinstance(stages, list) or not 1 <= len(stages) <= 24:
        raise ValueError('Expected 1..24 stages')
    seen = set()
    for stage in stages:
        if set(stage) - {'bindings'} != {'id', 'goal', 'executor', 'skill', 'completion', 'recoveries', 'max_blocks'}:
            raise ValueError('Invalid stage fields')
        checked_text(stage['id']); checked_text(stage['goal'])
        if stage['id'] in seen:
            raise ValueError('Duplicate stage')
        seen.add(stage['id'])
        if stage['executor'] not in EXECUTORS or stage['executor'] not in available:
            raise ValueError('Unavailable executor')
        if type(stage['max_blocks']) is not int or not 1 <= stage['max_blocks'] <= 128:
            raise ValueError('Invalid stage budget')
        if not isinstance(stage['completion'], list) or not stage['completion']:
            raise ValueError('Completion evidence required')
        if stage['skill'] is not None:
            skill = skills[stage['skill']]
            if skill['executor'] != stage['executor']:
                raise ValueError('Skill/executor mismatch')
        elif stage['executor'] == 'cap':
            raise ValueError('CaP requires a registered executable skill')
        if not isinstance(stage['recoveries'], list):
            raise ValueError('Invalid recovery list')
        for name in stage['recoveries']:
            skill = skills[name]
            if not skill.get('validated') or not skill.get('recovery'):
                raise ValueError('Recovery is not validated')
        for key in stage['completion']:
            checked_text(key, 200)
        for name, binding in stage.get('bindings', {}).items():
            checked_text(name, 100)
            if set(binding) != {'value', 'source', 'evidence'}:
                raise ValueError('A bound parameter requires provenance')
            if binding['source'] not in ('current_observation', 'derived_from_observation', 'constant'):
                raise ValueError('Invalid binding source')
            checked_text(binding['evidence'])
            if type(binding['value']) not in (int, float, str) or isinstance(binding['value'], float) and not math.isfinite(binding['value']):
                raise ValueError('Invalid bound value')
    for key in plan['preconditions']:
        checked_text(key, 200)
    return plan

@dataclass
class Evidence:
    revision: int
    steps: int
    remaining: int
    facts: dict = field(default_factory=dict)
    motion: dict = field(default_factory=dict)
    refs: list = field(default_factory=list)
    native_success: bool = False
    ended: bool = False

    def decision_state(self):
        # Explicit allowlist. No task IDs, layout, seed, reward or privileged object poses.
        return {'revision': self.revision, 'steps': self.steps, 'remaining': self.remaining,
                'facts': self.facts, 'motion': self.motion, 'evidence_refs': self.refs}

    def satisfies(self, conditions):
        return bool(conditions) and all(self.facts.get(k) is True for k in conditions)

@dataclass(frozen=True)
class Ticket:
    epoch: int
    stage: str
    revision: int
    executor: str

class Ownership:
    def __init__(self):
        self.epoch = 0
        self.executor = None

    def transfer(self, executor, cancel):
        cancel()  # Cancellation failure must not grant the next writer ownership.
        self.epoch += 1
        self.executor = executor

    def ticket(self, stage, evidence):
        return Ticket(self.epoch, stage, evidence.revision, self.executor)

    def validate(self, ticket, stage, evidence):
        if ticket != self.ticket(stage, evidence):
            raise RuntimeError('Stale action or decision: ownership/phase/observation changed')

def validate_choice(answer, choices):
    if answer.get('choice') not in choices:
        raise ValueError('Unregistered choice')
    probs = answer.get('probabilities')
    if not isinstance(probs, dict) or set(probs) != set(choices):
        raise ValueError('Incomplete probability distribution')
    values = list(probs.values()) + [answer.get('confidence')]
    if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('Invalid probability')
    if abs(sum(probs.values()) - 1) > .01:
        raise ValueError('Probabilities must sum to one')
    return answer
