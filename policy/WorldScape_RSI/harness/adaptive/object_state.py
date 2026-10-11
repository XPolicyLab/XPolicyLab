"""Episode-local, entity-scoped evidence. Unknown and stale never mean false."""
from dataclasses import dataclass
import re

ALLOWED = {'held', 'supported', 'aligned', 'seated', 'stable', 'visible',
           'path_clear', 'release_allowed', 'goal_met', 'role_valid', 'hand_free'}
ID = re.compile(r'^[A-Za-z0-9_.-]{1,80}$')

def predicate(relation, subject, target):
    if relation not in ALLOWED or not ID.fullmatch(subject) or not ID.fullmatch(target):
        raise ValueError('Invalid scoped predicate')
    if relation == 'held' and subject not in ('left', 'right'):
        raise ValueError('held requires an explicit hand')
    return '{}({},{})'.format(relation, subject, target)

@dataclass
class Fact:
    value: object
    step: int
    revision: int
    expires: int
    evidence: str
    source: str
    entities: tuple

class ObjectState:
    def __init__(self):
        self.facts = {}
        self.events = []

    def put(self, relation, subject, target, value, step, revision, evidence,
            source='astra_rgb', ttl=24):
        key = predicate(relation, subject, target)
        if type(value) not in (bool, type(None)) or not evidence or not 0 <= ttl <= 120:
            raise ValueError('Invalid evidence')
        if source not in ('astra_rgb', 'registered_perception'):
            raise ValueError('Object relationships require perception, not command feedback')
        if key in self.facts and revision < self.facts[key].revision:
            raise ValueError('Out-of-order evidence')
        self.facts[key] = Fact(value, step, revision, step+ttl, evidence, source, (subject,target))
        self.events.append({'kind':'fact', 'key':key, **vars(self.facts[key])})
        return key

    def get(self, key, step):
        f = self.facts.get(key)
        return f.value if f and step <= f.expires else None

    def invalidate(self, entities, reason):
        for key, f in self.facts.items():
            if not entities or set(entities).intersection(f.entities):
                f.value = None
        self.events.append({'kind':'invalidate','entities':list(entities),'reason':reason})

    def snapshot(self, step):
        return {k:{**vars(f), 'value':self.get(k,step)} for k,f in self.facts.items()}
