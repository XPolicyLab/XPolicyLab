"""Frozen libraries + candidate experience; promotion is a separate reviewed operation."""
import json
import os
import re
import uuid
from pathlib import Path
from .contracts import digest, validate_plan, EXECUTORS

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_new(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: no overwriting a past candidate or evidence record.
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

class Library:
    def __init__(self, root, available, experimental=()):
        self.root = Path(root)
        self.skills = read(self.root/'skills.json')
        self.programs = read(self.root/'programs.json') if (self.root/'programs.json').exists() else []
        self.task_programs = read(self.root/'task_programs.json') if (self.root/'task_programs.json').exists() else {}
        self.plans = read(self.root/'plans.json')
        self.available = available
        self.experimental = tuple(experimental)
        for plan in self.plans:
            validate_plan(plan, self.skills, EXECUTORS)
        self.hash = digest({'plans':self.plans, 'skills':self.skills,'programs':self.programs,'task_programs':self.task_programs})

    def catalog(self, task=None, goal='', limit=12):
        from .code_skills import enabled
        words=set(re.findall(r'[a-z]+',goal.lower()))
        rows=[(name,s) for name,s in self.skills.items() if enabled(s,self.experimental)]
        def score(row):
            name,s=row
            return (len(words & set(re.findall(r'[a-z]+',s.get('description','').lower())))+
                    (2 if s.get('source_task')==task else 0))
        rows.sort(key=lambda row:(-score(row),row[0]))
        return dict(rows[:limit])

    def executable_candidates(self, stage, evidence):
        from .code_skills import eligible
        if 'cap' not in self.available: return {}
        return {name:skill for name,skill in self.skills.items()
                if eligible(skill,stage,evidence,self.experimental)}

    def candidates(self, evidence):
        # Unknown prerequisites are not treated as true. Keep them for explicit verification.
        return [p for p in self.plans if all(s['executor'] in self.available for s in p['stages'])
                and not any(evidence.facts.get(k) is False for k in p['preconditions'])]

    def recoveries(self, stage, evidence):
        return {name:self.skills[name] for name in stage['recoveries']
                if self.skills[name].get('validated')
                and evidence.satisfies(self.skills[name]['preconditions'])}

def save_candidate(root, run_id, plan, events_path, outcome):
    ident = uuid.uuid4().hex
    obj = {'id':ident, 'status':'candidate', 'source_run':run_id, 'plan':plan,
           'events':str(events_path), 'outcome':outcome, 'validated':False}
    atomic_new(Path(root)/(ident+'.json'), obj)
    return ident

def promote(candidate_path, validation_path, destination):
    candidate, validation = read(candidate_path), read(validation_path)
    if validation.get('candidate_sha256') != digest(candidate):
        raise ValueError('Validation is for another candidate')
    trials = validation.get('trials', [])
    heldout = [t for t in trials if t.get('source_run') != candidate['source_run']
               and t.get('split') == 'heldout' and t.get('native_success') is True
               and t.get('within_budget') is True and t.get('feedback_mode') == 'no_feedback']
    if not heldout or validation.get('regression_passed') is not True:
        raise ValueError('Independent validation and regression are required')
    # Every cited result must exist and agree, not merely a hand-entered success flag.
    for trial in trials:
        actual = read(trial['result_path'])
        if actual.get('success') != trial.get('native_success'):
            raise ValueError('Result mismatch')
        if actual.get('steps_used', float('inf')) > actual.get('native_step_limit', -1):
            raise ValueError('Invalid native budget')
        if actual.get('evaluation_track') != trial.get('feedback_mode'):
            raise ValueError('Feedback provenance mismatch')
    atomic_new(destination, {**candidate, 'status':'validated', 'validated':True,
                             'validation':validation, 'activation':'next_episode'})
