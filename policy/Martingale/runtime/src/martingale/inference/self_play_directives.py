"""Explicit, value-free practice objectives and evidence-backed candidate reports."""
import hashlib
import json
from pathlib import Path

from .control import atomic_json


def load_directives(path):
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if len(raw) > 128 * 1024:
        raise ValueError('Self-play directives exceed 128 KiB')
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {'task', 'objective', 'targets'}:
        raise ValueError('Directives require task, objective and targets only')
    if any(not isinstance(data[k], str) or not data[k].strip() for k in ('task', 'objective')):
        raise ValueError('Directive task and objective must be nonempty text')
    if not isinstance(data['targets'], list) or not data['targets']:
        raise ValueError('Provide at least one target')
    seen = set()
    for target in data['targets']:
        if not isinstance(target, dict) or set(target) != {'id', 'description'}:
            raise ValueError('Each target requires only id and description')
        if any(not isinstance(v, str) or not v.strip() for v in target.values()):
            raise ValueError('Target fields must be nonempty text')
        if target['id'] in seen:
            raise ValueError('Duplicate directive target')
        seen.add(target['id'])
    return data


def prompt(data):
    return '''Self-play directives: prioritize these investigations during this episode.
Use only permitted observations, robot proprioception and observed action outcomes.
Robot geometry may come from URDF. Derive object geometry, scene positions and
control settings from this episode; do not invent values.
These directives do not change observation boundaries or episode termination.
For each target, record a candidate as soon as there is evidence, before risky
actions or task completion. Reobserve and revise it when needed. Report every
component of a vector, mapping or sequence, with its units and coordinate frame.
Action settings require an observed trial; an IK preview alone is insufficient.
Distinguish scene-specific estimates from reusable settings. An unsuccessful or
interrupted episode is not evidence of impossibility. Leave unmeasured targets
unresolved with the attempted approaches and blockers; never fill in guesses.

robo.directive_progress() returns all target statuses.
robo.directive_record(target_id, status='estimated' or 'unresolved', value=...,
    units='...', frame='...', uncertainty='...', method='...',
    evidence_cells=[completed_cell_numbers], limitations='...')

For unresolved targets omit value and explain attempts and blockers in method
and limitations. For estimated targets all fields are required, including units
and frame (use 'categorical' / 'not applicable' for an ordering). uncertainty must
explain empirical bounds, variation, or insufficient repeat trials. Completed
evidence cells must contain the actual observations, calculations and trials.
The host retains those cell records; these are candidates for later review, not
automatic source-code replacements or proof of accuracy. No constants are deleted.

''' + json.dumps(data, indent=2)


class DirectiveReports:
    def __init__(self, data, directory):
        self.data = data
        self.directory = Path(directory)
        self.records = {t['id']: {'status': 'unresolved', 'reason': 'not investigated'}
                        for t in data['targets']}
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self.directory / 'directives.json', data)
        self.save()

    def progress(self):
        return self.records.copy()

    def save(self):
        atomic_json(self.directory / 'estimates.json', {
            'schema': 'martingale-self-play-estimates-v1', 'task': self.data['task'],
            'source': '--self-play', 'review_status': 'recorded',
            'directives_sha256': hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest(),
            'targets': self.records})

    def record(self, target_id, *, status, evidence, method, limitations,
               value=None, units=None, frame=None, uncertainty=None):
        if target_id not in self.records or status not in ('estimated', 'unresolved'):
            raise ValueError('Unknown target or invalid status')
        if not evidence or not isinstance(method, str) or not method.strip() or not isinstance(limitations, str) or not limitations.strip():
            raise ValueError('Supply observation evidence, method and limitations')
        if status == 'estimated' and (value is None or any(
                not isinstance(v, str) or not v.strip() for v in (units, frame, uncertainty))):
            raise ValueError('An estimate needs value, units, frame and uncertainty')
        if status == 'unresolved' and value is not None:
            raise ValueError('Unresolved targets must not contain an estimate')
        record = dict(status=status, value=value, units=units, frame=frame,
                      uncertainty=uncertainty, method=method, limitations=limitations, evidence=evidence)
        encoded = json.dumps(record, allow_nan=False)
        if len(encoded) > 1024 * 1024:
            raise ValueError('Directive report exceeds 1 MiB')
        self.records[target_id] = json.loads(encoded)
        self.save()
        return {'target_id': target_id, 'status': status, 'review_status': 'recorded'}
