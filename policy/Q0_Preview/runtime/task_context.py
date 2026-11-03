"""Human-readable task clarification; no runtime reward/pose access."""
import json
from pathlib import Path
import re


def task_card(name):
    return json.loads((Path(__file__).parent.parent/'assets/tasks/task_cards.json').read_text())['cards'][name]


def task_step_limit(name, configured_limit=None):
    """Static task budget audited offline against the frozen benchmark sources."""
    try:
        limit = task_card(name)['control_step_limit']
    except (KeyError, TypeError) as exc:
        raise ValueError(f'No audited control-step limit for task {name!r}') from exc
    if type(limit) is not int or limit <= 0:
        raise ValueError(f'Invalid audited control-step limit for task {name!r}')
    if configured_limit is not None and (type(configured_limit) is not int or configured_limit != limit):
        raise ValueError(f'Configured max_control_steps={configured_limit!r} disagrees with '
                         f'task {name!r} limit {limit}; update the audited task metadata explicitly')
    return limit


def task_reference(name):
    """Return only this task's public definitions, never live reward or object state."""
    data = json.loads((Path(__file__).parent.parent/'assets/tasks/task_reference.json').read_text())
    if name not in data['cards']:
        raise ValueError(f'No audited task reference for {name}')
    card = data['cards'][name]
    if card['source_sha256'] != task_card(name)['source_sha256']:
        raise RuntimeError(f'Task brief/reference source mismatch: {name}')
    return {'task': name, 'notice': data['notice'],
            'completion_details': card['completion_details'], 'process_rules': card['process_rules'],
            'assessment_notes': card.get('assessment_notes', '')}


def task_context(name, benchmark_root=None, verify=False):
    """Static task cards; never read live task/reward source or state."""
    card = task_card(name)
    reference = task_reference(name)
    text = card['requirements'] + '\n' + card['finish']
    for title, key in (('Completion details', 'completion_details'),
                       ('Process rules', 'process_rules'), ('Evaluation notes', 'assessment_notes')):
        extra = []
        for sentence in re.split(r'(?<=[.!?])\s+', reference[key].strip()):
            # Omit inherited placeholders and facts already stated in the brief.
            if (sentence and not sentence.startswith('No additional ')
                    and sentence not in text and sentence not in extra):
                extra.append(sentence)
        if extra:
            text += '\n' + title + ': ' + ' '.join(extra)
    return text
