"""Aggregate authoritative episodes and distinguish diagnostics from evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from coap.audit import audit_result, verify_source

TARGETS = {'fill_egg_holder': .247, 'fill_pen_holder': .24, 'fold_clothes': .90,
           'hang_mugs': .32, 'imitate_sorting_sequence': 0.0,
           'insert_key': .393, 'insert_tubes': .947,
           'store_tools_in_toolbox': .5, 'pour_by_language': .5}

SCORE_TARGETS = {'fill_egg_holder':.508,'fill_pen_holder':.49,'fold_clothes':.916,
                 'hang_mugs':.419,'imitate_sorting_sequence':.70,'insert_key':.484,'insert_tubes':.968,
                 'store_tools_in_toolbox':.5,'pour_by_language':.5}

CLASSIFICATION_TASKS = frozenset({'classify_objects', 'classify_objects_by_language'})
TARGETS.update({task: 0.0 for task in CLASSIFICATION_TASKS})
SCORE_TARGETS.update({'classify_objects': .70, 'classify_objects_by_language': .70})

MIXED_TASKS = ('fold_clothes', 'hang_mugs')
OPEN_TASKS = frozenset({'store_tools_in_toolbox', 'pour_by_language'})
DEVELOPMENT_PANEL_ID = 'seed0-ten-v1'
DEVELOPMENT_MIN_SR = {task: (.9 if task in ('fold_clothes','insert_tubes') else .7)
                      for task in TARGETS}
DEVELOPMENT_LAYOUTS = {task:list(range(5 if task in MIXED_TASKS else 10)) for task in TARGETS}
# These replacements were fixed from audited teacher exclusions before this panel ran.
DEVELOPMENT_LAYOUTS['imitate_sorting_sequence'] = [0,1,2,3,5,6,7,9,10,11]
DEVELOPMENT_MIN_SR.update({task: .5 for task in OPEN_TASKS})


def meets_target(task, success_rate, score):
    """Use the user's strict alternative thresholds for the two new tasks."""
    if task == 'imitate_sorting_sequence':
        return score > SCORE_TARGETS[task]
    if task in CLASSIFICATION_TASKS:
        return score > SCORE_TARGETS[task]
    if task in OPEN_TASKS:
        return success_rate > .5 or score > .5
    return (round(success_rate*100, 1) >= round(TARGETS[task]*100, 1)
            and round(score*100, 1) >= round(SCORE_TARGETS[task]*100, 1))


def canonical_task(task):
    """Map the official random task variant to its fixed policy program."""
    return task[:-7] if task in tuple(name+'_random' for name in MIXED_TASKS) else task


def task_layout_count(task):
    """Use the official per-variant evaluation counts."""
    return 25 if canonical_task(task) in MIXED_TASKS else 50


def protocol_identities(task, seeds=(0, 1, 2)):
    """Require both official variants for mixed generalization tasks."""
    if task in OPEN_TASKS:
        return {(task, 0, layout) for layout in range(10)}
    variants = (task, task+'_random') if task in MIXED_TASKS else (task,)
    return {(variant, seed, layout) for variant in variants
            for seed in seeds for layout in range(task_layout_count(variant))}


def development_identities(task):
    """Return a fixed ten-episode seed-zero panel, including both mixed variants."""
    task = canonical_task(task)
    variants = (task,task+'_random') if task in MIXED_TASKS else (task,)
    return {(variant,0,layout) for variant in variants for layout in DEVELOPMENT_LAYOUTS[task]}


def development_summary(group):
    """Require ten actual official outcomes before allowing fifty-layout expansion."""
    required = development_identities(group['task'])
    def identity(episode):
        return episode['evaluation_task'],episode['seed'],episode['layout_id']
    episodes = [episode for episode in group['episodes'] if identity(episode) in required]
    excluded = [episode for episode in group['excluded'] if identity(episode) in required]
    if not episodes and not excluded:
        return None
    observed = {identity(episode) for episode in episodes}
    complete = observed == required and len(episodes)==10 and not excluded
    sr = sum(episode['success'] for episode in episodes)/len(episodes) if episodes else None
    score = sum(episode['score'] for episode in episodes)/len(episodes) if episodes else None
    passed = complete and sr+1e-12>=DEVELOPMENT_MIN_SR[group['task']] and score+1e-12>=SCORE_TARGETS[group['task']]
    if group['task'] in OPEN_TASKS:
        passed = complete and meets_target(group['task'], sr, score)
    return dict(version=group['version'],task=group['task'],numerical_ik=group['numerical_ik'],
                panel_id=DEVELOPMENT_PANEL_ID,seeds=[0],episodes=episodes,excluded=excluded,
                required_identities=sorted(required),valid_episodes=len(episodes),
                success_rate=sr,score=score,min_success_rate=DEVELOPMENT_MIN_SR[group['task']],
                min_score=SCORE_TARGETS[group['task']],panel_complete=complete,
                panel_passed=bool(passed),allow_full_evaluation=bool(passed))


def seed_zero_summary(group):
    """Evaluate the user's selected seed without mixing other seed outcomes."""
    episodes = [episode for episode in group['episodes'] if episode['seed'] == 0]
    if not episodes:
        return None
    excluded = [episode for episode in group['excluded'] if episode['seed'] == 0]
    identities = {(episode['evaluation_task'], episode['seed'], episode['layout_id'])
                  for episode in episodes+excluded}
    sr = sum(episode['success'] for episode in episodes)/len(episodes)
    score = sum(episode['score'] for episode in episodes)/len(episodes)
    complete = identities == protocol_identities(group['task'], seeds=(0,))
    observed_met = meets_target(group['task'], sr, score)
    return dict(version=group['version'], task=group['task'],
                numerical_ik=group['numerical_ik'], seeds=[0], episodes=episodes,
                excluded=excluded, valid_episodes=len(episodes),
                attempted_layouts=len(identities), success_rate=sr, score=score,
                target_success_rate=TARGETS[group['task']],
                target_score=SCORE_TARGETS[group['task']],
                observed_target_met=observed_met, evaluation_protocol_complete=complete,
                evaluation_target_met=complete and observed_met)


def report(root):
    grouped = {}
    jobs = []
    verified_sources = {}
    for run in sorted((Path(root) / 'runs').glob('*')):
        path = run / 'manifest.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        status_path = run / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'phase': 'queued'}
        row = {'job_id': run.name, 'task': manifest['task'], 'version': manifest['version'],
               'seed': manifest['seed'], 'layouts': manifest['layouts'], 'probe_only': manifest['probe_only'],
               'phase': status['phase'], 'error': status.get('error')}
        if manifest.get('diagnostic_mode'):
            row['diagnostic_mode'] = manifest['diagnostic_mode']
        jobs.append(row)
        if manifest['probe_only'] or manifest.get('diagnostic_mode') or status['phase'] != 'complete':
            continue
        source_key = (manifest['version'],manifest['code_digest'])
        if source_key not in verified_sources:
            verified_sources[source_key] = verify_source(Path(root)/'versions'/manifest['version'],manifest['code_digest'])
        official = Path(status['official_result'])
        if hashlib.sha256(official.read_bytes()).hexdigest() != status['result_digest']:
            raise ValueError('Official result changed: ' + str(official))
        result = json.loads(official.read_text())
        if result != status['result']:
            raise ValueError('Stored result differs from official result')
        audit = audit_result(manifest, result, (run / 'execution.log').read_text())
        row.update(audit)
        details = result['details']
        key = (manifest['version'], canonical_task(manifest['task']), bool(manifest.get('numerical_ik', False)))
        group = grouped.setdefault(key, {'version':key[0], 'task':key[1], 'numerical_ik':key[2], 'episodes':[], 'excluded':[]})
        group['excluded'].extend({'evaluation_task':manifest['task'], 'seed': manifest['seed'], 'layout_id': layout, 'job_id': run.name, 'reason': audit['exclusion_reason']} for layout in audit['excluded_layouts'])
        for episode in details.values():
            score = episode['score']
            if type(episode['success']) is not bool or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid official score')
            group['episodes'].append(dict(episode, evaluation_task=manifest['task'], seed=manifest['seed'], job_id=run.name))
    summaries = []
    for group in grouped.values():
        episodes = group['episodes']
        if not episodes:
            continue
        identities = {(e['evaluation_task'], e['seed'], e['layout_id']) for e in episodes}
        if len(identities) != len(episodes):
            raise ValueError('Duplicate official layout: ' + str(group))
        sr = sum(e['success'] for e in episodes) / len(episodes)
        audited_identities = identities | {(e['evaluation_task'],e['seed'],e['layout_id']) for e in group['excluded']}
        protocol_complete = audited_identities == protocol_identities(group['task'])
        target_met = meets_target(group['task'], sr, sum(e['score'] for e in episodes)/len(episodes))
        group.update(valid_episodes=len(episodes), success_rate=sr, score=sum(e['score'] for e in episodes)/len(episodes),
                     attempted_layouts=len(audited_identities),
                     target_success_rate=TARGETS[group['task']], target_score=SCORE_TARGETS[group['task']],
                     observed_target_met=target_met,
                     paper_target_met=protocol_complete and target_met and group['task'] not in OPEN_TASKS,
                     paper_protocol_complete=protocol_complete and group['task'] not in OPEN_TASKS)
        summaries.append(group)
    seed_zero_summaries = [summary for group in grouped.values()
                          if (summary := seed_zero_summary(group)) is not None]
    development_summaries = [summary for group in grouped.values()
                             if (summary := development_summary(group)) is not None]
    return {'targets':TARGETS, 'jobs':jobs, 'summaries':summaries,
            'evaluation_seeds':[0], 'evaluation_summaries':seed_zero_summaries,
            'development_panel_id':DEVELOPMENT_PANEL_ID,
            'development_summaries':development_summaries,
            'paper_reference':'https://arxiv.org/html/2610.12369v1#A4',
            'scope':'Diagnostics excluded; development success does not establish paper-protocol replication.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    args=parser.parse_args()
    print(json.dumps(report(args.root),indent=2))
