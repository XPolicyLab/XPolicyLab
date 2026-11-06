"""Validate an explicit continuation of the SDK's own cumulative result.

These receipts are evaluator-side evidence. No result/layout data enters the
policy configuration, and no completed success or failure may be replayed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import socket


def document(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Continuation needs an ordinary evidence file: ' + str(path))
    return json.loads(path.read_text())


def official_details(result):
    details = result.get('details')
    if not isinstance(details, dict) or result.get('eval_time') != len(details):
        raise ValueError('Official detail count differs from its summary')
    layouts = [d.get('layout_id') for d in details.values()]
    if (any(type(v) is not int or v < 0 for v in layouts)
            or len(set(layouts)) != len(layouts)):
        raise ValueError('Missing or repeated completed official layout')
    if any(type(d.get('success')) is not bool or type(d.get('score')) not in (int, float)
           or not math.isfinite(d['score']) or not 0 <= d['score'] <= 1 for d in details.values()):
        raise ValueError('Invalid official success/reward detail')
    if details:
        wins = sum(d['success'] for d in details.values())
        score = sum(d['score'] for d in details.values()) / len(details) * 100
        if (not math.isclose(result.get('success_rate', -1), wins / len(details), abs_tol=1e-9)
                or not math.isclose(result.get('score', -1), score, abs_tol=1e-6)):
            raise ValueError('Official aggregate and details disagree')
    return details


def inspect_resume(previous, identity, args, route, run_id):
    previous = Path(previous).resolve()
    for name in ('source-identity.json', 'source-identity-final.json'):
        if document(previous / name) != identity or identity.get('source_dirty') is not False:
            raise ValueError('Continue only the same clean, finished source invocation')
    lease = document(previous / 'lease.json')
    if lease.get('node') != socket.gethostname():
        raise ValueError('Check the former owner on its actual compute node')
    pid = lease.get('owner_pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('Missing former launcher identity')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('The former launcher is still live; do not resume it')
    deploy = document(previous / 'deploy.json')
    expected = dict(task_name=args.task, seed=int(args.seed), action_type=args.action,
                    env_cfg_type=args.robot, policy_name='Riemann-RSI-Preview', bench_name=args.bench)
    if any(deploy.get(key) != value for key, value in expected.items()):
        raise ValueError('Continuation differs from the original public deployment')
    results = previous / 'official-client/eval_result'
    parent = (results / args.bench / args.task / 'Riemann-RSI-Preview' / args.robot /
              (args.seed + '_ckpt_name=' + args.checkpoint + ',action_type=' + args.action))
    manifest_path = parent / ('_resume_' + run_id + '.json')
    result_path = parent / run_id / '_result.json'
    manifest, result = document(manifest_path), document(result_path)
    fields = dict(run_id=run_id, task_name=args.task, policy_name='Riemann-RSI-Preview',
                  config_name=args.robot, eval_seed=int(args.seed),
                  additional_info='ckpt_name=' + args.checkpoint + ',action_type=' + args.action,
                  save_dir=str(result_path.parent.relative_to(previous / 'official-client')))
    if any(manifest.get(key) != value for key, value in fields.items()):
        raise ValueError('SDK checkpoint identity differs from the original invocation')
    details = official_details(result)
    count = len(details)
    expected_count = route.get('official_eval_nums', route.get('official_episodes'))
    if type(expected_count) is not int or not 0 < count < expected_count:
        raise ValueError('Resume requires an incomplete official cohort with scored evidence')
    completed = sorted(d['layout_id'] for d in details.values())
    if (manifest.get('details') != details or manifest.get('completed_layout_ids') != completed
            or manifest.get('success_nums') != sum(d['success'] for d in details.values())
            or manifest.get('fail_nums') != sum(not d['success'] for d in details.values())
            or not math.isclose(manifest.get('total_score', -1),
                                sum(d['score'] for d in details.values()), abs_tol=1e-8)):
        raise ValueError('SDK checkpoint and scored official result disagree')
    if set(manifest.get('abandoned_layout_ids', [])) & set(completed):
        raise ValueError('SDK checkpoint abandons a completed layout')
    return dict(deploy=deploy, result_root=str(results), official_result=str(result_path),
                manifest_path=str(manifest_path), sdk_manifest=manifest, official_prefix=result,
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                source_identity=identity, expected_episodes=expected_count,
                policy_server_restarted=True, rng_stream_continuity_claimed=False)


def verify_extension(before, result):
    """The SDK may add details; it must preserve every previously scored case."""
    old, new = official_details(before), official_details(result)
    if any(new.get(key) != value for key, value in old.items()):
        raise ValueError('A continuation changed a previously scored success or failure')
    return len(new)
