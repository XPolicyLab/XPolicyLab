"""Finite three-seed campaign; the official SDK selects and scores every case."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from .policy_catalog import catalog, runtime_settings
from .deployment import locations

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def worker_lock(folder, node):
    # Only the assigned physical node may schedule this partition. No shared
    # JSON read/modify/rename transaction is used between NFS clients.
    path = folder / ('worker-' + node + '.lock')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(folder, slots):
    if (folder / 'plan.json').exists():
        raise FileExistsError('A campaign plan is immutable; choose a fresh campaign directory')
    from rsi.skills.git_versioning import require_clean
    source = require_clean(ROOT)
    parsed = []
    for value in slots:
        seed, job, node = value.split(':')
        parsed.append(dict(seed=int(seed), job=job, node=node))
    counts = {s: sum(r['seed'] == s for r in parsed) for s in range(3)}
    if len(parsed) not in (9, 12) or any(counts[s] != len(parsed) // 3 for s in range(3)):
        raise ValueError('Supply three or four distinct nodes per eval seed 0/1/2')
    if len({r['node'] for r in parsed}) != len(parsed):
        raise ValueError('Seed groups must use distinct physical nodes')
    common = runtime_settings()
    units = {}
    for seed in range(3):
        for mode, declaration in catalog().items():
            backend = declaration['backend']
            checkpoint = '312000' if backend == 'learned' else 'rule'
            action = 'ee' if backend == 'learned' else 'joint'
            key = 's%d-%s' % (seed, mode)
            run_id = folder.name + '-' + key
            result = (folder / 'runs' / key / 'official-client/eval_result/RoboDojo' / mode /
                      common['policy_name'] / 'arx_x5' /
                      ('%d_ckpt_name=%s,action_type=%s' % (seed, checkpoint, action)) / run_id / '_result.json')
            units[key] = dict(task=mode, seed=seed, backend=backend,
                              expected=declaration.get('official_eval_nums', declaration.get('official_episodes')),
                              checkpoint=checkpoint, action_type=action, run_id=run_id,
                              official_result=str(result))
    if len(units) != 162 or sum(r['expected'] for r in units.values()) != 6300:
        raise ValueError('The official 54-mode / 2100-per-seed catalog changed')
    for seed in range(3):
        owners = [row for row in parsed if row['seed'] == seed]
        for backend in ('rule', 'learned'):
            identities = sorted(key for key, row in units.items()
                                if row['seed'] == seed and row['backend'] == backend)
            for index, key in enumerate(identities):
                units[key]['assigned_node'] = owners[index % len(owners)]['node']
    plan = dict(schema='riemann-official54-three-seeds-v2', created_at=now(),
                source_commit=source['commit'], source_tree=str(ROOT), slots=parsed,
                runtime=common, tasks=units, expected=6300,
                case_selection='unmodified official SeedManager and default eval_num',
                scorer='unmodified official run_eval/finalcheck', retries=0,
                probe_results_imported=False,
                scheduling='fixed disjoint node partitions; one writer per node state')
    write(folder / 'plan.json', plan)
    for slot in parsed:
        write(folder / ('node-' + slot['node'] + '-state.json'),
              dict(node=slot['node'], seed=slot['seed'], source_commit=source['commit'],
                   updated_at=now(), worker_pid=None,
                   tasks={key: dict(status='pending') for key, row in units.items()
                          if row['assigned_node'] == slot['node']}))
    return dict(tasks=len(units), episodes=6300, source_commit=source['commit'])


def collect(folder):
    plan = read(folder / 'plan.json')
    states = {}
    for slot in plan['slots']:
        snapshot = read(folder / ('node-' + slot['node'] + '-state.json'))
        expected = {key for key, row in plan['tasks'].items()
                    if row['assigned_node'] == slot['node']}
        if (snapshot['node'] != slot['node'] or snapshot['seed'] != slot['seed']
                or snapshot['source_commit'] != plan['source_commit']
                or set(snapshot['tasks']) != expected):
            raise ValueError('Node state changed its fixed ownership or source')
        for key, state in snapshot['tasks'].items():
            if state['status'] in ('starting', 'running'):
                if (state.get('node') != slot['node']
                        or state.get('worker_pid') != snapshot['worker_pid']
                        or not isinstance(state.get('gpus'), list)
                        or len(state['gpus']) != (2 if plan['tasks'][key]['backend'] == 'learned' else 1)
                        or len(set(state['gpus'])) != len(state['gpus'])
                        or any(type(gpu) is not int or not 0 <= gpu < 8 for gpu in state['gpus'])):
                    raise ValueError('Active task lost its node/GPU ownership: ' + key)
        states.update(snapshot['tasks'])
    rows = []
    for key, unit in plan['tasks'].items():
        state = states[key]
        row = dict(unit, identity=key, source_commit=plan['source_commit'],
                   status=state['status'], node=state.get('node'), gpus=state.get('gpus'),
                   done=0, successes=0, failures=0, score=None, complete=False,
                   error=state.get('error'))
        if state.get('node') and state['status'] in ('starting', 'running'):
            exit_path = folder / ('worker-' + state['node'] + '-exit.json')
            if exit_path.is_file() and read(exit_path).get('pid') == state.get('worker_pid'):
                row.update(status='needs_review', error='Owned worker exited before reaping this task; inspect its retained process evidence')
        result = Path(unit['official_result'])
        if result.is_file():
            try:
                raw = read(result)
                details = list(raw['details'].values())
                count = raw['eval_time']
                if type(count) is not int or len(details) != count or not 0 <= count <= unit['expected']:
                    raise ValueError('Official detail count differs from its summary')
                if len({d['layout_id'] for d in details}) != count:
                    raise ValueError('An official layout was scored more than once')
                if any(type(d['success']) is not bool for d in details):
                    raise ValueError('Use official success booleans, not a reward threshold')
                wins = sum(d['success'] for d in details)
                if count and abs(raw['success_rate'] - wins / count) > 1e-8:
                    raise ValueError('Official success summary and details disagree')
                if count and (not isinstance(raw['score'], (int, float)) or not math.isfinite(raw['score'])):
                    raise ValueError('The official mean reward must be finite')
                row.update(done=count, successes=wins, failures=count-wins,
                           score=raw['score'] if count else None,
                           official_details=details,
                           complete=count == unit['expected'] and state['status'] == 'finished')
            except (ValueError, TypeError, KeyError, OSError) as error:
                row['read_error'] = str(error)
        if state['status'] == 'finished' and not row['complete']:
            row.update(status='needs_review', error='Official client exited with an incomplete or invalid result')
        rows.append(row)
    graded = sum(row['done'] for row in rows)
    wins = sum(row['successes'] for row in rows)
    complete = all(row['complete'] for row in rows)
    report = dict(schema='riemann-official54-three-seeds-progress-v1', updated_at=now(),
                  source_commit=plan['source_commit'], expected=6300, tasks=rows,
                  done=graded, successes=wins, failures=graded-wins,
                  completed_tasks=sum(row['complete'] for row in rows), complete=complete,
                  provisional_success_rate=100. * wins / graded if graded else None,
                  provisional_mean_score=sum((r['score'] or 0) * r['done'] for r in rows) / graded if graded else None,
                  final_success_rate=100. * wins / 6300 if complete else None,
                  final_mean_score=sum((r['score'] or 0) * r['done'] for r in rows) / 6300 if complete else None,
                  no_probes_imported=True)
    write(folder / 'progress.json', report)
    return report


def run_task(folder, identity, gpus, cpus):
    plan = read(folder / 'plan.json')
    unit = plan['tasks'][identity]
    owners = [row for row in plan['slots'] if row['node'] == unit['assigned_node']]
    if (socket.gethostname() != unit['assigned_node'] or len(owners) != 1
            or os.environ.get('SLURM_JOB_ID') != owners[0]['job']):
        raise ValueError('Task may run only in its fixed physical-node allocation')
    from rsi.skills.git_versioning import require_clean
    if require_clean(ROOT)['commit'] != plan['source_commit']:
        raise ValueError('The immutable campaign source changed')
    if cpus:
        os.sched_setaffinity(0, {int(v) for v in cpus.split(',')})
    def terminate(signum, frame):
        raise KeyboardInterrupt('Campaign task received SIGTERM; clean up owned children')
    signal.signal(signal.SIGTERM, terminate)
    if unit['backend'] == 'learned':
        policy_python = locations()['policy_python']
    else:
        policy_python = plan['runtime']['python']
    chosen = [int(value) for value in gpus.split(',')]
    if len(chosen) != (2 if unit['backend'] == 'learned' else 1):
        raise ValueError('Wrong backend-specific GPU lease size')
    output = folder / 'runs' / identity
    from tools.official_launch import evaluate
    evaluate(['RoboDojo', unit['task'], unit['checkpoint'], 'arx_x5', unit['action_type'],
              str(unit['seed']), str(chosen[0]), str(chosen[-1]), policy_python,
              plan['runtime']['python']], output=output, run_id=unit['run_id'])
    result = read(unit['official_result'])
    if result['eval_time'] != unit['expected'] or len(result['details']) != unit['expected']:
        raise RuntimeError('Official client exited without the complete requested cohort; preserve partial results for review')


def stop_task_processes(processes, grace_seconds=90):
    from rsi.gpu_resources import stop_owned
    # Each task must reap two separately owned client/server groups. Give all
    # task parents the signal first and allow their nested cleanup to finish
    # before applying the shorter process-group fallback.
    for process in processes:
        if process.poll() is None:
            try:
                if os.getpgid(process.pid) != process.pid:
                    raise RuntimeError('Refusing to signal a task outside its owned session')
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            stop_owned(process)


def worker(folder, seed, wall_seconds=None):
    plan = read(folder / 'plan.json')
    node = socket.gethostname()
    slots = [r for r in plan['slots'] if r['seed'] == seed and r['node'] == node]
    if len(slots) != 1 or os.environ.get('SLURM_JOB_ID') != slots[0]['job']:
        raise ValueError('Worker must run inside its exact owned Slurm allocation')
    with worker_lock(folder, node):
        _worker(folder, plan, slots[0], wall_seconds)


def _worker(folder, plan, slot, wall_seconds):
    from rsi.gpu_resources import gpu_leases
    node, seed = slot['node'], slot['seed']
    path = folder / ('node-' + node + '-state.json')
    snapshot = read(path)
    expected = {key for key, row in plan['tasks'].items() if row['assigned_node'] == node}
    if (snapshot['source_commit'] != plan['source_commit'] or set(snapshot['tasks']) != expected
            or any(row['status'] != 'pending' for row in snapshot['tasks'].values())):
        raise ValueError('An existing partition is never implicitly restarted or retried')
    state = snapshot['tasks']
    snapshot['worker_pid'] = os.getpid()
    available = set(range(8))
    running = {}
    affinity = sorted(os.sched_getaffinity(0))
    deadline = None if wall_seconds is None else time.monotonic() + wall_seconds
    stop_reason = None
    last_write = 0.

    def persist():
        nonlocal last_write
        snapshot['updated_at'] = now()
        write(path, snapshot)
        last_write = time.monotonic()

    def request_stop(signum, frame):
        nonlocal stop_reason
        stop_reason = 'Owned controller received ' + signal.Signals(signum).name

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    write(folder / ('worker-' + node + '.json'), dict(node=node, pid=os.getpid(),
          job=slot['job'], seed=seed, started_at=now(), source_commit=plan['source_commit']))
    persist()
    try:
        while (deadline is None or time.monotonic() < deadline) and stop_reason is None:
            for key, (process, log, chosen) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                state[key].update(status='finished' if code == 0 else 'needs_review',
                                  returncode=code, finished_at=now(),
                                  error=None if code == 0 else 'Task process exited %d; see retained evidence' % code)
                available.update(chosen)
                del running[key]
                persist()
            healthy = []
            for gpu in sorted(available):
                if stop_reason is not None:
                    break
                try:
                    with gpu_leases([gpu]):
                        healthy.append(gpu)
                except (RuntimeError, subprocess.SubprocessError):
                    pass  # No lock or workload belonging to another owner is changed.
            pending = [key for key, value in state.items() if value['status'] == 'pending']
            local_wam = any(plan['tasks'][key]['backend'] == 'learned' for key in running)
            rules_waiting = any(plan['tasks'][key]['backend'] == 'rule' for key in pending)
            eligible = [key for key in pending if plan['tasks'][key]['backend'] == 'rule'
                        or (not local_wam or not rules_waiting) and len(healthy) >= 2]
            eligible.sort(key=lambda key: (plan['tasks'][key]['backend'] != 'learned', key))
            key = eligible[0] if eligible and healthy and stop_reason is None else None
            if key is not None:
                size = 2 if plan['tasks'][key]['backend'] == 'learned' else 1
                chosen = healthy[:size]
                state[key] = dict(status='starting', node=node, gpus=chosen,
                                  worker_pid=os.getpid(), started_at=now())
                persist()
                cpus = sorted({cpu for gpu in chosen for cpu in affinity[gpu::8]})
                log_path = folder / 'logs' / (key + '.log')
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('x')
                command = [sys.executable, '-B', str(ROOT / 'tools/run_evaluation_batch.py'),
                           'task', '--campaign', str(folder), '--identity', key,
                           '--gpus', ','.join(map(str, chosen)), '--cpus', ','.join(map(str, cpus))]
                env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='8',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running[key] = process, log, chosen
                available.difference_update(chosen)
                state[key].update(status='running', task_pid=process.pid, log=str(log_path))
                persist()
                continue
            if not pending and not running:
                break
            if time.monotonic() - last_write >= 10:
                persist()
            time.sleep(3)
        if any(row['status'] not in ('finished', 'needs_review') for row in state.values()) and stop_reason is None:
            stop_reason = 'Owned controller wall-time budget expired'
    except BaseException:
        stop_reason = traceback.format_exc()
        raise
    finally:
        stop_task_processes([process for process, log, chosen in running.values()])
        for key, (process, log, chosen) in running.items():
            log.close()
            state[key].update(status='finished' if process.returncode == 0 else 'needs_review',
                              returncode=process.returncode, finished_at=now(),
                              error=None if process.returncode == 0 else stop_reason)
        for row in state.values():
            if row['status'] in ('pending', 'starting', 'running'):
                row.update(status='needs_review', error=stop_reason or 'Owned controller exited before completion')
        persist()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write(folder / ('worker-' + node + '-exit.json'),
              dict(node=node, pid=os.getpid(), ended_at=now(), error=stop_reason))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'task', 'report', 'monitor'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--slots', nargs='*')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--identity')
    parser.add_argument('--gpus')
    parser.add_argument('--cpus', default='')
    parser.add_argument('--wall-seconds', type=int, help='Optional explicit operator time cap; default: no wall-time cap')
    args = parser.parse_args(argv)
    if args.wall_seconds is not None and args.wall_seconds <= 0:
        parser.error('--wall-seconds must be positive when explicitly supplied')
    if args.command == 'prepare':
        print(json.dumps(prepare(args.campaign.resolve(), args.slots or [])))
    elif args.command == 'worker':
        worker(args.campaign.resolve(), args.seed, args.wall_seconds)
    elif args.command == 'task':
        try:
            run_task(args.campaign.resolve(), args.identity, args.gpus, args.cpus)
        except BaseException:
            write(args.campaign / 'errors' / (args.identity + '.json'), dict(at=now(), traceback=traceback.format_exc()))
            raise
    elif args.command == 'monitor':
        deadline = None if args.wall_seconds is None else time.monotonic() + args.wall_seconds
        while deadline is None or time.monotonic() < deadline:
            report = collect(args.campaign.resolve())
            if all(row['status'] in ('finished', 'needs_review') for row in report['tasks']):
                break
            time.sleep(10)
    else:
        report = collect(args.campaign.resolve())
        print(json.dumps({k: v for k, v in report.items() if k != 'tasks'}))
