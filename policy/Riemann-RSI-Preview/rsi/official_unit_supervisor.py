"""Supervise registered in-flight official units without changing their actors.

This is an evaluator-side bridge for already-running older launchers. Recovery
uses inspect_resume and the same official run namespace, never a fresh cohort.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .official_runtime_monitor import RuntimeMonitor, OfficialRuntimeFault
from .official_resume import official_details, verify_extension, inspect_resume


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def client_transcript_path(output):
    paths = list(Path(output).glob('official-client-attempt-*.log'))
    return max(paths, key=lambda p: int(p.stem.rsplit('-', 1)[1])) if paths else Path(output)/'official-client-attempt-1.log'


def proc(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, uid=path.stat().st_uid, state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), start=fields[19], exit_code=int(fields[49]),
                cmdline=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())


class ProcessView:
    def __init__(self, row):
        self.pid, self.start = row['pid'], row['start']

    def poll(self):
        row = proc(self.pid)
        if row is None or row['start'] != self.start:
            return 1
        if row['state'] == 'Z':
            return os.waitstatus_to_exitcode(row['exit_code'])
        return None


def owned_peers(spec, unit_path=None):
    output = Path(spec['output'])
    lease = read(output / 'lease.json')
    owner = proc(lease['owner_pid'])
    if owner is None or owner['state'] == 'Z':
        return None
    if (owner['uid'] != os.getuid() or owner['pgid'] != owner['pid']
            or str(output.parent.parent) not in owner['cmdline']):
        raise ValueError('Unverified unit owner; no signal permitted')
    if unit_path is not None and str(Path(unit_path).resolve()) not in owner['cmdline'].split():
        raise ValueError('Owner command differs from this exact unit invocation')
    for descriptor in lease['held_descriptors']:
        target = os.readlink(Path('/proc') / str(owner['pid']) / 'fd' / str(descriptor))
        if '/robodojo-shared-gpu-locks/' not in target or not target.endswith('.lock'):
            raise ValueError('Standard lease descriptor identity changed')
    result = subprocess.run(['pgrep', '-P', str(owner['pid'])], text=True, capture_output=True)
    peers = {}
    for pid in result.stdout.split():
        row = proc(int(pid))
        if row is None or row['uid'] != os.getuid():
            continue
        text = row['cmdline']
        if 'serve_official_policy.py' in text and '--mode ' + spec['mode'] in text:
            peers['server'] = row
        elif 'run_official_eval.py' in text and '--task_name ' + spec['mode'] in text:
            peers['client'] = row
    # A zombie has an empty cmdline. Retain the previously verified peer
    # identity so an in-flight server SIGSEGV cannot disappear from monitoring.
    proof_path = output / 'supervisor-owner-proof.json'
    if set(peers) != {'server', 'client'} and proof_path.is_file():
        previous = read(proof_path)
        if previous['owner']['pid'] == owner['pid'] and previous['owner']['start'] == owner['start']:
            for role in ('server', 'client'):
                if role in peers:
                    continue
                prior = previous[role]
                current = proc(prior['pid'])
                if current is None:
                    peers[role] = prior  # ProcessView reports that exact peer absent.
                elif current['start'] == prior['start'] and current['uid'] == os.getuid():
                    peers[role] = current
    if set(peers) != {'server', 'client'}:
        return None
    return dict(owner=owner, **peers)


def recover(registry_path, unit_path, spec, fault, launcher):
    attempt = spec.get('infrastructure_recovery_attempt', 0) + 1
    folder = Path(unit_path).parent / ('runtime-recovery-' + str(attempt))
    if folder.exists():
        return
    result = read(spec['official_result']) if Path(spec['official_result']).is_file() else None
    rows = official_details(result) if result is not None else {}
    output = Path(spec['output'])
    lease = read(output / 'lease.json')
    owner = proc(lease['owner_pid'])
    if owner is not None and owner['state'] != 'Z':
        proof = owned_peers(spec, unit_path)
        if proof is None:
            return
        # A newer launcher detects these faults itself, then reaps its owned
        # groups in finally while retaining the leases. A second SIGTERM could
        # interrupt that cleanup. Older launchers still need one controlled stop.
        embedded_path = output / 'runtime-health.json'
        embedded = read(embedded_path) if embedded_path.is_file() else {}
        self_cleanup = (embedded.get('status') == 'fault' and embedded.get('fault') == fault
                        and embedded.get('client_pid') == proof['client']['pid']
                        and embedded.get('server_pid') == proof['server']['pid'])
        write(output / 'supervisor-stop-proof.json', dict(fault=fault, peers=proof,
                                                        launcher_cleanup_observed=self_cleanup))
        if not self_cleanup:
            os.kill(owner['pid'], signal.SIGTERM)  # The handler reaps its groups with leases held.
    deadline = time.monotonic() + 90
    while proc(lease['owner_pid']) is not None and time.monotonic() < deadline:
        time.sleep(.5)
    if proc(lease['owner_pid']) is not None:
        return
    if not 0 < len(rows) < spec['expected'] or attempt > 2:
        return  # Stop a proven broken owner, but never replay zero/full/unknown cases.
    verify_extension(result, read(spec['official_result']))
    from rsi.skills.git_versioning import runtime_source_identity, require_clean
    from tools.official_launch import arguments
    from .official_source import read as source_catalog
    from .deployment import locations, slurm_srun
    evaluation_python = locations(launcher)['python']
    argv = ['RoboDojo', spec['mode'], spec['checkpoint'], 'arx_x5', spec['action_type'],
            str(spec['seed']), str(spec['gpus'][0]), str(spec['gpus'][-1]),
            spec['policy_python'], evaluation_python]
    checked = inspect_resume(output, runtime_source_identity(Path(spec['actor_source'])),
                             arguments(argv), source_catalog(spec['actor_source'], spec['mode'])['route'], spec['run_id'])
    folder.mkdir()
    (folder / 'official-prefix.json').write_text(json.dumps(checked['official_prefix'], indent=2) + '\n')
    (folder / 'sdk-prefix.json').write_text(json.dumps(checked['sdk_manifest'], indent=2) + '\n')
    new = dict(spec, kind='continuation', origin_kind=spec.get('origin_kind', spec['kind']),
               previous_run=str(output), output=str(output) + '-recovery-' + str(attempt),
               prefix=str(folder / 'official-prefix.json'), launcher_source=str(launcher),
               launcher_commit=require_clean(launcher)['commit'], infrastructure_recovery_attempt=attempt,
               previous_runtime_fault=fault)
    path = folder / 'unit.json'
    write(path, new)
    # Different nodes can recover simultaneously; registry writes share one lock.
    descriptor = os.open(registry_path.with_suffix('.lock'), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        registry = read(registry_path)
        if registry['units'].get(spec['identity']) != str(unit_path):
            raise ValueError('A newer registration already owns this unit')
        registry['units'][spec['identity']] = str(path)
        write(registry_path, registry)
    runner = registry_path.parent / 'run_unit.py'
    command = [slurm_srun(launcher), '--overlap', '--jobid=' + spec['job'],
               '-N1', '-n1', '-w', spec['node'], '--cpu-bind=none',
               '--cpus-per-task=' + str(32 * len(spec['gpus'])),
               evaluation_python, '-B', str(runner), str(path)]
    with (folder / 'run.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    write(folder / 'dispatch.json', dict(pid=child.pid, at=time.time(), explicit_fault=fault,
                                       scored_prefix_count=len(rows), actor_unchanged=True))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    launcher = Path(__file__).resolve().parents[1]
    monitors = {}
    while True:
        pending = False
        registry = read(args.registry)
        for identity, name in registry['units'].items():
            path = Path(name)
            spec = read(path)
            if spec['node'] != socket.gethostname():
                continue
            state_path = path.with_name('status.json')
            state = read(state_path) if state_path.is_file() else dict(status='pending')
            if state['status'] == 'complete':
                continue
            output = Path(spec['output'])
            health_path = output / 'runtime-health.json'
            fault = read(health_path).get('fault') if health_path.is_file() else None
            try:
                if state['status'] == 'fault':
                    if fault in ('policy_server_exited', 'client_cuda_context_failed', 'client_kit_hang'):
                        if recover(args.registry, path, spec, fault, launcher):
                            pending = True
                    continue
                pending = True
                peers = owned_peers(spec, path) if (output / 'lease.json').is_file() else None
                if peers is None:
                    continue
                key = (name, peers['client']['pid'])
                if key not in monitors:
                    client, server = ProcessView(peers['client']), ProcessView(peers['server'])
                    monitors[key] = (RuntimeMonitor(client, server, spec['official_result'],
                                                   output / 'runtime-supervisor-health.json'), 0)
                    write(output / 'supervisor-owner-proof.json', peers)
                monitor, offset = monitors[key]
                transcript = client_transcript_path(output)
                if transcript.is_file():
                    with transcript.open('rb') as log:
                        log.seek(offset)
                        text = log.read(1024 * 1024)
                        monitors[key] = monitor, log.tell()
                    monitor.feed(text.decode(errors='replace'))
                monitor.check()
            except OfficialRuntimeFault as error:
                monitor.write(error)
                recover(args.registry, path, spec, error.kind, launcher)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                write(path.with_name('supervisor-review.json'), dict(at=time.time(), error=repr(error),
                                                                 automatic_score_or_unknown_retry=False))
        if not pending:
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
