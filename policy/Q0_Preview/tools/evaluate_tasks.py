"""Optional local task admission: one official eval.sh process per slot, no retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from runtime import POLICY, resolve_output_path, save
from processes import stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True, help='Comma-separated task variants; each is launched once')
    p.add_argument('--slots', default='0:0', help='Comma-separated policy_gpu:env_gpu pairs')
    p.add_argument('--policy-env', required=True)
    p.add_argument('--eval-env', required=True)
    p.add_argument('--output', required=True,
                   help='New directory for scheduling logs; relative paths use the XPolicyLab root')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    names = a.tasks.split(',')
    known = json.loads((POLICY/'assets/tasks/task_cards.json').read_text())['cards']
    if len(set(names)) != len(names) or any(n not in known for n in names):
        p.error('Provide unique supported task variant names')
    slots = [s.split(':') for s in a.slots.split(',')]
    if any(len(s) != 2 or any(not x.isdecimal() for x in s) for s in slots):
        p.error('Slots must be policy_gpu:env_gpu pairs')
    # Admit longer tasks first; the official client owns trials, layouts and scores.
    names.sort(key=lambda n: known[n]['control_step_limit'], reverse=True)
    def command(name, slot):
        return ['bash', str(POLICY/'eval.sh'), 'RoboDojo', name, 'api', 'arx_x5', 'joint',
                str(a.seed), *slot, a.policy_env, a.eval_env]
    root = resolve_output_path(a.output)
    if a.dry_run:
        print(json.dumps({'tasks': names, 'slots': slots, 'output_directory': str(root),
                          'first_command': command(names[0], slots[0])}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    running = {}
    results = []
    cancelled = False
    def cancel(_sig, _frame):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        while names or running:
            if cancelled:
                break
            for i, slot in enumerate(slots):
                if i in running or not names:
                    continue
                task = names.pop(0)
                folder = root/task
                folder.mkdir()
                env = os.environ.copy()
                env['Q0_PREVIEW_RUN_DIR'] = str(folder/'policy')
                cmd = command(task, slot)
                save(folder/'launch.json', {'task': task, 'slot': slot, 'command': cmd})
                log = (folder/'evaluation.log').open('w')
                try:
                    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running[i] = (task, child, log)
            for i, (task, child, log) in list(running.items()):
                rc = child.poll()
                if rc is not None:
                    log.close()
                    result = {'task': task, 'process_exit_code': rc, 'finished_unix_s': time.time()}
                    save(root/task/'process_result.json', result)
                    results.append(result)
                    del running[i]
            time.sleep(.2)
    finally:
        for task, child, log in running.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for task, child, log in running.values():
            try:
                stop(child, grace_seconds=60)
            finally:
                log.close()
        save(root/'schedule.json', {'cancelled': cancelled, 'unlaunched': names,
             'interrupted': [t for t, _, _ in running.values()], 'finished': results,
             'note': 'Process status only. Official evaluator records determine validity and scores.'})
    return 130 if cancelled else int(any(r['process_exit_code'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
