"""Portable all-task, native-count, seeds 0/1/2 evaluation; preview by default."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

POLICY = Path(__file__).resolve().parent
BENCH = POLICY.parents[2]
# No training checkout on PYTHONPATH, even when invoked from a different cwd.
sys.path.insert(0, str(BENCH))
from XPolicyLab.policy.Era0_Preview.checkpoint import resolve_assets
from XPolicyLab.policy.Era0_Preview.native_protocol import native_counts, default_tasks
from XPolicyLab.policy.Era0_Preview.sdk_compat import describe as execution_spec, validate as validate_sdk


def result_status(sdk_root, task, seed, expected, checkpoint_name, run_id):
    root = Path(sdk_root) / 'eval_result' / 'RoboDojo' / task / POLICY.name / 'arx_x5'
    run = root / f'{seed}_ckpt_name={checkpoint_name},action_type=ee' / run_id
    files = [run / '_result.json'] if (run / '_result.json').is_file() else []
    if len(files) != 1:
        return dict(actual=0, valid=False, reason=f'Expected one result file, found {len(files)}')
    try:
        payload = json.loads(files[0].read_text())
        details = payload['details']
        rows = list(details.values()) if isinstance(details, dict) else details
        valid = isinstance(rows, list) and all(isinstance(row, dict) and
            isinstance(row.get('success'), bool) and math.isfinite(float(row.get('score', 0))) for row in rows)
        actual = len(rows)
        complete = valid and type(payload['eval_time']) is int and payload['eval_time'] == expected == actual
        return dict(actual=actual, valid=complete, result=str(files[0]),
                    reason='' if complete else 'Incomplete or invalid trajectory results')
    except (KeyError, ValueError, TypeError) as error:
        return dict(actual=0, valid=False, reason=str(error))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from XPolicyLab.policy.Era0_Preview.obs_lag import observation_lag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Portable bundle root, model directory, or checkpoint name')
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for launch logs and protocol metadata; SDK result/video paths are unchanged')
    parser.add_argument('--checkpoint-name', default='era0-preview', help='Path-safe checkpoint label used in the SDK result directory')
    parser.add_argument('--policy-env', default=os.environ.get('POLICY_ENV', str(POLICY/'.venv')))
    parser.add_argument('--sim-env', default='RoboDojo', help='Conda env, venv directory, or Python executable')
    parser.add_argument('--robodojo-root', type=Path, default=Path(os.environ.get('ROBODOJO_ROOT', BENCH)))
    parser.add_argument('--gpu-ids', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--execution-profile', choices=['local','official'], default='local',
                        help='local: SDK execution with IK guard; official: unmodified SDK execution')
    parser.add_argument('--render-mode', choices=['legacy', 'sdk'], default=None)
    parser.add_argument('--obs-lag', type=observation_lag, default=observation_lag(),
                        help='Image-only delay in observation updates; default HY_OBS_LAG or 1')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--idle-timeout', type=int, default=600)
    parser.add_argument('--tasks-file', type=Path, help='Diagnostic subset; report is labelled partial')
    args = parser.parse_args()
    args.render_mode = args.render_mode or 'sdk'
    if args.obs_lag and args.render_mode == 'legacy':
        parser.error('--obs-lag > 0 requires --render-mode sdk')
    args.output = args.output.expanduser().resolve()
    args.robodojo_root = args.robodojo_root.expanduser().resolve()
    assets = resolve_assets(dict(ckpt_name=args.checkpoint))
    all_tasks = default_tasks(args.robodojo_root)
    if args.tasks_file:
        tasks = args.tasks_file.read_text().splitlines()
    else:
        tasks = all_tasks
    import re
    if not re.fullmatch('[A-Za-z0-9_.-]+', args.checkpoint_name) or args.checkpoint_name in {'.', '..'}:
        parser.error('--checkpoint-name must be a path-safe identifier')
    summarizer = args.robodojo_root / 'scripts/internal/summarize_result.py'
    if not summarizer.is_file(): parser.error('Selected SDK is missing scripts/internal/summarize_result.py')
    if not tasks or len(set(tasks)) != len(tasks) or not all(re.fullmatch('[A-Za-z0-9_]+', t) for t in tasks):
        parser.error('Task list must contain unique task identifiers')
    if not set(tasks).issubset(all_tasks):
        parser.error('Requested tasks must belong to the SDK default task inventory')
    counts = native_counts(args.robodojo_root, tasks)
    execution = validate_sdk(args.robodojo_root, args.execution_profile)
    gpus = args.gpu_ids.split(',')
    if not all(re.fullmatch('[0-9]+', gpu) for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU IDs must be distinct nonnegative integers')
    if args.idle_timeout <= 0: parser.error('--idle-timeout must be positive')
    def revision(root):
        result = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'

    plan = dict(model_name='Era0-Preview', policy=POLICY.name, checkpoint=str(assets['model']), seeds=[0,1,2],
        eval_num='native', task_counts=counts, expected_episodes=3*sum(counts.values()),
        full_suite=set(tasks)==set(all_tasks), sdk_default_tasks=all_tasks, task_inventory_sha256=fingerprint(args.robodojo_root/'scripts/internal/task_inventory.py'), num_envs=10, render_mode=args.render_mode,
        policy_image_lag_updates=args.obs_lag, physics='sdk_unchanged', execution=execution, norm_sha256=fingerprint(assets['norm']),
        config_sha256=fingerprint(assets['model']/'config.json'),
        sdk_revision=revision(args.robodojo_root), xpolicylab_revision=revision(POLICY.parents[1]),
        task_count_config_sha256=fingerprint(args.robodojo_root/'task/RoboDojo/config/_task.yml'),
        source_manifest_sha256=fingerprint(Path(__file__).parent/'source_manifest.json'),
        adapter_sha256=fingerprint(assets['model']/'hy_xvla_adapter.json'),
        qwen_history_time_rope=assets['config'].get('qwen_history_time_rope'),
        history=[8,1,1], history_interval=40, execute_slots=[1,10], denoise_steps=10,
        gpu_ids=gpus, checkpoint_name=args.checkpoint_name,
        result_root=str(args.robodojo_root/'eval_result/RoboDojo'),
        summary_script_sha256=fingerprint(summarizer),
        primary_metric='eval_result/RoboDojo/_summary.md Overview Avg Score')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run: return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'evaluation_contract.json').write_text(json.dumps(plan,indent=2)+'\n')
    run_id = time.strftime('%Y-%m-%d_%H-%M-%S')
    base_env = dict(os.environ, ROBODOJO_ROOT=str(args.robodojo_root),
        ERA0_CHECKPOINT_PATH=str(assets['model']), ROBODOJO_RUN_ID=run_id,
        ROBODOJO_NUM_ENVS='10', ROBODOJO_EVAL_LEGACY_RENDER=str(int(args.render_mode=='legacy')),
        HY_OBS_LAG=str(args.obs_lag), ERA0_EXECUTION_PROFILE=args.execution_profile, EVAL_NUM='native', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled')
    base_env.pop('ROBODOJO_EVAL_OUTPUT_DIR', None)
    base_env.pop('ROBODOJO_EVAL_ROOT', None)
    # A caller's local debug/subsample settings must not reduce the full benchmark.
    for key in ('ROBODOJO_SAMPLE_RATIO','ROBODOJO_FATAL_RESTART_COUNT','EVAL_ENV_TYPE',
                'ROBODOJO_DISABLE_CAMERAS','ROBODOJO_DISABLE_RENDER'):
        base_env.pop(key, None)
    base_env['PYTHONPATH']=f'{BENCH}:{BENCH}/XPolicyLab'
    # Record provenance once before parallel workers can write any results.
    subprocess.run([sys.executable,str(POLICY/'simulator_entry.py'),'--describe','--output',
                    str(args.output/'render_config.json')],env=base_env,check=True)
    cells = [(task,seed,n) for seed in (0,1,2) for task,n in counts.items()]
    queue = iter(cells)
    lock = threading.Lock()
    children = set()
    cancelled = threading.Event()

    def terminate(proc):
        if proc.poll() is not None: return
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()

    def interrupt(signum, frame):
        cancelled.set()
        with lock: active=list(children)
        for proc in active: terminate(proc)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def worker(gpu):
        rows=[]
        while not cancelled.is_set():
            with lock:
                cell=next(queue,None)
            if cell is None: break
            task,seed,n=cell
            log_path=args.output/'logs'/f'{task}_seed{seed}.log'
            log_path.parent.mkdir(exist_ok=True)
            env=dict(base_env)
            # Preserve the user's no-retry exception for unstable toast scenes.
            if task=='make_toast_random': env['ROBODOJO_FATAL_RESTART_COUNT']='3'
            command=['bash',str(POLICY/'eval.sh'),'RoboDojo',task,args.checkpoint_name,
                     'arx_x5','ee',str(seed),gpu,gpu,args.policy_env,args.sim_env]
            with log_path.open('w') as log:
                proc=subprocess.Popen(command,env=env,cwd=args.robodojo_root,stdout=log,
                                      stderr=subprocess.STDOUT,start_new_session=True)
                with lock: children.add(proc)
                last_size=0;last_activity=time.monotonic();completed_at=None;timeout=False;grace_terminated=False
                try:
                    while proc.poll() is None and not cancelled.wait(5):
                        size=log_path.stat().st_size
                        if size!=last_size: last_size=size;last_activity=time.monotonic()
                        status=result_status(args.robodojo_root,task,seed,n,args.checkpoint_name,run_id)
                        if status['valid']:
                            completed_at=completed_at or time.monotonic()
                            if time.monotonic()-completed_at>60: grace_terminated=True;terminate(proc);break
                        elif time.monotonic()-last_activity>args.idle_timeout:
                            timeout=True;terminate(proc);break
                finally:
                    terminate(proc)
                    with lock:children.discard(proc)
            status=result_status(args.robodojo_root,task,seed,n,args.checkpoint_name,run_id)
            if proc.returncode != 0 and not grace_terminated:
                status['valid']=False
                status['reason']=f'Launcher exited {proc.returncode}: '+status.get('reason','')
            status.update(task=task,seed=seed,expected=n,exit_code=proc.returncode,timeout=timeout,log=str(log_path))
            rows.append(status)
            print(f"{task} seed{seed}: {status['actual']}/{n}, exit={proc.returncode}",flush=True)
        return rows

    rows=[]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for future in futures: rows.extend(future.result())
    incomplete=[row for row in rows if not row['valid']]
    report=dict(plan,cells=rows,valid_episodes=sum(r['actual'] for r in rows if r['valid']),
                complete=len(rows)==len(cells) and not incomplete,incomplete=incomplete)
    (args.output/'evaluation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # The installed SDK owns score aggregation and video inventory checks.
    subprocess.run([sys.executable, str(summarizer)], cwd=args.robodojo_root,
                   env=base_env, check=True)
    if not report['complete']: raise SystemExit(1)

if __name__=='__main__':
    main()
