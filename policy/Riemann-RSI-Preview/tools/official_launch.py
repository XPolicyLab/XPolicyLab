"""XPolicyLab's ten-argument server/client lifecycle for both policy backends."""
from contextlib import nullcontext
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def interpreter(environment):
    path = Path(environment).expanduser()
    if path.is_file():
        return [str(path)]
    if path.is_dir() and (path / 'bin/python').is_file():
        return [str(path / 'bin/python')]
    return ['conda', 'run', '--no-capture-output', '-n', environment, 'python']


def devices(value):
    values = value.split(',')
    if not values or any(not item.isdecimal() for item in values):
        raise ValueError('Supply explicit comma-separated physical GPU numbers')
    result = [int(item) for item in values]
    if len(set(result)) != len(result):
        raise ValueError('Physical GPU numbers must be unique')
    return result


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bench', 'task', 'checkpoint', 'robot', 'action', 'seed',
                 'policy_gpu', 'env_gpu', 'policy_env', 'eval_env'):
        parser.add_argument(name)
    return parser.parse_args(argv)


def evaluate(argv, *, episodes=None, output=None, run_id=None, timeout=None,
             policy_source=None, resume_from=None):
    args = arguments(argv)
    source = Path(policy_source).resolve() if policy_source is not None else ROOT
    if args.bench != 'RoboDojo' or args.robot != 'arx_x5':
        raise ValueError('The current policies support RoboDojo/arx_x5')
    from rsi.policy_catalog import resolve_task, runtime_settings
    if source == ROOT:
        _, route = resolve_task(None, args.task)
        settings = runtime_settings()
    else:
        from rsi.official_source import read as source_catalog
        historical = source_catalog(source, args.task)
        route, settings = historical['route'], historical['settings']
        current = runtime_settings()
        if any(settings.get(key) != current.get(key) for key in ('official_root', 'python', 'policy_name')):
            raise ValueError('A historical actor must use the same current official SDK and evaluator')
    if args.action not in ('joint', 'ee') or route['backend'] == 'learned' and args.action != 'ee':
        raise ValueError('Use the declared action space (WAM requires ee)')
    from rsi.deployment import interpreter_python
    resolved_eval = interpreter_python(args.eval_env)
    if (Path(resolved_eval).parent.parent.resolve() != Path(settings['python']).parent.parent.resolve()):
        raise ValueError('The evaluator interpreter is fixed to ' + settings['python'])
    from rsi.deployment import evaluator_path
    eval_path = evaluator_path(dict(os.environ), settings['python'],
                               require_ffmpeg=os.environ.get('EVAL_ENV_TYPE') != 'debug')
    from rsi.gpu_resources import gpu_leases, stop_owned, remember_owned_group
    debug = os.environ.get('EVAL_ENV_TYPE') == 'debug'
    from rsi.skills.git_versioning import runtime_source_identity
    identity = runtime_source_identity(source)
    continuation = None
    if resume_from is not None:
        if debug or episodes is not None or not run_id:
            raise ValueError('SDK continuation needs the original run ID and official default count')
        from rsi.official_resume import inspect_resume
        continuation = inspect_resume(resume_from, identity, args, route, run_id)
    selected = sorted(set(devices(args.policy_gpu) + devices(args.env_gpu)))
    if len(devices(args.env_gpu)) != 1:
        raise ValueError('One simulator GPU is required per episode')
    # Debug transport has no GPU or simulated success; normal execution must
    # retain the standard EX lease until both owned process groups have exited.
    synthetic = debug and os.environ.get('RIEMANN_DEBUG_HOLD') == '1'
    lease_fds = []
    leases = nullcontext([]) if synthetic else gpu_leases(selected, lease_descriptors=lease_fds)
    if episodes is not None and (type(episodes) is not int or episodes <= 0):
        raise ValueError('A bounded diagnostic needs an explicit positive SDK episode count')
    if output is None:
        output = Path(tempfile.mkdtemp(prefix='riemann-official-'))
    else:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    (output / 'source-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    if continuation is not None:
        (output / 'sdk-continuation-input.json').write_text(json.dumps(continuation, indent=2) + '\n')
    if policy_source is not None:
        (output / 'launcher-source-identity.json').write_text(
            json.dumps(runtime_source_identity(ROOT), indent=2) + '\n')
    from rsi.official_client_lifecycle import OfficialClientRestarts, client_transcript, wait_client
    result_path = (output / 'official-client/eval_result' / args.bench / args.task /
                   'Riemann-RSI-Preview' / args.robot /
                   (args.seed + '_ckpt_name=' + args.checkpoint + ',action_type=' + args.action) /
                   (run_id or output.name) / '_result.json')
    restarts = None if debug else OfficialClientRestarts(
        settings['official_root'], output, os.environ, run_id or output.name, result_path,
        expected=dict(run_id=run_id or output.name, task_name=args.task, policy_name='Riemann-RSI-Preview',
                      config_name=args.robot, eval_seed=int(args.seed),
                      save_dir=str(result_path.parent.relative_to(output / 'official-client')),
                      additional_info='ckpt_name=' + args.checkpoint + ',action_type=' + args.action),
        expected_count=episodes or route.get('official_eval_nums', route.get('official_episodes')))
    with leases as inventory:
        import datetime
        import socket as host_socket
        (output / 'lease.json').write_text(json.dumps({
            'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'node': host_socket.gethostname(), 'owner_pid': os.getpid(),
            'job': os.environ.get('SLURM_JOB_ID'), 'gpus': inventory,
            'held_descriptors': lease_fds}, indent=2) + '\n')
        identities = {row['index']: row['uuid'] for row in inventory}
        from rsi.gpu_resources import cuda_visibility
        visible = lambda value: (','.join(str(gpu) for gpu in devices(value)) if synthetic
                                 else cuda_visibility(devices(value), inventory))
        policy_devices = devices(args.policy_gpu)
        if route['backend'] == 'learned' and len(policy_devices) == 1:
            # Preserve the original WAM/T5 two-device topology. The simulator
            # shares the second leased card, as in the existing WAM launcher.
            policy_devices += [gpu for gpu in devices(args.env_gpu) if gpu not in policy_devices]
        policy_visible = (','.join(str(gpu) for gpu in policy_devices) if synthetic
                          else cuda_visibility(policy_devices, inventory))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        from rsi.deployment import apply_native_libraries
        native_dirs = apply_native_libraries(env)
        if not debug:
            from rsi.graphics_runtime import configure as configure_graphics
            physical = devices(args.env_gpu)[0]
            try:
                graphics = configure_graphics(env, identities[physical])
            except subprocess.CalledProcessError as error:
                (output / 'graphics-admission-error.json').write_text(json.dumps(dict(
                    command=error.cmd, returncode=error.returncode, stdout=error.stdout,
                    stderr=error.stderr, native_library_dirs=native_dirs), indent=2) + '\n')
                raise RuntimeError('Renderer admission failed before SDK/model startup: '
                                   + (error.stderr or str(error))[-2000:]) from error
            graphics['physical_gpu'] = physical
            graphics['native_library_dirs'] = native_dirs
            (output / 'graphics-runtime.json').write_text(json.dumps(graphics, indent=2) + '\n')
        for key in ('RIEMANN_COMPOSITION_ROOT', 'RIEMANN_SKILL_MODE', 'RIEMANN_SKILL_FREEZE_SHA256',
                    'RIEMANN_SKILL_REGISTRY', 'RIEMANN_SKILL_TRACE_DIR', 'PYTHONPATH'):
            env.pop(key, None)
        if not debug:
            env.pop('EVAL_NUM', None)
            env['ROBODOJO_RUN_ID'] = run_id or output.name
            if episodes is not None:
                env['EVAL_NUM'] = str(episodes)  # Explicit legacy diagnostic only, never the normal CLI default.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        config = dict(policy_name='Riemann-RSI-Preview', task_name=args.task, action_type=args.action,
                      env_cfg_type=args.robot, bench_name=args.bench, ckpt_name=args.checkpoint,
                      seed=int(args.seed), protocol='ws', output_dir=str(output / 'policy'))
        if debug and os.environ.get('RIEMANN_DEBUG_HOLD') == '1':
            config['debug_hold'] = True
        local = source / 'deploy.local.json'
        if continuation is not None:
            config = dict(continuation['deploy'], output_dir=str(output / 'policy'))
        elif local.is_file():
            config.update(json.loads(local.read_text()))
        # Explicit invocation always wins over local model asset locations.
        config.update(task_name=args.task, action_type=args.action, env_cfg_type=args.robot)
        (output / 'deploy.json').write_text(json.dumps(config, indent=2) + '\n')
        policy_command = [settings['python']] if route['backend'] == 'rule' else interpreter(args.policy_env)
        server_cmd = [*policy_command, '-B', str(source / 'tools/serve_official_policy.py'),
                      '--mode', args.task, '--port', str(port), '--config', str(output / 'deploy.json'),
                      '--output', str(output / 'policy'), '--action_type', args.action,
                      '--seed', args.seed]
        client_cmd = [settings['python'], '-B', str(ROOT / 'tools/run_official_eval.py'),
                      '--task_name', args.task, '--env_cfg_type', args.robot, '--policy_name', 'Riemann-RSI-Preview',
                      '--port', str(port), '--protocol', 'ws', '--host', '127.0.0.1',
                      '--seed', args.seed, '--device_id', '0', '--headless', '--enable_cameras',
                      '--additional_info', 'ckpt_name=' + args.checkpoint + ',action_type=' + args.action]
        # The installed SDK is immutable. Its evaluator intentionally writes
        # relative eval_result paths, so give it a private writable cwd with
        # ordinary read-only links to the same official static inputs.
        client_root = output / 'official-client'
        client_root.mkdir()
        if continuation is not None:
            # Reuse only the official result namespace. SDK loads its own
            # manifest and filters completed IDs; fresh policy diagnostics stay
            # in this new attempt directory. No result is copied or regraded.
            (client_root / 'eval_result').symlink_to(continuation['result_root'], target_is_directory=True)
        for name in ('Assets', 'env_cfg', 'task', 'utils', 'XPolicyLab', 'third_party'):
            target = Path(settings['official_root']) / name
            if target.is_dir():
                (client_root / name).symlink_to(target, target_is_directory=True)
        server = client = None
        try:
            with (output / 'server.log').open('x') as log:
                server = subprocess.Popen(server_cmd, cwd=source,
                    env=dict(env, CUDA_VISIBLE_DEVICES=policy_visible,
                             PATH=(str(Path(policy_command[0]).parent) + os.pathsep + eval_path
                                   if len(policy_command) == 1 else eval_path)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(lease_fds))
                remember_owned_group(server)
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Policy server exited; see ' + str(output / 'server.log'))
                    # The common server emits this only after binding its WS
                    # listener. A bare TCP connect creates a spurious failed WS
                    # handshake/traceback in otherwise healthy server logs.
                    with (output / 'server.log').open('rb') as status:
                        status.seek(max(0, os.fstat(status.fileno()).st_size - 16384))
                        ready = False
                        for line in status.read().splitlines():
                            try:
                                event = json.loads(line)
                                if (event.get('event') in ('official_policy_ready', 'official_rule_policy_ready')
                                        and event.get('mode') == args.task):
                                    ready = True
                            except (ValueError, AttributeError):
                                pass
                    if ready:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Policy server startup timed out')
                    time.sleep(.5)
                client_deadline = None if timeout is None else time.monotonic() + timeout
                while True:
                    if client_deadline is not None and time.monotonic() >= client_deadline:
                        raise subprocess.TimeoutExpired(client_cmd, timeout)
                    with client_transcript(output, restarts) as (transcript, log_path):
                        client = subprocess.Popen(client_cmd, cwd=client_root,
                            env=dict(env, CUDA_VISIBLE_DEVICES=visible(args.env_gpu),
                                     RIEMANN_PHYSICAL_ENV_GPU=args.env_gpu, PATH=eval_path),
                            stdout=transcript, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=tuple(lease_fds))
                        remember_owned_group(client)
                        remaining = None if client_deadline is None else max(0., client_deadline - time.monotonic())
                        code = wait_client(client, log_path, timeout=remaining, server=server,
                                           result_path=result_path,
                                           health_path=output / 'runtime-health.json',
                                           policy_episode_path=output / 'policy/episode-0000'
                                               if route['backend'] == 'rule' else None)
                    if debug or not restarts.should_restart(code, client_log=log_path):
                        break
                    # Same model server, GPU lease, command, cwd and run ID.
                    # Only the SDK reads its checkpoint or chooses layouts.
                    time.sleep(restarts.delay_seconds)
                if code:
                    raise subprocess.CalledProcessError(code, client_cmd)
                if not debug:
                    from rsi.official_resume import document, official_details
                    expected_count = episodes or route.get('official_eval_nums', route.get('official_episodes'))
                    if len(official_details(document(result_path))) != expected_count:
                        raise RuntimeError('SDK exited without the complete official cohort; see client transcript')
                if continuation is not None:
                    from rsi.official_resume import document, verify_extension
                    count = verify_extension(continuation['official_prefix'],
                                             document(continuation['official_result']))
                    if count != continuation['expected_episodes']:
                        raise RuntimeError('SDK continuation exited with an incomplete official cohort')
        finally:
            stop_owned(client)
            stop_owned(server)
            (output / 'source-identity-final.json').write_text(
                json.dumps(runtime_source_identity(source), indent=2) + '\n')
    print('[MAIN] eval finished; diagnostics: ' + str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(evaluate(sys.argv[1:]))
