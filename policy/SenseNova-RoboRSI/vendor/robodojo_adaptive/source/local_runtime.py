"""Machine-specific launcher only; use the supplied framework and adapter unchanged."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from policylab import codex

BWRAP = ROOT.parents[2] / "runtime/codex/codex-resources/bwrap"
ORIGINAL_COMMAND = codex.command


def command(cfg, session=None):
    cmd = ORIGINAL_COMMAND(cfg, session)
    extra = ['-c', 'features.multi_agent=false', '-c', 'model_provider="tokenhub"',
             '-c', 'model_providers.tokenhub.name="TokenHub"',
             '-c', 'model_providers.tokenhub.base_url="https://tokenhub.sensetime.com/v1"',
             '-c', 'model_providers.tokenhub.env_key="ROBOPROBE_MODEL_KEY"',
             '-c', 'model_providers.tokenhub.wire_api="responses"',
             '-c', 'model_providers.tokenhub.stream_max_retries=8',
             '-c', 'model_providers.tokenhub.request_max_retries=8']
    return cmd[:-1] + extra + cmd[-1:]


def sandbox_command(cfg, run, private_home, cmd):
    binary = codex.native_binary(cfg['codex_binary'])
    argv = [str(BWRAP), '--die-with-parent', '--new-session', '--unshare-user',
            '--unshare-pid', '--unshare-ipc', '--unshare-uts']
    for p in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc/ssl',
              '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/ld.so.cache']:
        if Path(p).exists():
            argv += ['--ro-bind', p, p]
    python = Path(cfg['agent_python'])
    runtime, base = python.parent.parent, python.resolve().parent.parent
    argv += ['--ro-bind', str(base), str(base), '--ro-bind', str(runtime), '/runtime',
             '--ro-bind', str(binary), '/codex',
             '--ro-bind', str(binary.with_name('codex-code-mode-host')), '/codex-code-mode-host',
             '--ro-bind', str(binary.parent.parent / 'codex-resources'), '/codex-resources',
             '--dir', '/proc', '--dir', '/proc/self', '--symlink', '/codex', '/proc/self/exe',
             '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/home/agent',
             '--bind', str(private_home), '/home/agent/.codex',
             '--ro-bind', str(Path(cfg['training_set']).resolve()), '/training',
             '--ro-bind', str(run / 'bridge'), '/bridge',
             '--ro-bind', str(run / 'tools'), '/tools',
             '--bind', str((run / 'workspace').resolve()), '/workspace',
             '--clearenv', '--setenv', 'HOME', '/home/agent',
             '--setenv', 'CODEX_HOME', '/home/agent/.codex',
             '--setenv', 'PATH', '/runtime/bin:/usr/bin:/bin',
             '--setenv', 'PYTHONPATH', '/tools', '--setenv', 'LANG', 'C.UTF-8',
             '--setenv', 'TERM', 'xterm',
             '--setenv', 'ROBOPROBE_MODEL_KEY', os.environ['ROBOPROBE_MODEL_KEY'],
             '--chdir', '/workspace']
    episode = cfg.get('_observation_episode')
    if episode:
        argv += ['--dir', '/observations', '--ro-bind', str(run/'observations'/episode), '/observations/'+episode]
    else:
        argv += ['--ro-bind', str(run/'observations'), '/observations']
    for name in ['NO_PROXY', 'no_proxy']:
        argv += ['--setenv', name, 'localhost,127.0.0.1,tokenhub.sensetime.com']
    return argv + ['--'] + cmd


def main():
    from policylab.common import load_config
    from policylab.runner import run_experiment
    import hybrid_host
    hybrid_host.install()
    codex.command = command
    codex.sandbox_command = sandbox_command
    cfg = load_config(sys.argv[1])
    run,summary=run_experiment(cfg, sys.argv[2] if len(sys.argv)>2 else str(Path.cwd()/"runs"))
    print(json.dumps({'run': str(run), **summary}, indent=2), flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
