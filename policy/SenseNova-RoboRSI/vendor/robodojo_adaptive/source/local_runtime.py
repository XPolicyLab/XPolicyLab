"""Portable deployment launcher; policy/controller behavior is unchanged."""
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code'))
from policylab import codex
from adaptive_steps import apply_adaptive_steps

BWRAP = ROOT.parents[2] / "runtime/codex/codex-resources/bwrap"
ORIGINAL_COMMAND = codex.command


def resolve_runtime_config(cfg, *, expected_model=None):
    """Resolve required deployment settings without embedding service defaults."""
    values = {}
    for key, field in (("ROBORSI_CODEX_BASE_URL", "model_base_url"),
                       ("ROBORSI_CODEX_MODEL", "model")):
        value = os.environ.get(key, "").strip()
        if not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError(f"{key} must be a nonempty value without whitespace")
        values[field] = value
    try:
        endpoint = urlsplit(values["model_base_url"])
        valid = (
            endpoint.scheme in {"http", "https"}
            and endpoint.hostname
            and endpoint.username is None
            and endpoint.password is None
            and not endpoint.query
            and not endpoint.fragment
        )
        endpoint.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("ROBORSI_CODEX_BASE_URL must be an HTTP(S) URL without credentials, query or fragment")
    if expected_model is not None and values["model"] != expected_model:
        raise ValueError("ROBORSI_CODEX_MODEL differs from the prepared model identity")
    return apply_adaptive_steps({**cfg, **values})


def command(cfg, session=None):
    cmd = ORIGINAL_COMMAND(cfg, session)
    extra = ['-c', 'features.multi_agent=false', '-c', 'model_provider="runtime"',
             '-c', 'model_providers.runtime.name="Runtime"',
             '-c', 'model_providers.runtime.base_url=' + json.dumps(cfg['model_base_url']),
             '-c', 'model_providers.runtime.env_key="CODEX_API_KEY"',
             '-c', 'model_providers.runtime.wire_api="responses"',
             '-c', 'model_providers.runtime.stream_max_retries=8',
             '-c', 'model_providers.runtime.request_max_retries=8']
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
             '--setenv', 'CODEX_API_KEY', os.environ['CODEX_API_KEY'],
             '--chdir', '/workspace']
    episode = cfg.get('_observation_episode')
    if episode:
        argv += ['--dir', '/observations', '--ro-bind', str(run/'observations'/episode), '/observations/'+episode]
    else:
        argv += ['--ro-bind', str(run/'observations'), '/observations']
    # Forward only caller-selected proxy behavior; no private routing defaults.
    for name in ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY',
                 'https_proxy', 'http_proxy', 'all_proxy', 'no_proxy'):
        if name in os.environ:
            argv += ['--setenv', name, os.environ[name]]
    return argv + ['--'] + cmd


def main():
    from policylab.common import load_config
    from policylab.runner import run_experiment
    import hybrid_host
    hybrid_host.install()
    codex.command = command
    codex.sandbox_command = sandbox_command
    cfg = resolve_runtime_config(load_config(sys.argv[1]))
    run,summary=run_experiment(cfg, sys.argv[2] if len(sys.argv)>2 else str(Path.cwd()/"runs"))
    print(json.dumps({'run': str(run), **summary}, indent=2), flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
