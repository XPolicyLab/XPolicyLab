"""Regression tests for the isolated public policy installer."""
import ast
import builtins
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import tempfile
import hashlib
import io
import json
from types import SimpleNamespace
from unittest import mock

import pytest

PATH = Path(__file__).resolve().parents[1] / 'environment' / 'install.py'
SPEC = importlib.util.spec_from_file_location('public_installer', PATH)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
RUN_IDENTITY = PATH.parents[1] / 'run_identity.py'
SPEC = importlib.util.spec_from_file_location('vap_run_identity', RUN_IDENTITY)
run_identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_identity)


def write_request_signature(run, arguments):
    payload = run_identity.serialize(run_identity.server_request(run_identity.canonical_request(arguments)))
    (run / '.vap_request.json').write_text(payload + '\n')


def test_shared_server_request_allows_tasks_but_binds_eval_contract(monkeypatch, tmp_path):
    arguments = ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee', '0']
    monkeypatch.delenv('EVAL_NUM', raising=False)
    monkeypatch.delenv('EVAL_ENV_TYPE', raising=False)
    monkeypatch.delenv('DEBUG_OBS_ENCODED', raising=False)
    request = run_identity.canonical_request(arguments)
    shared = run_identity.server_request(request)
    assert request['eval_num'] == 'native'
    assert 'task' not in shared
    assert 'eval_num' not in shared
    assert run_identity.run_id(request).startswith('vap_stack_bowls_seed0_')
    other_task = arguments.copy()
    other_task[1] = 'push_T'
    assert run_identity.server_request(run_identity.canonical_request(other_task)) == shared
    assert run_identity.run_id(run_identity.canonical_request(other_task)) != run_identity.run_id(request)
    for position, value in ((2, '/other-checkpoint'), (3, 'other_robot'), (4, 'joint'), (5, '1')):
        other = arguments.copy()
        other[position] = value
        assert run_identity.server_request(run_identity.canonical_request(other)) != shared
    for name, value in (('EVAL_ENV_TYPE', 'debug'), ('DEBUG_OBS_ENCODED', '1')):
        monkeypatch.setenv(name, value)
        assert run_identity.server_request(run_identity.canonical_request(arguments)) != shared
        monkeypatch.delenv(name)
    monkeypatch.setenv('EVAL_NUM', '1')
    counted = run_identity.canonical_request(arguments)
    assert run_identity.server_request(counted) == shared
    assert run_identity.run_id(counted) != run_identity.run_id(request)
    monkeypatch.setenv('VAP_POLICY_DIR', '/workspace/RoboDojo/XPolicyLab/policy/vap')
    relative = arguments.copy()
    relative[2] = './checkpoints/baseline'
    relative_request = run_identity.canonical_request(relative)
    assert relative_request['checkpoint'] == '/workspace/RoboDojo/XPolicyLab/policy/vap/checkpoints/baseline'
    expected_id = run_identity.run_id(relative_request)
    monkeypatch.chdir(tmp_path)
    assert run_identity.run_id(run_identity.canonical_request(relative)) == expected_id
    environment = dict(os.environ, EVAL_NUM='1', VAP_POLICY_DIR='/workspace/RoboDojo/XPolicyLab/policy/vap')
    cli_request = subprocess.check_output(
        [sys.executable, '-I', str(RUN_IDENTITY), '--request', *arguments],
        env=environment, text=True,
    )
    assert json.loads(cli_request) == shared
    cli_shell = subprocess.check_output(
        [sys.executable, '-I', str(RUN_IDENTITY), '--shell', *arguments],
        env=environment, text=True,
    )
    assert cli_shell == f'export ROBODOJO_RUN_ID={run_identity.run_id(counted)}\n'
    subprocess.run([sys.executable, '-I', str(RUN_IDENTITY), '--write', str(tmp_path), *arguments],
                   env=environment, check=True)
    assert json.loads((tmp_path / '.vap_request.json').read_text()) == shared
    subprocess.run([sys.executable, '-I', str(RUN_IDENTITY), '--validate', str(tmp_path), *arguments],
                   env=environment, check=True)
    policy_dir = tmp_path / 'RoboDojo/XPolicyLab/policy/vap'
    combined = policy_dir / 'checkpoints/RoboDojo-baseline-arx_x5-ee-0'
    combined.mkdir(parents=True)
    monkeypatch.setenv('VAP_POLICY_DIR', str(policy_dir))
    bare = arguments.copy()
    bare[2] = 'baseline'
    bare_request = run_identity.canonical_request(bare)
    assert bare_request['checkpoint'] == str(combined)
    assert run_identity.run_id(bare_request) != run_identity.run_id(request)
    verbatim = policy_dir / 'checkpoints/baseline'
    verbatim.mkdir()
    assert run_identity.canonical_request(bare)['checkpoint'] == str(combined)
    combined.rmdir()
    fallback_request = run_identity.canonical_request(bare)
    assert fallback_request['checkpoint'] == str(verbatim)
    monkeypatch.chdir(tmp_path)
    assert run_identity.canonical_request(bare) == fallback_request
    subprocess.run([sys.executable, '-I', str(RUN_IDENTITY), '--write', str(verbatim), *bare],
                   env=dict(os.environ, VAP_POLICY_DIR=str(policy_dir)), check=True)
    assert json.loads((verbatim / '.vap_request.json').read_text())['checkpoint'] == str(verbatim)


def test_pip_environment_is_sanitized():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.dict(os.environ, {
            'PYTHONPATH': '/private/editable', 'PYTHONHOME': '/private/python',
            'PIP_TARGET': '/private/target', 'PIP_PREFIX': '/private/prefix',
            'PIP_ROOT': '/private/root', 'PIP_INDEX_URL': 'https://private.invalid/simple',
            'PIP_CUSTOM_SETTING': 'private',
            'VAP_INSTALL_CACHE': tmp,
        }):
            env = installer.public_environ()
        assert 'PYTHONPATH' not in env
        assert 'PYTHONHOME' not in env
        assert not any(key.startswith('PIP_') for key in env
                       if key not in {'PIP_DISABLE_PIP_VERSION_CHECK', 'PIP_ROOT_USER_ACTION',
                                      'PIP_CONFIG_FILE', 'PIP_CACHE_DIR'})
        for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_INDEX_URL', 'PIP_CUSTOM_SETTING'):
            assert key not in env
        assert env['PIP_CONFIG_FILE'] == os.devnull
        assert env['PYTHONNOUSERSITE'] == '1'


def test_lock_preflight_and_existing_target():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'new-env'
        args = SimpleNamespace(target=target, python=sys.executable, index_url='unused')
        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', Path(tmp) / 'missing.lock'), \
             mock.patch.object(installer, 'run', side_effect=lambda *a, **kw: calls.append(a)):
            try:
                installer.install(args)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('missing lock was accepted')
        assert not target.exists()
        assert not calls


def test_failed_install_is_retryable():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'new-env'
        (Path(tmp) / 'requirements.lock').write_text('numpy==2.2.6 --hash=sha256:' + '0' * 64 + '\n')
        args = SimpleNamespace(target=target, python=sys.executable,
                               index_url='https://pypi.org/simple')
        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', Path(tmp) / 'requirements.lock'), \
             mock.patch.object(installer, 'public_environ', side_effect=lambda: {'PIP_CACHE_DIR': str(Path(tmp) / 'cache')}), \
             mock.patch.object(installer, 'bootstrap_pip', side_effect=subprocess.CalledProcessError(1, 'bootstrap')), \
             mock.patch.object(installer, 'run') as run:
            run.side_effect = [None, subprocess.CalledProcessError(1, 'bootstrap')]
            try:
                installer.install(args)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('simulated pip failure was ignored')
        assert not target.exists()
        assert not list(Path(tmp).glob('new-env-installing-*'))


def test_isolated_source_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'XPolicyLab').mkdir()
        (root / 'XPolicyLab/__init__.py').write_text('')
        (root / 'XPolicyLab/model_template.py').write_text('class ModelTemplate: pass\n')
        code = ('import sys; sys.path[:0] = sys.argv[1:]; '
                'import XPolicyLab.model_template as module; print(module.__file__)')
        result = subprocess.run([sys.executable, '-I', '-c', code, str(root)],
                                env={'PYTHONPATH': '/private/invalid'},
                                check=True, capture_output=True, text=True)
    assert Path(result.stdout.strip()) == root / 'XPolicyLab/model_template.py'


def test_existing_target_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'existing'
        target.mkdir()
        (target / 'keep.txt').write_text('unchanged')
        args = SimpleNamespace(target=target, python=sys.executable, index_url='unused')
        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', Path(tmp) / 'requirements.lock'), \
             mock.patch.object(installer, 'run') as run:
            (Path(tmp) / 'requirements.lock').write_text('numpy==2.2.6 --hash=sha256:' + '0' * 64 + '\n')
            try:
                installer.install(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError('existing target was overwritten')
        assert (target / 'keep.txt').read_text() == 'unchanged'
        run.assert_not_called()


def test_dangling_target_symlink_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'target-link'
        target.symlink_to(Path(tmp) / 'missing-target')
        args = SimpleNamespace(target=target, python=sys.executable, index_url='unused')
        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', Path(tmp) / 'requirements.lock'):
            (Path(tmp) / 'requirements.lock').write_text('numpy==2.2.6 --hash=sha256:' + '0' * 64 + '\n')
            try:
                installer.install(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError('dangling symlink target was accepted')
        assert target.is_symlink()


def test_check_uses_target_python_without_path_interpreter():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / 'venv'
        target.mkdir()
        commands = root / 'commands'
        commands.mkdir()
        fake = root / 'fake_install.py'
        fake.write_text('import sys; print("CHECK_PYTHON", sys.executable)\n')
        wrapper = commands / 'install.sh'
        source = (policy / 'install.sh').read_text().replace(
            'INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            f'INSTALL_DIR={root}',
        )
        wrapper.write_text(source)
        python = commands / 'python3.11'
        python.write_text('#!/bin/sh\nexit 97\n')
        python.chmod(0o755)
        result = subprocess.run(
            ['bash', str(wrapper), 'check', str(target)],
            executable='/bin/bash',
            env={'PATH': str(commands), 'INSTALL_PYTHON': str(python)},
            capture_output=True, text=True,
        )
        assert result.returncode == 127
        assert '/venv/bin/python: No such file or directory' in result.stderr
        assert result.returncode != 97

        (target / 'bin').mkdir()
        (target / 'bin/python').write_text(
            '#!/bin/sh\nprintf "CHECK_ARGV"\nprintf " <%s>" "$@"\n'
        )
        (target / 'bin/python').chmod(0o755)
        result = subprocess.run(
            ['bash', str(wrapper), 'check', str(target), '--report', 'report.json'],
            executable='/bin/bash',
            env={'PATH': str(commands), 'INSTALL_PYTHON': str(python)},
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout == (
            f'CHECK_ARGV <{root}/environment/install.py> <check> <{target}> '
            '<--report> <report.json>'
        )


def test_policy_launcher_uses_explicit_isolated_paths():
    policy = PATH.parents[1]
    script = (policy / 'setup_eval_policy_server.sh').read_text()
    assert 'bin/python" -I ' in script
    assert '"$VAP_POLICY_DIR/audit_no_wan.py" --config_path' in script
    assert 'os.environ["VAP_POLICY_DIR"]' not in script
    common = (policy / 'common.sh').read_text()
    assert 'PYTHONPATH:+:$PYTHONPATH' not in common
    assert 'third_party/IsaacLab' not in common
    assert 'third_party/curobo' not in common
    assert 'host.env' not in common
    assert '${SIM_ENV/bin/python' not in common
    client = (policy / 'setup_eval_env_client.sh').read_text()
    assert 'export POLICY_ENV="${POLICY_ENV:-${ROBODOJO_SIM_ENV:-$SIM_ENV}}"' in client


def test_official_client_defaults_only_its_launcher_to_sim_environment():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / 'run'
        run.mkdir()
        (run / '.vap_run_id').write_text('official\n')
        write_request_signature(run, ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee', '0'])
        sim = root / 'sim'
        (sim / 'bin').mkdir(parents=True)
        (sim / 'bin/python').symlink_to(Path(sys.executable).resolve())
        launcher = (policy / 'setup_eval_env_client.sh').read_text().replace(
            'source "$(dirname "$0")/common.sh"', f'source "{policy / "common.sh"}"',
        ).replace(
            'exec bash "$XPL_ROOT/utils/setup_env_client.sh"',
            'printf "%s\\n" "$POLICY_ENV"; exit 0\n#',
        )
        script = root / 'client.sh'
        script.write_text(launcher)
        environment = {key: value for key, value in os.environ.items() if key != 'POLICY_ENV'}
        environment.update(VAP_QWEN_MODEL='/explicit/qwen', VAP_RUN_DIR=str(run),
                           ROBODOJO_RUN_ID='official', VAP_CACHE_ROOT=str(root / 'cache'))
        result = subprocess.run(
            ['bash', str(script), 'RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5',
             'ee', '0', '0', str(sim), 'ckpt_name=/checkpoint,action_type=ee', '34567', 'localhost'],
            env=environment, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(sim)


def test_failed_install_does_not_delete_a_concurrent_target():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'new-env'
        (Path(tmp) / 'requirements.lock').write_text('numpy==2.2.6 --hash=sha256:' + '0' * 64 + '\n')
        args = SimpleNamespace(target=target, python=sys.executable, index_url='unused')

        def simulate_race(*args, **kwargs):
            target.mkdir()
            (target / 'other-install').write_text('keep')
            raise subprocess.CalledProcessError(1, 'pip')

        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', Path(tmp) / 'requirements.lock'), \
             mock.patch.object(installer, 'public_environ', return_value={}), \
             mock.patch.object(installer, 'bootstrap_pip'), \
             mock.patch.object(installer, 'run', side_effect=simulate_race):
            try:
                installer.install(args)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('simulated pip failure was ignored')
        assert (target / 'other-install').read_text() == 'keep'


def test_relative_environment_paths_are_canonicalized():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy_env = root / 'policy'
        sim_env = Path(tmp) / 'sim'
        for environment in (policy_env, sim_env):
            (environment / 'bin').mkdir(parents=True)
            (environment / 'bin/python').symlink_to(Path(sys.executable).resolve())
        env = dict(os.environ, VAP_QWEN_MODEL='/explicit/qwen',
                   WALLX_MODEL_ROOT='/explicit/models', VAP_WAN22_TOKENIZER_PATH='/explicit/t5',
                   VAP_OUTPUT_ROOT=str(root / 'runs'), VAP_CACHE_ROOT=str(root / 'cache'),
                   ROBODOJO_RUN_ID='canonical')
        command = (f'export POLICY_ENV={policy_env} SIM_ENV={sim_env}; '
                   f'cd {policy.parent}; set -- RoboDojo stack_bowls /checkpoint arx_x5 ee 0; '
                   f'source {policy / "common.sh"} >/dev/null; '
                   'printf "%s\\n%s\\n" "$POLICY_ENV" "$SIM_ENV"')
        result = subprocess.run(['bash', '-c', command], env=env, check=True,
                                capture_output=True, text=True)
        assert result.stdout.splitlines() == [str(policy_env), str(sim_env)]
        audit_env = dict(env, VAP_REQUIRE_NO_WAN='1', ROBODOJO_RUN_ID='audit')
        command = (f'export POLICY_ENV={policy_env} SIM_ENV={sim_env} VAP_REQUIRE_NO_WAN=1; '
                   f'cd {policy.parent}; set -- RoboDojo stack_bowls /checkpoint arx_x5 ee 0; '
                   f'source {policy / "common.sh"} >/dev/null; '
                   'test -z "${WALLX_MODEL_ROOT+x}" && test -z "${VAP_WAN22_TOKENIZER_PATH+x}"')
        subprocess.run(['bash', '-c', command], env=audit_env, check=True)


def test_shared_run_initialization_and_split_reuse():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ('policy', 'sim'):
            (root / name / 'bin').mkdir(parents=True)
            (root / name / 'bin/python').symlink_to(Path(sys.executable).resolve())
        env = dict(os.environ, POLICY_ENV=str(root / 'policy'), SIM_ENV=str(root / 'sim'),
                   VAP_QWEN_MODEL='/explicit/qwen', VAP_OUTPUT_ROOT=str(root / 'results'),
                   VAP_CACHE_ROOT=str(root / 'cache'), ROBODOJO_RUN_ID='shared',
                   EVAL_NUM='1')
        args = 'RoboDojo stack_bowls /checkpoint arx_x5 ee 0 0 policy-env port'
        def source(role, extra=None):
            environment = dict(env, VAP_LAUNCH_ROLE=role, **(extra or {}))
            command = (f'cd {policy}; set -- {args}; source ./common.sh; '
                       'printf "%s\\n%s\\n%s\\n" "$VAP_RUN_DIR" "$VAP_EVAL_TASK" "$VAP_EVAL_SEED"')
            return subprocess.run(['bash', '-c', command], env=environment,
                                  capture_output=True, text=True)
        result = source('server')
        assert result.returncode == 0, result.stderr
        run = root / 'results/shared'
        assert result.stdout.splitlines() == [str(run), 'stack_bowls', '0']
        assert (run / '.vap_run_id').read_text() == 'shared\n'
        assert (run / 'task.txt').read_text() == 'stack_bowls\n'
        assert source('server').returncode == 2
        assert (run / '.vap_run_id').read_text() == 'shared\n'
        result = source('client', {'VAP_RUN_DIR': str(run)})
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == str(run)
        result = source('server', {'VAP_RUN_DIR': str(run), 'VAP_RUN_INITIALIZED': '1'})
        assert result.returncode == 0, result.stderr
        missing = source('client', {'VAP_RUN_DIR': str(root / 'missing')})
        assert missing.returncode == 2 and not (root / 'missing').exists()
        unrelated = root / 'unrelated'
        unrelated.mkdir()
        assert source('client', {'VAP_RUN_DIR': str(unrelated)}).returncode == 2
        (run / '.vap_run_id').write_text('wrong\n')
        assert source('client', {'VAP_RUN_DIR': str(run)}).returncode == 2
        (run / '.vap_run_id').write_text('shared\n')
        (run / '.vap_request.json').write_text('{}\n')
        mismatched_request = source('client', {'VAP_RUN_DIR': str(run)})
        assert mismatched_request.returncode == 2
        assert 'VAP request signature mismatch' in mismatched_request.stderr
        dry = source('server', {'VAP_RUN_DIR': str(root / 'dry-run'), 'VAP_DRY_RUN': '1',
                                'VAP_CACHE_ROOT': str(root / 'dry-cache')})
        assert dry.returncode == 0, dry.stderr
        assert not (root / 'dry-run').exists() and not (root / 'dry-cache').exists()


def test_official_split_client_uses_the_policy_launcher_and_validates_identity():
    policy = PATH.parents[1]
    root = policy.parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / 'RoboDojo'
        package = workspace / 'XPolicyLab/policy/vap'
        package.mkdir(parents=True)
        for filename in ('common.sh', 'run_identity.py', 'setup_eval_env_client.sh', 'setup_eval_policy_server.sh', 'deploy.py'):
            (package / filename).write_text((policy / filename).read_text())
        (workspace / 'scripts').mkdir()
        (workspace / 'scripts/robodojo.sh').write_text((root / 'scripts/robodojo.sh').read_text())
        sim = Path(tmp) / 'sim'
        (sim / 'bin').mkdir(parents=True)
        (sim / 'bin/python').symlink_to(Path(sys.executable).resolve())
        handoff = Path(tmp) / 'setup_env_client.sh'
        handoff.write_text('#!/usr/bin/env bash\nexit 0\n')
        client = (package / 'setup_eval_env_client.sh').read_text().replace(
            'exec bash "$XPL_ROOT/utils/setup_env_client.sh"', f'exec bash "{handoff}"',
        ).replace(
            'source "$(dirname "$0")/common.sh"',
            'source "$(dirname "$0")/common.sh"\npwd > "$VAP_RUN_DIR/client_cwd.txt"',
        )
        (package / 'setup_eval_env_client.sh').write_text(client)
        run = Path(tmp) / 'run'
        run.mkdir()
        (run / '.vap_run_id').write_text('split-identity\n')
        write_request_signature(run, ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee', '0'])
        env = {key: value for key, value in os.environ.items()
               if key not in {'POLICY_ENV', 'SIM_ENV', 'VAP_RUN_INITIALIZED'}}
        env.update(ROBODOJO_RUN_ID='split-identity', VAP_RUN_DIR=str(run),
                   ROBODOJO_SIM_ENV=str(sim), VAP_QWEN_MODEL='/explicit/qwen',
                   VAP_CACHE_ROOT=str(Path(tmp) / 'cache'))
        command = ['bash', str(workspace / 'scripts/robodojo.sh'), 'client',
                   '--task', 'stack_bowls', '--policy-dir', str(package),
                   '--policy-host', '127.0.0.1', '--policy-port', '34567',
                   '--ckpt', '/checkpoint']
        result = subprocess.run([*command, '--dry-run'], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '/XPolicyLab/policy/vap ' in result.stdout
        parsed = subprocess.run(
            ['bash', '-c', 'eval "argv=($DRY_COMMAND)"; printf "%s\\0" "${argv[@]:0:7}"'],
            env=dict(os.environ, DRY_COMMAND=result.stdout.split('dry-run: ', 1)[1]),
            check=True, capture_output=True, text=True,
        ).stdout.split('\0')
        assert parsed[:7] == [
            'bash', '-c', 'cd "$1" && shift && exec "$@"', '_', str(package),
            'bash', str(package / 'setup_eval_env_client.sh'),
        ]
        assert str(package) in result.stdout
        assert 'RoboDojo stack_bowls /checkpoint arx_x5 ee 0 0' in result.stdout
        assert 'ckpt_name=/checkpoint\\,action_type=ee 34567 127.0.0.1' in result.stdout
        assert not (Path(tmp) / 'cache').exists()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert (run / 'client_cwd.txt').read_text() == str(package) + '\n'
        env['VAP_RUN_DIR'] = str(Path(tmp) / 'missing-run')
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'Missing VAP run identity' in result.stderr
        assert not Path(env['VAP_RUN_DIR']).exists()
        env['VAP_RUN_DIR'] = str(run)
        env['ROBODOJO_RUN_ID'] = 'mismatched'
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'VAP run identity mismatch' in result.stderr
        without_sim = dict(env, ROBODOJO_RUN_ID='split-identity', SIM_ENV='RoboDojo')
        del without_sim['ROBODOJO_SIM_ENV']
        result = subprocess.run(command, env=without_sim, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'requires an absolute simulator environment path' in result.stderr


def test_native_result_directory_reservation_preserves_existing_and_resume():
    source = Path(__file__).resolve().parents[4] / 'src/eval_client/eval_env.py'
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == 'prepare_result_directory')
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    prepare = namespace['prepare_result_directory']
    args = ('RoboDojo', 'stack_bowls', 'vap', 'arx_x5', 0,
            'ckpt_name=/checkpoint,action_type=ee', 'reserved-run')
    with tempfile.TemporaryDirectory() as tmp:
        previous = Path.cwd()
        os.chdir(tmp)
        try:
            created = Path(prepare(*args))
            assert created.is_dir()
            original = created / '_result.json'
            original.write_text('{"eval_time": 1}\n')
            try:
                prepare(*args)
            except FileExistsError:
                pass
            else:
                raise AssertionError('existing result path was not rejected')
            assert original.read_text() == '{"eval_time": 1}\n'
            assert 'if resume_state is None:\n                self.save_dir = prepare_result_directory(' in source.read_text()
        finally:
            os.chdir(previous)


def test_eval_main_failure_propagates_without_masking_restart_exits():
    source = Path(__file__).resolve().parents[4] / 'src/eval_client/main.py'
    tree = ast.parse(source.read_text())
    entry = next(item for item in tree.body
                 if isinstance(item, ast.If) and isinstance(item.test, ast.Compare)
                 and isinstance(item.test.left, ast.Name) and item.test.left.id == '__name__')
    calls = []

    def exit_with_status(error):
        os._exit(error.code)

    def main():
        calls.append('main')
        raise FileExistsError('existing native result directory')

    namespace = {'__name__': '__main__', 'main': main, 'os': os, 'sys': sys,
                 '_exit_with_status': exit_with_status}
    with mock.patch('os._exit') as exit_process:
        exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), 'exec'), namespace)
    exit_process.assert_called_once_with(1)
    assert calls == ['main']

    for expected in (0, 99, 134, 139):
        def restart_exit(code=expected):
            raise SystemExit(code)

        namespace = {
            '__name__': '__main__', 'main': restart_exit, 'os': os, 'sys': sys,
            '_exit_with_status': exit_with_status,
        }
        with mock.patch('traceback.print_exc') as print_exc, mock.patch('os._exit') as exit_process:
            exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), 'exec'), namespace)
            exit_process.assert_called_once_with(expected)
            print_exc.assert_not_called()


def test_post_app_import_failures_use_the_nonzero_exit_handler():
    source = Path(__file__).resolve().parents[4] / 'src/eval_client/main.py'
    tree = ast.parse(source.read_text())
    imports = next(item for item in tree.body if isinstance(item, ast.Try)
                   and isinstance(item.body[0], ast.ImportFrom)
                   and item.body[0].module == 'omegaconf')
    original_import = builtins.__import__

    def fail_post_app_import(name, *args, **kwargs):
        if name == 'omegaconf':
            raise ModuleNotFoundError('intentional post-AppLauncher import failure')
        return original_import(name, *args, **kwargs)

    with mock.patch('builtins.__import__', side_effect=fail_post_app_import), \
         mock.patch('traceback.print_exc') as print_exc, \
         mock.patch('os._exit') as exit_process:
        namespace = {'os': os, 'sys': sys, '_exit_with_status':
                     lambda error: os._exit(error.code)}
        exec(compile(ast.Module(body=[imports], type_ignores=[]), str(source), 'exec'), namespace)
        exit_process.assert_called_once_with(1)
        print_exc.assert_called_once()


def test_batch_split_preserves_explicit_policy_name(tmp_path):
    root = PATH.parents[1].parents[2]
    package = tmp_path / 'custom-policy-location'
    package.mkdir()
    for script in ('setup_eval_env_client.sh', 'setup_eval_policy_server.sh'):
        (package / script).write_text('#!/bin/bash\nexit 0\n')
    smoke = (root / 'scripts/internal/smoke_all_tasks.sh').read_text().replace(
        'log_dir="${ROOT_DIR}/smoke_results/${run_id}/logs"',
        'log_dir="${SMOKE_TEST_LOG_DIR}"',
    ).replace(
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
        'ROOT_DIR="${SMOKE_TEST_ROOT_DIR}"',
    )
    fixture = tmp_path / 'smoke_all_tasks.sh'
    fixture.write_text(smoke)
    command = [
        'bash', str(fixture),
        '--mode', 'client', '--policy-dir', str(package),
        '--policy-name', 'vap', '--policy-host', '127.0.0.1',
        '--policy-port', '34567', '--only', 'stack_bowls',
        '--ckpt', '/checkpoint', '--dry-run',
        '--summary', str(tmp_path / 'summary.json'),
        '--markdown', str(tmp_path / 'summary.md'),
        '--run-id', 'batch-policy-name',
    ]
    result = subprocess.run(command, cwd=tmp_path, env={
                                'PATH': os.environ['PATH'],
                                'SMOKE_TEST_ROOT_DIR': str(root),
                                'SMOKE_TEST_LOG_DIR': str(tmp_path / 'logs'),
                            }, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / 'logs/stack_bowls.log').read_text()
    assert 'policy=vap' in text
    assert str(package / 'setup_eval_env_client.sh') in text


@pytest.mark.parametrize('mode', ['sequential', 'sequential-resume', 'parallel-resume'])
def test_batch_split_shares_bootstrap_identity_and_reuses_group_run(tmp_path, mode):
    """Run the real split launchers up to an explicitly stubbed transport boundary."""
    resume = mode != 'sequential'
    parallel = mode == 'parallel-resume'
    source_root = PATH.parents[1].parents[2]
    policy = PATH.parents[1]
    package = tmp_path / 'XPolicyLab/policy/vap'
    package.mkdir(parents=True)
    (package / 'deploy.py').write_text('# test adapter marker\n')
    for name in ('common.sh', 'run_identity.py', 'setup_eval_env_client.sh'):
        (package / name).write_text((policy / name).read_text())
    (package / 'setup_eval_policy_server.sh').write_text(
        '#!/bin/bash\nset -Eeuo pipefail\n'
        'export VAP_LAUNCH_ROLE=server POLICY_ENV="$8"\n'
        'source "$(dirname "$0")/common.sh"\n'
        'printf "%s\\n%s\\n%s\\n" "$ROBODOJO_RUN_ID" "$VAP_RUN_DIR" "$PWD" '
        '> "$VAP_RUN_DIR/server-contract.txt"\n'
    )
    (tmp_path / 'XPolicyLab/utils').mkdir(parents=True)
    (tmp_path / 'XPolicyLab/utils/setup_env_client.sh').write_text(
        '#!/bin/bash\nset -Eeuo pipefail\n'
        '[[ "${ROBODOJO_BATCH_TRANSPORT_PROBE:-0}" == 1 ]] || exit 99\n'
        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n" '
        '"$ROBODOJO_RUN_ID" "$VAP_RUN_DIR" "$2" "$6" "$PWD" '
        '>> "$VAP_RUN_DIR/client-contract.txt"\n'
        '# Stub transport result: this is not a model or simulator episode.\n'
        'result="${10}/eval_result/RoboDojo/$6/vap/arx_x5/${11}_$9/$ROBODOJO_RUN_ID"\n'
        'mkdir -p "$result"\n'
        'printf \'{"eval_time": 1}\\n\' > "$result/_result.json"\n'
    )
    (tmp_path / 'scripts/internal').mkdir(parents=True)
    for name in ('robodojo.sh', 'internal/smoke_all_tasks.sh',
                 'internal/run_policy_server.sh', 'internal/task_inventory.py'):
        target = tmp_path / 'scripts' / name
        target.write_text((source_root / 'scripts' / name).read_text())
    # Inventory input is read-only; execution remains under the temporary root.
    (tmp_path / 'task').symlink_to(source_root / 'task', target_is_directory=True)
    policy_env = tmp_path / 'policy-env'
    sim_env = tmp_path / 'simulator-env'
    for environment_path in (policy_env, sim_env):
        (environment_path / 'bin').mkdir(parents=True)
        (environment_path / 'bin/python').symlink_to(Path(sys.executable).resolve())
    environment = {
        'PATH': os.environ['PATH'],
        'VAP_OUTPUT_ROOT': str(tmp_path / 'external-results'),
        'VAP_CACHE_ROOT': str(tmp_path / 'external-cache'),
        'VAP_QWEN_MODEL': str(tmp_path / 'qwen'),
        'ROBODOJO_SIM_ENV': str(sim_env),
        'ROBODOJO_BATCH_TRANSPORT_PROBE': '1',
        # Batch scheduling, not an inherited process identity, selects the group.
        'ROBODOJO_RUN_ID': 'unrelated-parent-identity',
    }
    shared = [
        '--policy-dir', str(package), '--policy-name', 'vap',
        '--only', 'stack_bowls,push_T' + (',general_pickup' if parallel else ''),
        '--ckpt', '/checkpoint', '--run-id', 'batch-group', '--eval-num', '1',
    ]
    workers = ['--gpu-ids', '0,1'] if parallel else []
    ports = '34567,34568' if parallel else '34567'
    previous_result = tmp_path / (
        'eval_result/RoboDojo/stack_bowls/vap/arx_x5/0_ckpt_name=/checkpoint,'
        'action_type=ee/batch-group_stack_bowls/_result.json'
    )
    if resume:
        # In the sequential case, skip the highest-weighted bootstrap task.
        previous_result.parent.mkdir(parents=True)
        previous_result.write_text('{"eval_time": 1}\n')
    server_command = [
        'bash', str(tmp_path / 'scripts/internal/smoke_all_tasks.sh'),
        '--mode', 'server', '--policy-env', str(policy_env),
        '--policy-port', ports, *workers, *shared,
    ]
    if resume:
        server_command.append('--resume')
        (tmp_path / 'smoke_results').mkdir()
        (tmp_path / 'smoke_results/batch-group_server.json').write_text(json.dumps({
            'counts': {'PASS': 1},
            'results': [{'status': 'PASS', 'task': 'stack_bowls'}],
        }))
    server = subprocess.run(server_command, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert server.returncode == 0, server.stdout + server.stderr
    runs = {}
    for contract in (tmp_path / 'external-results').glob('*/server-contract.txt'):
        identity, run_dir, cwd = contract.read_text().splitlines()
        assert identity == contract.parent.name
        assert run_dir == str(contract.parent)
        assert cwd == str(package)
        runs[identity.removeprefix('batch-group_')] = contract.parent
    assert set(runs) == ({'stack_bowls'} if not parallel else {'stack_bowls', 'push_T'})
    for task, run in runs.items():
        assert (run / 'server-contract.txt').read_text().splitlines() == [
            f'batch-group_{task}', str(run), str(package),
        ]
    client_summary = tmp_path / 'smoke_results/batch-group.json'
    # A successful long-lived server execs the policy and never returns a summary.
    assert not client_summary.exists()
    if resume:
        previous = {
            'run_id': 'batch-group', 'eval_num': 1, 'dimensions': ['all'],
            'counts': {'PASS': 1, 'FAIL': 0, 'SKIP': 0, 'DRY_RUN': 0},
            'results': [{
                'status': 'PASS', 'task': 'stack_bowls', 'exit_code': '0',
                'eval_time': '1', 'elapsed_sec': '1',
                'result_path': str(previous_result),
                'log_path': 'previous.log', 'message': 'ok',
            }],
        }
        client_summary.write_text(json.dumps(previous))
    client = subprocess.run([
        'bash', str(tmp_path / 'scripts/internal/smoke_all_tasks.sh'),
        '--mode', 'client', '--policy-host', '127.0.0.1',
        '--policy-port', ports, *workers,
        *(['--resume'] if resume else []), *shared,
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    logs = [path.read_text() for path in (tmp_path / 'smoke_results/batch-group/logs').glob('*.log')]
    assert client.returncode == 0, client.stdout + client.stderr + '\n'.join(logs)
    bootstrap = next(task for task, run in runs.items()
                     if (run / 'client-contract.txt').exists())
    active_tasks = ['push_T', 'general_pickup'] if parallel else ['push_T'] if resume else ['push_T', 'stack_bowls']
    lines = (runs[bootstrap] / 'client-contract.txt').read_text().splitlines()
    assert lines[0::5] == [f'batch-group_{bootstrap}'] * len(active_tasks)
    assert lines[1::5] == [str(runs[bootstrap])] * len(active_tasks)
    assert lines[2::5] == [str(package / 'deploy.yml')] * len(active_tasks)
    assert lines[3::5] == active_tasks
    assert lines[4::5] == [str(package)] * len(active_tasks)
    if parallel:
        assert not (runs['stack_bowls'] / 'client-contract.txt').exists()
    assert not (package / '.cache').exists()
    summary = json.loads(client_summary.read_text())
    assert summary['counts']['FAIL'] == 0
    assert summary['counts']['PASS'] == len(active_tasks) + int(resume)
    assert summary['counts']['SKIP'] == 0
    expected_tasks = {'push_T', 'stack_bowls'} | ({'general_pickup'} if parallel else set())
    assert {row['task'] for row in summary['results']} == expected_tasks
    for row in summary['results']:
        if row['status'] == 'PASS':
            result = tmp_path / Path(row['result_path']).relative_to(tmp_path)
            assert result.is_file()
            assert f'/RoboDojo/{row["task"]}/' in row['result_path']
            identity = 'stack_bowls' if resume and row['task'] == 'stack_bowls' else bootstrap
            assert f'/batch-group_{identity}/_result.json' in row['result_path']
    assert not (tmp_path / 'smoke_results/batch-group_push_T.json').exists()
    assert not (tmp_path / 'smoke_results/batch-group_stack_bowls.json').exists()


def test_batch_combined_parallel_resume_keeps_active_only_assignment(tmp_path):
    source_root = PATH.parents[1].parents[2]
    package = tmp_path / 'XPolicyLab/policy/vap'
    package.mkdir(parents=True)
    (package / 'eval.sh').write_text('# transport is not used by this dry-run\n')
    (tmp_path / 'scripts/internal').mkdir(parents=True)
    for name in ('robodojo.sh', 'internal/smoke_all_tasks.sh', 'internal/task_inventory.py'):
        target = tmp_path / 'scripts' / name
        target.write_text((source_root / 'scripts' / name).read_text())
    (tmp_path / 'task').symlink_to(source_root / 'task', target_is_directory=True)
    summary_dir = tmp_path / 'smoke_results'
    summary_dir.mkdir()
    (summary_dir / 'combined-resume.json').write_text(json.dumps({
        'results': [{'status': 'PASS', 'task': 'stack_bowls'}],
    }))
    result = subprocess.run([
        'bash', str(tmp_path / 'scripts/internal/smoke_all_tasks.sh'),
        '--mode', 'eval', '--policy-dir', str(package), '--policy-env', '/policy-env',
        '--gpu-ids', '0,1', '--only', 'stack_bowls,push_T,general_pickup',
        '--ckpt', '/checkpoint', '--run-id', 'combined-resume', '--resume', '--dry-run',
    ], cwd=tmp_path, env={'PATH': os.environ['PATH']}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'group=0' in result.stdout and 'group=1' in result.stdout
    summary = json.loads((summary_dir / 'combined-resume.json').read_text())
    assert {row['task'] for row in summary['results'] if row['status'] == 'DRY_RUN'} == {'push_T', 'general_pickup'}
    assert all('/combined-resume_stack_bowls/' not in row['result_path']
               for row in summary['results'] if row['status'] == 'DRY_RUN')


def test_official_eval_parent_shares_run_identity():
    """The unmodified parent must work without supplying any run identity."""
    policy = PATH.parents[1]
    source_root = policy.parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / 'RoboDojo/XPolicyLab/policy/vap'
        package.mkdir(parents=True)
        for name in ('common.sh', 'run_identity.py', 'config_env.py', 'deploy.yml'):
            (package / name).write_text((policy / name).read_text())
        (root / 'RoboDojo/scripts/internal').mkdir(parents=True)
        (root / 'RoboDojo/XPolicyLab/utils').mkdir(parents=True)
        launcher = subprocess.run(
            ['git', 'show', 'HEAD:scripts/internal/run_policy_eval.sh'],
            cwd=source_root, check=True, capture_output=True, text=True,
        ).stdout
        (root / 'RoboDojo/scripts/internal/run_policy_eval.sh').write_text(launcher)
        (root / 'RoboDojo/XPolicyLab/utils/get_free_port.sh').write_text('#!/bin/bash\necho 34567\n')
        # The server needs three minutes to load; the parent starts the client later.
        (root / 'RoboDojo/XPolicyLab/utils/wait_for_policy_server.sh').write_text(
            '#!/bin/bash\nsleep "${RECORD_DELAY:-0}"\n'
            'for _ in {1..200}; do\n'
            '  [[ ! -s "$RECORD_DIR/server.$3" ]] || exit 0\n'
            '  kill -0 "$3" 2>/dev/null || exit 2\n'
            '  read -r _ _ state _ < "/proc/$3/stat" || exit 2\n'
            '  [[ "$state" != Z ]] || exit 2\n'
            '  sleep 0.05\n'
            'done\n'
            'echo "Fixture server did not become ready" >&2; exit 2\n'
        )
        policy_env = root / 'policy-env'
        sim_env = root / 'sim-env'
        for environment in (policy_env, sim_env):
            (environment / 'bin').mkdir(parents=True)
            (environment / 'bin/python').symlink_to(Path(sys.executable).resolve())
        server = '''#!/bin/bash
set -Eeuo pipefail
export VAP_LAUNCH_ROLE=server POLICY_ENV="$8"
source "$(dirname "$0")/common.sh"
printf "%s\\n%s\\n" "$ROBODOJO_RUN_ID" "$VAP_RUN_DIR" > "$RECORD_DIR/server.$$"
printf "%s\\n%s\\n" "$ROBODOJO_RUN_ID" "$VAP_RUN_DIR" > "$RECORD_DIR/server"
sleep 220
'''
        client = '''#!/bin/bash
set -Eeuo pipefail
export VAP_LAUNCH_ROLE=client POLICY_ENV="${POLICY_ENV:-$8}"
source "$(dirname "$0")/common.sh"
printf "%s\\n%s\\n" "$ROBODOJO_RUN_ID" "$VAP_RUN_DIR" > "$RECORD_DIR/client"
'''
        (package / 'setup_eval_policy_server.sh').write_text(server)
        (package / 'setup_eval_env_client.sh').write_text(client)
        record = root / 'record'
        record.mkdir()
        environment = {
            'PATH': os.environ['PATH'], 'RECORD_DIR': str(record),
            'POLICY_ENV': str(policy_env), 'ROBODOJO_SIM_ENV': str(sim_env),
            'VAP_QWEN_MODEL': '/explicit/qwen', 'VAP_OUTPUT_ROOT': str(root / 'results'),
            'VAP_CACHE_ROOT': str(root / 'cache'),
            'RECORD_DELAY': os.environ.get('RECORD_DELAY', '181'),
        }
        command = ['bash', str(root / 'RoboDojo/scripts/internal/run_policy_eval.sh'), str(package),
                   'RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee', '0', '0', '0',
                   str(policy_env), str(sim_env)]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        identity = (record / 'server').read_text().splitlines()
        assert identity[0].startswith('vap_stack_bowls_seed0_')
        assert (record / 'client').read_text().splitlines() == identity
        run = Path(identity[1])
        assert run.parent == root / 'results'
        assert (run / '.vap_run_id').read_text() == identity[0] + '\n'
        signature = (run / '.vap_request.json').read_text()
        stored_request = json.loads(signature)
        assert stored_request['checkpoint'] == '/checkpoint'
        assert stored_request['mode'] == 'sim'
        assert 'eval_num' not in stored_request
        assert 'task' not in stored_request
        assert (run / 'task.txt').read_text() == 'stack_bowls\n'
        assert '[MAIN] eval finished' in result.stdout

        # Only the first launch needs to reproduce the three-minute model startup.
        environment['RECORD_DELAY'] = '1'
        # Depending on readiness/server scheduling, the unmodified parent exits
        # 2 or masks the client failure with cleanup's return 0. Never accept 1.
        repeated = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
        assert repeated.returncode in {0, 2}, repeated.stdout + repeated.stderr
        assert 'Run exists:' in repeated.stderr
        assert 'Missing VAP run identity' not in repeated.stderr
        assert (run / '.vap_request.json').read_text() == signature
        assert (run / '.vap_run_id').read_text() == identity[0] + '\n'
        for position, value in ((2, 'push_T'), (3, '/other-checkpoint'), (4, 'other_robot'),
                                (5, 'joint'), (6, '1')):
            changed = command.copy()
            changed[position + 2] = value
            different = subprocess.run(changed, env=environment, capture_output=True, text=True, timeout=30)
            assert different.returncode in {0, 2}, different.stdout + different.stderr
            assert 'Run exists:' not in different.stderr
            assert (record / 'server').read_text().splitlines()[0] != identity[0]
        changed_env = dict(environment, EVAL_ENV_TYPE='debug')
        different = subprocess.run(command, env=changed_env, capture_output=True, text=True, timeout=30)
        assert different.returncode == 0, different.stderr
        assert (record / 'client').read_text().splitlines()[0] != identity[0]
        counted = subprocess.run(command, env=dict(environment, EVAL_NUM='1'),
                                 capture_output=True, text=True, timeout=30)
        assert counted.returncode == 0, counted.stderr
        assert (record / 'client').read_text().splitlines()[0] != identity[0]
        relative = command.copy()
        relative[5] = './relative-checkpoint'
        first = subprocess.run(relative, env=environment, capture_output=True, text=True, timeout=30)
        assert first.returncode == 0, first.stderr
        assert (record / 'client').read_text().splitlines() == (record / 'server').read_text().splitlines()
        relative_run = Path((record / 'client').read_text().splitlines()[1])
        assert json.loads((relative_run / '.vap_request.json').read_text())['checkpoint'] == str(package / 'relative-checkpoint')
        assert (relative_run / 'task.txt').read_text() == 'stack_bowls\n'

        # Explicit identities still follow the pristine parent unchanged.
        explicit = dict(environment, ROBODOJO_RUN_ID='explicit-smoke-id')
        result = subprocess.run(command, env=explicit, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (record / 'server').read_text().splitlines() == [
            'explicit-smoke-id', str(root / 'results/explicit-smoke-id'),
        ]
        assert (record / 'client').read_text() == (record / 'server').read_text()
        dry = subprocess.run(command, env=dict(environment, VAP_DRY_RUN='1'),
                             capture_output=True, text=True, timeout=240)
        assert dry.returncode == 0, dry.stderr
        assert not (root / 'dry-cache').exists()
        identities = {path.read_text() for path in (root / 'results').glob('*/.vap_run_id')}
        assert len(identities) == 10
        assert 'explicit-smoke-id\n' in identities

        # An early server failure is observable even when the pristine parent
        # subsequently masks the client status. No stale readiness file is used.
        environment['VAP_OUTPUT_ROOT'] = str(root / 'immediate-results')
        immediate_env = dict(environment, ROBODOJO_RUN_ID='immediate-exit')
        immediate = subprocess.run(command, env=immediate_env,
                                   capture_output=True, text=True, timeout=30)
        assert immediate.returncode == 0, immediate.stderr
        immediate = subprocess.run(command, env=immediate_env,
                                   capture_output=True, text=True, timeout=30)
        assert 'Run exists:' in immediate.stderr
        assert 'Missing VAP run identity' not in immediate.stderr
        assert (root / 'immediate-results/immediate-exit/.vap_run_id').read_text() == 'immediate-exit\n'


def test_direct_eval_uses_shared_owner_server_and_client_lifecycle():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / 'RoboDojo/XPolicyLab/policy/vap'
        package.mkdir(parents=True)
        (root / 'RoboDojo/XPolicyLab/utils').mkdir(parents=True)
        (root / 'RoboDojo/env_cfg').mkdir()
        (root / 'RoboDojo/env_cfg/arx_x5.yml').write_text('robot: test\n')
        for filename in ('eval.sh', 'common.sh', 'run_identity.py', 'config_env.py', 'deploy.yml'):
            (package / filename).write_text((policy / filename).read_text())
        (root / 'RoboDojo/XPolicyLab/utils/get_free_port.sh').write_text(
            (policy.parents[1] / 'utils/get_free_port.sh').read_text(),
        )
        (root / 'RoboDojo/XPolicyLab/utils/wait_for_policy_server.sh').write_text(
            (policy.parents[1] / 'utils/wait_for_policy_server.sh').read_text(),
        )
        policy_env = root / 'policy-env'
        sim_env = root / 'sim-env'
        for environment in (policy_env, sim_env):
            subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
            (environment / 'pyvenv.cfg').write_text(
                f'home = {Path(sys.prefix) / "bin"}\ninclude-system-site-packages = true\n',
            )
        for role in ('server', 'client'):
            filename = 'setup_eval_policy_server.sh' if role == 'server' else 'setup_eval_env_client.sh'
            (package / filename).write_text(
                f'#!/usr/bin/env bash\n'
                f'set -Eeuo pipefail\n'
                f'export VAP_LAUNCH_ROLE={role}\n'
                f'[[ $# -eq {9 if role == "server" else 11} ]] || '
                f'{{ echo "wrong {role} argument count: $#"; exit 2; }}\n'
                f'POLICY_ENV="$8"\n'
                f'source "$(dirname "$0")/common.sh"\n'
                f'printf "%s\\n" "$VAP_RUN_DIR" >> "$VAP_RUN_DIR/{role}_args.txt"\n'
                f'printf "%s\\n" "$@" >> "$VAP_RUN_DIR/{role}_args.txt"\n'
                + (f'exec {Path(sys.executable).resolve()} -I -c '
                   '"import socket,sys,time; s=socket.socket(); '
                   's.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); '
                   's.bind((\'localhost\',int(sys.argv[1]))); s.listen(); time.sleep(30)" "$9"\n'
                   if role == 'server' else ''),
            )
        environment = {key: value for key, value in os.environ.items()
                       if key not in {'ROBODOJO_RUN_ID', 'VAP_RUN_DIR', 'VAP_RUN_INITIALIZED'}}
        environment.update(VAP_QWEN_MODEL='/explicit/qwen', VAP_OUTPUT_ROOT=str(root / 'results'),
                           VAP_CACHE_ROOT=str(root / 'cache'), ROBODOJO_RUN_ID='direct-owner')
        command = [str(package / 'eval.sh'), 'RoboDojo', 'stack_bowls', '/checkpoint',
                   'arx_x5', 'ee', '0', '0', '0', str(policy_env), str(sim_env)]
        result = subprocess.run(['bash', *command], env=environment,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        run = root / 'results/direct-owner'
        assert (run / '.vap_run_id').read_text() == 'direct-owner\n'
        request = json.loads((run / 'request.json').read_text())
        assert request['task'] == 'stack_bowls'
        assert request['policy_env'] == str(policy_env)
        assert request['sim_env'] == str(sim_env)
        assert (run / 'exit_code.txt').read_text() == '0'
        for role in ('server', 'client'):
            lines = (run / f'{role}_args.txt').read_text().splitlines()
            assert lines[0] == str(run)
            assert lines[1:7] == ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee', '0']
            if role == 'client':
                assert lines[1:] == ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee',
                                     '0', '0', str(sim_env), 'ckpt_name=/checkpoint,action_type=ee',
                                     lines[-2], 'localhost']
                assert lines[-2].isdigit()
            else:
                assert lines[1:] == ['RoboDojo', 'stack_bowls', '/checkpoint', 'arx_x5', 'ee',
                                     '0', '0', str(policy_env), lines[-1]]
                assert lines[-1].isdigit()
        assert '[MAIN] eval finished' in result.stdout
        assert 'Run exists:' not in result.stdout + result.stderr


def test_wan_audit_is_installed_before_server_import():
    policy = PATH.parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / 'source/XPolicyLab/policy/vap'
        package.mkdir(parents=True)
        for parent in (package.parent, package.parent.parent, package.parent.parent.parent):
            (parent / '__init__.py').write_text('')
        run_dir = root / 'run'
        run_dir.mkdir()
        env = dict(os.environ, VAP_SOURCE_PATH=str(root / 'source'),
                   VAP_RUN_DIR=str(run_dir), WALLX_MODEL_ROOT=str(root),
                   VAP_WAN22_TOKENIZER_PATH=str(root / 't5'))
        server = package / 'server.py'
        server.write_text('print("SERVER_IMPORTED", flush=True)\n')
        result = subprocess.run([sys.executable, '-I', str(policy / 'audit_no_wan.py')],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert 'SERVER_IMPORTED' in result.stdout
        report = json.loads((run_dir / 'wan_free_server.json').read_text())
        assert report['status'] == 'PASS' and report['attempts'] == []
        server.write_text('from pathlib import Path\nPath("Wan2.2_VAE.pth").read_bytes()\n')
        result = subprocess.run([sys.executable, '-I', str(policy / 'audit_no_wan.py')],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert 'SERVER_IMPORTED' not in result.stdout
        report = json.loads((run_dir / 'wan_free_server.json').read_text())
        assert report['status'] == 'FAIL'
        assert report['attempts'] == ['Wan2.2_VAE.pth']
        assert 'Wan asset access attempted' in report['error']


def test_piptools_continuations_and_interpreter_origin():
    with tempfile.TemporaryDirectory() as tmp:
        lock = Path(tmp) / 'requirements.lock'
        lock.write_text('torch @ https://download.pytorch.org/whl/torch-2.12.1%2Bcu130-cp311-cp311-linux.whl'
                        '#sha256=' + 'a' * 64 + ' \\\n    --hash=sha256:' + 'a' * 64 + '\n')
        assert installer.locked_versions(lock) == {'torch': '2.12.1+cu130'}
        assert installer.locked_hashes(lock) == {'torch': 'a' * 64}
    env = {'python_prefix': '/new/venv', 'sys_executable': '/new/venv/bin/python',
           'resolved_executable': '/usr/bin/python3.11', 'sys_path': [], 'packages': {}}
    allowed = (Path('/new/venv/lib'), Path('/workspace/RoboDojo'))
    external = {}
    for name, path in env.items():
        if name in {'python_prefix', 'sys_executable', 'resolved_executable', 'sys_path', 'packages'}:
            continue
        if isinstance(path, str) and path.startswith('/'):
            source = Path(path).resolve()
            if not any(source.is_relative_to(parent) for parent in allowed):
                external[name] = path
    assert not external


def test_report_accepts_standard_venv_interpreter():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'venv'
        (target / 'bin').mkdir(parents=True)
        (target / 'bin/python').symlink_to(sys.executable)
        lock = Path(tmp) / 'requirements.lock'
        lock.write_text('torch==2.12.1+cu130 \\\n    --hash=sha256:' + 'a' * 64 + '\n')
        data = {'python_prefix': str(target), 'sys_executable': str(target / 'bin/python'),
                'resolved_executable': str(Path(sys.executable).resolve()), 'sys_path': [], 'python': '3.11.18',
                'packages': {'torch': '2.12.1+cu130'}, 'torch_version': '2.12.1+cu130', 'cuda': '13.0'}
        args = SimpleNamespace(target=target, gpu=False, report=None)
        with mock.patch.object(installer, 'LOCK', lock), \
             mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'module_report', return_value=data), \
             mock.patch.object(installer, 'run'):
            installer.report(args)


def test_no_replace_publication():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lock = root / 'requirements.lock'
        lock.write_text('numpy==2.2.6 --hash=sha256:' + '0' * 64 + '\n')
        target = root / 'new-env'
        args = SimpleNamespace(target=target, python=sys.executable, index_url='unused')

        def competing_target(*command, **kwargs):
            if '--requirement' in command[0]:
                raise FileExistsError('simulated concurrent target')
            if '--prefix' in command[0]:
                prefix = Path(command[0][command[0].index('--prefix') + 1])
                packages = prefix / 'lib/python3.11/site-packages'
                packages.mkdir(parents=True)
                (packages / 'demo.py').write_text('# test fixture\n')

        with mock.patch.object(installer, 'check_interpreter'), \
             mock.patch.object(installer, 'LOCK', lock), \
             mock.patch.object(installer, 'public_environ', return_value={}), \
             mock.patch.object(installer, 'bootstrap_pip'), \
             mock.patch.object(installer, 'report'), \
             mock.patch.object(installer, 'run', side_effect=competing_target):
            try:
                installer.install(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError('concurrent target failure was ignored')
        assert not target.exists()
        assert not list(root.glob('new-env-installing-*'))


def test_flock_rejects_a_second_installer():
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / 'install.lock'
        code = '''import fcntl,sys,time
with open(sys.argv[1],"a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print("locked",flush=True)
 time.sleep(30)
'''
        with subprocess.Popen([sys.executable, '-I', '-c', code, lock_path],
                              stdout=subprocess.PIPE, text=True) as owner:
            assert owner.stdout.readline().strip() == 'locked'
            with lock_path.open('a') as competing:
                try:
                    import fcntl
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError('a second installer acquired the active lock')
            owner.terminate()
            owner.wait(timeout=10)


def test_staged_pip_preserves_final_venv_identity():
    """Exercise real bootstrap and staged pip installs with an official wheel."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / 'final-env'
        python = target / 'bin/python'
        version = installer.locked_versions(installer.HERE / 'bootstrap.lock')['pip']
        expected = installer.locked_hashes(installer.HERE / 'bootstrap.lock')['pip']
        metadata_url = f'https://pypi.org/pypi/pip/{version}/json'
        with installer.urllib.request.urlopen(metadata_url, timeout=60) as response:
            metadata_bytes = response.read()
        metadata = json.loads(metadata_bytes)
        name = f'pip-{version}-py3-none-any.whl'
        record = next(item for item in metadata['urls'] if item['filename'] == name)
        wheel_url = record['url']
        # An optional previously downloaded *public* wheel avoids a duplicate
        # transfer in validation; a clean test run fetches the same PyPI URL.
        fixture = os.environ.get('VAP_TEST_PIP_WHEEL')
        if fixture:
            payload = Path(fixture).read_bytes()
        else:
            with installer.urllib.request.urlopen(wheel_url, timeout=60) as response:
                payload = response.read()
        assert len(payload) == record['size']
        assert hashlib.sha256(payload).hexdigest() == expected
        with mock.patch.dict(os.environ, {'VAP_INSTALL_CACHE': str(root / 'cache')}):
            env = installer.public_environ()
            installer.run([sys.executable, '-I', '-m', 'venv', '--without-pip', target])

            class Response(io.BytesIO):
                status = 200
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.close()

            def offline_urlopen(request, timeout):
                url = request.full_url if hasattr(request, 'full_url') else request
                if url == metadata_url:
                    return Response(metadata_bytes)
                if url == wheel_url:
                    return Response(payload)
                raise AssertionError(f'unexpected network access: {url}')

            def offline_bootstrap(command, **kwargs):
                # The complete round-04 install exercised the other bootstrap
                # dependencies; this regression isolates final interpreter paths.
                if '--requirement' in command:
                    assert Path(command[command.index('--requirement') + 1]).name == 'bootstrap.lock'
                    return
                if 'uninstall' in command:
                    return
                subprocess.run(command, check=True, **kwargs)

            with mock.patch.object(installer.urllib.request, 'urlopen',
                                   side_effect=offline_urlopen), \
                 mock.patch.object(installer, 'run', side_effect=offline_bootstrap):
                installer.bootstrap_pip(python, target, env)
            wheel = Path(env['PIP_CACHE_DIR']) / name
            assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected
            installer.run([python, '-I', '-m', 'pip', '--python', python, 'install',
                           '--ignore-installed', '--prefix', root / 'staged',
                           '--no-deps', '--no-index', wheel], env=env)
        command = (root / 'staged/bin/pip').read_text()
        assert command.startswith(f'#!{python}\n')
        activate = (target / 'bin/activate').read_text()
        assert f'VIRTUAL_ENV={target}' in activate
        assert '-installing-' not in command


def run_all():
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            function()
            print('PASS', name)
    print('PASS public installer preflight, retry and launcher isolation')


if __name__ == '__main__':
    run_all()
