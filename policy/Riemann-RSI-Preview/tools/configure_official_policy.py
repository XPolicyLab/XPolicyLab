"""Install an isolated policy environment and bind verified external locations."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def environment_commands(target, base_python, root=ROOT):
    target = Path(target)
    if target.exists():
        raise ValueError('The new policy environment must not already exist')
    return [[str(base_python), '-m', 'venv', str(target)],
            [str(target / 'bin/python'), '-m', 'pip', 'install', '--require-virtualenv',
             '-r', str(Path(root) / 'requirements.txt')]]


@contextmanager
def configured_environment(binding):
    from rsi.resource_locations import overrides, aliases
    values = dict(RIEMANN_OFFICIAL_ROOT=binding['official_runtime']['official_root'],
                  RIEMANN_EVAL_PYTHON=binding['official_runtime']['python'],
                  RIEMANN_POLICY_PYTHON=binding['policy_python'],
                  RIEMANN_RESOURCES_BASE=binding['resources_base'],
                  RIEMANN_RESOURCE_ROOTS=json.dumps(binding['resource_roots']),
                  RIEMANN_RESOURCE_ALIASES=json.dumps(binding.get('resource_aliases', {})))
    for key, variable in (('wam_checkpoint', 'RIEMANN_WAM_CHECKPOINT'),
                          ('encoder_path', 'RIEMANN_WAM_ENCODER')):
        if key in binding: values[variable] = binding[key]
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        overrides()  # Reject unknown resource IDs before installation.
        aliases()
        yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def python_identity(executable):
    # Python 3.11 discovers a venv prefix in site initialization; -S would
    # incorrectly identify its base installation as the selected environment.
    result = subprocess.check_output([str(executable), '-I', '-B', '-c',
        'import json,sys; print(json.dumps(dict(version=sys.version.split()[0], prefix=sys.prefix)))'], text=True, timeout=30)
    return json.loads(result)



def policy_preflight(executable, root=ROOT):
    """Import the actual server/model code and compile slots, without weights."""
    code = r"""import importlib.metadata as metadata, importlib.util, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); sys.path[:0]=[str(root),str(root/'policies/riemann_1/implementation')]
from tools.serve_official_policy import configure_imports
configure_imports()
import torch, torchvision, diffusers, transformers
from skyrobot.deploy.policy import SkyrobotWAMPolicy
from skyrobot.deploy.model_dp import SkyrobotWAMForInference
from rsi.skills.runtime import Library
library=Library(root,'stack_bowls')
path=root/'policies/riemann_1/adapter/riemann_1.0_robodojo/model.py'
spec=importlib.util.spec_from_file_location('_riemann_install_probe',path); module=importlib.util.module_from_spec(spec)
module.__dict__['__riemann_skills__']=library; spec.loader.exec_module(module); library.trace.close()
versions={}
for line in (root/'policies/riemann_1/implementation/requirements.txt').read_text().splitlines():
    if line and not line.startswith(('#','-')):
        name,expected=line.split('==',1); actual=metadata.version(name)
        if actual!=expected:raise ValueError('Declared policy dependency differs: '+name+' '+actual+' != '+expected)
        versions[name]=actual
assert not torch.cuda.is_initialized()
print(json.dumps(dict(versions=versions,slots_compiled=len(library.bindings),models_loaded=False,cuda_initialized=False)))
"""
    output = subprocess.check_output([str(executable),'-B','-c',code,str(root)],
        text=True, timeout=180, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
    return json.loads(output.strip().splitlines()[-1])

def main(argv=None):
    from rsi.deployment import absolute, locations, native_library_dirs
    current = locations(ROOT)
    from rsi.deployment import model_options
    defaults = model_options(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robodojo', type=Path, default=Path(current['official_root']))
    parser.add_argument('--eval-python', type=Path, default=Path(current['python']))
    parser.add_argument('--policy-python', type=Path, help='Bind an existing WAM Python without changing it')
    parser.add_argument('--create-policy-env', type=Path, help='Create a new, isolated WAM/server venv')
    parser.add_argument('--base-python', type=Path, help='Python 3.11 used to create the new policy venv')
    parser.add_argument('--wam-checkpoint', type=Path, default=Path(defaults.get('wam_checkpoint', ROOT / 'policies/riemann_1/checkpoint')))
    parser.add_argument('--wam-encoder', type=Path, default=Path(defaults.get('encoder_path', current['encoder_path'])))
    parser.add_argument('--resources-base', type=Path, default=Path(current['resources_base']))
    parser.add_argument('--resource-root', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--native-library-dir', action='append', default=None, metavar='PATH',
                        help='Existing simulator runtime library directory; never installed or modified')
    parser.add_argument('--cluster', choices=('local','nx'), default=current['cluster'])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Print locations and install commands without changing files')
    args = parser.parse_args(argv)
    if args.policy_python and args.create_policy_env: parser.error('Choose existing --policy-python or --create-policy-env')
    if args.create_policy_env and args.check_only: parser.error('--check-only never creates environments; use --plan to inspect commands')
    try:
        sdk = absolute(args.robodojo).resolve(); evaluation = absolute(args.eval_python)
        policy = absolute(args.policy_python or (args.create_policy_env / 'bin/python' if args.create_policy_env else current['policy_python']))
        from rsi.deployment import local
        resource_roots = dict(local(ROOT).get('resource_roots', {}))
        inherited_roots = json.loads(os.environ.get('RIEMANN_RESOURCE_ROOTS', '{}'))
        if not isinstance(inherited_roots, dict): raise ValueError('Resource roots must be a location mapping')
        resource_roots.update(inherited_roots)
        for value in args.resource_root:
            name, separator, location = value.partition('=')
            if not separator or not name: raise ValueError('Use --resource-root NAME=/absolute/path')
            resource_roots[name] = str(absolute(location))
        binding = dict(official_runtime=dict(official_root=str(sdk), python=str(evaluation)),
                       policy_python=str(policy), resources_base=str(absolute(args.resources_base)),
                       resource_roots=resource_roots, cluster=args.cluster)
        resource_aliases = dict(local(ROOT).get('resource_aliases', {}))
        inherited_aliases = json.loads(os.environ.get('RIEMANN_RESOURCE_ALIASES', '{}'))
        if not isinstance(inherited_aliases, dict): raise ValueError('Resource aliases must be a location mapping')
        resource_aliases.update(inherited_aliases)
        if resource_aliases: binding['resource_aliases'] = resource_aliases
        if local(ROOT).get('slurm_bin'): binding['slurm_bin'] = local(ROOT)['slurm_bin']
        native = args.native_library_dir if args.native_library_dir is not None else native_library_dirs(ROOT)
        if native: binding['native_library_dirs'] = [str(absolute(value)) for value in native]
        for key, value in (('wam_checkpoint',args.wam_checkpoint),('encoder_path',args.wam_encoder)):
            if value is not None: binding[key] = str(absolute(value).resolve())
        commands = environment_commands(absolute(args.create_policy_env), args.base_python or evaluation) if args.create_policy_env else []
        with configured_environment(binding):
            if args.plan:
                print(json.dumps(dict(binding=binding, install_commands=commands, files_changed=False), indent=2));return 0
            output = ROOT / 'deploy.local.json'
            if not args.check_only and output.exists(): raise ValueError('deploy.local.json already exists; review it explicitly instead of overwriting')
            runtime = json.loads((ROOT / 'configs/official-rule-evaluation.json').read_text())
            actual = subprocess.check_output(['git','-c','safe.directory='+str(sdk),'-C',str(sdk),'rev-parse','HEAD'],text=True).strip()
            if actual != runtime['official_commit']: raise ValueError('Use the declared official SDK revision')
            identity = python_identity(evaluation)
            if identity['version'] != runtime['official_python_version']: raise ValueError('Use official evaluator Python '+runtime['official_python_version'])
            if Path(identity['prefix']).resolve() != evaluation.parent.parent.resolve():
                raise ValueError('Select the evaluator environment own bin/python, not an unrelated wrapper')
            if not commands and not policy.is_file(): raise ValueError('Missing policy Python: '+str(policy))
            from tools.official_rule_integrity import verify_policy
            from rsi.task_contract import verify
            from rsi.policy_catalog import catalog
            from rsi.model_assets import verify as verify_assets, verify_rule
            proof = dict(runtime_inputs_checked=verify_policy(sdk=sdk),
                         task_contracts={mode:verify(ROOT,mode) for mode in catalog()},
                         model_preflight=verify_assets(wam_checkpoint=args.wam_checkpoint, encoder_path=args.wam_encoder, root=ROOT),
                         rule_model_preflight=verify_rule(root=ROOT),
                         evaluator_python=identity, source_versioning='git', models_loaded=False)
            # Only a newly created environment is mutated. The evaluator and
            # any existing policy environment remain read-only.
            if commands:
                base = python_identity(args.base_python or evaluation)
                if not base['version'].startswith('3.11.'): raise ValueError('The pinned model wheels require Python 3.11')
                target = absolute(args.create_policy_env).resolve()
                if target.is_relative_to(sdk) or target.is_relative_to(evaluation.parent.parent.resolve()):
                    raise ValueError('Create the policy venv outside the SDK and evaluator environment')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir()  # Atomically reserve the new directory; never reuse a racing writer.
                for command in commands: subprocess.run(command, check=True)
                proof['policy_environment_created'] = str(args.create_policy_env)
            proof['policy_python'] = python_identity(policy)
            if not proof['policy_python']['version'].startswith('3.11.'):
                raise ValueError('The supported policy environment uses Python 3.11')
            proof['policy_preflight'] = policy_preflight(policy)
            if not args.check_only:
                descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor,'w') as stream: json.dump(binding,stream,indent=2);stream.write('\n')
                proof['local_binding'] = str(output)
            print(json.dumps(proof,indent=2))
        return 0
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
