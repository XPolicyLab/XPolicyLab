"""Load the existing WAM adapter without process-wide import monkeypatches."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configuration(mode, overrides, root=ROOT):
    root = Path(root)
    base = root / 'policies/riemann_1'
    spec = json.loads((base / 'policy.json').read_text())
    if mode not in spec['tasks']:
        raise ValueError('Task is outside the registered WAM policy')
    from .deployment import model_options
    from .resource_locations import relocate_path
    from XPolicyLab.utils.checkpoint_resolver import ckpt_name_is_path, resolve_checkpoint_root
    overrides = dict(model_options(root), **overrides)
    deployed = spec['assets']
    checkpoint = overrides.get('wam_checkpoint') or overrides.get('checkpoint_path')
    if checkpoint is None:
        requested = overrides.get('ckpt_name')
        checkpoint = requested if ckpt_name_is_path(requested) else deployed['checkpoint']
    # This eval-only release declares one published checkpoint root. Preserve
    # its explicit binding; the official helper owns path expansion/relocation.
    checkpoint = resolve_checkpoint_root({'checkpoint_path': checkpoint}, root / 'checkpoints',
                                         policy_dir=root, must_exist=False)
    cfg = dict(spec['inference'])
    cfg.update(policy_name=spec['adapter_name'], task_name=mode, bench_name='RoboDojo',
               seed=overrides.get('seed', 0), ckpt_name=spec['step'],
               env_cfg_type='arx_x5', action_type='ee', exp_dir=str(checkpoint),
               model_path=str(checkpoint / spec['model_directory']),
               norm_stats_path=str(checkpoint / spec['norm_stats']),
               train_config_path=str(checkpoint / spec['train_config']),
               skyrobot_root=str(base / 'implementation'),
               encoder_path=relocate_path(deployed['encoder']), device='cuda:0', t5_device='cuda:1')
    # Only documented policy-side settings can replace inference defaults.
    for key in (*spec['inference'], 'encoder_path', 'device', 't5_device'):
        if key in overrides:
            cfg[key] = overrides[key]
    return cfg


def create(mode, declaration, config):
    if config.get('action_type', 'ee') != 'ee':
        raise ValueError('WAM retains its original ee action space; use action_type=ee')
    from rsi.skills.runtime import Library
    from .model_records import verify_records
    source_spec = json.loads((ROOT / 'policies/riemann_1/policy.json').read_text())
    cfg = configuration(mode, config)
    from .model_assets import verify as verify_assets
    verify_assets(wam_checkpoint=cfg['exp_dir'], encoder_path=cfg['encoder_path'], root=ROOT)
    verify_records(Path(cfg['exp_dir']), source_spec['checkpoint_files'])
    library = Library(ROOT, mode)
    filename = ROOT / 'policies/riemann_1/adapter/riemann_1.0_robodojo/model.py'
    name = '_riemann_wam_adapter_' + mode
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__riemann_skills__'] = library
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module.Model(cfg)
    except BaseException:
        sys.modules.pop(name, None)
        library.trace.close()
        raise
