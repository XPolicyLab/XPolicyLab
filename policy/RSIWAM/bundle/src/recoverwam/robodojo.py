"""Portable RoboDojo release configuration and checkpoint preflight."""
import json
import hashlib
import math
from pathlib import Path
import re


INFERENCE = dict(shift=1.0, steps=10, horizon=32, replan=24, history_stride=25, seed=1)


def evaluation_settings(config):
    settings = {**INFERENCE, **config.get('inference', {})}
    if not math.isfinite(float(settings['shift'])) or float(settings['shift']) <= 0:
        raise ValueError('Inference shift must be finite and positive')
    if int(settings['steps']) != settings['steps'] or settings['steps'] < 1:
        raise ValueError('Inference steps must be a positive integer')
    for key in ['horizon', 'replan', 'history_stride', 'seed']:
        if settings[key] != INFERENCE[key]:
            raise ValueError(f'RoboDojo reference {key} must be {INFERENCE[key]}')
    if config.get('seed', 1) != 1 or config.get('policy_name', 'RecoverWAM') != 'RecoverWAM':
        raise ValueError('Expected seed 1 and the RecoverWAM adapter')
    return settings


def validate_bundle(bundle, expected_step):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    tag = manifest.get('format', '')
    suffix = '-robodojo-bundle-v1'
    if not isinstance(tag, str) or not tag.endswith(suffix) or not tag[:-len(suffix)].isidentifier():
        raise ValueError('Unsupported RecoverWAM bundle manifest')
    if manifest.get('step') != expected_step:
        raise ValueError(f'Expected checkpoint step {expected_step}, got {manifest.get("step")}')
    for name in ['action.pt', 'video.pt', 'dataset_stats.json']:
        path = bundle / name
        size = manifest.get('files', {}).get(name)
        if not isinstance(size, int) or size <= 0 or path.stat().st_size != size:
            raise ValueError(f'Bundle file size mismatch: {name}')
    stats = json.loads((bundle / 'dataset_stats.json').read_text())
    if not isinstance(stats, dict) or not stats:
        raise ValueError('Missing normalization statistics')
    # The policy loader subsequently verifies tensor shapes and matching Video/Action steps.
    return manifest


def server_settings(args):
    config = {}
    if args.eval_config:
        config = json.loads(Path(args.eval_config).read_text())
        if Path(config['robodojo_root']).resolve() != Path(args.robodojo_root).resolve():
            raise ValueError('Server and evaluation use different RoboDojo roots')
        if args.port not in {int(s['policy_port']) for s in config['slots']}:
            raise ValueError('Server port is absent from the evaluation slots')
        if config.get('checkpoint_step') != args.expected_step:
            raise ValueError('Server and evaluation checkpoint steps differ')
    settings = evaluation_settings(config)
    for key in ['shift', 'steps']:
        value = getattr(args, key)
        if value is not None:
            if args.eval_config and value != settings[key]:
                raise ValueError(f'Server {key} conflicts with evaluation config')
            settings[key] = value
    settings = evaluation_settings({'inference': settings})
    return dict(settings, checkpoint_step=args.expected_step,
                checkpoint_label=config.get('ckpt_name', f'joint2b_s{args.expected_step:06d}'))


def write_eval_config(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('Use a new run ID containing only letters, digits, _, - and .')
    if args.sim_gpu < 0 or not 1 <= args.policy_port <= 65535 or args.step <= 0:
        raise ValueError('Invalid GPU, policy port or checkpoint step')
    config = json.loads(Path(args.template).read_text())
    config.update(
        run_id=args.run_id,
        output_root=str(Path(args.output_root or Path('runs') / args.run_id).resolve()),
        robodojo_root=str(Path(args.robodojo_root).resolve()),
        sim_python=str(Path(args.sim_python).absolute()),
        tasks_file=str(Path(config['tasks_file']).resolve()),
        official_tasks_file=str(Path(config['official_tasks_file']).resolve()),
        sim_ssh_host='local', sim_ssh_port=22,
        ckpt_name=f'joint2b_s{args.step:06d}', checkpoint_step=args.step,
        session_prefix=args.run_id,
        inference={**INFERENCE, 'shift': args.shift, 'steps': args.steps},
        slots=[dict(gpu=args.sim_gpu, policy_port=args.policy_port)],
        sim_bin_dirs=[str(Path(p).resolve()) for p in args.sim_bin_dir],
        topology='external_single_client',
    )
    if getattr(args, 'recovery_adapter', None):
        adapter = Path(args.recovery_adapter).resolve()
        sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
        config.update(recovery_adapter=str(adapter), recovery_adapter_sha256=sha,
                      ckpt_name=config['ckpt_name']+'-recovery-'+sha[:12])
    else:
        config.pop('recovery_adapter', None)
        config.pop('recovery_adapter_sha256', None)
    evaluation_settings(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    print(str(output.resolve()))
