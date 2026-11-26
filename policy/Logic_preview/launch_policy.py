"""Portable launcher for the complete paired LogicPreview 746-case step-10000 release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'model_code'
WEIGHTS = Path(os.environ.get('LOGIC_PREVIEW_WEIGHTS_DIR') or ROOT / 'weights').resolve()


def command(args):
    if (args.bench, args.env_cfg, args.action_type) != ('RoboDojo', 'arx_x5', 'ee'):
        raise ValueError('This release uses RoboDojo/arx_x5/absolute EE16')
    if (args.steps, args.replan, args.shift) != (10, 24, 1.0):
        raise ValueError('The 746-case release requires steps=10, replan=24, shift=1')
    if args.seed < 0:
        raise ValueError('The evaluation seed must be non-negative')
    if args.bundle and not Path(args.bundle).expanduser().exists():
        raise FileNotFoundError(f'Explicit bundle does not exist: {args.bundle}')
    xpl = Path(args.xpl_root).resolve() if args.xpl_root else ROOT.parent.parent
    sys.path.insert(0, str(xpl.parent))
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
    bundle = resolve_checkpoint_root(
        dict(ckpt_name=args.ckpt, model_path=str(Path(args.bundle).expanduser().resolve()) if args.bundle else None, bench_name=args.bench,
             env_cfg_type=args.env_cfg, action_type=args.action_type, seed=args.seed),
        WEIGHTS, policy_dir=ROOT,
    )
    if bundle.is_file():
        bundle = bundle.parent
    wan = Path(args.wan).expanduser().resolve() if args.wan else WEIGHTS / 'Wan2.1-I2V-14B-480P'
    manifest_bytes = (bundle / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if (hashlib.sha256(manifest_bytes).hexdigest() != '451ee3544dbfb347fdd3bbde044bbf58b52755c0cbe8a0b75cacdedeb0b47b34'
            or manifest.get('step') != 10000 or manifest.get('training_mode') != 'joint'
            or manifest.get('inference', {}).get('horizon') != 36):
        raise ValueError('Select the paired joint2b_746_s010000 bundle (joint, step 10000, horizon 36)')
    for name in ['action.pt', 'video.pt', 'dataset_stats.json']:
        path = bundle / name
        if not path.is_file() or path.stat().st_size != manifest['files'][name]:
            raise ValueError(f'Missing or incomplete bundle file: {path}')
    for name in ['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 'google/umt5-xxl/tokenizer_config.json']:
        if not (wan / name).is_file():
            raise FileNotFoundError(wan / name)
    if not (xpl / 'setup_policy_server.py').is_file():
        raise FileNotFoundError('Place this policy under XPolicyLab/policy/Logic_preview')
    if not (SOURCE / 'src/logic_preview_core/policy.py').is_file():
        raise FileNotFoundError('Keep the bundled model_code directory with this adapter')
    overrides = dict(policy_name=ROOT.name, bench_name=args.bench, task_name=args.task,
                     ckpt_name=args.ckpt, env_cfg_type=args.env_cfg, action_type='ee', action_dim=16,
                     seed=args.seed, host=args.host, port=args.port,
                     checkpoint_path=str(bundle / 'action.pt'), video_checkpoint_path=str(bundle / 'video.pt'),
                     dataset_stats_path=str(bundle / 'dataset_stats.json'), wan_model_dir=str(wan),
                     num_inference_steps=args.steps, sigma_shift=args.shift, replan_steps=args.replan,
                     action_horizon=36, history_stride=25, video_seed_offset=1000003)
    if args.device_map:
        overrides['inference_device_map'] = args.device_map
    server = ROOT / 'packed_server.py' if os.getenv('LOGIC_PREVIEW_PACKED_SERVER') == '1' else xpl / 'setup_policy_server.py'
    cmd = [sys.executable, '-u', str(server), '--config_path',
           str(ROOT / 'deploy.yml'), '--overrides'] + [f'{k}={v}' for k, v in overrides.items()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, LOGIC_PREVIEW_TORCH_LOAD_MMAP='1',
               PYTHONPATH=os.pathsep.join([str(SOURCE / 'src'), str(xpl.parent), str(xpl), os.environ.get('PYTHONPATH', '')]))
    return cmd, env, xpl.parent, overrides


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', default='RoboDojo')
    p.add_argument('--task', default='cover_blocks')
    p.add_argument('--ckpt', default='joint2b_746_s010000')
    p.add_argument('--env-cfg', default='arx_x5')
    p.add_argument('--action-type', default='ee')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gpu', default='0')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--shift', type=float, default=1)
    p.add_argument('--replan', type=int, default=24)
    p.add_argument('--bundle', default=os.environ.get('LOGIC_PREVIEW_BUNDLE'))
    p.add_argument('--wan', default=os.environ.get('LOGIC_PREVIEW_WAN_ROOT'))
    p.add_argument('--xpl-root')
    p.add_argument('--device-map', default=os.environ.get('LOGIC_PREVIEW_INFERENCE_DEVICE_MAP'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cmd, env, cwd, settings = command(args)
    print(json.dumps(dict(method='Logic_preview', checkpoint_step=10000, settings=settings), indent=2), flush=True)
    if not args.dry_run:
        os.chdir(cwd)
        os.execvpe(cmd[0], cmd, env)


if __name__ == '__main__':
    main()
