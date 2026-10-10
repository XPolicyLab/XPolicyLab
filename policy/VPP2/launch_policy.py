"""Portable XPolicyLab launcher for the paired VPP2 joint+Action2B 100k bundle."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def command(args):
    if (args.bench, args.env_cfg, args.action_type) != ('RoboDojo', 'arx_x5', 'ee'):
        raise ValueError('This release uses RoboDojo/arx_x5/absolute EE16')
    if args.steps <= 0 or not 0 < args.replan <= 32 or not 0 < args.shift < float('inf'):
        raise ValueError('Require steps > 0, finite shift > 0, and 1 <= replan <= 32')
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
    bundle = resolve_checkpoint_root(
        dict(ckpt_name=args.ckpt, model_path=args.bundle, bench_name=args.bench,
             env_cfg_type=args.env_cfg, action_type=args.action_type, seed=args.seed),
        ROOT / 'checkpoints', policy_dir=ROOT,
    )
    if bundle.is_file():
        bundle = bundle.parent
    wan = Path(args.wan).expanduser().resolve() if args.wan else ROOT / 'checkpoints/Wan2.1-I2V-14B-480P'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('format') != 'vpp2-robodojo-bundle-v1' or manifest.get('step') != 100000:
        raise ValueError('Select the paired joint2b_s100000 bundle for this release')
    for name in ['action.pt', 'video.pt', 'dataset_stats.json']:
        path = bundle / name
        if not path.is_file() or path.stat().st_size != manifest['files'][name]:
            raise ValueError(f'Missing or incomplete bundle file: {path}')
    for name in ['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 'google/umt5-xxl/tokenizer_config.json']:
        if not (wan / name).is_file():
            raise FileNotFoundError(wan / name)
    xpl = Path(args.xpl_root).resolve() if args.xpl_root else ROOT.parent.parent
    if not (xpl / 'setup_policy_server.py').is_file():
        raise FileNotFoundError('Download this policy to RoboDojo/XPolicyLab/policy/VPP2')
    overrides = dict(policy_name=ROOT.name, bench_name=args.bench, task_name=args.task,
                     ckpt_name=args.ckpt, env_cfg_type=args.env_cfg, action_type='ee', action_dim=16,
                     seed=args.seed, host=args.host, port=args.port,
                     checkpoint_path=str(bundle / 'action.pt'), video_checkpoint_path=str(bundle / 'video.pt'),
                     dataset_stats_path=str(bundle / 'dataset_stats.json'), wan_model_dir=str(wan),
                     num_inference_steps=args.steps, sigma_shift=args.shift, replan_steps=args.replan,
                     action_horizon=32, history_stride=25, video_seed_offset=1000003)
    cmd = [sys.executable, '-u', str(xpl / 'setup_policy_server.py'), '--config_path',
           str(ROOT / 'deploy.yml'), '--overrides'] + [f'{k}={v}' for k, v in overrides.items()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, VPP2_TORCH_LOAD_MMAP='1',
               PYTHONPATH=os.pathsep.join([str(xpl.parent), str(xpl), os.environ.get('PYTHONPATH', '')]))
    return cmd, env, xpl.parent, overrides


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', default='RoboDojo')
    p.add_argument('--task', default='cover_blocks')
    p.add_argument('--ckpt', default='joint2b_s100000')
    p.add_argument('--env-cfg', default='arx_x5')
    p.add_argument('--action-type', default='ee')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gpu', default='0')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--shift', type=float, default=1)
    p.add_argument('--replan', type=int, default=24)
    p.add_argument('--bundle')
    p.add_argument('--wan')
    p.add_argument('--xpl-root')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cmd, env, cwd, settings = command(args)
    print(json.dumps(dict(method='VPP2', checkpoint_step=100000, settings=settings), indent=2), flush=True)
    if not args.dry_run:
        os.chdir(cwd)
        os.execvpe(cmd[0], cmd, env)


if __name__ == '__main__':
    main()
