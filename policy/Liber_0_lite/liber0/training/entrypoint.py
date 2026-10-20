import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name, resolve_checkpoint_root

MODULE = 'XPolicyLab.policy.Liber_0_lite.liber0.training'
LITE_DIR = Path(__file__).resolve().parents[2]
REPO = LITE_DIR.parent.parent


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--policy-dir', type=Path, required=True)
    modes = result.add_subparsers(dest='mode', required=True)
    for mode in ('train', 'data'):
        command = modes.add_parser(mode)
        command.add_argument('bench_name', choices=['RoboDojo'])
        command.add_argument('ckpt_name')
        command.add_argument('env_cfg_type', choices=['arx_x5'])
        command.add_argument('action_type', choices=['joint'])
        command.add_argument('--dataset', type=Path, required=True)
        command.add_argument('--checkpoint', type=Path, required=True)
        command.add_argument('--dry-run', action='store_true')
        if mode == 'data':
            command.add_argument('expert_data_num', nargs='?', type=int)
        else:
            command.add_argument('seed', type=int)
            command.add_argument('gpu_id')
            command.add_argument('--config', type=Path, default=LITE_DIR / 'configs/robodojo_sft.yaml')
            command.add_argument('--assets', type=Path, required=True)
            command.add_argument('--init-weights', type=Path)
            command.add_argument('--batch-size', type=int, required=True)
            command.add_argument('--accumulation', type=int, required=True)
            command.add_argument('--num-workers', type=int, required=True)
            command.add_argument('--resume', type=Path)
            command.add_argument('--stop-after', type=int)
    return result


def command_for(args):
    policy_dir = args.policy_dir.resolve()
    if policy_dir.name not in {'Liber_0_lite', 'Liber_0_preview'} or policy_dir.parent != LITE_DIR.parent:
        raise ValueError('Unsupported policy directory.')
    if not args.ckpt_name or Path(args.ckpt_name).name != args.ckpt_name or args.ckpt_name in {'.', '..'}:
        raise ValueError('ckpt_name must be a run name, not a path.')
    dimension = int(subprocess.check_output(
        ['bash', str(REPO / 'utils/get_action_dim.sh'), str(REPO.parent), args.env_cfg_type], text=True).strip())
    if dimension != 14:
        raise ValueError('Registered robot dimensions do not match this model.')

    def resolve(path):
        return (policy_dir / path).resolve() if not path.is_absolute() else path.resolve()

    fields = dict(bench_name=args.bench_name, ckpt_name=args.ckpt_name,
                  env_cfg_type=args.env_cfg_type, action_type=args.action_type)
    checkpoint = resolve_checkpoint_root(dict(checkpoint_path=str(args.checkpoint)),
        policy_dir / 'checkpoints', policy_dir=policy_dir, must_exist=not args.dry_run)
    dataset = resolve(args.dataset)
    if args.mode == 'data':
        if args.expert_data_num is not None:
            raise ValueError('Select episodes with the official converter; this preflight does not truncate data.')
        return [sys.executable, '-m', MODULE + '.check_data', '--dataset', str(dataset),
                '--checkpoint', str(checkpoint)]

    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32).')
    gpu_ids = args.gpu_id.split(',')
    if any(not value.strip() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('gpu_id must contain distinct comma-separated device identifiers.')
    nodes = int(os.environ.get('NNODES', '1'))
    if nodes < 1 or min(args.batch_size, args.accumulation) < 1 or args.num_workers < 0:
        raise ValueError('Invalid node/batch/worker configuration.')
    config = resolve(args.config)
    settings = yaml.safe_load(config.read_text())
    actual_batch = nodes * len(gpu_ids) * args.batch_size * args.accumulation
    if actual_batch != settings['global_batch_size']:
        raise ValueError(f'Global batch must be {settings["global_batch_size"]}, got {actual_batch}.')
    fields['seed'] = args.seed
    output = policy_dir / 'checkpoints' / build_run_dir_name(fields)
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(len(gpu_ids))]
    if nodes == 1:
        command += ['--standalone']
    else:
        rank = int(os.environ['NODE_RANK'])
        port = int(os.environ['MASTER_PORT'])
        if not 0 <= rank < nodes or not 0 < port < 65536:
            raise ValueError('Invalid node rank or rendezvous port.')
        command += ['--nnodes', str(nodes), '--node_rank', str(rank),
                    '--master_addr', os.environ['MASTER_ADDR'], '--master_port', str(port)]
    command += ['-m', MODULE + '.train', '--config', str(config), '--dataset', str(dataset),
                '--checkpoint', str(checkpoint), '--assets', str(resolve(args.assets)),
                '--output', str(output), '--seed', str(args.seed),
                '--batch-size', str(args.batch_size), '--accumulation', str(args.accumulation),
                '--num-workers', str(args.num_workers)]
    for key in ('init_weights', 'resume', 'stop_after'):
        value = getattr(args, key)
        if value is not None:
            command += ['--' + key.replace('_', '-'), str(resolve(value) if isinstance(value, Path) else value)]
    return command


def main():
    args = parser().parse_args()
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') != '0':
        raise RuntimeError('Use the policy launcher or export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0.')
    command = command_for(args)
    if args.mode == 'train':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.dry_run:
        print(json.dumps(dict(policy=args.policy_dir.name, command=command,
                              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
        return
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
