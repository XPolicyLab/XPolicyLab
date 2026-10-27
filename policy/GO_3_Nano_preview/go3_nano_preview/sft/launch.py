"""Reference CLI for a scheduler that runs one command instance per node.

WORLD_SIZE/RANK here mean node count/node rank at the train.sh entry point.
torchrun replaces them with GPU process world size/global rank for the worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from go3_nano_preview.sft.paths import POLICY_ROOT, add_standard_arguments, standard_name, prepared_training_paths, check_link


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def gpu_count(env):
    if 'CUDA_VISIBLE_DEVICES' in env:
        devices = [d.strip() for d in env['CUDA_VISIBLE_DEVICES'].split(',') if d.strip()]
        return 0 if devices == ['-1'] else len(devices)
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, check=True)
    return len(result.stdout.splitlines())


def build_command(args, env):
    if 'LOCAL_RANK' in env:
        raise ValueError('Run train.sh once per node, not inside an existing torchrun GPU worker')
    nodes = int(env.get('WORLD_SIZE', '1'))
    node_rank = int(env.get('RANK', '0'))
    process_count = env.get('NPROC_PER_NODE') or env.get('SFT_NPROC')
    per_node = int(process_count) if process_count else gpu_count(env)
    if nodes < 1 or per_node < 1 or not 0 <= node_rank < nodes:
        raise ValueError('Invalid WORLD_SIZE/RANK/NPROC_PER_NODE node topology')
    if nodes > 1 and (not env.get('MASTER_ADDR') or 'RANK' not in env):
        raise ValueError('Multi-node jobs require scheduler-provided MASTER_ADDR and node RANK')
    master = env.get('MASTER_ADDR', '127.0.0.1')
    port = int(env.get('MASTER_PORT', '12345'))
    if not 1 <= port <= 65535:
        raise ValueError('MASTER_PORT must be between 1 and 65535')
    config = Path(args.config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    standard = standard_name(args)
    name = standard or args.run_name or config.stem
    if standard and args.run_name and args.run_name != standard:
        raise ValueError('The standard interface fixes the run name; use the simple interface for --run-name')
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('run_name must be a simple directory name')
    run_root = Path(env.get('RUN_ROOT') or str(Path.home() / '.local/state/robotwin')).expanduser()
    output = Path(getattr(args, 'output', None) or env.get('SFT_OUTPUT_DIR') or
                  str(run_root / 'checkpoints' / name)).expanduser().resolve()
    if output.is_relative_to(POLICY_ROOT):
        raise ValueError('Training output must be outside the policy checkout')
    link = POLICY_ROOT / 'checkpoints' / name
    if not args.skip_save and not args.data_only:
        check_link(link, output / 'step_0', checkpoint_output=output)
    command = [sys.executable, '-m', 'torch.distributed.run',
        f'--nnodes={nodes}', f'--nproc-per-node={per_node}']
    if nodes == 1:
        command.append('--standalone')
    else:
        command.extend([f'--node-rank={node_rank}', f'--master-addr={master}', f'--master-port={port}'])
    command.extend(['-m', 'go3_nano_preview.sft.train', '--config', str(config),
                    '--output', str(output), '--seed', str(args.seed)])
    if not args.skip_save and not args.data_only:
        command.extend(['--checkpoint-link', str(link)])
    for attr, flag in [('steps','--steps'),('batch_size','--batch-size'),('workers','--workers'),
                       ('save_every','--save-every'),('resume','--resume')]:
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    for attr in ('overwrite','data_only','skip_validation','skip_save','gradient_checkpointing','synchronize_timing'):
        if getattr(args, attr):
            command.append('--' + attr.replace('_','-'))
    return command, {'nodes':nodes,'node_rank':node_rank,'gpus_per_node':per_node,
                     'world_size':nodes*per_node,'output':str(output),
                     'config':str(config),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
                     'checkpoint_link':str(link) if not args.skip_save and not args.data_only else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?',
        help='Training config; relative paths are resolved from the calling directory')
    parser.add_argument('--config', dest='config_override', help='Override the configured training recipe')
    parser.add_argument('--env', choices=['GO_3_Nano_preview', 'go3_nano_preview', 'WAM_policy'], default='GO_3_Nano_preview')
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--run_name', '--run-name')
    parser.add_argument('--output', help='Override the training output directory')
    parser.add_argument('--overwrite', type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--debug', type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SFT_SEED') or 1016))
    parser.add_argument('--steps', type=int, default=int(os.environ['SFT_STEPS']) if os.environ.get('SFT_STEPS') else None)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--resume')
    add_standard_arguments(parser)
    for option in ('data-only','skip-validation','skip-save','gradient-checkpointing','synchronize-timing','dry-run'):
        parser.add_argument('--' + option, action='store_true')
    args = parser.parse_args()
    args.config = (args.config_override or args.config or os.environ.get('SFT_CONFIG') or
                   str(Path(__file__).resolve().parents[2] / 'configs' / 'robotwin_sft_ka_step.py'))
    data_name = standard_name(args, include_seed=False)
    if data_name:
        data_paths = prepared_training_paths(data_name)
    else:
        args.run_name = args.run_name or os.environ.get('SFT_RUN_NAME') or None
        data_paths = {key: os.environ.get(key, '') for key in ('SFT_JSONL_PATH', 'SFT_DATA_ROOT')}
    command, topology = build_command(args, os.environ)
    print(json.dumps({'event':'distributed_launch', **topology, 'data':data_paths, 'command':command}), flush=True)
    if args.dry_run:
        return
    if args.debug:
        os.environ.setdefault('TORCH_DISTRIBUTED_DEBUG', 'DETAIL')
    os.environ.update(data_paths)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
