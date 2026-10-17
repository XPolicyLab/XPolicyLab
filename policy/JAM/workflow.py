"""Optional scheduler-independent reproduction. --dry-run never starts work."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
from delivery import POLICY_DIR, WORKSPACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('download', 'convert', 'stats', 'text', 'cache', 'train'))
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--method', choices=('jam', 'fastwam'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus-per-node', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29500)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data = args.workspace.expanduser().resolve()
    runtime = POLICY_DIR / 'runtime'
    # The release recipe writes elsewhere and never touches supplied model bundles.
    if data == WORKSPACE or data in (WORKSPACE / 'models').parents or data == WORKSPACE / 'models':
        raise ValueError('Use a separate data workspace')
    env = dict(os.environ)
    env.update(JAM_DATA_ROOT=str(data), DIFFSYNTH_MODEL_BASE_PATH=str(runtime / 'checkpoints'),
               DIFFSYNTH_SKIP_DOWNLOAD='true', FASTWAM_VIDEO_BACKEND='pyav',
               WANDB_MODE='disabled', PYTHONPATH=os.pathsep.join([str(runtime / 'src'), str(runtime / 'scripts'),
                                                               str(WORKSPACE), str(WORKSPACE / 'XPolicyLab')]),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    command = [sys.executable]
    if args.stage in ('download', 'convert', 'stats'):
        script = {'download': 'robodojo_piperx_download.py', 'convert': 'robodojo_piperx_prepare.py',
                  'stats': 'robodojo_piperx_stats.py'}[args.stage]
        command += [str(runtime / 'scripts' / script), '--root', str(data)]
    elif args.stage == 'text':
        command += ['scripts/precompute_text_embeds.py', 'task=robodojo_piperx_fastwam', 'wandb.enabled=false']
    elif args.stage == 'cache':
        command = [str(Path(sys.executable).parent / 'torchrun'), '--standalone',
                   f'--nproc_per_node={args.gpus_per_node}', 'scripts/extract_foresight_cache.py',
                   'task=robodojo_piperx_jam_wan', f'+extract.out_roots=[{data}/cache/targets_train]',
                   '+extract.layers=[0]', '+extract.offsets=[1,2]',
                   '+extract.batch_size=16', '+extract.num_workers=4', '+extract.shard_size=1024',
                   'wandb.enabled=false', f'output_dir={data}/cache/extraction_work']
    else:
        if args.method is None or args.output is None:
            parser.error('train requires --method and --output')
        output = args.output.expanduser().resolve()
        if output.exists() and (args.nodes == 1 or args.node_rank == 0):
            raise FileExistsError('Training output must be a new path; frozen weights cannot be overwritten')
        if args.gpus_per_node * args.nodes != 8:
            raise ValueError('The archived fixed recipe requires world_size=8 and batch 16/GPU (global 128)')
        if not 0 <= args.node_rank < args.nodes:
            raise ValueError('Invalid node rank')
        task = {'jam': 'robodojo_piperx_jam_wan', 'fastwam': 'robodojo_piperx_fastwam'}[args.method]
        env.update(NNODES=str(args.nodes), NODE_RANK=str(args.node_rank), MASTER_ADDR=args.master_addr,
                   MASTER_PORT=str(args.master_port), RUN_ID=output.name, RUNS_ROOT=str(output.parent))
        command = ['bash', 'scripts/train_zero1.sh', str(args.gpus_per_node),
                   f'task={task}', f'output_dir={output}', 'max_steps=28630', 'lr_schedule_steps=28630',
                   'batch_size=16', 'gradient_accumulation_steps=1', 'num_workers=4', 'eval_every=0',
                   'save_every=5000', 'save_full_state=true', 'keep_latest_state_only=true',
                   'seed=42', 'wandb.enabled=false', 'prefetch_factor=1', 'log_cuda_memory=true']
        if args.method == 'jam':
            command += [f'model.foresight.target_cache_roots=[{data}/cache/targets_train]',
                        '+data.train.first_frame_only=true']
        if not args.dry_run:
            split = json.loads((data / 'stats/split_audit.json').read_text())
            if split['optimizer_steps'] != 28630 or split['seed'] != 42:
                raise ValueError('Unexpected training split or schedule')
            if args.method == 'jam' and not (data / 'cache/targets_train').is_dir():
                raise FileNotFoundError('Run the cache stage first')
    print(json.dumps({'stage': args.stage, 'command': command, 'cwd': str(runtime),
                      'data_workspace': str(data), 'dry_run': args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return
    data.mkdir(parents=True, exist_ok=True)
    if args.stage == 'stats' and (data / 'stats/train90_seed42.json').exists():
        raise FileExistsError('Statistics already exist; preserve them or use a new workspace')
    if args.stage in ('convert', 'stats'):
        for name in ('XPolicyLab', 'env_cfg'):
            link = data / name
            target = WORKSPACE / name
            if link.exists() and link.resolve() != target:
                raise FileExistsError(f'Unexpected existing path: {link}')
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
    subprocess.run(command, cwd=runtime, env=env, check=True)


if __name__ == '__main__':
    main()
