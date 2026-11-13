"""Standard XPolicyLab arguments for the packaged, unchanged NPU recipe."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed', 'npu_id'):
        parser.add_argument(name)
    parser.add_argument('--mode', choices=('train', 'smoke', 'dry-run'), default='train')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    xpl = Path(os.environ.get('CLOUDROBO_XPOLICYLAB_ROOT', root.parents[1])).resolve()
    if (args.bench_name, args.env_cfg_type, args.action_type) != ('RoboTwin', 'arx_x5', 'joint'):
        parser.error('This recipe supports RoboTwin / arx_x5 / joint only.')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.ckpt_name) is None:
        parser.error('ckpt_name must be a simple run label, not a path.')
    config = json.loads((root / 'posttrain/configs/train_fp32.json').read_text())
    if args.seed != str(config['runtime']['seed']):
        parser.error('The sealed training recipe uses seed 42; other seeds are unsupported.')
    devices = args.npu_id.split(',')
    if (len(devices) != 16 or any(re.fullmatch(r'0|[1-9][0-9]*', d) is None for d in devices)
            or len(set(devices)) != 16):
        parser.error('npu_id selects exactly 16 distinct logical NPU IDs for this recipe.')
    action_dim = int(subprocess.check_output(
        ['bash', str(xpl / 'utils/get_action_dim.sh'), str(xpl.parent), args.env_cfg_type], text=True))
    if action_dim != 14:
        parser.error('The packaged Joint14 training data requires the shared robot dimension to be 14.')
    spec = importlib.util.spec_from_file_location('_xpl_checkpoint_resolver', xpl / 'utils/checkpoint_resolver.py')
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)
    run_name = resolver.build_run_dir_name(vars(args))
    env = dict(os.environ)
    for key in ('CLOUDROBO_WM_DATA_ROOT', 'CLOUDROBO_WM_ASSET_ROOT',
                'CLOUDROBO_WM_INIT_WEIGHTS', 'CLOUDROBO_WM_OUTPUT_ROOT'):
        if not env.get(key):
            parser.error('Set ' + key + ' explicitly; output must be a fresh external shared POSIX directory.')
    env.update(CLOUDROBO_WM_NPU_DEVICES=args.npu_id, CLOUDROBO_WM_RUN_NAME=run_name,
               PYTHONDONTWRITEBYTECODE='1')
    # FULL_TRAIN.sh calls python3. Keep the interpreter selected for this wrapper.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    print(json.dumps({'run_name': run_name, 'robot_action_dimension': action_dim,
                      'output_root': env['CLOUDROBO_WM_OUTPUT_ROOT'], 'mode': args.mode}), flush=True)
    os.execvpe('bash', ['bash', str(root / 'posttrain/FULL_TRAIN.sh'), args.mode], env)


if __name__ == '__main__':
    main()
