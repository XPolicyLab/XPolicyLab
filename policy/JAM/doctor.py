"""CPU preflight. Does not load checkpoints, contact a robot, or initialize CUDA."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
from delivery import POLICY_DIR, WORKSPACE, TASKS, load_deployment, validate_overrides, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true', help='Also SHA-256 all staged assets')
    args = parser.parse_args()
    for path in [WORKSPACE, WORKSPACE / 'XPolicyLab', POLICY_DIR / 'runtime/src', POLICY_DIR / 'runtime/scripts']:
        sys.path.insert(0, str(path))
    for name in ['torch', 'torchvision', 'accelerate', 'hydra', 'transformers', 'h5py',
                 'websockets', 'msgpack_numpy', 'cv2', 'fastwam.runtime',
                 'fastwam.models.wan22.fastwam', 'XPolicyLab.policy.JAM.model']:
        importlib.import_module(name)
    from infer_aloha_stationary import _compose_task
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from XPolicyLab.utils.process_data import get_robot_action_dim_info
    dimensions = get_robot_action_dim_info('piper_x')
    assert dimensions['arm_dim'] == [6, 6] and dimensions['ee_dim'] == [1, 1]
    configs = {}
    for method, task in [('jam', 'robodojo_piperx_jam_wan'), ('fastwam', 'robodojo_piperx_fastwam')]:
        manifest = load_deployment(WORKSPACE / 'models' / method, verify_checkpoint=False)
        for robot_task in TASKS:
            validate_overrides({'task_name': robot_task, 'seed': 42, 'replan_steps': 24}, manifest)
        cfg = _compose_task(task)
        OmegaConf.to_container(cfg, resolve=True)
        processor = instantiate(cfg.data.train.processor)
        configs[method] = {'training_config': task, 'processor': type(processor).__name__,
                           'deployment_config': 'robodojo_piperx_fastwam', 'normalization_sha256_verified': True}
    if args.assets:
        verify_assets(hashes=True)
    import torch
    result = {'status': 'passed', 'scope': 'CPU imports, Hydra configs, processor, path relocation and fixed manifests',
              'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda_initialized': torch.cuda.is_initialized(),
              'workspace': str(WORKSPACE), 'configs': configs,
              'assets_sha256_verified': args.assets, 'robot_trials': 0, 'success_rate': None}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
