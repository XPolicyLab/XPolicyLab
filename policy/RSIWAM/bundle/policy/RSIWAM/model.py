"""XPolicyLab interface for the exact RSIWAM release candidate."""
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE_ROOT / 'src'), str(PACKAGE_ROOT / 'runtime')]

from rsiwam_native_wrapper import Model as NativeModel
from rsiwam_loader import BASE, check_base, check_package, compose


class Model(NativeModel):
    def __init__(self, model_cfg):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the policy process')
        cfg = dict(model_cfg)
        root = cfg.get('rsiwam_weights_root') or os.environ.get('RSIWAM_BASE_WEIGHTS')
        if not root:
            raise ValueError('Set RSIWAM_BASE_WEIGHTS to the external checkpoints directory')
        artifacts = check_package()
        weights = check_base(root)
        fixed = dict(action_type='ee', action_dim=16, deployment_adapter='robodojo_ee16',
                     env_cfg_type='arx_x5', num_inference_steps=10, sigma_shift=1,
                     seed=1, replan_steps=24, action_horizon=32, mixed_precision='bf16')
        for key, value in fixed.items():
            if cfg.get(key) is not None and cfg[key] != value:
                raise ValueError(f'RSIWAM frozen configuration requires {key}={value!r}')
        cfg.update(fixed, policy_name='RSIWAM', device='cuda', base_identity=BASE,
                   recovery_adapter=None,
                   checkpoint_path=str(weights / 'joint2b_s100000/action.pt'),
                   video_checkpoint_path=str(weights / 'joint2b_s100000/video.pt'),
                   dataset_stats_path=str(weights / 'joint2b_s100000/dataset_stats.json'),
                   wan_model_dir=str(weights / 'Wan2.1-I2V-14B-480P'))
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        super().__init__(cfg)
        self.model.model.requires_grad_(False)
        self.model.model.eval()
        compose(self.model, artifacts)

    def get_action(self, obs=None):
        import torch
        with torch.inference_mode():
            return super().get_action(obs)
