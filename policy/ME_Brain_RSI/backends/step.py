"""Bridge preserved Policy.step implementations to the XPL model API."""
import importlib
import json
import os
import time
from pathlib import Path
import numpy as np
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from ..core.runtime import ROOT


class StepModel:
    def __init__(self, config, route, scope):
        self.task = route.get('policy_task', route['canonical_task'])
        self.factory = scope.resolve(route['model'])
        self.dimensions = get_robot_action_dim_info(config['env_cfg_type'])
        if len(self.dimensions['arm_dim']) != 2:
            raise ValueError('Step policies require two arms')
        options_file = ROOT / 'tasks' / route['canonical_task'] / 'options.json'
        options = json.loads(options_file.read_text())
        self.feedback_mode = 'stop'
        if self.task == 'swap_T':
            self.feedback_mode = config.get('swap_t_feedback_mode') or os.environ.get(
                'ME_BRAIN_SWAP_T_FEEDBACK_MODE', 'stop')
            if self.feedback_mode not in ('stop', 'hold_until_budget'):
                raise ValueError('Unknown swap_T feedback mode: ' + str(self.feedback_mode))
            if self.feedback_mode == 'hold_until_budget':
                # The frozen policy already holds its current joints after a
                # failed visual gate. Keep that action until the evaluator's
                # normal episode limit; never retry the gate or invent a score.
                options['stop_on_upstream_failure'] = False
        static_file = config.get('robot_static_path') or os.environ.get('ME_BRAIN_ROBOT_STATIC')
        if not static_file:
            raise ValueError('Set robot_static_path or ME_BRAIN_ROBOT_STATIC to robot-only static JSON')
        static_file = Path(static_file).expanduser().resolve(strict=True)
        static = json.loads(static_file.read_text())
        if set(static) != {'left_arm', 'right_arm'}:
            raise ValueError('robot_static must contain only left_arm and right_arm')
        for arm in static.values():
            urdf = Path(arm['urdf_path']).expanduser()
            if not urdf.is_absolute():
                urdf = static_file.parent / urdf
            arm['urdf_path'] = str(urdf.resolve(strict=True))
        options['trace_dir'] = os.environ['B88R42WQ_EVIDENCE_DIR']
        if self.task == 'sort_nesting_dolls_by_size':
            assets = config.get('nesting_assets_root') or os.environ.get('ME_BRAIN_NESTING_ASSETS')
            if not assets:
                raise ValueError('Set nesting_assets_root or ME_BRAIN_NESTING_ASSETS to DINO/SAM3 assets')
            options['assets_root'] = str(Path(assets).expanduser().resolve(strict=True))
        if self.task in ('stack_bowls_random', 'stack_blocks_random'):
            assets = config.get('stack_assets_root') or os.environ.get('ME_BRAIN_STACK_ASSETS')
            if not assets:
                raise ValueError('Set ME_BRAIN_STACK_ASSETS to verified SAM3 assets root')
            options['assets_root'] = str(Path(assets).expanduser().resolve(strict=True))
        self.config = dict(options=options, robot_static=static, task=self.task,
                           observation_contract='rgb_visual_v1')
        self.reset()

    def reset(self):
        close = getattr(getattr(self, 'policy', None), 'close', None)
        if callable(close):
            close()
        self.policy = self.factory(self.config)
        error = getattr(self.policy, 'initialization_error', None)
        if error:
            raise RuntimeError(error)
        self.observation = None
        self.index = 0
        self.fresh = False
        self.feedback_reported = False

    def record_feedback_hold(self, response):
        if self.feedback_reported:
            return
        evidence = Path(os.environ['B88R42WQ_EVIDENCE_DIR'])
        directory = evidence / ('feedback_' + str(time.time_ns()))
        directory.mkdir()
        record = dict(task=self.task, step_index=self.index - 1,
                      feedback_mode=self.feedback_mode, response=response,
                      scoring='Unmodified evaluator after normal episode termination')
        (directory / 'feedback.json').write_text(json.dumps(record, indent=2))
        # Store the policy's existing RGB observations through the shared codec.
        from XPolicyLab.utils.process_data import encode_image_bit
        for name, rgb in self.observation['images'].items():
            (directory / (name + '.jpg')).write_bytes(encode_image_bit(rgb))
        self.feedback_reported = True
        print('[ME_Brain_RSI] swap_T visual gate failed; holding frozen-policy '
              'joints until normal episode limit. Evidence: ' + str(directory), flush=True)

    def update_obs(self, obs):
        images = {}
        for target, aliases in {
            'head': ('cam_head', 'cam_high', 'head_camera'),
            'left_wrist': ('cam_left_wrist', 'left_camera', 'left_wrist'),
            'right_wrist': ('cam_right_wrist', 'right_camera', 'right_wrist'),
        }.items():
            camera = next((obs['vision'][key] for key in aliases if key in obs['vision']), None)
            if camera is None and getattr(self, 'task', None) == 'swap_blocks':
                images[target] = None
                continue
            if camera is None:
                raise ValueError(f'Missing RGB camera: {target}')
            rgb = camera.get('color', camera.get('rgb')) if isinstance(camera, dict) else camera
            array = np.asarray(rgb)
            if array.dtype != np.uint8 or array.shape != (480, 640, 3):
                raise ValueError(f'{target} requires unresized HWC 640x480 uint8 RGB')
            images[target] = np.ascontiguousarray(array).copy()
        keys = [f'{side}_{suffix}' for side in ('left', 'right')
                for suffix in ('arm_joint_state', 'ee_joint_state', 'ee_pose')]
        robot = {key: np.asarray(obs['state'][key], dtype=float).copy() for key in keys}
        self.observation = dict(images=images, robot=robot, step_index=self.index,
                                instruction=obs.get('instruction', obs.get('instructions', '')))
        self.fresh = True

    def get_action(self):
        if not self.fresh:
            raise RuntimeError('A fresh observation is required for every step')
        self.fresh = False
        response = self.policy.step(self.observation)
        self.index += 1
        if 'stop_for_feedback' in response:
            evidence = os.environ.get('B88R42WQ_EVIDENCE_DIR')
            if evidence:
                (Path(evidence) / 'step_failure.json').write_text(json.dumps(response, indent=2))
            raise RuntimeError('Policy stopped for feedback: ' + str(response['stop_for_feedback']))
        if (getattr(self, 'feedback_mode', 'stop') == 'hold_until_budget' and
                response.get('diagnostics', {}).get('upstream_failed') is True):
            self.record_feedback_hold(response)
        action = response['action']
        expected = {f'{side}_{kind}' for side in ('left', 'right')
                    for kind in ('arm_joint_state', 'ee_joint_state')}
        if set(action) != expected:
            raise ValueError('Step policy returned invalid action keys')
        result = {}
        for i, side in enumerate(('left', 'right')):
            for suffix, dimension in [('arm_joint_state', self.dimensions['arm_dim'][i]),
                                      ('ee_joint_state', self.dimensions['ee_dim'][i])]:
                key = f'{side}_{suffix}'
                value = np.asarray(action[key], dtype=np.float32)
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid action shape/value: {key}')
                result[key] = value
        return [result]

    def update_obs_batch(self, obs):
        if len(obs) != 1:
            raise ValueError('Step policies support one environment per process')
        self.update_obs(obs[0])

    def get_action_batch(self, indices=None):
        if indices is not None and len(indices) != 1:
            raise ValueError('Step policies support one environment per process')
        return [self.get_action()]
