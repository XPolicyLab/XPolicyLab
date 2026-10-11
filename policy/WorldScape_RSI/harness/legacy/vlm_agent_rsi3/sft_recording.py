"""Record aligned pre-action RGB/state and executed joint targets, never fabricate success."""
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from .recording import EpisodeRecording
from XPolicyLab.utils.process_data import encode_image_bit

FIELDS = {'left_arm_joint_state': 6, 'left_ee_joint_state': 1,
          'right_arm_joint_state': 6, 'right_ee_joint_state': 1}
CAMERAS = ('cam_head', 'cam_right_wrist', 'cam_left_wrist')

class SFTRecording(EpisodeRecording):
    def __enter__(self):
        root = Path(os.environ['ROBODAWN_SFT_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        self.sft_path = root / f'episode_{time.time_ns()}.hdf5'
        self.sft = h5py.File(self.sft_path, 'x')
        self.sft.attrs.update(success=False, recording_valid=False, control_dt_s=self.control_dt,
                              native_step_limit=int(self.env.step_lim), budget_mode='native_strict',
                              action_semantics='absolute_joint_targets', image_color='RGB',
                              camera_order=json.dumps(CAMERAS), planner='codex-desktop-session',
                              source='RoboDawn desktop transport; not API reproduction')
        self.sft.attrs["uses_privileged_reward_feedback"] = getattr(self.env, "_rsi3_reward_feedback", None) is not None
        coverage_path = os.environ.get('RSI3_ACTUAL_LAYOUT')
        if coverage_path:
            coverage = json.loads(Path(coverage_path).read_text())
            if int(self.env.env_seeds[0]) != coverage['layout_id']:
                raise RuntimeError('Recording actual layout differs from scheduled layout')
            self.sft.attrs['rsi3_coverage'] = json.dumps(coverage)
        self.rows = 0
        self.sft_errors = []
        self.sft_done = False
        return super().__enter__()

    def append(self, name, value, encoded=False):
        if name not in self.sft:
            if encoded:
                self.sft.create_dataset(name, (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            else:
                value = np.asarray(value)
                self.sft.create_dataset(name, (0,)+value.shape, maxshape=(None,)+value.shape, dtype=value.dtype)
        ds = self.sft[name]
        ds.resize(ds.shape[0]+1, axis=0)
        ds[-1] = np.frombuffer(value, dtype=np.uint8) if encoded else value

    def take_action(self, action):
        obs = self.get_obs()
        before = int(self.env.take_action_cnt[0])
        state, targets, images = {}, {}, {}
        for key, width in FIELDS.items():
            state[key] = np.asarray(obs['state'][key], dtype=np.float32).reshape(-1)
            targets[key] = np.asarray(action[key], dtype=np.float32).reshape(-1)
            for value in (state[key], targets[key]):
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid SFT field {key}: {value.shape}')
        for camera in CAMERAS:
            images[camera] = encode_image_bit(obs['vision'][camera]['color'])
            if not isinstance(images[camera], bytes):
                raise ValueError(f'Expected one frame for {camera}')
        result = super().take_action(action)
        after = int(self.env.take_action_cnt[0])
        if after == before:
            return result
        if after != before + 1:
            self.sft_errors.append(f'Non-unit control transition {before}->{after}')
            raise ValueError(self.sft_errors[-1])
        for key in FIELDS:
            self.append('state/'+key, state[key])
            self.append('action/'+key, targets[key])
        for camera, value in images.items():
            self.append('vision/'+camera+'/colors', value, encoded=True)
        self.append('control_step', np.asarray(before, dtype=np.int64))
        self.append('timestamp', np.asarray(before*self.control_dt, dtype=np.float64))
        self.sft.attrs['instruction'] = str(obs.get('instruction', ''))
        self.rows += 1
        self.sft.flush()
        return result

    def finish(self, reason):
        super().finish(reason)
        if getattr(self, 'sft_done', True):
            return
        self.sft_done = True
        valid = self.rows > 0 and not self.sft_errors and reason not in ('exception', 'interrupted')
        limit = int(self.sft.attrs['native_step_limit'])
        within_budget = 0 < self.rows <= limit < 10**9 and int(self.env.take_action_cnt[0]) <= limit
        self.sft.attrs.update(success=bool(self.env.success[0]), recording_valid=valid,
                              within_native_budget=within_budget,
                              transitions=self.rows, termination_reason=str(reason),
                              eligible_for_sft=bool(self.env.success[0]) and valid and within_budget)
        self.sft.close()
