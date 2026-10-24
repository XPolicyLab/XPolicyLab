"""XPolicyLab adapter for frozen LILO programmatic policies.

A frozen policy is a directory `<ckpt_root>/scripts/<task>/policy.py` defining `Policy(task, seed)` with
`reset(obs)` and `act(obs)`. `act` returns a macro command
    {'targets': {'left_ee_pose': [x, y, z, qw, qx, qy, qz], 'right_ee_joint_state': [g], ...},
     'frames': n, 'phase': str}
or {'op': 'stop'}. This adapter expands each macro into an n-step ee-pose action chunk with the same
smoothstep position / slerp rotation interpolation the policies were developed against.
Checkpoints declaring action_type=joint instead emit native arm_joint_state targets, interpolated
without inverse kinematics. Their manifest may override depth_provider for RGB-only policies.

Observation contract: RoboDojo eval sends RGB + robot state only. The policies were written against
RGB-D with per-camera `intrinsic_matrix` / `extrinsic_matrix`; the adapter reconstructs both matrices
from public camera config and the observed ee poses (`camera_calib.json`, produced by
scripts/lilo_experiment/xpl_calib.py in the RoboDojo workspace). `depth_provider: mono` estimates depth from
RGB in the policy child (stereo.MonoPlaneDepth: Depth Anything V2 Large + table-plane calibration);
`depth_provider: none` passes observations through without a depth key.

After a policy stops (or raises), the adapter holds the current pose until the environment ends the
episode, because success is only decided by the environment's own episode-end check.
"""
import json
import os
from pathlib import Path
import pickle
import select
import struct
import subprocess
import sys
import time

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
ARMS = ('left', 'right')


def quat_mul(a, b):
    """Hamilton product of [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def slerp(q0, q1, t):
    """Shortest-path geodesic interpolation of [w, x, y, z] quaternions (same path as scipy Slerp)."""
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    rel = quat_mul(q0 * [1, -1, -1, -1], q1)
    if rel[0] < 0:
        rel = -rel
    s = np.linalg.norm(rel[1:])
    if s < 1e-12:
        return q0.copy()
    angle = 2 * np.arctan2(s, rel[0]) * t
    return quat_mul(q0, np.concatenate([[np.cos(angle / 2)], rel[1:] / s * np.sin(angle / 2)]))


def pose_matrix(pose):
    """[x, y, z, qw, qx, qy, qz] -> 4x4 world transform."""
    x, y, z, w, qx, qy, qz = np.asarray(pose, float)
    n = np.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    w, qx, qy, qz = w / n, qx / n, qy / n, qz / n
    t = np.eye(4)
    t[:3, :3] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)],
                 [2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx)],
                 [2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)]]
    t[:3, 3] = x, y, z
    return t


def to_wire(value):
    """numpy-version-neutral encoding for the child (which may run a different numpy than the server)."""
    if isinstance(value, np.ndarray):
        return ('__nd__', value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_wire(v) for v in value)
    return value


class FrozenPolicyProcess:
    """One child process per episode; replies are JSON, observations go down as pickles."""

    def __init__(self, python, policy_path, task, reset_timeout, act_timeout, depth_provider='none', model_dir=''):
        self.reset_timeout, self.act_timeout = reset_timeout, act_timeout
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1')
        self.proc = subprocess.Popen([python, '-u', str(POLICY_DIR / 'runtime.py'), str(policy_path), task,
                                      depth_provider, model_dir],
                                     cwd=str(Path(policy_path).parent), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)

    def call(self, op, obs):
        deadline = time.monotonic() + (self.reset_timeout if op == 'reset' else self.act_timeout)
        payload = pickle.dumps((op, to_wire(obs)), protocol=4)
        self.proc.stdin.write(struct.pack('!Q', len(payload)) + payload)
        self.proc.stdin.flush()
        response = bytearray()
        while b'\n' not in response:
            remain = deadline - time.monotonic()
            if remain <= 0 or not select.select([self.proc.stdout], [], [], remain)[0]:
                raise TimeoutError(f'policy_timeout during {op}')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('policy_process_exited')
            response.extend(chunk)
        value = json.loads(response)
        if not value.get('ok'):
            raise RuntimeError('policy_exception: ' + str(value.get('error'))[:1000])
        return value.get('result')

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        if model_cfg['env_cfg_type'] != 'arx_x5':
            raise ValueError('REAL_RSI camera calibration supports env_cfg_type=arx_x5 only')
        self.task_name = str(model_cfg['task_name'])
        self.action_type = model_cfg.get('action_type') or 'ee'
        if self.action_type not in ('ee', 'joint'):
            raise ValueError('REAL_RSI supports action_type=ee or joint')
        dims = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        if len(dims['arm_dim']) != 2:
            raise ValueError('REAL_RSI policies are dual-arm')
        suffix = 'arm_joint_state' if self.action_type == 'joint' else 'ee_pose'
        self.action_shapes = {f'{arm}_{suffix}': (int(dim) if self.action_type == 'joint' else 7,)
                              for arm, dim in zip(ARMS, dims['arm_dim'])}
        self.action_shapes.update({f'{arm}_ee_joint_state': (int(dim),)
                                   for arm, dim in zip(ARMS, dims['ee_dim'])})
        root = resolve_checkpoint_root(model_cfg, POLICY_DIR, policy_dir=POLICY_DIR,
                                       include_seed=False)
        routes = json.loads((root / 'routes.json').read_text())
        if self.task_name not in routes:
            raise ValueError(f'Unsupported REAL_RSI task: {self.task_name}')
        route = routes[self.task_name]
        task_dir = root / route['program']
        self.policy_path = task_dir / 'policy.py'
        if not self.policy_path.is_file():
            raise FileNotFoundError(f'no frozen policy for {self.task_name} under {root}')
        manifest_path = task_dir / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        required_action = route['action_type']
        if self.action_type != required_action:
            raise ValueError(f'Checkpoint requires action_type={required_action}, got {self.action_type}')
        self.calib = json.loads((POLICY_DIR / model_cfg.get('camera_calib', 'camera_calib.json')).read_text())
        self.depth_provider = route['depth_provider']
        if self.depth_provider not in ('none', 'mono'):
            raise ValueError(f'unknown depth_provider {self.depth_provider!r}')
        self.mono_model_dir = model_cfg.get('mono_model_dir') or ''   # '' -> stereo.DA2_LARGE
        self.hold_chunk = int(model_cfg.get('hold_chunk', 50))
        self.reset_timeout = float(model_cfg.get('reset_timeout_s', 240))
        self.act_timeout = float(model_cfg.get('act_timeout_s', 120))
        self.policy_python = model_cfg.get('policy_python') or os.environ.get('REAL_RSI_POLICY_PYTHON') or sys.executable
        self.strict_errors = bool(model_cfg.get('strict_errors', True))
        self.process = None
        self.reset()

    # observation -------------------------------------------------------------------------------------------
    def _augment(self, obs):
        state = obs['state']
        vision = {}
        for cam, data in obs.get('vision', {}).items():
            data = dict(data)
            cal = self.calib['cameras'].get(cam)
            if cal is not None:
                data['intrinsic_matrix'] = np.asarray(cal['K'], float)
                if 'mount' in cal:
                    data['extrinsic_matrix'] = pose_matrix(state[cal['ee_key']]) @ np.asarray(cal['mount'], float)
                else:
                    data['extrinsic_matrix'] = np.asarray(cal['T'], float)
            vision[cam] = data
        return dict(obs, vision=vision)

    def update_obs(self, obs):
        self.obs = self._augment(obs)
        if self.started or self.failed:
            return
        self.started = True
        try:
            self.process = FrozenPolicyProcess(self.policy_python, self.policy_path, self.task_name, self.reset_timeout,
                                               self.act_timeout, self.depth_provider, self.mono_model_dir)
            self.process.call('reset', self.obs)
        except Exception as exc:
            self._fail(exc)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('REAL_RSI runs with eval_batch: false')

    # action ------------------------------------------------------------------------------------------------
    def _current(self):
        cur = {k: np.asarray(self.obs['state'][k], float) for k in self.action_shapes if k not in self.grips}
        cur.update({k: np.asarray(v, float) for k, v in self.grips.items()})
        return cur

    def _hold(self):
        cur = self._current()
        return [{k: v.astype(np.float32) for k, v in cur.items()} for _ in range(self.hold_chunk)]

    def _fail(self, exc):
        self.failed = True
        self.error = repr(exc)[:500]
        print(f'[REAL_RSI] {self.task_name}: policy failed: {self.error}',
              flush=True)
        if self.process is not None:
            self.process.close()
        if self.strict_errors:
            raise RuntimeError(f'REAL_RSI policy {self.task_name} failed: {exc}') from exc

    def get_action(self):
        if self.stopped or self.failed:
            return self._hold()
        try:
            command = self.process.call('act', self.obs)
            if command.get('op') == 'stop':
                self.stopped = True
                self.process.close()
                return self._hold()
            frames = command.get('frames', 1)
            if not isinstance(frames, int) or frames <= 0:
                raise ValueError('frames must be a positive integer')
            targets = command.get('targets', {})
            allowed = set(self.action_shapes)
            if not set(targets).issubset(allowed):
                raise ValueError('Unsupported actuator key')
            begin = self._current()
            goals = {k: np.asarray(targets.get(k, v), float) for k, v in begin.items()}
            for k, value in goals.items():
                if value.shape != self.action_shapes[k] or not np.isfinite(value).all():
                    raise ValueError(f'Invalid action shape/value: {k}')
        except Exception as exc:
            self._fail(exc)
            return self._hold()
        chunk = []
        for frame in range(frames):
            t = (frame + 1) / frames
            t = t * t * (3 - 2 * t)
            action = {}
            for key, a in begin.items():
                b = goals[key]
                val = a * (1 - t) + b * t
                if key.endswith('_ee_pose'):
                    val[3:] = slerp(a[3:], b[3:], t)
                action[key] = val.astype(np.float32)
            chunk.append(action)
        self.grips = {k: goals[k] for k in self.grips}
        return chunk

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('REAL_RSI runs with eval_batch: false')

    def reset(self):
        if self.process is not None:
            self.process.close()
        self.process = None
        self.obs = None
        self.started = self.stopped = self.failed = False
        self.error = None
        self.grips = {k: np.ones(shape) for k, shape in self.action_shapes.items() if k.endswith('_ee_joint_state')}
