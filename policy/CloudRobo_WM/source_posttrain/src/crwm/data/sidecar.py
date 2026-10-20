from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .schema import file_sha256, within
URDF_SHA256 = '097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a'
EMBODIMENT_SHA256 = 'a4f043d2d4b44cca5ec47fb498a3194da3f87ff9530dcbfdcee1539a44423d57'

class RobotwinSidecar:

    def __init__(self, dataset_root, sidecar_root, urdf_sha256=URDF_SHA256, embodiment_sha256=EMBODIMENT_SHA256):
        self.root, self.dataset_root = (Path(sidecar_root), Path(dataset_root))
        if self.root.name == 'episodes':
            self.root = self.root.parent
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m, p = (self.manifest, self.manifest['provenance'])
        ph = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        if not m.get('complete') or m.get('schema_version') != 'robotwin_fk_sidecars_v1' or ph != m['provenance_sha256']:
            raise ValueError('Incomplete/invalid FK sidecar manifest/provenance')
        expected = dict(urdf_sha256=urdf_sha256, embodiment_config_sha256=embodiment_sha256, action_alignment='action[t]=state[min(t+1,T-1)]', action_alignment_verified=True, joint_semantics='raw_absolute_drive_target_radians_gripper_absolute', eef_frame='raw_fl_link6_fr_link6_world', pose_layout='left_right_xyz_quaternion_wxyz', pose_dtype='float32', arm_joint_indices=[[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], ignored_gripper_indices=[6, 13])
        for key, value in expected.items():
            if p.get(key) != value:
                raise ValueError(f'FK sidecar {key} mismatch')
        for relative, digest in p['dataset_metadata_sha256'].items():
            if file_sha256(within(self.dataset_root, relative)) != digest:
                raise ValueError(f'FK metadata identity mismatch: {relative}')
        seen, total = (set(), 0)
        with (self.dataset_root / 'meta/episodes.jsonl').open('rb') as stream:
            for line in stream:
                raw = line.rstrip(b'\r\n')
                if not raw:
                    continue
                row = json.loads(raw)
                eid = str(row['episode_index'])
                entry = m['episodes'][eid]
                if eid in seen or entry['length'] != row['length'] or entry['episode_metadata_sha256'] != hashlib.sha256(raw).hexdigest():
                    raise ValueError(f'FK episode identity mismatch: {eid}')
                if entry['provenance_sha256'] != ph:
                    raise ValueError(f'Stale FK episode: {eid}')
                seen.add(eid)
                total += row['length']
        if seen != set(m['episodes']) or len(seen) != p['total_episodes'] or total != p['total_frames']:
            raise ValueError('FK sidecar incomplete inventory')
        self.digest = file_sha256(self.root / 'manifest.json')

    def load(self, ep, columns):
        entry = self.manifest['episodes'][str(ep['episode_id'])]
        path = within(self.root, entry['pose_path'])
        if ep['data_sha256'] != entry['parquet_sha256'] or file_sha256(path) != entry['pose_sha256']:
            raise ValueError('FK source/pose file hash mismatch')
        n = ep['length']
        state = np.asarray(columns['observation.state'], dtype='<f4')
        action = np.asarray(columns['action'], dtype='<f4')
        if state.shape != (n, 14) or action.shape != state.shape or (not np.isfinite(state).all()) or (not np.isfinite(action).all()):
            raise ValueError('FK raw joint shape/finiteness mismatch')
        sb, ab = (state.tobytes(), action.tobytes())
        for key, raw in (('state_sha256', sb), ('action_sha256', ab), ('trajectory_sha256', sb + ab)):
            if hashlib.sha256(raw).hexdigest() != entry[key]:
                raise ValueError(f'FK raw joint identity mismatch: {key}')
        if not np.array_equal(action, state[np.minimum(np.arange(n) + 1, n - 1)]):
            raise ValueError('RoboTwin action is not the next ORIGINAL state frame')
        poses = np.load(path, mmap_mode='r', allow_pickle=False)
        if poses.shape != (n, 2, 7) or poses.dtype != np.float32 or (not np.isfinite(poses).all()):
            raise ValueError('Invalid FK pose array')
        if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-05):
            raise ValueError('Invalid FK quaternion norm')
        return poses
