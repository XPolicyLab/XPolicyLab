from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from .codec import ActionCodec, GripperCalibration
from .geometry import quaternion_to_matrix, euler_xyz_to_matrix, rotvec_to_matrix
from .manifest import WindowIndex
from .schema import file_sha256, within

def field(column, indices=None, **kwargs):
    return dict(column=column, **{'indices': indices} if indices is not None else {}, **kwargs)

def robotwin_fields():
    result = {}
    for kind, col in (('state', 'observation.state'), ('action', 'action')):
        result[kind] = {}
        for arm, off in (('left', 0), ('right', 7)):
            result[kind][arm] = dict(joints=field(col, list(range(off, off + 6)), command='absolute_position'), gripper=field(col, [off + 6], calibration=dict(closed=0, opened=1)))
    return result

class SourceAdapter:

    def __init__(self, manifest, *, split='train', horizon=32, stride=1, window_stride=1, language=None):
        self.manifest = manifest
        self.source = manifest['source']
        self.root = Path(manifest['root'])
        self.index = WindowIndex(manifest, split, window_stride, language)
        self.horizon, self.stride = (int(horizon), int(stride))
        if self.horizon < 1 or self.stride < 1:
            raise ValueError('Invalid chunk horizon/stride')
        self.codec = ActionCodec(self.source['protocol'])
        for required in ('source_id', 'stats_key', 'embodiment', 'coordinate_frame', 'endpoint', 'fields'):
            if required not in self.source:
                raise ValueError(f'Source contract requires {required}')
        for relative, digest in manifest.get('metadata_hashes', {}).items():
            if file_sha256(within(self.root, relative)) != digest:
                raise ValueError(f'Stale source manifest: {relative}')
        self._cache, self._verified_files = (OrderedDict(), set())
        self.sidecar = None
        from .robotwin_joint14 import NATIVE_PROFILES, CLEAN_PROFILE, validate_source
        joint14 = self.source.get('supervision_profile') in NATIVE_PROFILES
        if joint14:
            validate_source(self.source)
            if self.source['supervision_profile'] == CLEAN_PROFILE:
                from .robotwin_cleanonly import validate_manifest
                validate_manifest(manifest)
        if self.source.get('adapter') == 'robotwin' and (not joint14):
            from .sidecar import RobotwinSidecar
            self.sidecar = RobotwinSidecar(self.root, self.source['sidecar_root'])
        for arm in self.source['fields'].get('action', {}).values():
            if 'joints' in arm and arm['joints'].get('command') != 'absolute_position':
                raise ValueError('Joint actions require an explicit absolute_position protocol')
        for kind in ('state', 'action'):
            base = self.source['fields'].get(kind, {}).get('base')
            if base is not None and (set(base) != {'values'} or not base['values'].get('protocol')):
                raise ValueError('Base fields require values with an explicit position/velocity protocol')

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'], state['_verified_files'] = (OrderedDict(), set())
        return state

    def episode(self, ep):
        eid = ep['episode_id']
        if eid in self._cache:
            self._cache.move_to_end(eid)
            return self._cache[eid]
        columns, companion, poses = ({}, {}, None)
        if 'data_path' in ep:
            import pyarrow.parquet as pq
            path = within(self.root, ep['data_path'])
            if str(path) not in self._verified_files:
                if file_sha256(path) != ep['data_sha256']:
                    raise ValueError(f'Source parquet changed: {path}')
                self._verified_files.add(str(path))
            required = {'frame_index', 'episode_index', 'timestamp'}
            for arms in self.source['fields'].values():
                for fields in arms.values():
                    for spec in fields.values():
                        required.add(spec['column'])
                        if 'valid_column' in spec:
                            required.add(spec['valid_column'])
            if self.sidecar is not None:
                required.update(('observation.state', 'action'))
            schema = pq.ParquetFile(path).schema_arrow.names
            table = pq.read_table(path, columns=sorted(required.intersection(schema)), filters=[('episode_index', '=', eid)])
            if table.num_rows != ep['length']:
                raise ValueError(f'Wrong episode population in shard: {eid}')
            columns = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
            if not np.array_equal(columns['frame_index'], np.arange(ep['length'])) or not np.all(columns['episode_index'] == eid):
                raise ValueError(f'Episode/frame alignment mismatch: {eid}')
            if self.sidecar is not None:
                poses = self.sidecar.load(ep, columns)
            calibration = ep.get('metadata', {}).get('calibration_path')
            if calibration:
                companion = json.loads(within(self.root, calibration).read_text())
                if companion.get('frame_mapping', {}).get('count') != ep['length']:
                    raise ValueError('Companion frame mapping mismatch')
        result = (columns, companion, poses)
        self._cache[eid] = result
        if len(self._cache) > 4:
            self._cache.popitem(last=False)
        return result

    def fields(self, kind, columns, indices, companion):
        result = {}
        for arm, specs in self.source['fields'].get(kind, {}).items():
            result[arm] = {'valid': {}}
            for name, spec in specs.items():
                col = spec['column']
                if col not in columns:
                    continue
                provenance = spec.get('native_alignment')
                if provenance and companion.get('native_alignment', {}).get(provenance, {}).get('mode') != 'source':
                    continue
                x = np.asarray(columns[col][indices], dtype=np.float64)
                leading = () if np.ndim(indices) == 0 else (len(indices),)
                x = x.reshape(*leading, -1)
                if 'expected_width' in spec and x.shape[-1] != spec['expected_width']:
                    raise ValueError(f"{col}: expected {spec['expected_width']} channels, got {x.shape[-1]}; check source variant")
                if 'indices' in spec:
                    x = x[..., spec['indices']]
                m = np.isfinite(x)
                if 'valid_column' in spec:
                    m &= np.asarray(columns[spec['valid_column']][indices], bool).reshape(*leading, -1)
                x = np.where(m, x, 0) * float(spec.get('scale', 1))
                if 'calibration' in spec:
                    x, gv = GripperCalibration(**spec['calibration']).encode(x)
                    m &= gv
                representation = spec.get('rotation')
                if representation:
                    good = m.all(-1)
                    raw = x.copy()
                    if representation in ('xyzw', 'wxyz'):
                        good &= np.linalg.norm(raw, axis=-1) > 1e-08
                        raw[~good] = [0, 0, 0, 1] if representation == 'xyzw' else [1, 0, 0, 0]
                        x = quaternion_to_matrix(raw, representation)
                    elif representation == 'euler_xyz':
                        x = euler_xyz_to_matrix(raw)
                    elif representation == 'rotvec':
                        x = rotvec_to_matrix(raw)
                    elif representation == 'matrix':
                        x = raw.reshape(*leading, 3, 3)
                        good &= np.max(np.abs(np.swapaxes(x, -1, -2) @ x - np.eye(3)), axis=(-2, -1)) < 0.0001
                        good &= np.abs(np.linalg.det(x) - 1) < 0.0001
                    elif representation in ('6d_rows', '6d_columns'):
                        u = raw[..., :3]
                        v = raw[..., 3:]
                        norm = np.linalg.norm(u, axis=-1, keepdims=True)
                        good &= norm[..., 0] > 1e-08
                        u = u / np.maximum(norm, 1e-08)
                        v = v - (u * v).sum(-1, keepdims=True) * u
                        norm = np.linalg.norm(v, axis=-1, keepdims=True)
                        good &= norm[..., 0] > 1e-08
                        v = v / np.maximum(norm, 1e-08)
                        x = np.stack([u, v, np.cross(u, v)], axis=-1 if representation == '6d_columns' else -2)
                    else:
                        raise ValueError(f'Unknown rotation convention: {representation}')
                    m = np.broadcast_to(np.asarray(good)[..., None, None], x.shape)
                result[arm][name] = x
                result[arm]['valid'][name] = m
            if arm == 'base':
                block = result.pop('base')
                if 'values' in block:
                    result['base'] = block['values']
                    result.setdefault('valid', {})['base'] = block['valid']['values']
        return result

    def action_fields(self, columns, indices, companion, *, length):
        return self.fields('action', columns, indices, companion)

    def labels(self, index):
        ep, anchor, end = self.index.locate(index)
        columns, companion, poses = self.episode(ep)
        raw_indices = anchor + np.arange(self.horizon) * self.stride
        valid = raw_indices < end
        target_offset = int(self.source.get('action_target_offset_frames', 1 if self.sidecar is not None else 0))
        if target_offset < 0:
            raise ValueError('Action target offsets cannot precede the observation')
        if end < ep['length']:
            valid &= raw_indices + target_offset < end
        if self.source.get('terminal_action_valid', True) is False:
            valid &= raw_indices + target_offset < ep['length']
        rows = np.minimum(raw_indices, end - 1)
        state = self.fields('state', columns, anchor, companion)
        actions = self.action_fields(columns, rows, companion, length=ep['length'])
        if poses is not None:
            targets = np.minimum(rows + 1, ep['length'] - 1)
            for i, arm in enumerate(('left', 'right')):
                state[arm].update(position=poses[anchor, i, :3], rotation=quaternion_to_matrix(poses[anchor, i, 3:]))
                actions[arm].update(position=poses[targets, i, :3], rotation=quaternion_to_matrix(poses[targets, i, 3:]))
        result = self.codec.encode(state, actions, self.horizon, time_valid=valid)
        result.update(episode_id=ep['episode_id'], frame_index=anchor)
        return (result, ep, anchor, end)

class RoboTwinAdapter(SourceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.source.get('supervision_profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
            from .robotwin_eef_cleanonly import validate_source
            from .robotwin_cleanonly import validate_manifest
            validate_source(self.source)
            validate_manifest(self.manifest)
            if self.sidecar is None:
                raise ValueError('EEF sidecars are required')

    def labels(self, index):
        result, ep, anchor, end = super().labels(index)
        if self.source.get('supervision_profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
            from .robotwin_eef_cleanonly import ACTION_EEF
            targets = anchor + np.arange(self.horizon) * self.stride + 1
            good = targets < min(end, ep['length'])
            mask = result['action_valid'][:, ACTION_EEF] & good[:, None]
            result['action_valid'][:, ACTION_EEF] = mask
            result['action'][:, ACTION_EEF] = np.where(mask, result['action'][:, ACTION_EEF], 0)
        return (result, ep, anchor, end)
ADAPTERS = {'robotwin': RoboTwinAdapter}
