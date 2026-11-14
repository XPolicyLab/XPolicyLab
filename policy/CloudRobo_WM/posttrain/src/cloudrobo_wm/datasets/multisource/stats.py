import json
from cloudrobo_wm.storage import DataPath as Path
import numpy as np
from .schema import STATE, ACTION, STATE_DIM, ACTION_DIM, SCHEMA_VERSION, CODEC_VERSION, atomic_json, fingerprint

class StreamingStats:

    def __init__(self, dim, capacity=100000, seed=0, compute_quantiles=True):
        if capacity < 1:
            raise ValueError('reservoir capacity must be positive')
        self.dim, self.capacity, self.seed = (dim, capacity, seed)
        self.compute_quantiles = bool(compute_quantiles)
        self.count = np.zeros(dim, np.int64)
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.values = [np.empty(0) for _ in range(dim)]
        self.keys = [np.empty(0) for _ in range(dim)]
        self.rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(dim)]

    def update(self, values, valid):
        x, m = (np.asarray(values).reshape(-1, self.dim), np.asarray(valid, bool).reshape(-1, self.dim))
        for d in range(self.dim):
            v = x[m[:, d] & np.isfinite(x[:, d]), d].astype(np.float64)
            n = len(v)
            if not n:
                continue
            old, avg = (self.count[d], v.mean())
            total = old + n
            delta = avg - self.mean[d]
            self.m2[d] += ((v - avg) ** 2).sum() + delta ** 2 * old * n / total
            self.mean[d] += delta * n / total
            self.count[d] = total
            if not self.compute_quantiles:
                continue
            keys = np.concatenate((self.keys[d], self.rng[d].random(n)))
            vals = np.concatenate((self.values[d], v))
            if len(keys) > self.capacity:
                take = np.argpartition(keys, self.capacity - 1)[:self.capacity]
                keys, vals = (keys[take], vals[take])
            self.keys[d], self.values[d] = (keys, vals)

    def finish(self):
        q = np.array([np.quantile(v, [0.01, 0.99]) if len(v) else [0, 0] for v in self.values])
        return dict(count=self.count.tolist(), mean=self.mean.tolist(), std=np.sqrt(self.m2 / np.maximum(self.count, 1)).tolist(), q01=q[:, 0].tolist(), q99=q[:, 1].tolist(), reservoir_capacity=self.capacity, reservoir_seed=self.seed, quantiles_approximate=self.compute_quantiles and bool((self.count > self.capacity).any()), quantiles_computed=self.compute_quantiles)

    def merge_moments(self, other):
        if self.compute_quantiles or other.compute_quantiles:
            raise ValueError('Moment merging requires quantiles disabled')
        self._merge_moments(other)

    def _merge_moments(self, other):
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta ** 2 * self.count * other.count / np.maximum(total, 1)
        self.mean += delta * other.count / np.maximum(total, 1)
        self.count = total

    def merge_priority(self, other):
        if not self.compute_quantiles or not other.compute_quantiles or self.dim != other.dim or (self.capacity != other.capacity):
            raise ValueError('Priority merging needs matching quantile reservoirs')
        self._merge_moments(other)
        for d in range(self.dim):
            keys = np.concatenate((self.keys[d], other.keys[d]))
            values = np.concatenate((self.values[d], other.values[d]))
            if len(keys) > self.capacity:
                indices = np.argpartition(keys, self.capacity - 1)[:self.capacity]
                keys, values = (keys[indices], values[indices])
            self.keys[d], self.values[d] = (keys, values)

def default_contract(**kwargs):
    return dict(schema=SCHEMA_VERSION, codec=CODEC_VERSION, **kwargs)

class StatsRegistry:

    def __init__(self, groups, contract):
        self.groups, self.contract = (groups, contract)
        for key, group in groups.items():
            if group.get('profile'):
                from .robotwin_joint14 import validate_group
                validate_group(group)
            if group['mode'] not in {'quantile', 'z-score'}:
                raise ValueError(f'{key}: unknown normalization mode')
            if group['mode'] == 'quantile' and any((group[k].get('quantiles_computed') is False for k in ('state', 'action'))):
                raise ValueError(f'{key}: quantile normalization requires computed quantiles')
            for kind, dim in (('state', STATE_DIM), ('action', ACTION_DIM)):
                for stat in ('count', 'mean', 'std', 'q01', 'q99'):
                    x = np.asarray(group[kind][stat])
                    if x.shape != (dim,) or not np.isfinite(x).all():
                        raise ValueError(f'{key}/{kind}/{stat}: invalid statistics')

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return dict(version=1, contract=self.contract, groups=self.groups)

    def save(self, path):
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path, expected_contract=None):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('version') != 1:
            raise ValueError('Unsupported statistics version')
        if expected_contract is not None and payload['contract'] != expected_contract:
            raise ValueError('Statistics contract mismatch; regenerate from the training split')
        return cls(payload['groups'], payload['contract'])

    def _affine(self, key, kind):
        group = self.groups[key]
        stats, layout = (group[kind], STATE if kind == 'state' else ACTION)
        if group['mode'] == 'quantile':
            low, high = (np.asarray(stats['q01']), np.asarray(stats['q99']))
            center, scale, bound = ((low + high) / 2, (high - low) / 2, 1.0)
        else:
            center, scale, bound = (np.asarray(stats['mean']), np.asarray(stats['std']), 5.0)
        scale = np.where(scale > 1e-08, scale, 1.0)
        clip = np.ones_like(center, bool)
        for side in ('left', 'right'):
            identity = [layout[side + '.gripper']]
            if kind == 'state':
                identity.append(layout[side + '.rotation'])
            else:
                clip[layout[side + '.rotation']] = False
            for sl in identity:
                center[sl], scale[sl], clip[sl] = (0, 1, False)
        return (center, scale, clip, bound, np.asarray(stats['count']) > 0)

    def normalize(self, key, kind, values, valid, *, report=False):
        x, valid = (np.asarray(values), np.asarray(valid, bool))
        if self.groups[key].get('profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
            from .robotwin_eef_cleanonly import normalize
            return normalize(self.groups[key], kind, values, valid, report=report)
        if self.groups[key].get('profile'):
            from .robotwin_joint14 import native_affine
            scale, offset, known = native_affine(self.groups[key], kind)
            m = valid & np.isfinite(x)
            if (m & ~known).any():
                raise ValueError('Native joint14 received valid labels outside its 14 slots')
            y = np.where(m, x, 0).astype(np.float32) * scale + offset
            clipped = m & (np.abs(y) > 5)
            y = np.where(m, np.clip(y, -5, 5), 0).astype(np.float32)
            return (y, dict(clipped=int(clipped.sum()), valid=int(m.sum()))) if report else y
        center, scale, clip, bound, known = self._affine(key, kind)
        m = valid & np.isfinite(x)
        if (m & ~known).any():
            raise ValueError(f'{key}/{kind}: valid label has no training statistics')
        y = (np.where(m, x, center) - center) / scale
        clipped = m & clip & (np.abs(y) > bound)
        y = np.where(clip, np.clip(y, -bound, bound), y)
        y = np.where(m, y, 0).astype(np.float32)
        if report:
            return (y, dict(clipped=int(clipped.sum()), valid=int(m.sum())))
        return y

    def denormalize(self, key, kind, values, valid=None):
        if self.groups[key].get('profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
            from .robotwin_eef_cleanonly import denormalize
            return denormalize(self.groups[key], kind, values, valid)
        x = np.asarray(values)
        if self.groups[key].get('profile'):
            from .robotwin_joint14 import native_affine
            scale, offset, known = native_affine(self.groups[key], kind)
            m = known if valid is None else np.asarray(valid, bool) & known
            return np.where(m, (x.astype(np.float32) - offset) / scale, 0).astype(np.float32)
        center, scale, _, _, known = self._affine(key, kind)
        m = known if valid is None else np.asarray(valid, bool) & known
        return np.where(m, x * scale + center, 0).astype(np.float32)
