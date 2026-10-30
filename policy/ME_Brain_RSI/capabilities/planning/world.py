"""Shared measured object state with explicit provenance and observability."""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np


def _plain(value):
    """Copy public evidence into finite JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite measured evidence')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported measured evidence type: '+type(value).__name__)


@dataclass(frozen=True)
class Measurement:
    object_id: str
    category: str
    values: dict
    error: dict
    source: str
    cameras: tuple
    measurement_step: int
    accepted: bool
    observable_fields: tuple
    reason: str = ''
    asset: dict = None
    relations: dict = None
    role: str = 'geometric_measurement'
    assumptions: dict = None
    identity: dict = None

    def __post_init__(self):
        if not self.object_id or not self.category or not self.source:
            raise ValueError('Measurements require identity, category and provenance')
        if type(self.measurement_step) is not int or self.measurement_step < 0:
            raise ValueError('Invalid measurement step')
        if type(self.accepted) is not bool:
            raise TypeError('Measurement acceptance must be explicit')
        for name in ('values', 'error', 'asset', 'relations', 'assumptions', 'identity'):
            object.__setattr__(self, name, _plain(getattr(self, name) or {}))
        object.__setattr__(self, 'cameras', tuple(self.cameras))
        object.__setattr__(self, 'observable_fields', tuple(self.observable_fields))
        if 'position' in self.values and np.asarray(self.values['position']).shape != (3,):
            raise ValueError('Measured position must have three components')
        if 'quaternion' in self.values:
            quaternion = np.asarray(self.values['quaternion'], float)
            if quaternion.shape != (4,) or abs(np.linalg.norm(quaternion)-1.) > .001:
                raise ValueError('Measured quaternion must be normalized')
        if any(field not in self.values for field in self.observable_fields):
            raise ValueError('Observable fields must exist in the measurement')
        if not isinstance(self.role, str) or not self.role:
            raise ValueError('A measurement role is required')
        if self.accepted and (not self.source.startswith('rgb_') or not self.cameras
                              or not self.error or not self.observable_fields):
            raise ValueError('Accepted RGB state requires cameras, errors and observable fields')

    def as_dict(self):
        return deepcopy({name: _plain(getattr(self, name)) for name in self.__dataclass_fields__})


class WorldState:
    """A rejected current observation supersedes an older accepted measurement."""
    def __init__(self):
        self.records = {}
        self.detection_batches = []
        self.detection_sequence = 0

    def update(self, measurement):
        if not isinstance(measurement, Measurement):
            raise TypeError('World state accepts measured records only')
        previous = self.records.get(measurement.object_id)
        if previous is not None and measurement.measurement_step < previous.measurement_step:
            raise ValueError('An old observation cannot replace current object state')
        self.records[measurement.object_id] = measurement
        return measurement

    def require(self, object_id, current_step, max_age_steps=0, fields=('position',)):
        if type(current_step) is not int or current_step < 0:
            raise ValueError('Invalid current observation step')
        if type(max_age_steps) is not int or max_age_steps < 0:
            raise ValueError('Invalid measurement age allowance')
        record = self.records.get(object_id)
        if record is None:
            raise RuntimeError('No measured state for '+object_id)
        age = current_step-record.measurement_step
        if age < 0 or age > max_age_steps:
            raise RuntimeError('Object measurement is not current: '+object_id)
        if not record.accepted:
            raise RuntimeError('Current object state is not accepted: '+object_id+'; '+record.reason)
        role = record.error.get('measurement_role', record.role)
        if role in ('coarse_detection', 'coarse_observation') and any(
                field in ('position', 'quaternion') for field in fields):
            raise RuntimeError('Coarse observation cannot authorize a precise object operation: '+object_id)
        if any(field not in record.observable_fields for field in fields):
            raise RuntimeError('Required object fields are not independently observable: '+object_id)
        return record

    def as_dict(self):
        return {key: value.as_dict() for key, value in self.records.items()}

    def detections_as_dict(self):
        """Return bounded observation-local candidates, separate from tracked objects."""
        return deepcopy(self.detection_batches)

    def record_detection_batch(self, batch, max_batches=16):
        if type(max_batches) is not int or max_batches <= 0:
            raise ValueError('Invalid detection history limit')
        batch = _plain(batch)
        if type(batch.get('measurement_step')) is not int or batch['measurement_step'] < 0:
            raise ValueError('Invalid detection observation step')
        self.detection_batches.append(batch)
        del self.detection_batches[:-max_batches]
        return deepcopy(batch)


def record_measurement(ctx, object_id, category, values, *, error, source,
                       cameras, accepted, observable_fields=(), reason='',
                       asset=None, relations=None, role='geometric_measurement',
                       assumptions=None, identity=None):
    """Record only values supported by the current public RGB observation."""
    if not hasattr(ctx, 'world'):
        ctx.world = WorldState()
    record = Measurement(object_id, category, values, error, source, tuple(cameras),
                         int(ctx.steps), accepted, tuple(observable_fields), reason,
                         asset, relations, role, assumptions, identity)
    return ctx.world.update(record)


def require_current(ctx, object_id, max_age_steps=0, fields=('position',)):
    return ctx.world.require(object_id, int(ctx.steps), max_age_steps, fields)
