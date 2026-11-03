"""Publish and verify exact observation files before any paid model request."""
import hashlib
import json
from pathlib import Path
import shutil
from ..runtime.io_utils import atomic, read_json

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_numbers(value, digits=10):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [compact_numbers(x, digits) for x in value]
    if isinstance(value, dict):
        return {k: compact_numbers(v, digits) for k, v in value.items()}
    return value


def publish_manifest(folder, request, obs):
    files = {'state': {'name': 'observation.json', 'sha256': digest(folder/'observation.json')}}
    for camera in CAMERAS:
        path = folder/(camera+'.jpg')
        if not path.is_file() or path.stat().st_size < 20:
            raise RuntimeError('Missing observation image: ' + camera)
        files[camera] = {'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size}
        rgb = folder/(camera+'.rgb.npy')
        files[camera+'_rgb'] = {'name': rgb.name, 'sha256': digest(rgb), 'bytes': rgb.stat().st_size}
    value = {k: request[k] for k in ('run_id', 'task', 'request_id', 'observation_id', 'kind')}
    value.update(sim_time_s=obs['sim_time_s'], cameras=list(CAMERAS), files=files)
    atomic(folder/'manifest.json', value)
    return value


def stage_observation(root, sandbox, req):
    source = root/'observations'/req['observation_id']
    m = read_json(source/'manifest.json')
    if not m or any(m.get(k) != req[k] for k in ('run_id', 'task', 'request_id', 'observation_id', 'kind')):
        raise RuntimeError('Observation manifest attribution mismatch')
    target = sandbox.observations/req['observation_id']
    target.mkdir(exist_ok=True)
    for item in m['files'].values():
        name = item['name']
        if Path(name).name != name or digest(source/name) != item['sha256']:
            raise RuntimeError('Observation source integrity mismatch: '+name)
        if (target/name).exists() and digest(target/name) != item['sha256']:
            raise RuntimeError('Refusing to replace conflicting observation: '+name)
        shutil.copy2(source/name, target/name)
        if digest(target/name) != item['sha256']:
            raise RuntimeError('Observation sandbox copy integrity mismatch: '+name)
    shutil.copy2(source/'manifest.json', target/'manifest.json')
    # Backward-compatible directory name, pointing to the read-only observation mount.
    alias = sandbox.work/req['observation_id']
    if not alias.exists():
        alias.symlink_to('observations/'+req['observation_id'], target_is_directory=True)
    return source, m, read_json(target/'observation.json')


def precise_json(value):
    """JSON numbers with ten fractional digits, without converting numbers to strings."""
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise ValueError('Nonfinite observation')
        return f'{value:.10f}'
    if isinstance(value, dict):
        return '{'+','.join(json.dumps(k, ensure_ascii=False)+':'+precise_json(v) for k,v in value.items())+'}'
    if isinstance(value, (list, tuple)):
        return '['+','.join(precise_json(v) for v in value)+']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
