"""Child process that hosts one frozen LILO policy for one episode.

Run as `python runtime.py <policy.py> <task> [depth_provider] [model_dir]`. depth_provider 'mono' fills
obs['vision'][cam]['depth'] from RGB + camera matrices (stereo.MonoPlaneDepth: Depth Anything V2 + table-plane
calibration) before every call, so policies written for RGB-D run on RGB-only observations. The parent sends length-prefixed pickles of (op, obs) with op in
{'reset', 'act'}; the child answers one JSON line {"ok": bool, "result"|"error": ...}. A fresh process per episode
keeps module-level state from leaking between episodes, matching how the policies were developed and tested.
"""
import contextlib
import importlib.util
import json
import os
import pickle
import struct
import sys

import numpy as np


def from_wire(value):
    if isinstance(value, tuple) and len(value) == 4 and value[0] == '__nd__':
        return np.frombuffer(value[3], dtype=np.dtype(value[1])).reshape(value[2]).copy()
    if isinstance(value, dict):
        return {k: from_wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(from_wire(v) for v in value)
    return value


def main(policy_path, task, depth_provider='none', model_dir=None):
    from real_rsi_assets import model_path
    os.environ['LILO_MODELS'] = model_path()
    source, sink = sys.stdin.buffer, sys.stdout
    null = open(os.devnull, 'w')
    estimator = None
    with contextlib.redirect_stdout(null):
        if depth_provider == 'mono':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from stereo import DA2_LARGE, MonoPlaneDepth
            estimator = MonoPlaneDepth(model_dir or DA2_LARGE)
        sys.path.insert(0, os.path.dirname(os.path.abspath(policy_path)))
        spec = importlib.util.spec_from_file_location('frozen_policy', policy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy = module.Policy(task=task, seed=0)
    while True:
        header = source.read(8)
        if not header:
            return
        length, = struct.unpack('!Q', header)
        op, obs = pickle.loads(source.read(length))
        obs = from_wire(obs)
        try:
            if estimator is not None:
                cams = {c: v for c, v in obs['vision'].items() if 'color' in v and 'intrinsic_matrix' in v}
                with contextlib.redirect_stdout(null):
                    for c, d in estimator.estimate(cams).items():
                        obs['vision'][c]['depth'] = d
            with contextlib.redirect_stdout(null):
                result = getattr(policy, op)(obs)
            line = json.dumps(dict(ok=True, result=result), allow_nan=False)
        except Exception as exc:
            line = json.dumps(dict(ok=False, error=repr(exc)))
        sink.write(line + '\n')
        sink.flush()


if __name__ == '__main__':
    main(*sys.argv[1:5])
