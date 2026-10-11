"""RSI2-only preflight: reject whole paths; never rewrite planned angles."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

VERSION = "rsi2-path-preflight-v1"
MAX_NET_RAD = 2 * math.pi
MAX_TRAVEL_RAD = 2 * math.pi
START_TOL_RAD = 0.05


def assess(start, positions):
    q = np.asarray(start, dtype=float)
    p = np.asarray(positions, dtype=float)
    if q.ndim != 1 or p.ndim != 2 or not len(p) or p.shape[1] != len(q):
        raise ValueError("path/start shape mismatch")
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite path/start")
    start_error = np.abs(p[0] - q)
    net = np.abs(p[-1] - q)
    # Include the initial state-to-first-waypoint segment; inspect full path,
    # including portions that native budget truncation would not execute.
    travel = np.abs(np.diff(np.vstack((q, p)), axis=0)).sum(axis=0)
    reasons = []
    if np.any(start_error > START_TOL_RAD):
        reasons.append("start_mismatch")
    if np.any(net > MAX_NET_RAD):
        reasons.append("excessive_net_rotation")
    if np.any(travel > MAX_TRAVEL_RAD):
        reasons.append("excessive_total_rotation")
    return dict(version=VERSION, accepted=not reasons, reasons=reasons,
                start_error_rad=start_error.tolist(), net_rad=net.tolist(),
                travel_rad=travel.tolist(), start_tolerance_rad=START_TOL_RAD,
                max_net_rad=MAX_NET_RAD, max_travel_rad=MAX_TRAVEL_RAD)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def begin(env, request, start, positions, policy_file, planner_source_file=None):
    """Save raw planner output and return a zero-step decision before sampling."""
    if getattr(env, "task_name", None) != "classify_objects_by_language":
        raise ValueError("RSI2 path guard task mismatch")
    run = Path(os.environ["RSI2_CLASSIFY_RUN"]).resolve()
    if run.parent != Path("/manifold-obs/ken/rsi2_classify_work"):
        raise ValueError("RSI2 path guard output outside own run root")
    root = run / "path_preflight"
    root.mkdir(exist_ok=True)
    source_dir = Path(policy_file).resolve().parent
    snapshots = root / "policy_sources"
    snapshots.mkdir(exist_ok=True)
    sources = {}
    for src in sorted(source_dir.iterdir()):
        if src.is_file() and src.suffix in (".py", ".yml"):
            data = src.read_bytes()
            dst = snapshots / src.name
            try:
                with dst.open("xb") as out:
                    out.write(data)
            except FileExistsError:
                if dst.read_bytes() != data:
                    raise ValueError("policy source changed during episode")
            sources[src.name] = hashlib.sha256(data).hexdigest()
    if planner_source_file:
        original = Path(planner_source_file)
        data = original.read_bytes()
        dst = snapshots / 'native_planner_source.py'
        try:
            with dst.open('xb') as out:
                out.write(data)
        except FileExistsError:
            if dst.read_bytes() != data:
                raise ValueError('native planner source changed during episode')
        sources['native_planner_source.py'] = hashlib.sha256(data).hexdigest()
    stem = root / (str(int(env.take_action_cnt[0])) + "_" + request["arm"] + "_" + uuid.uuid4().hex)
    raw = stem.with_suffix(".npz")
    with raw.open("xb") as out:
        np.savez_compressed(out, planner_positions=np.asarray(positions, dtype=float),
                            actual_start_joints=np.asarray(start, dtype=float))
    verdict = assess(start, positions)
    record = dict(verdict, native_step=int(env.take_action_cnt[0]), arm=request["arm"],
                  target_pose=request["target_pose"], start_pose=request.get('start_pose'),
                  target_joints=request.get("target_joints"), planner_source_file=planner_source_file,
                  raw_planner_file=str(raw), raw_planner_sha256=_sha(raw),
                  policy_sources_sha256=sources, emitted_actions=0,
                  actual_execution_source="join original HDF5 state/action by native_step; no replay inferred")
    meta = stem.with_suffix(".json")
    with meta.open("x") as out:
        json.dump(record, out, indent=2)
    return verdict, str(meta)


def record_resampled(meta_path, actions, arm_key):
    """Separate emitted targets from raw planner points; no actual-state claim."""
    meta = Path(meta_path)
    sampled = meta.with_name(meta.stem + "_resampled.npz")
    with sampled.open("xb") as out:
        np.savez_compressed(out, resampled_targets=np.asarray([a[arm_key] for a in actions]))
    record = json.loads(meta.read_text())
    record.update(emitted_actions=len(actions), resampled_file=str(sampled),
                  resampled_sha256=_sha(sampled))
    temporary = meta.with_suffix(".tmp")
    with temporary.open("x") as out:
        json.dump(record, out, indent=2)
    os.replace(temporary, meta)
