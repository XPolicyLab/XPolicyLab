"""Pose-plan / joint-plan playback with one speed cap and short endpoint ramps.

Targets are signed, ordered frames. No jaw flip, stage-generated waypoints,
speed classes or contact-only Cartesian controller is selected internally.
"""
import copy
import itertools
import math
import time
import numpy as np
from ..runtime.protocol import ARMS, MAX_GROUP_SECONDS, frame
from .robot_geometry import matrix_to_quat_wxyz
from .executor import Replan, angle

# Reference FREE_STEP_RAD=.08 at 25 Hz. Use physical time for other control frequencies.
JOINT_SPEED_RAD_S = 2.
START_STOP_RAMP_S = .12
SWEEP_LIMIT_M = .12
PATH_SAMPLE_RAD = .05
MAX_PATH_SAMPLES = 10000
ROTATION_ALLOWANCE_RAD = math.radians(20)
ROTATION_ALLOWANCE_RATIO = .5
# Conservative first guard: the archived 4.5 cm / 0.18 degree move swung joints
# >150 degrees. A 4.7 cm / 7.31 degree move used 63.7 degrees and stays eligible.
LOCAL_MOVE_M = .05
LOCAL_ROTATION_DEG = 10.
LOCAL_JOINT_EXCURSION_DEG = 90.
LIMIT_PREFERENCE_DEG = 5.
MAX_IK_PATH_CANDIDATES = 4
MAX_GROUP_PATH_COMBINATIONS = (1 + MAX_IK_PATH_CANDIDATES) ** len(ARMS)


def ramped_progress(fraction, duration_s, ramp_s):
    """Normalized path progress with linear speed ramps at the two endpoints.

    This smooths startup and stopping, not changes of direction at interior
    planner knots. The caller reserves ramp time before enforcing step budgets.
    """
    if not ramp_s:
        return fraction
    r = ramp_s / duration_s
    if fraction < r:
        return fraction*fraction / (2*r*(1-r))
    if fraction > 1-r:
        return 1-(1-fraction)**2 / (2*r*(1-r))
    return (fraction-r/2) / (1-r)


def joint_path_metrics(b, arm, start, path):
    samples = np.vstack([start, path])
    travel = np.abs(np.diff(samples, axis=0)).sum(axis=0)
    excursion = np.abs(samples-np.asarray(start)).max(axis=0)
    metrics = {'joint_travel_deg': np.rad2deg(travel).tolist(),
               'max_joint_excursion_deg': float(np.rad2deg(excursion.max())),
               'joint_displacement_rad': float(np.linalg.norm(samples[-1]-start)),
               'joint_path_length_rad': float(np.linalg.norm(np.diff(samples,axis=0),axis=1).sum())}
    limits = getattr(b, 'joint_limits', {}).get(arm)
    if limits is not None:
        margins = np.minimum(samples-limits.bounds[:,0], limits.bounds[:,1]-samples)
        # Continuous joints have no endstop; omit infinite margins from JSON.
        for name, value in [('min_joint_limit_margin_deg', margins.min()),
                            ('goal_joint_limit_margin_deg', margins[-1].min())]:
            metrics[name] = float(np.rad2deg(value)) if np.isfinite(value) else None
    return metrics


def candidate_cost(metrics):
    margin = metrics.get('goal_joint_limit_margin_deg')
    limit_penalty = 0. if margin is None else max(0., 1.-margin/LIMIT_PREFERENCE_DEG)
    return (metrics['joint_displacement_rad'] + .25*math.radians(metrics['max_joint_excursion_deg'])
            + .15*metrics['joint_path_length_rad'] + .5*limit_penalty)


def path_diagnostics(b, arm, start, path, goal_p, goal_r):
    """Sample joint transitions, not just endpoints, to catch 30 -> 150 -> 30 detours.

    This is a sampled geometry guard, not a continuous collision proof or a global
    shortest-path solver. Bounded joints may make the short route unavailable.
    """
    previous = np.asarray(start, float)
    p0, r0 = b.fk(arm, previous)
    direct = angle(r0, goal_r)
    canonical_dot = float(matrix_to_quat_wxyz(r0) @ matrix_to_quat_wxyz(goal_r))
    allowance = max(ROTATION_ALLOWANCE_RAD, ROTATION_ALLOWANCE_RATIO * direct)
    rotation_total, sweep, count = 0., 0., 0
    previous_r = r0
    delta_p = np.asarray(goal_p) - p0
    for q in path:
        n = max(1, math.ceil(float(np.max(abs(q-previous))) / PATH_SAMPLE_RAD))
        count += n
        if count > MAX_PATH_SAMPLES:
            raise Replan('joint path is excessively complex; add an explicit intermediate pose')
        for u in np.linspace(0, 1, n+1)[1:]:
            p, r = b.fk(arm, (1-u)*previous+u*q)
            if not np.isfinite(p).all() or not np.isfinite(r).all():
                raise Replan('non-finite FK during path inspection')
            rotation_total += angle(previous_r, r)
            previous_r = r
            progress = np.clip((p-p0)@delta_p/max(float(delta_p@delta_p), 1e-12), 0, 1)
            sweep = max(sweep, float(np.linalg.norm(p-p0-progress*delta_p)))
        previous = q
    end_p, end_r = b.fk(arm, path[-1])
    if np.linalg.norm(end_p-goal_p) > .005 or angle(end_r, goal_r) > math.radians(5):
        raise Replan('IK pose mismatch; ordered target frame was not reached')
    return {**joint_path_metrics(b, arm, start, path),
            'requested_displacement_m': float(np.linalg.norm(delta_p)),
            'requested_rotation_deg': math.degrees(direct),
            # A negative canonical dot is a representation sign change, not proof
            # of a physical flip. Rotation totals below are sign-invariant SO(3).
            'canonical_quaternion_dot': canonical_dot,
            'target_quaternion_sign_adjusted': canonical_dot < 0,
            'path_rotation_deg': math.degrees(rotation_total),
            'rotation_budget_deg': math.degrees(direct+allowance),
            'grasp_sweep_m': sweep, 'samples': count}


def path_rejection(metrics):
    if metrics['path_rotation_deg'] > metrics['rotation_budget_deg'] + 1e-6:
        return (f"rotation detour: requested {metrics['requested_rotation_deg']:.1f} deg, "
                f"path {metrics['path_rotation_deg']:.1f} deg; use an explicit intermediate "
                'pose or another reachable ordered frame, without swapping finger points')
    if metrics['grasp_sweep_m'] > SWEEP_LIMIT_M:
        return 'arm path sweeps over 12 cm; add an explicit intermediate pose'
    if (metrics['requested_displacement_m'] <= LOCAL_MOVE_M+1e-8
            and metrics['requested_rotation_deg'] <= LOCAL_ROTATION_DEG
            and metrics['max_joint_excursion_deg'] > LOCAL_JOINT_EXCURSION_DEG):
        return (f"small pose change requires {metrics['max_joint_excursion_deg']:.1f} deg joint excursion "
                f"(limit {LOCAL_JOINT_EXCURSION_DEG:g}); change approach, arm or explicit intermediate pose")
    return None


class PosePathSearch:
    """Retain same-frame candidates for one group; expand at most once per arm."""
    def __init__(self, b, arm, start, p, r):
        self.b, self.arm, self.planner = b, arm, b.planners[arm]
        self.start, self.p, self.r = [np.array(v, float, copy=True) for v in (start, p, r)]
        self.candidates, self.attempts, self.goals = [], [], []
        self.expanded = False
        _, self.first = self.inspect(self.planner.plan_path(start, b.planner_target_pose(arm, start, p, r),
            real_robot_pose=copy.deepcopy(b.robots[arm].entity_origin_pose)), 'pose_plan')

    def unpack(self, result):
        b, arm, start, planner = self.b, self.arm, self.start, self.planner
        if result.get('status') != 'Success' or result.get('position') is None:
            return None
        path = np.asarray(result['position'], float)
        if path.ndim != 2 or not len(path) or path.shape[1:] != np.shape(start) or not np.isfinite(path).all():
            raise Replan('invalid planned path')
        goal = b.nearest_joint_equivalent(arm, path[-1], start)
        if not np.array_equal(goal, path[-1]):
            result = planner.plan_joint(start, goal)
            if result.get('status') != 'Success' or result.get('position') is None:
                raise Replan('short legal equivalent joint path unavailable')
            path = np.asarray(result['position'], float)
        if path.ndim != 2 or not len(path) or path.shape[1:] != np.shape(start) or not np.isfinite(path).all():
            raise Replan('invalid equivalent joint path')
        for q in path:
            b.joint_limits[arm].validate(q)
        return path

    def inspect(self, result, route):
        metrics = {}
        try:
            path = self.unpack(result)
            if path is None:
                raise Replan('planner found no path')
            metrics = path_diagnostics(self.b, self.arm, self.start, path, self.p, self.r)
            error = path_rejection(metrics)
        except Replan as exc:
            path, error = None, str(exc)
        record = dict(metrics, route=route, rejection=error, candidate_index=len(self.attempts))
        if metrics:
            record['candidate_cost'] = candidate_cost(metrics)
        self.attempts.append(record)
        if error is None:
            self.candidates.append((record['candidate_cost'], path, record))
        return path, record

    def expand(self, check_deadline=lambda: None):
        if self.expanded:
            return
        check_deadline()
        self.expanded = True
        b, arm, start, p, r = self.b, self.arm, self.start, self.p, self.r
        try:
            # Preserve the exact signed frame and all physical limits. No finger swap.
            self.goals = b.ik_candidates(arm, start, p, r) if hasattr(b, 'ik_candidates') else [b.ik(arm, start, p, r)]
        except Replan as exc:
            self.attempts.append({'route':'seeded_ik_joint_plan', 'rejection':str(exc)})
            return
        ranked = sorted(self.goals, key=lambda q: candidate_cost(joint_path_metrics(b, arm, start, [q])))
        for goal in ranked[:MAX_IK_PATH_CANDIDATES]:
            check_deadline()
            try:
                result = self.planner.plan_joint(start, goal)
            except Replan as exc:
                self.attempts.append({'route':'seeded_ik_joint_plan', 'rejection':str(exc)})
                continue
            self.inspect(result, 'seeded_ik_joint_plan')

    def diagnostics(self, record):
        result = dict(record, explicit_ik_calls=int(self.expanded),
                      candidate_attempts=copy.deepcopy(self.attempts))
        if self.expanded:
            result.update(original_rejection=self.first['rejection'], fallback_attempted=True,
                          distinct_ik_candidates=len(self.goals))
        return result


def plan_path(b, arm, start, p, r):
    """Fast first choice, with a reusable bounded same-pose search for group checks."""
    if not hasattr(b, 'path_diagnostics'):
        b.path_diagnostics = {}
    if not hasattr(b, '_path_searches'):
        b._path_searches = {}
    b.path_diagnostics[arm] = {}
    # Never reuse a search from another action or measured start state.
    b._path_searches.pop(arm, None)
    search = PosePathSearch(b, arm, start, p, r)
    b._path_searches[arm] = search
    first = search.first
    margin = first.get('goal_joint_limit_margin_deg')
    if not (first['rejection'] is None and first['max_joint_excursion_deg'] <= 60.
            and (margin is None or margin >= LIMIT_PREFERENCE_DEG)):
        search.expand()
    chosen = min(search.candidates, key=lambda item:item[0]) if search.candidates else None
    fallback_record = first if first['rejection'] != 'planner found no path' else search.attempts[-1]
    rejected = [x for x in search.attempts if 'candidate_cost' in x]
    record = chosen[2] if chosen else min(rejected, key=lambda x:x['candidate_cost'], default=fallback_record)
    b.path_diagnostics[arm] = search.diagnostics(record)
    if chosen is None:
        raise Replan(record['rejection'] or f'{arm} target cannot be planned; change the explicit pose')
    return chosen[1]


def path_alternatives(b, arm, start, p, r, check_deadline):
    search = getattr(b, '_path_searches', {}).get(arm)
    if search is None or not all(np.array_equal(a, v) for a, v in
                                ((search.start, start), (search.p, p), (search.r, r))):
        raise Replan('no matching current-group path search; replan from the current state')
    search.expand(check_deadline)
    return [(path, search.diagnostics(record)) for _, path, record in
            sorted(search.candidates, key=lambda item:item[0])]


def compile_motion(e, group):
    b = e.b
    started = time.monotonic()
    active = ARMS if group['active_arms'] == 'both' else (() if group['active_arms'] == 'none' else (group['active_arms'],))
    starts = {a: b.pose(a) for a in ARMS}
    goals = {a: frame(group[a]) for a in active}
    constrained = [a for a in active if group[a]['contact_constraint'] is not None]
    for a in constrained:
        e._contact(group[a], starts[a], 'start')
        e._contact(group[a], goals[a], 'goal')
    q0 = b.joints()
    raw_paths, diagnostics = {}, {}
    # Publish diagnostics as they become available, including preparation failures.
    e.last_feedback['path_diagnostics'] = diagnostics
    calls = 0
    for a in active:
        e.deadline_check()
        if np.linalg.norm(goals[a][0]-starts[a][0]) <= 1e-6 and angle(starts[a][1], goals[a][1]) <= 1e-5:
            continue
        try:
            raw_paths[a] = np.asarray(b.plan_free(a, q0[a], *goals[a]), float)
        finally:
            diagnostics[a] = copy.deepcopy(getattr(b, 'path_diagnostics', {}).get(a, {}))
        calls += 1
        e.deadline_check()
    attempts = []
    e.last_feedback['group_path_attempts'] = attempts

    def check(paths, records):
        e.deadline_check()
        trial = copy.deepcopy(records)
        e.last_feedback['path_diagnostics'] = trial
        attempt = {'candidates':{a:d.get('candidate_index') for a,d in trial.items()}, 'rejection':None}
        attempts.append(attempt)
        try:
            result = compile_path_combination(e, group, active, goals, constrained, q0, paths, trial, started, calls)
        except Replan as exc:
            attempt['rejection'] = str(exc)
            raise
        e.last_timing['group_path_attempts'] = copy.deepcopy(attempts)
        if hasattr(b, 'path_diagnostics'):
            b.path_diagnostics.update(copy.deepcopy(trial))
        return result

    try:
        return check(raw_paths, diagnostics)
    except Replan as initial_error:
        if not raw_paths or not hasattr(b, 'alternative_paths'):
            raise
        moving = tuple(raw_paths)
        try:
            options = [b.alternative_paths(a, q0[a], *goals[a], e.deadline_check) for a in moving]
        finally:
            # Keep rejected searches visible even if none reach group validation,
            # or cancellation interrupts expansion before a new combination.
            for a in moving:
                search = getattr(b, '_path_searches', {}).get(a)
                if search is not None:
                    records = e.last_feedback['path_diagnostics']
                    records[a] = search.diagnostics(records.get(a, {}))
        combinations = sorted(itertools.product(*options),
                              key=lambda items:sum(d['candidate_cost'] for _,d in items))
        for items in combinations:
            if all(np.array_equal(path, raw_paths[a]) for a,(path,_) in zip(moving,items)):
                continue
            if len(attempts) >= MAX_GROUP_PATH_COMBINATIONS:
                break
            try:
                return check({a:path for a,(path,_) in zip(moving,items)},
                             {a:d for a,(_,d) in zip(moving,items)})
            except Replan:
                continue
        raise initial_error


def compile_path_combination(e, group, active, goals, constrained, q0, raw_paths, diagnostics, started, calls):
    """Apply every existing timing, contact and paired-robot guard before any action."""
    b = e.b
    commands = b.gripper_commands()
    before = e._gripper_targets(group, 'before', commands)
    after = e._gripper_targets(group, 'after', before)
    if hasattr(b, 'gripper_estimates'):
        widths_before = b.gripper_estimates(group, 'before')
        widths_after = b.gripper_estimates(group, 'after', widths_before)
        if before != commands:
            b.set_collision_grippers(commands, before, b.held_width_mm, widths_before)
            b.collision_context = {'phase': 'before_gripper', 'path_fraction': 0.}
            b.validate_pair(q0)
        b.set_collision_grippers(before, before, widths_before, widths_before)
    paths, clocks = {}, {}
    for a, raw in raw_paths.items():
        e.deadline_check()
        if raw.ndim != 2 or not len(raw) or raw.shape[1:] != q0[a].shape or not np.isfinite(raw).all():
            raise Replan('invalid planned path')
        # Applies to every backend, including unscored probes and test backends.
        metrics = path_diagnostics(b, a, q0[a], raw, *goals[a])
        error = path_rejection(metrics)
        diagnostics[a].update(metrics, rejection=error)
        if error:
            raise Replan(error)
        path = np.vstack([q0[a], raw])
        if a in constrained:
            for q in path:
                e._contact(group[a], b.fk(a, q), 'planned path')
        # Arc-length clocks cap every joint even with uneven planner samples.
        times = np.r_[0., np.cumsum(np.max(abs(np.diff(path, axis=0)), axis=1)/JOINT_SPEED_RAD_S)]
        keep = np.r_[True, np.diff(times) > 1e-12]
        paths[a], clocks[a] = path[keep], times[keep]
    base_seconds = max([0.] + [c[-1] for c in clocks.values()])
    ramp_s = START_STOP_RAMP_S if base_seconds > 1e-12 else 0.
    # A ramped profile covers duration-ramp seconds at its peak speed. Reserve
    # that lost area, so adding a ramp never increases the existing speed cap.
    seconds = max(group['duration_s'], base_seconds+ramp_s, 2*ramp_s)
    n = max(0, math.ceil(seconds/b.control_dt-1e-10))
    grip = e._gripper_step_budget(group)
    if n*b.control_dt > MAX_GROUP_SECONDS+1e-8:
        phase = 'motion' if active else 'wait'
        advice = 'use a nearer intermediate pose' if active else 'shorten the wait'
        raise Replan(f'{phase} phase needs {n*b.control_dt:.3f} simulation seconds, '
                     f'exceeding the {MAX_GROUP_SECONDS:g} s limit; {advice}')
    if n+grip > b.remaining_steps():
        raise Replan(f'group including gripper events exceeds remaining environment steps: {n} motion + {grip} gripper; {b.remaining_steps()} remain')
    qs = []
    previous_progress = 0.
    for t in np.linspace(0, 1, n+1)[1:]:
        e.deadline_check()
        progress = ramped_progress(t, n*b.control_dt, ramp_s)
        q = {a: q0[a].copy() for a in ARMS}
        for a, path in paths.items():
            q[a] = np.array([np.interp(progress*clocks[a][-1], clocks[a], path[:,j]) for j in range(path.shape[1])])
        previous = qs[-1] if qs else q0
        for u in (.25, .5, .75, 1.):
            swept = {a: (1-u)*previous[a]+u*q[a] for a in ARMS}
            if hasattr(b, 'collision_context'):
                b.collision_context = {'phase': 'motion', 'path_fraction': previous_progress+u*(progress-previous_progress)}
            b.validate_pair(swept)
            for a in constrained:
                e._contact(group[a], b.fk(a, swept[a]), 'sampled path')
        qs.append(q)
        previous_progress = progress
    if hasattr(b, 'gripper_estimates') and after != before:
        b.set_collision_grippers(before, after, widths_before, widths_after)
        b.collision_context = {'phase': 'after_gripper', 'path_fraction': 1.}
        b.validate_pair(qs[-1] if qs else q0)
    # Playback can skip tiny planner knots. Check the actual command interpolation too.
    for a in paths:
        if qs:
            metrics = path_diagnostics(b, a, q0[a], np.array([q[a] for q in qs]), *goals[a])
            error = path_rejection(metrics)
            diagnostics[a]['playback'] = dict(metrics, rejection=error)
            if error:
                raise Replan(error)
    e.last_timing = {'explicit_duration_s': group['duration_s'], 'nominal_motion_steps': n,
        'actual_planned_motion_steps': len(qs), 'gripper_steps': grip,
        'planner_calls': calls, 'preparation_wall_seconds': time.monotonic()-started,
        'ik_calls': sum(d.get('explicit_ik_calls', 0) for d in diagnostics.values()),
        'control_dt_s': b.control_dt, 'joint_speed_rad_s': JOINT_SPEED_RAD_S,
        'timing_mode': 'ramped', 'endpoint_ramp_s': ramp_s,
        'unramped_minimum_motion_s': base_seconds, 'contact_checked_arms': constrained,
        'path_diagnostics': diagnostics}
    if constrained:
        peak = {a: 0. for a in constrained}
        previous = q0
        for q in qs:
            for a in constrained:
                offset = np.asarray(group[a]['contact_constraint']['working_point_local'])
                p0,r0 = b.fk(a, previous[a]); p1,r1 = b.fk(a, q[a])
                peak[a] = max(peak[a], float(np.linalg.norm(p1+r1@offset-p0-r0@offset))/b.control_dt)
            previous = q
        e.last_timing['sampled_contact_point_speed_m_s'] = peak
    return active, goals, qs
