# Copyright © 2026 Martingale. All Rights Reserved.
import time
import numpy as np

def preview(robo,xyz,*,arm,pitch=None,yaw=None,roll=None,path='auto'):
    """Plan a move_to from the current joints without moving: is the pose reachable, and by which route.

    Same arguments as move_to: xyz in world metres, arm 'left' or 'right', pitch/yaw/roll in degrees (None keeps
    the current angle), path 'auto' or 'line'. No action. Returns reachable (an exact solution and a route
    exist), ik_error (why not), note (set when only a nearby pose is reachable; reachable is then False), path
    ('line' or 'joint'), planned_actions, tcp_waypoints (the grasp-point positions the route passes) and
    limitation. It checks the robot's own kinematics and the table only: not objects, not what is held, not the
    other arm. Look at the current images and the swept route before executing.
    """
    from martingale.hosts import robodojo_tools as tools
    from martingale.hosts import robodojo_kinematics as kin
    if arm not in tools.ARMS or path not in ('auto','line'):raise ValueError('Invalid arm/path')
    if pitch is None or yaw is None or roll is None:
        current=robo.state().arm(arm)
        pitch=current.pitch if pitch is None else pitch
        yaw=current.yaw if yaw is None else yaw
        roll=current.roll if roll is None else roll
    kit=robo._backend._toolkit
    with kit._lock:
        state=tools._obs(kit._primitives)['state'];starts={a:tools._joints(state,a) for a in tools.ARMS}
        target,_=tools._target(arm,starts[arm],xyz=xyz,pitch=pitch,yaw=yaw,roll=roll)
        ways,taken,error,note=tools._plan(arm,starts[arm],target,path,starts)
        positions=[kin.tcp(kin.fk(arm,q)).tolist() for q in ways]
        return dict(reachable=bool(ways) and not error and not note,ik_error=error,note=note,
                    path=taken,planned_actions=len(ways),tcp_waypoints=positions,
                    episode_actions_taken=0,
                    limitation='Own robot kinematics/table check only; no object or full inter-arm collision oracle. Inspect current images and swept volumes before execution.')


def preview_candidates(robo, coarse, *, refine=(), charge=None):
    """Ordered, bounded first-feasible search from one locked joint state.

    Candidates use preview() arguments: xyz in world metres; pitch/yaw/roll
    in the existing tool convention, arm left/right, path auto/line. Refinements
    are explicit alternate poses, tried only if coarse search fails. No action,
    scene/object collision guarantee, waypoint FK or automatic execution.
    """
    from martingale.hosts import robodojo_tools as tools
    candidates = list(coarse) + list(refine)
    if not candidates or len(candidates) > 32:
        raise ValueError('Provide 1..32 coarse plus refinement candidates')
    allowed = {'xyz', 'arm', 'pitch', 'yaw', 'roll', 'path'}
    for candidate in candidates:
        if (not isinstance(candidate, dict) or set(candidate)-allowed or
                not {'xyz', 'arm', 'pitch', 'yaw', 'roll'} <= set(candidate) or
                candidate['arm'] not in tools.ARMS or candidate.get('path', 'auto') not in ('auto', 'line')):
            raise ValueError('Explicit xyz, arm and all angles required for every candidate')
        values = np.asarray([*candidate['xyz'], candidate['pitch'], candidate['yaw'], candidate['roll']], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Candidate pose must contain six finite coordinates')
    kit = robo._backend._toolkit
    failures = []
    with kit._lock:
        state = tools._obs(kit._primitives)['state']
        starts = {a: tools._joints(state, a) for a in tools.ARMS}
        for index, candidate in enumerate(candidates):
            if charge: charge()
            arm = candidate['arm']
            target, _ = tools._target(arm, starts[arm], **{k: candidate[k] for k in ('xyz','pitch','yaw','roll')})
            ways, taken, error, note = tools._plan(arm, starts[arm], target, candidate.get('path','auto'), starts)
            if ways and not error and not note:
                return {'status': 'candidate_found', 'candidate': candidate, 'index': index,
                        'tested': index+1, 'planned_actions': len(ways), 'path': taken,
                        'episode_actions_taken': 0, 'collision_clearance': 'unknown; inspect RGB before execution'}
            failures.append({'index': index, 'ik_error': error, 'note': note})
    return {'status': 'needs_replanning', 'tested': len(candidates), 'failures': failures,
            'episode_actions_taken': 0, 'episode_termination': False}
