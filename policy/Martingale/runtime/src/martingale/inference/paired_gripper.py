# Copyright © 2026 Martingale. All Rights Reserved.
import time
import numpy as np

def set_grippers(robo,*,left,right,min_actions=2,max_actions=10):
    """Open or close BOTH grippers in the same actions, holding both arms' joints.

    left / right: robo.OPEN (-1) or robo.CLOSE (1). The commands repeat until both measured openings settle, at
    least min_actions and at most max_actions (2 to 10); every action counts. Returns steps_used, the commands,
    the held joint targets, measured_openings and the joint drift. A reading is not proof of a grasp or a
    release: check the images.
    """
    from martingale.hosts import robodojo_tools as tools
    if any(isinstance(v,bool)or v not in(-1,1)for v in(left,right)):
        raise ValueError('Both grippers require OPEN (-1) or CLOSE (1)')
    if type(min_actions)is not int or type(max_actions)is not int or not 2<=min_actions<=max_actions<=10:
        raise ValueError('Require 2 <= min_actions <= max_actions <= 10')
    kit=robo._backend._toolkit;primitives=kit._primitives
    record=robo._ledger.open('set_grippers',dict(left=left,right=right,min_actions=min_actions,max_actions=max_actions),stateful=True)
    started=time.perf_counter();used=0
    try:
        with kit._lock:
            robo._guard();state=tools._obs(primitives)['state']
            targets={a:list(tools._joints(state,a))for a in tools.ARMS}
            commands={'left':tools._command(left),'right':tools._command(right)}
            last={a:tools._opening(state,a)for a in tools.ARMS};status={};stopped=None
            for _ in range(max_actions):
                primitives._check_cancelled()
                if tools._spent(kit._status()):stopped='episode ended';break
                state,status=tools._act(primitives,state,targets,commands);used+=1
                now={a:tools._opening(state,a)for a in tools.ARMS}
                if tools._spent(status):stopped='episode ended';break
                if used>=min_actions and all(abs(now[a]-last[a])<tools.GRIP_STILL for a in tools.ARMS):break
                last=now
            result={'steps_used':used,'requested_gripper_commands':commands,'fixed_joint_targets':targets,'measured_openings':{a:tools._opening(state,a)for a in tools.ARMS},'max_joint_drift_rad':{a:float(np.max(np.abs(np.asarray(tools._joints(state,a))-targets[a])))for a in tools.ARMS},'stopped':stopped,'scope':'Both grippers commanded simultaneously; fixed achieved joint posture, not scene collision or object-retention proof.'}
            record.ok=True;record.summary=result;return result
    except BaseException as error:
        record.error=f'{type(error).__name__}: {error}';raise
    finally:
        record.elapsed_s=time.perf_counter()-started;record.env_steps=used
        obs=primitives.env.get_obs();primitives._last_obs=obs
        kit.state.record(obs,kit._status());robo._last_state=None
