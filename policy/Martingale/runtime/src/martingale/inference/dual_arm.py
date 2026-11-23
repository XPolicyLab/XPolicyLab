# Copyright © 2026 Martingale. All Rights Reserved.
import time
import copy
import hashlib
import json
import secrets
import numpy as np

def _fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,
        default=lambda x:np.asarray(x).tolist(),allow_nan=False).encode()).hexdigest()

def synchronize(starts,paths):
    count=max(len(paths[a]) for a in ('left','right'))
    if count<1:raise ValueError('Empty dual trajectory')
    result=[]
    for step in range(1,count+1):
        row={}
        for arm in ('left','right'):
            points=np.asarray([starts[arm],*paths[arm]],float)
            if points.ndim!=2 or points.shape[1]!=6 or not np.isfinite(points).all():raise ValueError('Invalid joint trajectory')
            t=step/count*(len(points)-1);lo=min(int(t),len(points)-2);fraction=t-lo
            row[arm]=(points[lo]*(1-fraction)+points[lo+1]*fraction).tolist()
        result.append(row)
    return result

def move_both(robo,left,right,*,clearance_evidence,max_actions=40,own_robot_margin_m=None,
              preview_only=False,approved_plan=None):
    """Plan or execute bounded synchronized motion, preserving both grippers.

    preview_only=True returns zero-action planned joint/TCP samples and a
    single-use plan_id. Inspect that actual route against current RGB and held
    material, then call again with approved_plan=plan_id and identical poses,
    max_actions and margin. Dispatch consumes the stored route without replanning.
    Changed robot state/status or another preview invalidates the ticket.
    Errors return to the caller before motion for stale/mismatched plans; partial
    execution errors require observation/reconciliation, never blind retry.
    TCP/optional sampled robot separation is not scene/cloth collision checking.
    Calls without a ticket plan automatically.
    """
    from martingale.hosts import robodojo_tools as tools
    from martingale.hosts import robodojo_kinematics as kin
    if not isinstance(clearance_evidence,str) or not clearance_evidence.strip():raise ValueError('Describe current visual swept-path clearance')
    if not isinstance(max_actions,int) or not 1<=max_actions<=100:raise ValueError('Invalid motion length bound')
    if type(preview_only)is not bool or (preview_only and approved_plan is not None):raise ValueError('Preview and approved dispatch are exclusive')
    if own_robot_margin_m is not None and (isinstance(own_robot_margin_m,bool) or not np.isfinite(own_robot_margin_m) or own_robot_margin_m<.005):
        raise ValueError('Opt-in sampled own-robot margin must be finite and at least 5mm')
    kit=robo._backend._toolkit;primitives=kit._primitives
    arguments={'left':left,'right':right,'max_actions':max_actions,'clearance_evidence':clearance_evidence}
    if own_robot_margin_m is not None:arguments['own_robot_margin_m']=own_robot_margin_m
    request_fingerprint=_fingerprint(arguments | {'clearance_evidence':None})
    if preview_only:arguments['preview_only']=True
    if approved_plan is not None:arguments['approved_plan']=approved_plan
    record=robo._ledger.open('move_both',arguments,stateful=True)
    started=time.perf_counter();used=0;failed=False
    try:
      with kit._lock:
        robo._guard();state=tools._obs(primitives)['state'];starts={a:tools._joints(state,a) for a in tools.ARMS};paths={};goals={}
        state_fingerprint=_fingerprint({'state':state,'status':kit._status()})
        if approved_plan is not None:
            cached=getattr(kit,'_dual_plan',None)
            kit._dual_plan=None
            if (not cached or cached['plan_id']!=approved_plan
                    or cached['request']!=request_fingerprint
                    or cached['state']!=state_fingerprint):
                raise ValueError('Paired preview stale or mismatched; preview and inspect again')
            paths,goals,trajectory=copy.deepcopy(cached['route'])
        else:
            if preview_only:kit._dual_plan=None
            for arm,pose in [('left',left),('right',right)]:
                if set(pose)!={'xyz','pitch','yaw','roll'}:raise ValueError('Both poses require xyz/pitch/yaw/roll')
                target,_=tools._target(arm,starts[arm],**pose);goals[arm]=target
                ways,taken,why,note=tools._plan(arm,starts[arm],target,'auto',starts)
                if why or note:raise ValueError('Exact dual pose unavailable: '+str(why or note))
                paths[arm]=ways
            trajectory=synchronize(starts,paths)
        if len(trajectory)>max_actions:raise ValueError('Plan exceeds requested motion length bound')
        for waypoint in trajectory:
            separation=np.linalg.norm(kin.tcp(kin.fk('left',waypoint['left']))-kin.tcp(kin.fk('right',waypoint['right'])))
            if separation<.10:raise ValueError('TCP separation below 10cm; full collision checking still required')
        own_robot_screen=None
        if own_robot_margin_m is not None:
            from .robot_envelope import screen
            own_robot_screen=screen(starts,trajectory,margin=float(own_robot_margin_m))
            record.summary={'own_robot_screen':own_robot_screen,'steps_used':0}
            if not own_robot_screen['passed']:
                raise RuntimeError('Sampled own-robot paired trajectory rejected before dispatch: '+repr(own_robot_screen))
        if preview_only:
            ticket=secrets.token_hex(16)
            kit._dual_plan=dict(plan_id=ticket,request=request_fingerprint,
                state=state_fingerprint,route=copy.deepcopy((paths,goals,trajectory)))
            result=dict(plan_id=ticket,episode_actions_taken=0,planned_actions=len(trajectory),
                joint_waypoints=copy.deepcopy(trajectory),
                tcp_waypoints=[{a:kin.tcp(kin.fk(a,q)).tolist() for a,q in row.items()} for row in trajectory],
                own_robot_screen=own_robot_screen,
                limitation='Exact stored route, not scene/held-material collision clearance; inspect before approved dispatch')
            record.ok=True;record.summary=result;return result
        status={};stopped=None;lag_count=0
        for waypoint in trajectory:
            if tools._spent(kit._status()):stopped='episode ended';break
            primitives._check_cancelled()
            state,status=tools._act(primitives,state,waypoint,{a:None for a in tools.ARMS});used+=1
            lag=max(np.max(np.abs(np.asarray(tools._joints(state,a))-waypoint[a])) for a in tools.ARMS)
            lag_count=lag_count+1 if lag>.2 else 0
            if lag_count>=3:stopped='persistent joint tracking lag';break
        for _ in range(min(tools.SETTLE_ACTIONS,max_actions-used)):
            if stopped or tools._spent(status):break
            if all(max(abs(np.asarray(tools._joints(state,a))-paths[a][-1]))<.01 for a in tools.ARMS):break
            state,status=tools._act(primitives,state,{a:paths[a][-1] for a in tools.ARMS},{a:None for a in tools.ARMS});used+=1
        arms={}
        for arm in tools.ARMS:
            fk=kin.fk(arm,tools._joints(state,arm));distance=float(np.linalg.norm(kin.tcp(fk)-goals[arm][:3,3]));angle=float(kin.rotation_error(fk[:3,:3],goals[arm][:3,:3]))
            arms[arm]={'distance_m':distance,'orientation_error_degrees':angle,'reached':distance<=tools.ARRIVE_M and angle<=tools.ARRIVE_DEG}
        result={'steps_used':used,'arms':arms,'reached':all(x['reached'] for x in arms.values()),'stopped':stopped,'gripper_commands':'preserved'}
        if own_robot_screen is not None:result['own_robot_screen']=own_robot_screen
        record.ok=True;record.summary=result;return result
    except BaseException as error:
        failed=True
        record.error=f'{type(error).__name__}: {error}';raise
    finally:
        record.elapsed_s=time.perf_counter()-started;record.env_steps=used
        robo._last_state=None
        if not failed:
            obs=primitives.env.get_obs();primitives._last_obs=obs
            kit.state.record(obs,kit._status())
