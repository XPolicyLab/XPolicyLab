"""ANS8: ANS7 routes with empirical reach routing, stall recovery and bounded supervisor degradation.

Fixes the ANS7 observation-move contract. The limit-free public IK is not a
reach certificate: routing uses the horizontal reach observed in real runs.
Every terminal path opens the gripper; a stalled loaded carry puts the digit
back where it was lifted instead of holding until the step cap.
"""
import re
import numpy as np
from .ans7_policy import Policy as ANS7
from .ans6_policy import Policy as ANS6
from .ans1_policy import Policy as ANS1
from .contracts import ARMS

REACH_BASE={'left':np.array([-.30,-.45]),'right':np.array([.30,-.45])}
# 19/19 historical carry stalls occurred at >=0.466 m horizontal TCP distance
# from the arm base; same-side inner pads (<=0.471 m) were always reached.
REACH_SAFE=.460
REACH_LIMIT=.472
STALL_STEPS=25
STALL_IMPROVEMENT=.001
ROUTE_PATTERN=re.compile(r'_(\d+)(?:_stage|_retry|_repair)?$')


def route_digit(name):
    match=ROUTE_PATTERN.search(name)
    return None if match is None else int(match[1])


class DegradingSupervisor:
    """A failed or exhausted review degrades to the ANS4 local decision, never to an episode-long hold."""
    def __init__(self,inner):
        self.inner=inner;self.degraded=[]

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def review(self,stage,step,payload,frames,allowed):
        answer=self.inner.review(stage,step,payload,frames,allowed)
        if not answer.get('error'):return answer
        local=(payload.get('local_evidence') or {}).get('local_status')
        if stage in ('post_lift','pre_place','transport_stall'):
            if local=='held':action,possession,placement='continue','held_clear','aligned'
            elif local=='empty' and 'skip_empty' in allowed:action,possession,placement='skip_empty','empty','not_applicable'
            else:action,possession,placement='observe','unknown','unknown'
        elif stage=='post_release':action,possession,placement='continue','empty','unknown'
        else:action,possession,placement='continue','not_applicable','not_applicable'
        if action not in allowed:action='observe' if 'observe' in allowed else 'hold'
        self.degraded.append(dict(step=int(step),stage=stage,error=answer['error'],local_status=local,action=action))
        return dict(answer,action=action,possession=possession,placement=placement,confidence=0.,
            evidence='local_fallback_after_'+str(answer['error']),degraded_from=answer['error'],error=None)


class Policy(ANS7):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor)
        self.supervisor=DegradingSupervisor(self.supervisor)
        self.routing=[];self.stall_events=[];self.put_backs=[]
        self.tracker=dict(index=None,name=None,best=None,count=0)

    # ---- (a) observation contract: ANS6 passes the review context, not the pending record ----
    def _observe_move(self,obs,context):
        if self.pending is not None and self.pending.get('stage')=='pre_place':return None
        return ANS6._observe_move(self,obs,context)

    # ---- (c) reach-aware routing ----
    def reach(self,side,xy):
        return float(np.linalg.norm(np.asarray(xy,float)[:2]-REACH_BASE[side]))

    def options(self,xy,goal,yaw):
        choices=[]
        for cost,side,height in ANS1.options(self,xy,goal,yaw):
            worst=max(self.reach(side,xy),self.reach(side,goal))
            cross=(side=='left' and goal[0]>.03) or (side=='right' and goal[0]<-.03)
            if worst>REACH_LIMIT or cross:
                self.routing.append(dict(side=side,source=np.round(xy,3).tolist(),goal=np.round(goal,3).tolist(),
                    reach=round(worst,3),rejected='cross_centre' if cross else 'empirical_reach_limit'))
                continue
            choices.append((cost+(.05 if worst>REACH_SAFE else 0.),side,height))
        return sorted(choices)

    # ---- (d) progress-based stall handling ----
    def _stall(self,w,obs):
        current=np.asarray(obs['state'][w['side']+'_ee_pose']);err=float(np.linalg.norm(current[:3]-w['pose'][:3]))
        t=self.tracker
        if t['index']!=self.index or t['name']!=w['name']:
            t.update(index=self.index,name=w['name'],best=err,count=0);return None
        if err<t['best']-STALL_IMPROVEMENT:t['best']=err;t['count']=0
        else:t['count']+=1
        return err if t['count']>=STALL_STEPS else None

    def _drop_route(self,digit):
        self.plan[self.index:]=[p for p in self.plan[self.index:] if route_digit(p['name'])!=digit]

    def _put_back(self,w,cause):
        side=w['side'];label=w['name'].split('_',1)[1];digit=route_digit(w['name'])
        lift=next((p for p in reversed(self.plan[:self.index]) if p['name']=='lift_'+label),None)
        descend=next((p for p in reversed(self.plan[:self.index]) if p['name']=='descend_'+label),None)
        if lift is None or descend is None or digit is None or self.home is None:return False
        park=[dict(side=side,pose=lift['pose'].copy(),gripper=0.,name='putback_lift_'+label,dwell=2),
              dict(side=side,pose=descend['pose'].copy(),gripper=0.,name='descend_putback_'+label,dwell=2),
              dict(side=side,pose=descend['pose'].copy(),gripper=1.,name='release_putback_'+label,dwell=6),
              dict(side=side,pose=lift['pose'].copy(),gripper=1.,name='putback_retreat_'+label,dwell=2),
              dict(side=side,pose=self.home[side].copy(),gripper=1.,name='putback_home_'+label,dwell=3)]
        kept=[p for p in self.plan[self.index:] if route_digit(p['name'])!=digit]
        self.plan=self.plan[:self.index]+park+kept;self.age=self.dwell=0
        self.route_states[str(digit)]='put_back_'+cause
        self.put_backs.append(dict(step=self.step,label=label,cause=cause,landing=np.round(descend['pose'][:3],4).tolist()))
        self._event('put_back_'+cause,dict(label=label))
        return True

    def _accept_contact(self,w,obs):
        label=w['name'].split('_',1)[1];current=np.asarray(obs['state'][w['side']+'_ee_pose']).copy()
        for j in range(self.index+1,len(self.plan)):
            p=self.plan[j]
            if p['name'] in ('close_'+label,'place_'+label,'softopen_'+label,'release_'+label):p['pose']=current.copy()
            elif p['name']=='clearance_'+label:
                clear=current.copy();clear[2]+=.015;p['pose']=clear
        self.index+=1;self.age=self.dwell=0

    def _handle_stall(self,obs,w,err):
        side=w['side'];name=w['name'];kind=name.split('_')[0];digit=route_digit(name)
        current=np.asarray(obs['state'][side+'_ee_pose'])
        xy_err=float(np.linalg.norm(current[:2]-w['pose'][:2]));z_err=float(abs(current[2]-w['pose'][2]))
        event=dict(step=self.step,phase=name,position_error=round(err,4),xy_error=round(xy_err,4),z_error=round(z_err,4))
        if kind in ('place','lower','descend') and xy_err<.006 and z_err<.016:
            # Resting contact above the nominal height (pad thickness, held depth): do not press further.
            self._accept_contact(w,obs);event['outcome']='accepted_resting_contact'
        elif w['gripper']==0. and digit is not None and self._put_back(w,'unreachable_'+kind):
            event['outcome']='put_back_at_lift_point'
        elif w['gripper']==0.:
            event['outcome']='loaded_stall_no_landmarks_hold';self.stall_events.append(event)
            return self._freeze(obs,'stall_'+event['outcome'])
        elif kind in ('approach','descend') and digit is not None:
            self._drop_route(digit);self.route_states[str(digit)]='unreachable_source_skipped'
            self.plan.insert(self.index,dict(side=side,pose=self.home[side].copy(),gripper=1.,name='unreachable_home_'+str(digit),dwell=3))
            self.age=self.dwell=0;event['outcome']='route_skipped_unreachable_source'
        else:
            self.index+=1;self.age=self.dwell=0;event['outcome']='advanced_past_stalled_free_motion'
        self.stall_events.append(event);self.tracker.update(index=None)
        self._event('stall_'+event['outcome']);return self._stationary(obs['state'])

    # ---- (3) terminal paths: never end clamping a digit ----
    def _freeze(self,obs,reason):
        w=self.plan[self.index] if (self.planned and self.index<len(self.plan)) else None
        closed=[s for s in ARMS if float(np.asarray(self.last_gripper_commands.get(s,[1.]))[0])<.5]
        if w is not None and w['side'] in closed and w['gripper']==0. and self._put_back(w,'terminal_'+reason):
            self._event(reason);self.pending=None;self.gate=None;self.auxiliary=None;self.tracker.update(index=None)
            return self._stationary(obs['state'])
        result=super()._freeze(obs,reason)
        for side in closed:
            self.hold[side+'_ee_joint_state']=np.array([1.])
            lifted=self.hold[side+'_ee_pose'].copy();lifted[2]+=.03;self.hold[side+'_ee_pose']=lifted
        return {k:v.copy() for k,v in self.hold.items()}

    def _act(self,obs):
        active=self.hold is None and self.pending is None and self.gate is None and self.auxiliary is None and self.planned and self.index<len(self.plan)
        if active:
            w=self.plan[self.index]
            if w['name'].startswith(('carry_','place_','lift_')):self.stall_reviewed.add((w['name'],self.index))
            err=self._stall(w,obs)
            if err is not None:
                self.step+=1;return self._handle_stall(obs,w,err)
        return super()._act(obs)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS8',routing_rejections=self.routing[-16:],stall_events=self.stall_events,put_backs=self.put_backs,
            degraded_reviews=list(self.supervisor.degraded),reach_model=dict(safe=REACH_SAFE,limit=REACH_LIMIT,bases={k:v.tolist() for k,v in REACH_BASE.items()}),
            progress_stall_steps=STALL_STEPS,terminal_gripper_open=True)
        return result
