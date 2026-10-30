"""ANS7: ANS4 normal routes with steady release, angle repair, checked recovery.

RGB/proprioception only. Neither reached TCP poses nor visual stillness proves
contact or world collision clearance. Unknown loaded states never authorize release.
"""
import re
import numpy as np
from scipy.spatial.transform import Rotation
from .ans6_policy import Policy as ANS6
from .ans4_policy import Policy as ANS4
from .ans7_supervisor import Supervisor
from .contracts import ARMS, STATE_KEYS, validate_observation
from .geometry import bounded_pose, quat_matrix


class Policy(ANS6):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration,supervisor or Supervisor())
        self.release_checks=[];self.reached_routes={};self.recovery_outcomes=[]
        self.steady_events=[];self.prechecked_carries=set()

    def _context(self,waypoint):
        context=super()._context(waypoint)
        context['target_kind']='staging_table' if '_stage' in context['label'] else 'final_pad'
        return context

    def _queue_review(self,stage,context=None,**extra):
        if context is not None:
            context=dict(context,target_kind='staging_table' if '_stage' in context['label'] else 'final_pad')
        super()._queue_review(stage,context,**extra)

    def transfer(self,side,xy,goal,yaw,height,label):
        # Preserve original source, target, arm and rotation; change ONLY the
        # final descent/release timing and short vertical withdrawal.
        ANS4.transfer(self,side,xy,goal,yaw,height,label)
        i=next(i for i in range(len(self.plan)-1,-1,-1) if self.plan[i]['name']=='place_'+label)
        place=self.plan[i];release=self.plan[i+1]
        above=place['pose'].copy();above[2]+=.012
        clear=place['pose'].copy();clear[2]+=.015
        self.plan[i:i+2]=[
            dict(place,pose=above,name='lower_'+label,dwell=2,steady=True,
                 translation_step=self.calibration['precision_translation_step'],rotation_step=self.calibration['max_rotation_step']),
            dict(place,dwell=8,steady=True),
            dict(release,name='softopen_'+label,gripper=.4,dwell=5,steady=True),
            dict(release,dwell=8,steady=True),
            dict(release,pose=clear,name='clearance_'+label,dwell=2,steady=True)]
        # Last 12mm descent uses 2mm increments; no extra lowering of the
        # ANS4 TCP height (24mm) or supposed contact-force measurement.

    def _at_carry_goal(self,obs,context):
        # Observation moves may have displaced the TCP. Do not descend from a
        # displaced observation pose; a checked restoration is handled below.
        side=context['side']
        goal=next((w['pose'] for w in reversed(self.plan[:self.index])
                   if w['name']=='carry_'+context['label']),None)
        if goal is None:return False
        current=np.asarray(obs['state'][side+'_ee_pose'])
        return bool(np.linalg.norm(current[:3]-goal[:3])<.009 and
            (Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(goal[3:])).magnitude()<.12)

    def _observe_move(self,obs,pending):
        # Pre-place target ambiguity gets fresh stationary views. Moving off a
        # reached carry goal must not silently change the descent origin.
        if pending['stage']=='pre_place':return None
        return super()._observe_move(obs,pending)

    @staticmethod
    def _stable_pose(samples):
        if len(samples)<3 or any(p is None for p in samples[-3:]):return None
        last=samples[-3:]
        if any(p.get('confidence',0)<.65 for p in last):return None
        xy=np.asarray([p['xy'] for p in last],float);yaw=np.asarray([p['yaw'] for p in last])
        angle=np.arctan2(np.sin(yaw-yaw[-1]),np.cos(yaw-yaw[-1]))
        if np.max(np.linalg.norm(xy-xy[-1],axis=1))>.003 or np.max(np.abs(angle))>.07:return None
        return dict(last[-1],xy=np.mean(xy,axis=0).tolist(),confidence=min(p['confidence'] for p in last))

    def _release_observation(self,obs,context,pending):
        found=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],
            np.asarray(context['target']),context['family'],height=.014,max_distance=.05)
        pending.setdefault('release_samples',[]).append(found)
        return self._stable_pose(pending['release_samples'])

    def _repair_plan(self,obs,context,found):
        identity=context['digit']
        if identity in self.repaired or '_stage' in context['label'] or '_repair' in context['label'] or found is None:return None
        error=float(np.linalg.norm(np.asarray(found['xy'])-context['target']));angle=abs(float(found['yaw']))
        # A pure-angle error must not require a >=12mm centre error.
        if found['confidence']<.70 or error>.045 or angle>.80 or (error<.012 and angle<.16):return None
        side=context['side'];xy=np.asarray(found['xy']);goal=np.asarray(context['target'])
        if (side=='left' and max(xy[0],goal[0])>.06) or (side=='right' and min(xy[0],goal[0])<-.06):return None
        saved_plan,saved_transfers=self.plan,self.transfers;self.plan=[];self.transfers=[]
        try:
            self.transfer(side,xy,goal,found['yaw'],context['height'],str(identity)+'_repair')
            proposed=self.plan;transfer=self.transfers[0]
        finally:self.plan,self.transfers=saved_plan,saved_transfers
        if self.step+self._remaining_cost(obs,proposed)>=980:return None
        if not self._checked_path(obs,side,[w['pose'] for w in proposed]):return None
        for w in proposed:w['checked_slow']=True
        return dict(plan=proposed,transfer=transfer,kind='angle_only' if error<.012 else 'centre_and_angle')

    def _remaining_cost(self,obs,extra):
        # Estimate the actual segment rates, with explicit review/dwell reserve.
        # This is not a guarantee of convergence; the native 1050 cap remains.
        positions={s:np.asarray(obs['state'][s+'_ee_pose']).copy() for s in ARMS};cost=20
        for w in extra+self.plan[self.index:]:
            start=positions[w['side']];end=w['pose']
            if w.get('steady'):
                translation=w.get('translation_step',.002);rotation=w.get('rotation_step',.04)
            else:
                precision=w.get('checked_slow') or w['name'].startswith(('descend','close','place','release'))
                translation=self.calibration['precision_translation_step'] if precision else self.calibration['max_translation_step']
                rotation=self.calibration['max_rotation_step']
            cost+=int(np.ceil(max(np.linalg.norm(start[:3]-end[:3])/translation,
                (Rotation.from_quat(start[3:]).inv()*Rotation.from_quat(end[3:])).magnitude()/rotation)))+w['dwell']+2
            if w['name'].startswith('approach_'):cost+=12
            positions[w['side']]=end
        return cost

    def _reverse_reached(self,context):
        # Only poses actually reached in THIS leg, never a planned-but-unreached
        # waypoint. Carry/lift landmarks retain load clearance; no blind descent.
        rows=self.reached_routes.get(context['label'],[])
        return [np.asarray(row['pose']).copy() for row in reversed(rows[-4:])
                if row['name'].startswith(('lift_','carry_','lower_'))]

    def _detour(self,obs,context):
        if context['label'] in self.recovered or self.step>850:return None
        w=self.plan[self.index];side=context['side']
        if not w['name'].startswith(('carry_','place_','lift_','lower_')):return None
        current=np.asarray(obs['state'][side+'_ee_pose'])
        old=super()._detour(obs,context)
        if old:
            resume=old.pop('resume_pose',None)
            return dict(points=[old['pose']],resume=resume,kind='checked_local_channel')
        if not w['name'].startswith(('carry_','place_','lower_')):return None
        back=self._reverse_reached(context)
        if not back:return None
        # Return toward achieved lift position, then a low transit channel.
        # All samples use the CURRENT joint branch, not cached feasibility.
        for height in (.08,.065):
            source=np.asarray(back[-1]).copy()
            tcp=source[:3]+quat_matrix(source[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
            source[2]+=self.calibration['table_z']+height-tcp[2]
            goal=self.pose(context['target'],height,np.pi/2)
            final=self.pose(context['target'],.024,np.pi/2)
            poses=back+[source,goal,final]
            if self._within_side(side,poses) and self._checked_path(obs,side,poses):
                return dict(points=back+[source],resume=goal if w['name'].startswith('carry_') else None,
                    extra_goal=None if w['name'].startswith('carry_') else goal,kind='reverse_reached_low_channel')
        self.path_rejections.append(dict(step=self.step,label=context['label'],phase=w['name'],reason='reverse_reached_path_rejected'))
        return None

    def _install_recovery(self,recovery,context):
        side=context['side'];label=context['label'];original=dict(self.plan[self.index])
        resume=recovery.get('resume')
        if resume is not None:
            original.update(pose=resume,checked_slow=True)
            for w in self.plan[self.index+1:]:
                if w['name']=='retreat_'+label:w.update(pose=resume.copy(),checked_slow=True);break
        original['recovery_goal']=label
        points=list(recovery['points'])
        if recovery.get('extra_goal') is not None:points.append(recovery['extra_goal'])
        extra=[dict(side=side,pose=p.copy(),gripper=0.,name='recovery_'+label,dwell=2,
                    checked_slow=True,recovery_point=label) for p in points]
        self.plan[self.index:self.index+1]=extra+[original]
        self.recovered.add(label);self.age=self.dwell=0;self.progress=[]
        self.recovery_outcomes.append(dict(label=label,kind=recovery['kind'],requested_step=self.step,
            points_reached=0,original_goal_reached=False))
        self.correction_paths.append(dict(kind=recovery['kind'],step=self.step,label=label,poses=[p.tolist() for p in points]))
        self._event('checked_recovery_started_not_yet_succeeded',context)

    def _skip_safely(self,obs,context):
        side=context['side'];lift=np.asarray(obs['state'][side+'_ee_pose']).copy()
        lift[2]+=max(0.,.10-(self._tcp_z(obs['state'],side)-self.calibration['table_z']))
        candidates=[[lift,self.home[side]]]
        reached=self._reverse_reached(context)
        if reached:candidates.append(reached+[self.home[side]])
        poses=next((p for p in candidates if self._within_side(side,p) and self._checked_path(obs,side,p)),None)
        if poses is None:return self._freeze(obs,'confirmed_empty_no_checked_exit')
        kept=[]
        for w in self.plan[self.index:]:
            match=re.search(r'_(\d+)(?:_|$)',w['name'])
            if not match or int(match[1])!=context['digit']:kept.append(w)
        park=[dict(side=side,pose=p.copy(),gripper=1.,name='empty_park',dwell=3,checked_slow=True) for p in poses]
        self.plan[self.index:]=park+kept;self.age=self.dwell=0;self.pending=None;self.gate=None
        self.route_states[str(context['digit'])]='empty_skipped'
        self.skipped.append(dict(digit=context['digit'],reason='confirmed_empty_checked_exit'))
        self._event('confirmed_empty_skip_independent',context);return self._stationary(obs['state'])

    def _review(self, obs):
        self.step+=1;pending=self.pending;stage=pending['stage'];context=pending['context']
        action=self._stationary(obs['state']);allowed=['continue','observe','hold']
        payload=dict(robot_state={k:np.asarray(obs['state'][k]).tolist() for k in STATE_KEYS if k in obs['state']},
            task_instruction=obs['instruction'][:2000],target=context,local_evidence={},route_states=self.route_states,
            action_scope='Only listed checked actions. No model coordinates. Unknown is not empty.',observation_attempt=pending['attempt']+1)
        evidence=None;retry=None;detour=None;repair=None
        if stage=='initial':
            if pending['attempt']:self.plan=[];self.transfers=[];self.build(obs['vision']['cam_head']['color'])
            payload.update(detected_digits=self.digits,targets_left_to_right=self.targets,intended_order=sorted([d['digit'] for d in self.digits],reverse=True))
        elif stage in ('post_lift','pre_place','transport_stall'):
            evidence=self._evidence(obs,context,pending['observations']);payload['local_evidence']=evidence
            if len(pending['observations'])<2:return action
            if evidence['local_status']=='empty':
                allowed.append('skip_empty');retry=self._retry_plan(obs,context,evidence['source_pose'])
                if retry:allowed.append('retry')
            if stage=='transport_stall':
                detour=self._detour(obs,context);payload['local_evidence']=dict(evidence,motion=self.motion,detour_available=detour is not None)
                if detour:allowed.append('recover')
        elif stage=='post_release':
            found=self._release_observation(obs,context,pending)
            if len(pending['release_samples'])<3 or (found is None and len(pending['release_samples'])<6):return action
            repair=self._repair_plan(obs,context,found);payload['local_evidence']=dict(local_digit_pose=found,repair_available=repair is not None,
                release_samples=pending['release_samples'],stable_pose=found is not None,
                correction_kind=repair and repair['kind'],target_kind=context['target_kind'])
            if repair:allowed.append('repair')
        elif stage=='final':payload.update(targets_left_to_right=self.targets,detected_initial_digits=self.digits)
        frames=[('before_close_'+n,im) for n,im in self.reference_frames.get(context['label'],[])] if context else []
        frames += [('current_'+n,im) for n,im in self._images(obs,context)]
        answer=self.supervisor.review(stage,self.step,payload,frames,allowed)
        self.visual_checks.append(dict(step=self.step,stage=stage,label=context and context['label'],answer=answer));pending['attempt']+=1
        if answer.get('error'):self.supervisor_failure=answer['error'];return self._freeze(obs,'astra_unavailable_hold')
        if stage=='initial':
            if answer['action']!='continue' or answer['placement']=='wrong_digit' or not self.plan:return self._defer_or_stop(obs,pending,'initial_identity_unresolved')
        elif stage in ('post_lift','pre_place','transport_stall'):
            model=answer['possession'];local=evidence['local_status']
            if local=='empty' and model in ('empty','dropped') and answer['action'] in ('retry','skip_empty'):
                if answer['action']=='retry' and retry:
                    label=context['label'];end=self.index
                    while end<len(self.plan) and self.plan[end]['name'].endswith('_'+label):end+=1
                    self.plan[self.index:end]=retry['plan'];self.transfers.append(retry['transfer']);self.retry_digits.add(context['digit']);self.age=self.dwell=0
                    self._event('confirmed_empty_retry_once',context);self.pending=None;return action
                return self._skip_safely(obs,context)
            held=(local=='held' and model in ('held_clear','held_occluded','unknown')) or (evidence['coarse_stable_contact'] and model in ('held_clear','held_occluded') and answer['confidence']>=.75)
            if not held:return self._defer_or_stop(obs,pending,'possession_unresolved')
            if answer['placement']=='wrong_digit':return self._defer_or_stop(obs,pending,'identity_conflict')
            if stage=='pre_place':
                # Target visibility is not possession. One fresh check for a coarse
                # mismatch; staging is a tabletop handoff, never a final pad.
                conflict=answer['placement']=='misaligned' or answer['action']=='hold'
                if conflict:
                    return self._defer_or_stop(obs,pending,'target_conflict')
                if answer['action']=='observe' and pending['attempt']<2:
                    self._event('target_visibility_reobserve',context);return action
                if not self._at_carry_goal(obs,context):
                    return self._defer_or_stop(obs,pending,'carry_goal_not_reached')
                self._event('held_target_visibility_separated',context,placement=answer['placement'])
            elif stage=='post_lift' and answer['action'] in ('hold','observe'):
                return self._defer_or_stop(obs,pending,'held_visual_request')

            if stage=='transport_stall':
                if answer['action']=='recover' and detour:
                    self._install_recovery(detour,context)
                else:return self._defer_or_stop(obs,pending,'no_selected_feasible_recovery')
            else:
                self.held_status[context['label']]=model;self.route_states[str(context['digit'])]='held_confirmed';self._event('held_resume_ANS4',context)
        elif stage=='post_release':
            if answer['action']=='observe' and pending['attempt']<2:self._event('release_reobserve',context);return action
            if answer['action']=='repair' and answer['placement']!='wrong_digit' and repair:
                self.plan[self.index:self.index]=repair['plan'];self.transfers.append(repair['transfer']);self.repaired.add(context['digit']);self.age=self.dwell=0
                self._event('checked_placement_repair_once',context,kind=repair['kind'])
            verified=found is not None and np.linalg.norm(np.asarray(found['xy'])-context['target'])<.012 and abs(found['yaw'])<.16 and found['confidence']>=.65 and answer['placement']=='aligned'
            self.route_states[str(context['digit'])]='placed_verified' if verified and context['target_kind']=='final_pad' else 'released_unverified'
            self.release_checks.append(dict(step=self.step,label=context['label'],stable_pose=found,verified=bool(verified),target_kind=context['target_kind']))
        elif stage=='final':self.final_checked=True;self._event('program_complete_visual_review_not_success_claim')
        self.pending=None;return action


    def _steady_act(self,obs,w):
        self.step+=1;side=w['side'];current=np.asarray(obs['state'][side+'_ee_pose'])
        action=self._stationary(obs['state'])
        action[side+'_ee_pose']=bounded_pose(current,w['pose'],w.get('translation_step',.002),w.get('rotation_step',.04))
        action[side+'_ee_joint_state']=np.array([w['gripper']])
        pos=float(np.linalg.norm(current[:3]-w['pose'][:3]))
        angle=float((Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(w['pose'][3:])).magnitude())
        self.motion=dict(position_error=pos,orientation_error=angle,side=side,current_ee=current.tolist(),goal_ee=w['pose'].tolist())
        self.age+=1;self.dwell=self.dwell+1 if pos<.003 and angle<.08 else 0
        if self.dwell>=w['dwell']:
            self.steady_events.append(dict(step=self.step,phase=w['name'],arrived=True,dwell=self.dwell,gripper=w['gripper']))
            self.index+=1;self.age=self.dwell=0
        elif self.age>=90:
            if w['gripper']==0.:
                self._queue_review('transport_stall',self._context(w));self.age=0
            else:return self._freeze(obs,'release_motion_unresolved_no_transport')
        return action

    def _record_arrival(self,w,obs):
        match=re.match(r'(?:lift|carry|lower|place)_(.+)$',w['name'])
        if match:
            label=match[1]
            self.reached_routes.setdefault(label,[]).append(dict(step=self.step,name=w['name'],
                pose=np.asarray(obs['state'][w['side']+'_ee_pose']).tolist()))
        if w.get('recovery_point') or w.get('recovery_goal'):
            label=w.get('recovery_point') or w['recovery_goal']
            record=next(r for r in reversed(self.recovery_outcomes) if r['label']==label)
            if w.get('recovery_point'):record['points_reached']+=1
            if w.get('recovery_goal'):
                record.update(original_goal_reached=True,reached_step=self.step)
                self._event('recovery_original_goal_actually_reached',dict(label=label))

    def _act(self,obs):
        validate_observation(obs)
        active=self.hold is None and self.pending is None and self.gate is None and self.auxiliary is None and self.planned and self.index<len(self.plan)
        w=self.plan[self.index] if active else None;old_index=self.index
        if w is not None and w.get('steady'):
            result=self._steady_act(obs,w)
        else:
            result=super()._act(obs)
        if w is not None and self.index==old_index+1 and self.hold is None:
            self._record_arrival(w,obs)
        return result

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS7',steady_events=self.steady_events,release_checks=self.release_checks,
            reached_routes=self.reached_routes,recovery_outcomes=self.recovery_outcomes,
            angle_only_repair_enabled=True,steady_release_enabled=True,
            contact_force_known=False,target_visibility_separate_from_possession=True)
        return result
