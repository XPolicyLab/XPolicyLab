"""ANS6: ANS4 motion, refreshed visual evidence and bounded recovery.

No model coordinates. Continuous IK is not collision or contact certification.
An unresolved possibly-loaded state still holds after bounded observations.
"""
import re
import numpy as np
from scipy.spatial.transform import Rotation
from .ans5_policy import Policy as ANS5
from .ans4_policy import Policy as ANS4
from .ans6_supervisor import Supervisor
from .contracts import ARMS, STATE_KEYS, validate_observation
from .geometry import bounded_pose, quat_matrix
from .kinematics import pose_matrix
from .wrist_grasp import decide


class Policy(ANS5):
    def __init__(self, task, calibration, supervisor=None):
        super().__init__(task, calibration, supervisor or Supervisor())
        self.auxiliary=None; self.retry_digits=set(); self.route_states={}
        self.transitions=[]; self.path_rejections=[]; self.path_checks=[]; self.stall_reviewed=set()

    def _event(self, state, context=None, **extra):
        self.reason=state
        self.transitions.append(dict(step=self.step,state=state,label=context and context['label'],**extra))

    def _freeze(self, obs, reason):
        self._event(reason); self.pending=None; self.gate=None; self.auxiliary=None
        return super()._freeze(obs,reason)

    def _queue_review(self, stage, context=None, **extra):
        super()._queue_review(stage,context,**extra)
        self.pending.update(observations=[],observation_move_used=False)

    def _evidence(self, obs, context, observations):
        """Every new image refreshes wrist AND source; old local status is never reused."""
        features=self._wrist(obs,context); observations.append(features); del observations[:-2]
        reference=self.references.get(context['label'],{})
        lift=self._tcp_z(obs['state'],context['side'])-reference.get('tcp_z',self._tcp_z(obs['state'],context['side']))
        source=self._table_digit(obs,context,features)
        local=decide(reference.get('features'),observations,lift,source is not None)
        coarse=len(observations)==2 and lift>=.035 and all(f.get('available') and f.get('contact') and
            (f.get('object') or {}).get('area',0)>.01 for f in observations)
        if coarse:
            a,b=[f['object'] for f in observations]
            coarse=np.linalg.norm(np.asarray(a['centre'])-b['centre'])<.035 and .65<a['area']/b['area']<1.55
        result=dict(local_status=local,wrist_observations=list(observations),lift_metres=float(lift),
            source_pose=source,coarse_stable_contact=bool(coarse),evidence_step=self.step)
        self.grasp_checks.append(dict(step=self.step,digit=context['digit'],side=context['side'],status=local,**result))
        return result

    def _table_digit(self, obs, context, features):
        # Source first, then the current TCP's tabletop projection and target:
        # a dropped digit need not remain at its original source. Conflicting
        # template locations are rejected rather than selecting the best score.
        side=context['side'];ee=np.asarray(obs['state'][side+'_ee_pose'])
        tcp=ee[:3]+quat_matrix(ee[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
        centres=[context['source']]
        if features.get('available') and not features.get('contact'):
            centres += [tcp[:2],context['target']]
        found=[]
        for xy in centres:
            candidate=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],
                np.asarray(xy),context['family'],height=.014,max_distance=.05)
            if candidate is not None and candidate['confidence']>=.65:found.append(candidate)
        if not found:return None
        if any(np.linalg.norm(np.asarray(a['xy'])-b['xy'])>.015 for a in found for b in found):return None
        return found[0]

    def _checked_path(self, obs, side, poses):
        q=np.asarray(obs['state'].get(side+'_arm_joint_state',[]),float)
        current=np.asarray(obs['state'][side+'_ee_pose'],float).copy();samples=0
        def finish(ok,reason,**extra):
            self.path_checks.append(dict(step=self.step,side=side,passed=ok,reason=reason,samples=samples,**extra))
            del self.path_checks[:-32]
            return ok
        if q.shape!=(6,) or not np.isfinite(q).all():return finish(False,'missing_joints')
        # Same residual and branch-jump thresholds as ANS5. The bounded sample
        # budget covers a full retry transfer, not just a short transit segment.
        for goal in poses:
            for _ in range(150):
                nxt=bounded_pose(current,goal,self.calibration['precision_translation_step'],self.calibration['max_rotation_step'])
                fit=self.arms[side].inverse(pose_matrix(nxt),initial=q,restarts=0);new=np.asarray(fit['joints']);samples+=1
                jump=float(np.max(np.abs(new-q)))
                if fit['position_error_m']>=.004 or fit['orientation_error_rad']>=.08 or jump>.35:
                    return finish(False,'IK_residual_or_branch_jump',position_error=fit['position_error_m'],rotation_error=fit['orientation_error_rad'],joint_jump=jump)
                q=new;current=nxt
                if samples>240:return finish(False,'sample_budget')
                if np.linalg.norm(current[:3]-goal[:3])<1e-7 and (Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(goal[3:])).magnitude()<1e-7:break
            else:return finish(False,'segment_budget')
        return finish(True,'continuous_IK_only_not_collision_proof')

    def _grasp_gate(self, obs):
        self.step+=1; gate=self.gate
        self._evidence(obs,gate['context'],gate['observations'])
        if len(gate['observations'])>=2:
            self._queue_review('post_lift',gate['context']); self.pending['observations']=list(gate['observations']); self.gate=None
        self.reason='fresh_wrist_observation'
        return self._stationary(obs['state'])

    def _within_side(self, side, poses):
        for pose in poses:
            tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
            if side=='left' and tcp[0]>.06 or side=='right' and tcp[0]<-.06:return False
        return True

    def _observe_move(self, obs, context):
        if not context or self.step>900:return None
        side=context['side']; goal=np.asarray(obs['state'][side+'_ee_pose']).copy()
        if not .065<=self._tcp_z(obs['state'],side)-self.calibration['table_z']<=.18:return None
        goal[0]+= -.015 if side=='left' else .015
        if not self._within_side(side,[goal]) or not self._checked_path(obs,side,[goal]):return None
        return dict(side=side,goal=goal,age=0,dwell=0,kind='observe')

    def _aux_act(self, obs):
        self.step+=1; aux=self.auxiliary; side=aux['side']; goal=aux['goal']
        action=self._stationary(obs['state']); current=action[side+'_ee_pose']
        action[side+'_ee_pose']=bounded_pose(current,goal,self.calibration['precision_translation_step'],self.calibration['max_rotation_step'])
        pos=np.linalg.norm(current[:3]-goal[:3]); angle=(Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(goal[3:])).magnitude()
        aux['age']+=1; aux['dwell']=aux['dwell']+1 if pos<.005 and angle<.12 else 0
        if aux['dwell']>=2:
            self.auxiliary=None
            if self.pending:self.pending['observations']=[]
            self._event('observation_pose_reached')
        elif aux['age']>=60:return self._freeze(obs,'observation_motion_failed_hold_closed')
        return action

    def _defer_or_stop(self, obs, pending, reason):
        if pending['attempt']<2:
            self._event('reobserve_fresh',pending['context'],cause=reason);return self._stationary(obs['state'])
        if not pending['observation_move_used']:
            pending['observation_move_used']=True; move=self._observe_move(obs,pending['context'])
            if move is not None:
                self.auxiliary=move;self._event('bounded_observation_move',pending['context'],cause=reason)
                return self._stationary(obs['state'])
        return self._freeze(obs,reason+'_bounded_observation_exhausted')

    def _detour(self, obs, context):
        label=context['label'];side=context['side']
        if label in self.recovered or self.step>850:return None
        goal=self.plan[self.index]
        if not goal['name'].startswith(('carry_','place_','lift_')):return None
        current=np.asarray(obs['state'][side+'_ee_pose']).copy()
        z=self._tcp_z(obs['state'],side)-self.calibration['table_z']
        if not .035<=z<=.20:return None
        candidates=[]
        for dx,dy,dz in ((-.025 if side=='left' else .025,-.02,.015),(0.,-.025,.025),(0.,0.,.025)):
            if not .065<=z+dz<=.20:continue
            middle=current.copy();middle[:3]+=[dx,dy,dz];candidates.append(middle)
            if self._within_side(side,[middle,goal['pose']]) and self._checked_path(obs,side,[middle,goal['pose']]):
                return dict(side=side,pose=middle,gripper=0.,name='supervisor_detour',dwell=2)
        if goal['name'].startswith('carry_'):
            tcp=current[:3]+quat_matrix(current[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
            for height in (.08,.065):
                middle=self.pose(tcp[:2],height,np.pi/2)
                resume=self.pose(context['target'],height,np.pi/2)
                place=self.pose(context['target'],.024,np.pi/2)
                if self._within_side(side,[middle,resume,place]) and self._checked_path(obs,side,[middle,resume,place]):
                    return dict(side=side,pose=middle,gripper=0.,name='supervisor_detour',dwell=2,resume_pose=resume)
        self.path_rejections.append(dict(step=self.step,label=label,candidates=len(candidates),reason='side_or_continuous_IK_rejected',collision_certified=False))
        return None

    def _retry_plan(self, obs, context, source):
        if source is None or context['digit'] in self.retry_digits or source['confidence']<.65 or self.step>800:return None
        side=context['side'];saved_plan,saved_transfers=self.plan,self.transfers;self.plan=[];self.transfers=[]
        try:
            self.transfer(side,np.asarray(source['xy']),np.asarray(context['target']),source['yaw'],context['height'],context['label']+'_retry')
            proposed=self.plan;transfer=self.transfers[0]
        finally:self.plan,self.transfers=saved_plan,saved_transfers
        poses=[w['pose'] for w in proposed]
        if not self._within_side(side,poses) or not self._checked_path(obs,side,poses):return None
        end=self.index
        while end<len(self.plan) and self.plan[end]['name'].endswith('_'+context['label']):end+=1
        saved_plan,saved_index=self.plan,self.index
        try:
            self.plan=self.plan[end:];self.index=0
            cost=self._remaining_cost(obs,proposed)
        finally:self.plan,self.index=saved_plan,saved_index
        if self.step+cost>=980:return None
        for w in proposed:w['checked_slow']=True
        return dict(plan=proposed,transfer=transfer)

    def _repair_plan(self, obs, context, found):
        # A retried staging leg is still staging; never treat it as final placement.
        if '_stage' in context['label'] or '_repair' in context['label']:return None
        result=super()._repair_plan(obs,context,found)
        if result:
            for w in result['plan']:w['checked_slow']=True
        return result

    def _skip_safely(self, obs, context):
        side=context['side'];lift=np.asarray(obs['state'][side+'_ee_pose']).copy()
        lift[2]+=max(0.,.10-(self._tcp_z(obs['state'],side)-self.calibration['table_z']))
        poses=[lift,self.home[side]]
        if not self._within_side(side,poses) or not self._checked_path(obs,side,poses):return self._freeze(obs,'confirmed_empty_no_checked_exit')
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
            found=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],np.asarray(context['target']),context['family'],height=.014,max_distance=.05)
            repair=self._repair_plan(obs,context,found);payload['local_evidence']=dict(local_digit_pose=found,repair_available=repair is not None)
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
            if answer['action'] in ('hold','observe') or not held:return self._defer_or_stop(obs,pending,'possession_unresolved')
            if stage=='pre_place' and answer['placement'] in ('misaligned','wrong_digit','unknown'):return self._defer_or_stop(obs,pending,'target_unresolved')
            if stage=='transport_stall':
                if answer['action']=='recover' and detour:
                    resume=detour.pop('resume_pose',None)
                    if resume is not None:
                        self.plan[self.index]=dict(self.plan[self.index],pose=resume,checked_slow=True)
                        for j in range(self.index+1,len(self.plan)):
                            if self.plan[j]['name']=='retreat_'+context['label']:
                                self.plan[j]=dict(self.plan[j],pose=resume.copy(),checked_slow=True);break
                    detour['checked_slow']=True
                    self.plan.insert(self.index,detour);self.recovered.add(context['label']);self.age=self.dwell=0;self.progress=[]
                    self.correction_paths.append(dict(kind='transit_detour',step=self.step,label=context['label'],pose=detour['pose'].tolist()))
                    self._event('checked_detour_once',context)
                else:return self._defer_or_stop(obs,pending,'no_selected_feasible_recovery')
            else:
                self.held_status[context['label']]=model;self.route_states[str(context['digit'])]='held_confirmed';self._event('held_resume_ANS4',context)
        elif stage=='post_release':
            if answer['action']=='observe' and pending['attempt']<2:self._event('release_reobserve',context);return action
            if answer['action']=='repair' and answer['placement']=='misaligned' and repair:
                self.plan[self.index:self.index]=repair['plan'];self.transfers.append(repair['transfer']);self.repaired.add(context['digit']);self.age=self.dwell=0
                self._event('checked_placement_repair_once',context)
            verified=found is not None and np.linalg.norm(np.asarray(found['xy'])-context['target'])<.012 and abs(found['yaw'])<.16 and found['confidence']>=.65 and answer['placement']=='aligned'
            self.route_states[str(context['digit'])]='placed_verified' if verified else 'released_unverified'
        elif stage=='final':self.final_checked=True;self._event('program_complete_visual_review_not_success_claim')
        self.pending=None;return action

    def _act(self, obs):
        validate_observation(obs)
        if self.hold is not None:self.step+=1;return {k:v.copy() for k,v in self.hold.items()}
        if self.auxiliary is not None:return self._aux_act(obs)
        if self.pending is not None:return self._review(obs)
        if self.gate is not None:return self._grasp_gate(obs)
        if not self.planned:
            self.home={s:np.asarray(obs['state'][s+'_ee_pose']).copy() for s in ARMS};self.joints={s:np.asarray(obs['state'].get(s+'_arm_joint_state',[])).copy() for s in ARMS}
            self.build(obs['vision']['cam_head']['color']);self.planned=True
            if self.plan:self._queue_review('initial');return self._review(obs)
        if self.index>=len(self.plan):
            if self.plan and not self.final_checked:self._queue_review('final');return self._review(obs)
            self.step+=1;return self._stationary(obs['state'])
        w=self.plan[self.index];old_index=self.index
        if self.step>=940 and w['name'].startswith('approach_'):self.step+=1;return self._freeze(obs,'budget_no_new_transfer_hold')
        if w['name'].startswith(('carry_','place_','lift_')):
            position=np.asarray(obs['state'][w['side']+'_ee_pose'][:3]);self.progress.append(position.copy());self.progress=self.progress[-12:]
            stagnant=len(self.progress)==12 and np.max(np.linalg.norm(np.asarray(self.progress)-self.progress[0],axis=1))<.002
            trigger=(self.age>=45 and stagnant and np.linalg.norm(position-w['pose'][:3])>=.012) or self.age>=95
            key=(w['name'],self.index)
            if trigger and key not in self.stall_reviewed:
                self.stall_reviewed.add(key);self._queue_review('transport_stall',self._context(w));return self._review(obs)
        else:self.progress=[]
        action=ANS4.act(self,obs)
        if self.reason=='motion_timeout_hold':return self._freeze(obs,'motion_timeout_no_safe_recovery')
        if w.get('checked_slow'):
            key=w['side']+'_ee_pose'
            action[key]=bounded_pose(np.asarray(obs['state'][key]),w['pose'],self.calibration['precision_translation_step'],self.calibration['max_rotation_step'])
        if self.index!=old_index:
            if w['name'].startswith('close_'):
                context=self._context(w);self.reference_frames[context['label']]=self._images(obs,context)
            elif w['name'].startswith('carry_'):self._queue_review('pre_place',self._context(w))
            elif w['name'].startswith('retreat_'):self._queue_review('post_release',self._context(w))
        return action

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS6',recovery_state=self.reason,recovery_transitions=self.transitions,
            retry_digits=sorted(self.retry_digits),route_states=self.route_states,path_rejections=self.path_rejections,
            path_checks=self.path_checks,
            auxiliary_motion=None if self.auxiliary is None else self.auxiliary['kind'],fresh_evidence_each_observation=True,regrasp_enabled=True)
        return result
