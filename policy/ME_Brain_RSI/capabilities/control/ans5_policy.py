"""ANS5: ANS4 segments with bounded key-node Astra supervision.

No model-generated coordinates/code. Extra transit/correction motion requires
current-joint incremental IK checks, not a world-collision guarantee.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .ans4_policy import Policy as ANS4
from .astra_supervisor import Supervisor, MODEL
from .contracts import ARMS, STATE_KEYS, validate_observation, validate_action
from .geometry import quat_matrix, bounded_pose
from .kinematics import pose_matrix
from .wrist_grasp import decide


class Policy(ANS4):
    def __init__(self,task,calibration,supervisor=None):
        super().__init__(task,calibration)
        self.supervisor=supervisor if supervisor is not None else Supervisor()
        self.pending=None;self.reference_frames={};self.final_checked=False
        self.recovered=set();self.repaired=set();self.progress=[];self.visual_checks=[]
        self.supervisor_failure=None;self.held_status={};self.correction_paths=[]
        self.last_gripper_commands={}

    def _stationary(self,state):
        action=ANS4._stationary(state)
        for side,value in self.last_gripper_commands.items():action[side+'_ee_joint_state']=value.copy()
        return action

    def _context(self,waypoint):
        label=waypoint['name'].split('_',1)[1];identity=int(label.split('_')[0])
        transfer=next(t for t in reversed(self.transfers) if t['digit']==label)
        digit=next(d for d in self.digits if d['digit']==identity)
        return dict(digit=identity,label=label,side=waypoint['side'],source=transfer['source'],
            target=transfer['target'],family=digit['family'],height=transfer['height'])

    def _images(self,obs,context):
        names=['cam_head']+(['cam_'+context['side']+'_wrist'] if context else ['cam_left_wrist','cam_right_wrist'])
        return [(name,obs['vision'][name]['color'].copy()) for name in names if name in obs['vision']]

    def _skip_empty(self,context):
        # Include correction legs as dependencies; never carry/release an empty repair.
        import re
        kept=[]
        for item in self.plan[self.index:]:
            match=re.search(r'_(\d+)(?:_stage|_repair)?$',item['name'])
            if match and int(match[1])==context['digit']:continue
            kept.append(item)
        park=dict(side=context['side'],pose=self.home[context['side']].copy(),gripper=1.,name='empty_park',dwell=3)
        self.plan=self.plan[:self.index]+[park]+kept;self.age=self.dwell=0
        self.skipped.append(dict(digit=context['digit'],reason='local_and_astra_confirmed_empty'))

    def _freeze(self,obs,reason):
        self.hold=self._stationary(obs['state']);self.reason=reason
        return {k:v.copy() for k,v in self.hold.items()}

    def _checked_path(self,obs,side,poses):
        joints=np.asarray(obs['state'].get(side+'_arm_joint_state',[]),float)
        if joints.shape!=(6,) or not np.isfinite(joints).all():return False
        current=np.asarray(obs['state'][side+'_ee_pose'],float).copy();q=joints.copy();samples=0
        for goal in poses:
            for _ in range(100):
                nxt=bounded_pose(current,goal,self.calibration['precision_translation_step'],self.calibration['max_rotation_step'])
                fit=self.arms[side].inverse(pose_matrix(nxt),initial=q,restarts=0);new=np.asarray(fit['joints'])
                if fit['position_error_m']>=.004 or fit['orientation_error_rad']>=.08 or np.max(np.abs(new-q))>.35:return False
                q=new;current=nxt;samples+=1
                if samples>120:return False
                if np.linalg.norm(current[:3]-goal[:3])<1e-7 and (Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(goal[3:])).magnitude()<1e-7:break
            else:return False
        return True

    def _detour(self,obs,context):
        if context['label'] in self.recovered or self.step>850:return None
        w=self.plan[self.index]
        if not w['name'].startswith('carry_'):return None
        side=context['side'];pose=np.asarray(obs['state'][side+'_ee_pose']).copy()
        tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(self.calibration['tool_offset_in_ee'])
        height=tcp[2]-self.calibration['table_z']
        if not .085<=height<=.18:return None
        shift=np.array([-.018 if side=='left' else .018,-.012,-min(.02,height-.065)])
        if (side=='left' and tcp[0]+shift[0]>.06) or (side=='right' and tcp[0]+shift[0]<-.06):return None
        intermediate=pose.copy();intermediate[:3]+=shift
        if not self._checked_path(obs,side,[intermediate,w['pose']]):return None
        return dict(side=side,pose=intermediate,gripper=0.,name='supervisor_detour',dwell=2)

    def _remaining_cost(self,obs,extra):
        positions={s:np.asarray(obs['state'][s+'_ee_pose']).copy() for s in ARMS};cost=0
        for w in extra+self.plan[self.index:]:
            start=positions[w['side']];end=w['pose']
            distance=np.linalg.norm(start[:3]-end[:3])/self.calibration['precision_translation_step']
            angle=(Rotation.from_quat(start[3:]).inv()*Rotation.from_quat(end[3:])).magnitude()/self.calibration['max_rotation_step']
            cost+=int(np.ceil(max(distance,angle)))+w['dwell']+3;positions[w['side']]=end
        return cost+25

    def _repair_plan(self,obs,context,found):
        identity=context['digit']
        if identity in self.repaired or context['label'].endswith(('_stage','_repair')) or found is None:return None
        error=float(np.linalg.norm(np.asarray(found['xy'])-context['target']))
        if not .012<=error<=.045 or abs(found['yaw'])>.35 or found['confidence']<.65:return None
        side=context['side'];xy=np.asarray(found['xy']);goal=np.asarray(context['target'])
        if (side=='left' and max(xy[0],goal[0])>.06) or (side=='right' and min(xy[0],goal[0])<-.06):return None
        saved_plan=self.plan;saved_transfers=self.transfers;self.plan=[];self.transfers=[]
        try:
            self.transfer(side,xy,goal,found['yaw'],context['height'],str(identity)+'_repair')
            proposed=self.plan;transfer=self.transfers[0]
        finally:self.plan=saved_plan;self.transfers=saved_transfers
        if self.step+self._remaining_cost(obs,proposed)>=980:return None
        if not self._checked_path(obs,side,[w['pose'] for w in proposed]):return None
        return dict(plan=proposed,transfer=transfer)

    def _queue_review(self,stage,context=None,**extra):
        self.pending=dict(stage=stage,context=context,attempt=0,**extra)

    def _review(self,obs):
        self.step+=1;pending=self.pending;stage=pending['stage'];context=pending['context']
        action=self._stationary(obs['state']);allowed=['continue','observe','hold']
        payload=dict(robot_state={k:np.asarray(obs['state'][k]).tolist() for k in STATE_KEYS if k in obs['state']},task_instruction=obs['instruction'][:2000],
            target=context,local_evidence={},action_scope='No arbitrary coordinates, releases, retries or threshold changes.')
        if stage=='initial':payload.update(detected_digits=self.digits,targets_left_to_right=self.targets,intended_order=sorted([d['digit'] for d in self.digits],reverse=True))
        elif stage=='post_lift':
            payload['local_evidence']=pending['evidence']
            if pending['local_status']=='empty':allowed.append('skip_empty')
        elif stage=='transport_stall':
            detour=self._detour(obs,context);pending['detour']=detour
            payload['local_evidence']=dict(motion=self.motion,detour_available=detour is not None)
            if detour is not None:
                allowed.append('recover');payload['detour_description']='18mm toward active-arm side, 12mm toward robot, at most 20mm lower; at least 65mm table clearance; incremental current-joint IK passed, not collision certification.'
        elif stage=='post_release':
            found=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],np.asarray(context['target']),context['family'],height=.014,max_distance=.05)
            repair=self._repair_plan(obs,context,found);pending.update(found=found,repair=repair)
            payload['local_evidence']=dict(local_digit_pose=found,repair_available=repair is not None)
            if repair is not None:allowed.append('repair')
        elif stage=='pre_place':payload['local_evidence']=dict(motion=self.motion,previous_possession=self.held_status.get(context['label']))
        elif stage=='final':payload.update(targets_left_to_right=self.targets,detected_initial_digits=self.digits)
        frames=[('before_close_'+name,image) for name,image in self.reference_frames.get(context['label'],[])] if context else []
        frames += [('current_'+name,image) for name,image in self._images(obs,context)]
        answer=self.supervisor.review(stage,self.step,payload,frames,allowed)
        self.visual_checks.append(dict(step=self.step,stage=stage,label=context and context['label'],answer=answer))
        if answer.get('error'):
            self.supervisor_failure=answer['error'];self.pending=None;return self._freeze(obs,'astra_unavailable_hold')
        pending['attempt']+=1
        if answer['action']=='observe' and pending['attempt']<2:self.reason='astra_stationary_observation';return action
        self.pending=None
        if stage=='initial':
            if answer['action']=='hold' or answer['placement']=='wrong_digit':return self._freeze(obs,'astra_initial_conflict_hold')
        elif stage=='post_lift':
            local=pending['local_status'];model=answer['possession']
            if local=='empty' and model=='empty' and answer['action']=='skip_empty':
                self._skip_empty(context);self.reason='astra_confirmed_empty_skip';return action
            coarse=pending['evidence']['coarse_stable_contact']
            held=(local=='held' and model in ('held_clear','held_occluded','unknown')) or (coarse and model in ('held_clear','held_occluded') and answer['confidence']>=.75)
            if not held or answer['action']=='hold':return self._freeze(obs,'astra_grasp_conflict_hold_closed')
            self.held_status[context['label']]=model;self.reason='astra_held_resume_ANS4'
        elif stage=='pre_place':
            if answer['action']=='hold' or answer['possession'] in ('empty','dropped') or answer['placement'] in ('misaligned','wrong_digit'):return self._freeze(obs,'astra_pre_place_conflict_hold_closed')
        elif stage=='transport_stall':
            detour=pending.get('detour')
            if answer['action']=='recover' and detour is not None and answer['possession'] in ('held_clear','held_occluded'):
                self.plan.insert(self.index,detour);self.recovered.add(context['label']);self.age=self.dwell=0;self.progress=[]
                self.correction_paths.append(dict(kind='transit_detour',step=self.step,label=context['label'],pose=detour['pose'].tolist()))
            elif answer['action']=='hold' or answer['possession'] in ('empty','dropped'):return self._freeze(obs,'astra_transport_hold_closed')
            else:self.recovered.add(context['label'])
        elif stage=='post_release':
            repair=pending.get('repair')
            if answer['action']=='repair' and answer['placement']=='misaligned' and repair is not None:
                self.plan[self.index:self.index]=repair['plan'];self.transfers.append(repair['transfer']);self.repaired.add(context['digit']);self.age=self.dwell=0
                self.correction_paths.append(dict(kind='placement_repair',step=self.step,label=context['label'],source=pending['found'],target=context['target']))
            self.reason='astra_post_release_recorded'
        elif stage=='final':self.final_checked=True;self.reason='program_complete_visual_review_not_success_claim'
        validate_action(action);return action

    def _grasp_gate(self,obs):
        self.step+=1;gate=self.gate;context=gate['context'];features=self._wrist(obs,context)
        gate['observations'].append(features);reference=self.references.get(context['label'],{})
        lift=self._tcp_z(obs['state'],context['side'])-reference.get('tcp_z',self._tcp_z(obs['state'],context['side']))
        source=None
        if len(gate['observations'])>=2 and not features.get('contact'):source=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],np.asarray(context['source']),context['family'],height=.014,max_distance=.035)
        local=decide(reference.get('features'),gate['observations'],lift,source is not None)
        action=self._stationary(obs['state']);action[context['side']+'_ee_joint_state']=np.array([0.])
        self.grasp_checks.append(dict(step=self.step,digit=context['digit'],side=context['side'],status=local,wrist=features,lift_metres=float(lift),source_pose=source))
        if len(gate['observations'])>=2:
            observations=gate['observations'][-2:]
            coarse=all(f.get('available') and f.get('contact') and f.get('object',{}).get('area',0)>.01 for f in observations)
            if coarse:coarse=np.linalg.norm(np.asarray(observations[0]['object']['centre'])-observations[1]['object']['centre'])<.04 and lift>=.035
            self.gate=None;self._queue_review('post_lift',context,local_status=local,evidence=dict(local_status=local,wrist_observations=observations,reference=reference,lift_metres=float(lift),source_pose=source,coarse_stable_contact=bool(coarse)))
        return action

    def _act(self,obs):
        validate_observation(obs)
        if self.hold is not None:self.step+=1;return {k:v.copy() for k,v in self.hold.items()}
        if self.pending is not None:return self._review(obs)
        if self.gate is not None:return self._grasp_gate(obs)
        if not self.planned:
            self.home={s:np.asarray(obs['state'][s+'_ee_pose']).copy() for s in ARMS}
            self.joints={s:np.asarray(obs['state'].get(s+'_arm_joint_state',[])).copy() for s in ARMS}
            self.build(obs['vision']['cam_head']['color']);self.planned=True
            if self.plan:self._queue_review('initial');return self._review(obs)
        if self.index>=len(self.plan):
            if self.plan and not self.final_checked:self._queue_review('final');return self._review(obs)
            self.step+=1;return self._stationary(obs['state'])
        waypoint=self.plan[self.index];old_index=self.index
        if self.step>=940 and waypoint['name'].startswith('approach_'):self.step+=1;return self._freeze(obs,'budget_no_new_transfer_hold')
        if waypoint['name'].startswith('carry_'):
            context=self._context(waypoint);position=np.asarray(obs['state'][waypoint['side']+'_ee_pose'][:3])
            self.progress.append(position.copy());self.progress=self.progress[-12:]
            if self.age>=45 and len(self.progress)==12 and context['label'] not in self.recovered and np.max(np.linalg.norm(np.asarray(self.progress)-self.progress[0],axis=1))<.002 and np.linalg.norm(position-waypoint['pose'][:3])>=.012:
                self._queue_review('transport_stall',context);return self._review(obs)
        else:self.progress=[]
        action=super().act(obs)
        if self.reason=='motion_timeout_hold':self.hold=self._stationary(obs['state']);return {k:v.copy() for k,v in self.hold.items()}
        if self.index!=old_index:
            if waypoint['name'].startswith('close_'):
                context=self._context(waypoint);self.reference_frames[context['label']]=self._images(obs,context)
            elif waypoint['name'].startswith('carry_'):self._queue_review('pre_place',self._context(waypoint))
            elif waypoint['name'].startswith('retreat_'):self._queue_review('post_release',self._context(waypoint))
        validate_action(action);return action

    def act(self,obs):
        action=self._act(obs);validate_action(action)
        self.last_gripper_commands={s:action[s+'_ee_joint_state'].copy() for s in ARMS}
        return action

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS5',astra_calls=self.supervisor.calls,astra_elapsed_seconds=self.supervisor.elapsed,
            astra_records=self.supervisor.records,astra_failure=self.supervisor_failure,visual_checks=self.visual_checks,
            pending_review=None if self.pending is None else self.pending['stage'],bounded_recoveries=sorted(self.recovered),
            placement_corrections=sorted(self.repaired),correction_paths=self.correction_paths,placement_correction_enabled=True,
            runtime_models=[MODEL] if MODEL else [],model_generated_coordinates=False)
        return result
