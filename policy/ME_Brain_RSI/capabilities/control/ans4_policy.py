"""ANS4 = byte-preserved ANS1 plans plus wrist grasp confirmation (A).

No regrasp, new grasp geometry, pose correction, release check or learned model.
Confirmed held preserves the trajectory; empty skips only its dependent route.
Unknown cannot authorize blind transport or release.
"""
import re
import numpy as np
from .ans1_policy import Policy as ANS1
from .contracts import ARMS,validate_observation,validate_action
from .geometry import quat_matrix
from .placement_feedback import PlacementFeedback
from .wrist_grasp import wrist_features,decide


class Policy(ANS1):
    def __init__(self,task,calibration):
        super().__init__(task,calibration)
        self.source_checker=PlacementFeedback(self.perception)
        self.references={};self.gate=None;self.grasp_checks=[];self.skipped=[];self.hold=None

    def _context(self,waypoint):
        match=re.fullmatch(r'(?:close|lift)_(\d+)(?:_stage)?',waypoint['name'])
        identity=int(match[1]);label=waypoint['name'].split('_',1)[1]
        transfer=next(t for t in self.transfers if t['digit']==label)
        digit=next(d for d in self.digits if d['digit']==identity)
        return dict(digit=identity,label=label,side=waypoint['side'],source=transfer['source'],
            family=digit['family'],height=transfer['height'])

    def _tcp_z(self,state,side):
        ee=state[side+'_ee_pose']
        return float((np.asarray(ee[:3])+quat_matrix(ee[3:])@np.asarray(self.calibration['tool_offset_in_ee']))[2])

    @staticmethod
    def _stationary(state):
        return {s+'_'+key:np.array(state.get(s+'_'+key,[1.]),float).copy()
                for s in ARMS for key in ('ee_pose','ee_joint_state')}

    def _wrist(self,obs,context):
        frame=obs['vision'].get('cam_'+context['side']+'_wrist',{})
        return wrist_features(frame.get('color'),context['family'])

    def _skip_empty(self,context):
        # No retry/new grasp point (B). Park the confirmed empty arm using
        # ANS1's home pose; then execute unchanged independent transfers.
        pattern=re.compile(r'_(\d+)(?:_stage)?$');kept=[]
        for item in self.plan[self.index:]:
            match=pattern.search(item['name'])
            if match and int(match[1])==context['digit']:continue
            kept.append(item)
        park=dict(side=context['side'],pose=self.home[context['side']].copy(),gripper=1.,
            name='empty_park',dwell=3)
        self.plan=self.plan[:self.index]+[park]+kept;self.age=self.dwell=0
        self.skipped.append(dict(digit=context['digit'],reason='source_present_empty_wrist'))

    def act(self,obs):
        validate_observation(obs)
        if self.hold is not None:
            self.step+=1
            return {k:v.copy() for k,v in self.hold.items()}
        if self.gate is not None:
            self.step+=1;gate=self.gate;context=gate['context'];state=obs['state']
            action=self._stationary(state);action[context['side']+'_ee_joint_state']=np.array([0.])
            features=self._wrist(obs,context);gate['observations'].append(features)
            reference=self.references.get(context['label'],{})
            lift=self._tcp_z(state,context['side'])-reference.get('tcp_z',self._tcp_z(state,context['side']))
            source=None
            if len(gate['observations'])>=2 and not features.get('contact'):
                source=self.source_checker.locate(obs['vision']['cam_head']['color'],context['digit'],
                    np.asarray(context['source']),context['family'],height=.014,max_distance=.035)
            status=decide(reference.get('features'),gate['observations'],lift,source is not None)
            self.grasp_checks.append(dict(step=self.step,digit=context['digit'],side=context['side'],
                status=status,wrist=features,reference=reference,lift_metres=float(lift),source_pose=source,
                exact_held_pose_estimated=False))
            if status=='held':
                self.gate=None;self.reason='wrist_held_continue_ANS1'
            elif status=='empty':
                self._skip_empty(context);self.gate=None;self.reason='empty_route_skipped_no_retry'
            elif len(gate['observations'])>=3:
                self.hold={k:v.copy() for k,v in action.items()};self.reason='grasp_unresolved_hold_closed'
            validate_action(action);return action
        previous=self.plan[self.index] if self.index<len(self.plan) else None
        old_index=self.index;action=super().act(obs)
        if previous and self.index!=old_index and self.reason!='motion_timeout_hold':
            if previous['name'].startswith('close_'):
                context=self._context(previous)
                self.references[context['label']]=dict(features=self._wrist(obs,context),
                    tcp_z=self._tcp_z(obs['state'],context['side']))
            elif previous['name'].startswith('lift_'):
                self.gate=dict(context=self._context(previous),observations=[])
        return action

    def diagnostics(self):
        data=super().diagnostics()
        data.update(candidate='ANS4',grasp_checks=self.grasp_checks[-24:],skipped=self.skipped,
            grasp_gate=None if self.gate is None else self.gate['context'],
            exact_held_pose_estimated=False,regrasp_enabled=False,placement_correction_enabled=False)
        return data
