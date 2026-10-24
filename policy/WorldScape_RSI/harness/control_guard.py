"""Generic delegation, progress and initial-pose contract; no task metadata."""
from __future__ import annotations
import copy, math
import numpy as np
WAM_IDS={'worldscape-wam-memory','worldscape-wam-nomemory'}
PROGRESS={'initial','advancing','needs_more_motion','stalled','wrong_direction','uncertain'}
class ControlGuard:
    def __init__(self): self.reset()
    def reset(self):
        self.initial=self.latest=None; self.step=0; self.permit=None
        self.homing=False; self.native_required=False; self.stalls=0
        self.last_progress_step=-1; self.last_assessment_step=-1; self.last_wam_step=None
        self.last_reason=''; self.expected=''; self.history=[]
    def observe(self,observation,step,executed_action=None,was_wam=False):
        state=observation['state']; old=self.latest
        if self.initial is None: self.initial=copy.deepcopy(state)
        self.latest=copy.deepcopy(state); self.step=step; self.permit=None
        if executed_action is not None:
            if was_wam: self.last_wam_step=step
            elif old is not None and any(np.max(np.abs(np.asarray(executed_action[a+'_arm_joint_state'])-np.asarray(old[a+'_arm_joint_state'])))>1e-4 and np.max(np.abs(np.asarray(state[a+'_arm_joint_state'])-np.asarray(old[a+'_arm_joint_state'])))>1e-4 for a in ('left','right')):
                self.native_required=False; self.stalls=0
    def _text(self,value,label):
        if not isinstance(value,str) or not 8<=len(value.strip())<=1600: raise ValueError(label+' needs concise observable evidence')
        return value.strip()
    def _progress(self,progress,evidence):
        if progress not in PROGRESS: raise ValueError('Unknown progress')
        if progress=='initial' and self.last_wam_step is not None: raise ValueError('Initial cannot erase earlier execution feedback')
        evidence=self._text(evidence,'evidence')
        if self.last_progress_step!=self.step:
            if progress=='stalled': self.stalls+=1
            elif progress=='advancing': self.stalls=0
            self.last_progress_step=self.step
        if progress in ('wrong_direction','uncertain') or self.stalls>=2:
            self.native_required=True; self.last_reason='Delegation withheld until an acknowledged native correction and a fresh assessment'
        self.history.append({'step':self.step,'progress':progress,'evidence':evidence}); self.history=self.history[-64:]
    def assess(self,args):
        required={'mode','progress','evidence','expected_change','instruction_sufficient','requires_precise_control','requires_agent_memory','history_available'}
        if set(args)!=required: raise ValueError('Use exactly the advertised assessment fields')
        if args['mode'] not in {'native','flash','memory','return_home'}: raise ValueError('Unknown mode')
        for key in ('instruction_sufficient','requires_precise_control','requires_agent_memory','history_available'):
            if type(args[key]) is not bool: raise ValueError(key+' must be boolean')
        expected=self._text(args['expected_change'],'expected_change')
        if self.last_assessment_step==self.step: raise ValueError('One assessment per fresh observation; choose native control if rejected')
        self._progress(args['progress'],args['evidence']); self.last_assessment_step=self.step; self.permit=None; self.expected=expected
        mode=args['mode']
        if mode=='return_home': self.homing=True
        if mode in ('native','return_home'): return self.status()
        if self.homing or self.native_required: raise ValueError('Use native control: '+self.last_reason)
        if not args['instruction_sufficient'] or args['requires_precise_control'] or args['requires_agent_memory']:
            self.native_required=True; raise ValueError('This stage needs constraints unavailable in the WAM input; use native Tools')
        self.permit={'step':self.step,'skill':'worldscape-wam-memory' if mode=='memory' else 'worldscape-wam-nomemory','history_available':args['history_available']}
        return self.status()
    def consume(self,skill_id,continuous=None):
        if self.homing or self.native_required or self.permit is None or self.permit['step']!=self.step or self.permit['skill']!=skill_id:
            raise ValueError('WAM requires a fresh matching assess_control decision; no variant cycling after stalled/wrong/uncertain progress')
        if skill_id=='worldscape-wam-memory' and self.permit['history_available'] and continuous!=skill_id:
            self.permit=None; raise ValueError('Memory starts with the current frame after a switch; unavailable history cannot justify delegation')
        self.permit=None
    def review(self,args):
        if set(args)!={'decision','progress','evidence'} or args['decision'] not in {'continue','takeover'} or args['progress']=='initial': raise ValueError('Invalid review decision')
        self._progress(args['progress'],args['evidence'])
        if args['decision']=='takeover': self.native_required=True; self.last_reason='Agent took control of an unfinished WAM chunk'
        return args['decision']=='takeover' or self.native_required or self.homing
    def home_status(self):
        if self.initial is None or self.latest is None: return {'verified':False,'reason':'No initial public pose'}
        arms={}
        for a in ('left','right'):
            p=np.asarray(self.initial[a+'_ee_pose'],float); q=np.asarray(self.latest[a+'_ee_pose'],float)
            dist=float(np.linalg.norm(p[:3]-q[:3])); denom=float(np.linalg.norm(p[3:7])*np.linalg.norm(q[3:7]))
            deg=math.degrees(2*math.acos(min(1.,abs(float(np.dot(p[3:7],q[3:7])))/denom))) if denom else float('inf')
            grip=float(np.max(np.abs(np.asarray(self.initial[a+'_ee_joint_state'])-np.asarray(self.latest[a+'_ee_joint_state']))))
            arms[a]={'position_error_m':dist,'orientation_error_deg':deg,'at_initial_pose':dist<=.005 and deg<=5.,'gripper_error':grip,'gripper_restored':grip<=.02}
        return {'verified':all(v['at_initial_pose'] and v['gripper_restored'] for v in arms.values()),'arms':arms,'scope':'Public robot pose tolerance, not task success or support verification'}
    def status(self):
        return {'step':self.step,'permit':copy.deepcopy(self.permit),'native_required':self.native_required,'homing':self.homing,'expected_change':self.expected,'reason':self.last_reason,'home':self.home_status()}


def policy_review_tool():
    """The one additional read-only tool for an acknowledged WAM boundary."""
    return {'type': 'function', 'function': {
        'name': 'review_policy_chunk',
        'description': 'Review a paused queued action suffix; no robot motion and no new WAM inference.',
        'parameters': {'type': 'object', 'additionalProperties': False,
            'properties': {
                'decision': {'type': 'string', 'enum': ['continue', 'takeover']},
                'progress': {'type': 'string', 'enum': [
                    'advancing', 'needs_more_motion', 'stalled', 'wrong_direction', 'uncertain']},
                'evidence': {'type': 'string'}},
            'required': ['decision', 'progress', 'evidence']}}}
