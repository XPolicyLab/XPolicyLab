"""Finite operation programs. Geometry is bound once; visual checkpoints remain explicit.

No action replay, reward feedback, generated Python, or implicit perception.
Each motion re-computes relative displacement from the current measured TCP.
"""
import math
from .object_state import ID, predicate

def move(waypoint): return {'kind':'move','waypoint':waypoint}
def grip(value): return {'kind':'gripper','value':value}
def check(relation, subject, target): return {'kind':'check','fact':[relation,subject,target]}

OPERATIONS = {
 'grasp_probe': [grip('open'),move('approach'),move('grasp'),grip('close'),move('probe'),check('held','arm','object')],
 'transfer_place': [check('held','arm','object'),move('transit'),move('place'),
                    check('release_allowed','object','target'),grip('open'),move('retreat'),check('stable','object','target')],
 'align_insert': [check('held','arm','object'),move('prealign'),check('aligned','object','target'),
                  move('entry'),move('seat'),check('seated','object','target'),
                  check('release_allowed','object','target'),grip('open'),move('retreat'),check('stable','object','target')],
 'tool_stroke': [check('held','arm','object'),move('start'),move('finish'),move('retreat'),check('goal_met','object','target')],
 'tilt_restore': [check('held','arm','object'),move('pour'),{'kind':'rotate','angle':'tilt_deg'},
                  {'kind':'wait','steps':'settle_steps'},check('goal_met','object','target'),
                  {'kind':'rotate','angle':'restore_deg'},move('retreat'),check('held','arm','object')],
 # Direct handover is intentionally distinct from a table-supported regrasp.
 'handover': [check('held','support_arm','object'),grip('open'),move('approach'),move('grasp'),grip('close'),
              check('held','arm','object'),{'kind':'gripper','value':'open','arm':'support_arm'},
              move('probe'),check('held','arm','object')],
 'table_place': [check('held','arm','object'),move('place'),check('supported','object','target'),
                 grip('open'),move('retreat'),check('stable','object','target')],
 'press_once': [move('approach'),move('press'),move('retreat'),check('goal_met','object','target')],
 'observe_event': [check('goal_met','object','target')],
}
OPERATIONS['pick_place'] = OPERATIONS['grasp_probe'] + OPERATIONS['transfer_place'][1:]
OPERATIONS['pick_insert'] = OPERATIONS['grasp_probe'] + OPERATIONS['align_insert'][1:]

def validate_binding(b, revision):
    if set(b) != {'operation','arm','object','target','support_arm','support_object',
                 'waypoints_cm','parameters','evidence','revision','max_steps','reserve_steps'}:
        raise ValueError('Invalid operation binding fields')
    if b['operation'] not in OPERATIONS or b['arm'] not in ('left','right'):
        raise ValueError('Unknown operation/arm')
    for name in ('object','target'):
        if not isinstance(b[name],str) or not ID.fullmatch(b[name]): raise ValueError('Invalid entity ID')
    if b['support_arm'] not in (None,'left','right') or b['support_arm']==b['arm']:
        raise ValueError('Hands must have distinct roles')
    if b['support_arm'] and (not isinstance(b['support_object'],str) or not ID.fullmatch(b['support_object'])):
        raise ValueError('Support object required')
    if b['operation']=='handover' and (not b['support_arm'] or b['support_object']!=b['object']):
        raise ValueError('Handover must identify the same object in both hands')
    if b['revision'] != revision or not isinstance(b['evidence'],str) or not b['evidence'].strip():
        raise ValueError('Current binding evidence required')
    for k in ('max_steps','reserve_steps'):
        if type(b[k]) is not int or not 0 <= b[k] <= 1200:raise ValueError('Invalid budget')
    if b['max_steps']==0:raise ValueError('Positive operation budget required')
    required={s['waypoint'] for s in OPERATIONS[b['operation']] if s['kind']=='move'}
    if set(b['waypoints_cm']) != required:raise ValueError('Explicit current geometry required')
    for p in b['waypoints_cm'].values():
        if not isinstance(p,list) or len(p)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>200 for v in p):
            raise ValueError('Invalid world TCP waypoint in cm')
    expected={'tilt_deg','restore_deg','settle_steps'} if b['operation']=='tilt_restore' else set()
    if set(b['parameters'])!=expected:raise ValueError('Unexpected parameters')
    if expected:
        for k in ('tilt_deg','restore_deg'):
            x=b['parameters'][k]
            if type(x) not in (int,float) or not math.isfinite(x) or abs(x)>90 or x%15:
                raise ValueError('Rotation must match existing grammar')
        if b['parameters']['tilt_deg']+b['parameters']['restore_deg']!=0:raise ValueError('Restore rotation must undo tilt')
        x=b['parameters']['settle_steps']
        if type(x) is not int or not 1<=x<=50:raise ValueError('Invalid settle')
    return b

class ProgramRunner:
    """IO owns physical limits, measured arrival, single writer, and fresh visual checks."""
    def __init__(self, io, audit):self.io,self.audit=io,audit

    def run(self, binding):
        b=validate_binding(binding,self.io.evidence().revision)
        start=self.io.evidence().steps; completed=[]; motion_calls=0; checks=0
        def end(status,code):
            result={'status':status,'failure_code':code,'operation':b['operation'],
                    'object':b['object'],'target':b['target'],'completed_nodes':completed,
                    'steps':self.io.evidence().steps-start,'motion_batches':motion_calls,'visual_checks':checks}
            self.audit({'kind':'program_result',**result});return result
        nodes=OPERATIONS[b['operation']];index=0
        if self.io.evidence().remaining<=b['reserve_steps']:
            return end('blocked','operation_budget')
        # Ground applicable path/roles before the first motion, in a single call.
        prerequisites=[('path_clear',b['object'],b['target'])]
        if b['operation'] in ('grasp_probe','handover','pick_place','pick_insert'):
            prerequisites.append(('hand_free',b['arm'],'gripper'))
        if b['support_arm']:
            prerequisites.append(('held',b['support_arm'],b['support_object']))
        checks+=1
        initial=self.io.program_check(prerequisites,b)
        if any(initial.get(predicate(*q)) is not True for q in prerequisites):
            return end('uncertain','preconditions_unverified')
        while index<len(nodes):
            e=self.io.evidence()
            if e.ended or e.remaining<=b['reserve_steps'] or e.steps-start>=b['max_steps']:
                return end('blocked','operation_budget')
            # Group adjacent semantic checks into ONE perception request.
            if nodes[index]['kind']=='check':
                requested=[];ids=[]
                while index<len(nodes) and nodes[index]['kind']=='check':
                    rel,sub,obj=nodes[index]['fact'];requested.append((rel,b[sub],b[obj]));ids.append(index);index+=1
                checks+=1
                values=self.io.program_check(requested,b)
                if any(v is not True for v in values.values()) or any(values.get(predicate(*q)) is not True for q in requested):
                    return end('uncertain','visual_condition_unverified')
                completed.extend(ids);continue
            # Keep a physical boundary after each primitive, with no Astra call for arrival.
            node=nodes[index]
            report=self.io.program_motion(node,b,min(e.remaining-b['reserve_steps'],b['max_steps']-(e.steps-start)))
            motion_calls+=1
            if report.get('status')!='completed':return end(report.get('status','blocked'),report.get('failure_code','motion_failure'))
            completed.append(index);index+=1
        return end('completed',None)

class EventMemory:
    """Symbolic observations are supplied by perception; rules never invent their labels."""
    def __init__(self):self.events=[];self.completed=set()
    def record(self, identity, evidence_ref):
        if not evidence_ref:raise ValueError('Event evidence required')
        self.events.append({'identity':identity,'evidence':evidence_ref})
    def finish(self, obj, target, evidence_ref):
        self.record((obj,target),evidence_ref);self.completed.add((obj,target))
    @staticmethod
    def descending_digits(digits):
        if any(type(x) is not int or not 0<=x<=9 for x in digits):raise ValueError('Recognized digits required')
        return sorted(digits,reverse=True)
    @staticmethod
    def solve_missing_addend(total,known):
        if type(total) is not int or type(known) is not int:raise ValueError('Recognized integers required')
        return total-known
