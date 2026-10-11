"""Bridge reusable programs to the existing native command executor."""
import math
from .object_state import ObjectState, predicate
from .program_policy import OPERATIONS, ProgramRunner, validate_binding, EventMemory

class ProgramBudgetReached(RuntimeError):pass

class ProgramAdapter:
    def init_programs(self):
        self.object_state=ObjectState();self.event_memory=EventMemory()
        self.program_attempts={};self.program_results=[]
        self.program_abstained=set();self.program_bind_calls={}
        self.program_parent=None;self.program_checked_step=None
        self.program_failed_bindings=set()

    def program_check(self, requested, binding):
        if binding.get('support_arm') and binding['operation']!='handover':
            requested=list(requested)+[('held',binding['support_arm'],binding['support_object'])]
        names=list(dict.fromkeys(predicate(*q) for q in requested))
        cached={k:self.object_state.get(k,self.evidence().steps) for k in names}
        if all(v is True for v in cached.values()):return cached
        # No generic object_held boolean is accepted here.
        completion=names+list((self.program_parent or {}).get('completion',[]))
        e=self.inspect({'goal':binding['operation'],'parent_stage':self.program_parent,'completion':completion},
            'Verify only these object/hand-specific relations using CURRENT RGB. release_allowed means supported deliberate release at the stated target; held needs object evidence, not finger closure. Unknown=null.')
        for rel,sub,obj in requested:
            key=predicate(rel,sub,obj)
            self.object_state.put(rel,sub,obj,e.facts.get(key),e.steps,e.revision,
                ';'.join(e.refs) or 'current RGB inspection',ttl=0)
        self.program_checked_step=e.steps
        self.audit({'kind':'object_state','state':self.object_state.snapshot(e.steps)})
        return {k:self.object_state.get(k,e.steps) for k in names}

    def _tcp_cm(self, arm):
        from .. import main_route
        offset=float(self.config.get('main',{}).get('tcp_offset_m',main_route.TCP_OFFSET_M))
        return (main_route.tcp_of(*main_route.live_poses(self.env)[arm],offset)*100).tolist()

    def _program_line(self,line,limit):
        from .. import open_restore,motion
        grammar=open_restore.module(self.env.task_name,'commands')
        commands,errors=grammar.parse_command_list([line])
        if errors or not commands or any(c.kind in ('done','home','homeplan') for c in commands):
            return {'status':'blocked','failure_code':'unsupported_command'}
        before=self.evidence().steps;take_action=self.env.take_action
        # Preserve the native limit; impose a smaller controller-local cap on submissions.
        def bounded(*args,**kwargs):
            if self.evidence().steps-before>=limit or self.stopped():raise ProgramBudgetReached()
            return take_action(*args,**kwargs)
        proposal={'envelope':motion.sequence([c.to_dict() for c in commands],{'program':True},
                  {'max_plan_steps':min(24,limit),'hold_steps':0,'gripper_steps':min(8,limit),'motion_settle_steps':min(4,limit)})}
        self.pending=proposal;self.validate(proposal,'cap',self.evidence().remaining)
        self.env.take_action=bounded
        try:
            report=self.execute(proposal,'cap')
        except ProgramBudgetReached:
            self.pending=None;self.revision+=1;self.facts={}
            self.motion={'hard_error':True,'program_budget_reached':True}
            return {'status':'blocked','failure_code':'operation_budget'}
        finally:
            self.env.take_action=take_action
        self.audit({'kind':'program_motion','command':line,'report':report})
        if self.motion.get('hard_error'):return {'status':'blocked','failure_code':'motion_failure'}
        return {'status':'completed'}

    def program_motion(self,node,b,limit):
        arm=b.get(node.get('arm','arm'));start=self.evidence().steps
        support=b.get('support_arm')
        # A hold role is a physical pose constraint, NOT proof the object is still held.
        support_before=self._tcp_cm(support) if support and arm!=support else None
        self.object_state.invalidate([arm,b['object']], 'program_motion')
        if node['kind']=='move':
            target=b['waypoints_cm'][node['waypoint']]
            for attempt in range(16):
                current=self._tcp_cm(arm);delta=[t-c for t,c in zip(target,current)]
                distance=math.sqrt(sum(v*v for v in delta))
                if distance<=.4:return {'status':'completed'}
                remaining=limit-(self.evidence().steps-start)
                if remaining<=0:return {'status':'blocked','failure_code':'operation_budget'}
                # Subdivide long transit; use finer segments at contact waypoints.
                cap=1.0 if node['waypoint'] in ('entry','seat','press') else 5.0
                scale=min(1,cap/distance);values=[v*scale for v in delta]
                line='{} move3d {}'.format(arm,' '.join('{:.5f}'.format(v) for v in values))
                result=self._program_line(line,remaining)
                if result['status']!='completed':return result
                after=self._tcp_cm(arm)
                if math.sqrt(sum((a-c)**2 for a,c in zip(after,current)))<.05:
                    return {'status':'blocked','failure_code':'no_measured_motion'}
                if support_before and math.dist(support_before,self._tcp_cm(support))>.5:
                    self.object_state.invalidate([support,b['support_object']],'support_pose_drift')
                    return {'status':'blocked','failure_code':'support_pose_drift'}
            return {'status':'blocked','failure_code':'waypoint_segment_limit'}
        if node['kind']=='gripper':line='{} gripper {}'.format(arm,node['value'])
        elif node['kind']=='rotate':line='{} rotate pitch {}'.format(arm,b['parameters'][node['angle']])
        elif node['kind']=='wait':line='wait {}'.format(b['parameters'][node['steps']])
        else:return {'status':'blocked','failure_code':'unsupported_program_node'}
        result=self._program_line(line,limit)
        if support_before and math.dist(support_before,self._tcp_cm(support))>.5:
            return {'status':'blocked','failure_code':'support_pose_drift'}
        return result

    def bind_operation(self,stage,evidence,recovery=False):
        recipe=self.library.task_programs.get(self.env.task_name)
        if not recipe:return None
        attempt_key=(stage['id'],recovery)
        if attempt_key in self.program_abstained or self.program_bind_calls.get(attempt_key,0)>=self.cfg.get('max_program_bindings_per_stage',6):return None
        self.program_bind_calls[attempt_key]=self.program_bind_calls.get(attempt_key,0)+1
        offered={k:OPERATIONS[k] for k in recipe['operations']}
        result=self.call('bind_operation',{'stage':stage,'recipe':recipe,'operations':offered,
             'recovery':recovery,'revision':evidence.revision,
             'object_state':self.object_state.snapshot(evidence.steps),
             'events':self.event_memory.events[-12:],
             'instructions':'Choose a complete registered operation or null. Bind every WORLD TCP waypoint in centimeters from CURRENT observation/calibration; preserve current orientation unless this operation explicitly rotates. If orientation needs changing first, or a perception/coordination capability is missing, return null for Astra. No historic-coordinate replay. Keep the other hand in its assigned role. max_steps is a local budget within remaining native steps. No blind retries.'})
        b=result['binding']
        if b is None:
            self.program_abstained.add(attempt_key)
            return None
        validate_binding(b,evidence.revision)
        if b['operation'] not in offered:raise ValueError('Operation not offered')
        # Consume only explicitly scoped facts from the same binding observation.
        allowed={predicate('path_clear',b['object'],b['target']):('path_clear',b['object'],b['target'])}
        if b['operation'] in ('grasp_probe','handover','pick_place','pick_insert'):
            q=('hand_free',b['arm'],'gripper');allowed[predicate(*q)]=q
        if b['support_arm']:
            q=('held',b['support_arm'],b['support_object']);allowed[predicate(*q)]=q
        for node in OPERATIONS[b['operation']]:
            if node['kind']!='check':continue
            rel,sub,obj=node['fact'];q=(rel,b[sub],b[obj]);allowed[predicate(*q)]=q
        for fact in result['facts']:
            if fact['name'] in allowed:
                self.object_state.put(*allowed[fact['name']],fact['value'],evidence.steps,evidence.revision,
                    fact['evidence'],ttl=0)
        key=(stage['id'],b['operation'],b['object'],b['target'])
        from .contracts import digest
        signature=digest({k:b[k] for k in ('operation','arm','object','target','support_arm','support_object','waypoints_cm','parameters')})
        if signature in self.program_failed_bindings:
            self.audit({'kind':'program_rejected','reason':'same_failed_geometry'});return None
        if self.program_attempts.get(key,0)>=self.cfg.get('max_program_attempts_per_object',2):
            self.audit({'kind':'program_rejected','reason':'attempt_limit','key':key});return None
        return b

    def run_operation(self,binding,stage):
        calls=self.astra_calls
        self.program_parent=stage;self.program_checked_step=None
        result=ProgramRunner(self,self.audit).run(binding)
        result['astra_checks']=self.astra_calls-calls
        result['binding_calls']=1
        self.program_results.append(result)
        if result['status']!='completed':
            from .contracts import digest
            self.program_failed_bindings.add(digest({k:binding[k] for k in ('operation','arm','object','target','support_arm','support_object','waypoints_cm','parameters')}))
        if result['status']=='completed':
            ref=self.refs[-1] if self.refs else 'program_execution'
            self.event_memory.record({'operation':binding['operation'],'object':binding['object'],'target':binding['target']},ref)
            if binding['operation'] in ('transfer_place','align_insert','table_place','pick_place','pick_insert'):
                self.event_memory.finish(binding['object'],binding['target'],ref)
        return result
