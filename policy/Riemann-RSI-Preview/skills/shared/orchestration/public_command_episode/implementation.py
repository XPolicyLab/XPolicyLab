"""Acknowledged transport for original public phase programs, without a model selector."""
from collections import deque
from copy import deepcopy
import importlib
import numpy as np
from harness.execution import ExecutionError
from harness.runtime.public_policy_recording import PublicEvidence as BasePublicEvidence


class PublicCommandEvidence(BasePublicEvidence):
    def append(self,observation,robots):
        super().append(observation,robots)
        self.instruction=observation.get("instruction")



class PublicCommandActor:
    def __init__(self, mode, observation, declaration, planner, *, controller=None):
        self.mode=mode
        self.controller=controller or importlib.import_module('tasks.'+mode+'.controller')
        self.public=importlib.import_module(declaration['public_api_module'])
        self.contract=importlib.import_module(declaration['contract_module'])
        self.motion=importlib.import_module(declaration['motion_module'])
        self.planner=planner
        self.home=self.public.PublicRobotAPI(observation).home_targets()
        self.home_joints={side:np.asarray(observation['state'][side+'_arm_joint_state']).copy() for side in ('left','right')}
        self.history=deque(maxlen=32)
        self.events=[]
        self.executor=self.motion.NativeJointExecutor(planner.plan,self.record)
        self.actions=0
        self.program=None
        self.request=None
        self.workflow=self.controller.execution() if callable(getattr(self.controller,'execution',None)) else None
        self.last_action={side+suffix:np.asarray(observation['state'][side+suffix]).copy() for side in ('left','right') for suffix in ('_arm_joint_state','_ee_joint_state')}
        self.phase='public_command_reset'
        self.holding_reason=None
        self.finished_workflow=False
        self.boundaries=0

    def record(self,event):
        self.events.append(deepcopy(event))

    def api(self,observation):
        return self.public.PublicRobotAPI(observation,self.home,{},trajectory=self.planner.plan,
            home_joints=self.home_joints,robot_bases=self.planner.robot_bases,
            command_memory=list(self.history),issued_action_count=self.actions)

    def select(self,observation):
        api=self.api(observation)
        if self.workflow is not None:
            request=self.workflow.next_request(api)
            if request is None:
                self.finished_workflow=True
                self.holding_reason='original_workflow_commands_complete'
                return
            factory=getattr(self.workflow.adapter,'action_program',None)
            if callable(factory):
                program=factory(api,deepcopy(request))
            else:
                program=self.contract.action_program(observation,request,self.home,{},trajectory=self.planner.plan,
                    home_joints=self.home_joints,robot_bases=self.planner.robot_bases,
                    command_memory=list(self.history),issued_action_count=self.actions)
            if hasattr(self.workflow,'prepare_program'):program=self.workflow.prepare_program(program)
        else:
            result=self.controller.resolve(api,list(self.history))
            operations=result['operations']
            if len(operations)!=1:
                self.finished_workflow=result.get('relation',{}).get('workflow_state')=='commands_complete_awaiting_verification' if isinstance(result.get('relation'),dict) else False
                self.holding_reason=result['reason']
                return
            operation=operations[0];skill=operation['skill'];parameters=operation['parameters']
            # Same singleton-only qualification as the source executable_request; never guess among alternatives.
            if (skill not in self.contract.ACTION_BINDINGS or set(parameters)!=set(self.contract.ACTION_PARAMETERS[skill])
                    or any(not isinstance(v,list) or len(v)!=1 for v in parameters.values())):
                self.holding_reason='Original resolver did not qualify a unique complete command'
                return
            request={'skill':skill,'steps':self.contract.ACTION_LIMITS[skill],
                'reason':result['reason'][:240],**{k:deepcopy(v[0]) for k,v in parameters.items()}}
            program=self.contract.action_program(observation,request,self.home,{},trajectory=self.planner.plan,
                home_joints=self.home_joints,robot_bases=self.planner.robot_bases,
                command_memory=list(self.history),issued_action_count=self.actions)
        self.request=self.contract.validate_action_parameters(request)
        if not isinstance(program,self.public.PhasedProgram):raise TypeError('The original tool did not return its declared public phase program')
        if hasattr(program,'enforce_command_order'):program.enforce_command_order(list(self.history))
        self.program=program
        self.record({'event':'skill_selected','request':deepcopy(self.request),'public_action_epoch':self.actions})

    def finish_boundary(self,observation):
        program=self.program
        feedback=self.contract.validate_action_feedback(self.request,program.feedback(observation,budget=program.status=='budget'))
        outcome={'completed':'public_tool_completed','precondition_unmet':'public_precondition_unmet','budget':'tool_action_budget'}.get(program.status,'public_tool_uncertain')
        event={'event':'skill_outcome','skill':self.request['skill'],'outcome':outcome,
            'actions':program.actions,'request':deepcopy(self.request),'execution_feedback':feedback,
            'public_action_epoch':self.actions,'official_success_claimed':False}
        if feedback['actions_executed']!=program.actions:raise RuntimeError('Public boundary action accounting changed')
        self.record(event);self.history.append(deepcopy(event));self.boundaries+=1
        if self.workflow is not None:self.workflow.accept(event)
        self.program=None;self.request=None

    def next_action(self,observation):
        # At most64 zero-action boundaries in one frame; actual task recovery remains in its original workflow.
        for _ in range(64):
            if self.holding_reason is not None:
                self.phase='official_final_hold'
                return deepcopy(self.last_action)
            try:
                if self.program is None:self.select(observation)
                if self.program is None:continue
                self.program.observe(observation)
                if not self.program.finished and self.program.actions>=self.request['steps']:
                    self.program.feedback(observation,budget=True)
                if self.program.finished:
                    self.finish_boundary(observation);continue
                action=self.executor.action(self.program,observation,min(self.request['steps']-self.program.actions,400))
                if self.program.finished:
                    self.finish_boundary(observation);continue
                if action is None:raise RuntimeError('An active original phase omitted its command and terminal receipt')
                self.phase=self.program.phases[self.program.phase_index]['name']
                self.program.record_action();self.actions+=1;self.last_action=deepcopy(action)
                return action
            except ExecutionError as error:
                self.holding_reason=type(error).__name__+': '+str(error)
                self.record({'event':'original_public_boundary_stop','reason':self.holding_reason,
                    'public_action_epoch':self.actions,'official_success_claimed':False})
        raise RuntimeError('Original command workflow exceeded its zero-action boundary bound')

    def close(self):
        if self.program is not None:
            cancel=getattr(self.program,'cancel',None)
            if callable(cancel):cancel()
            worker=getattr(self.program,'worker',None)
            if worker is not None and worker.is_alive():worker.join(timeout=2.)

    def result(self):
        return {'task':self.mode,'actions':self.actions,'boundaries':self.boundaries,
            'phase':self.phase,'original_workflow_commands_complete':self.finished_workflow,
            'public_boundary_hold_reason':self.holding_reason,'events':self.events,
            'physical_finger_or_contact_available':False,'model_calls':0,'official_success_claimed':False}
