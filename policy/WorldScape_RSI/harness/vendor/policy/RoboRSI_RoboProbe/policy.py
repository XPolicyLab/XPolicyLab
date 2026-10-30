"""RoboProbe EEF agent with a shared library of observation-grounded capabilities."""
import json,math,time
from pathlib import Path
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import CapabilityFailure,InfrastructureFailure,MotionOutcome,AzureAgentClient,_compact_message_history_in_place
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action,ActionChunk
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.policy import EefAgentPolicy
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF import pose
from .contracts import RULE,public_observation,strict_settings
from .library import SkillLibrary
from . import tools as skill_tools

EMBODIMENT="""Two ARX X5 arms with parallel grippers. Use world metres and wxyz quaternions
in call_tool. position is the world grasp point; quaternion_wxyz is flange
orientation, never Euler angles. Raw *_ee_pose XYZ is the flange origin.
The current tool_poses give matching position/quaternion_wxyz pairs derived from the
official flange pose using RoboProbe's fixed robot geometry. Never add an offset
twice. Gripper 0 closes and 1 opens; its observation is a command target.
Use current RGB to locate objects and support surfaces. There are no supplied
scene heights, object coordinates, collision maps or hidden scoring thresholds.
Robot-only planning checks robot self-collision; inspect objects and both full
arm corridors before moving. Arrival does not prove grasp or task completion."""

CONTROL_CONTRACT="Every episode must end with both arms at their recorded initial poses with initial empty-gripper openings restored after safe support, release and collision-free withdrawal. This universal requirement applies even if the short environment instruction omits it. Use home_status and return_home; never invent home coordinates or claim unobserved return. Do not send motions after the native environment has terminated. Before every new WAM chunk call assess_control using a capability hypothesis, public evidence and expected next change. WAM arguments remain empty: only the original environment instruction and the checkpoint's declared formatting reach Policy; Agent plans, selected object references and Skill memory do not. Use native control when such missing information or precise orientation/contact constraints are required. Repeated stalled, wrong-direction or uncertain progress requires actual native correction, not switching between WAM variants. After each acknowledged WAM action, the Agent reviews fresh RGB/state before another queued WAM action can execute, and may take over and discard the remaining suffix. A needs_more_motion judgment is allowed only with visible evidence of the necessary approach/transport phase, never as a substitute for checking progress. Read and apply generic geometry/constraint Skills. No benchmark labels or task-specific catalogs are available. This contract overrides older text that postpones review until the end of a chunk or makes returning optional. "

class SkillEefPolicy(EefAgentPolicy):
    def __init__(self,*,action_spec,env,planner,client=None,skills_root=None):
        from dataclasses import replace
        self.library=SkillLibrary(skills_root or Path(__file__).parent/'skills')
        self.extra_images=[];self.last_observation=None
        settings=strict_settings(env)
        if client is None:
            from .provider import build_client
            client=build_client(settings)
        settings.setdefault('L3_INSPECT_BASE_URL',getattr(client,'config',{}).get('url','http://injected-client.invalid/v1'))
        super().__init__(action_spec=replace(action_spec,docs=EMBODIMENT),env=settings,planner=planner,client=client)
        self._max_llm_calls=0
        self._client_config['timeout_s']=None;self._client_config['hard_timeout_s']=0
    def reset(self):
        super().reset();self.extra_images=[];self.last_observation=None;self._model_refusal=None
    @property
    def _max_steps(self):return math.inf
    def _axis_bounds(self,label):
        axis=label.partition('_')[2]
        if axis in pose.POSITION_AXES:return -math.inf,math.inf
        if axis in pose.ANGLE_BOUNDS:return pose.ANGLE_BOUNDS[axis]
        if axis=='gripper':return 0.,1.
        raise ValueError('Unknown target dimension')
    def _bounds_text(self):return 'finite XYZ; canonical orientation angles; gripper in [0,1]. Robot reachability comes from robot-only IK'
    def _build_tools(self,action_spec):
        # Keep RoboProbe's motion generator internal. Models choose a reviewed
        # Tool, so the primary execution Skill cannot be bypassed by move_eef.
        return skill_tools.specs(self.library)
    def _system_message(self):
        return (RULE+'\n'+EMBODIMENT+'\n'+CONTROL_CONTRACT+'\nUse one advertised tool call at a time. Read relevant Skills and Tool interfaces. '
                'There is no harness wall-clock, model-call, repair-count or per-motion duration limit. '
                'Only native environment actions consume the task budget. Do not declare done or request resets. '
                'The worldscape-wam-memory and worldscape-wam-nomemory Skills invoke the two checkpoint-backed World Action Models inside this Agent. Read its interface with read_tool, then call_tool with its id and empty arguments. Apply the preloaded policy-applicability Skill before choosing WAM or native Tools. WAM is optional, not the default; use a specific capability hypothesis and verify observable progress. The WAM interface receives the original environment instruction, not an Agent-authored subgoal or its conversation memory. Choose from current RGB, instruction, state and feedback. Every chunk returns control to this Agent. '
                'Apply the preloaded behavioral-checkpoints contract: before the first action and after a material stage change, read the relevant generic behavior Skill by matching visible geometry, constraints and feedback, not task labels. Preserve its concrete pose, contact, visibility, release and recovery conditions. Reading a Skill does not transmit it to WAM. Select and compose relevant Skills by reasoning about the current instruction, visible relationships, uncertainty and execution feedback. Descriptions indicate possible applicability; they are not automatic triggers or mandatory plans. Read a relevant Skill before applying it, and reassess after each observation. If none fits, use the common tools from current evidence. Honor the public objective and constraints together with universal safe return of both arms. For move/grip/grip_both/move_both/visual_align read robot-action-timing-tool then use call_tool with that id. No direct move_eef entry is available. Backend coordinates and arguments come from the current Tool interface, never legacy names. Movement is executed by official RoboProbe joint interpolation.\n'
                +'Memory rules:\n'+self._memory_rules()+'\nInitial Skills:\n'+self.library.initial()+'\nGeneral Skill catalog:\n'+json.dumps(self.library.catalog('skill'),ensure_ascii=False))
    def _memory_rules(self):
        root = Path(__file__).parents[3] / 'memory'
        return '\n\n'.join(f.read_text(encoding='utf8') for f in sorted(root.glob('*.md')))
    def _state_block(self,observation):
        public=skill_tools.execute(self,'view_env_state',{},public_observation(observation))
        return ('Instruction: '+(observation.instruction or '')+'\n'
                'Current tool_poses: position is world grasp-point XYZ in metres; '
                'quaternion_wxyz is world flange orientation, not Euler angles. '
                'These are current poses, not requested actions. Save these pairs '
                'for mandatory return of both arms in every episode. The saved initial poses are available through home_status/return_home. Raw state '
                '*_ee_pose XYZ is flange origin; do not add an offset to tool_poses.\n'
                +json.dumps({'tool_poses':public['tool_poses'],'state':public['state']},ensure_ascii=False))
    def _arrival_text(self,observation):
        return super()._arrival_text(observation).replace('previous move_eef target','previous Tool target')
    def prepare(self,observation):return super().prepare(public_observation(observation))
    def native_move(self,targets,observation):
        self.last_observation=observation
        return super()._handle_motion('move_eef',{'targets':targets,'note':'Reviewed Skill translation using current official observation'},observation)
    def _is_motion_tool(self,name):return name in {t['function']['name'] for t in self._tools}
    def _handle_motion(self,name,arguments,observation):
        try:
            self.last_observation=observation
            result=skill_tools.execute(self,name,arguments,observation)
            if isinstance(result,MotionOutcome):return result
            return MotionOutcome(chunk=None,tool_result=json.dumps(result,ensure_ascii=False),repairable=False)
        except (ValueError,KeyError,TypeError,OverflowError,TimeoutError,OSError) as error:
            return MotionOutcome(chunk=None,tool_result=json.dumps({'error':str(error),'executed_steps':0},ensure_ascii=False),repairable=True)
    def act(self,observation):
        # RoboProbe's conversation protocol and motion expansion, with its optional
        # call/repair caps removed. Tool calls cannot invoke native stop/reset.
        observation=public_observation(observation);self.last_observation=observation
        if self._model_refusal is not None:return self._refusal_hold(observation)
        self.prepare(observation);self._messages.append(self._observation_message(observation));repairs=0;format_repair_start=None
        while True:
            # Read-only tools and format repairs may continue without a native
            # action. Persist their completed records before the next request.
            checkpoint=getattr(self,'trace_checkpoint',None)
            if checkpoint is not None:checkpoint()
            if not self._keep_all_images:_compact_message_history_in_place(self._messages,image_horizon=self._image_horizon)
            started=time.monotonic()
            try:response=self._client.complete(self._messages,self._tools)
            except CapabilityFailure as error:
                # A model refusal is not an auth/network failure. Keep native
                # scoring and termination: hold the robot through normal actions.
                response={'choices':[{'message':{'content':None,'refusal':str(error),'tool_calls':[]}}]}
            except InfrastructureFailure:raise
            except Exception as error:raise InfrastructureFailure('Inference backend failed: '+type(error).__name__) from None
            if not isinstance(response,dict) or not isinstance(response.get('choices'),list) or not response['choices'] or not isinstance(response['choices'][0].get('message'),dict):raise InfrastructureFailure('Malformed inference response')
            self._calls+=1
            record=self._record_turn(response,policy_step=observation.step,repair_attempt=repairs,latency_s=time.monotonic()-started)
            message=response['choices'][0]['message'];content=message.get('content')
            if message.get('refusal'):
                self._model_refusal=str(message['refusal']);record['model_refusal']=self._model_refusal
                record['accepted']=False;return self._refusal_hold(observation)
            raw_calls=message.get('tool_calls') or []
            valid=isinstance(raw_calls,list) and all(isinstance(c,dict) and isinstance(c.get('id'),str) and isinstance(c.get('function'),dict) and isinstance(c['function'].get('name'),str) and isinstance(c['function'].get('arguments'),str) for c in raw_calls)
            # RoboProbe merges parallel move_eef calls before dispatch. Validate
            # object shape first: JSON null/arrays otherwise crash that merger.
            if valid:
                try:valid=all(isinstance(json.loads(c['function']['arguments']),dict) for c in raw_calls)
                except (ValueError,TypeError):valid=False
            calls=self._normalize_tool_calls(raw_calls) if valid else []
            if len(calls)!=1:
                record['validation_error']='Exactly one advertised tool call is required'
                # Failed formatting did not change the scene. Preserve every raw
                # response in the transcript, but do not teach the next request
                # dozens of malformed examples or keep growing unchanged context.
                if format_repair_start is None:format_repair_start=len(self._messages)
                else:pass  # Preserve completed API history during repeated format repairs.
                self._messages.append({'role':'user','content':'Response format invalid: exactly one advertised function call with JSON object arguments is required. No action was executed; the current observation is unchanged. Use the tool channel, not JSON in prose.'});repairs+=1;continue
            if format_repair_start is not None:
                format_repair_start=None  # Prior repair turns remain in the sticky conversation.
            call=calls[0];name=call['function']['name'];record['tool']=name
            try:
                arguments=json.loads(call['function'].get('arguments') or '{}')
                if not isinstance(arguments,dict):raise ValueError('Tool arguments must be an object')
            except (ValueError,TypeError) as error:
                self._append_tool_repair(call,content,str(error));record['validation_error']=str(error);repairs+=1;continue
            record['arguments']=arguments
            self._messages.append({'role':'assistant','content':content,'tool_calls':calls})
            outcome=(self._handle_motion(name,arguments,observation) if self._is_motion_tool(name) else
                     MotionOutcome(chunk=None,tool_result='Unavailable tool; use only the advertised list. Native evaluator alone ends the episode.',repairable=True))
            record['tool_result']=outcome.tool_result
            self._messages.append(self._tool_result(call['id'],outcome.tool_result))
            if self.extra_images:
                self._messages.append({'role':'user','content':[{'type':'text','text':'Requested crop of the current official RGB:'},*self.extra_images]});self.extra_images=[]
            if outcome.chunk is not None:record['accepted']=True;return outcome.chunk
            if outcome.repairable:record['validation_error']=outcome.tool_result;repairs+=1
            else:record['accepted']=True
    def review_wam_execution(self,observation,control_status):
        """Read fresh public feedback before exposing any remaining WAM action."""
        observation=public_observation(observation)
        self._messages.append(self._observation_message(observation))
        self._messages.append({'role':'user','content':'A WAM action was acknowledged; its queued suffix has not executed. Compare current RGB/state with the previous frame and expected change. Choose continue only if this local phase remains appropriate; choose takeover for wrong target/direction, contact/visibility risk, uncertain state, or completion requiring safe return. Do not infer visual progress from a planned endpoint. Control status: '+json.dumps(control_status,ensure_ascii=False)})
        from XPolicyLab.policy.WorldScape_RSI.harness.control_guard import policy_review_tool
        tool=policy_review_tool()
        started=time.monotonic()
        try: response=self._client.complete(self._messages,[tool])
        except Exception as error: raise InfrastructureFailure('Policy review request failed; no queued suffix released: '+type(error).__name__) from None
        self._calls+=1
        record=self._record_turn(response,policy_step=observation.step,repair_attempt=0,latency_s=time.monotonic()-started)
        message=response.get('choices',[{}])[0].get('message',{}) if isinstance(response,dict) else {}
        calls=message.get('tool_calls') or []
        fallback={'decision':'takeover','progress':'uncertain','evidence':'Agent review response did not provide a valid evidence-grounded decision'}
        try:
            if len(calls)!=1 or calls[0]['function']['name']!='review_policy_chunk' or not isinstance(calls[0]['id'],str): raise ValueError('One review call required')
            args=json.loads(calls[0]['function']['arguments'])
            if not isinstance(args,dict) or set(args)!={'decision','progress','evidence'} or args['decision'] not in ('continue','takeover') or args['progress'] not in ('advancing','needs_more_motion','stalled','wrong_direction','uncertain') or not isinstance(args['evidence'],str) or not 8<=len(args['evidence'].strip())<=1600: raise ValueError('Invalid review')
        except (ValueError,TypeError,KeyError):
            record['accepted']=False; record['validation_error']='Malformed review; suffix withheld'
            self._messages.append({'role':'user','content':'Review invalid. No additional queued action executed. Native control is required from this observation.'})
            return fallback
        self._messages.append({'role':'assistant','content':message.get('content'),'tool_calls':calls})
        self._messages.append(self._tool_result(calls[0]['id'],'Review recorded; the execution boundary enforces the decision before releasing another action.'))
        record.update(tool='review_policy_chunk',arguments=args,accepted=True)
        return args

    def _refusal_hold(self,observation):
        steps=1 # No internal budget is passed to the policy.
        return ActionChunk(actions=[Action(data=self._held_action_data(observation),meta={}) for _ in range(steps)],control_hz=self.action_spec.control_hz,meta={'model_refusal':self._model_refusal})
    def execution_feedback(self, observation, played, planned, interruption):
        self.last_observation = observation
        summary = {'executed_actions':played, 'planned_actions':planned,
                   'interrupted':bool(interruption), 'reasons':interruption or []}
        if self._transcript:
            self._transcript[-1]['execution'] = summary
        if interruption:
            # Planning latches future gripper goals. A cancelled suffix must not
            # close/open the gripper on the next unrelated action.
            self._last_targets = {}
            self._gripper_goals = {arm:float(observation.state[arm+'_ee_joint_state'][0]) for arm in ('left','right')}
            self._messages.append({'role':'user', 'content':
                'Execution monitor interrupted the previous motion from official public robot state: '
                + ', '.join(interruption) + '. Remaining queued actions were discarded. '
                'The next observation is current. Reassess the motion from RGB and robot state; '
                'the planned endpoint and gripper phase were not confirmed executed.'})
    def audit_config(self):
        result=super().audit_config()
        result['adapter']='RoboRSI_RoboProbe-v1.8.2-execution-monitored'
        if hasattr(self._client,'config'):
            result['inference_transport']={k:self._client.config.get(k) for k in ('kind','model','effort')}
            result['inference_transport']['external_tool_calls_allowed']=False
        for key in ('low','high'):result['embodiment'][key]=[v if math.isfinite(v) else None for v in result['embodiment'][key]]
        result['policy_config'].update(max_llm_calls=None,max_duration_s=None,repair_attempt_limit=None,episode_wall_timeout_s=None,depth='off',learned_action_service=False)
        result['skills']={'selection':'agent_from_current_context','catalog':self.library.catalog('skill'),'tools':self.library.catalog('tool')}
        result['observation_contract']='official_rgb_and_six_robot_state_channels_only'
        result['model_refusal']=self._model_refusal
        return result
