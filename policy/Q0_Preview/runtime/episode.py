"""Policy-side planning lifecycle. File IPC stays entirely on the policy host."""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from ..controller.backend import array
from ..controller.executor import Executor, Replan, Ended, StopExecution, StepBudgetExhausted
from .protocol import validate_plan, validate_decision, ContractError, MAX_GROUPS
from .task_context import task_context, task_card
from .termination import validate_termination
from .io_utils import atomic, read_json, json_default
from ..harness.observations import publish_manifest
POLICY = Path(__file__).resolve().parents[1]
CAMERAS = ('cam_head','cam_left_wrist','cam_right_wrist')
class Deadline(StopExecution):
    pass

class Episode:
    def __init__(self, backend, root, config):
        self.b, self.root = backend, Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        self.run_id = self.root.name
        self.task = str(config['task_name'])
        self.context = task_context(self.task)
        self.started = time.monotonic()
        self.started_unix = time.time()
        self.wall_limit = float(config.get('wall_seconds',10800))
        self.turn_limit = float(config.get('turn_seconds',1800))
        self.check_limit = float(config.get('checkpoint_seconds',60))
        self.max_plans = int(config.get('max_plans',100))
        self.counter = self.plan_no = 0
        self.last_sim = self.b.sim_time()
        self.last_wall = time.monotonic()
        self.last_env_step = self.b.steps
        self.notice = 'Fresh observation is authoritative. Gripper state is a command echo, not a measurement.'
        self.log = (self.root/'execution.jsonl').open('a',buffering=1)
        self.executor = Executor(self.b,self.trace,self.check_deadline)
        atomic(self.root/'episode.json', {'policy':'Q0_Preview','run_id':self.run_id,'task':self.task,
            'policy':POLICY.name,'interface':'standard_remote_joint','wall_limit_s':self.wall_limit,
            'max_groups':MAX_GROUPS, 'checkpoint_limit_s':self.check_limit,'max_plans':self.max_plans,'wall_started_unix_s':self.started_unix,
            'wall_deadline_unix_s':self.started_unix+self.wall_limit,'turn_limit_s':self.turn_limit,
            'control_step_limit':self.b.max_control_steps,
            'control_step_limit_source':'audited static task definition'})
        self.b.on_step_acknowledged = self.record_progress
        self.record_progress()

    def trace(self, event):
        self.log.write(json.dumps({'run_id': self.run_id, 'time': time.time(), **event}, default=json_default)+'\n')

    def record_progress(self):
        # Called after EVERY acknowledged action, including grippers, convergence
        # and post-planning holds. An issued terminal action is not an acknowledgement.
        progress = {'run_id': self.run_id, 'task': self.task,
                    'updated_unix': time.time(), 'phase': self.b.step_phase,
                    **self.b.budget_state()}
        self.trace({'event': 'action_acknowledged' if self.b.steps else 'episode_progress', **progress})
        atomic(self.root/'PROGRESS.json', progress)

    def write_end(self, reason, final):
        atomic(self.root/'END.json', {'run_id':self.run_id,'task':self.task,'reason':reason,
            'final':final, 'control_steps':self.b.steps, **self.b.budget_state(),
            'groups_completed':self.executor.group_count,
            'wall_seconds':time.monotonic()-self.started,'official_result':'not available to policy'})

    def remaining(self):
        return self.wall_limit-(time.monotonic()-self.started)

    def check_deadline(self):
        if self.b.ended():
            raise Ended()
        if self.b.remaining_steps() == 0:
            raise StepBudgetExhausted('task_control_step_limit')
        if self.remaining() <= 0:
            raise Deadline('episode_wall_limit')
        if (self.root/'STOP').exists():
            raise Deadline('external_stop')

    def publish(self, kind, **extra):
        from XPolicyLab.utils.process_data import encode_image_bit, decode_image_bit
        import numpy as np
        self.check_deadline()
        obs, vision = self.b.observation()
        if self.executor.last_feedback is not None:
            # Measured diagnostics only; never a possession or official-success verdict.
            obs['last_execution']=self.executor.last_feedback
        now = time.monotonic()
        obs.update(delta_sim_time_s=obs['sim_time_s']-self.last_sim,
                   delta_wall_time_s=now-self.last_wall,
                   delta_control_steps=obs['control_steps_used']-self.last_env_step,
                   wall_seconds_remaining=max(0, self.remaining()),
                   planning_rounds_remaining=max(0, self.max_plans-self.plan_no),
                   task_requirements=self.context,
                   completion_requirements={'return_home_required': task_card(self.task)['return_home_required']},
                   state_notice=self.notice)
        self.last_sim, self.last_wall, self.last_env_step = obs['sim_time_s'], now, obs['control_steps_used']
        obs_id = f'obs_{self.counter:05d}'
        obs.update(observation_id=obs_id, run_id=self.run_id, task=self.task,
                   capture_unix_s=time.time(), observation_kind=kind)
        self.counter += 1
        folder = self.root/'observations'/obs_id
        folder.mkdir(parents=True)
        for camera in CAMERAS:
            rgb = array(vision[camera]['color'])[..., :3].astype('uint8')
            bits = encode_image_bit(rgb, quality=95)
            (folder/(camera+'.jpg')).write_bytes(bits)
            np.save(folder/(camera+'.rgb.npy'), decode_image_bit(bits), allow_pickle=False)
        atomic(folder/'observation.json', obs)
        seconds = min(self.remaining(), self.check_limit if kind == 'checkpoint' else self.turn_limit)
        req = {'run_id': self.run_id, 'task': self.task, 'observation_id': obs_id,
               'request_id': uuid.uuid4().hex, 'kind': kind, 'deadline': time.time()+seconds, **extra}
        atomic(folder/'request.json', req)
        publish_manifest(folder, req, obs)
        atomic(self.root/'READY.json', req)
        self.trace({'event': 'observation', **req})
        return folder, req

    def request(self, kind, **extra):
        folder, req = self.publish(kind, **extra)
        while time.time() < req['deadline']:
            self.check_deadline()
            reply = read_json(folder/'reply.json')
            if reply:
                if all(reply.get(k) == req[k] for k in ('run_id', 'request_id', 'observation_id')):
                    self.trace({'event': 'reply_accepted', 'request_id': req['request_id'], 'kind': kind})
                    return reply.get('payload')
                self.trace({'event': 'stale_reply_rejected', 'request_id': req['request_id']})
                raise ContractError('reply attribution mismatch')
            time.sleep(.1)
        self.trace({'event': 'planning_window_elapsed', 'request_id': req['request_id']})
        if kind == 'checkpoint':
            return {'decision': 'REPLAN', 'reason': 'checkpoint timeout; pending group discarded'}
        return {'tool': 'harness_replan', 'reason': 'planning_observation_wait_limit'}


    def checkpoint(self, index, group, remaining):
        # No action/physics step while waiting: images are from the final acknowledged action.
        self.notice = 'First group finished. Review its visible outcome before releasing the pending second group.'
        try:
            reply = self.request('checkpoint', plan_group_index=index,
                                 completed_group=group, expectation=group['expectation'], remaining_groups=remaining)
            if isinstance(reply, dict) and reply.get('tool') == 'harness_stop':
                raise Deadline(str(reply.get('reason', 'harness_stop')))
            return validate_decision(reply)
        except (ContractError, TypeError, KeyError) as exc:
            self.trace({'event': 'checkpoint_rejected', 'reason': str(exc)})
            return {'decision': 'REPLAN', 'reason': ('invalid checkpoint reply: '+str(exc))[:300]}

    def run(self):
        reason = 'planning_round_limit'
        no_progress_windows = 0
        try:
            for plan_no in range(self.max_plans):
                self.plan_no = plan_no
                self.check_deadline()
                steps_before = self.b.steps
                try:
                    reply = self.request('plan')
                    if isinstance(reply,dict) and reply.get('tool') == 'harness_stop':
                        reason=reply.get('reason','harness_stop')
                        break
                    if isinstance(reply,dict) and reply.get('tool')=='harness_replan':
                        self.notice = 'No motion executed. Replan from the current observation: '+str(reply.get('reason'))
                    else:
                        if isinstance(reply,dict) and reply.get('tool') == 'end_attempt':
                            try:
                                validate_termination({k:v for k,v in reply.items() if k != 'tool'}, {})
                            except ValueError as exc:
                                raise ContractError(str(exc)) from exc
                            reason = 'abandoned'
                            break
                        validate_plan(reply)
                        self.trace({'event':'plan','plan_number':plan_no,'payload':reply})
                        outcome=self.executor.execute(reply, self.checkpoint)
                        self.notice=('Action interrupted: '+self.executor.last_replan+'. Recompute from current images and robot state.'
                            if outcome=='REPLAN' else 'Previous action plan finished; verify results from this fresh observation.')
                except (ContractError,Replan) as exc:
                    self.notice='Plan rejected: '+str(exc)+'. Use current observations; old targets will not resume.'
                    self.trace({'event':'plan_rejected','reason':str(exc)})
                # Protocol acceptance and zero-step COMPLETE are not execution progress.
                # Count every main planning window, including rejected/expired replies.
                if self.b.steps > steps_before:
                    no_progress_windows = 0
                else:
                    no_progress_windows += 1
                    self.trace({'event':'planning_no_progress','plan_number':plan_no,
                                'control_steps':self.b.steps,'consecutive_windows':no_progress_windows})

        except StepBudgetExhausted:
            reason='task_control_step_limit'
        except Ended:
            reason='client_closed'
        except Deadline as exc:
            reason=str(exc)
        except BaseException as exc:
            reason='policy_error'
            self.b.fail(exc)
            raise
        finally:
            # END stops the API harness immediately. Keep recording holds until the
            # official client ends/resets; then replace this with the final totals.
            self.write_end(reason, final=False)
            try:
                if reason != 'policy_error':
                    self.hold_until_closed()
            finally:
                self.write_end(reason, final=True)
                self.b.on_step_acknowledged = lambda: None
                self.log.close()

    def hold_until_closed(self):
        try:
            self.b.step_phase = 'post_planning_hold'
            while not self.b.ended() and self.b.remaining_steps() > 0:
                self.b.step(self.b.joints(),self.b.gripper_commands())
        except (Ended, StepBudgetExhausted):
            pass
