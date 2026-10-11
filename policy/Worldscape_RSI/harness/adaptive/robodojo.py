"""Real RoboDojo adapter; reuses native recording, action grammar and WAM wire protocol."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from .contracts import Evidence
from .engine import Engine
from .jev import Jev
from .library import Library, read, save_candidate
from .program_adapter import ProgramAdapter

class Adapter(ProgramAdapter):
    def __init__(self, env, rpc, config, capture, session, output, settings, audit):
        from .. import deploy, main_route
        self.env, self.rpc, self.config, self.capture = env, rpc, config, capture
        self.session, self.output, self.cfg, self.audit = session, Path(output), settings, audit
        self.deadline = time.monotonic()+min(float(config.get('episode_budget_s',9000)),float(settings.get('development_wall_cap_s',9000)))
        self.stack = ExitStack(); self.wam = None; self.history = None; self.pending = None
        self.revision = self.astra_calls = 0; self.facts = {}; self.motion = {}; self.refs = []
        self.last_report = {}; self.commanded = {}; self.local_end = False; self.recent=[]
        self.instruction = env.get_obs().get('instruction')
        self.available = ['astra','cap'] + (['wam'] if settings.get('wam_url') else [])
        self.guard_checks = 0
        self.shadow = None
        self.init_programs()
        if settings.get('force_astra'):
            self.available = ['astra','cap']
        # Mem is deliberately advertised only when a separate, tested provider is installed.
        self.session['home_poses'] = {arm:[*p,*q] for arm,(p,q) in main_route.live_poses(env).items()}
        self.session['home_joints'] = main_route._read_joints(env, SimpleNamespace(joint_module=deploy._joint_module()))

    def stopped(self):
        return self.local_end or self.env.is_episode_end() or int(self.env.take_action_cnt[0]) >= int(self.env.step_lim) or time.monotonic() >= self.deadline or self.astra_calls >= self.cfg['max_astra_calls']

    def evidence(self):
        ended = self.stopped(); steps = int(self.env.take_action_cnt[0])
        return Evidence(self.revision, steps, max(0,int(self.env.step_lim)-steps),
            dict(self.facts),dict(self.motion),list(self.refs),
            bool(self.env.end_flag[0]) and bool(self.env.success[0]),ended)

    def finalize(self):
        from .. import deploy
        # Native success starts True as a running sentinel. Only end_flag confirms settlement.
        if not self.env.is_episode_end():
            deploy._end_unsuccessful(self.env)
        return self.evidence()

    def call(self, mode, context):
        from .. import deploy, observation_contract
        if self.stopped() or self.astra_calls >= self.cfg['max_astra_calls']:
            raise RuntimeError('Astra/episode budget exhausted')
        obs = observation_contract.filter_observation(self.env.get_obs())
        obs['policy_budget_s'] = max(0,self.deadline-time.monotonic())
        obs['gripper_real'] = deploy._gripper_real(self.env)
        obs['gripper_closed_floor'] = deploy._gripper_closed_floor(self.env)
        if deploy._wants_calibration(self.config):
            obs['calibration'] = deploy._camera_calibration(self.env,tuple(self.config.get('cameras') or ['cam_head']))
        self.astra_calls += 1
        steps = int(self.env.take_action_cnt[0])
        context = dict(context, runtime_budget={'steps_used':steps,
            'native_step_limit':int(self.env.step_lim),
            'remaining_steps':max(0,int(self.env.step_lim)-steps)})
        result = self.rpc.call(func_name='adaptive_call',obs={'mode':mode,'observation':obs,'context':context})
        self.audit({'kind':'astra_result','call':self.astra_calls,'mode':mode,'revision':self.revision,'result':result})
        return result

    def plan(self, skills, evidence):
        from .code_skills import enabled
        skills = self.library.catalog(self.env.task_name,self.instruction or '',24)
        return self.call('plan',{'skills':skills,'available_executors':self.available,
            'task_program':self.library.task_programs.get(self.env.task_name),
            'programs':[p for p in self.library.programs if p['task']==self.env.task_name],
            'evidence':evidence.decision_state(), 'constraints':'Prefer original task instruction for WAM. Use explicit observable stage predicates. No evaluator state.'})

    def bind_code_subgoal(self, stage, evidence, recovery=False):
        from .code_skills import bind
        skills = self.library.catalog(self.env.task_name,str(stage.get('goal','')))
        if not skills or 'cap' not in self.available: return None, evidence
        result = self.call('bind_skill', {'stage':stage,'skills':skills,
            'recovery':recovery,'evidence':evidence.decision_state(),
            'instructions':'Choose one useful bounded subgoal, not a whole-stage claim. Source traces are historical hypotheses, not current facts. Rebind every geometric parameter from current images; never replay source coordinates. Return null if no capability fits. A successful source episode does not establish this wrapper is validated.'})
        name=result['skill_id']
        if name is None:return None, self.evidence()
        skill=skills[name];bind(skill,result['bindings'])
        facts={f['name']:f['value'] for f in result['facts'] if f['name'] in skill['preconditions']}
        self.facts=facts;self.revision+=1;self.refs=[f'astra_call:{self.astra_calls}:skill_binding']
        refreshed=self.evidence()
        if not refreshed.satisfies(skill['preconditions']):return None,refreshed
        sub=dict(stage,executor='cap',skill=name,goal=skill['description'],completion=skill['postconditions'],bindings=result['bindings'])
        self.audit({'kind':'code_subgoal_bound','skill':name,'recovery':recovery,'reason':result['reason'],
                    'revision':refreshed.revision,'bindings':result['bindings']})
        return sub,refreshed

    def inspect(self, stage, reason):
        requested = set((stage or {}).get('completion',[])) | {'release_authorized','stop_recommended','recovery_complete','wam_applicable','astra_applicable','progressing','object_held'}
        result = self.call('inspect',{'stage':stage,'reason':reason,'requested_facts':sorted(requested),
            'recent_motion':self.motion})
        facts = {}
        for fact in result['facts']:
            # Do not let model output inject score/seed/layout or arbitrary privileged fields.
            if fact['name'] in requested:
                facts[fact['name']] = fact['value']
        self.facts = facts; self.revision += 1
        self.refs = [f'astra_call:{self.astra_calls}:observation_revision:{self.revision}']
        return self.evidence()

    def cancel_pending(self):
        self.pending = None

    def reset_context(self, executor, reason):
        self.history = None
        self.audit({'kind':'context_reset','executor':executor,'reason':reason,'method':'real_current_observation'})

    def connect_wam(self):
        import fcntl
        from ..wam.client import RoboDojoClient
        if self.wam is None:
            path = Path(self.cfg['wam_lock']); path.parent.mkdir(parents=True,exist_ok=True)
            lock = self.stack.enter_context(path.open('a+'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.wam = self.stack.enter_context(RoboDojoClient(self.cfg['wam_url'],self.cfg['wam_timeout_s']))

    def propose(self, executor, stage, evidence, recovery=None):
        if executor not in self.available:
            raise ValueError('Executor excluded by frozen experiment configuration')
        if executor == 'wam':
            import numpy as np
            from ..wam.robodojo_adapter import capture_frame, build_inputs
            self.connect_wam()
            current = capture_frame(self.env.get_obs())
            frames = self.history if self.history is not None else [current]
            if not np.array_equal(frames[-1].state,current.state) or frames[-1].instruction != current.instruction:
                raise RuntimeError('WAM context does not match real state')
            images,state = build_inputs(frames)
            reply = (self.wam.predict_current(images,state,instruction=current.instruction,
                      task=None,seed=0,source_color_order='rgb') if self.history is None else self.wam.predict(images,state))
            self.history = None  # server prediction advanced; reset unless whole chunk executes
            self.pending = {'actions':np.asarray(reply['result']['action']).tolist()}
            from .chunk_gate import release_candidate
            shadow = release_candidate(self.pending['actions'], current.state.tolist())
            self.shadow = shadow
            self.audit({'kind':'chunk_gate_shadow', 'stage':stage.get('id'),
                        'revision':evidence.revision, 'steps':evidence.steps, **shadow})
            return self.pending
        if executor == 'cap':
            from .. import open_restore, motion
            skill = self.library.skills[recovery or stage['skill']]
            from .code_skills import enabled, bind
            if not enabled(skill,self.cfg.get('experimental_cap_skills',[])):
                raise ValueError('CaP skill is not enabled for this frozen experiment')
            if not evidence.satisfies(skill['preconditions']):
                raise ValueError('CaP prerequisites not established')
            if skill.get('implementation') == 'command_template':
                bindings = {k:v['value'] for k,v in stage.get('bindings',{}).items()}
                lines = [s.format_map(bindings) for s in skill['commands']]
            else:
                lines = bind(skill,stage.get('bindings',{}))
            grammar = open_restore.module(self.env.task_name,'commands')
            commands,errors = grammar.parse_command_list(lines)
            if errors or not commands or any(c.kind=='done' for c in commands):
                raise ValueError('Invalid CaP commands')
            self.pending = {'envelope':motion.sequence([c.to_dict() for c in commands], {'skill':recovery or stage['skill']}, skill.get('options',{}))}
            return self.pending
        if executor != 'astra':
            raise ValueError('Provider not installed: '+executor)
        self.pending = self.call('action',{'stage':stage,'recovery':recovery,
            'evidence':evidence.decision_state(),'last_execution':self.last_report})
        return self.pending

    def guard_proposal(self, proposal, stage, evidence):
        from .contracts import digest
        arms = self.cfg.get('release_guard_arms', [])
        event = self.shadow or {}
        relevant = [x for x in event.get('events', []) if x['arm'] in arms]
        if not relevant:
            return evidence, True
        if self.guard_checks >= self.cfg.get('max_release_checks', 4):
            self.cancel_pending(); self.local_end = True
            self.audit({'kind':'development_cost_stop','reason':'release_check_cap'})
            return self.evidence(), False
        self.guard_checks += 1
        expected = event['proposal_sha256']
        guarded = dict(stage, proposed_release=relevant,
                       release_requirement=self.cfg['release_requirement'])
        refreshed = self.inspect(guarded, 'Before executing proposed gripper opening, judge release_authorized using CURRENT RGB and the stated release requirement. Legitimate handoff can be allowed when visibly supported. Unknown is null, not proof of failure. Do not use future frames.')
        allowed = refreshed.facts.get('release_authorized') is True
        if proposal is not self.pending or digest(proposal['actions']) != expected:
            raise RuntimeError('Proposal changed during release inspection')
        self.audit({'kind':'release_guard','proposal_sha256':expected,'revision':refreshed.revision,
                    'allowed':allowed,'facts':refreshed.facts,'check':self.guard_checks})
        if not allowed:
            self.cancel_pending(); self.reset_context('wam','release_not_authorized')
            self.motion = dict(self.motion, hard_error=True, guard_blocked=True,
                               guard_reason='release_not_authorized_not_proven_physical_failure')
            refreshed = self.evidence()
        return refreshed, allowed

    def validate(self, proposal, executor, remaining):
        if proposal is not self.pending or remaining <= 0:
            raise RuntimeError('Expired proposal')
        if executor == 'wam':
            import numpy as np
            a = np.asarray(proposal['actions'])
            if a.shape != (24,14) or not np.isfinite(a).all():
                raise ValueError('Invalid WAM action block')
            from ..wam.robodojo_adapter import action_to_robodojo
            for action in a:
                action_to_robodojo(action)
        elif proposal['envelope'].get('kind') not in ('sequence','stop'):
            raise ValueError('Expected bounded command sequence')

    def execute(self, proposal, executor):
        from .. import deploy, gripper_settle
        import numpy as np
        from ..wam.robodojo_adapter import capture_frame
        before_frame = capture_frame(self.env.get_obs())
        start = int(self.env.take_action_cnt[0])
        report = {}
        if proposal.get('envelope',{}).get('kind') == 'stop':
            self.local_end=True;self.pending=None
            return {'executed':0,'stop':'astra_policy_stop'}
        if executor == 'wam':
            from ..wam.robodojo_adapter import capture_frame, action_to_robodojo
            observed=[]
            for action in proposal['actions']:
                if self.stopped(): break
                before = int(self.env.take_action_cnt[0])
                self.env.take_action(action_to_robodojo(action))
                if int(self.env.take_action_cnt[0])-before != 1:
                    raise RuntimeError('Unexpected native control step count')
                observed.append(capture_frame(self.env.get_obs()))
            if len(observed)==24:self.history=observed
            report={'executed':len(observed),'partial':len(observed)!=24}
        else:
            self.commanded = gripper_settle.action_targets(self.env,self.env.get_obs()['state'])
            report, halt = deploy._run_sequence(self.env,proposal['envelope'],self.config,self.capture,
                self.deadline,self.commanded,self.session,gripper_settle)
            report['halt']=halt
        self.pending=None;self.revision+=1;self.facts={}
        if hasattr(self,'object_state'):
            self.object_state.invalidate([], 'new physical action; relationship evidence requires renewal')
        self.motion={'executed_steps':int(self.env.take_action_cnt[0])-start,
                     'hard_error':bool(report.get('ik_failed') or report.get('halt')),
                     'command_failures':[r.get('kind','error') for r in report.get('command_results',[]) if r.get('ok') is False]}
        self.motion['hard_error'] |= bool(self.motion['command_failures'])
        after_frame = capture_frame(self.env.get_obs())
        self.motion['joint_delta_l2'] = float(np.linalg.norm(after_frame.state-before_frame.state))
        self.motion['image_change_mean_abs'] = {
            name:float(np.mean(np.abs(after_frame.images[name][::16,::16].astype(float)-image[::16,::16].astype(float))))
            for name,image in before_frame.images.items()}
        self.motion['interpretation'] = 'Motion/image change are diagnostics, not proof of grasp, progress or success.'
        self.last_report=report
        self.recent.append({'revision':self.revision,'executor':executor,'motion':dict(self.motion)})
        self.recent=self.recent[-12:]
        self.motion['recent_execution'] = list(self.recent)
        if executor != 'wam':
            feedback=dict(report,steps_used=int(self.env.take_action_cnt[0]),task_name=self.env.task_name)
            self.rpc.call(func_name='report_execution',obs=feedback)
        self.capture.decisions+=1
        return report

def execute(env,rpc,config,capture,output,session):
    from .. import deploy
    root=Path(__file__).resolve().parents[2]
    cfg=read(root/'configs/adaptive/runtime.json')
    cfg.update(cfg.get('task_experiments', {}).get(env.task_name, {}))
    if cfg.get('trajectory_cap_enabled'):
        registered=read(root/'configs/adaptive/skills.json')
        cfg['experimental_cap_skills']=list(set(cfg.get('experimental_cap_skills',[])) |
            {k for k,s in registered.items() if s.get('implementation')=='trajectory_template'})
    # Runtime transport endpoints are not policy decisions or plan inputs.
    import os
    cfg['wam_url']=os.environ.get('RDA_ADAPTIVE_WAM_URL') or cfg.get('wam_url')
    cfg['wam_lock']=os.environ.get('RDA_ADAPTIVE_WAM_LOCK') or cfg['wam_lock']
    path=Path(output)/'adaptive_events.jsonl'
    def audit(event):
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False,default=str)+'\n')
    adapter=Adapter(env,rpc,config,capture,session,output,cfg,audit)
    library=Library(root/'configs/adaptive',adapter.available,cfg.get('experimental_cap_skills',[]));adapter.library=library
    engine=Engine(library,Jev(cfg['jev_model'],cfg['jev_timeout_s'],audit=audit),adapter,cfg,audit)
    result=None
    try:
        result=engine.run()
        result['astra_calls']=adapter.astra_calls
        # Optional post-episode review is a candidate proposal only; never mutates this library.
        if cfg.get('review_after_episode') and adapter.astra_calls < cfg['max_astra_calls']:
            from .. import observation_contract
            obs=observation_contract.filter_observation(env.get_obs())
            obs['policy_budget_s']=min(120,max(0,adapter.deadline-time.monotonic()))
            if obs['policy_budget_s']>0:
                try:
                    review=rpc.call(func_name='adaptive_call',obs={'mode':'review','observation':obs,
                        'context':{'plan':engine.plan,'outcome':result,'recent_execution':adapter.recent,
                                   'instruction':'Propose evidence-linked experience only. Do not claim unseen frames or independent validation.'}})
                    (Path(output)/'experience_review.json').write_text(json.dumps(review,indent=2))
                    result['astra_calls']+=1
                except Exception as exc:audit({'kind':'review_failed','error':type(exc).__name__})
        if not env.is_episode_end():deploy._end_unsuccessful(env)
        result['success']=bool(env.success[0])
        return result
    finally:
        adapter.cancel_pending();adapter.stack.close()
        if engine.plan is not None:
            save_candidate(Path(output)/'experience_candidates',Path(output).name,engine.plan,path,
                           result or {'termination_reason':'exception','success':False})
