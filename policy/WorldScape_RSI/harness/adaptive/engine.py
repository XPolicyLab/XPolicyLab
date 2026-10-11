"""Synchronous single-writer orchestrator. Inference occurs only at control boundaries."""
from .contracts import Ownership, validate_plan

PROCESS_CHOICES = {
 'continue':'Evidence supports continued execution toward the CURRENT stage goal.',
 'complete':'All CURRENT stage completion conditions are supported by current evidence.',
 'observe':'Evidence missing, stale, occluded or conflicting; do not assume success.',
 'recover':'A listed validated recovery matches the observed failure and prerequisites.',
 'escalate':'Unknown failure or known recovery unsuitable; request Astra diagnosis.',
 'stop':'Irrecoverable failure or exhausted budget; never assert native success.'}

class Engine:
    def __init__(self, library, jev, adapter, config, audit):
        self.library, self.jev, self.io, self.cfg, self.audit = library, jev, adapter, config, audit
        self.ownership = Ownership()
        self.plan = None
        self.calls = self.recoveries = self.observations = self.blocks = 0
        self.last_escalation = -1000
        self.wam_return_block = None
        self.wam_return_disabled = False

    def emit(self, kind, **fields):
        self.audit({'kind':kind, **fields})

    def choose(self, purpose, evidence, choices, extra=None):
        if self.calls >= self.cfg['max_jev_calls']:
            return 'stop' if 'stop' in choices else 'new_plan'
        self.calls += 1
        state = {'observation':evidence.decision_state(), **(extra or {})}
        try:
            answer = self.jev.choose(purpose, state, choices)
        except Exception as exc:
            self.emit('jev_unavailable', error=type(exc).__name__)
            return 'observe' if 'observe' in choices else 'new_plan'
        if answer['confidence'] < self.cfg['min_confidence']:
            return 'observe' if 'observe' in choices else 'new_plan'
        return answer['choice']

    def plan_episode(self, evidence):
        candidates = self.library.candidates(evidence)
        if candidates:
            choices = {'new_plan':'None covers the goal/constraints; ask Astra.',
                       'observe':'Need current visual evidence before selecting a plan.'}
            choices.update({p['id']:p for p in candidates})
            selected = self.choose('Select a reusable task plan, not a benchmark policy score.', evidence,
                                   choices, {'instruction':self.io.instruction})
            if selected == 'observe':
                predicates = sorted({q for p in candidates for q in p['preconditions']})
                evidence = self.io.inspect({'completion':predicates}, 'Verify plan prerequisites from current images')
                self.observations += 1
                selected = self.choose('Select only when prerequisites are supported.', evidence, choices,
                                       {'instruction':self.io.instruction})
            chosen = next((p for p in candidates if p['id'] == selected), None)
            if chosen and (not chosen['preconditions'] or evidence.satisfies(chosen['preconditions'])):
                self.plan = chosen
        if self.plan is None:
            self.plan = validate_plan(self.io.plan(self.library.skills, evidence), self.library.skills,
                                      self.library.available)
            if self.plan['preconditions'] and not evidence.satisfies(self.plan['preconditions']):
                evidence = self.io.inspect({'completion':self.plan['preconditions'], 'goal':self.plan['goal']}, 'Verify new plan prerequisites')
                if not evidence.satisfies(self.plan['preconditions']):
                    raise RuntimeError('New plan prerequisites are not established')
        self.emit('plan_bound', plan=self.plan, library_hash=self.library.hash)
        return evidence

    def switch(self, executor, stage, evidence, reason):
        old = self.ownership.executor
        self.ownership.transfer(executor, self.io.cancel_pending)
        if old != executor or reason == 'recovered':
            self.io.reset_context(executor, reason)
        self.emit('ownership', old=old, executor=executor, epoch=self.ownership.epoch,
                  stage=stage['id'], revision=evidence.revision, reason=reason)

    def execute(self, executor, stage, evidence, recovery=None):
        if self.ownership.executor != executor:
            self.switch(executor, stage, evidence, 'recovery' if recovery else 'stage')
        ticket = self.ownership.ticket(stage['id'], evidence)
        proposal = self.io.propose(executor, stage, evidence, recovery)
        current = self.io.evidence()
        self.ownership.validate(ticket, stage['id'], current)
        if current.ended or current.remaining <= 0:
            return current
        self.io.validate(proposal, executor, current.remaining)
        if executor == 'wam' and hasattr(self.io, 'guard_proposal'):
            current, allowed = self.io.guard_proposal(proposal, stage, current)
            if not allowed:
                self.emit('chunk_blocked', stage=stage['id'], revision=current.revision)
                return current
            # Inspection changes evidence revision, not the authorized proposal.
            ticket = self.ownership.ticket(stage['id'], current)
            self.ownership.validate(ticket, stage['id'], self.io.evidence())
            self.io.validate(proposal, executor, current.remaining)
            if current.ended or current.remaining <= 0:
                return current
        result = self.io.execute(proposal, executor)
        self.blocks += 1
        self.emit('execution', ticket=vars(ticket), report=result)
        return self.io.evidence()

    def repair(self, stage, evidence):
        if self.recoveries >= self.cfg['max_recoveries']:
            return evidence, False
        self.recoveries += 1
        operation,evidence=self.try_program(stage,evidence,recovery=True)
        if operation is not None:
            if evidence.ended:return evidence,False
            if evidence.facts.get('recovery_complete') is True:return evidence,True
        if self.cfg.get('code_subgoal_routing') and hasattr(self.io,'bind_code_subgoal'):
            sub,evidence=self.io.bind_code_subgoal(stage,evidence,recovery=True)
            if sub is not None:
                candidates=self.library.executable_candidates(sub,evidence)
                decision=self.choose('Select a grounded recovery subgoal.',evidence,
                    {'keep_executor':'Use Astra recovery instead',**candidates},{'stage':sub})
                if decision in candidates:
                    evidence=self.execute('cap',sub,evidence)
                    if evidence.ended:return evidence,False
                    evidence=self.io.inspect(sub,'Verify code recovery subgoal; not necessarily entire fault repaired')
                    self.emit('cap_recovery_result',skill=decision,passed=evidence.satisfies(sub['completion']))
                    if evidence.satisfies(sub['completion']):
                        evidence=self.io.inspect(stage,'Separately verify original fault recovery_complete and next executor applicability')
                        if evidence.facts.get('stop_recommended') is True:return evidence,False
                        if evidence.facts.get('recovery_complete') is True:return evidence,True
        matches = self.library.recoveries(stage, evidence)
        selected = None
        if len(matches) == 1:
            selected = next(iter(matches))
        elif matches:
            choices = {'escalate':'No listed recovery fits this situation', **matches}
            selected = self.choose('Select a validated recovery.', evidence, choices,
                                   {'stage':stage})
        if selected in matches:
            skill = matches[selected]
            evidence = self.execute(skill['executor'], stage, evidence, recovery=selected)
            evidence = self.io.inspect(dict(stage, completion=skill['postconditions']), 'Verify recovery postconditions')
            if evidence.facts.get('stop_recommended') is True:
                return evidence, False
            if evidence.satisfies(skill['postconditions']):
                evidence = self.io.inspect(stage, 'Recovery postconditions passed. Separately verify recovery_complete, wam_applicable and astra_applicable from current evidence.')
                if evidence.facts.get('stop_recommended') is True:
                    return evidence, False
                if evidence.facts.get('recovery_complete') is True:
                    return evidence, True
        # Astra diagnoses and emits only existing command-grammar actions.
        for _ in range(self.cfg['max_astra_recovery_blocks']):
            if evidence.ended or evidence.remaining <= 0:
                return evidence, False
            evidence = self.execute('astra', stage, evidence, recovery='astra_diagnosis')
            evidence = self.io.inspect(stage, 'Separately assess recovery_complete (fault repaired), wam_applicable and astra_applicable (next executor suitable for remaining stage). Recovery alone does not establish WAM applicability; unknown is null.')
            if evidence.facts.get('stop_recommended') is True:
                return evidence, False
            if evidence.facts.get('recovery_complete') is True:
                return evidence, True
        return evidence, False

    def next_executor(self, stage, evidence):
        if evidence.facts.get('recovery_complete') is not True:
            return None
        original = stage['executor']
        if original == 'wam' and not self.wam_return_disabled and evidence.facts.get('wam_applicable') is True:
            return 'wam'
        if evidence.facts.get('astra_applicable') is True:
            return 'astra'
        return None

    def try_program(self,stage,evidence,recovery=False):
        if not self.cfg.get('operation_programs_enabled') or not hasattr(self.io,'bind_operation'):
            return None,evidence
        try:
            binding=self.io.bind_operation(stage,evidence,recovery)
        except (ValueError,KeyError,TypeError) as exc:
            self.emit('program_binding_rejected',error=str(exc),stage=stage['id'])
            return None,self.io.evidence()
        if binding is None:return None,self.io.evidence()
        # JEV makes a bounded local dispatch decision after schema/geometry binding;
        # it does not manufacture probability of physical success.
        key=(stage['id'],binding['operation'],binding['object'],binding['target'])
        decision=self.choose('Dispatch bound operation or retain executor.',evidence,
            {'keep_executor':'retain', 'run_program':{'preconditions':['program_binding_valid']}},
            {'observation':dict(evidence.decision_state(),facts={'program_binding_valid':True})})
        if decision!='run_program':return None,self.io.evidence()
        self.io.program_attempts[key]=self.io.program_attempts.get(key,0)+1
        self.switch('cap',stage,evidence,'registered_operation')
        result=self.io.run_operation(binding,stage)
        self.blocks+=result['motion_batches']
        e=self.io.evidence()
        if not e.ended and getattr(self.io,'program_checked_step',None)!=e.steps:
            e=self.io.inspect(stage,'Check original stage after registered operation. Local completed does NOT imply whole stage complete. If operation returned uncertain/blocked, determine recovery and executor applicability independently.')
        if result['status']=='blocked':e.motion['hard_error']=True
        self.emit('operation_dispatch_result',result=result,stage=stage['id'])
        return result,e

    def run(self):
        evidence = self.io.evidence()
        if evidence.native_success:
            return self.finish('native', evidence)
        if evidence.ended or evidence.remaining <= 0:
            return self.finish('budget_or_end', evidence)
        evidence = self.plan_episode(evidence)
        for stage in self.plan['stages']:
            executor = 'astra' if self.wam_return_disabled and stage['executor']=='wam' else stage['executor']
            if executor != stage['executor']:
                evidence = self.io.inspect(stage, 'WAM re-entry disabled after immediate recurrence. Verify Astra applicability for this stage.')
                if evidence.facts.get('astra_applicable') is not True:
                    return self.finish('no_suitable_executor', evidence)
            self.switch(executor, stage, evidence, 'stage_start')
            inspections = 0
            for block in range(stage['max_blocks']):
                if evidence.native_success or evidence.ended or evidence.remaining <= 0:
                    return self.finish('native' if evidence.native_success else 'budget_or_end', evidence)
                operation,evidence=self.try_program(stage,evidence)
                if operation is None:
                    selected = stage.get('skill') if executor == 'cap' else None
                    execution_stage = stage
                    if executor != 'cap' and block == 0 and self.cfg.get('code_subgoal_routing') and hasattr(self.io,'bind_code_subgoal'):
                        sub,evidence=self.io.bind_code_subgoal(stage,evidence)
                        if sub is not None:
                            candidates=self.library.executable_candidates(sub,evidence)
                            choice=self.choose('Select grounded code subgoal.',evidence,
                                {'keep_executor':'Keep current executor',**candidates},{'stage':sub})
                            if choice in candidates:selected=choice;execution_stage=sub
                    if selected is None and executor != 'cap' and hasattr(self.library, 'executable_candidates'):
                        candidates = self.library.executable_candidates(stage, evidence)
                        if candidates:
                            choice = self.choose('Select a bounded executable capability for this stage.', evidence,
                                {'keep_executor':'Keep current executor', **candidates}, {'stage':stage})
                            if choice in candidates: selected = choice
                    if selected:
                        skill = self.library.skills[selected]
                        if not evidence.satisfies(skill['preconditions']):
                            evidence = self.io.inspect(dict(stage,completion=skill['preconditions']), 'Verify code skill preconditions')
                        if not evidence.satisfies(skill['preconditions']):
                            return self.finish('skill_preconditions_unverified', evidence)
                        self.emit('cap_selected', skill=selected, version=skill.get('version'),
                                  validated=skill.get('validated',False), bindings=execution_stage.get('bindings',{}))
                        cap_stage = dict(execution_stage, executor='cap', skill=selected)
                        evidence = self.execute('cap', cap_stage, evidence)
                        if not evidence.ended:
                            evidence = self.io.inspect(dict(execution_stage,completion=skill['postconditions']), 'Verify code skill postconditions from NEW observations; command completion is not physical success')
                        self.emit('cap_result', skill=selected, passed=evidence.satisfies(skill['postconditions']),
                                  evidence=evidence.decision_state())
                        if not evidence.satisfies(skill['postconditions']):
                            evidence.motion['hard_error'] = True
                        elif execution_stage is not stage:
                            evidence = self.io.inspect(stage,'Subgoal passed. Separately inspect progress and completion of the original stage; do not equate them.')
                    else:
                        evidence = self.execute(executor, stage, evidence)
                if evidence.native_success:
                    return self.finish('native', evidence)
                if evidence.ended or evidence.remaining <= 0:
                    return self.finish('budget_or_end', evidence)
                # Known hard errors can skip a model decision. Unknown visual progress cannot.
                if evidence.motion.get('hard_error'):
                    decision = 'recover'
                elif evidence.satisfies(stage['completion']):
                    decision = 'complete'
                else:
                    decision = self.choose('Judge CURRENT stage progress from actual execution evidence.',
                                           evidence, PROCESS_CHOICES, {'stage':stage})
                # Periodic semantic refresh bounds how long stale visual facts can be trusted.
                if decision == 'continue' and (block+1) % self.cfg['semantic_refresh_blocks'] == 0:
                    decision = 'observe'
                if decision in ('observe','complete') and not evidence.satisfies(stage['completion']):
                    if inspections >= self.cfg['max_inspections_per_stage']:
                        return self.finish('inspection_budget', evidence)
                    evidence = self.io.inspect(stage, 'Inspect progress and completion from current images')
                    inspections += 1
                    decision = 'complete' if evidence.satisfies(stage['completion']) else self.choose(
                        'Evidence refreshed. Continue only if appropriate.', evidence, PROCESS_CHOICES, {'stage':stage})
                    if decision == 'observe' or decision == 'complete' and not evidence.satisfies(stage['completion']):
                        decision = 'escalate'  # bounded unknown loop; never accept unverified completion
                self.emit('process_decision', stage=stage['id'], decision=decision,
                          evidence=evidence.decision_state())
                if decision == 'complete':
                    break
                if decision == 'stop':
                    return self.finish('policy_stop', evidence)
                if operation is not None and decision=='continue' and executor=='wam':
                    # A completed code operation does not prove WAM is suitable now.
                    if not self.wam_return_disabled and evidence.facts.get('wam_applicable') is True:
                        self.switch('wam',stage,evidence,'recovered')
                        self.wam_return_block=self.blocks
                    elif evidence.facts.get('astra_applicable') is True:
                        executor='astra';self.switch(executor,stage,evidence,'program_continuation')
                    else:
                        return self.finish('no_suitable_executor',evidence)
                if decision in ('recover','escalate'):
                    if executor == 'wam' and self.wam_return_block is not None and self.blocks-self.wam_return_block <= 2:
                        # Conservative episode-local lockout: any immediate recurrence, not an invented diagnosis of identical cause.
                        self.wam_return_disabled = True
                        self.emit('wam_return_disabled', reason='anomaly_within_two_blocks_after_return', stage=stage['id'])
                    evidence, ok = self.repair(stage, evidence)
                    if evidence.native_success:
                        return self.finish('native', evidence)
                    if not ok:
                        return self.finish('policy_stop' if evidence.facts.get('stop_recommended') is True else 'recovery_failed', evidence)
                    executor = self.next_executor(stage, evidence)
                    self.emit('recovery_handoff', recovery_complete=evidence.facts.get('recovery_complete'),
                              wam_applicable=evidence.facts.get('wam_applicable'),
                              astra_applicable=evidence.facts.get('astra_applicable'), selected=executor)
                    if executor is None:
                        return self.finish('no_suitable_executor', evidence)
                    self.switch(executor, stage, evidence, 'recovered')
                    if executor == 'wam':
                        self.wam_return_block = self.blocks
            else:
                return self.finish('stage_budget', evidence)
        return self.finish('plan_finished', evidence)

    def finish(self, reason, evidence):
        self.io.cancel_pending()
        if hasattr(self.io, 'finalize'):
            evidence = self.io.finalize()
            if evidence.native_success:
                reason = 'native'
        result = {'termination_reason':reason, 'success':evidence.native_success,
                  'steps_used':evidence.steps, 'blocks':self.blocks, 'jev_calls':self.calls,
                  'recovery_attempts':self.recoveries, 'evaluation_track':'no_feedback',
                  'execution_mode':'adaptive_v04'}
        self.emit('end', **result)
        return result
