"""Deterministic execution of frozen native skill boundaries, without model reselection."""
from collections import Counter, deque
from copy import deepcopy
from harness.execution import ExecutionError
from skills.shared.orchestration.observed_start_action_dispatch.implementation import ACTION_BINDINGS

class WorkflowStopped(ExecutionError):

    def __init__(self, code, stage, reason, phase=None):
        self.failure = {'code': code, 'stage': stage, 'reason': str(reason)[:4096], 'phase': phase, 'official_success_claimed': False}
        super().__init__(self.failure['reason'])

class DeterministicTaskWorkflow:

    def __init__(self, definition, adapter):
        execution = definition.get('execution', {})
        stages = definition.get('stages', [])
        if not set(adapter.allowed_skills) <= set(ACTION_BINDINGS) or not set(adapter.recovery_skills) <= set(adapter.allowed_skills) or execution != {'executor': 'deterministic_native', 'maximum_recovery_attempts_per_stage': 2, 'partial_irreversible_replay': False} or (definition.get('completion_is_official_success') is not False) or (not isinstance(stages, list)) or (not 1 <= len(stages) <= 64):
            raise ExecutionError('A native workflow requires a bounded frozen execution contract')
        identifiers = []
        for stage in stages:
            if not isinstance(stage, dict) or set(stage) - {'id', 'skill', 'side', 'move_index'} or (not isinstance(stage.get('id'), str)) or (not 1 <= len(stage['id']) <= 128) or ('move_index' in stage and (type(stage['move_index']) is not int or not 1 <= stage['move_index'] <= 64)) or ('side' in stage and stage['side'] not in ('left', 'right')) or (stage.get('skill') not in adapter.allowed_skills):
                raise ExecutionError('A workflow stage must use a declared native tool and bounded identifier')
            identifiers.append(stage['id'])
        if len(set(identifiers)) != len(identifiers):
            raise ExecutionError('A workflow cannot repeat a stage identifier')
        self.definition = deepcopy(definition)
        self.adapter = adapter
        self.cursor = 0
        self.context = None
        self.pending = None
        self.recovery = deque()
        self.attempts = Counter()
        self.state = 'INITIALISE'
        self.failure = None
        self.last_epoch = 0
        self.completed_stages = []

    @property
    def stage(self):
        return self.definition['stages'][self.cursor] if self.cursor < len(self.definition['stages']) else None

    def stop(self, code, reason, phase=None):
        self.state = 'FAILED'
        error = WorkflowStopped(code, self.stage['id'] if self.stage else 'complete', reason, phase)
        self.failure = deepcopy(error.failure)
        raise error

    def next_request(self, api):
        if self.failure is not None:
            raise WorkflowStopped(**{key: self.failure[key] for key in ('code', 'stage', 'reason', 'phase')})
        if self.pending is not None:
            self.stop('active_boundary_interrupted', 'The current native tool must finish before workflow advancement.')
        if self.context is None:
            try:
                self.context = deepcopy(self.adapter.initialise(api))
            except ExecutionError as error:
                self.stop('initial_public_qualification_failed', str(error))
        if self.stage is None:
            self.state = 'DONE'
            return None
        repairing = bool(self.recovery)
        request = self.recovery.popleft() if repairing else self.adapter.request(self.stage, self.context, api)
        if not isinstance(request, dict) or request.get('skill') not in self.adapter.allowed_skills or request.get('skill') not in ACTION_BINDINGS or (type(request.get('steps')) is not int) or (request['steps'] < 1):
            self.stop('undeclared_workflow_tool', 'Workflow dispatch cannot introduce a policy or undeclared tool.')
        self.pending = {'request': deepcopy(request), 'recovery': repairing, 'epoch': api.issued_action_count()}
        self.state = 'NEEDS_REPAIR' if repairing else 'RUNNING'
        return deepcopy(request)

    def accept(self, event):
        pending = self.pending
        feedback = event.get('execution_feedback', {})
        actions = event.get('actions')
        epoch = event.get('public_action_epoch')
        if pending is None or event.get('event') != 'skill_outcome' or event.get('request') != pending['request'] or (event.get('skill') != pending['request']['skill']) or (type(actions) is not int) or (actions < 0) or (actions != feedback.get('actions_executed')) or (actions > pending['request']['steps']) or (type(epoch) is not int) or (epoch != pending['epoch'] + actions) or (epoch < self.last_epoch) or (event.get('official_success_claimed') is not False) or (feedback.get('official_success_claimed') is not False):
            self.stop('invalid_boundary_receipt', 'Only the exact active request and its own action receipt may advance a workflow.')
        self.pending = None
        self.last_epoch = epoch
        if pending['recovery']:
            if not self.adapter.recovery_completed(event):
                self.stop('rule_recovery_failed', 'The declared native recovery did not complete; no policy takeover is permitted.', feedback.get('phase'))
            self.state = 'NEEDS_REPAIR' if self.recovery else 'READY'
            return
        if actions > 0 and self.adapter.boundary_completed(self.stage, self.context, event):
            self.completed_stages.append(self.stage['id'])
            self.cursor += 1
            self.state = 'DONE' if self.stage is None else 'READY'
            return
        if actions > 0:
            self.stop('partial_irreversible_stage', 'Commands were issued but the native boundary was not completed; stop without replaying the transfer or confirmation.', feedback.get('phase'))
        stage_id = self.stage['id']
        if self.attempts[stage_id] >= self.definition['execution']['maximum_recovery_attempts_per_stage']:
            self.stop('rule_recovery_exhausted', 'The same native stage remains unqualified after its bounded public recovery.', feedback.get('phase'))
        recovery = self.adapter.recover(self.stage, self.context, self.attempts[stage_id], event)
        if not isinstance(recovery, list) or not 1 <= len(recovery) <= 3 or any((request.get('skill') not in self.adapter.recovery_skills for request in recovery)):
            self.stop('rule_recovery_unavailable', 'No declared safe native recovery exists for this boundary.', feedback.get('phase'))
        self.attempts[stage_id] += 1
        self.recovery.extend(deepcopy(recovery))
        self.state = 'NEEDS_REPAIR'

    def snapshot(self):
        return {'executor': 'deterministic_native', 'state': self.state, 'cursor': self.cursor, 'stage': self.stage['id'] if self.stage else 'complete', 'completed_stages': list(self.completed_stages), 'recovery_attempts': dict(self.attempts), 'active_skill': self.pending['request']['skill'] if self.pending else None, 'failure': deepcopy(self.failure), 'official_success_claimed': False}
