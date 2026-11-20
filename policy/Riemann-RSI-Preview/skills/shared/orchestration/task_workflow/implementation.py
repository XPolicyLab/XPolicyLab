"""Deterministic execution of frozen native skill boundaries, without model reselection."""
from collections import Counter, deque
from copy import deepcopy

from harness.execution import ExecutionError
from harness.runtime.manipulation import ACTION_BINDINGS, ACTION_LIMITS
from skills.shared.motion.rigid_transfer_program.implementation import RigidClock


class WorkflowStopped(ExecutionError):
    def __init__(self, code, stage, reason, phase=None):
        self.failure = {'code': code, 'stage': stage, 'reason': str(reason)[:4096],
                        'phase': phase, 'official_success_claimed': False}
        super().__init__(self.failure['reason'])


class DeterministicTaskWorkflow:
    def __init__(self, definition, adapter, *, action_bindings=None):
        self.action_bindings = dict(ACTION_BINDINGS if action_bindings is None else action_bindings)
        execution = definition.get('execution', {})
        stages = definition.get('stages', [])
        if (not set(adapter.allowed_skills) <= set(self.action_bindings)
                or not set(adapter.recovery_skills) <= set(adapter.allowed_skills)
                or execution != {'executor': 'deterministic_native', 'maximum_recovery_attempts_per_stage': 2,
                          'partial_irreversible_replay': False}
                or definition.get('completion_is_official_success') is not False
                or not isinstance(stages, list) or not 1 <= len(stages) <= 64):
            raise ExecutionError('A native workflow requires a bounded frozen execution contract')
        identifiers = []
        for stage in stages:
            if (not isinstance(stage, dict) or set(stage) - {'id', 'skill', 'side', 'move_index'}
                    or not isinstance(stage.get('id'), str) or not 1 <= len(stage['id']) <= 128
                    or 'move_index' in stage and (type(stage['move_index']) is not int or not 1 <= stage['move_index'] <= 64)
                    or 'side' in stage and stage['side'] not in ('left', 'right')
                    or stage.get('skill') not in adapter.allowed_skills):
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
        try:
            return self.select_command(api)
        except WorkflowStopped:
            raise
        except (ExecutionError, ValueError, KeyError, TypeError) as error:
            self.stop('native_continuation_unqualified', str(error))

    def select_command(self, api):
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
        if (not isinstance(request, dict) or request.get('skill') not in self.adapter.allowed_skills
                or request.get('skill') not in self.action_bindings
                or not repairing and request.get('skill') != self.stage['skill']
                or type(request.get('steps')) is not int or not 1 <= request['steps'] <= ACTION_LIMITS[request['skill']]):
            self.stop('undeclared_workflow_tool', 'Workflow dispatch cannot introduce a policy or undeclared tool.')
        self.pending = {'request': deepcopy(request), 'recovery': repairing, 'epoch': api.issued_action_count()}
        self.state = 'NEEDS_REPAIR' if repairing else 'RUNNING'
        return deepcopy(request)

    def verified_receipt(self, event):
        if not isinstance(event, dict) or not isinstance(event.get('execution_feedback'), dict):
            self.stop('invalid_boundary_receipt', 'The active native boundary requires a structured public receipt.')
        pending = self.pending
        feedback = event.get('execution_feedback', {})
        actions = event.get('actions')
        epoch = event.get('public_action_epoch')
        if (pending is None or event.get('event') != 'skill_outcome'
                or event.get('request') != pending['request'] or event.get('skill') != pending['request']['skill']
                or type(actions) is not int or actions < 0 or actions != feedback.get('actions_executed')
                or actions > pending['request']['steps'] or type(epoch) is not int
                or epoch != pending['epoch'] + actions or epoch < self.last_epoch
                or event.get('official_success_claimed') is not False
                or feedback.get('official_success_claimed') is not False):
            self.stop('invalid_boundary_receipt', 'Only the exact active request and its own action receipt may advance a workflow.')
        self.pending = None
        self.last_epoch = epoch
        return pending, feedback

    def accept(self, event):
        pending, feedback = self.verified_receipt(event)
        actions = event['actions']
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
            self.stop('partial_irreversible_stage',
                      'Commands were issued but the native boundary was not completed; stop without replaying the transfer or confirmation.', feedback.get('phase'))
        stage_id = self.stage['id']
        if self.attempts[stage_id] >= self.definition['execution']['maximum_recovery_attempts_per_stage']:
            self.stop('rule_recovery_exhausted', 'The same native stage remains unqualified after its bounded public recovery.', feedback.get('phase'))
        recovery = self.adapter.recover(self.stage, self.context, self.attempts[stage_id], event)
        if (not isinstance(recovery, list) or not 1 <= len(recovery) <= 3
                or any(request.get('skill') not in self.adapter.recovery_skills for request in recovery)):
            self.stop('rule_recovery_unavailable', 'No declared safe native recovery exists for this boundary.', feedback.get('phase'))
        self.attempts[stage_id] += 1
        self.recovery.extend(deepcopy(recovery))
        self.state = 'NEEDS_REPAIR'

    def snapshot(self):
        return {'executor': 'deterministic_native', 'state': self.state, 'cursor': self.cursor,
                'stage': self.stage['id'] if self.stage else 'complete',
                'completed_stages': list(self.completed_stages), 'recovery_attempts': dict(self.attempts),
                'active_skill': self.pending['request']['skill'] if self.pending else None,
                'failure': deepcopy(self.failure), 'official_success_claimed': False}


class AcknowledgedCommandWorkflow(DeterministicTaskWorkflow):
    def prepare_program(self, program):
        adapt = getattr(self.adapter, 'prepare_program', None)
        if adapt is None:
            return program
        if self.pending is None:
            raise ExecutionError('A control adapter may prepare only its exact pending native request')
        expected = deepcopy(self.pending['request'])
        if expected.get('side') == 'auto' and program.request.get('side') in ('left', 'right'):
            expected['side'] = program.request['side']
        effective = deepcopy(program.request)
        origin = deepcopy(vars(program).get('dispatch_request', effective))
        limit = effective.get('steps')
        internal_limit = vars(program).get('maximum_actions')
        if (origin != expected or type(limit) is not int or not 1 <= limit <= expected['steps']
                or effective != dict(expected, steps=limit)
                or limit != expected['steps'] and internal_limit != limit or program.actions != 0):
            raise ExecutionError('A control adapter may prepare only its exact pending native request')
        prepared = adapt(program, self.context, deepcopy(effective))
        refused = (type(prepared) is RigidClock and prepared.api is program.api
                   and prepared.finished is True and prepared.status == 'precondition_unmet')
        if (program.request != effective or program.actions != 0
                or vars(program).get('dispatch_request', program.request) != origin
                or vars(program).get('maximum_actions') != internal_limit
                or prepared.request != effective or prepared.actions != 0
                or not refused and (vars(prepared).get('dispatch_request', prepared.request) != origin
                                    or vars(prepared).get('maximum_actions') != internal_limit)):
            raise ExecutionError('A control adapter cannot rewrite the request or issue unacknowledged actions')
        return prepared

    def __init__(self, definition, adapter, *, action_bindings=None):
        maximum = definition.get('maximum_tool_boundaries')
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ExecutionError('A command workflow requires a frozen finite boundary budget')
        super().__init__(definition, adapter, action_bindings=action_bindings)
        self.maximum_tool_boundaries = maximum
        self.tool_boundaries = 0
        self.command_boundaries = 0
        self.observation_boundaries = 0

    def position(self):
        try:
            position = self.adapter.position(self.context)
        except ExecutionError as error:
            self.stop('native_phase_unqualified', str(error))
        if type(position) is not int or not 0 <= position <= len(self.definition['stages']):
            self.stop('invalid_command_transition', 'Own-command continuation selected an undeclared workflow boundary.')
        self.cursor = position

    def select_command(self, api):
        if self.failure is not None or self.pending is not None:
            return super().select_command(api)
        if self.context is None:
            self.context = deepcopy(self.adapter.initialise(api))
            self.position()
        if self.tool_boundaries >= self.maximum_tool_boundaries and self.stage is not None:
            self.stop('command_boundary_budget_exhausted', 'The frozen native boundary budget is exhausted; no policy takeover is permitted.')
        if self.stage is not None and not self.recovery:
            prepare = getattr(self.adapter, 'prepare', None)
            if callable(prepare):
                prepare(self.context, api)
                self.position()
        return super().select_command(api)

    def accept(self, event):
        try:
            self.accept_command(event)
        except WorkflowStopped:
            raise
        except (ExecutionError, ValueError, KeyError, TypeError) as error:
            self.stop('invalid_native_command_state', str(error))

    def accept_command(self, event):
        pending, feedback = self.verified_receipt(event)
        self.tool_boundaries += 1
        if pending['recovery']:
            if not self.adapter.recovery_completed(event):
                self.stop('rule_recovery_failed', 'The declared native recovery did not complete.', feedback.get('phase'))
            self.adapter.advance(self.context, event)
            if not self.recovery:
                self.position()
            self.state = 'NEEDS_REPAIR' if self.recovery else 'READY'
            return
        observation = getattr(self.adapter, 'zero_action_observation', lambda *arguments: False)
        if (event['actions'] > 0 or observation(self.stage, self.context, event)) and self.adapter.boundary_completed(self.stage, self.context, event):
            self.completed_stages.append(self.stage['id'])
            self.command_boundaries += int(event['actions'] > 0)
            self.observation_boundaries += int(event['actions'] == 0)
            self.adapter.advance(self.context, event)
            self.position()
            self.state = 'DONE' if self.stage is None else 'READY'
            return
        if event['actions'] > 0:
            self.stop('partial_irreversible_stage', 'The native command boundary is incomplete; do not replay it or switch backend.', feedback.get('phase'))
        key = self.stage['id'] + ':' + str(self.command_boundaries)
        if self.attempts[key] >= self.definition['execution']['maximum_recovery_attempts_per_stage']:
            self.stop('rule_recovery_exhausted', 'The current boundary exhausted its declared native recovery.', feedback.get('phase'))
        recovery = self.adapter.recover(self.stage, self.context, self.attempts[key], event)
        if (not isinstance(recovery, list) or not 1 <= len(recovery) <= 3
                or any(not isinstance(request, dict) or request.get('skill') not in self.adapter.recovery_skills for request in recovery)):
            self.stop('rule_recovery_unavailable', 'No declared safe native recovery exists for the current boundary.', feedback.get('phase'))
        self.attempts[key] += 1
        self.recovery.extend(deepcopy(recovery))
        self.state = 'NEEDS_REPAIR'

    def snapshot(self):
        return dict(super().snapshot(), tool_boundaries=self.tool_boundaries,
                    command_boundaries=self.command_boundaries,
                    observation_boundaries=self.observation_boundaries,
                    maximum_tool_boundaries=self.maximum_tool_boundaries,
                    resolver_calls=0, language_model_calls=0, policy_calls=0)


def configured_workflow(task_name):
    from importlib import import_module
    import json
    from pathlib import Path

    if not isinstance(task_name, str) or not task_name.isidentifier():
        raise ExecutionError('A deterministic workflow requires a registered deployment mode')
    root = Path(__file__).resolve().parents[4]
    path = root / 'tasks' / task_name / 'workflow.json'
    if path.is_symlink():
        raise ExecutionError('Use the contained frozen task workflow')
    definition = json.loads(path.read_text())
    if definition.get('execution', {}).get('executor') != 'deterministic_native':
        return None
    return import_module('tasks.' + task_name + '.controller').execution()
