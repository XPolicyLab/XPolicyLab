"""XPolicyLab entrypoint using PhysicalRSI Operations and immutable memory."""
import os
import uuid
from copy import deepcopy
from numbers import Integral
from pathlib import Path

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.storage import atomic_json, read_json, digest
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.skills.episode_plan import catalogue as operation_catalogue, freeze_episode_plan, SCHEMA as OPERATION_PLAN_SCHEMA
from .agent_planner import AgentPlanner
from .capability_memory import attach_capability_memory
from .execution_skills import load_skill, compose_skill, compose_observer


class Model:
    def __init__(self, model_cfg):
        path = os.environ.get('PHYSICALRSI_SKILL_CONFIG') or model_cfg.get('physicalrsi_skill_config')
        if not path:
            raise ValueError('Set PHYSICALRSI_SKILL_CONFIG to a frozen skill library configuration')
        self.config = read_json(Path(path))
        if self.config.get('schema') != 'physicalrsi.skill-library/v1':
            raise ValueError('Versioned skill library configuration required')
        self.deployment = deepcopy(model_cfg)
        catalog = read_json(Path(__file__).parent / 'configs/skill_compositions.json')['compositions']
        extensions = self.config.get('composition_definitions', {})
        if set(extensions) & set(catalog):
            raise ValueError('Custom compositions cannot replace built-in definitions')
        catalog.update(extensions)
        self.catalog = {}
        for name in self.config['compositions']:
            entry = catalog[name]
            if (not isinstance(entry.get('description'), str)
                    or not entry['description'].strip()
                    or not isinstance(entry.get('steps'), list)
                    or len(entry['steps']) != 2
                    or entry['steps'][0] != 'memory.snapshot'
                    or (entry['steps'][-1] not in self.config['skills']
                        and not (entry['steps'][-1] == 'memory-program-library'
                                 and self.config.get('program_registry')))):
                raise ValueError('Composition has no configured executable skill: ' + name)
            entry = deepcopy(entry)
            skill_entry = self.config['skills'].get(entry['steps'][-1])
            memories = skill_entry['configuration'].get('operation_memories') if skill_entry else None
            if memories is not None:
                if not isinstance(memories, dict) or not memories:
                    raise ValueError('Structured skill requires registered operation memories')
                for revision, memory in memories.items():
                    if digest(memory) != revision:
                        raise ValueError('Configured operation memory revision mismatch')
                entry['operation_memory'] = [operation_catalogue(memory) for memory in memories.values()]
            entry = attach_capability_memory(entry, entry['steps'][-1])
            self.catalog[name] = entry
        if not self.catalog:
            raise ValueError('At least one configured skill composition required')
        registry = self.config.get('program_registry', {})
        self.operation_program = None
        if registry:
            operations = {}
            for operation, entry in registry.items():
                if not isinstance(entry, dict) or not isinstance(entry.get('configuration'), dict):
                    raise ValueError('Invalid operation memory registry entry: ' + str(operation))
                operations[operation] = {
                    'input': entry.get('input', 'robodojo.observation-batch/v1'),
                    'output': entry.get('output', 'robodojo.action-chunks/v1'),
                    'description': entry.get('description', ''),
                    'memory_primitives': entry.get('memory_primitives', []),
                }
                operations[operation] = attach_capability_memory(
                    operations[operation], entry.get('source_name', operation))
                memories = entry['configuration'].get('operation_memories', {})
                for revision, memory in memories.items():
                    if digest(memory) != revision:
                        raise ValueError('Configured operation memory revision mismatch')
                if memories:
                    operations[operation]['operation_memory'] = [
                        operation_catalogue(memory) for memory in memories.values()]
            self.operation_program = {
                'input': 'robodojo.observation-batch/v1',
                'output': 'robodojo.action-chunks/v1',
                'operations': operations,
            }
        self.planner = AgentPlanner(**self.config.get('agent', {}))
        self.root = Path(os.environ.get('PHYSICALRSI_EVAL_OUTPUT', './results/physicalrsi')) / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.store = MemoryStore(self.root / 'memory')
        library_memory = read_json(Path(__file__).parent / 'configs/policy_memory.json')
        self.memory = self.store.snapshot({'task_specific_policy_memory': {
            'library': library_memory, 'run_context': self.config.get('memory', {})}})
        self.identity = dict(schema='physicalrsi.skill-model/v1', library_sha256=digest(self.config),
                             composition_catalog_sha256=digest(self.catalog),
                             memory_revision=self.memory.revision, qualification=False, baseline_mode='task-aware-skill-library',
                             agent_role='memory_conditioned_skill_composition')
        atomic_json(self.root / 'identity.json', self.identity)
        self.action_queues = {}
        self.groups = {}
        self.sessions = {}
        self.pending = {}
        self.failed = False
        self.closed = False
        self.indices = []

        # Load configured checkpoint-backed skills before the policy server is ready.
        # The observation-dependent agent selects among ready implementations.
        self.preloaded = {}
        self.preloaded_plan_revisions = {}
        try:
            for name, entry in self.config['skills'].items():
                if entry.get('implementation', name) not in {'pi05', 'pi05-sparse-memory'}:
                    continue
                descriptor = deepcopy(entry)
                memories = descriptor['configuration'].get('operation_memories')
                if memories:
                    plan = dict(schema=OPERATION_PLAN_SCHEMA, episode='initialization',
                                rationale='Initialize the registered immutable programs.', programs={
                        revision: {method: program['steps'] for method, program in memory['programs'].items()}
                        for revision, memory in memories.items()})
                    environment = freeze_episode_plan(self.root / 'initialization' / name / 'operation-plan.json',
                                                      plan, memories)
                    descriptor['configuration'].setdefault('environment', {}).update(environment)
                    self.preloaded_plan_revisions[name] = digest(plan)
                self.preloaded[name] = load_skill(descriptor, self.deployment)
        except BaseException:
            for skill in self.preloaded.values():
                skill.close()
            raise

    def physicalrsi_identity(self):
        return deepcopy(self.identity)

    def _start_episode(self, index, observation):
        instruction = observation.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Natural-language instruction required')
        plan = self.planner.plan(instruction=instruction, observation=observation,
                                 memory=self.memory.read(), previous_plan=None,
                                 compositions=self.catalog, operation_program=self.operation_program)
        expected = {'composition', 'operations', 'rationale'} if self.operation_program else {'composition', 'rationale'}
        if (set(plan) != expected
                or plan.get('composition') not in self.catalog
                or not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip()):
            raise ValueError('Agent must choose a registered skill composition with rationale')
        name = plan['composition']
        episode = uuid.uuid4().hex
        atomic_json(self.root / 'episodes' / episode / 'agent-decision.json',
                    dict(env_idx=index, instruction=instruction, plan=plan,
                         status='selected_before_episode_binding', **self.identity))
        selected_operation = None
        if name == 'memory-guided-program-library':
            operations = plan.get('operations')
            if not isinstance(operations, list) or len(operations) != 1:
                raise ValueError('The operation library requires one complete capability at episode start')
            selected_operation = operations[0]
            if selected_operation not in self.config.get('program_registry', {}):
                raise ValueError('Agent selected an unregistered operation memory')
            registered = self.config['program_registry'][selected_operation]
            selected = registered.get('source_name', selected_operation)
            descriptor = {
                'name': selected,
                'implementation': 'code-policy',
                'configuration': deepcopy(registered['configuration']),
            }
        else:
            if plan.get('operations'):
                raise ValueError('Operations cannot be attached to an independent execution skill')
            selected = self.catalog[name]['steps'][-1]
            descriptor = deepcopy(self.config['skills'][selected])
        if descriptor.get('name') != selected:
            raise ValueError('Skill descriptor identity mismatch')
        operation_plan_revision = None
        memories = descriptor['configuration'].get('operation_memories')
        if memories is not None:
            # The sole agent choice binds the complete registered stage programs.
            # It does not authorize modifying their numerical implementations.
            operation_plan = dict(schema=OPERATION_PLAN_SCHEMA, episode=episode,
                                  rationale=plan['rationale'], programs={
                revision: {method: program['steps'] for method, program in memory['programs'].items()}
                for revision, memory in memories.items()})
            environment = freeze_episode_plan(self.root / 'episodes' / episode / 'operation-plan.json',
                                              operation_plan, memories)
            descriptor['configuration'].setdefault('environment', {}).update(environment)
            operation_plan_revision = digest(operation_plan)
        group_key = selected if name == 'memory-guided-program-library' or selected in self.preloaded else name
        group = self.groups.get(group_key)
        if group is None:
            group_id = uuid.uuid4().hex
            worker_plan_revision = None
            if memories is not None:
                worker_plan = dict(schema=OPERATION_PLAN_SCHEMA, episode=group_id,
                                   rationale='Execute the registered composition independently selected by each member episode.',
                                   programs=operation_plan['programs'])
                environment = freeze_episode_plan(self.root / 'groups' / group_id / 'operation-plan.json',
                                                  worker_plan, memories)
                descriptor['configuration'].setdefault('environment', {}).update(environment)
                worker_plan_revision = digest(worker_plan)
            skill = self.preloaded.get(selected)
            if skill is None:
                skill = load_skill(descriptor, self.deployment)
            else:
                worker_plan_revision = self.preloaded_plan_revisions.get(selected)
            try:
                group = dict(skill=skill, operation=compose_skill(group_key, skill, self.memory),
                             observer=compose_observer(group_key, skill, self.memory),
                             context=Context(group_id), operation_plan_revision=worker_plan_revision,
                             indices=[])
                self.groups[group_key] = group
            except BaseException:
                skill.close()
                raise
        skill = group['skill']
        operation = group['operation']
        episode_memory = self.store.snapshot({
            'parent_revision': self.memory.revision, 'episode': episode,
            'instruction': instruction, 'plan': plan, 'skill': skill.describe(),
            'composition_revision': operation.revision,
            'episode_operation_plan_revision': operation_plan_revision,
            'operation_plan_revision': group['operation_plan_revision'],
            'execution_group': group['context'].episode,
            'evidence_status': 'skill_composition_applied; outcome_not_yet_observed',
        })
        atomic_json(self.root / 'episodes' / episode / 'episode.json',
                    dict(env_idx=index, plan=plan, skill=skill.describe(),
                         composition_revision=operation.revision,
                         episode_operation_plan_revision=operation_plan_revision,
                         operation_plan_revision=group['operation_plan_revision'],
                         execution_group=group['context'].episode,
                         episode_memory_revision=episode_memory.revision, **self.identity))
        return dict(skill=skill, operation=operation, observer=group['observer'],
                    context=Context(episode), group=group_key, steps=0, delivered_chunks=0)

    def update_obs(self, obs):
        return self.update_obs_batch([obs if "env_idx" in obs else dict(obs, env_idx=0)])

    def update_obs_batch(self, observations):
        if self.closed or self.failed:
            raise RuntimeError('Skill runtime closed or failed; reset required')
        rows = list(observations)
        indices = [row.get('env_idx') for row in rows]
        if (not 1 <= len(rows) <= 10 or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('One to ten distinct explicit environment observations required')
        self.pending.clear()
        try:
            grouped = {}
            for index, observation in zip(indices, rows):
                if index not in self.sessions:
                    self.sessions[index] = self._start_episode(index, observation)
                grouped.setdefault(self.sessions[index]['group'], []).append(observation)
            for name, batch in grouped.items():
                group = self.groups[name]
                group['observer'](batch, group['context'])
                group['indices'] = [row['env_idx'] for row in batch]
            self.pending.update(zip(indices, rows))
            self.indices = indices
        except BaseException:
            self.pending.clear()
            self.failed = True
            raise

    def get_action(self):
        if self.closed or self.failed:
            raise RuntimeError("Skill runtime closed or failed; reset required")
        if len(self.indices) != 1:
            raise ValueError('Single action requires one environment')
        return self.get_action_batch(self.indices)[0]

    def get_action_batch(self, env_idx_list=None):
        if self.closed or self.failed:
            raise RuntimeError('Skill runtime closed or failed; reset required')
        indices = list(self.indices if env_idx_list is None else env_idx_list)
        if (not indices or len(set(indices)) != len(indices)
                or not set(indices) <= set(self.indices) or not set(indices) <= set(self.pending)):
            raise ValueError('Fresh observations for every active environment required')
        try:
            self.memory.read()
            # Omitted environments have terminated according to the native caller.
            # Their remaining actions must never be delivered to another episode.
            for index in set(self.action_queues) - set(indices):
                del self.action_queues[index]
            names = list(dict.fromkeys(self.sessions[index]['group'] for index in indices))
            for name in names:
                group = self.groups[name]
                active = [index for index in group['indices']
                          if index in set(indices) and not self.action_queues.get(index)]
                if not active:
                    continue
                chunks = group['operation'](dict(observations=[self.pending[i] for i in active],
                                                  indices=active), group['context'])
                if (not isinstance(chunks, list) or len(chunks) != len(active)
                        or any(not isinstance(c, (list, tuple)) or not c
                               or any(not isinstance(a, dict) for a in c) for c in chunks)):
                    raise ValueError('Skill must return one nonempty action chunk per active environment')
                for index, chunk in zip(active, chunks):
                    # Own a stable copy: later backend calls cannot mutate a tail
                    # that has already been generated but not yet delivered.
                    self.action_queues[index] = deepcopy(list(chunk))
                    self.sessions[index]['steps'] += 1
            horizon = min(len(self.action_queues[index]) for index in indices)
            results = []
            for index in indices:
                queue = self.action_queues[index]
                results.append(queue[:horizon])
                self.action_queues[index] = queue[horizon:]
                session = self.sessions[index]
                session['delivered_chunks'] += 1
                atomic_json(self.root / 'episodes' / session['context'].episode / 'execution.json',
                            dict(action_chunks=session['steps'], delivered_chunks=session['delivered_chunks'],
                                 buffered_actions=len(self.action_queues[index]),
                                 composition_revision=session['operation'].revision,
                                 memory_revision=self.memory.revision,
                                 execution_group=self.groups[session['group']]['context'].episode))
            self.pending.clear()
            return results
        except BaseException:
            self.failed = True
            self.pending.clear()
            self.action_queues.clear()
            raise

    def reset(self):
        self.pending.clear()
        self.action_queues.clear()
        failure = None
        for group in self.groups.values():
            skill = group['skill']
            try:
                if skill in self.preloaded.values():
                    skill.reset()
                else:
                    skill.close()
            except BaseException as error:
                failure = failure or error
        self.groups.clear()
        self.sessions.clear()
        self.indices = []
        self.failed = failure is not None
        if failure is not None:
            raise failure

    def on_trial_end(self, result=None):
        return self.reset()

    def close(self):
        if self.closed:
            return
        try:
            self.reset()
        finally:
            for skill in self.preloaded.values():
                skill.close()
            self.closed = True
