"""XPolicyLab entrypoint using physicalRSI Operations and immutable memory."""
import os
import uuid
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.storage import atomic_json, read_json, digest
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from .agent_planner import AgentPlanner
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
                    or entry['steps'][-1] not in self.config['skills']):
                raise ValueError('Composition has no configured executable skill: ' + name)
            self.catalog[name] = entry
        if not self.catalog:
            raise ValueError('At least one configured skill composition required')
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
        self.sessions = {}
        self.pending = {}
        self.failed = False
        self.closed = False
        self.indices = []

    def physicalrsi_identity(self):
        return deepcopy(self.identity)

    def _start_episode(self, index, observation):
        instruction = observation.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Natural-language instruction required')
        plan = self.planner.plan(instruction=instruction, observation=observation,
                                 memory=self.memory.read(), previous_plan=None, compositions=self.catalog)
        if (set(plan) != {'composition', 'rationale'}
                or plan.get('composition') not in self.catalog
                or not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip()):
            raise ValueError('Agent must choose a registered skill composition with rationale')
        name = plan['composition']
        episode = uuid.uuid4().hex
        selected = self.catalog[name]['steps'][-1]
        descriptor = self.config['skills'][selected]
        if descriptor.get('name') != selected:
            raise ValueError('Skill descriptor identity mismatch')
        skill = load_skill(descriptor, self.deployment)
        try:
            operation = compose_skill(name, skill, self.memory)
            episode_memory = self.store.snapshot({
                'parent_revision': self.memory.revision, 'episode': episode,
                'instruction': instruction, 'plan': plan, 'skill': skill.describe(),
                'composition_revision': operation.revision,
                'evidence_status': 'skill_composition_applied; outcome_not_yet_observed',
            })
            observer = compose_observer(name, skill, self.memory)
            atomic_json(self.root / 'episodes' / episode / 'episode.json',
                        dict(env_idx=index, plan=plan, skill=skill.describe(),
                             composition_revision=operation.revision, episode_memory_revision=episode_memory.revision, **self.identity))
        except BaseException:
            skill.close()
            raise
        return dict(skill=skill, operation=operation, observer=observer, context=Context(episode), steps=0)

    def update_obs(self, obs):
        return self.update_obs_batch([obs if "env_idx" in obs else dict(obs, env_idx=0)])

    def update_obs_batch(self, observations):
        if self.closed or self.failed:
            raise RuntimeError('Skill runtime closed or failed; reset required')
        rows = list(observations)
        indices = [row.get('env_idx', 0) for row in rows]
        if not rows or len(set(indices)) != len(indices):
            raise ValueError('Distinct environment observations required')
        self.pending.clear()
        try:
            for index, observation in zip(indices, rows):
                if index not in self.sessions:
                    self.sessions[index] = self._start_episode(index, observation)
                session = self.sessions[index]
                session['observer']([observation], session['context'])
                self.pending[index] = observation
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
        if not indices or len(set(indices)) != len(indices) or not set(indices) <= self.pending.keys():
            raise ValueError('Fresh observations and agent guidance required')
        results = []
        try:
            for index in indices:
                session = self.sessions[index]
                observation = self.pending.pop(index)
                chunks = session['operation'](dict(observations=[observation], indices=[index]), session['context'])
                if not isinstance(chunks, list) or len(chunks) != 1 or not chunks[0]:
                    raise ValueError('Skill must return one nonempty action chunk')
                session['steps'] += 1
                atomic_json(self.root / 'episodes' / session['context'].episode / 'execution.json',
                            dict(action_chunks=session['steps'], composition_revision=session['operation'].revision,
                                 memory_revision=self.memory.revision))
                results.append(chunks[0])
            return results
        except BaseException:
            self.failed = True
            self.pending.clear()
            raise

    def reset(self):
        self.pending.clear()
        for index, session in list(self.sessions.items()):
            session['skill'].close()
            del self.sessions[index]
        self.indices = []
        self.failed = False

    def on_trial_end(self, result=None):
        return self.reset()

    def close(self):
        self.reset()
        self.closed = True
