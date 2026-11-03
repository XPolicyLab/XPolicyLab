"""Executable skills with a common observation/action/lifecycle contract."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import hashlib

from PhysicalRSI_core.infra.storage import digest


class ExecutionSkill:
    input_contract = 'robodojo.observation-batch/v1'
    output_contract = 'robodojo.action-chunks/v1'

    def __init__(self, name, implementation, revision):
        self.name = name
        self.implementation = implementation
        self.revision = revision
        self.closed = False
        for method in ('update_obs_batch', 'get_action_batch', 'reset'):
            if not callable(getattr(implementation, method, None)):
                raise ValueError('Skill implementation requires ' + method)

    def describe(self):
        return dict(name=self.name, revision=self.revision, input=self.input_contract,
                    output=self.output_contract, state='closed' if self.closed else 'ready')

    def observe(self, observations):
        if self.closed:
            raise RuntimeError('Skill is closed')
        return self.implementation.update_obs_batch(observations)

    def execute(self, indices):
        if self.closed:
            raise RuntimeError('Skill is closed')
        return self.implementation.get_action_batch(indices)

    def reset(self):
        if self.closed:
            raise RuntimeError('Skill is closed')
        return self.implementation.reset()

    def close(self):
        if not self.closed:
            close = getattr(self.implementation, 'close', None)
            if close:
                close()
            self.closed = True


def load_skill(descriptor, deployment):
    """Bind an explicit skill once. Task observations cannot change its identity."""
    descriptor = deepcopy(descriptor)
    name = descriptor['name']
    implementation_name = descriptor.get('implementation', name)
    configuration = deepcopy(descriptor['configuration'])
    if implementation_name in {'pi05', 'pi05-sparse-memory'}:
        if not configuration.get('model_path'):
            raise ValueError('Skill requires an explicit checkpoint model_path')
        module = 'pi05' if implementation_name == 'pi05' else 'sparse'
        cls = import_module('PhysicalRSI_baselines.robodojo.skills.' + module).Model
        # Dataset identity and weights belong to the frozen skill configuration.
        # Environment fields do not override model/training/checkpoint settings.
        for field in ('task_name', 'env_cfg_type', 'action_type'):
            configuration[field] = deployment[field]
        implementation = cls(configuration)
    elif implementation_name == 'code-policy':
        from .skills.code_policy import load
        implementation = load(configuration)
    else:
        raise ValueError('Unknown execution skill: ' + str(name))
    try:
        sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                   for source in sorted((Path(__file__).parent / "skills").glob("*.py"))}
        return ExecutionSkill(name, implementation, digest(dict(descriptor=descriptor, sources=sources)))
    except BaseException:
        close = getattr(implementation, 'close', None)
        if close:
            close()
        raise


def compose_skill(name, skill, snapshot):
    """Compose memory access and execution using the repository's Operation type."""
    from PhysicalRSI_core.contracts import Contract, Operation
    from PhysicalRSI.Embodied_Harness.skills.composition import sequence
    packet = Contract('robodojo.skill-request/v1')
    memory_type = Contract('robodojo.skill-request-with-memory/v1')
    reader = snapshot.reader('task_specific_policy_memory', Contract('policy-memory'))

    def retrieve(request, context):
        return dict(request, memory=reader(None, context))

    def execute(request, context):
        return skill.execute(request['indices'])

    memory = Operation('memory.snapshot', snapshot.revision, packet, memory_type, retrieve,
                       children=(reader,))
    execution = Operation(skill.name, skill.revision, memory_type,
                          Contract(skill.output_contract), execute,
                          frozenset({'execution-skill-state'}))
    return sequence(name, memory, execution)


def compose_observer(name, skill, snapshot):
    """Deliver every physical observation once, including within action chunks."""
    from PhysicalRSI_core.contracts import Contract, Operation
    from PhysicalRSI.Embodied_Harness.skills.composition import sequence
    rows = Contract('robodojo.observation-batch/v1')
    def checked(observations, context):
        snapshot.read()
        return observations
    guard = Operation('memory.observe-check', snapshot.revision, rows, rows, checked)
    observe = Operation(skill.name + '.observe', skill.revision, rows,
                        Contract('robodojo.observed-state/v1'),
                        lambda observations, context: skill.observe(observations),
                        frozenset({'execution-skill-state'}))
    return sequence(name + '.observe', guard, observe)
