"""Construct an isolated primitive-program skill from a validated assembly."""
from PhysicalRSI_core.infra.resources import ResourcePool
from ..code_policy_factory import code_policy_factory


def load(configuration):
    """Use the standard frozen harness, source revision, and primitive contract."""
    if configuration.get('runtime') == 'frozen-program-service':
        from .program_service import Model
        return Model(configuration)
    pool = ResourcePool(configuration['resources'])
    factory = code_policy_factory(configuration['spec'], pool=pool)
    return factory(harness=configuration['harness'], binding=configuration['binding'],
                   output=configuration['output'])
