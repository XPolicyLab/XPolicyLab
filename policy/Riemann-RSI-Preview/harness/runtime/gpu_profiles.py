"""Current single-card SAM binding; the outer official launcher owns GPU leases."""
from contextlib import contextmanager
import json
from pathlib import Path

RULE_SINGLE_PROFILE = 'debug-rule-only-single-gpu-v1'
_PUBLIC_PERCEPTION_BINDINGS = {}


def rule_only_single_gpu(config):
    """Validate the explicit model-free profile before any CUDA or worker startup."""
    if config.get('native_gpu_resource_profile') != RULE_SINGLE_PROFILE:
        return False
    expected = {'client_runtime_profile': 'debug', 'node_simulator_concurrency': 1,
                'model_initialization': 'disabled', 'planner_initialization': 'disabled',
                'debug_episode_planner_residency': False, 'planner_gpus': [],
                'motion_with_environment': True}
    for name, value in expected.items():
        if type(config.get(name)) is not type(value) or config[name] != value:
            raise ValueError('Rule-only single GPU requires its exact binding: ' + name)
    gpu = config.get('environment_physical_gpu')
    if (type(gpu) is not int or not 0 <= gpu <= 255
            or any(type(config.get(name)) is not int or config[name] != gpu for name in
                   ('policy_physical_gpu', 'perception_physical_gpu', 'motion_gpu'))
            or config.get('planner_service') is not None
            or config.get('model_resource_broker') is not None):
        raise ValueError('Rule-only simulation, SAM and motion must share one leased physical GPU')
    return True


def prepare_public_perception_resource(root, config):
    key = str(Path(root).absolute())
    if config.get('native_gpu_resource_profile') not in (None, RULE_SINGLE_PROFILE):
        raise ValueError('Only the current single-card public perception binding is supported')
    binding = None
    if rule_only_single_gpu(config):
        fields = ('native_gpu_resource_profile', 'client_runtime_profile',
                  'node_simulator_concurrency', 'model_initialization',
                  'planner_initialization', 'debug_episode_planner_residency',
                  'planner_gpus', 'policy_physical_gpu', 'perception_physical_gpu',
                  'environment_physical_gpu', 'motion_gpu', 'motion_with_environment')
        binding = {name: config[name] for name in fields}
    payload = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if key in _PUBLIC_PERCEPTION_BINDINGS and _PUBLIC_PERCEPTION_BINDINGS[key] != payload:
        raise ValueError('The prepared public perception binding cannot change during execution')
    _PUBLIC_PERCEPTION_BINDINGS[key] = payload


def public_perception_cuda_device(root, default):
    key = str(Path(root).absolute())
    if key not in _PUBLIC_PERCEPTION_BINDINGS:
        raise RuntimeError('Public perception requires its execution binding before access isolation')
    binding = json.loads(_PUBLIC_PERCEPTION_BINDINGS[key])
    return 0 if binding is not None and rule_only_single_gpu(binding) else default


@contextmanager
def public_perception_resource(root):
    key = str(Path(root).absolute())
    if key not in _PUBLIC_PERCEPTION_BINDINGS:
        raise RuntimeError('Public perception requires its execution binding before access isolation')
    config = json.loads(_PUBLIC_PERCEPTION_BINDINGS[key])
    if config is not None and not rule_only_single_gpu(config):
        raise ValueError('Only the current single-card public perception binding is supported')
    yield
