"""The existing public Rule factory behind the standard unified Model."""
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys

from .rule_configuration import settings
ROOT = Path(__file__).resolve().parents[1]

def prepare_policy_cache(output, environment=None):
    """Keep Numba's generated CPU cache beside this run, outside frozen skills."""
    environment = os.environ if environment is None else environment
    cache = Path(output).resolve().parent / 'numba-cache'
    environment['NUMBA_CACHE_DIR'] = str(cache)
    return cache

def prepare_task_resources(mode, root=ROOT, *, query=subprocess.check_output):
    """Reuse the original Fold SAM declaration on the actually leased UUID."""
    if mode not in ('fold_clothes', 'fold_clothes_random'):
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible:
        raise ValueError('Fold public perception needs one explicitly leased GPU UUID')
    lines = query(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    matches = [int(index.strip()) for line in lines.splitlines() if ',' in line
               for index, uuid in [line.split(',', 1)] if uuid.strip() == visible]
    if len(matches) != 1:
        raise ValueError('Cannot map the leased UUID to one actual physical GPU')
    module = importlib.import_module('tasks.' + mode + '.skills.native_binding.implementation')
    return module.prepare_single_gpu_perception(root, matches[0])


def create(mode, declaration, config):
    output_root = Path(config["output_dir"]).resolve()
    configuration = settings(mode)
    prepare_policy_cache(output_root)
    from rsi.task_contract import verify
    verify(ROOT, mode)
    from rsi.resource_locations import relocate_path, load_manifest
    sys.path.insert(0, str(Path(relocate_path('resource://protocol_dependencies/site'))))
    declaration = configuration['tasks'][mode]
    from .rule_reuse import implementation_mode
    requested_mode = mode
    mode = implementation_mode(ROOT, mode)
    overrides = {}
    if requested_mode != mode:
        overrides = json.loads((ROOT / 'tasks' / requested_mode / 'composition.json').read_text())['parameter_overrides']
    task = ROOT / 'tasks' / mode
    metadata = json.loads((task / 'task.json').read_text())
    family = declaration['adapter_family']
    if family == 'public_mature_workflow':
        profile = load_manifest(task / metadata.get('native_evaluation_profile', 'workflow.json'))
        contract = json.loads((task / 'bindings/observation-contract.json').read_text())
        if contract.get('public_factory') != 'controller.create_public_environment' or contract.get('measured_finger_joints_required') is not False:
            raise ValueError('This adapter requires the declared public mature factory')
        controller = importlib.import_module('tasks.' + mode + '.controller')
    else:
        from skills.shared.orchestration.public_task_workflow.implementation import FAMILIES
        if family not in FAMILIES:
            raise ValueError('No public adapter for the declared task family')
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'rule-implementation.json').write_text(json.dumps(dict(
        evaluated_mode=requested_mode, implementation_mode=mode, parameter_overrides=overrides,
        learned_fallback=False), indent=2) + '\n')
    from skills.shared.orchestration.public_rule_policy_interface.implementation import BlockingRulePolicy
    from skills.shared.orchestration.public_mature_workflow.implementation import run, save
    resource_binding = prepare_task_resources(mode)
    if resource_binding is not None:
        save(output_root / 'public-perception-resource-binding.json', resource_binding)
    # Horizon is task-declared and independently checked against official source
    # against pinned official files. No scorer or terminal-success state enters the actor.
    counter = 0
    def runner(initial, exchange):
        nonlocal counter
        index = counter; counter += 1
        output = output_root / ('episode-%04d' % index)
        try:
            if family == 'public_mature_workflow':
                run(controller, profile, initial, exchange, output,
                    horizon=declaration['official_action_horizon'])
            else:
                from skills.shared.orchestration.public_task_workflow.implementation import run as run_task
                run_task(mode, declaration, initial, exchange, output, public_parameters=overrides)
        finally:
            # The real policy server is reused across official episodes.
            # Release unreachable episode tools after their worker has exited;
            # only this process's unreferenced allocator blocks are released.
            try:
                import gc
                gc.collect()
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception as error:
                print(json.dumps({'event': 'policy_episode_allocator_cleanup_error',
                                  'error': repr(error)}), flush=True)
    policy = BlockingRulePolicy(runner, wait_timeout=600, join_timeout=10,
                                planning_poll_seconds=.5)
    return policy
