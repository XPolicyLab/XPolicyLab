"""Create an evaluation configuration from installed, explicitly identified assets."""
import argparse
import hashlib
import json
from pathlib import Path


def configure(assets, *, python, output, evidence, endpoint, model, framework=None):
    assets = Path(assets).resolve(strict=True)
    # Preserve a virtualenv's executable path: resolving its symlink can select
    # the base interpreter and lose every dependency installed in the environment.
    python = Path(python).absolute()
    if not python.is_file():
        raise ValueError('A policy Python executable is required')
    example = Path(__file__).parent / 'configs/skill_library.example.json'
    config = json.loads(example.read_text())
    config['agent'] = {'endpoint': endpoint, 'model': model}
    for name, folder in [('pi05', 'pi05'), ('pi05-sparse-memory', 'pi05-sparse-memory')]:
        checkpoint = assets / 'checkpoints' / folder
        if not (checkpoint / 'params').exists():
            raise FileNotFoundError('Missing checkpoint params: ' + str(checkpoint))
        config['skills'][name]['configuration']['model_path'] = str(checkpoint)
    programs = assets / 'implementations/code-skills'
    manifest = programs / 'programs.json'
    if manifest.exists():
        package = json.loads(manifest.read_text())
        if package.get('schema') != 'physicalrsi.frozen-program-library/v1':
            raise ValueError('Versioned code skill manifest required')
        replacements = {'{root}': str(programs), '{python}': str(python),
                        '{evidence}': str(Path(evidence).resolve())}
        if '{framework}' in json.dumps(package):
            framework_root = Path(framework).resolve(strict=True) if framework else Path.cwd().resolve()
            if not (framework_root / 'model_template.py').is_file() or not (framework_root / 'client_server').is_dir():
                raise ValueError('Provide --framework pointing to the official XPolicyLab checkout')
            replacements['{framework}'] = str(framework_root)
        def expand(value):
            if isinstance(value, str):
                for key, replacement in replacements.items():
                    value = value.replace(key, replacement)
                return value
            if isinstance(value, list):
                return [expand(v) for v in value]
            if isinstance(value, dict):
                return {expand(k): expand(v) for k, v in value.items()}
            return value
        config['composition_definitions'] = {}
        program_registry = {}
        capability_names = {
            'align-blocks': ('edge-alignment', 'Visual geometric grounding and bounded relative-pose alignment for rigid objects and tools.'),
            'arrange-largest-number-random': ('ordered-object-placement', 'Visual symbol grounding, ordering, and reachable object placement under observed geometry.'),
            'classify-objects': ('visual-attribute-placement', 'Visual attribute grounding with category-aware localization and bounded grasp-and-place control.'),
            'classify-objects-by-language': ('language-conditioned-placement', 'Language-conditioned attribute grounding with visual localization and bounded object placement.'),
            'cover-blocks': ('footprint-relative-placement', 'Visual shape grounding and footprint-relative placement with reachable grasp control.'),
            'deposit-coin': ('opening-directed-transfer', 'Small-object and opening geometry grounding with direct or assisted transfer control.'),
            'general-pickup': ('instruction-grounded-pickup', 'Instruction-grounded object localization with appearance-conditioned grasp and reachable lifting.'),
            'insert-key': ('keyed-insertion', 'Two-object pose grounding with possession checks, alignment, insertion, and bounded contact control.'),
            'insert-tubes': ('slot-insertion', 'Object-slot geometry grounding with reachable assignment, alignment refresh, and insertion release.'),
            'make-kong': ('visual-match-press-place', 'Visual role matching with temporal identity binding, qualified contact, and upright placement.'),
            'play-tic-tac-toe': ('board-cell-placement', 'Board-state grounding with reachable cell selection and staged end-effector placement.'),
            'plug-in-charger': ('connector-alignment', 'Connector-socket geometry grounding with grasp, handoff, alignment, and contact control.'),
            'pour-balls-into-vase': ('vessel-transfer', 'Container-opening grounding with geometry-ranked grasp, supported transfer, and tilt control.'),
            'pour-by-language': ('language-conditioned-pour', 'Container transfer selected from language-specified source and destination relations, such as color, ordinal, or named roles, with tilt control.'),
            'pour-liquid-into-cup': ('spout-anchored-transfer', 'Direct bottle or spout to cup transfer from visually observed container geometry, with grasp, tilt, and restoration control.'),
            'press-by-number': ('counted-button-interaction', 'Visual symbol grounding with bounded counted contact and confirmation control.'),
            'push-t': ('planar-object-push', 'Planar object-target geometry grounding with reachable rotation and translation control.'),
            'solve-equation': ('symbol-selection-placement', 'Visual symbol and relation grounding with constrained selection and geometric placement.'),
            'sort-nesting-dolls-by-size': ('size-ordered-placement', 'Visual size grounding and ordering with reachable grasp and staged placement.'),
            'stack-blocks-by-language': ('language-conditioned-stacking', 'Language-conditioned order and relation grounding for labeled blocks, with staged block placement.'),
            'store-tools-in-toolbox': ('oriented-container-placement', 'Object and container geometry grounding with orientation-preserving transport control.'),
            'swap-t': ('identity-preserving-transfer', 'Visual identity tracking with qualified intermediate placement and identity-preserving transfer.'),
        }
        primitive_tags = {
            'align-blocks': ['visual_grounding', 'relative_pose', 'alignment_motion'],
            'arrange-largest-number-random': ['symbol_reading', 'ordering', 'grasp_place'],
            'classify-objects': ['attribute_grounding', 'object_localization', 'grasp_place'],
            'classify-objects-by-language': ['language_grounding', 'object_localization', 'grasp_place'],
            'cover-blocks': ['shape_grounding', 'footprint_relation', 'grasp_place'],
            'deposit-coin': ['object_localization', 'opening_geometry', 'grasp', 'handoff', 'insertion', 'return_home'],
            'general-pickup': ['instruction_grounding', 'appearance_grasp', 'reachable_lift'],
            'insert-key': ['object_localization', 'possession_check', 'alignment', 'insertion', 'contact_rotation'],
            'insert-tubes': ['object_localization', 'slot_geometry', 'alignment', 'insertion', 'release'],
            'make-kong': ['visual_matching', 'temporal_identity', 'push', 'handoff', 'upright_place'],
            'play-tic-tac-toe': ['board_reading', 'cell_selection', 'grasp_place'],
            'plug-in-charger': ['connector_localization', 'handoff', 'alignment', 'contact_motion'],
            'pour-balls-into-vase': ['opening_geometry', 'grasp', 'supported_transfer', 'tilt', 'return_home'],
            'pour-by-language': ['language_grounding', 'container_localization', 'tilt', 'setdown'],
            'pour-liquid-into-cup': ['container_localization', 'spout_geometry', 'grasp', 'tilt', 'return_home'],
            'press-by-number': ['symbol_reading', 'counting', 'button_contact', 'confirmation'],
            'push-t': ['planar_geometry', 'rotation_translation', 'push', 'geometry_refresh'],
            'solve-equation': ['symbol_reading', 'relation_reasoning', 'object_selection', 'grasp_place'],
            'sort-nesting-dolls-by-size': ['size_estimation', 'ordering', 'grasp_place'],
            'stack-blocks-by-language': ['language_grounding', 'object_localization', 'ordered_place'],
            'store-tools-in-toolbox': ['object_localization', 'orientation_transport', 'release', 'retract'],
            'swap-t': ['temporal_identity', 'intermediate_place', 'qualified_transfer'],
        }
        for entry in package['programs']:
            name = entry['name']
            settings = expand(entry['configuration'])
            for path, expected in settings['files'].items():
                target = Path(path).resolve(strict=True)
                if not target.is_relative_to(programs):
                    raise ValueError('Code dependency escapes installed asset root')
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError('Code dependency changed: ' + path)
            if name in config['skills']:
                raise ValueError('Duplicate skill: ' + name)
            # Keep implementations available as skills for inspection and
            # compatibility. Only the generic library below is an agent
            # composition, so these names are not offered as task choices.
            config['skills'][name] = {'name': name, 'implementation': 'code-policy',
                                      'configuration': settings}
            operation_name, description = capability_names.get(name, (name, entry['description']))
            if operation_name in program_registry:
                raise ValueError('Duplicate capability operation: ' + operation_name)
            program_registry[operation_name] = {
                'source_name': name,
                'description': description,
                'memory_primitives': primitive_tags.get(name, []),
                'action_type': entry['action_type'],
                'input': 'robodojo.observation-batch/v1',
                'output': 'robodojo.action-chunks/v1',
                'configuration': settings,
            }
        if program_registry:
            # The agent receives one operation library.  Individual programs remain
            # immutable implementation memories inside that library and are resolved
            # only after the agent has returned a contract-checked operation plan.
            config['program_registry'] = program_registry
            composition = 'memory-guided-program-library'
            config['composition_definitions'][composition] = {
                'description': 'Observation-grounded operation memory for reusable visual manipulation capabilities.',
                'action_types': sorted({entry['action_type'] for entry in program_registry.values()}),
                'action_contract': 'Each action preserves its operation representation. The evaluation client infers joint or end-effector control from the action keys.',
                'steps': ['memory.snapshot', 'memory-program-library']}
            config['compositions'].append(composition)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--framework', type=Path,
                        help='Official XPolicyLab checkout (defaults to current directory)')
    args = parser.parse_args()
    configuration = configure(**vars(args))
    print('Configured executable skills:', ', '.join(configuration['skills']))
