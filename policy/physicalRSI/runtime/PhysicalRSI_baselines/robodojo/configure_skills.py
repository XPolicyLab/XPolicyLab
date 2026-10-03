"""Create an evaluation configuration from installed, explicitly identified assets."""
import argparse
import hashlib
import json
from pathlib import Path


def configure(assets, *, python, output, evidence, endpoint, model):
    assets = Path(assets).resolve(strict=True)
    python = Path(python).resolve(strict=True)
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
        for entry in package['programs']:
            name = entry['name']
            if name in config['skills']:
                raise ValueError('Duplicate skill: ' + name)
            settings = expand(entry['configuration'])
            for path, expected in settings['files'].items():
                target = Path(path).resolve(strict=True)
                if not target.is_relative_to(programs):
                    raise ValueError('Code dependency escapes installed asset root')
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError('Code dependency changed: ' + path)
            config['skills'][name] = {'name': name, 'implementation': 'code-policy',
                                      'configuration': settings}
            composition = 'memory-guided-' + name
            config['composition_definitions'][composition] = {
                'description': entry['description'],
                'action_type': entry['action_type'],
                'steps': ['memory.snapshot', name]}
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
    args = parser.parse_args()
    configuration = configure(**vars(args))
    print('Configured executable skills:', ', '.join(configuration['skills']))
