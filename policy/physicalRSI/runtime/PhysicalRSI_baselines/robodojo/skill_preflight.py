"""Check executable skill dependencies without loading weights or using API keys."""
import argparse
import importlib
import json
import os
from pathlib import Path
from .execution_skills import load_skill


def inspect(path):
    config = json.loads(Path(path).read_text())
    errors = []
    if config.get('schema') != 'physicalrsi.skill-library/v1':
        errors.append('Expected physicalrsi.skill-library/v1')
    catalog = json.loads((Path(__file__).parent / 'configs/skill_compositions.json').read_text())['compositions']
    extensions = config.get('composition_definitions', {})
    if set(extensions) & set(catalog):
        errors.append('Custom compositions cannot replace built-in definitions')
    catalog.update(extensions)
    for name in config.get('compositions', []):
        if name not in catalog or catalog[name]['steps'][-1] not in config.get('skills', {}):
            errors.append('Missing executable composition: ' + name)
    if not config.get('compositions'):
        errors.append('No skill compositions configured')
    for name, entry in config.get('skills', {}).items():
        if entry.get('name') != name:
            errors.append('Skill descriptor identity mismatch: ' + name)
        implementation = entry.get('implementation', name)
        settings = entry.get('configuration', {})
        if implementation in ('pi05', 'pi05-sparse-memory'):
            checkpoint = Path(settings.get('model_path', '/missing-checkpoint'))
            if not checkpoint.is_dir() or not (checkpoint / 'params').exists():
                errors.append('Missing checkpoint params: ' + str(checkpoint))
            module = 'pi05' if implementation == 'pi05' else 'sparse'
            try:
                importlib.import_module('PhysicalRSI_baselines.robodojo.skills.' + module)
                training = importlib.import_module('PhysicalRSI_baselines.robodojo.skills.inference_configs')
                training.get_config(settings['train_config_name'], checkpoint)
            except Exception as exc:
                errors.append(f'{name} import failed: {type(exc).__name__}: {exc}')
        elif implementation == 'code-policy':
            from .code_policy_factory import verify_spec
            if settings.get('runtime') == 'frozen-program-service':
                from PhysicalRSI_core.infra.storage import file_digest
                try:
                    if not settings.get('command') or not settings.get('files'):
                        raise ValueError('Frozen command and file hashes required')
                    for path, expected in settings['files'].items():
                        if file_digest(Path(path)) != expected:
                            raise ValueError('Code skill dependency changed: ' + path)
                    if not Path(settings['cwd']).is_dir():
                        raise ValueError('Code skill working directory missing')
                except Exception as exc:
                    errors.append(f'code-policy service: {exc}')
                continue
            try:
                verify_spec(settings['spec'])
                for key in ('harness', 'binding', 'output', 'resources'):
                    if key not in settings:
                        errors.append('code-policy missing ' + key)
            except Exception as exc:
                errors.append(f'code-policy specification: {type(exc).__name__}: {exc}')
        else:
            errors.append('Unknown skill: ' + name)
    if not any(os.environ.get(key) for key in ('PHYSICALRSI_AGENT_API_KEY', 'OPENAI_API_KEY', 'ARK_API_KEY')):
        errors.append('Missing agent API key')
    agent = config.get('agent', {})
    if not (agent.get('endpoint') or os.environ.get('PHYSICALRSI_AGENT_ENDPOINT')):
        errors.append('Missing agent endpoint')
    if not (agent.get('model') or os.environ.get('PHYSICALRSI_AGENT_MODEL')):
        errors.append('Missing agent model')
    return dict(schema='physicalrsi.skill-preflight/v1', ready=not errors, errors=errors,
                qualification=False, scope='dependency checks; no policy inference or simulator execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration')
    args = parser.parse_args()
    result = inspect(args.configuration)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
