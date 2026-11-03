"""Bind verified code-program templates to an installation directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ''):
    # Support the shipped script from any working directory, including a clean
    # interpreter before the evaluation environment has been installed.
    runtime_root = Path(__file__).resolve().parents[2]
    if not (runtime_root / 'PhysicalRSI_core' / 'infra' / 'storage.py').is_file():
        raise RuntimeError('Run preparation from the complete PhysicalRSI runtime tree')
    sys.path.insert(0, str(runtime_root))

from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI.Embodied_Harness.skills.episode_plan import catalogue

MARKER = '/PHYSICALRSI_PROGRAM_ASSETS'


def prepare(root):
    root = Path(root).resolve(strict=True)
    if any(c in str(root) for c in ('"', "'", '\n', '\\')):
        raise ValueError('Program installation path must not contain quotes, backslashes or newlines')
    manifest = root / 'program-templates.json'
    template = json.loads(manifest.read_text())
    if template.get('schema') != 'physicalrsi.program-templates/v1':
        raise ValueError('Versioned program templates required')
    completed = root / 'programs.json'
    if completed.exists():
        raise FileExistsError('Code programs already prepared; use a fresh asset installation')
    verified = {}
    replacements = {}
    relocated = {}
    for relative, expected in template['text_files'].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Program source escapes installation directory')
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in verified:
            content = path.read_bytes()
            verified[identity] = hashlib.sha256(content).hexdigest()
            if MARKER.encode() in content:
                relocated[identity] = content.replace(MARKER.encode(), str(root).encode())
        if identity in relocated:
            replacements[relative] = relocated[identity]
        if verified[identity] != expected:
            raise ValueError('Program template changed: ' + relative)
    # Replace whole files so shared content in the asset store stays immutable.
    for relative, after in replacements.items():
        path = root / relative
        temporary = path.with_name(path.name+'.installing')
        temporary.write_bytes(after)
        shutil.copymode(path, temporary)
        temporary.replace(path)
    programs = []
    for entry in template['programs']:
        for relative in entry['dependency_files']:
            if not (root / relative).resolve(strict=True).is_relative_to(root):
                raise ValueError('Program dependency escapes installation directory')
        if 'worker_descriptor' in entry:
            relative = entry['worker_descriptor']
            path = (root / relative).resolve(strict=True)
            if (not path.is_relative_to(root) or relative not in template['text_files']
                    or relative not in entry['dependency_files']):
                raise ValueError('Worker descriptor must be a verified program dependency')
            descriptor = json.loads(path.read_text())
            if descriptor.get('schema') != 'physicalrsi.skill-worker/v1':
                raise ValueError('Versioned worker descriptor required')
            identity = descriptor['implementation']
            source = identity['source']
            if not isinstance(source, str) or not source.startswith('{assets}/'):
                raise ValueError('Worker implementation must be relative to the asset root')
            source_relative = source.removeprefix('{assets}/')
            source_path = (root / source_relative).resolve(strict=True)
            if (not source_path.is_relative_to(root) or source_relative not in template['text_files']
                    or source_relative not in entry['dependency_files']):
                raise ValueError('Worker implementation must be a verified program dependency')
            if identity['sha256'] != template['text_files'][source_relative]:
                raise ValueError('Worker source identity differs from verified template')
            identity['sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            initialization = descriptor.get('transport', {}).get('initialization', {})
            if 'controller_source' in initialization:
                controller = initialization['controller_source']
                if not isinstance(controller, str) or not controller.startswith('{assets}/'):
                    raise ValueError('Startup controller must be relative to the asset root')
                controller_relative = controller.removeprefix('{assets}/')
                controller_path = (root / controller_relative).resolve(strict=True)
                if (not controller_path.is_relative_to(root)
                        or controller_relative not in template['text_files']
                        or controller_relative not in entry['dependency_files']):
                    raise ValueError('Startup controller must be a verified program dependency')
                if initialization['controller_sha256'] != template['text_files'][controller_relative]:
                    raise ValueError('Startup controller differs from verified template')
                initialization['controller_sha256'] = hashlib.sha256(controller_path.read_bytes()).hexdigest()
            temporary = path.with_name(path.name + '.installing')
            temporary.write_text(json.dumps(descriptor, indent=2) + '\n')
            shutil.copymode(path, temporary)
            temporary.replace(path)
        settings = entry['configuration']
        settings['files'] = {
            '{root}/'+relative: hashlib.sha256((root/relative).read_bytes()).hexdigest()
            for relative in entry['dependency_files']}
        if 'operation_memory_files' in entry:
            memories = {}
            for relative in entry['operation_memory_files']:
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root) or relative not in template['text_files']:
                    raise ValueError('Operation memory must be a verified asset file')
                document = json.loads(path.read_text())
                if document.get('schema') == 'physicalrsi.structured-module/v1':
                    classes = document.get('classes')
                    if not isinstance(classes, dict) or not classes:
                        raise ValueError('Structured module requires class memory documents')
                    documents = classes.values()
                else:
                    documents = [document]
                for memory in documents:
                    catalogue(memory)
                    memories[digest(memory)] = memory
            if not memories:
                raise ValueError('Declared operation memory cannot be empty')
            settings['operation_memories'] = memories
        programs.append({key: entry[key] for key in ('name', 'description', 'action_type', 'configuration')})
    with completed.open('x') as stream:
        json.dump({'schema': 'physicalrsi.frozen-program-library/v1', 'programs': programs},stream,indent=2)
        stream.write('\n')
    return len(programs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print('Prepared executable programs:', prepare(parser.parse_args().root))
