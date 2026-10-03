"""Bind verified code-program templates to an installation directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

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
        settings = entry['configuration']
        settings['files'] = {
            '{root}/'+relative: hashlib.sha256((root/relative).read_bytes()).hexdigest()
            for relative in entry['dependency_files']}
        programs.append({key: entry[key] for key in ('name', 'description', 'action_type', 'configuration')})
    with completed.open('x') as stream:
        json.dump({'schema': 'physicalrsi.frozen-program-library/v1', 'programs': programs},stream,indent=2)
        stream.write('\n')
    return len(programs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print('Prepared executable programs:', prepare(parser.parse_args().root))
