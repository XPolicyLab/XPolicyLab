"""Assemble the portable skill runtime source tree for an evaluation release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def build(destination):
    repository = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new source export directory')
    checksums = {}
    for package in ('PhysicalRSI', 'PhysicalRSI_core', 'PhysicalRSI_baselines'):
        source = repository / package
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if any(part in {'releases', '__pycache__', '.pytest_cache'} for part in relative.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix not in {'.py', '.json', '.yml', '.yaml', '.sh', '.md'} and not path.name.startswith(('LICENSE', 'NOTICE')):
                continue
            target = destination / package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            checksums[str(target.relative_to(destination))] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ('LICENSE', 'NOTICE'):
        if (repository / name).is_file():
            shutil.copyfile(repository / name, destination / name)
            checksums[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    (destination / 'files.sha256.json').write_text(json.dumps(checksums, indent=2)+'\n')
    return len(checksums)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print('Exported source files:', build(args.destination))
