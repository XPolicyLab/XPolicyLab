"""Download, verify and install versioned skill archives from a release manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


def relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Asset paths must be safe relative paths')
    return path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def install(asset, root):
    if asset['format'] not in {'tar-parts', 'deduplicated-tar-parts'} or not asset['parts']:
        raise ValueError('A nonempty tar-parts asset is required')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / relative(asset['destination'])
    if destination.exists():
        raise FileExistsError('Refusing to replace existing asset: ' + str(destination))
    if not destination.resolve().is_relative_to(root):
        raise ValueError('Asset destination escapes installation root')
    cache = root / '.downloads'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='skill-', dir=root) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / 'asset.tar'
        with archive_path.open('wb') as assembled:
            for part in asset['parts']:
                name = relative(part['name'])
                if len(name.parts) != 1 or len(part['sha256']) != 64:
                    raise ValueError('Invalid asset part identity')
                target = cache / name
                if not target.exists() or sha256(target) != part['sha256']:
                    partial = temporary / name
                    with urllib.request.urlopen(part['url'], timeout=120) as response, partial.open('wb') as output:
                        shutil.copyfileobj(response, output, 1024**2)
                    if partial.stat().st_size != part['size'] or sha256(partial) != part['sha256']:
                        raise ValueError('Asset checksum or size mismatch: ' + str(name))
                    partial.replace(target)
                with target.open('rb') as source:
                    shutil.copyfileobj(source, assembled, 1024**2)
        unpacked = temporary / 'unpacked'
        unpacked.mkdir()
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                relative(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Links and special archive entries are forbidden')
            archive.extractall(unpacked, filter='data')
        if asset['format'] == 'deduplicated-tar-parts':
            tree = json.loads((unpacked / 'tree.json').read_text())
            if tree.get('schema') != 'physicalrsi.asset-tree/v1':
                raise ValueError('Versioned asset tree required')
            restored = temporary / 'restored'
            restored.mkdir()
            for directory in tree.get('directories', []):
                (restored / relative(directory)).mkdir(parents=True, exist_ok=True)
            verified = set()
            installed = {}
            for entry in tree['files']:
                path = relative(entry['path'])
                sha = entry['sha256']
                if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('Invalid asset blob identity')
                blob = unpacked / 'blobs' / sha
                if sha not in verified:
                    if sha256(blob) != sha:
                        raise ValueError('Asset blob checksum mismatch')
                    verified.add(sha)
                target = restored / path
                target.parent.mkdir(parents=True, exist_ok=True)
                mode = entry['mode']
                if type(mode) is not int or mode < 0 or mode > 0o777:
                    raise ValueError('Invalid asset file permissions')
                identity = (sha, mode)
                if identity in installed:
                    os.link(installed[identity], target)
                else:
                    with target.open('xb') as stream, blob.open('rb') as source:
                        shutil.copyfileobj(source, stream, 1024**2)
                    target.chmod(mode)
                    installed[identity] = target
            unpacked = restored
        destination.parent.mkdir(parents=True, exist_ok=True)
        unpacked.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('assets.json'))
    parser.add_argument('--output', type=Path, default=Path('skill-assets'))
    parser.add_argument('--asset', action='append', help='Select one or more asset names; default all')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = set(args.asset or [asset['name'] for asset in manifest['assets']])
    available = {asset['name'] for asset in manifest['assets']}
    if selected - available:
        parser.error('Unknown asset: ' + ', '.join(sorted(selected - available)))
    for asset in manifest['assets']:
        if asset['name'] in selected:
            print(install(asset, args.output))
