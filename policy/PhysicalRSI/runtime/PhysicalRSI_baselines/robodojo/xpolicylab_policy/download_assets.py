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


# A default release reference is replaceable; asset inventories belong to releases.
DEFAULT_MANIFEST = 'https://github.com/PhysicalRSI/XPolicyLab/releases/download/physicalrsi-skill-assets-v4/assets.json'
DEFAULT_MANIFEST_SHA256 = '0eed83202427dc6b4c75cde613246f8f6e606cac6c8da3bd71ebc6798d6963d2'


def load_manifest(source, output, expected_sha256=None):
    """Fetch release metadata and retain its exact bytes with the downloaded assets."""
    if str(source).startswith(('https://', 'http://', 'file://')):
        with urllib.request.urlopen(str(source), timeout=120) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    else:
        raw = Path(source).read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Asset manifest is too large')
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Manifest checksum mismatch')
    manifest = json.loads(raw)
    assets = manifest.get('assets')
    if not isinstance(assets, list) or not assets:
        raise ValueError('A nonempty asset manifest is required')
    names = [asset['name'] for asset in assets]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate asset names')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'assets.json'
    if target.exists() and target.read_bytes() != raw:
        raise FileExistsError('Different manifest already installed; use a new output directory')
    if not target.exists():
        with target.open('xb') as stream:
            stream.write(raw)
    provenance = output / 'assets.provenance.json'
    if not provenance.exists():
        with provenance.open('x') as stream:
            json.dump({'source': str(source), 'sha256': digest}, stream, indent=2)
    return manifest


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
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST, help='Release manifest URL or local path')
    parser.add_argument('--manifest-sha256', help='Expected SHA-256; the default release is pinned automatically')
    parser.add_argument('--output', type=Path, default=Path('skill-assets'))
    parser.add_argument('--asset', action='append', help='Select one or more asset names; default all')
    args = parser.parse_args()
    expected = args.manifest_sha256 or (DEFAULT_MANIFEST_SHA256 if args.manifest == DEFAULT_MANIFEST else None)
    manifest = load_manifest(args.manifest, args.output, expected)
    selected = set(args.asset or [asset['name'] for asset in manifest['assets']])
    available = {asset['name'] for asset in manifest['assets']}
    if selected - available:
        parser.error('Unknown asset: ' + ', '.join(sorted(selected - available)))
    for asset in manifest['assets']:
        if asset['name'] in selected:
            print(install(asset, args.output))
