"""Build reproducible, checksummed release assets without adding weights to Git."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import tarfile


class Parts:
    def __init__(self, root, name, base_url, part_bytes):
        self.root, self.name, self.base_url, self.limit = root, name, base_url, part_bytes
        self.parts = []
        self.output = None
        self.count = 0
    def write(self, data):
        original = len(data)
        while data:
            if self.output is None:
                filename = f'{self.name}.tar.part{len(self.parts):03d}'
                self.output = (self.root / filename).open('xb')
                self.current = dict(name=filename, url=self.base_url.rstrip('/') + '/' + filename)
                self.hash = hashlib.sha256()
                self.count = 0
            block = data[:self.limit-self.count]
            self.output.write(block)
            self.hash.update(block)
            self.count += len(block)
            data = data[len(block):]
            if self.count == self.limit:
                self.finish()
        return original
    def finish(self):
        if self.output is not None:
            self.output.close()
            self.current.update(size=self.count, sha256=self.hash.hexdigest())
            self.parts.append(self.current)
            print('Prepared', self.current['name'], flush=True)
            self.output = None
    def flush(self):
        if self.output:
            self.output.flush()


def pack(source, output, name, base_url, destination, part_bytes=1024**3):
    output.mkdir(parents=True, exist_ok=True)
    writer = Parts(output, name, base_url, part_bytes)
    def clean(info):
        if not (info.isfile() or info.isdir()):
            raise ValueError('Assets must contain regular files and directories')
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        return info
    try:
        with tarfile.open(fileobj=writer, mode='w|', dereference=True) as archive:
            for child in sorted(source.iterdir()):
                if child.name in {'__pycache__', '.git', '.venv', 'training_provenance.json'}:
                    continue
                archive.add(child, arcname=child.name, filter=clean)
    finally:
        writer.finish()
    return dict(name=name, format='tar-parts', destination=destination, parts=writer.parts)


def pack_deduplicated(source, output, name, base_url, destination, part_bytes=1024**3):
    """Store every distinct file content once, preserving the complete file tree."""
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries, inodes, blobs, directories = [], {}, {}, []
    with tempfile.TemporaryDirectory(prefix='deduplicated-', dir=output) as folder:
        root = Path(folder)
        (root / 'blobs').mkdir()
        for path in sorted(source.rglob('*')):
            if any(part in {'__pycache__', '.git', '.pytest_cache', '.venv'} for part in path.relative_to(source).parts):
                continue
            if path.is_symlink():
                raise ValueError('Materialize dependencies before deduplicated packaging')
            if path.is_dir():
                directories.append(str(path.relative_to(source)))
                continue
            if not path.is_file():
                raise ValueError('Unsupported special asset file: ' + str(path))
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            sha = inodes.get(identity)
            if sha is None:
                value = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        value.update(block)
                sha = inodes[identity] = value.hexdigest()
            if sha not in blobs:
                os.link(path, root / 'blobs' / sha)
                blobs[sha] = stat.st_size
            entries.append({'path': str(path.relative_to(source)), 'sha256': sha,
                            'mode': stat.st_mode & 0o777})
        (root / 'tree.json').write_text(json.dumps({'schema': 'physicalrsi.asset-tree/v1',
            'files': entries, 'directories': directories}, indent=2)+'\n')
        asset = pack(root, output, name, base_url, destination, part_bytes)
        asset.update(format='deduplicated-tar-parts', files=len(entries), unique_blobs=len(blobs),
                     unique_content_bytes=sum(blobs.values()))
        return asset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    asset = pack(args.source, args.output, args.name, args.base_url, args.destination)
    (args.output / (args.name + '.manifest.json')).write_text(json.dumps(asset, indent=2)+'\n')
