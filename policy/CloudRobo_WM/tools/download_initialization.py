"""Verify the initialization artifact and emit the packaged launcher's input receipt."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-B', str(root / 'download_checkpoint.py'),
               '--manifest', str(root / 'initialization_manifest.json'), '--output-dir', str(args.output_dir)]
    if args.source_dir:
        command += ['--source-dir', str(args.source_dir)]
    if args.verify_only:
        command += ['--verify-only']
    subprocess.run(command, check=True)
    row = json.loads((root / 'initialization_manifest.json').read_text())['files'][0]
    target = args.output_dir / row['path']
    with target.open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(source.read(length))
    if '__metadata__' in header or len(header) != 2509 or any(v['dtype'] != 'BF16' for v in header.values()):
        raise ValueError('Initialization must contain 2509 BF16 tensors without metadata')
    receipt = dict(complete=True, sha256=row['sha256'], bytes=row['bytes'], tensor_count=len(header),
                   verification='Pinned complete-file SHA256 and safetensors header',
                   purpose='post-training initialization; no training execution claimed')
    path = target.with_name(target.name + '.manifest.json')
    text = json.dumps(receipt, indent=2) + '\n'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('complete') is not True or existing.get('sha256') != row['sha256']:
            raise ValueError('Existing initialization receipt differs; inspect it before replacing')
    else:
        with path.open('x') as stream:
            stream.write(text)
    print('Verified initialization receipt:', path)


if __name__ == '__main__':
    main()
