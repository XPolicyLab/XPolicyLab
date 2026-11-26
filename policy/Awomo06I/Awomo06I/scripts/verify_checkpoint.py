#!/usr/bin/env python3
"""Verify every file against the published checkpoint SHA256SUMS."""
import argparse
import hashlib
from pathlib import Path


PUBLISHED_MANIFEST_SHA256 = 'aee13441a40a1c96800352257e5b2a737e91e3ea7b30396d834a0209bced6e25'


def verify_checkpoint(checkpoint, manifest_sha256=PUBLISHED_MANIFEST_SHA256):
    root = Path(checkpoint).resolve()
    manifest_bytes = (root / 'SHA256SUMS').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError('checkpoint manifest does not match the published release')
    count = 0
    seen = set()
    for line in manifest_bytes.decode().splitlines():
        expected, name = line.split(maxsplit=1)
        if name in seen or len(expected) != 64:
            raise ValueError(f'invalid or duplicate checkpoint member: {name}')
        seen.add(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'invalid checkpoint member: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'checksum mismatch: {name}')
        count += 1
    if count != 73:
        raise ValueError(f'expected 73 files, got {count}')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--manifest-sha256', default=PUBLISHED_MANIFEST_SHA256)
    args = parser.parse_args()
    count = verify_checkpoint(args.checkpoint, manifest_sha256=args.manifest_sha256)
    print(f'Verified {count} checkpoint files')


if __name__ == '__main__':
    main()
