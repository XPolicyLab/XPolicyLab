#!/usr/bin/env python3
"""Configure the published checkpoint for the RoboTwin inference service."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True,
                        help='directory containing the full historical Automodel runtime')
    parser.add_argument('--flux2-base', type=Path, required=True)
    parser.add_argument('--qwen-vl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    root = Path(__file__).resolve().parents[1]
    runtime = args.runtime_root.resolve()
    base = args.flux2_base.resolve()
    for path in (checkpoint / 'config.yaml', checkpoint / '.complete', base / 'transformer', base / 'vae', args.qwen_vl):
        if not path.exists():
            parser.error(f'required model asset not found: {path}')
    manifest = json.loads((root / 'runtime_snapshot' / 'SHA256.json').read_text())
    if not isinstance(manifest, dict) or len(manifest) != 48:
        parser.error('expected 48 historical runtime hashes')
    runtime_source = (runtime / 'Automodel').resolve()
    for name, expected in manifest.items():
        path = (runtime_source / name).resolve()
        if not path.is_relative_to(runtime_source) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f'historical runtime file missing or different: {name}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {'checkpoint': str(checkpoint), 'model_config': str(checkpoint / 'config.yaml'), 'runtime_root': str(runtime)}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    os.environ.update({
        'FLUX2_TRANSFORMER_DIR': str(base / 'transformer'),
        'FLUX2_VAE_DIR': str(base / 'vae'),
        'FLUX2_BASE_DIR': str(base),
        'QWEN3_VL_DIR': str(args.qwen_vl.resolve()),
        'AWOMO_BIND_HOST': args.host,
    })
    src = str(runtime / 'Automodel')
    os.environ['PYTHONPATH'] = src + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.execv(sys.executable, [sys.executable, str(root / 'scripts/server.py'), str(output), str(args.worker)])


if __name__ == '__main__':
    main()
