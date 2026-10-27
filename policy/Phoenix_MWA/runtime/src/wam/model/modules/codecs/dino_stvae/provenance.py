"""Bundled encoder paths. Runtime file fingerprints are not required."""
from pathlib import Path

BUNDLED_SOURCE = "bundled://dino-stvae-v1"
PACKAGE = Path(__file__).resolve().parent
WAM_ROOT = PACKAGE.parents[3]
RUNTIME = WAM_ROOT / "third_party/dino_runtime"


def verify_bundle():
    """Retained API name; check required source files, not their hashes."""
    for path in (PACKAGE / 'online_dinov3_stvae_encoder.py', PACKAGE / 'stvae.py',
                 RUNTIME / 'STARVLA-LICENSE', RUNTIME / 'dinov3/LICENSE.md'):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {'schema': 'wam.bundled_encoder.v1', 'source_sha256': None, 'hash_validation': False}


def bundled_config(config):
    import copy
    result = copy.deepcopy(dict(config))
    verify_bundle()
    result.update(source_root=BUNDLED_SOURCE, source_sha256=None)
    result['dino_kwargs'] = dict(result['dino_kwargs'])
    result['dino_kwargs'].update(starvla_source_path=str(RUNTIME), dinov3_source_path=str(RUNTIME / 'dinov3'))
    return result


def canonical_source_identity(encoder):
    """Compatibility metadata only; never a runtime admission check."""
    return None
