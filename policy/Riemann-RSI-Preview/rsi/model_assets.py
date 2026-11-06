"""Cheap completeness preflight for declared WAM assets; never load models.

Sizes catch missing shards and empty bindings. They do not replace the existing
checkpoint SHA verification or the download manifest's content validation.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(kind, *, root=ROOT):
    root = Path(root)
    if kind == 'wam_checkpoint':
        data = json.loads((root / 'policies/riemann_1/policy.json').read_text())['checkpoint_files']
    elif kind == 'encoder_path':
        manifest = json.loads((root / 'configs/models/downloads.json').read_text())
        data = {}
        for item in manifest['files']:
            if not item['id'].startswith('public-encoder:'):
                continue
            relative = item['id'].removeprefix('public-encoder:')
            if item['path'] != 'encoder/' + relative or relative in data:
                raise ValueError('Encoder download declaration differs from its resource ID')
            data[relative] = {key: item[key] for key in ('bytes', 'sha256')}
    else:
        raise ValueError('Unknown model binding: ' + kind)
    if not data:
        raise ValueError('Model assets require explicit nonempty file declarations')
    return data


def verify(*, wam_checkpoint=None, encoder_path=None, root=ROOT):
    from .model_records import verify_records
    checked = {}
    for kind, directory in (('wam_checkpoint', wam_checkpoint), ('encoder_path', encoder_path)):
        if directory is not None:
            directory = Path(directory)
            if not directory.is_dir():
                raise ValueError('Missing asset directory: ' + str(directory))
            checked[kind] = verify_records(directory, records(kind, root=root), deep=False)
    return {'files_size_checked': checked, 'content_hashes_verified': False, 'models_loaded': False}


def verify_rule(*, root=ROOT):
    """Check completeness of all common Rule models without loading weights."""
    from .resource_locations import load_manifest
    from .model_records import verify_records
    root=Path(root)
    sam=load_manifest(root/'configs/models/sam3.json')
    models=load_manifest(root/'tasks/classify_objects/resources.json')['models']
    declarations={'sam3':sam,'clip':models['clip_model_manifest.json'],
                  'grounding_dino':models['grounding_model_manifest.json']}
    checked={name:verify_records(Path(item['path']),item['files'],deep=False)
             for name,item in declarations.items()}
    return dict(files_size_checked=checked,content_hashes_verified=False,models_loaded=False)
