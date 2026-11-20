"""Hash-checked static public resources, stored outside the source repository."""
from rsi.resource_locations import load_manifest as load_resource_manifest
import hashlib
import json
from pathlib import Path


def asset_root(task_root, manifest=None):
    task_root = Path(task_root)
    manifest = load_resource_manifest(task_root / 'resources.json') if manifest is None else manifest
    return Path(manifest['default_root']) / task_root.name


def resource(task_root, name):
    task_root=Path(task_root)
    manifest=load_resource_manifest(task_root/'resources.json')
    base=asset_root(task_root, manifest)
    record=manifest['files'][name]
    path=base/record['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Missing or changed public rule asset: '+str(path))
    return path


def portable_coin_assets(task_root, destination):
    """Only relocate the pinned public URDF path; leave geometry unchanged."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    metadata=json.loads(resource(task_root,'public_assets.json').read_text())
    metadata['urdf']['path']=str(resource(task_root,'X5A.urdf'))
    (destination/'public_assets.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (destination/'public_templates.npz').symlink_to(resource(task_root,'public_templates.npz'))
    return destination
