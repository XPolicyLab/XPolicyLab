"""Materialize declared public resources without changing the SDK or environments."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def contained(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Resource member must stay inside its destination')
    root = Path(root).resolve()
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root):
        raise ValueError('Resource destination escapes through a symlink')
    for parent in (result, *result.parents):
        if parent == root: break
        if parent.is_symlink(): raise ValueError('Resource destinations cannot be symlinks')
    return result



def official_file(root, relative):
    """Official installations may link geometry outside their Assets tree.

    Only the source can follow those installer-owned links. Its bytes still
    have to match the declared public digest before any output is published.
    """
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Invalid official resource member')
    path = Path(root).joinpath(*relative.parts)
    if not path.is_file(): raise ValueError('Missing official asset: ' + str(relative))
    return path

def verify(path, record):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): return False
    if path.stat().st_size != record['bytes']: return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']


def put(root, relative, content, record):
    path = contained(root, relative)
    if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError('Resource content identity differs: ' + relative)
    if path.exists():
        if not verify(path, record): raise ValueError('Preserve nonmatching resource: ' + relative)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.resource-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
            # Publish without replacing a racing writer.
            stream.close(); path.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)


def extract_tar(archive, root, members):
    """Require the exact regular member set; do not use tar.extractall."""
    with tarfile.open(archive, 'r:gz') as source:
        seen = set()
        for item in source:
            if item.name not in members or item.name in seen or not item.isfile():
                raise ValueError('Unexpected, repeated or nonregular archive member')
            record = members[item.name]
            if item.size != record['bytes']: raise ValueError('Archive member size differs')
            seen.add(item.name)
            content = source.extractfile(item).read(record['bytes'] + 1)
            for path in record['paths']: put(root, path, content, record)
        if seen != set(members): raise ValueError('Archive is missing declared members')


def finger_triangles(sources):
    """The original numeric triangles are the raw binary public STL vertices."""
    import numpy as np
    arrays = {}
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    for index, source in zip((7, 8), sources):
        raw = Path(source).read_bytes()
        if len(raw) < 84: raise ValueError('Truncated public binary STL')
        count = struct.unpack('<I', raw[80:84])[0]
        if len(raw) != 84 + count * 50: raise ValueError('Unexpected public binary STL length')
        arrays['link' + str(index)] = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].copy()
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
    return stream.getvalue()



def official_records(root=ROOT):
    """Derive the static public asset subset from current task declarations."""
    from .resource_locations import reference_for
    from .policy_catalog import catalog
    root=Path(root); found={}; prefix='resource://official_assets/'
    def add(path, digest):
        if not isinstance(path,str) or not isinstance(digest,str): return
        reference=reference_for(path)
        if not reference.startswith(prefix): return
        member=reference.removeprefix(prefix)
        if member in found and found[member]!=digest:
            raise ValueError('Conflicting official asset identities: '+member)
        found[member]=digest
    def walk(value):
        if isinstance(value,dict):
            path=value.get('resolved_path',value.get('path'))
            if isinstance(path,str):
                add(path,value.get('sha256'))
                members=value.get('files',{})
                for name,rec in (members.items() if isinstance(members,dict) else ()):
                    if isinstance(rec,str): add(path.rstrip('/')+'/'+name,rec)
                    elif isinstance(rec,dict): add(path.rstrip('/')+'/'+name,rec.get('sha256'))
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    for mode in catalog(root):
        filename=root/'tasks'/mode/'resources.json'
        if not filename.is_file():continue
        data=json.loads(filename.read_text());walk(data)
        urdf=data.get('public_robot_urdf',{}).get('path')
        if urdf:
            parent = (urdf.rsplit('/', 1)[0] if urdf.startswith('resource://')
                      else str(PurePosixPath(urdf).parent))
            for name,digest in data.get('public_robot_meshes',{}).items():
                add(parent + '/' + name,digest)
    lock=json.loads((root/'configs/official-runtime-inputs.json').read_text())
    for path,digest in lock['external_files'].items():add(path,digest)
    return found

def prepare(destination, *, official_assets, root=ROOT, check_only=False):
    root, destination, official_assets = Path(root), Path(destination).resolve(), Path(official_assets).resolve()
    plan = json.loads((root / 'configs/resource-install.json').read_text())
    if plan['schema'] != 'riemann-resource-install-v1': raise ValueError('Unknown resource install declaration')
    files = list(plan['files'])
    asset_directory = json.loads((root/'configs/resource-roots.json').read_text())['roots']['official_assets']['default']
    for member,digest in official_records(root).items():
        source=official_file(official_assets,member)
        files.append(dict(path=asset_directory+'/'+member,sha256=digest,bytes=source.stat().st_size,
                          kind='official',source='resource://official_assets/'+member))
    sam = json.loads((root / 'configs/models/sam3.json').read_text())
    runtime = plan['sam_runtime']
    for name, record in sam['files'].items():
        if name != 'sam3.pt':
            files.append(dict(record, path=runtime['target'] + '/' + name, kind='sam_runtime', member=name))
    pending = []
    for record in files:
        path = contained(destination, record['path'])
        if verify(path, record): continue
        if path.exists(): raise ValueError('Preserve nonmatching resource: ' + record['path'])
        pending.append(record)
    if check_only: return dict(complete=not pending, pending=[x['path'] for x in pending], files_checked=len(files))
    # Validate archive identities before opening any members.
    if any(x['kind'] == 'cached' for x in pending):
        archive = plan['archive']; path = contained(destination, archive['path'])
        if not verify(path, archive): raise ValueError('Download the exact template archive first')
        members = {}
        for item in files:
            if item['kind'] != 'cached': continue
            old = members.setdefault(item['object'], dict(bytes=item['bytes'], sha256=item['sha256'], paths=[]))
            if (old['bytes'], old['sha256']) != (item['bytes'], item['sha256']): raise ValueError('Conflicting resource object')
            old['paths'].append(item['path'])
        extract_tar(path, destination, members)
    if any(x['kind'] == 'sam_runtime' for x in pending):
        if not verify(contained(destination, runtime['path']), runtime): raise ValueError('Download the exact SAM runtime first')
        members = {x['member']:dict(bytes=x['bytes'],sha256=x['sha256'],paths=[x['path']])
                   for x in files if x['kind']=='sam_runtime'}
        extract_tar(contained(destination,runtime['path']),destination,members)
    for item in pending:
        kind = item['kind']
        if kind in ('cached','sam_runtime'): continue
        if kind == 'official':
            prefix = 'resource://official_assets/'
            if not item['source'].startswith(prefix): raise ValueError('Unknown official resource origin')
            source = official_file(official_assets,item['source'].removeprefix(prefix))
            put(destination,item['path'],source.read_bytes(),item)
        elif kind == 'download':
            put(destination,item['path'],contained(destination,item['source']).read_bytes(),item)
        elif kind == 'finger_triangles':
            prefix = 'resource://official_assets/'
            if any(not x.startswith(prefix) for x in item['sources']): raise ValueError('Unknown geometry origin')
            sources = [official_file(official_assets,x.removeprefix(prefix)) for x in item['sources']]
            # Each original official STL has its own hash in the current task contract.
            put(destination,item['path'],finger_triangles(sources),item)
        elif kind == 'wheel':
            wheel = contained(destination,item['source'])
            if not verify(wheel,plan['protocol_wheel']): raise ValueError('Protocol wheel changed')
            with zipfile.ZipFile(wheel) as archive:
                member=archive.getinfo(item['member'])
                if member.file_size != item['bytes']: raise ValueError('Wheel member size differs')
                put(destination,item['path'],archive.read(member),item)
        else: raise ValueError('Unknown resource preparation method: '+kind)
    result = prepare(destination,official_assets=official_assets,root=root,check_only=True)
    if not result['complete']: raise ValueError('Resource preparation did not complete')
    return result


@contextmanager
def relocated_robot_yaml(filename):
    """Bind public geometry locations in an owned temporary planner configuration."""
    import yaml
    from .resource_locations import relocate_manifest
    filename = Path(filename)
    data = relocate_manifest(yaml.safe_load(filename.read_text()))
    with tempfile.TemporaryDirectory(prefix='riemann-robot-') as directory:
        output=Path(directory)/filename.name
        output.write_text(yaml.safe_dump(data))
        yield output


def robot_yaml_digest(filename):
    """Hash YAML bytes after normalizing only declared absolute resource paths.

    Numeric configuration, whitespace, collision fields and every other byte
    remain significant. This permits the identical official asset at a new root.
    """
    import re
    from .resource_locations import reference_for
    raw=Path(filename).read_text()
    normalized=re.sub(r"/(?:[^\s\'\"\],}#]+)",lambda match:reference_for(match[0]),raw)
    return hashlib.sha256(normalized.encode()).hexdigest()
