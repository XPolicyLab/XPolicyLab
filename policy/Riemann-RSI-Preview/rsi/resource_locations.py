"""Relocate declared external inputs; callers retain their content checks."""
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .deployment import absolute, local, locations

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def catalog():
    data = json.loads((ROOT / 'configs/resource-roots.json').read_text())
    if data.get('schema') != 'riemann-resource-roots-v2' or set(data) != {'schema', 'roots'}:
        raise ValueError('Unknown resource root declaration')
    for name, record in data['roots'].items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
            raise ValueError('Invalid resource root identity')
        if set(record) != {'default'} or not isinstance(record.get('default'), str):
            raise ValueError('Resource roots declare locations only')
    return data


def aliases(environment=None):
    """Optional machine-local relocation of immutable external legacy records."""
    environment = os.environ if environment is None else environment
    values = dict(local(ROOT).get('resource_aliases', {}))
    explicit = json.loads(environment.get('RIEMANN_RESOURCE_ALIASES', '{}'))
    if not isinstance(explicit, dict): raise ValueError('Resource aliases must be a location mapping')
    values.update(explicit)
    if set(values) - set(catalog()['roots']): raise ValueError('Unknown resource alias identity')
    for paths in values.values():
        if not isinstance(paths, list): raise ValueError('Resource aliases must be path lists')
        for path in paths: absolute(path)
    return values


def overrides(environment=None):
    environment = os.environ if environment is None else environment
    values = dict(local(ROOT).get('resource_roots', {}))
    explicit = json.loads(environment.get('RIEMANN_RESOURCE_ROOTS', '{}'))
    if not isinstance(explicit, dict): raise ValueError('Resource roots must be a location mapping')
    values.update(explicit)
    if set(values) - set(catalog()['roots']):
        raise ValueError('Resource overrides require declared root identities')
    return {name: absolute(value) for name, value in values.items()}


def _relative_reference(value):
    parsed = urlsplit(value); relative = parsed.path[1:]
    if (parsed.scheme != 'resource' or parsed.netloc not in catalog()['roots']
            or parsed.query or parsed.fragment or '%' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts):
        raise ValueError('Invalid declared resource reference: ' + value)
    return parsed.netloc, relative


def roots(environment=None):
    environment = os.environ if environment is None else environment
    binding = locations(ROOT, environment=environment); explicit = overrides(environment)
    resolved = {}; pending = set()
    def resolve(name):
        if name in resolved: return resolved[name]
        if name in pending: raise ValueError('Cyclic resource location declaration')
        pending.add(name)
        value = catalog()['roots'][name]['default']
        if name in explicit:
            result = explicit[name]
        elif value.startswith('deployment:'):
            key = value.removeprefix('deployment:')
            if key not in ('official_root','eval_prefix','policy_prefix','encoder_path'):
                raise ValueError('Unknown deployment resource location')
            result = absolute(binding[key])
        elif value.startswith('resource:'):
            parent, relative = _relative_reference(value); result = resolve(parent) / relative
        else:
            relative = PurePosixPath(value)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Default resources must be contained below resources_base')
            result = absolute(binding['resources_base']) / relative
        result = result.resolve()
        resolved[name] = result; pending.remove(name)
        return result
    for name in catalog()['roots']: resolve(name)
    return resolved


def relocate_path(value, *, environment=None):
    """Resolve logical IDs or historical manifest locations idempotently."""
    value = os.fspath(value)
    if '..' in Path(value).parts: raise ValueError('Resource paths cannot contain parent traversal')
    configured = roots(environment)
    if value.startswith('resource:'):
        name, relative = _relative_reference(value)
        return str(configured[name] / relative)
    path = Path(value)
    if not path.is_absolute(): return value
    matches = []
    historical = aliases(environment)
    for name, target in configured.items():
        # Prefer an already resolved location over an equally long alias.
        try:
            path.relative_to(target); matches.append((len(target.parts), 1, value))
        except ValueError: pass
        for alias in historical.get(name, []):
            prefix = Path(alias)
            try: relative = path.relative_to(prefix)
            except ValueError: continue
            matches.append((len(prefix.parts), 0, str(target / relative)))
    return max(matches, key=lambda row: row[:2])[2] if matches else value


def reference_for(value, *, environment=None):
    value = os.fspath(value)
    if '..' in Path(value).parts: raise ValueError('Resource paths cannot contain parent traversal')
    if value.startswith('resource:'):
        _relative_reference(value); return value
    matches = []
    historical = aliases(environment)
    for name, target in roots(environment).items():
        for prefix in (target, *(Path(x) for x in historical.get(name, []))):
            try: relative = Path(value).relative_to(prefix)
            except ValueError: continue
            matches.append((len(prefix.parts), 'resource://' + name + '/' + relative.as_posix()))
    return max(matches, key=lambda row: row[0])[1] if matches else value


def relocate_manifest(value, *, environment=None):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            target = relocate_path(key, environment=environment) if isinstance(key,str) and key.startswith(('/', 'resource:')) else key
            item = relocate_manifest(item, environment=environment)
            if target in result and result[target] != item:
                raise ValueError('Conflicting relocated resource declarations: ' + str(target))
            result[target] = item
        return result
    if isinstance(value,list): return [relocate_manifest(v, environment=environment) for v in value]
    if isinstance(value,str) and value.startswith(('/', 'resource:')): return relocate_path(value, environment=environment)
    return value


def load_manifest(path, *, environment=None):
    return relocate_manifest(json.loads(Path(path).read_text()), environment=environment)
