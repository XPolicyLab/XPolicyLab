from copy import deepcopy
import json
from cloudrobo_wm.storage import DataPath as Path
from .curation import VERSION, validate_policy
from .schema import file_sha256, fingerprint, within

def verify_curation(root, path, expected_sha256=None, *, require_visual=True):
    root, path = (Path(root).resolve(), Path(path))
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError('Curation report changed; rebuild manifest')
    report = json.loads(path.read_text())
    if report.get('version') not in (VERSION, 'multisource.curation.v2') or not report.get('complete'):
        raise ValueError('Incomplete/unknown curation report')
    validate_policy(report['policy'])
    if report['policy_sha256'] != fingerprint(report['policy']) or report['source_root'] != str(root):
        raise ValueError('Curation source/policy mismatch')
    if report['checks'].get('signals') != 'complete':
        raise ValueError('Missing signal QC coverage')
    if require_visual and report['checks'].get('visual') != 'complete':
        raise ValueError('Visual QC incomplete; explicitly request signal-only curation if intended')
    for rel, digest in report['metadata_hashes'].items():
        if file_sha256(within(root, rel)) != digest:
            raise ValueError('Curation metadata changed: ' + rel)
    for rel, spec in report['source_files'].items():
        stat = within(root, rel).stat()
        if stat.st_size != spec['size'] or stat.st_mtime_ns != spec['mtime_ns']:
            raise ValueError('Curation source file changed: ' + rel)
    for spec in report.get('dependencies', []):
        if file_sha256(spec['path']) != spec['sha256']:
            raise ValueError('Curation dependency changed')
    return report

def bind_curation(source, root, path, *, require_visual=True):
    report = verify_curation(root, path, require_visual=require_visual)
    out = deepcopy(source)
    out['curation'] = dict(version=report['version'], path=str(Path(path).resolve()), sha256=file_sha256(path), require_visual=require_visual, checks=report['checks'])
    suffix = '_qc_' + fingerprint(dict(policy=report['policy_sha256'], profile=report['profile_sha256'], checks=report['checks']))[:12]
    label = '_next_state_v1'
    key = out['stats_key']
    if suffix not in key:
        out['stats_key'] = key[:-len(label)] + suffix + label if key.endswith(label) else key + suffix
    return out

def curated_population(root, source, episodes, intervals, excluded):
    bound = source['curation']
    if bound['version'] not in (VERSION, 'multisource.curation.v2'):
        raise ValueError('Unknown bound curation version')
    report = verify_curation(root, bound['path'], bound['sha256'], require_visual=bound['require_visual'])
    rows = {str(e['episode_index']): e for e in episodes}
    if set(rows) != set(report['episodes']):
        raise ValueError('Curation episode coverage mismatch')
    intervals, excluded = (dict(intervals), dict(excluded))
    for eid, ep in rows.items():
        result = report['episodes'][eid]
        if result['length'] != ep['length']:
            raise ValueError('Curation episode length mismatch')
        if not result['keep']:
            excluded.setdefault(eid, 'curation:' + result['reason'])
            continue
        old = intervals.get(eid, [[0, ep['length']]])
        new = result['intervals']
        for ranges in (old, new):
            previous = 0
            for a, b in ranges:
                if not 0 <= a < b <= ep['length'] or a < previous:
                    raise ValueError('Invalid curation interval')
                previous = b
        selected = [[max(a, c), min(b, d)] for a, b in old for c, d in new if max(a, c) < min(b, d)]
        if selected:
            intervals[eid] = selected
        else:
            excluded.setdefault(eid, 'curation:empty_intersection')
    return (intervals, excluded)
