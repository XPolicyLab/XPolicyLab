"""Reproduce the v4 geometry-prior findings without importing policy code.

This is a targeted evidence collector, not a general proof that arbitrary Python
is free of privileged information. A zero finding count is not certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

RULES = {
    'privileged_pickup_geometry_override': re.compile(r'_CALIBRATED_PRIMARY_GEOMETRY|_PRIVILEGED_PRIMARY_KEYWORDS|same-task privileged trajectories'),
    'fixed_table_height': re.compile(r'(?<![\d.])(?:0)?\.765(?!\d)|_TABLE_Z\s*=\s*0\.74\b'),
    'exact_tube_height': re.compile(r'0\.7808752236133516'),
    'exact_coin_height': re.compile(r'0\.780(?:4|3999822616577)\b'),
    'known_object_scale': re.compile(r'known_side_m\s*=|use_known_square_scale.*True|_BLOCK_HALF_(?:SIZES|HEIGHTS)\s*='),
    'asset_shape_template': re.compile(r'(?:public_)?shapes\.(?:npz|json)'),
    'object_cad_archive': re.compile(r'\b(?:digit_meshes(?:_provenance)?\.(?:npz|json)|box_public\.npz|doll_profiles\.json)\b'),
    'rank_specific_doll_heights': re.compile(r'(?<![\d.])0?\.0501\s*,\s*0?\.0703\s*,\s*0?\.0903\s*,\s*0?\.1103\s*,\s*0?\.1305'),
    'known_mahjong_dimensions': re.compile(r'\b(?:HEIGHT|WIDTH|THICK)\s*=\s*0?\.(?:0647|0449|0328)\b'),
    'key_asset_insertion_offset': re.compile(r'(?<![\d.])0?\.02325(?!\d)'),
    'key_asset_tip_offset': re.compile(r'(?<![\d.])0?\.05284674(?!\d)'),
    'bottle_asset_radial_profile': re.compile(r'public_bottle_profile\.json|(?<![\d.])0?\.(?:0381942|0399855|0415241|0375717|0309621|0121638|0148733|2180715)(?!\d)'),
}


def audit(root, official_objects=None):
    root = Path(root).resolve(strict=True)
    template = json.loads((root / 'program-templates.json').read_text())
    if template.get('schema') != 'physicalrsi.program-templates/v1':
        raise ValueError('Audit requires original versioned program templates')
    dependencies = {}
    import_roots = {}
    for entry in template['programs']:
        descriptor_path = entry.get('worker_descriptor')
        if descriptor_path:
            descriptor = json.loads((root / descriptor_path).read_text())
            import_roots[entry['name']] = [path.removeprefix('{assets}/').rstrip('/') + '/'
                for path in descriptor.get('pythonpath', [])
                if path.startswith('{assets}/implementations/')]
        for path in entry['dependency_files']:
            dependencies.setdefault(path, []).append(entry['name'])
    grouped = {}
    generic_fonts=[]
    font_manifest=Path(__file__).parent/'asset_revision/physicalrsi_observed_geometry/fonts/provenance.json'
    known_fonts={item['file']:item['sha256'] for item in json.loads(font_manifest.read_text())['fonts']} if font_manifest.exists() else {}
    for relative, expected in template['text_files'].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Source path escapes asset tree')
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != expected:
            raise ValueError('Original source hash mismatch: ' + relative)
        if relative.startswith('physicalrsi_observed_geometry/fonts/') and path.suffix=='.ttf':
            if known_fonts.get(path.name)!=sha:
                raise ValueError('Unrecognized generic font content: '+relative)
            generic_fonts.append({'path':relative,'sha256':sha,'kind':'unmodified generic typography; no object metric geometry'})
            continue
        if sha not in grouped:
            findings = []
            for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
                for rule, pattern in RULES.items():
                    if pattern.search(line):
                        findings.append({'rule': rule, 'line': number, 'excerpt': line.strip()[:600]})
            grouped[sha] = {'sha256': sha, 'paths': [], 'programs': set(), 'importable_by': set(), 'findings': findings}
        grouped[sha]['paths'].append(relative)
        grouped[sha]['programs'].update(dependencies.get(relative, []))
        grouped[sha]['importable_by'].update(name for name, prefixes in import_roots.items()
                                            if any(relative.startswith(prefix) for prefix in prefixes))
    sources = []
    for record in grouped.values():
        if record['findings']:
            record['programs'] = sorted(record['programs'])
            record['importable_by'] = sorted(record['importable_by'])
            sources.append(record)
    report = {
        'schema': 'physicalrsi.legacy-geometry-audit/v1',
        'program_templates_sha256': hashlib.sha256((root / 'program-templates.json').read_bytes()).hexdigest(),
        'scope': 'Targeted static evidence; neither proof of runtime reachability for every hit nor compliance certification.',
        'qualified': False,
        'rule_patterns': {name: pattern.pattern for name, pattern in RULES.items()},
        'programs': [e['name'] for e in template['programs']],
        'sources': sources,
        'generic_font_files':generic_fonts,
    }
    if official_objects is not None:
        official = Path(official_objects)
        metadata = [(p, json.loads(p.read_text())) for p in sorted(official.rglob('metadata.json'))]
        comparisons = []
        seen = set()
        for path in sorted(root.glob('implementations/PourBallsRGB/*/PourBallsRGB/shapes.json')):
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            for label, shape in json.loads(path.read_text()).items():
                # JSON number comparison is exact; no tolerance or rounding.
                matches = [{'path': str(p.relative_to(official)),
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p, data in metadata
                           if data.get('geometry', {}).get('aligned_bbox', {}).get('vertices') == shape['bbox']]
                comparisons.append({'label': label, 'shape_sha256': sha, 'exact_matches': matches})
        report['official_bbox_comparison'] = comparisons
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--official-objects', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.official_objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'programs': len(report['programs']), 'flagged_sources': len(report['sources']), 'qualified': False}))
