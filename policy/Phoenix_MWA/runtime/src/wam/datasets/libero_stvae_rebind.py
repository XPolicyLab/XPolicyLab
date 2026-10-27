from wam.utils.runtime_contract import semantic_contract
"Explicit, immutable reuse of a validated Wan cache with a new STVAE cache.\n\nNever modify old contracts/receipts. Validate the original pair first, then\nbind the new pair by content hashes, identical dataset/window identity, and\nthe new STVAE's full-dataset validation receipt.\n"
import hashlib
import json
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def require(value, message):
    if not value:
        raise ValueError('LIBERO STVAE rebind: ' + message)

def binding_content(vae_root, stvae_root, pair_contracts):
    vae, stvae = (Path(vae_root).resolve(), Path(stvae_root).resolve())
    vr = read(vae / 'contract.json')
    old = Path(vr['reference_stvae_root'])
    _, original = pair_contracts(vae, [old / 'head', old / 'wrist'])
    newer = [read(stvae / view / 'contract.json') for view in ('head', 'wrist')]
    require(semantic_contract({k: v for k, v in newer[0].items() if k != 'view'}) == semantic_contract({k: v for k, v in newer[1].items() if k != 'view'}), 'new views differ')
    for view, c in zip(('head', 'wrist'), newer):
        require(c['view'] == view and c['selection'] == 'full_dataset', 'new cache must be full')
        require(c['dataset_length'] == vr['dataset_length'] == 277713, 'dataset length mismatch')
        require(c['posterior'] == 'mean' and c['extra_latent_scaling'] == 'none' and (c['cache_dtype'] == 'bfloat16'), 'new latent format mismatch')
        require(c['encoder']['teacher_identity_verified'], 'expected verified teacher identity')
        for key in ('state_frames', 'action_horizon', 'rgb_offsets', 'latent_frames', 'pad_policy'):
            require(c['window'][key] == vr['window'][key], 'window mismatch: ' + key)
        for key in ('teacher_sha256', 'teacher_checkpoint_state', 'teacher_output_source', 'teacher_preprocessing', 'dinov3_source_commit'):
            require(c['encoder'][key] == original[2]['encoder'][key], 'DINO contract changed: ' + key)
    reports = sorted((stvae / '.launch').glob('*/VALIDATED.json'))
    matched = []
    for path in reports:
        r = read(path)
        if r.get('status') == 'passed' and r.get('scope') == 'full_dataset' and r.get('feature_hashes_verified') and (r.get('samples_per_view') == 277713) and (r.get('world_size') == newer[0]['world_size']) and all((r.get('views', {}).get(v, {}).get('contract_sha256') == digest(c) for v, c in zip(('head', 'wrist'), newer))):
            matched.append((str(path.resolve()), digest(r)))
    require(matched, 'no matching full-dataset STVAE checksum receipt')
    return {'schema': 'phoenix.libero_stvae_rebind.v1', 'vae_root': str(vae), 'stvae_root': str(stvae), 'original_stvae_root': str(old.resolve()), 'vae_contract_sha256': digest(vr), 'old_contract_sha256': [digest(c) for c in original], 'new_contract_sha256': [digest(c) for c in newer], 'dataset_snapshot_sha256': vr['dataset_snapshot_sha256'], 'new_validation_sha256': matched[-1][1]}

def load_bound_pair(vae_root, stvae_root, binding_file, pair_contracts):
    vae, stvae = (Path(vae_root).resolve(), Path(stvae_root).resolve())
    roots = [vae / 'head', vae / 'wrist', stvae / 'head', stvae / 'wrist']
    pair_contracts(vae, [stvae / 'head', stvae / 'wrist'])
    return roots, [read(p / 'contract.json') for p in roots]
